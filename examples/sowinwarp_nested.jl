# ==============================================================================================
# sowinwarp_nested.jl -- SOWinWaRP as a one-way nest in B-SOSE, on NumericalEarth's
# ss/nested-ocean-model branch. Modelled on examples/bouvet_nested_ocean_simulation.jl.
#
#     SMOKE=1 julia --project=. examples/sowinwarp_nested.jl     # 1.5 days, proves it starts
#     julia --project=. examples/sowinwarp_nested.jl             # Jun-Aug 2014
#     END_DATE=2016-06-01T12:00 julia --project=. examples/sowinwarp_nested.jl   # 2 years
#
# END_DATE can be any time up to 2016-12-31T22:00 (the ERA5 converted so far covers 2014-2016).
# OUTPUT follows src/model.jl, so `./run_animate_simulation RUN_NAME` works on it unchanged:
# RUN_NAME_surface.nc, _midlon.nc, _free_surface.nc and _eke.nc in the working directory.
# RUN_NAME defaults to nested_smoke or nested_<start>_<end>, so runs do not overwrite each other.
#
# RUN examples/nested_probe.jl FIRST. This script needs BSOSEMonthly to satisfy the parent
# dataset interface; the probe says whether it does and what is missing. Starting here instead
# means discovering the same gaps one GPU job at a time.
#
# WHAT THIS REPLACES from src/model.jl. `nested_ocean_simulation` builds, in one call, what
# bsose_open_boundary_conditions + bsose_sponge_layer_forcing + the transport pin do now:
#
#   open boundaries      ocean_open_boundary_conditions from the parent's own strips
#   the sponge           ocean_boundary_relaxation, rate 1/6hours over ~8/12 deg by default --
#                        note that is 2x STRONGER than our SPONGE_TIMESCALE = 12hours, and a
#                        band rather than a raised-cosine taper. Set `relaxation_rate` and
#                        `relaxation_width` explicitly before comparing against the old runs.
#   the barotropic mode  sample_parent_depth_mean! computes the Flather exterior from the
#                        parent every step. That retires `target_transport`, which is
#                        materialised ONCE in the current model and is the live suspect for the
#                        north-face anomaly.
#   boundary bathymetry  blend_parent_bathymetry tapers the child's seafloor onto the parent's
#                        over ~4/12 deg. That is the thin/stepped boundary column problem that
#                        caused the west-face pileup, handled by construction.
#
# TWO THINGS TO DECIDE BEFORE READING ANY RESULT:
#
#   1. NO SEA ICE. `OceanOnlyModel` is ocean-only, and ERA5 gives ATMOSPHERIC state, not the
#      post-ice ocean fluxes B-SOSE's TFLUX/SFLUX carry. Over 70-45S that deletes the brine
#      rejection that makes Winter Water -- the variable this project exists to study. At
#      Bouvet (~54S, largely ice-free) it is fine; here it is not. The branch has
#      `OceanSeaIceModel(ocean, sea_ice; atmosphere, radiation)`; going there means ClimaSeaIce.
#      This script is ocean-only ON PURPOSE, as the first thing to get running, NOT as physics.
#
#   2. STATIC Z ON B-SOSE'S OWN LEVELS. The 52 levels of B-SOSE iteration 156 (4.2 m at the
#      surface to 400 m, 6000 m), not mutable. The parent's state then lands on the child level
#      for level, with no vertical interpolation at the boundaries, and the top cell is the same
#      4.2 m layer B-SOSE's surface fields describe. Static z as in src/model.jl, whose own grid
#      (8 m to 250 m, then stretched, 53 levels) is close but not identical. The first 2-year run
#      (2Y_nested_test) used a 40-level z-star ExponentialDiscretization instead: 11 m at the top,
#      11 levels in the upper 200 m. Its output names the vertical coordinate r_aac rather than
#      z_aac; tools/compare_surface_bsose.jl reads either.
# ==============================================================================================

using CUDA            # loads Oceananigans' CUDA extension, which is what defines `GPU()`
using CopernicusClimateDataStore    # an ERA5 backend: the ERA5 constructors always call `download`, and
                                    # without one that errors even when every file is on disk
using NumericalEarth
using NumericalEarth.DataWrangling: metadata_path
using Oceananigans
using Oceananigans.Units
using Oceananigans.Diagnostics: NaNChecker
using Oceanostics.ProgressMessengers: TimedMessenger
using NCDatasets      # loads Oceananigans' NetCDFWriter
using Dates, Printf

include(joinpath(@__DIR__, "..", "src", "setup_bsose.jl"))   # BSOSEMonthly and its file map

const SMOKE = get(ENV, "SMOKE", "0") == "1"
const ARCH  = get(ENV, "ARCH", "GPU") == "CPU" ? CPU() : GPU()
ARCH isa GPU && !CUDA.functional() &&
    (CUDA.functional(true); error("CUDA is not functional on this node. Check the GPU allocation, or set ARCH=CPU."))
ARCH isa GPU && @info "Running on GPU: $(CUDA.name(CUDA.device()))"
const BDIR  = get(ENV, "BSOSE_DIR", default_bsose_directory())

# ── domain: the production SOWinWaRP box at 1/6 deg ───────────────────────────────────────────
const λ₁, λ₂ = 90.0, 150.0
const φ₁, φ₂ = -70.0, -45.0
const SCALING = 6
const Nx, Ny = Int(SCALING * (λ₂ - λ₁)), Int(SCALING * (φ₂ - φ₁))   # 360 x 150

# B-SOSE iteration 156's level thicknesses, top down: `DRF` in its grid.nc (identical in i105's
# Theta/Salt files). 52 levels, 4.2 m to 400 m, summing to 6000 m.
const BSOSE_DRF = [4.2, 5.0, 5.9, 6.9, 8.5, 9.5, 10.0, 10.0, 10.0, 10.0,
                   10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 13.0, 17.0, 20.0, 20.0,
                   20.0, 20.0, 20.0, 20.0, 22.0, 30.0, 38.0, 45.0, 50.0, 50.0,
                   53.0, 72.0, 100.0, 100.0, 100.0, 100.0, 100.0, 150.0, 200.0, 200.0,
                   200.0, 220.0, 300.0, 380.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0,
                   400.0, 400.0]

# Faces bottom to top, as LatitudeLongitudeGrid wants them: -6000 m ... 0
z = reverse(-vcat(0.0, cumsum(BSOSE_DRF))) .+ 0.0     # .+ 0.0 turns the top face's -0.0 into 0.0
const Nz = length(z) - 1

# Must be a NATIVE B-SOSE stamp (records sit at odd times: 2014-05-02T02:00, 2014-06-01T12:00, ...).
# The initial snapshot goes through `bracketing_dates`, which looks `[start_date]` up in
# `all_dates` by exact match and crashes on `nothing` when it is not there. Workaround until
# that is fixed upstream on ss/nested-ocean-model.
start_date = DateTime(2014, 6, 1, 12)
end_date   = SMOKE ? DateTime(2014, 6, 3) : DateTime(get(ENV, "END_DATE", "2014-08-01T00:00"))
const RUN_NAME = get(ENV, "RUN_NAME", SMOKE ? "nested_smoke" :
                     "nested_" * Dates.format(start_date, "yyyymmdd") * "_" * Dates.format(end_date, "yyyymmdd"))

@info "SOWinWaRP nested: $(Nx)x$(Ny)x$(Nz) over $(λ₁)-$(λ₂)E, $(φ₁)-$(φ₂)S, $start_date to $end_date"

underlying_grid = LatitudeLongitudeGrid(ARCH;
                                        size = (Nx, Ny, Nz),
                                        halo = (7, 7, 7),
                                        z,
                                        longitude = (λ₁, λ₂),
                                        latitude = (φ₁, φ₂))

# The knobs match src/model.jl:452 so the nested run differs from the current model in the
# NESTING, not in the seafloor. (model.jl passes no `major_basins`, so it is the default 1.)
bottom_height = regrid_bathymetry(underlying_grid;
                                  dataset = ETOPO2022(),
                                  height_above_water = 1,
                                  minimum_depth = 10,
                                  interpolation_passes = 5,
                                  major_basins = 1)

grid = ImmersedBoundaryGrid(underlying_grid,
                            PartialCellBottom(bottom_height; minimum_fractional_cell_height = 0.2);
                            active_cells_map = true)

# ── the nest ──────────────────────────────────────────────────────────────────────────────────
# `dates` is a (start, end) tuple expanded to B-SOSE's native cadence. open_sides defaults to
# every bounded side of the grid, which for this box is all four. The south edge has no wet
# columns (Antarctica), so in practice it is the same three src/model.jl opens: W, E, N.
dataset = BSOSEMonthly(dir = BDIR, iteration = 156)

# EVERY parent record resident, as src/model.jl's `resident_fts` does, not the branch's default
# of 2 streamed from disk. Loading (and inpainting) then all happens here, before the run, and the
# stale-reload trap the `resident_fts` docstring measured cannot occur. The parent is only strips
# along the open sides, so even ~26 monthly records of 5 variables is a small fraction of the GPU.
parent_records = length(NumericalEarth.DataWrangling.bracketing_dates(dataset, :temperature, (start_date, end_date)))
@info "B-SOSE parent: $parent_records monthly records held in memory"

ocean = nested_ocean_simulation(grid, dataset;
                                dates = (start_date, end_date),
                                dir = BDIR,
                                parent_time_indices_in_memory = parent_records)

child = ocean.model.child

# ── atmosphere ────────────────────────────────────────────────────────────────────────────────
# Yearly files cropped to the region, converted from the Globus copy of NCAR's ERA5 by
# tools/download_era5_ncar.jl -- not downloaded from the CDS, whose queue made that impractical.
# Same region and directory as that script, which names the files after them.
#
# Mean sea level pressure, not surface pressure: interpolated onto coastal ocean cells, the
# surface pressure over the Antarctic ice sheet (tens of kPa lower) acts like a sea level step of
# several meters. NumericalEarth's docstring recommends this for ocean runs.
era5_dir = get(ENV, "ERA5_DIR", joinpath(BDIR, "era5"))
era5_region = BoundingBox(longitude = (λ₁ - 1.0, λ₂ + 1.0), latitude = (φ₁ - 1.0, φ₂ + 1.0))
era5_dataset = ERA5YearlySingleLevel()
era5_pressure = :mean_sea_level_pressure

# One extra hour at each end: ERA5 values are means over the hour ending at their stamp, so
# NumericalEarth places them half an hour earlier, and a series on exactly [start, end] leaves the
# first and last 30 minutes to `Cyclical()` wrap-around from the other end of the record.
era5_start = start_date - Hour(1)
era5_end   = end_date + Hour(1)

# Fail here rather than let the backend start a year-long CDS request for a missing file
era5_names = (:eastward_velocity, :northward_velocity, :temperature, :dewpoint_temperature, era5_pressure,
              :total_precipitation, :downwelling_shortwave_radiation, :downwelling_longwave_radiation)
missing_era5 = [path for name in era5_names
                for path in unique(metadata_path(Metadata(name; dataset = era5_dataset,
                                                          start_date = era5_start, end_date = era5_end,
                                                          dir = era5_dir, region = era5_region)))
                if !isfile(path)]
isempty(missing_era5) ||
    error("Missing ERA5 files (run tools/download_era5_ncar.jl):\n  " * join(missing_era5, "\n  "))

atmosphere = ERA5PrescribedAtmosphere(ARCH; dataset = era5_dataset, start_date = era5_start, end_date = era5_end,
                                      region = era5_region, dir = era5_dir, pressure = era5_pressure)
radiation  = ERA5PrescribedRadiation(ARCH; dataset = era5_dataset, start_date = era5_start, end_date = era5_end,
                                     region = era5_region, dir = era5_dir)

coupled_model = OceanOnlyModel(ocean; atmosphere, radiation)      # see note 1 in the header

# ── stepping and progress, as src/model.jl ────────────────────────────────────────────────────
# Start with a very small Δt so the first CATKE evaluation is numerically gentle;
# the wizard will ramp this up within a few iterations.
simulation = Simulation(coupled_model; Δt = 1seconds,
                        stop_time = (end_date - start_date).value / 1000)

# Adaptive time step on the advective CFL, with src/model.jl's settings. The wizard asks the
# simulation's model for its advective timescale, and the model here is the coupled
# EarthSystemModel, which has no such method for an ocean-only nest (NumericalEarth forwards it
# only for NestedModel itself and for nested ATMOSPHERES). So hand it the child ocean's directly.
child_advection_timescale(_) = Oceananigans.Advection.cell_advection_timescale(child)
wizard = TimeStepWizard(cfl = 0.6, max_change = 1.1, min_Δt = 0.1,
                        cell_advection_timescale = child_advection_timescale)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

# Fail loudly and early on NaN: every field, every step, and throw (see src/model.jl).
simulation.callbacks[:nan_checker] =
    Callback(NaNChecker(fields = merge(child.velocities, child.tracers), erroring = true),
             IterationInterval(1))

# Oceanostics' TimedMessenger, as src/model.jl's BSOSE path uses. It reads velocities, the closure
# and the CFL numbers off `simulation.model`, and the coupled model has none of them -- which is
# why src/model.jl swaps it out on its coupled path. This view hands it the child ocean as the
# model and the coupled simulation for everything else (Δt, time, iteration, stop time).
struct ChildView{S}
    simulation :: S
end
Base.getproperty(view::ChildView, name::Symbol) =
    name === :model ? child : getproperty(getfield(view, :simulation), name)
Oceananigans.Simulations.iteration(view::ChildView) = iteration(getfield(view, :simulation))   # the one Oceanostics calls
Base.time(view::ChildView) = time(getfield(view, :simulation))

output_interval = SMOKE ? 6hours : 1days

progress = TimedMessenger()
simulation.callbacks[:progress] = Callback(sim -> progress(ChildView(sim)), TimeInterval(output_interval))

# ── output ────────────────────────────────────────────────────────────────────────────────────
# The same four files src/model.jl writes, with the same names, fields, cadences and global
# attributes, so tools/animate_simulation.jl and the analysis tools read both runs alike:
#
#   RUN_NAME_surface.nc       u, v, T, S at k = Nz, daily
#   RUN_NAME_midlon.nc        u, v, T, S on the mid-longitude section, daily
#   RUN_NAME_free_surface.nc  η, daily
#   RUN_NAME_eke.nc           30-day means of u, v, T, S, uu, vv at cell centres, full depth
#
# Written off `child`, NOT off the NestedModel: the nest's prognostic state is the child.
const EKE_WINDOW = 30days
const EKE_STRIDE = 1

u, v, w = child.velocities
T, S = child.tracers.T, child.tracers.S

# Stamped on every file so the animation captions frames with calendar dates
output_metadata = Dict("run_name" => RUN_NAME,
    "start_date" => string(start_date),
    "end_date" => string(end_date),
    "dataset" => "BSOSE",
    "boundaries" => "one-way nest in B-SOSE (NumericalEarth nested_ocean_simulation)",
    "atmosphere" => "ERA5 (NCAR ds633.0), $(era5_pressure)",
    "vertical_coordinate" => "static z, B-SOSE i156's 52 levels")

ocean.output_writers[:surface] = NetCDFWriter(
    child,
    (; u, v, T, S),
    filename="$(RUN_NAME)_surface.nc",
    schedule=TimeInterval(output_interval),
    indices=(:, :, Nz),
    global_attributes=output_metadata,
    overwrite_files=true
)

mid_lon_idx = div(Nx, 2)
ocean.output_writers[:mid_lon] = NetCDFWriter(
    child,
    (; u, v, T, S),
    filename="$(RUN_NAME)_midlon.nc",
    schedule=TimeInterval(output_interval),
    indices=(mid_lon_idx, :, :),
    global_attributes=output_metadata,
    overwrite_files=true
)

# η lives at (Center, Center, Face) on a single level, which NetCDFWriter cannot write
# directly; reducing over the (length-one) vertical gives an ordinary 2D field.
η_2d = Field(Average(child.free_surface.displacement, dims=3))

ocean.output_writers[:free_surface] = NetCDFWriter(
    child,
    (; η=η_2d),
    filename="$(RUN_NAME)_free_surface.nc",
    schedule=TimeInterval(output_interval),
    global_attributes=output_metadata,
    overwrite_files=true
)

# Co-located velocities, squared after interpolating (see src/model.jl's EKE block). T and S
# ride along: this is the only full-depth output.
u_cc = @at (Center, Center, Center) u
v_cc = @at (Center, Center, Center) v
eke_outputs = (; u=u_cc, v=v_cc, T, S, uu=u_cc^2, vv=v_cc^2)

eke_metadata = merge(output_metadata, Dict(
    "averaging_window_seconds" => Float64(EKE_WINDOW),
    "averaging_stride" => EKE_STRIDE,
    "moment_location" => "Center, Center, Center",
    "eke_recipe" => "EKE = 0.5*((uu - u^2) + (vv - v^2)); MKE = 0.5*(u^2 + v^2)"))

simulation_seconds = (end_date - start_date).value / 1000
if EKE_WINDOW > simulation_seconds
    @warn "EKE_WINDOW ($(EKE_WINDOW / 1days) days) is longer than the run: $(RUN_NAME)_eke.nc will be created and left empty."
end

ocean.output_writers[:eke] = NetCDFWriter(
    child,
    eke_outputs,
    filename="$(RUN_NAME)_eke.nc",
    schedule=AveragedTimeInterval(EKE_WINDOW, window=EKE_WINDOW, stride=EKE_STRIDE),
    indices=(:, :, :),
    global_attributes=eke_metadata,
    overwrite_files=true
)

@info "output: $(RUN_NAME)_{surface,midlon,free_surface,eke}.nc in $(pwd())"

@info "starting"
run!(simulation)
@info "done"
