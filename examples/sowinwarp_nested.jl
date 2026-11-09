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
# RUN_NAME_surface.nc, _midlon.nc, _free_surface.nc and _eke.nc in the working directory, plus
# _sea_ice.nc (h, ℵ, snow, ice u/v daily) and _surface_fluxes.nc (what the coupler gives the ocean, daily means).
# RUN_NAME defaults to nested_smoke or nested_<start>_<end>, so runs do not overwrite each other.
# A checkpoint every 365 days and at the end; PICKUP=1 continues RUN_NAME from the latest one.
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
#   1. SEA ICE, STARTED FROM B-SOSE. `OceanSeaIceModel` with ClimaSeaIce through NumericalEarth's
#      `sea_ice_simulation`, on the child's grid, defaults but for advection and coasts (below):
#      slab ice + snow, ice salinity 4 g/kg (B-SOSE implies 4.1), EVP dynamics, three-equation
#      ice-ocean heat flux. The
#      nest's ocean coupling forwards to the child (NestedModels/nested_component.jl), so the ice
#      sees the child ocean. The ice starts from B-SOSE's SIarea/SIheff on start_date (5-day files,
#      read in this script, not through setup_bsose.jl), bare of snow. The ice model's sides are
#      walls, so in the 4-cell relaxation band along the open W, E and N sides the ice is relaxed
#      to B-SOSE's every step: ice leaves through the band and B-SOSE's ice enters from it.
#      Up to fix3 the ice started at zero on 1 June and the band's ice was deleted, so all of it
#      was frozen locally (ICE_INIT = "none", ICE_BOUNDARY = "delete" in the switches give that back). The wizard also limits Δt on the ICE
#      speed, and a run stops if the ice volume h·ℵ passes 50 m. 2Y_nested_seaice_full had none of these
#      three: its ice piled against the east edge from day 110 and grew to 10⁶ m.
#      The ice is advected by IncrementalRemapping, not WENO. ClimaSeaIce carries volume h·ℵ and ℵ
#      separately and takes h = volume/ℵ; WENO reconstructs the two differently next to a thick
#      cell, so outflow strips ℵ faster than volume, h inflates, and the ice strength
#      (∝ h·exp(-20(1-ℵ))) collapses. 2Y_nested_seaice_fix2 died that way in a coastal corner at
#      97.1E 65.1S: ℵ 0.71 → 0.14 in a day, h 54 m, then volume 7.7 → 30.9 m in two days.
#      Remapping moves ℵ and volume across a face together, keeping h within the local range.
#      The coasts are free-slip, MITgcm's default (SEAICE_no_slip = .FALSE.), not NumericalEarth's
#      no-slip: that corner had land W and S, and no-slip held the ice still along both walls, so
#      ice could leave only straight out N or E.
#      The ocean-only runs (2Y_nested_test, 2Y_nested_bsosez) had none of this. Their coast went
#      warm and salty every summer, with nothing melting into it: the missing sea-ice cycle.
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
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Operators: Δzᶜᶜᶜ
using NumericalEarth.SeaIces.ClimaSeaIce: IncrementalRemapping   # via NumericalEarth, not a direct dependency
using Oceanostics.ProgressMessengers: TimedMessenger
using NCDatasets      # loads Oceananigans' NetCDFWriter
using Dates, Printf

include(joinpath(@__DIR__, "..", "src", "setup_bsose.jl"))   # BSOSEMonthly and its file map

const SMOKE = get(ENV, "SMOKE", "0") == "1"
const PICKUP = get(ENV, "PICKUP", "0") == "1"   # continue RUN_NAME from its latest checkpoint; see "checkpoints"
const ARCH  = get(ENV, "ARCH", "GPU") == "CPU" ? CPU() : GPU()
ARCH isa GPU && !CUDA.functional() &&
    (CUDA.functional(true); error("CUDA is not functional on this node. Check the GPU allocation, or set ARCH=CPU."))
ARCH isa GPU && @info "Running on GPU: $(CUDA.name(CUDA.device()))"
const BDIR  = get(ENV, "BSOSE_DIR", default_bsose_directory())

# ── experiment switches: set HERE, not on the sbatch line ─────────────────────────────────────
# What the run does to the physics is chosen in the code. The environment only says which run this
# is and where things are (RUN_NAME, END_DATE, SMOKE, PICKUP, ARCH, BSOSE_DIR, ERA5_DIR,
# CHECKPOINT_DAYS). An environment switch that is not set falls back silently, and fix5's first
# submission ran without its freshwater that way. The values are logged at startup and stamped on
# every output file.
const COASTAL_FRESHWATER      = true                            # B-SOSE's coastal freshwater; see "coastal freshwater"
const COASTAL_FRESHWATER_FILE = "coastal_freshwater_bsose.nc"   # written by tools/coastal_freshwater.jl
const ICE_INIT                = "bsose"   # "bsose": start from B-SOSE's ice; "none": ice-free (up to fix3)
const ICE_BOUNDARY            = "bsose"   # "bsose": relax the open-side bands to B-SOSE's ice; "delete" (up to fix3)
const ICE_RELAXATION_HOURS    = 6         # the bands' relaxation time, with ICE_BOUNDARY = "bsose"
const PENETRATING_SHORTWAVE   = true      # shortwave through the column (TwoColorRadiation); false: all in the top cell (up to fix5_flux)
const ICE_GROUNDING           = false     # Lemieux (2015) landfast basal stress where depth < 30 m; true up to fix6 (see "sea ice")

ICE_INIT in ("bsose", "none") || error("ICE_INIT must be \"bsose\" or \"none\", got $ICE_INIT")
ICE_BOUNDARY in ("bsose", "delete") || error("ICE_BOUNDARY must be \"bsose\" or \"delete\", got $ICE_BOUNDARY")
for name in ("COASTAL_FRESHWATER", "COASTAL_FRESHWATER_FILE", "ICE_INIT", "ICE_BOUNDARY", "ICE_RELAXATION_HOURS",
             "PENETRATING_SHORTWAVE", "ICE_GROUNDING")
    haskey(ENV, name) && @warn "$name is set in the environment, which this script no longer reads: set it in the switches block of examples/sowinwarp_nested.jl"
end
@info "switches: COASTAL_FRESHWATER = $COASTAL_FRESHWATER ($COASTAL_FRESHWATER_FILE), ICE_INIT = $ICE_INIT, " *
      "ICE_BOUNDARY = $ICE_BOUNDARY, ICE_RELAXATION_HOURS = $ICE_RELAXATION_HOURS, " *
      "PENETRATING_SHORTWAVE = $PENETRATING_SHORTWAVE, ICE_GROUNDING = $ICE_GROUNDING"

# ── penetrating shortwave (PENETRATING_SHORTWAVE, in the switches) ────────────────────────────
# The coupler finds the ocean's TwoColorRadiation with `get_radiative_forcing(ocean)`
# (Radiations/apply_air_sea_radiative_fluxes.jl). NumericalEarth's ss/nested-ocean-model branch
# (PR #676) has that method for a HydrostaticFreeSurfaceModel but not for a NestedModel, so for
# this nest it fell through to `get_radiative_forcing(something) = nothing`: ALL the shortwave went
# into the top cell's heat flux, and the TwoColorRadiation the child carries (hydrostatic_ocean_model's
# default) was never fed. Every nested run up to 2Y_nested_seaice_fix5_flux ran that way; its
# _surface_fluxes.nc heat_flux follows the shortwave through the seasons. Forward to the child, as
# NestedModels/nested_component.jl already forwards net_fluxes. Only defined when the package has
# no method of its own, so this goes quiet by itself once #676 carries the fix. Checked after the
# coupled model is built (search "penetrating shortwave: ON").
if PENETRATING_SHORTWAVE &&
   Base.unwrap_unionall(which(NumericalEarth.Oceans.get_radiative_forcing, (NumericalEarth.NestedModel,)).sig).parameters[2] === Any
    NumericalEarth.Oceans.get_radiative_forcing(nm::NumericalEarth.NestedModel) =
        NumericalEarth.Oceans.get_radiative_forcing(nm.child)
end

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

# Fill dead-end pockets: an ocean column with land on 3 or 4 of its 4 sides becomes land, repeated
# until none is left, so one-cell inlets close from their dead end outward. 2Y_nested_seaice_fix1
# had one at 142.6E 67.1S (i = 316, j = 18): 10 m deep (minimum_depth), open only to the north,
# under a polynya. Ice formed there all the time and its brine could neither sink nor mix out, so
# the surface went from 34.5 to 46 g/kg in 26 days. Off the grid counts as OCEAN, so the open
# sides are never filled. Land is written as 0, which the immersed boundary treats as dry.
function fill_dead_end_pockets!(bottom_height)
    z = Array(interior(bottom_height))[:, :, 1]
    nx, ny = size(z)
    land(i, j) = 1 <= i <= nx && 1 <= j <= ny && z[i, j] >= 0
    filled = 0
    while true
        pockets = [(i, j) for i in 1:nx, j in 1:ny
                   if z[i, j] < 0 && count((land(i-1, j), land(i+1, j), land(i, j-1), land(i, j+1))) >= 3]
        isempty(pockets) && break
        for (i, j) in pockets
            z[i, j] = 0
        end
        filled += length(pockets)
    end
    set!(bottom_height, reshape(z, nx, ny, 1))
    fill_halo_regions!(bottom_height)
    return filled
end

@info "bathymetry: $(fill_dead_end_pockets!(bottom_height)) dead-end pocket cells filled"

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

# ── coastal freshwater (COASTAL_FRESHWATER, in the switches) ────────────────────────────────────────────────
# Our only surface freshwater is ERA5 precipitation: no ice-shelf melt, iceberg melt or runoff.
# B-SOSE prescribes Antarctic melt and calving (Hammond & Jones 2016, per the 2024 GMD evaluation of
# MITgcm reanalyses), and every nested run has a salty strip along the coast that the sea-ice
# changes of fix4 barely touched. tools/coastal_freshwater.jl measures what B-SOSE's coast gets
# that ours does not, oceFWflx - SFLUX/S_ICE - ERA5 P, minus its offshore value, and writes it on
# this grid to COASTAL_FRESHWATER_FILE. Run that first. Raw oceFWflx is NOT used: it holds B-SOSE's
# own sea-ice freezing and melting, which our ice model already does.
#
# Applied as a virtual salt flux in the top cell, -S F / (ρ₀ Δz), the way the coupler applies the
# rest of the freshwater, and constant in time like B-SOSE's prescribed melt. A caller's `forcing`
# is ADDED to the nest's boundary relaxation (NestedModels/nested_ocean_model.jl), so this does not
# displace it. The flux is concentrated: up to ~44 m/yr at the Shackleton front (96-97E 65.1-65.3S)
# and at 103.25E 65.25S, i.e. ~1 g/kg per day in an unmixed 4.2 m top cell.
const COASTAL_ρ₀ = 1026.0

@inline function coastal_freshwater_dilution(i, j, k, grid, clock, fields, p)
    @inbounds S = fields.S[i, j, k]
    @inbounds F = p.F[i, j]
    return ifelse(k == grid.Nz, -S * F / (p.ρ₀ * Δzᶜᶜᶜ(i, j, k, grid)), zero(S))
end

function coastal_freshwater_forcing()
    COASTAL_FRESHWATER || return NamedTuple()
    isfile(COASTAL_FRESHWATER_FILE) ||
        error("$COASTAL_FRESHWATER_FILE not found: run tools/coastal_freshwater.jl first, or set COASTAL_FRESHWATER = false in the switches")
    ds = NCDataset(COASTAL_FRESHWATER_FILE)
    lon, lat = Float64.(ds["lon"][:]), Float64.(ds["lat"][:])
    F = Float64.(coalesce.(ds["freshwater_flux"][:, :], 0.0))
    attributes = Dict(ds.attrib)
    close(ds)
    λc = λ₁ .+ ((1:Nx) .- 0.5) ./ SCALING
    φc = φ₁ .+ ((1:Ny) .- 0.5) ./ SCALING
    (size(F) == (Nx, Ny) && maximum(abs, lon .- λc) < 1e-6 && maximum(abs, lat .- φc) < 1e-6) ||
        error("$COASTAL_FRESHWATER_FILE is not on this grid ($(Nx)x$(Ny), $(λ₁)-$(λ₂)E, $(φ₁)-$(φ₂)S)")
    land = Array(interior(bottom_height))[:, :, 1] .>= 0
    F[land] .= 0
    area = [(6.371e6 * deg2rad(1 / SCALING))^2 * cosd(φ) for _ in λc, φ in φc]
    @info @sprintf("coastal freshwater from %s: %.0f Gt/yr, up to %.1f m/yr in a cell (%s)",
                   COASTAL_FRESHWATER_FILE, sum(F .* area) * 365.25 * 86400 / 1e12,
                   maximum(F) * 365.25 * 86400 / 1000, get(attributes, "method", "no method attribute"))
    get(attributes, "era5_precipitation_subtracted", "true") == "true" ||
        @warn "$COASTAL_FRESHWATER_FILE was made without ERA5 precipitation subtracted"
    return (; S = Forcing(coastal_freshwater_dilution; discrete_form = true,
                          parameters = (; F = on_architecture(ARCH, F), ρ₀ = COASTAL_ρ₀)))
end

ocean = nested_ocean_simulation(grid, dataset;
                                dates = (start_date, end_date),
                                dir = BDIR,
                                parent_time_indices_in_memory = parent_records,
                                forcing = coastal_freshwater_forcing())

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

# ── sea ice: see note 1 in the header ─────────────────────────────────────────────────────────
# On the child's grid, so ice and ocean share cells and the exchange grid is the child's own.
# IncrementalRemapping needs the forward-Euler stepper (the sea_ice_simulation default), a halo of 2
# and an ice Courant number below 1, which the wizard's ice CFL keeps.
#
# GROUNDING (ICE_GROUNDING). NumericalEarth's `sea_ice_dynamics` turns on ClimaSeaIce's
# LandfastBasalStress (Lemieux et al. 2015: ice thicker than depth*ℵ/8 is held by the sea floor,
# wherever the water is under 30 m), although ClimaSeaIce's own default is none. MITgcm has the same
# scheme off by default. In our grid it acts in shallow pockets, and at 97.4E 65.2S, a 10.4 m cell
# walled by land on two sides, it held every floe over ~1.3 m: up to fix6 the pile there reached 47 m
# (B-SOSE: ~2 m, often open). ICE_GROUNDING = false builds the same dynamics with `basal_stress =
# nothing`. The coast condition is applied by sea_ice_simulation itself, not by the dynamics, so it
# stays free-slip.
ice_dynamics = ICE_GROUNDING ? NumericalEarth.SeaIces.sea_ice_dynamics(child.grid, ocean) :
                               NumericalEarth.SeaIces.sea_ice_dynamics(child.grid, ocean; basal_stress = nothing)
sea_ice = sea_ice_simulation(child.grid, ocean; advection = IncrementalRemapping(),
                             lateral_boundary_condition = :free_slip,
                             dynamics = ice_dynamics)
@info "sea-ice grounding: $(ICE_GROUNDING ? "ON (LandfastBasalStress)" : "OFF")" *
      " -- basal_stress is $(nameof(typeof(sea_ice.model.dynamics.basal_stress)))"

# ── B-SOSE sea ice: the starting state and the open-side bands (note 1) ──────────────────────
# SIarea is ℵ; SIheff is ice volume per unit area of CELL, i.e. h·ℵ, not h. Both come from the
# 5-day files tools/ice_production.jl reads, <token>_bsoseI156_<year>_5day.nc in BSOSE_DIR, and
# are regridded onto the child grid once, here, for the whole run. Snow is not in those files, so
# the ice starts bare and the band ice carries no snow.
const ICE_ℵ_MIN = 0.01   # less ice area than this counts as open water: h = volume/ℵ is noise there
const ICE_FILL_CELLS = 6  # B-SOSE cells (~1 deg of longitude) searched for ocean to fill B-SOSE land

bsose_ice_file(token, year) = joinpath(BDIR, "$(token)_bsoseI156_$(year)_5day.nc")

"""
    load_bsose_ice(first_date, last_date)

B-SOSE ℵ and volume h·ℵ on the child grid, `(; dates, ℵ, V)` with `dates` the 5-day window
CENTRES (a 5-day stamp is the END of its window; see tools/ice_production.jl) and `ℵ`, `V` of
size (Nx, Ny, records). B-SOSE's land is first given the ice of its nearest B-SOSE ocean cell
within ICE_FILL_CELLS, then the result is bilinear in longitude and latitude. Our coastline is
not B-SOSE's (at 100E theirs lies ~1.5 degrees further north), and without the fill a child
coastal cell over B-SOSE land would start with no ice at all. Child land, and ice of less than
ICE_ℵ_MIN, become 0.
"""
function load_bsose_ice(first_date, last_date)
    geometry = joinpath(BDIR, resolve_bsose_filename(BDIR, :surface_heat_flux, 156))
    ds = NCDataset(geometry)
    xc, yc = load_coord(ds, "XC", 1), load_coord(ds, "YC", 2)
    iB = findall(x -> λ₁ - 1 <= x <= λ₂ + 1, xc)
    jB = findall(y -> φ₁ - 1 <= y <= φ₂ + 1, yc)
    iB, jB = minimum(iB):maximum(iB), minimum(jB):maximum(jB)
    wetB = Float64.(coalesce.(ds["Depth"][iB, jB], 0.0)) .> 0
    close(ds)
    xs, ys = xc[iB], yc[jB]

    # Each B-SOSE cell's source: itself if wet, else its nearest wet cell, else 0 (none in reach)
    lin = LinearIndices(wetB)
    source = zeros(Int, size(wetB))
    for I in CartesianIndices(wetB)
        best, distance = 0, Inf
        for di in -ICE_FILL_CELLS:ICE_FILL_CELLS, dj in -ICE_FILL_CELLS:ICE_FILL_CELLS
            J = I + CartesianIndex(di, dj)
            (checkbounds(Bool, wetB, J) && wetB[J] && di^2 + dj^2 < distance) || continue
            best, distance = lin[J], di^2 + dj^2
        end
        source[I] = best
    end

    # Child cell centres (the grid is uniform at 1/SCALING degree) and the 4-point stencil
    λc = λ₁ .+ ((1:Nx) .- 0.5) ./ SCALING
    φc = φ₁ .+ ((1:Ny) .- 0.5) ./ SCALING
    bracket(s, x) = (a = clamp(searchsortedlast(s, x), 1, length(s) - 1);
                     (a, clamp((x - s[a]) / (s[a+1] - s[a]), 0, 1)))
    wet = Array(interior(bottom_height))[:, :, 1] .< 0
    regrid(A) = [wet[i, j] ? stencil(A, bracket(xs, λc[i])..., bracket(ys, φc[j])...) : 0.0
                 for i in 1:Nx, j in 1:Ny]
    function stencil(A, a, wx, b, wy)
        num = den = 0.0
        for (di, w₁) in ((0, 1 - wx), (1, wx)), (dj, w₂) in ((0, 1 - wy), (1, wy))
            s = source[a+di, b+dj]
            s > 0 || continue
            num += w₁ * w₂ * A[s]; den += w₁ * w₂
        end
        return den > 0 ? num / den : 0.0
    end

    unreached = count(wet .& (regrid(ones(size(wetB))) .== 0))
    unreached > 0 && @warn "$unreached child ocean cells are over $ICE_FILL_CELLS B-SOSE cells from B-SOSE ocean: no B-SOSE ice there"

    # The years the run needs, plus either neighbour that is on disk, so a run starting or ending
    # within 2.5 days of New Year is still bracketed
    years = [y for y in year(first_date)-1:year(last_date)+1
             if year(first_date) <= y <= year(last_date) || isfile(bsose_ice_file("SIArea", y))]
    dates, ℵs, Vs = DateTime[], Matrix{Float64}[], Matrix{Float64}[]
    for y in years, (token, names, out) in (("SIArea", ("SIarea", "SIArea", "SIAREA"), ℵs), ("SIHeff", ("SIheff", "SIHeff", "SIHEFF"), Vs))
        path = bsose_ice_file(token, y)
        isfile(path) || error("$path not found: the B-SOSE 5-day $token files are needed for " *
                              "ICE_INIT/ICE_BOUNDARY = \"bsose\" (or set them to \"none\"/\"delete\" in the switches)")
        ds = NCDataset(path)
        found = filter(n -> haskey(ds, n), names)
        isempty(found) && error("no $(join(names, " or ")) in $path; it holds $(collect(keys(ds)))")
        name = first(found)
        A = Float64.(coalesce.(ds[name][iB, jB, :], 0.0))
        token == "SIArea" && append!(dates, DateTime.(ds["time"][:]) .- Hour(60))   # stamp - 2.5 days
        close(ds)
        append!(out, [regrid(A[:, :, n]) for n in axes(A, 3)])
    end
    length(dates) == length(ℵs) == length(Vs) || error("SIArea and SIHeff hold different records")
    issorted(dates) || error("B-SOSE ice records are out of order: $(first(dates)) .. $(last(dates))")
    (first(dates) <= first_date && last(dates) >= last_date) ||
        error("B-SOSE ice covers $(first(dates)) to $(last(dates)), not the run's $first_date to $last_date")

    ℵ, V = stack(ℵs), stack(Vs)
    for n in eachindex(ℵ)
        open_water = !(ℵ[n] >= ICE_ℵ_MIN)
        ℵ[n] = open_water ? 0.0 : min(ℵ[n], 1.0)
        V[n] = open_water ? 0.0 : max(V[n], 0.0)
    end
    return (; dates, ℵ, V)
end

# ℵ and V at `date`, linear in time between the two window centres around it
function bsose_ice_at(ice, date)
    n = searchsortedlast(ice.dates, date)
    1 <= n < length(ice.dates) || error("$date is outside the B-SOSE ice records")
    w = (date - ice.dates[n]) / (ice.dates[n+1] - ice.dates[n])
    return (1 - w) .* ice.ℵ[:, :, n] .+ w .* ice.ℵ[:, :, n+1],
           (1 - w) .* ice.V[:, :, n] .+ w .* ice.V[:, :, n+1]
end

const BSOSE_ICE = (ICE_INIT == "bsose" || ICE_BOUNDARY == "bsose") ? load_bsose_ice(start_date, end_date) : nothing

# Ice area and volume over the domain, to read against tools/plot_sea_ice.jl's series
function ice_totals(ℵ, V)
    A = [(6.371e6 * deg2rad(1 / SCALING))^2 * cosd(φ₁ + (j - 0.5) / SCALING) for i in 1:Nx, j in 1:Ny]
    return sum(ℵ .* A) / 1e12, sum(V .* A) / 1e9   # 10⁶ km², km³
end

# Start from B-SOSE's ice on start_date, not from none. Starting ice-free on 1 June made the model
# freeze a whole early winter's ice out of an ocean that B-SOSE had already balanced against that
# ice, and fix3's interior surface salinity rose 34.04 → 34.20 in the first 50 days. A pickup
# takes its ice from the checkpoint instead.
if ICE_INIT == "bsose" && !PICKUP
    ℵ₀, V₀ = bsose_ice_at(BSOSE_ICE, start_date)
    h₀ = map((a, v) -> a >= ICE_ℵ_MIN ? v / a : 0.0, ℵ₀, V₀)
    ℵ₀[h₀ .== 0] .= 0
    set!(sea_ice.model.ice_concentration, reshape(ℵ₀, Nx, Ny, 1))
    set!(sea_ice.model.ice_thickness, reshape(h₀, Nx, Ny, 1))
    fill_halo_regions!(sea_ice.model.ice_concentration)
    fill_halo_regions!(sea_ice.model.ice_thickness)
    @info @sprintf("sea ice from B-SOSE on %s: area %.2f 10⁶ km², volume %.0f km³, h up to %.1f m",
                   start_date, ice_totals(ℵ₀, V₀)..., maximum(h₀))
end

coupled_model = OceanSeaIceModel(ocean, sea_ice; atmosphere, radiation)

let R = NumericalEarth.Oceans.get_radiative_forcing(ocean)
    if PENETRATING_SHORTWAVE
        R isa NumericalEarth.Oceans.TwoColorRadiation ||
            error("PENETRATING_SHORTWAVE = true, but the coupler finds $(typeof(R)) for this ocean, not the child's " *
                  "TwoColorRadiation: the shortwave would all go into the top cell. See \"penetrating shortwave\".")
        @info "penetrating shortwave: ON (the coupler feeds the child's TwoColorRadiation)"
    else
        @warn "PENETRATING_SHORTWAVE = false: all the shortwave is absorbed in the top cell, as up to fix5_flux"
    end
end

# ── stepping and progress, as src/model.jl ────────────────────────────────────────────────────
# Start with a very small Δt so the first CATKE evaluation is numerically gentle;
# the wizard will ramp this up within a few iterations.
simulation = Simulation(coupled_model; Δt = 1seconds,
                        stop_time = (end_date - start_date).value / 1000)

# Adaptive time step on the advective CFL, with src/model.jl's settings. The wizard asks the
# simulation's model for its advective timescale, and the model here is the coupled
# EarthSystemModel, which has no such method for an ocean-only nest (NumericalEarth forwards it
# only for NestedModel itself and for nested ATMOSPHERES). So hand it the child ocean's directly.
#
# The ICE is advected on the same Δt and is faster: 2Y_nested_seaice_full reached 1-4 m/s ice
# against Δt of 20-40 min, i.e. ice CFL above 1 on these 6-8 km cells, and the thickness grew
# without bound (8 m on day 105, 1 km by day 180, hundreds of km after). Oceananigans'
# cell_advection_timescale wants 3-D velocities, so the ice gets a conservative bound instead:
# its fastest u and v over the smallest Δx and Δy anywhere on the grid. The wizard takes the
# shorter of the two, so Δt only drops when fast ice demands it; it is not a fixed cap.
const ICE_Δx_MIN = minimum_xspacing(child.grid.underlying_grid)
const ICE_Δy_MIN = minimum_yspacing(child.grid.underlying_grid)

function ice_advection_timescale()
    uᵢ, vᵢ = sea_ice.model.velocities
    inverse = maximum(abs, uᵢ) / ICE_Δx_MIN + maximum(abs, vᵢ) / ICE_Δy_MIN
    return inverse > 0 ? 1 / inverse : Inf
end

coupled_advection_timescale(_) = min(Oceananigans.Advection.cell_advection_timescale(child),
                                     ice_advection_timescale())
wizard = TimeStepWizard(cfl = 0.6, max_change = 1.1, min_Δt = 0.1,
                        cell_advection_timescale = coupled_advection_timescale)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

# Fail loudly and early on NaN: every field, every step, and throw (see src/model.jl).
simulation.callbacks[:nan_checker] =
    Callback(NaNChecker(fields = merge(child.velocities, child.tracers,
                                       (; h = sea_ice.model.ice_thickness, ℵ = sea_ice.model.ice_concentration)),
                        erroring = true),
             IterationInterval(1))

# ── ice at the open sides ─────────────────────────────────────────────────────────────────────
# The ocean's W, E and N sides are open, but the ice model's are walls: ice drifting west along
# the coast piled against the west edge, ice carried east by the ACC against the east edge, and
# the first runaway thickness in 2Y_nested_seaice_full started at the east edge (i = 360, day 110).
# So the ice is taken in hand in the band where the nest relaxes the ocean to B-SOSE (8/12 deg,
# 4 cells here), every step.
#
# ICE_BOUNDARY = bsose (the default): ℵ and volume h·ℵ are relaxed to B-SOSE's in that band, on
# ICE_RELAXATION_HOURS (6, the nest's ocean relaxation). Ice piling against a side is drained
# toward B-SOSE's, and ice drifts IN from the band where B-SOSE has it: the westward coastal drift
# brings ice in across the east side, which deletion never could (fix3's East next band held half
# the interior's ice). The relaxation edits the ice state directly, so it exchanges no heat, salt or
# freshwater with the ocean, as ice arriving from or leaving for outside the domain would not.
#
# ICE_BOUNDARY = delete (up to fix3): the band's ice is set to zero. Ice leaves, none enters, and
# ice frozen inside the band is deleted at once, its brine left behind (tools/check_edge_bands.jl:
# fix3's bands salted 0.03-0.09 g/kg more each winter than the interior).
const ICE_EDGE = 4
const ICE_FIELDS = (sea_ice.model.ice_thickness, sea_ice.model.ice_concentration, sea_ice.model.snow_thickness)

function remove_edge_ice!(_)
    for field in ICE_FIELDS
        data = interior(field)
        data[1:ICE_EDGE, :, :] .= 0                # west
        data[Nx-ICE_EDGE+1:Nx, :, :] .= 0          # east
        data[:, Ny-ICE_EDGE+1:Ny, :] .= 0          # north
    end
    return nothing
end

const ICE_RELAXATION = ICE_RELAXATION_HOURS * hours
const ICE_TARGET_EVERY = 1hours   # B-SOSE's ice is 5-day means; refreshing hourly is plenty
const ICE_BAND = on_architecture(ARCH, [i <= ICE_EDGE || i > Nx - ICE_EDGE || j > Ny - ICE_EDGE
                                        for i in 1:Nx, j in 1:Ny])
const ICE_TARGET_ℵ = on_architecture(ARCH, zeros(Nx, Ny))
const ICE_TARGET_V = on_architecture(ARCH, zeros(Nx, Ny))
const ICE_TARGET_TIME = Ref(-Inf)

function refresh_ice_target!(t)
    t - ICE_TARGET_TIME[] < ICE_TARGET_EVERY && return nothing
    ℵ★, V★ = bsose_ice_at(BSOSE_ICE, start_date + Millisecond(round(Int, 1000t)))
    copyto!(ICE_TARGET_ℵ, ℵ★)
    copyto!(ICE_TARGET_V, V★)
    ICE_TARGET_TIME[] = t
    return nothing
end

@inline relaxed_area(ℵ, ℵ★, r) = (1 - r) * ℵ + r * ℵ★
@inline function relaxed_thickness(h, ℵ, ℵ★, V★, r)
    a = relaxed_area(ℵ, ℵ★, r)
    return ifelse(a < ICE_ℵ_MIN, zero(h), ((1 - r) * h * ℵ + r * V★) / max(a, ICE_ℵ_MIN))
end

# One fused broadcast per field over the whole surface, changing only the band. h is updated
# before ℵ because both use the old ℵ; snow goes where the ice does.
function relax_edge_ice!(sim)
    refresh_ice_target!(time(sim))
    r = min(1.0, sim.Δt / ICE_RELAXATION)
    h, ℵ, hs = (view(interior(f), :, :, 1) for f in ICE_FIELDS)
    ℵ★, V★, band = ICE_TARGET_ℵ, ICE_TARGET_V, ICE_BAND
    @. hs = ifelse(band & (relaxed_area(ℵ, ℵ★, r) < ICE_ℵ_MIN), zero(hs), hs)
    @. h  = ifelse(band, relaxed_thickness(h, ℵ, ℵ★, V★, r), h)
    @. ℵ  = ifelse(band, ifelse(relaxed_area(ℵ, ℵ★, r) < ICE_ℵ_MIN, zero(ℵ), relaxed_area(ℵ, ℵ★, r)), ℵ)
    return nothing
end

simulation.callbacks[:edge_ice] = ICE_BOUNDARY == "bsose" ?
    Callback(relax_edge_ice!, IterationInterval(1)) :
    Callback(remove_edge_ice!, IterationInterval(1))

# Ice can run away without ever going NaN (the NaN checker did not fire on 2Y_nested_seaice_full,
# whose ice reached 10⁶ m). Watched as VOLUME per unit area, h·ℵ: h alone is the thickness of the
# ice that is there, and at the ice edge a cell with ℵ ≈ 0 can carry a meaningless h (21.6 m at
# ℵ = 0.000 on day 22 of 2Y_nested_seaice_fix1). No real Antarctic ice holds 50 m per square metre.
const ICE_MAX_VOLUME = 50
function check_ice_volume(sim)
    v_max = maximum(interior(sea_ice.model.ice_thickness) .* interior(sea_ice.model.ice_concentration))
    v_max > ICE_MAX_VOLUME &&
        error(@sprintf("sea ice volume %.1f m (h·ℵ) at t = %.2f days, iteration %d: the ice is blowing up",
                       v_max, time(sim) / 86400, iteration(sim)))
    return nothing
end
simulation.callbacks[:ice_volume] = Callback(check_ice_volume, IterationInterval(10))

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

# ── health check: one line per output interval, a @warn when something looks wrong ─────────────
# 2Y_nested_seaice_full went wrong for months without a NaN, and the surface only showed it ~130
# days after the ice did. This prints, once per output interval, the numbers that would have shown
# it: surface S and T ranges, the largest change since the last check, the domain-mean drift since
# the start, and the ice. It never stops the run (the NaN checker and the 50 m ice-volume guard do that).
#
# Cheap by construction: the top level and the ice are copied to the CPU (about 2 MB) once per
# interval and summarised there, a few ms against the tens of seconds a model day takes.
#
# The thresholds are deliberately loose. The top cell is 4.2 m, so freezing 0.1 m of ice adds
# ~0.8 g/kg to it before mixing, melting 0.5 m removes ~3.5 g/kg, and a front crossing a cell can
# move it ~1 g/kg in a day: a single cell over the limit is weather, many at once are not.
const HEALTH_S_RANGE   = (30.0, 36.0)   # g/kg, surface
const HEALTH_T_RANGE   = (-2.5, 25.0)   # degC, surface
const HEALTH_ΔS_CELL   = 1.0            # g/kg per check: a cell counts as "jumping" past this
const HEALTH_ΔT_CELL   = 2.0            # degC per check
const HEALTH_JUMPERS   = 100            # warn when more cells than this jump at once (of ~46,000 wet)
const HEALTH_DRIFT_S   = 0.25           # g/kg, |domain-mean surface S - its starting value|
const HEALTH_DRIFT_T   = 1.0            # degC
const HEALTH_ICE_SOUTH = -55.0          # warn on ice north of this latitude...
const HEALTH_ICE_ℵ     = 0.15           # ...at more than this concentration

# The child's own seafloor (after the nest blended it onto the parent's): stored as a Field or as
# its bare offset array depending on how the grid was built, and both index the interior as 1:N
health_wet = Array(child.grid.immersed_boundary.bottom_height[1:Nx, 1:Ny, 1]) .< 0
health_λ   = Array(λnodes(child.grid, Center()))
health_φ   = Array(φnodes(child.grid, Center()))
health_w   = [health_wet[i, j] * cosd(health_φ[j]) for i in 1:Nx, j in 1:Ny]   # area weights, 0 on land
health_north = [φ > HEALTH_ICE_SOUTH for _ in 1:Nx, φ in health_φ]
const HEALTH_STATE = Dict{Symbol, Any}()

surface_slice(field) = Array(view(interior(field), :, :, Nz:Nz))[:, :, 1]   # copies the top level only
ice_slice(field)     = Array(interior(field))[:, :, 1]
weighted_mean(X, w)  = sum(X .* w) / sum(w)
where_is(X, k)       = @sprintf("%.2f at %.1fE %.1fS", X[k], health_λ[k[1]], -health_φ[k[2]])

function health_check(sim)
    S, T = surface_slice(child.tracers.S), surface_slice(child.tracers.T)
    h, ℵ = ice_slice(sea_ice.model.ice_thickness), ice_slice(sea_ice.model.ice_concentration)
    day  = time(sim) / 86400
    S̄, T̄ = weighted_mean(S, health_w), weighted_mean(T, health_w)

    if isempty(HEALTH_STATE)
        HEALTH_STATE[:S₀], HEALTH_STATE[:T₀] = S̄, T̄
        HEALTH_STATE[:S], HEALTH_STATE[:T] = S, T
        return nothing
    end

    ΔS = abs.(S .- HEALTH_STATE[:S]) .* health_wet
    ΔT = abs.(T .- HEALTH_STATE[:T]) .* health_wet
    HEALTH_STATE[:S], HEALTH_STATE[:T] = S, T

    Sw = ifelse.(health_wet, S, NaN); Tw = ifelse.(health_wet, T, NaN)
    lo(X) = argmin(map(x -> isnan(x) ? Inf : x, X))     # CartesianIndex of the wet minimum
    hi(X) = argmax(map(x -> isnan(x) ? -Inf : x, X))
    kSmin, kSmax, kTmin, kTmax = lo(Sw), hi(Sw), lo(Tw), hi(Tw)
    hℵ = h .* ℵ
    kΔS, kΔT, kv = argmax(ΔS), argmax(ΔT), argmax(hℵ)
    nΔS, nΔT     = count(>(HEALTH_ΔS_CELL), ΔS), count(>(HEALTH_ΔT_CELL), ΔT)
    ice_area     = weighted_mean(ℵ, health_w)
    ice_volume   = weighted_mean(hℵ, health_w)                           # m of ice, domain mean
    ice_north    = sum(ℵ .* health_w .* health_north) / sum(health_w .* health_north)

    @info @sprintf("health day %.1f, Δt %.1f min | S %.2f-%.2f, mean %+.3f since start, max Δ %.2f (%d cells > %.1f) | T %.2f-%.2f, mean %+.2f, max Δ %.2f (%d > %.1f) | ice: area %.3f, volume %.3f m, max h·ℵ %s m, north of %.0fS %.3f",
                   day, sim.Δt / 60, S[kSmin], S[kSmax], S̄ - HEALTH_STATE[:S₀], ΔS[kΔS], nΔS, HEALTH_ΔS_CELL,
                   T[kTmin], T[kTmax], T̄ - HEALTH_STATE[:T₀], ΔT[kΔT], nΔT, HEALTH_ΔT_CELL,
                   ice_area, ice_volume, where_is(hℵ, kv), -HEALTH_ICE_SOUTH, ice_north)

    S[kSmin] < HEALTH_S_RANGE[1] && @warn "health: surface S below $(HEALTH_S_RANGE[1]) g/kg: $(where_is(S, kSmin))"
    S[kSmax] > HEALTH_S_RANGE[2] && @warn "health: surface S above $(HEALTH_S_RANGE[2]) g/kg: $(where_is(S, kSmax))"
    T[kTmin] < HEALTH_T_RANGE[1] && @warn "health: surface T below $(HEALTH_T_RANGE[1]) degC: $(where_is(T, kTmin))"
    T[kTmax] > HEALTH_T_RANGE[2] && @warn "health: surface T above $(HEALTH_T_RANGE[2]) degC: $(where_is(T, kTmax))"
    nΔS > HEALTH_JUMPERS && @warn "health: $nΔS surface cells changed S by > $(HEALTH_ΔS_CELL) g/kg since the last check; largest $(where_is(ΔS, kΔS))"
    nΔT > HEALTH_JUMPERS && @warn "health: $nΔT surface cells changed T by > $(HEALTH_ΔT_CELL) degC since the last check; largest $(where_is(ΔT, kΔT))"
    abs(S̄ - HEALTH_STATE[:S₀]) > HEALTH_DRIFT_S && @warn @sprintf("health: domain-mean surface S has drifted %+.3f g/kg since the start", S̄ - HEALTH_STATE[:S₀])
    abs(T̄ - HEALTH_STATE[:T₀]) > HEALTH_DRIFT_T && @warn @sprintf("health: domain-mean surface T has drifted %+.2f degC since the start", T̄ - HEALTH_STATE[:T₀])
    ice_north > HEALTH_ICE_ℵ && @warn @sprintf("health: mean ice concentration north of %.0fS is %.3f; there should be almost none", -HEALTH_ICE_SOUTH, ice_north)
    hℵ[kv] > 10 && @warn "health: most ice is $(where_is(hℵ, kv)) (m of h·ℵ); the run stops at $(ICE_MAX_VOLUME) m"
    return nothing
end
simulation.callbacks[:health] = Callback(health_check, TimeInterval(output_interval))

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
    "sea_ice" => "ClimaSeaIce via sea_ice_simulation defaults, IncrementalRemapping advection, free-slip coasts, " *
                 (ICE_INIT == "bsose" ? "started from B-SOSE SIarea/SIheff" : "started ice-free") * ", " *
                 (ICE_BOUNDARY == "bsose" ? "open-side bands relaxed to B-SOSE ice on $(ICE_RELAXATION / hours) h" :
                                            "open-side bands ice deleted"),
    "coastal_freshwater" => COASTAL_FRESHWATER ? "B-SOSE coastal excess from $(COASTAL_FRESHWATER_FILE)" : "none",
    "ice_grounding" => ICE_GROUNDING ? "LandfastBasalStress (NumericalEarth default)" : "off (basal_stress = nothing)",
    "penetrating_shortwave" => PENETRATING_SHORTWAVE ? "yes: TwoColorRadiation fed by the coupler" :
                                                       "no: all shortwave in the top cell's heat flux",
    "vertical_coordinate" => "static z, B-SOSE i156's 52 levels")

ocean.output_writers[:surface] = NetCDFWriter(
    child,
    (; u, v, T, S),
    filename="$(RUN_NAME)_surface.nc",
    schedule=TimeInterval(output_interval),
    indices=(:, :, Nz),
    global_attributes=output_metadata,
    overwrite_files=!PICKUP
)

mid_lon_idx = div(Nx, 2)
ocean.output_writers[:mid_lon] = NetCDFWriter(
    child,
    (; u, v, T, S),
    filename="$(RUN_NAME)_midlon.nc",
    schedule=TimeInterval(output_interval),
    indices=(mid_lon_idx, :, :),
    global_attributes=output_metadata,
    overwrite_files=!PICKUP
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
    overwrite_files=!PICKUP
)

# ── what the coupler hands the ocean at its surface, daily means ──────────────────────────────
# tools/mixing_energetics.jl's `supplied` is B-SOSE's own forcing; this run is forced by ERA5 bulk
# formulae and the ice instead, so without these there is no way to say whether OUR surface removes
# more buoyancy than B-SOSE's (fix5: the 60-55S column is 15% harder to mix than B-SOSE's, yet its
# mixed layer is 25-70 m deeper). Read off NumericalEarth's net ocean fluxes, the fields the ocean's
# top boundary conditions take, all kinematic and positive OUT of the ocean (Oceananigans' sign):
#
#   heat_flux        Jᵀ, °C m/s: turbulent + longwave + the shortwave absorbed AT the surface, over
#                    open water and under ice. Penetrating shortwave enters through a separate
#                    forcing and is not in it; `shortwave` below is the whole transmitted beam.
#   ice_salt_flux    Jˢ, g/kg m/s: the sea-ice salt flux ONLY.
#   freshwater_flux  Jʷ, m/s, + INTO the ocean: rain, snow, evaporation and ice melt/freeze. The
#                    salinity condition adds S_top·Jʷ live, so the salt flux the ocean sees is
#                    Jˢ + S_top·Jʷ, with S_top from RUN_surface.nc.
#   shortwave        W/m², NEGATIVE downward: shortwave transmitted into the ocean, (1 - ℵ)-weighted.
#
# The coastal freshwater (COASTAL_FRESHWATER) is a Forcing, not part of these; its field is the
# constant COASTAL_FRESHWATER_FILE. Daily MEANS, not snapshots: a midnight snapshot of a flux with
# a diurnal shortwave cycle is not a daily flux. Missing fields are warned about and skipped, so a
# NumericalEarth that names them differently loses this file, not the run.
function surface_flux_outputs()
    candidates = (
        (:heat_flux,       () -> coupled_model.interfaces.net_fluxes.ocean.T),
        (:ice_salt_flux,   () -> coupled_model.interfaces.net_fluxes.ocean.S),
        (:freshwater_flux, () -> coupled_model.interfaces.net_fluxes.ocean.η),
        (:shortwave,       () -> coupled_model.radiation.interface_fluxes.ocean.downwelling_shortwave))
    found = Pair{Symbol,Any}[]
    for (name, get_field) in candidates
        f = try get_field() catch err; err end
        if !(f isa Oceananigans.Fields.AbstractField)
            @warn "surface fluxes: no $name ($(f isa Exception ? sprint(showerror, f) : typeof(f))); not written"
        elseif size(f, 1) != Nx || size(f, 2) != Ny
            @warn "surface fluxes: $name is $(size(f)), not on the child's $(Nx)x$(Ny) grid; not written"
        else
            push!(found, name => f)
        end
    end
    @info "surface fluxes: writing $(isempty(found) ? "nothing" : join(first.(found), ", ")) as daily means"
    return NamedTuple(found)
end

surface_fluxes = surface_flux_outputs()
isempty(surface_fluxes) || (ocean.output_writers[:surface_fluxes] = NetCDFWriter(
    child,
    surface_fluxes,
    filename="$(RUN_NAME)_surface_fluxes.nc",
    schedule=AveragedTimeInterval(output_interval, window=output_interval),
    global_attributes=merge(output_metadata, Dict(
        "flux_convention" => "kinematic, positive OUT of the ocean, except freshwater_flux (+ into the ocean) and shortwave (W/m2, negative down)",
        "shortwave_in_heat_flux" => PENETRATING_SHORTWAVE ? "no" : "yes",
        "salt_flux_seen_by_ocean" => "ice_salt_flux + S_top * freshwater_flux, S_top from the surface file",
        "averaging_window_seconds" => Float64(output_interval))),
    overwrite_files=!PICKUP
))

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
    overwrite_files=!PICKUP
)

# Sea ice, daily: thickness, concentration, snow and the ice velocities
sea_ice_outputs = merge((; h = sea_ice.model.ice_thickness,
                           ℵ = sea_ice.model.ice_concentration,
                           hs = sea_ice.model.snow_thickness),
                        sea_ice.model.velocities)

sea_ice.output_writers[:sea_ice] = NetCDFWriter(
    sea_ice.model,
    sea_ice_outputs,
    filename="$(RUN_NAME)_sea_ice.nc",
    schedule=TimeInterval(output_interval),
    global_attributes=output_metadata,
    overwrite_files=!PICKUP
)

@info "output: $(RUN_NAME)_{surface,midlon,free_surface,surface_fluxes,eke,sea_ice}.nc in $(pwd())"

# ── checkpoints ───────────────────────────────────────────────────────────────────────────────
# Oceananigans' Checkpointer on the coupled simulation: the ocean child (the nest's parent is data,
# reread on restart), the ice, the atmosphere's clock, the interfaces, and every writer's and
# schedule's own state, so on pickup the NetCDF files are APPENDED to (overwrite_files = !PICKUP
# above) and the 30-day EKE window resumes where it was. Every CHECKPOINT_DAYS (365; 1 in smoke
# mode), and once more at the end, so a finished run can also be extended by raising END_DATE.
# Files: RUN_NAME_checkpoint_iteration<N>.jld2 in the working directory, all kept.
#
#     PICKUP=1 RUN_NAME=... END_DATE=... julia --project=. examples/sowinwarp_nested.jl
#
# On pickup, Δt restarts at 1 s and the wizard ramps it back up (a few hundred iterations), and the
# health check takes its drift baseline from the pickup time, not the original start.
const CHECKPOINT_DAYS = parse(Float64, get(ENV, "CHECKPOINT_DAYS", SMOKE ? "1" : "365"))
simulation.output_writers[:checkpointer] = Checkpointer(coupled_model;
                                                        schedule = TimeInterval(CHECKPOINT_DAYS * days),
                                                        prefix = "$(RUN_NAME)_checkpoint")

PICKUP && @info "picking up $RUN_NAME from its latest checkpoint"
@info "starting"
run!(simulation; pickup = PICKUP, checkpoint_at_end = true)
@info "done"
