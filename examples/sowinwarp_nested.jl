# ==============================================================================================
# sowinwarp_nested.jl -- SOWinWaRP as a one-way nest in B-SOSE, on NumericalEarth's
# ss/nested-ocean-model branch. Modelled on examples/bouvet_nested_ocean_simulation.jl.
#
#     SMOKE=1 julia --project=. examples/sowinwarp_nested.jl     # 2 days, proves it starts
#     julia --project=. examples/sowinwarp_nested.jl             # the real window
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
#   2. Z-STAR COMES FREE. `ExponentialDiscretization(...; mutable = true)` is a
#      MutableVerticalDiscretization, so the free surface moves the levels. Our current model is
#      static-z, which is exactly why freshwater enters as a VIRTUAL SALT FLUX with no volume:
#      B-SOSE adds water, eta rises, divergence exports the anomaly; we remove salt and the
#      anomaly can only leave as an advected tracer. Moving here changes that mechanism, so the
#      `nofw` testbed arm and this branch are asking the same question two different ways.
# ==============================================================================================

using CUDA            # loads Oceananigans' CUDA extension, which is what defines `GPU()`
using CopernicusClimateDataStore    # activates the ERA5 download backend (needs ~/.cdsapirc)
using NumericalEarth
using Oceananigans
using Oceananigans.Units
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
const Nz = 40
const DEPTH = 5500meters

# Must be a NATIVE B-SOSE stamp (records sit at odd times: 2014-05-02T02:00, 2014-06-01T12:00, ...).
# The initial snapshot goes through `bracketing_dates`, which looks `[start_date]` up in
# `all_dates` by exact match and crashes on `nothing` when it is not there. Workaround until
# that is fixed upstream on ss/nested-ocean-model.
start_date = DateTime(2014, 6, 1, 12)
end_date   = SMOKE ? DateTime(2014, 6, 3) : DateTime(2014, 8, 1)

@info "SOWinWaRP nested: $(Nx)x$(Ny)x$(Nz) over $(λ₁)-$(λ₂)E, $(φ₁)-$(φ₂)S, $start_date to $end_date"

z = ExponentialDiscretization(Nz, -DEPTH, 0; scale = DEPTH / 4, mutable = true)

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

ocean = nested_ocean_simulation(grid, dataset;
                                dates = (start_date, end_date),
                                dir = BDIR)

child = ocean.model.child

# ── atmosphere ────────────────────────────────────────────────────────────────────────────────
# As src/model.jl does it: same directory and region, so hours the main runs already fetched
# are read from cache. NEEDS CDS CREDENTIALS (~/.cdsapirc or .copernicus_env).
era5_dir = get(ENV, "ERA5_DIR", joinpath(BDIR, "era5"))
mkpath(era5_dir)
era5_region = BoundingBox(longitude = (λ₁ - 1.0, λ₂ + 1.0), latitude = (φ₁ - 1.0, φ₂ + 1.0))

# Serially first: the constructors download one thread per variable, which CDS rejects.
prefetch_era5!(; start_date, end_date, region = era5_region, dir = era5_dir, threads = 1)

atmosphere = ERA5PrescribedAtmosphere(ARCH; start_date, end_date, region = era5_region, dir = era5_dir)
radiation  = ERA5PrescribedRadiation(ARCH; start_date, end_date, region = era5_region, dir = era5_dir)

coupled_model = OceanOnlyModel(ocean; atmosphere, radiation)      # see note 1 in the header

simulation = Simulation(coupled_model; Δt = 5minutes,
                        stop_time = (end_date - start_date).value / 1000)

wall = Ref(time_ns())
function progress(sim)
    u, v, w = child.velocities
    T, S = child.tracers.T, child.tracers.S
    elapsed = 1e-9 * (time_ns() - wall[])
    @info @sprintf("t = %s | Δt = %s | |u| = %.2e %.2e %.2e | T %.2f..%.2f | S %.2f..%.2f | %.1f s",
                   prettytime(sim), prettytime(sim.Δt),
                   maximum(abs, u), maximum(abs, v), maximum(abs, w),
                   minimum(T), maximum(T), minimum(S), maximum(S), elapsed)
    wall[] = time_ns()
    return nothing
end
add_callback!(simulation, progress, TimeInterval(SMOKE ? 6hours : 1days))

# ── output ────────────────────────────────────────────────────────────────────────────────────
# Written off `child`, NOT off the NestedModel: the nest's prognostic state is the child.
outputs = merge(child.velocities, child.tracers)
ocean.output_writers[:surface] =
    JLD2Writer(child, outputs;
               schedule = TimeInterval(SMOKE ? 6hours : 1days),
               filename = "sowinwarp_nested_surface",
               indices = (:, :, Nz),
               overwrite_files = true)

ocean.output_writers[:free_surface] =
    JLD2Writer(child, (; η = child.free_surface.displacement);
               schedule = TimeInterval(SMOKE ? 6hours : 1days),
               filename = "sowinwarp_nested_free_surface",
               overwrite_files = true)

@info "starting"
run!(simulation)
@info "done"
