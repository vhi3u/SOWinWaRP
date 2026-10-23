# ==============================================================================
# model_simple.jl -- the simplest boundary configuration that can work.
#
#     julia --project=. src/model_simple.jl
#
# This is not a reduced copy of `model.jl`. It is a different experiment: one boundary
# condition, one forcing dataset, nothing switchable. Everything `model.jl` can turn on --
# BSOSE forcing, ERA5, sponges, penetrating shortwave, the BSOSE boundary path, the scheme
# comparison -- is absent here rather than set to `false`, because the point is to have no
# configuration left to blame.
#
#   BOUNDARIES   hard Dirichlet from GLORYS on u, v (normal AND tangential), T, S and η,
#                on all four faces. No scheme, no relaxation timescale, no sponge.
#   FORCING      none. No wind stress, no heat or salt flux, no shortwave, no atmosphere.
#   INTERIOR     initialised from GLORYS at `start_date`.
#   LENGTH       20 days.
#
# WHY THIS SHAPE. MITgcm's `obcs` prescribes every boundary field as a hard Dirichlet value and
# compiles out Orlanski, Stevens and the sponge entirely -- `useOBCSprescribe` plus
# `useOBCSbalance` plus `OBCSfixtopo`, and nothing else. That is what Ito & Timmermans' SODA-forced
# Gulf of Alaska setup runs, and it is decades-proven on domains like this one. Every scheme more
# elaborate than that has been tried here and none of them helped.
#
# THE ONE PIECE THAT IS NOT DIRICHLET, AND WHY IT CANNOT BE. Oceananigans' split-explicit free
# surface carries its own prognostic (U, V, η) substepped at √(gH) ≈ 198 m/s, and `target_transport`
# -- the net-transport pin, our `useOBCSbalance` -- exists only as a field of the Flather scheme.
# A scheme-free `NormalFlowBoundaryCondition` on U/V would throw the pin away. So U and V keep
# Flather, and η is prescribed instead, which collapses Flather to Dirichlet exactly:
#
#     Flather west fill:  ηᵇ = ℑxᶠᵃᵃ(1, j, ...) = (η[0,j] + η[1,j]) / 2
#     Value halo fill:    η[0,j] = 2v - η[1,j]        ⟹  ηᵇ = v identically
#
# Feeding the same GLORYS series as both the prescribed value and Flather's ηᵉˣᵗ makes the
# √(gH)(ηᵇ - ηᵉˣᵗ) term identically zero, so U = Uᵉˣᵗ, then rescaled to the balanced net. The
# result is MITgcm's `useOBCSprescribe` + `useOBCSbalance`, reached the only way this solver allows.
#
# WHAT THAT COSTS. A prescribed η has no radiating path for barotropic gravity waves. MITgcm does
# not need one -- its implicit free surface (deltaTfreesurf = deltaTmom = 3600 s) does not resolve
# them at all -- but ours does. A fully reflecting basin this size seiches at 2L/c ≈ 10.1 h zonal,
# 7.8 h meridional. Daily output aliases that completely, so this script logs the free surface
# every 30 minutes. If mean η develops a ~10 h oscillation that does not decay, that is the
# reflection, and a sponge on η is the answer. Until it shows up, there is no sponge here.
#
# WHAT TO LOOK AT FIRST. The boundary problem is an impulse at step one, not a slow instability:
# in `20D_glorys_pinned` the domain-wide boundary maximum went 0.94 → 3.54 m/s between day 0 and
# day 1, then decayed for nineteen days. So day 1 decides this run. If the boundary band is still
# near 1 m/s at day 1 instead of 3.5, it worked.
#
# Outputs follow the same conventions as `model.jl` so the existing tools read them unchanged:
#   RUN_NAME_surface.nc       surface u, v, T, S, daily
#   RUN_NAME_midlon.nc        mid-longitude transect, daily
#   RUN_NAME_free_surface.nc  η, daily
# ==============================================================================

using Pkg
Pkg.instantiate()

using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids
using NumericalEarth
using CopernicusMarine                     # activates the GLORYS download backend
using CUDA: has_cuda_gpu
using NCDatasets
using Dates
using Printf
using Statistics: mean
using Oceananigans.TurbulenceClosures
using Oceananigans.Operators: Ax_qᶠᶜᶜ, Ay_qᶜᶠᶜ
using Oceananigans.Diagnostics: NaNChecker
using Oceanostics.ProgressMessengers: TimedMessenger

# `setup_bsose.jl` is included for shared helpers only -- `default_bsose_directory` (which
# `glorys_directory` builds on, so the GLORYS cache lands on scratch rather than $HOME) and
# `fill_bathymetry_gaps!` with the physical T/S bounds. No BSOSE data is read by this script.
include("setup_bsose.jl")
include("glorys_obc.jl")

# ------------------------------------------------------------------------------
# Configuration -- the whole of it
# ------------------------------------------------------------------------------
const RUN_NAME = "20D_simple_dirichlet"

const start_date = DateTime("2014-06-01")
const end_date   = DateTime("2014-06-21")   # 20 days

const SCALING    = 6     # 1/6 degree
const DZ_SURFACE = 5     # m
const DZ_BOTTOM  = 200   # m
const N_MATCH    = 4     # cells over which to blend the bathymetry toward GLORYS at the edges

const GLORYS_DATASET = GLORYSMonthly()

arch = has_cuda_gpu() ? GPU() : CPU()
@info "Architecture: $arch"

# ------------------------------------------------------------------------------
# Grid
# ------------------------------------------------------------------------------
λ₁, λ₂ = (90, 150)
φ₁, φ₂ = (-70, -45)

z = ReferenceToStretchedDiscretization(; extent = 5800,
                                       constant_spacing = DZ_SURFACE,
                                       maximum_spacing = DZ_BOTTOM,
                                       constant_spacing_extent = 500,
                                       stretching = PowerLawStretching(1.12))

Nx = Int(SCALING * (λ₂ - λ₁))
Ny = Int(SCALING * (φ₂ - φ₁))
Nz = length(z)

grid = LatitudeLongitudeGrid(arch;
                             size = (Nx, Ny, Nz),
                             longitude = (λ₁, λ₂),
                             latitude  = (φ₁, φ₂),
                             z,
                             halo = (7, 7, 7))

bottom_height = regrid_bathymetry(grid;
                                  height_above_water = 1,
                                  minimum_depth = 10,
                                  interpolation_passes = 5)

# U = ∫u dz, so a depth mismatch between us and GLORYS at an open edge IS a transport error no
# matter how carefully Uᵉˣᵗ is built. This also applies MITgcm's `OBCSfixtopo`: the boundary column
# is never deeper than its interior neighbour. Must run before the immersed grid is built, since
# that is what freezes the wet column the barotropic transport is computed over.
match_boundary_bathymetry!(bottom_height, grid;
                           region = BoundingBox(longitude = (λ₁ - 1.0, λ₂ + 1.0),
                                                latitude  = (φ₁ - 1.0, φ₂ + 1.0)),
                           n_match = N_MATCH, minimum_depth = 10)

grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom_height))
@info grid

# ------------------------------------------------------------------------------
# Boundary conditions -- Dirichlet, everywhere, from GLORYS
# ------------------------------------------------------------------------------
simulation_seconds = Dates.value(Second(end_date - start_date))
@info @sprintf("Run: %s to %s (%.1f days)", start_date, end_date, simulation_seconds / 86400)

glorys_dates = glorys_boundary_dates(start_date, end_date; dataset = GLORYS_DATASET)
@info "GLORYS boundary records: $(length(glorys_dates)), $(first(glorys_dates)) to $(last(glorys_dates))"

glorys_obc = glorys_open_boundary_conditions(grid;
    dates         = glorys_dates,
    dataset       = GLORYS_DATASET,
    form          = :dirichlet,   # u, v, T, S: hard Dirichlet, no scheme
    prescribe_eta = true,         # η: GLORYS zos, which makes U/V Dirichlet too
    pin_transport = true,         # MITgcm's useOBCSbalance; without it η ramps forever
    uext_mode     = :native)

boundary_conditions = glorys_obc.boundary_conditions

# ------------------------------------------------------------------------------
# Closure: Ito et al. (2026), biharmonic Ah = 3e9 m⁴/s + CATKE
# ------------------------------------------------------------------------------
closures = (NumericalEarth.Oceans.default_ocean_closure(),
            HorizontalScalarBiharmonicDiffusivity(ν = 3e9, κ = 3e9),
            VerticalScalarDiffusivity(ν = 1e-5, κ = 1e-5))

@info "Constructing ocean simulation..."
ocean = ocean_simulation(grid;
                         boundary_conditions,
                         closure = closures)

# Initialise the interior from the same dataset that drives the boundaries. Mixing them puts a
# discontinuity on the boundary at t = 0 that the open conditions then have to argue with.
@info "Initialising from GLORYS at $start_date..."
glorys_initial_conditions!(ocean.model; date = start_date, dataset = GLORYS_DATASET,
                           region = glorys_obc.region, remove_eta_mean = false)
fill_bathymetry_gaps!(parent(ocean.model.tracers.S), S_MIN_PHYSICAL, S_MAX_PHYSICAL)
fill_bathymetry_gaps!(parent(ocean.model.tracers.T), T_MIN_PHYSICAL, T_MAX_PHYSICAL)

# ------------------------------------------------------------------------------
# Simulation
# ------------------------------------------------------------------------------
simulation = Simulation(ocean.model, Δt = 1seconds, stop_time = Float64(simulation_seconds))

simulation.callbacks[:wizard] =
    Callback(TimeStepWizard(cfl = 0.6, max_change = 1.1, min_Δt = 0.1), IterationInterval(10))

# Throw on the first NaN in any field rather than letting it reach the wizard, where it surfaces
# as `InexactError: Int64(NaN)` inside the free-surface substep and names neither field nor time.
simulation.callbacks[:nan_checker] =
    Callback(NaNChecker(fields = merge(ocean.model.velocities, ocean.model.tracers), erroring = true),
             IterationInterval(1))

const output_interval = 1days

simulation.callbacks[:progress] = Callback(TimedMessenger(), TimeInterval(output_interval))

# ------------------------------------------------------------------------------
# Output
# ------------------------------------------------------------------------------
u, v, w = ocean.model.velocities
T, S = ocean.model.tracers

output_metadata = Dict("run_name" => RUN_NAME,
                       "start_date" => string(start_date),
                       "end_date" => string(end_date),
                       "dataset" => "GLORYS",
                       "boundaries" => "Dirichlet (u, v, T, S, eta) + pinned transport")

simulation.output_writers[:surface] = NetCDFWriter(ocean.model, (; u, v, T, S);
    filename = "$(RUN_NAME)_surface.nc",
    schedule = TimeInterval(output_interval),
    indices = (:, :, grid.Nz),
    global_attributes = output_metadata,
    overwrite_files = true)

simulation.output_writers[:mid_lon] = NetCDFWriter(ocean.model, (; u, v, T, S);
    filename = "$(RUN_NAME)_midlon.nc",
    schedule = TimeInterval(output_interval),
    indices = (div(grid.Nx, 2), :, :),
    global_attributes = output_metadata,
    overwrite_files = true)

# η lives at (Center, Center, Face) on a single level, which NetCDFWriter cannot write directly;
# reducing over the length-one vertical gives an ordinary 2-D field.
η_2d = Field(Average(ocean.model.free_surface.displacement, dims = 3))

simulation.output_writers[:free_surface] = NetCDFWriter(ocean.model, (; η = η_2d);
    filename = "$(RUN_NAME)_free_surface.nc",
    schedule = TimeInterval(output_interval),
    global_attributes = output_metadata,
    overwrite_files = true)

# ------------------------------------------------------------------------------
# Diagnostics
# ------------------------------------------------------------------------------
u_transport = KernelFunctionOperation{Face,Center,Center}(Ax_qᶠᶜᶜ, grid, u)
v_transport = KernelFunctionOperation{Center,Face,Center}(Ay_qᶜᶠᶜ, grid, v)

west_transport  = Field(u_transport; indices = (1, :, :))
east_transport  = Field(u_transport; indices = (grid.Nx + 1, :, :))
south_transport = Field(v_transport; indices = (:, 1, :))
north_transport = Field(v_transport; indices = (:, grid.Ny + 1, :))

function log_transport(sim)
    Sv = 1e6
    west  = sum(compute!(west_transport))  / Sv
    east  = sum(compute!(east_transport))  / Sv
    south = sum(compute!(south_transport)) / Sv
    north = sum(compute!(north_transport)) / Sv

    # Outward-positive, so `net` is the volume leaving the domain per unit time.
    net   = -west + east - south + north
    gross = abs(west) + abs(east) + abs(south) + abs(north)

    @info @sprintf("        transport (Sv): W %+8.2f | E %+8.2f | S %+8.2f | N %+8.2f || net out %+8.3f (%.2f%% of gross)",
                   west, east, south, north, net, gross == 0 ? 0.0 : 100 * abs(net) / gross)
    flush(stdout)
end

simulation.callbacks[:transport] = Callback(log_transport, TimeInterval(output_interval))

# The reflection watch. A prescribed η cannot radiate barotropic gravity waves, so if this domain
# is going to ring it will do so at 2L/c ≈ 10.1 h zonal and 7.8 h meridional -- periods that daily
# output aliases into noise. Thirty minutes resolves them ~16x over. Read the mean, not the max:
# `maximum(abs, η)` includes immersed cells and is not a physical number.
#
# A steady or decaying mean η means the Dirichlet free surface is fine and no sponge is needed.
# A persistent oscillation with a ~8-10 h period is the reflection, and then η needs damping.
function log_free_surface(sim)
    η = sim.model.free_surface.displacement
    @info @sprintf("        [eta] t = %8.3f d   mean η = %+.5f m", sim.model.clock.time / 86400, mean(η))
    flush(stdout)
end

simulation.callbacks[:free_surface_watch] = Callback(log_free_surface, TimeInterval(30minutes))

# ------------------------------------------------------------------------------
@info "--- Setup complete ---"
@info "Grid: Nx=$(grid.Nx), Ny=$(grid.Ny), Nz=$(grid.Nz)"
@info "Boundaries: Dirichlet on u, v, T, S and η from GLORYS; transport pinned; no sponge"
@info "Forcing: none (no wind, no surface fluxes, no atmosphere)"
@info "Run name: $RUN_NAME"

run!(simulation)

@info "--- Done: $RUN_NAME ---"
