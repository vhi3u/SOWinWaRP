using Pkg
Pkg.instantiate()

using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids
using NumericalEarth
using NumericalEarth.ECCO
using CUDA: has_cuda_gpu, allowscalar
using NCDatasets
using Dates
using Printf
using Statistics: mean
using SeawaterPolynomials
using Oceananigans.TurbulenceClosures
using Oceananigans.BoundaryConditions: PerturbationAdvection, NormalRadiation
using Oceanostics.ProgressMessengers: TimedMessenger

# Include our BSOSE helper module
include("setup_bsose.jl")

import CUDA

if CUDA.functional()
    arch = GPU()
    @info "Running on GPU: $(CUDA.name(CUDA.device()))"
elseif haskey(ENV, "FORCE_CPU")
    arch = CPU()
    @warn "Running on CPU (forced by FORCE_CPU environment variable)"
else
    # Call functional(true) to output the exact reason why CUDA failed to initialize
    CUDA.functional(true)
    error("CUDA is not functional on this node! If you are running via sbatch, check your GPU allocation or logs. To force CPU testing, export FORCE_CPU=1.")
end

# flags

const OBCS = true # open boundary conditions (NormalFlow with PerturbationAdvection)
const SPONGE_LAYERS = true # sponge layer restoring (DatasetRestoring) near open boundary edges
const WINDS = false # time-varying surface wind forcing from BSOSE data (oceTAUX and oceTAUY)
const CHECKPOINTS = false # save state and restart if the model crashes. If false, the model will start from scratch. 

# domain related parameters
const SCALING = 6 # horizontal resolution = 1 / SCALING degrees. 1/6 = ~15km, 1/2 = 50km
const DZ_SURFACE = 2 # m 
const DZ_BOTTOM = 200 # m
const CIRCUMPOLAR = false # if true, we will use a circumpolar domain

const DATASET = "BSOSE" # by default, our forcings and BCs will be BSOSE, but if we want to test GLORYS or ECCO, we should be able to do that. 

# Domain setup

# horizontal domain extent
if CIRCUMPOLAR
    λ₁, λ₂ = (0, 360)
    φ₁, φ₂ = (-78, 30)
else
    λ₁, λ₂ = (90, 150)
    φ₁, φ₂ = (-70, -40)
end

# z stretching so that the upper 500 meters of the ocean has dz = 2 meters, and the deeper ocean will gradually stretch to 200 m vertical resolution. 
if arch isa CPU
    z = ReferenceToStretchedDiscretization(; extent=5800,
        constant_spacing=10,
        maximum_spacing=1000,
        constant_spacing_extent=500,
        stretching=PowerLawStretching(1.15))
else
    z = ReferenceToStretchedDiscretization(; extent=5800,
        constant_spacing=DZ_SURFACE,
        maximum_spacing=DZ_BOTTOM,
        constant_spacing_extent=500,
        stretching=PowerLawStretching(1.15))
end

# grid
# if running on CPU, do a 1 degree horizontal resolution run:
if arch isa CPU
    Nx = Int((λ₂ - λ₁))
    Ny = Int((φ₂ - φ₁))
else
    Nx = Int(SCALING * (λ₂ - λ₁))
    Ny = Int(SCALING * (φ₂ - φ₁))
end
Nz = length(z)

grid = LatitudeLongitudeGrid(arch;
    size=(Nx, Ny, Nz),
    latitude=(φ₁, φ₂),
    longitude=(λ₁, λ₂),
    z,
    halo=(7, 7, 7))

# bathymetry: automatically regrid from NumericalEarth
bottom_height = regrid_bathymetry(grid,
    height_above_water=1,
    minimum_depth=10,
    interpolation_passes=5)

grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom_height))

@info grid

# dataset conditonal

if DATASET == "BSOSE"
    dataset = BSOSEMonthly()
elseif DATASET == "GLORYS"
    dataset = GLORYSMonthly()
elseif DATASET == "ECCO"
    dataset = ECCO4Monthly()
else
    error("Unknown dataset: $DATASET. Choose 'BSOSE', 'GLORYS', or 'ECCO'.")
end

@info "Selected dataset: $(summary(dataset))"

# Simulation time window (1 year of forcing): local device has a different dataset than HPC GPU.
if arch isa CPU
    start_date = DateTime(2009, 1, 1)
    end_date = DateTime(2009, 12, 31)
else
    start_date = DateTime(2014, 1, 1)
    end_date = DateTime(2014, 12, 31)
end
dates = (start_date, end_date)

# ==============================================================================
# Boundary Conditions & Forcing
# ==============================================================================

if OBCS && DATASET == "BSOSE"
    @info "Configuring BSOSE Open Boundary Conditions with radiation scheme..."
    # In an open-ocean regional domain with strong outflow (ACC carrying ~130 Sv eastward through 150°E),
    # rigid Dirichlet boundaries reflect outgoing eddies and waves, leading to tracer blowup at the boundary.
    # PerturbationAdvection / NormalRadiation with outflow_timescale=Inf allows internal waves/eddies to freely
    # radiate out of the domain, while inflow_timescale=1days nudges incoming boundary flow (West) smoothly to BSOSE.
    obc_scheme_type = get(ENV, "OBC_SCHEME", "PerturbationAdvection")
    obc_scheme = if obc_scheme_type == "PerturbationAdvection"
        PerturbationAdvection(inflow_timescale = 1days, outflow_timescale = Inf)
    elseif obc_scheme_type == "NormalRadiation"
        NormalRadiation(inflow_timescale = 1days, outflow_timescale = Inf)
    elseif obc_scheme_type == "clamped" || obc_scheme_type == "none"
        nothing
    else
        error("Unknown OBC_SCHEME: $obc_scheme_type. Choose 'PerturbationAdvection', 'NormalRadiation', or 'clamped'.")
    end
    @info "  -> OBC Scheme: $(obc_scheme === nothing ? "Clamped Dirichlet" : summary(obc_scheme))"
    boundary_conditions = bsose_open_boundary_conditions(grid; dataset=dataset, dates=dates, winds=WINDS, scheme=obc_scheme)

    if SPONGE_LAYERS
        @info "Configuring boundary edge sponge layers (3.0° width, 5-day restoring timescale)..."
        forcings = bsose_sponge_layer_forcing(grid; dataset=dataset, dates=dates, sponge_width=3.0, timescale=5days, restore_velocities=true)
    else
        forcings = NamedTuple()
    end
elseif !OBCS
    @info "Configuring whole-region DatasetRestoring (30-day restoring for T and S from $DATASET)..."
    wind_bcs = (WINDS && DATASET == "BSOSE") ? bsose_surface_wind_stress(grid; dataset=dataset, dates=dates) : nothing
    boundary_conditions = wind_bcs !== nothing ? (u=FieldBoundaryConditions(top=wind_bcs.u), v=FieldBoundaryConditions(top=wind_bcs.v)) : NamedTuple()

    restoring_region = DATASET == "BSOSE" ? bsose_region(grid) : nothing
    temperature_metadata = Metadata(:temperature; dataset, start_date, end_date, region=restoring_region)
    salinity_metadata = Metadata(:salinity; dataset, start_date, end_date, region=restoring_region)

    FT = DatasetRestoring(temperature_metadata, grid; rate=1 / 30days)
    FS = DatasetRestoring(salinity_metadata, grid; rate=1 / 30days)

    forcings = (T=FT, S=FS)
else
    boundary_conditions = NamedTuple()
    forcings = NamedTuple()
end

# Turbulence closures:
# 1. Vertical: CATKE / default NumericalEarth vertical closure
vertical_closure = NumericalEarth.Oceans.default_ocean_closure()

# 2. Horizontal: Dissipate grid-scale enstrophy and stabilize boundary shear without
# over-damping physical mesoscale eddies. Biharmonic diffusivity (∇⁴) is standard for
# eddy-permitting resolutions (~1/6°). Harmonic diffusivity can also be selected.
horizontal_closure_type = get(ENV, "HORIZONTAL_CLOSURE", "biharmonic")
horizontal_closure = if horizontal_closure_type == "biharmonic"
    HorizontalScalarBiharmonicDiffusivity(ν=1e10, κ=1e10)
elseif horizontal_closure_type == "harmonic"
    HorizontalScalarDiffusivity(ν=100.0, κ=100.0)
elseif horizontal_closure_type == "none"
    nothing
else
    error("Unknown HORIZONTAL_CLOSURE: $horizontal_closure_type. Choose 'biharmonic', 'harmonic', or 'none'.")
end

closures = horizontal_closure !== nothing ? (vertical_closure, horizontal_closure) : vertical_closure
@info "Turbulence closures: $(summary(closures))"


# build the ocean model

@info "Constructing ocean simulation model..."
ocean = ocean_simulation(grid;
    boundary_conditions=boundary_conditions,
    forcing=forcings,
    closure=closures
)

# initial conditions based on either BSOSE or the other datasets

if DATASET == "BSOSE"
    bsose_initial_conditions!(ocean.model; dataset=dataset, date=start_date, velocities=true)
else
    set!(ocean.model,
        MetadataSet(:temperature; dataset=dataset, date=start_date),
        MetadataSet(:salinity; dataset=dataset, date=start_date)
    )
    fill_bathymetry_gaps!(parent(ocean.model.tracers.S), S_MIN_PHYSICAL, S_MAX_PHYSICAL)
    fill_bathymetry_gaps!(parent(ocean.model.tracers.T), T_MIN_PHYSICAL, T_MAX_PHYSICAL)
    fill_halo_regions!(ocean.model.tracers.T, ocean.model.clock, fields(ocean.model))
    fill_halo_regions!(ocean.model.tracers.S, ocean.model.clock, fields(ocean.model))
    fill_bathymetry_gaps!(parent(ocean.model.tracers.S), S_MIN_PHYSICAL, S_MAX_PHYSICAL)
    fill_bathymetry_gaps!(parent(ocean.model.tracers.T), T_MIN_PHYSICAL, T_MAX_PHYSICAL)
end

# ── TEOS10 sqrt guard ─────────────────────────────────────────────────────────
# The TEOS10 coordinate s(Sᴬ) = √((Sᴬ + ΔS) / Sₐᵤ) crashes for Sᴬ < -32.
# Guard with max(..., 0) so that any residual bad values from immersed/halo cells
# don't crash the run — they will return zero (freshwater) density rather than NaN.
import SeawaterPolynomials.TEOS10 as TEOS10_mod
@eval SeawaterPolynomials.TEOS10 begin
    @inline s(Sᴬ::FT) where FT = √(max((Sᴬ + FT(ΔS)) / FT(Sₐᵤ), zero(FT)))
end

# some housekeeping to set up the simulation (progress checks and adaptive timestep)

stop_time = haskey(ENV, "STOP_TIME") ? parse(Float64, ENV["STOP_TIME"]) : 365days
stop_iteration = haskey(ENV, "STOP_ITERATION") ? parse(Int, ENV["STOP_ITERATION"]) : Inf
# Start with a very small Δt so the first CATKE evaluation is numerically gentle;
# the wizard will ramp this up within a few iterations.
simulation = Simulation(ocean.model, Δt=1seconds, stop_time=stop_time, stop_iteration=stop_iteration)

# adaptive timestep wizard based on CFL (following mediterranean.jl: cfl=0.2, max_change=1.1)
wizard = TimeStepWizard(cfl=0.4, max_change=1.3, min_Δt=0.1)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

# Safety callback: clamp salinity/temperature to physically valid ranges before each time step.
# Operates on parent() to cover all cells including halos and immersed cells.
# Prevents DomainError in TEOS10's sqrt(S) from any numerical noise that accumulates
# in boundary-adjacent or immersed cells during time stepping.
# function clamp_salinity!(sim)
#     clamp!(parent(sim.model.tracers.S), S_MIN_PHYSICAL, S_MAX_PHYSICAL)
#     clamp!(parent(sim.model.tracers.T), T_MIN_PHYSICAL, T_MAX_PHYSICAL)
# end
# simulation.callbacks[:clamp_S] = Callback(clamp_salinity!, IterationInterval(1))

# progress logger: Oceanostics TimedMessenger reports wall-clock timing,
# max velocities and CFL/diffusive stability numbers each interval.
callback_interval = haskey(ENV, "CALLBACK_INTERVAL") ? parse(Float64, ENV["CALLBACK_INTERVAL"]) : 1hours
progress = TimedMessenger()
simulation.callbacks[:progress] = Callback(progress, TimeInterval(callback_interval))

# With Dirichlet open boundaries there is no wave radiation, so any net
# volume-flux imbalance across the four faces accumulates in the free surface.
# TimedMessenger does not report it, so track it separately.
function log_free_surface(sim)
    eta = sim.model.free_surface.displacement
    @info @sprintf("        max|eta| = %.4f m, mean(eta) = %+.4f m",
        maximum(abs, eta), mean(eta))
    flush(stdout)
end
simulation.callbacks[:free_surface] = Callback(log_free_surface, TimeInterval(callback_interval))

# output writers

u, v, w = ocean.model.velocities
T, S = ocean.model.tracers
simulation.output_writers[:surface] = NetCDFWriter(
    ocean.model,
    (; u, v, T, S),
    filename="model_surface_fields.nc",
    schedule=TimeInterval(1days),
    indices=(:, :, grid.Nz),
    overwrite_existing=true
)

mid_lon_idx = div(grid.Nx, 2)
simulation.output_writers[:mid_lon] = NetCDFWriter(
    ocean.model,
    (; u, v, T, S),
    filename="model_mid_lon.nc",
    schedule=TimeInterval(1days),
    indices=(mid_lon_idx, :, :),
    overwrite_existing=true
)

# checkpointing

if CHECKPOINTS
    simulation.output_writers[:checkpointer] = Checkpointer(
        ocean.model,
        schedule=TimeInterval(30days),
        prefix="checkpoint",
        overwrite_existing=true
    )
end

@info "--- Model Setup Complete ---"
@info "Grid Resolution: Nx=$(grid.Nx), Ny=$(grid.Ny), Nz=$(grid.Nz)"
@info "OBCS: $OBCS | WINDS: $WINDS | DATASET: $DATASET"

# red button 

@info "Starting simulation..."
run!(simulation)
@info "Simulation completed successfully!"
