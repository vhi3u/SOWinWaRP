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
using SeawaterPolynomials
using SeawaterPolynomials.TEOS10: TEOS10EquationOfState
using Oceananigans.TurbulenceClosures

# Include our BSOSE helper module
include("setup_bsose.jl")

if has_cuda_gpu()
    arch = GPU()
else
    arch = CPU()
end

# flags

const OBCS = true # open boundary conditions; if false, we will use sponge layers instead? 
const WINDS = false # time-varying surface wind forcing from BSOSE data (oceTAUX and oceTAUY)
const TEOS = true # use TEOS because the nonlinearity from the full equation of state allows for AAIW formation ; if false uses linear equation of state
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
z = ReferenceToStretchedDiscretization(; extent=5000,
    constant_spacing=DZ_SURFACE,
    maximum_spacing=DZ_BOTTOM,
    constant_spacing_extent=500,
    stretching=PowerLawStretching(1.15))

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
    interpolation_passes=25)

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
    @info "Configuring BSOSE Open Boundary Conditions..."
    boundary_conditions = bsose_open_boundary_conditions(grid; dataset=dataset, dates=dates, winds=WINDS)
    forcings = NamedTuple()
elseif !OBCS && DATASET == "BSOSE"
    @info "Configuring BSOSE Sponge Layer Relaxation..."
    wind_bcs = WINDS ? bsose_surface_wind_stress(grid; dataset=dataset, dates=dates) : nothing
    boundary_conditions = WINDS ? (u=FieldBoundaryConditions(top=wind_bcs.u), v=FieldBoundaryConditions(top=wind_bcs.v)) : NamedTuple()
    forcings = bsose_sponge_forcing(grid; dataset=dataset, dates=dates, rate=1 / 30days, sponge_distance=2.0)
else
    boundary_conditions = NamedTuple()
    forcings = NamedTuple()
end

# equation of state: choosing between TEOS or linear equation of state
equation_of_state = TEOS ? TEOS10EquationOfState() : LinearEquationOfState()

vertical_closure = NumericalEarth.Oceans.default_ocean_closure()
# Note: IsopycnalSkewSymmetricDiffusivity does not support ImmersedBoundaryGrid and causes
# division by zero / slope blowup across immersed topographic cells.
# NumericalEarth's default CATKE closure combined with WENO advection provides stable mixing.
closures = vertical_closure

# build the ocean model

@info "Constructing ocean simulation model..."
ocean = ocean_simulation(grid;
    boundary_conditions=boundary_conditions,
    forcing=forcings,
    closure=closures,
    equation_of_state=equation_of_state
)

# initial conditions based on either BSOSE or the other datasets

if DATASET == "BSOSE"
    bsose_initial_conditions!(ocean.model; dataset=dataset, date=start_date)
else
    set!(ocean.model,
        MetadataSet(:temperature; dataset=dataset, date=start_date),
        MetadataSet(:salinity; dataset=dataset, date=start_date)
    )
end

# some housekeeping to set up the simulation (progress checks and adaptive timestep)

stop_time = haskey(ENV, "STOP_TIME") ? parse(Float64, ENV["STOP_TIME"]) : 365days
stop_iteration = haskey(ENV, "STOP_ITERATION") ? parse(Int, ENV["STOP_ITERATION"]) : Inf
simulation = Simulation(ocean.model, Δt=2minutes, stop_time=stop_time, stop_iteration=stop_iteration)

# adaptive timestep wizard based on CFL
wizard = TimeStepWizard(cfl=0.7, max_Δt=1hours)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

# progress logger
function progress(sim)
    u, v, w = sim.model.velocities
    T, S = sim.model.tracers
    @info @sprintf("Time: %s, Iter: %d, Δt: %s, max(|u|,|v|,|w|): (%.2e, %.2e, %.2e), T range: (%.2f, %.2f), S range: (%.2f, %.2f)",
        prettytime(sim.model.clock.time),
        sim.model.clock.iteration,
        prettytime(sim.Δt),
        maximum(abs, u), maximum(abs, v), maximum(abs, w),
        minimum(T), maximum(T),
        minimum(S), maximum(S))
end
simulation.callbacks[:progress] = Callback(progress, TimeInterval(1hours))

# output writers

u, v, w = ocean.model.velocities
T, S = ocean.model.tracers

simulation.output_writers[:surface] = NetCDFWriter(
    ocean.model,
    (; u, v, T, S),
    filename="bsose_surface_fields.nc",
    schedule=TimeInterval(1days),
    indices=(:, :, grid.Nz),
    overwrite_existing=true
)

mid_lon_idx = div(grid.Nx, 2)
simulation.output_writers[:mid_lon] = NetCDFWriter(
    ocean.model,
    (; u, v, T, S),
    filename="bsose_mid_lon.nc",
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
@info "OBCS: $OBCS | WINDS: $WINDS | TEOS: $TEOS | DATASET: $DATASET"

# red button 

@info "Starting simulation..."
run!(simulation)
@info "Simulation completed successfully!"
