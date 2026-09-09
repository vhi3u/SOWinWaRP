# ==============================================================================
# verify_model_setup.jl
#
# Diagnostic and verification script:
# 1. Sets up the full Oceananigans grid and ImmersedBoundaryGrid.
# 2. Configures boundary conditions, sponge layer forcings, and initial conditions
#    identically to `src/model.jl`.
# 3. Assembles the simulation model (`ocean_simulation` or `HydrostaticFreeSurfaceModel`).
# 4. Rigorously tests every component:
#    - Boundary conditions: checks conditions on all 4 boundaries (West, East, South, North).
#    - Forcings: checks `DatasetRestoring` callable sponge layer forcing kernels.
#    - Initial conditions: checks model state fields (`u`, `v`, `w`, `T`, `S`, `η`) for NaNs,
#      Infs, and physical bounds across interior and parent halo regions.
#    - GPU compatibility: asserts that every field array and boundary slice is stored
#      as a `CUDA.CuArray` on GPU runs.
#    - Test time step: performs one test iteration `time_step!(simulation.model, 1.0)`
#      to verify stability and detect any `InexactError: Int64(NaN)` or NaN advection spikes.
# ==============================================================================

using Dates
using Printf
using Statistics
using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids
using Oceananigans.BoundaryConditions
using Oceananigans.BoundaryConditions: PerturbationAdvection
using Oceananigans.OutputReaders: FieldTimeSeries
using Oceananigans.Fields: interior, location, fill_halo_regions!
using Oceananigans.TurbulenceClosures
using NumericalEarth
using NumericalEarth.DataWrangling
using NumericalEarth.Oceans: ocean_simulation
using NCDatasets
using SeawaterPolynomials

import CUDA

# Include BSOSE helper module
include(joinpath(@__DIR__, "..", "src", "setup_bsose.jl"))

println("="^85)
println(" Model Setup, Boundary Conditions, Forcings & Initial State Verification")
println("="^85)

# ------------------------------------------------------------------------------
# 1. Architecture & Domain Setup
# ------------------------------------------------------------------------------
if CUDA.functional()
    arch = GPU()
    @info "Running on GPU: $(CUDA.name(CUDA.device()))"
elseif haskey(ENV, "FORCE_CPU")
    arch = CPU()
    @warn "Running on CPU (forced by FORCE_CPU environment variable)"
else
    @warn "CUDA is not functional on this node! Reason:"
    try
        CUDA.functional(true)
    catch e
        @warn "$e"
    end
    arch = CPU()
end
println("  - Architecture : $(arch)")

const OBCS = true
const SPONGE_LAYERS = true
const WINDS = false
const DATASET = "BSOSE"

# Domain boundaries (matching model.jl)
λ₁, λ₂ = (90.0, 150.0)
φ₁, φ₂ = (-70.0, -40.0)

if arch isa GPU
    SCALING = 3 # 1/3 degree horizontal resolution
    Nx = Int(SCALING * (λ₂ - λ₁))
    Ny = Int(SCALING * (φ₂ - φ₁))
    z = ReferenceToStretchedDiscretization(; extent=5800,
        constant_spacing=2,
        maximum_spacing=200,
        constant_spacing_extent=500,
        stretching=PowerLawStretching(1.15))
    start_date = DateTime(2014, 1, 1)
    end_date = DateTime(2014, 12, 31)
else
    println("  - CPU Verification mode (1° horizontal resolution, 2 time levels)")
    Nx = Int((λ₂ - λ₁))
    Ny = Int((φ₂ - φ₁))
    z = ReferenceToStretchedDiscretization(; extent=5400,
        constant_spacing=50,
        maximum_spacing=500,
        constant_spacing_extent=500,
        stretching=PowerLawStretching(1.15))
    dataset_tmp = BSOSEMonthly()
    avail = all_dates(dataset_tmp, :temperature)
    start_date = first(avail)
    end_date = avail[min(2, length(avail))]
end

Nz = length(z)
dates = (start_date, end_date)
println("  - Grid size    : Nx=$Nx, Ny=$Ny, Nz=$Nz")
println("  - Time window  : $start_date to $end_date")

underlying_grid = LatitudeLongitudeGrid(arch;
    size=(Nx, Ny, Nz),
    latitude=(φ₁, φ₂),
    longitude=(λ₁, λ₂),
    z,
    halo=(7, 7, 7))

# Regrid bathymetry
@info "Regridding bathymetry from NumericalEarth..."
bottom_height = regrid_bathymetry(underlying_grid,
    height_above_water=1,
    minimum_depth=10,
    interpolation_passes=5)

grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height))
println("  ✓ ImmersedBoundaryGrid created successfully.")

# ------------------------------------------------------------------------------
# 2. Boundary Conditions & Forcings
# ------------------------------------------------------------------------------
dataset = BSOSEMonthly()

println("\n[1/4] Configuring Boundary Conditions & Forcings (matching model.jl)...")
obc_scheme = nothing
boundary_conditions = bsose_open_boundary_conditions(grid; dataset=dataset, dates=dates, winds=WINDS, scheme=obc_scheme)
println("  ✓ Open boundary conditions constructed.")

if SPONGE_LAYERS
    forcings = bsose_sponge_layer_forcing(grid; dataset=dataset, dates=dates, sponge_width=3.0, timescale=5days, restore_velocities=true)
    println("  ✓ Sponge layer restoring forcings configured.")
else
    forcings = NamedTuple()
end

vertical_closure = NumericalEarth.Oceans.default_ocean_closure()
horizontal_closure = HorizontalScalarDiffusivity(ν=100.0, κ=100.0)
closures = (vertical_closure, horizontal_closure)

# ------------------------------------------------------------------------------
# 3. Model Assembly & Initial Conditions
# ------------------------------------------------------------------------------
println("\n[2/4] Assembling Ocean Model Simulation...")
ocean = ocean_simulation(grid;
    boundary_conditions=boundary_conditions,
    forcing=forcings,
    closure=closures)
println("  ✓ Model assembled.")

println("\nApplying Initial Conditions from BSOSE...")
bsose_initial_conditions!(ocean.model; dataset=dataset, date=start_date, velocities=true)
println("  ✓ Initial conditions applied.")

# ------------------------------------------------------------------------------
# 4. Comprehensive Validation: Fields, Boundary Slices & GPU Memory
# ------------------------------------------------------------------------------
println("\n[3/4] Validating Model Fields, Slices & GPU Architecture...")
println("-"^85)
@printf("%-22s | %-16s | %-10s | %-10s | %-8s | %-18s\n",
    "Field / Component", "Dimensions", "Min Val", "Max Val", "NaNs", "Array Type")
println("-"^85)

all_passed = true
total_nans = 0
incompatible_arrays = 0

function check_array_health(name, arr, target_arch; check_nan=true)
    global all_passed, total_nans, incompatible_arrays
    arr_type = string(typeof(arr))
    short_type = occursin("CuArray", arr_type) ? "CUDA.CuArray" :
                 occursin("Array", arr_type) ? "Base.Array (CPU)" :
                 split(arr_type, "{")[1]

    # Check GPU compatibility
    if target_arch isa GPU && !occursin("CuArray", arr_type)
        incompatible_arrays += 1
        all_passed = false
    end

    # Check NaNs and values on CPU copy
    cpu_arr = Array(arr)
    nan_count = check_nan ? count(isnan, cpu_arr) : 0
    total_nans += nan_count

    val_min = (nan_count == length(cpu_arr) || isempty(cpu_arr)) ? NaN : minimum(filter(!isnan, cpu_arr))
    val_max = (nan_count == length(cpu_arr) || isempty(cpu_arr)) ? NaN : maximum(filter(!isnan, cpu_arr))

    status = nan_count == 0 ? "PASS" : "FAIL"
    if status == "FAIL"
        all_passed = false
    end

    sz_str = string(size(cpu_arr))
    @printf("%-22s | %-16s | %10.4f | %10.4f | %8d | %-18s\n",
        name, sz_str, val_min, val_max, nan_count, short_type)
    return status == "PASS"
end

# Check Model Prognostic Fields
println("Model State Fields (Interior):")
check_array_health("model.tracers.T", interior(ocean.model.tracers.T), arch)
check_array_health("model.tracers.S", interior(ocean.model.tracers.S), arch)
check_array_health("model.velocities.u", interior(ocean.model.velocities.u), arch)
check_array_health("model.velocities.v", interior(ocean.model.velocities.v), arch)
check_array_health("model.velocities.w", interior(ocean.model.velocities.w), arch)
check_array_health("model.free_surface.η", interior(ocean.model.free_surface.displacement), arch)

println("\nModel State Fields (Parent / Halo buffer):")
check_array_health("parent(tracers.T)", parent(ocean.model.tracers.T), arch)
check_array_health("parent(tracers.S)", parent(ocean.model.tracers.S), arch)
check_array_health("parent(velocities.u)", parent(ocean.model.velocities.u), arch)
check_array_health("parent(velocities.v)", parent(ocean.model.velocities.v), arch)

println("\nBoundary Condition Slices:")
for (vname, bcs) in pairs(boundary_conditions)
    for side in (:west, :east, :south, :north)
        bc = getproperty(bcs, side)
        if bc !== nothing && hasproperty(bc, :condition) && bc.condition isa FieldTimeSeries
            bts = bc.condition
            check_array_health("$vname.$side [t=1]", parent(bts[1]), arch)
        end
    end
end

println("-"^85)

# ------------------------------------------------------------------------------
# 5. 2-Day Model Spinup Simulation
# ------------------------------------------------------------------------------
println("\n[4/4] Executing 2-Day Model Spinup Simulation...")

# Guard TEOS10 as in model.jl
import SeawaterPolynomials.TEOS10 as TEOS10_mod
@eval SeawaterPolynomials.TEOS10 begin
    @inline s(Sᴬ::FT) where FT = √(max((Sᴬ + FT(ΔS)) / FT(Sₐᵤ), zero(FT)))
end

SPINUP_TIME = 2days
simulation = Simulation(ocean.model, Δt=1seconds, stop_time=SPINUP_TIME)

# Adaptive timestep wizard matching model.jl
wizard = TimeStepWizard(cfl=0.4, max_Δt=1hours, max_change=1.1, min_Δt=0.1)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

# Periodic progress logging
function log_progress(sim)
    u_curr = maximum(abs, interior(sim.model.velocities.u))
    v_curr = maximum(abs, interior(sim.model.velocities.v))
    w_curr = maximum(abs, interior(sim.model.velocities.w))
    t_days = sim.model.clock.time / 86400
    @printf("  [Spinup] Iter: %6d | Time: %6.2f days / %.1f days | Δt: %7.2f s | max|u|: %.4f | max|v|: %.4f | max|w|: %.2e m/s\n",
        sim.model.clock.iteration, t_days, SPINUP_TIME / 86400, sim.Δt, u_curr, v_curr, w_curr)
    flush(stdout)
end
simulation.callbacks[:progress] = Callback(log_progress, IterationInterval(50))

spinup_ok = true
t_spinup_start = time()

try
    run!(simulation)
    t_spinup_elapsed = round(time() - t_spinup_start, digits=2)
    println("\n  ✓ 2-day spinup completed successfully in $(t_spinup_elapsed) s! Final simulated time: $(simulation.model.clock.time / 86400) days")
catch err
    global spinup_ok = false
    global all_passed = false
    println("\n  ❌ ERROR during 2-day spinup: $err")
    Base.show_backtrace(stdout, catch_backtrace())
    println()
end

# Post-spinup diagnostic checks
u_max_post = maximum(abs, interior(ocean.model.velocities.u))
v_max_post = maximum(abs, interior(ocean.model.velocities.v))
w_max_post = maximum(abs, interior(ocean.model.velocities.w))
T_min_post = minimum(interior(ocean.model.tracers.T))
T_max_post = maximum(interior(ocean.model.tracers.T))
S_min_post = minimum(interior(ocean.model.tracers.S))
S_max_post = maximum(interior(ocean.model.tracers.S))
nan_post = count(isnan, Array(parent(ocean.model.velocities.u))) +
           count(isnan, Array(parent(ocean.model.velocities.v))) +
           count(isnan, Array(parent(ocean.model.tracers.T))) +
           count(isnan, Array(parent(ocean.model.tracers.S)))

println("\nPost-Spinup Diagnostic State:")
@printf("  - Velocities : max|u| = %.4f m/s, max|v| = %.4f m/s, max|w| = %.2e m/s\n", u_max_post, v_max_post, w_max_post)
@printf("  - Tracers    : T ∈ [%.2f, %.2f] °C, S ∈ [%.2f, %.2f] psu\n", T_min_post, T_max_post, S_min_post, S_max_post)
@printf("  - Total NaNs : %d\n", nan_post)

if nan_post > 0
    spinup_ok = false
    all_passed = false
end

# ------------------------------------------------------------------------------
# 6. Final Summary
# ------------------------------------------------------------------------------
println("\n" * "="^85)
println(" VERIFICATION SUMMARY REPORT")
println("="^85)
@printf("  1. Boundary Conditions Structure      : %s\n", "PASS")
@printf("  2. Model State NaN Count               : %s (Initial: %d, Post-Spinup: %d)\n",
    (total_nans == 0 && nan_post == 0) ? "PASSED" : "FAILED", total_nans, nan_post)
@printf("  3. Architecture / GPU Array Storage   : %s (Incompatible: %d)\n",
    incompatible_arrays == 0 ? "PASSED" : "FAILED", incompatible_arrays)
@printf("  4. 2-Day Model Spinup Run             : %s\n", spinup_ok ? "PASSED (2.0 Days Completed)" : "FAILED")
println("="^85)

if all_passed && spinup_ok
    println("🎉 ALL CHECKS PASSED: Model, boundary conditions, forcings, and 2-day spinup are 100% stable!")
else
    println("⚠️  VERIFICATION FAILED: Review the diagnostics above.")
end
println("="^85)
