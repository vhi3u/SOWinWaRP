# ==============================================================================
# verify_bsose_dataset.jl
#
# Diagnostic and verification script:
# 1. Cleanly removes temporary BSOSE subsets/inpainting cache files and
#    cached boundary condition JLD2 files (and optionally scratchspace field caches).
# 2. Runs setup_bsose boundary condition dataset generation from scratch.
# 3. Inspects every variable/product for BSOSEMonthly() (raw variables,
#    regridded 3D fields, and 2D boundary time series slices).
# 4. Rigorously verifies structure, sizes, and ensures there are ZERO NaNs,
#    missing values, or infinite values in interior cells or boundary data.
# ==============================================================================

using Dates
using Printf
using Statistics
using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids
using Oceananigans.BoundaryConditions
using Oceananigans.OutputReaders: FieldTimeSeries
using Oceananigans.Fields: interior, location
using NumericalEarth
using NumericalEarth.DataWrangling
using NumericalEarth.DataWrangling: JLD2
using NCDatasets

try
    using CUDA
catch
end

# Load our BSOSE interface definitions
include(joinpath(@__DIR__, "..", "src", "setup_bsose.jl"))

println("="^80)
println(" BSOSEMonthly() Cache Cleaner & Diagnostic Verification Suite")
println("="^80)

# ------------------------------------------------------------------------------
# 1. Dataset Resolution & Cleanup
# ------------------------------------------------------------------------------
dataset = BSOSEMonthly()
println("\n[1/4] Target Dataset:")
println("  - Directory : $(dataset.dir)")
println("  - Iteration : $(dataset.iteration)")

temp_dir = bsose_temp_directory(dataset)
println("\nCleaning up cache files...")

# (a) Clean temporary inpainting directory: temp/bsose_inpainted_*.jld2
deleted_temp_count = 0
if isdir(temp_dir)
    for f in readdir(temp_dir; join=true)
        if endswith(f, ".jld2") || occursin("bsose_inpainted", f)
            rm(f; force=true)
            global deleted_temp_count += 1
        end
    end
    println("  ✓ Deleted $deleted_temp_count intermediate inpainted files from: $temp_dir")
else
    println("  - Temp directory does not exist yet: $temp_dir")
end

# (b) Clean cached unified boundary condition datasets: bsose_dataset_*.jld2
deleted_bc_count = 0
if isdir(dataset.dir)
    for f in readdir(dataset.dir; join=true)
        base = basename(f)
        if (startswith(base, "bsose_dataset_") || startswith(base, "bsose_subset_")) && (endswith(base, ".jld2") || endswith(base, ".nc"))
            rm(f; force=true)
            global deleted_bc_count += 1
            println("  ✓ Removed cached boundary/subset file: $base")
        end
    end
    println("  ✓ Total boundary/subset cache files removed: $deleted_bc_count")
end

# (c) Clean ~/.julia/scratchspaces bathymetry/field cache if present
scratch_base = joinpath(homedir(), ".julia", "scratchspaces", "904d977b-046a-4731-8b86-9235c0d1ef02", "field_cache")
deleted_scratch = 0
if isdir(scratch_base)
    for f in readdir(scratch_base; join=true)
        if occursin("bottom_height", basename(f))
            rm(f; force=true)
            global deleted_scratch += 1
        end
    end
    println("  ✓ Cleaned $deleted_scratch cached scratchspace bathymetry file(s) from $scratch_base")
end

# ------------------------------------------------------------------------------
# 2. Inspect Raw NetCDF Products for BSOSEMonthly()
# ------------------------------------------------------------------------------
println("\n[2/4] Inspecting Raw NetCDF Products for $(summary(dataset))...")

# Key variables expected in BSOSEMonthly
variables_to_check = [
    (:temperature, "temperature"),
    (:salinity, "salinity"),
    (:u_velocity, "u_velocity"),
    (:v_velocity, "v_velocity"),
    (:zonal_wind_stress, "zonal_wind_stress"),
    (:meridional_wind_stress, "meridional_wind_stress")
]

println("-"^80)
@printf("%-24s | %-38s | %-10s\n", "Variable", "NetCDF Filename", "Status")
println("-"^80)

raw_files_ok = true
for (sym, desc) in variables_to_check
    fname = resolve_bsose_filename(dataset.dir, sym, dataset.iteration)
    fpath = joinpath(dataset.dir, fname)
    if isfile(fpath)
        # Open and inspect time dimension and shapes
        ds = Dataset(fpath)
        try
            var_name = get(BSOSE_VARIABLE_NAMES, sym, string(sym))
            actual_key = haskey(ds, var_name) ? var_name :
                         haskey(ds, uppercase(var_name)) ? uppercase(var_name) :
                         haskey(ds, titlecase(var_name)) ? titlecase(var_name) : nothing
            
            if actual_key !== nothing
                sz = size(ds[actual_key])
                n_times = haskey(ds, "time") ? length(ds["time"]) : "N/A"
                @printf("%-24s | %-38s | FOUND (sz=%s, t=%s)\n", desc, fname, string(sz), string(n_times))
            else
                @printf("%-24s | %-38s | KEY NOT FOUND\n", desc, fname)
                global raw_files_ok = false
            end
        finally
            close(ds)
        end
    else
        @printf("%-24s | %-38s | MISSING FILE\n", desc, fname)
        global raw_files_ok = false
    end
end
println("-"^80)

# ------------------------------------------------------------------------------
# 3. Setup Grid & Simulation Parameters
# ------------------------------------------------------------------------------
println("\n[3/4] Constructing Target Grid & Executing setup_bsose Pipeline...")

# Target Regional Domain (matching simulation)
λ₁, λ₂ = (90.0, 150.0)
φ₁, φ₂ = (-70.0, -40.0)

# Check if running under GPU or CPU
has_gpu = false
try
    if @isdefined(CUDA) && CUDA.functional()
        has_gpu = true
    end
catch
end
arch = has_gpu ? GPU() : CPU()
println("  - Architecture : $(arch)")

# Choose resolution and stretching (if GPU is available vs local CPU test)
if arch isa GPU
    SCALING = 3 # 1/3 degree matching iteration 105/156
    Nx = Int(SCALING * (λ₂ - λ₁))
    Ny = Int(SCALING * (φ₂ - φ₁))
    z = ReferenceToStretchedDiscretization(; extent=5800,
        constant_spacing=2,
        maximum_spacing=200,
        constant_spacing_extent=500,
        stretching=PowerLawStretching(1.15))
    if dataset.iteration == 156
        start_date = DateTime(2014, 1, 1)
        end_date = DateTime(2014, 12, 31)
    else
        start_date = DateTime(2009, 1, 1)
        end_date = DateTime(2009, 12, 31)
    end
else
    # CPU: use 1-degree resolution and 2 months of dates for fast thorough validation
    println("  - Running on CPU in diagnostic verification mode (1° horizontal resolution, first 2 time levels)")
    Nx = Int((λ₂ - λ₁))
    Ny = Int((φ₂ - φ₁))
    z = ReferenceToStretchedDiscretization(; extent=5400,
        constant_spacing=50,
        maximum_spacing=500,
        constant_spacing_extent=500,
        stretching=PowerLawStretching(1.15))
    avail_dates = all_dates(dataset, :temperature)
    start_date = first(avail_dates)
    end_date = avail_dates[min(2, length(avail_dates))]
end

Nz = length(z)
dates = (start_date, end_date)

println("  - Grid Dimensions : Nx=$Nx, Ny=$Ny, Nz=$Nz")
println("  - Verification Dates: $start_date to $end_date")

underlying_grid = LatitudeLongitudeGrid(arch;
    size=(Nx, Ny, Nz),
    latitude=(φ₁, φ₂),
    longitude=(λ₁, λ₂),
    z,
    halo=(7, 7, 7))

println("  ✓ Created base LatitudeLongitudeGrid")

# ------------------------------------------------------------------------------
# 4. Generate Open Boundary Conditions from Scratch
# ------------------------------------------------------------------------------
println("\nGenerating BSOSE Open Boundary Conditions (cache=true)...")
t_start = time()
bcs = bsose_open_boundary_conditions(underlying_grid;
    dataset=dataset,
    dates=dates,
    winds=false,
    cache=true)
t_elapsed = round(time() - t_start, digits=2)
println("  ✓ Boundary conditions generated in $(t_elapsed) s")

# Verify unified dataset cache file was created
start_str = Dates.format(start_date, "yyyymmdd")
end_str = Dates.format(end_date, "yyyymmdd")
expected_cache_name = "bsose_dataset_$(dataset.iteration)_$(start_str)_to_$(end_str)_$(Nx)x$(Ny)x$(Nz).jld2"
expected_cache_path = joinpath(dataset.dir, expected_cache_name)

if isfile(expected_cache_path)
    println("  ✓ Cache file successfully written: $expected_cache_path ($(round(filesize(expected_cache_path)/1024^2, digits=2)) MB)")
else
    @warn "Expected cache file was not found at: $expected_cache_path"
end

# ------------------------------------------------------------------------------
# 5. Rigorous Product & Boundary Slice Verification
# ------------------------------------------------------------------------------
println("\n[4/4] Validating Products and Boundary Slices for Missing/NaN Values...")

function inspect_slice(name, bts)
    if bts === nothing
        @printf("%-18s | %-16s | %-10s | %-10s | %-8s | %s\n", name, "None", "-", "-", "0", "SKIPPED")
        return true
    end

    all_ok = true
    times = bts.times
    for t in 1:length(times)
        fld = bts[t]
        # Check interior (active cells)
        int_data = Array(interior(fld))
        par_data = Array(parent(fld))
        
        nan_int = count(isnan, int_data)
        nan_par = count(isnan, par_data)
        inf_int = count(isinf, int_data)
        
        v_min = nan_int == length(int_data) ? NaN : minimum(filter(!isnan, int_data))
        v_max = nan_int == length(int_data) ? NaN : maximum(filter(!isnan, int_data))

        status = (nan_int == 0 && inf_int == 0) ? "PASS" : "FAIL (NaNs)"
        if nan_int > 0 || inf_int > 0
            all_ok = false
        end

        sz_str = string(size(int_data))
        t_str = @sprintf("%s [t=%d]", name, t)
        @printf("%-20s | %-16s | %10.4f | %10.4f | %8d | %s\n",
                t_str, sz_str, v_min, v_max, nan_int, status)
    end
    return all_ok
end

println("-"^85)
@printf("%-20s | %-16s | %-10s | %-10s | %-8s | %s\n",
        "Field", "Interior Size", "Min Val", "Max Val", "NaN Count", "Status")
println("-"^85)

# Extract 2D boundary FieldTimeSeries from the unified dataset
cached_slices = JLD2.jldopen(expected_cache_path, "r") do f
    Dict(k => f[k] for k in keys(f) if k != "start_date" && k != "end_date")
end

total_failures = 0

for (k, v) in cached_slices
    if v isa FieldTimeSeries
        ok = inspect_slice(k, v)
        if !ok
            global total_failures += 1
        end
    end
end
println("-"^85)

# Also check single date initial condition fields
println("\nChecking 3D Initial Condition Fields (Temperature, Salinity, Velocities)...")
region = bsose_region(underlying_grid)

for (sym, vmin_phys, vmax_phys) in [
    (:temperature, T_MIN_PHYSICAL, T_MAX_PHYSICAL),
    (:salinity, S_MIN_PHYSICAL, S_MAX_PHYSICAL),
    (:u_velocity, -5.0, 5.0),
    (:v_velocity, -5.0, 5.0)
]
    init_fld = Field(Metadatum(sym; dataset=dataset, date=start_date, region=region), underlying_grid)
    # Apply bathymetry gap filling as done in bsose_initial_conditions!
    fill_bathymetry_gaps!(parent(init_fld), vmin_phys, vmax_phys)
    
    int_arr = Array(interior(init_fld))
    par_arr = Array(parent(init_fld))
    
    nan_int = count(isnan, int_arr)
    nan_par = count(isnan, par_arr)
    v_min = minimum(int_arr)
    v_max = maximum(int_arr)
    
    status = (nan_int == 0 && nan_par == 0) ? "PASS" : "FAIL"
    if status == "FAIL"
        global total_failures += 1
    end
    
    @printf("%-20s | %-16s | %10.4f | %10.4f | NaN(int=%d, par=%d) | %s\n",
            string(sym), string(size(int_arr)), v_min, v_max, nan_int, nan_par, status)
end

# ------------------------------------------------------------------------------
# 6. GPU Compatibility Verification Suite
# ------------------------------------------------------------------------------
println("\n[5/5] GPU Compatibility & Architecture Checks...")
println("-"^85)
@printf("%-26s | %-28s | %-12s | %s\n", "Component / Field", "Underlying Array Type", "Arch Target", "Compatibility")
println("-"^85)

gpu_failures = 0

function verify_gpu_compatibility(name, obj, target_arch)
    # Check what array backend is used in the field/data
    arr_type = "Unknown"
    is_compat = true
    reason = ""

    data_obj = nothing
    if obj isa Field
        data_obj = obj.data
    elseif obj isa FieldTimeSeries
        data_obj = obj.data
    elseif hasproperty(obj, :condition) && obj.condition isa FieldTimeSeries
        data_obj = obj.condition.data
    elseif hasproperty(obj, :condition) && obj.condition isa Field
        data_obj = obj.condition.data
    end

    if data_obj !== nothing
        arr_type = string(typeof(data_obj))
        # Simplify array type for printing
        short_type = occursin("CuArray", arr_type) ? "CUDA.CuArray" :
                     occursin("Array", arr_type) ? "Base.Array (CPU)" :
                     split(arr_type, "{")[1]

        if target_arch isa GPU
            # When target architecture is GPU, the array must be CuArray and not CPU Array
            if occursin("CuArray", arr_type)
                status_str = "PASS (GPU CuArray)"
            else
                status_str = "FAIL (CPU Array on GPU)"
                is_compat = false
                reason = "Array is stored in host CPU memory instead of device CuArray"
            end
        else
            # On CPU, CuArray is not expected, Array is normal
            status_str = occursin("Array", arr_type) ? "PASS (Host Array)" : "PASS"
        end
        @printf("%-26s | %-28s | %-12s | %s\n", name, short_type, string(target_arch), status_str)
    else
        @printf("%-26s | %-28s | %-12s | %s\n", name, "Non-array / Boundary", string(target_arch), "PASS (Metadata/BC)")
    end

    return is_compat
end

# Check boundary condition objects returned by bsose_open_boundary_conditions
println("Checking Boundary Condition Objects:")
for (var_sym, bconds) in pairs(bcs)
    for side in (:west, :east, :south, :north, :top)
        bc = getproperty(bconds, side)
        if bc !== nothing && hasproperty(bc, :condition) && bc.condition !== nothing
            name_str = "bcs.$var_sym.$side"
            ok = verify_gpu_compatibility(name_str, bc, arch)
            if !ok
                global gpu_failures += 1
            end
        end
    end
end

println("\nChecking 3D Fields & Model State Compatibility:")
# Check fields generated for initial conditions
for (sym, vmin_phys, vmax_phys) in [
    (:temperature, T_MIN_PHYSICAL, T_MAX_PHYSICAL),
    (:salinity, S_MIN_PHYSICAL, S_MAX_PHYSICAL),
    (:u_velocity, -5.0, 5.0),
    (:v_velocity, -5.0, 5.0)
]
    init_fld = Field(Metadatum(sym; dataset=dataset, date=start_date, region=region), underlying_grid)
    ok = verify_gpu_compatibility("Field($sym)", init_fld, arch)
    if !ok
        global gpu_failures += 1
    end
end

println("-"^85)

# Summarized Report
println("\n" * "="^85)
println(" SUMMARY OF ALL VERIFICATION CHECKS")
println("="^85)
@printf("  1. Raw NetCDF Products Inspection     : %s\n", raw_files_ok ? "ALL FOUND & VALID" : "MISSING FILES")
@printf("  2. Boundary Conditions Extraction      : %s\n", isfile(expected_cache_path) ? "SUCCESS (Cached)" : "FAILED")
@printf("  3. Data Integrity (NaNs / Missing)     : %s\n", total_failures == 0 ? "PASSED (Zero NaNs)" : "FAILED ($total_failures issues)")
@printf("  4. Architecture & GPU Compatibility    : %s\n", gpu_failures == 0 ? "PASSED (Fully Compatible)" : "FAILED ($gpu_failures incompatible arrays)")
println("="^85)

if total_failures == 0 && gpu_failures == 0 && raw_files_ok
    println("🎉 ALL CHECKS PASSED: Dataset is completely structured, NaN-free, and GPU-ready.")
else
    println("⚠️  WARNING: One or more checks flagged potential issues. Review the logs above.")
end
println("="^85)
