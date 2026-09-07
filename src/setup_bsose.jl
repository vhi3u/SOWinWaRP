# ==============================================================================
# setup_bsose.jl
# 
# Helper script to integrate BSOSE (Biogeochemical Southern Ocean State Estimate)
# into NumericalEarth.jl and Oceananigans.jl.
#
# Provides:
#  - BSOSEMonthly() dataset type compatible with NumericalEarth.jl (like ECCOMonthly() / GLORYSMonthly())
#  - Metadata, Metadatum, Field, FieldTimeSeries, and DatasetRestoring integration
#  - bsose_open_boundary_conditions() for open boundaries (NormalFlow & Value boundary conditions)
#  - bsose_initial_conditions!() to initialize model state from BSOSE
#  - bsose_surface_wind_stress() for top boundary flux forcing
#  - bsose_sponge_forcing() for sponge layer relaxation
# ==============================================================================

using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids
using Oceananigans.Grids: topology
using Oceananigans.BoundaryConditions
using Oceananigans.BoundaryConditions: PerturbationAdvection
using Oceananigans.OutputReaders: FieldTimeSeries, Cyclical
using Oceananigans.Architectures: architecture, CPU, GPU, on_architecture
using Oceananigans.Fields: interior, location, fill_halo_regions!
using NumericalEarth
using NumericalEarth.DataWrangling
using NCDatasets
using Dates
using CFTime
using Downloads
using NumericalEarth.DataWrangling: JLD2

import NumericalEarth.DataWrangling:
    default_download_directory,
    dataset_variable_name,
    dataset_location,
    all_dates,
    first_date,
    last_date,
    metadata_filename,
    is_three_dimensional,
    reversed_vertical_axis,
    longitude_interfaces,
    latitude_interfaces,
    z_interfaces,
    longitude_name,
    latitude_name,
    retrieve_data,
    available_variables,
    default_inpainting,
    centers_to_interfaces,
    metadata_path

import Downloads: download

# ==============================================================================
# 1. BSOSE Dataset Types & Path Resolution
# ==============================================================================

abstract type BSOSEDataset end

"""
    BSOSEMonthly(; dir = default_bsose_directory(), iteration = 156)

Construct a `BSOSEMonthly` dataset descriptor.

# Keyword Arguments
- `dir`: Path to the directory containing BSOSE NetCDF files.
         Defaults to `ENV["BSOSE_DIR"]`, `/storage/scratch1/1/vnguyen480/SOWinWaRP/data` (HPC),
         or `./data`.
- `iteration`: BSOSE iteration number. Iteration 156 spans 2013–2024, iteration 105 spans 2008–2012.
"""
struct BSOSEMonthly <: BSOSEDataset
    dir::String
    iteration::Int
end

function default_bsose_directory()
    if haskey(ENV, "BSOSE_DIR") && isdir(ENV["BSOSE_DIR"])
        return ENV["BSOSE_DIR"]
    elseif isdir("/storage/scratch1/1/vnguyen480/SOWinWaRP/data")
        return "/storage/scratch1/1/vnguyen480/SOWinWaRP/data"
    elseif isdir(normpath(joinpath(@__DIR__, "..", "data")))
        return normpath(joinpath(@__DIR__, "..", "data"))
    elseif isdir("data")
        return abspath("data")
    else
        return abspath("data")
    end
end

function BSOSEMonthly(; dir=default_bsose_directory(), iteration=nothing)
    # Auto-detect iteration if not explicitly given
    if isnothing(iteration)
        if isdir(dir)
            files = readdir(dir)
            if any(f -> occursin("156", f) || occursin("I156", f), files)
                iteration = 156
            elseif any(f -> occursin("105", f) || occursin("i105", f), files)
                iteration = 105
            else
                iteration = 156
            end
        else
            iteration = 156
        end
    end
    return BSOSEMonthly(abspath(dir), iteration)
end

Base.summary(ds::BSOSEMonthly) = "BSOSEMonthly(iteration=$(ds.iteration), dir=\"$(ds.dir)\")"

# Variable name dictionary
const BSOSE_VARIABLE_NAMES = Dict(
    :temperature => "THETA",
    :salinity => "SALT",
    :u_velocity => "UVEL",
    :v_velocity => "VVEL",
    :zonal_wind_stress => "oceTAUX",
    :meridional_wind_stress => "oceTAUY",
    :surface_heat_flux => "TFLUX",
    # Aliases
    :Theta => "THETA",
    :Salt => "SALT",
    :Uvel => "UVEL",
    :Vvel => "VVEL",
    :oceTAUX => "oceTAUX",
    :oceTAUY => "oceTAUY",
    :surfTflx => "TFLUX",
    :T => "THETA",
    :S => "SALT",
    :u => "UVEL",
    :v => "VVEL"
)

# Variable spatial locations on Arakawa C-grid
const BSOSE_LOCATIONS = Dict(
    :temperature => (Center, Center, Center),
    :salinity => (Center, Center, Center),
    :u_velocity => (Face, Center, Center),
    :v_velocity => (Center, Face, Center),
    :zonal_wind_stress => (Face, Center, Nothing),
    :meridional_wind_stress => (Center, Face, Nothing),
    :surface_heat_flux => (Center, Center, Nothing),
    :Theta => (Center, Center, Center),
    :Salt => (Center, Center, Center),
    :Uvel => (Face, Center, Center),
    :Vvel => (Center, Face, Center),
    :oceTAUX => (Face, Center, Nothing),
    :oceTAUY => (Center, Face, Nothing),
    :surfTflx => (Center, Center, Nothing),
    :T => (Center, Center, Center),
    :S => (Center, Center, Center),
    :u => (Face, Center, Center),
    :v => (Center, Face, Center)
)

function resolve_bsose_filename(dir::String, var_name::Symbol, iteration::Int)
    shortname = get(BSOSE_VARIABLE_NAMES, var_name, string(var_name))

    # Candidate filename patterns for Iteration 156 (2013-2024)
    cands_156 = [
        "$(shortname)_bsoseI156_2013to2024_monthly.nc",
        "$(titlecase(shortname))_bsoseI156_2013to2024_monthly.nc",
        "$(uppercase(shortname))_bsoseI156_2013to2024_monthly.nc",
        "$(lowercase(shortname))_bsoseI156_2013to2024_monthly.nc",
        "bsose_i156_2013to2024_monthly_$(shortname).nc",
        "bsose_i156_2013to2024_monthly_$(titlecase(shortname)).nc",
    ]

    # Candidate filename patterns for Iteration 105 (2008-2012)
    cands_105 = [
        "bsose_i105_2008to2012_monthly_$(shortname).nc",
        "bsose_i105_2008to2012_monthly_$(titlecase(shortname)).nc",
        "bsose_i105_2008to2012_monthly_$(uppercase(shortname)).nc",
        "$(titlecase(shortname))_bsoseI105_2008to2012_monthly.nc",
        "$(shortname)_bsoseI105_2008to2012_monthly.nc",
    ]

    candidates = iteration == 156 ? vcat(cands_156, cands_105) : vcat(cands_105, cands_156)

    for cand in candidates
        if isfile(joinpath(dir, cand))
            return cand
        end
    end

    # Fallback directory search matching target token
    if isdir(dir)
        files = readdir(dir)
        t_low = lowercase(shortname)
        for f in files
            endswith(f, ".nc") || continue
            if occursin(t_low, lowercase(f))
                return f
            end
        end
    end

    return iteration == 156 ? "$(titlecase(shortname))_bsoseI156_2013to2024_monthly.nc" : "bsose_i105_2008to2012_monthly_$(titlecase(shortname)).nc"
end

# ==============================================================================
# 2. NumericalEarth.DataWrangling Interface Extensions
# ==============================================================================

const BSOSEMetadata{D} = Metadata{<:BSOSEDataset,D}
const BSOSEMetadatum = Metadatum{<:BSOSEDataset}

DataWrangling.default_download_directory(dataset::BSOSEMonthly) = dataset.dir
DataWrangling.available_variables(::BSOSEDataset) = BSOSE_VARIABLE_NAMES

function DataWrangling.dataset_variable_name(metadata::BSOSEMetadata)
    name = metadata.name
    return get(BSOSE_VARIABLE_NAMES, name, string(name))
end

function DataWrangling.dataset_location(::BSOSEDataset, name)
    return get(BSOSE_LOCATIONS, name, (Center, Center, Center))
end

function DataWrangling.is_three_dimensional(metadata::BSOSEMetadata)
    loc = dataset_location(metadata.dataset, metadata.name)
    return loc[3] !== Nothing
end

# BSOSE stores vertical dimension surface-to-bottom (Z[1] = -2.1m, Z[Nz] = -5800m).
# Oceananigans uses bottom-to-surface indexing (k=1 at bottom, k=Nz at surface).
DataWrangling.reversed_vertical_axis(::BSOSEDataset) = true

DataWrangling.longitude_interfaces(::BSOSEMetadata) = (0.0, 360.0)
DataWrangling.latitude_interfaces(::BSOSEMetadata) = (-78.0, -29.7)

function DataWrangling.longitude_name(metadata::BSOSEMetadata)
    loc = dataset_location(metadata.dataset, metadata.name)
    return loc[1] === Face ? "XG" : "XC"
end

function DataWrangling.latitude_name(metadata::BSOSEMetadata)
    loc = dataset_location(metadata.dataset, metadata.name)
    return loc[2] === Face ? "YG" : "YC"
end

function DataWrangling.metadata_filename(dataset::BSOSEMonthly, name, date, region)
    return resolve_bsose_filename(dataset.dir, name, dataset.iteration)
end

# BSOSE has valid, complete ocean coverage within Southern Ocean domains.
# Disabling inpainting prevents NumericalEarth from generating dozens of separate per-slice JLD2 files.
DataWrangling.default_inpainting(::BSOSEMetadata) = nothing
DataWrangling.default_inpainting(::BSOSEMetadatum) = nothing

function DataWrangling.all_dates(dataset::BSOSEMonthly, var)
    fn = resolve_bsose_filename(dataset.dir, var, dataset.iteration)
    fp = joinpath(dataset.dir, fn)
    if isfile(fp)
        try
            ds = Dataset(fp)
            dates = [DateTime(t) for t in ds["time"][:]]
            close(ds)
            return dates
        catch
        end
    end
    # Fallback based on iteration
    if dataset.iteration == 156
        return collect(DateTime(2013, 1, 1):Month(1):DateTime(2024, 12, 1))
    else
        return collect(DateTime(2008, 1, 1):Month(1):DateTime(2012, 12, 1))
    end
end

function DataWrangling.z_interfaces(metadata::BSOSEMetadata)
    fn = resolve_bsose_filename(metadata.dataset.dir, metadata.name, metadata.dataset.iteration)
    fp = joinpath(metadata.dataset.dir, fn)
    if isfile(fp)
        ds = Dataset(fp)
        zc = Float64.(reverse(ds["Z"][:])) # bottom-first
        close(ds)
        return centers_to_interfaces(zc)
    else
        # Standard BSOSE 52-level centers reversed
        zc_default = [-5800.0, -5400.0, -5000.0, -4600.0, -4200.0, -3800.0, -3400.0, -3000.0, -2610.0, -2270.0, -2010.0,
            -1800.0, -1600.0, -1400.0, -1225.0, -1100.0, -1000.0, -900.0, -800.0, -700.0, -614.0, -551.5,
            -500.0, -450.0, -402.5, -361.0, -327.0, -301.0, -280.0, -260.0, -240.0, -220.0, -200.0, -180.0,
            -161.5, -146.5, -135.0, -125.0, -115.0, -105.0, -95.0, -85.0, -75.0, -65.0, -55.0, -45.0,
            -35.25, -26.25, -18.55, -12.15, -6.7, -2.1]
        return centers_to_interfaces(zc_default)
    end
end

function Base.size(metadata::Metadata{<:BSOSEDataset})
    Nx = 1080
    Ny = 294
    Nz = is_three_dimensional(metadata) ? 52 : 1
    Nt = metadata.dates isa AbstractArray ? length(metadata.dates) : 1
    return (Nx, Ny, Nz, Nt)
end

function Downloads.download(metadata::Metadata{<:BSOSEDataset})
    path = metadata_path(metadata)
    paths = path isa AbstractVector ? path : [path]
    for p in paths
        if !isfile(p)
            error("BSOSE file not found at $p. Ensure BSOSE NetCDF files are downloaded to $(metadata.dir).")
        end
    end
    return path
end

# Time-index resolver matching year & month
function find_bsose_time_index(ds::Dataset, target_date)
    if isnothing(target_date)
        return 1
    end
    if target_date isa Integer
        return clamp(target_date, 1, length(ds["time"]))
    end

    time_raw = ds["time"][:]
    target_dt = DateTime(target_date)
    target_y = Dates.year(target_dt)
    target_m = Dates.month(target_dt)

    # 1. Match year and month
    for (i, t) in enumerate(time_raw)
        dt = DateTime(t)
        if Dates.year(dt) == target_y && Dates.month(dt) == target_m
            return i
        end
    end

    # 2. Match closest timestamp
    diffs = [abs((DateTime(t) - target_dt).value) for t in time_raw]
    return argmin(diffs)
end

"""
    extrapolate_bsose_3d!(data::Array{Float32, 3}, var_name::AbstractString)

In BSOSE NetCDF data, cells below bathymetry and dry land columns are masked with 0.0 (or NaN).
When regridded onto an Oceananigans grid with deeper or higher-resolution bathymetry, these
zero values cause massive unphysical density shocks (e.g., freshwater S = 0 PSU beneath 34.5 PSU seawater),
triggering catastrophic convective velocities and DomainErrors in TEOS-10.

This function vertically extrapolates each ocean column downwards by extending the bottom-most
valid ocean value down to the deepest level (k = Nz in BSOSE coordinates).
For completely dry columns (such as continental Antarctica), it fills with physically realistic
Southern Ocean background values (S = 34.6 PSU, T = 0.0°C, u = 0.0, v = 0.0).
"""
function extrapolate_bsose_3d!(data::Array{Float32,3}, var_name::AbstractString)
    Nx, Ny, Nz = size(data)
    is_salt = uppercase(var_name) in ("SALT", "SALINITY", "S")
    is_theta = uppercase(var_name) in ("THETA", "TEMPERATURE", "T")
    default_bg = is_salt ? 34.6f0 : (is_theta ? 0.0f0 : 0.0f0)

    @inbounds for j in 1:Ny, i in 1:Nx
        first_valid = 0
        last_valid = 0

        if is_salt
            for k in 1:Nz
                val = data[i, j, k]
                if val > 10.0f0 && !isnan(val)
                    first_valid == 0 && (first_valid = k)
                    last_valid = k
                end
            end
        else # theta, u, v
            for k in 1:Nz
                val = data[i, j, k]
                if val != 0.0f0 && !isnan(val)
                    first_valid == 0 && (first_valid = k)
                    last_valid = k
                end
            end
        end

        if last_valid == 0
            # Entire column is dry/land: fill with default background
            for k in 1:Nz
                data[i, j, k] = default_bg
            end
        else
            # Fill upward if surface cells were unpopulated
            top_val = data[i, j, first_valid]
            for k in 1:(first_valid-1)
                data[i, j, k] = top_val
            end
            # Fill downward below bathymetry with bottom-most ocean value
            bot_val = data[i, j, last_valid]
            for k in (last_valid+1):Nz
                data[i, j, k] = bot_val
            end
        end
    end
    return data
end

function DataWrangling.retrieve_data(metadatum::BSOSEMetadatum)
    path = metadata_path(metadatum)
    name = dataset_variable_name(metadatum)

    ds = Dataset(path)

    # In BSOSE files, if variable name casing differs (e.g. SALT vs Salt)
    var_key = haskey(ds, name) ? name : (haskey(ds, uppercase(name)) ? uppercase(name) : (haskey(ds, titlecase(name)) ? titlecase(name) : name))

    time_idx = find_bsose_time_index(ds, metadatum.dates)

    if is_three_dimensional(metadatum)
        raw = ds[var_key][:, :, :, time_idx]
        data = Array{Float32}(undef, size(raw))
        @inbounds for i in eachindex(raw)
            val = raw[i]
            data[i] = (ismissing(val) || isnan(val)) ? 0.0f0 : Float32(val)
        end
        # Vertically extrapolate ocean columns downwards to eliminate 0.0 below bathymetry
        extrapolate_bsose_3d!(data, name)
        if reversed_vertical_axis(metadatum.dataset)
            data = reverse(data, dims=3)
        end
    else
        raw = ds[var_key][:, :, time_idx]
        data = Array{Float32}(undef, size(raw))
        @inbounds for i in eachindex(raw)
            val = raw[i]
            data[i] = (ismissing(val) || isnan(val)) ? NaN32 : Float32(val)
        end
    end

    close(ds)
    return data
end

# ==============================================================================
# 3. Boundary Slice Extraction & Open Boundary Conditions (OBCs)
# ==============================================================================

"""
    boundary_slice_time_series(fts::FieldTimeSeries, side::Symbol)

Extract a 2D boundary `FieldTimeSeries` along `side` (:west, :east, :south, :north)
from a 3D `FieldTimeSeries`.

The resulting 2D FieldTimeSeries has the reduced dimensionality and index structure
required by Oceananigans boundary conditions (`YZFTS` for west/east, `XZFTS` for south/north).
"""
function boundary_slice_time_series(fts::FieldTimeSeries, side::Symbol)
    grid = fts.grid
    times = fts.times
    time_indexing = fts.time_indexing
    LX, LY, LZ = location(fts)
    Nx, Ny, Nz = size(grid)

    if side === :west
        # West boundary: i = 1, location = (Nothing, LY, LZ)
        bts = FieldTimeSeries{Nothing,LY,LZ}(grid, times; indices=(1, :, :), time_indexing)
        for t in 1:length(times)
            interior(bts[t], 1, :, :) .= interior(fts[t], 1, :, :)
            fill_halo_regions!(bts[t])
        end
        return bts
    elseif side === :east
        # East boundary: i = Nx (Center) or Nx+1 (Face), location = (Nothing, LY, LZ)
        i_east = LX === Face ? Nx + 1 : Nx
        bts = FieldTimeSeries{Nothing,LY,LZ}(grid, times; indices=(1, :, :), time_indexing)
        for t in 1:length(times)
            interior(bts[t], 1, :, :) .= interior(fts[t], i_east, :, :)
            fill_halo_regions!(bts[t])
        end
        return bts
    elseif side === :south
        # South boundary: j = 1, location = (LX, Nothing, LZ)
        bts = FieldTimeSeries{LX,Nothing,LZ}(grid, times; indices=(:, 1, :), time_indexing)
        for t in 1:length(times)
            interior(bts[t], :, 1, :) .= interior(fts[t], :, 1, :)
            fill_halo_regions!(bts[t])
        end
        return bts
    elseif side === :north
        # North boundary: j = Ny (Center) or Ny+1 (Face), location = (LX, Nothing, LZ)
        j_north = LY === Face ? Ny + 1 : Ny
        bts = FieldTimeSeries{LX,Nothing,LZ}(grid, times; indices=(:, 1, :), time_indexing)
        for t in 1:length(times)
            interior(bts[t], :, 1, :) .= interior(fts[t], :, j_north, :)
            fill_halo_regions!(bts[t])
        end
        return bts
    elseif side === :top
        # Top boundary: k = Nz (Center) or Nz+1 (Face), location = (LX, LY, Nothing)
        bts = FieldTimeSeries{LX,LY,Nothing}(grid, times; indices=(:, :, 1), time_indexing)
        for t in 1:length(times)
            interior(bts[t], :, :, 1) .= interior(fts[t], :, :, 1)
            fill_halo_regions!(bts[t])
        end
        return bts
    else
        throw(ArgumentError("Unknown side: $side. Valid options are :west, :east, :south, :north, :top."))
    end
end

# ==============================================================================
# 3. Surface Wind Stress Helper (Standalone)
# ==============================================================================

"""
    bsose_surface_wind_stress(grid;
                              dataset = BSOSEMonthly(),
                              dates = all_dates(dataset, :temperature)[1:12],
                              ρ₀ = 1026.0)

Load BSOSE surface wind stress (`oceTAUX` and `oceTAUY`), convert to kinematic
momentum flux (`τ / ρ₀`), and return a `NamedTuple` of top `FluxBoundaryCondition`s:
`(; u = FluxBoundaryCondition(top_u), v = FluxBoundaryCondition(top_v))`.

Can be used standalone or passed to `bsose_open_boundary_conditions`.
"""
function bsose_surface_wind_stress(grid;
    dataset=BSOSEMonthly(),
    dates=all_dates(dataset, :temperature)[1:12],
    ρ₀=1026.0)
    @info "Loading BSOSE surface wind stress (oceTAUX, oceTAUY)..."
    taux_fts = FieldTimeSeries(Metadata(:zonal_wind_stress; dataset, dates), grid)
    tauy_fts = FieldTimeSeries(Metadata(:meridional_wind_stress; dataset, dates), grid)

    # Convert stress (N/m²) to kinematic momentum flux (m²/s²): divide by ρ₀
    for t in 1:length(dates)
        interior(taux_fts[t]) ./= ρ₀
        interior(tauy_fts[t]) ./= ρ₀
    end

    top_u_slice = boundary_slice_time_series(taux_fts, :top)
    top_v_slice = boundary_slice_time_series(tauy_fts, :top)

    return (u=FluxBoundaryCondition(top_u_slice),
        v=FluxBoundaryCondition(top_v_slice))
end

# ==============================================================================
# 4. Open Boundary Conditions (OBCs)
# ==============================================================================

"""
    bsose_open_boundary_conditions(grid;
                                   dataset = BSOSEMonthly(),
                                   dates = all_dates(dataset, :temperature)[1:12],
                                   scheme = nothing,
                                   winds = nothing,
                                   ρ₀ = 1026.0)

Construct a `NamedTuple` of `FieldBoundaryConditions` `(; u, v, T, S)` configured with
open boundary conditions from BSOSE.

# Arguments
- `grid`: The simulation `LatitudeLongitudeGrid`.
- `dataset`: `BSOSEMonthly()` instance.
- `dates`: Range or collection of dates for open boundary forcing.
- `scheme`: Radiation/matching scheme for open boundary normal flows.
            Defaults to `Oceananigans.BoundaryConditions.PerturbationAdvection()`.
- `winds`: Surface wind forcing. Options:
           - `nothing` or `false` (default): Disables surface wind forcing (no-flux top boundary).
           - `true`: Automatically calls `bsose_surface_wind_stress` and applies top flux.
           - A `NamedTuple` `(; u, v)` returned from `bsose_surface_wind_stress(grid; ...)`.
- `ρ₀`: Seawater density for wind stress conversion (kg/m³). Default: 1026.0.
- `cache`: If `true`, saves and loads all boundary conditions to/from a SINGLE unified dataset file.
- `cache_file`: Optional custom file path for the unified dataset file.
"""
function bsose_open_boundary_conditions(grid;
    dataset=BSOSEMonthly(),
    dates=all_dates(dataset, :temperature)[1:12],
    scheme=PerturbationAdvection(),
    winds=nothing,
    ρ₀=1026.0,
    cache=true,
    cache_file=nothing)
    @info "Setting up BSOSE Open Boundary Conditions..."

    # Determine simulation start and end dates
    start_d = dates isa Tuple ? dates[1] : first(dates)
    end_d = dates isa Tuple ? dates[2] : last(dates)
    start_str = Dates.format(start_d, "yyyymmdd")
    end_str = Dates.format(end_d, "yyyymmdd")

    underlying = grid isa ImmersedBoundaryGrid ? grid.underlying_grid : grid
    Nx, Ny, Nz = size(underlying)
    is_x_periodic = topology(underlying, 1) === Periodic

    # Unified dataset filename
    default_dataset_name = "bsose_dataset_$(dataset.iteration)_$(start_str)_to_$(end_str)_$(Nx)x$(Ny)x$(Nz).jld2"
    dataset_path = cache_file !== nothing ? cache_file : joinpath(dataset.dir, default_dataset_name)

    # 1. Attempt loading from single unified dataset
    if cache && isfile(dataset_path)
        @info "Loading all BSOSE boundary conditions from single unified dataset: $dataset_path"
        cached = JLD2.jldopen(dataset_path, "r") do f
            (
                u_west=haskey(f, "u_west") ? f["u_west"] : nothing,
                u_east=haskey(f, "u_east") ? f["u_east"] : nothing,
                u_south=haskey(f, "u_south") ? f["u_south"] : nothing,
                u_north=haskey(f, "u_north") ? f["u_north"] : nothing,
                v_west=haskey(f, "v_west") ? f["v_west"] : nothing,
                v_east=haskey(f, "v_east") ? f["v_east"] : nothing,
                v_south=haskey(f, "v_south") ? f["v_south"] : nothing,
                v_north=haskey(f, "v_north") ? f["v_north"] : nothing,
                T_west=haskey(f, "T_west") ? f["T_west"] : nothing,
                T_east=haskey(f, "T_east") ? f["T_east"] : nothing,
                T_south=haskey(f, "T_south") ? f["T_south"] : nothing,
                T_north=haskey(f, "T_north") ? f["T_north"] : nothing,
                S_west=haskey(f, "S_west") ? f["S_west"] : nothing,
                S_east=haskey(f, "S_east") ? f["S_east"] : nothing,
                S_south=haskey(f, "S_south") ? f["S_south"] : nothing,
                S_north=haskey(f, "S_north") ? f["S_north"] : nothing,
                wind_u=haskey(f, "wind_u") ? f["wind_u"] : nothing,
                wind_v=haskey(f, "wind_v") ? f["wind_v"] : nothing,
            )
        end
        u_west, u_east, u_south, u_north = cached.u_west, cached.u_east, cached.u_south, cached.u_north
        v_west, v_east, v_south, v_north = cached.v_west, cached.v_east, cached.v_south, cached.v_north
        T_west, T_east, T_south, T_north = cached.T_west, cached.T_east, cached.T_south, cached.T_north
        S_west, S_east, S_south, S_north = cached.S_west, cached.S_east, cached.S_south, cached.S_north
        top_u_cached, top_v_cached = cached.wind_u, cached.wind_v
    else
        # 1. Load 3D FieldTimeSeries for state variables
        @info "Extracting BSOSE fields for simulation window ($start_d to $end_d)..."
        @info " -> Loading BSOSE u_velocity..."
        u_fts = FieldTimeSeries(Metadata(:u_velocity; dataset, dates), grid)
        @info " -> Loading BSOSE v_velocity..."
        v_fts = FieldTimeSeries(Metadata(:v_velocity; dataset, dates), grid)
        @info " -> Loading BSOSE temperature..."
        T_fts = FieldTimeSeries(Metadata(:temperature; dataset, dates), grid)
        @info " -> Loading BSOSE salinity..."
        S_fts = FieldTimeSeries(Metadata(:salinity; dataset, dates), grid)

        # 2. Extract 2D boundary slices
        @info " -> Extracting boundary slices (West, East, South, North)..."
        u_west = boundary_slice_time_series(u_fts, :west)
        u_east = boundary_slice_time_series(u_fts, :east)
        u_south = boundary_slice_time_series(u_fts, :south)
        u_north = boundary_slice_time_series(u_fts, :north)

        v_west = boundary_slice_time_series(v_fts, :west)
        v_east = boundary_slice_time_series(v_fts, :east)
        v_south = boundary_slice_time_series(v_fts, :south)
        v_north = boundary_slice_time_series(v_fts, :north)

        T_west = boundary_slice_time_series(T_fts, :west)
        T_east = boundary_slice_time_series(T_fts, :east)
        T_south = boundary_slice_time_series(T_fts, :south)
        T_north = boundary_slice_time_series(T_fts, :north)

        S_west = boundary_slice_time_series(S_fts, :west)
        S_east = boundary_slice_time_series(S_fts, :east)
        S_south = boundary_slice_time_series(S_fts, :south)
        S_north = boundary_slice_time_series(S_fts, :north)

        top_u_cached = nothing
        top_v_cached = nothing

        # Compute winds if enabled so they can be saved into the single dataset
        if winds === true
            wind_bcs = bsose_surface_wind_stress(grid; dataset, dates, ρ₀)
            top_u_cached = wind_bcs.u
            top_v_cached = wind_bcs.v
        end

        # Save all boundary conditions together into ONE single dataset
        if cache
            @info "Saving all boundary conditions together into single dataset: $dataset_path"
            JLD2.jldopen(dataset_path, "w") do f
                f["start_date"] = string(start_d)
                f["end_date"] = string(end_d)
                f["u_west"] = u_west
                f["u_east"] = u_east
                f["u_south"] = u_south
                f["u_north"] = u_north
                f["v_west"] = v_west
                f["v_east"] = v_east
                f["v_south"] = v_south
                f["v_north"] = v_north
                f["T_west"] = T_west
                f["T_east"] = T_east
                f["T_south"] = T_south
                f["T_north"] = T_north
                f["S_west"] = S_west
                f["S_east"] = S_east
                f["S_south"] = S_south
                f["S_north"] = S_north
                if top_u_cached !== nothing && top_v_cached !== nothing
                    f["wind_u"] = top_u_cached
                    f["wind_v"] = top_v_cached
                end
            end
        end
    end

    # Ensure all boundary slices have valid ocean tracer ranges (no 0.0 salinity or extreme T)
    for s_slice in (S_west, S_east, S_south, S_north)
        if s_slice !== nothing
            for t in 1:length(s_slice.times)
                arr = interior(s_slice[t])
                @. arr = ifelse(arr < 25.0f0, 34.6f0, arr)
                fill_halo_regions!(s_slice[t])
            end
        end
    end
    for t_slice in (T_west, T_east, T_south, T_north)
        if t_slice !== nothing
            for t in 1:length(t_slice.times)
                arr = interior(t_slice[t])
                @. arr = ifelse(arr < -3.0f0, -1.8f0, arr)
                fill_halo_regions!(t_slice[t])
            end
        end
    end

    # 3. Top boundary condition (surface wind stress)
    top_u_bc = FluxBoundaryCondition(nothing)
    top_v_bc = FluxBoundaryCondition(nothing)

    if winds === true
        if top_u_cached !== nothing && top_v_cached !== nothing
            top_u_bc = top_u_cached
            top_v_bc = top_v_cached
        else
            wind_bcs = bsose_surface_wind_stress(grid; dataset, dates, ρ₀)
            top_u_bc = wind_bcs.u
            top_v_bc = wind_bcs.v
        end
    elseif winds isa NamedTuple
        top_u_bc = get(winds, :u, FluxBoundaryCondition(nothing))
        top_v_bc = get(winds, :v, FluxBoundaryCondition(nothing))
    else
        @info " -> Surface wind stress is DISABLED (top boundary is no-flux)."
    end

    # 4. Construct FieldBoundaryConditions
    # Note: On East/West boundaries, normal velocity is u (NormalFlow), tangential is v (Value).
    #       On South/North boundaries, normal velocity is v (NormalFlow), tangential is u (Value).
    # If the domain is longitudinally periodic (e.g. Circumpolar), only South and North BCs are applied.
    if is_x_periodic
        @info " -> Longitude is Periodic (circumpolar): applying South & North boundary conditions."
        u_bcs = FieldBoundaryConditions(
            south=ValueBoundaryCondition(u_south),
            north=ValueBoundaryCondition(u_north),
            top=top_u_bc
        )

        v_bcs = FieldBoundaryConditions(
            south=NormalFlowBoundaryCondition(v_south; scheme),
            north=NormalFlowBoundaryCondition(v_north; scheme),
            top=top_v_bc
        )

        T_bcs = FieldBoundaryConditions(
            south=ValueBoundaryCondition(T_south),
            north=ValueBoundaryCondition(T_north)
        )

        S_bcs = FieldBoundaryConditions(
            south=ValueBoundaryCondition(S_south),
            north=ValueBoundaryCondition(S_north)
        )
    else
        @info " -> Longitude is Bounded (regional): applying West, East, South, and North boundary conditions."
        u_bcs = FieldBoundaryConditions(
            west=NormalFlowBoundaryCondition(u_west; scheme),
            east=NormalFlowBoundaryCondition(u_east; scheme),
            south=ValueBoundaryCondition(u_south),
            north=ValueBoundaryCondition(u_north),
            top=top_u_bc
        )

        v_bcs = FieldBoundaryConditions(
            west=ValueBoundaryCondition(v_west),
            east=ValueBoundaryCondition(v_east),
            south=NormalFlowBoundaryCondition(v_south; scheme),
            north=NormalFlowBoundaryCondition(v_north; scheme),
            top=top_v_bc
        )

        T_bcs = FieldBoundaryConditions(
            west=ValueBoundaryCondition(T_west),
            east=ValueBoundaryCondition(T_east),
            south=ValueBoundaryCondition(T_south),
            north=ValueBoundaryCondition(T_north)
        )

        S_bcs = FieldBoundaryConditions(
            west=ValueBoundaryCondition(S_west),
            east=ValueBoundaryCondition(S_east),
            south=ValueBoundaryCondition(S_south),
            north=ValueBoundaryCondition(S_north)
        )
    end

    @info "BSOSE Open Boundary Conditions setup complete."
    return (u=u_bcs, v=v_bcs, T=T_bcs, S=S_bcs)
end

# ==============================================================================
# 5. Initial Conditions Helper
# ==============================================================================

"""
    bsose_initial_conditions!(model;
                               dataset = BSOSEMonthly(),
                               date = nothing,
                               dates = nothing)

Initialize `model` tracers (`T`, `S`) and velocities (`u`, `v`) from BSOSE at `date`.
If `dates` is passed, initializes at the start of the simulation window.
"""
function bsose_initial_conditions!(model;
    dataset=BSOSEMonthly(),
    date=nothing,
    dates=nothing)
    init_date = date !== nothing ? date :
                dates !== nothing ? (dates isa Tuple ? dates[1] : first(dates)) :
                first_date(dataset, :temperature)

    @info "Initializing model state from BSOSE at date: $init_date..."
    grid = model.grid

    T_init = Field(Metadatum(:temperature; dataset, date=init_date), grid)
    S_init = Field(Metadatum(:salinity; dataset, date=init_date), grid)
    u_init = Field(Metadatum(:u_velocity; dataset, date=init_date), grid)
    v_init = Field(Metadatum(:v_velocity; dataset, date=init_date), grid)

    # Sanitize initial tracer fields to guarantee no unphysical values exist in open or immersed cells
    S_int = interior(S_init)
    T_int = interior(T_init)
    @. S_int = ifelse(S_int < 25.0, 34.6, S_int)
    @. T_int = ifelse(T_int < -3.0, -1.8, T_int)

    set!(model; u=u_init, v=v_init, T=T_init, S=S_init)

    @info "Model successfully initialized with BSOSE fields."
    return nothing
end

# ==============================================================================
# 6. Single NetCDF Simulation Dataset Creation Helper
# ==============================================================================

"""
    create_bsose_netcdf_dataset(grid;
                                dataset = BSOSEMonthly(),
                                dates = (DateTime(2008, 1, 1), DateTime(2008, 12, 31)),
                                output_path = nothing,
                                buffer = 2.0)

Extract all raw BSOSE variables (`UVEL`, `VVEL`, `THETA`, `SALT`, `oceTAUX`, `oceTAUY`) for the
simulation's spatial domain and time window into a SINGLE unified NetCDF file.
"""
function create_bsose_netcdf_dataset(grid;
    dataset=BSOSEMonthly(),
    dates=(DateTime(2008, 1, 1), DateTime(2008, 12, 31)),
    output_path=nothing,
    buffer=2.0)
    start_d = dates isa Tuple ? dates[1] : first(dates)
    end_d = dates isa Tuple ? dates[2] : last(dates)
    start_str = Dates.format(start_d, "yyyymmdd")
    end_str = Dates.format(end_d, "yyyymmdd")

    if output_path === nothing
        output_path = joinpath(dataset.dir, "bsose_subset_$(dataset.iteration)_$(start_str)_to_$(end_str).nc")
    end

    @info "Creating unified BSOSE NetCDF dataset for $start_d to $end_d at: $output_path"

    underlying = grid isa ImmersedBoundaryGrid ? grid.underlying_grid : grid
    λ_min = minimum(underlying.λᶠᵃᵃ) - buffer
    λ_max = maximum(underlying.λᶠᵃᵃ) + buffer
    φ_min = minimum(underlying.φᵃᶠᵃ) - buffer
    φ_max = maximum(underlying.φᵃᶠᵃ) + buffer

    # Reference dimensions from Theta file
    theta_fp = joinpath(dataset.dir, resolve_bsose_filename(dataset.dir, :temperature, dataset.iteration))
    ds_ref = NCDataset(theta_fp)
    times = ds_ref["time"][:]
    t_idx = findall(t -> (start_d - Month(1)) <= t <= (end_d + Month(1)), times)
    if isempty(t_idx)
        t_idx = 1:min(12, length(times))
    end
    lon_c = ds_ref["XC"][:]
    lat_c = ds_ref["YC"][:]
    z_c = ds_ref["Z"][:]

    ilon_c = findall(x -> λ_min <= x <= λ_max, lon_c)
    ilat_c = findall(y -> φ_min <= y <= φ_max, lat_c)
    close(ds_ref)

    NCDataset(output_path, "c") do ds_out
        defDim(ds_out, "XC", length(ilon_c))
        defDim(ds_out, "YC", length(ilat_c))
        defDim(ds_out, "Z", length(z_c))
        defDim(ds_out, "time", length(t_idx))

        defVar(ds_out, "XC", lon_c[ilon_c], ("XC",))
        defVar(ds_out, "YC", lat_c[ilat_c], ("YC",))
        defVar(ds_out, "Z", z_c, ("Z",))
        defVar(ds_out, "time", times[t_idx], ("time",))

        var_files = [
            (:THETA, :temperature, ("XC", "YC", "Z", "time")),
            (:SALT, :salinity, ("XC", "YC", "Z", "time")),
            (:UVEL, :u_velocity, ("XC", "YC", "Z", "time")),
            (:VVEL, :v_velocity, ("XC", "YC", "Z", "time")),
        ]

        for (vname_nc, sym, dims) in var_files
            fp = joinpath(dataset.dir, resolve_bsose_filename(dataset.dir, sym, dataset.iteration))
            if isfile(fp)
                @info " -> Extracting $vname_nc..."
                NCDataset(fp) do ds_in
                    v = ds_in[string(vname_nc)][ilon_c, ilat_c, :, t_idx]
                    v_out = defVar(ds_out, string(vname_nc), eltype(v), dims)
                    v_out[:, :, :, :] = v
                end
            end
        end

        for (vname_nc, sym) in [(:oceTAUX, :u_wind_stress), (:oceTAUY, :v_wind_stress)]
            fp = joinpath(dataset.dir, resolve_bsose_filename(dataset.dir, sym, dataset.iteration))
            if isfile(fp)
                @info " -> Extracting $vname_nc..."
                NCDataset(fp) do ds_in
                    v = ds_in[string(vname_nc)][ilon_c, ilat_c, t_idx]
                    v_out = defVar(ds_out, string(vname_nc), eltype(v), ("XC", "YC", "time"))
                    v_out[:, :, :] = v
                end
            end
        end
    end

    @info "Unified BSOSE NetCDF dataset successfully created: $output_path"
    return output_path
end

# ==============================================================================
# 6. Sponge Layer Relaxation Forcing Helper (Alternative or complement to OBCs)
# ==============================================================================

"""
    bsose_sponge_forcing(grid;
                         dataset = BSOSEMonthly(),
                         dates = all_dates(dataset, :temperature)[1:12],
                         rate = 1 / 30days,
                         sponge_distance = 2.0)

Construct 3D relaxation forcings nudging the model toward BSOSE within `sponge_distance` degrees
of the domain boundaries.
"""
function bsose_sponge_forcing(grid;
    dataset=BSOSEMonthly(),
    dates=all_dates(dataset, :temperature)[1:12],
    rate=1 / 30days,
    sponge_distance=2.0)
    @info "Setting up BSOSE sponge layer nudging (timescale = $(prettytime(1/rate)), width = $(sponge_distance)°)..."

    u_target = FieldTimeSeries(Metadata(:u_velocity; dataset, dates), grid)
    v_target = FieldTimeSeries(Metadata(:v_velocity; dataset, dates), grid)
    T_target = FieldTimeSeries(Metadata(:temperature; dataset, dates), grid)
    S_target = FieldTimeSeries(Metadata(:salinity; dataset, dates), grid)

    g = grid isa ImmersedBoundaryGrid ? grid.underlying_grid : grid
    λ₁, λ₂ = extrema(λnodes(g, Face(), Center(), Center()))
    φ₁, φ₂ = extrema(φnodes(g, Center(), Face(), Center()))

    @inline function sponge_mask(λ, φ, z)
        west_ramp = λ < λ₁ + sponge_distance ? (λ₁ + sponge_distance - λ) / sponge_distance : 0.0
        east_ramp = λ > λ₂ - sponge_distance ? (λ - (λ₂ - sponge_distance)) / sponge_distance : 0.0
        south_ramp = φ < φ₁ + sponge_distance ? (φ₁ + sponge_distance - φ) / sponge_distance : 0.0
        north_ramp = φ > φ₂ - sponge_distance ? (φ - (φ₂ - sponge_distance)) / sponge_distance : 0.0
        return clamp(west_ramp + east_ramp + south_ramp + north_ramp, 0.0, 1.0)
    end

    forcings = (
        u=Relaxation(rate=rate, mask=sponge_mask, target=u_target),
        v=Relaxation(rate=rate, mask=sponge_mask, target=v_target),
        T=Relaxation(rate=rate, mask=sponge_mask, target=T_target),
        S=Relaxation(rate=rate, mask=sponge_mask, target=S_target)
    )

    return forcings
end
