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
    centers_to_interfaces

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
    dir :: String
    iteration :: Int
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

function BSOSEMonthly(; dir = default_bsose_directory(), iteration = nothing)
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
    :temperature            => "THETA",
    :salinity               => "SALT",
    :u_velocity             => "UVEL",
    :v_velocity             => "VVEL",
    :zonal_wind_stress      => "oceTAUX",
    :meridional_wind_stress => "oceTAUY",
    :surface_heat_flux      => "TFLUX",
    # Aliases
    :Theta                  => "THETA",
    :Salt                   => "SALT",
    :Uvel                   => "UVEL",
    :Vvel                   => "VVEL",
    :oceTAUX                => "oceTAUX",
    :oceTAUY                => "oceTAUY",
    :surfTflx               => "TFLUX",
    :T                      => "THETA",
    :S                      => "SALT",
    :u                      => "UVEL",
    :v                      => "VVEL"
)

# Variable spatial locations on Arakawa C-grid
const BSOSE_LOCATIONS = Dict(
    :temperature            => (Center, Center, Center),
    :salinity               => (Center, Center, Center),
    :u_velocity             => (Face,   Center, Center),
    :v_velocity             => (Center, Face,   Center),
    :zonal_wind_stress      => (Face,   Center, Nothing),
    :meridional_wind_stress => (Center, Face,   Nothing),
    :surface_heat_flux      => (Center, Center, Nothing),
    :Theta                  => (Center, Center, Center),
    :Salt                   => (Center, Center, Center),
    :Uvel                   => (Face,   Center, Center),
    :Vvel                   => (Center, Face,   Center),
    :oceTAUX                => (Face,   Center, Nothing),
    :oceTAUY                => (Center, Face,   Nothing),
    :surfTflx               => (Center, Center, Nothing),
    :T                      => (Center, Center, Center),
    :S                      => (Center, Center, Center),
    :u                      => (Face,   Center, Center),
    :v                      => (Center, Face,   Center)
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

const BSOSEMetadata{D} = Metadata{<:BSOSEDataset, D}
const BSOSEMetadatum   = Metadatum{<:BSOSEDataset}

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
DataWrangling.latitude_interfaces(::BSOSEMetadata)  = (-78.0, -29.7)

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
            data[i] = (ismissing(val) || isnan(val)) ? NaN32 : Float32(val)
        end
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
        bts = FieldTimeSeries{Nothing, LY, LZ}(grid, times; indices=(1, :, :), time_indexing)
        for t in 1:length(times)
            interior(bts[t], 1, :, :) .= interior(fts[t], 1, :, :)
            fill_halo_regions!(bts[t])
        end
        return bts
    elseif side === :east
        # East boundary: i = Nx (Center) or Nx+1 (Face), location = (Nothing, LY, LZ)
        i_east = LX === Face ? Nx + 1 : Nx
        bts = FieldTimeSeries{Nothing, LY, LZ}(grid, times; indices=(1, :, :), time_indexing)
        for t in 1:length(times)
            interior(bts[t], 1, :, :) .= interior(fts[t], i_east, :, :)
            fill_halo_regions!(bts[t])
        end
        return bts
    elseif side === :south
        # South boundary: j = 1, location = (LX, Nothing, LZ)
        bts = FieldTimeSeries{LX, Nothing, LZ}(grid, times; indices=(:, 1, :), time_indexing)
        for t in 1:length(times)
            interior(bts[t], :, 1, :) .= interior(fts[t], :, 1, :)
            fill_halo_regions!(bts[t])
        end
        return bts
    elseif side === :north
        # North boundary: j = Ny (Center) or Ny+1 (Face), location = (LX, Nothing, LZ)
        j_north = LY === Face ? Ny + 1 : Ny
        bts = FieldTimeSeries{LX, Nothing, LZ}(grid, times; indices=(:, 1, :), time_indexing)
        for t in 1:length(times)
            interior(bts[t], :, 1, :) .= interior(fts[t], :, j_north, :)
            fill_halo_regions!(bts[t])
        end
        return bts
    elseif side === :top
        # Top boundary: k = Nz (Center) or Nz+1 (Face), location = (LX, LY, Nothing)
        bts = FieldTimeSeries{LX, LY, Nothing}(grid, times; indices=(:, :, 1), time_indexing)
        for t in 1:length(times)
            interior(bts[t], :, :, 1) .= interior(fts[t], :, :, 1)
            fill_halo_regions!(bts[t])
        end
        return bts
    else
        throw(ArgumentError("Unknown side: $side. Valid options are :west, :east, :south, :north, :top."))
    end
end

"""
    bsose_open_boundary_conditions(grid;
                                   dataset = BSOSEMonthly(),
                                   dates = all_dates(dataset, :temperature)[1:12],
                                   scheme = nothing,
                                   surface_winds = false,
                                   ρ₀ = 1026.0)

Construct a `NamedTuple` of `FieldBoundaryConditions` `(; u, v, T, S)` configured with
open boundary conditions from BSOSE.

# Arguments
- `grid`: The simulation `LatitudeLongitudeGrid`.
- `dataset`: `BSOSEMonthly()` instance.
- `dates`: Range or collection of `DateTime` dates to load.
- `scheme`: Radiation/matching scheme for open boundaries. Can be `nothing` (clamped Dirichlet)
            or an instance of `Oceananigans.BoundaryConditions.PerturbationAdvection`.
- `surface_winds`: If `true`, loads BSOSE `oceTAUX` and `oceTAUY` and sets top `FluxBoundaryCondition`
                   with kinematic stress `τ / ρ₀`.
- `ρ₀`: Seawater density for wind stress conversion (kg/m³). Default: 1026.0.
"""
function bsose_open_boundary_conditions(grid;
                                       dataset = BSOSEMonthly(),
                                       dates = all_dates(dataset, :temperature)[1:12],
                                       scheme = nothing,
                                       surface_winds = false,
                                       ρ₀ = 1026.0)
    @info "Setting up BSOSE Open Boundary Conditions..."
    
    # 1. Load 3D FieldTimeSeries for state variables
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
    u_west  = boundary_slice_time_series(u_fts, :west)
    u_east  = boundary_slice_time_series(u_fts, :east)
    u_south = boundary_slice_time_series(u_fts, :south)
    u_north = boundary_slice_time_series(u_fts, :north)

    v_west  = boundary_slice_time_series(v_fts, :west)
    v_east  = boundary_slice_time_series(v_fts, :east)
    v_south = boundary_slice_time_series(v_fts, :south)
    v_north = boundary_slice_time_series(v_fts, :north)

    T_west  = boundary_slice_time_series(T_fts, :west)
    T_east  = boundary_slice_time_series(T_fts, :east)
    T_south = boundary_slice_time_series(T_fts, :south)
    T_north = boundary_slice_time_series(T_fts, :north)

    S_west  = boundary_slice_time_series(S_fts, :west)
    S_east  = boundary_slice_time_series(S_fts, :east)
    S_south = boundary_slice_time_series(S_fts, :south)
    S_north = boundary_slice_time_series(S_fts, :north)

    # 3. Top boundary condition (surface wind stress)
    top_u_bc = FluxBoundaryCondition(nothing)
    top_v_bc = FluxBoundaryCondition(nothing)

    if surface_winds
        @info " -> Loading BSOSE surface wind stress (oceTAUX, oceTAUY)..."
        taux_fts = FieldTimeSeries(Metadata(:zonal_wind_stress; dataset, dates), grid)
        tauy_fts = FieldTimeSeries(Metadata(:meridional_wind_stress; dataset, dates), grid)
        
        # Scale by 1 / ρ₀ to get kinematic momentum flux
        for t in 1:length(dates)
            interior(taux_fts[t]) ./= ρ₀
            interior(tauy_fts[t]) ./= ρ₀
        end

        top_u_slice = boundary_slice_time_series(taux_fts, :top)
        top_v_slice = boundary_slice_time_series(tauy_fts, :top)

        top_u_bc = FluxBoundaryCondition(top_u_slice)
        top_v_bc = FluxBoundaryCondition(top_v_slice)
    end

    # 4. Construct FieldBoundaryConditions
    # Note: On East/West boundaries, normal velocity is u (NormalFlow), tangential is v (Value).
    #       On South/North boundaries, normal velocity is v (NormalFlow), tangential is u (Value).
    # If the domain is longitudinally periodic (e.g. Circumpolar), only South and North BCs are applied.
    underlying = grid isa ImmersedBoundaryGrid ? grid.underlying_grid : grid
    is_x_periodic = underlying.topology[1] === Periodic

    if is_x_periodic
        @info " -> Longitude is Periodic (circumpolar): applying South & North boundary conditions."
        u_bcs = FieldBoundaryConditions(
            south = ValueBoundaryCondition(u_south; scheme),
            north = ValueBoundaryCondition(u_north; scheme),
            top   = top_u_bc
        )

        v_bcs = FieldBoundaryConditions(
            south = NormalFlowBoundaryCondition(v_south; scheme),
            north = NormalFlowBoundaryCondition(v_north; scheme),
            top   = top_v_bc
        )

        T_bcs = FieldBoundaryConditions(
            south = ValueBoundaryCondition(T_south; scheme),
            north = ValueBoundaryCondition(T_north; scheme)
        )

        S_bcs = FieldBoundaryConditions(
            south = ValueBoundaryCondition(S_south; scheme),
            north = ValueBoundaryCondition(S_north; scheme)
        )
    else
        @info " -> Longitude is Bounded (regional): applying West, East, South, and North boundary conditions."
        u_bcs = FieldBoundaryConditions(
            west  = NormalFlowBoundaryCondition(u_west; scheme),
            east  = NormalFlowBoundaryCondition(u_east; scheme),
            south = ValueBoundaryCondition(u_south; scheme),
            north = ValueBoundaryCondition(u_north; scheme),
            top   = top_u_bc
        )

        v_bcs = FieldBoundaryConditions(
            west  = ValueBoundaryCondition(v_west; scheme),
            east  = ValueBoundaryCondition(v_east; scheme),
            south = NormalFlowBoundaryCondition(v_south; scheme),
            north = NormalFlowBoundaryCondition(v_north; scheme),
            top   = top_v_bc
        )

        T_bcs = FieldBoundaryConditions(
            west  = ValueBoundaryCondition(T_west; scheme),
            east  = ValueBoundaryCondition(T_east; scheme),
            south = ValueBoundaryCondition(T_south; scheme),
            north = ValueBoundaryCondition(T_north; scheme)
        )

        S_bcs = FieldBoundaryConditions(
            west  = ValueBoundaryCondition(S_west; scheme),
            east  = ValueBoundaryCondition(S_east; scheme),
            south = ValueBoundaryCondition(S_south; scheme),
            north = ValueBoundaryCondition(S_north; scheme)
        )
    end

    @info "BSOSE Open Boundary Conditions setup complete."
    return (u = u_bcs, v = v_bcs, T = T_bcs, S = S_bcs)
end

# ==============================================================================
# 4. Initial Conditions Helper
# ==============================================================================

"""
    bsose_initial_conditions!(model;
                              dataset = BSOSEMonthly(),
                              date = first_date(dataset, :temperature))

Initialize `model` tracers (`T`, `S`) and velocities (`u`, `v`) from BSOSE at `date`.
"""
function bsose_initial_conditions!(model;
                                   dataset = BSOSEMonthly(),
                                   date = first_date(dataset, :temperature))
    @info "Initializing model state from BSOSE at date: $date..."
    grid = model.grid
    
    T_init = Field(Metadatum(:temperature; dataset, date), grid)
    S_init = Field(Metadatum(:salinity; dataset, date), grid)
    u_init = Field(Metadatum(:u_velocity; dataset, date), grid)
    v_init = Field(Metadatum(:v_velocity; dataset, date), grid)

    set!(model; u=u_init, v=v_init, T=T_init, S=S_init)
    @info "Model successfully initialized with BSOSE fields."
    return nothing
end

# ==============================================================================
# 5. Sponge Layer Relaxation Forcing Helper (Alternative or complement to OBCs)
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
                              dataset = BSOSEMonthly(),
                              dates = all_dates(dataset, :temperature)[1:12],
                              rate = 1 / 30days,
                              sponge_distance = 2.0)
    @info "Setting up BSOSE sponge layer nudging (timescale = $(prettytime(1/rate)), width = $(sponge_distance)°)..."
    
    u_target = FieldTimeSeries(Metadata(:u_velocity; dataset, dates), grid)
    v_target = FieldTimeSeries(Metadata(:v_velocity; dataset, dates), grid)
    T_target = FieldTimeSeries(Metadata(:temperature; dataset, dates), grid)
    S_target = FieldTimeSeries(Metadata(:salinity; dataset, dates), grid)

    g = grid isa ImmersedBoundaryGrid ? grid.underlying_grid : grid
    λ₁, λ₂ = extrema(λnodes(g, Face(), Center(), Center()))
    φ₁, φ₂ = extrema(φnodes(g, Center(), Face(), Center()))

    @inline function sponge_mask(λ, φ, z)
        west_ramp  = λ < λ₁ + sponge_distance ? (λ₁ + sponge_distance - λ) / sponge_distance : 0.0
        east_ramp  = λ > λ₂ - sponge_distance ? (λ - (λ₂ - sponge_distance)) / sponge_distance : 0.0
        south_ramp = φ < φ₁ + sponge_distance ? (φ₁ + sponge_distance - φ) / sponge_distance : 0.0
        north_ramp = φ > φ₂ - sponge_distance ? (φ - (φ₂ - sponge_distance)) / sponge_distance : 0.0
        return clamp(west_ramp + east_ramp + south_ramp + north_ramp, 0.0, 1.0)
    end

    forcings = (
        u = Relaxation(rate=rate, mask=sponge_mask, target=u_target),
        v = Relaxation(rate=rate, mask=sponge_mask, target=v_target),
        T = Relaxation(rate=rate, mask=sponge_mask, target=T_target),
        S = Relaxation(rate=rate, mask=sponge_mask, target=S_target)
    )

    return forcings
end
