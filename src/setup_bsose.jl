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

bsose_iteration_tokens(iteration::Int) = ("i$(iteration)", "$(iteration)")

"""
    bsose_names_iteration(filename, iteration)

Whether `filename` identifies itself as belonging to `iteration`. Used to avoid
trusting a file that was only matched by its variable name.
"""
bsose_names_iteration(filename, iteration::Int) =
    any(t -> occursin(t, lowercase(filename)), bsose_iteration_tokens(iteration))

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

    # Fallback directory search matching the variable token, preferring files that
    # also name this iteration. Without the iteration preference, a directory
    # holding both iterations would silently serve iteration 105 data to an
    # iteration 156 run.
    if isdir(dir)
        t_low = lowercase(shortname)
        matches = filter(readdir(dir)) do f
            endswith(f, ".nc") && occursin(t_low, lowercase(f))
        end

        for token in bsose_iteration_tokens(iteration)
            for f in matches
                occursin(token, lowercase(f)) && return f
            end
        end

        isempty(matches) || return first(matches)
    end

    return iteration == 156 ? "$(titlecase(shortname))_bsoseI156_2013to2024_monthly.nc" : "bsose_i105_2008to2012_monthly_$(titlecase(shortname)).nc"
end

# ==============================================================================
# 2. NumericalEarth.DataWrangling Interface Extensions
#
# BSOSE is an MITgcm product on a lat-lon grid that is uniform in longitude and
# variably spaced (Mercator-like) in latitude. We mirror the ECCO pipeline in
# NumericalEarth: build a Field on the NATIVE BSOSE grid, mark land with NaN
# using the model's own hFac masks, let NumericalEarth inpaint on that native
# grid, and only then interpolate onto the Oceananigans target grid. Inpainting
# before interpolation is what keeps the bilinear stencil from ever mixing a
# land value into an ocean cell.
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

# BSOSE stores the vertical dimension surface-to-bottom (Z[1] = -2.1 m).
# Oceananigans indexes bottom-to-surface, so `retrieve_data` reverses dim 3.
DataWrangling.reversed_vertical_axis(::BSOSEDataset) = true

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

# ------------------------------------------------------------------------------
# 2a. Native grid geometry
#
# Each iteration ships an MITgcm grid file holding 2D coordinate arrays and the
# hFac masks, and is used when present. Iteration 105 additionally carries 1D
# coordinates and its matching hFac inside every data file, which serves as a
# fallback. `BSOSEGeometry` normalizes either source into 1D node vectors plus
# the interface vectors Oceananigans needs.
# ------------------------------------------------------------------------------

struct BSOSEGeometry
    λc::Vector{Float64}   # Nx    longitude cell centers   (XC)
    λf::Vector{Float64}   # Nx    longitude western faces  (XG)
    φc::Vector{Float64}   # Ny    latitude cell centers    (YC)
    φf::Vector{Float64}   # Ny    latitude southern faces  (YG)
    λi::Vector{Float64}   # Nx+1  longitude interfaces
    φi::Vector{Float64}   # Ny+1  latitude interfaces
    zi::Vector{Float64}   # Nz+1  z interfaces, bottom-first
    Nx::Int
    Ny::Int
    Nz::Int
end

const BSOSE_GEOMETRY_CACHE = Dict{Tuple{String,Int},BSOSEGeometry}()
const BSOSE_HFAC_CACHE = Dict{Tuple{String,Int,String},BitArray{3}}()

"""
    bsose_grid_file(dataset)

Path to the MITgcm grid file matching `dataset`, or `nothing` when none is present.

Each BSOSE iteration ships its own grid file at its own resolution (iteration 156
is 1/6°, iteration 105 is 1/3°), and they routinely sit in the same directory.
Candidates are therefore checked against the dimensions the data files advertise,
so a grid file is never paired with data at a different resolution.
"""
function bsose_grid_file(dataset::BSOSEDataset)
    candidates = dataset.iteration == 156 ?
                 ["grid.nc", "grid156.nc", "grid_i156.nc"] :
                 ["grid$(dataset.iteration).nc", "grid_i$(dataset.iteration).nc", "grid.nc"]

    expected = bsose_data_file_size(dataset)

    for name in candidates
        path = joinpath(dataset.dir, name)
        isfile(path) || continue
        actual = bsose_grid_file_size(path)
        isnothing(actual) && continue
        (isnothing(expected) || actual == expected) && return path
    end

    return nothing
end

# Horizontal dimensions advertised by a BSOSE data file, or `nothing` when it
# carries no coordinate variables. Handles both the 1D and 2D coordinate layouts.
function bsose_data_file_size(dataset::BSOSEDataset)
    filename = resolve_bsose_filename(dataset.dir, :temperature, dataset.iteration)

    # A file matched only on its variable name may belong to another iteration, in
    # which case its dimensions say nothing about this one.
    bsose_names_iteration(filename, dataset.iteration) || return nothing

    path = joinpath(dataset.dir, filename)
    isfile(path) || return nothing
    ds = Dataset(path)
    try
        (haskey(ds, "XC") && haskey(ds, "YC")) || return nothing
        xc, yc = ds["XC"], ds["YC"]
        return (size(xc, 1), ndims(yc) == 1 ? length(yc) : size(yc, 2))
    catch
        return nothing
    finally
        close(ds)
    end
end

function bsose_grid_file_size(path::AbstractString)
    ds = Dataset(path)
    try
        haskey(ds, "XC") || return nothing
        return (size(ds["XC"], 1), size(ds["YC"], 2))
    catch
        return nothing
    finally
        close(ds)
    end
end

"""
    strictly_increasing(v)

Whether every step of `v` is positive. Used to reject the zero-padded rows and
columns that BSOSE's iteration 156 `grid.nc` carries in its `XG`/`YG` arrays.
"""
strictly_increasing(v) = all(>(0), diff(v))

"""
    separable_slice(A, dim, label)

Reduce the 2D coordinate array `A` from `grid.nc` to the 1D vector that varies
along `dim`, taking the first slice that is strictly increasing.

`XC`/`YC` are perfectly separable, but `XG`/`YG` in the iteration 156 `grid.nc`
are zero-padded across most of their rows/columns, so a naive `A[:, 1]` silently
yields a broken axis. Scanning for a monotonic slice and erroring when none
exists turns that into a loud failure at setup instead of a blowup at runtime.
"""
function separable_slice(A::AbstractMatrix, dim::Int, label::AbstractString)
    n = dim == 1 ? size(A, 2) : size(A, 1)
    for s in 1:n
        v = Float64.(dim == 1 ? A[:, s] : A[s, :])
        strictly_increasing(v) && return v
    end
    error("No strictly increasing slice found for $label in the BSOSE grid file; it may be corrupt.")
end

function bsose_geometry(dataset::BSOSEDataset)
    key = (dataset.dir, dataset.iteration)
    haskey(BSOSE_GEOMETRY_CACHE, key) && return BSOSE_GEOMETRY_CACHE[key]

    grid_file = bsose_grid_file(dataset)

    if !isnothing(grid_file)
        geom = bsose_geometry_from_grid_file(grid_file)
    elseif dataset.iteration == 156
        error("""
              BSOSE iteration 156 requires an MITgcm grid file in $(dataset.dir).
              It supplies the native coordinates and the hFacC/hFacW/hFacS land masks,
              neither of which is present in the iteration 156 data files.
              Expected one of: grid.nc, grid156.nc, grid_i156.nc.
              """)
    else
        geom = bsose_geometry_from_data_files(dataset)
    end

    BSOSE_GEOMETRY_CACHE[key] = geom
    return geom
end

function bsose_geometry_from_grid_file(grid_file::AbstractString)
    ds = Dataset(grid_file)
    try
        λc = load_coord(ds, "XC", 1)
        φc = load_coord(ds, "YC", 2)
        φf = load_coord(ds, "YG", 2)

        Nx = length(λc)
        Ny = length(φc)

        # `XG` is zero-padded in grid.nc, so rebuild the longitude faces from the
        # uniform lattice BSOSE actually uses and check them against XC.
        λf = bsose_reconstructed_longitude_faces(λc)

        zi = bsose_z_interfaces_from(ds)

        return bsose_build_geometry(λc, λf, φc, φf, zi, Nx, Ny)
    finally
        close(ds)
    end
end

# Coordinate arrays appear as 1D vectors in BSOSE data files and as 2D arrays in
# the grid files; read either as the 1D axis varying along `dim`.
function load_coord(ds, name::AbstractString, dim::Int)
    v = ds[name]
    ndims(v) == 1 && return Float64.(v[:])
    return separable_slice(v[:, :], dim, name)
end

function bsose_geometry_from_data_files(dataset::BSOSEDataset)
    dir = dataset.dir
    center_path = joinpath(dir, resolve_bsose_filename(dir, :temperature, dataset.iteration))
    isfile(center_path) || error("Cannot resolve BSOSE geometry: $center_path not found.")

    ds = Dataset(center_path)
    local λc, φc, zi
    try
        λc = load_coord(ds, "XC", 1)
        φc = load_coord(ds, "YC", 2)
        zi = bsose_z_interfaces_from(ds)
    finally
        close(ds)
    end

    Nx = length(λc)
    Ny = length(φc)

    λf = bsose_reconstructed_longitude_faces(λc)
    φf = bsose_latitude_faces_from_data_files(dir, dataset.iteration, φc)

    return bsose_build_geometry(λc, λf, φc, φf, zi, Nx, Ny)
end

# Read `YG` from the v-velocity file, which is the only iteration 105 file that
# carries it. Fall back to midpoints between centers when that file is absent.
function bsose_latitude_faces_from_data_files(dir, iteration, φc)
    v_path = joinpath(dir, resolve_bsose_filename(dir, :v_velocity, iteration))
    if isfile(v_path)
        ds = Dataset(v_path)
        try
            if haskey(ds, "YG")
                φf = load_coord(ds, "YG", 2)
                if length(φf) == length(φc) && strictly_increasing(φf)
                    return φf
                end
            end
        finally
            close(ds)
        end
    end

    @warn "Falling back to center-derived BSOSE latitude faces; YG was unavailable."
    interfaces = centers_to_interfaces(φc)
    return Float64.(interfaces[1:end-1])
end

# BSOSE is uniform in longitude, so the western faces are an exact lattice.
function bsose_reconstructed_longitude_faces(λc)
    Nx = length(λc)
    Δλ = 360 / Nx
    λf = collect(range(0.0, step=Δλ, length=Nx))

    # The reconstruction must place every center at its cell midpoint. BSOSE stores
    # coordinates at Float32 precision and accumulates rounding along the axis, so
    # compare in fractions of a cell rather than in absolute degrees.
    deviation = maximum(abs, λc .- (λf .+ Δλ / 2)) / Δλ
    deviation < 0.05 || error("BSOSE longitude centers are not on a uniform $(Δλ)° lattice " *
                              "(max deviation $(round(100 * deviation, digits=2))% of a cell); " *
                              "cannot reconstruct faces.")
    return λf
end

function bsose_z_interfaces_from(ds)
    if haskey(ds, "RF")
        rf = Float64.(ds["RF"][:])          # 0 at the surface, descending
        return reverse(rf)                   # bottom-first for Oceananigans
    elseif haskey(ds, "drF")
        drf = Float64.(ds["drF"][:])
        return reverse([0.0; -cumsum(drf)])
    elseif haskey(ds, "DRF")
        drf = Float64.(ds["DRF"][:])
        return reverse([0.0; -cumsum(drf)])
    else
        error("BSOSE file provides neither RF nor drF; cannot build z interfaces.")
    end
end

function bsose_build_geometry(λc, λf, φc, φf, zi, Nx, Ny)
    length(φf) == Ny || error("BSOSE latitude faces have length $(length(φf)), expected $Ny.")
    strictly_increasing(φf) || error("BSOSE latitude faces are not strictly increasing.")
    strictly_increasing(zi) || error("BSOSE z interfaces are not strictly increasing.")

    Δλ = 360 / Nx
    λi = [λf; λf[end] + Δλ]

    # `φf[j]` is the southern face of cell j, so the northern face of the last
    # cell has to be extrapolated from its center.
    φi = [φf; φf[end] + 2 * (φc[end] - φf[end])]
    strictly_increasing(φi) || error("BSOSE latitude interfaces are not strictly increasing.")

    return BSOSEGeometry(λc, λf, φc, φf, λi, φi, zi, Nx, Ny, length(zi) - 1)
end

DataWrangling.longitude_interfaces(metadata::BSOSEMetadata) = bsose_geometry(metadata.dataset).λi
DataWrangling.latitude_interfaces(metadata::BSOSEMetadata) = bsose_geometry(metadata.dataset).φi
DataWrangling.z_interfaces(metadata::BSOSEMetadata) = bsose_geometry(metadata.dataset).zi

function Base.size(metadata::Metadata{<:BSOSEDataset})
    geom = bsose_geometry(metadata.dataset)
    Nz = is_three_dimensional(metadata) ? geom.Nz : 1
    Nt = metadata.dates isa AbstractArray ? length(metadata.dates) : 1
    return (geom.Nx, geom.Ny, Nz, Nt)
end

# Node coordinates at the variable's own C-grid location, matching how
# `set_region_data!` aligns the raw file array against the native field.
function DataWrangling.read_file_coords(metadatum::BSOSEMetadatum)
    geom = bsose_geometry(metadatum.dataset)
    loc = dataset_location(metadatum.dataset, metadatum.name)
    λ = loc[1] === Face ? geom.λf : geom.λc
    φ = loc[2] === Face ? geom.φf : geom.φc
    return copy(λ), copy(φ)
end

# ------------------------------------------------------------------------------
# 2b. Land masks from hFac
#
# MITgcm's hFac is the wet fraction of a cell: 0 is land, and anything above 0 is
# ocean (BSOSE uses partial cells extensively, so testing `> 0` rather than `== 1`
# matters). Masking on hFac instead of on the value itself is what lets a genuine
# zero velocity survive as data rather than being mistaken for land.
# ------------------------------------------------------------------------------

function bsose_hfac_name(metadatum::BSOSEMetadatum)
    loc = dataset_location(metadatum.dataset, metadatum.name)
    loc[1] === Face && return "hFacW"
    loc[2] === Face && return "hFacS"
    return "hFacC"
end

function bsose_hfac_candidate_paths(dataset::BSOSEDataset, hfac_name)
    dir = dataset.dir
    var = hfac_name == "hFacW" ? :u_velocity :
          hfac_name == "hFacS" ? :v_velocity : :temperature
    data_file = joinpath(dir, resolve_bsose_filename(dir, var, dataset.iteration))
    grid_file = bsose_grid_file(dataset)

    # Prefer the grid file, which holds all three hFac arrays; iteration 105 data
    # files each carry their own as a fallback.
    return isnothing(grid_file) ? [data_file] : [grid_file, data_file]
end

"""
    bsose_wet_mask(dataset, hfac_name)

Boolean array over the native BSOSE grid, `true` where the cell is ocean.
Cached per dataset because the underlying hFac arrays are hundreds of megabytes.
"""
function bsose_wet_mask(dataset::BSOSEDataset, hfac_name::AbstractString)
    key = (dataset.dir, dataset.iteration, hfac_name)
    haskey(BSOSE_HFAC_CACHE, key) && return BSOSE_HFAC_CACHE[key]

    for path in bsose_hfac_candidate_paths(dataset, hfac_name)
        isfile(path) || continue
        ds = Dataset(path)
        try
            haskey(ds, hfac_name) || continue
            raw = ds[hfac_name][:, :, :]
            mask = BitArray(undef, size(raw))
            @inbounds for i in eachindex(raw)
                val = raw[i]
                mask[i] = !ismissing(val) && !isnan(val) && val > 0
            end
            BSOSE_HFAC_CACHE[key] = mask
            return mask
        finally
            close(ds)
        end
    end

    error("Could not find $hfac_name for $(summary(dataset)). Expected it in grid.nc " *
          "or in the corresponding BSOSE data file under $(dataset.dir).")
end

# ------------------------------------------------------------------------------
# 2c. Data retrieval and inpainting
# ------------------------------------------------------------------------------

# NaN is what `compute_mask` already treats as missing, so writing NaN into land
# cells is all it takes for NumericalEarth to inpaint them.
DataWrangling.default_mask_value(::BSOSEDataset) = NaN

DataWrangling.default_inpainting(::BSOSEMetadata) = DataWrangling.NearestNeighborInpainting(Inf)
DataWrangling.default_inpainting(::BSOSEMetadatum) = DataWrangling.NearestNeighborInpainting(Inf)

# Inpainted native fields are large intermediates, so keep them out of the data
# directory proper.
function bsose_temp_directory(dataset::BSOSEDataset)
    dir = joinpath(dataset.dir, "temp")
    isdir(dir) || mkpath(dir)
    return dir
end

function DataWrangling.inpainted_metadata_path(metadata::BSOSEMetadatum)
    geom = bsose_geometry(metadata.dataset)
    dstr = metadata.dates isa Dates.AbstractDateTime ? Dates.format(metadata.dates, "yyyymmdd") : "all"
    name = "bsose_inpainted_i$(metadata.dataset.iteration)_$(metadata.name)_$(dstr)_$(geom.Nx)x$(geom.Ny).jld2"
    return joinpath(bsose_temp_directory(metadata.dataset), name)
end

function DataWrangling.inpainted_metadata_path(metadata::BSOSEMetadata)
    geom = bsose_geometry(metadata.dataset)
    start_str = Dates.format(first(metadata.dates), "yyyymmdd")
    end_str = Dates.format(last(metadata.dates), "yyyymmdd")
    name = "bsose_inpainted_i$(metadata.dataset.iteration)_$(metadata.name)_$(start_str)_to_$(end_str)_$(geom.Nx)x$(geom.Ny).jld2"
    return joinpath(bsose_temp_directory(metadata.dataset), name)
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
    if dataset.iteration == 156
        return collect(DateTime(2013, 1, 1):Month(1):DateTime(2024, 12, 1))
    else
        return collect(DateTime(2008, 1, 1):Month(1):DateTime(2012, 12, 1))
    end
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

    for (i, t) in enumerate(time_raw)
        dt = DateTime(t)
        if Dates.year(dt) == target_y && Dates.month(dt) == target_m
            return i
        end
    end

    diffs = [abs((DateTime(t) - target_dt).value) for t in time_raw]
    return argmin(diffs)
end

function DataWrangling.retrieve_data(metadatum::BSOSEMetadatum)
    path = metadata_path(metadatum)
    name = dataset_variable_name(metadatum)

    ds = Dataset(path)
    local raw, time_idx
    try
        var_key = haskey(ds, name) ? name :
                  haskey(ds, uppercase(name)) ? uppercase(name) :
                  haskey(ds, titlecase(name)) ? titlecase(name) : name
        haskey(ds, var_key) || error("Variable $name not found in $path.")

        time_idx = find_bsose_time_index(ds, metadatum.dates)
        raw = is_three_dimensional(metadatum) ? ds[var_key][:, :, :, time_idx] :
              ds[var_key][:, :, time_idx]
    finally
        close(ds)
    end

    wet = bsose_wet_mask(metadatum.dataset, bsose_hfac_name(metadatum))

    if is_three_dimensional(metadatum)
        size(raw) == size(wet) || error("BSOSE $name has size $(size(raw)) but its hFac mask " *
                                        "has size $(size(wet)); grid.nc does not match the data files.")
        data = Array{Float32}(undef, size(raw))
        @inbounds for i in eachindex(raw)
            val = raw[i]
            data[i] = (!wet[i] || ismissing(val) || isnan(val)) ? NaN32 : Float32(val)
        end
        return reverse(data, dims=3)
    else
        # Surface forcing lives on the top model level, so mask it with k = 1 of hFac.
        surface = view(wet, :, :, 1)
        size(raw) == size(surface) || error("BSOSE $name has size $(size(raw)) but its surface hFac " *
                                            "mask has size $(size(surface)).")
        data = Array{Float32}(undef, size(raw))
        @inbounds for i in eachindex(raw)
            val = raw[i]
            data[i] = (!surface[i] || ismissing(val) || isnan(val)) ? NaN32 : Float32(val)
        end
        return data
    end
end

"""
    bsose_region(grid; padding = 1.0)

`BoundingBox` covering `grid` widened by `padding` degrees, for restricting BSOSE
reads to the model domain. Without it every read materializes and inpaints the
full circumpolar native grid, which at iteration 156 is 2160×588×52.
"""
function bsose_region(grid; padding=1.0)
    underlying = grid isa ImmersedBoundaryGrid ? grid.underlying_grid : grid
    return DataWrangling.BoundingBox(underlying; padding)
end

# ==============================================================================
# 3. Boundary Slice Extraction & Open Boundary Conditions (OBCs)
# ==============================================================================

# Southern Ocean physical parameter bounds
const S_MIN_PHYSICAL = 30.0
const S_MAX_PHYSICAL = 36.5
const T_MIN_PHYSICAL = -2.2
const T_MAX_PHYSICAL = 20.0

"""
    fill_bathymetry_gaps!(data, valid_min, valid_max)

When the model bathymetry is deeper than the BSOSE bathymetry (or at steep slopes),
regridding leaves missing/unphysical values in cells that are ocean in the model
but land/unmapped in BSOSE. Rather than inserting arbitrary bounds, we propagate
the nearest valid physical ocean values downwards (and laterally) through the water column.
"""
function fill_bathymetry_gaps!(data, valid_min=S_MIN_PHYSICAL, valid_max=S_MAX_PHYSICAL)
    Nx, Ny, Nz = size(data)
    cpu_data = Array(data)

    # 1. Vertical propagation: extend the deepest valid water mass downward
    for i in 1:Nx, j in 1:Ny
        # Downward pass: fill deep cells from the valid ocean level directly above
        for k in Nz-1:-1:1
            val = cpu_data[i, j, k]
            if val < valid_min || val > valid_max || isnan(val)
                above = cpu_data[i, j, k+1]
                if valid_min <= above <= valid_max && !isnan(above)
                    cpu_data[i, j, k] = above
                end
            end
        end
        # Upward pass: in case surface level had missing data
        for k in 2:Nz
            val = cpu_data[i, j, k]
            if val < valid_min || val > valid_max || isnan(val)
                below = cpu_data[i, j, k-1]
                if valid_min <= below <= valid_max && !isnan(below)
                    cpu_data[i, j, k] = below
                end
            end
        end
    end

    # 2. Horizontal propagation: fill any entirely isolated columns from adjacent horizontal neighbors
    for k in 1:Nz
        for i in 1:Nx
            for j in 2:Ny
                val = cpu_data[i, j, k]
                if val < valid_min || val > valid_max || isnan(val)
                    prev = cpu_data[i, j-1, k]
                    if valid_min <= prev <= valid_max && !isnan(prev)
                        cpu_data[i, j, k] = prev
                    end
                end
            end
            for j in Ny-1:-1:1
                val = cpu_data[i, j, k]
                if val < valid_min || val > valid_max || isnan(val)
                    nxt = cpu_data[i, j+1, k]
                    if valid_min <= nxt <= valid_max && !isnan(nxt)
                        cpu_data[i, j, k] = nxt
                    end
                end
            end
        end
        for j in 1:Ny
            for i in 2:Nx
                val = cpu_data[i, j, k]
                if val < valid_min || val > valid_max || isnan(val)
                    prev = cpu_data[i-1, j, k]
                    if valid_min <= prev <= valid_max && !isnan(prev)
                        cpu_data[i, j, k] = prev
                    end
                end
            end
            for i in Nx-1:-1:1
                val = cpu_data[i, j, k]
                if val < valid_min || val > valid_max || isnan(val)
                    nxt = cpu_data[i+1, j, k]
                    if valid_min <= nxt <= valid_max && !isnan(nxt)
                        cpu_data[i, j, k] = nxt
                    end
                end
            end
        end
    end

    # Copy filled values back to original array buffer
    copyto!(data, cpu_data)
    return data
end

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
    region = bsose_region(grid)
    taux_fts = FieldTimeSeries(Metadata(:zonal_wind_stress; dataset, dates, region), grid)
    tauy_fts = FieldTimeSeries(Metadata(:meridional_wind_stress; dataset, dates, region), grid)

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
- `scheme`: Matching scheme for the open boundaries. Defaults to `nothing`, which
            imposes the prescribed BSOSE values directly. `PerturbationAdvection()`
            is NOT usable here: it drives the boundary faces to several times the
            prescribed velocity and the run goes to NaN within four time steps.
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
    scheme=nothing,
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
        region = bsose_region(grid)
        @info " -> Loading BSOSE u_velocity..."
        u_fts = FieldTimeSeries(Metadata(:u_velocity; dataset, dates, region), grid)
        @info " -> Loading BSOSE v_velocity..."
        v_fts = FieldTimeSeries(Metadata(:v_velocity; dataset, dates, region), grid)
        @info " -> Loading BSOSE temperature..."
        T_fts = FieldTimeSeries(Metadata(:temperature; dataset, dates, region), grid)
        @info " -> Loading BSOSE salinity..."
        S_fts = FieldTimeSeries(Metadata(:salinity; dataset, dates, region), grid)

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

        # Sanitize boundary velocity and tracer slices so bathymetric gaps are filled with nearby valid ocean values
        for (bts, lo, hi) in ((u_west, -5.0, 5.0),
            (u_east, -5.0, 5.0),
            (u_south, -5.0, 5.0),
            (u_north, -5.0, 5.0),
            (v_west, -5.0, 5.0),
            (v_east, -5.0, 5.0),
            (v_south, -5.0, 5.0),
            (v_north, -5.0, 5.0),
            (T_west, T_MIN_PHYSICAL, T_MAX_PHYSICAL),
            (T_east, T_MIN_PHYSICAL, T_MAX_PHYSICAL),
            (T_south, T_MIN_PHYSICAL, T_MAX_PHYSICAL),
            (T_north, T_MIN_PHYSICAL, T_MAX_PHYSICAL),
            (S_west, S_MIN_PHYSICAL, S_MAX_PHYSICAL),
            (S_east, S_MIN_PHYSICAL, S_MAX_PHYSICAL),
            (S_south, S_MIN_PHYSICAL, S_MAX_PHYSICAL),
            (S_north, S_MIN_PHYSICAL, S_MAX_PHYSICAL))
            if bts isa FieldTimeSeries
                for t in 1:length(bts.times)
                    fill_bathymetry_gaps!(parent(bts[t]), lo, hi)
                end
            end
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

    # Ensure all boundary slices match the grid architecture (e.g., GPU/CuArray)
    arch = architecture(grid)
    u_west = u_west isa FieldTimeSeries ? on_architecture(arch, u_west) : u_west
    u_east = u_east isa FieldTimeSeries ? on_architecture(arch, u_east) : u_east
    u_south = u_south isa FieldTimeSeries ? on_architecture(arch, u_south) : u_south
    u_north = u_north isa FieldTimeSeries ? on_architecture(arch, u_north) : u_north

    v_west = v_west isa FieldTimeSeries ? on_architecture(arch, v_west) : v_west
    v_east = v_east isa FieldTimeSeries ? on_architecture(arch, v_east) : v_east
    v_south = v_south isa FieldTimeSeries ? on_architecture(arch, v_south) : v_south
    v_north = v_north isa FieldTimeSeries ? on_architecture(arch, v_north) : v_north

    T_west = T_west isa FieldTimeSeries ? on_architecture(arch, T_west) : T_west
    T_east = T_east isa FieldTimeSeries ? on_architecture(arch, T_east) : T_east
    T_south = T_south isa FieldTimeSeries ? on_architecture(arch, T_south) : T_south
    T_north = T_north isa FieldTimeSeries ? on_architecture(arch, T_north) : T_north

    S_west = S_west isa FieldTimeSeries ? on_architecture(arch, S_west) : S_west
    S_east = S_east isa FieldTimeSeries ? on_architecture(arch, S_east) : S_east
    S_south = S_south isa FieldTimeSeries ? on_architecture(arch, S_south) : S_south
    S_north = S_north isa FieldTimeSeries ? on_architecture(arch, S_north) : S_north

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
            south=ValueBoundaryCondition(u_south; scheme),
            north=ValueBoundaryCondition(u_north; scheme),
            top=top_u_bc
        )

        v_bcs = FieldBoundaryConditions(
            south=NormalFlowBoundaryCondition(v_south; scheme),
            north=NormalFlowBoundaryCondition(v_north; scheme),
            top=top_v_bc
        )

        T_bcs = FieldBoundaryConditions(
            south=ValueBoundaryCondition(T_south; scheme),
            north=ValueBoundaryCondition(T_north; scheme)
        )

        S_bcs = FieldBoundaryConditions(
            south=ValueBoundaryCondition(S_south; scheme),
            north=ValueBoundaryCondition(S_north; scheme)
        )

    else
        @info " -> Longitude is Bounded (regional): applying West, East, South, and North boundary conditions."
        u_bcs = FieldBoundaryConditions(
            west=NormalFlowBoundaryCondition(u_west; scheme),
            east=NormalFlowBoundaryCondition(u_east; scheme),
            south=ValueBoundaryCondition(u_south; scheme),
            north=ValueBoundaryCondition(u_north; scheme),
            top=top_u_bc
        )

        v_bcs = FieldBoundaryConditions(
            west=ValueBoundaryCondition(v_west; scheme),
            east=ValueBoundaryCondition(v_east; scheme),
            south=NormalFlowBoundaryCondition(v_south; scheme),
            north=NormalFlowBoundaryCondition(v_north; scheme),
            top=top_v_bc
        )

        T_bcs = FieldBoundaryConditions(
            west=ValueBoundaryCondition(T_west; scheme),
            east=ValueBoundaryCondition(T_east; scheme),
            south=ValueBoundaryCondition(T_south; scheme),
            north=ValueBoundaryCondition(T_north; scheme)
        )

        S_bcs = FieldBoundaryConditions(
            west=ValueBoundaryCondition(S_west; scheme),
            east=ValueBoundaryCondition(S_east; scheme),
            south=ValueBoundaryCondition(S_south; scheme),
            north=ValueBoundaryCondition(S_north; scheme)
        )
    end

    @info "BSOSE Open Boundary Conditions setup complete."
    return (u=u_bcs, v=v_bcs, T=T_bcs, S=S_bcs)
end

# ==============================================================================
# 4b. Sponge Layer Boundary Restoring Helper
# ==============================================================================

"""
    bsose_sponge_layer_forcing(grid;
                               dataset = BSOSEMonthly(),
                               dates = nothing,
                               sponge_width = 3.0,
                               timescale = 5days,
                               restore_velocities = true)

Construct sponge layer forcing using `DatasetRestoring` for all open boundaries.
A smooth cosine taper relaxes variables toward BSOSE values within `sponge_width`
degrees of the boundary edges on a timescale of `timescale`, dropping to zero
relaxation in the interior domain.
"""
function bsose_sponge_layer_forcing(grid;
    dataset=BSOSEMonthly(),
    dates=nothing,
    sponge_width=3.0, # degrees
    timescale=5days,
    restore_velocities=true)

    underlying = grid isa ImmersedBoundaryGrid ? grid.underlying_grid : grid
    is_x_periodic = topology(underlying, 1) === Periodic

    λ_min = underlying.λᶠᵃᵃ[1]
    λ_max = underlying.λᶠᵃᵃ[underlying.Nx+1]
    φ_min = underlying.φᵃᶠᵃ[1]
    φ_max = underlying.φᵃᶠᵃ[underlying.Ny+1]

    # Sponge mask: 1 at boundary edge, smoothly tapering to 0 at distance >= sponge_width.
    # Oceananigans forcing kernels call mask functions as either mask(λ, φ, z, t) or mask(λ, φ, z).
    compute_sponge = (λ, φ, z) -> begin
        dist_s = φ - φ_min
        dist_n = φ_max - φ
        d_edge = min(dist_s, dist_n)
        if !is_x_periodic
            dist_w = λ - λ_min
            dist_e = λ_max - λ
            d_edge = min(d_edge, dist_w, dist_e)
        end

        if d_edge >= sponge_width
            return 0.0
        elseif d_edge <= 0.0
            return 1.0
        else
            return 0.5 * (1.0 + cos(π * d_edge / sponge_width))
        end
    end

    sponge_mask = (λ, φ, z, args...) -> compute_sponge(λ, φ, z)

    start_date = dates isa Tuple ? dates[1] : (dates !== nothing ? first(dates) : first_date(dataset, :temperature))
    end_date = dates isa Tuple ? dates[2] : (dates !== nothing ? last(dates) : last_date(dataset, :temperature))

    region = bsose_region(grid)
    rate = 1 / timescale

    @info "Configuring BSOSE Sponge Layers (width=$(sponge_width)°, timescale=$(timescale/86400) days)..."
    T_meta = Metadata(:temperature; dataset, start_date, end_date, region)
    S_meta = Metadata(:salinity; dataset, start_date, end_date, region)

    FT = DatasetRestoring(T_meta, grid; rate=rate, mask=sponge_mask)
    FS = DatasetRestoring(S_meta, grid; rate=rate, mask=sponge_mask)

    if restore_velocities
        @info " -> Restoring u and v velocities in boundary sponge layers to damp accelerations."
        u_meta = Metadata(:u_velocity; dataset, start_date, end_date, region)
        v_meta = Metadata(:v_velocity; dataset, start_date, end_date, region)
        Fu = DatasetRestoring(u_meta, grid; rate=rate, mask=sponge_mask)
        Fv = DatasetRestoring(v_meta, grid; rate=rate, mask=sponge_mask)
        return (u=Fu, v=Fv, T=FT, S=FS)
    else
        return (T=FT, S=FS)
    end
end

# ==============================================================================
# 5. Initial Conditions Helper
# ==============================================================================

"""
    bsose_initial_conditions!(model;
                               dataset = BSOSEMonthly(),
                               date = nothing,
                               dates = nothing,
                               velocities = true)

Initialize `model` tracers (`T`, `S`) from BSOSE at `date`, and by default its
velocities (`u`, `v`) as well.

The ACC carries ~100-150 Sv of zonal transport through this domain, so starting
from rest while the open boundaries inject that transport at t = 0 drives an
artificial divergence spike and barotropic wave pile-up. Initializing `u` and `v`
from the same BSOSE snapshot keeps the interior consistent with the inflow.
"""
function bsose_initial_conditions!(model;
    dataset=BSOSEMonthly(),
    date=nothing,
    dates=nothing,
    velocities=true)
    init_date = date !== nothing ? date :
                dates !== nothing ? (dates isa Tuple ? dates[1] : first(dates)) :
                first_date(dataset, :temperature)

    @info "Initializing model state from BSOSE at date: $init_date (velocities: $velocities)..."
    grid = model.grid
    region = bsose_region(grid)

    T_init = Field(Metadatum(:temperature; dataset, date=init_date, region), grid)
    S_init = Field(Metadatum(:salinity; dataset, date=init_date, region), grid)

    if velocities
        u_init = Field(Metadatum(:u_velocity; dataset, date=init_date, region), grid)
        v_init = Field(Metadatum(:v_velocity; dataset, date=init_date, region), grid)
        set!(model; u=u_init, v=v_init, T=T_init, S=S_init)
    else
        set!(model; T=T_init, S=S_init)
    end

    if velocities
        fill_bathymetry_gaps!(parent(model.velocities.u), -5.0, 5.0)
        fill_bathymetry_gaps!(parent(model.velocities.v), -5.0, 5.0)
        fill_halo_regions!(model.velocities.u, model.clock, fields(model))
        fill_halo_regions!(model.velocities.v, model.clock, fields(model))
        fill_bathymetry_gaps!(parent(model.velocities.u), -5.0, 5.0)
        fill_bathymetry_gaps!(parent(model.velocities.v), -5.0, 5.0)
    end

    # Propagate valid ocean values into bathymetry discrepancies / deep slope trenches
    fill_bathymetry_gaps!(parent(model.tracers.S), S_MIN_PHYSICAL, S_MAX_PHYSICAL)
    fill_bathymetry_gaps!(parent(model.tracers.T), T_MIN_PHYSICAL, T_MAX_PHYSICAL)
    fill_halo_regions!(model.tracers.T, model.clock, fields(model))
    fill_halo_regions!(model.tracers.S, model.clock, fields(model))
    fill_bathymetry_gaps!(parent(model.tracers.S), S_MIN_PHYSICAL, S_MAX_PHYSICAL)
    fill_bathymetry_gaps!(parent(model.tracers.T), T_MIN_PHYSICAL, T_MAX_PHYSICAL)

    @info "Model successfully initialized with BSOSE fields and filled bathymetric gaps."
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

    region = bsose_region(grid)
    u_target = FieldTimeSeries(Metadata(:u_velocity; dataset, dates, region), grid)
    v_target = FieldTimeSeries(Metadata(:v_velocity; dataset, dates, region), grid)
    T_target = FieldTimeSeries(Metadata(:temperature; dataset, dates, region), grid)
    S_target = FieldTimeSeries(Metadata(:salinity; dataset, dates, region), grid)

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
