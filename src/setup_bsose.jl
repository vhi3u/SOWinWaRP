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
using Oceananigans.Grids: topology, znodes
using Oceananigans.BoundaryConditions
using Oceananigans.BoundaryConditions: PerturbationAdvection, NormalRadiation, ObliqueRadiation,
                                       GravityWaveRadiationBoundaryCondition
using Oceananigans.OutputReaders: FieldTimeSeries, Cyclical, TimeSeriesInterpolation
using Oceananigans.Architectures: architecture, CPU, GPU, on_architecture
using Oceananigans.Fields: interior, location, fill_halo_regions!
using Oceananigans.Operators: Δyᶠᶜᶜ, Δxᶜᶠᶜ
using Oceananigans.ImmersedBoundaries: immersed_peripheral_node, immersed_cell
using NumericalEarth
using NumericalEarth.DataWrangling
using NCDatasets
using Dates
using CFTime
using Printf: @sprintf
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
    :surface_salt_flux => "SFLUX",
    # `oceQsw` is the solar part of `TFLUX`, not a flux in addition to it. Both follow the
    # same MITgcm convention ("net Short-Wave radiation (+=down), >0 increases theta"), so
    # TFLUX - oceQsw is the non-solar remainder; see `bsose_penetrating_shortwave`.
    :surface_shortwave => "oceQsw",
    # The variable inside SSH_bsoseI156_*.nc is ETAN, not SSH; see BSOSE_FILE_TOKENS below.
    # Unlike every other field here it marks land with 0.0 rather than NaN -- see
    # `bsose_exterior_free_surface`, which masks from Depth before using it.
    :sea_surface_height => "ETAN",
    # Aliases
    :oceQsw => "oceQsw",
    :Theta => "THETA",
    :Salt => "SALT",
    :Uvel => "UVEL",
    :Vvel => "VVEL",
    :oceTAUX => "oceTAUX",
    :oceTAUY => "oceTAUY",
    :surfTflx => "TFLUX",
    :surfSflx => "SFLUX",
    :T => "THETA",
    :S => "SALT",
    :u => "UVEL",
    :v => "VVEL"
)

# Some BSOSE products name their file with a token that differs from the variable inside it:
# the surface flux files are `surfTflx`/`surfSflx` while the variables they hold are TFLUX/SFLUX.
const BSOSE_FILE_TOKENS = Dict(
    :surface_heat_flux => "surfTflx",
    :surface_salt_flux => "surfSflx",
    :surfTflx => "surfTflx",
    :surfSflx => "surfSflx",
    :sea_surface_height => "SSH",
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
    :surface_salt_flux => (Center, Center, Nothing),
    :surface_shortwave => (Center, Center, Nothing),
    :sea_surface_height => (Center, Center, Nothing),
    :oceQsw => (Center, Center, Nothing),
    :Theta => (Center, Center, Center),
    :Salt => (Center, Center, Center),
    :Uvel => (Face, Center, Center),
    :Vvel => (Center, Face, Center),
    :oceTAUX => (Face, Center, Nothing),
    :oceTAUY => (Center, Face, Nothing),
    :surfTflx => (Center, Center, Nothing),
    :surfSflx => (Center, Center, Nothing),
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

"""
    BSOSE_FILES

The one filename per variable, per iteration: `(token, span)` builds

    iteration 156   <token>_bsoseI156_<span>.nc
    iteration 105   bsose_i105_<span>_<token>.nc

matching how each distribution actually ships. The iteration 156 entries are confirmed
against the cluster's `data/` listing (2026-10-22), not inferred. There is deliberately no
list of candidate spellings and no directory scan. Guessing produced two silent failures in this project -- a
missing iteration 156 file resolving to the iteration 105 file beside it, and a variable
resolving to whatever `.nc` happened to contain its name -- and both were only caught by
noticing wrong numbers much later. If a token here is wrong the run stops and says so.
"""
const BSOSE_FILES = Dict(
    156 => Dict(
        :temperature            => ("Theta",    "2013to2024_monthly"),
        :salinity               => ("Salt",     "2013to2024_monthly"),
        :u_velocity             => ("Uvel",     "2013to2024_monthly"),
        :v_velocity             => ("Vvel",     "2013to2024_monthly"),
        :surface_heat_flux      => ("surfTflx", "2013to2024_monthly"),
        :surface_salt_flux      => ("surfSflx", "2013to2024_monthly"),
        :surface_shortwave      => ("oceQsw",   "2013to2024_monthly"),
        :sea_surface_height     => ("SSH",      "2013to2024_monthly"),
        # The cluster carries BOTH `oceTAUX_bsoseI156_2013to2024_monthly.nc` and
        # `oceTAUX_bsoseI156_2014_5day.nc`. Monthly is the one this project uses: the 5-day
        # wind forcing was taken out deliberately. Switch the span to "2014_5day" to go back.
        :zonal_wind_stress      => ("oceTAUX",  "2013to2024_monthly"),
        :meridional_wind_stress => ("oceTAUY",  "2013to2024_monthly"),
    ),
    105 => Dict(
        :temperature            => ("Theta",    "2008to2012_monthly"),
        :salinity               => ("Salt",     "2008to2012_monthly"),
        :u_velocity             => ("Uvel",     "2008to2012_monthly"),
        :v_velocity             => ("Vvel",     "2008to2012_monthly"),
        :surface_heat_flux      => ("surfTflx", "2008to2012_monthly"),
        :surface_salt_flux      => ("surfSflx", "2008to2012_monthly"),
        :zonal_wind_stress      => ("oceTAUX",  "2008to2012_monthly"),
        :meridional_wind_stress => ("oceTAUY",  "2008to2012_monthly"),
    ),
)

bsose_filename(token, span, iteration) =
    iteration == 156 ? "$(token)_bsoseI156_$(span).nc" :
                       "bsose_i$(iteration)_$(span)_$(token).nc"

"""
    resolve_bsose_filename(dir, var_name, iteration)

The bare filename for `var_name` at `iteration`, or an error.

Errors rather than falling back, by design -- see [`BSOSE_FILES`](@ref). The message names the
exact path that was expected and lists any `.nc` file in `dir` carrying the same token, so a
wrong token in the table is a one-line fix rather than a debugging session.
"""
function resolve_bsose_filename(dir::String, var_name::Symbol, iteration::Int)
    haskey(BSOSE_FILES, iteration) ||
        error("No BSOSE filename table for iteration $iteration. Known: $(sort(collect(keys(BSOSE_FILES)))).")

    table = BSOSE_FILES[iteration]
    haskey(table, var_name) ||
        error("""BSOSE iteration $iteration has no filename for `$var_name`.
                 Known variables: $(sort(string.(collect(keys(table))))).
                 Add it to `BSOSE_FILES` in src/setup_bsose.jl.""")

    token, span = table[var_name]
    fname = bsose_filename(token, span, iteration)
    isfile(joinpath(dir, fname)) && return fname

    # Not found: say exactly what was expected, and what is there under the same token.
    nearby = isdir(dir) ?
             filter(f -> endswith(f, ".nc") && occursin(lowercase(token), lowercase(f)), readdir(dir)) :
             String[]
    hint = isempty(nearby) ? "No .nc file in that directory mentions \"$token\"." :
           "Files there mentioning \"$token\":\n    " * join(nearby, "\n    ") *
           "\n  If one of those is the right file, fix the token for `$var_name` in `BSOSE_FILES`."

    error("""BSOSE iteration $iteration: `$var_name` not found.
             Expected: $(joinpath(dir, fname))
             $hint""")
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

# Each BSOSE monthly file holds a calendar-month average stamped at the END of the month it
# averages (to within a day: the stamps sit at uniform year/12 intervals, so they drift either
# side of the month boundary). Snapping the stamp to the nearest month boundary recovers the
# month the average was taken over, which is what `native_times` needs to place each sample at
# its own middle rather than at its end, and what gives a twelve-month series a `Cyclical`
# period of exactly one year instead of one inferred from the spacing of the stamps.
"""
    bsose_averaging_window(stamp)

The `(start, stop)` dates of the month a BSOSE monthly record averages, recovered from the
date it is stamped with by snapping that stamp to the nearest month boundary.
"""
function bsose_averaging_window(stamp)
    window_stop = round(DateTime(stamp), Month)
    return (window_stop - Month(1), window_stop)
end

"""
    bsose_window_center(stamp)

The middle of the month a BSOSE monthly record averages, which is where a linearly
interpolating `FieldTimeSeries` places that record's value. Half a month earlier than the
date the record is stamped with, so it is the right thing to compare a simulation date
against when choosing which records a run needs.
"""
function bsose_window_center(stamp)
    window_start, window_stop = bsose_averaging_window(stamp)
    half = Dates.value(Millisecond(window_stop - window_start)) ÷ 2
    return window_start + Millisecond(half)
end

DataWrangling.averaging_window(metadatum::BSOSEMetadatum) = bsose_averaging_window(metadatum.dates)

"""
    model_clock_offset(metadata, reference_date)

The shift in seconds that puts a `FieldTimeSeries` built from `metadata` onto the model clock.

`native_times` measures a series from the stamp of its first record, so a series used as-is
makes model time zero mean "the first record of the dataset". Adding this offset to its times
makes model time `t` mean the calendar date `reference_date + t` instead, which is the clock
the initial condition, the output writers and the analysis scripts all assume. `nothing` leaves
the series on the dataset's own clock.
"""
function model_clock_offset(metadata, reference_date)
    isnothing(reference_date) && return 0.0
    first_stamp = DateTime(first(metadata).dates)
    return Dates.value(Dates.Millisecond(first_stamp - DateTime(reference_date))) / 1000
end

function DataWrangling.longitude_name(metadata::BSOSEMetadata)
    loc = dataset_location(metadata.dataset, metadata.name)
    return loc[1] === Face ? "XG" : "XC"
end

function DataWrangling.latitude_name(metadata::BSOSEMetadata)
    loc = dataset_location(metadata.dataset, metadata.name)
    return loc[2] === Face ? "YG" : "YC"
end


function DataWrangling.default_download_directory(dataset::BSOSEDataset)
    return dataset.dir
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

bsose_bbox_str(::Nothing) = ""
bsose_bbox_str(c::Number) = @sprintf("_%.1f", c)
function bsose_bbox_str(c)
    return @sprintf("_%.1f_%.1f", c[1], c[2])
end

function bsose_region_suffix(region::DataWrangling.BoundingBox)
    w_e = bsose_bbox_str(region.longitude)
    s_n = bsose_bbox_str(region.latitude)
    return string(w_e, s_n)
end
bsose_region_suffix(::Nothing) = ""
bsose_region_suffix(other) = ""

function DataWrangling.inpainted_metadata_path(metadata::BSOSEMetadatum)
    geom = bsose_geometry(metadata.dataset)
    dstr = metadata.dates isa Dates.AbstractDateTime ? Dates.format(metadata.dates, "yyyymmdd") : "all"
    suffix = bsose_region_suffix(metadata.region)
    geom_str = isempty(suffix) ? "_$(geom.Nx)x$(geom.Ny)" : suffix
    name = "bsose_inpainted_i$(metadata.dataset.iteration)_$(metadata.name)_$(dstr)$(geom_str).jld2"
    return joinpath(bsose_temp_directory(metadata.dataset), name)
end

function DataWrangling.inpainted_metadata_path(metadata::BSOSEMetadata)
    geom = bsose_geometry(metadata.dataset)
    start_str = Dates.format(first(metadata.dates), "yyyymmdd")
    end_str = Dates.format(last(metadata.dates), "yyyymmdd")
    suffix = bsose_region_suffix(metadata.region)
    geom_str = isempty(suffix) ? "_$(geom.Nx)x$(geom.Ny)" : suffix
    name = "bsose_inpainted_i$(metadata.dataset.iteration)_$(metadata.name)_$(start_str)_to_$(end_str)$(geom_str).jld2"
    return joinpath(bsose_temp_directory(metadata.dataset), name)
end

function DataWrangling.all_dates(dataset::BSOSEMonthly, var)
    # Read the dates out of the file, or fail. There used to be a `catch` here that fell back
    # to a synthetic `2013-01-01:Month(1):2024-12-01` range, which meant an unreadable or
    # truncated file produced a plausible-looking date axis that did not match its own
    # contents -- a silent error of exactly the kind this module is trying to stop making.
    fp = joinpath(dataset.dir, resolve_bsose_filename(dataset.dir, var, dataset.iteration))
    ds = Dataset(fp)
    try
        return [DateTime(t) for t in ds["time"][:]]
    finally
        close(ds)
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

    # Pick the record whose averaging window contains `target_date`, using the same
    # convention as `averaging_window`: a BSOSE monthly mean is stamped at the end of the
    # month it averages. Matching the calendar month of the stamp instead would be wrong
    # whenever the stamp drifts past the month boundary — the February mean of iteration
    # 105 is stamped 2008-03-01T20:00, so a run starting 2008-03-01 would initialise from
    # February data while its boundary conditions and winds are already in March.
    for (i, t) in enumerate(time_raw)
        window_start, window_stop = bsose_averaging_window(t)
        if window_start <= target_dt < window_stop
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
    resident_fts(metadata, grid; kw...)

`FieldTimeSeries` from `Metadata` with EVERY record held in memory, instead of the default 2.

NumericalEarth's `FieldTimeSeries(::Metadata, grid)` defaults `time_indices_in_memory = 2`: two
snapshots resident, the rest reloaded from disk on demand. That is a sensible memory default and
a bad one for us, because `boundary_slice_time_series` below copies `fts[t]` EAGERLY for every
`t`, so a stale read is baked into the copy permanently and silently.

It has been measured. `tools/face_transports.jl` asked for 12 monthly records and got **9
distinct ones, in three bit-identical consecutive pairs**. NestingValidation's recipe lists the
same trap: "Set `time_indices_in_memory` to the full length ... The default of 2 reloads cached
slices and undoes the repair."

The 20-day runs used 3 records and never showed it. A 2-year run uses ~26 and is well inside the
regime where it bites, which would mean months of boundary and sponge forcing silently repeating
other months.

Cost is memory: one regridded 360x150x53 Float64 field is ~23 MB, so ~26 records of one variable
is ~0.6 GB. Four variables is a few GB on a 32 GB node -- affordable, and the alternative is a
run whose forcing is quietly wrong.
"""
resident_fts(metadata, grid; kw...) =
    FieldTimeSeries(metadata, grid; time_indices_in_memory = max(1, length(metadata)), kw...)

"""
    boundary_slice_time_series(fts::FieldTimeSeries, side::Symbol)

Extract a 2D boundary `FieldTimeSeries` along `side` (:west, :east, :south, :north)
from a 3D `FieldTimeSeries`.

The resulting 2D FieldTimeSeries has the reduced dimensionality and index structure
required by Oceananigans boundary conditions (`YZFTS` for west/east, `XZFTS` for south/north).

`time_offset` (seconds) shifts the slice onto the model clock; see [`model_clock_offset`](@ref).
"""
function boundary_slice_time_series(fts::FieldTimeSeries, side::Symbol; time_offset=0)
    grid = fts.grid
    times = fts.times .+ time_offset
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

"""
    mask_immersed_boundary_normal_flow!(bts, side, grid)

Zero out prescribed normal boundary flow at any point where the boundary face or adjacent cell
is immersed (solid bathymetry). This ensures open boundary conditions do not inject momentum
directly into or across solid land walls.
"""
function mask_immersed_boundary_normal_flow!(bts, side, grid)
    (bts === nothing || !(grid isa ImmersedBoundaryGrid)) && return bts

    # If grid or bts is on GPU, transfer to CPU for indexing and mask evaluation to avoid scalar GPU indexing
    arch_bts = architecture(bts)
    cpu_grid = architecture(grid) isa GPU ? on_architecture(CPU(), grid) : grid
    cpu_bts = arch_bts isa GPU ? on_architecture(CPU(), bts) : bts

    underlying = cpu_grid.underlying_grid
    Nx, Ny, Nz = size(underlying)
    masked_count = 0

    if side === :west
        # Normal velocity is u at i=1: (Face, Center, Center)
        for j in 1:Ny, k in 1:Nz
            if immersed_peripheral_node(1, j, k, cpu_grid, Face(), Center(), Center()) || immersed_cell(1, j, k, cpu_grid)
                for t in 1:length(cpu_bts.times)
                    cpu_bts[t][1, j, k] = 0.0
                end
                masked_count += 1
            end
        end
    elseif side === :east
        # Normal velocity is u at i=Nx+1: (Face, Center, Center)
        for j in 1:Ny, k in 1:Nz
            if immersed_peripheral_node(Nx + 1, j, k, cpu_grid, Face(), Center(), Center()) || immersed_cell(Nx, j, k, cpu_grid)
                for t in 1:length(cpu_bts.times)
                    cpu_bts[t][1, j, k] = 0.0
                end
                masked_count += 1
            end
        end
    elseif side === :south
        # Normal velocity is v at j=1: (Center, Face, Center)
        for i in 1:Nx, k in 1:Nz
            if immersed_peripheral_node(i, 1, k, cpu_grid, Center(), Face(), Center()) || immersed_cell(i, 1, k, cpu_grid)
                for t in 1:length(cpu_bts.times)
                    cpu_bts[t][i, 1, k] = 0.0
                end
                masked_count += 1
            end
        end
    elseif side === :north
        # Normal velocity is v at j=Ny+1: (Center, Face, Center)
        for i in 1:Nx, k in 1:Nz
            if immersed_peripheral_node(i, Ny + 1, k, cpu_grid, Center(), Face(), Center()) || immersed_cell(i, Ny, k, cpu_grid)
                for t in 1:length(cpu_bts.times)
                    cpu_bts[t][i, 1, k] = 0.0
                end
                masked_count += 1
            end
        end
    end

    if masked_count > 0
        for t in 1:length(cpu_bts.times)
            fill_halo_regions!(cpu_bts[t])
        end
        @info "Masked $masked_count immersed/solid face points to zero on $side normal boundary."
    end

    return arch_bts isa GPU ? on_architecture(arch_bts, cpu_bts) : cpu_bts
end

# ==============================================================================
# 2d. Barotropic transport for the Flather open boundary conditions
#
# The split-explicit free surface solves separately for the depth-integrated
# transport (U, V) and the surface displacement η, and its corrector REPLACES the
# depth mean of u and v at the boundary with whatever the barotropic solver says.
# A NormalFlow condition on `u` is therefore only half an open boundary: while `U`
# keeps its default impenetrable `NormalFlow(nothing)`, the depth mean of any
# prescribed inflow is driven back to zero, and the baroclinic condition can only
# redistribute flow in the vertical — never let a net transport cross the face.
# That is what pins a jet along the western edge and holds every face transport at
# 0.00 Sv no matter which scheme `u` carries.
#
# Flather supplies the missing half,
#
#     Uᵇ = Uᵉˣᵗ ± √(gH) (ηᵇ − ηᵉˣᵗ)
#
# and is the only place data enters the barotropic mode: the companion Chapman
# condition on η radiates but carries no data, and Oceananigans 0.113 pairs it onto
# η automatically from the `U`/`V` conditions (see
# `implicit_gravity_wave_companion_boundary_conditions`), so η needs nothing here.
# ==============================================================================

"""
    boundary_vertical_spacing(grid)

Cell vertical spacings `Δz[k]` of `grid`, as a plain CPU vector.

The vertical discretization of a `LatitudeLongitudeGrid` does not vary horizontally, and
`GridFittedBottom` leaves each cell wholly wet or wholly dry rather than partially filled,
so a single column of spacings describes every boundary cell.
"""
function boundary_vertical_spacing(grid)
    underlying = grid isa ImmersedBoundaryGrid ? grid.underlying_grid : grid
    cpu_grid = architecture(underlying) isa GPU ? on_architecture(CPU(), underlying) : underlying
    return diff(collect(znodes(cpu_grid, Face())))
end

"""
    bsose_barotropic_transport(bts, side, grid)

Depth-integrated transport `Uᵉˣᵗ = ∫ u·dz` (m² s⁻¹) along one open boundary, returned as a
`FieldTimeSeries` shaped for the barotropic velocity's halo on that side — `(Nothing,
Center, Center)` on west/east, `(Center, Nothing, Center)` on south/north, full depth, with
every level carrying the same value. Both the `Center` in `z` and the full-depth window are
load-bearing; see the comment at the construction.

`bts` must be the boundary slice of the *normal* velocity for `side` **after**
[`mask_immersed_boundary_normal_flow!`](@ref) has zeroed its dry cells, so that summing the
whole column already gives the transport through the model's wet column alone.

That ordering is what keeps this integral honest, and it is the easiest thing to get wrong.
BSOSE land is inpainted with nearest-neighbour ocean values
(`DataWrangling.default_inpainting`), so the raw slice carries plausible-looking velocity in
cells below the sea floor; integrating those and then dividing by a shallow shelf depth is
how a boundary comes to demand metres per second of depth-mean inflow. The reported
`max |Uᵉˣᵗ|/H` makes that visible — it should be a sane depth-mean velocity (order
0.1 m/s here), not order 1.
"""
function bsose_barotropic_transport(bts, side, grid)
    bts === nothing && return nothing

    Δz = boundary_vertical_spacing(grid)
    cpu_bts = architecture(bts) isa GPU ? on_architecture(CPU(), bts) : bts
    cpu_grid = architecture(grid) isa GPU ? on_architecture(CPU(), grid) : grid
    underlying = grid isa ImmersedBoundaryGrid ? grid.underlying_grid : grid
    Nx, Ny, Nz = size(underlying)

    times = cpu_bts.times
    time_indexing = cpu_bts.time_indexing

    along_x = side === :south || side === :north   # transport varies along i, not j
    N = along_x ? Nx : Ny

    # Wet depth of each boundary column. Only used to report the implied depth-mean
    # velocity: Flather forms √(gH) from the model's own H, not from this.
    H = zeros(N)
    for n in 1:N, k in 1:Nz
        if cpu_grid isa ImmersedBoundaryGrid
            i = along_x ? n : (side === :west ? 1 : Nx)
            j = along_x ? (side === :south ? 1 : Ny) : n
            immersed_cell(i, j, k, cpu_grid) && continue
        end
        H[n] += Δz[k]
    end

    # The z location must be `Center`, not `Nothing`, even though these series carry no depth.
    #
    # `getbc` dispatches on the location triple. The west/east halo fill calls
    # `getbc(bc, j, k, grid, clock)` with k = 1, because the barotropic fields are `Nothing` in
    # z and the kernel is one cell deep. {Nothing, Center, Nothing} is `YFTS`, whose `getbc` is
    #
    #     getbc(f::YFTS, i1, i2, grid, clock) = (Nx, _, Nz = size(grid);
    #                                            j = ifelse(Nz == 1, i1, i2); f[1, j, 1, t])
    #
    # a guess the Oceananigans source itself calls "not really well defined". It asks whether
    # the GRID is two-dimensional -- but these are built on the full 3-D grid (Nz = 135), so it
    # takes i2 = k = 1 and EVERY column of the west and east faces reads element j = 1. On this
    # domain j = 1 is in the dry Antarctic band, so both zonal faces were fed a single constant.
    #
    # {Nothing, Center, Center} is `YZFTS`, whose `getbc` is unambiguous, f[1, j, k, t], and the
    # one-cell z window makes k = 1 the only valid index. South/north were always correct
    # (`XFTS` uses only the first index); they move to `XZFTS` for symmetry.
    # Full-depth window, and `Center` in z -- both load-bearing.
    #
    # `Center` in z: `getbc` dispatches on the location triple. With {Nothing,Center,Nothing}
    # (`YFTS`) it resolves the index through `j = ifelse(size(grid)[3] == 1, i1, i2)`, a guess
    # the Oceananigans source calls "not really well defined". The barotropic U/V/η live on the
    # full 3-D grid, so it took i2 = k = 1 and EVERY column of the west and east faces read
    # element j = 1 -- which on this domain is in the dry Antarctic band. {Nothing,Center,Center}
    # is `YZFTS`, whose `getbc` is plainly f[1,j,k,t].
    #
    # FULL depth rather than a one-cell z window: a one-cell window is a concrete type this GPU
    # had never compiled, and every variant of it failed on the cluster -- `malloc_consolidate`
    # abort, SIGBUS, then an indefinite hang, on both the model grid and a flat 1-level grid.
    # `boundary_slice_time_series` has been building exactly this full-z shape on this GPU all
    # along without trouble, so reuse it. The cost is 135x a very small array.
    #
    # The barotropic halo fill asks for k = 1, so every level is filled with the same value.
    Uts = along_x ?
          FieldTimeSeries{Center,Nothing,Center}(cpu_bts.grid, times; indices=(:, 1, :), time_indexing) :
          FieldTimeSeries{Nothing,Center,Center}(cpu_bts.grid, times; indices=(1, :, :), time_indexing)

    # Width of each boundary column, for the face integral reported below. The barotropic
    # transport lives at (Face, Center) on west/east and (Center, Face) on south/north, so
    # the spacing accessor differs between them. Spacings are geometric, so they come from
    # the underlying grid on the host.
    cpu_underlying = cpu_grid isa ImmersedBoundaryGrid ? cpu_grid.underlying_grid : cpu_grid
    column_width(n) = along_x ? Δxᶜᶠᶜ(n, side === :south ? 1 : Ny + 1, 1, cpu_underlying) :
                      Δyᶠᶜᶜ(side === :west ? 1 : Nx + 1, n, 1, cpu_underlying)

    peak = 0.0
    face_transports = Float64[]
    for t in 1:length(times)
        slice = along_x ? interior(cpu_bts[t], :, 1, :) : interior(cpu_bts[t], 1, :, :)
        # Masked exactly as `H` above is. BSOSE inpaints land with nearest-neighbour OCEAN
        # values, so an UNMASKED integral sums real velocities through solid rock. Before this
        # was fixed (2026-10-25) the all-land south face reported -32.4 Sv over 0 wet columns,
        # which is how the bug was caught -- but the faces that mattered were west and east,
        # with 19 and 10 dry columns along the Antarctic coast. `target_transport` pins the face
        # INTEGRAL, so a phantom in those dry columns was being spread across the wet ones by
        # the rescale, corrupting the PROFILE while the net stayed perfect. The north face is
        # 360/360 wet, so this changes nothing there.
        U = zeros(N)
        for n in 1:N, k in 1:Nz
            if cpu_grid isa ImmersedBoundaryGrid
                i = along_x ? n : (side === :west ? 1 : Nx)
                j = along_x ? (side === :south ? 1 : Ny) : n
                immersed_cell(i, j, k, cpu_grid) && continue
            end
            U[n] += Δz[k] * slice[n, k]
        end

        if along_x
            interior(Uts[t], :, 1, :) .= reshape(U, :, 1)   # same value at every level
        else
            interior(Uts[t], 1, :, :) .= reshape(U, :, 1)
        end
        fill_halo_regions!(Uts[t])

        for n in 1:N
            H[n] > 0 && (peak = max(peak, abs(U[n]) / H[n]))
        end

        # Face total, which is the number the Flather condition is actually being asked to
        # drive. `max |U|/H` only says that no single column is absurd; it says nothing about
        # whether the face carries the right transport, and that distinction cost a lot of
        # debugging. Verified targets for the 90-150E / 70-45S domain, from BSOSE integrated
        # directly and from the model's own iteration-0 diagnostic independently:
        # west +180 Sv, east +205 Sv, north -25 Sv.
        push!(face_transports, sum(U[n] * column_width(n) for n in 1:N) / 1e6)
    end

    T̄ = sum(face_transports) / length(face_transports)
    @info @sprintf(" -> %-5s Uᵉˣᵗ: %+7.1f Sv (range %+.1f to %+.1f), max |U|/H = %.3f m/s over %d wet columns",
        string(side), T̄, minimum(face_transports), maximum(face_transports),
        peak, count(>(0), H))

    arch = architecture(grid)
    return arch isa GPU ? on_architecture(arch, Uts) : Uts
end

# ==============================================================================
# 3. Surface Wind Stress Helper (Standalone)
# ==============================================================================

"""
    bsose_surface_wind_stress(grid;
                              dataset = BSOSEMonthly(),
                              dates = all_dates(dataset, :temperature)[1:12],
                              ρ₀ = 1026.0,
                              reference_date = nothing)

Load BSOSE surface wind stress (`oceTAUX` and `oceTAUY`), convert to the kinematic
momentum flux Oceananigans expects at a top boundary (`-τ / ρ₀`), and return a
`NamedTuple` of top `FluxBoundaryCondition`s:
`(; u = FluxBoundaryCondition(top_u), v = FluxBoundaryCondition(top_v))`.

`reference_date` is the calendar date of model time zero; see [`model_clock_offset`](@ref).

Can be used standalone or passed to `bsose_open_boundary_conditions`.
"""
function bsose_surface_wind_stress(grid;
    dataset=BSOSEMonthly(),
    dates=nothing,
    ρ₀=1026.0,
    reference_date=nothing)
    if isnothing(dates)
        available = all_dates(dataset, :zonal_wind_stress)
        dates = available[1:min(12, length(available))]
    end
    @info "Loading surface wind stress ($(summary(dataset)))..."
    region = bsose_region(grid)
    taux_meta = Metadata(:zonal_wind_stress; dataset, dates, region)
    tauy_meta = Metadata(:meridional_wind_stress; dataset, dates, region)
    taux_fts = resident_fts(taux_meta, grid)
    tauy_fts = resident_fts(tauy_meta, grid)

    time_offset = model_clock_offset(taux_meta, reference_date)
    top_u_slice = boundary_slice_time_series(taux_fts, :top; time_offset)
    top_v_slice = boundary_slice_time_series(tauy_fts, :top; time_offset)

    # Convert stress (N/m²) to the kinematic momentum flux a top boundary condition takes
    # (m²/s²). A flux at the top is positive upward, so an eastward stress τˣ > 0 — momentum
    # carried *into* the ocean — enters as -τˣ/ρ₀.
    #
    # This has to happen here, on the sliced series, rather than on `taux_fts` above: the
    # series returned by `FieldTimeSeries(::Metadata, grid)` keeps only two snapshots in
    # memory and reloads them from disk whenever a third is indexed, so scaling it in place
    # is silently undone. The slices are ordinary in-memory series, so the scaling sticks.
    # Scaling `parent` covers the halos too, which were filled during slicing.
    for t in 1:length(top_u_slice.times)
        parent(top_u_slice[t]) .*= -1 / ρ₀
        parent(top_v_slice[t]) .*= -1 / ρ₀
    end

    return (u=FluxBoundaryCondition(top_u_slice),
        v=FluxBoundaryCondition(top_v_slice))
end

"""
    bsose_surface_tracer_fluxes(grid;
                                dataset = BSOSEMonthly(),
                                dates = nothing,
                                ρ₀ = 1026.0,
                                cₚ = 3991.86795711963,
                                reference_date = nothing)

Load the BSOSE surface heat and salt fluxes (`TFLUX` and `SFLUX`), convert them to the
kinematic fluxes Oceananigans takes at a top boundary, and return a `NamedTuple` of top
`FluxBoundaryCondition`s: `(; T = FluxBoundaryCondition(Jᵀ), S = FluxBoundaryCondition(Jˢ))`.

Without these the ocean exchanges no heat or freshwater with the atmosphere at all: it cannot
cool in winter or warm in summer, and its surface temperature is left to advection and mixing.

`reference_date` is the calendar date of model time zero; see [`model_clock_offset`](@ref).
"""
function bsose_surface_tracer_fluxes(grid;
    dataset=BSOSEMonthly(),
    dates=nothing,
    ρ₀=1026.0,
    cₚ=3991.86795711963,   # TEOS-10 reference heat capacity, J kg⁻¹ °C⁻¹
    subtract_shortwave=false,
    reference_date=nothing)

    if isnothing(dates)
        available = all_dates(dataset, :surface_heat_flux)
        dates = available[1:min(12, length(available))]
    end

    @info "Loading surface heat and salt fluxes ($(summary(dataset)))..."
    region = bsose_region(grid)
    heat_metadata = Metadata(:surface_heat_flux; dataset, dates, region)
    salt_metadata = Metadata(:surface_salt_flux; dataset, dates, region)
    heat_fts = resident_fts(heat_metadata, grid)
    salt_fts = resident_fts(salt_metadata, grid)

    time_offset = model_clock_offset(heat_metadata, reference_date)
    top_T_slice = boundary_slice_time_series(heat_fts, :top; time_offset)
    top_S_slice = boundary_slice_time_series(salt_fts, :top; time_offset)

    # With `subtract_shortwave`, what stays at the surface is the NON-SOLAR part of the heat
    # flux. TFLUX is the total and already contains the solar part, so handing the whole of it
    # to the top boundary while `bsose_penetrating_shortwave` also distributes oceQsw through
    # the column would count the sunlight twice. The two fields share the MITgcm ">0 increases
    # theta" convention, so the remainder is a plain difference, taken before the sign flip
    # below and on the sliced series for the reason given there.
    if subtract_shortwave
        shortwave_metadata = Metadata(:surface_shortwave; dataset, dates, region)
        shortwave_fts = resident_fts(shortwave_metadata, grid)
        top_Q_slice = boundary_slice_time_series(shortwave_fts, :top; time_offset)

        for t in 1:length(top_T_slice.times)
            parent(top_T_slice[t]) .-= parent(top_Q_slice[t])
        end

        Nlo, Nbar, Nhi = series_statistics(top_T_slice)
        @info @sprintf(" -> Non-solar surface heat flux (TFLUX - oceQsw): mean %+.1f W m⁻², range %+.1f to %+.1f",
            Nbar, Nlo, Nhi)
    end

    # BSOSE follows the MITgcm convention, stated in the files themselves: TFLUX "> 0
    # increases theta" and SFLUX "> 0 increases salt". An Oceananigans top flux is positive
    # UPWARD, out of the domain, so both enter with a minus sign. TFLUX is W m⁻², which
    # becomes the °C m s⁻¹ a temperature flux condition takes after dividing by ρ₀cₚ; SFLUX
    # is g m⁻² s⁻¹, which becomes psu m s⁻¹ after dividing by ρ₀.
    #
    # As with the wind stress, the scaling has to happen here, on the sliced series. The
    # series `FieldTimeSeries(::Metadata, grid)` returns holds only two snapshots in memory
    # and reloads them from disk, so scaling it in place is silently discarded.
    for t in 1:length(top_T_slice.times)
        parent(top_T_slice[t]) .*= -1 / (ρ₀ * cₚ)
        parent(top_S_slice[t]) .*= -1 / ρ₀
    end

    return (T=FluxBoundaryCondition(top_T_slice),
        S=FluxBoundaryCondition(top_S_slice))
end


"""
    series_statistics(fts)

`(minimum, mean, maximum)` over every snapshot of `fts`, for reporting.

Each slice is copied to the host before being reduced. `interior` hands back a view of an
`OffsetArray` wrapping the underlying array, and on the GPU those wrappers defeat the CUDA
reduction methods: `Statistics._mean` falls through to the generic implementation, which
begins with `first(A)` and so trips the scalar-indexing guard. The slices are single surface
levels read once at setup, so the copy costs nothing worth measuring.
"""
function series_statistics(fts)
    lo, hi, total, n = Inf, -Inf, 0.0, 0
    for t in 1:length(fts.times)
        a = Array(interior(fts[t]))
        lo = min(lo, minimum(a))
        hi = max(hi, maximum(a))
        total += sum(a)
        n += length(a)
    end
    return lo, total / n, hi
end


"""
    face_geometry(side, grid)

`(widths, depths)` per boundary column: the along-face cell width and the wet depth of each
column of `side`, on the host.
"""
function face_geometry(side, grid)
    underlying = grid isa ImmersedBoundaryGrid ? grid.underlying_grid : grid
    cpu_grid = architecture(grid) isa GPU ? on_architecture(CPU(), grid) : grid
    cpu_underlying = cpu_grid isa ImmersedBoundaryGrid ? cpu_grid.underlying_grid : cpu_grid
    Nx, Ny, Nz = size(underlying)
    Δz = boundary_vertical_spacing(grid)
    along_x = side === :south || side === :north
    N = along_x ? Nx : Ny

    widths = [along_x ? Δxᶜᶠᶜ(n, side === :south ? 1 : Ny + 1, 1, cpu_underlying) :
              Δyᶠᶜᶜ(side === :west ? 1 : Nx + 1, n, 1, cpu_underlying) for n in 1:N]

    depths = zeros(N)
    for n in 1:N, k in 1:Nz
        i = along_x ? n : (side === :west ? 1 : Nx)
        j = along_x ? (side === :south ? 1 : Ny) : n
        cpu_grid isa ImmersedBoundaryGrid && immersed_cell(i, j, k, cpu_grid) && continue
        depths[n] += Δz[k]
    end
    return widths, depths
end


"""
    balance_barotropic_transports!(Us, grid)

Remove the net volume imbalance from the prescribed barotropic transports, in place, record
by record. This is MITgcm's `useOBCSbalance` and it is standard practice in regional modelling.

BSOSE's velocities are masked onto a bathymetry that is not BSOSE's, so the four faces do not
sum to zero even though BSOSE's own do. Over this domain the residual is about +3 Sv of
outflow, which sounds small against ~200 Sv of gross transport and is not: unopposed it drains
the free surface by 2.9 m over ninety days. Flather does oppose it, but only by letting η sit
low enough for `√(gH)(ηᵇ - ηᵉˣᵗ)` to cancel it — and with `ηᵉˣᵗ = 0` that offset reached
-0.65 m by day 50, which is worth several hundred Sverdrups of correction and was enough to
reverse the eastern face from +203 Sv of outflow to -119 Sv of inflow.

The correction is a uniform velocity across every open face, so each column is adjusted in
proportion to its own depth and the vertical structure of the prescribed transport is left
alone. Faces that are entirely land contribute no area and take no correction.
"""
function balance_barotropic_transports!(Us, grid)
    # A face with no wet area cannot carry transport and must not take part. It would
    # contribute to `net` through its WIDTHS -- which come from the underlying grid and are
    # always positive -- while receiving none of the correction, which is distributed by DEPTH
    # and is therefore zero for it. So an all-land face injects a phantom imbalance that the
    # real faces then pay for. Found 2026-10-25 on the all-land south face, where the
    # pre-balance net read -63.92 Sv against BSOSE's own documented closure of +0.151 Sv. The
    # pinned-target code in `bsose_open_boundary_conditions` already filters this way; this
    # makes the balance agree with it.
    sides = [s for s in (:west, :east, :south, :north)
             if getproperty(Us, s) !== nothing && sum(face_geometry(s, grid)[2]) > 0]
    isempty(sides) && return Us

    outward = (west=-1.0, east=+1.0, south=-1.0, north=+1.0)
    geom = Dict(s => face_geometry(s, grid) for s in sides)
    total_area = sum(sum(w .* d) for (w, d) in values(geom))

    if total_area <= 0
        @warn " -> no wet open-boundary area; barotropic transports left unbalanced."
        return Us
    end

    cpu = Dict(s => (architecture(getproperty(Us, s)) isa GPU ?
                     on_architecture(CPU(), getproperty(Us, s)) : getproperty(Us, s)) for s in sides)
    nt = length(first(values(cpu)).times)
    before, after = Float64[], Float64[]

    for t in 1:nt
        face_U(s) = (fts = cpu[s]; s in (:south, :north) ? interior(fts[t], :, 1, 1) : interior(fts[t], 1, :, 1))
        net = sum(outward[s] * sum(face_U(s) .* geom[s][1]) for s in sides)
        push!(before, net / 1e6)

        Δu = net / total_area
        for s in sides
            face_U(s) .-= outward[s] * Δu .* geom[s][2]
        end
        push!(after, sum(outward[s] * sum(face_U(s) .* geom[s][1]) for s in sides) / 1e6)
        for s in sides
            fill_halo_regions!(cpu[s][t])
        end
    end

    @info @sprintf(" -> barotropic balance: mean net %+.2f -> %+.2e Sv, worst record %+.2f -> %+.2e Sv, uniform correction up to %.4f m/s",
        sum(before)/nt, sum(after)/nt, before[argmax(abs.(before))], after[argmax(abs.(after))],
        maximum(abs.(before)) * 1e6 / total_area)

    arch = architecture(grid)
    return NamedTuple(s => (arch isa GPU ? on_architecture(arch, cpu[s]) : cpu[s]) for s in sides)
end


"""
    mean_face_transport(fts, side, grid)

Time-mean net transport through one face, `∮U·dl` in m³ s⁻¹, signed positive in the positive
coordinate direction -- the convention `GravityWaveRadiation`'s `target_transport` expects.

`fts` is a depth-integrated transport series (m² s⁻¹) as `bsose_barotropic_transport` returns,
so multiplying by the column width and summing along the face gives m³ s⁻¹ directly.
"""
function mean_face_transport(fts, side, grid)
    series = face_transport_series(fts, side, grid)
    return isempty(series) ? 0.0 : sum(series) / length(series)
end


"""
    face_transport_series(fts, side, grid)

Net transport through one face for **every record**, `∮U·dl` in m³ s⁻¹, signed positive in the
positive coordinate direction. Empty if `fts === nothing`.

[`mean_face_transport`](@ref) is the time mean of this, and the pinned `target_transport` is
that mean -- so this series is exactly what the pin averages away. `tools/face_transports.jl`
reports it per record, which is the only way to see whether the single pinned number is
representative of the run window or whether it is holding a face against its own seasonal sign
reversal. North swings -31 to +16 Sv over a year, so that is not a hypothetical.

Kept as one code path with `mean_face_transport` deliberately: a tool that re-derived this sum
could drift from what the model actually pins, and then the diagnostic would be measuring
something the model does not do.
"""
function face_transport_series(fts, side, grid)
    fts === nothing && return Float64[]
    widths, _ = face_geometry(side, grid)
    cpu = architecture(fts) isa GPU ? on_architecture(CPU(), fts) : fts
    return [sum((side in (:south, :north) ? interior(cpu[t], :, 1, 1) :
                                            interior(cpu[t], 1, :, 1)) .* widths)
            for t in 1:length(cpu.times)]
end


"""
    bsose_exterior_free_surface(grid; dataset, dates, reference_date, remove_datum = true)

External sea surface `ηᵉˣᵗ` for the Flather condition, as a `NamedTuple` of per-face
`FieldTimeSeries` shaped to match what [`bsose_barotropic_transport`](@ref) returns.

Flather does not prescribe the transport. Rearranged,

    Uᵇ - √(gH) ηᵇ  =  Uᵉˣᵗ - √(gH) ηᵉˣᵗ

it prescribes the incoming Riemann invariant, a *combination* of transport and sea level.
Passing `ηᵉˣᵗ = 0` therefore does not omit a refinement, it specifies a different invariant:
√(gH) is 198 m/s here, so a half-metre of neglected sea surface is 99 m² s⁻¹ against a
prescribed transport of 81 m² s⁻¹ per unit length. The term dropped is larger than the signal
kept, which is why a face asked for +203 Sv of outflow delivered -176 Sv of inflow.

### Why this reads the file directly

BSOSE's `ETAN` is the one field in this module that does **not** mark land with NaN. It
declares `_FillValue = NaN` and then never uses it: land is stored as exactly `0.0`, matching
`Depth == 0` point for point. The inpainting the other variables rely on would see those
zeros as valid data, and zero is a *plausible* sea level sitting next to ocean values near
-2 m. Land is therefore masked here, from `Depth`, before anything else happens.

### Datum

`ETAN` is an anomaly about BSOSE's own reference, and over this domain its ocean-only mean is
about -1 m, while the model's `η` starts at zero domain mean. Fed in raw, Flather would spend
the first days of the run dragging the free surface down to BSOSE's datum. `remove_datum`
subtracts a single scalar — the mean over every wet boundary point and every record — so the
spatial structure and the seasonal variation survive and only the offset goes. Removing a
time-varying mean instead would also remove the seasonal barotropic signal, which is real.
"""
function bsose_exterior_free_surface(grid;
    dataset=BSOSEMonthly(),
    dates=nothing,
    reference_date=nothing,
    remove_datum=true)

    underlying = grid isa ImmersedBoundaryGrid ? grid.underlying_grid : grid
    cpu_grid = architecture(underlying) isa GPU ? on_architecture(CPU(), underlying) : underlying
    Nx, Ny, _ = size(cpu_grid)

    λc = collect(λnodes(cpu_grid, Center()))
    φc = collect(φnodes(cpu_grid, Center()))
    λf = collect(λnodes(cpu_grid, Face()))
    φf = collect(φnodes(cpu_grid, Face()))

    # `resolve_bsose_filename` returns a bare filename; callers join it with the directory.
    path = joinpath(dataset.dir, resolve_bsose_filename(dataset.dir, :sea_surface_height, dataset.iteration))
    @info "Loading exterior free surface (ETAN, $(basename(path)))..."

    stack, times = NCDataset(path) do ds
        XC = Float64.(ds["XC"][:])
        YC = Float64.(ds["YC"][:])
        stamps = DateTime.(ds["time"][:])
        Depth = Float64.(coalesce.(ds["Depth"][:, :], 0.0))

        want = dates === nothing ? collect(eachindex(stamps)) :
               [argmin(abs.(Dates.value.(Millisecond.(stamps .- DateTime(d))))) for d in dates]
        nearest(xs, x) = argmin(abs.(xs .- x))

        # One source for everything. The faces are sliced out of this below rather than read
        # separately: reading the west face at λf and the interior at λc put them in
        # neighbouring BSOSE columns with different coastlines, and the land fill then chose
        # different neighbours, leaving ηᵇ and ηᵉˣᵗ disagreeing by up to 0.12 m near Antarctica
        # -- 58 Sv through √(gH). Half a cell of a smooth field is worth about a millimetre;
        # a fill discontinuity is worth a hundred times that, so consistency wins.
        iall = [nearest(XC, λ) for λ in λc]
        jall = [nearest(YC, φ) for φ in φc]

        η = fill(NaN, Nx, Ny, length(want))
        for (t, rec) in enumerate(want)
            blk = Float64.(coalesce.(ds["ETAN"][iall, jall, rec], NaN))
            for i in 1:Nx, j in 1:Ny
                # ETAN marks land with exactly 0.0, never NaN; Depth is the truth.
                η[i, j, t] = Depth[iall[i], jall[j]] == 0 ? NaN : blk[i, j]
            end
        end

        (η, [Dates.value(Second(stamps[rec] - (reference_date === nothing ? stamps[want[1]] :
                                               DateTime(reference_date)))) for rec in want])
    end

    # Datum, from ocean points only and before any filling. ETAN is an anomaly about BSOSE's
    # own reference; the model's η starts at zero domain mean. Removing the domain mean lines
    # the two up while leaving the spatial structure and the seasonal variation intact.
    datum = 0.0
    if remove_datum
        wet = filter(!isnan, vec(stack))
        datum = sum(wet) / length(wet)
        stack .-= datum
    end

    # Fill land along latitude. These points are inside the immersed boundary and never enter
    # the dynamics, but a NaN would propagate through `set!` and a zero would stand a 2 m
    # cliff against the neighbouring ocean.
    gaps = 0
    for t in axes(stack, 3), i in axes(stack, 1)
        col = @view stack[i, :, t]
        valid = findall(!isnan, col)
        if isempty(valid)
            col .= 0
        else
            for j in eachindex(col)
                isnan(col[j]) || continue
                col[j] = col[valid[argmin(abs.(valid .- j))]]
                gaps += 1
            end
        end
    end
    gaps > 0 && @info @sprintf(" -> filled %d land points from the nearest wet neighbour in latitude", gaps)
    @info @sprintf(" -> ηᵉˣᵗ: datum %+.3f m removed, residual range %+.3f to %+.3f m (span %.3f m)",
        datum, minimum(stack), maximum(stack), maximum(stack) - minimum(stack))

    arch = architecture(grid)
    faces = (west = stack[1, :, :], east = stack[Nx, :, :],
             south = stack[:, 1, :], north = stack[:, Ny, :])

    series(side) = begin
        along_x = side === :south || side === :north
        # Same shape and the same two reasons as `bsose_barotropic_transport`; see the note there.
        fts = along_x ?
              FieldTimeSeries{Center,Nothing,Center}(cpu_grid, times; indices=(:, 1, :), time_indexing=Cyclical()) :
              FieldTimeSeries{Nothing,Center,Center}(cpu_grid, times; indices=(1, :, :), time_indexing=Cyclical())
        for t in eachindex(times)
            along_x ? (interior(fts[t], :, 1, :) .= reshape(faces[side][:, t], :, 1)) :
                      (interior(fts[t], 1, :, :) .= reshape(faces[side][:, t], :, 1))
            fill_halo_regions!(fts[t])
        end
        arch isa GPU ? on_architecture(arch, fts) : fts
    end

    return (west=series(:west), east=series(:east), south=series(:south), north=series(:north),
        initial=stack[:, :, 1], datum=datum)
end


"""
    bsose_penetrating_shortwave(grid;
                                dataset = BSOSEMonthly(),
                                dates = nothing,
                                ρ₀ = 1026.0,
                                cₚ = 3991.86795711963,
                                reference_date = nothing)

Penetrating solar radiation from BSOSE's `oceQsw`, as `(radiation, update_shortwave!)`.

`radiation` is a `TwoColorRadiation` to hand to `ocean_simulation(grid; radiative_forcing)`.
It spreads the surface shortwave through the column by Beer's law in two bands — red, 58% of
it absorbed within 0.35 m, and blue-green, the other 42%, decaying over 23 m — the Paulson
and Simpson (1977) Jerlov Type I constants that MITgcm's `SWFRAC` also defaults to. For the
5 m top cell here that leaves 66% of the sunlight in the first cell and sends 34% deeper,
against the 100% a plain surface flux deposits.

`update_shortwave!(sim)` interpolates `oceQsw` onto the model clock and writes it into
`radiation.surface_flux`. **It has to be registered as a callback.** `ocean_simulation`
already builds a `TwoColorRadiation` by default, but nothing fills its surface flux outside a
coupled `EarthSystemModel`, so the field sits at zero and the scheme silently does nothing —
which is the state this function exists to correct.

    radiation, update_shortwave! = bsose_penetrating_shortwave(grid; dataset, dates)
    ocean = ocean_simulation(grid; radiative_forcing=radiation, ...)
    simulation.callbacks[:shortwave] = Callback(update_shortwave!, IterationInterval(1))

Pair it with `bsose_surface_tracer_fluxes(...; subtract_shortwave=true)`, which takes the
solar part back out of the top boundary condition. `oceQsw` is a *component* of `TFLUX`, not
a flux in addition to it, so using one without the other either counts the sunlight twice or
drops it entirely.

### Sign convention

`surface_flux` takes the opposite sign to a top `FluxBoundaryCondition`, which is the easiest
thing to get wrong here. An Oceananigans top flux is positive UPWARD, out of the ocean. The
radiation integrates its own flux divergence back to `J₀`, so it wants a kinematic flux
positive DOWNWARD, into the ocean. BSOSE's oceQsw is already "+= down", so it enters as
`+oceQsw / (ρ₀ cₚ)` where TFLUX enters as `-TFLUX / (ρ₀ cₚ)`.
"""
function bsose_penetrating_shortwave(grid;
    dataset=BSOSEMonthly(),
    dates=nothing,
    ρ₀=1026.0,
    cₚ=3991.86795711963,
    reference_date=nothing)

    if isnothing(dates)
        available = all_dates(dataset, :surface_shortwave)
        dates = available[1:min(12, length(available))]
    end

    @info "Loading penetrating shortwave radiation (oceQsw, $(summary(dataset)))..."
    region = bsose_region(grid)
    shortwave_metadata = Metadata(:surface_shortwave; dataset, dates, region)
    shortwave_fts = resident_fts(shortwave_metadata, grid)

    time_offset = model_clock_offset(shortwave_metadata, reference_date)
    Qsw = boundary_slice_time_series(shortwave_fts, :top; time_offset)

    # Kinematic and positive DOWNWARD; see the sign note above. As with the wind stress and the
    # tracer fluxes the scaling goes on the slice, because the dataset-backed series that
    # `FieldTimeSeries(::Metadata, grid)` returns holds two snapshots and reloads them from
    # disk, so scaling that one is silently discarded.
    for t in 1:length(Qsw.times)
        parent(Qsw[t]) .*= 1 / (ρ₀ * cₚ)
    end

    W(x) = x * ρ₀ * cₚ   # back to W m⁻² for the messages
    Qmin, Q̄, Qmax = series_statistics(Qsw)

    # Net shortwave into the ocean cannot be negative. If it is, the file's convention is not
    # the one assumed above, and the run would heat the ocean in winter and cool it in summer.
    # Cheaper to find out here than twelve hours into a job.
    if Q̄ <= 0
        error("""BSOSE oceQsw averages $(@sprintf("%.1f", W(Q̄))) W m⁻² over this domain and period, which is not \
                 positive. `bsose_penetrating_shortwave` assumes the MITgcm convention the file \
                 states for itself ("net Short-Wave radiation (+=down), >0 increases theta"); \
                 check oceQsw's `long_name` attribute before going further.""")
    end
    Qmin < 0 && @warn @sprintf("BSOSE oceQsw dips to %.2f W m⁻²; net shortwave into the ocean should not go below zero.", W(Qmin))

    @info @sprintf(" -> Penetrating shortwave: mean %+.1f W m⁻², peak %+.1f W m⁻²", W(Q̄), W(Qmax))

    radiation = TwoColorRadiation(grid)
    @info " -> $(sprint(show, radiation))"

    # Interpolating the series onto the model clock looks like it should be
    # `Qsw[Time(clock.time)]`, but that path is unavailable here: the sliced series is
    # `Nothing`-located in z while carrying a non-colon z index, and Oceananigans'
    # `restrict_index_on_location` has no method for that pair, so building the lazy
    # interpolation throws. `TimeSeriesInterpolation` reaches the same result through
    # `getindex(i, j, k)`, honours the same `Cyclical` time indexing, and adapts to the GPU.
    #
    # It needs the model's clock, which does not exist until `ocean_simulation` has been
    # called, so it is built on the first invocation and held. The clock is mutable and
    # updated in place, so one interpolator serves the whole run.
    interpolator = Ref{Any}(nothing)

    function update_shortwave!(sim)
        if interpolator[] === nothing
            interpolator[] = TimeSeriesInterpolation(Qsw, grid; clock=sim.model.clock)
        end
        set!(radiation.surface_flux, interpolator[])
        return nothing
    end

    return radiation, update_shortwave!
end


# ==============================================================================
# 4. Open Boundary Conditions (OBCs)
# ==============================================================================

"""
    bsose_tangential_boundary_condition(value, kind; gravity_wave_speed = 210.0)

The boundary condition for the velocity component *tangential* to a face — `u` on
south/north, `v` on west/east — given the BSOSE `value` there.

This is the setting that governs the meridional jet pinned along the western boundary. A
tangential velocity is Center-located in the boundary-normal direction, so it is the same
case the schemes document for tracers, and the split-explicit free surface gives the
barotropic tangential velocity a zero-gradient (NoFlux) condition at these faces whatever
the 3D field uses.

- `:gradient` — `∂v/∂x = 0`, imposing nothing. Cannot pin the flow, matches what the
  barotropic mode does, and is the conventional tangential treatment in regional models.
  The BSOSE value still reaches the boundary through the sponge layer.
- `:clamped` — plain Dirichlet to the BSOSE value. Fills the halo by mirroring the interior
  about it (`v_halo = 2 v̄ - v_interior`), which over-specifies outflow and drives a
  boundary overshoot against the biharmonic viscosity.
- `:radiation` — `NormalRadiation`, an Orlanski phase-speed condition. Documented for
  tangential velocities, and it takes its advecting velocity one cell into the interior and
  checks for immersed nodes, neither of which the PerturbationAdvection `Value` path does.
- `:radiation_boundary_velocity` — the same, advecting with the boundary-face velocity.
- `:perturbation_advection` — `PerturbationAdvection` as-is. Known to diverge on a
  tangential component within ~5 time steps; kept so a test matrix can show that.
- `:oblique` — `ObliqueRadiation`, the Raymond & Kuo (1984) two-dimensional radiation
  condition. Unlike the one-dimensional schemes it carries a boundary-tangential phase speed
  as well as a normal one, so a feature crossing the boundary obliquely — which is most of
  them at a corner — is radiated along its own direction of travel rather than only normal to
  the face. This is the configuration the MAB sandbox runs: oblique on the normal flow and on
  the tangential component alike, nudged over `oblique_inflow_timescale` on inflow and
  `oblique_outflow_timescale` on outflow.
- `:oblique_boundary_velocity` — the same, deciding inflow from the boundary-face velocity
  rather than one cell in.
- `:perturbation_advection_gravity` — `PerturbationAdvection` with a nonzero
  `gravity_wave_speed`. The phase speed then saturates the scheme's `Ũ` at 1 rather than
  the ~10⁻³ a 0.1 m/s tangential flow gives, so the halo relaxes toward the interior
  instead of holding its own stale value. This is the form most likely to make
  PerturbationAdvection usable here; √(gH) ≈ 210 m/s for a 4.5 km water column.
"""
function bsose_tangential_boundary_condition(value, kind; gravity_wave_speed=210.0,
    oblique_inflow_timescale=3days, oblique_outflow_timescale=360days,
    radiation_inflow_timescale=1days)
    kind === :gradient && return GradientBoundaryCondition(0)
    kind === :clamped && return ValueBoundaryCondition(value)

    kind === :oblique && return ValueBoundaryCondition(value;
        scheme=ObliqueRadiation(inflow_timescale=oblique_inflow_timescale,
            outflow_timescale=oblique_outflow_timescale))

    kind === :oblique_boundary_velocity && return ValueBoundaryCondition(value;
        scheme=ObliqueRadiation(inflow_timescale=oblique_inflow_timescale,
            outflow_timescale=oblique_outflow_timescale, use_boundary_velocity=true))

    kind === :radiation && return ValueBoundaryCondition(value;
        scheme=NormalRadiation(inflow_timescale=radiation_inflow_timescale, outflow_timescale=Inf))

    kind === :radiation_boundary_velocity && return ValueBoundaryCondition(value;
        scheme=NormalRadiation(inflow_timescale=radiation_inflow_timescale, outflow_timescale=Inf,
            use_boundary_velocity=true))

    kind === :perturbation_advection && return ValueBoundaryCondition(value;
        scheme=PerturbationAdvection(inflow_timescale=1days, outflow_timescale=Inf))

    kind === :perturbation_advection_gravity && return ValueBoundaryCondition(value;
        scheme=PerturbationAdvection(inflow_timescale=1days, outflow_timescale=Inf,
            gravity_wave_speed=gravity_wave_speed))

    error("Unknown tangential_bc_kind: $kind. Choose :gradient, :clamped, :radiation, " *
          ":radiation_boundary_velocity, :oblique, :oblique_boundary_velocity, " *
          ":perturbation_advection or :perturbation_advection_gravity.")
end

"""
    usable_boundary_cache(path)

Whether the unified boundary-condition dataset at `path` can be opened and holds a complete
set of slices. A job killed partway through writing one leaves a file that exists but cannot
be read, and several jobs submitted together share this path, so an existing file is not by
itself evidence of a good one.
"""
function usable_boundary_cache(path)
    try
        return JLD2.jldopen(path, "r") do f
            all(haskey(f, k) for k in ("u_west", "u_east", "v_south", "v_north",
                "T_west", "T_east", "S_south", "S_north"))
        end
    catch err
        @warn "Boundary condition cache $path could not be read ($(typeof(err))); rebuilding it."
        return false
    end
end

"""
    bsose_open_boundary_conditions(grid;
                                   dataset = BSOSEMonthly(),
                                   dates = all_dates(dataset, :temperature)[1:12],
                                   scheme = nothing,
                                   winds = nothing,
                                   ρ₀ = 1026.0)

Construct a `NamedTuple` of `FieldBoundaryConditions` `(; u, v, T, S)` configured with
open boundary conditions from BSOSE — plus `(; U, V)` when `barotropic = true`.

# Arguments
- `grid`: The simulation `LatitudeLongitudeGrid`.
- `dataset`: `BSOSEMonthly()` instance.
- `dates`: Range or collection of dates for open boundary forcing.
- `scheme`: Matching scheme for the open boundaries. Defaults to `nothing` (clamped Dirichlet).
            Can be an `OpenBoundaryScheme` such as `PerturbationAdvection(inflow_timescale=1days, outflow_timescale=Inf)`
            or `NormalRadiation(inflow_timescale=1days, outflow_timescale=Inf)` to allow interior waves/eddies
            to radiate cleanly out of outflow boundaries (e.g. the eastern ACC outflow) without reflection.
- `winds`: Surface wind forcing. Options:
           - `nothing` or `false` (default): Disables surface wind forcing (no-flux top boundary).
           - `true`: Automatically calls `bsose_surface_wind_stress` and applies top flux.
           - A `NamedTuple` `(; u, v)` returned from `bsose_surface_wind_stress(grid; ...)`.
- `penetrating_shortwave`: With `surface_fluxes = true`, whether the solar part of the heat
                flux is being distributed through the column by
                [`bsose_penetrating_shortwave`](@ref) rather than applied at the surface. When
                `true` the top temperature flux is built from `TFLUX - oceQsw` instead of
                `TFLUX`, so the sunlight is not counted twice. Setting this without also
                constructing the radiation, or the reverse, silently drops or doubles the
                shortwave — they are two halves of one change.
- `ρ₀`: Seawater density for wind stress conversion (kg/m³). Default: 1026.0.
- `reference_date`: Calendar date of model time zero, so that the boundary and wind series are
                    stepped on the model clock; see [`model_clock_offset`](@ref). `nothing`
                    (default) leaves them on the dataset's own clock.
- `barotropic`: If `true`, also returns Flather conditions on the depth-integrated transport
                `U` and `V`, fed the BSOSE transport through each face, and lets Oceananigans
                pair the companion Chapman condition onto `η`. **This is not an optional
                refinement.** The split-explicit corrector replaces the depth mean of `u` and
                `v` at the boundary with the barotropic solution, so while `U`/`V` keep their
                default impenetrable `NormalFlow(nothing)` the net transport through every face
                is held at zero whatever `scheme` is doing — see the section comment above
                [`bsose_barotropic_transport`](@ref).
- `exterior_free_surface`: `ηᵉˣᵗ` in the Flather condition. `0` (default) drives the barotropic
                mode by transport alone, which is all BSOSE can do here: its `ETAN` is not among
                the files this module reads. A `FieldTimeSeries` of exterior sea surface height
                (GLORYS `zos`, say) can be passed instead.
- `cache`: If `true`, saves and loads all boundary conditions to/from a SINGLE unified dataset file.
           The barotropic transports are derived from the cached slices rather than stored, so
           turning `barotropic` on does not invalidate an existing cache.
- `cache_file`: Optional custom file path for the unified dataset file.
"""
function bsose_open_boundary_conditions(grid;
    dataset=BSOSEMonthly(),
    dates=all_dates(dataset, :temperature)[1:12],
    scheme=nothing,
    winds=nothing,
    surface_fluxes=false,
    penetrating_shortwave=false,
    tangential_bc_kind=:gradient,
    tracer_scheme=:same,
    radiation_inflow_timescale=1days,
    barotropic=false,
    barotropic_faces=:zonal,
    barotropic_scheme=:pinned,
    balance_transports=true,
    exterior_free_surface=0,
    tangential_gravity_wave_speed=210.0,
    oblique_inflow_timescale=3days,
    oblique_outflow_timescale=360days,
    ρ₀=1026.0,
    reference_date=nothing,
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
    wind_tag = winds === true ? "_winds" : ""
    default_dataset_name = "bsose_dataset_v2_$(dataset.iteration)_$(start_str)_to_$(end_str)_$(Nx)x$(Ny)x$(Nz)$(wind_tag).jld2"
    dataset_path = cache_file !== nothing ? cache_file : joinpath(dataset.dir, default_dataset_name)

    # 1. Attempt loading from single unified dataset
    if cache && isfile(dataset_path) && usable_boundary_cache(dataset_path)
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
        u_meta = Metadata(:u_velocity; dataset, dates, region)
        time_offset = model_clock_offset(u_meta, reference_date)
        u_fts = resident_fts(u_meta, grid)
        @info " -> Loading BSOSE v_velocity..."
        v_fts = resident_fts(Metadata(:v_velocity; dataset, dates, region), grid)
        @info " -> Loading BSOSE temperature..."
        T_fts = resident_fts(Metadata(:temperature; dataset, dates, region), grid)
        @info " -> Loading BSOSE salinity..."
        S_fts = resident_fts(Metadata(:salinity; dataset, dates, region), grid)

        # 2. Extract 2D boundary slices
        @info " -> Extracting boundary slices (West, East, South, North)..."
        u_west = boundary_slice_time_series(u_fts, :west; time_offset)
        u_east = boundary_slice_time_series(u_fts, :east; time_offset)
        u_south = boundary_slice_time_series(u_fts, :south; time_offset)
        u_north = boundary_slice_time_series(u_fts, :north; time_offset)

        v_west = boundary_slice_time_series(v_fts, :west; time_offset)
        v_east = boundary_slice_time_series(v_fts, :east; time_offset)
        v_south = boundary_slice_time_series(v_fts, :south; time_offset)
        v_north = boundary_slice_time_series(v_fts, :north; time_offset)

        T_west = boundary_slice_time_series(T_fts, :west; time_offset)
        T_east = boundary_slice_time_series(T_fts, :east; time_offset)
        T_south = boundary_slice_time_series(T_fts, :south; time_offset)
        T_north = boundary_slice_time_series(T_fts, :north; time_offset)

        S_west = boundary_slice_time_series(S_fts, :west; time_offset)
        S_east = boundary_slice_time_series(S_fts, :east; time_offset)
        S_south = boundary_slice_time_series(S_fts, :south; time_offset)
        S_north = boundary_slice_time_series(S_fts, :north; time_offset)

        top_u_cached = nothing
        top_v_cached = nothing

        # Compute winds if enabled so they can be saved into the single dataset
        if winds === true
            wind_bcs = bsose_surface_wind_stress(grid; dataset, dates, ρ₀, reference_date)
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

        # Save all boundary conditions together into ONE single dataset.
        #
        # Written to a private temporary file and renamed into place. Jobs submitted together
        # share this path — it is keyed on the dataset, the dates and the grid size, none of
        # which vary between boundary-condition experiments — and JLD2 writes through an
        # mmap, so two processes creating the same file at once die with SIGBUS rather than
        # raising an error. A rename within one filesystem is atomic, so each writer builds
        # its own copy and the last to finish wins, while a reader sees either the previous
        # complete file or the new one, never a half-written mapping.
        if cache
            @info "Saving all boundary conditions together into single dataset: $dataset_path"
            tmp_path = string(dataset_path, ".tmp.", getpid(), ".", string(rand(UInt32), base=16))
            JLD2.jldopen(tmp_path, "w") do f
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
            mv(tmp_path, dataset_path; force=true)
        end
    end

    # Step 1: Zero out normal velocity components on immersed (solid bathymetry) boundary faces
    # so that open boundaries do not force flow into or across solid land/bottom topography.
    if grid isa ImmersedBoundaryGrid
        @info "Masking normal boundary flow on immersed boundary faces..."
        u_west = mask_immersed_boundary_normal_flow!(u_west, :west, grid)
        u_east = mask_immersed_boundary_normal_flow!(u_east, :east, grid)
        v_south = mask_immersed_boundary_normal_flow!(v_south, :south, grid)
        v_north = mask_immersed_boundary_normal_flow!(v_north, :north, grid)
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

    # 2d. Barotropic (depth-integrated) open boundary conditions.
    #
    # Without these the split-explicit corrector overwrites the depth mean of the
    # prescribed u/v at every face with the impenetrable barotropic solution, so no net
    # transport can cross the boundary however the baroclinic condition is configured.
    # Built from the slices AFTER immersed masking, so each integral already covers only
    # the model's wet column — see `bsose_barotropic_transport`.
    U_bcs = nothing
    V_bcs = nothing

    if barotropic === true
        @info "Building barotropic (Flather) boundary conditions for U and V..."

        # Flather prescribes the incoming Riemann invariant U - √(gH)η, so Uᵉˣᵗ and ηᵉˣᵗ are
        # one pair, not two settings. A per-face `NamedTuple` from
        # `bsose_exterior_free_surface` supplies the real sea surface; a bare number (the old
        # default of 0) asserts a flat boundary, which contradicts any prescribed transport --
        # at √(gH) = 198 m/s the neglected term outweighs the transport itself.
        η_at(side) = exterior_free_surface isa NamedTuple ?
                     getproperty(exterior_free_surface, side) : exterior_free_surface

        if !(exterior_free_surface isa NamedTuple) && barotropic_scheme === :flather
            @info """ -> ηᵉˣᵗ is a bare $(typeof(exterior_free_surface)): every open face is told its sea \
                     surface is flat, and Flather carries transport only. Measured in \
                     obc_tests/flather_sandbox.jl, a uniform ηᵉˣᵗ only shifts the datum (0 to 0.04 m moved \
                     the transport 0.3%), whereas a tilt adds 60.8 Sv per metre regardless of Uᵉˣᵗ. Supply \
                     `bsose_exterior_free_surface(grid; ...)` only when the model's own η can be shown to \
                     hold BSOSE's 2 m tilt; otherwise this is the conservative setting."""
        end

        V_south_transport = bsose_barotropic_transport(v_south, :south, grid)
        V_north_transport = bsose_barotropic_transport(v_north, :north, grid)

        U_west_transport = is_x_periodic ? nothing : bsose_barotropic_transport(u_west, :west, grid)
        U_east_transport = is_x_periodic ? nothing : bsose_barotropic_transport(u_east, :east, grid)

        # Take the net imbalance out before any of these reach a boundary condition. See
        # `balance_barotropic_transports!`: +3 Sv left in drains the domain, and Flather pays
        # for that with a free-surface offset large enough to reverse a face.
        if balance_transports
            # Only the faces that can actually pass flux take part. With `:zonal` the
            # north/south depth-integrated flux is zero by construction, so including them
            # would hand correction to faces that cannot carry it and leave west/east
            # unbalanced -- and with the meridional faces shut, west and east are the only
            # places the domain can conserve volume.
            open_faces = barotropic_faces === :all ?
                (west=U_west_transport, east=U_east_transport,
                 south=V_south_transport, north=V_north_transport) :
                (west=U_west_transport, east=U_east_transport,
                 south=nothing, north=nothing)
            balanced = balance_barotropic_transports!(open_faces, grid)
            U_west_transport  = get(balanced, :west,  U_west_transport)
            U_east_transport  = get(balanced, :east,  U_east_transport)
            V_south_transport = get(balanced, :south, V_south_transport)
            V_north_transport = get(balanced, :north, V_north_transport)
        else
            @info " -> barotropic transports left UNBALANCED (balance_transports = false)."
        end

        # Flather on all four faces by default; `:zonal` closes north/south barotropically.
        #
        # `:zonal` exists but is not recommended. See the long note on BAROTROPIC_FACES in
        # src/model.jl: tools/inspect_bsose_boundaries.jl measures the north face carrying
        # -31 Sv (18% of the zonal transport, feeding the +31 Sv by which east exceeds west),
        # not the ~1% it was assumed to carry, so closing it imposes a real distortion rather
        # than removing a spurious one. Keep it for the diagnostic case where you want the
        # meridional exchange removed deliberately.
        # How the barotropic mode is constrained: `:pinned` or `:flather`.
        #
        # Flather alone does not prescribe transport. It prescribes Uᵇ - √(gH)ηᵇ, so it is only
        # as good as ηᵉˣᵗ, and the stiffness here is brutal:
        # tools/inspect_bsose_boundaries.jl measures 5.1 Sv per cm of (ηᵇ - ηᵉˣᵗ) error on
        # west/east and 9.5 Sv per cm on north. The model's free surface would have to track
        # BSOSE's ETAN to the centimetre for that to be well posed, and it does not. Initialised
        # from BSOSE, max|η| falls 3.40 -> 1.32 m within a day as the field radiates into balance
        # with the model's own dynamics (√(gH) = 200 m/s crosses this domain in ~4 h). What is
        # left is the model's own η, over a metre from BSOSE's at the north face.
        #
        # Two Flather runs confirm it, and they bracket zero:
        #
        #     ηᵉˣᵗ = 0        north +213 to +219 Sv   (prescribed -31)
        #     ηᵉˣᵗ = BSOSE    north -393 to -437 Sv   (prescribed -31)
        #
        # Same magnitude, opposite sign: too stiff, not mis-signed. Volume stayed conserved
        # throughout (net out within ±0.5 Sv), so it was never a leak -- it was a ~400 Sv
        # spurious overturning cell, in at the north and out through both zonal faces, which is
        # the corner pileup and the 5.3 m/s |u|ₘₐₓ.
        #
        # `:pinned` keeps Flather but hands it a `target_transport`. Oceananigans then shifts the
        # face transport uniformly after the Flather fill of every barotropic substep so its
        # integral along the boundary matches the target: the Flather profile sets the *shape*,
        # the target sets the *net*. The free-surface error can no longer move net volume, only
        # redistribute it along the face. Radiation is retained, unlike a plain Dirichlet.
        #
        # The one compromise: `boundary_transport` is materialised once when the free surface is
        # built, so the target is a single number per face for the whole run. We pin the time
        # mean and lose the seasonal swing of the net (west +183 to +212, north -31 to +16).
        # The shape, and all the baroclinic structure, still vary in time.
        #
        # Note a plain `NormalFlowBoundaryCondition(transport_series)` does NOT work here: a
        # `FieldTimeSeries` falls into `getbc`'s `AbstractArray` branch, which reads
        # `condition[i, j, 1]` -- no time interpolation, and it throws on the adapted type.
        barotropic_bc = if barotropic_scheme === :pinned
            # Targets must close to zero or the pinned net is a permanent volume source and η
            # ramps linearly forever. The balanced series already closes per record; this makes
            # the time means close exactly.
            tt = Dict(s => mean_face_transport(getfield((west=U_west_transport, east=U_east_transport,
                                                         south=V_south_transport, north=V_north_transport), s),
                                               s, grid)
                      for s in (:west, :east, :south, :north))
            open_sides = [s for s in (:west, :east, :south, :north)
                          if (barotropic_faces === :all || s in (:west, :east)) && sum(face_geometry(s, grid)[2]) > 0]
            outward = (west=-1.0, east=+1.0, south=-1.0, north=+1.0)
            areas = Dict(s => sum(face_geometry(s, grid)[1] .* face_geometry(s, grid)[2]) for s in open_sides)
            net = sum(outward[s] * tt[s] for s in open_sides)
            total_area = sum(values(areas))
            for s in open_sides
                tt[s] -= outward[s] * net * areas[s] / total_area
            end
            @info @sprintf(" -> barotropic scheme: Flather + pinned transport. Targets (Sv): W %+.2f, E %+.2f, S %+.2f, N %+.2f; net %+.2e",
                tt[:west]/1e6, tt[:east]/1e6, tt[:south]/1e6, tt[:north]/1e6,
                sum(outward[s] * tt[s] for s in open_sides)/1e6)
            (transport, side) -> GravityWaveRadiationBoundaryCondition((transport, η_at(side));
                                                                       target_transport = tt[side])
        elseif barotropic_scheme === :flather
            @info " -> barotropic scheme: Flather, transport NOT pinned."
            (transport, side) -> GravityWaveRadiationBoundaryCondition((transport, η_at(side)))
        else
            error("Unknown barotropic_scheme: $barotropic_scheme. Choose :pinned or :flather.")
        end

        if barotropic_faces === :all
            V_bcs = FieldBoundaryConditions(grid, (Center(), Face(), nothing);
                south=barotropic_bc(V_south_transport, :south),
                north=barotropic_bc(V_north_transport, :north))
        elseif barotropic_faces === :zonal
            @info " -> barotropic faces: west and east only; north/south depth-integrated flux held at zero."
        else
            error("Unknown barotropic_faces: $barotropic_faces. Choose :zonal or :all.")
        end

        if !is_x_periodic
            U_bcs = FieldBoundaryConditions(grid, (Face(), Center(), nothing);
                west=barotropic_bc(U_west_transport, :west),
                east=barotropic_bc(U_east_transport, :east))
        end

        @info " -> η gets its companion Chapman condition automatically from U/V."
    else
        @info " -> Barotropic boundary conditions are DISABLED: the depth-integrated " *
              "transport through every face is held at zero by the default impenetrable U/V."
    end

    # 3. Top boundary condition (surface wind stress)
    top_u_bc = FluxBoundaryCondition(nothing)
    top_v_bc = FluxBoundaryCondition(nothing)

    if winds === true
        if top_u_cached !== nothing && top_v_cached !== nothing
            top_u_bc = top_u_cached
            top_v_bc = top_v_cached
        else
            wind_bcs = bsose_surface_wind_stress(grid; dataset, dates, ρ₀, reference_date)
            top_u_bc = wind_bcs.u
            top_v_bc = wind_bcs.v
        end
    elseif winds isa NamedTuple
        top_u_bc = get(winds, :u, FluxBoundaryCondition(nothing))
        top_v_bc = get(winds, :v, FluxBoundaryCondition(nothing))
    else
        @info " -> Surface wind stress is DISABLED (top boundary is no-flux)."
    end

    # Surface heat and salt fluxes
    top_T_bc = FluxBoundaryCondition(nothing)
    top_S_bc = FluxBoundaryCondition(nothing)

    if surface_fluxes === true
        tracer_flux_bcs = bsose_surface_tracer_fluxes(grid; dataset, dates, ρ₀, reference_date,
            subtract_shortwave=penetrating_shortwave)
        top_T_bc = tracer_flux_bcs.T
        top_S_bc = tracer_flux_bcs.S
    elseif surface_fluxes isa NamedTuple
        top_T_bc = get(surface_fluxes, :T, FluxBoundaryCondition(nothing))
        top_S_bc = get(surface_fluxes, :S, FluxBoundaryCondition(nothing))
    else
        @info " -> Surface heat and salt fluxes are DISABLED (no air-sea exchange)."
    end


    # 4. Construct FieldBoundaryConditions
    # Note: On East/West boundaries, normal velocity is u (NormalFlow), tangential is v (Value).
    #       On South/North boundaries, normal velocity is v (NormalFlow), tangential is u (Value).
    # If the domain is longitudinally periodic (e.g. Circumpolar), only South and North BCs are applied.
    #
    # The tangential condition is selected by `tangential_bc_kind`; see
    # [`bsose_tangential_boundary_condition`](@ref) for what each option does and why it
    # matters. Clamping to a prescribed value over-specifies an outflow boundary and leaves
    # a fast meridional jet pinned along the western edge.
    tangential_bc(value) = bsose_tangential_boundary_condition(value, tangential_bc_kind;
        gravity_wave_speed=tangential_gravity_wave_speed,
        oblique_inflow_timescale, oblique_outflow_timescale, radiation_inflow_timescale)

    # Tracers can take a different scheme from the normal flow.
    #
    # `:same` keeps the historical behaviour -- T and S inherit whatever `scheme` the normal
    # velocity uses. A `TracerReservoir` instead carries a reservoir value just outside each
    # boundary point (after MOM6's tracer reservoirs) and relaxes it toward the upstream value
    # over a length scale: the exterior value on inflow, the adjacent interior value on outflow.
    # With `outflow_length_scale = 0` an outflowing cell simply takes the interior value, which
    # is the zero-gradient behaviour you want when water is leaving.
    #
    # This is what NestingValidation's canary setup runs, and it is also the configuration that
    # sidesteps their ObliqueRadiation-on-tracers divergence: the tracer boundary no longer
    # carries a tangential phase speed at all, so a land-adjacent cell has nothing to go wrong.
    tracer_bc(value) = tracer_scheme === :same ?
        ValueBoundaryCondition(value; scheme) :
        ValueBoundaryCondition(value; scheme=tracer_scheme)
    if is_x_periodic
        @info " -> Longitude is Periodic (circumpolar): applying South & North boundary conditions."
        u_bcs = FieldBoundaryConditions(
            south=tangential_bc(u_south),
            north=tangential_bc(u_north),
            top=top_u_bc
        )

        v_bcs = FieldBoundaryConditions(
            south=NormalFlowBoundaryCondition(v_south; scheme),
            north=NormalFlowBoundaryCondition(v_north; scheme),
            top=top_v_bc
        )

        T_bcs = FieldBoundaryConditions(
            south=tracer_bc(T_south),
            north=tracer_bc(T_north),
            top=top_T_bc
        )

        S_bcs = FieldBoundaryConditions(
            south=tracer_bc(S_south),
            north=tracer_bc(S_north),
            top=top_S_bc
        )

    else
        @info " -> Longitude is Bounded (regional): applying West, East, South, and North boundary conditions."
        u_bcs = FieldBoundaryConditions(
            west=NormalFlowBoundaryCondition(u_west; scheme),
            east=NormalFlowBoundaryCondition(u_east; scheme),
            south=tangential_bc(u_south),
            north=tangential_bc(u_north),
            top=top_u_bc
        )

        v_bcs = FieldBoundaryConditions(
            west=tangential_bc(v_west),
            east=tangential_bc(v_east),
            south=NormalFlowBoundaryCondition(v_south; scheme),
            north=NormalFlowBoundaryCondition(v_north; scheme),
            top=top_v_bc
        )

        T_bcs = FieldBoundaryConditions(
            west=tracer_bc(T_west),
            east=tracer_bc(T_east),
            south=tracer_bc(T_south),
            north=tracer_bc(T_north),
            top=top_T_bc
        )

        S_bcs = FieldBoundaryConditions(
            west=tracer_bc(S_west),
            east=tracer_bc(S_east),
            south=tracer_bc(S_south),
            north=tracer_bc(S_north),
            top=top_S_bc
        )
    end

    @info "BSOSE Open Boundary Conditions setup complete."
    bcs = (u=u_bcs, v=v_bcs, T=T_bcs, S=S_bcs)
    V_bcs !== nothing && (bcs = merge(bcs, (; V=V_bcs)))
    U_bcs !== nothing && (bcs = merge(bcs, (; U=U_bcs)))
    return bcs
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
                               restore_velocities = true,
                               reference_date = nothing)

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
    restore_velocities=true,
    reference_date=nothing)

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

    FT = restore_on_model_clock!(DatasetRestoring(T_meta, grid; rate=rate, mask=sponge_mask,
                                            time_indices_in_memory=max(1, length(T_meta))), reference_date)
    FS = restore_on_model_clock!(DatasetRestoring(S_meta, grid; rate=rate, mask=sponge_mask,
                                            time_indices_in_memory=max(1, length(S_meta))), reference_date)

    if restore_velocities
        @info " -> Restoring u and v velocities in boundary sponge layers to damp accelerations."
        u_meta = Metadata(:u_velocity; dataset, start_date, end_date, region)
        v_meta = Metadata(:v_velocity; dataset, start_date, end_date, region)
        Fu = restore_on_model_clock!(DatasetRestoring(u_meta, grid; rate=rate, mask=sponge_mask,
                                            time_indices_in_memory=max(1, length(u_meta))), reference_date)
        Fv = restore_on_model_clock!(DatasetRestoring(v_meta, grid; rate=rate, mask=sponge_mask,
                                            time_indices_in_memory=max(1, length(v_meta))), reference_date)
        return (u=Fu, v=Fv, T=FT, S=FS)
    else
        return (T=FT, S=FS)
    end
end

"""
    restore_on_model_clock!(restoring::DatasetRestoring, reference_date)

Shift the times of the series inside `restoring` onto the model clock, so that a sponge layer
reads the same calendar date as the open boundaries and the wind stress at any model time.
`DatasetRestoring` builds its own `FieldTimeSeries`, so the shift is applied after the fact.
"""
function restore_on_model_clock!(restoring, reference_date)
    isnothing(reference_date) && return restoring
    fts = restoring.field_time_series

    # Rebind rather than broadcast in place. `FieldTimeSeries` is a mutable struct, but its
    # `times` are not always a Vector — for uniformly spaced records they can be an
    # immutable range, and `.+=` on one of those throws `CanonicalIndexError: setindex!
    # not defined for StepRangeLen`. Adding a scalar to a range returns a range, so the
    # rebinding form works for either.
    fts.times = fts.times .+ model_clock_offset(fts.backend.metadata, reference_date)
    return restoring
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
    u_target = resident_fts(Metadata(:u_velocity; dataset, dates, region), grid)
    v_target = resident_fts(Metadata(:v_velocity; dataset, dates, region), grid)
    T_target = resident_fts(Metadata(:temperature; dataset, dates, region), grid)
    S_target = resident_fts(Metadata(:salinity; dataset, dates, region), grid)

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


# ==============================================================================
# ERA5 pre-fetch
#
# Rescued from src/glorys_obc.jl when that file was deleted (2026-10-23). It has nothing to do
# with GLORYS -- it only happened to live there because that was where the ERA5 path was first
# wired up -- and `src/model.jl` still calls it on the `ATMOSPHERE = :era5` path.
# ==============================================================================

using NumericalEarth.DataWrangling: MetadataSet

"""
    prefetch_era5!(; start_date, end_date, region, dir, threads = 1,
                   dataset = ERA5HourlySingleLevel())

Download every ERA5 variable the atmosphere and radiation need, `threads` requests at a time.

`ERA5PrescribedAtmosphere` and `prescribed_radiation` each bundle their variables into one
`MetadataSet` and call `Downloads.download(mset)` with no keyword arguments, and the
CopernicusClimateDataStore backend then defaults to one era5cli thread PER VARIABLE -- six
concurrent CDS requests for the atmosphere, two for the radiation. CDS refuses that:

    Number queued requests for this dataset is temporarily limited.
    Please configure your scripts accordingly

and all six come back `Rejected` within seconds. Worse, the client only raises on status
`failed`, so a `rejected` request is polled until `max_wait` (an hour) instead of erroring.

The backend does take a `threads` keyword, it just is not reachable through those constructors.
So the download happens here first, serially, and because `skip_existing = true` the
constructors then find every file cached and issue no requests of their own.

One variable at a time is slower but it is the difference between finishing and not. Raise
`threads` if CDS loosens the limit.
"""
function prefetch_era5!(; start_date, end_date, region, dir,
                        threads = 1, dataset = ERA5HourlySingleLevel())
    atmos_vars = (:eastward_velocity, :northward_velocity, :temperature,
                  :dewpoint_temperature, :surface_pressure, :total_precipitation)
    rad_vars   = (:downwelling_shortwave_radiation, :downwelling_longwave_radiation)

    all_vars = (atmos_vars..., rad_vars...)
    @info @sprintf(" -> pre-fetching %d ERA5 variables %d request(s) at a time (CDS limits concurrent queued requests)",
                   length(all_vars), threads)

    for (n, v) in enumerate(all_vars)
        @info "    [$n/$(length(all_vars))] $v"
        mset = MetadataSet(v; dataset, start_date, end_date, dir, region)
        Downloads.download(mset; threads)
    end
    @info " -> ERA5 pre-fetch complete; the atmosphere and radiation will read from cache."
    return nothing
end


# ==============================================================================
# Boundary topography conditioning
# ==============================================================================

"""
    fix_boundary_topography!(bottom_height, grid; minimum_boundary_depth = 100)

Condition the sea floor along the open boundaries. Two operations, in order:

1. **Mask columns too thin to carry transport.** The barotropic corrector sets
   `depth_mean(u) = U` whatever the column depth, so a transport correction ΔU becomes a
   velocity `ΔU/H`. On the east face `(360,10)` is **42 m** against neighbours of ~438 m --
   a 75x amplifier on a face averaging 3146 m, carrying ~0.00008 Sv. Masking it loses nothing
   and removes the amplification. The default 100 m threshold catches that column and nothing
   else: `(1,20)` at 297 m and `(1,21)` at 483 m are real shelf-break and are kept.

2. **MITgcm's `OBCSfixtopo`.** Raise any boundary column deeper than the cell inside it, so
   `d(topo)/dn >= 0` outward. MITgcm applies this unconditionally whatever the parent dataset.
   Measured here: 79 such columns on west, 68 on east, 218 on north -- 365 in all, worst 924 m.

Must be called BEFORE the `ImmersedBoundaryGrid` is built, since that is what freezes the wet
column the barotropic transport is computed over. Returns `(n_masked, n_raised)`.

This is a deliberate modification of the bathymetry and re-keys nothing on its own, but the
immersed grid built from it differs, so runs before and after are not directly comparable.
"""
function fix_boundary_topography!(bottom_height, grid; minimum_boundary_depth = 100)
    underlying = grid isa ImmersedBoundaryGrid ? grid.underlying_grid : grid
    Nx, Ny, _ = size(underlying)

    B = Array(interior(bottom_height))          # (Nx, Ny, 1), negative below sea level
    depth(i, j) = -B[i, j, 1]

    # which (i,j) sit on an open boundary, and which cell is "inward" from each
    edges = Tuple{Int,Int,Int,Int}[]            # (i, j, i_inward, j_inward)
    for j in 1:Ny
        push!(edges, (1, j, 2, j))
        push!(edges, (Nx, j, Nx - 1, j))
    end
    for i in 1:Nx
        push!(edges, (i, 1, i, 2))
        push!(edges, (i, Ny, i, Ny - 1))
    end

    # ---- 1. mask columns too thin to carry transport ----
    masked = Tuple{Int,Int,Float64}[]
    for (i, j, _, _) in edges
        d = depth(i, j)
        if 0 < d < minimum_boundary_depth
            push!(masked, (i, j, d))
            B[i, j, 1] = 0.0                    # land
        end
    end

    # ---- 2. OBCSfixtopo ----
    raised = Float64[]
    for (i, j, ii, jj) in edges
        db, di = depth(i, j), depth(ii, jj)
        if db > 0 && di > 0 && db > di
            push!(raised, db - di)
            B[i, j, 1] = -di
        end
    end

    set!(bottom_height, B)
    fill_halo_regions!(bottom_height)

    if isempty(masked)
        @info @sprintf(" -> boundary topography: no column shallower than %d m to mask", minimum_boundary_depth)
    else
        @info @sprintf(" -> boundary topography: masked %d column(s) under %d m: %s",
                       length(masked), minimum_boundary_depth,
                       join([@sprintf("(%d,%d) %.0fm", i, j, d) for (i, j, d) in masked], ", "))
    end
    if isempty(raised)
        @info " -> OBCSfixtopo: no boundary column was deeper than its inward neighbour"
    else
        @info @sprintf(" -> OBCSfixtopo: raised %d boundary column(s), mean %.0f m, worst %.0f m",
                       length(raised), sum(raised) / length(raised), maximum(raised))
    end

    return length(masked), length(raised)
end
