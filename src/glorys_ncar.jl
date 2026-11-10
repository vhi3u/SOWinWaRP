# glorys_ncar.jl -- NSF NCAR's copy of GLORYS12V1 (GDEX d010049) cut into the files NumericalEarth's
# GLORYSDaily reads. NCAR holds one global file per day with every variable,
#
#     <source>/<YYYY>/mercatorglorys12v1_gl12_mean_<YYYYMMDD>_R<YYYYMMDD>.nc    (or flat in <source>)
#
# the same product, variable names, dimensions and packing as Copernicus' cmems_mod_glo_phy_my_0.083deg_P1D-m.
# A cut keeps one variable over a region plus two native cells, as NumericalEarth's Copernicus backend pads its
# requests, under the file name NumericalEarth gives that Metadatum, so NumericalEarth finds it and downloads
# nothing. Used by tools/prefetch_glorys.jl, which cuts everything ahead of the run, and by
# examples/sowinwarp_glorys_baseline.jl, which checks it is all there. Needs Dates, NCDatasets, Logging,
# Oceananigans and NumericalEarth loaded.

const NCAR_MARGIN = 2 / 12            # two native (1/12 deg) cells
const GLORYS_ICE_RECORD_DAYS = 5      # the baseline samples GLORYS ice every 5 days (B-SOSE's cadence)

"The NCAR daily file for `date`; the latest release if there are several."
function ncar_glorys_file(source, date)
    prefix = "mercatorglorys12v1_gl12_mean_" * Dates.format(date, "yyyymmdd") * "_R"
    for folder in (joinpath(source, string(year(date))), source)
        isdir(folder) || continue
        found = sort(filter(f -> startswith(f, prefix) && endswith(f, ".nc"), readdir(folder)))
        isempty(found) || return joinpath(folder, last(found))
    end
    error("no NCAR GLORYS file for $(Date(date)) in $source (looked for $(year(date))/$(prefix)*.nc)")
end

"""
    cut_ncar_glorys(src, name, region, path)

Write variable `name` (NCAR's own name: thetao, so, uo, vo, zos, siconc, sithick, ...) of the open NCAR
file `src`, over `region` (a BoundingBox, or `nothing` for the globe) plus NCAR_MARGIN, to `path`. Values
stay packed as NCAR stores them, with every attribute. Written to `path.part<pid>` and renamed, so a file at
`path` is always complete, even with several jobs cutting at once.
"""
function cut_ncar_glorys(src, name, region, path)
    λ, φ = src["longitude"][:], src["latitude"][:]
    pick(x, bounds) = isnothing(bounds) ? (1:length(x)) :
        (k = findall(c -> bounds[1] - NCAR_MARGIN <= c <= bounds[2] + NCAR_MARGIN, x);
         isempty(k) && error("$name: $bounds is outside the NCAR file"); minimum(k):maximum(k))
    i = pick(λ, isnothing(region) ? nothing : region.longitude)
    j = pick(φ, isnothing(region) ? nothing : region.latitude)
    v = src[name]
    dims = dimnames(v)
    index = Tuple(d == "longitude" ? i : d == "latitude" ? j : Colon() for d in dims)

    part = path * ".part" * string(getpid())   # per process: two jobs cutting one file never share a .part
    isfile(part) && rm(part)
    NCDataset(part, "c") do out
        for d in dims
            n = d == "longitude" ? length(i) : d == "latitude" ? length(j) : src.dim[d]
            defDim(out, d, n)
            c = src[d]
            cv = defVar(out, d, eltype(c.var), (d,); attrib = collect(c.attrib))
            cv.var[:] = d == "longitude" ? c.var[i] : d == "latitude" ? c.var[j] : c.var[:]
        end
        ov = defVar(out, name, eltype(v.var), dims; attrib = collect(v.attrib))
        ov.var[ntuple(_ -> Colon(), length(dims))...] = v.var[index...]
    end
    mv(part, path; force = true)
    return path
end

"The region the baseline reads GLORYS sea ice over: the domain plus a degree."
glorys_ice_region(λ₁, λ₂, φ₁, φ₂) = BoundingBox(longitude = (λ₁ - 1, λ₂ + 1), latitude = (φ₁ - 1, φ₂ + 1))

"""
The days the baseline samples GLORYS ice on, for a run from `first_date` to `last_date`: the day before
`first_date`, every `every` days after it, and the day after `last_date`, so the records (placed at noon)
bracket the run. Anchored to the RUN's first date: a converter cutting part of a run must call this with
the run's dates and keep the days in its window, not call it with the window.
"""
function glorys_ice_record_starts(first_date, last_date; every = GLORYS_ICE_RECORD_DAYS)
    starts = collect(floor(first_date, Day) - Day(1):Day(every):last_date)
    last(starts) + Hour(12) < last_date && push!(starts, floor(last_date, Day) + Day(1))
    return starts
end

# ── The GLORYS files a run reads, and their inpainted caches ─────────────────────────────────────
# Shared by tools/prefetch_glorys.jl (cuts them ahead of the run) and examples/sowinwarp_glorys_baseline.jl
# (checks them at startup, then holds them all in memory).
# Needs Oceananigans and NumericalEarth loaded.

const GLORYS_OCEAN_NAMES = (:temperature, :salinity, :u_velocity, :v_velocity, :free_surface)
const GLORYS_OPEN_SIDES  = (:west, :east, :north)   # the south row is Antarctica: NumericalEarth's open_boundary_sides drops it

"""
    glorys_run_files(dir, run_start, run_end; λ₁, λ₂, φ₁, φ₂, Nx, Ny)

Every GLORYS daily Metadatum a nested run over `run_start`-`run_end` reads, deduplicated by file:
T, S, u, v, η on each open side's strip (the regions NumericalEarth's BoundaryPrescribedComponent asks
for, computed with its own functions); the same five over the domain on the first days (the starting
state, and the temperature GLORYS' seafloor is taken from when there is no deptho); and the ice days.
Returns `(; metadata, first_strip_records, deptho, strips, width, w, padding)`: `first_strip_records`
is each strip variable's first record (to size the parent held in memory), `deptho` the Metadatum of
GLORYS' static seafloor over the nest's snapshot region (fetched from Copernicus only, so not in
`metadata`). The regions depend only on the horizontal extent and resolution, so a one-level stand-in
grid is used.
"""
function glorys_run_files(dir, run_start, run_end; λ₁, λ₂, φ₁, φ₂, Nx, Ny)
    DW = NumericalEarth.DataWrangling
    dataset = DW.GLORYS.GLORYSDaily()
    grid = LatitudeLongitudeGrid(CPU(); size = (Nx, Ny, 1), longitude = (λ₁, λ₂), latitude = (φ₁, φ₂), z = (-1, 0))

    padding = DW.default_horizontal_padding(dataset)
    width   = NumericalEarth.NestedModels.cells_spanning(grid, NumericalEarth.NestedModels.default_relaxation_degrees)
    w       = NumericalEarth.EarthSystemModels.relaxation_zone_width(grid, width)
    δλ, δφ  = DW.axis_padding(padding)
    λ, φ    = (λ₁ - δλ, λ₂ + δλ), (φ₁ - δφ, φ₂ + δφ)
    strips  = (west  = DW.BoundingBox(longitude = (λ₁ - δλ, λ₁ + w + δλ), latitude = φ),
               east  = DW.BoundingBox(longitude = (λ₂ - w - δλ, λ₂ + δλ), latitude = φ),
               north = DW.BoundingBox(longitude = λ, latitude = (φ₂ - w - δφ, φ₂ + δφ)))
    domain  = DW.BoundingBox(grid; padding)

    files = Dict{String, Any}()
    add!(m) = for d in m
        files[joinpath(d.dir, d.filename)] = d
    end
    strip_dates = DW.bracketing_dates(dataset, :temperature, (run_start, run_end))
    first_strip_records = []
    for side in GLORYS_OPEN_SIDES, name in GLORYS_OCEAN_NAMES
        series = DW.Metadata(name; dataset, dates = strip_dates, region = strips[side], dir)
        add!(series)
        push!(first_strip_records, first(series))
    end
    for name in GLORYS_OCEAN_NAMES
        add!(DW.Metadata(name; dataset, dates = DW.bracketing_dates(dataset, :temperature, [run_start]), region = domain, dir))
    end
    add!(DW.Metadatum(:temperature; dataset, date = run_start, region = domain, dir))
    ice_region = glorys_ice_region(λ₁, λ₂, φ₁, φ₂)
    for date in glorys_ice_record_starts(run_start, run_end), name in (:sea_ice_concentration, :sea_ice_thickness)
        add!(DW.Metadatum(name; dataset, date, region = ice_region, dir))
    end
    metadata = [files[k] for k in sort(collect(keys(files)))]
    deptho = DW.Metadatum(:depth; dataset = DW.GLORYS.GLORYSStatic(), region = domain, dir)
    return (; metadata, first_strip_records, deptho, strips, width, w, padding)
end

"Does `m` get inpainted when read (ocean: yes; ice: never), and is its cache still missing?"
needs_inpainting(m) = !isnothing(NumericalEarth.DataWrangling.default_inpainting(m)) &&
                      !isfile(NumericalEarth.DataWrangling.inpainted_metadata_path(m))
