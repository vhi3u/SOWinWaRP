# glorys_ncar.jl -- NSF NCAR's copy of GLORYS12V1 (GDEX d010049) cut into the files NumericalEarth's
# GLORYSDaily reads. NCAR holds one global file per day with every variable,
#
#     <source>/<YYYY>/mercatorglorys12v1_gl12_mean_<YYYYMMDD>_R<YYYYMMDD>.nc    (or flat in <source>)
#
# the same product, variable names, dimensions and packing as Copernicus' cmems_mod_glo_phy_my_0.083deg_P1D-m.
# A cut keeps one variable over a region plus two native cells, as NumericalEarth's Copernicus backend pads its
# requests, under the file name NumericalEarth gives that Metadatum, so NumericalEarth finds it and downloads
# nothing. Used by examples/sowinwarp_glorys_baseline.jl (cuts on demand) and tools/convert_glorys_ncar.jl
# (cuts everything ahead of the run). Needs Dates, NCDatasets and NumericalEarth's BoundingBox in scope.

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
stay packed as NCAR stores them, with every attribute. Written to `path.part` and renamed, so a file at
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

    part = path * ".part"
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
