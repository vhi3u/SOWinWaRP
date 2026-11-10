# Convert the hourly ERA5 single-level fields that NumericalEarth's `ERA5PrescribedAtmosphere` (10 m winds, 2 m
# temperature and dewpoint, surface and mean sea level pressure, total precipitation, snowfall) and `ERA5PrescribedRadiation`
# (downwelling shortwave and longwave) read, from a Globus copy of NSF NCAR's ERA5 (RDA ds633.0, GDEX /d633000/), into
# the yearly files NumericalEarth's `ERA5YearlySingleLevel` reads, cropped to the SOWinWaRP box. Nothing is downloaded.
#
#   ./create_job.sh era5_ncar cpu 12 none 32G tools/download_era5_ncar.jl
#   sbatch slurm_era5_ncar.sh
#
# Each output is <ERA5_DIR>/<CDS variable>_<YYYY>_<W>_<E>_<S>_<N>.nc, named by NumericalEarth's own `metadata_filename`
# for the region examples/sowinwarp_nested.jl passes, so the nested run finds them without downloading. The files hold
# the region padded by two 0.25° cells, as a CDS area request does. Only whole years are written, and an output already
# on disk is skipped, so rerun after a failure.
#
# SOURCE_DIR is the Globus copy, as <SOURCE_DIR>/<YYYYMM>/<file> (all products in one folder) or NCAR's own
# <SOURCE_DIR>/<product>/<YYYYMM>/<file>. It is only read, never deleted. Each year needs all twelve months of both
# products, plus the previous December's last forecast files (e.g. 201312 for 2014; see below). Only run it once the
# copy has finished: a year with a file missing is skipped with a warning naming the variables.
#
# NCAR stores the analysed fields (winds, temperatures, pressure) as hourly monthly files, and the accumulations
# (precipitation, radiation) in forecast form: 12 hourly steps after each 06 and 18 UTC forecast start, in half-month
# files. Each step is the accumulation over the hour ending at start + step, as ERA5 on the CDS, so it is written at that
# time. The first hours of 1 January come from the previous December's last file. ERA5's total precipitation is NCAR's
# large-scale plus convective precipitation; snowfall (in it) is NCAR's 128_144_sf. A year is only kept if every hour was filled.
#
# Units as NCAR stores them: winds m s⁻¹ (eastward/northward positive), temperatures K, pressure Pa, precipitation and snowfall m (of
# water equivalent) and radiation J m⁻² (positive downwards) accumulated over the hour.
#
#     SOURCE_DIR=<BSOSE_DIR>/ncar-era5  ERA5_DIR=<BSOSE_DIR>/era5  YEARS=2014:2016   (or the years as arguments)

using Dates
using NCDatasets: NCDatasets
using NumericalEarth
using NumericalEarth.DataWrangling: metadata_filename

data_dir   = get(ENV, "BSOSE_DIR", "/storage/scratch1/1/vnguyen480/SOWinWaRP/data")
source_dir = get(ENV, "SOURCE_DIR", joinpath(data_dir, "ncar-era5"))
dir        = get(ENV, "ERA5_DIR", joinpath(data_dir, "era5"))
year_range = parse.(Int, split(get(ENV, "YEARS", "2014:2016"), ':'))
years      = isempty(ARGS) ? (first(year_range):last(year_range)) : parse.(Int, ARGS)

# MUST MATCH examples/sowinwarp_nested.jl: the region is part of every file name
const λ₁, λ₂ = 90.0, 150.0
const φ₁, φ₂ = -70.0, -45.0
region  = BoundingBox(longitude = (λ₁ - 1.0, λ₂ + 1.0), latitude = (φ₁ - 1.0, φ₂ + 1.0))
padding = 2 * 0.25

analysis = "e5.oper.an.sfc"
forecast = "e5.oper.fc.sfc.accumu"

# Each output's NumericalEarth name, its NetCDF name in NumericalEarth's tables, the NCAR product, and the NCAR
# parameters summed into it
variables = [(name = :eastward_velocity,               short = "u10",  product = analysis, parameters = ["128_165_10u"]),
             (name = :northward_velocity,              short = "v10",  product = analysis, parameters = ["128_166_10v"]),
             (name = :temperature,                     short = "t2m",  product = analysis, parameters = ["128_167_2t"]),
             (name = :dewpoint_temperature,            short = "d2m",  product = analysis, parameters = ["128_168_2d"]),
             (name = :surface_pressure,                short = "sp",   product = analysis, parameters = ["128_134_sp"]),
             (name = :mean_sea_level_pressure,         short = "msl",  product = analysis, parameters = ["128_151_msl"]),
             (name = :total_precipitation,             short = "tp",   product = forecast, parameters = ["128_142_lsp", "128_143_cp"]),
             (name = :snowfall,                        short = "sf",   product = forecast, parameters = ["128_144_sf"]),
             (name = :downwelling_shortwave_radiation, short = "ssrd", product = forecast, parameters = ["128_169_ssrd"]),
             (name = :downwelling_longwave_radiation,  short = "strd", product = forecast, parameters = ["128_175_strd"])]

output_path(variable, y) = joinpath(dir, metadata_filename(ERA5YearlySingleLevel(), variable.name, DateTime(y), region))

# The first ERA5 forecast starts at 06 UTC on 1 January 1940, so its first accumulation ends an hour later
first_accumulation = DateTime(1940, 1, 1, 7)

# Paths (relative to SOURCE_DIR) of one product's files in one month, from either layout
listings = Dict{Tuple{String, Date}, Vector{String}}()

function month_files(product, month)
    get!(listings, (product, month)) do
        yyyymm = Dates.format(month, "yyyymm")
        files = String[]
        for folder in (joinpath(product, yyyymm), yyyymm)
            isdir(joinpath(source_dir, folder)) || continue
            append!(files, [joinpath(folder, file) for file in readdir(joinpath(source_dir, folder))
                            if startswith(file, product * ".") && endswith(file, ".nc")])
        end
        sort!(files; by = basename)
    end
end

# The files holding one parameter's hours of year `y` in time order, or `nothing` if SOURCE_DIR does not have them all
function parameter_files(product, parameter, y)
    files = [file for m in 1:12 for file in month_files(product, Date(y, m)) if occursin(".$parameter.", file)]

    if product == forecast
        december = [file for file in month_files(product, Date(y - 1, 12))
                    if occursin(".$parameter.", file) && endswith(file, "_$(y)010106.nc")]
        files = vcat(december, files)
    end

    expected = product == analysis ? 12 : y == year(first_accumulation) ? 24 : 25
    return length(files) == expected ? files : nothing
end

# The one variable in an NCAR file laid out on the grid (its name, e.g. VAR_2T or SSRD, differs per parameter)
field_name(ds, dims) = only(name for name in keys(ds) if NCDatasets.dimnames(ds[name]) == dims)

hour_index(time, times) = Dates.value(time - first(times)) ÷ 3_600_000 + 1

# The contiguous index range of `coordinates` inside `(lo, hi)`, padded
function crop(coordinates, (lo, hi))
    inside = findall(c -> lo - padding - 1e-6 <= c <= hi + padding + 1e-6, coordinates)
    isempty(inside) && error("no NCAR grid points in $lo..$hi")
    all(diff(inside) .== 1) || error("the crop of $lo..$hi is not contiguous (does it cross the date line?)")
    return first(inside):last(inside)
end

function write_year(variable, y, sources)
    path    = output_path(variable, y)
    start   = variable.product == forecast ? max(DateTime(y), first_accumulation) : DateTime(y)
    times   = start:Hour(1):DateTime(y, 12, 31, 23)
    written = falses(length(times))
    files   = [NCDatasets.Dataset(joinpath(source_dir, source[f])) for source in sources, f in eachindex(first(sources))]

    try
        grid = files[1, 1]
        i = crop(grid["longitude"][:], region.longitude)
        j = crop(grid["latitude"][:], region.latitude)   # NCAR's latitude runs 90 → -90, like the CDS's; kept as is

        NCDatasets.Dataset(path * ".part", "c") do dst
            NCDatasets.defDim(dst, "longitude", length(i))
            NCDatasets.defDim(dst, "latitude", length(j))
            NCDatasets.defDim(dst, "valid_time", length(times))
            NCDatasets.defVar(dst, "longitude", grid["longitude"][i], ("longitude",); attrib = ["units" => "degrees_east"])
            NCDatasets.defVar(dst, "latitude", grid["latitude"][j], ("latitude",); attrib = ["units" => "degrees_north"])
            NCDatasets.defVar(dst, "valid_time", [Dates.value(t - DateTime(1900)) ÷ 3_600_000 for t in times], ("valid_time",);
                              attrib = ["units" => "hours since 1900-01-01 00:00:00", "calendar" => "proleptic_gregorian"])

            if variable.product == analysis
                dims = ("longitude", "latitude", "time")
                source_name = field_name(grid, dims)
                field = NCDatasets.defVar(dst, variable.short, Float32, ("longitude", "latitude", "valid_time");
                                          chunksizes = [length(i), length(j), 1], shuffle = true, deflatelevel = 1,
                                          attrib = ["units" => grid[source_name].attrib["units"],
                                                    "long_name" => grid[source_name].attrib["long_name"]])

                for src in files
                    file_times = src["time"][:]
                    all(diff(file_times) .== Hour(1)) || error("an NCAR $(variable.name) file for $y is not hourly")

                    for block in Iterators.partition(eachindex(file_times), 24)
                        n = hour_index(file_times[first(block)], times) .+ (0:length(block) - 1)
                        field.var[:, :, n] = src[source_name].var[i, j, block]
                        written[n] .= true
                    end
                end
            else
                dims = ("longitude", "latitude", "forecast_hour", "forecast_initial_time")
                source_names = [field_name(src, dims) for src in files[:, 1]]
                field = NCDatasets.defVar(dst, variable.short, Float32, ("longitude", "latitude", "valid_time");
                                          chunksizes = [length(i), length(j), 1], shuffle = true, deflatelevel = 1,
                                          attrib = ["units" => grid[source_names[1]].attrib["units"],
                                                    "long_name" => String(variable.name)])

                for f in axes(files, 2)
                    srcs = files[:, f]
                    starts = srcs[1]["forecast_initial_time"][:]
                    steps  = srcs[1]["forecast_hour"][:]

                    for (k, start) in enumerate(starts)
                        valid = [start + Hour(step) for step in steps]
                        inside = findall(t -> first(times) <= t <= last(times), valid)
                        isempty(inside) && continue

                        steps_inside = first(inside):last(inside)
                        n = hour_index(valid[first(inside)], times) .+ (0:length(inside) - 1)
                        field.var[:, :, n] = sum(src[name].var[i, j, steps_inside, k] for (src, name) in zip(srcs, source_names))
                        written[n] .= true
                    end
                end
            end
        end
    finally
        foreach(close, files)
    end

    all(written) || error("$path is missing $(count(!, written)) hours, from $(times[findfirst(!, written)])")
    mv(path * ".part", path; force = true)
    @info "Wrote $path"
    return nothing
end

function process_year(y)
    todo = [variable for variable in variables if !isfile(output_path(variable, y))]
    isempty(todo) && return nothing

    sources = Dict(variable.name => [parameter_files(variable.product, parameter, y) for parameter in variable.parameters]
                   for variable in todo)

    incomplete = [string(variable.name) for variable in todo if any(isnothing, sources[variable.name])]
    if !isempty(incomplete)
        @warn "$source_dir does not have all of $y for $(join(incomplete, ", ")); skipping $y"
        return nothing
    end

    for variable in todo
        write_year(variable, y, sources[variable.name])
    end

    return nothing
end

# Run only as a script, so the functions can also be included and tested
if abspath(PROGRAM_FILE) == @__FILE__
    isdir(source_dir) || error("SOURCE_DIR $source_dir does not exist")
    mkpath(dir)
    @info "ERA5 from $source_dir into $dir for $(collect(years)), region $(region.longitude) x $(region.latitude)"
    foreach(process_year, years)
end
