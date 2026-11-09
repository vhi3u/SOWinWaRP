# Cut, from a Globus copy of NSF NCAR's GLORYS12V1 (GDEX d010049), every GLORYS daily file that
# examples/sowinwarp_glorys_baseline.jl reads, under the names NumericalEarth's `GLORYSDaily` gives them, so the
# baseline finds them on disk and downloads nothing. The GLORYS counterpart of tools/download_era5_ncar.jl.
#
#   ./create_job.sh glorys_ncar cpu 12 none 32G tools/convert_glorys_ncar.jl
#   sbatch slurm_glorys_ncar.sh
#
# Split across jobs that run at once with START/END (a window of the run; the files are per day, so windows
# never write the same file):
#
#   START=2014-05-31 END=2015-01-01 sbatch slurm_glorys_ncar_1.sh
#   START=2015-01-01 END=2015-09-01 sbatch slurm_glorys_ncar_2.sh
#   START=2015-09-01 END=2016-06-03 sbatch slurm_glorys_ncar_3.sh
#
# For each day it opens that day's NCAR file ONCE and cuts from it:
#   - the boundary strips: T, S, u, v and η (thetao, so, uo, vo, zos) on the west, east and north strips, the
#     regions NumericalEarth's BoundaryPrescribedComponent asks for (its own functions compute them here);
#   - on the run's first days, the same five over the whole domain: the nest's starting state, and the
#     temperature the baseline takes GLORYS' seafloor from without Copernicus credentials;
#   - on the baseline's ice days (every 5 days, src/glorys_ncar.jl), siconc and sithick over the domain + 1 deg.
# A file already in GLORYS_DIR is skipped, so rerun after a failure; a file is written whole or not at all.
# The baseline cuts anything missed itself at startup (e.g. a south strip, if NumericalEarth ever opens the
# south side), from the same NCAR files.
#
# SOURCE is the Globus copy, as <SOURCE>/<YYYY>/mercatorglorys12v1_gl12_mean_<YYYYMMDD>_R<YYYYMMDD>.nc or flat
# in <SOURCE>. It is only read, never deleted.
#
#     GLORYS_SOURCE=<BSOSE_DIR>/ncar-glorys  GLORYS_DIR=<BSOSE_DIR>/glorys  RUN_START=2014-06-01  RUN_END=2016-06-01T12:00
#     DRY_RUN=1   lists what it would cut on the first two days, and the strip regions, and stops (seconds; login node)

using Dates, Printf
using NCDatasets
using Oceananigans
using NumericalEarth
using NumericalEarth.DataWrangling: Metadata, Metadatum, BoundingBox, dataset_variable_name, bracketing_dates,
                                    default_horizontal_padding, axis_padding
using NumericalEarth.DataWrangling.GLORYS: GLORYSDaily

include(joinpath(@__DIR__, "..", "src", "glorys_ncar.jl"))

data_dir  = get(ENV, "BSOSE_DIR", "/storage/scratch1/1/vnguyen480/SOWinWaRP/data")
source    = get(ENV, "GLORYS_SOURCE", joinpath(data_dir, "ncar-glorys"))
dir       = get(ENV, "GLORYS_DIR", joinpath(data_dir, "glorys"))
run_start = DateTime(get(ENV, "RUN_START", "2014-06-01"))
run_end   = DateTime(get(ENV, "RUN_END", "2016-06-01T12:00"))
window    = (Date(DateTime(get(ENV, "START", string(run_start - Day(2))))), Date(DateTime(get(ENV, "END", string(run_end + Day(2))))))
isdir(source) || error("$source not found: set GLORYS_SOURCE to the Globus copy of NCAR's d010049")
mkpath(dir)

# MUST MATCH examples/sowinwarp_glorys_baseline.jl: the domain and resolution set every region, and the region
# is part of every file name
const λ₁, λ₂ = 90.0, 150.0
const φ₁, φ₂ = -70.0, -45.0
const Nx, Ny = 360, 150
open_sides = (:west, :east, :north)   # the south row is Antarctica: NumericalEarth's open_boundary_sides drops it

# A horizontal stand-in for the baseline's grid: the regions only depend on its extent and spacing
grid = LatitudeLongitudeGrid(CPU(); size = (Nx, Ny, 1), longitude = (λ₁, λ₂), latitude = (φ₁, φ₂), z = (-1, 0))
dataset = GLORYSDaily()

# The strips, as NumericalEarth's BoundaryPrescribedComponent (DataWrangling/boundary_prescribed_component.jl)
# builds them for nested_ocean_model's defaults: `width` cells of relaxation, padded by the dataset's padding
padding = default_horizontal_padding(dataset)
width   = NumericalEarth.NestedModels.cells_spanning(grid, NumericalEarth.NestedModels.default_relaxation_degrees)
w       = NumericalEarth.EarthSystemModels.relaxation_zone_width(grid, width)
δλ, δφ  = axis_padding(padding)
λ, φ    = (λ₁ - δλ, λ₂ + δλ), (φ₁ - δφ, φ₂ + δφ)
strips  = (west  = BoundingBox(longitude = (λ₁ - δλ, λ₁ + w + δλ), latitude = φ),
           east  = BoundingBox(longitude = (λ₂ - w - δλ, λ₂ + δλ), latitude = φ),
           north = BoundingBox(longitude = λ, latitude = (φ₂ - w - δφ, φ₂ + δφ)))
domain  = BoundingBox(grid; padding)
ocean_names = (:temperature, :salinity, :u_velocity, :v_velocity, :free_surface)

# Every file to cut, grouped by day: (NCAR variable, region, path). Built from NumericalEarth's own Metadata,
# so the dates and names are the ones the run asks for.
cuts = Dict{Date, Vector{Tuple{String, Any, String}}}()
add!(m) = for d in m
    day = Date(d.dates)
    window[1] <= day <= window[2] || continue
    entry = (dataset_variable_name(d), d.region, joinpath(d.dir, d.filename))
    list = get!(cuts, day, Tuple{String, Any, String}[])
    entry in list || push!(list, entry)
end

strip_dates = bracketing_dates(dataset, :temperature, (run_start, run_end))
for side in open_sides, name in ocean_names
    add!(Metadata(name; dataset, dates = strip_dates, region = strips[side], dir))
end
for name in ocean_names
    add!(Metadata(name; dataset, dates = bracketing_dates(dataset, :temperature, [run_start]), region = domain, dir))
end
add!(Metadatum(:temperature; dataset, date = run_start, region = domain, dir))
ice_region = glorys_ice_region(λ₁, λ₂, φ₁, φ₂)
for date in glorys_ice_record_starts(run_start, run_end), name in (:sea_ice_concentration, :sea_ice_thickness)
    add!(Metadatum(name; dataset, date, region = ice_region, dir))
end

days = sort(collect(keys(cuts)))
total = sum(length, values(cuts); init = 0)
@info "GLORYS from NCAR: $(length(days)) days, $total files, $(window[1]) to $(window[2]) of the run " *
      "$run_start to $run_end, from $source into $dir"

if get(ENV, "DRY_RUN", "0") == "1"
    @info "strips: $(width) cells = $(round(w, digits = 4)) deg, padded by $(round(padding, digits = 4)) deg on the $(join(open_sides, ", ")) sides"
    foreach(side -> println("  $side: ", strips[side]), open_sides)
    for day in days[1:min(2, end)]
        println("$day ($(length(cuts[day])) files):")
        foreach(c -> println("  ", rpad(c[1], 8), basename(c[3]), isfile(c[3]) ? "  (there)" : ""), cuts[day])
    end
    exit(0)
end

done, skipped, missing_days = 0, 0, Date[]
t₀ = time()
for (n, day) in enumerate(days)
    todo = filter(c -> !isfile(c[3]), cuts[day])
    global skipped += length(cuts[day]) - length(todo)
    isempty(todo) && continue
    path = try
        ncar_glorys_file(source, DateTime(day))
    catch
        push!(missing_days, day)
        continue
    end
    NCDataset(path) do src
        for (name, region, out) in todo
            cut_ncar_glorys(src, name, region, out)
            global done += 1
        end
    end
    (n % 30 == 0 || n == length(days)) &&
        @info @sprintf("%s: %d of %d days, %d files cut, %d already there, %.2f h", day, n, length(days), done, skipped, (time() - t₀) / 3600)
end

isempty(missing_days) ||
    @warn "$(length(missing_days)) days have no NCAR file in $source, so nothing was cut for them: " * join(string.(missing_days), ", ")
@info @sprintf("GLORYS from NCAR: %d files cut, %d already there, %d days missing, %.2f h", done, skipped, length(missing_days), (time() - t₀) / 3600)
