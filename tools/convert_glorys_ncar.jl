# Cut, from a Globus copy of NSF NCAR's GLORYS12V1 (GDEX d010049), every GLORYS daily file that
# examples/sowinwarp_glorys_baseline.jl reads, under the names NumericalEarth's `GLORYSDaily` gives them, so the
# baseline finds them on disk and downloads nothing. The GLORYS counterpart of tools/download_era5_ncar.jl.
#
#   ./create_job.sh glorys_ncar cpu 12 none 32G tools/convert_glorys_ncar.jl
#   sbatch slurm_glorys_ncar.sh
#
# Split across jobs that run at once with START/END (a window of the run, START included and END not, so
# adjacent windows never cut the same day):
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
# Then it inpaints each ocean file as the run would and writes NumericalEarth's cache of it beside the file
# (<file>_inpainted.jld2), so the run loads the inpainted fields and inpaints nothing (INPAINT=0 to skip).
# A file already in GLORYS_DIR is skipped, so rerun after a failure; a file is written whole or not at all.
# A rerun over days already cut only inpaints.
# The baseline cuts anything missed itself at startup (e.g. a south strip, if NumericalEarth ever opens the
# south side), from the same NCAR files.
#
# SOURCE is the Globus copy, as <SOURCE>/<YYYY>/mercatorglorys12v1_gl12_mean_<YYYYMMDD>_R<YYYYMMDD>.nc or flat
# in <SOURCE>. It is only read, never deleted.
#
#     GLORYS_SOURCE=<BSOSE_DIR>/ncar-glorys  GLORYS_DIR=<BSOSE_DIR>/glorys  RUN_START=2014-06-01  RUN_END=2016-06-01T12:00
#     DRY_RUN=1   lists what it would cut on the first two days, and the strip regions, and stops (seconds; login node)

using Dates, Printf, Logging
using NCDatasets
using Oceananigans
using NumericalEarth
using NumericalEarth.DataWrangling: Metadatum, BoundingBox, dataset_variable_name, default_inpainting, inpainted_metadata_path
using NumericalEarth.DataWrangling.GLORYS: GLORYSDaily

include(joinpath(@__DIR__, "..", "src", "glorys_ncar.jl"))

# NumericalEarth's Field(metadatum) calls download first, even for a file on disk, and with no backend loaded
# its fallback errors. Here every file is cut before it is read, so "downloading" is finding it.
function NumericalEarth.DataWrangling.Downloads.download(meta::Metadatum{<:GLORYSDaily}; kwargs...)
    path = joinpath(meta.dir, meta.filename)
    isfile(path) || error("$path is not cut: the converter reads only files it has cut")
    return path
end

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

# INPAINT (default on): also write the inpainted cache NumericalEarth keeps beside each file it inpaints
# (src/glorys_ncar.jl, inpaint_glorys!), so the run inpaints nothing. Ice is never inpainted.
const INPAINT = get(ENV, "INPAINT", "1") == "1"
to_inpaint(m) = INPAINT && needs_inpainting(m)

# Every file to cut, grouped by day: path => Metadatum, from the list the baseline checks at startup
run_files = glorys_run_files(dir, run_start, run_end; λ₁, λ₂, φ₁, φ₂, Nx, Ny)
strips, width, w, padding = run_files.strips, run_files.width, run_files.w, run_files.padding
open_sides = GLORYS_OPEN_SIDES
cuts = Dict{Date, Dict{String, Any}}()
for m in run_files.metadata
    day = Date(m.dates)
    window[1] <= day < window[2] || continue   # END is exclusive, so adjacent windows share no day
    get!(cuts, day, Dict{String, Any}())[joinpath(m.dir, m.filename)] = m
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
        for (out, m) in sort(collect(cuts[day]); by = first)
            println("  ", rpad(dataset_variable_name(m), 8), basename(out), isfile(out) ? "  (there)" : "",
                    isnothing(default_inpainting(m)) ? "" : isfile(inpainted_metadata_path(m)) ? "  (inpainted)" : "  (to inpaint)")
        end
    end
    exit(0)
end

done, skipped, inpainted, missing_days = 0, 0, 0, Date[]
t₀ = time()
for (n, day) in enumerate(days)
    todo = [(out, m) for (out, m) in cuts[day] if !isfile(out)]
    global skipped += length(cuts[day]) - length(todo)
    if !isempty(todo)
        path = try
            ncar_glorys_file(source, DateTime(day))
        catch
            push!(missing_days, day)
            continue
        end
        NCDataset(path) do src
            for (out, m) in todo
                cut_ncar_glorys(src, dataset_variable_name(m), m.region, out)
                global done += 1
            end
        end
    end
    for m in values(cuts[day])
        to_inpaint(m) || continue
        inpaint_glorys!(m)
        global inpainted += 1
    end
    (n % 30 == 0 || n == length(days)) &&
        @info @sprintf("%s: %d of %d days, %d files cut, %d already there, %d inpainted, %.2f h", day, n, length(days), done, skipped, inpainted, (time() - t₀) / 3600)
end

isempty(missing_days) ||
    @warn "$(length(missing_days)) days have no NCAR file in $source, so nothing was cut for them: " * join(string.(missing_days), ", ")
@info @sprintf("GLORYS from NCAR: %d files cut, %d already there, %d inpainted, %d days missing, %.2f h", done, skipped, inpainted, length(missing_days), (time() - t₀) / 3600)
