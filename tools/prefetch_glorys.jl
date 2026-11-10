# Make everything examples/sowinwarp_glorys_baseline.jl reads from GLORYS ready, so the run starts by loading it
# and does nothing else with it. The GLORYS counterpart of tools/download_era5_ncar.jl. One CPU job:
#
#   ./create_job.sh prefetch_glorys cpu 12 none 32G tools/prefetch_glorys.jl
#   sbatch slurm_prefetch_glorys.sh
#
# It ends with "GLORYS READY" or with an error saying what is missing. Rerun it after any failure: whatever is
# already done is skipped. In order:
#
#   1. check (only with CHECK=1): every GLORYS file the run reads that is on disk but does not open is deleted,
#      so step 2 makes it again. Off by default: every file is written whole (below), and opening ~11,000
#      files on scratch takes a while;
#   2. cut: each missing file the run reads (src/glorys_ncar.jl, glorys_run_files: T, S, u, v and η on the
#      west, east and north strips for every day, the same five over the domain on the first days, and siconc
#      and sithick every 5 days), from that day's file in the Globus copy of NSF NCAR's GLORYS12V1 (GDEX
#      d010049), under the name NumericalEarth gives it, so NumericalEarth finds it and downloads nothing;
#   3. seafloor: GLORYS' static deptho from Copernicus, if COPERNICUSMARINE_SERVICE_USERNAME and _PASSWORD are set
#      (create_job.sh sources .copernicus_env); without them the run takes GLORYS' seafloor from where its
#      temperature is missing on the first day, which needs nothing more;
#   4. ERA5: lists any ERA5 file the run reads that is missing (tools/download_era5_ncar.jl makes them).
#
# Nothing is inpainted here: the baseline inpaints each record on the GPU as it loads them all at startup
# (~0.25 s a day, minutes for the run; far slower on a CPU), and NumericalEarth keeps each as its cache.
#
# Every file is written to a .part file and renamed, so a file on disk is always whole.
#
#     GLORYS_SOURCE=<BSOSE_DIR>/ncar-glorys (the NCAR copy, <YYYY>/mercatorglorys12v1_gl12_mean_<YYYYMMDD>_R*.nc or flat;
#     only read)   GLORYS_DIR=<BSOSE_DIR>/glorys   ERA5_DIR=<BSOSE_DIR>/era5

using Dates, Printf, Logging
using NCDatasets
using Oceananigans
using CopernicusMarine     # only for deptho (step 3)
using NumericalEarth
using NumericalEarth.DataWrangling: Metadata, Metadatum, BoundingBox, dataset_variable_name, metadata_path
using NumericalEarth.DataWrangling.GLORYS: GLORYSDaily

include(joinpath(@__DIR__, "..", "src", "glorys_ncar.jl"))

data_dir = get(ENV, "BSOSE_DIR", "/storage/scratch1/1/vnguyen480/SOWinWaRP/data")
source   = get(ENV, "GLORYS_SOURCE", joinpath(data_dir, "ncar-glorys"))
dir      = get(ENV, "GLORYS_DIR", joinpath(data_dir, "glorys"))
era5_dir = get(ENV, "ERA5_DIR", joinpath(data_dir, "era5"))
mkpath(dir)

# MUST MATCH examples/sowinwarp_glorys_baseline.jl: the dates set which days are read, and the domain and
# resolution set every region, which is part of every file name
const RUN_START = DateTime(2014, 6, 1)
const RUN_END   = DateTime(2016, 6, 1, 12)   # the baseline's END_DATE
const λ₁, λ₂ = 90.0, 150.0
const φ₁, φ₂ = -70.0, -45.0
const Nx, Ny = 360, 150

# A daily GLORYS file is "downloaded" by finding it: step 2 has cut it by the time anything reads it
function NumericalEarth.DataWrangling.Downloads.download(meta::Metadatum{<:GLORYSDaily}; kwargs...)
    path = joinpath(meta.dir, meta.filename)
    isfile(path) || error("$path is not cut")
    return path
end

t₀ = time()
elapsed() = @sprintf("%.2f h", (time() - t₀) / 3600)
run_files = glorys_run_files(dir, RUN_START, RUN_END; λ₁, λ₂, φ₁, φ₂, Nx, Ny)
files = run_files.metadata
@info "GLORYS prefetch for the baseline, $RUN_START to $RUN_END: $(length(files)) files in $dir"

# ── 1. check what is on disk ─────────────────────────────────────────────────────────────────
opens(path) = try
    NCDataset(close, path); true
catch
    false
end
if get(ENV, "CHECK", "0") == "1"
    broken = filter(m -> isfile(metadata_path(m)) && !opens(metadata_path(m)), files)
    foreach(m -> rm(metadata_path(m)), broken)
    @info "1. check: $(length(broken)) broken files deleted, to be made again ($(elapsed()))"
end

# ── 2. cut the missing files from NCAR's daily files ─────────────────────────────────────────
to_cut = Dict{Date, Vector{Any}}()
for m in files
    isfile(metadata_path(m)) || push!(get!(to_cut, Date(m.dates), Any[]), m)
end
if !isempty(to_cut)
    isdir(source) || error("$(sum(length, values(to_cut))) GLORYS files are not cut, and $source (GLORYS_SOURCE, " *
                           "the Globus copy of NCAR's d010049) is not there to cut them from")
    no_ncar = Date[]
    for (n, day) in enumerate(sort(collect(keys(to_cut))))
        path = try
            ncar_glorys_file(source, DateTime(day))
        catch
            push!(no_ncar, day)
            continue
        end
        NCDataset(path) do src
            for m in to_cut[day]
                cut_ncar_glorys(src, dataset_variable_name(m), m.region, metadata_path(m))
            end
        end
        n % 30 == 0 && @info "2. cut: $n of $(length(to_cut)) days ($(elapsed()))"
    end
    isempty(no_ncar) || error("no NCAR file in $source for $(length(no_ncar)) days the run needs: " * join(string.(no_ncar), ", "))
end
@info "2. cut: $(sum(length, values(to_cut); init = 0)) files cut ($(elapsed()))"

# ── 3. GLORYS' seafloor ───────────────────────────────────────────────────────────────────────
deptho = run_files.deptho
copernicus = haskey(ENV, "COPERNICUSMARINE_SERVICE_USERNAME") && haskey(ENV, "COPERNICUSMARINE_SERVICE_PASSWORD")
if !isfile(metadata_path(deptho)) && copernicus
    NumericalEarth.DataWrangling.Downloads.download(deptho)
end
@info "3. seafloor: " * (isfile(metadata_path(deptho)) ? "deptho, $(basename(metadata_path(deptho)))" :
                         "no deptho (no Copernicus credentials): the run takes it from GLORYS' first-day temperature")

# ── 4. ERA5 (the same files and region as the baseline reads) ────────────────────────────────
era5_region = BoundingBox(longitude = (λ₁ - 1.0, λ₂ + 1.0), latitude = (φ₁ - 1.0, φ₂ + 1.0))
era5_names  = (:eastward_velocity, :northward_velocity, :temperature, :dewpoint_temperature, :mean_sea_level_pressure,
               :total_precipitation, :downwelling_shortwave_radiation, :downwelling_longwave_radiation)
missing_era5 = unique([path for name in era5_names
                       for path in metadata_path(Metadata(name; dataset = ERA5YearlySingleLevel(),
                                                          start_date = RUN_START - Hour(1), end_date = RUN_END + Hour(1),
                                                          dir = era5_dir, region = era5_region))
                       if !isfile(path)])

# ── ready? ─────────────────────────────────────────────────────────────────────────────────────
not_cut = count(m -> !isfile(metadata_path(m)), files)
not_cut == 0 && isempty(missing_era5) ||
    error("NOT READY: $not_cut GLORYS files not cut, $(length(missing_era5)) ERA5 files missing" *
          (isempty(missing_era5) ? "" : " (run tools/download_era5_ncar.jl):\n  " * join(missing_era5, "\n  ")))
@info "GLORYS READY: all $(length(files)) files cut, ERA5 complete ($(elapsed())). Submit the baseline."
