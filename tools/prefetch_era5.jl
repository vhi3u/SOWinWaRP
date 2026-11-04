# ==============================================================================================
# prefetch_era5.jl -- download the ERA5 cache the nested run reads, as its own CPU job.
#
#     ./create_job.sh era5_prefetch cpu 48 none 8G tools/prefetch_era5.jl
#     sbatch slurm_era5_prefetch.sh
#
# WHY A SEPARATE JOB. CDS can leave a request queued for over an hour, and the client gives up
# at 3600 s. Inside the GPU job that wait burns GPU hours and then kills the run. Here it only
# costs a CPU slot, and a timed-out variable is simply retried: `prefetch_era5!` skips files
# already on disk, so each retry asks only for what is still missing.
#
# MUST MATCH examples/sowinwarp_nested.jl -- same region and directory, or the GPU job will not
# find these files and will download again. Defaults cover the full Jun-Aug window, which also
# covers the smoke run.
#
#     START_DATE=2014-06-01T12:00  END_DATE=2014-08-01  ERA5_DIR=...  MAX_ATTEMPTS=20
# ==============================================================================================

using CopernicusClimateDataStore    # the ERA5 download backend
using NumericalEarth
using Dates

include(joinpath(@__DIR__, "..", "src", "setup_bsose.jl"))   # prefetch_era5!, default_bsose_directory

const λ₁, λ₂ = 90.0, 150.0
const φ₁, φ₂ = -70.0, -45.0

start_date   = DateTime(get(ENV, "START_DATE", "2014-06-01T12:00"))
end_date     = DateTime(get(ENV, "END_DATE",   "2014-08-01T00:00"))
era5_dir     = get(ENV, "ERA5_DIR", joinpath(get(ENV, "BSOSE_DIR", default_bsose_directory()), "era5"))
max_attempts = parse(Int, get(ENV, "MAX_ATTEMPTS", "20"))

era5_region = BoundingBox(longitude = (λ₁ - 1.0, λ₂ + 1.0), latitude = (φ₁ - 1.0, φ₂ + 1.0))
mkpath(era5_dir)

@info "ERA5 prefetch: $start_date to $end_date into $era5_dir (up to $max_attempts attempts)"

for attempt in 1:max_attempts
    try
        prefetch_era5!(; start_date, end_date, region = era5_region, dir = era5_dir, threads = 1)
        @info "ERA5 prefetch complete on attempt $attempt."
        exit(0)
    catch e
        msg = first(split(sprint(showerror, e), '\n'))
        @warn "Attempt $attempt/$max_attempts failed: $msg"
        attempt < max_attempts && sleep(120)
    end
end

error("ERA5 prefetch did not finish after $max_attempts attempts; resubmit to continue where it stopped.")
