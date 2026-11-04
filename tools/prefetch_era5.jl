# ==============================================================================================
# prefetch_era5.jl -- download the ERA5 cache the nested run reads, as its own CPU job.
#
#     ./create_job.sh era5_prefetch cpu 48 none 8G tools/prefetch_era5.jl
#     sbatch slurm_era5_prefetch.sh
#
# BACKEND: CDSAPI, NOT CopernicusClimateDataStore. Two differences, both of which bit us:
#
#   - No timeout. CopernicusClimateDataStore gives up on a queued request after 3600 s and
#     leaves it queued on CDS, so every retry joined the back of the queue behind its own
#     orphan. CDSAPI.retrieve polls until the request succeeds or CDS reports it failed.
#   - One request for all variables. NumericalEarth's CDSAPI extension bundles every variable of
#     a MetadataSet into one request per calendar month (split to stay under 5000 fields), so
#     2 years is ~48 requests instead of 24 months x 8 variables = 192.
#
# NEVER load CopernicusClimateDataStore in the same session: both NumericalEarth extensions
# define `download(::MetadataSet{<:ERA5Dataset})`, and whichever loads last silently wins.
# That is also why this does not call `prefetch_era5!` -- that goes one variable at a time,
# and src/model.jl depends on it as is.
#
# One-time setup on Phoenix (no version changes, no login-node precompile):
#     JULIA_PKG_PRECOMPILE_AUTO=0 julia --project=. -e 'using Pkg; Pkg.add("CDSAPI"; preserve=Pkg.PRESERVE_ALL)'
#
# Credentials: CDSAPI_URL / CDSAPI_KEY (sourced from .copernicus_env by the job) or ~/.cdsapirc.
#
# MUST MATCH examples/sowinwarp_nested.jl -- same region and directory, or the GPU job will not
# find these files and will download again. Files already on disk are skipped, so a resubmit
# continues where the last one stopped.
#
#     START_DATE=2014-06-01T12:00  END_DATE=2014-08-01  ERA5_DIR=...  MAX_ATTEMPTS=5
#     CDS_POLL_SECONDS=30
# ==============================================================================================

using CDSAPI                         # the ERA5 download backend -- see the header
using NumericalEarth
using NumericalEarth.DataWrangling: MetadataSet
using Downloads
using Dates

include(joinpath(@__DIR__, "..", "src", "setup_bsose.jl"))   # default_bsose_directory

const λ₁, λ₂ = 90.0, 150.0
const φ₁, φ₂ = -70.0, -45.0

# The eight ERA5PrescribedAtmosphere + ERA5PrescribedRadiation read; same list as prefetch_era5!.
const ERA5_VARIABLES = (:eastward_velocity, :northward_velocity, :temperature,
                        :dewpoint_temperature, :surface_pressure, :total_precipitation,
                        :downwelling_shortwave_radiation, :downwelling_longwave_radiation)

start_date   = DateTime(get(ENV, "START_DATE", "2014-06-01T12:00"))
end_date     = DateTime(get(ENV, "END_DATE",   "2014-08-01T00:00"))
era5_dir     = get(ENV, "ERA5_DIR", joinpath(get(ENV, "BSOSE_DIR", default_bsose_directory()), "era5"))
max_attempts = parse(Int, get(ENV, "MAX_ATTEMPTS", "5"))
poll_seconds = parse(Float64, get(ENV, "CDS_POLL_SECONDS", "30"))

era5_region = BoundingBox(longitude = (λ₁ - 1.0, λ₂ + 1.0), latitude = (φ₁ - 1.0, φ₂ + 1.0))
mkpath(era5_dir)

mset = MetadataSet(ERA5_VARIABLES...; dataset = ERA5HourlySingleLevel(),
                   start_date, end_date, dir = era5_dir, region = era5_region)

# CDSAPI polls every second by default; a request can queue for hours, so ask less often.
retrieve(product, request, path) = CDSAPI.retrieve(product, request, path; wait = poll_seconds)

@info "ERA5 prefetch via CDSAPI: $(length(ERA5_VARIABLES)) variables, $start_date to $end_date, into $era5_dir"

# The retries are for real failures now (a dropped connection during a long wait, a CDS-side
# "failed"), not for the queue: a slow queue just takes longer.
for attempt in 1:max_attempts
    try
        Downloads.download(mset; retrieve)
        @info "ERA5 prefetch complete on attempt $attempt."
        exit(0)
    catch e
        msg = first(split(sprint(showerror, e), '\n'))
        @warn "Attempt $attempt/$max_attempts failed: $msg"
        attempt < max_attempts && sleep(300)
    end
end

error("ERA5 prefetch did not finish after $max_attempts attempts; resubmit to continue where it stopped.")
