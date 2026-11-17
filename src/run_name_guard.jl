# run_name_guard.jl -- refuse to start a run whose output files another job is still writing.
#
#     include(joinpath(@__DIR__, "..", "src", "run_name_guard.jl")); guard_run_name(RUN_NAME)
#
# Two jobs with the same RUN_NAME write the same NetCDF files, and with overwrite_files = true the second
# truncates them under the first. On 2026-11-16 a job submitted as "ggl" picked up CLOSURE = "catke" from the
# script (edited after submission, read at start), so it and the real "catke" job both wrote
# 2Y_nested_bsose_catke_hj_pr207_*.nc; the first died with an HDF error at its day 15. A file written in the
# last `minutes` means a run is (or just was) live under this name: stop, and say which file.

function guard_run_name(run_name; dir = pwd(), minutes = 15)
    live = [f for f in readdir(dir) if startswith(f, run_name * "_") && endswith(f, ".nc") &&
                                        time() - mtime(joinpath(dir, f)) < 60 * minutes]
    isempty(live) ||
        error("$(join(live, ", ")) in $dir was written in the last $minutes minutes: another job is probably running " *
              "as $run_name. Change RUN_NAME (or the setting it is built from), or wait for that job to end.")
    return nothing
end
