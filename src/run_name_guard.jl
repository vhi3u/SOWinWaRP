# run_name_guard.jl -- one job per RUN_NAME: a run claims its name with a lock directory and stops at startup if
# another job holds it.
#
#     include(joinpath(@__DIR__, "..", "src", "run_name_guard.jl")); guard_run_name(RUN_NAME)
#
# Two jobs with the same RUN_NAME write the same NetCDF files, and the second truncates them under the first. On
# 2026-11-16 that happened twice. First a job submitted as "ggl" picked up CLOSURE = "catke" from a script edited
# after submission (a job reads its script when it STARTS) and ran beside the real "catke" job: HDF error -101 on
# day 15. Then two jobs started together as 2Y_nested_bsose_ggl_hj_pr207: both passed a check on the files' ages,
# both created the files, and one died at its first write (HDF error -105). A lock directory closes that race:
# `mkdir` either creates it or fails, atomically, so only one job can hold a name.
#
# The lock, <RUN_NAME>.lock/, holds the job id, node and start time, and is removed when Julia exits (normally or on
# an error). A job killed outright (scancel, a node failure, the wall-time limit) can leave it behind: then the
# error below says so, and `rm -r <RUN_NAME>.lock` frees the name once no job is running as it.

function guard_run_name(run_name; dir = pwd())
    lock = joinpath(dir, run_name * ".lock")
    try
        mkdir(lock)
    catch err
        err isa Base.IOError || rethrow()
        holder = isfile(joinpath(lock, "holder")) ? strip(read(joinpath(lock, "holder"), String)) : "unknown"
        error("$run_name is locked by another job ($holder): $lock exists. If no job is running as $run_name " *
              "(check squeue), it was left by a killed run: `rm -r $lock` and resubmit. Otherwise change RUN_NAME " *
              "(or the setting it is built from).")
    end
    write(joinpath(lock, "holder"), "job $(get(ENV, "SLURM_JOB_ID", "none")) on $(gethostname()), started $(Dates.now())\n")
    atexit(() -> rm(lock; recursive = true, force = true))
    @info "run name $run_name locked ($lock; removed when the run exits)"
    return nothing
end
