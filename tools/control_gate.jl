# ==============================================================================================
# control_gate.jl -- is the testbed `control` arm the same model as the reference run?
#
#     julia --project=. tools/control_gate.jl
#     REF_RUN=2Y_wind_fw_sw_5day ARM_RUN=testbed/control julia --project=. tools/control_gate.jl
#     WINDOWS=mldmin julia --project=. tools/control_gate.jl        # one window, half the time
#
# WHY THIS EXISTS. testbed_report.jl prints `control` at +45.0% and its own gate text demands
# +56.8% from the reference run. That comparison is NOT VALID as stated, because the two numbers
# were produced by different invocations of mixing_energetics.jl and the excess depends on which
# B-SOSE month the integral is read at:
#
#     ke    = argmax of B-SOSE band-mean MLD over Aug-Nov      -> sets h_end
#     ks    = WINDOW == "mldmin" ? kmin : deepening_start(...) -> sets WHICH COLUMN is integrated
#     excess = req_model(ks, h_end) / req_BSOSE(ks, h_end) - 1
#
# So BOTH columns of the ratio move when `ks` or `h_end` moves, and two things moved between the
# reference number and today:
#
#   1. WINDOW. mixing_energetics defaults to "deepening"; testbed_report.jl hardcodes "mldmin",
#      which its own ENV block marks "(pre-10.29.2026)". Different `ks`, different column.
#   2. The B-SOSE date offset fixed 2026-10-28. That changes which file record maps to which
#      date, hence `ke`, hence h_end -- which is 227.0 m today against the 196.4 m the +56.8%
#      was computed at. A 30 m deeper limit pulls more of the warm/salty 200-600 m layer into
#      BOTH integrals.
#
# WHAT THIS DOES. Runs mixing_energetics.jl once per (window, run) with a SINGLE pinned
# environment -- identical BSOSE_DIR, ITER, YEAR, ZMAX, TRIM_SPONGE, S0, MLD_DT, CADENCE and
# MODEL_START, varying only MODEL_RUN -- and prints the 70S-65S row of each side by side. The
# comparison is apples-to-apples exactly when `ks`, `ke` and `h_end` agree across the two runs,
# and the table says so explicitly rather than leaving it to be eyeballed.
#
# `req BSOSE` is a B-SOSE-only quantity and MUST come out identical for the two runs at the same
# window. It is checked, because if it does not the pinning has a hole in it and nothing else
# here can be trusted.
#
# ENV
#   REF_RUN    the reference run's name          (default: 2Y_wind_fw_sw_5day)
#   ARM_RUN    the testbed arm to gate           (default: testbed/control)
#   WINDOWS    comma-separated windows to sweep  (default: deepening,mldmin)
#   TOL        excess agreement tolerance, pp    (default: 2.0)
#   plus every mixing_energetics knob, which is passed through UNCHANGED to both runs:
#   BSOSE_DIR ITER YEAR ZMAX TRIM_SPONGE S0 MLD_DT CADENCE MODEL_DIR MODEL_START
#
# LOGIN-NODE SAFE, but it is 2 runs x 2 windows = 4 invocations and each one reads 3-D monthly
# Theta and Salt. Use WINDOWS=mldmin to halve it.
# ==============================================================================================

using Printf, Dates

const REF_RUN = get(ENV, "REF_RUN", "2Y_wind_fw_sw_5day")
const ARM_RUN = get(ENV, "ARM_RUN", "testbed/control")
const WINDOWS = [strip(w) for w in split(get(ENV, "WINDOWS", "deepening,mldmin"), ",") if !isempty(strip(w))]
const TOL     = parse(Float64, get(ENV, "TOL", "2.0"))
const BAND    = "70S-65S"

# The knobs that move `ks`, `ke` or `h_end`. Pinned once here and handed to every invocation, so
# the two runs cannot be analysed differently. MODEL_START is shared: both runs begin 2014-06-01,
# and if that ever stops being true this is the line to split.
const PINNED = Dict(
    "BSOSE_DIR"   => get(ENV, "BSOSE_DIR", "data"),
    "ITER"        => get(ENV, "ITER", "156"),
    "YEAR"        => get(ENV, "YEAR", "2015"),
    "ZMAX"        => get(ENV, "ZMAX", "600"),
    "TRIM_SPONGE" => get(ENV, "TRIM_SPONGE", "3.0"),
    "S0"          => get(ENV, "S0", "34.0"),
    "MLD_DT"      => get(ENV, "MLD_DT", "0.03"),
    "CADENCE"     => get(ENV, "CADENCE", "monthly"),
    "MODEL_DIR"   => get(ENV, "MODEL_DIR", "."),
    "MODEL_START" => get(ENV, "MODEL_START", "2014-06-01"),
)

const MODEL_DIRS = [PINNED["MODEL_DIR"], "surface_tests"]

"""Where mixing_energetics would find this run, or `nothing`. Same resolution it uses."""
runfile(name) = (h = findfirst(d -> isfile(joinpath(d, "$(name)_eke.nc")), MODEL_DIRS);
                 h === nothing ? nothing : joinpath(MODEL_DIRS[h], "$(name)_eke.nc"))

"""Every `*_eke.nc` visible, for an error message that is actually actionable."""
function available()
    out = String[]
    for d in MODEL_DIRS, sub in ("", "testbed")
        p = isempty(sub) ? d : joinpath(d, sub)
        isdir(p) || continue
        for f in readdir(p)
            endswith(f, "_eke.nc") || continue
            push!(out, (isempty(sub) ? "" : sub * "/") * replace(f, "_eke.nc" => ""))
        end
    end
    return sort(unique(out))
end

# ── fail before spending four invocations on a typo ────────────────────────────────────────────
# mixing_energetics only WARNS on a missing run and reports `req model` as `--`. That would read
# here as "the reference could not be derived" when the truth is the name was wrong, so the
# existence check happens up front.
missing_runs = [r for r in (REF_RUN, ARM_RUN) if runfile(r) === nothing]
if !isempty(missing_runs)
    av = available()
    error("no _eke.nc for: $(join(missing_runs, ", "))\n" *
          "  looked under: $(join(MODEL_DIRS, ", ")) (and their testbed/ subdirectory)\n" *
          "  available:    " * (isempty(av) ? "NOTHING -- wrong working directory?" : join(av, ", ")) *
          "\n  set REF_RUN / ARM_RUN to two of those.")
end

"""The 70S-65S row of one invocation, as a NamedTuple, or `nothing` with the reason logged."""
function energetics_row(name, window)
    env = copy(ENV)
    for (k, v) in PINNED; env[k] = v; end
    env["WINDOW"]    = window
    env["MODEL_RUN"] = name
    buf = IOBuffer()
    cmd = setenv(`julia --project=. $(joinpath(@__DIR__, "mixing_energetics.jl"))`, env)
    ok = try run(pipeline(cmd; stdout = buf, stderr = buf)); true catch; false end
    out = String(take!(buf))
    if !ok
        lines = [strip(l) for l in split(out, '\n') if !isempty(strip(l))]
        why = something(findfirst(l -> occursin("ERROR", l) || occursin("needs", l), lines), 0)
        @warn "mixing_energetics failed: $name at WINDOW=$window" *
              "\n    " * (why == 0 ? (isempty(lines) ? "no output" : last(lines)) : lines[why])
        return nothing
    end
    # The model column is a WARNING, not an error, so catch it here or read NaNs as physics.
    dropped = findfirst(l -> occursin("skipping the model column", l), split(out, '\n'))
    dropped === nothing || @warn "$name at WINDOW=$window: mixing_energetics dropped the model column"
    for ln in split(out, '\n')
        f = split(ln)
        length(f) == 10 && f[1] == BAND || continue
        p(s) = (v = tryparse(Float64, s); v === nothing ? NaN : v)
        return (run = name, window = window, ks = f[2], ke = f[3],
                h0 = p(f[4]), h_end = p(f[5]), req_b = p(f[6]), req_m = p(f[7]))
    end
    @warn "no $BAND row for $name at WINDOW=$window"
    return nothing
end

excess(r) = (isfinite(r.req_b) && r.req_b != 0 && isfinite(r.req_m)) ?
            100 * (r.req_m / r.req_b - 1) : NaN

# ── collect ───────────────────────────────────────────────────────────────────────────────────
@info "pinned: " * join(["$k=$v" for (k, v) in sort(collect(PINNED), by = first)], " ")
@info "reference: $(runfile(REF_RUN))"
@info "arm:       $(runfile(ARM_RUN))"
rows = Dict{Tuple{String,String},Any}()
for w in WINDOWS, r in (REF_RUN, ARM_RUN)
    @info "running $r at WINDOW=$w"
    v = energetics_row(r, w)
    v === nothing || (rows[(w, r)] = v)
end
isempty(rows) && error("no invocation produced a $BAND row")

# ── report ────────────────────────────────────────────────────────────────────────────────────
println("\n" * "="^100)
println("CONTROL GATE -- $BAND, `$ARM_RUN` against `$REF_RUN`")
println("="^100)
@printf("\n  %-10s %-22s %-9s %-9s %8s %11s %11s %9s\n",
        "window", "run", "ks", "ke", "h_end", "req BSOSE", "req model", "excess")
@printf("  %-10s %-22s %-9s %-9s %8s %11s %11s %9s\n", "", "", "", "", "(m)", "(m2/s2)", "(m2/s2)", "")
for w in WINDOWS, r in (REF_RUN, ARM_RUN)
    haskey(rows, (w, r)) || (@printf("  %-10s %-22s %s\n", w, r, "-- did not run"); continue)
    v = rows[(w, r)]
    ex = isfinite(excess(v)) ? @sprintf("%+8.1f%%", excess(v)) : lpad("--", 9)
    @printf("  %-10s %-22s %-9s %-9s %8.1f %11.4f %11.4f %s\n",
            w, r, v.ks, v.ke, v.h_end, v.req_b, v.req_m, ex)
end

println("\n  VERDICT")
for w in WINDOWS
    a, b = get(rows, (w, REF_RUN), nothing), get(rows, (w, ARM_RUN), nothing)
    if a === nothing || b === nothing
        @printf("    %-10s incomplete -- one side did not produce a row\n", w); continue
    end
    # The pinning self-check. req BSOSE is B-SOSE-only and cannot depend on MODEL_RUN.
    if !isapprox(a.req_b, b.req_b; rtol = 1e-6)
        @printf("    %-10s PINNING IS LEAKING: req BSOSE differs (%.4f vs %.4f) at the same\n",
                w, a.req_b, b.req_b)
        println("               window. It is a B-SOSE-only quantity, so something outside")
        println("               PINNED is reaching the analysis. Nothing below is trustworthy.")
        continue
    end
    same = a.ks == b.ks && a.ke == b.ke && isapprox(a.h_end, b.h_end; rtol = 1e-6)
    d = excess(b) - excess(a)
    if !same
        @printf("    %-10s NOT COMPARABLE: ks %s/%s, ke %s/%s, h_end %.1f/%.1f\n",
                w, a.ks, b.ks, a.ke, b.ke, a.h_end, b.h_end)
    elseif abs(d) <= TOL
        @printf("    %-10s PASS: %+.1f%% vs %+.1f%%, difference %+.1f pp (within %.1f)\n",
                w, excess(b), excess(a), d, TOL)
    else
        @printf("    %-10s FAIL: %+.1f%% vs %+.1f%%, difference %+.1f pp (tolerance %.1f)\n",
                w, excess(b), excess(a), d, TOL)
        println("               Same ks, ke and h_end, so this is a real difference between the")
        println("               two MODEL runs -- the testbed is not reproducing the reference.")
    end
end

println()
println("  HOW TO READ THIS")
println("    The number to quote from now on is whatever the REFERENCE row says at the window the")
println("    testbed uses. +56.8% was computed at h_end = 196.4 m, before the 2026-10-28 date fix,")
println("    and is superseded by this table -- do not compare any arm to it again.")
println()
println("    PASS at the testbed's window means `control` reproduces the reference and the other")
println("    arms can be read. FAIL with matching ks/ke/h_end means the testbed itself diverged:")
println("    check START_DATE, END_DATE and the consts run_testbed rewrote into")
println("    testbed/model_control.jl.")
println()
println("    If the two windows disagree about the excess, that is the window sensitivity the tool")
println("    warns about, not a model result. Pick one window, record it, keep it for every arm.")
