# ==============================================================================
# compare_obc_logs.jl -- rank the run_obc_tests arms from their LOGS, not their surface files.
#
#     julia --project=. tools/compare_obc_logs.jl
#     julia --project=. tools/compare_obc_logs.jl logs           # another directory
#
# WHY NOT compare_obc_tests.jl. That script reads `ds["u"][:, :, 1, t]` -- the SURFACE slice --
# and this model's worst feature has twice been subsurface. In the 2026-10-24 run the surface
# map showed a clean ~1.4 m/s plateau while the 3-D maximum went 1.82 -> 4.05 m/s in a BOTTOM
# cell at (1,21,39) and kept climbing. A surface-only comparison would have ranked that run as
# fine. Use both: this for what is happening, that one for the spatial pattern.
#
# Everything this needs is already printed per output interval and found nowhere else:
#
#     |u⃗|ₘₐₓ = [1.53e+00,  2.15e+00,  2.09e-02] m/s,  advective CFL = 0.65
#     [max|u|]   1.528 m/s at (i=  7, j=149, k=132)  z = -17.5 m  seafloor = -3501.9 m | ...
#     [max|eta|] raw 1.504 at (i= 89, j=150) seafloor -3501.9 m [wet] | wet ... 0 cells from ...
#     [eta face] north : row mean -0.0712 m, ... | one cell in -0.0840 m | offset +0.0128 m
#     transport (Sv): W  +186.83 | E  +214.95 | S +0.00 | N  -28.12 || net out -0.000
#
# THE HEADLINE METRIC IS NOT A VELOCITY. It is whether the maximum is PINNED to one cell or
# WANDERS. That distinction is what separated the two problems this model has had:
#
#   pinned at one (i,j) for days on end  => a boundary/geometry artifact
#   a different (i,j) every day          => resolved turbulence, however fast
#
# The bathymetry fix did not lower the velocity much at first glance -- but it moved the maximum
# from eight consecutive days at (1,21) to a new location daily, and that was the result. So
# this reports `pin%` (the share of outputs the modal location holds) and `dface` (mean cells to
# the nearest face) alongside the magnitudes, and sorts on them.
# ==============================================================================

using Printf, Statistics

const LOGDIR = length(ARGS) > 0 ? ARGS[1] : "logs"
const SPINUP_DAY = 5.0   # growth rates measured from here, to skip the start-up transient

# logs/ accumulates. It still holds the 2026-10-03 batch (`clamped_90d`, `oblique_90d`, ...),
# whose results are VOID -- they predate the barotropic boundary condition existing at all and
# the west/east index bug. A plain `grep RESULT logs/*` mixes them in with today's, and the old
# ones even arrive twice because the same arm was submitted under two job ids.
#
# So: keep only the most recent BATCH (every log within SINCE_HOURS of the newest one), and
# within it keep only the newest log per arm. Both exclusions are printed rather than silent,
# because a comparison quietly dropping arms is worse than one that shows too many.
const SINCE_HOURS = parse(Float64, get(ENV, "SINCE_HOURS", "24"))

isdir(LOGDIR) || error("no $LOGDIR/ directory -- run from the repository root or pass one")

# ---------------------------------------------------------------- parsing
const RE_CONFIG = r"^\s*Config\s*:\s*(\S+)"
const RE_RESULT = r"^RESULT\s+(\S+):\s+(completed|FAILED)"
const RE_ITER   = r"time = (\d+(?:\.\d+)?) days?,\s+Δt = ([\d.]+) minutes"
const RE_UMAX   = r"\|u⃗\|ₘₐₓ = \[([\d.eE+-]+),\s+([\d.eE+-]+),\s+([\d.eE+-]+)\].*advective CFL = ([\d.]+)"
const RE_ARGMAX = r"\[max\|(u|v)\|\]\s+([\d.eE+-]+) m/s at \(i=\s*(\d+), j=\s*(\d+), k=\s*(\d+)\).*?seafloor =\s*(-?[\d.]+) m\s+\|\s+(\d+) cells from west/east,\s+(\d+) from south/north"
const RE_ETAMAX = r"\[max\|eta\|\] raw\s+([\d.eE+-]+) at \(i=\s*(\d+), j=\s*(\d+)\) seafloor\s+(-?[\d.]+) m \[(\w+)\]\s+\|\s+wet\s+([\d.]+) at \(i=\s*(\d+), j=\s*(\d+)\),\s+(\d+) cells from west/east,\s+(\d+) from south/north"
const RE_ETAFACE = r"\[eta face\]\s+(\w+)\s+: row mean\s+(\S+) m.*offset\s+(\S+) m"
const RE_TRANS  = r"transport \(Sv\): W\s+(\S+) \| E\s+(\S+) \| S\s+(\S+) \| N\s+(\S+) \|\| net out\s+(\S+)"
const RE_WET    = r"->\s+(\w+)\s+face:\s+(\d+)/\s*(\d+) columns wet"
const RE_FLAT   = r"-> FLAT BOTTOM: (\d+) wet columns set to (\d+) m"
const RE_FIXTOPO = r"-> OBCSfixtopo: raised (\d+) boundary column"
const RE_NAN    = r"NaN|InexactError|isnan"

f(x) = parse(Float64, x)
i(x) = parse(Int, x)

"""Everything worth knowing from one arm's log."""
function parse_log(paths)
    path = first(paths)
    name = replace(basename(path), r"^obc_" => "", r"-\d+$" => "", r"\.(out|err)$" => "")
    result, days = "no RESULT line", 0.0
    dts, cfls, umax3, vmax3 = Float64[], Float64[], Float64[], Float64[]
    daylist = Float64[]
    uloc, vloc = Tuple{Int,Int,Int,Int}[], Tuple{Int,Int,Int,Int}[]   # i, j, k, dist-to-face
    etawet, etaface = Any[], Dict{String,Float64}()
    trans = nothing
    wet = Dict{String,Tuple{Int,Int}}()
    flat, fixtopo, sawnan = nothing, nothing, false

    for line in Iterators.flatten(eachline(p) for p in paths if isfile(p))
        if (m = match(RE_CONFIG, line)) !== nothing;  name = m[1]
        elseif (m = match(RE_RESULT, line)) !== nothing; result = m[2]
        elseif (m = match(RE_WET, line)) !== nothing;   wet[m[1]] = (i(m[2]), i(m[3]))
        elseif (m = match(RE_FLAT, line)) !== nothing;  flat = (i(m[1]), i(m[2]))
        elseif (m = match(RE_FIXTOPO, line)) !== nothing; fixtopo = i(m[1])
        elseif (m = match(RE_ITER, line)) !== nothing
            push!(daylist, f(m[1])); push!(dts, f(m[2])); days = max(days, f(m[1]))
        elseif (m = match(RE_UMAX, line)) !== nothing
            push!(umax3, f(m[1])); push!(vmax3, f(m[2])); push!(cfls, f(m[4]))
        elseif (m = match(RE_ARGMAX, line)) !== nothing
            rec = (i(m[3]), i(m[4]), i(m[5]), min(i(m[7]), i(m[8])))
            m[1] == "u" ? push!(uloc, rec) : push!(vloc, rec)
        elseif (m = match(RE_ETAMAX, line)) !== nothing
            push!(etawet, (f(m[6]), i(m[7]), i(m[8]), min(i(m[9]), i(m[10])), m[5]))
        elseif (m = match(RE_ETAFACE, line)) !== nothing
            etaface[m[1]] = f(replace(m[3], "+" => ""))
        elseif (m = match(RE_TRANS, line)) !== nothing
            trans = (f(m[1]), f(m[2]), f(m[3]), f(m[4]), f(m[5]))
        elseif occursin(RE_NAN, line); sawnan = true
        end
    end

    return (; name, path, result, days, dts, cfls, umax3, vmax3, daylist,
            uloc, vloc, etawet, etaface, trans, wet, flat, fixtopo, sawnan)
end

"""Growth of `series` in units per day, measured from `SPINUP_DAY` onward."""
function growth(days, series)
    n = min(length(days), length(series))
    n < 2 && return NaN
    keep = [t for t in 1:n if days[t] >= SPINUP_DAY]
    length(keep) < 2 && return NaN
    Δd = days[keep[end]] - days[keep[1]]
    Δd <= 0 ? NaN : (series[keep[end]] - series[keep[1]]) / Δd
end

"""
Pinning, measured PER AXIS, plus the mean distance to the nearest face.

Per axis and not on the (i,j) pair, because the pair would have missed the real thing. In the
2026-10-25 `tgrad_eta` run max|v| sat at **i = 1 on sixteen consecutive days** while j migrated
north about a cell a day: (1,144) (1,145) (1,129) (1,131) (1,132) (1,134) (1,136) (1,137). The
modal PAIR there holds roughly 15% of the outputs, which would read as "wandering, healthy" --
exactly backwards. The modal i holds 100%.

Pinned in one axis and free in the other is the signature of a disturbance traveling ALONG a
wall, which is the thing to look for. Pinned in both is a single stuck cell. Free in both is
turbulence.
"""
function pinning(locs)
    isempty(locs) && return (NaN, NaN, NaN, "")
    share(v) = (c = Dict{Int,Int}(); for x in v; c[x] = get(c, x, 0) + 1; end;
                m = argmax(c); (100c[m] / length(v), m))
    pi_, mi = share([l[1] for l in locs])
    pj_, mj = share([l[2] for l in locs])
    return (pi_, pj_, mean(l[4] for l in locs), @sprintf("(%d,%d)", mi, mj))
end

# One ARM is a stem with up to two files beside it: NAME-JOBID.out and NAME-JOBID.err.
#
# Julia's `@info` writes to STDERR, so every diagnostic -- the progress blocks, [max|u|],
# [max|eta|], [eta face], the transports -- is in `.err`, and only the shell `echo` lines
# (Config, RESULT) reach `.out`. Reading `.out` alone yields a full list of arms with every
# number NaN, which is what the first version of this script printed as if it were data.
raw = filter(p -> occursin(r"\.(out|err)$", p) && occursin("obc_", basename(p)),
             readdir(LOGDIR; join = true))
isempty(raw) && error("no obc_*.{out,err} files in $LOGDIR/")
all_logs = sort(unique(replace.(raw, r"\.(out|err)$" => "")))

# Report the raw file count FIRST. The first version of this script printed "1 arms in the most
# recent batch" while the directory held twenty-one logs, and because the dropped ones appeared
# in neither the kept list nor the excluded list there was no way to see that from the output.
# A filter that can discard most of its input must show its working.
@printf("found %d arm log(s) in %s/ (%d files, .out + .err)\n",
        length(all_logs), LOGDIR, length(raw))
stem_mtime(st) = maximum(mtime(f) for f in (st * ".out", st * ".err") if isfile(f))
if get(ENV, "VERBOSE", "0") != "0"
    for p in all_logs
        @printf("    %-52s  %s  %s\n", basename(p),
                Libc.strftime("%Y-%m-%d %H:%M", stem_mtime(p)),
                join([e for e in (".out", ".err") if isfile(p * e)], " "))
    end
end

newest = maximum(stem_mtime, all_logs)
cutoff = newest - SINCE_HOURS * 3600
recent = filter(p -> stem_mtime(p) >= cutoff, all_logs)
stale  = filter(p -> stem_mtime(p) < cutoff, all_logs)

parsed = [parse_log([st * ".out", st * ".err"]) for st in recent]
arms = filter(a -> !isempty(a.dts) || a.result != "no RESULT line", parsed)
dropped = setdiff(basename.(getproperty.(parsed, :path)), basename.(getproperty.(arms, :path)))
isempty(dropped) || @printf("skipped %d log(s) with no progress output (prep jobs, empties): %s\n",
                            length(dropped), join(sort(dropped), ", "))

# One arm resubmitted keeps only its newest log.
by_name = Dict{String,Any}()
dupes = String[]
for a in arms
    if haskey(by_name, a.name)
        older, newer = mtime(by_name[a.name].path) < mtime(a.path) ? (by_name[a.name], a) : (a, by_name[a.name])
        push!(dupes, @sprintf("%s (kept %s, dropped %s)", a.name,
                              basename(newer.path), basename(older.path)))
        by_name[a.name] = newer
    else
        by_name[a.name] = a
    end
end
arms = collect(values(by_name))

@printf("%d arm(s) kept, within %.0f h of the newest log: %s\n",
        length(arms), SINCE_HOURS, join(sort(getproperty.(arms, :name)), ", "))
if !isempty(stale)
    @printf("EXCLUDED %d older log(s) from an earlier batch: %s\n", length(stale),
            join(sort(unique(replace.(basename.(stale), r"-\d+$" => ""))), ", "))
    println("         pass SINCE_HOURS=100000 to include them -- but pre-2026-10-23 results are void.")
end
isempty(dupes) || println("DEDUPLICATED: ", join(dupes, "; "))

# ------------------------------------------------------------------ the table
println()
@printf("%-24s %-9s %5s  %5s %5s  %6s %6s  %7s  %5s %5s %5s %-10s\n",
        "arm", "result", "days", "Δt1", "Δtn", "max|u|", "max|v|",
        "dv/dt", "pinI%", "pinJ%", "dface", "modal(i,j)")
println("-"^114)

for a in sort(arms, by = x -> (x.result != "completed", -(isempty(x.vmax3) ? 0.0 : last(x.vmax3))))
    dt1 = isempty(a.dts) ? NaN : first(a.dts)
    dtn = isempty(a.dts) ? NaN : last(a.dts)
    umx = isempty(a.umax3) ? NaN : last(a.umax3)
    vmx = isempty(a.vmax3) ? NaN : last(a.vmax3)
    dv  = growth(a.daylist, a.vmax3)
    pinI, pinJ, dface, modal = pinning(a.vloc)
    @printf("%-24s %-9s %5.0f  %5.1f %5.1f  %6.2f %6.2f  %+7.3f  %5.0f %5.0f %5.1f %-10s%s\n",
            a.name, a.result, a.days, dt1, dtn, umx, vmx, dv, pinI, pinJ, dface, modal,
            a.sawnan ? "  NaN" : "")
end

# ------------------------------------------------------------- what each column means
println("""

  Δt1/Δtn  first and last timestep, minutes. A collapsing Δt is the cheapest sign of trouble.
  max|u|   max|v|   the 3-D maxima at the last output -- NOT the surface values.
  dv/dt    growth of max|v| per day from day $(Int(SPINUP_DAY)) on. Near zero is the goal; the
           reference runs sat at +0.187.
  pinI%    share of outputs the modal i of max|v| holds -- and pinJ% the modal j, SEPARATELY.
           Measured per axis on purpose: in the reference run i was pinned at 1 for sixteen
           days while j migrated, so the modal (i,j) PAIR scored ~15% and would have read as
           healthy. High pinI% with low pinJ% = a disturbance traveling along a wall. Both
           high = one stuck cell. Both low = turbulence.
  dface    mean cells from the argmax to the nearest face. LOW IS BAD for the same reason.
""")

# ------------------------------------------------------------------ per-arm detail
println("="^108)
println("DETAIL -- per-face eta offset (boundary row minus one cell in) and transport closure")
println("A face held away from the interior beside it is being driven by its own condition.")
println("="^108)

for a in sort(arms, by = x -> x.name)
    isempty(a.etaface) && a.trans === nothing && continue
    @printf("\n%-24s  %s\n", a.name, a.result)
    if !isempty(a.etaface)
        print("    eta offset:")
        for s in ("west", "east", "south", "north")
            haskey(a.etaface, s) && @printf("  %s %+.4f m", s, a.etaface[s])
        end
        println()
    end
    if !isempty(a.etawet)
        v, ei, ej, ed, land = last(a.etawet)
        @printf("    max|eta| wet %.3f m at (%d,%d), %d cells from a face; raw argmax on %s\n",
                v, ei, ej, ed, land)
    end
    if a.trans !== nothing
        w, e, s_, n_, net = a.trans
        @printf("    transport   W %+8.2f  E %+8.2f  S %+8.2f  N %+8.2f   net %+8.3f Sv\n",
                w, e, s_, n_, net)
    end
    a.flat !== nothing && @printf("    FLAT BOTTOM %d columns at %d m; OBCSfixtopo raised %s\n",
                                  a.flat[1], a.flat[2],
                                  a.fixtopo === nothing ? "?" : string(a.fixtopo))
    if !isempty(a.wet)
        print("    wet columns:")
        for s in ("west", "east", "south", "north")
            haskey(a.wet, s) && @printf("  %s %d/%d", s, a.wet[s][1], a.wet[s][2])
        end
        println()
    end
end

println("""

Reading this:
  * A FLAT arm whose `OBCSfixtopo raised` is not 0 did not actually flatten -- that arm is
    meaningless and should be rerun, not interpreted.
  * `normal_clamped` is expected to FAIL around day 3 and `bt_flather` to invert the
    circulation (east and north reversing sign). Those two are controls; a FAILED line from
    them confirms the harness, it does not indicate a problem.
  * Compare `flat_upper` against `base` FIRST. If flat_upper has pin% near the others and
    dv/dt near zero, the boundary machinery is sound and the whole problem is depth/geometry.
    If flat_upper still pins at i=1, the machinery is implicated on trivial geometry, which
    would be the single most useful thing these runs could tell us.
  * Then `tools/boundary_components.jl` (with TRIM_CORNERS=30) on whichever arms disagree, to
    see whether the surviving signal is normal or tangential.
""")
