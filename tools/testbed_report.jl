# ==============================================================================================
# testbed_report.jl -- one report over every arm of the testbed sweep.
#
#     julia --project=. tools/testbed_report.jl
#     ARMS=control,sponge1 julia --project=. tools/testbed_report.jl
#     SKIP_ENERGETICS=1 julia --project=. tools/testbed_report.jl     # tables+profiles only, fast
#
# WHAT IT ASSERTS. Five numbers per arm, each tied to a specific claim about the model, so an
# arm is read as "did this move the number it was meant to move, and did it break the others".
#
#   excess       req model / req BSOSE - 1 at 70-65S. THE HEADLINE. 2Y_wind_fw_sw_5day reads
#                +56.8% trimmed. Computed by shelling out to tools/mixing_energetics.jl rather
#                than reimplementing `deficit_profile`, so the two cannot drift apart.
#   dS 0-100     upper-ocean freshening at 67-65S. Seasonal and NOT accumulating in the
#                reference run (15-03 -0.2214 against 16-03 -0.2127).
#   dS 200-400   the only quantity still running away: +0.03 g/kg/yr, no plateau. Sits BELOW
#                h_end = 196 m, so it does not pay into `excess` -- a separate failure.
#   dT 100-200   the Winter Water warm bias, +0.11 -> +0.30 monotonic. The project's variable.
#   surf drift   domain-mean surface dS, the compare_bsose section-4 number.
#
# THE SPONGE IS TRIMMED (tools/sponge.jl, TRIM_SPONGE=3 deg default). Without it the 70-65S
# band is ~30% water held onto B-SOSE on a 12-24 h timescale and every anomaly is understated.
# The `sponge1` arm narrows the southern sponge to 1 deg, so for a like-for-like comparison
# across arms the trim must be the SAME for all of them -- it stays at 3 here whatever an arm
# ran with, and that is why `covers` is printed.
#
# ENV
#   TESTBED_DIR   where the arms live                  (default: testbed)
#   ARMS          comma-separated subset               (default: every arm found)
#   BSOSE_DIR     monthly Theta/Salt                   (default: data)
#   MODEL_START   simulation time zero                 (default: 2014-06-01)
#   YEAR          year for the required pair           (default: 2015)
#   CADENCE       fiveday | monthly, for the required pair (default: fiveday, as the arms run)
#   SKIP_ENERGETICS  1 to skip the mixing_energetics shell-outs
#   OUTDIR        figures                              (default: testbed/figs)
#
# LOGIN-NODE SAFE: NCDatasets and CairoMakie only, one month resident at a time.
# ==============================================================================================

using NCDatasets, Dates, Printf, Statistics, CairoMakie

const TBDIR  = get(ENV, "TESTBED_DIR", "testbed")
const DIR    = get(ENV, "BSOSE_DIR", "data")
const MSTART = DateTime(get(ENV, "MODEL_START", "2014-06-01"))
const YEAR   = parse(Int, get(ENV, "YEAR", "2015"))
const OUTDIR = get(ENV, "OUTDIR", joinpath(TBDIR, "figs"))
const SKIPE  = get(ENV, "SKIP_ENERGETICS", "0") == "1"
# The arms run SURFACE_CADENCE = :fiveday, so that is the default -- but it needs the per-year
# 5-day flux files, and an arm run at :monthly must be read at :monthly or the shell-out dies.
const CADENCE = get(ENV, "CADENCE", "fiveday")

const LON   = (90.0, 150.0)
const LAT   = (-70.0, -45.0)
const BANDS = [(-70.0, -65.0), (-65.0, -60.0), (-60.0, -55.0), (-55.0, -50.0), (-50.0, -45.0)]
const ICE   = BANDS[1]
const BINS  = [(0.0, 100.0), (100.0, 200.0), (200.0, 400.0)]

include(joinpath(@__DIR__, "sponge.jl"))
const KEEP = sponge_trimmer(LON, LAT)

bfile(tok) = joinpath(DIR, "$(tok)_bsoseI156_2013to2024_monthly.nc")

function coordvar(ds, cands, what)
    for c in cands
        haskey(ds, c) && return Float64.(coalesce.(ds[c][:], NaN))
    end
    error("no $what coordinate in $(path(ds)); tried $(join(cands, ", "))")
end

"""B-SOSE monthly stamps sit at the END of the window; its centre is the middle of that month."""
function window_center(stamp)
    stop = round(DateTime(stamp), Month)
    start = stop - Month(1)
    return start + Millisecond(Dates.value(Millisecond(stop - start)) ÷ 2)
end

"""Linear interpolation onto positive depths `t`, holding the nearest value outside the range."""
function interp_depth(d, v, t)
    ord = sortperm(d); d, v = d[ord], v[ord]
    keep = findall(i -> isfinite(d[i]) && isfinite(v[i]), eachindex(d))
    isempty(keep) && return fill(NaN, length(t))
    d, v = d[keep], v[keep]
    map(t) do z
        z <= d[1] && return v[1]
        z >= d[end] && return v[end]
        k = searchsortedfirst(d, z)
        w = (z - d[k - 1]) / (d[k] - d[k - 1])
        return (1 - w) * v[k - 1] + w * v[k]
    end
end

"""Cos-lat weighted band-mean profile over the UNSPONGED cells, surface-first on return."""
function band_profile(A, lons, lats, lo, hi; surface_first)
    nk = size(A, 3); num = zeros(nk); den = zeros(nk)
    for j in axes(A, 2)
        (lo <= lats[j] < hi) || continue
        w = cosd(lats[j])
        for i in axes(A, 1)
            KEEP(lons[i], lats[j]) || continue
            for k in 1:nk
                v = A[i, j, k]
                # Oceananigans writes immersed cells as EXACT zeros, B-SOSE's land is NaN;
                # `isfinite && != 0` is the one wet test right for both.
                (isfinite(v) && v != 0) || continue
                num[k] += w * v; den[k] += w
            end
        end
    end
    p = [den[k] == 0 ? NaN : num[k] / den[k] for k in 1:nk]
    return surface_first ? p : reverse(p)
end

bin_mean(prof, z, lo, hi) =
    (idx = findall(k -> lo <= z[k] < hi && isfinite(prof[k]), eachindex(z));
     isempty(idx) ? NaN : mean(prof[idx]))

# ── which arms are present ────────────────────────────────────────────────────────────────────
isdir(TBDIR) || error("no $TBDIR directory -- run ./run_testbed first")
found = sort([replace(f, "_eke.nc" => "") for f in readdir(TBDIR) if endswith(f, "_eke.nc")])
const ARMS = let want = get(ENV, "ARMS", "")
    isempty(want) ? found : [a for a in split(want, ",") if (a in found || (@warn "no $(a)_eke.nc in $TBDIR"; false))]
end
isempty(ARMS) && error("no <arm>_eke.nc files in $TBDIR")
@info "arms: $(join(ARMS, ", "))"

# ── B-SOSE geometry, read once ────────────────────────────────────────────────────────────────
probe = Dataset(bfile("Salt"))
bx = coordvar(probe, ["XC"], "longitude"); by = coordvar(probe, ["YC"], "latitude")
bis = findall(x -> LON[1] <= x <= LON[2], bx)
bjs = findall(y -> LAT[1] <= y <= LAT[2], by)
bstamps = [DateTime(t) for t in probe["time"][:]]
close(probe)
const BLON, BLAT = bx[bis], by[bjs]

gds = Dataset(joinpath(DIR, "grid.nc")); RC_all = Float64.(coalesce.(gds["RC"][:], NaN)); close(gds)
const NZ = something(findlast(z -> -z <= 600.0, RC_all), length(RC_all))
const ZB = -RC_all[1:NZ]

# ── per-arm profile metrics ───────────────────────────────────────────────────────────────────
"""Monthly dS/dT profiles at the ice-zone band for one arm, paired to B-SOSE within 20 days."""
function arm_profiles(arm)
    mp = joinpath(TBDIR, "$(arm)_eke.nc")
    mds = Dataset(mp)
    mz_up = coordvar(mds, ["z_aac", "zC", "z_c", "z", "r_aac"], "z")
    mlat  = coordvar(mds, ["y_aca", "φ_aca", "φ_cca", "yC", "latitude", "lat", "y"], "latitude")
    mlon  = let c = ["x_caa", "λ_caa", "λ_fca", "xC", "longitude", "lon", "x"],
                h = findfirst(k -> haskey(mds, k), c)
        h === nothing ?
            [LON[1] + (i - 0.5) * (LON[2] - LON[1]) / size(mds["S"], 1) for i in 1:size(mds["S"], 1)] :
            Float64.(coalesce.(mds[c[h]][:], NaN))
    end
    mt  = Float64.(mds["time"][:])
    if isempty(mt)
        close(mds)
        # Oceananigans CREATES the file at init and appends, so zero records means the run died
        # (walltime/OOM) or has not reached its first write -- not that the file is corrupt.
        # Guard here rather than downstream: `minimum(mcen)` on an empty Vector{DateTime} throws
        # `reducing over an empty collection`, which kills the whole report over one bad arm.
        @warn "$(arm)_eke.nc has ZERO time records -- that run wrote no snapshot. Skipping $arm."
        return (labels = String[], dS = Vector{Float64}[], dT = Vector{Float64}[])
    end
    win = haskey(mds.attrib, "averaging_window_seconds") ?
          Float64(mds.attrib["averaging_window_seconds"]) : 30 * 86400.0
    mcen = [MSTART + Second(round(Int, x - win / 2)) for x in mt]
    mz = -reverse(mz_up)

    lo_t, hi_t = minimum(mcen) - Day(20), maximum(mcen) + Day(20)
    sel = [k for k in eachindex(bstamps) if lo_t <= window_center(bstamps[k]) <= hi_t]

    labels = String[]; dS = Vector{Float64}[]; dT = Vector{Float64}[]
    bS = Dataset(bfile("Salt")); bT = Dataset(bfile("Theta"))
    sv = haskey(bS, "SALT") ? "SALT" : "Salt"
    tv = haskey(bT, "THETA") ? "THETA" : "Theta"
    try
        for k in sel
            c = window_center(bstamps[k])
            gap, m = findmin(abs(Dates.value(x - c)) for x in mcen)
            gap / 86400_000 > 20 && continue
            BS = Float64.(coalesce.(bS[sv][bis, bjs, 1:NZ, k], NaN))
            BT = Float64.(coalesce.(bT[tv][bis, bjs, 1:NZ, k], NaN))
            MS = Float64.(coalesce.(mds["S"][:, :, :, m], NaN))
            MM = Float64.(coalesce.(mds["T"][:, :, :, m], NaN))
            bs = band_profile(BS, BLON, BLAT, ICE...; surface_first = true)
            bt = band_profile(BT, BLON, BLAT, ICE...; surface_first = true)
            ms = interp_depth(mz, band_profile(MS, mlon, mlat, ICE...; surface_first = false), ZB)
            mm = interp_depth(mz, band_profile(MM, mlon, mlat, ICE...; surface_first = false), ZB)
            push!(labels, Dates.format(c, "yy-mm")); push!(dS, ms .- bs); push!(dT, mm .- bt)
        end
    finally
        close(bS); close(bT); close(mds)
    end
    return (labels = labels, dS = dS, dT = dT)
end

# ── the required pair, from the verified tool rather than a second implementation ─────────────
"""`(req_bsose, req_model, h_end)` at 70-65S for one arm, or NaNs. Shells out so the physics
in `deficit_profile` has exactly one implementation."""
function arm_energetics(arm)
    SKIPE && return (NaN, NaN, NaN)
    env = copy(ENV)
    env["MODEL_RUN"] = joinpath(TBDIR, arm)
    env["MODEL_DIR"] = "."
    env["CADENCE"] = CADENCE; env["WINDOW"] = "mldmin"; env["YEAR"] = string(YEAR)
    env["OUTDIR"] = OUTDIR
    # Capture BOTH streams into a buffer and test success explicitly. Letting `read` throw gives
    # a ProcessExited whose message is the entire environment -- forty lines of PATH that bury
    # the one line saying which file was missing.
    buf = IOBuffer()
    cmd = setenv(`julia --project=. $(joinpath(@__DIR__, "mixing_energetics.jl"))`, env)
    ok = try
        run(pipeline(cmd; stdout = buf, stderr = buf))
        true
    catch
        false
    end
    out = String(take!(buf))
    if !ok
        lines = [strip(l) for l in split(out, '\n') if !isempty(strip(l))]
        why = something(findfirst(l -> occursin("ERROR", l) || occursin("needs", l), lines), 0)
        @warn "mixing_energetics failed for $arm -- `excess` and `h_end` will read `--`" *
              "\n    " * (why == 0 ? (isempty(lines) ? "no output" : last(lines)) : lines[why])
        return (NaN, NaN, NaN)
    end
    for ln in split(out, '\n')
        f = split(ln)
        # the table row, not the "bands therefore cover ..." line: ten fields and the band first
        length(f) == 10 && f[1] == "70S-65S" || continue
        p(s) = (v = tryparse(Float64, s); v === nothing ? NaN : v)
        return (p(f[6]), p(f[7]), p(f[5]))
    end
    @warn "no 70S-65S row in mixing_energetics output for $arm"
    return (NaN, NaN, NaN)
end

# ── domain-mean surface drift, the compare_bsose section-4 number ─────────────────────────────
function arm_surface_drift(arm)
    sp = joinpath(TBDIR, "$(arm)_surface.nc")
    isfile(sp) || return NaN
    ds = Dataset(sp)
    try
        haskey(ds, "S") || return NaN
        nt = size(ds["S"], ndims(ds["S"]))
        nt == 0 && return NaN
        A = Float64.(coalesce.(ndims(ds["S"]) == 4 ? ds["S"][:, :, 1, nt] : ds["S"][:, :, nt], NaN))
        v = filter(x -> isfinite(x) && x != 0, vec(A))
        return isempty(v) ? NaN : mean(v)
    finally
        close(ds)
    end
end

# ── collect ───────────────────────────────────────────────────────────────────────────────────
R = Dict{String,Any}()
for a in ARMS
    @info "reading $a"
    pr = arm_profiles(a)
    isempty(pr.labels) && (@warn "$a produced no usable month pairs; skipping"; continue)
    rb, rm, he = arm_energetics(a)
    ser(v, bin, prof) = [bin_mean(prof[n], ZB, bin...) for n in eachindex(prof)]
    R[a] = (labels = pr.labels,
            s0100 = ser(nothing, BINS[1], pr.dS), s100200 = ser(nothing, BINS[2], pr.dS),
            s200400 = ser(nothing, BINS[3], pr.dS), t100200 = ser(nothing, BINS[2], pr.dT),
            dS = pr.dS, req_b = rb, req_m = rm, h_end = he, surf = arm_surface_drift(a))
end
PRESENT = [a for a in ARMS if haskey(R, a)]
isempty(PRESENT) && error("no arm produced usable output")

# ── table ─────────────────────────────────────────────────────────────────────────────────────
println("\n" * "="^104)
println("TESTBED REPORT -- $(length(PRESENT)) arm(s), ice zone = $(Int(-ICE[1]))S to $(Int(-ICE[2]))S")
println("="^104)
println("\n  " * sponge_note(LON, LAT))
tb = trimmed_band(LAT, ICE...)
println("  the ice-zone band therefore covers " *
        (tb === nothing ? "NOTHING" : @sprintf("%.0fS-%.0fS", -tb[1], -tb[2])) *
        " in EVERY arm, including the ones that ran a narrower sponge.\n")

@printf("  %-15s %10s %10s %9s %8s %10s %10s %10s %9s\n", "arm", "req BSOSE", "req model",
        "excess", "h_end", "dS 0-100", "dS 200-400", "dT 100-200", "last mo")
@printf("  %-15s %10s %10s %9s %8s %10s %10s %10s %9s\n", "", "(m2/s2)", "(m2/s2)", "", "(m)",
        "(g/kg)", "(g/kg)", "(degC)", "")
for a in PRESENT
    r = R[a]
    ex = (isfinite(r.req_b) && r.req_b != 0 && isfinite(r.req_m)) ?
         @sprintf("%+8.1f%%", 100 * (r.req_m / r.req_b - 1)) : lpad("--", 9)
    f(v) = isfinite(v) ? @sprintf("%10.4f", v) : lpad("--", 10)
    h(v) = isfinite(v) ? @sprintf("%8.1f", v) : lpad("--", 8)
    @printf("  %-15s %s %s %s %s %s %s %s %9s\n", a, f(r.req_b), f(r.req_m), ex, h(r.h_end),
            f(last(r.s0100)), f(last(r.s200400)), f(last(r.t100200)), last(r.labels))
end

println("\n  HOW TO READ THIS")
println("    `control` must reproduce 2Y_wind_fw_sw_5day (+56.8% over the same window). If it")
println("    does not, stop -- the testbed is wrong and nothing else here means anything.")
println("    sponge1  : excess survives -> the result is real. Collapses -> we studied the sponge.")
println("    nofw     : dS 0-100 AND excess both collapse -> z-star is worth building.")
println("    gm       : read dS 200-400 and its TREND, not the excess. Different failure.")
println("    bsose_*  : read h_end and the excess together. bsose_nocatke has no boundary-layer")
println("               scheme at all, so a shallow mixed layer there is expected, not a finding.")
println("    wind5d   : excess falls -> monthly stress was under-mixing the ice zone.")

# ── drift trend, the thing that is actually running away ──────────────────────────────────────
println("\n  200-400 m SALT TREND (g/kg/yr, least squares over the run)")
for a in PRESENT
    r = R[a]
    y = filter(isfinite, r.s200400)
    if length(y) < 4
        @printf("    %-15s %s\n", a, "too few months"); continue
    end
    x = collect(0:(length(y) - 1)) ./ 12
    b = (length(x) * sum(x .* y) - sum(x) * sum(y)) / (length(x) * sum(x .^ 2) - sum(x)^2)
    @printf("    %-15s %+8.4f   (%d months, %s to %s)\n", a, b, length(y), first(r.labels), last(r.labels))
end

# ── figures ───────────────────────────────────────────────────────────────────────────────────
mkpath(OUTDIR)
cols = Makie.wong_colors()
armcolor(i) = cols[mod1(i, length(cols))]

# 1. the decision figure: four metrics, one point per arm
fig = Figure(size = (1180, 720))
Label(fig[0, 1:2], "Testbed arms against the control — ice zone, sponge trimmed",
      fontsize = 17, font = :bold)
metrics = [("stratification excess (%)", [isfinite(R[a].req_b) && R[a].req_b != 0 ?
                                          100 * (R[a].req_m / R[a].req_b - 1) : NaN for a in PRESENT]),
           ("h_end (m)", [R[a].h_end for a in PRESENT]),
           ("dS 0-100 m, last month (g/kg)", [last(R[a].s0100) for a in PRESENT]),
           ("dS 200-400 m, last month (g/kg)", [last(R[a].s200400) for a in PRESENT])]
for (n, (title, vals)) in enumerate(metrics)
    ax = Axis(fig[fld(n - 1, 2) + 1, mod(n - 1, 2) + 1]; title, titlesize = 12,
              xticks = (1:length(PRESENT), PRESENT), xticklabelrotation = π / 4,
              xticklabelsize = 10, ylabel = "")
    finite = [isfinite(v) ? v : 0.0 for v in vals]
    barplot!(ax, 1:length(PRESENT), finite,
             color = [isfinite(v) ? armcolor(i) : RGBAf(0.8, 0.8, 0.8, 0.4) for (i, v) in enumerate(vals)])
    hlines!(ax, [0], color = :gray50, linewidth = 1)
    # the control's value as the line every other arm is read against
    ci = findfirst(==("control"), PRESENT)
    ci === nothing || isfinite(vals[ci]) &&
        hlines!(ax, [vals[ci]], color = :black, linestyle = :dash, linewidth = 1.5)
end
Label(fig[3, 1:2],
      "Dashed line is the `control` arm. Grey bars are arms whose number could not be computed — " *
      "usually an arm that did not reach the\nseason `mixing_energetics` needs, not a physics result.",
      fontsize = 10, color = :gray30)
save(joinpath(OUTDIR, "testbed_summary.png"), fig)
@info "wrote $(joinpath(OUTDIR, "testbed_summary.png"))"

# 2. the three time series that carry the claims
fig = Figure(size = (1180, 420))
Label(fig[0, 1:3], "What each arm did to the three drifts", fontsize = 16, font = :bold)
series = [("dS 0-100 m  (seasonal, should NOT accumulate)", :s0100, "g/kg"),
          ("dS 200-400 m  (the runaway)", :s200400, "g/kg"),
          ("dT 100-200 m  (Winter Water warm bias)", :t100200, "degC")]
for (c, (title, key, unit)) in enumerate(series)
    nmax = maximum(length(R[a].labels) for a in PRESENT)
    ref = R[PRESENT[argmax([length(R[a].labels) for a in PRESENT])]].labels
    step = max(1, cld(nmax, 8))
    ax = Axis(fig[1, c]; title, titlesize = 11, xlabel = "month", ylabel = c == 1 ? unit : "",
              xticks = (1:step:nmax, ref[1:step:nmax]), xticklabelrotation = π / 4,
              xticklabelsize = 9)
    hlines!(ax, [0], color = :gray60, linewidth = 1)
    for (i, a) in enumerate(PRESENT)
        y = getproperty(R[a], key)
        lines!(ax, 1:length(y), y, color = armcolor(i), linewidth = a == "control" ? 3 : 1.8,
               linestyle = a == "control" ? :solid : :dash, label = a)
    end
    c == 3 && axislegend(ax; position = :lt, framevisible = false, labelsize = 9)
end
save(joinpath(OUTDIR, "testbed_drifts.png"), fig)
@info "wrote $(joinpath(OUTDIR, "testbed_drifts.png"))"

# 3. where the salinity error lives, one Hovmoller per arm, on ONE shared colour scale
fig = Figure(size = (300 * length(PRESENT) + 160, 420))
Label(fig[0, 1:length(PRESENT)],
      "Where the salinity error lives — model − B-SOSE, ice zone, by arm", fontsize = 16, font = :bold)
amp = maximum([maximum(filter(isfinite, abs.(vcat(R[a].dS...))); init = 0.0) for a in PRESENT])
amp = amp == 0 ? 0.1 : amp
# A Ref, not a bare global: `hm = ...` inside a top-level `for` creates a NEW LOCAL, so the
# Colorbar below would be handed `nothing` and die with a MethodError that names
# initialize_block! rather than the scope. Parses clean, fails at run time -- the usual trap.
hm = Ref{Any}(nothing)
for (c, a) in enumerate(PRESENT)
    M = reduce(hcat, R[a].dS)                    # (depth, month)
    lab = R[a].labels
    step = max(1, cld(length(lab), 5))
    ax = Axis(fig[1, c], title = a, titlesize = 12, xlabel = "month",
              ylabel = c == 1 ? "depth (m)" : "", yreversed = true,
              xticks = (1:step:length(lab), lab[1:step:length(lab)]),
              xticklabelrotation = π / 4, xticklabelsize = 9,
              yticklabelsvisible = c == 1)
    hm[] = heatmap!(ax, 1:length(lab), ZB, permutedims(M), colormap = :balance,
                    colorrange = (-amp, amp))
end
Colorbar(fig[1, length(PRESENT) + 1], hm[], label = "model − B-SOSE  (g/kg)")
save(joinpath(OUTDIR, "testbed_profiles.png"), fig)
@info "wrote $(joinpath(OUTDIR, "testbed_profiles.png"))"

println("\n  figures in $OUTDIR\n")
