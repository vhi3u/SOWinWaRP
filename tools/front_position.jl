# ==============================================================================================
# front_position.jl -- where are the ACC fronts in the run, in B-SOSE and in the altimetry, month by month?
#
#     BSOSE_DIR=/storage/scratch1/1/vnguyen480/SOWinWaRP/data julia --project=. tools/front_position.jl RUN_NAME
#
# FRONTS. The CTOH weekly fronts (Sallée et al. 2008 method on Copernicus ADT; in BSOSE_DIR):
#     PF   Polar Front                  ADT = -0.3 m   CTOH_PolarFront_weekly_1993_2025.nc (PF)
#     SAF  Subantarctic Front           ADT = -0.1 m   CTOH_SubantarcticFront_weekly_1993_2025.nc (SAF)
#     NSAF northern branch of the SAF   ADT =  0.0 m   CTOH_NorthernSubantarcticFront_weekly_1993_2025.nc (SAF_N)
# The same contours are traced in B-SOSE's monthly ETAN and in the run's daily η (<RUN>_free_surface.nc) averaged to
# calendar months. At each longitude the front is the mean latitude of the contour's crossings between LAT_S and LAT_N;
# a month's front is the median over longitudes. The 1° next to the west and east faces is left out (SIDE_DEG), for
# all three sources alike.
#
# DATUM. ETAN is ADT up to a constant. The constant (OFFSET, ADT = ETAN + OFFSET) is fitted ONCE, so that B-SOSE's
# three mean fronts sit on CTOH's over the months both cover; one number for three fronts, so B-SOSE's front
# SPACING is still tested. Set OFFSET= to fix it instead. The run starts from ETAN and its Flather boundary pulls
# toward ETAN, so it shares the datum; its domain-mean drift against B-SOSE is removed month by month (printed).
#
# STRENGTH. Δη = zonal-mean η at STRENGTH_N minus at STRENGTH_S (default 47S and 55S): the surface geostrophic
# transport across the SAF band. Independent of the datum.
#
# WRITES  OUTDIR/<RUN>_fronts.png: each front's latitude month by month (CTOH, B-SOSE, run), the run-mean front lines
#         against longitude, and Δη. STDOUT: the monthly table and the mean offsets, whole run, year 1, year 2.
#
# ENV  DIR (.), OUTDIR (plots), BSOSE_DIR (required), SIDE_DEG (1.0), LAT_S (-62), LAT_N (-45.5), OFFSET (fitted),
#      STRENGTH_S (-55), STRENGTH_N (-47), MIN_DAYS (20, days of run output for a month to count).
# ==============================================================================================

using NCDatasets, CairoMakie, Dates, Printf, Statistics

const RUN        = isempty(ARGS) ? get(ENV, "RUN", "") : ARGS[1]
isempty(RUN) && error("usage: BSOSE_DIR=... julia --project=. tools/front_position.jl RUN_NAME")
const BSOSE_DIR  = get(ENV, "BSOSE_DIR", "")
isempty(BSOSE_DIR) && error("set BSOSE_DIR to the directory with SSH_bsoseI156_2013to2024_monthly.nc and the CTOH fronts")
const DIR        = get(ENV, "DIR", ".")
const OUTDIR     = get(ENV, "OUTDIR", "plots")
const SIDE_DEG   = parse(Float64, get(ENV, "SIDE_DEG", "1.0"))
const LAT_S      = parse(Float64, get(ENV, "LAT_S", "-62"))
const LAT_N      = parse(Float64, get(ENV, "LAT_N", "-45.5"))
const STRENGTH_S = parse(Float64, get(ENV, "STRENGTH_S", "-55"))
const STRENGTH_N = parse(Float64, get(ENV, "STRENGTH_N", "-47"))
const MIN_DAYS   = parse(Int, get(ENV, "MIN_DAYS", "20"))
const FRONTS     = (("PF", -0.3, "CTOH_PolarFront_weekly_1993_2025.nc", "PF"),
                    ("SAF", -0.1, "CTOH_SubantarcticFront_weekly_1993_2025.nc", "SAF"),
                    ("NSAF", 0.0, "CTOH_NorthernSubantarcticFront_weekly_1993_2025.nc", "SAF_N"))

f64(A) = Float64.(coalesce.(A, NaN))
nanmedian(v) = (w = filter(isfinite, v); isempty(w) ? NaN : median(w))
nanmean(v) = (w = filter(isfinite, v); isempty(w) ? NaN : mean(w))
monthkey(d) = (year(d), month(d))

"Mean latitude of the crossings of `v` in one column (η along lat, ascending), between LAT_S and LAT_N; NaN if none."
function crossing_lat(η, lat, v)
    s, n = 0.0, 0
    for j in 1:length(lat)-1
        a, b = η[j], η[j+1]
        (isfinite(a) && isfinite(b)) || continue
        (LAT_S <= lat[j] && lat[j+1] <= LAT_N) || continue
        if (a - v) * (b - v) < 0 || (a == v)
            s += lat[j] + (v - a) / (b - a) * (lat[j+1] - lat[j]); n += 1
        end
    end
    return n == 0 ? NaN : s / n
end

"Per-longitude front latitudes of contour v on a (lon, lat) field, lat ascending."
front_line(H, lat, v) = [crossing_lat(view(H, i, :), lat, v) for i in axes(H, 1)]

"Zonal-mean η at the row nearest each latitude: Δη = η(STRENGTH_N) - η(STRENGTH_S)."
function strength(H, lat)
    jn, js = argmin(abs.(lat .- STRENGTH_N)), argmin(abs.(lat .- STRENGTH_S))
    return nanmean(H[:, jn]) - nanmean(H[:, js])
end

function main()
    # ── the run: daily η to calendar-month means ───────────────────────────────────────────────
    path = joinpath(DIR, "$(RUN)_free_surface.nc")
    isfile(path) || error("no $path")
    fs = NCDataset(path)
    λ, φ = f64(fs["λ_caa"][:]), f64(fs["φ_aca"][:])
    start = DateTime(string(fs.attrib["start_date"]))
    tsec = f64(fs["time"][:])
    days = [start + Millisecond(round(Int, 1000s)) for s in tsec]
    landmask = if haskey(fs, "bottom_height")
        b = f64(fs["bottom_height"][:, :]); .!(b .< 0)
    else
        sp = joinpath(DIR, "$(RUN)_surface.nc")
        isfile(sp) && NCDataset(d -> haskey(d, "bottom_height"), sp) ?
            NCDataset(d -> .!(f64(d["bottom_height"][:, :]) .< 0), sp) : nothing
    end
    keepi = findall(x -> x - λ[1] >= SIDE_DEG && λ[end] - x >= SIDE_DEG, λ)
    rd(n) = (A = fs["η"]; f64(ndims(A) == 4 ? A[:, :, 1, n] : A[:, :, n]))
    months = unique(monthkey.(days[2:end]))                             # the t = 0 snapshot left out
    runm = Dict{Tuple{Int, Int}, Matrix{Float64}}()
    for mk in months
        ns = [n for n in 2:length(days) if monthkey(days[n]) == mk]
        length(ns) < MIN_DAYS && continue
        acc = zeros(length(λ), length(φ))
        for n in ns; acc .+= rd(n); end
        H = acc ./ length(ns)
        landmask === nothing ? (H[H .== 0] .= NaN) : (H[landmask] .= NaN)
        runm[mk] = H
    end
    close(fs)
    isempty(runm) && error("no calendar month with $MIN_DAYS days of output in $path")
    rmonths = sort(collect(keys(runm)))
    @info "run: $(length(rmonths)) months, $(rmonths[1]) to $(rmonths[end])"

    # ── B-SOSE monthly ETAN on the same columns (nearest cells), and its own months ────────────
    bpath = joinpath(BSOSE_DIR, "SSH_bsoseI156_2013to2024_monthly.nc")
    isfile(bpath) || error("no $bpath")
    bs = NCDataset(bpath)
    XC, YC = mod.(f64(bs["XC"][:]), 360), f64(bs["YC"][:])
    depth = f64(bs["Depth"][:, :])
    bmk = [monthkey(DateTime(s) - Day(14)) for s in bs["time"][:]]       # stamps sit at month ends
    ib = [argmin(abs.(XC .- mod(x, 360))) for x in λ]
    jb = [argmin(abs.(YC .- y)) for y in φ]; jr = minimum(jb):maximum(jb)
    bsose_on_run(n) = (A = f64(bs["ETAN"][:, jr, n]); H = A[ib, jb .- first(jr) .+ 1];
                       H[depth[ib, jb] .<= 0] .= NaN; H)

    # ── CTOH: per calendar month, the median latitude over 91-149E of every weekly record in it ───
    lon_lo, lon_hi = λ[first(keepi)], λ[last(keepi)]
    ctoh = Dict{String, Dict{Tuple{Int, Int}, Float64}}()
    for (name, _, file, var) in FRONTS
        p = joinpath(BSOSE_DIR, file); isfile(p) || error("no $p")
        NCDataset(p) do d
            t = DateTime.(d["time"][:]); acc = Dict{Tuple{Int, Int}, Vector{Float64}}()
            for n in eachindex(t)
                year(t[n]) < 2013 && continue
                lo = f64(d["longitude"][:, n]); la = f64(d[var][:, n])
                sel = [k for k in eachindex(lo) if isfinite(lo[k]) && isfinite(la[k]) && lon_lo <= mod(lo[k], 360) <= lon_hi]
                isempty(sel) && continue
                push!(get!(acc, monthkey(t[n]), Float64[]), median(la[sel]))
            end
            ctoh[name] = Dict(k => mean(v) for (k, v) in acc)
        end
    end

    # ── the datum: one offset putting B-SOSE's mean fronts on CTOH's over the months both cover ──
    common = [k for k in bmk if all(haskey(ctoh[f[1]], k) for f in FRONTS)]
    isempty(common) && error("B-SOSE and CTOH share no month")
    Hmean = zeros(length(λ), length(φ))
    for k in common; Hmean .+= bsose_on_run(findfirst(==(k), bmk)); end
    Hmean ./= length(common)
    keepH = Hmean[keepi, :]
    ctoh_mean = [mean(ctoh[f[1]][k] for k in common) for f in FRONTS]
    bfront(c) = [nanmedian(front_line(keepH, φ, f[2] - c)) for f in FRONTS]
    offset = if haskey(ENV, "OFFSET")
        parse(Float64, ENV["OFFSET"])
    else
        cs = -2.0:0.005:2.0
        cost = [(b = bfront(c); all(isfinite, b) ? sum(abs2, b .- ctoh_mean) : Inf) for c in cs]
        isfinite(minimum(cost)) || error("no offset puts all three B-SOSE fronts between $LAT_S and $LAT_N; set OFFSET")
        cs[argmin(cost)]
    end
    bfit = bfront(offset)

    # ── month by month ─────────────────────────────────────────────────────────────────────────
    nf = length(FRONTS)
    rows = NamedTuple[]
    for mk in rmonths
        nb = findfirst(==(mk), bmk); nb === nothing && continue
        Hb = bsose_on_run(nb); Hr = runm[mk]
        both = isfinite.(Hb) .& isfinite.(Hr)
        drift = mean(Hr[both]) - mean(Hb[both])                    # the run's domain-mean η against B-SOSE's
        Hr = Hr .- drift
        b = [nanmedian(front_line(Hb[keepi, :], φ, f[2] - offset)) for f in FRONTS]
        r = [nanmedian(front_line(Hr[keepi, :], φ, f[2] - offset)) for f in FRONTS]
        c = [get(ctoh[f[1]], mk, NaN) for f in FRONTS]
        push!(rows, (; mk, c, b, r, drift, sb = strength(Hb[keepi, :], φ), sr = strength(Hr[keepi, :], φ),
                       lb = [front_line(Hb[keepi, :], φ, f[2] - offset) for f in FRONTS],
                       lr = [front_line(Hr[keepi, :], φ, f[2] - offset) for f in FRONTS]))
    end
    close(bs)
    isempty(rows) && error("B-SOSE has none of the run's months")

    # ── stdout ─────────────────────────────────────────────────────────────────────────────────
    bar = "="^118
    println("\n", bar)
    println("ACC FRONTS: $RUN against B-SOSE and CTOH altimetry, $(lon_lo)-$(lon_hi)E, contours crossed between $(LAT_S) and $(LAT_N)")
    @printf("  datum: ADT = ETAN %+.3f m (%s); B-SOSE's mean fronts over %d months then sit at", offset,
            haskey(ENV, "OFFSET") ? "OFFSET" : "fitted", length(common))
    for (q, f) in enumerate(FRONTS); @printf("  %s %.2f (CTOH %.2f)", f[1], bfit[q], ctoh_mean[q]); end
    println("\n  latitude °N (negative = south); Δη = zonal-mean η at $(STRENGTH_N) minus $(STRENGTH_S), m; drift = run's domain-mean η - B-SOSE's")
    println(bar)
    @printf("  %-8s", "month")
    for f in FRONTS; @printf(" | %-6s %7s %7s %7s", f[1], "CTOH", "B-SOSE", "run"); end
    @printf(" | %6s %6s | %6s\n", "Δη B", "Δη run", "drift")
    for r in rows
        @printf("  %d-%02d  ", r.mk...)
        for q in 1:nf; @printf(" | %-6s %7.2f %7.2f %7.2f", "", r.c[q], r.b[q], r.r[q]); end
        @printf(" | %6.3f %6.3f | %+6.3f\n", r.sb, r.sr, r.drift)
    end
    y1 = [r for r in rows if DateTime(r.mk..., 1) < start + Day(365)]
    y2 = [r for r in rows if DateTime(r.mk..., 1) >= start + Day(365)]
    println("\n  means, ° (positive = the run's front NORTH of the other's)")
    @printf("  %-8s", "")
    for f in FRONTS; @printf(" | %-4s %12s %12s", f[1], "run-B-SOSE", "run-CTOH"); end
    @printf(" | %12s\n", "Δη run/B-SOSE")
    for (name, set) in (("all", rows), ("year 1", y1), ("year 2", y2))
        isempty(set) && continue
        @printf("  %-8s", name)
        for q in 1:nf
            @printf(" | %-4s %12.2f %12.2f", "", nanmean([r.r[q] - r.b[q] for r in set]), nanmean([r.r[q] - r.c[q] for r in set]))
        end
        @printf(" | %12.2f\n", nanmean([r.sr for r in set]) / nanmean([r.sb for r in set]))
    end
    println()

    # ── figure ─────────────────────────────────────────────────────────────────────────────────
    mkpath(OUTDIR)
    x = [DateTime(r.mk..., 15) for r in rows]
    xt = Dates.value.(x .- x[1]) ./ 86_400_000
    ticks = (xt[1:3:end], [Dates.format(d, "u yy") for d in x[1:3:end]])
    fig = Figure(size = (1400, 1000), fontsize = 13)
    Label(fig[0, 1:2], "$RUN: ACC fronts from sea surface height, $(round(lon_lo, digits = 1))-$(round(lon_hi, digits = 1))E"; font = :bold)
    for (q, f) in enumerate(FRONTS)
        ax = Axis(fig[q, 1]; title = "$(f[1]) (ADT $(f[2]) m)", ylabel = "latitude", xticks = ticks)
        for (v, lab, col) in (([r.c[q] for r in rows], "CTOH altimetry", :black),
                              ([r.b[q] for r in rows], "B-SOSE", :darkorange),
                              ([r.r[q] for r in rows], "run", :steelblue))
            lines!(ax, xt, v; color = col, linewidth = 2, label = lab)
        end
        q == 1 && axislegend(ax; position = :lt, backgroundcolor = (:white, 0.8))
    end
    axm = Axis(fig[1:2, 2]; title = "mean front lines over the run (solid B-SOSE, dashed run)", xlabel = "longitude", ylabel = "latitude")
    for (q, (f, col)) in enumerate(zip(FRONTS, (:purple, :darkorange, :teal)))
        lines!(axm, λ[keepi], [nanmean([r.lb[q][i] for r in rows]) for i in eachindex(keepi)]; color = col, linewidth = 2, label = f[1])
        lines!(axm, λ[keepi], [nanmean([r.lr[q][i] for r in rows]) for i in eachindex(keepi)]; color = col, linewidth = 2, linestyle = :dash)
    end
    axislegend(axm; position = :rb, backgroundcolor = (:white, 0.8))
    axs = Axis(fig[3, 2]; title = "Δη = η($(STRENGTH_N)) - η($(STRENGTH_S)), zonal mean", ylabel = "m", xticks = ticks)
    lines!(axs, xt, [r.sb for r in rows]; color = :darkorange, linewidth = 2, label = "B-SOSE")
    lines!(axs, xt, [r.sr for r in rows]; color = :steelblue, linewidth = 2, label = "run")
    axislegend(axs; position = :lb, backgroundcolor = (:white, 0.8))
    out = joinpath(OUTDIR, "$(RUN)_fronts.png")
    save(out, fig); @info "wrote $out"
end

main()
