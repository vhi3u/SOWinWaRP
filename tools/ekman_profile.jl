# ==============================================================================================
# ekman_profile.jl -- is the run's weak surface drift too little wind momentum, or the same momentum mixed deeper?
#
#     BSOSE_DIR=/storage/scratch1/1/vnguyen480/SOWinWaRP/data julia --project=. tools/ekman_profile.jl RUN
#
# WHY. tools/ice_drift_drivers.jl on _iceobc2_mitconc (Jun-Oct, 60S): the run's top-cell ocean moves north at
# 0.4 cm/s, B-SOSE's at 2.7; its ice at 0.5 against 5.2 cm/s, under the same ERA5 wind (B-SOSE's open-water oceTAU is
# the ERA5 bulk stress to within 10%). A weak top cell is either too little momentum (stress) or the right momentum
# spread over more depth (mixing). The depth-integrated wind-driven transport tells them apart:
#     same transport, weaker top cell   -> mixing (CATKE carries the wind's momentum deeper)
#     smaller transport                 -> stress (less momentum gets in)
#
# WHAT, per line (EKMAN_LINES, default 63S and 60S), month and column class:
#   v'(z) = v(z) - v_ref, v_ref the mean v over REF_TOP-REF_BOT (default 200-300 m), so the depth-independent
#   (geostrophic, barotropic) part of the meridional flow drops out and the wind-driven part is left;
#   V_E = ∫ v' dz over 0-REF_TOP (m²/s, + north), the Ekman transport per metre of line;
#   v'₀ the top cell's v' (cm/s); D = V_E / v'₀ (m), the depth the transport is spread over;
#   V_th = -τx / (ρ₀ f) from B-SOSE's own oceTAUX over the same faces and month: what its stress implies.
#   run: <RUN>_eke.nc 30-day means of u, v at cell centres (each window to the month of its centre);
#   B-SOSE: monthly Vvel (on its YG face nearest the line) and the 5-day oceTAUX, SIArea.
#   The run and B-SOSE share their 52 levels, so level k is the same depth in both.
# CLASSES, per column and month: "open" = run monthly mean ℵ < 0.05 AND B-SOSE SIArea < 0.05; "ice" = both > 0.8;
#   "all" = every column. Columns wet in both below REF_BOT, SIDE_DEG (1°) away from the west and east faces.
#
# WRITES OUTDIR/<RUN>_ekman_profile.png: mean v'(z) over the WINTER months, run against B-SOSE, per line and class;
# and V_E month by month with V_th. STDOUT: the monthly table and the WINTER means with run/B-SOSE ratios.
# ENV  DIR (.), OUTDIR (plots), BSOSE_DIR (required), EKMAN_LINES (-63,-60), WINTER (6,7,8,9,10), REF_TOP (200),
#      REF_BOT (300), SIDE_DEG (1.0), RHO0 (1027)
# ==============================================================================================

using NCDatasets, CairoMakie, Dates, Printf, Statistics

include(joinpath(@__DIR__, "model_grid.jl"))     # file_levels

ENV["HDF5_USE_FILE_LOCKING"] = "FALSE"

const RUN = isempty(ARGS) ? get(ENV, "RUN", "") : ARGS[1]
isempty(RUN) && error("usage: BSOSE_DIR=... julia --project=. tools/ekman_profile.jl RUN")
const BSOSE_DIR = get(ENV, "BSOSE_DIR", "")
isempty(BSOSE_DIR) && error("set BSOSE_DIR")
const DIR      = get(ENV, "DIR", ".")
const OUTDIR   = get(ENV, "OUTDIR", "plots")
const LINES    = sort(parse.(Float64, split(get(ENV, "EKMAN_LINES", "-63,-60"), ',')))
const WINTER   = parse.(Int, split(get(ENV, "WINTER", "6,7,8,9,10"), ','))
const REF_TOP  = parse(Float64, get(ENV, "REF_TOP", "200"))
const REF_BOT  = parse(Float64, get(ENV, "REF_BOT", "300"))
const SIDE_DEG = parse(Float64, get(ENV, "SIDE_DEG", "1.0"))
const ρ₀       = parse(Float64, get(ENV, "RHO0", "1027"))
const Ω        = 7.2921e-5
const CLASSES  = (:all, :open, :ice)

g(A) = Float64.(coalesce.(A, NaN))
nanmean(x) = (y = filter(isfinite, x); isempty(y) ? NaN : mean(y))
ym_of(d) = (year(d), month(d))
label(ym) = @sprintf("%04d-%02d", ym...)
lname(L) = @sprintf("%.0fS", -L)

function bfile(token, tag)
    want = lowercase("$(token)_bsoseI156_$(tag).nc")
    hits = filter(f -> lowercase(f) == want, readdir(BSOSE_DIR))
    isempty(hits) ? "" : joinpath(BSOSE_DIR, first(hits))
end
pick(d, names) = d[first(k for k in names if haskey(d, k))]

function main()
    # ── the run: grid, levels, months ───────────────────────────────────────────────────────────
    ice_path, eke_path = joinpath(DIR, "$(RUN)_sea_ice.nc"), joinpath(DIR, "$(RUN)_eke.nc")
    isfile(ice_path) || error("no $ice_path"); isfile(eke_path) || error("no $eke_path")
    di = NCDataset(ice_path)
    λ = g(di["λ_caa"][:]); φ = g(di["φ_aca"][:]); Nx = length(λ)
    start = DateTime(string(di.attrib["start_date"]))
    bottom = g(di["bottom_height"][:, :])
    Δφ = φ[2] - φ[1]; φS = φ[1] - Δφ / 2
    jm = [round(Int, (L - φS) / Δφ) + 1 for L in LINES]                 # run y-face j: south face of cell j
    φm = [φS + (j - 1) * Δφ for j in jm]
    datetime(t) = start + Millisecond(round(Int, 1000t))

    de = NCDataset(eke_path)
    zc, zf, _ = file_levels(de)
    Nz = length(zc)
    Δz = reverse(diff(zf))                                               # top first
    ztop = cumsum(Δz) .- Δz ./ 2                                         # cell-centre depth, top first (m, +down)
    kE  = findall(d -> d <= REF_TOP, ztop)                               # the Ekman layer's levels
    kR  = findall(d -> REF_TOP < d <= REF_BOT, ztop)                     # the reference levels
    (isempty(kE) || isempty(kR)) && error("no levels in 0-$REF_TOP or $REF_TOP-$REF_BOT m")
    K = maximum(kR)
    cols = [i for i in 1:Nx if λ[i] - λ[1] >= SIDE_DEG && λ[end] - λ[i] >= SIDE_DEG]
    deep(j) = [bottom[i, j-1] < -REF_BOT - 10 && bottom[i, j] < -REF_BOT - 10 for i in 1:Nx]

    # run ℵ: daily snapshots to calendar-month means at the line (the two rows either side)
    ts = g(di["time"][:])
    months = unique(ym_of.(datetime.(ts[2:end])))
    kof = Dict(ym => k for (k, ym) in enumerate(months)); nm = length(months)
    aR = [zeros(Nx) for _ in LINES, _ in 1:nm]; nR = zeros(Int, length(LINES), nm)
    for n in 2:length(ts)
        k = kof[ym_of(datetime(ts[n]))]
        for (l, j) in enumerate(jm)
            A = g(di["ℵ"][:, j-1:j, n]); A[.!isfinite.(A)] .= 0
            aR[l, k] .+= (A[:, 1] .+ A[:, 2]) ./ 2; nR[l, k] += 1
        end
    end
    for l in eachindex(LINES), k in 1:nm; nR[l, k] > 0 && (aR[l, k] ./= nR[l, k]); end
    close(di)

    # run v profiles: 30-day windows to the month of their centre (top-first levels)
    vR = [fill(NaN, Nx, K) for _ in LINES, _ in 1:nm]; cR = zeros(Int, length(LINES), nm)
    te = g(de["time"][:])
    win = haskey(de.attrib, "averaging_window_seconds") ? Float64(de.attrib["averaging_window_seconds"]) : 30 * 86400.0
    for (n, t) in enumerate(te)
        t < 0.99win && continue
        k = get(kof, ym_of(datetime(t - win / 2)), 0); k == 0 && continue
        for (l, j) in enumerate(jm)
            V = g(de["v"][:, j-1:j, Nz-K+1:Nz, n])                     # bottom-to-top levels
            P = reverse((V[:, 1, :] .+ V[:, 2, :]) ./ 2; dims = 2)       # (Nx, K), top first
            if cR[l, k] == 0
                vR[l, k] = P
            else
                vR[l, k] = (vR[l, k] .* cR[l, k] .+ P) ./ (cR[l, k] + 1)
            end
            cR[l, k] += 1
        end
    end
    close(de)

    # ── B-SOSE: monthly Vvel on the YG face nearest each line; 5-day SIArea and oceTAUX to months ──
    geom = joinpath(BSOSE_DIR, "surfTflx_bsoseI156_2013to2024_monthly.nc")
    xc, yc, depth = NCDataset(geom) do d
        x = g(d["XC"][:]); y = g(d["YC"][:])
        (ndims(x) == 1 ? x : x[:, 1]), (ndims(y) == 1 ? y : y[1, :]), g(d["Depth"][:, :])
    end
    Ir = [argmin(abs.(xc .- x)) for x in λ]                              # B-SOSE's columns are the run's centres
    yface = [(yc[j-1] + yc[j]) / 2 for j in 2:length(yc)]
    jb = [1 + argmin(abs.(yface .- L)) for L in LINES]                   # YG row = south face of YC row jb
    vB = [fill(NaN, Nx, K) for _ in LINES, _ in 1:nm]
    pV = bfile("Vvel", "2013to2024_monthly"); isempty(pV) && error("no Vvel_bsoseI156_2013to2024_monthly.nc in $BSOSE_DIR")
    NCDataset(pV) do dv
        Vo = pick(dv, ("VVEL", "Vvel"))
        for (n, s) in enumerate(DateTime.(dv["time"][:]))
            k = get(kof, ym_of(s - Day(15)), 0); k == 0 && continue
            for (l, j) in enumerate(jb)
                vB[l, k] = g(Vo[minimum(Ir):maximum(Ir), j, 1:K, n])[Ir .- minimum(Ir) .+ 1, :]
            end
        end
    end
    aB = [zeros(Nx) for _ in LINES, _ in 1:nm]; τB = [zeros(Nx) for _ in LINES, _ in 1:nm]; nB = zeros(Int, length(LINES), nm)
    for y in sort(unique(first.(months)))
        pa, pt = bfile("SIArea", "$(y)_5day"), bfile("oceTAUX", "$(y)_5day")
        (isempty(pa) || isempty(pt)) && (@warn "B-SOSE $y: SIArea or oceTAUX 5-day file missing; that year has no classes or V_th"; continue)
        NCDataset(pa) do da; NCDataset(pt) do dt
            Av = pick(da, ("SIarea", "SIArea", "SIAREA")); Tx = pick(dt, ("oceTAUX", "OCETAUX"))
            for (n, s) in enumerate(DateTime.(da["time"][:]))
                k = get(kof, ym_of(s - Hour(60)), 0); k == 0 && continue
                r = minimum(Ir):maximum(Ir); sel = Ir .- minimum(Ir) .+ 1
                for (l, j) in enumerate(jb)
                    A = g(Av[r, j-1:j, n])[sel, :]; A[.!isfinite.(A)] .= 0
                    T = g(Tx[r, j-1:j, n])[sel, :]
                    aB[l, k] .+= (A[:, 1] .+ A[:, 2]) ./ 2
                    τB[l, k] .+= (T[:, 1] .+ T[:, 2]) ./ 2
                    nB[l, k] += 1
                end
            end
        end end
    end
    for l in eachindex(LINES), k in 1:nm
        nB[l, k] > 0 ? (aB[l, k] ./= nB[l, k]; τB[l, k] ./= nB[l, k]) : (aB[l, k] .= NaN; τB[l, k] .= NaN)
    end

    # ── per line, month, class: v' profile, V_E, top v', V_th ─────────────────────────────────────
    res = Dict{Tuple{Int, Int, Symbol}, NamedTuple}()
    prof(P, sel) = begin
        ok = sel .& [all(isfinite, P[i, :]) for i in 1:Nx]
        count(ok) == 0 && return nothing
        m = vec(mean(P[ok, :]; dims = 1))
        m .- mean(m[kR]), count(ok)
    end
    for (l, L) in enumerate(LINES), k in 1:nm
        cR[l, k] == 0 && continue
        wetboth = deep(jm[l]) .& [depth[Ir[i], jb[l]-1] > REF_BOT && depth[Ir[i], jb[l]] > REF_BOT for i in 1:Nx]
        base = falses(Nx); base[cols] .= true; base .&= wetboth
        f = 2Ω * sind(φm[l])
        for cls in CLASSES
            sel = cls === :all ? base :
                  cls === :open ? base .& (aR[l, k] .< 0.05) .& (aB[l, k] .< 0.05) :
                                  base .& (aR[l, k] .> 0.8) .& (aB[l, k] .> 0.8)
            pr = prof(vR[l, k], sel); pb = prof(vB[l, k], sel)
            (pr === nothing || pb === nothing) && continue
            VE(p) = sum(p[1][kE] .* Δz[kE])
            τ = nanmean(τB[l, k][sel])
            res[(l, k, cls)] = (; pr = pr[1], pb = pb[1], n = min(pr[2], pb[2]),
                                VEr = VE(pr), VEb = VE(pb), v0r = pr[1][1], v0b = pb[1][1], Vth = -τ / (ρ₀ * f))
        end
    end

    # ── stdout ─────────────────────────────────────────────────────────────────────────────────
    bar = "="^128
    println("\n", bar)
    println("WIND-DRIVEN MERIDIONAL FLOW, $RUN against B-SOSE: v' = v - mean v over $(Int(REF_TOP))-$(Int(REF_BOT)) m")
    println("  V_E = ∫ v' dz over 0-$(Int(REF_TOP)) m (m²/s, + north); v'₀ top cell (cm/s); D = V_E / v'₀ (m); V_th = -τx/(ρ₀ f) from B-SOSE's oceTAUX")
    println("  classes: all columns; open (run ℵ and B-SOSE SIArea < 0.05); ice (both > 0.8). n = columns used")
    println(bar)
    for (l, L) in enumerate(LINES), cls in CLASSES
        @printf("\n  %s, %s  (run face %.3f, B-SOSE YG row %d)\n", lname(L), cls, φm[l], jb[l])
        println("  month    |    n |  V_E run  B-SOSE    V_th |  v'₀ run  B-SOSE |   D run  B-SOSE")
        for k in 1:nm
            r = get(res, (l, k, cls), nothing); r === nothing && continue
            @printf("  %s  | %4d |  %7.3f  %6.3f  %6.3f |  %7.2f  %6.2f |  %6.0f  %6.0f\n", label(months[k]), r.n,
                    r.VEr, r.VEb, r.Vth, 100r.v0r, 100r.v0b, r.VEr / r.v0r, r.VEb / r.v0b)
        end
        wk = [k for k in 1:nm if months[k][2] in WINTER && haskey(res, (l, k, cls))]
        isempty(wk) && continue
        m(f) = mean(f(res[(l, k, cls)]) for k in wk)
        @printf("  months %s (%d): V_E run %.3f B-SOSE %.3f V_th %.3f (run/B-SOSE %.2f) | v'₀ run %.2f B-SOSE %.2f cm/s (ratio %.2f)\n",
                join(WINTER, ','), length(wk), m(r -> r.VEr), m(r -> r.VEb), m(r -> r.Vth), m(r -> r.VEr) / m(r -> r.VEb),
                100m(r -> r.v0r), 100m(r -> r.v0b), m(r -> r.v0r) / m(r -> r.v0b))
    end
    println("\n  Reading it: V_E ratio ≈ 1 with v'₀ ratio << 1 -> the run spreads the same momentum deeper (mixing);")
    println("  V_E ratio << 1 -> less momentum gets in (stress, ice-ocean drag). V_th says what B-SOSE's own stress should drive.\n")

    # ── figure ─────────────────────────────────────────────────────────────────────────────────
    mkpath(OUTDIR)
    fig = Figure(size = (1500, 380 * length(LINES) + 120), fontsize = 13)
    Label(fig[0, 1:4], "$RUN: wind-driven meridional flow v' = v - v($(Int(REF_TOP))-$(Int(REF_BOT)) m), months $(join(WINTER, ','))"; font = :bold)
    zz = -ztop[1:K]
    for (l, L) in enumerate(LINES)
        for (c, cls) in enumerate(CLASSES)
            ax = Axis(fig[l, c]; title = "$(lname(L)), $cls", xlabel = "v' (cm/s)", ylabel = c == 1 ? "depth (m)" : "")
            vlines!(ax, [0]; color = :grey70)
            wk = [k for k in 1:nm if months[k][2] in WINTER && haskey(res, (l, k, cls))]
            isempty(wk) && (text!(ax, 0.5, 0.5; text = "no $cls columns in these months", space = :relative, align = (:center, :center)); continue)
            lines!(ax, 100 .* mean(res[(l, k, cls)].pr for k in wk), zz; color = :royalblue, linewidth = 2.5, label = "run")
            lines!(ax, 100 .* mean(res[(l, k, cls)].pb for k in wk), zz; color = :black, linewidth = 2.5, linestyle = :dash, label = "B-SOSE")
            l == 1 && c == 1 && axislegend(ax; position = :rb)
        end
        ax = Axis(fig[l, 4]; title = "$(lname(L)), all: V_E by month", ylabel = "m²/s",
                  xticks = (1:3:nm, label.(months)[1:3:end]), xticklabelrotation = π/6)
        hlines!(ax, [0]; color = :grey70)
        ks = [k for k in 1:nm if haskey(res, (l, k, :all))]
        lines!(ax, ks, [res[(l, k, :all)].VEr for k in ks]; color = :royalblue, linewidth = 2, label = "run")
        lines!(ax, ks, [res[(l, k, :all)].VEb for k in ks]; color = :black, linewidth = 2, linestyle = :dash, label = "B-SOSE")
        lines!(ax, ks, [res[(l, k, :all)].Vth for k in ks]; color = :darkorange, linewidth = 1.5, label = "-τx/(ρ₀f), B-SOSE τ")
        l == 1 && axislegend(ax; position = :lt, labelsize = 10)
    end
    out = joinpath(OUTDIR, "$(RUN)_ekman_profile.png"); save(out, fig); @info "wrote $out"
end

main()
