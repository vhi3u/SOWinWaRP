# ==============================================================================================
# zonal_bias.jl -- is the 50-45S temperature bias made at the NORTHERN BOUNDARY, or in the interior?
#
#     BSOSE_DIR=$D julia --project=. tools/zonal_bias.jl RUN_NAME
#
# Zonal means of model - B-SOSE temperature and salinity, by latitude and depth, from the run's 30-day 3-D means
# (<RUN>_eke.nc) against B-SOSE's monthly Theta and Salt over the same windows (tools/bsose_windows.jl: each window
# the overlap-weighted blend of its calendar months). B-SOSE is taken at the nearest cell; the nest runs on
# B-SOSE's own levels, so levels are matched by depth (an error if any is more than 1 m off). Cells count only where
# both are wet. The west and east relaxation bands are left out of the zonal mean (SIDE_DEG, default 1°: the band is
# 2/3°), so what is left near 45S is the north face's doing alone.
#
# READ IT LIKE THIS (the NumericalEarth nest relaxes T and S toward monthly B-SOSE at 1/6 h over the outer 2/3°):
#   bias largest just inside the band and fading southward, through the water column  -> the north boundary
#   bias spread evenly over 50-45S and confined to the top ~100 m                     -> surface flux or mixing
#   bias following the fronts at depth, either sign, not anchored to 45S              -> front position
#   year 2 against year 1: a boundary-made bias appears fast and stays; interior drift grows.
#
# WRITES  OUTDIR/<RUN>_zonal_bias.png: latitude-depth ΔT and ΔS over the whole run, and ΔT, ΔS against latitude
#         for three layers, the relaxation band shaded. STDOUT: the same by distance from the north face, per year.
#
# ENV  DIR (run files, default .), OUTDIR (plots), BSOSE_DIR (required), SIDE_DEG (1.0), ZMAX (m drawn, 1500),
#      BAND_DEG (the north relaxation band shaded, 2/3). Runs on the login node: one window (~25 MB) and one
#      B-SOSE record (~150 MB at full width) at a time.
# ==============================================================================================

using NCDatasets, CairoMakie, Dates, Printf, Statistics

include(joinpath(@__DIR__, "bsose_windows.jl"))  # bsose_records, bsose_window_mean, bsose_path, BSOSE_SOURCE
include(joinpath(@__DIR__, "model_grid.jl"))     # file_levels, file_kbot

const RUN       = isempty(ARGS) ? get(ENV, "RUN", "") : ARGS[1]
isempty(RUN) && error("usage: BSOSE_DIR=\$D julia --project=. tools/zonal_bias.jl RUN_NAME")
const BSOSE_DIR = get(ENV, "BSOSE_DIR", "")
isempty(BSOSE_DIR) && error("set BSOSE_DIR to the directory with B-SOSE's monthly Theta and Salt")
BSOSE_SOURCE == "monthly" || error("zonal_bias.jl reads B-SOSE's monthly files; unset SOURCE")
const DIR      = get(ENV, "DIR", ".")
const OUTDIR   = get(ENV, "OUTDIR", "plots")
const SIDE_DEG = parse(Float64, get(ENV, "SIDE_DEG", "1.0"))
const ZMAX     = parse(Float64, get(ENV, "ZMAX", "1500"))
const BAND_DEG = parse(Float64, get(ENV, "BAND_DEG", string(2 / 3)))
const LAYERS   = ((0.0, 100.0), (100.0, 500.0), (500.0, 1500.0))
const DIST     = ((0.0, BAND_DEG), (BAND_DEG, 1.5), (1.5, 3.0), (3.0, 5.0), (5.0, 7.5), (7.5, 10.0))   # ° from the face

f64(A) = Float64.(coalesce.(A, NaN))

function main()
    path = joinpath(DIR, "$(RUN)_eke.nc")
    isfile(path) || error("no $path")
    eke = NCDataset(path)
    λ, φ = f64(eke["λ_caa"][:]), f64(eke["φ_aca"][:])
    zc, zf, _ = file_levels(eke); kbot = file_kbot(eke, zc, zf)[1]
    Nx, Ny, Nz = length(λ), length(φ), length(zc)
    dz = abs.(diff(zf))
    start = DateTime(string(eke.attrib["start_date"]))
    win = haskey(eke.attrib, "averaging_window_seconds") ? Float64(eke.attrib["averaging_window_seconds"]) : 30 * 86400.0
    tend = f64(eke["time"][:]); keepn = findall(t -> t >= 0.99win, tend)
    isempty(keepn) && error("$path has no complete 30-day window")
    wins = [(start + Millisecond(round(Int, 1000 * (tend[n] - win))), start + Millisecond(round(Int, 1000 * tend[n])), n) for n in keepn]

    # B-SOSE: nearest cells, levels matched by depth
    recT = bsose_records(BSOSE_DIR, "Theta", ("THETA", "Theta"), nothing)
    recS = bsose_records(BSOSE_DIR, "Salt", ("SALT", "Salt"), nothing)
    XC, YC, RC = NCDataset(bsose_path(BSOSE_DIR, "Theta")) do ds
        xc = f64(ds["XC"][:]); yc = f64(ds["YC"][:])
        rf = haskey(ds, "RF") ? f64(ds["RF"][:]) : [0.0; -cumsum(f64(ds[haskey(ds, "drF") ? "drF" : "DRF"][:]))]
        (ndims(xc) == 1 ? xc : xc[:, 1]), (ndims(yc) == 1 ? yc : yc[1, :]), [(rf[k] + rf[k+1]) / 2 for k in 1:length(rf)-1]
    end
    kb_of = [argmin(abs.(RC .- zc[k])) for k in 1:Nz]               # B-SOSE level of model level k
    bad = [k for k in 1:Nz if abs(RC[kb_of[k]] - zc[k]) > 1]
    isempty(bad) || error("model levels $(zc[bad]) have no B-SOSE level within 1 m: is this run on B-SOSE's levels?")
    nlev = maximum(kb_of)
    ib = [argmin(abs.(mod.(XC, 360) .- mod(x, 360))) for x in λ]
    jn = [argmin(abs.(YC .- y)) for y in φ]; jr = minimum(jn):maximum(jn); jj = jn .- first(jr) .+ 1

    # columns kept: not within SIDE_DEG of the west or east face
    keepi = findall(x -> x - λ[1] >= SIDE_DEG && λ[end] - x >= SIDE_DEG, λ)
    wet = [kbot[i, j] !== nothing && k >= kbot[i, j] for i in 1:Nx, j in 1:Ny, k in 1:Nz]

    nw = length(wins)
    dT, dS = fill(NaN, Ny, Nz, nw), fill(NaN, Ny, Nz, nw)          # zonal means per window
    rT = fill(NaN, Ny, Nz, nw)                                       # zonal RMS of ΔT per window
    for (q, (w0, w1, n)) in enumerate(wins)
        Bt = bsose_window_mean(recT, w0, w1, ib, jr, nlev)
        Bs = bsose_window_mean(recS, w0, w1, ib, jr, nlev)
        (Bt === nothing || Bs === nothing) && (@warn "B-SOSE does not cover $(Date(w0))-$(Date(w1)); skipped"; continue)
        Tm = f64(eke["T"][:, :, :, n]); Sm = f64(eke["S"][:, :, :, n])
        for j in 1:Ny, k in 1:Nz
            kb = kb_of[k]; a = Float64[]; b = Float64[]
            for i in keepi
                wet[i, j, k] || continue
                sb = Bs[i, jj[j], kb]; (sb == 0 || !isfinite(sb)) && continue
                t = Tm[i, j, k] - Bt[i, jj[j], kb]; s = Sm[i, j, k] - sb
                (isfinite(t) && isfinite(s)) || continue
                push!(a, t); push!(b, s)
            end
            isempty(a) && continue
            dT[j, k, q] = mean(a); dS[j, k, q] = mean(b); rT[j, k, q] = sqrt(mean(abs2, a))
        end
        @info @sprintf("window %d/%d %s to %s", q, nw, Date(w0), Date(w1))
    end
    close(eke)

    nanmean(v) = (w = filter(isfinite, v); isempty(w) ? NaN : mean(w))
    tmean(A, qs) = [nanmean(A[j, k, qs]) for j in 1:Ny, k in 1:Nz]
    allq = 1:nw
    y1 = [q for q in 1:nw if wins[q][2] <= start + Day(365)]
    y2 = [q for q in 1:nw if wins[q][1] >= start + Day(365)]
    dist = φ[end] .- φ .+ (φ[end] - φ[end-1]) / 2                     # ° south of the north face, row centres

    "Thickness-weighted mean over a layer of a (Ny, Nz) field, per row."
    function layer(A, (ztop, zbot))
        ks = [k for k in 1:Nz if ztop <= -zc[k] < zbot]
        [(v = [(A[j, k], dz[k]) for k in ks if isfinite(A[j, k])]; isempty(v) ? NaN : sum(first.(v) .* last.(v)) / sum(last.(v))) for j in 1:Ny]
    end
    dmean(prof, (a, b)) = nanmean([prof[j] for j in 1:Ny if a <= dist[j] < b])

    # ── stdout ─────────────────────────────────────────────────────────────────────────────────
    bar = "="^110
    println("\n", bar)
    println("ZONAL BIAS, model - B-SOSE: $RUN, $(Date(wins[1][1])) to $(Date(wins[end][2])), $(nw) 30-day windows")
    @printf("  zonal mean over %.1f-%.1fE (%.1f° off the west and east faces); %s\n", λ[first(keepi)], λ[last(keepi)], SIDE_DEG, bsose_source_note())
    println("  distance = ° south of the north face ($(round(φ[end] + (φ[end]-φ[end-1])/2, digits = 2))°); the first bin is the relaxation band")
    println(bar)
    for (name, A, unit, fmt) in (("ΔT", dT, "°C", "%7.3f"), ("ΔS", dS, "g/kg", "%7.3f"))
        println("\n  $name ($unit), thickness-weighted layer means")
        @printf("  %-22s %-7s", "south of the face", "period")
        for L in LAYERS; @printf(" | %9s", @sprintf("%.0f-%.0f m", L...)); end
        name == "ΔT" && @printf(" | %11s", "RMS 0-100 m")
        println()
        face = φ[end] + 0.5(φ[end] - φ[end-1])
        for d in DIST
            any(j -> d[1] <= dist[j] < d[2], 1:Ny) || continue
            label = @sprintf("%.1f-%.1f° (%.1f-%.1fS)", d[1], d[2], -(face - d[1]), -(face - d[2]))
            for (pname, qs) in (("all", allq), ("year 1", y1), ("year 2", y2))
                isempty(qs) && continue
                @printf("  %-22s %-7s", pname == "all" ? label : "", pname)
                M = tmean(A, qs)
                for L in LAYERS; @printf(" | %9.3f", dmean(layer(M, L), d)); end
                name == "ΔT" && @printf(" | %11.3f", dmean(layer(tmean(rT, qs), LAYERS[1]), d))
                println()
            end
        end
    end
    println()

    # ── figure ─────────────────────────────────────────────────────────────────────────────────
    mkpath(OUTDIR)
    MT, MS = tmean(dT, allq), tmean(dS, allq)
    kz = reverse([k for k in 1:Nz if -zc[k] <= ZMAX])                 # top down: depth increasing
    lim(A) = (v = filter(isfinite, abs.(A[:, kz])); isempty(v) ? 1.0 : max(quantile(v, 0.99), 1e-6))
    band_lo = φ[end] + 0.5(φ[end] - φ[end-1]) - BAND_DEG
    fig = Figure(size = (1400, 900), fontsize = 13)
    Label(fig[0, 1:4], "$RUN: zonal-mean model - B-SOSE, $(Date(wins[1][1])) to $(Date(wins[end][2])), $(round(λ[first(keepi)], digits = 1))-$(round(λ[last(keepi)], digits = 1))E"; font = :bold)
    for (c, (A, name, unit)) in enumerate(((MT, "ΔT", "°C"), (MS, "ΔS", "g/kg")))
        l = lim(A)
        ax = Axis(fig[1, 2c-1]; title = "$name, whole run", xlabel = "latitude", ylabel = "depth (m)", yreversed = true)
        hm = heatmap!(ax, φ, -zc[kz], A[:, kz]; colormap = :balance, colorrange = (-l, l), nan_color = :gray80)
        vspan!(ax, band_lo, φ[end] + 0.5(φ[end] - φ[end-1]); color = (:black, 0.08))
        Colorbar(fig[1, 2c], hm; label = unit)
        ax2 = Axis(fig[2, 2c-1]; title = "$name by layer (shaded: north relaxation band)", xlabel = "latitude", ylabel = unit)
        vspan!(ax2, band_lo, φ[end] + 0.5(φ[end] - φ[end-1]); color = (:black, 0.08))
        hlines!(ax2, 0; color = :gray50, linewidth = 1)
        for (L, col) in zip(LAYERS, (:black, :darkorange, :steelblue))
            lines!(ax2, φ, layer(A, L); color = col, linewidth = 2, label = @sprintf("%.0f-%.0f m", L...))
        end
        axislegend(ax2; position = :lb, backgroundcolor = (:white, 0.8))
    end
    out = joinpath(OUTDIR, "$(RUN)_zonal_bias.png")
    save(out, fig); @info "wrote $out"
end

main()
