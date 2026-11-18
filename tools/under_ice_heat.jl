# ==============================================================================================
# under_ice_heat.jl -- how much heat the water under the ice holds above freezing, run against
# B-SOSE, by latitude band and 30-day window, and at what depth the difference sits.
#
#     BSOSE_DIR=/storage/scratch1/1/vnguyen480/SOWinWaRP/data julia --project=. tools/under_ice_heat.jl 2Y_nested_bsose_ggl_hj_pr207_mcphee
#
# WHY. 2Y_nested_bsose_ggl_hj_pr207_mcphee took MITgcm's ocean-to-ice heat transfer (8.75e-4 m/s, ~46x
# NumericalEarth's default) and its ice volume fell to ~75% of B-SOSE's at the peaks and ~25% at the
# minima: the ocean melted 646 km³ in the first month alone. B-SOSE's own coupling is that strong, so
# its water under the ice must sit near freezing; ours, it seems, does not. This says how far above
# freezing each side's water is, from the first window on, and whether the excess is at the surface
# (the initial state, the freezing point) or below it (mixed up from the warm deep water).
#
# WHERE. Both sides are compared on the run's own grid, at the cells under B-SOSE's ice: B-SOSE's
# 30-day mean SIarea >= ICE_MIN. So both are measured where the ice SHOULD be, whether or not the
# run has ice there (its own mean ℵ there is printed beside B-SOSE's). B-SOSE is taken at the nearest
# B-SOSE cell (no interpolation; 1/6° both, B-SOSE's latitudes Mercator-spaced); the run is on
# B-SOSE's 52 levels, so the levels match one for one (checked). Cells count where both are wet.
#
# WHEN. The run's <RUN>_eke.nc holds its only 3-D T and S: 30-day means, stamped at each window's end.
# B-SOSE is the mean of its 5-day Theta, Salt and SIarea records whose centres (stamp - 2.5 days) fall
# in the same window. The run's ℵ is the mean of its daily <RUN>_sea_ice.nc snapshots in the window.
#
# FREEZING. Each side against ITS OWN model's surface freezing point, the one its ice melts against:
#     run     ClimaSeaIce 0.7.2 LinearLiquidus:  Tf = -0.0542 S   (its depth term left out)
#     B-SOSE  MITgcm pkg/seaice defaults:        Tf = 0.0901 - 0.0575 S
# Ours is ~0.02 °C warmer at S = 34: B-SOSE water at its freezing point is below ours. The offset at
# each band's salinity is printed.
#
# PRINTS, per band and window: the cells; mean ℵ, run and B-SOSE; the top cell's T - Tf; the heat above
# freezing ρ₀cₚ∫(T - Tf)dz over 0-HC_DEPTH m (columns at least that deep, wet in both), MJ/m², run and
# B-SOSE, and the difference split at SPLIT m (above / below), with the difference as the ice it
# could melt (1 MJ/m² melts 0.33 cm).
# WRITES OUTDIR/<RUN>_under_ice_heat.png: per band, the 0-HC_DEPTH heat of both, and run - B-SOSE
#   (T - Tf) against depth and time to PROFILE_ZMAX.
#
# LOGIN-NODE SAFE: NCDatasets and CairoMakie only; B-SOSE is read one record at a time, the box and the
# top PROFILE_ZMAX metres only.
#
# ENV: DIR (run files, default .), OUTDIR (plots), BSOSE_DIR (required), START (default: the file's
#      start_date), BANDS (-70,-65,-60,-55), ICE_MIN (0.15), HC_DEPTH (100), SPLIT (40),
#      PROFILE_ZMAX (300), MIN_RECORDS (B-SOSE records per window, default 4)
# ==============================================================================================

using NCDatasets, CairoMakie, Dates, Printf, Statistics

ENV["HDF5_USE_FILE_LOCKING"] = "FALSE"   # safe on a run that is still writing
include(joinpath(@__DIR__, "model_grid.jl"))   # file_levels, file_kbot

const RUN = isempty(ARGS) ? get(ENV, "RUN", "") : ARGS[1]
isempty(RUN) && error("give the run name: julia --project=. tools/under_ice_heat.jl <RUN>")
const DIR          = get(ENV, "DIR", ".")
const OUTDIR       = get(ENV, "OUTDIR", "plots")
const BSOSE_DIR    = get(ENV, "BSOSE_DIR", "")
isempty(BSOSE_DIR) && error("set BSOSE_DIR to the directory with B-SOSE's 5-day Theta, Salt and SIarea files")
const EDGES        = parse.(Float64, split(get(ENV, "BANDS", "-70,-65,-60,-55"), ','))
const ICE_MIN      = parse(Float64, get(ENV, "ICE_MIN", "0.15"))
const HC_DEPTH     = parse(Float64, get(ENV, "HC_DEPTH", "100"))
const SPLIT        = parse(Float64, get(ENV, "SPLIT", "40"))
const PROFILE_ZMAX = parse(Float64, get(ENV, "PROFILE_ZMAX", "300"))
const MIN_RECORDS  = parse(Int, get(ENV, "MIN_RECORDS", "4"))
mkpath(OUTDIR)

const ρ₀, cₚ = 1026.0, 3991.9            # J/m³/K: ρ₀cₚ ≈ 4.1e6
const ρᵢL    = 900.0 * 334e3             # J/m³ of ice
Tf_run(S)   = -0.0542 * S
Tf_bsose(S) = 0.0901 - 0.0575 * S

f64(A) = Float64.(coalesce.(A, NaN))
coord(ds, name, dim) = (v = f64(ds[name][:]); ndims(v) == 1 ? v : dim == 1 ? v[:, 1] : v[1, :])
first_var(ds, names, path) = (found = filter(n -> haskey(ds, n), names);
                              isempty(found) ? error("none of $(names) in $path") : first(found))
band_name(b) = @sprintf("%.0f-%.0fS", -EDGES[b], -EDGES[b+1])
label(d) = Dates.format(d, "yyyy-mm-dd")

"The file `<token>_bsoseI156_<year>_5day.nc` in BSOSE_DIR, matching the token whatever its case."
function bsose_file(token, y)
    want = lowercase("$(token)_bsoseI156_$(y)_5day.nc")
    hits = filter(f -> lowercase(f) == want, readdir(BSOSE_DIR))
    isempty(hits) && error("no $(token)_bsoseI156_$(y)_5day.nc (any case) in $BSOSE_DIR")
    return joinpath(BSOSE_DIR, first(hits))
end

"Every 5-day record of a token over the given years: (path, variable, record, window centre)."
function bsose_records(token, names, years)
    recs = Tuple{String, String, Int, DateTime}[]
    for y in years
        path = bsose_file(token, y)
        NCDataset(path) do ds
            var = first_var(ds, names, path)
            for (n, stamp) in enumerate(ds["time"][:])
                push!(recs, (path, var, n, DateTime(stamp) - Hour(60)))
            end
        end
    end
    return recs
end

function main()
    # ── the run: grid, levels, 30-day T and S windows ─────────────────────────────────────────
    eke_path, ice_path = joinpath(DIR, "$(RUN)_eke.nc"), joinpath(DIR, "$(RUN)_sea_ice.nc")
    isfile(eke_path) || error("$eke_path not found; set DIR to the run directory")
    isfile(ice_path) || error("$ice_path not found; set DIR to the run directory")

    eke = NCDataset(eke_path)
    λ, φ = f64(eke["λ_caa"][:]), f64(eke["φ_aca"][:])
    Nx, Ny = length(λ), length(φ)
    lv = file_levels(eke); lv === nothing && error("$eke_path carries no vertical levels")
    zc, zf, _ = lv
    Nz = length(zc)
    fk = file_kbot(eke, zc, zf); fk === nothing && error("$eke_path carries no sea floor")
    kbot = fk[1]
    start = haskey(ENV, "START") ? DateTime(ENV["START"]) : DateTime(string(eke.attrib["start_date"]))
    win = haskey(eke.attrib, "averaging_window_seconds") ? Float64(eke.attrib["averaging_window_seconds"]) : 30 * 86400.0
    tend = f64(eke["time"][:])
    windows = [(start + Millisecond(round(Int, 1000 * (t - win))), start + Millisecond(round(Int, 1000t)))
               for t in tend if t >= 0.99win]
    keepn = findall(t -> t >= 0.99win, tend)

    # Levels top-down, to PROFILE_ZMAX; their thickness, and which make up 0-HC_DEPTH and 0-SPLIT
    ks = [k for k in Nz:-1:1 if -zc[k] <= PROFILE_ZMAX]
    nk = length(ks)
    depth = [-zc[k] for k in ks]
    Δz = [zf[k+1] - zf[k] for k in ks]
    in_hc = [-zf[k] <= HC_DEPTH + 1e-6 for k in ks]
    upper = [-zf[k] <= SPLIT + 1e-6 for k in ks]
    abs(sum(Δz[in_hc]) - HC_DEPTH) > 0.5 &&
        @warn @sprintf("the levels make 0-%.1f m, not 0-%.0f m (HC_DEPTH)", sum(Δz[in_hc]), HC_DEPTH)
    # Wet in the run, top-down: Oceananigans' k counts up from the floor
    wet_run = [kbot[i, j] !== nothing && ks[m] >= kbot[i, j] for i in 1:Nx, j in 1:Ny, m in 1:nk]

    ice = NCDataset(ice_path)
    tice = f64(ice["time"][:])
    ℵvar = ice["ℵ"]
    function run_ice(w0, w1)
        idx = findall(t -> w0 < start + Millisecond(round(Int, 1000t)) <= w1, tice)
        isempty(idx) && return fill(NaN, Nx, Ny)
        A = f64(ndims(ℵvar) == 4 ? ℵvar[:, :, 1, idx] : ℵvar[:, :, idx])
        return dropdims(mean(replace(A, NaN => 0.0); dims = 3); dims = 3)
    end

    # ── B-SOSE: 5-day records, nearest cell to each run cell, matching levels ────────────────
    years = year(first(windows)[1]):year(last(windows)[2])
    Trecs = bsose_records("Theta", ("THETA", "Theta"), years)
    Srecs = bsose_records("Salt", ("SALT", "Salt"), years)
    Arecs = bsose_records("SIarea", ("SIarea", "SIArea", "SIAREA"), years)
    [r[4] for r in Trecs] == [r[4] for r in Srecs] || error("B-SOSE Theta and Salt hold different 5-day records")

    XC, YC, RF = NCDataset(Trecs[1][1]) do ds
        rf = haskey(ds, "RF") ? f64(ds["RF"][:]) : [0.0; -cumsum(f64(ds[haskey(ds, "drF") ? "drF" : "DRF"][:]))]
        coord(ds, "XC", 1), coord(ds, "YC", 2), rf
    end
    ix = [argmin(abs.(XC .- x)) for x in λ]
    jy = [argmin(abs.(YC .- y)) for y in φ]
    ir, jr = minimum(ix):maximum(ix), minimum(jy):maximum(jy)
    il, jl = ix .- first(ir) .+ 1, jy .- first(jr) .+ 1
    dB = [-(RF[k] + RF[k+1]) / 2 for k in 1:nk]
    maximum(abs.(dB .- depth)) < 0.05 ||
        error("B-SOSE's top $nk level centres differ from the run's: $(round.(dB .- depth, digits = 2))")
    @info @sprintf("%s: %dx%d, %d levels to %.0f m; B-SOSE cells %d:%d x %d:%d; %d windows %s to %s",
                   RUN, Nx, Ny, nk, depth[end], first(ir), last(ir), first(jr), last(jr), length(windows),
                   label(first(windows)[1]), label(last(windows)[2]))

    function bsose_mean(recs, w0, w1, levels)
        sel = filter(r -> w0 < r[4] <= w1, recs)
        length(sel) < MIN_RECORDS && return nothing
        acc = nothing
        for (path, var, n, _) in sel
            A = NCDataset(path) do ds
                v = ds[var]
                f64(levels === nothing ? (ndims(v) == 4 ? v[ir, jr, 1, n] : v[ir, jr, n]) : v[ir, jr, 1:levels, n])
            end
            acc = acc === nothing ? A : acc .+ A
        end
        return acc ./ length(sel)
    end

    # ── per window ───────────────────────────────────────────────────────────────────────────
    nb, nw = length(EDGES) - 1, length(windows)
    band_of = [findfirst(b -> EDGES[b] <= φ[j] < EDGES[b+1], 1:nb) for j in 1:Ny]
    wlat = cosd.(φ)
    keys_ = (:cells, :ℵ_run, :ℵ_bsose, :sst_run, :sst_bsose, :hc_run, :hc_bsose, :Δhc_upper, :Δhc_lower, :S_bsose)
    res = Dict(k => fill(NaN, nb, nw) for k in keys_)
    prof_run, prof_bsose = fill(NaN, nb, nw, nk), fill(NaN, nb, nw, nk)

    for (w, (w0, w1)) in enumerate(windows)
        Tb = bsose_mean(Trecs, w0, w1, nk); Sb = bsose_mean(Srecs, w0, w1, nk); Ab = bsose_mean(Arecs, w0, w1, nothing)
        if Tb === nothing || Sb === nothing || Ab === nothing
            @warn "$(label(w0)) to $(label(w1)): fewer than $MIN_RECORDS B-SOSE records, skipped"
            continue
        end
        Tb[Sb .== 0] .= NaN; Sb[Sb .== 0] .= NaN               # B-SOSE land is zero
        Tb, Sb, Ab = Tb[il, jl, :], Sb[il, jl, :], Ab[il, jl]  # onto the run's cells
        n = keepn[w]
        lo = minimum(ks)
        Tr = f64(eke["T"][:, :, lo:Nz, n])[:, :, end:-1:1][:, :, 1:nk]
        Sr = f64(eke["S"][:, :, lo:Nz, n])[:, :, end:-1:1][:, :, 1:nk]
        Tr[.!wet_run] .= NaN; Sr[.!wet_run] .= NaN
        Xr = Tr .- Tf_run.(Sr)                                 # above freezing, °C
        Xb = Tb .- Tf_bsose.(Sb)
        ℵr = run_ice(w0, w1)

        for b in 1:nb
            cells = [(i, j) for i in 1:Nx, j in 1:Ny if band_of[j] == b && Ab[i, j] >= ICE_MIN &&
                                                        isfinite(Xr[i, j, 1]) && isfinite(Xb[i, j, 1])]
            isempty(cells) && continue
            W = [wlat[j] for (i, j) in cells]
            wm = f -> sum(W .* [f(i, j) for (i, j) in cells]) / sum(W)
            res[:cells][b, w] = length(cells)
            res[:ℵ_run][b, w]     = wm((i, j) -> ℵr[i, j])
            res[:ℵ_bsose][b, w]   = wm((i, j) -> Ab[i, j])
            res[:sst_run][b, w]   = wm((i, j) -> Xr[i, j, 1])
            res[:sst_bsose][b, w] = wm((i, j) -> Xb[i, j, 1])
            res[:S_bsose][b, w]   = wm((i, j) -> Sb[i, j, 1])
            # Heat above freezing over 0-HC_DEPTH: columns that deep and wet in both
            deep = [(i, j) for (i, j) in cells if all(m -> isfinite(Xr[i, j, m]) && isfinite(Xb[i, j, m]), findall(in_hc))]
            if !isempty(deep)
                Wd = [wlat[j] for (i, j) in deep]
                hc = (X, i, j, sel) -> ρ₀ * cₚ * sum(X[i, j, m] * Δz[m] for m in findall(sel)) / 1e6   # MJ/m²
                dm = f -> sum(Wd .* [f(i, j) for (i, j) in deep]) / sum(Wd)
                res[:hc_run][b, w]   = dm((i, j) -> hc(Xr, i, j, in_hc))
                res[:hc_bsose][b, w] = dm((i, j) -> hc(Xb, i, j, in_hc))
                res[:Δhc_upper][b, w] = dm((i, j) -> hc(Xr, i, j, in_hc .& upper) - hc(Xb, i, j, in_hc .& upper))
                res[:Δhc_lower][b, w] = dm((i, j) -> hc(Xr, i, j, in_hc .& .!upper) - hc(Xb, i, j, in_hc .& .!upper))
            end
            for m in 1:nk
                both = [(i, j) for (i, j) in cells if isfinite(Xr[i, j, m]) && isfinite(Xb[i, j, m])]
                isempty(both) && continue
                Wm = [wlat[j] for (i, j) in both]
                prof_run[b, w, m]   = sum(Wm .* [Xr[i, j, m] for (i, j) in both]) / sum(Wm)
                prof_bsose[b, w, m] = sum(Wm .* [Xb[i, j, m] for (i, j) in both]) / sum(Wm)
            end
        end
        @info "window $w/$nw ($(label(w1))) done"
    end
    close(eke); close(ice)

    # ── report ───────────────────────────────────────────────────────────────────────────────
    centre(w) = windows[w][1] + (windows[w][2] - windows[w][1]) ÷ 2
    bar = "="^128
    println("\n", bar)
    println("HEAT ABOVE FREEZING UNDER B-SOSE'S ICE (30-day mean SIarea >= $ICE_MIN), $RUN against B-SOSE")
    println("  T - Tf, each against its own model's freezing point; heat = ρ₀cₚ∫(T - Tf)dz over 0-$(Int(HC_DEPTH)) m, MJ/m²")
    println("  Δ = run - B-SOSE, split at $(Int(SPLIT)) m; 1 MJ/m² melts $(round(1e6 / ρᵢL * 100, digits = 2)) cm of ice")
    println(bar)
    for b in 1:nb
        all(isnan, res[:cells][b, :]) && continue
        Sm = mean(filter(isfinite, res[:S_bsose][b, :]))
        @printf("\n  %s   (freezing points at S = %.2f: run %.3f, B-SOSE %.3f °C)\n", band_name(b), Sm, Tf_run(Sm), Tf_bsose(Sm))
        println("  window centre | cells |  ℵ run  B-SOSE | top-cell T-Tf run  B-SOSE |   heat run  B-SOSE |  Δ 0-$(Int(SPLIT))  Δ $(Int(SPLIT))-$(Int(HC_DEPTH))   Δ as ice")
        for w in 1:nw
            isnan(res[:cells][b, w]) && continue
            Δ = res[:hc_run][b, w] - res[:hc_bsose][b, w]
            @printf("  %s    | %5d |  %5.2f   %5.2f |         %6.3f  %6.3f |  %6.2f   %6.2f |   %+6.2f    %+6.2f   %+6.1f cm\n",
                    label(centre(w)), res[:cells][b, w], res[:ℵ_run][b, w], res[:ℵ_bsose][b, w],
                    res[:sst_run][b, w], res[:sst_bsose][b, w], res[:hc_run][b, w], res[:hc_bsose][b, w],
                    res[:Δhc_upper][b, w], res[:Δhc_lower][b, w], Δ * 1e6 / ρᵢL * 100)
        end
    end

    # ── figure ───────────────────────────────────────────────────────────────────────────────
    bands = [b for b in 1:nb if !all(isnan, res[:cells][b, :])]
    isempty(bands) && return
    x = [Dates.value(centre(w) - start) / 86_400_000 for w in 1:nw]
    ticks = (x[1:3:end], [Dates.format(centre(w), "yyyy-mm") for w in 1:3:nw])
    Δprof = prof_run .- prof_bsose
    lim = let v = filter(isfinite, abs.(Δprof)); isempty(v) ? 1.0 : max(quantile(v, 0.98), 0.05) end
    fig = Figure(size = (1500, 380 * length(bands)))
    Label(fig[0, 1:3], "$RUN: water under B-SOSE's ice, above each model's freezing point"; fontsize = 18, font = :bold)
    local hm
    for (r, b) in enumerate(bands)
        ax1 = Axis(fig[r, 1]; title = "$(band_name(b)): heat above freezing, 0-$(Int(HC_DEPTH)) m",
                   ylabel = "MJ/m²", xticks = ticks, xticklabelrotation = π / 6)
        lines!(ax1, x, res[:hc_run][b, :]; color = :royalblue, linewidth = 2, label = "run")
        lines!(ax1, x, res[:hc_bsose][b, :]; color = :black, linestyle = :dash, linewidth = 2, label = "B-SOSE")
        r == 1 && axislegend(ax1; position = :lc, backgroundcolor = (:white, 0.8))
        ax2 = Axis(fig[r, 2]; title = "$(band_name(b)): run - B-SOSE, T - Tf (°C)", ylabel = "depth (m)",
                   xticks = ticks, xticklabelrotation = π / 6, yreversed = true)
        hm = heatmap!(ax2, x, depth, Δprof[b, :, :]; colormap = :balance, colorrange = (-lim, lim))
    end
    Colorbar(fig[1:length(bands), 3], hm; label = "Δ(T - Tf), °C")
    out = joinpath(OUTDIR, "$(RUN)_under_ice_heat.png")
    save(out, fig)
    @info "wrote $out"
end

main()
