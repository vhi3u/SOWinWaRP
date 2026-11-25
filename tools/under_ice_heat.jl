# ==============================================================================================
# under_ice_heat.jl -- how much heat the water under the ice holds above freezing, run against
# B-SOSE, by latitude band and 30-day window, and at what depth the difference sits.
#
#     BSOSE_DIR=/storage/scratch1/1/vnguyen480/SOWinWaRP/data julia --project=. tools/under_ice_heat.jl 2Y_nested_bsose_ggl_hj_pr207_mcphee
#     SOURCE=5day BSOSE_DIR=... julia --project=. tools/under_ice_heat.jl <RUN>     # the 5-day files, as before 2026-11-22
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
# B-SOSE (bsose_windows.jl, SOURCE): monthly (default) = its Theta, Salt and SIArea *_2013to2024_monthly.nc,
# each window the overlap-weighted blend of its calendar months; 5day = the mean of its 5-day records whose
# centres (stamp - 2.5 days) fall in the window. The run's ℵ is the mean of its daily <RUN>_sea_ice.nc
# snapshots in the window.
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
# could melt (1 MJ/m² melts 0.33 cm). Then, over the columns wet to DEEP_DEPTH in both (a subset: the
# shelf drops out), the 0-HC_DEPTH difference again and the HC_DEPTH-DEEP_DEPTH heat of both and their
# difference: a run short above HC_DEPTH but warm below it keeps its deep heat down (too little mixing);
# short in both lacks the heat itself (sides, or drift from the initial state).
# STRATIFICATION, on the same deep columns: the density step from the top cell down to the
# HC_DEPTH-DEEP_DEPTH layer mean, linearised: Δσ = ∂ρ/∂T·ΔT + ∂ρ/∂S·ΔS, with the
# TEOS-10 sensitivities (SeawaterPolynomials) at each column's mid T, S and mid depth, so the run - B-SOSE
# difference splits exactly into a temperature part and a salinity part. Larger = harder for winter
# mixing to reach the warm water below; the top-cell and deep salinities of both are printed with it.
# B-SOSE's θ and practical S go through the same TEOS-10 formula: the ~0.16 g/kg Sᴬ - Sₚ offset is
# common to both layers and drops out of ΔS.
# OVERTURNING, on the same deep columns (Gülk et al. 2024 eqs. 8-9, after Martinson & Iannuzzi 1998 and
# Campbell et al. 2019): how much it takes to mix each column down to the HC_DEPTH-DEEP_DEPTH layer.
#   MLD  first level below MLD_REF (10 m) whose σ₀ exceeds σ₀(MLD_REF) by MLD_DSIGMA (0.03 kg/m³); the
#        level's centre depth, DEEP_DEPTH if none (same levels in both, so comparable one for one)
#   SD*  salt deficit, m of ice: MLD × [(S_d - S_ml) - (α/β)(θ_d - T_f)] / DSI, with the mixed layer
#        cooled to its freezing point, d the deep-layer mean, α, β TEOS-10 at the midpoint and DSI
#        (30 g/kg) the ocean-minus-ice salinity. The ice the column must grow to overturn; set it against
#        the brine in tools/model_surface_buoyancy.jl and the growth in tools/ice_growth_budget.jl.
#   CR   convective resistance, g/ρ₀ ∫₀ᴴ (σ₀(H) - σ₀(z)) dz to the bottom of the deep layer, m²/s²: the
#        buoyancy loss to overturn to H, temperature included as it stands.
# WRITES OUTDIR/<RUN>_under_ice_heat.png (SOURCE=5day) or <RUN>_under_ice_heat_monthly.png: per band, the 0-HC_DEPTH heat of both; run - B-SOSE heat by
#   layer (0-SPLIT, SPLIT-HC_DEPTH, HC_DEPTH-DEEP_DEPTH, the last on the deep columns); and run - B-SOSE
#   (T - Tf) against depth and time to PROFILE_ZMAX.
#
# LOGIN-NODE SAFE: NCDatasets and CairoMakie only; B-SOSE is read one record at a time, the box's rows at
# full width (contiguous on disk) and the top PROFILE_ZMAX metres only.
#
# ENV: DIR (run files, default .), OUTDIR (plots), BSOSE_DIR (required), SOURCE (monthly | 5day), START (default: the file's
#      start_date), BANDS (-70,-65,-60,-55), ICE_MIN (0.15), HC_DEPTH (100), SPLIT (40),
#      DEEP_DEPTH (default: the bottom of the deepest level within PROFILE_ZMAX), PROFILE_ZMAX (300), MIN_RECORDS (5-day records per window, default 4),
#      MLD_REF (10), MLD_DSIGMA (0.03), DSI (30)
# ==============================================================================================

using NCDatasets, CairoMakie, Dates, Printf, Statistics
using SeawaterPolynomials: thermal_sensitivity, haline_sensitivity
import SeawaterPolynomials
using SeawaterPolynomials.TEOS10: TEOS10EquationOfState

ENV["HDF5_USE_FILE_LOCKING"] = "FALSE"   # safe on a run that is still writing
include(joinpath(@__DIR__, "model_grid.jl"))   # file_levels, file_kbot
include(joinpath(@__DIR__, "bsose_windows.jl")) # BSOSE_SOURCE, bsose_records, bsose_window_mean

const RUN = isempty(ARGS) ? get(ENV, "RUN", "") : ARGS[1]
isempty(RUN) && error("give the run name: julia --project=. tools/under_ice_heat.jl <RUN>")
const DIR          = get(ENV, "DIR", ".")
const OUTDIR       = get(ENV, "OUTDIR", "plots")
const BSOSE_DIR    = get(ENV, "BSOSE_DIR", "")
isempty(BSOSE_DIR) && error("set BSOSE_DIR to the directory with B-SOSE's Theta, Salt and SIArea files")
const EDGES        = parse.(Float64, split(get(ENV, "BANDS", "-70,-65,-60,-55"), ','))
const ICE_MIN      = parse(Float64, get(ENV, "ICE_MIN", "0.15"))
const HC_DEPTH     = parse(Float64, get(ENV, "HC_DEPTH", "100"))
const SPLIT        = parse(Float64, get(ENV, "SPLIT", "40"))
const PROFILE_ZMAX = parse(Float64, get(ENV, "PROFILE_ZMAX", "300"))
const MIN_RECORDS  = parse(Int, get(ENV, "MIN_RECORDS", "4"))
const MLD_REF      = parse(Float64, get(ENV, "MLD_REF", "10"))
const MLD_DSIGMA   = parse(Float64, get(ENV, "MLD_DSIGMA", "0.03"))
const DSI          = parse(Float64, get(ENV, "DSI", "30"))
mkpath(OUTDIR)

const ρ₀, cₚ = 1026.0, 3991.9            # J/m³/K: ρ₀cₚ ≈ 4.1e6
const ρᵢL    = 900.0 * 334e3             # J/m³ of ice
Tf_run(S)   = -0.0542 * S
Tf_bsose(S) = 0.0901 - 0.0575 * S
const EOS = TEOS10EquationOfState()

"Density step (kg/m³) from (T₁, S₁) at depth d₁ down to (T₂, S₂) at d₂, linearised about the midpoint: (T part, S part)."
function density_step(T₁, S₁, d₁, T₂, S₂, d₂)
    T, S, Z = (T₁ + T₂) / 2, (S₁ + S₂) / 2, -(d₁ + d₂) / 2
    return (-thermal_sensitivity(T, S, Z, EOS) * (T₂ - T₁), haline_sensitivity(T, S, Z, EOS) * (S₂ - S₁))
end

σ₀(T, S) = SeawaterPolynomials.ρ(T, S, 0.0, EOS)

"""
Overturning measures for one column, levels top-down (T, S, depth, Δz to the deep layer's last level `ml`,
the deep layer's levels `md`, the freezing point `Tf`): (MLD m, salt deficit SD* m of ice, CR m²/s²).
"""
function overturning(T, S, depth, Δz, md, kref, Tf)
    ml = last(md)
    σ = [σ₀(T[m], S[m]) for m in 1:ml]
    k = findfirst(m -> m > kref && σ[m] - σ[kref] > MLD_DSIGMA, 1:ml)
    mld = k === nothing ? depth[ml] + Δz[ml] / 2 : depth[k]
    mix = [m for m in 1:ml if depth[m] <= mld]
    isempty(mix) && (mix = [1])
    Sml = sum(S[m] * Δz[m] for m in mix) / sum(Δz[m] for m in mix)
    Td = sum(T[m] * Δz[m] for m in md) / sum(Δz[m] for m in md)
    Sd = sum(S[m] * Δz[m] for m in md) / sum(Δz[m] for m in md)
    dd = sum(depth[m] * Δz[m] for m in md) / sum(Δz[m] for m in md)
    T₀ = Tf(Sml)
    Tm, Sm, Zm = (T₀ + Td) / 2, (Sml + Sd) / 2, -dd / 2
    α, β = thermal_sensitivity(Tm, Sm, Zm, EOS), haline_sensitivity(Tm, Sm, Zm, EOS)
    sd = mld * ((Sd - Sml) - α / β * (Td - T₀)) / DSI
    cr = 9.81 / ρ₀ * sum((σ[ml] - σ[m]) * Δz[m] for m in 1:ml)
    return (mld, sd, cr)
end

f64(A) = Float64.(coalesce.(A, NaN))
coord(ds, name, dim) = (v = f64(ds[name][:]); ndims(v) == 1 ? v : dim == 1 ? v[:, 1] : v[1, :])
band_name(b) = @sprintf("%.0f-%.0fS", -EDGES[b], -EDGES[b+1])
label(d) = Dates.format(d, "yyyy-mm-dd")

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
    deep_depth = haskey(ENV, "DEEP_DEPTH") ? parse(Float64, ENV["DEEP_DEPTH"]) : -zf[ks[end]]
    in_deep = [!in_hc[m] && -zf[ks[m]] <= deep_depth + 1e-6 for m in 1:nk]
    kref = argmin(abs.(depth .- MLD_REF))             # the MLD's reference level
    any(in_deep) || error(@sprintf("no level between HC_DEPTH %.0f m and DEEP_DEPTH %.0f m", HC_DEPTH, deep_depth))
    deep_bot = -zf[ks[findlast(in_deep)]]
    abs(deep_bot - deep_depth) > 0.5 &&
        @warn @sprintf("the levels make %.0f-%.1f m, not %.0f-%.0f m (DEEP_DEPTH)", HC_DEPTH, deep_bot, HC_DEPTH, deep_depth)
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

    # ── B-SOSE: monthly or 5-day records, nearest cell to each run cell, matching levels ─────
    # 5-day: a year counts only if its Theta, Salt and SIArea files all open (a truncated download drops that
    # year's windows with a warning). Monthly: a missing or unreadable file is an error naming it.
    years = bsose_usable_years(BSOSE_DIR, ("Theta", "Salt", "SIArea"), year(first(windows)[1]):year(last(windows)[2]))
    Trecs = bsose_records(BSOSE_DIR, "Theta", ("THETA", "Theta"), years)
    Srecs = bsose_records(BSOSE_DIR, "Salt", ("SALT", "Salt"), years)
    Arecs = bsose_records(BSOSE_DIR, "SIArea", ("SIarea", "SIArea", "SIAREA"), years)
    [(r.t0, r.t1) for r in Trecs] == [(r.t0, r.t1) for r in Srecs] || error("B-SOSE Theta and Salt hold different records")

    XC, YC, RF = NCDataset(Trecs[1].path) do ds
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
    @info "SOURCE=$BSOSE_SOURCE: $(bsose_source_note())"
    bsose_mean(recs, w0, w1, levels) = bsose_window_mean(recs, w0, w1, ir, jr, levels; min_records = MIN_RECORDS)

    # ── per window ───────────────────────────────────────────────────────────────────────────
    nb, nw = length(EDGES) - 1, length(windows)
    band_of = [findfirst(b -> EDGES[b] <= φ[j] < EDGES[b+1], 1:nb) for j in 1:Ny]
    wlat = cosd.(φ)
    keys_ = (:cells, :ℵ_run, :ℵ_bsose, :sst_run, :sst_bsose, :hc_run, :hc_bsose, :Δhc_upper, :Δhc_lower, :S_bsose,
             :deep_cells, :Δhc_sub, :dhc_run, :dhc_bsose,
             :σ_run, :σ_bsose, :Δσ_T, :Δσ_S, :Stop_run, :Stop_bsose, :Sdeep_run, :Sdeep_bsose,
             :mld_run, :mld_bsose, :sd_run, :sd_bsose, :cr_run, :cr_bsose)
    res = Dict(k => fill(NaN, nb, nw) for k in keys_)
    prof_run, prof_bsose = fill(NaN, nb, nw, nk), fill(NaN, nb, nw, nk)

    for (w, (w0, w1)) in enumerate(windows)
        Tb = bsose_mean(Trecs, w0, w1, nk); Sb = bsose_mean(Srecs, w0, w1, nk); Ab = bsose_mean(Arecs, w0, w1, nothing)
        if Tb === nothing || Sb === nothing || Ab === nothing
            @warn "$(label(w0)) to $(label(w1)): B-SOSE does not cover this window ($BSOSE_SOURCE), skipped"
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
                # Below HC_DEPTH: the columns wet to DEEP_DEPTH in both, with their own 0-HC_DEPTH difference
                both_deep = [(i, j) for (i, j) in deep if all(m -> isfinite(Xr[i, j, m]) && isfinite(Xb[i, j, m]), findall(in_deep))]
                if !isempty(both_deep)
                    Wb = [wlat[j] for (i, j) in both_deep]
                    bm = f -> sum(Wb .* [f(i, j) for (i, j) in both_deep]) / sum(Wb)
                    res[:deep_cells][b, w] = length(both_deep)
                    res[:Δhc_sub][b, w]   = bm((i, j) -> hc(Xr, i, j, in_hc) - hc(Xb, i, j, in_hc))
                    res[:dhc_run][b, w]   = bm((i, j) -> hc(Xr, i, j, in_deep))
                    res[:dhc_bsose][b, w] = bm((i, j) -> hc(Xb, i, j, in_deep))
                    # Stratification: top cell against the deep-layer mean, each model
                    md = findall(in_deep)
                    ddeep = sum(depth[m] * Δz[m] for m in md) / sum(Δz[m] for m in md)
                    lay = (X, i, j) -> sum(X[i, j, m] * Δz[m] for m in md) / sum(Δz[m] for m in md)
                    step = (T, S, i, j) -> density_step(T[i, j, 1], S[i, j, 1], depth[1], lay(T, i, j), lay(S, i, j), ddeep)
                    sr = Dict((i, j) => step(Tr, Sr, i, j) for (i, j) in both_deep)
                    sb = Dict((i, j) => step(Tb, Sb, i, j) for (i, j) in both_deep)
                    res[:σ_run][b, w]   = bm((i, j) -> sum(sr[(i, j)]))
                    res[:σ_bsose][b, w] = bm((i, j) -> sum(sb[(i, j)]))
                    res[:Δσ_T][b, w]    = bm((i, j) -> sr[(i, j)][1] - sb[(i, j)][1])
                    res[:Δσ_S][b, w]    = bm((i, j) -> sr[(i, j)][2] - sb[(i, j)][2])
                    res[:Stop_run][b, w]    = bm((i, j) -> Sr[i, j, 1])
                    res[:Stop_bsose][b, w]  = bm((i, j) -> Sb[i, j, 1])
                    res[:Sdeep_run][b, w]   = bm((i, j) -> lay(Sr, i, j))
                    res[:Sdeep_bsose][b, w] = bm((i, j) -> lay(Sb, i, j))
                    # Overturning: MLD, salt deficit, convective resistance
                    ov(T, S, i, j, Tf) = overturning(view(T, i, j, :), view(S, i, j, :), depth, Δz, md, kref, Tf)
                    orr = Dict((i, j) => ov(Tr, Sr, i, j, Tf_run) for (i, j) in both_deep)
                    obb = Dict((i, j) => ov(Tb, Sb, i, j, Tf_bsose) for (i, j) in both_deep)
                    for (q, key) in enumerate((:mld, :sd, :cr))
                        res[Symbol(key, :_run)][b, w]   = bm((i, j) -> orr[(i, j)][q])
                        res[Symbol(key, :_bsose)][b, w] = bm((i, j) -> obb[(i, j)][q])
                    end
                end
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
    println("  SOURCE=$BSOSE_SOURCE: $(bsose_source_note())")
    println("  Δ = run - B-SOSE, split at $(Int(SPLIT)) m; 1 MJ/m² melts $(round(1e6 / ρᵢL * 100, digits = 2)) cm of ice")
    D0, D1 = Int(HC_DEPTH), round(Int, deep_bot)
    println("  right of ||: columns wet to $D1 m in both (deep cols), their Δ 0-$D0, the $D0-$D1 m heat of both and Δ $D0-$D1")
    println(bar)
    for b in 1:nb
        all(isnan, res[:cells][b, :]) && continue
        Sm = mean(filter(isfinite, res[:S_bsose][b, :]))
        @printf("\n  %s   (freezing points at S = %.2f: run %.3f, B-SOSE %.3f °C)\n", band_name(b), Sm, Tf_run(Sm), Tf_bsose(Sm))
        println("  window centre | cells |  ℵ run  B-SOSE | top-cell T-Tf run  B-SOSE |   heat run  B-SOSE |  Δ 0-$(Int(SPLIT))  Δ $(Int(SPLIT))-$(Int(HC_DEPTH))   Δ as ice || deep cols | Δ 0-$D0 | $D0-$D1 run  B-SOSE |  Δ $D0-$D1")
        for w in 1:nw
            isnan(res[:cells][b, w]) && continue
            Δ = res[:hc_run][b, w] - res[:hc_bsose][b, w]
            @printf("  %s    | %5d |  %5.2f   %5.2f |         %6.3f  %6.3f |  %6.2f   %6.2f |   %+6.2f    %+6.2f   %+6.1f cm || %9s | %+7.2f |   %7.2f  %7.2f |  %+7.2f\n",
                    label(centre(w)), res[:cells][b, w], res[:ℵ_run][b, w], res[:ℵ_bsose][b, w],
                    res[:sst_run][b, w], res[:sst_bsose][b, w], res[:hc_run][b, w], res[:hc_bsose][b, w],
                    res[:Δhc_upper][b, w], res[:Δhc_lower][b, w], Δ * 1e6 / ρᵢL * 100,
                    isnan(res[:deep_cells][b, w]) ? "-" : string(Int(res[:deep_cells][b, w])), res[:Δhc_sub][b, w],
                    res[:dhc_run][b, w], res[:dhc_bsose][b, w], res[:dhc_run][b, w] - res[:dhc_bsose][b, w])
        end
    end

    println("\n", bar)
    println("STRATIFICATION on the deep columns: density step from the top cell to the $D0-$D1 m mean (kg/m³, linearised TEOS-10)")
    println("  Δ = run - B-SOSE, split into its temperature and salinity parts; larger = winter mixing reaches the warm water less easily")
    println(bar)
    for b in 1:nb
        all(isnan, res[:σ_run][b, :]) && continue
        @printf("\n  %s\n", band_name(b))
        println("  window centre | deep cols |  step run  B-SOSE |   Δ from T  from S |  top S run  B-SOSE | $D0-$D1 S run  B-SOSE")
        for w in 1:nw
            isnan(res[:σ_run][b, w]) && continue
            @printf("  %s    | %9d |   %6.3f  %6.3f |    %+6.3f  %+6.3f |   %7.3f  %7.3f |      %7.3f  %7.3f\n",
                    label(centre(w)), res[:deep_cells][b, w], res[:σ_run][b, w], res[:σ_bsose][b, w],
                    res[:Δσ_T][b, w], res[:Δσ_S][b, w], res[:Stop_run][b, w], res[:Stop_bsose][b, w],
                    res[:Sdeep_run][b, w], res[:Sdeep_bsose][b, w])
        end
    end

    println("\n", bar)
    println("OVERTURNING on the deep columns, to the $D0-$D1 m layer (Gülk et al. 2024 eqs. 8-9)")
    println("  MLD: σ₀ > σ₀($(Int(MLD_REF)) m) + $(MLD_DSIGMA) kg/m³ (level centre); SD*: ice to grow (m, ΔSᵢ = $(Int(DSI)) g/kg) for the")
    println("  mixed layer, cooled to freezing, to match the deep layer's density; CR: g/ρ₀∫(σ₀(H) - σ₀)dz to $D1 m, m²/s²")
    println("  run/B-SOSE > 1 in SD* = the run's column needs more ice growth than B-SOSE's to overturn")
    println(bar)
    for b in 1:nb
        all(isnan, res[:sd_run][b, :]) && continue
        @printf("\n  %s\n", band_name(b))
        println("  window centre | deep cols |  MLD run  B-SOSE |  SD* run  B-SOSE  ratio |   CR run  B-SOSE  ratio")
        for w in 1:nw
            isnan(res[:sd_run][b, w]) && continue
            @printf("  %s    | %9d |   %6.1f  %6.1f |   %5.2f   %5.2f  %5.2f |   %6.3f  %6.3f  %5.2f\n",
                    label(centre(w)), res[:deep_cells][b, w], res[:mld_run][b, w], res[:mld_bsose][b, w],
                    res[:sd_run][b, w], res[:sd_bsose][b, w], res[:sd_run][b, w] / res[:sd_bsose][b, w],
                    res[:cr_run][b, w], res[:cr_bsose][b, w], res[:cr_run][b, w] / res[:cr_bsose][b, w])
        end
    end

    # ── figure ───────────────────────────────────────────────────────────────────────────────
    bands = [b for b in 1:nb if !all(isnan, res[:cells][b, :])]
    isempty(bands) && return
    x = [Dates.value(centre(w) - start) / 86_400_000 for w in 1:nw]
    ticks = (x[1:3:end], [Dates.format(centre(w), "yyyy-mm") for w in 1:3:nw])
    Δprof = prof_run .- prof_bsose
    lim = let v = filter(isfinite, abs.(Δprof)); isempty(v) ? 1.0 : max(quantile(v, 0.98), 0.05) end
    fig = Figure(size = (2000, 380 * length(bands)))
    Label(fig[0, 1:4], "$RUN: water under B-SOSE's ice, above each model's freezing point"; fontsize = 18, font = :bold)
    local hm
    for (r, b) in enumerate(bands)
        ax1 = Axis(fig[r, 1]; title = "$(band_name(b)): heat above freezing, 0-$(Int(HC_DEPTH)) m",
                   ylabel = "MJ/m²", xticks = ticks, xticklabelrotation = π / 6)
        lines!(ax1, x, res[:hc_run][b, :]; color = :royalblue, linewidth = 2, label = "run")
        lines!(ax1, x, res[:hc_bsose][b, :]; color = :black, linestyle = :dash, linewidth = 2, label = "B-SOSE")
        r == 1 && axislegend(ax1; position = :lc, backgroundcolor = (:white, 0.8))
        axd = Axis(fig[r, 2]; title = "$(band_name(b)): run - B-SOSE heat by layer", ylabel = "MJ/m²",
                   xticks = ticks, xticklabelrotation = π / 6)
        hlines!(axd, 0; color = :gray, linewidth = 1)
        lines!(axd, x, res[:Δhc_upper][b, :]; color = :darkorange, linewidth = 2, label = "0-$(Int(SPLIT)) m")
        lines!(axd, x, res[:Δhc_lower][b, :]; color = :seagreen, linewidth = 2, label = "$(Int(SPLIT))-$(Int(HC_DEPTH)) m")
        lines!(axd, x, res[:dhc_run][b, :] .- res[:dhc_bsose][b, :]; color = :purple, linewidth = 2,
               label = "$(Int(HC_DEPTH))-$(round(Int, deep_bot)) m (deep cols)")
        r == 1 && axislegend(axd; position = :lb, backgroundcolor = (:white, 0.8))
        ax2 = Axis(fig[r, 3]; title = "$(band_name(b)): run - B-SOSE, T - Tf (°C)", ylabel = "depth (m)",
                   xticks = ticks, xticklabelrotation = π / 6, yreversed = true)
        hm = heatmap!(ax2, x, depth, Δprof[b, :, :]; colormap = :balance, colorrange = (-lim, lim))
    end
    Colorbar(fig[1:length(bands), 4], hm; label = "Δ(T - Tf), °C")
    out = joinpath(OUTDIR, BSOSE_SOURCE == "5day" ? "$(RUN)_under_ice_heat.png" : "$(RUN)_under_ice_heat_monthly.png")
    save(out, fig)
    @info "wrote $out"
end

main()
