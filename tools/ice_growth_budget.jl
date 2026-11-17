# ==============================================================================================
# ice_growth_budget.jl -- what grows and melts a run's sea ice, the air above or the ocean below,
# by latitude band and calendar month, against B-SOSE's change in ice volume.
#
#     julia --project=. tools/ice_growth_budget.jl 2Y_nested_bsose_ggl_hj_pr207
#     BSOSE_DIR=/storage/scratch1/1/vnguyen480/SOWinWaRP/data julia --project=. tools/ice_growth_budget.jl 2Y_nested_bsose_ggl_hj_pr207
#
# NEEDS <RUN>_sea_ice.nc (daily h, ℵ snapshots) and <RUN>_sea_ice_fluxes.nc (daily means), the
# second written by examples/sowinwarp_nested_bsose_{ggl,catke}.jl from 2026-11-17 on.
#
# THE BUDGET. Ice volume V = Σ h·ℵ·A over a band, in km³ per month, positive = more ice:
#     total    ΔV between the snapshots that bound the month's daily windows
#     thermo   ∫ ice_mass_tendency / ρᵢ: freezing and melting, all of it (ClimaSeaIce's own diagnostic)
#     ocean    -∫ interface_heat / (ρᵢ L): ice melted from below by ocean heat (always ≤ 0)
#     frazil   -∫ frazil_heat / (ρᵢ L): ice formed from water cooled below freezing (≥ 0)
#     air      thermo - ocean - frazil: what the air does through the top of the ice
#     drift    total - thermo: ice carried in or out of the band. Over the whole domain it is the
#              ice crossing the open sides (and any remapping error).
# ClimaSeaIce's slab grows at (top - bottom)/(ρᵢ L), every flux positive upward, and the bottom
# flux is interface + frazil, so the split is exact up to the mass tendency's bookkeeping of
# new ice in open water. As a check, `air≈` is top_heat/(ρᵢ L) integrated the same way; it and
# `air` should agree to within the growth that conduction lags.
#
# THE AIR, IN W/m² (winter months only, per ICE area, weighted by ℵ·A each day): sensible, latent,
# upwelling longwave, absorbed longwave and shortwave (both < 0, downward), and their sum. Positive
# = the ice loses heat to the air = it grows. Compare a band's sum with what its ice needs: 1 cm/day
# of growth is 35 W/m² (ρᵢ L = 3.0e8 J/m³).
#
# B-SOSE (BSOSE_DIR, optional): ΔV per band and month from its 5-day SIheff, the window centres
# interpolated to the same month boundaries. B-SOSE's total is all we have of its budget unless its
# SIdHb* diagnostics are on disk.
#
# WRITES (OUTDIR, default plots/): <RUN>_ice_growth_budget.png, one panel per band, km³/month.
# ENV: DIR (run files, default .), OUTDIR, BSOSE_DIR, BANDS (default -70,-65,-60,-55,-50,-45),
#      WINTER (months for the W/m² table, default 5,6,7,8,9).
# ==============================================================================================

using NCDatasets, CairoMakie, Dates, Printf, Statistics

ENV["HDF5_USE_FILE_LOCKING"] = "FALSE"   # safe on a run that is still writing

RUN = isempty(ARGS) ? get(ENV, "RUN", "") : ARGS[1]
isempty(RUN) && error("give the run name: julia --project=. tools/ice_growth_budget.jl <RUN>")
DIR       = get(ENV, "DIR", ".")
OUTDIR    = get(ENV, "OUTDIR", "plots")
BSOSE_DIR = get(ENV, "BSOSE_DIR", "")
EDGES     = parse.(Float64, split(get(ENV, "BANDS", "-70,-65,-60,-55,-50,-45"), ','))
WINTER    = parse.(Int, split(get(ENV, "WINTER", "5,6,7,8,9"), ','))

ice_path  = joinpath(DIR, "$(RUN)_sea_ice.nc")
flux_path = joinpath(DIR, "$(RUN)_sea_ice_fluxes.nc")
isfile(ice_path)  || error("$ice_path not found; set DIR to the run directory")
isfile(flux_path) || error("$flux_path not found: this run predates the sea ice flux output (re-run it with the current script)")

g(A) = Float64.(coalesce.(A, NaN))
z(A) = replace(A, NaN => 0.0)

# ── the run's ice ────────────────────────────────────────────────────────────────────────────
ds = NCDataset(ice_path)
λ  = g(ds["λ_caa"][:]); φ = g(ds["φ_aca"][:])
A  = g(ds["Δx_cca"][:, :]) .* g(ds["Δy_cca"][:, :])
wet = g(ds["bottom_height"][:, :]) .< 0
start = DateTime(ds.attrib["start_date"])
ts = g(ds["time"][:])                                   # seconds since start
V  = z(g(ds["h"][:, :, :]) .* g(ds["ℵ"][:, :, :]))     # h·ℵ, m
ℵs = z(g(ds["ℵ"][:, :, :]))
close(ds)
Nx, Ny = size(A)
for n in axes(V, 3); @views V[:, :, n][.!wet] .= 0; @views ℵs[:, :, n][.!wet] .= 0; end

fs = NCDataset(flux_path)
ρᵢ = Float64(get(fs.attrib, "ice_density", 900.0))
L  = Float64(get(fs.attrib, "latent_heat_of_fusion", 334e3))
tf = g(fs["time"][:])
readday(name, n) = (v = fs[name]; z(g(ndims(v) == 4 ? v[:, :, 1, n] : v[:, :, n])))
keep = findall(>(0), tf)                                # a t = 0 record holds no average

# ── bands and months ─────────────────────────────────────────────────────────────────────────
nb = length(EDGES) - 1
band_name(b) = @sprintf("%.0f-%.0fS", -EDGES[b], -EDGES[b+1])
band_of = [findfirst(b -> EDGES[b] <= φ[j] < EDGES[b+1], 1:nb) for j in 1:Ny]
bandmask = [[wet[i, j] && band_of[j] == b for i in 1:Nx, j in 1:Ny] for b in 1:nb]
bandsum(F, b) = sum(F[bandmask[b]] .* A[bandmask[b]])

datetime(t) = start + Millisecond(round(Int, 1000t))
window_month(t) = (d = datetime(t - 43200); (year(d), month(d)))   # the daily window's centre
months = unique(window_month.(tf[keep]))

const KM3 = 1e9
terms = (:total, :thermo, :air, :ocean, :frazil, :drift, :air_check)
budget = Dict(k => fill(NaN, nb, length(months)) for k in terms)
air_parts = (:sensible_heat, :latent_heat, :upwelling_longwave, :absorbed_longwave, :absorbed_shortwave)
winter_W  = Dict(k => zeros(nb) for k in (air_parts..., :weight))
icesnap(t) = findfirst(s -> abs(s - t) < 60, ts)        # the snapshot at a window's edge

for (m, ym) in enumerate(months)
    recs = [n for n in keep if window_month(tf[n]) == ym]
    acc = Dict(k => zeros(nb) for k in (:thermo, :ocean, :frazil, :air_check))
    for n in recs
        Δt = n > 1 ? tf[n] - tf[n-1] : 86400.0
        M  = readday("ice_mass_tendency", n)
        Qi = readday("interface_heat", n); Qf = readday("frazil_heat", n); Qt = readday("top_heat", n)
        for b in 1:nb
            acc[:thermo][b]    += bandsum(M, b) / ρᵢ * Δt
            acc[:ocean][b]     -= bandsum(Qi, b) / (ρᵢ * L) * Δt
            acc[:frazil][b]    -= bandsum(Qf, b) / (ρᵢ * L) * Δt
            acc[:air_check][b] += bandsum(Qt, b) / (ρᵢ * L) * Δt
        end
        if ym[2] in WINTER
            k = something(icesnap(tf[n]), icesnap(tf[n] - Δt), 0)
            k == 0 && continue
            w = ℵs[:, :, k] .* A
            parts = Dict(p => readday(String(p), n) for p in air_parts)
            for b in 1:nb
                wb = sum(w[bandmask[b]])
                winter_W[:weight][b] += wb
                for p in air_parts
                    winter_W[p][b] += sum(parts[p][bandmask[b]] .* w[bandmask[b]])
                end
            end
        end
    end
    a, e = icesnap(tf[first(recs)] - (first(recs) > 1 ? tf[first(recs)] - tf[first(recs)-1] : 86400.0)), icesnap(tf[last(recs)])
    for b in 1:nb
        for k in keys(acc); budget[k][b, m] = acc[k][b] / KM3; end
        budget[:air][b, m] = budget[:thermo][b, m] - budget[:ocean][b, m] - budget[:frazil][b, m]
        if a !== nothing && e !== nothing
            budget[:total][b, m] = (bandsum(V[:, :, e], b) - bandsum(V[:, :, a], b)) / KM3
            budget[:drift][b, m] = budget[:total][b, m] - budget[:thermo][b, m]
        end
    end
end
close(fs)

# ── B-SOSE's ΔV (optional) ───────────────────────────────────────────────────────────────────
bsose = fill(NaN, nb, length(months))
if !isempty(BSOSE_DIR)
    file(token, y) = joinpath(BSOSE_DIR, "$(token)_bsoseI156_$(y)_5day.nc")
    geom = joinpath(BSOSE_DIR, "surfTflx_bsoseI156_2013to2024_monthly.nc")
    years = (first(months)[1] - 1):(last(months)[1] + 1)
    have = [y for y in years if isfile(file("SIHeff", y))]
    if !isfile(geom) || isempty(have)
        @warn "B-SOSE not drawn: need $(basename(geom)) and SIHeff_bsoseI156_<year>_5day.nc in $BSOSE_DIR"
    else
        gds = NCDataset(geom)
        xc, yc = g(gds["XC"][:]), g(gds["YC"][:])
        IR = let r = findall(x -> λ[1] - 1/12 <= x <= λ[end] + 1/12, xc); minimum(r):maximum(r) end
        JR = let r = findall(y -> EDGES[1] <= y < EDGES[end], yc); minimum(r):maximum(r) end
        bwet = g(gds["Depth"][IR, JR]) .> 0
        rA = haskey(gds, "rA") ? g(gds["rA"][IR, JR]) : [(6.371e6 * deg2rad(xc[2] - xc[1]))^2 * cosd(y) for _ in IR, y in yc[JR]]
        close(gds)
        bb = [findfirst(b -> EDGES[b] <= y < EDGES[b+1], 1:nb) for y in yc[JR]]
        centres, vols = DateTime[], Vector{Float64}[]
        for y in have
            d = NCDataset(file("SIHeff", y))
            name = first(k for k in ("SIheff", "SIHeff", "SIHEFF") if haskey(d, k))
            H = z(g(d[name][IR, JR, :]))
            for (n, s) in enumerate(d["time"][:])
                push!(centres, DateTime(s) - Hour(60))
                push!(vols, [sum(H[i, j, n] * rA[i, j] for i in axes(H, 1), j in axes(H, 2) if bwet[i, j] && bb[j] == b; init = 0.0) / KM3 for b in 1:nb])
            end
            close(d)
        end
        o = sortperm(centres); centres, vols = centres[o], vols[o]
        function vol_at(t)
            k = searchsortedlast(centres, t)
            (k < 1 || k >= length(centres)) && return fill(NaN, nb)
            w = Dates.value(t - centres[k]) / Dates.value(centres[k+1] - centres[k])
            return (1 - w) .* vols[k] .+ w .* vols[k+1]
        end
        for (m, (y, mo)) in enumerate(months)
            bsose[:, m] = vol_at(DateTime(y, mo, 1) + Month(1)) .- vol_at(DateTime(y, mo, 1))
        end
    end
end

# ── tables ───────────────────────────────────────────────────────────────────────────────────
label(ym) = @sprintf("%04d-%02d", ym...)
println("\n", "="^104)
println("SEA ICE VOLUME BUDGET, $RUN -- km³ per month, positive = more ice")
println("  thermo = air + ocean + frazil;  total = thermo + drift;  air≈ = top_heat/(ρᵢL), a check on air")
println("="^104)
for b in 1:nb
    all(x -> abs(x) < 0.05 || isnan(x), budget[:thermo][b, :]) && continue
    println("\n  ", band_name(b))
    println("  month    |    total    drift   thermo |      air     ocean   frazil |    air≈ |  B-SOSE total")
    for m in eachindex(months)
        f(k) = budget[k][b, m]
        println(@sprintf("  %s  | %8.1f %8.1f %8.1f | %8.1f %8.1f %8.1f | %7.1f | %9.1f",
                         label(months[m]), f(:total), f(:drift), f(:thermo), f(:air), f(:ocean), f(:frazil), f(:air_check), bsose[b, m]))
    end
end

println("\n  WHOLE DOMAIN (drift here = ice crossing the open sides)")
println("  month    |    total    drift   thermo |      air     ocean   frazil |  B-SOSE total")
for m in eachindex(months)
    s(k) = sum(budget[k][:, m]); sb = sum(bsose[:, m])
    println(@sprintf("  %s  | %8.1f %8.1f %8.1f | %8.1f %8.1f %8.1f | %9.1f",
                     label(months[m]), s(:total), s(:drift), s(:thermo), s(:air), s(:ocean), s(:frazil), sb))
end

println("\n", "="^104)
println("THE AIR OVER THE ICE, months $(join(WINTER, ',')) -- W/m² per ice area, ℵ·A-weighted, + = ice loses heat upward")
println("  35 W/m² sustains 1 cm/day of growth")
println("="^104)
println("  band     | sensible   latent  LW up  LW absorbed  SW absorbed |    net")
for b in 1:nb
    w = winter_W[:weight][b]
    w > 0 || continue
    p = [winter_W[k][b] / w for k in air_parts]
    println(@sprintf("  %-8s | %8.1f %8.1f %6.1f %12.1f %12.1f | %6.1f", band_name(b), p..., sum(p)))
end
println()

# ── figure ───────────────────────────────────────────────────────────────────────────────────
mkpath(OUTDIR)
shown = [b for b in 1:nb if any(x -> abs(x) >= 0.05, filter(!isnan, budget[:thermo][b, :]))]
x = 1:length(months)
fig = Figure(size = (1100, 260 * length(shown) + 80), fontsize = 13)
Label(fig[0, 1], "$RUN: sea ice volume budget by band (km³/month, + = more ice)"; font = :bold, fontsize = 15)
style = ((:total, :black, :solid, 2.5), (:air, :steelblue, :solid, 2), (:ocean, :firebrick, :solid, 2),
         (:frazil, :seagreen, :solid, 2), (:drift, :darkorange, :solid, 2))
for (r, b) in enumerate(shown)
    ax = Axis(fig[r, 1]; title = band_name(b), ylabel = "km³/month",
              xticks = (x[1:3:end], label.(months)[1:3:end]), xticklabelrotation = π/6)
    hlines!(ax, [0]; color = :grey70)
    for (k, c, ls, lw) in style
        lines!(ax, x, budget[k][b, :]; color = c, linestyle = ls, linewidth = lw, label = String(k))
    end
    all(isnan, bsose[b, :]) || lines!(ax, x, bsose[b, :]; color = :black, linestyle = :dash, linewidth = 2, label = "B-SOSE total")
    r == 1 && Legend(fig[1, 2], ax; framevisible = false)
end
out = joinpath(OUTDIR, "$(RUN)_ice_growth_budget.png")
save(out, fig)
@info "wrote $out"
