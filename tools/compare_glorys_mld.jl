# ==============================================================================
# compare_glorys_mld.jl -- a run's mixed layer and near-surface stratification against GLORYS's, south of
# 60S, month by month. Does the run mix its summer fresh water too deep?
#
#     source .copernicus_env      # only while GLORYS files are missing
#     julia --project=. tools/compare_glorys_mld.jl 2Y_nested_glorys_hj
#
# WHY. tools/coastal_freshwater_budget.jl found GLORYS's summer coastal freshening a sharp cap (1.6-2.7
# g/kg at the surface, nearly gone by 50 m) and the run's a uniform 0.4-0.5 g/kg over 0-50 m. Part of the
# gap is a missing source; the shape says the run may also mix too deep. This measures that directly.
#
# WHAT. Both sides as the MIXED LAYER OF THE MONTHLY-MEAN STATE (tools/compare_mld.jl's reason: the mld of
# a mean is not the mean of mlds, so both sides are means): GLORYS's monthly means, averaged onto the run's
# cells and levels as compare_glorys.jl does, against the run's 30-day means (_eke.nc), the window nearest
# each month's 15th. MLD by tools/mld.jl (de Boyer Montegut: σ exceeds its 10 m value by MLD_DT), on the
# same levels and the same equation of state for both, down to Z_MAX or the shallower of the two floors.
# A column with no crossing is "mixed to the bottom" (or to Z_MAX): counted, and given that depth in the
# medians, so they are lower bounds there.
# Also the density step Δσ = σ(50 m) - σ(top cell): the strength of the fresh cap, independent of any
# threshold.
#
# REGIONS as coastal_freshwater_budget.jl: coast (within COAST_CELLS of land, south of COAST_SOUTH_OF),
# coast 95-110E, offshore; TRIM degrees left out at the open sides; cells wet in both.
#
# DOWNLOADS: GLORYS temperature and salinity 0-(Z_MAX+50) m south of COAST_SOUTH_OF + 1, a month a file.
#
# WRITES (OUTDIR, default plots/): <RUN>_mld_vs_glorys.png, the DJF-mean MLD maps (GLORYS, run, run minus
# GLORYS) for each summer, and the regional series.
#
# LOGIN-NODE SAFE: NCDatasets, one record at a time.
#
# ENV
#     DIR, OUTDIR, GLORYS_DIR      as compare_glorys.jl (., plots, <data>/glorys_monthly)
#     Z_MAX                        deepest MLD sought, m (default 500)
#     MLD_DT                       density threshold, kg/m3 (default 0.03, in tools/mld.jl)
#     COAST_CELLS, COAST_SOUTH_OF  default 2, -60
#     TRIM                         default 0.67
# ==============================================================================

using NumericalEarth
using NumericalEarth.DataWrangling: BoundingBox, Metadatum, metadata_path, dataset_variable_name
using NumericalEarth.DataWrangling.GLORYS: GLORYSMonthly
using CopernicusMarine          # NumericalEarth downloads GLORYS through it
import Downloads
using NCDatasets, CairoMakie, Printf, Statistics, Dates

include(joinpath(@__DIR__, "..", "src", "setup_bsose.jl"))   # default_bsose_directory
include(joinpath(@__DIR__, "model_grid.jl"))                 # file_levels
include(joinpath(@__DIR__, "mld.jl"))                        # mld, rho_theta, MLD_DT

const RUN = length(ARGS) > 0 ? ARGS[1] : error("give the run: julia --project=. tools/compare_glorys_mld.jl <RUN>")
const DIR        = get(ENV, "DIR", ".")
const OUTDIR     = get(ENV, "OUTDIR", "plots")
const GLORYS_DIR = get(ENV, "GLORYS_DIR", joinpath(default_bsose_directory(), "glorys_monthly"))
const Z_MAX      = parse(Float64, get(ENV, "Z_MAX", "500"))
const COAST_CELLS    = parse(Int, get(ENV, "COAST_CELLS", "2"))
const COAST_SOUTH_OF = parse(Float64, get(ENV, "COAST_SOUTH_OF", "-60"))
const TRIM           = parse(Float64, get(ENV, "TRIM", "0.67"))
const CAP_DEPTH      = 50.0
mkpath(OUTDIR); mkpath(GLORYS_DIR)

f64(A) = Float64.(coalesce.(A, NaN))

const SURF = joinpath(DIR, "$(RUN)_surface.nc")
const EKE  = joinpath(DIR, "$(RUN)_eke.nc")
for f in (SURF, EKE)
    isfile(f) || error("no $f: this needs the run's surface and eke files (set DIR)")
end

# ── the run's grid, levels within Z_MAX, and the regions ──────────────────────────────────────
ds = Dataset(SURF)
λc, φc = f64(ds["λ_caa"][:]), f64(ds["φ_aca"][:])
Nx, Ny = length(λc), length(φc)
Δλ, Δφ = λc[2] - λc[1], φc[2] - φc[1]
λW, φS = λc[1] - Δλ / 2, φc[1] - Δφ / 2
λE, φN = λW + Nx * Δλ, φS + Ny * Δφ
start = DateTime(string(ds.attrib["start_date"]))
bottom = f64(ds["bottom_height"][:, :])
close(ds)

ke = Dataset(EKE)
zc, zf, _ = file_levels(ke)
eke_times = f64(ke["time"][:])
window = haskey(ke.attrib, "averaging_window_seconds") ? Float64(ke.attrib["averaging_window_seconds"]) : 30 * 86400.0
close(ke)

# The levels whose top is above Z_MAX, bottom to top as Oceananigans stores them
const KS = let ks = [k for k in eachindex(zc) if -zf[k + 1] < Z_MAX]; first(ks):last(ks) end
const NK = length(KS)
const ZS = zc[KS]
const K_CAP = argmin(abs.(ZS .+ CAP_DEPTH))          # the level nearest 50 m
# deepest level of the subset whose centre is above the floor (PartialCellBottom keeps a cut cell; its
# T and S are still that cell's)
kbot_run = [bottom[i, j] < 0 ? something(findfirst(n -> zf[KS[n] + 1] > bottom[i, j], 1:NK), NK + 1) : NK + 1
            for i in 1:Nx, j in 1:Ny]

wet_top = bottom .< 0
inside = [λW + TRIM <= λc[i] <= λE - TRIM && φc[j] <= φN - TRIM for i in 1:Nx, j in 1:Ny]
coastal = falses(Nx, Ny)
for i in 1:Nx, j in 1:Ny
    (wet_top[i, j] && φc[j] < COAST_SOUTH_OF) || continue
    coastal[i, j] = any(.!wet_top[max(1, i - COAST_CELLS):min(Nx, i + COAST_CELLS),
                                  max(1, j - COAST_CELLS):min(Ny, j + COAST_CELLS)])
end
south = [φ < COAST_SOUTH_OF for _ in 1:Nx, φ in φc]
const REGIONS = [("coast", coastal .& inside),
                 ("coast 95-110E", coastal .& inside .& [95 <= λ < 110 for λ in λc, _ in 1:Ny]),
                 ("offshore", wet_top .& south .& .!coastal .& inside)]

# ── months, and the run's window for each ─────────────────────────────────────────────────────
center(t) = start + Millisecond(round(Int, 1000t))
eke_keep = findall(>(0), eke_times)
eke_dates = [center(t - window / 2) for t in eke_times[eke_keep]]
months = Tuple{Int,Int}[]
let d = Date(year(start), month(start), 1), last_centre = maximum(eke_dates)
    while DateTime(d) + Day(14) <= last_centre + Day(15)
        push!(months, (year(d), month(d))); d += Month(1)
    end
end
label(k) = @sprintf("%d-%02d", k...)
window_for(k) = eke_keep[argmin(abs.(Dates.value.(eke_dates .- DateTime(k[1], k[2], 15))))]

# ── GLORYS T and S, 0-(Z_MAX+50) m south of COAST_SOUTH_OF + 1 ─────────────────────────────────
r2(x) = round(x, digits = 6)
const REGION = BoundingBox(longitude = (r2(λW), r2(λE)), latitude = (r2(φS), r2(COAST_SOUTH_OF + 1)),
                           z = (-(Z_MAX + 50), 0.0))
glorys_meta(var, k) = Metadatum(var; dataset = GLORYSMonthly(), date = DateTime(k[1], k[2], 1),
                                region = REGION, dir = GLORYS_DIR)
missing_files = [(v, k) for v in (:temperature, :salinity) for k in months if !isfile(metadata_path(glorys_meta(v, k)))]
if !isempty(missing_files)
    haskey(ENV, "COPERNICUSMARINE_SERVICE_USERNAME") && haskey(ENV, "COPERNICUSMARINE_SERVICE_PASSWORD") ||
        error("$(length(missing_files)) GLORYS files to download, but no Copernicus credentials in the " *
              "environment: run `source .copernicus_env` first")
    @info "downloading $(length(missing_files)) GLORYS monthly T/S files (0-$(Int(Z_MAX + 50)) m) into $GLORYS_DIR"
    foreach(((v, k),) -> Downloads.download(glorys_meta(v, k)), missing_files)
end

# Averaging GLORYS onto the run's cells and levels, as compare_glorys.jl
shares(p) = (n = round(Int, p); abs(p - n) < 1e-6 ? ((n, 0.5), (n + 1, 0.5)) : ((floor(Int, p) + 1, 1.0),))
function cell_means(values, lons, lats)
    total, weight = zeros(Nx, Ny), zeros(Nx, Ny)
    for b in axes(values, 2), a in axes(values, 1)
        v = values[a, b]
        isfinite(v) || continue
        for (i, wi) in shares((lons[a] - λW) / Δλ), (j, wj) in shares((lats[b] - φS) / Δφ)
            (1 <= i <= Nx && 1 <= j <= Ny) || continue
            total[i, j] += wi * wj * v; weight[i, j] += wi * wj
        end
    end
    return [weight[i, j] > 0 ? total[i, j] / weight[i, j] : NaN for i in 1:Nx, j in 1:Ny]
end
glorys_edges(d) = vcat(0.0, (d[1:end-1] .+ d[2:end]) ./ 2, d[end] + (length(d) > 1 ? (d[end] - d[end-1]) / 2 : d[end]))
function layer_mean(x, edges, top, bottom)
    num, den = zeros(size(x, 1), size(x, 2)), zeros(size(x, 1), size(x, 2))
    for n in axes(x, 3)
        w = max(0.0, min(edges[n + 1], bottom) - max(edges[n], top))
        w == 0 && continue
        ok = isfinite.(x[:, :, n])
        num .+= w .* ifelse.(ok, x[:, :, n], 0.0); den .+= w .* ok
    end
    return ifelse.(den .> 0, num ./ max.(den, eps()), NaN)
end
"""GLORYS `var` for month `k` on the run's (Nx, Ny, NK) cells and levels; NaN where GLORYS has no water."""
function glorys_field(var, k)
    md = glorys_meta(var, k)
    x, lon, lat, depth = Dataset(metadata_path(md)) do d
        v = d[dataset_variable_name(md)]
        (f64(ndims(v) == 4 ? v[:, :, :, 1] : v[:, :, :]), f64(d["longitude"][:]), f64(d["latitude"][:]), f64(d["depth"][:]))
    end
    edges = glorys_edges(depth)
    F = fill(NaN, Nx, Ny, NK)
    for (n, kk) in enumerate(KS)
        F[:, :, n] = cell_means(layer_mean(x, edges, -zf[kk + 1], -zf[kk]), lon, lat)
    end
    return F
end
run_field(name, r) = Dataset(EKE) do d
    f64(d[name][:, :, KS, r])
end

# ── MLD and the cap, per cell ─────────────────────────────────────────────────────────────────
"""
MLD (m), whether it is a lower bound (no crossing above the floor or Z_MAX), and Δσ(50 m - top), for
every cell, from T and S on the run's levels. `kbot` is the deepest usable level of each column: the run's
floor, raised to where both sides have water.
"""
function column_stats(T, S, kbot)
    h, bound, cap = fill(NaN, Nx, Ny), falses(Nx, Ny), fill(NaN, Nx, Ny)
    for i in 1:Nx, j in 1:Ny
        kb = kbot[i, j]
        kb <= NK || continue
        Tc, Sc = view(T, i, j, :), view(S, i, j, :)
        m = mld(Tc, Sc, ZS, kb)
        if isfinite(m)
            h[i, j] = m
        else
            h[i, j] = -ZS[kb]; bound[i, j] = true      # mixed to the floor or to Z_MAX
        end
        K_CAP >= kb && (cap[i, j] = rho_theta(Tc[K_CAP], Sc[K_CAP]) - rho_theta(Tc[NK], Sc[NK]))
    end
    return h, bound, cap
end

"The deepest level where the run is wet AND GLORYS has water, per column."
function common_kbot(gT)
    kb = copy(kbot_run)
    for i in 1:Nx, j in 1:Ny
        kb[i, j] > NK && continue
        n = findfirst(n -> isfinite(gT[i, j, n]), kb[i, j]:NK)
        kb[i, j] = n === nothing ? NK + 1 : kb[i, j] + n - 1
        # the run's top must also be reachable with GLORYS water all the way up
        all(isfinite, view(gT, i, j, kb[i, j]:NK)) || (kb[i, j] = NK + 1)
    end
    return kb
end

@info "$RUN: $(NK) levels in 0-$(Int(Z_MAX)) m (cap level $(round(-ZS[K_CAP], digits = 1)) m); $(length(months)) months; " *
      join(["$(name) $(count(m)) cells" for (name, m) in REGIONS], ", ")

results = map(months) do k
    gT, gS = glorys_field(:temperature, k), glorys_field(:salinity, k)
    r = window_for(k)
    rT, rS = run_field("T", r), run_field("S", r)
    kb = common_kbot(gT)
    (; k, run_date = eke_dates[findfirst(==(r), eke_keep)], kb,
       glorys = column_stats(gT, gS, kb), run = column_stats(rT, rS, kb))
end

# ── print ─────────────────────────────────────────────────────────────────────────────────────
function region_stats(res, side, mask)
    h, bound, cap = getproperty(res, side)
    cells = [I for I in CartesianIndices(mask) if mask[I] && res.kb[I] <= NK && isfinite(h[I])]
    isempty(cells) && return (NaN, NaN, NaN)
    w = [cosd(φc[I[2]]) for I in cells]
    capw = [(isfinite(cap[I]) ? (cap[I] * cosd(φc[I[2]]), cosd(φc[I[2]])) : (0.0, 0.0)) for I in cells]
    return median(h[cells]), 100 * sum(w .* bound[cells]) / sum(w),
           sum(first.(capw)) / max(sum(last.(capw)), eps())
end

println("\n", "="^104)
println("MIXED LAYER AND THE FRESH CAP, $RUN against GLORYS (monthly-mean states, on the run's levels)")
@printf("  MLD: median depth (m) where σ exceeds its 10 m value by %.2f kg/m3; %%bot: share of cells mixed to the\n", MLD_DT)
@printf("  floor or %d m (their depth used, so medians are lower bounds there). Δσ: σ(%.0f m) - σ(top cell), kg/m3,\n", Int(Z_MAX), -ZS[K_CAP])
println("  the strength of the near-surface stratification (bigger: a stronger cap). Cells wet in both.")
println("="^104)
for (name, mask) in REGIONS
    @printf("\n  %s\n  %-8s %-10s | %8s %8s %6s %6s | %8s %8s\n", uppercase(name), "month", "run window",
            "MLD GLO", "MLD run", "%bot G", "%bot R", "Δσ GLO", "Δσ run")
    for res in results
        gh, gb, gc = region_stats(res, :glorys, mask)
        rh, rb, rc = region_stats(res, :run, mask)
        @printf("  %-8s %-10s | %8.0f %8.0f %6.0f %6.0f | %8.3f %8.3f\n", label(res.k),
                Dates.format(res.run_date, "yyyy-mm-dd"), gh, rh, gb, rb, gc, rc)
    end
end

# ── plot: DJF-mean MLD per summer, and the regional series ────────────────────────────────────
summers = unique([month(DateTime(r.k[1], r.k[2])) == 12 ? r.k[1] : r.k[1] - 1 for r in results if r.k[2] in (12, 1, 2)])
summers = [s for s in summers if count(r -> (r.k == (s, 12) || r.k == (s + 1, 1) || r.k == (s + 1, 2)), results) == 3]
jS = findall(<(COAST_SOUTH_OF), φc)
djf_mean(side, s) = begin
    sel = [r for r in results if r.k in ((s, 12), (s + 1, 1), (s + 1, 2))]
    A = [getproperty(r, side)[1] for r in sel]
    [all(a -> isfinite(a[I]), A) ? mean(a[I] for a in A) : NaN for I in CartesianIndices(A[1])]
end
fig = Figure(size = (1500, 260 * (length(summers) + length(REGIONS)) + 80))
for (row, s) in enumerate(summers)
    g, r = djf_mean(:glorys, s), djf_mean(:run, s)
    for (col, (A, title, cmap, range)) in enumerate(((g, "GLORYS", :deep, (0, 150)), (r, RUN, :deep, (0, 150)),
                                                   (r .- g, "run - GLORYS", :balance, (-100, 100))))
        ax = Axis(fig[row, 2col - 1]; title = "MLD (m), Dec $s - Feb $(s + 1): $title", aspect = DataAspect())
        heatmap!(ax, λc, φc[jS], A[:, jS]; colormap = cmap, colorrange = range, nan_color = :gray80)
        Colorbar(fig[row, 2col]; colormap = cmap, colorrange = range)
    end
end
dates = [DateTime(r.k[1], r.k[2], 15) for r in results]
for (n, (name, mask)) in enumerate(REGIONS)
    ax = Axis(fig[length(summers) + n, 1:6]; ylabel = "MLD (m), median", title = name, yreversed = true)
    lines!(ax, datetime2unix.(dates), [region_stats(r, :glorys, mask)[1] for r in results]; color = :black, label = "GLORYS")
    lines!(ax, datetime2unix.(dates), [region_stats(r, :run, mask)[1] for r in results]; color = :firebrick, label = RUN)
    ax.xticks = (datetime2unix.(dates[1:3:end]), [label(r.k) for r in results[1:3:end]])
    n == 1 && axislegend(ax; position = :rb)
end
out = joinpath(OUTDIR, "$(RUN)_mld_vs_glorys.png")
save(out, fig)
@info "wrote $out"
