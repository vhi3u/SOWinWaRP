# ==============================================================================
# coastal_freshwater_budget.jl -- how much fresh water the coast gains over summer: in GLORYS, in a run,
# and from the run's own surface fluxes. Is the run's summer coast salty because GLORYS has a source of
# fresh water the run does not?
#
#     source .copernicus_env      # only while GLORYS files are missing
#     julia --project=. tools/coastal_freshwater_budget.jl                     # 2Y_nested_glorys_baseline
#     H=100 julia --project=. tools/coastal_freshwater_budget.jl <RUN>
#
# FRESHWATER CONTENT over 0-H (default 150 m, below the summer mixed layer), per cell, against each
# dataset's own first record S₀:
#     FWC(t) = Σₖ Δzₖ (S₀ₖ - Sₖ(t)) / S₀ₖ      metres of fresh water; up = fresher
# on the run's levels and cells. GLORYS's monthly means are averaged onto them as compare_glorys.jl does
# (its points inside each 1/6-degree cell, its levels thickness-weighted into each of the run's). A level
# counts where the run is wet AND GLORYS has water, so both sides integrate the same water.
#     GLORYS  monthly means (cmems_mod_glo_phy_my_0.083deg_P1M-m), at the 15th of each month
#     run     <RUN>_eke.nc's S, 30-day means, at each window's centre (this file has the 3-D S)
#     input   <RUN>_surface_fluxes.nc, the daily fresh water the coupler handed the ocean, summed from
#             the run's first window centre: J_w + J_S / S_top (m), with J_w = rain, snow, evaporation and
#             ice melt/growth (+ in), J_S the sea-ice salt flux (+ out), S_top from <RUN>_surface.nc.
#             ERA5's P - E is in J_w: this is what the ocean actually received.
#
# READING IT. The run's FWC minus its input is what its own currents and mixing brought in or took away
# through the sides and the floor of the region. GLORYS's FWC minus the run's is the fresh water GLORYS's
# coast gains that the run's does not; if the run's input already matches the run's FWC, that difference
# is a source the run lacks (GLORYS adds land-ice and iceberg fresh water, and assimilates SST, sea ice and
# T/S profiles). For each summer the gain from the window nearest mid-November to the one nearest
# mid-February is printed.
#
# REGIONS, cos(lat)-weighted, south of COAST_SOUTH_OF and outside TRIM degrees of the open sides (the
# nest's relaxation band):
#     coast           within COAST_CELLS cells of land, as compare_glorys.jl
#     coast 95-110E   the same, where both runs are saltiest against GLORYS (the Shackleton Ice Shelf)
#     offshore        the rest
#
# DOWNLOADS: GLORYS salinity 0-(H+40) m over the run's box south of COAST_SOUTH_OF + 1, a month a file,
# ~10 MB each, into GLORYS_DIR (credentials from `source .copernicus_env`).
#
# WRITES (OUTDIR, default plots/): <RUN>_coastal_freshwater.png, FWC month by month per region: GLORYS
# black, the run red, the run's cumulative input red dashed. On stdout: the series and the Nov-Feb gains.
#
# LOGIN-NODE SAFE: NCDatasets, one record at a time.
#
# ENV
#     DIR, OUTDIR, GLORYS_DIR      as compare_glorys.jl (., plots, <data>/glorys_monthly)
#     H                            depth integrated, m (default 150)
#     COAST_CELLS, COAST_SOUTH_OF  default 2, -60
#     TRIM                         degrees left out at the open west, east and north sides (default 0.67)
# ==============================================================================

using NumericalEarth
using NumericalEarth.DataWrangling: BoundingBox, Metadatum, metadata_path, dataset_variable_name
using NumericalEarth.DataWrangling.GLORYS: GLORYSMonthly
using CopernicusMarine          # NumericalEarth downloads GLORYS through it
import Downloads
using NCDatasets, CairoMakie, Printf, Statistics, Dates

include(joinpath(@__DIR__, "..", "src", "setup_bsose.jl"))   # default_bsose_directory
include(joinpath(@__DIR__, "model_grid.jl"))                 # file_levels

const RUN = length(ARGS) > 0 ? ARGS[1] : "2Y_nested_glorys_baseline"
const DIR        = get(ENV, "DIR", ".")
const OUTDIR     = get(ENV, "OUTDIR", "plots")
const GLORYS_DIR = get(ENV, "GLORYS_DIR", joinpath(default_bsose_directory(), "glorys_monthly"))
const H          = parse(Float64, get(ENV, "H", "150"))
const COAST_CELLS    = parse(Int, get(ENV, "COAST_CELLS", "2"))
const COAST_SOUTH_OF = parse(Float64, get(ENV, "COAST_SOUTH_OF", "-60"))
const TRIM           = parse(Float64, get(ENV, "TRIM", "0.67"))
mkpath(OUTDIR); mkpath(GLORYS_DIR)

f64(A) = Float64.(coalesce.(A, NaN))

const SURF = joinpath(DIR, "$(RUN)_surface.nc")
const EKE  = joinpath(DIR, "$(RUN)_eke.nc")
const FLUX = joinpath(DIR, "$(RUN)_surface_fluxes.nc")
for f in (SURF, EKE, FLUX)
    isfile(f) || error("no $f: this needs the run's surface, eke and surface_fluxes files (set DIR)")
end

# ── the run's grid, levels within H, and the regions ──────────────────────────────────────────
ds = Dataset(SURF)
λc, φc = f64(ds["λ_caa"][:]), f64(ds["φ_aca"][:])
Nx, Ny = length(λc), length(φc)
Δλ, Δφ = λc[2] - λc[1], φc[2] - φc[1]
λW, φS = λc[1] - Δλ / 2, φc[1] - Δφ / 2
λE, φN = λW + Nx * Δλ, φS + Ny * Δφ
start = DateTime(string(ds.attrib["start_date"]))
bottom = f64(ds["bottom_height"][:, :])
surf_times = f64(ds["time"][:])
close(ds)

ke = Dataset(EKE)
zc, zf, _ = file_levels(ke)
eke_times = f64(ke["time"][:])
window = haskey(ke.attrib, "averaging_window_seconds") ? Float64(ke.attrib["averaging_window_seconds"]) : 30 * 86400.0
close(ke)

# The run's levels that reach into 0-H, top first, and each one's thickness inside 0-H and above the floor
# (the partial bottom cell counts only its wet part)
const KS = let ks = [k for k in eachindex(zc) if -zf[k + 1] < H]; first(ks):last(ks) end   # the top levels, contiguous
const NK = length(KS)
thickness = zeros(Nx, Ny, NK)
for i in 1:Nx, j in 1:Ny, (n, k) in enumerate(KS)
    bottom[i, j] < 0 || continue
    top_depth, base_depth = -zf[k + 1], min(-zf[k], H, -bottom[i, j])
    thickness[i, j, n] = max(0.0, base_depth - top_depth)
end

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

# ── GLORYS: salinity 0-(H+40) m south of COAST_SOUTH_OF, monthly ──────────────────────────────
center(t) = start + Millisecond(round(Int, 1000t))
eke_keep = findall(>(0), eke_times)
eke_dates = [center(t - window / 2) for t in eke_times[eke_keep]]   # each 30-day window's centre
first_month = (year(start), month(start))
run_end = center(maximum(surf_times))
months = Tuple{Int,Int}[]
let d = Date(first_month..., 1)
    while Date(d) + Month(1) <= Date(run_end)
        push!(months, (year(d), month(d))); d += Month(1)
    end
end
label(k) = @sprintf("%d-%02d", k...)

r2(x) = round(x, digits = 6)
const REGION = BoundingBox(longitude = (r2(λW), r2(λE)), latitude = (r2(φS), r2(COAST_SOUTH_OF + 1)),
                           z = (-(H + 40), 0.0))
glorys_meta(k) = Metadatum(:salinity; dataset = GLORYSMonthly(), date = DateTime(k[1], k[2], 1),
                           region = REGION, dir = GLORYS_DIR)
missing_files = filter(k -> !isfile(metadata_path(glorys_meta(k))), months)
if !isempty(missing_files)
    haskey(ENV, "COPERNICUSMARINE_SERVICE_USERNAME") && haskey(ENV, "COPERNICUSMARINE_SERVICE_PASSWORD") ||
        error("$(length(missing_files)) GLORYS files to download, but no Copernicus credentials in the " *
              "environment: run `source .copernicus_env` first")
    @info "downloading $(length(missing_files)) GLORYS monthly salinity files (0-$(Int(H + 40)) m) into $GLORYS_DIR"
    foreach(k -> Downloads.download(glorys_meta(k)), missing_files)
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

"""GLORYS salinity for month `k` on the run's (Nx, Ny, NK) cells and levels within 0-H."""
function glorys_S(k)
    md = glorys_meta(k)
    x, lon, lat, depth = Dataset(metadata_path(md)) do d
        v = d[dataset_variable_name(md)]
        (f64(ndims(v) == 4 ? v[:, :, :, 1] : v[:, :, :]), f64(d["longitude"][:]), f64(d["latitude"][:]), f64(d["depth"][:]))
    end
    edges = glorys_edges(depth)
    S = fill(NaN, Nx, Ny, NK)
    for (n, kk) in enumerate(KS)
        S[:, :, n] = cell_means(layer_mean(x, edges, -zf[kk + 1], -zf[kk]), lon, lat)
    end
    return S
end

"""The run's 30-day mean salinity, record `r` of the eke file, on the levels within 0-H."""
run_S(r) = Dataset(EKE) do d
    v = d["S"]
    f64(v[:, :, KS, r])
end

# ── freshwater content, per cell, against each side's first record ─────────────────────────────
@info "$RUN: $(NK) levels in 0-$(Int(H)) m; $(length(months)) GLORYS months, $(length(eke_keep)) run windows; " *
      join(["$(name) $(count(m)) cells" for (name, m) in REGIONS], ", ")

gS = [glorys_S(k) for k in months]
rS = [run_S(r) for r in eke_keep]
both = isfinite.(gS[1]) .& isfinite.(rS[1]) .& (thickness .> 0)   # the same water on both sides
weights = ifelse.(both, thickness, 0.0)

fwc(S, S₀) = dropdims(sum(ifelse.(both, weights .* (S₀ .- S) ./ S₀, 0.0), dims = 3), dims = 3)
function region_mean(A, mask)
    num = den = 0.0
    for i in 1:Nx, j in 1:Ny
        (mask[i, j] && isfinite(A[i, j]) && any(@view both[i, j, :])) || continue
        w = cosd(φc[j]); num += w * A[i, j]; den += w
    end
    return den == 0 ? NaN : num / den
end

g_dates = [DateTime(k[1], k[2], 15) for k in months]
g_fwc = [[region_mean(fwc(S, gS[1]), m) for S in gS] for (_, m) in REGIONS]
r_fwc = [[region_mean(fwc(S, rS[1]), m) for S in rS] for (_, m) in REGIONS]

# ── the run's surface fresh water, summed from its first window centre ─────────────────────────
fl = Dataset(FLUX)
flux_times = f64(fl["time"][:])
flux_window = haskey(fl.attrib, "averaging_window_seconds") ? Float64(fl.attrib["averaging_window_seconds"]) : 86400.0
recs = findall(>(0), flux_times)
flux_dates = [center(t - flux_window / 2) for t in flux_times[recs]]
readday(d, name, n) = (v = d[name]; f64(ndims(v) == 4 ? v[:, :, 1, n] : v[:, :, n]))
sf = Dataset(SURF)
input_dates = DateTime[]
input = [Float64[] for _ in REGIONS]
cumulative = [0.0 for _ in REGIONS]
for (n, d) in zip(recs, flux_dates)
    d < eke_dates[1] && continue
    Jw, JS = readday(fl, "freshwater_flux", n), readday(fl, "ice_salt_flux", n)
    s = argmin(abs.(surf_times .- (flux_times[n] - flux_window / 2)))
    Stop = readday(sf, "S", s)
    F = (Jw .+ JS ./ Stop) .* flux_window            # m of fresh water into the ocean this day
    for (r, (_, m)) in enumerate(REGIONS)
        cumulative[r] += region_mean(F, m)
        push!(input[r], cumulative[r])
    end
    push!(input_dates, d)
end
close(fl); close(sf)

# ── print ─────────────────────────────────────────────────────────────────────────────────────
nearest(dates, d) = argmin(abs.(Dates.value.(dates .- d)))
println("\n", "="^100)
println("COASTAL FRESHWATER, 0-$(Int(H)) m  ($RUN against GLORYS)")
println("  FWC: metres of fresh water gained since each side's first record (up = fresher).")
println("  input: the run's surface fresh water (J_w + J_S/S_top), summed from its first window centre.")
println("="^100)
for (r, (name, _)) in enumerate(REGIONS)
    @printf("\n  %s\n  %-8s | %9s | %9s %9s %10s\n", uppercase(name), "month", "GLORYS", "run", "input", "run-input")
    for (n, k) in enumerate(months)
        d = g_dates[n]
        rr = nearest(eke_dates, d); ii = nearest(input_dates, eke_dates[rr])
        far = abs(eke_dates[rr] - d) > Day(16)
        @printf("  %-8s | %+9.3f | %s\n", label(k), g_fwc[r][n],
                far ? "      (no run window)" :
                @sprintf("%+9.3f %+9.3f %+10.3f", r_fwc[r][rr], input[r][ii], r_fwc[r][rr] - input[r][ii]))
    end
end

println("\n  SUMMER GAIN, mid-November to mid-February (m of fresh water)")
@printf("  %-14s %-9s | %8s | %8s %8s %10s | %13s\n", "region", "summer", "GLORYS", "run", "input", "run-input", "GLORYS-run")
for (r, (name, _)) in enumerate(REGIONS), y in unique(first.(months))
    a, b = DateTime(y, 11, 15), DateTime(y + 1, 2, 15)
    (a >= g_dates[1] && b <= g_dates[end] && b <= eke_dates[end] + Day(16)) || continue
    ga, gb = nearest(g_dates, a), nearest(g_dates, b)
    ra, rb = nearest(eke_dates, a), nearest(eke_dates, b)
    ia, ib = nearest(input_dates, eke_dates[ra]), nearest(input_dates, eke_dates[rb])
    G = g_fwc[r][gb] - g_fwc[r][ga]
    R = r_fwc[r][rb] - r_fwc[r][ra]
    I = input[r][ib] - input[r][ia]
    @printf("  %-14s %d-%02d   | %+8.3f | %+8.3f %+8.3f %+10.3f | %+13.3f   (run windows %s, %s)\n",
            name, y, (y + 1) % 100, G, R, I, R - I, G - R, Date(eke_dates[ra]), Date(eke_dates[rb]))
end

# ── where in the column the summer fresh water goes ───────────────────────────────────────────
# Per level, mid-November to mid-February: the region-mean salinity change, and the fresh water that level
# gains (m), summed down from the surface. Fresh water put in at the surface (melt, icebergs, runoff) stays
# in the summer mixed layer, the top few tens of metres; freshening spread to 100-150 m is water brought in
# from elsewhere, or an assimilation increment.
function level_mean(A, mask, n)
    num = den = 0.0
    for i in 1:Nx, j in 1:Ny
        (mask[i, j] && both[i, j, n] && isfinite(A[i, j])) || continue
        w = cosd(φc[j]); num += w * A[i, j]; den += w
    end
    return den == 0 ? NaN : num / den
end
println("\n  WHERE IN THE COLUMN, mid-November to mid-February: ΔS (g/kg) per level, and the fresh water gained")
println("  from the surface down to that level (m), region means")
for (name, m) in REGIONS, y in unique(first.(months))
    a, b = DateTime(y, 11, 15), DateTime(y + 1, 2, 15)
    (a >= g_dates[1] && b <= g_dates[end] && b <= eke_dates[end] + Day(16)) || continue
    ga, gb = nearest(g_dates, a), nearest(g_dates, b)
    ra, rb = nearest(eke_dates, a), nearest(eke_dates, b)
    @printf("\n  %s, %d-%02d\n  %8s %7s | %8s %8s | %10s %10s\n", uppercase(name), y, (y + 1) % 100,
            "depth", "Δz", "ΔS GLO", "ΔS run", "fresh GLO", "fresh run")
    fg = fr = 0.0
    for (n, k) in reverse(collect(enumerate(KS)))   # KS runs up from the deepest level: start at the top
        dg = level_mean(gS[gb][:, :, n] .- gS[ga][:, :, n], m, n)
        dr = level_mean(rS[rb][:, :, n] .- rS[ra][:, :, n], m, n)
        fg += level_mean(weights[:, :, n] .* (gS[ga][:, :, n] .- gS[gb][:, :, n]) ./ gS[ga][:, :, n], m, n)
        fr += level_mean(weights[:, :, n] .* (rS[ra][:, :, n] .- rS[rb][:, :, n]) ./ rS[ra][:, :, n], m, n)
        @printf("  %6.1f m %5.1f m | %+8.3f %+8.3f | %+10.3f %+10.3f\n", -zc[k], zf[k + 1] - zf[k], dg, dr, fg, fr)
    end
end

# ── plot ──────────────────────────────────────────────────────────────────────────────────────
fig = Figure(size = (1100, 330 * length(REGIONS)))
for (r, (name, m)) in enumerate(REGIONS)
    ax = Axis(fig[r, 1]; title = "$name ($(count(m)) cells): fresh water gained over 0-$(Int(H)) m",
              ylabel = "m of fresh water", xticklabelrotation = π / 4)
    x(ds) = Dates.value.(ds .- start) ./ 86400e3
    lines!(ax, x(g_dates), g_fwc[r]; color = :black, label = "GLORYS")
    scatter!(ax, x(g_dates), g_fwc[r]; color = :black, markersize = 6)
    lines!(ax, x(eke_dates), r_fwc[r]; color = :firebrick, label = "$RUN")
    scatter!(ax, x(eke_dates), r_fwc[r]; color = :firebrick, markersize = 6)
    lines!(ax, x(input_dates), input[r]; color = :firebrick, linestyle = :dash, label = "run's surface input")
    hlines!(ax, [0]; color = :gray, linewidth = 0.5)
    ticks = [DateTime(k[1], k[2], 1) for k in months if k[2] in (6, 9, 12, 3)]
    ax.xticks = (x(ticks), [Dates.format(t, "yyyy-mm") for t in ticks])
    r == 1 && axislegend(ax; position = :lt)
end
out = joinpath(OUTDIR, "$(RUN)_coastal_freshwater.png")
save(out, fig)
@info "wrote $out"
