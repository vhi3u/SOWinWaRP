# ==============================================================================
# compare_glorys_sea_ice.jl -- a GLORYS-nested run's sea ice against GLORYS's, month by month, at the coast
# and offshore. Does GLORYS keep ice at the coast into summer and melt it there, where the run has none?
# (tools/coastal_freshwater_budget.jl: GLORYS's coast gains ~2 m of fresh water over 0-150 m each
# Nov-Feb, the baseline's ~0.8 m, and almost none of the run's comes through its surface.)
#
#     julia --project=. tools/compare_glorys_sea_ice.jl                        # 2Y_nested_glorys_baseline
#
# WHAT IS COMPARED, on the run's 1/6-degree cells (GLORYS's 1/12-degree points averaged into each, as
# compare_glorys.jl):
#     ℵ        concentration                           GLORYS siconc        run ℵ
#     h        floe thickness, where ℵ > 0.15          GLORYS sithick       run h
#     V = h ℵ  volume per area (m)                     sithick × siconc     h × ℵ
# GLORYS monthly means against the run's daily snapshots (<RUN>_sea_ice.nc) averaged over the same calendar
# month (months with fewer than MIN_DAYS days left out). Cells where the run is wet and GLORYS has ocean.
#
# MELT IN PLACE. For each summer the volume lost from mid-November to mid-February is printed, and the fresh
# water it would give if it melted where it is: ΔV × (ρᵢ/ρ_w) × (1 - Sᵢ/S) = ΔV × 0.75 (ρᵢ = 900, ρ_w = 1025,
# Sᵢ = 5, S = 34). Ice carried in or out by drift is not separated, so this is an upper bound on local melt
# when the region exports ice, and the number to put beside the freshwater budget's summer gain.
#
# GLORYS FILES. The siconc and sithick files the baseline downloaded (nested_sea_ice_simulation) are found in
# GLORYS_DIR by their variable and by their own coordinates: any file of the month that covers the run's box
# is used. A month with none is downloaded over the run's box (needs `source .copernicus_env`; ~1 MB a file).
#
# REGIONS (south of COAST_SOUTH_OF, outside TRIM degrees of the open sides), as coastal_freshwater_budget.jl:
#     coast, coast 95-110E, offshore
#
# WRITES (OUTDIR, default plots/):
#     <RUN>_sea_ice_vs_glorys_series.png   ℵ and V month by month per region, GLORYS black, the run red
#     <RUN>_sea_ice_vs_glorys_summer.png   ℵ maps, Oct-Mar of each summer, GLORYS above the run
# and on stdout: ℵ, h and V per region and month, and the Nov-Feb volume loss.
#
# LOGIN-NODE SAFE: NCDatasets, 2-D fields.
#
# ENV
#     DIR, OUTDIR, GLORYS_DIR, MIN_DAYS   as compare_glorys.jl (., plots, <data>/glorys_monthly, 20)
#     COAST_CELLS, COAST_SOUTH_OF, TRIM   default 2, -60, 0.67
# ==============================================================================

using NumericalEarth
using NumericalEarth.DataWrangling: BoundingBox, Metadatum, metadata_path
using NumericalEarth.DataWrangling.GLORYS: GLORYSMonthly
using CopernicusMarine          # NumericalEarth downloads GLORYS through it
import Downloads
using NCDatasets, CairoMakie, Printf, Statistics, Dates

include(joinpath(@__DIR__, "..", "src", "setup_bsose.jl"))   # default_bsose_directory

const RUN = length(ARGS) > 0 ? ARGS[1] : "2Y_nested_glorys_baseline"
const DIR        = get(ENV, "DIR", ".")
const OUTDIR     = get(ENV, "OUTDIR", "plots")
const GLORYS_DIR = get(ENV, "GLORYS_DIR", joinpath(default_bsose_directory(), "glorys_monthly"))
const MIN_DAYS   = parse(Int, get(ENV, "MIN_DAYS", "20"))
const COAST_CELLS    = parse(Int, get(ENV, "COAST_CELLS", "2"))
const COAST_SOUTH_OF = parse(Float64, get(ENV, "COAST_SOUTH_OF", "-60"))
const TRIM           = parse(Float64, get(ENV, "TRIM", "0.67"))
const MELT_TO_FRESH  = (900 / 1025) * (1 - 5 / 34)
const ICE_EDGE       = 0.15
mkpath(OUTDIR); mkpath(GLORYS_DIR)

f64(A) = Float64.(coalesce.(A, NaN))

const SURF = joinpath(DIR, "$(RUN)_surface.nc")
const ICE  = joinpath(DIR, "$(RUN)_sea_ice.nc")
for f in (SURF, ICE)
    isfile(f) || error("no $f (set DIR)")
end

# ── the run's grid, regions and months ────────────────────────────────────────────────────────
ds = Dataset(SURF)
λc, φc = f64(ds["λ_caa"][:]), f64(ds["φ_aca"][:])
Nx, Ny = length(λc), length(φc)
Δλ, Δφ = λc[2] - λc[1], φc[2] - φc[1]
λW, φS = λc[1] - Δλ / 2, φc[1] - Δφ / 2
λE, φN = λW + Nx * Δλ, φS + Ny * Δφ
start = DateTime(string(ds.attrib["start_date"]))
wet = f64(ds["bottom_height"][:, :]) .< 0
close(ds)

inside = [λW + TRIM <= λc[i] <= λE - TRIM && φc[j] <= φN - TRIM for i in 1:Nx, j in 1:Ny]
coastal = falses(Nx, Ny)
for i in 1:Nx, j in 1:Ny
    (wet[i, j] && φc[j] < COAST_SOUTH_OF) || continue
    coastal[i, j] = any(.!wet[max(1, i - COAST_CELLS):min(Nx, i + COAST_CELLS), max(1, j - COAST_CELLS):min(Ny, j + COAST_CELLS)])
end
south = [φ < COAST_SOUTH_OF for _ in 1:Nx, φ in φc]
const REGIONS = [("coast", coastal .& inside),
                 ("coast 95-110E", coastal .& inside .& [95 <= λ < 110 for λ in λc, _ in 1:Ny]),
                 ("offshore", wet .& south .& .!coastal .& inside)]

ice = Dataset(ICE)
ice_times = f64(ice["time"][:])
groups = Dict{Tuple{Int,Int},Vector{Int}}()
for (n, t) in enumerate(ice_times)
    t == 0 && continue
    d = start + Millisecond(round(Int, 1000t))
    push!(get!(groups, (year(d), month(d)), Int[]), n)
end
months = sort([k for (k, v) in groups if length(v) >= MIN_DAYS])
label(k) = @sprintf("%d-%02d", k...)

readice(name, n) = (v = ice[name]; f64(ndims(v) == 4 ? v[:, :, 1, n] : v[:, :, n]))
function run_month(k)
    n = groups[k]
    ℵ = sum(readice("ℵ", m) for m in n) ./ length(n)
    V = sum(readice("h", m) .* readice("ℵ", m) for m in n) ./ length(n)
    return ifelse.(wet, ℵ, NaN), ifelse.(wet, V, NaN)
end

# ── GLORYS: the baseline's own siconc / sithick files, found by their coordinates ─────────────
const VARS = (:sea_ice_concentration => "siconc", :sea_ice_thickness => "sithick")

"""Every GLORYS file in GLORYS_DIR holding `var` and covering the run's box, by its (year, month)."""
function find_files(var)
    found = Dict{Tuple{Int,Int},String}()
    for f in readdir(GLORYS_DIR; join = true)
        (startswith(basename(f), var * "_") && endswith(f, ".nc")) || continue
        try
            Dataset(f) do d
                haskey(d, var) || return
                lon, lat = f64(d["longitude"][:]), f64(d["latitude"][:])
                (minimum(lon) <= λW + Δλ && maximum(lon) >= λE - Δλ &&
                 minimum(lat) <= φS + Δφ && maximum(lat) >= COAST_SOUTH_OF) || return
                for t in d["time"][:]
                    found[(year(t), month(t))] = f
                end
            end
        catch err
            @warn "skipping $(basename(f)): $(sprint(showerror, err))"
        end
    end
    return found
end

r2(x) = round(x, digits = 6)
const REGION = BoundingBox(longitude = (r2(λW), r2(λE)), latitude = (r2(φS), r2(φN)))
glorys_meta(name, k) = Metadatum(name; dataset = GLORYSMonthly(), date = DateTime(k[1], k[2], 1),
                                 region = REGION, dir = GLORYS_DIR)

files = Dict(var => find_files(var) for (_, var) in VARS)
missing_files = [(name, k) for (name, var) in VARS for k in months if !haskey(files[var], k)]
if !isempty(missing_files)
    haskey(ENV, "COPERNICUSMARINE_SERVICE_USERNAME") && haskey(ENV, "COPERNICUSMARINE_SERVICE_PASSWORD") ||
        error("$(length(missing_files)) GLORYS sea-ice months not found in $GLORYS_DIR, and no Copernicus " *
              "credentials in the environment to download them: run `source .copernicus_env` first")
    @info "downloading $(length(missing_files)) GLORYS sea-ice files into $GLORYS_DIR"
    for (name, k) in missing_files
        Downloads.download(glorys_meta(name, k))
    end
    files = Dict(var => find_files(var) for (_, var) in VARS)
end
@info "$RUN against GLORYS sea ice: $(length(months)) months $(label(first(months)))..$(label(last(months))); " *
      "GLORYS files: $(length(unique(values(files["siconc"])))) siconc, $(length(unique(values(files["sithick"])))) sithick; " *
      join(["$(name) $(count(m)) cells" for (name, m) in REGIONS], ", ")

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

"""GLORYS `var` for month `k`, (lon, lat) values with their coordinates."""
function glorys_field(var, k)
    Dataset(files[var][k]) do d
        n = findfirst(t -> (year(t), month(t)) == k, d["time"][:])
        v = d[var]
        x = f64(ndims(v) == 4 ? v[:, :, 1, n] : v[:, :, n])
        (x, f64(d["longitude"][:]), f64(d["latitude"][:]))
    end
end
function glorys_month(k)
    c, lon, lat = glorys_field("siconc", k)
    h, lon2, lat2 = glorys_field("sithick", k)
    size(c) == size(h) || error("siconc and sithick for $(label(k)) are on different points")
    ℵ = cell_means(c, lon, lat)
    V = cell_means(ifelse.(isfinite.(c), coalesce.(c, 0.0) .* ifelse.(isfinite.(h), h, 0.0), NaN), lon, lat)
    return ifelse.(wet, ℵ, NaN), ifelse.(wet, V, NaN)
end

gℵ, gV, rℵ, rV = Dict(), Dict(), Dict(), Dict()
for k in months
    gℵ[k], gV[k] = glorys_month(k)
    rℵ[k], rV[k] = run_month(k)
end
close(ice)
both = reduce(.&, (isfinite.(gℵ[k]) .& isfinite.(rℵ[k]) for k in months))

function region_mean(A, mask)
    num = den = 0.0
    for i in 1:Nx, j in 1:Ny
        (mask[i, j] && both[i, j] && isfinite(A[i, j])) || continue
        w = cosd(φc[j]); num += w * A[i, j]; den += w
    end
    return den == 0 ? NaN : num / den
end
"""Mean floe thickness where the concentration passes ICE_EDGE: volume over concentration, area-weighted."""
function thickness(ℵ, V, mask)
    num = den = 0.0
    for i in 1:Nx, j in 1:Ny
        (mask[i, j] && both[i, j] && isfinite(ℵ[i, j]) && ℵ[i, j] > ICE_EDGE) || continue
        w = cosd(φc[j]); num += w * V[i, j]; den += w * ℵ[i, j]
    end
    return den == 0 ? NaN : num / den
end

# ── print ─────────────────────────────────────────────────────────────────────────────────────
cell(v, fmt) = isnan(v) ? lpad("--", 6) : Printf.format(Printf.Format(fmt), v)
println("\n", "="^96)
println("SEA ICE, $RUN against GLORYS, monthly means")
println("  ℵ concentration; h floe thickness (m) where ℵ > $ICE_EDGE; V = h ℵ volume per area (m). cos(lat)-weighted.")
println("="^96)
ice_series = Dict()
for (name, m) in REGIONS
    @printf("\n  %s\n  %-8s | %6s %6s | %6s %6s | %6s %6s\n", uppercase(name), "month", "ℵ GLO", "ℵ run", "h GLO", "h run", "V GLO", "V run")
    s = (gℵ = Float64[], rℵ = Float64[], gV = Float64[], rV = Float64[])
    for k in months
        vals = (region_mean(gℵ[k], m), region_mean(rℵ[k], m), thickness(gℵ[k], gV[k], m),
                thickness(rℵ[k], rV[k], m), region_mean(gV[k], m), region_mean(rV[k], m))
        @printf("  %-8s | %s %s | %s %s | %s %s\n", label(k), cell(vals[1], "%6.2f"), cell(vals[2], "%6.2f"),
                cell(vals[3], "%6.2f"), cell(vals[4], "%6.2f"), cell(vals[5], "%6.3f"), cell(vals[6], "%6.3f"))
        push!(s.gℵ, vals[1]); push!(s.rℵ, vals[2]); push!(s.gV, vals[5]); push!(s.rV, vals[6])
    end
    ice_series[name] = s
end

println("\n  ICE VOLUME LOST, November to February (monthly means), and the fresh water it gives if it melts in place")
@printf("  %-14s %-9s | %9s %9s | %12s %12s\n", "region", "summer", "ΔV GLORYS", "ΔV run", "fresh GLORYS", "fresh run")
for (name, _) in REGIONS, y in unique(first.(months))
    a, b = findfirst(==((y, 11)), months), findfirst(==((y + 1, 2)), months)
    (a === nothing || b === nothing) && continue
    s = ice_series[name]
    dg, dr = s.gV[a] - s.gV[b], s.rV[a] - s.rV[b]
    @printf("  %-14s %d-%02d   | %9.3f %9.3f | %12.3f %12.3f\n", name, y, (y + 1) % 100, dg, dr,
            dg * MELT_TO_FRESH, dr * MELT_TO_FRESH)
end
println("  (ΔV > 0: volume lost. Drift is not separated: a region that exports ice loses volume without melting it.)")

# ── plots ─────────────────────────────────────────────────────────────────────────────────────
x = [Dates.value(DateTime(k[1], k[2], 15) - start) / 86400e3 for k in months]
ticks = [(Dates.value(DateTime(k[1], k[2], 1) - start) / 86400e3, label(k)) for k in months if k[2] in (6, 9, 12, 3)]
fig = Figure(size = (1200, 300 * length(REGIONS)))
for (r, (name, m)) in enumerate(REGIONS)
    s = ice_series[name]
    for (c, (g, rr, what)) in enumerate(((s.gℵ, s.rℵ, "concentration"), (s.gV, s.rV, "volume per area (m)")))
        ax = Axis(fig[r, c]; title = "$name ($(count(m .& both)) cells): $what", xticklabelrotation = π / 4,
                  xticks = (first.(ticks), last.(ticks)))
        scatterlines!(ax, x, g; color = :black, markersize = 6, label = "GLORYS")
        scatterlines!(ax, x, rr; color = :firebrick, markersize = 6, label = RUN)
        r == 1 && c == 1 && axislegend(ax; position = :rt)
    end
end
out = joinpath(OUTDIR, "$(RUN)_sea_ice_vs_glorys_series.png")
save(out, fig); @info "wrote $out"

# ℵ maps over each summer, Oct-Mar, south of 58S: GLORYS above the run
summers = [y for y in unique(first.(months)) if all(k -> k in months, ((y, 10), (y, 11), (y, 12), (y + 1, 1), (y + 1, 2), (y + 1, 3)))]
if !isempty(summers)
    jj = findall(φ -> φ < -58, φc)
    fig = Figure(size = (1700, 280 * 2 * length(summers)))
    hm = nothing
    for (s, y) in enumerate(summers)
        for (c, k) in enumerate(((y, 10), (y, 11), (y, 12), (y + 1, 1), (y + 1, 2), (y + 1, 3)))
            for (row, (A, who)) in enumerate(((gℵ, "GLORYS"), (rℵ, "run")))
                ax = Axis(fig[2(s - 1) + row, c]; title = "$who $(label(k))", aspect = DataAspect(),
                          xticklabelsvisible = false, yticklabelsvisible = c == 1)
                hm = heatmap!(ax, λc, φc[jj], A[k][:, jj]; colormap = :ice, colorrange = (0, 1), nan_color = :gray80)
            end
        end
    end
    Colorbar(fig[:, 7], hm; label = "sea-ice concentration")
    out = joinpath(OUTDIR, "$(RUN)_sea_ice_vs_glorys_summer.png")
    save(out, fig); @info "wrote $out"
end
