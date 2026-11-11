# ==============================================================================
# compare_glorys_bsose.jl -- the two reanalyses against each other at the surface, month by month:
# GLORYS monthly minus B-SOSE monthly over the SOWinWaRP domain (90-150E, 70-45S, the nest's 1/6-degree
# cells). No model run is read: this is the parents alone.
#
#     source .copernicus_env      # only while GLORYS files are missing: the GLORYS surface is downloaded
#     julia --project=. tools/compare_glorys_bsose.jl
#     FIRST_MONTH=2014-06 LAST_MONTH=2016-05 S_LIM=0.5 julia --project=. tools/compare_glorys_bsose.jl
#
# WHAT IS COMPARED. Each month's mean of the top level: GLORYS's (0.49 m, cmems_mod_glo_phy_my_0.083deg_P1M-m,
# stamped on the 1st of its month) and B-SOSE i156's (0-4.2 m, the record whose averaging window is that
# month). Both are put on the domain's 1/6-degree cells the same way, by averaging the wet source points
# inside each cell (GLORYS ~2x2 points, B-SOSE 1-3), so neither is interpolated differently from the other.
# Cells count where both have water. Potential temperature and practical salinity in both.
#
# DOWNLOADS. Only GLORYS's top level, for the domain and months, through NumericalEarth's CopernicusMarine
# extension: a few MB a month, into GLORYS_DIR, with "_-1.0_0.0" (the depth range) in the names so they
# never stand in for the full-depth files the runs use. Needs COPERNICUSMARINE_SERVICE_USERNAME and
# _PASSWORD in the environment (source .copernicus_env). B-SOSE is read straight from its files.
#
# WRITES (OUTDIR, default plots/):
#     glorys_minus_bsose_S.png, _T.png    one surface map per month
#     glorys_minus_bsose_mean.png         ΔT and ΔS over all months, and ΔS over Dec-Feb
# and on stdout: the domain and each latitude band, month by month; the Antarctic coast (cells within
# COAST_CELLS of land, south of COAST_SOUTH_OF) against the offshore water there, by 10-degree longitude;
# and the 15 coastal cells where GLORYS is saltiest against B-SOSE in Dec-Feb.
#
# ENV
#     FIRST_MONTH, LAST_MONTH      yyyy-mm (default 2014-06, 2016-05: the 2-year runs' span)
#     GLORYS_DIR                   default <data>/glorys_monthly
#     OUTDIR                       default plots
#     COAST_CELLS, COAST_SOUTH_OF  default 2, -60
#     S_LIM, T_LIM                 colour limits (default 0.3 g/kg and 1.0 degC: the broad pattern; larger
#                                  values, at the coast say, are drawn at the end colour)
# ==============================================================================

using Oceananigans
using NumericalEarth
using NumericalEarth.DataWrangling: BoundingBox, Metadatum, metadata_path, dataset_variable_name
using NumericalEarth.DataWrangling.GLORYS: GLORYSMonthly
using CopernicusMarine          # NumericalEarth downloads GLORYS through it
import Downloads
using NCDatasets, CairoMakie, Printf, Statistics, Dates

include(joinpath(@__DIR__, "..", "src", "setup_bsose.jl"))

const OUTDIR     = get(ENV, "OUTDIR", "plots")
const GLORYS_DIR = get(ENV, "GLORYS_DIR", joinpath(default_bsose_directory(), "glorys_monthly"))
const S_LIM      = parse(Float64, get(ENV, "S_LIM", "0.3"))
const T_LIM      = parse(Float64, get(ENV, "T_LIM", "1.0"))
const COAST_CELLS    = parse(Int, get(ENV, "COAST_CELLS", "2"))
const COAST_SOUTH_OF = parse(Float64, get(ENV, "COAST_SOUTH_OF", "-60"))
mkpath(OUTDIR); mkpath(GLORYS_DIR)

f64(A) = Float64.(coalesce.(A, NaN))

# ── the domain: the nest's cells (examples/sowinwarp_nested.jl) ───────────────────────────────
const λW, λE, φS, φN = 90.0, 150.0, -70.0, -45.0
const Δλ = Δφ = 1 / 6
const Nx, Ny = round(Int, (λE - λW) / Δλ), round(Int, (φN - φS) / Δφ)   # 360 x 150
λc = [λW + (i - 0.5) * Δλ for i in 1:Nx]
φc = [φS + (j - 0.5) * Δφ for j in 1:Ny]

month_of(s) = (d = Date(s, dateformat"yyyy-mm"); (year(d), month(d)))
first_month = month_of(get(ENV, "FIRST_MONTH", "2014-06"))
last_month  = month_of(get(ENV, "LAST_MONTH", "2016-05"))
bsose = BSOSEMonthly(iteration = 156)
bsose_record = Dict((year(bsose_window_center(d)), month(bsose_window_center(d))) => d
                    for d in all_dates(bsose, :temperature))
months = Tuple{Int,Int}[]
let d = Date(first_month..., 1)
    while (year(d), month(d)) <= last_month
        push!(months, (year(d), month(d))); d += Month(1)
    end
end
absent = filter(k -> !haskey(bsose_record, k), months)
isempty(absent) || @warn "B-SOSE has no record for $(length(absent)) months, left out: $(absent)"
filter!(k -> haskey(bsose_record, k), months)
isempty(months) && error("no month between $first_month and $last_month has a B-SOSE record")
label(k) = @sprintf("%d-%02d", k...)
@info "GLORYS - B-SOSE over $(λW)-$(λE)E, $(-φS)-$(-φN)S ($(Nx)x$(Ny) cells), $(length(months)) months $(label(first(months)))..$(label(last(months)))"

"""
Mean of the wet `values` at (`lons`, `lats`) inside each cell; NaN where none. A point on a cell edge
(GLORYS's 1/12-degree points fall on every other edge of the 1/6-degree cells) counts half in each
neighbour, or every mean would sit 1/24 degree off its cell's centre.
"""
function cell_means(values, lons, lats)
    total, weight = zeros(Nx, Ny), zeros(Nx, Ny)
    shares(p) = (n = round(Int, p); abs(p - n) < 1e-6 ? ((n, 0.5), (n + 1, 0.5)) : ((floor(Int, p) + 1, 1.0),))
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

# ── GLORYS: the top level, downloaded once ────────────────────────────────────────────────────
const GLORYS_REGION = BoundingBox(longitude = (λW, λE), latitude = (φS, φN), z = (-1.0, 0.0))
glorys_meta(name, k) = Metadatum(name; dataset = GLORYSMonthly(), date = DateTime(k[1], k[2], 1),
                                 region = GLORYS_REGION, dir = GLORYS_DIR)
missing_files = [metadata_path(glorys_meta(n, k)) for n in (:temperature, :salinity) for k in months
                 if !isfile(metadata_path(glorys_meta(n, k)))]
if !isempty(missing_files)
    haskey(ENV, "COPERNICUSMARINE_SERVICE_USERNAME") && haskey(ENV, "COPERNICUSMARINE_SERVICE_PASSWORD") ||
        error("$(length(missing_files)) GLORYS surface files to download, but no Copernicus credentials in the " *
              "environment: run `source .copernicus_env` first")
    @info "downloading $(length(missing_files)) GLORYS monthly surface files into $GLORYS_DIR"
    for n in (:temperature, :salinity), k in months
        Downloads.download(glorys_meta(n, k))
    end
end

function glorys_surface(name, k)
    md = glorys_meta(name, k)
    Dataset(metadata_path(md)) do d
        v = d[dataset_variable_name(md)]
        x = ndims(v) == 4 ? v[:, :, 1, 1] : ndims(v) == 3 ? v[:, :, 1] : v[:, :]
        return cell_means(f64(x), f64(d["longitude"][:]), f64(d["latitude"][:]))
    end
end

# ── B-SOSE: the top level, straight from its files ────────────────────────────────────────────
# Coordinates and sea floor from the surface-salt-flux file (as tools/bsose_masks.jl), cut to the domain:
# land is 0 in the T and S files, so it is masked by Depth, not by value.
bsose_box = let d = Dataset(joinpath(bsose.dir, resolve_bsose_filename(bsose.dir, :surface_salt_flux, bsose.iteration)))
    xc, yc = f64(d["XC"][:]), f64(d["YC"][:])
    is, js = findall(x -> λW <= x < λE, xc), findall(y -> φS <= y < φN, yc)
    is, js = first(is):last(is), first(js):last(js)
    wet = f64(d["Depth"][is, js]) .> 0
    close(d)
    (; is, js, λ = xc[is], φ = yc[js], wet)
end

function bsose_surface(name, k)
    md = Metadatum(name; dataset = bsose, date = bsose_record[k])
    Dataset(metadata_path(md)) do d
        key = dataset_variable_name(md)
        key = haskey(d, key) ? key : haskey(d, uppercase(key)) ? uppercase(key) : titlecase(key)
        t = find_bsose_time_index(d, md.dates)
        x = f64(d[key][bsose_box.is, bsose_box.js, 1, t])
        return cell_means(ifelse.(bsose_box.wet, x, NaN), bsose_box.λ, bsose_box.φ)
    end
end

# ── GLORYS - B-SOSE, month by month ───────────────────────────────────────────────────────────
ΔT, ΔS = Dict{Tuple{Int,Int},Matrix{Float64}}(), Dict{Tuple{Int,Int},Matrix{Float64}}()
wet = trues(Nx, Ny)                                  # water in both, every month
for k in months
    for (Δ, name) in ((ΔT, :temperature), (ΔS, :salinity))
        G, B = glorys_surface(name, k), bsose_surface(name, k)
        Δ[k] = G .- B
        wet .&= isfinite.(Δ[k])
    end
end
for k in months; ΔT[k][.!wet] .= NaN; ΔS[k][.!wet] .= NaN; end
@info "$(count(wet)) cells with water in both reanalyses"

# ── statistics ────────────────────────────────────────────────────────────────────────────────
const BANDS = [(-70.0, -65.0), (-65.0, -60.0), (-60.0, -55.0), (-55.0, -50.0), (-50.0, -45.0)]
function wmean(A, keep = wet; lo = -90.0, hi = 90.0)
    num = den = 0.0
    for i in 1:Nx, j in 1:Ny
        (keep[i, j] && lo <= φc[j] < hi && isfinite(A[i, j])) || continue
        w = cosd(φc[j]); num += w * A[i, j]; den += w
    end
    return den == 0 ? NaN : num / den
end
rms(A) = sqrt(wmean(A .^ 2))
cell(v, fmt) = isnan(v) ? lpad("--", 8) : Printf.format(Printf.Format(fmt), v)

# Coastal: wet cells south of COAST_SOUTH_OF within COAST_CELLS of a cell where either reanalysis has land
coastal = falses(Nx, Ny)
for i in 1:Nx, j in 1:Ny
    (wet[i, j] && φc[j] < COAST_SOUTH_OF) || continue
    coastal[i, j] = any(.!wet[max(1, i - COAST_CELLS):min(Nx, i + COAST_CELLS), max(1, j - COAST_CELLS):min(Ny, j + COAST_CELLS)])
end
offshore = wet .& .!coastal .& [φ < COAST_SOUTH_OF for _ in 1:Nx, φ in φc]
const LON_BINS = [(λ, λ + 10) for λ in Int(λW):10:Int(λE) - 10]
summer(k) = k[2] in (12, 1, 2)
smonths = filter(summer, months)
mean_over(Δ, ms) = isempty(ms) ? fill(NaN, Nx, Ny) : sum(Δ[k] for k in ms) ./ length(ms)

println("\n", "="^96)
println("SURFACE, monthly means, GLORYS minus B-SOSE, $(λW)-$(λE)E $(-φS)-$(-φN)S")
println("  cos(lat)-weighted over cells with water in both. Positive ΔS: GLORYS SALTIER; positive ΔT: WARMER.")
println("="^96)
@printf("\n  whole domain\n  %-8s | %8s %8s | %8s %8s\n", "month", "ΔT", "rms T", "ΔS", "rms S")
for k in months
    @printf("  %-8s | %s %s | %s %s\n", label(k), cell(wmean(ΔT[k]), "%+8.3f"), cell(rms(ΔT[k]), "%8.3f"),
            cell(wmean(ΔS[k]), "%+8.3f"), cell(rms(ΔS[k]), "%8.3f"))
end
for (Δ, name, fmt) in ((ΔT, "ΔT (°C)", "%+7.2f"), (ΔS, "ΔS (g/kg)", "%+7.3f"))
    @printf("\n  %s by latitude band\n  %-8s |", name, "month")
    for (lo, hi) in BANDS; @printf(" %7s", "$(Int(-lo))-$(Int(-hi))S"); end
    println()
    for k in months
        @printf("  %-8s |", label(k))
        for (lo, hi) in BANDS; print(" ", cell(wmean(Δ[k]; lo, hi), fmt)); end
        println()
    end
end

@printf("\n  ΔS (g/kg) south of %.0fS: the COAST (%d cells within %d of land) and OFFSHORE (%d cells), coast by longitude\n",
        -COAST_SOUTH_OF, count(coastal), COAST_CELLS, count(offshore))
@printf("  %-8s | %8s %8s |", "month", "coast", "offshore")
for b in LON_BINS; @printf(" %8s", "$(b[1])-$(b[2])E"); end
println()
for k in months
    @printf("  %-8s | %s %s |", label(k), cell(wmean(ΔS[k], coastal), "%+8.3f"), cell(wmean(ΔS[k], offshore), "%+8.3f"))
    for b in LON_BINS
        m = coastal .& [b[1] <= λc[i] < b[2] for i in 1:Nx, _ in 1:Ny]
        print(" ", cell(wmean(ΔS[k], m), "%+8.3f"))
    end
    println()
end

ΔS_all, ΔS_summer = mean_over(ΔS, months), mean_over(ΔS, smonths)
ranked = sort([(ΔS_summer[i, j], i, j) for i in 1:Nx, j in 1:Ny if coastal[i, j] && isfinite(ΔS_summer[i, j])], rev = true)
println("\n  The 15 coastal cells where GLORYS is saltiest against B-SOSE in Dec-Feb, and over all months")
@printf("  %8s %8s | %10s %10s\n", "lon", "lat", "ΔS DJF", "ΔS all")
for (v, i, j) in ranked[1:min(15, end)]
    @printf("  %7.2fE %7.2fS | %+10.3f %+10.3f\n", λc[i], -φc[j], v, ΔS_all[i, j])
end
println()

# ── figures ───────────────────────────────────────────────────────────────────────────────────
for (Δ, v, unit, lim) in ((ΔS, "S", "g/kg", S_LIM), (ΔT, "T", "°C", T_LIM))
    nm = length(months); ncol = min(nm, 6); nrow = cld(nm, ncol)
    fig = Figure(size = (260 * ncol + 120, 190 * nrow + 120))
    Label(fig[0, 1:ncol], "surface $v, monthly mean, GLORYS − B-SOSE", fontsize = 18, font = :bold)
    local hm
    for (n, k) in enumerate(months)
        r, c = fldmod1(n, ncol)
        ax = Axis(fig[r, c], title = label(k), titlesize = 12, xticklabelsvisible = r == nrow,
                  yticklabelsvisible = c == 1, xticklabelsize = 9, yticklabelsize = 9)
        hm = heatmap!(ax, λc, φc, Δ[k], colormap = :balance, colorrange = (-lim, lim), nan_color = :gray80)
    end
    Colorbar(fig[1:nrow, ncol + 1], hm, label = "Δ$v ($unit)")
    Label(fig[nrow + 1, 1:ncol], "red: GLORYS higher; blue: lower. Grey: land in either reanalysis. " *
          "Colour limit ±$lim $unit (set $(v)_LIM); larger values are drawn at the end colour.", fontsize = 11)
    f = joinpath(OUTDIR, "glorys_minus_bsose_$(v).png"); save(f, fig); @info "wrote $f"
end

let panels = (("ΔT, all months", mean_over(ΔT, months), T_LIM, "°C"),
              ("ΔS, all months", ΔS_all, S_LIM, "g/kg"),
              ("ΔS, Dec-Feb", ΔS_summer, S_LIM, "g/kg"))
    fig = Figure(size = (900, 1050))
    Label(fig[0, 1], "surface, GLORYS − B-SOSE, $(label(first(months)))..$(label(last(months)))", fontsize = 18, font = :bold)
    for (r, (title, A, lim, unit)) in enumerate(panels)
        ax = Axis(fig[r, 1], title = title, xticklabelsvisible = r == length(panels))
        hm = heatmap!(ax, λc, φc, A, colormap = :balance, colorrange = (-lim, lim), nan_color = :gray80)
        r > 1 && contour!(ax, λc, φc, Float64.(coastal), levels = [0.5], color = :black, linewidth = 0.6)
        Colorbar(fig[r, 2], hm, label = "$(first(title, 2)) ($unit)")
    end
    Label(fig[length(panels) + 1, 1], "red: GLORYS higher. Black outline (S panels): the coastal cells. " *
          "Colour limits ±$T_LIM °C, ±$S_LIM g/kg (T_LIM, S_LIM); larger values at the end colour.", fontsize = 11)
    f = joinpath(OUTDIR, "glorys_minus_bsose_mean.png"); save(f, fig); @info "wrote $f"
end
