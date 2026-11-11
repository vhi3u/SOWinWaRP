# ==============================================================================
# compare_glorys.jl -- a GLORYS-nested run against GLORYS, month by month: the SURFACE and the MIDLON
# SECTION. compare_bsose.jl's counterpart for a run whose parent is GLORYS: the two reanalyses differ at
# the Antarctic coast by more than most model biases (tools/compare_glorys_bsose.jl), so such a run is
# judged against its own parent.
#
#     source .copernicus_env      # only while GLORYS files are missing
#     julia --project=. tools/compare_glorys.jl                        # 2Y_nested_glorys_baseline
#     S_LIM=0.5 julia --project=. tools/compare_glorys.jl 2Y_nested_glorys_baseline
#
# WHAT IS COMPARED. GLORYS monthly means (cmems_mod_glo_phy_my_0.083deg_P1M-m, stamped on the 1st of the
# month they average) against the run's daily output averaged over the same calendar month (the t = 0
# snapshot and months with fewer than MIN_DAYS days left out).
#     surface   the run's top cell (read off its levels: 0-4.2 m on B-SOSE's, ~0-10.4 m on the Bouvet
#               example's) against GLORYS's levels averaged over that same layer, thickness-weighted (for
#               4.2 m, the same files to 5 m that compare_glorys_bsose.jl downloads)
#     midlon    the run's latitude-depth section (~120E) against GLORYS's points inside that column of
#               cells, each of the run's levels taking the thickness-weighted mean of the GLORYS levels it
#               overlaps
# Horizontally GLORYS's points are averaged inside each of the run's 1/6-degree cells (~2x2 points; a point
# on a cell edge counts half in each neighbour). Cells count where the run is wet and GLORYS has water.
# Potential temperature and practical salinity, as the run is initialised and forced.
#
# DOWNLOADS (through NumericalEarth's CopernicusMarine extension, into GLORYS_DIR, credentials from
# `source .copernicus_env`): the top 5 m over the run's box (shared with compare_glorys_bsose.jl), and the
# full depth of the one column of cells the section runs through. A few MB a month each.
#
# WRITES (OUTDIR, default plots/):
#     <RUN>_vs_glorys_surface_S.png, _T.png    one surface map per month, run - GLORYS
#     <RUN>_vs_glorys_surface_mean.png         ΔT and ΔS over all months, ΔS over Dec-Feb, coast outlined
#     <RUN>_vs_glorys_midlon_S.png, _T.png     one section panel per month, run - GLORYS
#     <RUN>_vs_glorys_series.png               surface and section bias, month by month
# and on stdout: the surface by latitude band, the coast against offshore south of 60S by 10 degrees of
# longitude, the 15 coastal cells where the run is saltiest in Dec-Feb, the section by layer, and the drift
# per year (last 12 months minus first 12) for 24 months or more.
#
# ENV
#     DIR, OUTDIR, MIN_DAYS        as compare_bsose.jl (., plots, 20)
#     GLORYS_DIR                   default <data>/glorys_monthly
#     COAST_CELLS, COAST_SOUTH_OF  default 2, -60
#     MIDLON_ZMAX                  depth drawn in the section maps, m, or "full" (default 1000)
#     S_LIM, T_LIM, SS_LIM, ST_LIM colour limits, surface and section (default 0.3 g/kg, 1.0 degC, 0.2 g/kg,
#                                  1.0 degC; larger values are drawn at the end colour)
# ==============================================================================

using NumericalEarth
using NumericalEarth.DataWrangling: BoundingBox, Metadatum, metadata_path, dataset_variable_name
using NumericalEarth.DataWrangling.GLORYS: GLORYSMonthly
using CopernicusMarine          # NumericalEarth downloads GLORYS through it
import Downloads
using NCDatasets, CairoMakie, Printf, Statistics, Dates

include(joinpath(@__DIR__, "..", "src", "setup_bsose.jl"))   # default_bsose_directory
include(joinpath(@__DIR__, "model_grid.jl"))                 # file_levels, file_kbot, section_column

const RUN = length(ARGS) > 0 ? ARGS[1] : "2Y_nested_glorys_baseline"
const DIR        = get(ENV, "DIR", ".")
const OUTDIR     = get(ENV, "OUTDIR", "plots")
const MIN_DAYS   = parse(Int, get(ENV, "MIN_DAYS", "20"))
const GLORYS_DIR = get(ENV, "GLORYS_DIR", joinpath(default_bsose_directory(), "glorys_monthly"))
const S_LIM, T_LIM   = parse(Float64, get(ENV, "S_LIM", "0.3")), parse(Float64, get(ENV, "T_LIM", "1.0"))
const SS_LIM, ST_LIM = parse(Float64, get(ENV, "SS_LIM", "0.2")), parse(Float64, get(ENV, "ST_LIM", "1.0"))
const ZMAX = (s = get(ENV, "MIDLON_ZMAX", "1000"); s == "full" ? Inf : parse(Float64, s))
const COAST_CELLS    = parse(Int, get(ENV, "COAST_CELLS", "2"))
const COAST_SOUTH_OF = parse(Float64, get(ENV, "COAST_SOUTH_OF", "-60"))
mkpath(OUTDIR); mkpath(GLORYS_DIR)

f64(A) = Float64.(coalesce.(A, NaN))

# ── the run's grid, mask and months ───────────────────────────────────────────────────────────
const SURF = joinpath(DIR, "$(RUN)_surface.nc")
const SECT = joinpath(DIR, "$(RUN)_midlon.nc")
isfile(SURF) || error("no $SURF")
HAVE_SECT = isfile(SECT)
HAVE_SECT || @warn "no $SECT: surface comparison only"

ds = Dataset(SURF)
λc, φc = f64(ds["λ_caa"][:]), f64(ds["φ_aca"][:])
Nx, Ny = length(λc), length(φc)
Δλ, Δφ = λc[2] - λc[1], φc[2] - φc[1]
λW, φS = λc[1] - Δλ / 2, φc[1] - Δφ / 2
λE, φN = λW + Nx * Δλ, φS + Ny * Δφ
surf_times = f64(ds["time"][:])
start = DateTime(string(ds.attrib["start_date"]))
model_wet = let m = ds["inactive_nodes_ccc"]; (ndims(m) == 3 ? m[:, :, end] : m[:, :]) .== 0 end
close(ds)

if HAVE_SECT
    g = Dataset(SECT)
    zc, zf, _ = file_levels(g)
    kb = file_kbot(g, zc, zf)[1]
    kbot = size(kb, 1) == 1 ? vec(kb) : kb[section_column(SECT, Nx), :]
    sect_times = f64(g["time"][:])
    close(g)
    MID = section_column(SECT, Nx)
    Nz = length(zc)
else
    sect_times = Float64[]
end

# The run's top cell, which the surface file holds, and the GLORYS layer it is compared with: 4.2 m on
# B-SOSE's levels (our runs), ~10.4 m on the Bouvet example's (nested_glorys_mini.jl)
const TOP = HAVE_SECT ? zf[end] - zf[end - 1] : (@warn("no section file for the levels: taking the top cell as 4.2 m"); 4.2)
# GLORYS levels down to SURFACE_Z: to 5 m for a 4.2 m top cell (the files compare_glorys_bsose.jl shares),
# else a few metres past the cell, so its lowest GLORYS level is whole
const SURFACE_Z = TOP <= 4.25 ? -5.0 : -ceil(TOP + 3)
const TOP_LABEL = @sprintf("%.1f", TOP)

function by_month(times)
    g = Dict{Tuple{Int,Int},Vector{Int}}()
    for (n, t) in enumerate(times)
        t == 0 && continue
        d = start + Millisecond(round(Int, 1000t))
        push!(get!(g, (year(d), month(d)), Int[]), n)
    end
    return g
end
whole(g) = sort([k for (k, v) in g if length(v) >= MIN_DAYS])
surf_groups, sect_groups = by_month(surf_times), by_month(sect_times)
surf_months = whole(surf_groups)
sect_months = HAVE_SECT ? whole(sect_groups) : Tuple{Int,Int}[]
all_months = sort(union(surf_months, sect_months))
isempty(all_months) && error("no month has $MIN_DAYS+ days of output")
label(k) = @sprintf("%d-%02d", k...)
@info "$RUN against GLORYS: $(Nx)x$(Ny) over $(round(λW, digits = 2))-$(round(λE, digits = 2))E, " *
      "$(length(all_months)) months $(label(first(all_months)))..$(label(last(all_months)))" *
      (HAVE_SECT ? @sprintf("; section at %.2fE (column %d)", λc[MID], MID) : "")

# ── GLORYS files, downloaded once ─────────────────────────────────────────────────────────────
r2(x) = round(x, digits = 6)
const SURFACE_REGION = BoundingBox(longitude = (r2(λW), r2(λE)), latitude = (r2(φS), r2(φN)), z = (SURFACE_Z, 0.0))
const COLUMN_REGION  = HAVE_SECT ? BoundingBox(longitude = (r2(λc[MID] - Δλ / 2), r2(λc[MID] + Δλ / 2)),
                                               latitude = (r2(φS), r2(φN))) : nothing
glorys_meta(name, k, region) = Metadatum(name; dataset = GLORYSMonthly(), date = DateTime(k[1], k[2], 1),
                                         region, dir = GLORYS_DIR)
wanted = Any[(n, k, SURFACE_REGION) for n in (:temperature, :salinity) for k in surf_months]
HAVE_SECT && append!(wanted, [(n, k, COLUMN_REGION) for n in (:temperature, :salinity) for k in sect_months])
missing_files = filter(w -> !isfile(metadata_path(glorys_meta(w...))), wanted)
if !isempty(missing_files)
    haskey(ENV, "COPERNICUSMARINE_SERVICE_USERNAME") && haskey(ENV, "COPERNICUSMARINE_SERVICE_PASSWORD") ||
        error("$(length(missing_files)) GLORYS files to download, but no Copernicus credentials in the " *
              "environment: run `source .copernicus_env` first")
    @info "downloading $(length(missing_files)) GLORYS monthly files into $GLORYS_DIR"
    foreach(w -> Downloads.download(glorys_meta(w...)), missing_files)
end

# ── averaging GLORYS onto the run's cells ─────────────────────────────────────────────────────
"""Cells a point at fractional index `p` falls in, with weights: half each on an edge, else all in one."""
shares(p) = (n = round(Int, p); abs(p - n) < 1e-6 ? ((n, 0.5), (n + 1, 0.5)) : ((floor(Int, p) + 1, 1.0),))

"""Mean of the finite `values[a, b]` at (`lons[a]`, `lats[b]`) inside each of the run's (Nx, Ny) cells; NaN where none."""
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

"""GLORYS level interfaces (m, positive down) from its level centres: half-way between centres, the first at 0."""
glorys_edges(d) = vcat(0.0, (d[1:end-1] .+ d[2:end]) ./ 2, d[end] + (length(d) > 1 ? (d[end] - d[end-1]) / 2 : d[end]))

"""
Thickness-weighted mean over depths `top`..`bottom` (m, positive down) of `x[:, :, n]` on GLORYS levels with
interfaces `edges`: each level counts by how much of it lies in the layer, over the levels with water.
"""
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

read_glorys(md) = Dataset(metadata_path(md)) do d
    v = d[dataset_variable_name(md)]
    x = f64(ndims(v) == 4 ? v[:, :, :, 1] : v[:, :, :])
    (x, f64(d["longitude"][:]), f64(d["latitude"][:]), f64(d["depth"][:]))
end

"""GLORYS over the run's top cell (0-TOP m), on its (Nx, Ny) cells."""
function glorys_surface(name, k)
    x, lon, lat, depth = read_glorys(glorys_meta(name, k, SURFACE_REGION))
    return cell_means(layer_mean(x, glorys_edges(depth), 0.0, TOP), lon, lat)
end

"""GLORYS on the run's section, (Ny, Nz): the column's points per latitude cell, then per model level."""
function glorys_section(name, k)
    x, lon, lat, depth = read_glorys(glorys_meta(name, k, COLUMN_REGION))
    edges = glorys_edges(depth)
    S = fill(NaN, Ny, Nz)
    for kk in 1:Nz                                   # the run's level kk spans depths -zf[kk+1] .. -zf[kk]
        layer = layer_mean(x, edges, -zf[kk + 1], -zf[kk])
        S[:, kk] = column_means(layer, lon, lat)
    end
    return S
end
"""Mean of the finite `layer[a, b]` inside column MID's cells, per latitude cell: the download is padded
beyond the column, so only the points inside it count (half for a point on its edge)."""
function column_means(layer, lon, lat)
    total, weight = zeros(Ny), zeros(Ny)
    west = λc[MID] - Δλ / 2
    for b in axes(layer, 2), a in axes(layer, 1)
        v = layer[a, b]; isfinite(v) || continue
        wa = sum((w for (i, w) in shares((lon[a] - west) / Δλ) if i == 1); init = 0.0)
        wa == 0 && continue
        for (j, wj) in shares((lat[b] - φS) / Δφ)
            1 <= j <= Ny || continue
            total[j] += wa * wj * v; weight[j] += wa * wj
        end
    end
    return [weight[j] > 0 ? total[j] / weight[j] : NaN for j in 1:Ny]
end

# ── run minus GLORYS, month by month ──────────────────────────────────────────────────────────
sΔT, sΔS = Dict{Tuple{Int,Int},Matrix{Float64}}(), Dict{Tuple{Int,Int},Matrix{Float64}}()   # (Nx, Ny)
cΔT, cΔS = Dict{Tuple{Int,Int},Matrix{Float64}}(), Dict{Tuple{Int,Int},Matrix{Float64}}()   # (Ny, Nz)
surface_mean(d, var, idx) = (v = d[var]; sum(f64(ndims(v) == 4 ? v[:, :, 1, n] : v[:, :, n]) for n in idx) ./ length(idx))
section_mean(d, var, idx) = sum(f64(d[var][1, :, :, n]) for n in idx) ./ length(idx)
wet_section = HAVE_SECT ? [kbot[j] !== nothing && k >= kbot[j] for j in 1:Ny, k in 1:Nz] : falses(0, 0)

dsurf = Dataset(SURF)
for k in surf_months, (D, name, var) in ((sΔT, :temperature, "T"), (sΔS, :salinity, "S"))
    D[k] = ifelse.(model_wet, surface_mean(dsurf, var, surf_groups[k]), NaN) .- glorys_surface(name, k)
end
close(dsurf)
if HAVE_SECT
    dsect = Dataset(SECT)
    for k in sect_months, (D, name, var) in ((cΔT, :temperature, "T"), (cΔS, :salinity, "S"))
        D[k] = ifelse.(wet_section, section_mean(dsect, var, sect_groups[k]), NaN) .- glorys_section(name, k)
    end
    close(dsect)
end
surface_wet = isempty(surf_months) ? model_wet : reduce(.&, (isfinite.(sΔS[k]) for k in surf_months))
@info "$(count(surface_wet)) surface cells wet in the run with GLORYS water"

# ── statistics ────────────────────────────────────────────────────────────────────────────────
const BANDS = [(-70.0, -65.0), (-65.0, -60.0), (-60.0, -55.0), (-55.0, -50.0), (-50.0, -45.0)]
function wmean(A, keep = nothing; lo = -90.0, hi = 90.0, ks = nothing)
    num = den = 0.0
    if ks === nothing                                            # (Nx, Ny) surface
        for i in axes(A, 1), j in axes(A, 2)
            (lo <= φc[j] < hi && isfinite(A[i, j]) && (keep === nothing || keep[i, j])) || continue
            w = cosd(φc[j]); num += w * A[i, j]; den += w
        end
    else                                                         # (Ny, Nz) section
        for j in axes(A, 1), k in ks
            (lo <= φc[j] < hi && isfinite(A[j, k])) || continue
            w = cosd(φc[j]); num += w * A[j, k]; den += w
        end
    end
    return den == 0 ? NaN : num / den
end
rms(A; ks = nothing) = sqrt(wmean(A .^ 2; ks))
cell(v, fmt) = isnan(v) ? lpad("--", 8) : Printf.format(Printf.Format(fmt), v)
function per_year(ms, v)
    length(ms) < 24 && return NaN
    yrs(k) = Dates.value(DateTime(k[1], k[2], 15) - start) / (86400e3 * 365.25)
    a, b = 1:12, (length(ms) - 11):length(ms)
    return (mean(v[b]) - mean(v[a])) / (mean(yrs.(ms[b])) - mean(yrs.(ms[a])))
end

coastal = falses(Nx, Ny)
for i in 1:Nx, j in 1:Ny
    (surface_wet[i, j] && φc[j] < COAST_SOUTH_OF) || continue
    coastal[i, j] = any(.!surface_wet[max(1, i - COAST_CELLS):min(Nx, i + COAST_CELLS), max(1, j - COAST_CELLS):min(Ny, j + COAST_CELLS)])
end
offshore = surface_wet .& .!coastal .& [φ < COAST_SOUTH_OF for _ in 1:Nx, φ in φc]
const LON_BINS = [(λ, λ + 10) for λ in 10 * floor(Int, λW / 10 + 1e-6):10:(10 * ceil(Int, λE / 10 - 1e-6) - 10)]
summer(k) = k[2] in (12, 1, 2)
mean_over(D, ms) = isempty(ms) ? fill(NaN, Nx, Ny) : sum(D[k] for k in ms) ./ length(ms)

println("\n", "="^96)
println("MONTHLY MEAN run minus GLORYS  ($RUN)")
println("  surface: the run's top cell against GLORYS over 0-$(TOP_LABEL) m. cos(lat)-weighted over cells wet in both.")
println("  Positive ΔT: the run too WARM; positive ΔS: too SALTY.")
println("="^96)
println("\n  SURFACE, whole domain")
@printf("  %-8s | %8s %8s | %8s %8s\n", "month", "ΔT", "rms T", "ΔS", "rms S")
for k in surf_months
    @printf("  %-8s | %s %s | %s %s\n", label(k), cell(wmean(sΔT[k]), "%+8.3f"), cell(rms(sΔT[k]), "%8.3f"),
            cell(wmean(sΔS[k]), "%+8.3f"), cell(rms(sΔS[k]), "%8.3f"))
end
@printf("  %-8s | %s %8s | %s\n", "per year", cell(per_year(surf_months, [wmean(sΔT[k]) for k in surf_months]), "%+8.3f"),
        "", cell(per_year(surf_months, [wmean(sΔS[k]) for k in surf_months]), "%+8.3f"))
for (name, D, fmt) in (("ΔT (°C)", sΔT, "%+7.2f"), ("ΔS (g/kg)", sΔS, "%+7.3f"))
    @printf("\n  SURFACE %s by latitude band\n  %-8s |", name, "month")
    for (lo, hi) in BANDS; @printf(" %7s", "$(Int(-lo))-$(Int(-hi))S"); end
    println()
    for k in surf_months
        @printf("  %-8s |", label(k))
        for (lo, hi) in BANDS; print(" ", cell(wmean(D[k]; lo, hi), fmt)); end
        println()
    end
end

@printf("\n  SURFACE ΔS (g/kg) south of %.0fS: the COAST (%d cells within %d of land) and OFFSHORE (%d cells), coast by longitude\n",
        -COAST_SOUTH_OF, count(coastal), COAST_CELLS, count(offshore))
@printf("  %-8s | %8s %8s |", "month", "coast", "offshore")
for b in LON_BINS; @printf(" %8s", "$(b[1])-$(b[2])E"); end
println()
for k in surf_months
    @printf("  %-8s | %s %s |", label(k), cell(wmean(sΔS[k], coastal), "%+8.3f"), cell(wmean(sΔS[k], offshore), "%+8.3f"))
    for b in LON_BINS
        print(" ", cell(wmean(sΔS[k], coastal .& [b[1] <= λc[i] < b[2] for i in 1:Nx, _ in 1:Ny]), "%+8.3f"))
    end
    println()
end

smonths = filter(summer, surf_months)
ΔS_all, ΔS_summer = mean_over(sΔS, surf_months), mean_over(sΔS, smonths)
ranked = sort([(ΔS_summer[i, j], i, j) for i in 1:Nx, j in 1:Ny if coastal[i, j] && isfinite(ΔS_summer[i, j])], rev = true)
println("\n  The 15 coastal cells where the run is saltiest against GLORYS in Dec-Feb, and over all months")
@printf("  %8s %8s | %10s %10s\n", "lon", "lat", "ΔS DJF", "ΔS all")
for (v, i, j) in ranked[1:min(15, end)]
    @printf("  %7.2fE %7.2fS | %+10.3f %+10.3f\n", λc[i], -φc[j], v, ΔS_all[i, j])
end

if !isempty(sect_months)
    UPPER = [k for k in 1:Nz if -zc[k] <= 200]
    DEEP  = [k for k in 1:Nz if -zc[k] > 200]
    @printf("\n  MIDLON SECTION at %.1fE, by layer\n", λc[MID])
    @printf("  %-8s | %8s %8s %8s | %8s %8s %8s\n", "month", "ΔT 0-200", "200-flr", "rms 0-200",
            "ΔS 0-200", "200-flr", "rms 0-200")
    for k in sect_months
        @printf("  %-8s | %s %s %s | %s %s %s\n", label(k),
                cell(wmean(cΔT[k]; ks = UPPER), "%+8.3f"), cell(wmean(cΔT[k]; ks = DEEP), "%+8.3f"),
                cell(rms(cΔT[k]; ks = UPPER), "%8.3f"),
                cell(wmean(cΔS[k]; ks = UPPER), "%+8.4f"), cell(wmean(cΔS[k]; ks = DEEP), "%+8.4f"),
                cell(rms(cΔS[k]; ks = UPPER), "%8.4f"))
    end
    py(D, ks) = per_year(sect_months, [wmean(D[k]; ks) for k in sect_months])
    @printf("  %-8s | %s %s %8s | %s %s\n", "per year", cell(py(cΔT, UPPER), "%+8.3f"), cell(py(cΔT, DEEP), "%+8.3f"),
            "", cell(py(cΔS, UPPER), "%+8.4f"), cell(py(cΔS, DEEP), "%+8.4f"))
end
println("\n  per year: (last 12 months - first 12) / their separation, so the seasonal cycle cancels (24+ months only).\n")

# ── figures ───────────────────────────────────────────────────────────────────────────────────
xticks_for(ms) = (n = length(ms); s = max(1, cld(n, 12)); (collect(1:s:n), [label(ms[i]) for i in 1:s:n]))

!isempty(surf_months) && for (D, v, unit, lim, key) in ((sΔS, "S", "g/kg", S_LIM, "S_LIM"), (sΔT, "T", "°C", T_LIM, "T_LIM"))
    nm = length(surf_months); ncol = min(nm, 6); nrow = cld(nm, ncol)
    fig = Figure(size = (260 * ncol + 120, 190 * nrow + 120))
    Label(fig[0, 1:ncol], "$RUN — surface $v, monthly mean, run − GLORYS (0-$(TOP_LABEL) m)", fontsize = 18, font = :bold)
    local hm
    for (n, k) in enumerate(surf_months)
        r, c = fldmod1(n, ncol)
        ax = Axis(fig[r, c], title = label(k), titlesize = 12, xticklabelsvisible = r == nrow,
                  yticklabelsvisible = c == 1, xticklabelsize = 9, yticklabelsize = 9)
        hm = heatmap!(ax, λc, φc, D[k], colormap = :balance, colorrange = (-lim, lim), nan_color = :gray80)
    end
    Colorbar(fig[1:nrow, ncol + 1], hm, label = "Δ$v ($unit)")
    Label(fig[nrow + 1, 1:ncol], "red: the run higher; blue: lower. Grey: land in the run, or no GLORYS water. " *
          "Colour limit ±$lim $unit (set $key); larger values are drawn at the end colour.", fontsize = 11)
    f = joinpath(OUTDIR, "$(RUN)_vs_glorys_surface_$(v).png"); save(f, fig); @info "wrote $f"
end

!isempty(surf_months) && let panels = (("ΔT, all months", mean_over(sΔT, surf_months), T_LIM, "°C"),
                                       ("ΔS, all months", ΔS_all, S_LIM, "g/kg"),
                                       ("ΔS, Dec-Feb", ΔS_summer, S_LIM, "g/kg"))
    fig = Figure(size = (900, 1050))
    Label(fig[0, 1], "$RUN − GLORYS, surface, $(label(first(surf_months)))..$(label(last(surf_months)))",
          fontsize = 18, font = :bold)
    for (r, (title, A, lim, unit)) in enumerate(panels)
        ax = Axis(fig[r, 1], title = title, xticklabelsvisible = r == length(panels))
        hm = heatmap!(ax, λc, φc, A, colormap = :balance, colorrange = (-lim, lim), nan_color = :gray80)
        r > 1 && contour!(ax, λc, φc, Float64.(coastal), levels = [0.5], color = :black, linewidth = 0.6)
        Colorbar(fig[r, 2], hm, label = "$(first(title, 2)) ($unit)")
    end
    Label(fig[length(panels) + 1, 1], "red: the run higher. Black outline (S panels): the coastal cells. " *
          "Colour limits ±$T_LIM °C, ±$S_LIM g/kg (T_LIM, S_LIM); larger values at the end colour.", fontsize = 11)
    f = joinpath(OUTDIR, "$(RUN)_vs_glorys_surface_mean.png"); save(f, fig); @info "wrote $f"
end

!isempty(sect_months) && for (D, v, unit, lim, key) in ((cΔS, "S", "g/kg", SS_LIM, "SS_LIM"), (cΔT, "T", "°C", ST_LIM, "ST_LIM"))
    nm = length(sect_months); ncol = min(nm, 6); nrow = cld(nm, ncol)
    zbottom = isinf(ZMAX) ? zf[1] : -ZMAX
    fig = Figure(size = (280 * ncol + 120, 200 * nrow + 120))
    Label(fig[0, 1:ncol], "$RUN — $v along $(round(λc[MID], digits = 1))°E, monthly mean, run − GLORYS",
          fontsize = 18, font = :bold)
    local hm
    for (n, k) in enumerate(sect_months)
        r, c = fldmod1(n, ncol)
        ax = Axis(fig[r, c], title = label(k), titlesize = 12, xticklabelsvisible = r == nrow,
                  yticklabelsvisible = c == 1, xticklabelsize = 9, yticklabelsize = 9,
                  xlabel = r == nrow ? "latitude" : "", ylabel = c == 1 ? "z (m)" : "")
        hm = heatmap!(ax, φc, zc, D[k], colormap = :balance, colorrange = (-lim, lim), nan_color = :gray80)
        ylims!(ax, zbottom, 0)
    end
    Colorbar(fig[1:nrow, ncol + 1], hm, label = "Δ$v ($unit)")
    Label(fig[nrow + 1, 1:ncol], "red: the run higher; blue: lower. Grey: land in the run, or no GLORYS water. " *
          (isinf(ZMAX) ? "Full depth. " : "Upper $(round(Int, ZMAX)) m (MIDLON_ZMAX). ") *
          "Colour limit ±$lim $unit (set $key).", fontsize = 11)
    f = joinpath(OUTDIR, "$(RUN)_vs_glorys_midlon_$(v).png"); save(f, fig); @info "wrote $f"
end

!isempty(surf_months) && let
    fig = Figure(size = (900, 700))
    for (row, (v, sD, cD, unit)) in enumerate((("T", sΔT, cΔT, "°C"), ("S", sΔS, cΔS, "g/kg")))
        ax = Axis(fig[row, 1], ylabel = "Δ$v ($unit)", xticks = xticks_for(all_months), xticklabelrotation = π / 3,
                  xticklabelsvisible = row == 2, title = row == 1 ? "$RUN − GLORYS, month by month" : "")
        hlines!(ax, [0], color = :gray60)
        x(ms) = [findfirst(==(k), all_months) for k in ms]
        scatterlines!(ax, x(surf_months), [wmean(sD[k]) for k in surf_months], color = :firebrick, label = "surface, domain")
        scatterlines!(ax, x(surf_months), [wmean(sD[k], coastal) for k in surf_months], color = :orange, label = "surface, coast")
        if !isempty(sect_months)
            UPPER = [k for k in 1:Nz if -zc[k] <= 200]
            scatterlines!(ax, x(sect_months), [wmean(cD[k]; ks = UPPER) for k in sect_months], color = :navy,
                          label = "section 0-200 m")
        end
        row == 1 && axislegend(ax; position = :lt, framevisible = false)
    end
    f = joinpath(OUTDIR, "$(RUN)_vs_glorys_series.png"); save(f, fig); @info "wrote $f"
end
