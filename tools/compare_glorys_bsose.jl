# ==============================================================================
# compare_glorys_bsose.jl -- the two parents against each other at the surface, month by month:
# GLORYS monthly minus B-SOSE monthly, beside the GLORYS-nested run minus each of them. Whether a
# bias of the GLORYS baseline (the salty Antarctic coast, say) is its parent's, or the model's own.
#
#     source .copernicus_env      # only the first time: the GLORYS surface is downloaded
#     julia --project=. tools/compare_glorys_bsose.jl                 # 2Y_nested_glorys_baseline
#     julia --project=. tools/compare_glorys_bsose.jl 2Y_nested_seaice_fix9
#
# WHAT IS COMPARED. Each month's mean of the top level: GLORYS's (0.49 m, cmems_mod_glo_phy_my_0.083deg_P1M-m),
# B-SOSE i156's (0-4.2 m), and the run's (its top cell, from <RUN>_surface.nc, over the calendar month). All
# three are put on the run's 1/6-degree cells the same way, by averaging the wet source points that fall
# inside each cell (GLORYS ~2x2 points, B-SOSE 1-3), so none is interpolated differently from the others.
# Cells count where the run's top cell is wet and both parents have water in it. Potential temperature
# and practical salinity in both parents, as the run is initialised and forced.
#
# DOWNLOADS. Only GLORYS's top level, for the run's box and months, through NumericalEarth's
# CopernicusMarine extension: a few MB a month, into GLORYS_DIR, with "_-1.0_0.0" (the depth range) in
# the names so they never stand in for the full-depth files the runs use. Later runs read them back.
# Needs COPERNICUSMARINE_SERVICE_USERNAME and _PASSWORD in the environment (source .copernicus_env).
# B-SOSE is read straight from its files: no regridding, no inpainting, no 3-D fields.
#
# WRITES (OUTDIR, default plots/):
#     <RUN>_glorys_minus_bsose_S.png, _T.png   one surface map per month, GLORYS - B-SOSE
#     <RUN>_parents_vs_run.png                 whole-run and Dec-Feb means: GLORYS - B-SOSE,
#                                              run - B-SOSE, run - GLORYS, side by side
# and on stdout: GLORYS - B-SOSE by latitude band; the coast (within COAST_CELLS of land, south of 60S)
# against the offshore water there, for all three differences; the 15 coastal cells where the run is
# saltiest against B-SOSE in Dec-Feb, with what GLORYS has there.
#
# ENV
#     DIR, OUTDIR, MIN_DAYS        as compare_bsose.jl (., plots, 20)
#     GLORYS_DIR                   default <data>/glorys_monthly
#     COAST_CELLS, COAST_SOUTH_OF  as compare_bsose.jl (2, -60)
#     S_LIM, T_LIM                 colour limits (default 0.3 g/kg and 1.0 degC: the broad pattern; the
#                                  coast and the eddies go past them, and are drawn at the end colour)
# ==============================================================================

using Oceananigans
using NumericalEarth
using NumericalEarth.DataWrangling: BoundingBox, Metadatum, metadata_path, dataset_variable_name
using NumericalEarth.DataWrangling.GLORYS: GLORYSMonthly
using CopernicusMarine          # NumericalEarth downloads GLORYS through it
import Downloads
using NCDatasets, CairoMakie, Printf, Statistics, Dates

include(joinpath(@__DIR__, "..", "src", "setup_bsose.jl"))

const RUN        = length(ARGS) > 0 ? ARGS[1] : "2Y_nested_glorys_baseline"
const DIR        = get(ENV, "DIR", ".")
const OUTDIR     = get(ENV, "OUTDIR", "plots")
const MIN_DAYS   = parse(Int, get(ENV, "MIN_DAYS", "20"))
const GLORYS_DIR = get(ENV, "GLORYS_DIR", joinpath(default_bsose_directory(), "glorys_monthly"))
const S_LIM      = parse(Float64, get(ENV, "S_LIM", "0.3"))
const T_LIM      = parse(Float64, get(ENV, "T_LIM", "1.0"))
const COAST_CELLS    = parse(Int, get(ENV, "COAST_CELLS", "2"))
const COAST_SOUTH_OF = parse(Float64, get(ENV, "COAST_SOUTH_OF", "-60"))
mkpath(OUTDIR); mkpath(GLORYS_DIR)

f64(A) = Float64.(coalesce.(A, NaN))

# ── the run's cells, mask and months ──────────────────────────────────────────────────────────
const SURF = joinpath(DIR, "$(RUN)_surface.nc")
isfile(SURF) || error("no $SURF")
ds = Dataset(SURF)
λc, φc = f64(ds["λ_caa"][:]), f64(ds["φ_aca"][:])
Nx, Ny = length(λc), length(φc)
Δλ, Δφ = λc[2] - λc[1], φc[2] - φc[1]
λW, φS = λc[1] - Δλ / 2, φc[1] - Δφ / 2
surf_times = f64(ds["time"][:])
start = DateTime(string(ds.attrib["start_date"]))
model_wet = let m = ds["inactive_nodes_ccc"]; (ndims(m) == 3 ? m[:, :, end] : m[:, :]) .== 0 end
close(ds)

groups = Dict{Tuple{Int,Int},Vector{Int}}()
for (n, t) in enumerate(surf_times)
    t == 0 && continue
    d = start + Millisecond(round(Int, 1000t))
    push!(get!(groups, (year(d), month(d)), Int[]), n)
end
bsose = BSOSEMonthly(iteration = 156)
bsose_record = Dict((year(bsose_window_center(d)), month(bsose_window_center(d))) => d
                    for d in all_dates(bsose, :temperature))
months = sort([k for (k, v) in groups if length(v) >= MIN_DAYS && haskey(bsose_record, k)])
isempty(months) && error("no month has $MIN_DAYS+ days of output and a B-SOSE record")
label(k) = @sprintf("%d-%02d", k...)
@info "$RUN: $(Nx)x$(Ny) cells, $(length(months)) months $(label(first(months)))..$(label(last(months)))"

"""
Mean of the wet `values` at (`lons`, `lats`) inside each of the run's cells; NaN where none. A point on a
cell edge (GLORYS's 1/12-degree points fall on every other edge of the 1/6-degree cells) counts half in
each neighbour, or every mean would sit 1/24 degree off its cell's centre.
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
const GLORYS_REGION = BoundingBox(longitude = (λW, λW + Nx * Δλ), latitude = (φS, φS + Ny * Δφ), z = (-1.0, 0.0))
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
# Its coordinates and sea floor from the surface-salt-flux file (as bsose_masks.jl), cut to the box:
# land is 0 in the T and S files, so it is masked by Depth, not by value.
bsose_box = let d = Dataset(joinpath(bsose.dir, resolve_bsose_filename(bsose.dir, :surface_salt_flux, bsose.iteration)))
    xc, yc = f64(d["XC"][:]), f64(d["YC"][:])
    is = findall(x -> λW <= x < λW + Nx * Δλ, xc)
    js = findall(y -> φS <= y < φS + Ny * Δφ, yc)
    wet = f64(d["Depth"][first(is):last(is), first(js):last(js)]) .> 0
    close(d)
    (; is = first(is):last(is), js = first(js):last(js), λ = xc[first(is):last(is)], φ = yc[first(js):last(js)], wet)
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

function run_surface(d, var, k)
    v = d[var]
    return sum(f64(ndims(v) == 4 ? v[:, :, 1, n] : v[:, :, n]) for n in groups[k]) ./ length(groups[k])
end

# ── the three differences, month by month ────────────────────────────────────────────────────
D = Dict{Symbol, Dict{Tuple{Int,Int},Matrix{Float64}}}()
for key in (:GB_T, :GB_S, :RB_T, :RB_S, :RG_T, :RG_S); D[key] = Dict(); end
dsurf = Dataset(SURF)
for k in months
    for (v, name, var) in ((:T, :temperature, "T"), (:S, :salinity, "S"))
        G, B = glorys_surface(name, k), bsose_surface(name, k)
        R = ifelse.(model_wet, run_surface(dsurf, var, k), NaN)
        ok = isfinite.(G) .& isfinite.(B) .& model_wet
        D[Symbol(:GB_, v)][k] = ifelse.(ok, G .- B, NaN)
        D[Symbol(:RB_, v)][k] = ifelse.(ok, R .- B, NaN)
        D[Symbol(:RG_, v)][k] = ifelse.(ok, R .- G, NaN)
    end
end
close(dsurf)
@info "compared $(count(isfinite, D[:GB_S][first(months)])) cells"

# ── statistics ────────────────────────────────────────────────────────────────────────────────
const BANDS = [(-70.0, -65.0), (-65.0, -60.0), (-60.0, -55.0), (-55.0, -50.0), (-50.0, -45.0)]
function wmean(A, keep = trues(Nx, Ny); lo = -90.0, hi = 90.0)
    num = den = 0.0
    for i in 1:Nx, j in 1:Ny
        (keep[i, j] && lo <= φc[j] < hi && isfinite(A[i, j])) || continue
        w = cosd(φc[j]); num += w * A[i, j]; den += w
    end
    return den == 0 ? NaN : num / den
end
cell(v, fmt) = isnan(v) ? lpad("--", 8) : Printf.format(Printf.Format(fmt), v)

coastal = falses(Nx, Ny)
for i in 1:Nx, j in 1:Ny
    (model_wet[i, j] && φc[j] < COAST_SOUTH_OF) || continue
    coastal[i, j] = any(.!model_wet[max(1, i - COAST_CELLS):min(Nx, i + COAST_CELLS),
                                    max(1, j - COAST_CELLS):min(Ny, j + COAST_CELLS)])
end
offshore = model_wet .& .!coastal .& [φ < COAST_SOUTH_OF for _ in 1:Nx, φ in φc]
summer(k) = k[2] in (12, 1, 2)
mean_over(Dk, ms) = isempty(ms) ? fill(NaN, Nx, Ny) : sum(Dk[k] for k in ms) ./ length(ms)

println("\n", "="^96)
println("SURFACE, monthly means: GLORYS - B-SOSE, and the run ($RUN) against each")
println("  Positive ΔS: the first is SALTIER; positive ΔT: WARMER. cos(lat)-weighted over cells wet in all three.")
println("="^96)
for (v, name, fmt) in ((:S, "ΔS (g/kg)", "%+7.3f"), (:T, "ΔT (°C)", "%+7.2f"))
    @printf("\n  GLORYS - B-SOSE, %s, by latitude band\n  %-8s | %7s |", name, "month", "all")
    for (lo, hi) in BANDS; @printf(" %7s", "$(Int(-lo))-$(Int(-hi))S"); end
    println()
    for k in months
        A = D[Symbol(:GB_, v)][k]
        @printf("  %-8s | %s |", label(k), cell(wmean(A), fmt))
        for (lo, hi) in BANDS; print(" ", cell(wmean(A; lo, hi), fmt)); end
        println()
    end
end

@printf("\n  SURFACE ΔS (g/kg) south of %.0fS: the COAST (%d cells within %d of land) and OFFSHORE (%d cells)\n",
        -COAST_SOUTH_OF, count(coastal), COAST_CELLS, count(offshore))
@printf("  %-8s | %26s | %26s\n", "", "coast", "offshore")
@printf("  %-8s | %8s %8s %8s | %8s %8s %8s\n", "month", "GLO-BSO", "run-BSO", "run-GLO", "GLO-BSO", "run-BSO", "run-GLO")
for k in months
    @printf("  %-8s |", label(k))
    for m in (coastal, offshore)
        for key in (:GB_S, :RB_S, :RG_S); print(" ", cell(wmean(D[key][k], m), "%+8.3f")); end
        m === coastal && print(" |")
    end
    println()
end

smonths = filter(summer, months)
RB, GB, RG = mean_over(D[:RB_S], smonths), mean_over(D[:GB_S], smonths), mean_over(D[:RG_S], smonths)
ranked = sort([(RB[i, j], i, j) for i in 1:Nx, j in 1:Ny if coastal[i, j] && isfinite(RB[i, j])], rev = true)
println("\n  The 15 coastal cells where the run is saltiest against B-SOSE in Dec-Feb, and GLORYS there")
@printf("  %8s %8s | %8s %8s %8s\n", "lon", "lat", "run-BSO", "GLO-BSO", "run-GLO")
for (v, i, j) in ranked[1:min(15, end)]
    @printf("  %7.2fE %7.2fS | %+8.3f %+8.3f %+8.3f\n", λc[i], -φc[j], v, GB[i, j], RG[i, j])
end
println("\n  If GLO-BSO is about as large as run-BSO, the run inherits the bias from GLORYS; if run-GLO")
println("  carries it, the run makes it itself.\n")

# ── figures ───────────────────────────────────────────────────────────────────────────────────
for (v, unit, lim) in ((:S, "g/kg", S_LIM), (:T, "°C", T_LIM))
    nm = length(months); ncol = min(nm, 6); nrow = cld(nm, ncol)
    fig = Figure(size = (260 * ncol + 120, 190 * nrow + 120))
    Label(fig[0, 1:ncol], "surface $v, monthly mean, GLORYS − B-SOSE", fontsize = 18, font = :bold)
    local hm
    for (n, k) in enumerate(months)
        r, c = fldmod1(n, ncol)
        ax = Axis(fig[r, c], title = label(k), titlesize = 12, xticklabelsvisible = r == nrow,
                  yticklabelsvisible = c == 1, xticklabelsize = 9, yticklabelsize = 9)
        hm = heatmap!(ax, λc, φc, D[Symbol(:GB_, v)][k], colormap = :balance, colorrange = (-lim, lim), nan_color = :gray80)
    end
    Colorbar(fig[1:nrow, ncol + 1], hm, label = "Δ$v ($unit)")
    Label(fig[nrow + 1, 1:ncol], "red: GLORYS higher; blue: lower. Grey: land, or no water in a parent. " *
          "Colour limit ±$lim $unit (set $(v)_LIM); larger values are drawn at the end colour.", fontsize = 11)
    f = joinpath(OUTDIR, "$(RUN)_glorys_minus_bsose_$(v).png"); save(f, fig); @info "wrote $f"
end

let rows = (("ΔT, whole run", :T, months, T_LIM, "°C"), ("ΔS, whole run", :S, months, S_LIM, "g/kg"),
            ("ΔS, Dec-Feb", :S, smonths, S_LIM, "g/kg"))
    cols = (("GLORYS − B-SOSE", :GB_), ("run − B-SOSE", :RB_), ("run − GLORYS", :RG_))
    fig = Figure(size = (1500, 900))
    Label(fig[0, 1:3], "$RUN: the parents and the run at the surface", fontsize = 18, font = :bold)
    for (r, (rname, v, ms, lim, unit)) in enumerate(rows)
        local hm
        for (c, (cname, prefix)) in enumerate(cols)
            ax = Axis(fig[r, c], title = r == 1 ? cname : "", ylabel = c == 1 ? rname : "",
                      xticklabelsvisible = r == length(rows), yticklabelsvisible = c == 1)
            hm = heatmap!(ax, λc, φc, mean_over(D[Symbol(prefix, v)], ms), colormap = :balance,
                          colorrange = (-lim, lim), nan_color = :gray80)
        end
        Colorbar(fig[r, 4], hm, label = "$(first(rname, 2)) ($unit)")
    end
    Label(fig[length(rows) + 1, 1:3], "Colour limits ±$T_LIM °C and ±$S_LIM g/kg (T_LIM, S_LIM); larger values are drawn at the end colour.",
          fontsize = 11)
    f = joinpath(OUTDIR, "$(RUN)_parents_vs_run.png"); save(f, fig); @info "wrote $f"
end
