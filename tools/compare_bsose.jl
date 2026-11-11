# ==============================================================================
# compare_bsose.jl -- a run against B-SOSE, month by month: the SURFACE and the MIDLON SECTION.
#
#     julia --project=. tools/compare_bsose.jl 2Y_nested_seaice_fix6
#     MIDLON_ZMAX=full T_LIM=2 julia --project=. tools/compare_bsose.jl 2Y_nested_seaice_fix6
#
# WHAT IS COMPARED. B-SOSE publishes MONTHLY MEANS, so the model side is made the same thing: the
# mean of its daily output over each calendar month, against the B-SOSE record whose averaging
# window is that month. The t = 0 snapshot (the initial condition, B-SOSE itself) is left out, and
# a month with fewer than MIN_DAYS days of output (the partial months at either end) is skipped.
# B-SOSE is regridded onto the run's own grid -- levels and sea floor read off its output
# (tools/model_grid.jl) -- so both sides are compared at the same depths. Cells count only where
# BOTH bathymetries are wet (tools/bsose_masks.jl).
#
#     surface   <RUN>_surface.nc, the top cell, against B-SOSE's top cell
#     midlon    <RUN>_midlon.nc, the latitude-depth section at the run's middle longitude (~120E),
#               against B-SOSE's column there
#
# WRITES (OUTDIR, default plots/):
#     <RUN>_surface_bias_T.png, _S.png     one map per month, model - B-SOSE
#     <RUN>_surface_bias_series.png        domain-mean bias and RMS difference, month by month
#     <RUN>_midlon_bias_T.png, _S.png      one latitude-depth panel per month, model - B-SOSE
#     <RUN>_midlon_bias_series.png         section-mean bias and RMS, 0-200 m and 200 m-floor
#     <RUN>_coastal_salinity_bias.png      surface ΔS south of 60S, whole run and Dec-Feb, coast outlined
# and the same numbers on stdout, by latitude band for the surface and by layer for the section,
# with the drift per year (last 12 months minus first 12, the seasonal cycle cancelled) for a run
# of 24 months or more.
#
# CONVENTIONS. The model runs TEOS10 but is initialised and forced from B-SOSE's potential
# temperature and practical salinity without conversion, so the difference is like with like.
#
# This replaced, on 2026-11-09, a five-section report (relaxation-band knee, MLD and Winter Water
# on the section, upper-200 m structure, surface and section drift, eight figure types). For the
# mixed layer use tools/compare_mld.jl, for Winter Water tools/compare_winter_water.jl, for the
# open-side bands tools/check_edge_bands.jl. The old report is in git history (commit 6746ee4).
#
# ENV
#     DIR          where the run's files are                       (default .)
#     OUTDIR       figures                                         (default plots)
#     START        model time zero                     (default: the file's start_date attribute)
#     MIN_DAYS     fewest days of output for a month               (default 20)
#     MIDLON_ZMAX  depth drawn in the section maps, m, or "full"   (default 1000)
#     T_LIM, S_LIM, ST_LIM, SS_LIM   colour limits for the surface / section maps (default: the
#                  99th percentile of |Δ| over the months drawn)
#     CHUNK        B-SOSE months regridded at once                 (default 6)
#     COAST_CELLS  a coastal cell is within this many cells of land (default 2)
#     COAST_SOUTH_OF  only the coast south of this latitude        (default -60)
#     COAST_S_LIM  colour limit of the coastal map, g/kg           (default: 99th percentile)
# ==============================================================================

using Oceananigans, Oceananigans.Units, Oceananigans.Grids
using NumericalEarth
using NCDatasets, CairoMakie, Printf, Statistics, Dates

include(joinpath(@__DIR__, "..", "src", "setup_bsose.jl"))
include(joinpath(@__DIR__, "bsose_masks.jl"))   # bsose_depth
include(joinpath(@__DIR__, "model_grid.jl"))    # file_levels, file_kbot, section_column

const RUN = length(ARGS) > 0 ? ARGS[1] : get(ENV, "RUN", "")
isempty(RUN) && error("usage: julia --project=. tools/compare_bsose.jl RUN_NAME")
const DIR      = get(ENV, "DIR", ".")
const OUTDIR   = get(ENV, "OUTDIR", "plots")
const MIN_DAYS = parse(Int, get(ENV, "MIN_DAYS", "20"))
const CHUNK    = parse(Int, get(ENV, "CHUNK", "6"))
const ZMAX     = (s = get(ENV, "MIDLON_ZMAX", "1000"); s == "full" ? Inf : parse(Float64, s))
mkpath(OUTDIR)

const SURF = joinpath(DIR, "$(RUN)_surface.nc")
const SECT = joinpath(DIR, "$(RUN)_midlon.nc")
const EKE  = joinpath(DIR, "$(RUN)_eke.nc")
isfile(SURF) || error("no $SURF")
HAVE_SECT = isfile(SECT)
HAVE_SECT || @warn "no $SECT: surface comparison only"

f64(A) = Float64.(coalesce.(A, NaN))

# ── the run's grid, off its own output ────────────────────────────────────────────────────────
ds = Dataset(SURF)
λc, φc = f64(ds["λ_caa"][:]), f64(ds["φ_aca"][:])
Nx, Ny = length(λc), length(φc)
surf_times = f64(ds["time"][:])
start = haskey(ENV, "START") ? DateTime(ENV["START"]) :
        haskey(ds.attrib, "start_date") ? DateTime(string(ds.attrib["start_date"])) :
        error("$SURF has no start_date attribute; pass START=yyyy-mm-ddTHH:MM")
haskey(ds, "inactive_nodes_ccc") || error("$SURF has no inactive_nodes_ccc mask")
model_surface_wet = let m = ds["inactive_nodes_ccc"]
    (ndims(m) == 3 ? m[:, :, end] : m[:, :]) .== 0
end
close(ds)

geom = HAVE_SECT ? SECT : isfile(EKE) ? EKE : error("need $(RUN)_midlon.nc or $(RUN)_eke.nc for the vertical levels")
levels, floor_ = let g = Dataset(geom)
    lv = file_levels(g)
    fl = lv === nothing ? nothing : file_kbot(g, lv[1], lv[2])
    close(g)
    lv, fl
end
(levels === nothing || floor_ === nothing) && error("$geom carries no vertical levels or sea floor")
zc, zf, _ = levels
Nz = length(zc)
MID = section_column(SECT, Nx)
kbot_model = size(floor_[1], 1) == 1 ? vec(floor_[1]) : floor_[1][MID, :]

Δλ, Δφ = λc[2] - λc[1], φc[2] - φc[1]
grid = LatitudeLongitudeGrid(CPU(); size = (Nx, Ny, Nz),
                             longitude = (λc[1] - Δλ / 2, λc[end] + Δλ / 2),
                             latitude = (φc[1] - Δφ / 2, φc[end] + Δφ / 2),
                             z = zf, halo = (7, 7, 7))
dataset = BSOSEMonthly(iteration = 156)
region  = bsose_region(grid)

# Wet in BOTH bathymetries: the surface, and every cell of the section
surface_wet = BitMatrix(model_surface_wet .& (bsose_depth(dataset, λc, φc) .> 0))
parent_depth = bsose_depth(dataset, [λc[MID]], φc)[1, :]
kbot_parent = [d <= 0 ? nothing : findfirst(k -> -zc[k] <= d, 1:Nz) for d in parent_depth]
wetboth(j, k) = kbot_model[j] !== nothing && kbot_parent[j] !== nothing &&
                k >= kbot_model[j] && k >= kbot_parent[j]
@info @sprintf("%s: %dx%dx%d from %s; %d surface cells wet in both; section at %.2fE (column %d)",
               RUN, Nx, Ny, Nz, Dates.format(start, "yyyy-mm-dd"), count(surface_wet), λc[MID], MID)

# ── calendar months, and the B-SOSE record for each ───────────────────────────────────────────
function by_month(times)
    g = Dict{Tuple{Int,Int},Vector{Int}}()
    for (n, t) in enumerate(times)
        t == 0 && continue                                       # the initial condition
        d = start + Millisecond(round(Int, 1000t))
        push!(get!(g, (year(d), month(d)), Int[]), n)
    end
    return g
end
record_of = Dict((year(bsose_window_center(d)), month(bsose_window_center(d))) => d
                 for d in all_dates(dataset, :temperature))
whole(g) = sort([k for (k, v) in g if length(v) >= MIN_DAYS && haskey(record_of, k)])

surf_groups = by_month(surf_times)
sect_times = HAVE_SECT ? let d = Dataset(SECT); t = f64(d["time"][:]); close(d); t end : Float64[]
sect_groups = by_month(sect_times)
surf_months = whole(surf_groups)
sect_months = HAVE_SECT ? whole(sect_groups) : Tuple{Int,Int}[]
all_months  = sort(union(surf_months, sect_months))
isempty(all_months) && error("no month has $MIN_DAYS+ days of output and a B-SOSE record")
label(k) = @sprintf("%d-%02d", k...)
@info "comparing $(length(all_months)) months, $(label(first(all_months))) to $(label(last(all_months)))"

# ── model monthly means minus B-SOSE, B-SOSE regridded CHUNK months at a time ─────────────────
sΔT, sΔS = Dict{Tuple{Int,Int},Matrix{Float64}}(), Dict{Tuple{Int,Int},Matrix{Float64}}()   # (Nx, Ny)
cΔT, cΔS = Dict{Tuple{Int,Int},Matrix{Float64}}(), Dict{Tuple{Int,Int},Matrix{Float64}}()   # (Ny, Nz)

"""Mean over records `idx` of a (Nx, Ny) surface field; the file's 4-D (x, y, 1, t) or 3-D layout."""
function surface_mean(d, var, idx)
    v = d[var]
    return sum(f64(ndims(v) == 4 ? v[:, :, 1, n] : v[:, :, n]) for n in idx) ./ length(idx)
end
"""Mean over records `idx` of the (Ny, Nz) section."""
section_mean(d, var, idx) = sum(f64(d[var][1, :, :, n]) for n in idx) ./ length(idx)

dsurf = Dataset(SURF)
dsect = HAVE_SECT ? Dataset(SECT) : nothing
for chunk in Iterators.partition(all_months, CHUNK)
    c = collect(chunk)
    load(var) = FieldTimeSeries(Metadata(var; dataset, dates = [record_of[k] for k in c], region), grid;
                                time_indices_in_memory = length(c))
    Tb, Sb = load(:temperature), load(:salinity)
    for (i, k) in enumerate(c)
        Tp, Sp = Array(interior(Tb[i])), Array(interior(Sb[i]))
        if k in surf_months
            sΔT[k] = ifelse.(surface_wet, surface_mean(dsurf, "T", surf_groups[k]) .- Tp[:, :, Nz], NaN)
            sΔS[k] = ifelse.(surface_wet, surface_mean(dsurf, "S", surf_groups[k]) .- Sp[:, :, Nz], NaN)
        end
        if k in sect_months
            wb = [wetboth(j, kk) for j in 1:Ny, kk in 1:Nz]
            cΔT[k] = ifelse.(wb, section_mean(dsect, "T", sect_groups[k]) .- Tp[MID, :, :], NaN)
            cΔS[k] = ifelse.(wb, section_mean(dsect, "S", sect_groups[k]) .- Sp[MID, :, :], NaN)
        end
    end
    Tb = Sb = nothing
    GC.gc()
    @info "  regridded B-SOSE for $(label(first(c)))..$(label(last(c)))"
end
close(dsurf); dsect === nothing || close(dsect)

# ── statistics ────────────────────────────────────────────────────────────────────────────────
const BANDS = [(-70.0, -65.0), (-65.0, -60.0), (-60.0, -55.0), (-55.0, -50.0), (-50.0, -45.0)]

"""cos(lat)-weighted mean of f.(A) over finite cells with latitude in [lo, hi) (and levels ks for a section)."""
function wmean(A, lo, hi; f = identity, ks = nothing)
    num = den = 0.0
    if ks === nothing                                            # (Nx, Ny) surface
        for i in axes(A, 1), j in axes(A, 2)
            (lo <= φc[j] < hi && isfinite(A[i, j])) || continue
            w = cosd(φc[j]); num += w * f(A[i, j]); den += w
        end
    else                                                         # (Ny, Nz) section
        for j in axes(A, 1), k in ks
            (lo <= φc[j] < hi && isfinite(A[j, k])) || continue
            w = cosd(φc[j]); num += w * f(A[j, k]); den += w
        end
    end
    return den == 0 ? NaN : num / den
end
bias(A; ks = nothing) = wmean(A, -90, 90; ks)
rms(A; ks = nothing)  = sqrt(wmean(A, -90, 90; f = abs2, ks))
const UPPER = [k for k in 1:Nz if -zc[k] <= 200]
const DEEP  = [k for k in 1:Nz if -zc[k] > 200]
cell(v, fmt) = isnan(v) ? lpad("--", 8) : Printf.format(Printf.Format(fmt), v)

"""(last 12 months - first 12) / their separation in years, of the series `v` over months `ms`; NaN if < 24."""
function per_year(ms, v)
    length(ms) < 24 && return NaN
    yrs(k) = Dates.value(DateTime(k[1], k[2], 15) - start) / (86400e3 * 365.25)
    a, b = 1:12, (length(ms) - 11):length(ms)
    return (mean(v[b]) - mean(v[a])) / (mean(yrs.(ms[b])) - mean(yrs.(ms[a])))
end

# ── tables ────────────────────────────────────────────────────────────────────────────────────
println()
println("="^96)
println("MONTHLY MEAN model minus B-SOSE  ($RUN)")
println("  cos(lat)-weighted over cells wet in both bathymetries. Positive ΔT: model too WARM; positive ΔS: too SALTY.")
println("="^96)
println("\n  SURFACE (top cell), whole domain")
@printf("  %-8s | %8s %8s | %8s %8s\n", "month", "ΔT", "rms T", "ΔS", "rms S")
for k in surf_months
    @printf("  %-8s | %s %s | %s %s\n", label(k), cell(bias(sΔT[k]), "%+8.3f"), cell(rms(sΔT[k]), "%8.3f"),
            cell(bias(sΔS[k]), "%+8.3f"), cell(rms(sΔS[k]), "%8.3f"))
end
@printf("  %-8s | %s %8s | %s\n", "per year", cell(per_year(surf_months, [bias(sΔT[k]) for k in surf_months]), "%+8.3f"),
        "", cell(per_year(surf_months, [bias(sΔS[k]) for k in surf_months]), "%+8.3f"))
for (name, D, fmt) in (("ΔT (°C)", sΔT, "%+7.2f"), ("ΔS (g/kg)", sΔS, "%+7.3f"))
    @printf("\n  SURFACE %s by latitude band\n  %-8s |", name, "month")
    for (lo, hi) in BANDS; @printf(" %7s", "$(Int(-lo))-$(Int(-hi))S"); end
    println()
    for k in surf_months
        @printf("  %-8s |", label(k))
        for (lo, hi) in BANDS; print(" ", cell(wmean(D[k], lo, hi), fmt)); end
        println()
    end
end
if !isempty(sect_months)
    @printf("\n  MIDLON SECTION at %.1fE, by layer\n", λc[MID])
    @printf("  %-8s | %8s %8s %8s | %8s %8s %8s\n", "month", "ΔT 0-200", "200-flr", "rms 0-200",
            "ΔS 0-200", "200-flr", "rms 0-200")
    for k in sect_months
        @printf("  %-8s | %s %s %s | %s %s %s\n", label(k),
                cell(bias(cΔT[k]; ks = UPPER), "%+8.3f"), cell(bias(cΔT[k]; ks = DEEP), "%+8.3f"),
                cell(rms(cΔT[k]; ks = UPPER), "%8.3f"),
                cell(bias(cΔS[k]; ks = UPPER), "%+8.4f"), cell(bias(cΔS[k]; ks = DEEP), "%+8.4f"),
                cell(rms(cΔS[k]; ks = UPPER), "%8.4f"))
    end
    py(D, ks) = per_year(sect_months, [bias(D[k]; ks) for k in sect_months])
    @printf("  %-8s | %s %s %8s | %s %s\n", "per year", cell(py(cΔT, UPPER), "%+8.3f"), cell(py(cΔT, DEEP), "%+8.3f"),
            "", cell(py(cΔS, UPPER), "%+8.4f"), cell(py(cΔS, DEEP), "%+8.4f"))
end
println("\n  per year: (last 12 months - first 12) / their separation, so the seasonal cycle cancels (24+ months only).\n")

# ── the Antarctic coast: where the surface ΔS sits ────────────────────────────────────────────
# Coastal cells: wet in both, south of COAST_SOUTH_OF, within COAST_CELLS cells (either axis) of the model's
# land. Offshore: the rest of the wet cells south of COAST_SOUTH_OF.
const COAST_CELLS    = parse(Int, get(ENV, "COAST_CELLS", "2"))
const COAST_SOUTH_OF = parse(Float64, get(ENV, "COAST_SOUTH_OF", "-60"))
coastal = falses(Nx, Ny)
for i in 1:Nx, j in 1:Ny
    (surface_wet[i, j] && φc[j] < COAST_SOUTH_OF) || continue
    near = (max(1, i - COAST_CELLS):min(Nx, i + COAST_CELLS), max(1, j - COAST_CELLS):min(Ny, j + COAST_CELLS))
    coastal[i, j] = any(.!model_surface_wet[near...])
end
offshore = surface_wet .& .!coastal .& [φ < COAST_SOUTH_OF for _ in 1:Nx, φ in φc]
masked_mean(A, m) = wmean(ifelse.(m, A, NaN), -90, 90)
const LON_BINS = [(λ, λ + 10) for λ in 10 * floor(Int, λc[1] / 10):10:(10 * floor(Int, λc[end] / 10))]
in_bin(i, (a, b)) = a <= λc[i] < b
summer(k) = k[2] in (12, 1, 2)

@printf("\n  SURFACE ΔS (g/kg) at the COAST: %d cells within %d of land south of %.0fS, against %d offshore cells there\n",
        count(coastal), COAST_CELLS, -COAST_SOUTH_OF, count(offshore))
@printf("  %-8s | %8s %8s |", "month", "coast", "offshore")
for b in LON_BINS; @printf(" %8s", "$(b[1])-$(b[2])E"); end
println()
for k in surf_months
    @printf("  %-8s | %s %s |", label(k), cell(masked_mean(sΔS[k], coastal), "%+8.3f"), cell(masked_mean(sΔS[k], offshore), "%+8.3f"))
    for b in LON_BINS
        m = coastal .& [in_bin(i, b) for i in 1:Nx, _ in 1:Ny]
        print(" ", cell(any(m) ? masked_mean(sΔS[k], m) : NaN, "%+8.3f"))
    end
    println()
end

mean_over(ms) = isempty(ms) ? fill(NaN, Nx, Ny) : sum(sΔS[k] for k in ms) ./ length(ms)
ΔS_all, ΔS_summer = mean_over(surf_months), mean_over(filter(summer, surf_months))
ranked = sort([(ΔS_summer[i, j], i, j) for i in 1:Nx, j in 1:Ny if coastal[i, j] && isfinite(ΔS_summer[i, j])], rev = true)
println("\n  The 15 coastal cells with the largest summer (Dec-Feb) ΔS, and their whole-run mean ΔS")
@printf("  %8s %8s | %10s %10s\n", "lon", "lat", "ΔS DJF", "ΔS all")
for (v, i, j) in ranked[1:min(15, end)]
    @printf("  %7.2fE %7.2fS | %+10.3f %+10.3f\n", λc[i], -φc[j], v, ΔS_all[i, j])
end
println()

!isempty(surf_months) && let
    jsouth = findall(φ -> φ < COAST_SOUTH_OF, φc)
    lim = haskey(ENV, "COAST_S_LIM") ? parse(Float64, ENV["COAST_S_LIM"]) :
          max(quantile(filter(isfinite, abs.(vcat(vec(ΔS_all[:, jsouth]), vec(ΔS_summer[:, jsouth])))), 0.99), 1e-6)
    fig = Figure(size = (1100, 640))
    Label(fig[0, 1], "$RUN — surface S, model − B-SOSE, south of $(Int(-COAST_SOUTH_OF))S", fontsize = 18, font = :bold)
    local hm
    for (row, (name, D)) in enumerate((("whole run", ΔS_all), ("Dec-Feb", ΔS_summer)))
        ax = Axis(fig[row, 1], title = "mean over $name; black: coastal cells (within $COAST_CELLS of land)",
                  titlesize = 12, ylabel = "latitude", xlabel = row == 2 ? "longitude" : "")
        hm = heatmap!(ax, λc, φc[jsouth], D[:, jsouth], colormap = :balance, colorrange = (-lim, lim), nan_color = :gray80)
        contour!(ax, λc, φc[jsouth], Float64.(coastal[:, jsouth]), levels = [0.5], color = :black, linewidth = 0.8)
    end
    Colorbar(fig[1:2, 2], hm, label = "ΔS (g/kg); red: model too salty")
    f = joinpath(OUTDIR, "$(RUN)_coastal_salinity_bias.png"); save(f, fig); @info "wrote $f"
end

# ── figures ───────────────────────────────────────────────────────────────────────────────────
function colour_limit(Ds, key)
    haskey(ENV, key) && return parse(Float64, ENV[key])
    v = filter(isfinite, abs.(reduce(vcat, vec.(Ds))))
    return isempty(v) ? 1.0 : max(quantile(v, 0.99), 1e-6)
end
xticks_for(ms) = (n = length(ms); s = max(1, cld(n, 12)); (collect(1:s:n), [label(ms[i]) for i in 1:s:n]))

# Surface maps, one per month
!isempty(surf_months) && let
    nm = length(surf_months); ncol = min(nm, 6); nrow = cld(nm, ncol)
    for (var, D, unit, key, hot, cold) in (("T", sΔT, "°C", "T_LIM", "warm", "cold"), ("S", sΔS, "g/kg", "S_LIM", "salty", "fresh"))
        lim = colour_limit([D[k] for k in surf_months], key)
        fig = Figure(size = (260 * ncol + 120, 190 * nrow + 120))
        Label(fig[0, 1:ncol], "$RUN — surface $var, monthly mean, model − B-SOSE", fontsize = 18, font = :bold)
        local hm
        for (n, k) in enumerate(surf_months)
            r, c = fldmod1(n, ncol)
            ax = Axis(fig[r, c], title = label(k), titlesize = 12, xticklabelsvisible = r == nrow,
                      yticklabelsvisible = c == 1, xticklabelsize = 9, yticklabelsize = 9)
            hm = heatmap!(ax, λc, φc, D[k], colormap = :balance, colorrange = (-lim, lim), nan_color = :gray80)
        end
        Colorbar(fig[1:nrow, ncol + 1], hm, label = "Δ$var ($unit)")
        Label(fig[nrow + 1, 1:ncol], "red: model too $hot; blue: too $cold. Grey: land in either bathymetry. " *
              "Colour limit ±$(round(lim, sigdigits = 2)) $unit" *
              (haskey(ENV, key) ? " (set by $key)." : " (99th percentile of |Δ$var|; set $key to fix it)."), fontsize = 11)
        f = joinpath(OUTDIR, "$(RUN)_surface_bias_$(var).png"); save(f, fig); @info "wrote $f"
    end

    nm = length(surf_months)
    fig = Figure(size = (900, 520))
    for (row, (var, D, unit)) in enumerate((("T", sΔT, "°C"), ("S", sΔS, "g/kg")))
        ax = Axis(fig[row, 1], ylabel = "Δ$var ($unit)", xticks = xticks_for(surf_months), xticklabelrotation = π / 3,
                  xticklabelsvisible = row == 2, title = row == 1 ? "$RUN — surface, model − B-SOSE, domain" : "")
        hlines!(ax, [0], color = :gray60)
        scatterlines!(ax, 1:nm, [bias(D[k]) for k in surf_months], color = :firebrick, label = "mean")
        scatterlines!(ax, 1:nm, [rms(D[k]) for k in surf_months], color = :black, linestyle = :dash, label = "RMS")
        row == 1 && axislegend(ax; position = :lt, framevisible = false)
    end
    f = joinpath(OUTDIR, "$(RUN)_surface_bias_series.png"); save(f, fig); @info "wrote $f"
end

# Section maps, one latitude-depth panel per month
!isempty(sect_months) && let
    nm = length(sect_months); ncol = min(nm, 6); nrow = cld(nm, ncol)
    kdraw = [k for k in 1:Nz if -zc[k] <= ZMAX]
    zbottom = isinf(ZMAX) ? zf[1] : -ZMAX
    for (var, D, unit, key, hot, cold) in (("T", cΔT, "°C", "ST_LIM", "warm", "cold"), ("S", cΔS, "g/kg", "SS_LIM", "salty", "fresh"))
        lim = colour_limit([D[k][:, kdraw] for k in sect_months], key)
        fig = Figure(size = (280 * ncol + 120, 200 * nrow + 120))
        Label(fig[0, 1:ncol], "$RUN — $var along $(round(λc[MID], digits = 1))°E, monthly mean, model − B-SOSE",
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
        Colorbar(fig[1:nrow, ncol + 1], hm, label = "Δ$var ($unit)")
        Label(fig[nrow + 1, 1:ncol], "red: model too $hot; blue: too $cold. Grey: land in either bathymetry. " *
              (isinf(ZMAX) ? "Full depth. " : "Upper $(round(Int, ZMAX)) m (MIDLON_ZMAX). ") *
              "Colour limit ±$(round(lim, sigdigits = 2)) $unit" *
              (haskey(ENV, key) ? " (set by $key)." : " (99th percentile of |Δ$var| drawn; set $key to fix it)."), fontsize = 11)
        f = joinpath(OUTDIR, "$(RUN)_midlon_bias_$(var).png"); save(f, fig); @info "wrote $f"
    end

    fig = Figure(size = (900, 520))
    for (row, (var, D, unit)) in enumerate((("T", cΔT, "°C"), ("S", cΔS, "g/kg")))
        ax = Axis(fig[row, 1], ylabel = "Δ$var ($unit)", xticks = xticks_for(sect_months), xticklabelrotation = π / 3,
                  xticklabelsvisible = row == 2,
                  title = row == 1 ? "$RUN — section at $(round(λc[MID], digits = 1))°E, model − B-SOSE" : "")
        hlines!(ax, [0], color = :gray60)
        for (ks, colour, name) in ((UPPER, :firebrick, "0-200 m"), (DEEP, :navy, "200 m-floor"))
            scatterlines!(ax, 1:nm, [bias(D[k]; ks) for k in sect_months], color = colour, label = "mean $name")
            lines!(ax, 1:nm, [rms(D[k]; ks) for k in sect_months], color = colour, linestyle = :dash, label = "RMS $name")
        end
        row == 1 && axislegend(ax; position = :lt, framevisible = false, nbanks = 2)
    end
    f = joinpath(OUTDIR, "$(RUN)_midlon_bias_series.png"); save(f, fig); @info "wrote $f"
end
