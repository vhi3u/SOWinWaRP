# ==============================================================================
# compare_runs.jl -- one run against another, day by day and month by month: the SURFACE, the
# SEA ICE and the MIDLON SECTION. The run-against-run counterpart of tools/compare_bsose.jl; made
# for the GLORYS baseline against the B-SOSE nest.
#
#     julia --project=. tools/compare_runs.jl                                   # fix8 vs the GLORYS baseline
#     julia --project=. tools/compare_runs.jl 2Y_nested_seaice_fix8 2Y_nested_glorys_baseline
#
# Differences are B - A: RUN_B (default 2Y_nested_glorys_baseline) minus RUN_A (default
# 2Y_nested_seaice_fix8). Red: B warmer / saltier / more ice than A.
#
# WHAT IS COMPARED. Both runs write daily output on the same 360x150x52 grid (checked), but over
# different sea floors (fix8: B-SOSE's; the baseline: ETOPO2022), so a cell counts only where BOTH
# are wet. Monthly means are over the days of each calendar month both runs have (MIN_DAYS or more),
# leaving out each run's t = 0 snapshot. Either run may still be running: only the days and months
# both have are compared.
#
#     surface   <RUN>_surface.nc, the top cell
#     sea ice   <RUN>_sea_ice.nc: area Σℵ·A and volume Σh·ℵ·A over the domain, and ℵ maps
#     midlon    <RUN>_midlon.nc, the latitude-depth section at the middle longitude (~120E)
#
# WRITES (OUTDIR, default plots/), prefix <A>_vs_<B>:
#     _series.png                daily domain-mean SST and SSS, ice area and volume, both runs
#     _surface_T.png, _S.png     one map per month, B - A
#     _ice_concentration.png     one map per month of mean ℵ, B - A
#     _midlon_T.png, _S.png      one latitude-depth panel per month, B - A
# and on stdout, month by month: the surface differences (domain and by latitude band), both runs'
# ice area and volume, and the section differences by layer.
#
# To compare either run with B-SOSE itself, use tools/compare_bsose.jl on each.
#
# ENV
#     DIR          where both runs' files are                       (default .)
#     DIR_A, DIR_B where each run's files are, if apart             (default DIR)
#     OUTDIR       figures                                          (default plots)
#     MIN_DAYS     fewest common days of output for a month         (default 20)
#     MIDLON_ZMAX  depth drawn in the section maps, m, or "full"    (default 1000)
#     T_LIM, S_LIM, A_LIM, ST_LIM, SS_LIM   colour limits for the surface T/S, ice concentration and
#                  section T/S maps (default: the 99th percentile of |Δ| over the months drawn)
# ==============================================================================

using NCDatasets, CairoMakie, Printf, Statistics, Dates

include(joinpath(@__DIR__, "model_grid.jl"))   # file_levels, file_kbot, section_column

const RUN_A  = length(ARGS) >= 1 ? ARGS[1] : get(ENV, "RUN_A", "2Y_nested_seaice_fix8")
const RUN_B  = length(ARGS) >= 2 ? ARGS[2] : get(ENV, "RUN_B", "2Y_nested_glorys_baseline")
const DIR    = get(ENV, "DIR", ".")
const DIR_A  = get(ENV, "DIR_A", DIR)
const DIR_B  = get(ENV, "DIR_B", DIR)
const OUTDIR = get(ENV, "OUTDIR", "plots")
const MIN_DAYS = parse(Int, get(ENV, "MIN_DAYS", "20"))
const ZMAX   = (s = get(ENV, "MIDLON_ZMAX", "1000"); s == "full" ? Inf : parse(Float64, s))
const PREFIX = "$(RUN_A)_vs_$(RUN_B)"
mkpath(OUTDIR)

f64(A) = Float64.(coalesce.(A, NaN))
file(dir, run, kind) = joinpath(dir, "$(run)_$(kind).nc")
for (dir, run) in ((DIR_A, RUN_A), (DIR_B, RUN_B)), kind in ("surface", "sea_ice")
    isfile(file(dir, run, kind)) || error("no $(file(dir, run, kind)); set DIR (or DIR_A/DIR_B) to where the runs are")
end
HAVE_SECT = isfile(file(DIR_A, RUN_A, "midlon")) && isfile(file(DIR_B, RUN_B, "midlon"))
HAVE_SECT || @warn "a midlon file is missing: no section comparison"

# ── each run: grid, dates, wet cells ──────────────────────────────────────────────────────────
"""A record `n` of a (Nx, Ny) field, in the file's 4-D (x, y, 1, t) or 3-D (x, y, t) layout."""
record(v, n) = f64(ndims(v) == 4 ? v[:, :, 1, n] : v[:, :, n])

function run_info(dir, run)
    ds = Dataset(file(dir, run, "surface"))
    λc, φc = f64(ds["λ_caa"][:]), f64(ds["φ_aca"][:])
    haskey(ds.attrib, "start_date") || error("$(file(dir, run, "surface")) has no start_date attribute")
    start = DateTime(string(ds.attrib["start_date"]))
    times = f64(ds["time"][:])
    wet = let m = ds["inactive_nodes_ccc"]
        (ndims(m) == 3 ? m[:, :, end] : m[:, :]) .== 0
    end
    close(ds)
    di = Dataset(file(dir, run, "sea_ice"))
    ice_times = f64(di["time"][:])
    area = f64(di["Δx_cca"][:, :]) .* f64(di["Δy_cca"][:, :])
    close(di)
    return (; dir, run, λc, φc, start, times, wet, ice_times, area)
end

A, B = run_info(DIR_A, RUN_A), run_info(DIR_B, RUN_B)
(length(A.λc), length(A.φc)) == (length(B.λc), length(B.φc)) && maximum(abs, A.λc .- B.λc) < 1e-6 &&
    maximum(abs, A.φc .- B.φc) < 1e-6 || error("$RUN_A and $RUN_B are not on the same horizontal grid")
const λc, φc = A.λc, A.φc
const Nx, Ny = length(λc), length(φc)
const WET = BitMatrix(A.wet .& B.wet)
const AREA = A.area   # m², the same grid
@info @sprintf("%s (A, from %s) vs %s (B, from %s): %dx%d, %d surface cells wet in both (%d only in A, %d only in B)",
               RUN_A, A.start, RUN_B, B.start, Nx, Ny, count(WET), count(A.wet .& .!B.wet), count(B.wet .& .!A.wet))

# Records by calendar day; a day's output is its last record that day (daily output: one record)
dates_of(r, times) = [r.start + Millisecond(round(Int, 1000t)) for t in times]
function by_day(r, times)
    d = Dict{Date, Int}()
    for (n, t) in enumerate(dates_of(r, times))
        times[n] == 0 && continue                                  # the initial condition
        d[Date(t)] = n
    end
    return d
end
common_days(da, db) = sort(collect(intersect(keys(da), keys(db))))

month_of(d) = (year(d), month(d))
label(k) = @sprintf("%d-%02d", k...)
function months_of(days)
    g = Dict{Tuple{Int,Int}, Vector{Date}}()
    foreach(d -> push!(get!(g, month_of(d), Date[]), d), days)
    return sort([k for (k, v) in g if length(v) >= MIN_DAYS]), g
end

# ── statistics ────────────────────────────────────────────────────────────────────────────────
const BANDS = [(-70.0, -65.0), (-65.0, -60.0), (-60.0, -55.0), (-55.0, -50.0), (-50.0, -45.0)]
"""cos(lat)-weighted mean of the finite cells of a (Nx, Ny) map with latitude in [lo, hi)."""
function wmean(X, lo = -90.0, hi = 90.0)
    num = den = 0.0
    for i in axes(X, 1), j in axes(X, 2)
        (lo <= φc[j] < hi && isfinite(X[i, j])) || continue
        w = cosd(φc[j]); num += w * X[i, j]; den += w
    end
    return den == 0 ? NaN : num / den
end
cell(v, fmt) = isnan(v) ? lpad("--", 8) : Printf.format(Printf.Format(fmt), v)
masked(X) = ifelse.(WET, X, NaN)

# ── surface: daily means, and monthly mean maps ───────────────────────────────────────────────
sA, sB = by_day(A, A.times), by_day(B, B.times)
surf_days = common_days(sA, sB)
isempty(surf_days) && error("$RUN_A and $RUN_B have no day of surface output in common")
surf_months, surf_groups = months_of(surf_days)
@info "surface: $(length(surf_days)) common days, $(first(surf_days)) to $(last(surf_days)); $(length(surf_months)) whole months"

series = Dict(k => (Float64[], Float64[]) for k in (:SST, :SSS, :area, :volume))
ΔT, ΔS, Δℵ = (Dict{Tuple{Int,Int}, Matrix{Float64}}() for _ in 1:3)

let da = Dataset(file(A.dir, A.run, "surface")), db = Dataset(file(B.dir, B.run, "surface"))
    acc = Dict(k => (zeros(Nx, Ny), zeros(Nx, Ny)) for k in surf_months)
    for d in surf_days
        TA, TB = record(da["T"], sA[d]), record(db["T"], sB[d])
        SA, SB = record(da["S"], sA[d]), record(db["S"], sB[d])
        push!(series[:SST][1], wmean(masked(TA))); push!(series[:SST][2], wmean(masked(TB)))
        push!(series[:SSS][1], wmean(masked(SA))); push!(series[:SSS][2], wmean(masked(SB)))
        k = month_of(d)
        haskey(acc, k) || continue
        acc[k][1] .+= TB .- TA
        acc[k][2] .+= SB .- SA
    end
    for k in surf_months
        n = length(surf_groups[k])
        ΔT[k], ΔS[k] = masked(acc[k][1] ./ n), masked(acc[k][2] ./ n)
    end
    close(da); close(db)
end

# ── sea ice: daily area and volume, monthly mean concentration maps ───────────────────────────
iA, iB = by_day(A, A.ice_times), by_day(B, B.ice_times)
ice_days = common_days(iA, iB)
ice_months, ice_groups = months_of(ice_days)
ice_month_area = Dict{Tuple{Int,Int}, NTuple{4,Float64}}()   # area A, area B (10⁶ km²), volume A, B (km³)
let da = Dataset(file(A.dir, A.run, "sea_ice")), db = Dataset(file(B.dir, B.run, "sea_ice"))
    acc = Dict(k => zeros(Nx, Ny) for k in ice_months)
    sums = Dict(k => zeros(4) for k in ice_months)
    for d in ice_days
        hA, hB = record(da["h"], iA[d]), record(db["h"], iB[d])
        ℵA, ℵB = record(da["ℵ"], iA[d]), record(db["ℵ"], iB[d])
        tot(X) = sum(x * a for (x, a) in zip(X, AREA) if isfinite(x))
        a = (tot(ℵA) / 1e12, tot(ℵB) / 1e12, tot(hA .* ℵA) / 1e9, tot(hB .* ℵB) / 1e9)
        push!(series[:area][1], a[1]); push!(series[:area][2], a[2])
        push!(series[:volume][1], a[3]); push!(series[:volume][2], a[4])
        k = month_of(d)
        haskey(acc, k) || continue
        acc[k] .+= ifelse.(isnan.(ℵB), 0.0, ℵB) .- ifelse.(isnan.(ℵA), 0.0, ℵA)
        sums[k] .+= collect(a)
    end
    for k in ice_months
        n = length(ice_groups[k])
        Δℵ[k] = masked(acc[k] ./ n)
        ice_month_area[k] = Tuple(sums[k] ./ n)
    end
    close(da); close(db)
end

# ── midlon section: monthly mean differences ──────────────────────────────────────────────────
cΔT, cΔS = Dict{Tuple{Int,Int}, Matrix{Float64}}(), Dict{Tuple{Int,Int}, Matrix{Float64}}()
sect_months = Tuple{Int,Int}[]
if HAVE_SECT
    pa, pb = file(A.dir, A.run, "midlon"), file(B.dir, B.run, "midlon")
    section_column(pa, Nx) == section_column(pb, Nx) || error("the two midlon sections are at different longitudes")
    global MID = section_column(pa, Nx)
    da, db = Dataset(pa), Dataset(pb)
    la, lb = file_levels(da), file_levels(db)
    (la === nothing || lb === nothing) && error("a midlon file carries no vertical levels")
    length(la[1]) == length(lb[1]) && maximum(abs, la[1] .- lb[1]) < 1e-6 ||
        error("$RUN_A and $RUN_B have different vertical levels")
    global zc, zf = la[1], la[2]
    global Nz = length(zc)
    kA, kB = file_kbot(da, la[1], la[2])[1], file_kbot(db, lb[1], lb[2])[1]
    kbA = size(kA, 1) == 1 ? vec(kA) : kA[MID, :]
    kbB = size(kB, 1) == 1 ? vec(kB) : kB[MID, :]
    wetboth = [kbA[j] !== nothing && kbB[j] !== nothing && k >= kbA[j] && k >= kbB[j] for j in 1:Ny, k in 1:Nz]
    cA, cB = by_day(A, f64(da["time"][:])), by_day(B, f64(db["time"][:]))
    global sect_months, sect_groups = months_of(common_days(cA, cB))
    sect(ds, n) = f64(ds["T"][1, :, :, n]), f64(ds["S"][1, :, :, n])
    for k in sect_months
        aT, aS = zeros(Ny, Nz), zeros(Ny, Nz)
        for d in sect_groups[k]
            (TA, SA), (TB, SB) = sect(da, cA[d]), sect(db, cB[d])
            aT .+= TB .- TA; aS .+= SB .- SA
        end
        n = length(sect_groups[k])
        cΔT[k] = ifelse.(wetboth, aT ./ n, NaN)
        cΔS[k] = ifelse.(wetboth, aS ./ n, NaN)
    end
    close(da); close(db)
    global UPPER = [k for k in 1:Nz if -zc[k] <= 200]
    global DEEP  = [k for k in 1:Nz if -zc[k] > 200]
end
"""cos(lat)-weighted mean over the finite cells of a (Ny, Nz) section at levels ks."""
function smean(X, ks)
    num = den = 0.0
    for j in axes(X, 1), k in ks
        isfinite(X[j, k]) || continue
        w = cosd(φc[j]); num += w * X[j, k]; den += w
    end
    return den == 0 ? NaN : num / den
end

# ── tables ────────────────────────────────────────────────────────────────────────────────────
println()
println("="^100)
println("MONTHLY MEAN  B - A   (A = $RUN_A, B = $RUN_B)")
println("  cos(lat)-weighted over cells wet in both. Positive ΔT: B warmer; positive ΔS: B saltier.")
println("="^100)
for (name, D, fmt) in (("SST, ΔT (°C)", ΔT, "%+7.2f"), ("SSS, ΔS (g/kg)", ΔS, "%+7.3f"))
    @printf("\n  SURFACE %s, domain and by latitude band\n  %-8s | %7s |", name, "month", "domain")
    for (lo, hi) in BANDS; @printf(" %7s", "$(Int(-lo))-$(Int(-hi))S"); end
    println()
    for k in surf_months
        @printf("  %-8s | %s |", label(k), cell(wmean(D[k]), fmt))
        for (lo, hi) in BANDS; print(" ", cell(wmean(D[k], lo, hi), fmt)); end
        println()
    end
end
println("\n  SEA ICE, monthly mean")
@printf("  %-8s | %9s %9s %9s | %9s %9s %9s\n", "month", "area A", "area B", "B/A", "vol A", "vol B", "B/A")
@printf("  %-8s | %9s %9s %9s | %9s %9s %9s\n", "", "10⁶ km²", "10⁶ km²", "", "km³", "km³", "")
for k in ice_months
    aA, aB, vA, vB = ice_month_area[k]
    ratio(b, a) = a > 0 ? @sprintf("%9.2f", b / a) : lpad("--", 9)
    @printf("  %-8s | %9.3f %9.3f %s | %9.0f %9.0f %s\n", label(k), aA, aB, ratio(aB, aA), vA, vB, ratio(vB, vA))
end
if !isempty(sect_months)
    @printf("\n  MIDLON SECTION at %.1fE, B - A by layer\n", λc[MID])
    @printf("  %-8s | %9s %9s | %9s %9s\n", "month", "ΔT 0-200", "200-flr", "ΔS 0-200", "200-flr")
    for k in sect_months
        @printf("  %-8s | %s  %s | %s  %s\n", label(k), cell(smean(cΔT[k], UPPER), "%+8.3f"), cell(smean(cΔT[k], DEEP), "%+8.3f"),
                cell(smean(cΔS[k], UPPER), "%+8.4f"), cell(smean(cΔS[k], DEEP), "%+8.4f"))
    end
end
println()

# ── figures ───────────────────────────────────────────────────────────────────────────────────
# Every Label has tellwidth = false: an Axis asks for no width, so a Label that did would set the column
# widths (a short title squeezes the panels; a long footnote runs off the page).
function colour_limit(Ds, key)
    haskey(ENV, key) && return parse(Float64, ENV[key])
    v = filter(isfinite, abs.(reduce(vcat, vec.(Ds))))
    return isempty(v) ? 1.0 : max(quantile(v, 0.99), 1e-6)
end

# Daily series, both runs
let fig = Figure(size = (1000, 950))
    Label(fig[0, 1], "$RUN_A (A) and $RUN_B (B), daily", fontsize = 18, font = :bold, tellwidth = false)
    rows = ((:SST, ("SST (°C)", surf_days)), (:SSS, ("SSS (g/kg)", surf_days)),
            (:area, ("ice area (10⁶ km²)", ice_days)), (:volume, ("ice volume (km³)", ice_days)))
    for (r, (key, (ylabel, days))) in enumerate(rows)
        ax = Axis(fig[r, 1]; ylabel, xticklabelsvisible = r == length(rows),
                  title = r == 1 ? "domain means over cells wet in both; ice totals over the domain" : "")
        x = [Dates.value(d - first(days)) for d in days]
        lines!(ax, x, series[key][1], color = :navy, label = "A  $RUN_A")
        lines!(ax, x, series[key][2], color = :firebrick, label = "B  $RUN_B")
        r == 1 && axislegend(ax; position = :rt, framevisible = false)
        r == length(rows) && (ax.xlabel = "days since $(first(days))")
    end
    f = joinpath(OUTDIR, "$(PREFIX)_series.png"); save(f, fig); @info "wrote $f"
end

# Maps, one per month, B - A
function month_maps(D, months, var, unit, key, more, less, fname)
    isempty(months) && return
    nm = length(months); ncol = min(nm, 6); nrow = cld(nm, ncol)
    lim = colour_limit([D[k] for k in months], key)
    fig = Figure(size = (260 * ncol + 120, 190 * nrow + 120))
    Label(fig[0, 1:ncol], "$var, monthly mean, $RUN_B − $RUN_A", fontsize = 18, font = :bold, tellwidth = false)
    local hm
    for (n, k) in enumerate(months)
        r, c = fldmod1(n, ncol)
        ax = Axis(fig[r, c], title = label(k), titlesize = 12, xticklabelsvisible = r == nrow,
                  yticklabelsvisible = c == 1, xticklabelsize = 9, yticklabelsize = 9)
        hm = heatmap!(ax, λc, φc, D[k], colormap = :balance, colorrange = (-lim, lim), nan_color = :gray80)
    end
    Colorbar(fig[1:nrow, ncol + 1], hm, label = isempty(unit) ? "Δ" : "Δ ($unit)")
    Label(fig[nrow + 1, 1:ncol], "red: B $more; blue: B $less. Grey: land in either run. Colour limit ±$(round(lim, sigdigits = 2))$(isempty(unit) ? "" : " " * unit)" *
          (haskey(ENV, key) ? " (set by $key)." : " (99th percentile of |Δ|; set $key to fix it)."), fontsize = 11, tellwidth = false, word_wrap = true)
    f = joinpath(OUTDIR, "$(PREFIX)_$(fname).png"); save(f, fig); @info "wrote $f"
end
month_maps(ΔT, surf_months, "surface T", "°C", "T_LIM", "warmer", "colder", "surface_T")
month_maps(ΔS, surf_months, "surface S", "g/kg", "S_LIM", "saltier", "fresher", "surface_S")
month_maps(Δℵ, ice_months, "ice concentration ℵ", "", "A_LIM", "more ice", "less ice", "ice_concentration")

# Sections, one latitude-depth panel per month, B - A
!isempty(sect_months) && let
    nm = length(sect_months); ncol = min(nm, 6); nrow = cld(nm, ncol)
    kdraw = [k for k in 1:Nz if -zc[k] <= ZMAX]
    zbottom = isinf(ZMAX) ? zf[1] : -ZMAX
    for (var, D, unit, key, more, less) in (("T", cΔT, "°C", "ST_LIM", "warmer", "colder"), ("S", cΔS, "g/kg", "SS_LIM", "saltier", "fresher"))
        lim = colour_limit([D[k][:, kdraw] for k in sect_months], key)
        fig = Figure(size = (280 * ncol + 120, 200 * nrow + 120))
        Label(fig[0, 1:ncol], "$var along $(round(λc[MID], digits = 1))°E, monthly mean, $RUN_B − $RUN_A", fontsize = 18, font = :bold, tellwidth = false)
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
        Label(fig[nrow + 1, 1:ncol], "red: B $more; blue: B $less. Grey: land in either run. " *
              (isinf(ZMAX) ? "Full depth. " : "Upper $(round(Int, ZMAX)) m (MIDLON_ZMAX). ") *
              "Colour limit ±$(round(lim, sigdigits = 2)) $unit" *
              (haskey(ENV, key) ? " (set by $key)." : " (99th percentile of |Δ$var| drawn; set $key to fix it)."), fontsize = 11, tellwidth = false, word_wrap = true)
        f = joinpath(OUTDIR, "$(PREFIX)_midlon_$(var).png"); save(f, fig); @info "wrote $f"
    end
end
