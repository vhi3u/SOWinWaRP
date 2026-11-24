# ==============================================================================
# compare_nsidc.jl -- a run's sea-ice concentration against satellite observations (NOAA/NSIDC CDR, G02202 v6,
# 25 km), month by month, with B-SOSE's alongside: area, extent, where the ice edge is and where the run gets it wrong.
#
#     BSOSE_DIR=... julia --project=. tools/compare_nsidc.jl RUN
#     REF=other_run BSOSE_DIR=... julia --project=. tools/compare_nsidc.jl RUN     # a second run on the series figure
#
# DATA. NSIDC monthly files and the ancillary grid file, fetched by tools/download_nsidc.jl into NSIDC_DIR (default
# $BSOSE_DIR/nsidc). B-SOSE's 5-day SIarea (SIArea_bsoseI156_<year>_5day.nc in BSOSE_DIR, geometry from the monthly
# surfTflx file) is drawn when those files are there and skipped otherwise.
#
# ON NSIDC'S GRID. Everything is compared on NSIDC's 25 km polar stereographic cells: each 1/6-degree cell of the run
# (and of B-SOSE) is placed in the NSIDC cell holding its centre (EPSG:3412, Hughes 1980 ellipsoid, true scale at 70S;
# checked against the ancillary file's latitude/longitude to 0 m), and an NSIDC cell's value is the mean over the wet
# 1/6-degree cells in it. A cell is compared when NSIDC calls it ocean, its centre is inside the run's box, and over
# half of the 1/6-degree cells placed in it are wet in the run (and in B-SOSE when drawn). Cell area is (25 km)²/k²,
# k the map scale (1 at 70S, 1.14 at 45S).
#   The run: daily snapshots of ℵ (<RUN>_sea_ice.nc) averaged over each calendar month with MIN_DAYS or more.
#   B-SOSE: the 5-day SIarea windows whose centre (stamp - 2.5 days) falls in the month.
#
# NUMBERS, per month and region (all, 90-120E, 120-150E, east side 140-150E, coast = 1-2 NSIDC cells from land,
# offshore = 3+ cells from land):
#   area    Σ ℵ × cell area, 10³ km²
#   extent  Σ cell area where ℵ ≥ 0.15, 10³ km²
#   IIEE    integrated ice-edge error against NSIDC: the area where one says ice (ℵ ≥ 0.15) and the other does not,
#           split into over (too much ice) and under; 10³ km² (Goessling et al. 2016)
#   edge    the ice-edge latitude: in each 1° of longitude the northernmost compared cell with ℵ ≥ 0.15, median over
#           longitudes
#
# CAVEATS. Passive microwave blurs land into the 1-3 cells next to the coast and reads thin new ice (polynyas) low:
# the coast region is the least certain part of the observations. NSIDC has no thickness: volume stays a B-SOSE
# comparison (tools/plot_sea_ice.jl).
#
# WRITES (OUTDIR, default plots/):
#   <RUN>_nsidc_series.png   area, extent and IIEE through time per region: NSIDC black, B-SOSE dashed, run red, REF blue
#   <RUN>_nsidc_maps.png     run - NSIDC concentration each month on longitude/latitude axes (south of MAP_NORTH, -55),
#                            fixed colour limit ±0.6, NSIDC's 15% edge black, the run's red; the IIEE in each title
#   <RUN>_nsidc_edge.png     ice-edge latitude by longitude in March and September (and June, December) of each year
#
# LOGIN-NODE SAFE: NCDatasets and 2-D fields.
#
# ENV
#   DIR, OUTDIR, MIN_DAYS        run files directory (.), figures (plots), fewest days for a month (20)
#   NSIDC_DIR                    the NSIDC files ($BSOSE_DIR/nsidc)
#   REF, REF_DIR                 a second run, and its directory (DIR)
# ==============================================================================

using NCDatasets, CairoMakie, Printf, Statistics, Dates

const RUN       = length(ARGS) > 0 ? ARGS[1] : error("usage: julia --project=. tools/compare_nsidc.jl RUN")
const DIR       = get(ENV, "DIR", ".")
const OUTDIR    = get(ENV, "OUTDIR", "plots")
const MIN_DAYS  = parse(Int, get(ENV, "MIN_DAYS", "20"))
const BSOSE_DIR = get(ENV, "BSOSE_DIR", "")
const NSIDC_DIR = get(ENV, "NSIDC_DIR", isempty(BSOSE_DIR) ? joinpath("data", "nsidc") : joinpath(BSOSE_DIR, "nsidc"))
const REF       = get(ENV, "REF", "")
const REF_DIR   = get(ENV, "REF_DIR", DIR)
const EDGE      = 0.15
const MAP_LIM   = 0.6
mkpath(OUTDIR)

f64(A) = Float64.(coalesce.(A, NaN))
label(k) = @sprintf("%d-%02d", k...)

# ── NSIDC's grid: EPSG:3412 ─────────────────────────────────────────────────────────────────────
const A_E, INV_F = 6378273.0, 298.279411123064
const ECC = sqrt(2 / INV_F - 1 / INV_F^2)
tsnyder(φ) = tan(π / 4 - φ / 2) / ((1 - ECC * sin(φ)) / (1 + ECC * sin(φ)))^(ECC / 2)
msnyder(φ) = cos(φ) / sqrt(1 - ECC^2 * sin(φ)^2)
const φTS = deg2rad(70.0)
"""South polar stereographic (x, y) in metres and the map scale k, for longitude/latitude in degrees (Snyder 1987)."""
function pss(lon, lat)
    φ, λ = deg2rad(-lat), deg2rad(-lon)                  # mirrored onto the north pole
    ρ = A_E * msnyder(φTS) * tsnyder(φ) / tsnyder(φTS)
    return -ρ * sin(λ), ρ * cos(λ), ρ / (A_E * msnyder(φ))
end

anc_path = joinpath(NSIDC_DIR, "G02202-ancillary-pss25-v06r00.nc")
isfile(anc_path) || error("no $anc_path: run tools/download_nsidc.jl first (DEST = NSIDC_DIR)")
anc = NCDataset(anc_path)
gx, gy = f64(anc["x"][:]), f64(anc["y"][:])
glat, glon = f64(anc["latitude"][:, :]), f64(anc["longitude"][:, :])
surface = Int.(coalesce.(anc["surface_type"][:, :], 0))
adjacency = Int.(coalesce.(anc["adj123"][:, :], 0))      # 1, 2, 3 cells from land; 255 farther; 0 land
close(anc)
const DX = gx[2] - gx[1]; const DY = gy[2] - gy[1]      # 25 km; y decreases
nsidc_index(x, y) = (round(Int, (x - gx[1]) / DX) + 1, round(Int, (y - gy[1]) / DY) + 1)

# ── the run's geometry ──────────────────────────────────────────────────────────────────────────
function run_geometry(dir, run)
    ds = NCDataset(joinpath(dir, "$(run)_surface.nc"))
    λc, φc = f64(ds["λ_caa"][:]), f64(ds["φ_aca"][:])
    start = DateTime(string(ds.attrib["start_date"]))
    bh = ds["bottom_height"]
    wet = f64(ndims(bh) == 3 ? bh[:, :, 1] : bh[:, :]) .< 0
    close(ds)
    return λc, φc, start, wet
end
λc, φc, start, wet = run_geometry(DIR, RUN)
Nx, Ny = length(λc), length(φc)
Δλ, Δφ = λc[2] - λc[1], φc[2] - φc[1]
λW, λE, φS, φN = λc[1] - Δλ / 2, λc[end] + Δλ / 2, φc[1] - Δφ / 2, φc[end] + Δφ / 2

"""For each cell of a lon/lat grid, its NSIDC cell (CartesianIndex) or nothing outside NSIDC's grid."""
function placement(lons, lats)
    P = Array{Union{Nothing, CartesianIndex{2}}}(nothing, length(lons), length(lats))
    for (i, λ) in enumerate(lons), (j, φ) in enumerate(lats)
        x, y, _ = pss(λ, φ)
        I, J = nsidc_index(x, y)
        (1 <= I <= length(gx) && 1 <= J <= length(gy)) && (P[i, j] = CartesianIndex(I, J))
    end
    return P
end
"""Mean of `values` (NaN = no value) over the cells placed in each NSIDC cell, with the number placed and with a value."""
function binned(values, P)
    total, count = zeros(length(gx), length(gy)), zeros(Int, length(gx), length(gy))
    for c in eachindex(values)
        p = P[c]; (p === nothing || !isfinite(values[c])) && continue
        total[p] += values[c]; count[p] += 1
    end
    return [count[c] > 0 ? total[c] / count[c] : NaN for c in CartesianIndices(total)], count
end
placed_count(P) = (n = zeros(Int, length(gx), length(gy)); for p in P; p === nothing || (n[p] += 1); end; n)

Prun = placement(λc, φc)
n_placed = placed_count(Prun)
_, n_wet = binned(ifelse.(wet, 1.0, NaN), Prun)
cell_area = [(DX^2) / pss(glon[c], glat[c])[3]^2 for c in CartesianIndices(glat)]     # m²
inbox = [λW <= mod(glon[c], 360) <= λE && φS <= glat[c] <= φN for c in CartesianIndices(glat)]
compared = inbox .& (surface .== 50) .& (2 .* n_wet .> n_placed) .& (n_placed .> 0)

# ── monthly run concentration ───────────────────────────────────────────────────────────────────
function run_months(dir, run, start, wet, P)
    ice = NCDataset(joinpath(dir, "$(run)_sea_ice.nc"))
    t = f64(ice["time"][:])
    groups = Dict{Tuple{Int,Int},Vector{Int}}()
    for (n, s) in enumerate(t)
        s == 0 && continue                                # the initial state, copied from B-SOSE
        d = start + Millisecond(round(Int, 1000s))
        push!(get!(groups, (year(d), month(d)), Int[]), n)
    end
    v = ice["ℵ"]
    out = Dict{Tuple{Int,Int},Matrix{Float64}}()
    for (k, ns) in groups
        length(ns) >= MIN_DAYS || continue
        ℵ = sum(f64(ndims(v) == 4 ? v[:, :, 1, n] : v[:, :, n]) for n in ns) ./ length(ns)
        out[k] = first(binned(ifelse.(wet, ℵ, NaN), P))
    end
    close(ice)
    return out
end
run_ℵ = run_months(DIR, RUN, start, wet, Prun)
months = sort(collect(keys(run_ℵ)))
isempty(months) && error("$RUN: no month with $MIN_DAYS days of sea-ice output")

ref_ℵ = if isempty(REF)
    nothing
else
    rλ, rφ, rstart, rwet = run_geometry(REF_DIR, REF)
    (rλ == λc && rφ == φc) || error("REF $REF is on a different grid")
    run_months(REF_DIR, REF, rstart, rwet, Prun)
end

# ── NSIDC ───────────────────────────────────────────────────────────────────────────────────────
nsidc_files = Dict{Tuple{Int,Int},String}()
for f in readdir(NSIDC_DIR)
    m = match(r"^sic_pss25_(\d{4})(\d{2})_.*\.nc$", f)
    m === nothing || (nsidc_files[(parse(Int, m[1]), parse(Int, m[2]))] = joinpath(NSIDC_DIR, f))
end
missing_months = [k for k in months if !haskey(nsidc_files, k)]
isempty(missing_months) || error("NSIDC months missing in $NSIDC_DIR: $(join(label.(missing_months), ", ")) " *
                                 "(YEARS=$(first(missing_months)[1])-$(last(missing_months)[1]) julia tools/download_nsidc.jl)")
nsidc_ℵ = Dict(k => NCDataset(nsidc_files[k]) do d
                   c = f64(d["cdr_seaice_conc_monthly"][:, :, 1])
                   ifelse.((c .>= 0) .& (c .<= 1), c, NaN)
               end for k in months)

# ── B-SOSE (optional) ───────────────────────────────────────────────────────────────────────────
bsose_ℵ = let
    geom = joinpath(BSOSE_DIR, "surfTflx_bsoseI156_2013to2024_monthly.nc")
    years = unique(first.(months)) ∪ unique(first.(months) .+ 1)
    files = Dict(y => joinpath(BSOSE_DIR, "SIArea_bsoseI156_$(y)_5day.nc") for y in years)
    needed = [files[y] for y in unique(first.(months))]
    if isempty(BSOSE_DIR) || !isfile(geom) || !all(isfile, needed)
        @warn "B-SOSE not drawn: needs BSOSE_DIR with $(basename(geom)) and $(join(basename.(needed), ", "))"
        nothing
    else
        gds = NCDataset(geom)
        xc, yc = f64(gds["XC"][:]), f64(gds["YC"][:])
        I = findall(x -> λW <= x <= λE, xc); J = findall(y -> φS <= y <= φN, yc)
        IR, JR = minimum(I):maximum(I), minimum(J):maximum(J)
        bwet = f64(gds["Depth"][IR, JR]) .> 0
        close(gds)
        Pb = placement(xc[IR], yc[JR])
        nb_placed = placed_count(Pb); _, nb_wet = binned(ifelse.(bwet, 1.0, NaN), Pb)
        global compared = compared .& (2 .* nb_wet .> nb_placed)
        sums = Dict{Tuple{Int,Int},Tuple{Matrix{Float64},Int}}()
        for y in years
            isfile(files[y]) || continue
            NCDataset(files[y]) do a
                av = first(k for k in ("SIarea", "SIArea", "SIAREA") if haskey(a, k))
                for (n, s) in enumerate(a["time"][:])
                    c = DateTime(s) - Hour(60)
                    k = (year(c), month(c)); k in months || continue
                    A = ifelse.(bwet, f64(a[av][IR, JR, n]), NaN)
                    S, m = get(sums, k, (zeros(size(A)), 0))
                    sums[k] = (S .+ ifelse.(isfinite.(A), A, 0.0), m + 1)
                end
            end
        end
        Dict(k => first(binned(ifelse.(bwet, S ./ m, NaN), Pb)) for (k, (S, m)) in sums)
    end
end

# ── metrics ─────────────────────────────────────────────────────────────────────────────────────
# Every product must have a value in a compared cell, in every month
for k in months, D in filter(!isnothing, (run_ℵ, ref_ℵ, bsose_ℵ))
    haskey(D, k) || continue
    global compared = compared .& isfinite.(D[k]) .& isfinite.(nsidc_ℵ[k])
end
count(compared) == 0 && error("no NSIDC cell is compared: check the run's box and wet mask")

lonE(c) = mod(glon[c], 360)
const REGIONS = [("all", compared),
                 ("90-120E", compared .& [lonE(c) < 120 for c in CartesianIndices(glat)]),
                 ("120-150E", compared .& [lonE(c) >= 120 for c in CartesianIndices(glat)]),
                 ("east 140-150E", compared .& [lonE(c) >= 140 for c in CartesianIndices(glat)]),
                 ("coast (1-2 cells)", compared .& (1 .<= adjacency .<= 2)),
                 ("offshore (3+ cells)", compared .& (adjacency .>= 3))]

ok(A, B) = isfinite(A) && isfinite(B)
area(A, mask)   = sum((mask[c] && isfinite(A[c])) ? A[c] * cell_area[c] : 0.0 for c in eachindex(A)) / 1e9
extent(A, mask) = sum((mask[c] && isfinite(A[c]) && A[c] >= EDGE) ? cell_area[c] : 0.0 for c in eachindex(A)) / 1e9
"""Integrated ice-edge error of A against the observations O: (over, under), 10³ km²."""
function iiee(A, O, mask)
    over = under = 0.0
    for c in eachindex(A)
        (mask[c] && ok(A[c], O[c])) || continue
        a, o = A[c] >= EDGE, O[c] >= EDGE
        a & !o && (over += cell_area[c]); o & !a && (under += cell_area[c])
    end
    return over / 1e9, under / 1e9
end
const EDGE_LONS = 90.5:1.0:149.5
"""Northernmost compared latitude with ℵ ≥ EDGE in each 1° of longitude (NaN where there is no ice)."""
function edge_by_longitude(A)
    e = fill(NaN, length(EDGE_LONS))
    for c in eachindex(A)
        (compared[c] && isfinite(A[c]) && A[c] >= EDGE) || continue
        b = clamp(floor(Int, lonE(c) - 90) + 1, 1, length(EDGE_LONS))
        e[b] = isnan(e[b]) ? glat[c] : max(e[b], glat[c])
    end
    return e
end
edge_median(A) = (e = filter(isfinite, edge_by_longitude(A)); isempty(e) ? NaN : median(e))

sources = [("NSIDC", nsidc_ℵ), ("run", run_ℵ)]
ref_ℵ === nothing || push!(sources, ("REF", ref_ℵ))
bsose_ℵ === nothing || push!(sources, ("B-SOSE", bsose_ℵ))
val(D, k, f) = haskey(D, k) ? f(D[k]) : NaN
cell(v, w = 7, d = 0) = isnan(v) ? lpad("--", w) : lpad(@sprintf("%.*f", d, v), w)

println("\n", "="^110)
println("SEA-ICE CONCENTRATION, $RUN against NSIDC CDR v6 (25 km)" * (isempty(REF) ? "" : "; REF = $REF") *
        (bsose_ℵ === nothing ? "" : "; B-SOSE alongside"))
println("  area and extent 10³ km²; IIEE over/under 10³ km² (ice where NSIDC has none / none where it has ice);")
println("  $(count(compared)) NSIDC cells compared ($(round(Int, sum(cell_area[compared]) / 1e9)) 10³ km²)")
println("="^110)
names_ = first.(sources)
for (rname, mask) in REGIONS
    count(mask) == 0 && continue
    println("\n  $(uppercase(rname)), $(count(mask)) cells")
    @printf("  %-8s |%s |%s |%s\n", "month", join(lpad.("A " .* names_, 9)), join(lpad.("E " .* names_, 9)),
            join(lpad.([n * " ov/un" for n in names_[2:end]], 14)))
    for k in months
        A = [val(D, k, X -> area(X, mask)) for (_, D) in sources]
        E = [val(D, k, X -> extent(X, mask)) for (_, D) in sources]
        I = [haskey(D, k) ? iiee(D[k], nsidc_ℵ[k], mask) : (NaN, NaN) for (_, D) in sources[2:end]]
        @printf("  %-8s |%s |%s |%s\n", label(k), join(cell.(A, 9)), join(cell.(E, 9)),
                join([cell(o, 7) * cell(u, 7) for (o, u) in I]))
    end
end

println("\n  ICE-EDGE LATITUDE (median over 1° of longitude of the northernmost ℵ ≥ $EDGE), °S")
@printf("  %-8s |%s\n", "month", join(lpad.(names_, 9)))
for k in months
    @printf("  %-8s |%s\n", label(k), join(cell.([-val(D, k, edge_median) for (_, D) in sources], 9, 1)))
end
println("\n  WHOLE RECORD, mean over months: area bias and IIEE (10³ km²), all compared cells")
for (n, D) in sources[2:end]
    ks = [k for k in months if haskey(D, k)]
    ab = mean(area(D[k], compared) - area(nsidc_ℵ[k], compared) for k in ks)
    ii = [iiee(D[k], nsidc_ℵ[k], compared) for k in ks]
    @printf("  %-8s area bias %+8.0f   IIEE %7.0f (over %6.0f, under %6.0f)\n", n, ab,
            mean(sum.(ii)), mean(first.(ii)), mean(last.(ii)))
end

# ── figures ─────────────────────────────────────────────────────────────────────────────────────
style = Dict("NSIDC" => (color = :black, linestyle = :solid), "run" => (color = :firebrick, linestyle = :solid),
             "REF" => (color = :royalblue, linestyle = :solid), "B-SOSE" => (color = :black, linestyle = :dash))
tlabel(n) = n == "run" ? RUN : n == "REF" ? REF : n
x = 1:length(months)
xt = (collect(x)[1:3:end], label.(months)[1:3:end])
series_regions = REGIONS[[1, 2, 3, 5]]
fig = Figure(size = (1500, 300 * length(series_regions) + 80))
Label(fig[0, 1:3], "$RUN: sea-ice concentration against NSIDC CDR v6", fontsize = 18, font = :bold)
for (r, (rname, mask)) in enumerate(series_regions)
    for (c, (what, f)) in enumerate((("area (10³ km²)", X -> area(X, mask)), ("extent, ℵ ≥ 0.15 (10³ km²)", X -> extent(X, mask))))
        ax = Axis(fig[r, c]; title = "$rname: $what", xticks = xt, xticklabelrotation = π / 4, xticklabelsvisible = r == length(series_regions))
        for (n, D) in sources
            lines!(ax, x, [val(D, k, f) for k in months]; linewidth = 2, label = tlabel(n), style[n]...)
        end
        r == 1 && c == 1 && axislegend(ax; position = :lt, framevisible = false, labelsize = 10)
    end
    ax = Axis(fig[r, 3]; title = "$rname: IIEE against NSIDC (10³ km²)", xticks = xt, xticklabelrotation = π / 4,
              xticklabelsvisible = r == length(series_regions))
    for (n, D) in sources[2:end]
        lines!(ax, x, [haskey(D, k) ? sum(iiee(D[k], nsidc_ℵ[k], mask)) : NaN for k in months]; linewidth = 2, style[n]...)
    end
end
f = joinpath(OUTDIR, "$(RUN)_nsidc_series.png"); save(f, fig); @info "wrote $f"

# Maps: NSIDC's cells drawn at their own longitude and latitude (a curvilinear surface), so they read like the other
# comparison figures; cropped to the compared cells and to south of MAP_NORTH, where the ice is.
const MAP_NORTH = parse(Float64, get(ENV, "MAP_NORTH", "-55"))
rows = findall(i -> any(compared[i, :]), axes(compared, 1)); cols = findall(j -> any(compared[:, j]), axes(compared, 2))
IR = max(minimum(rows) - 1, 1):min(maximum(rows) + 1, length(gx))
JR = max(minimum(cols) - 1, 1):min(maximum(cols) + 1, length(gy))
LON, LAT = mod.(glon[IR, JR], 360), glat[IR, JR]
crop(A) = A[IR, JR]
let nm = length(months), ncol = min(nm, 6), nrow = cld(nm, ncol)
fig = Figure(size = (330 * ncol + 120, 200 * nrow + 140))
Label(fig[0, 1:ncol], "$RUN − NSIDC, sea-ice concentration, monthly mean", fontsize = 18, font = :bold)
local hm
for (n, k) in enumerate(months)
    r, c = fldmod1(n, ncol)
    D = ifelse.(compared, run_ℵ[k] .- nsidc_ℵ[k], NaN)
    ov, un = iiee(run_ℵ[k], nsidc_ℵ[k], compared)
    ax = Axis(fig[r, c]; title = label(k), subtitle = @sprintf("IIEE %.0f (over %.0f, under %.0f) 10³ km²", ov + un, ov, un),
              titlesize = 12, subtitlesize = 9, xticklabelsvisible = r == nrow, yticklabelsvisible = c == 1,
              xticklabelsize = 9, yticklabelsize = 9, backgroundcolor = :gray85)
    hm = surface!(ax, LON, LAT, zeros(size(LON)); color = crop(D), colormap = :balance, colorrange = (-MAP_LIM, MAP_LIM),
                  nan_color = :gray85, shading = NoShading)
    for (A, col) in ((nsidc_ℵ[k], :black), (run_ℵ[k], :firebrick))
        contour!(ax, LON, LAT, crop(ifelse.(compared, A, 0.0)); levels = [EDGE], color = col, linewidth = 1)
    end
    limits!(ax, λW, λE, φS, MAP_NORTH)
end
Colorbar(fig[1:nrow, ncol + 1], hm; label = "Δℵ, run − NSIDC (red: run has more ice)")
Label(fig[nrow + 1, 1:ncol], "NSIDC's 25 km cells at their longitude and latitude. Black: NSIDC's 15% edge; red: the run's. " *
      "Grey: land or not compared. Colour limit fixed at ±$MAP_LIM for every run.", fontsize = 11)
f = joinpath(OUTDIR, "$(RUN)_nsidc_maps.png"); save(f, fig); @info "wrote $f"
end

edge_months = [k for k in months if k[2] in (3, 6, 9, 12)]
if !isempty(edge_months)
    ncol = min(length(edge_months), 4); nrow = cld(length(edge_months), ncol)
    fig = Figure(size = (max(400 * ncol, 760), 280 * nrow + 110))
    local ax1
    Label(fig[0, 1:ncol], "$RUN: ice edge (northernmost ℵ ≥ 0.15) by longitude", fontsize = 16, font = :bold)
    for (n, k) in enumerate(edge_months)
        r, c = fldmod1(n, ncol)
        ax = Axis(fig[r, c]; title = label(k), xlabel = r == nrow ? "longitude (°E)" : "", ylabel = c == 1 ? "latitude" : "")
        for (nm_, D) in sources
            haskey(D, k) && lines!(ax, collect(EDGE_LONS), edge_by_longitude(D[k]); linewidth = 2, label = tlabel(nm_), style[nm_]...)
        end
        n == 1 && (ax1 = ax)
    end
    Legend(fig[nrow + 1, 1:ncol], ax1; orientation = :horizontal, framevisible = false)
    f = joinpath(OUTDIR, "$(RUN)_nsidc_edge.png"); save(f, fig); @info "wrote $f"
end
