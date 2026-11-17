# ==============================================================================
# bsose_ice_extent.jl -- B-SOSE's sea-ice extent, area and volume over our box (90-150E, 70-45S), from its 5-day
# SIarea and SIheff, with each year's maximum and minimum: the yardstick for a run's ice as it goes.
#
#     julia --project=. tools/bsose_ice_extent.jl                  # 2014-2016
#     YEARS=2014,2015 julia --project=. tools/bsose_ice_extent.jl
#
#   extent  the area of cells with concentration above 0.15 (10⁶ km²), the usual definition
#   area    Σ ℵ · cell area (10⁶ km²), what the runs' startup line calls "area"
#   volume  Σ SIheff · cell area (km³); SIheff is ice volume per unit cell area
# Cell areas from the cell-centre coordinates (B-SOSE's latitude spacing is Mercator-like), so no grid file is needed.
#
# Reads <data>/SIArea_bsoseI156_<year>_5day.nc and SIHeff_..., the files the nested scripts read. LOGIN-NODE SAFE.
# ==============================================================================

using NCDatasets, Dates, Printf

const BDIR  = get(ENV, "BSOSE_DIR", "/storage/scratch1/1/vnguyen480/SOWinWaRP/data")
const YEARS = parse.(Int, split(get(ENV, "YEARS", "2014,2015,2016"), ","))
const LON, LAT = (90.0, 150.0), (-70.0, -45.0)
const R = 6.371e6

f64(A) = Float64.(coalesce.(A, NaN))
coord(ds, name, dim) = (v = f64(ds[name][:]); ndims(v) == 1 ? v : dim == 1 ? v[:, 1] : v[1, :])
first_var(ds, names) = first(filter(n -> haskey(ds, n), names))

rows = []   # (date, extent, area, volume)
for y in YEARS
    fa, fh = joinpath(BDIR, "SIArea_bsoseI156_$(y)_5day.nc"), joinpath(BDIR, "SIHeff_bsoseI156_$(y)_5day.nc")
    isfile(fa) || (@warn "no $fa; skipping $y"; continue)
    da, dh = NCDataset(fa), NCDataset(fh)
    x, yc = coord(da, "XC", 1), coord(da, "YC", 2)
    i = findall(λ -> LON[1] < λ < LON[2], x)
    j = findall(φ -> LAT[1] < φ < LAT[2], yc)
    edges = vcat(1.5yc[1] - 0.5yc[2], (yc[1:end-1] .+ yc[2:end]) ./ 2, 1.5yc[end] - 0.5yc[end-1])
    dλ = x[2] - x[1]
    cell = [R^2 * deg2rad(dλ) * abs(sind(edges[jj+1]) - sind(edges[jj])) for _ in i, jj in j]
    ℵname, hname = first_var(da, ("SIarea", "SIArea", "SIAREA")), first_var(dh, ("SIheff", "SIHeff", "SIHEFF"))
    times = DateTime.(da["time"][:]) .- Hour(60)            # 5-day stamp = window end; the window centre
    for n in eachindex(times)
        ℵ = replace(f64(da[ℵname][i, j, n]), NaN => 0.0)
        V = replace(f64(dh[hname][i, j, n]), NaN => 0.0)
        push!(rows, (times[n], sum(cell[ℵ .> 0.15]) / 1e12, sum(ℵ .* cell) / 1e12, sum(V .* cell) / 1e9))
    end
    close(da); close(dh)
end
isempty(rows) && error("no B-SOSE 5-day ice files found in $BDIR")

println("\nB-SOSE SEA ICE over 90-150E, 70-45S (5-day means, dated at the window centre)")
@printf("  %-6s | %-24s | %-24s | %-22s\n", "year", "max extent (10⁶ km²)", "min extent (10⁶ km²)", "max volume (km³)")
for y in YEARS
    r = filter(t -> year(t[1]) == y, rows)
    isempty(r) && continue
    kx, kn, kv = argmax(getindex.(r, 2)), argmin(getindex.(r, 2)), argmax(getindex.(r, 4))
    @printf("  %-6d | %5.2f on %s       | %5.2f on %s       | %6.0f on %s\n", y,
            r[kx][2], Dates.format(r[kx][1], "yyyy-mm-dd"), r[kn][2], Dates.format(r[kn][1], "yyyy-mm-dd"),
            r[kv][4], Dates.format(r[kv][1], "yyyy-mm-dd"))
end
println("\n  MONTHLY MEANS")
@printf("  %-8s | %8s %8s %8s\n", "month", "extent", "area", "volume")
for m in unique(Date.(year.(first.(rows)), month.(first.(rows))))
    r = filter(t -> Date(year(t[1]), month(t[1])) == m, rows)
    @printf("  %-8s | %8.2f %8.2f %8.0f\n", Dates.format(m, "yyyy-mm"), sum(getindex.(r, 2)) / length(r),
            sum(getindex.(r, 3)) / length(r), sum(getindex.(r, 4)) / length(r))
end
