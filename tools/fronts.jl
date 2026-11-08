# ==============================================================================
# fronts.jl -- the ACC fronts from the CTOH/LEGOS weekly altimetry product, put on a model grid.
#
#     include(joinpath(@__DIR__, "fronts.jl"))
#     F = load_fronts(dir; from = DateTime(2014), to = DateTime(2017))
#     pf = front_on_grid(F.PF, lon, DateTime(2015, 9), DateTime(2015, 10))   # latitude per lon
#     z  = front_zone(φ, pf[i], saf[i], nsaf[i])                             # 1..4, 0 unknown
#
# THE PRODUCT (doi:10.24400/527896/a01-2023.004, v2.0). Weekly front positions from 1993 to 2025,
# traced as ADT contours of the Copernicus DT global product with the Sallée et al. (2008) method:
#     PF    Polar Front                      ADT = -0.3 m   CTOH_PolarFront_weekly_1993_2025.nc
#     SAF   Subantarctic Front               ADT = -0.1 m   CTOH_SubantarcticFront_weekly_1993_2025.nc
#     NSAF  northern branch of the SAF       ADT =  0.0 m   CTOH_NorthernSubantarcticFront_weekly_1993_2025.nc
# These are the CTOH levels. They are NOT Park et al.'s (2019) -0.58 m PF; the CTOH PF sits north of
# it. Each week is ONE contour stored as a list of (longitude, latitude) points running 0 -> 360E,
# NaN-padded at the end; where the front meanders the contour folds back and a longitude carries
# several latitudes.
#
# ON A GRID. For each grid longitude the front latitude is the median of the contour points falling
# in that grid cell's longitude bin, then the median over the weeks inside the requested window.
# Bins no point lands in (the contour is ~2.5 points per 1/6 deg) are filled by linear interpolation
# in longitude, periodically when the grid goes all the way round. A fold in a meander collapses to
# its median latitude, so a column inside a detached loop is zoned by the front's mean path, not
# the loop -- the right trade for zone statistics, not for an eddy-by-eddy look.
#
# ENV
#     FRONTS_DIR   where the three CTOH files are   (default: BSOSE_DIR, then data)
# ==============================================================================

using NCDatasets, Dates, Statistics

const FRONTS_DIR  = get(ENV, "FRONTS_DIR", get(ENV, "BSOSE_DIR", "data"))
const FRONT_FILES = (PF   = ("CTOH_PolarFront_weekly_1993_2025.nc", "PF", -0.3),
                     SAF  = ("CTOH_SubantarcticFront_weekly_1993_2025.nc", "SAF", -0.1),
                     NSAF = ("CTOH_NorthernSubantarcticFront_weekly_1993_2025.nc", "SAF_N", 0.0))
const ZONE_NAMES  = ("S of PF", "PF-SAF", "SAF-NSAF", "N of NSAF")

"""One front's weekly contours between `from` and `to` (a week either side is kept)."""
function load_front(dir, name; from, to)
    file, var, adt = FRONT_FILES[name]
    path = joinpath(dir, file)
    isfile(path) || error("$path not found (set FRONTS_DIR)")
    ds = Dataset(path)
    t = [ismissing(x) ? DateTime(0) : DateTime(x) for x in ds["time"][:]]
    K = findall(x -> from - Day(7) <= x < to + Day(7), t)
    isempty(K) && (close(ds); error("$path has no week between $from and $to"))
    K = minimum(K):maximum(K)
    lon = Float64.(coalesce.(ds["longitude"][:, K], NaN))
    lat = Float64.(coalesce.(ds[var][:, K], NaN))
    close(ds)
    return (; name, adt, weeks = t[K], lon, lat)
end

"""All three fronts, as a NamedTuple (PF, SAF, NSAF)."""
load_fronts(dir = FRONTS_DIR; from, to) =
    NamedTuple{keys(FRONT_FILES)}(Tuple(load_front(dir, n; from, to) for n in keys(FRONT_FILES)))

# Linear interpolation over the NaN runs of `v`, round the end if `periodic`
function fill_gaps!(v, periodic)
    n = length(v)
    good = findall(isfinite, v)
    isempty(good) && return v
    for i in 1:n
        isfinite(v[i]) && continue
        r = findfirst(>(i), good)
        l = findlast(<(i), good)
        if l !== nothing && r !== nothing
            a, b = good[l], good[r]
        elseif periodic
            a = l === nothing ? good[end] - n : good[l]          # wrap: indices may leave 1..n
            b = r === nothing ? good[1] + n : good[r]
        else
            v[i] = v[l === nothing ? good[r] : good[l]]           # hold the end value
            continue
        end
        va, vb = v[mod1(a, n)], v[mod1(b, n)]
        v[i] = va + (vb - va) * (i - a) / (b - a)
    end
    return v
end

"""
    front_on_grid(f, glon, a, b) -> Vector

Latitude of front `f` (from `load_front`) at each of the uniformly spaced grid longitudes `glon`,
the median over the weeks in [a, b). NaN everywhere when no week falls in the window.
"""
function front_on_grid(f, glon, a, b)
    nb = length(glon)
    d = (glon[end] - glon[1]) / (nb - 1)
    periodic = nb * d >= 359.9
    ks = findall(t -> a <= t < b, f.weeks)
    weekly = fill(NaN, nb, length(ks))
    buf = [Float64[] for _ in 1:nb]
    for (c, k) in enumerate(ks)
        foreach(empty!, buf)
        for p in axes(f.lon, 1)
            λ, φ = f.lon[p, k], f.lat[p, k]
            (isfinite(λ) && isfinite(φ)) || continue
            i = round(Int, mod(λ - glon[1], 360) / d)
            periodic && (i = mod(i, nb))
            0 <= i < nb && push!(buf[i+1], φ)
        end
        for i in 1:nb
            isempty(buf[i]) || (weekly[i, c] = median(buf[i]))
        end
    end
    out = [(v = filter(isfinite, view(weekly, i, :)); isempty(v) ? NaN : median(v)) for i in 1:nb]
    return fill_gaps!(out, periodic)
end

"""The three fronts on `glon` for the calendar month containing `t`."""
function fronts_for_month(F, glon, t)
    a = DateTime(firstdayofmonth(t))
    b = a + Month(1)
    return map(f -> front_on_grid(f, glon, a, b), F)
end

"""Zone of a column at latitude `φ`: 1 south of the PF, 2 PF-SAF, 3 SAF-NSAF, 4 north of the NSAF."""
function front_zone(φ, pf, saf, nsaf)
    (isfinite(pf) && isfinite(saf) && isfinite(nsaf)) || return 0
    return φ < pf ? 1 : φ < saf ? 2 : φ < nsaf ? 3 : 4
end
