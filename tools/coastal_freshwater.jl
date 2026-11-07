# ==============================================================================
# coastal_freshwater.jl -- does B-SOSE's coast get freshwater that our model never sees?
#
#     julia --project=. tools/coastal_freshwater.jl
#     MODEL_RUN=2Y_nested_seaice_fix4 julia --project=. tools/coastal_freshwater.jl
#
# THE HYPOTHESIS. Every nested run (fix3, fix4) has a salty strip a few cells wide along the whole
# coast, worst in summer and growing from year to year, and changing the sea ice (fix4) barely
# touched it. Our only surface freshwater is ERA5 precipitation: no runoff, no ice-shelf basal
# melt, no iceberg melt. B-SOSE runs a real freshwater flux (`oceFWflx`) and is fitted to
# observations, so whatever the real coast receives ends up in it one way or another. This sector
# holds Shackleton, Denman and Totten, among the fastest-melting ice shelves of East Antarctica.
#
# THE TEST. Split B-SOSE's monthly `oceFWflx` (kg m⁻² s⁻¹, + into the ocean) into what our model
# also has and the rest:
#
#     residual = oceFWflx - ice - P
#
#     ice  the sea-ice melt (+) / freeze (-) water flux. Taken from B-SOSE's SFLUX, which is the
#          salt carried IN the ice: ice of salinity S_ICE melting releases S_ICE grams of salt per
#          kilogram, so ice = SFLUX / S_ICE. S_ICE = 4.1 g/kg is what the seasonal amplitude ratio
#          of SFLUX to oceFWflx gave (memory: SFLUX is the salt carried in the ice, not brine).
#     P    ERA5 total precipitation, the same files the nest is forced with (ERA5_DIR). If they
#          are not there, P is left out and the residual is (runoff + P - E) instead.
#
# So residual = runoff + (B-SOSE's adjusted P - ERA5 P) - E. Evaporation is not on disk; near the
# Antarctic coast it is small (a few tenths of a metre a year, less under ice), so a residual that
# is large and POSITIVE in the first tens of kilometres and fades offshore is freshwater our model
# is missing. Everything is binned by distance from the coast (the first ocean cell north of
# Antarctica in each column), because the bias is a strip, not a band of latitude.
#
# WITH MODEL_RUN, also the test itself: the model-minus-B-SOSE surface S in the same distance
# bins, month by month, against the ΔS the residual would cause if it were simply missing, mixed
# over MLD_REF metres: S_REF * cumulative(residual) / (ρ₀ MLD_REF), positive = we end up saltier. If the coastal bias grows at
# roughly that rate and the offshore bins do not, the missing freshwater accounts for it.
#
# THE FIELD FOR THE MODEL. It also writes FIELD_FILE (coastal_freshwater_bsose.nc), the freshwater
# examples/sowinwarp_nested.jl adds when COASTAL_FRESHWATER=1: the time-mean residual minus its
# offshore value (REF_KM, 300-600 km), within DIST_MAX km of the coast, never negative, on the
# nest's own grid. Subtracting the offshore value removes what the residual holds everywhere,
# B-SOSE's precipitation adjustment and its evaporation (our model computes its own), and keeps the
# coastal excess: the melt and runoff. Without ERA5 P the offshore value also absorbs P - E, which
# is lower at the coast, so the field is then a slight underestimate. Regridded like the B-SOSE
# ice in the nested script: B-SOSE land takes its nearest ocean value, then bilinear.
#
# LOGIN-NODE SAFE: NCDatasets and CairoMakie only, subset reads.
#
# Reads, from BSOSE_DIR: oceFWflx_, surfSflx_ and (with MODEL_RUN) Salt_bsoseI156_2013to2024_monthly.nc;
# from ERA5_DIR: total_precipitation_<year>*.nc; from DIR: <MODEL_RUN>_surface.nc.
#
# ENV
#     BSOSE_DIR    B-SOSE monthly files               (default: the usual search path)
#     ERA5_DIR     ERA5 yearly files                  (default: BSOSE_DIR/era5)
#     START_DATE   first month                        (default 2014-06-01)
#     END_DATE     end, exclusive                     (default 2016-06-01)
#     S_ICE        sea-ice salinity, g/kg             (default 4.1)
#     S_REF        surface salinity for the ΔS        (default 34.0)
#     MLD_REF      mixing depth for the ΔS, m         (default 100)
#     BANDS_KM     distance bins, "0,30,75,150,300,600"
#     MODEL_RUN    a nested run to test against       (default: none)
#     DIR          where <MODEL_RUN>_surface.nc is    (default ".")
#     OUTDIR       figures                            (default plots)
#     FIELD_FILE   the field for the model            (default coastal_freshwater_bsose.nc; "" to skip)
#     REF_KM       offshore reference band, km        (default "300,600")
#     DIST_MAX     outer edge of the field, km        (default 300)
# ==============================================================================

using NCDatasets, CairoMakie, Dates, Printf, Statistics

const LON = (90.0, 150.0)
const LAT = (-70.0, -45.0)
const ρ₀  = 1026.0
const YEAR_SECONDS = 365.25 * 86400

const START_DATE = DateTime(get(ENV, "START_DATE", "2014-06-01"))
const END_DATE   = DateTime(get(ENV, "END_DATE", "2016-06-01"))
const S_ICE      = parse(Float64, get(ENV, "S_ICE", "4.1"))
const S_REF      = parse(Float64, get(ENV, "S_REF", "34.0"))
const MLD_REF    = parse(Float64, get(ENV, "MLD_REF", "100"))
const EDGES      = parse.(Float64, split(get(ENV, "BANDS_KM", "0,30,75,150,300,600"), ","))
const BANDS      = [(EDGES[k], EDGES[k+1]) for k in 1:length(EDGES)-1]
const MODEL_RUN  = get(ENV, "MODEL_RUN", "")
const MODEL_DIR  = get(ENV, "DIR", ".")
const OUTDIR     = get(ENV, "OUTDIR", "plots")
const FIELD_FILE = get(ENV, "FIELD_FILE", "coastal_freshwater_bsose.nc")
const REF_KM     = Tuple(parse.(Float64, split(get(ENV, "REF_KM", "300,600"), ",")))
const DIST_MAX   = parse(Float64, get(ENV, "DIST_MAX", "300"))

# The nest's grid (examples/sowinwarp_nested.jl): the field is written on it, and the nested
# script checks the coordinates match before using it.
const CHILD_LON, CHILD_LAT, CHILD_SCALING = (90.0, 150.0), (-70.0, -45.0), 6

# Same search path as `default_bsose_directory` in src/setup_bsose.jl, repeated so this file
# loads without the project's heavy dependencies.
function bsose_directory()
    haskey(ENV, "BSOSE_DIR") && isdir(ENV["BSOSE_DIR"]) && return abspath(ENV["BSOSE_DIR"])
    for d in ("/storage/scratch1/1/vnguyen480/SOWinWaRP/data",
              normpath(joinpath(@__DIR__, "..", "data")), "data")
        isdir(d) && return abspath(d)
    end
    error("No BSOSE data directory found. Set BSOSE_DIR.")
end
const BDIR = bsose_directory()
const ERA5_DIR = get(ENV, "ERA5_DIR", joinpath(BDIR, "era5"))

bsose_path(token) = joinpath(BDIR, "$(token)_bsoseI156_2013to2024_monthly.nc")
function require(p)
    isfile(p) || error("Missing $p")
    filesize(p) > 0 || error("$p is zero bytes: the download has not finished")
    return p
end

# A B-SOSE monthly record is stamped at the END of the month it averages (src/setup_bsose.jl,
# `bsose_averaging_window`); this is the month it stands for.
averaged_month(stamp) = round(DateTime(stamp), Month) - Month(1)
month_seconds(m) = Dates.value(Second((m + Month(1)) - m))

g(A) = Float64.(coalesce.(A, NaN))

# ── B-SOSE geometry and distance from the coast ──────────────────────────────────────────────
fw_ds = NCDataset(require(bsose_path("oceFWflx")))
sf_ds = NCDataset(require(bsose_path("surfSflx")))
xc, yc = g(fw_ds["XC"][:]), g(fw_ds["YC"][:])
iB = findall(x -> LON[1] <= x <= LON[2], xc)
jB = findall(y -> LAT[1] <= y <= LAT[2], yc)
iB, jB = first(iB):last(iB), first(jB):last(jB)
λB, φB = xc[iB], yc[jB]
areaB = g(fw_ds["rA"][iB, jB])
wetB  = g(fw_ds["Depth"][iB, jB]) .> 0

"""
Distance (km) of each ocean cell north of the first ocean cell of its column, i.e. from the
Antarctic coast along the meridian. NaN on land, and for a whole column whose first ocean cell is
the domain's southern edge (the coast is then outside the box and the distance would be wrong).
"""
function coast_distance(φ, wet)
    d = fill(NaN, size(wet))
    for i in axes(wet, 1)
        j₀ = findfirst(view(wet, i, :))
        (j₀ === nothing || j₀ == 1) && continue
        for j in j₀:size(wet, 2)
            wet[i, j] && (d[i, j] = (φ[j] - φ[j₀]) * 111.2)
        end
    end
    return d
end
distB = coast_distance(φB, wetB)
in_band(d, (lo, hi)) = lo <= d < hi

function band_mean(A, dist, area, band)
    num = den = 0.0
    for I in eachindex(A)
        (in_band(dist[I], band) && isfinite(A[I])) || continue
        num += A[I] * area[I]; den += area[I]
    end
    return den > 0 ? num / den : NaN
end

months = [m for m in START_DATE:Month(1):END_DATE-Month(1)]
fw_rec = Dict(averaged_month(s) => n for (n, s) in enumerate(fw_ds["time"][:]))
sf_rec = Dict(averaged_month(s) => n for (n, s) in enumerate(sf_ds["time"][:]))
missing_months = [m for m in months if !(haskey(fw_rec, m) && haskey(sf_rec, m))]
isempty(missing_months) || error("B-SOSE has no oceFWflx/SFLUX record for $(missing_months)")

# ── ERA5 precipitation, onto the B-SOSE cells (nearest ERA5 point) ───────────────────────────
function era5_file(y)
    isdir(ERA5_DIR) || return nothing
    f = filter(n -> startswith(n, "total_precipitation_$(y)") && endswith(n, ".nc"), readdir(ERA5_DIR))
    return isempty(f) ? nothing : joinpath(ERA5_DIR, first(f))
end
const HAVE_P = all(y -> era5_file(y) !== nothing, year(START_DATE):year(END_DATE - Month(1)))
HAVE_P || @warn "ERA5 total_precipitation not found in $ERA5_DIR: P is left out, so the residual is runoff + P - E"

const ERA5_CACHE = Dict{Int, Any}()
function era5_year(y)
    get!(ERA5_CACHE, y) do
        ds = NCDataset(era5_file(y))
        name = first(filter(n -> haskey(ds, n), ("tp", "total_precipitation")))
        lon, lat = g(ds["longitude"][:]), g(ds["latitude"][:])
        iE = [argmin(abs.(lon .- x)) for x in λB]
        jE = [argmin(abs.(lat .- y)) for y in φB]
        (; ds, name, iE, jE, times = DateTime.(ds["time"][:]))
    end
end

"Mean ERA5 precipitation over month m on the B-SOSE cells, kg m⁻² s⁻¹ (accumulations ending in (m, m+1])."
function era5_precip(m)
    total = zeros(length(λB), length(φB))
    e = era5_year(year(m))
    idx = findall(t -> m < t <= m + Month(1), e.times)
    # the hour ending exactly at m + 1 month on New Year's Day sits in the next year's file
    if month(m) == 12 && era5_file(year(m) + 1) !== nothing
        e2 = era5_year(year(m) + 1)
        k = findall(t -> t == m + Month(1), e2.times)
        isempty(k) || (total .+= g(e2.ds[e2.name][:, :, only(k)])[e2.iE, e2.jE])
    end
    isempty(idx) && error("no ERA5 precipitation in $(Dates.format(m, "yyyy-mm"))")
    A = g(e.ds[e.name][:, :, first(idx):last(idx)])       # metres of water per hour, summed below
    total .+= dropdims(sum(A; dims = 3); dims = 3)[e.iE, e.jE]
    return total .* 1000 ./ month_seconds(m)
end

# ── the terms, month by month ────────────────────────────────────────────────────────────────
nb, nm = length(BANDS), length(months)
FWb, ICEb, Pb, RESb = (fill(NaN, nb, nm) for _ in 1:4)
RES_sum = zeros(length(λB), length(φB)); RES_time = zeros(length(λB), length(φB))
println("B-SOSE: $(length(iB)) x $(length(jB)) cells over $(LON[1])-$(LON[2])E, $(-LAT[2])-$(-LAT[1])S; " *
        "S_ICE = $S_ICE g/kg; $(nm) months from $(Dates.format(first(months), "yyyy-mm"))")
for (n, m) in enumerate(months)
    FW  = g(fw_ds["oceFWflx"][iB, jB, fw_rec[m]])
    ICE = g(sf_ds["SFLUX"][iB, jB, sf_rec[m]]) ./ S_ICE      # g m⁻² s⁻¹ / (g kg⁻¹) = kg m⁻² s⁻¹
    P   = HAVE_P ? era5_precip(m) : zeros(size(FW))
    RES = FW .- ICE .- P
    for A in (FW, ICE, P, RES)
        A[.!wetB] .= NaN
    end
    for (b, band) in enumerate(BANDS)
        FWb[b, n], ICEb[b, n] = band_mean(FW, distB, areaB, band), band_mean(ICE, distB, areaB, band)
        Pb[b, n], RESb[b, n]  = band_mean(P, distB, areaB, band), band_mean(RES, distB, areaB, band)
    end
    ok = isfinite.(RES)
    RES_sum[ok] .+= RES[ok] .* month_seconds(m); RES_time[ok] .+= month_seconds(m)
end
close(fw_ds); close(sf_ds)

# kg m⁻² s⁻¹ -> metres of freshwater per year, and the ΔS a year of it makes over MLD_REF
to_m_per_year(x) = x * YEAR_SECONDS / 1000
dS_per_year(x)   = S_REF * x * YEAR_SECONDS / (ρ₀ * MLD_REF)   # + : we end up saltier

years = [(START_DATE + Year(k), min(START_DATE + Year(k + 1), END_DATE)) for k in 0:(year(END_DATE) - year(START_DATE) - 1)]
println()
println("="^104)
println("B-SOSE SURFACE FRESHWATER BY DISTANCE FROM THE COAST, metres of freshwater per year (+ into the ocean)")
println("  oceFWflx = everything B-SOSE puts in; ice = SFLUX/S_ICE; P = ERA5 precipitation (our only freshwater)")
println("  residual = oceFWflx - ice - P" * (HAVE_P ? " = runoff + ΔP(B-SOSE - ERA5) - E" : " = runoff + P - E  (no ERA5 here)"))
println("  ΔS/yr = how much saltier a year without the residual leaves $(Int(MLD_REF)) m of water, if nothing mixes it away")
println("="^104)
for (y₀, y₁) in years
    sel = [y₀ <= m < y₁ for m in months]
    w = [month_seconds(m) for m in months] .* sel
    avg(row) = sum(row .* w) / sum(w)
    println("  $(Dates.format(y₀, "yyyy-mm")) to $(Dates.format(y₁ - Month(1), "yyyy-mm"))")
    @printf("    %-12s %10s %10s %10s %10s %12s\n", "band (km)", "oceFWflx", "ice", "P", "residual", "ΔS/yr")
    for (b, (lo, hi)) in enumerate(BANDS)
        r = avg(RESb[b, :])
        @printf("    %4.0f-%-7.0f %10.2f %10.2f %10.2f %10.2f %12.3f\n", lo, hi,
                to_m_per_year(avg(FWb[b, :])), to_m_per_year(avg(ICEb[b, :])),
                to_m_per_year(avg(Pb[b, :])), to_m_per_year(r), dS_per_year(r))
    end
end
println()
println("  Sensitivity to the ice salinity (whole period, m/yr). A wrong S_ICE leaves part of the ice's large")
println("  seasonal cycle in the residual, so also check the residual's own seasonal range in the figure.")
let w = [month_seconds(m) for m in months]
    avg(row) = sum(row .* w) / sum(w)
    @printf("    %-12s", "band (km)"); foreach(s -> @printf(" %12s", @sprintf("S_ICE=%.1f", s)), (3.0, S_ICE, 6.0)); println()
    for (b, (lo, hi)) in enumerate(BANDS)
        @printf("    %4.0f-%-7.0f", lo, hi)
        for s_ice in (3.0, S_ICE, 6.0)
            @printf(" %12.2f", to_m_per_year(avg(FWb[b, :]) - avg(ICEb[b, :]) * S_ICE / s_ice - avg(Pb[b, :])))
        end
        println()
    end
end

# ── the field for the model ──────────────────────────────────────────────────────────────────
if !isempty(FIELD_FILE)
    R = RES_sum ./ RES_time                                     # kg m⁻² s⁻¹, time mean, NaN on land
    R_ref = band_mean(R, distB, areaB, REF_KM)
    excess = [wetB[I] && isfinite(distB[I]) && distB[I] < DIST_MAX && isfinite(R[I]) ? max(R[I] - R_ref, 0.0) : 0.0
              for I in CartesianIndices(R)]

    # B-SOSE land takes its nearest B-SOSE ocean value (within 6 cells), then bilinear onto the nest
    reach = 6
    source = zeros(Int, size(wetB)); lin = LinearIndices(wetB)
    for I in CartesianIndices(wetB)
        best, dist2 = 0, Inf
        for di in -reach:reach, dj in -reach:reach
            J = I + CartesianIndex(di, dj)
            (checkbounds(Bool, wetB, J) && wetB[J] && di^2 + dj^2 < dist2) || continue
            best, dist2 = lin[J], di^2 + dj^2
        end
        source[I] = best
    end
    Nx = round(Int, CHILD_SCALING * (CHILD_LON[2] - CHILD_LON[1]))
    Ny = round(Int, CHILD_SCALING * (CHILD_LAT[2] - CHILD_LAT[1]))
    λc = CHILD_LON[1] .+ ((1:Nx) .- 0.5) ./ CHILD_SCALING
    φc = CHILD_LAT[1] .+ ((1:Ny) .- 0.5) ./ CHILD_SCALING
    bracket(v, x) = (a = clamp(searchsortedlast(v, x), 1, length(v) - 1); (a, clamp((x - v[a]) / (v[a+1] - v[a]), 0, 1)))
    function onto_child(i, j)
        a, wx = bracket(λB, λc[i]); b, wy = bracket(φB, φc[j])
        num = den = 0.0
        for (di, w₁) in ((0, 1 - wx), (1, wx)), (dj, w₂) in ((0, 1 - wy), (1, wy))
            src = source[a+di, b+dj]
            src > 0 || continue
            num += w₁ * w₂ * excess[src]; den += w₁ * w₂
        end
        return den > 0 ? num / den : 0.0
    end
    F = [onto_child(i, j) for i in 1:Nx, j in 1:Ny]

    total_gt = sum(excess .* areaB) * YEAR_SECONDS / 1e12
    NCDataset(FIELD_FILE, "c") do out
        defDim(out, "lon", Nx); defDim(out, "lat", Ny)
        defVar(out, "lon", λc, ("lon",); attrib = ["units" => "degrees_east"])
        defVar(out, "lat", φc, ("lat",); attrib = ["units" => "degrees_north"])
        defVar(out, "freshwater_flux", F, ("lon", "lat");
               attrib = ["units" => "kg m-2 s-1", "positive" => "into the ocean",
                         "long_name" => "B-SOSE coastal freshwater excess our model lacks"])
        out.attrib["method"] = "time mean of oceFWflx - SFLUX/S_ICE - ERA5 P, minus its mean over $(REF_KM) km from the coast, " *
                               "within $(DIST_MAX) km, floored at 0"
        out.attrib["S_ICE"] = S_ICE
        out.attrib["era5_precipitation_subtracted"] = string(HAVE_P)
        out.attrib["period"] = "$(Dates.format(START_DATE, "yyyy-mm")) to $(Dates.format(END_DATE - Month(1), "yyyy-mm"))"
        out.attrib["offshore_reference_m_per_year"] = to_m_per_year(R_ref)
        out.attrib["total_Gt_per_year"] = total_gt
    end
    println()
    @printf("Field for the model: %s. Offshore reference %.2f m/yr removed; within %.0f km of the coast;\n",
            FIELD_FILE, to_m_per_year(R_ref), DIST_MAX)
    @printf("  %.0f Gt/yr over 90-150E (on B-SOSE's cells), up to %.1f m/yr in a cell%s\n", total_gt,
            to_m_per_year(maximum(F)), HAVE_P ? "" : "; NO ERA5 P subtracted, rerun where ERA5 is")
end

# ── the model's own coastal bias, in the same bins ───────────────────────────────────────────
model = nothing
if !isempty(MODEL_RUN)
    mpath = joinpath(MODEL_DIR, "$(MODEL_RUN)_surface.nc")
    isfile(mpath) || error("$mpath not found; set DIR")
    ds = NCDataset(mpath)
    λM, φM = g(ds["λ_caa"][:]), g(ds["φ_aca"][:])
    tM = g(ds["time"][:])
    start = haskey(ds.attrib, "start_date") ? DateTime(ds.attrib["start_date"]) : START_DATE
    SM = g(ds["S"][:, :, 1, :])
    wetM = haskey(ds, "inactive_nodes_ccc") ? (ds["inactive_nodes_ccc"][:, :, 1] .== 0) :
                                              .!isnan.(SM[:, :, end])
    close(ds)
    distM = coast_distance(φM, wetM)
    areaM = [cosd(φ) for _ in λM, φ in φM]
    datesM = [start + Millisecond(round(Int, 1000t)) for t in tM]

    salt = NCDataset(require(bsose_path("Salt")))
    salt_rec = Dict(averaged_month(s) => n for (n, s) in enumerate(salt["time"][:]))
    ΔSb = fill(NaN, nb, nm)
    for (n, m) in enumerate(months)
        days = [k for k in 2:length(datesM) if m <= datesM[k] < m + Month(1)]   # k = 1 is the initial state
        (length(days) >= 20 && haskey(salt_rec, m)) || continue
        Smodel = dropdims(mean(SM[:, :, days]; dims = 3); dims = 3)
        Sb = g(salt["SALT"][iB, jB, 1, salt_rec[m]]); Sb[.!wetB] .= NaN
        for (b, band) in enumerate(BANDS)
            ΔSb[b, n] = band_mean(Smodel, distM, areaM, band) - band_mean(Sb, distB, areaB, band)
        end
    end
    close(salt)
    global model = (; ΔSb)

    println()
    println("MODEL ($MODEL_RUN) minus B-SOSE surface S by distance from the coast, g/kg")
    @printf("    %-12s", "band (km)"); foreach(m -> @printf(" %8s", Dates.format(m, "yy-mm")), months[2:3:end]); println()
    for (b, (lo, hi)) in enumerate(BANDS)
        @printf("    %4.0f-%-7.0f", lo, hi); foreach(n -> @printf(" %+8.3f", ΔSb[b, n]), 2:3:nm); println()
    end
    println("  Compare the growth of each row with the ΔS/yr column above. Their RATIO is the share of the")
    println("  missing freshwater that stays near the surface; a coastal row growing while offshore rows do not")
    println("  is the signature of a coastal source we lack.")
end

# ── figure ───────────────────────────────────────────────────────────────────────────────────
mkpath(OUTDIR)
tag = isempty(MODEL_RUN) ? "bsose" : MODEL_RUN
mid = [m + Day(14) for m in months]
xs = [Dates.value(Day(d - first(mid))) for d in mid]
fig = Figure(size = (1500, isempty(MODEL_RUN) ? 900 : 1300), fontsize = 14)
Label(fig[0, 1:2], "Coastal freshwater: B-SOSE minus what our model gets (S_ICE = $S_ICE g/kg" *
      (HAVE_P ? ", ERA5 P)" : ", no ERA5 P: residual = runoff + P - E)"), fontsize = 18, font = :bold)

ax1 = Axis(fig[1, 1]; title = "Nearest band, $(Int(BANDS[1][1]))-$(Int(BANDS[1][2])) km: terms",
           ylabel = "m freshwater / yr", xlabel = "days since $(Dates.format(first(months), "yyyy-mm-dd"))")
for (data, label, color) in ((FWb, "oceFWflx", :black), (ICEb, "ice = SFLUX/S_ICE", :royalblue),
                             (Pb, "ERA5 P", :seagreen), (RESb, "residual", :firebrick))
    lines!(ax1, xs, to_m_per_year.(data[1, :]); label, color, linewidth = data === RESb ? 3 : 1.5)
end
hlines!(ax1, [0]; color = :grey60)
axislegend(ax1; position = :lb, framevisible = false)

ax2 = Axis(fig[1, 2]; title = "Residual by distance from the coast, mean over the period",
           xlabel = "distance from the coast (km)", ylabel = "m freshwater / yr")
w = [month_seconds(m) for m in months]
res_mean = [to_m_per_year(sum(RESb[b, :] .* w) / sum(w)) for b in 1:nb]
centres = [(lo + hi) / 2 for (lo, hi) in BANDS]
barplot!(ax2, centres, res_mean; width = [hi - lo for (lo, hi) in BANDS] .* 0.9, color = :firebrick)
hlines!(ax2, [0]; color = :grey60)

ax3 = Axis(fig[2, 1:2]; title = "Residual, mean over the period (m freshwater / yr); dashed: $(Int(BANDS[1][2])) and $(Int(BANDS[2][2])) km from the coast",
           xlabel = "Longitude (°E)", ylabel = "Latitude (°N)", aspect = DataAspect())
res_map = to_m_per_year.(RES_sum ./ RES_time); res_map[.!wetB] .= NaN
lim = max(quantile(filter(isfinite, vec(abs.(res_map))), 0.98), 0.1)
hm = heatmap!(ax3, λB, φB, res_map; colormap = :balance, colorrange = (-lim, lim), nan_color = :grey70)
Colorbar(fig[2, 3], hm; label = "m / yr")
for (k, c) in ((BANDS[1][2], :black), (BANDS[2][2], :black))
    contour!(ax3, λB, φB, replace(distB, NaN => 1e4); levels = [k], color = c, linestyle = :dash)
end
ylims!(ax3, LAT[1], -55)

if model !== nothing
    ax4 = Axis(fig[3, 1:2]; title = "Model - B-SOSE surface S by distance from the coast (solid), and the ΔS the residual would cause over $(Int(MLD_REF)) m (dashed)",
               xlabel = "days since $(Dates.format(first(months), "yyyy-mm-dd"))", ylabel = "ΔS (g/kg)")
    colors = cgrad(:viridis, nb; categorical = true)
    for b in 1:nb
        lines!(ax4, xs, model.ΔSb[b, :]; color = colors[b], linewidth = 2,
               label = @sprintf("%.0f-%.0f km", BANDS[b]...))
        implied = S_REF .* cumsum(RESb[b, :] .* w) ./ (ρ₀ * MLD_REF)
        lines!(ax4, xs, implied; color = colors[b], linestyle = :dash)
    end
    hlines!(ax4, [0]; color = :grey60)
    axislegend(ax4; position = :lt, framevisible = false, nbanks = 2)
end

out = joinpath(OUTDIR, "coastal_freshwater_$(tag).png")
save(out, fig)
@info "wrote $out"
