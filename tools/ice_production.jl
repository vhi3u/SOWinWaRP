# ==============================================================================
# WAS IT A LOW-ICE YEAR?  Sea-ice area, volume and GROWTH from the B-SOSE 5-day files.
#
# `mixing_energetics.jl` showed the 70S-65S brine term collapse in 2016: the Apr-Oct salt
# contribution to B0 summed to -5.10 against -10.59 and -10.67 in 2014 and 2015, `supplied`
# fell to 0.1604 from 0.3443, and B-SOSE's OWN winter mixed layer fell with it, 239 m to
# 204 m. Forcing and response moved together, which is the signature of a genuinely weak ice
# year rather than anything wrong on our side -- but that was inferred from the salt flux.
# This checks it against the ice itself.
#
# WHY GROWTH, NOT AREA. Brine is rejected when ice FORMS, not while it sits there. A year can
# hold its area and still reject little brine if it froze early and then coasted. The quantity
# that sets the salt flux is d(volume)/dt, so `SIHeff` -- MITgcm's effective thickness, which
# is already ice volume per unit area of grid cell -- is the field to difference.
#
# AND WHY THE TWO HALVES ARE SUMMED SEPARATELY. Freezing rejects brine; melting returns fresh
# water. Those are the two sides of the drift we are chasing, and the NET hides them: a year
# that melts as much as ever while freezing less is a net freshwater gain even if its mean ice
# area looks ordinary. So the increments are split by sign.
#
# PER CELL, THEN AVERAGED. The split is done on each column before the band average, never
# after. A cell freezing beside one that is melting cancels in a band-mean thickness while both
# are really happening, and it is exactly that local brine that drives the convection.
#
# LOGIN-NODE SAFE: NCDatasets only, one regional subset per file, no Oceananigans.
#
# ENV
#     BSOSE_DIR   directory holding the B-SOSE files       (default: data)
#     YEARS       comma-separated years to compare         (default: 2014,2015,2016)
#     OUTDIR      where the figure goes                    (default: plots)
#
# Reads `SIArea_bsoseI156_<year>_5day.nc` and `SIHeff_bsoseI156_<year>_5day.nc`, and
# `surfSflx_..._5day.nc` as a cross-check when it is there. Geometry comes from the monthly
# surfTflx file, the same one every other tool in here probes, so the subset indices match.
# ==============================================================================

using NCDatasets, Dates, Printf, Statistics, CairoMakie

const DIR    = get(ENV, "BSOSE_DIR", "data")
const YEARS  = parse.(Int, strip.(split(get(ENV, "YEARS", "2014,2015,2016"), ",")))
const OUTDIR = get(ENV, "OUTDIR", "plots")
const S0     = parse(Float64, get(ENV, "S0", "34.0"))   # matches src/model.jl and mixing_energetics

# The freezing season, as the months the growth intervals are counted over. Apr-Oct is the
# window `mixing_energetics` reports the brine sum across, so the two are directly comparable.
const SEASON = 4:10

const LON   = (90.0, 150.0)
const LAT   = (-70.0, -45.0)
const BANDS = [(-70.0, -65.0), (-65.0, -60.0), (-60.0, -55.0), (-55.0, -50.0), (-50.0, -45.0)]
const MONTH_TICKS = (cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30]) .+ 1,
                     ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                      "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])

fiveday_filename(token, y) = "$(token)_bsoseI156_$(y)_5day.nc"

# A 5-day stamp carries NO drift and must not be snapped to anything -- the window is simply
# the 5 days behind it. Same trap `bsose_averaging_window` warns about in setup_bsose.jl.
fiveday_center(stamp) = DateTime(stamp) - Millisecond(Dates.value(Millisecond(Day(5))) ÷ 2)

function varname(ds, cands, what)
    for c in cands
        haskey(ds, c) && return c
    end
    error("no $what variable in $(path(ds)); tried $(join(cands, ", ")). " *
          "Keys: $(collect(keys(ds)))")
end

# ── geometry ──────────────────────────────────────────────────────────────────────────────────
const GEOM = joinpath(DIR, "surfTflx_bsoseI156_2013to2024_monthly.nc")
isfile(GEOM) || error("need $GEOM for the grid (XC, YC, Depth); set BSOSE_DIR")
probe = Dataset(GEOM)
xc = Float64.(coalesce.(probe["XC"][:], NaN))
yc = Float64.(coalesce.(probe["YC"][:], NaN))
iall = findall(x -> LON[1] <= x <= LON[2], xc)
jall = findall(y -> LAT[1] <= y <= LAT[2], yc)
const IR, JR = minimum(iall):maximum(iall), minimum(jall):maximum(jall)
const DEPTH = Float64.(coalesce.(probe["Depth"][IR, JR], 0.0))
close(probe)

const LATS = yc[JR]
const WET  = DEPTH .> 0

@info "domain: $(length(IR)) x $(length(JR)) cells, $(count(WET)) wet " *
      "($(round(100 * count(WET) / length(WET), digits = 1))%)"

function band_mean(A, lo, hi)
    num = den = 0.0
    for i in axes(A, 1), j in axes(A, 2)
        (lo <= LATS[j] < hi && WET[i, j] && isfinite(A[i, j])) || continue
        w = cosd(LATS[j]); num += w * A[i, j]; den += w
    end
    return den == 0 ? NaN : num / den
end

# A wet-cell accumulator. Land starts as NaN and stays NaN through every `+=`, so there is no
# post-hoc masking step to forget and `band_mean` skips those cells on the `isfinite` test.
blank() = [WET[i, j] ? 0.0 : NaN for i in axes(WET, 1), j in axes(WET, 2)]

"""
    read5day(token, year, cands)

`(field, centres)` for one 5-day token: the regional subset over the whole year, with land
blanked to NaN, and the 73 window CENTRES. Errors loudly on a record count that is not 73 or
on spacing that is not exactly 5 days, because either means the file is not what it claims.
"""
function read5day(token, y, cands)
    p = joinpath(DIR, fiveday_filename(token, y))
    isfile(p) || error("""$(p) is not there.
                          5-day files are per year, named <token>_bsoseI156_<year>_5day.nc;
                          check the token spelling against what is actually in $(DIR).""")
    ds = Dataset(p)
    try
        v = varname(ds, cands, token)
        A = Float64.(coalesce.(ds[v][IR, JR, :], NaN))
        s = [DateTime(t) for t in ds["time"][:]]
        length(s) == 73 || error("$(p) holds $(length(s)) records; a 5-day year is 73.")
        gaps = unique(round.(Dates.value.(Millisecond.(diff(s))) ./ 86400_000, digits = 4))
        gaps == [5.0] || error("$(p) has non-uniform 5-day spacing: gaps (days) = $gaps")
        for t in axes(A, 3), j in axes(A, 2), i in axes(A, 1)
            WET[i, j] || (A[i, j, t] = NaN)
        end
        return A, [fiveday_center(x) for x in s]
    finally
        close(ds)
    end
end

# ── per year ──────────────────────────────────────────────────────────────────────────────────
Y = Dict{Int,Any}()

for y in YEARS
    area, cen = read5day("SIArea", y, ("SIarea", "SIArea", "siarea", "SIAREA"))
    heff, _   = read5day("SIHeff", y, ("SIheff", "SIHeff", "siheff", "SIHEFF"))

    # Interval m runs from centre m to centre m+1 and belongs to the month holding its midpoint.
    # `srec` is the separate list of RECORDS in season, for quantities that are states (area)
    # rather than increments (growth).
    mid  = [cen[m] + Millisecond(Dates.value(Millisecond(cen[m + 1] - cen[m])) ÷ 2)
            for m in 1:(length(cen) - 1)]
    seas = [m for m in eachindex(mid) if month(mid[m]) in SEASON]
    srec = [t for t in eachindex(cen) if month(cen[t]) in SEASON]

    # Split by sign PER CELL, then band-average. See the header.
    frz, mlt = blank(), blank()
    for m in seas, j in axes(heff, 2), i in axes(heff, 1)
        d = heff[i, j, m + 1] - heff[i, j, m]
        isfinite(d) || continue
        d > 0 ? (frz[i, j] += d) : (mlt[i, j] += d)
    end

    # Cumulative freezing through the whole year, band-averaged, for the figure.
    cum  = blank()
    cumb = [zeros(length(mid)) for _ in BANDS]
    for m in eachindex(mid)
        for j in axes(heff, 2), i in axes(heff, 1)
            d = heff[i, j, m + 1] - heff[i, j, m]
            isfinite(d) && d > 0 && (cum[i, j] += d)
        end
        for (b, (lo, hi)) in enumerate(BANDS)
            cumb[b][m] = band_mean(cum, lo, hi)
        end
    end

    # The salt side, which is what the ice is being checked against. Two terms, and they are
    # the whole reason this is here: SFLUX is the real salt flux -- brine in, so negative-
    # buoyancy -- while oceFWflx is a WATER flux that changes salinity by dilution without
    # moving any salt. Their salt-equivalent sum `SFLUX - S0*oceFWflx` is what the buoyancy
    # budget in mixing_energetics.jl actually sees, so a year whose brine held but whose net
    # collapsed was freshened, not starved of ice.
    sfx = fwx = netx = nothing
    if isfile(joinpath(DIR, fiveday_filename("surfSflx", y))) &&
       isfile(joinpath(DIR, fiveday_filename("oceFWflx", y)))
        S, _ = read5day("surfSflx", y, ("SFLUX", "surfSflx", "sflux"))
        F, _ = read5day("oceFWflx", y, ("oceFWflx", "OCEFWFLX", "ocefwflx"))
        seasmean(A) = [mean(filter(isfinite, [band_mean(A[:, :, t], lo, hi) for t in srec]))
                       for (lo, hi) in BANDS]
        # oceFWflx is kg/m^2/s and positive DOWN (freshening), so S0*oceFWflx lands in the same
        # g/m^2/s as SFLUX and subtracts from it.
        sfx  = seasmean(S)
        fwx  = seasmean(F)
        netx = [sfx[b] - S0 * fwx[b] for b in eachindex(BANDS)]
    end

    Y[y] = (cen = cen, mid = mid, seas = seas, srec = srec,
            areaband = [[band_mean(area[:, :, t], lo, hi) for t in axes(area, 3)]
                        for (lo, hi) in BANDS],
            freeze = [band_mean(frz, lo, hi) for (lo, hi) in BANDS],
            melt   = [band_mean(mlt, lo, hi) for (lo, hi) in BANDS],
            cumb = cumb, sflux = sfx, fwflx = fwx, netsalt = netx)
    @info "$y: $(length(seas)) growth intervals and $(length(srec)) records in " *
          "months $(first(SEASON))-$(last(SEASON))" *
          (sfx === nothing ? " (no 5-day surfSflx, skipping the cross-check)" : "")
end

# ── the table ─────────────────────────────────────────────────────────────────────────────────
println("\n" * "="^104)
println("SEA ICE OVER THE FREEZING SEASON (months $(first(SEASON))-$(last(SEASON))), " *
        "cos-lat weighted over wet cells")
println("  freeze / melt are the per-cell thickness increments summed BY SIGN, so they are " *
        "metres of ice grown and lost;")
println("  brine tracks `freeze`, not `net` and not area.")
println("  SFLUX is the brine salt flux and S0*FW the salt-equivalent of the freshwater flux, " *
        "both g/m^2/s; `net salt` is\n  their difference, which is the term the buoyancy " *
        "budget sees. NOTE d(SIHeff)/dt is local THICKENING, so `freeze`\n  and `melt` carry " *
        "ice advection as well as thermodynamics -- read `net` and the area columns.")
println("="^104)

for (b, (lo, hi)) in enumerate(BANDS)
    @printf("\n  %.0fS to %.0fS\n", -lo, -hi)
    @printf("    %-6s %9s %9s %8s %8s %8s %12s %12s %12s\n",
            "year", "peak area", "mean area", "freeze", "melt", "net",
            "SFLUX", "S0*FW", "net salt")
    @printf("    %-6s %9s %9s %8s %8s %8s %12s %12s %12s\n",
            "", "(%)", "(%)", "(m)", "(m)", "(m)", "(g/m2/s)", "(g/m2/s)", "(g/m2/s)")
    for y in YEARS
        d  = Y[y]
        a  = filter(isfinite, d.areaband[b])
        as = filter(isfinite, d.areaband[b][d.srec])
        cell(v) = isfinite(v) ? @sprintf("%12.4e", v) : lpad("--", 12)
        sf = d.sflux   === nothing ? NaN : d.sflux[b]
        fw = d.fwflx   === nothing ? NaN : S0 * d.fwflx[b]
        ns = d.netsalt === nothing ? NaN : d.netsalt[b]
        @printf("    %-6d %9.1f %9.1f %8.3f %8.3f %8.3f %s %s %s\n",
                y,
                isempty(a) ? NaN : 100 * maximum(a),
                isempty(as) ? NaN : 100 * mean(as),
                d.freeze[b], d.melt[b], d.freeze[b] + d.melt[b],
                cell(sf), cell(fw), cell(ns))
    end
    # Ratios against the first year, so the cross-year comparison reads without arithmetic.
    ref = Y[YEARS[1]].freeze[b]
    if length(YEARS) > 1 && isfinite(ref) && ref != 0
        print("    freeze relative to $(YEARS[1]):")
        for y in YEARS
            @printf("  %d %.2f", y, Y[y].freeze[b] / ref)
        end
        println()
    end
end

println("\n  Read `net salt` against `net` ice. Ice volume up with `net salt` down means the")
println("  freshwater term won, not that the ice failed -- and freshwater is the half of the")
println("  budget our virtual salt flux handles differently from B-SOSE's real one.")

# ── figure ────────────────────────────────────────────────────────────────────────────────────
# Years overlaid on a day-of-year axis, because the question is which year differs, not when
# anything happened in absolute time.
mkpath(OUTDIR)
fig = Figure(size = (320 * length(BANDS), 640))
cols = [:black, :royalblue3, :firebrick3, :seagreen4, :darkorange3]

for (b, (lo, hi)) in enumerate(BANDS)
    ax1 = Axis(fig[1, b], title = @sprintf("%.0fS to %.0fS", -lo, -hi),
               ylabel = b == 1 ? "ice area  (%)" : "",
               xticks = MONTH_TICKS, xticklabelsize = 9, xticklabelrotation = pi / 4)
    ax2 = Axis(fig[2, b], xlabel = "month",
               ylabel = b == 1 ? "cumulative ice grown  (m)" : "",
               xticks = MONTH_TICKS, xticklabelsize = 9, xticklabelrotation = pi / 4)
    for (n, y) in enumerate(YEARS)
        d = Y[y]
        lines!(ax1, dayofyear.(d.cen), 100 .* d.areaband[b],
               color = cols[mod1(n, length(cols))], linewidth = 2, label = string(y))
        lines!(ax2, dayofyear.(d.mid), d.cumb[b],
               color = cols[mod1(n, length(cols))], linewidth = 2, label = string(y))
    end
    b == length(BANDS) && axislegend(ax1, position = :lt, labelsize = 10, framevisible = false)
end

Label(fig[0, :], "Sea ice area and cumulative growth -- B-SOSE i156, 5-day records",
      fontsize = 17, font = :bold)
Label(fig[3, :], "Top: ice concentration. Bottom: ice GROWN since Jan 1, counting only the " *
                 "freezing increments, which is what rejects brine.\n" *
                 "A year whose bottom curve flattens through winter is not making ice, " *
                 "whatever its area is doing.",
      fontsize = 11, padding = (0, 0, 0, 6))

out = joinpath(OUTDIR, "ice_production_$(join(YEARS, "_")).png")
save(out, fig)
@info "wrote $out"
