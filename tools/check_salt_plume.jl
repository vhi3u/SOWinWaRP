# ==============================================================================
# check_salt_plume.jl -- is the brine we apply at the surface the brine B-SOSE spreads?
#
#     julia --project=. tools/check_salt_plume.jl
#
# WHY. `src/setup_bsose.jl` forces salinity with SFLUX, applied entirely in the top cell.
# B-SOSE runs MITgcm's SALT_PLUME package (available_diagnostics.log 507-510), which takes the
# brine rejected during freezing and distributes it DOWNWARD instead. The depth-integrated
# buoyancy budget is identical either way, which is exactly why `mixing_energetics.jl` reports
# ratio B near 1.0 at 70-65S -- that budget has no depth information -- while the column still
# refuses to mix and `req model` runs ~40% above `req BSOSE` in that band and no other.
#
# This is the same class of error that PENETRATING_SHORTWAVE already fixed for heat: TFLUX is
# the total, sunlight is not absorbed at the surface, and depositing all of it in the top cell
# gave SST +7 to +9 C too warm over a 3.1 m mixed layer while the column-mean temperature was
# only +0.4 off. Right integral, wrong depth. See src/model.jl.
#
# THE TWO QUESTIONS, and neither is a matter of interpretation:
#
#   1. IS oceSPflx CONTAINED IN SFLUX?  If SFLUX already carries the plumed brine then this is
#      a REDISTRIBUTION problem -- we have the right amount at the wrong depth, and the fix is
#      a vertical profile, not a new flux. If oceSPflx instead EXCEEDS SFLUX, we are missing
#      brine outright, which is a larger and different problem. Judged on the pointwise rms,
#      pooled over months, because two fields with opposite errors average to agreement.
#
#   2. HOW DEEP DOES IT GO?  `oceSPDep` is NOT archived, so the depth scale cannot be read off.
#      `oceSPtnd` is, and is better: the plume's own vertical salt tendency, one value per
#      level. Summing it over z must return oceSPflx, which is this script's check that the
#      field means what the table says. The cumulative profile then gives the depth above which
#      half the brine is deposited -- the number that decides whether this is first-order.
#
# oceSPtnd is 3-D and priced like Theta (35.76 GB monthly), so section 3 is OPTIONAL and runs
# only if the file is present. Sections 1 and 2 need oceSPflx alone, 0.70 GB.
#
# LOGIN-NODE SAFE: NCDatasets only, subset reads, no Oceananigans.
#
# ENV
#     BSOSE_DIR    directory holding the monthly files     (default: data)
#     ITER         B-SOSE iteration                        (default: 156)
#     START_DATE   first month                             (default: 2014-06-01)
#     END_DATE     last month                              (default: 2016-06-01)
#     TOL          rms residual / rms target for CLOSED    (default: 0.01)
#     ZMAX         depth to read for the 3-D profile, m    (default: 600)
#     PROFILE_MONTHS  comma-separated yyyy-mm to profile   (default: the 3 most convective)
#     OUTDIR       where the figure goes                   (default: plots)
# ==============================================================================

using NCDatasets, Dates, Printf, Statistics, CairoMakie

const DIR   = get(ENV, "BSOSE_DIR", "data")
const ITER  = parse(Int, get(ENV, "ITER", "156"))
const START = DateTime(get(ENV, "START_DATE", "2014-06-01"))
const STOP  = DateTime(get(ENV, "END_DATE", "2016-06-01"))
const TOL   = parse(Float64, get(ENV, "TOL", "0.01"))
const ZMAX  = parse(Float64, get(ENV, "ZMAX", "600"))
const OUTDIR = get(ENV, "OUTDIR", "plots")

const LON   = (90.0, 150.0)
const LAT   = (-70.0, -45.0)
const BANDS = [(-70.0, -65.0), (-65.0, -60.0), (-60.0, -55.0), (-55.0, -50.0), (-50.0, -45.0)]

# Same convention as src/setup_bsose.jl: a monthly mean is stamped at the END of the month it
# averages, so snap the stamp to the nearest month boundary and step back one.
function window_center(stamp)
    stop = round(DateTime(stamp), Month)
    start = stop - Month(1)
    return start + Millisecond(Dates.value(Millisecond(stop - start)) ÷ 2)
end

filename(token) = "$(token)_bsoseI156_2013to2024_monthly.nc"

function varname(ds, token)
    for c in (token, uppercase(token), titlecase(token),
              token == "surfSflx" ? "SFLUX" : token)
        haskey(ds, c) && return c
    end
    error("no variable matching \"$token\" in the file; keys are $(collect(keys(ds)))")
end

ITER == 156 || error("salt-plume diagnostics are i156 only; got ITER=$ITER")

# ── geometry and the month axis ───────────────────────────────────────────────────────────────
probe = Dataset(joinpath(DIR, filename("surfSflx")))
xc = Float64.(coalesce.(probe["XC"][:], NaN))
yc = Float64.(coalesce.(probe["YC"][:], NaN))
is = findall(x -> LON[1] <= x <= LON[2], xc)
js = findall(y -> LAT[1] <= y <= LAT[2], yc)
depth  = Float64.(coalesce.(probe["Depth"][is, js], 0.0))
stamps = [DateTime(t) for t in probe["time"][:]]
close(probe)

const WET  = depth .> 0
const LATS = yc[js]
const SEL  = [k for k in eachindex(stamps) if START <= window_center(stamps[k]) <= STOP]
isempty(SEL) && error("no B-SOSE record between $START and $STOP")
const LABELS = [Dates.format(window_center(stamps[k]), "yyyy-mm") for k in SEL]

@info "$(length(SEL)) months, $(first(LABELS)) to $(last(LABELS)); " *
      "$(count(WET)) of $(length(WET)) cells wet in $(LON[1])-$(LON[2])E, $(LAT[1])-$(LAT[2])N"

plumefile = joinpath(DIR, filename("oceSPflx"))
isfile(plumefile) || error("""$(plumefile) is not there.
    Fetch it with:  ./download_bsose_5day.sh --monthly --vars oceSPflx
    It is 0.70 GB for the whole 2013-2024 record.""")

function load2d(token)
    ds = Dataset(joinpath(DIR, filename(token)))
    v = varname(ds, token)
    out = map(SEL) do k
        a = Float64.(coalesce.(ds[v][is, js, k], NaN))
        reshape([WET[i] ? a[i] : NaN for i in eachindex(a)], size(depth))
    end
    close(ds)
    return out
end

@info "reading surfSflx, oceSPflx ..."
SFLUX = load2d("surfSflx")
SPFLX = load2d("oceSPflx")

# ── is the field even populated? ─────────────────────────────────────────────
# A field of EXACT zeros is the signature of two very different things: a diagnostic that was
# compiled in but never activated, and a truncated NetCDF (a partial file opens without
# complaint and reads zeros past the cut -- see the warning in download_bsose_5day.sh). Both
# must be ruled out before a plumed fraction of -0.000 gets reported as if it meant something.
let nz = count(x -> isfinite(x) && x != 0, vcat(vec.(SPFLX)...))
    if nz == 0
        ds = Dataset(plumefile); v = varname(ds, "oceSPflx"); nrec = size(ds[v], 3)
        anywhere = 0
        for k in 1:nrec
            a = ds[v][is, js, k]
            anywhere += count(x -> !ismissing(x) && isfinite(x) && x != 0, a)
            anywhere > 0 && break
        end
        close(ds)
        println("\n" * "="^100)
        println("oceSPflx IS IDENTICALLY ZERO -- stopping, because every number below it would be")
        println("="^100)
        @printf("\n  file            %s\n", plumefile)
        @printf("  size on disk    %.2f GB   (the server reports 0.70 GB for the monthly record)\n",
                filesize(plumefile) / 1073741824)
        @printf("  records         %d\n", nrec)
        @printf("  nonzero cells   %d in %s..%s, %d anywhere in the whole record\n",
                nz, first(LABELS), last(LABELS), anywhere)
        println()
        println("  TWO READINGS, and the size above separates them.")
        println()
        println("    Size well under 0.70 GB -- the download was truncated. Re-fetch; the job")
        println("    byte-checks against Content-Length and will replace a short file:")
        println("        ./download_bsose_5day.sh --monthly --vars oceSPflx")
        println()
        println("    Size correct and still zero everywhere -- SALT_PLUME is compiled into i156")
        println("    but NOT ACTIVE. available_diagnostics.log lists what was BUILT, not what was")
        println("    switched on. Then B-SOSE deposits its brine at the surface exactly as we do,")
        println("    the vertical distribution of brine is NOT a difference between us, and the")
        println("    salt-plume hypothesis is dead. Do not spend 35.76 GB on oceSPtnd to confirm")
        println("    a zero.")
        println("\n" * "="^100)
        exit(anywhere == 0 ? 0 : 1)
    end
end

finite(a) = filter(isfinite, vec(a))
rms(a) = (v = finite(a); isempty(v) ? NaN : sqrt(mean(abs2, v)))
avg(a) = (v = finite(a); isempty(v) ? NaN : mean(v))

function band_mean(A, lo, hi)
    num = den = 0.0
    for i in axes(A, 1), j in axes(A, 2)
        (lo <= LATS[j] < hi && isfinite(A[i, j])) || continue
        w = cosd(LATS[j]); num += w * A[i, j]; den += w
    end
    return den == 0 ? NaN : num / den
end

"""Pointwise statistics of `got` against `want`, pooled over every month, in the given band."""
function identity_report(name, want, got, unit; lo = -90.0, hi = 0.0)
    keep(A) = [lo <= LATS[j] < hi ? A[i, j] : NaN for i in axes(A, 1), j in axes(A, 2)]
    res = [keep(want[k]) .- keep(got[k]) for k in eachindex(want)]
    r = rms(vcat(vec.(res)...))
    w = rms(vcat(vec.(keep.(want))...))
    ratio = w == 0 ? NaN : r / w
    @printf("\n  %s\n", name)
    @printf("    rms residual %12.5g %-10s rms of target %12.5g %s\n", r, unit, w, unit)
    @printf("    ratio        %12.4f %-10s mean residual %12.5g %s   -> %s\n",
            ratio, "", avg(vcat(vec.(res)...)), unit,
            isfinite(ratio) && ratio < TOL ? "CLOSES" : "DOES NOT CLOSE")
    return ratio
end

function band_table(title, series, unit, fmt)
    println("\n  $title  ($unit)")
    print("    " * rpad("", 9) * "|")
    for l in LABELS; print(lpad(l[3:end], 11)); end
    println()
    for (lo, hi) in BANDS
        @printf("    %4.0fS-%2.0fS |", -lo, -hi)
        for k in eachindex(SEL)
            print(lpad(Printf.format(fmt, band_mean(series[k], lo, hi)), 11))
        end
        println()
    end
end

println("\n" * "="^100)
println("1. IS THE PLUMED BRINE CONTAINED IN THE SFLUX WE ALREADY APPLY?")
println("  If yes, this is a REDISTRIBUTION problem: right amount, wrong depth, and the fix is")
println("  a vertical profile. If oceSPflx EXCEEDS SFLUX we are missing brine outright.")
println("="^100)

identity_report("SFLUX  vs  oceSPflx   (whole domain)", SFLUX, SPFLX, "g/m^2/s")
identity_report("SFLUX  vs  oceSPflx   (70S-65S, the ice zone)", SFLUX, SPFLX, "g/m^2/s";
                lo = -70.0, hi = -65.0)

const FMT = Printf.Format("%.3e")
band_table("SFLUX", SFLUX, "g/m^2/s", FMT)
band_table("oceSPflx", SPFLX, "g/m^2/s", FMT)
band_table("SFLUX - oceSPflx", [SFLUX[k] .- SPFLX[k] for k in eachindex(SEL)], "g/m^2/s", FMT)

println("\n  plumed fraction, oceSPflx / SFLUX, by band (pooled over the months above)")
@printf("    %-10s %12s %12s %10s\n", "band", "mean SFLUX", "mean SPflx", "fraction")
for (lo, hi) in BANDS
    a = mean(filter(isfinite, [band_mean(SFLUX[k], lo, hi) for k in eachindex(SEL)]))
    b = mean(filter(isfinite, [band_mean(SPFLX[k], lo, hi) for k in eachindex(SEL)]))
    @printf("    %4.0fS-%2.0fS  %12.4e %12.4e %10s\n", -lo, -hi, a, b,
            a == 0 || !isfinite(a) ? "--" : @sprintf("%10.3f", b / a))
end

# ── 2. what it is worth, as salinity ──────────────────────────────────────────────────────────
# A flux integral is an upper bound, not a prediction -- the circulation exports some of it.
# It is here to say whether the term is first-order or a rounding error.
println("\n" * "="^100)
println("2. WHAT THE PLUMED BRINE IS WORTH, as dS in a 100 m layer over the months above")
println("="^100)
month_seconds(c) = Dates.value(Date(year(c), month(c), 1) + Month(1) -
                                Date(year(c), month(c), 1)) * 86400.0
const SECS = sum(month_seconds(window_center(stamps[k])) for k in SEL)
const MASS100 = 100.0 * 1025.0            # kg of water per m^2 in a 100 m layer
@printf("\n    %-10s %14s %14s\n", "band", "from SFLUX", "from oceSPflx")
@printf("    %-10s %14s %14s\n", "", "(g/kg)", "(g/kg)")
for (lo, hi) in BANDS
    a = mean(filter(isfinite, [band_mean(SFLUX[k], lo, hi) for k in eachindex(SEL)]))
    b = mean(filter(isfinite, [band_mean(SPFLX[k], lo, hi) for k in eachindex(SEL)]))
    @printf("    %4.0fS-%2.0fS  %14.4f %14.4f\n", -lo, -hi,
            a * SECS / MASS100, b * SECS / MASS100)
end

# ── 3. the vertical distribution, if oceSPtnd is on disk ──────────────────────────────────────
tndfile = joinpath(DIR, filename("oceSPtnd"))
if !isfile(tndfile)
    println("\n" * "="^100)
    println("3. VERTICAL DISTRIBUTION -- skipped, $(basename(tndfile)) is not in $DIR")
    println("  oceSPDep is NOT published, so the depth scale cannot be read off directly.")
    println("  oceSPtnd is, and is better -- it IS the distribution. 3-D, 35.76 GB monthly:")
    println("      ./download_bsose_5day.sh --monthly --vars oceSPtnd")
    println("="^100)
else
    gridfile = joinpath(DIR, "grid.nc")
    isfile(gridfile) || error("need $gridfile for the vertical grid (RC)")
    gds = Dataset(gridfile); RC_all = Float64.(coalesce.(gds["RC"][:], NaN)); close(gds)
    NZ = something(findlast(z -> -z <= ZMAX, RC_all), length(RC_all))
    RC = RC_all[1:NZ]                                     # B-SOSE is SURFACE FIRST

    # Default to the three months whose ice-zone plume flux is largest -- the convective peak,
    # which is where the depth of deposition actually decides anything.
    months = if haskey(ENV, "PROFILE_MONTHS")
        want = strip.(split(ENV["PROFILE_MONTHS"], ","))
        idx = [findfirst(==(m), LABELS) for m in want]
        any(isnothing, idx) && error("PROFILE_MONTHS not in range: $(want) vs $(LABELS)")
        collect(idx)
    else
        sortperm([abs(band_mean(SPFLX[k], -70.0, -65.0)) for k in eachindex(SEL)],
                 rev = true)[1:min(3, length(SEL))]
    end
    @info "vertical profile from $(join(LABELS[months], ", ")); $NZ levels to $(round(-RC[end], digits=1)) m"

    ds = Dataset(tndfile); vt = varname(ds, "oceSPtnd")
    prof = Dict{Tuple{Float64,Float64},Vector{Float64}}((b => zeros(NZ)) for b in BANDS)
    colsum = [fill(NaN, size(depth)) for _ in months]
    for (n, kk) in enumerate(months)
        A = Float64.(coalesce.(ds[vt][is, js, 1:NZ, SEL[kk]], NaN))
        # Land must be blanked HERE, not left to `band_mean`'s isfinite test -- a land cell
        # carrying a fill value rather than a NaN is finite, and it would land in every level
        # of the profile and drag the median toward half the column depth.
        for z in axes(A, 3), j in axes(A, 2), i in axes(A, 1)
            WET[i, j] || (A[i, j, z] = NaN)
        end
        for i in axes(A, 1), j in axes(A, 2)
            WET[i, j] || continue
            colsum[n][i, j] = sum(x -> isfinite(x) ? x : 0.0, @view A[i, j, :])
        end
        for b in BANDS
            prof[b] .+= [band_mean(A[:, :, z], b[1], b[2]) for z in 1:NZ] ./ length(months)
        end
    end
    close(ds)

    println("\n" * "="^100)
    println("3. VERTICAL DISTRIBUTION OF THE PLUME, mean of $(join(LABELS[months], ", "))")
    println("="^100)

    # Does it sum to oceSPflx? If not, the field does not mean what the table says and
    # nothing below it should be believed.
    println("\n  consistency: sum_z oceSPtnd  vs  oceSPflx")
    identity_report("oceSPflx  vs  sum_z oceSPtnd", [SPFLX[kk] for kk in months], colsum,
                    "g/m^2/s")

    println("\n  depth above which HALF the plume salt is deposited")
    @printf("    %-10s %14s %14s\n", "band", "median depth", "deepest 10%")
    @printf("    %-10s %14s %14s\n", "", "(m)", "(m)")
    for b in BANDS
        p = prof[b]
        tot = sum(x -> isfinite(x) ? x : 0.0, p)
        if !isfinite(tot) || tot == 0
            @printf("    %4.0fS-%2.0fS  %14s %14s\n", -b[1], -b[2], "--", "--"); continue
        end
        c = cumsum([isfinite(x) ? x : 0.0 for x in p]) ./ tot
        d50 = -RC[something(findfirst(>=(0.5), c), NZ)]
        d90 = -RC[something(findfirst(>=(0.9), c), NZ)]
        @printf("    %4.0fS-%2.0fS  %14.1f %14.1f\n", -b[1], -b[2], d50, d90)
    end

    mkpath(OUTDIR)
    fig = Figure(size = (320 * length(BANDS), 420))
    for (n, b) in enumerate(BANDS)
        ax = Axis(fig[1, n], title = @sprintf("%.0fS to %.0fS", -b[1], -b[2]),
                  xlabel = "plume salt tendency  (g m⁻² s⁻¹)",
                  ylabel = n == 1 ? "depth (m)" : "")
        lines!(ax, prof[b], RC, color = :firebrick3, linewidth = 2)
        vlines!(ax, 0.0, color = :gray60, linewidth = 1)
    end
    Label(fig[0, :], "Where B-SOSE puts the rejected brine -- oceSPtnd, " *
          join(LABELS[months], " / "), fontsize = 16, font = :bold)
    Label(fig[2, :], "We apply all of this in the top cell. Anything below the first level is " *
          "buoyancy loss our column never receives at depth.", fontsize = 11)
    f = joinpath(OUTDIR, "salt_plume_profile.png"); save(f, fig); @info "wrote $f"
end

println("\n" * "="^100)
println("READ IT LIKE THIS")
println("  Section 1 ratio small and the fraction near 1 at 70-65S: SFLUX IS the plumed brine,")
println("  so we apply the right amount in the wrong place and the fix is a vertical profile.")
println("  Fraction well under 1: only part of the ice-zone salt flux is plumed, and the")
println("  redistribution matters proportionally less. oceSPflx exceeding SFLUX would instead")
println("  mean brine is missing from our forcing altogether -- a different, larger problem.")
println("="^100)
