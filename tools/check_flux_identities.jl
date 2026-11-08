# ==============================================================================
# check_flux_identities.jl -- are TFLUX and SFLUX the fields we think they are?
#
#     julia --project=. tools/check_flux_identities.jl
#
# WHY. `src/setup_bsose.jl` forces the surface with TFLUX (`surfTflx`) and SFLUX (`surfSflx`).
# B-SOSE also publishes its own component diagnostics of the same quantities:
#
#     oceQnet    W/m^2     net surface heat flux into the ocean (+=down), >0 increases theta
#     oceFreez   W/m^2     heating from freezing of sea-water (allowFreezing=T)
#     oceSflux   g/m^2/s   net surface Salt flux into the ocean (+=down), >0 increases salinity
#
# If TFLUX is not oceQnet + oceFreez, there is a heat term in B-SOSE's ocean that we either
# never apply or apply twice -- and `oceFreez` is by construction concentrated in the seasonal
# ice zone, which is exactly the band where our winter mixed layer is 100 m too shallow. Same
# question for SFLUX against oceSflux. This is arithmetic on B-SOSE's own diagnostics of the
# same quantity, so the answer is not a matter of interpretation.
#
# A MEAN RESIDUAL NEAR ZERO PROVES NOTHING -- two fields with opposite errors average to
# agreement. Every identity here is judged on the RMS of the POINTWISE difference, relative to
# the RMS of the field it is supposed to reproduce. The means are printed too, but as context.
#
# LOGIN-NODE SAFE: NCDatasets only, subset reads, no Oceananigans and no NumericalEarth.
#
# ENV
#     BSOSE_DIR   directory holding the monthly files   (default: data)
#     ITER        B-SOSE iteration                      (default: 156)
#     START_DATE  first month                           (default: 2014-06-01)
#     END_DATE    last month                            (default: 2016-06-01)
#     TOL         rms residual / rms target, below which an identity is called CLOSED
#                                                       (default: 0.01)
# ==============================================================================

using NCDatasets, Dates, Printf, Statistics

const DIR   = get(ENV, "BSOSE_DIR", "data")
const ITER  = parse(Int, get(ENV, "ITER", "156"))
const START = DateTime(get(ENV, "START_DATE", "2014-06-01"))
const STOP  = DateTime(get(ENV, "END_DATE", "2016-06-01"))
const TOL   = parse(Float64, get(ENV, "TOL", "0.01"))

const LON  = (90.0, 150.0)
const LAT  = (-70.0, -45.0)
const BANDS = [(-70.0, -65.0), (-65.0, -60.0), (-60.0, -55.0), (-55.0, -50.0), (-50.0, -45.0)]

# Same convention as src/setup_bsose.jl: a monthly mean is stamped at the END of the month it
# averages, so the month a record belongs to is recovered by snapping the stamp to the nearest
# month boundary and stepping back one.
function window_center(stamp)
    stop = round(DateTime(stamp), Month)
    start = stop - Month(1)
    return start + Millisecond(Dates.value(Millisecond(stop - start)) ÷ 2)
end

filename(token) = ITER == 156 ? "$(token)_bsoseI156_2013to2024_monthly.nc" :
                                "bsose_i105_2008to2012_monthly_$(token).nc"

"""The variable inside the file, which is not always the filename token (surfTflx holds TFLUX)."""
function varname(ds, token)
    for c in (token, uppercase(token), titlecase(token),
              token == "surfTflx" ? "TFLUX" : token == "surfSflx" ? "SFLUX" : token)
        haskey(ds, c) && return c
    end
    error("no variable matching \"$token\" in the file; keys are $(collect(keys(ds)))")
end

# ── geometry, from whichever file we open first ───────────────────────────────────────────────
probe = Dataset(joinpath(DIR, filename("surfTflx")))
xc = Float64.(coalesce.(probe["XC"][:], NaN))
yc = Float64.(coalesce.(probe["YC"][:], NaN))
is = findall(x -> LON[1] <= x <= LON[2], xc)
js = findall(y -> LAT[1] <= y <= LAT[2], yc)
depth = Float64.(coalesce.(probe["Depth"][is, js], 0.0))
stamps = [DateTime(t) for t in probe["time"][:]]
close(probe)

wet = depth .> 0
lat = yc[js]
sel = [k for k in eachindex(stamps) if START <= window_center(stamps[k]) <= STOP]
isempty(sel) && error("no B-SOSE record between $START and $STOP")
labels = [Dates.format(window_center(stamps[k]), "yyyy-mm") for k in sel]

@info "$(length(sel)) months, $(first(labels)) to $(last(labels)); " *
      "$(count(wet)) of $(length(wet)) cells wet in $(LON[1])-$(LON[2])E, $(LAT[1])-$(LAT[2])N"

"""Every selected month of `token`, as a Vector of (ni, nj) arrays with land set to NaN."""
function load(token)
    ds = Dataset(joinpath(DIR, filename(token)))
    v = varname(ds, token)
    out = map(sel) do k
        a = Float64.(coalesce.(ds[v][is, js, k], NaN))
        [wet[i] ? a[i] : NaN for i in eachindex(a)]
    end
    close(ds)
    return reshape.(out, Ref(size(depth)))
end

@info "reading surfTflx, oceQnet, oceFreez, surfSflx, oceSflux ..."
TFLUX   = load("surfTflx")
QNET    = load("oceQnet")
FREEZ   = load("oceFreez")
SFLUX   = load("surfSflx")
SFLUXO  = load("oceSflux")

# ── statistics ────────────────────────────────────────────────────────────────────────────────
finite(a) = filter(isfinite, vec(a))
rms(a) = (v = finite(a); isempty(v) ? NaN : sqrt(mean(abs2, v)))
avg(a) = (v = finite(a); isempty(v) ? NaN : mean(v))
mx(a)  = (v = finite(a); isempty(v) ? NaN : maximum(abs, v))

"""Pointwise residual statistics of `got` against `want`, pooled over every month."""
function identity_report(name, want, got, unit)
    res = [want[k] .- got[k] for k in eachindex(want)]
    r, w = rms(vcat(vec.(res)...)), rms(vcat(vec.(want)...))
    ratio = r / w
    verdict = ratio <= TOL ? "CLOSES" : ratio <= 0.1 ? "CLOSE, not exact" : "DOES NOT CLOSE"
    println()
    println("  ", name)
    @printf("    rms residual  %10.4g %s      rms of target %10.4g %s\n", r, unit, w, unit)
    @printf("    ratio         %10.4f                  max |residual| %10.4g %s\n", ratio, mx(vcat(vec.(res)...)), unit)
    @printf("    mean residual %10.4g %s      -> %s\n", avg(vcat(vec.(res)...)), unit, verdict)
    return res, ratio
end

println()
println("="^100)
println("DO B-SOSE'S OWN SURFACE DIAGNOSTICS RECONSTRUCT THE FIELDS WE FORCE WITH?")
println("  Judged on the rms of the pointwise difference, not on the mean.")
println("="^100)

res_q1, _ = identity_report("TFLUX  vs  oceQnet",            TFLUX, QNET, "W/m^2")
res_q2, _ = identity_report("TFLUX  vs  oceQnet + oceFreez", TFLUX,
                            [QNET[k] .+ FREEZ[k] for k in eachindex(QNET)], "W/m^2")
res_s1, _ = identity_report("SFLUX  vs  oceSflux",           SFLUX, SFLUXO, "g/m^2/s")

# ── where the freezing term actually lives ────────────────────────────────────────────────────
# This is the reason the script exists: `oceFreez` is a numerical correction concentrated in the
# seasonal ice zone, which is the band where our winter mixed layer fails.
println()
println("="^100)
println("oceFreez BY BAND AND MONTH (W/m^2, cos-lat weighted over wet cells)")
println("  Heat B-SOSE injects to stop sea water going below freezing. If the TFLUX identity")
println("  above needs this term, every W/m^2 here is heat our model has never received.")
println("="^100)

function band_mean(A, lo, hi)
    num = den = 0.0
    for i in axes(A, 1), j in axes(A, 2)
        (lo <= lat[j] < hi && isfinite(A[i, j])) || continue
        w = cosd(lat[j]); num += w * A[i, j]; den += w
    end
    return den == 0 ? NaN : num / den
end

cell(v) = isnan(v) ? lpad("--", 9) : @sprintf("%9.3f", v)

for (tag, F) in (("oceFreez", FREEZ), ("TFLUX-(oceQnet+oceFreez)", res_q2),
                 ("SFLUX-oceSflux", res_s1))
    println("\n  $tag")
    @printf("  %-10s |", "")
    for l in labels; @printf("%9s", l[3:end]); end
    println()
    for (lo, hi) in BANDS
        @printf("  %4.0fS-%2.0fS |", -lo, -hi)
        for k in eachindex(sel); print(cell(band_mean(F[k], lo, hi))); end
        println()
    end
end

println()
println("="^100)
println("READ IT LIKE THIS")
println("  A ratio under $(TOL) means the identity holds and that field adds nothing new.")
println("  A ratio that is large for `TFLUX vs oceQnet` but small for `TFLUX vs oceQnet +")
println("  oceFreez` means TFLUX already carries the freezing correction, so we are applying")
println("  it and nothing is missing. The reverse -- oceQnet alone reconstructing TFLUX while")
println("  oceFreez is nonzero in the ice zone -- means B-SOSE's ocean got heat we never did.")
println("="^100)
