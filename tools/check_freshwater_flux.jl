# ==============================================================================
# check_freshwater_flux.jl -- is B-SOSE's SFLUX already carrying freshwater dilution,
# or do we have to add `oceFWflx` ourselves as a virtual salt flux?
#
#     julia --project=. tools/check_freshwater_flux.jl
#
# WHY THIS EXISTS. `src/setup_bsose.jl` forces surface salinity with SFLUX alone:
#
#     parent(top_S_slice[t]) .*= -1 / ρ₀        # g/m^2/s -> psu m/s, sign flipped
#
# MITgcm describes the two candidate diagnostics as
#
#     SFLUX     g/m^2/s   total salt flux (match salt-CONTENT variations), >0 increases salt
#     oceFWflx  kg/m^2/s  net surface Fresh-Water flux into the ocean (+=down), >0 DECREASES salinity
#
# Those are different physics. With a nonlinear free surface and a real freshwater flux,
# precipitation / evaporation / runoff / ice melt move WATER, not salt: salt content is
# untouched and salinity changes only because the cell gets thicker or thinner. In that
# formulation SFLUX carries brine rejection and nothing else, and the dilution lives
# entirely in oceFWflx. Oceananigans has no real freshwater flux here -- S is an ordinary
# tracer with a flux boundary condition and the cell never changes thickness -- so the
# dilution has to be added by hand as a VIRTUAL salt flux:
#
#     F_virtual = -S_surface * oceFWflx     [g/kg] * [kg/m^2/s] = [g/m^2/s]
#
# which is in the same units as SFLUX and so goes through the same -1/ρ₀ conversion.
#
# BUT if B-SOSE instead ran the LINEAR free surface with a virtual salt flux of its own,
# SFLUX would ALREADY contain the P-E term and adding oceFWflx would double it. That is the
# question this script settles, from the data rather than from a reading of MITgcm's source.
#
# TWO INDEPENDENT TESTS.
#
#   TEST A -- the ice-free band. North of about 55S in this sector there is no sea ice at any
#   time of year, so there is no brine rejection, but net precipitation is strong. A salt-only
#   SFLUX must therefore be near zero there while -S*oceFWflx is large. If instead the two are
#   comparable in that band, SFLUX already holds the dilution. This is the discriminator: it is
#   local, it does not depend on any budget closing, and the two outcomes are far apart.
#
#   TEST B -- the budget. Integrate each candidate over the run window and over a reference
#   mixed layer to get an implied ΔS. Our 2-year run drifts about +0.09 g/kg SALTY against
#   B-SOSE, so the missing term has to be a FRESHENING one of roughly that size. If
#   -S*oceFWflx integrates to about -0.09 g/kg it accounts for the whole drift.
#
# This reads subsets only (one basin, the run window) and never loads NumericalEarth or
# Oceananigans, so it is safe on a login node -- unlike `compare_bsose.jl`, which was OOM-killed
# there for loading full records.
#
# ENV OVERRIDES
#     BSOSE_DIR    directory holding the monthly files (default: the usual search path)
#     START_DATE   run start, default 2014-06-01
#     END_DATE     run end,   default 2016-06-01
#     MLD_REF      reference mixed-layer depth in m for the implied ΔS, default 100
#     S_REF        fallback surface salinity if the Salt file is absent, default 34.0
#     ICE_FREE     northern band for Test A as "lat1,lat2", default "-55,-45"
# ==============================================================================

using NCDatasets, Dates, Printf, Statistics

const LON = (90.0, 150.0)
const LAT = (-70.0, -45.0)
const ρ₀  = 1026.0

const START_DATE = DateTime(get(ENV, "START_DATE", "2014-06-01"))
const END_DATE   = DateTime(get(ENV, "END_DATE",   "2016-06-01"))
const MLD_REF    = parse(Float64, get(ENV, "MLD_REF", "100"))
const S_REF      = parse(Float64, get(ENV, "S_REF",   "34.0"))
const ICE_FREE   = (p = parse.(Float64, split(get(ENV, "ICE_FREE", "-55,-45"), ","));
                    (minimum(p), maximum(p)))

# Same search path as `default_bsose_directory` in src/setup_bsose.jl, repeated rather than
# imported so this file stays loadable without the project's heavy dependencies.
function bsose_directory()
    haskey(ENV, "BSOSE_DIR") && isdir(ENV["BSOSE_DIR"]) && return abspath(ENV["BSOSE_DIR"])
    for d in ("/storage/scratch1/1/vnguyen480/SOWinWaRP/data",
              normpath(joinpath(@__DIR__, "..", "data")), "data")
        isdir(d) && return abspath(d)
    end
    error("No BSOSE data directory found. Set BSOSE_DIR.")
end

# A BSOSE monthly record is stamped at the END of the month it averages, to within a day.
# See `bsose_averaging_window` in src/setup_bsose.jl -- same convention, so the records this
# script picks are the records the model is forced with.
bsose_averaged_month(stamp) = round(DateTime(stamp), Month) - Month(1)
bsose_window_center(stamp)  = bsose_averaged_month(stamp) + Day(15)

const DIR = bsose_directory()

path(token) = joinpath(DIR, "$(token)_bsoseI156_2013to2024_monthly.nc")

function require(token)
    p = path(token)
    isfile(p) || error("""Missing $(basename(p)) in $DIR.""")
    filesize(p) > 0 || error("""$(basename(p)) is zero bytes -- the download has not finished.""")
    return p
end

# ── open, and say exactly what oceFWflx claims to be ─────────────────────────────────────────
println("B-SOSE directory: ", DIR, "\n")

fw_ds = NCDataset(require("oceFWflx"))
sf_ds = NCDataset(require("surfSflx"))

for (label, ds, var) in (("oceFWflx", fw_ds, "oceFWflx"), ("surfSflx", sf_ds, "SFLUX"))
    v = ds[var]
    @printf("%-9s  %-8s  units=%-10s  %s\n", label, var,
            get(v.attrib, "units", "?"), get(v.attrib, "long_name", "?"))
end
println()

# ── geometry: one subset of indices used for every field ─────────────────────────────────────
xc = Float64.(coalesce.(fw_ds["XC"][:], NaN))
yc = Float64.(coalesce.(fw_ds["YC"][:], NaN))
iu = findall(x -> LON[1] <= x <= LON[2], xc)
ju = findall(y -> LAT[1] <= y <= LAT[2], yc)
isempty(iu) && error("No BSOSE columns in longitude range $LON.")
isempty(ju) && error("No BSOSE rows in latitude range $LAT.")

lat   = yc[ju]
area  = Float64.(coalesce.(fw_ds["rA"][iu, ju], NaN))
depth = Float64.(coalesce.(fw_ds["Depth"][iu, ju], NaN))
wet   = isfinite.(depth) .& (depth .> 0) .& isfinite.(area)

@printf("domain  %d x %d cells spanning %.2f to %.2f E, %.2f to %.2f S -- %d wet (%.0f%%)\n",
        length(iu), length(ju), xc[first(iu)], xc[last(iu)],
        -lat[1], -lat[end], count(wet), 100count(wet) / length(wet))

band = [ICE_FREE[1] <= y <= ICE_FREE[2] for y in lat]
wet_band = wet .& reshape(band, 1, :)
@printf("ice-free band for Test A: %.0fS to %.0fS -- %d wet cells\n\n",
        -ICE_FREE[2], -ICE_FREE[1], count(wet_band))

# ── records covering the run window ──────────────────────────────────────────────────────────
# Both files are listed independently and matched by the month each record averages, rather
# than assuming the two files are written in the same order. Guessing that has silently
# mismatched fields in this project before.
fw_month = Dict(bsose_averaged_month(s) => n for (n, s) in enumerate(fw_ds["time"][:]))
sf_month = Dict(bsose_averaged_month(s) => n for (n, s) in enumerate(sf_ds["time"][:]))

months = sort(collect(intersect(keys(fw_month), keys(sf_month))))
months = filter(m -> START_DATE <= m + Day(15) <= END_DATE, months)
isempty(months) && error("No records shared by both files inside $START_DATE..$END_DATE.")

@printf("%d shared monthly records, %s to %s\n\n",
        length(months), Dates.format(first(months), "yyyy-mm"),
        Dates.format(last(months), "yyyy-mm"))

# ── surface salinity: the real field if we have it, a constant if we do not ───────────────────
salt_path = path("Salt")
have_salt = isfile(salt_path) && filesize(salt_path) > 0
if have_salt
    salt_ds = NCDataset(salt_path)
    salt_month = Dict(bsose_averaged_month(s) => n for (n, s) in enumerate(salt_ds["time"][:]))
    println("surface salinity: from ", basename(salt_path))
else
    @warn "No Salt file here -- using S = $S_REF g/kg everywhere. Surface S varies by about " *
          "3% across this basin, far less than the factor the tests below distinguish, so " *
          "this does not change either verdict. Rerun on the HPC for the real field."
end

surface_salinity(m) =
    have_salt ? (A = Float64.(coalesce.(salt_ds["SALT"][iu, ju, 1, salt_month[m]], NaN));
                 map(s -> isfinite(s) && s > 0 ? s : S_REF, A)) :
                fill(S_REF, length(iu), length(ju))

# ── the per-record table ─────────────────────────────────────────────────────────────────────
wmean(A, mask) = (sel = mask .& isfinite.(A);
                  count(sel) == 0 ? NaN : sum(A[sel] .* area[sel]) / sum(area[sel]))

nm = length(months)
sflux_mean = fill(NaN, nm); virt_mean = fill(NaN, nm); fw_mean = fill(NaN, nm)
sflux_band = fill(NaN, nm); virt_band = fill(NaN, nm)
dt = zeros(nm)

Σsflux = zeros(length(iu), length(ju)); Σvirt = zeros(length(iu), length(ju))

println("\n", "="^96)
println("PER-RECORD DOMAIN MEANS   (g/m^2/s; positive = salinifying)")
println("="^96)
@printf("%-9s %12s %12s %12s %12s %8s | %12s %12s %8s\n",
        "month", "SFLUX", "oceFWflx", "-S*FWflx", "SFLUX+virt", "ratio",
        "band SFLUX", "band -S*FW", "ratio")
println("-"^120)

for (n, m) in enumerate(months)
    S  = surface_salinity(m)
    FW = Float64.(coalesce.(fw_ds["oceFWflx"][iu, ju, fw_month[m]], NaN))
    SF = Float64.(coalesce.(sf_ds["SFLUX"][iu, ju, sf_month[m]], NaN))
    Fv = -S .* FW

    dt[n] = Dates.value(Second((m + Month(1)) - m))

    sflux_mean[n] = wmean(SF, wet);      virt_mean[n] = wmean(Fv, wet)
    fw_mean[n]    = wmean(FW, wet)
    sflux_band[n] = wmean(SF, wet_band); virt_band[n] = wmean(Fv, wet_band)

    for I in eachindex(Σsflux)
        wet[I] || continue
        isfinite(SF[I]) && (Σsflux[I] += SF[I] * dt[n])
        isfinite(Fv[I]) && (Σvirt[I]  += Fv[I] * dt[n])
    end

    @printf("%-9s %12.3e %12.3e %12.3e %12.3e %8.2f | %12.3e %12.3e %8.3f\n",
            Dates.format(m, "yyyy-mm"), sflux_mean[n], fw_mean[n], virt_mean[n],
            sflux_mean[n] + virt_mean[n],
            abs(virt_mean[n]) > 0 ? sflux_mean[n] / virt_mean[n] : NaN,
            sflux_band[n], virt_band[n],
            abs(virt_band[n]) > 0 ? sflux_band[n] / virt_band[n] : NaN)
end

# ── TEST A: the ice-free band ────────────────────────────────────────────────────────────────
mSF_band, mFv_band = mean(filter(isfinite, sflux_band)), mean(filter(isfinite, virt_band))
ratio_mean = abs(mFv_band) > 0 ? mSF_band / mFv_band : NaN
per_record = filter(isfinite, [abs(v) > 0 ? s / v : NaN for (s, v) in zip(sflux_band, virt_band)])
ratio_band = isempty(per_record) ? NaN : median(per_record)

println("\n", "="^120)
@printf("TEST A -- the ice-free band (%.0fS to %.0fS)\n", -ICE_FREE[2], -ICE_FREE[1])
println("="^120)
@printf("  SFLUX, window mean          %12.4e g/m^2/s\n", mSF_band)
@printf("  -S*oceFWflx, window mean    %12.4e g/m^2/s\n", mFv_band)
@printf("  ratio of the window means   %12.4f\n", ratio_mean)
@printf("  MEDIAN per-record ratio     %12.4f   <- the verdict reads this\n", ratio_band)
@printf("  worst single month          %12.4f\n\n", maximum(abs, per_record))

verdict_a = if !isfinite(ratio_band)
    :inconclusive
elseif abs(ratio_band) < 0.25
    :salt_only
elseif 0.5 <= ratio_band <= 2.0
    :already_included
else
    :inconclusive
end

if verdict_a === :salt_only
    println("  => SFLUX is SALT-ONLY. No sea ice in this band, so no brine, and SFLUX is")
    println("     correspondingly small while the freshwater term is not. The dilution is")
    println("     NOT in SFLUX, and -S*oceFWflx is a term we are missing. ADD IT.")
elseif verdict_a === :already_included
    println("  => SFLUX ALREADY CONTAINS the dilution: it tracks -S*oceFWflx in a band with no")
    println("     brine to explain it. B-SOSE used a virtual salt flux of its own. DO NOT ADD")
    println("     oceFWflx -- it would double the freshwater forcing.")
else
    println("  => INCONCLUSIVE at this ratio. Look at the per-record table before deciding:")
    println("     a ratio drifting with season suggests the band is not actually ice-free.")
end

# ── TEST B: the budget ───────────────────────────────────────────────────────────────────────
dS(Σ) = (sel = wet .& isfinite.(Σ);
         sum(Σ[sel] .* area[sel]) / sum(area[sel]) / (ρ₀ * MLD_REF))

years = sum(dt) / (365.25 * 86400)

println("\n", "="^96)
@printf("TEST B -- implied ΔS over %.2f years, mixed over %.0f m\n", years, MLD_REF)
println("="^96)
@printf("  from SFLUX alone (what we force with now)   %+8.4f g/kg\n", dS(Σsflux))
@printf("  from -S*oceFWflx (the candidate term)       %+8.4f g/kg\n", dS(Σvirt))
@printf("  from both                                   %+8.4f g/kg\n", dS(Σsflux .+ Σvirt))
println()
println("  Our 2-year run drifts about +0.09 g/kg SALTY against B-SOSE, so the missing term")
println("  has to FRESHEN by roughly that much. Compare the second line against -0.09.")

# ── where it acts, against the latitudinal structure we are trying to explain ────────────────
println("\n", "="^96)
@printf("BY LATITUDE BAND -- implied ΔS over the window, mixed over %.0f m\n", MLD_REF)
println("="^96)
@printf("%-14s %10s %14s %14s %14s\n", "band", "wet cells", "SFLUX", "-S*FWflx", "both")
println("-"^72)
for lo in -70:5:-50
    hi = lo + 5
    sel = wet .& reshape([lo <= y < hi for y in lat], 1, :)
    count(sel) == 0 && continue
    bandS(Σ) = (s = sel .& isfinite.(Σ);
                sum(Σ[s] .* area[s]) / sum(area[s]) / (ρ₀ * MLD_REF))
    @printf("%3.0fS to %3.0fS   %10d %+14.4f %+14.4f %+14.4f\n",
            -lo, -hi, count(sel), bandS(Σsflux), bandS(Σvirt), bandS(Σsflux .+ Σvirt))
end

close(fw_ds); close(sf_ds); have_salt && close(salt_ds)
