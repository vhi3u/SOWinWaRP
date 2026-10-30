# ==============================================================================
# mixing_energetics.jl -- can the surface forcing WE APPLY mix B-SOSE's own water column
#                         to the depth B-SOSE reaches?
#
#     julia --project=. tools/mixing_energetics.jl
#     YEAR=2015 ZMAX=600 julia --project=. tools/mixing_energetics.jl
#
# THE CIRCULARITY THIS BREAKS. Inside our model's output every quantity explains every other:
# the surface is too fresh because it is too stratified because it mixes too little because the
# surface is too fresh. Nothing in `compare_mld.jl` can separate cause from consequence, because
# every number in it comes from the same drifting state.
#
# This does not use our model's output at all. It takes B-SOSE's OWN water column at the start
# of the freezing season, and asks whether the surface buoyancy flux our code delivers supplies
# enough energy to homogenise it to the depth B-SOSE's mixed layer actually reaches. The answer
# is independent of anything our model did.
#
#     supplied / required  <<  1   the forcing cannot make that mixed layer. Forcing problem.
#     supplied / required  >=  1   the forcing is sufficient and our model is not using it.
#                                  Closure or advection problem.
#
# THE TWO SIDES, both in m^2/s^2 so they can be divided.
#
#   REQUIRED, non-penetrative convection. To deepen to h the mixed layer must remove the
#   buoyancy by which the column above -h is lighter than the water at -h:
#
#       Breq = integral from -h to 0 of [ b(z) - b(-h) ] dz ,    b = -g (rho - rho0) / rho0
#
#   from B-SOSE's T,S at the START of the season, with h = B-SOSE's mixed layer depth at the
#   END of it.
#
#   SUPPLIED, the time integral of the surface buoyancy flux as `setup_bsose.jl` delivers it,
#   over the DEEPENING PHASE -- from the month the band first starts losing buoyancy to the
#   month its mixed layer is deepest. Starting instead at the shallowest mixed layer charges
#   the season for the whole summer restratification and makes the ratio a statement about
#   that summer rather than about the forcing; see the season selection for the numbers:
#
#       B0 = g*alpha*Q/(rho0*cp) - g*beta*Fs/rho0        [m^2/s^3]
#       Bsup = - integral of B0 dt  over the season      [m^2/s^2], positive = buoyancy LOST
#
#   with Q and Fs exactly what the model gets:  Q = TFLUX - oceQsw  (the shortwave is taken out
#   of the surface and put through the column), Fs = SFLUX - S0*oceFWflx (the virtual salt flux).
#   alpha and beta are evaluated from the same TEOS10 polynomial, by finite difference, at the
#   local surface T and S rather than assumed constant -- alpha varies by an order of magnitude
#   across this domain and is nearly zero at the freezing point, which is the whole reason the
#   ice zone behaves differently from 50S.
#
# WHAT IT IS NOT. This is a one-dimensional energy argument. It ignores wind stirring (which
# ADDS energy, so the real supply is larger than reported), and lateral restratification by
# eddies (which REMOVES mixed layer depth, so the real requirement is larger too). It is
# therefore a bound, not a simulation: `supplied << required` is strong evidence the forcing is
# deficient, while `supplied >= required` says only that the energy is there to be used.
#
# AND READ `req model` AGAINST `req BSOSE` FIRST. With MODEL_RUN set, the same deficit
# integral is run on OUR column as well as B-SOSE's. That pair never touches `supplied`, so
# unlike either ratio it carries no window sensitivity and no wind-stirring caveat: it asks
# only whether our water is harder to mix than B-SOSE's at the same place and month. A large
# excess in one band and not the others is a drifted state; a uniform one is a closure.
#
# EVERYTHING IS B-SOSE. Both sides use B-SOSE's own fields and the same equation of state, so
# the potential-temperature versus conservative-temperature mismatch that matters when
# comparing across the two models does not enter here.
#
# LOGIN-NODE SAFE: NCDatasets and SeawaterPolynomials, subset reads, no Oceananigans.
#
# ENV
#     BSOSE_DIR   directory holding the monthly files        (default: data)
#     ITER        B-SOSE iteration                           (default: 156)
#     YEAR        the freezing season to analyse             (default: 2015)
#     ZMAX        depth to read and integrate over, m        (default: 600)
#     S0          reference salinity in the virtual flux     (default: 34.0, matches model.jl)
#     MLD_DT      density threshold, kg/m^3                  (default: 0.03)
#     WINDOW      where the freezing season starts          (default: deepening)
#                 "deepening" = the autumn buoyancy-flux sign change
#                 "mldmin"    = the shallowest mixed layer in Jan-Apr (pre-10.29.2026)
#     CADENCE     cadence of the SURFACE FLUX files         (default: monthly)
#                 "monthly" | "fiveday" -- see the two-axis note below
#     MODEL_RUN   run name, to add OUR column beside B-SOSE's   (default: none)
#
# TWO TIME AXES, AND WHY. `CADENCE=fiveday` reads the four surface flux tokens from the 73
# record/year 5-day files, which is the same switch `SURFACE_CADENCE` makes in `src/model.jl`,
# so this diagnostic can be run at the cadence the model was actually forced at. But only the
# 2-D fields have 5-day files: `Theta` and `Salt` are 3-D (18.29 GB/variable/year) and exist
# monthly only. So the record index is NOT shared between the two sides:
#
#     MONTHLY SPINE   Theta, Salt -> mldof, Tsurf, Ssurf (which set alpha and beta), `required`,
#                     `req model`. Identical in both arms, because the data allows nothing else.
#     FLUX AXIS       the four flux tokens -> B0, the supply integral, the sign-change scan.
#
# Each flux record takes its alpha, beta and mixed-layer depth from the MONTHLY record whose
# month contains it. That is a real approximation, and it is the right one here: alpha and beta
# are set by surface T and S, which only exist monthly, while the quantity being resolved is the
# FLUX. What 5-day cadence buys is therefore the time structure of B0, not a sharper alpha.
#
# `deepening_start` scans at the flux cadence but returns the MONTH containing the sign change,
# so `required` and `supplied` stay anchored to the same date. Anchoring the supply integral at
# a 5-day boundary while `required` can only be read from a monthly column would put the two
# sides up to 15 days apart and undo the 10.29.2026 window fix.
#
# AND NOTE WHAT `supplied` WILL NOT DO. It is a time integral, linear in B0, and the monthly
# means ARE averages of the 5-day records, so the two cadences give nearly the same integral by
# construction. Cadence changes the sign-change month, the shape of figure 1, and the new
# intra-month minimum row -- not the total. If `supplied` moves a lot between arms, suspect the
# window, not the physics.
# ==============================================================================

using NCDatasets, Dates, Printf, Statistics, CairoMakie
include(joinpath(@__DIR__, "mld.jl"))      # one implementation of the threshold, shared

const DIR  = get(ENV, "BSOSE_DIR", "data")
const ITER = parse(Int, get(ENV, "ITER", "156"))
const YEAR = parse(Int, get(ENV, "YEAR", "2015"))
const ZMAX = parse(Float64, get(ENV, "ZMAX", "600"))
const S0   = parse(Float64, get(ENV, "S0", "34.0"))
# This one never reads run output, only B-SOSE, so there is no sweep case: always `plots/`.
const OUTDIR = get(ENV, "OUTDIR", "plots")

# WINDOW. Which month the freezing season is taken to START at. This turns out to matter more
# than anything else in the script -- see the long comment above the season selection below.
#     "deepening" (default)  the autumn sign change, when the band first starts losing buoyancy
#     "mldmin"               the shallowest mixed layer in Jan-Apr; what this did before 10.29.2026
const WINDOW = lowercase(get(ENV, "WINDOW", "deepening"))
WINDOW in ("deepening", "mldmin") ||
    error("WINDOW must be \"deepening\" or \"mldmin\", got \"$WINDOW\"")

# CADENCE. Which files the four SURFACE FLUX tokens are read from. Default monthly, so every
# number recorded before 10.30.2026 reproduces unchanged. See the two-axis note in the header:
# the 3-D side has no 5-day files and stays monthly regardless of this setting.
const CADENCE = lowercase(get(ENV, "CADENCE", "monthly"))
CADENCE in ("monthly", "fiveday") ||
    error("CADENCE must be \"monthly\" or \"fiveday\", got \"$CADENCE\"")
const FIVEDAY = CADENCE == "fiveday"
const CADTAG  = FIVEDAY ? "_5day" : ""        # figure-filename suffix, so arms do not collide
FIVEDAY && ITER != 156 &&
    error("Only B-SOSE iteration 156 ships 5-day files; got ITER=$ITER with CADENCE=fiveday.")

# Optional second column: OUR model's own stratification instead of B-SOSE's. The B-SOSE
# column answers 'is the forcing adequate in principle'. This one answers 'is our drifted
# state still mixable by it', which is the other half and the one that separates a closure
# problem from a state problem.
const MODEL_RUN   = get(ENV, "MODEL_RUN", "")
const MODEL_START = DateTime(get(ENV, "MODEL_START", "2014-06-01"))
const MODEL_DIRS  = [get(ENV, "MODEL_DIR", ".") , "surface_tests"]

# `rho_theta` from tools/mld.jl is SeawaterPolynomials' rho-prime: a density ANOMALY about
# `reference_density`, which TEOS10 sets to 1020, so it comes back around 7, not 1027. mld.jl
# only ever differences it, so the offset is invisible there -- and that is exactly the trap.
# Any ABSOLUTE density, such as the denominator of alpha and beta, needs the full value.
rho_full(T, S) = MLD_EOS.reference_density + rho_theta(T, S)

# TwoColorRadiation's defaults, which is what `bsose_penetrating_shortwave` builds. Beer's law
# in two bands: a red fraction absorbed within DZ1, the rest blue-green over DZ2. DZ2 is the
# Jerlov Type I value for the clearest open-ocean water -- override it to ask what shallower,
# chlorophyll-rich optics would do, which is the SeaWiFS question reduced to one number.
const EPS1 = parse(Float64, get(ENV, "SW_RED_FRACTION", "0.58"))
const DZ1  = parse(Float64, get(ENV, "SW_RED_SCALE", "0.35"))
const DZ2  = parse(Float64, get(ENV, "SW_BLUE_SCALE", "23.0"))

# Fraction of the surface shortwave still travelling downward at depth h -- the part that
# escapes BELOW a mixed layer of that depth and so cannot stabilise it.
solar_below(h) = isfinite(h) && h > 0 ? EPS1 * exp(-h / DZ1) + (1 - EPS1) * exp(-h / DZ2) : 1.0

const LON   = (90.0, 150.0)
const LAT   = (-70.0, -45.0)
const BANDS = [(-70.0, -65.0), (-65.0, -60.0), (-60.0, -55.0), (-55.0, -50.0), (-50.0, -45.0)]
const ρ₀, cₚ, g = 1026.0, 3991.86795711963, 9.81

function window_center(stamp)
    stop = round(DateTime(stamp), Month)
    start = stop - Month(1)
    return start + Millisecond(Dates.value(Millisecond(stop - start)) ÷ 2)
end

filename(token) = ITER == 156 ? "$(token)_bsoseI156_2013to2024_monthly.nc" :
                                "bsose_i105_2008to2012_monthly_$(token).nc"

# ── the flux axis ─────────────────────────────────────────────────────────────────────────────
#
# The four 2-D tokens that have 5-day files. Everything else -- Theta, Salt, the geometry -- is
# monthly whatever CADENCE says, so `filename` above stays the resolver for those.
const FLUX_TOKENS = ("surfTflx", "surfSflx", "oceFWflx", "oceQsw")

# 5-day files are one PER YEAR, not one per variable: `<token>_bsoseI156_<year>_5day.nc`, 73
# records, each stamped at the END of the 5 days it averages. Only YEAR's file is needed -- the
# season runs Jan to Nov, and that year's first record (Jan 5) already averages from Dec 31 of
# the year before, so there is no window hanging off the start that a second file would supply.
fiveday_filename(token, y) = "$(token)_bsoseI156_$(y)_5day.nc"
fluxfile(token) = (FIVEDAY && token in FLUX_TOKENS) ? fiveday_filename(token, YEAR) :
                                                      filename(token)

# A 5-day stamp carries NO drift (spacing min and max are both exactly 5.0 days), so unlike the
# monthly stamps it must NOT be snapped to anything -- the window is simply the 5 days behind it.
# This is the same trap that `bsose_averaging_window` carries a warning about in setup_bsose.jl.
fiveday_center(stamp) = DateTime(stamp) - Millisecond(Dates.value(Millisecond(Day(5))) ÷ 2)

function varname(ds, token)
    for c in (token, uppercase(token), titlecase(token),
              token == "surfTflx" ? "TFLUX" : token == "surfSflx" ? "SFLUX" : token)
        haskey(ds, c) && return c
    end
    error("no variable matching \"$token\" in $(DIR); keys are $(collect(keys(ds)))")
end

have(token) = isfile(joinpath(DIR, fluxfile(token)))

# ── geometry and the month axis ───────────────────────────────────────────────────────────────
probe = Dataset(joinpath(DIR, filename("surfTflx")))
xc = Float64.(coalesce.(probe["XC"][:], NaN))
yc = Float64.(coalesce.(probe["YC"][:], NaN))
iall = findall(x -> LON[1] <= x <= LON[2], xc)
jall = findall(y -> LAT[1] <= y <= LAT[2], yc)
const IR, JR = minimum(iall):maximum(iall), minimum(jall):maximum(jall)
depth = Float64.(coalesce.(probe["Depth"][IR, JR], 0.0))
stamps = [DateTime(t) for t in probe["time"][:]]
close(probe)

const LATS = yc[JR]
wet = depth .> 0

# B-SOSE's vertical grid is SURFACE FIRST (RC = -2.1, -6.7, ...), the opposite of Oceananigans.
gridfile = joinpath(DIR, ITER == 156 ? "grid.nc" : "grid105.nc")
isfile(gridfile) || error("need $gridfile for the vertical grid (RC)")
gds = Dataset(gridfile)
RC_all = Float64.(coalesce.(gds["RC"][:], NaN))
close(gds)
const NZ = something(findlast(z -> -z <= ZMAX, RC_all), length(RC_all))
const RC = RC_all[1:NZ]
@info "vertical: $NZ levels to $(round(-RC[end], digits=1)) m, top cell centre $(RC[1]) m"

monthof(k) = window_center(stamps[k])
inyear(k, y) = year(monthof(k)) == y
rec(y, m) = findfirst(k -> year(monthof(k)) == y && month(monthof(k)) == m, eachindex(stamps))

# THE FLUX AXIS. `stamps` above is the monthly axis and stays the spine; `fstamps` is the axis
# of the files the four surface tokens are actually read from. In the monthly arm the two are
# the same array, so every expression downstream is unchanged and the numbers reproduce.
const fstamps = if FIVEDAY
    fp = joinpath(DIR, fluxfile("surfTflx"))
    isfile(fp) || error("""CADENCE=fiveday needs $(fp), which is not there.
                           The 5-day files are per year: $(fiveday_filename("surfTflx", YEAR)).""")
    fds = Dataset(fp)
    s = [DateTime(t) for t in fds["time"][:]]
    close(fds)
    # A wrong time axis is the one failure here that produces plausible numbers, so check it
    # rather than trusting it: 73 records a year, and every gap exactly 5 days.
    length(s) == 73 || error("$(fp) holds $(length(s)) records; a 5-day year is 73.")
    gaps = unique(round.(Dates.value.(Millisecond.(diff(s))) ./ 86400_000, digits = 4))
    gaps == [5.0] || error("$(fp) has non-uniform 5-day spacing: gaps (days) = $gaps")
    s
else
    stamps
end

"""Middle of the window record `kf` of the FLUX axis averages."""
fcenter(kf) = FIVEDAY ? fiveday_center(fstamps[kf]) : window_center(fstamps[kf])

"""
    monthly_for(kf)

The monthly record whose month contains flux record `kf`. This is where a 5-day record gets its
`alpha`, `beta` and mixed-layer depth from, because the 3-D fields those come from exist monthly
only -- see the two-axis note in the header. The identity in the monthly arm.
"""
function monthly_for(kf)
    FIVEDAY || return kf
    c = fcenter(kf)
    k = findfirst(k -> year(monthof(k)) == year(c) && month(monthof(k)) == month(c),
                  eachindex(stamps))
    k === nothing && error("no monthly record covers $(c), the centre of flux record $kf")
    return k
end

# ── the surface side: what we actually apply ──────────────────────────────────────────────────
# `k` indexes the FLUX axis, not the monthly one. They coincide unless CADENCE=fiveday.
function load2d(token, k)
    ds = Dataset(joinpath(DIR, fluxfile(token)))
    v = varname(ds, token)
    a = Float64.(coalesce.(ds[v][IR, JR, k], NaN))
    close(ds)
    return [wet[i] ? a[i] : NaN for i in eachindex(a)] |> x -> reshape(x, size(depth))
end

"""
    buoyancy_flux(k, Tsurf, Ssurf)

Surface buoyancy flux (m^2/s^3) for record `k`, as `setup_bsose.jl` delivers it. Returns
`(total, heat_part, salt_part, total_with_full_TFLUX)`. Negative drives convection.

`alpha` and `beta` come from differencing the TEOS10 polynomial at the local surface state, so
the near-zero thermal expansion at the freezing point is represented rather than assumed away.
"""
function buoyancy_flux(k, Tsurf, Ssurf, H)
    TFLUX = load2d("surfTflx", k)
    QSW   = have("oceQsw")   ? load2d("oceQsw", k)   : zeros(size(TFLUX))
    SFLUX = have("surfSflx") ? load2d("surfSflx", k) : fill(NaN, size(TFLUX))
    FW    = have("oceFWflx") ? load2d("oceFWflx", k) : fill(NaN, size(TFLUX))

    α = similar(TFLUX); β = similar(TFLUX)
    for i in eachindex(TFLUX)
        T, S = Tsurf[i], Ssurf[i]
        if isfinite(T) && isfinite(S)
            ρ = rho_full(T, S)
            α[i] = -(rho_theta(T + 0.01, S) - rho_theta(T - 0.01, S)) / 0.02 / ρ
            β[i] =  (rho_theta(T, S + 0.01) - rho_theta(T, S - 0.01)) / 0.02 / ρ
        else
            α[i] = NaN; β[i] = NaN
        end
    end

    # What the MIXING LAYER actually feels. The shortwave is taken off the surface and put
    # through the column, but the part absorbed ABOVE the mixed layer base still buoys that
    # layer, so only what escapes below it is truly gone from the surface budget. In deep
    # winter at these latitudes oceQsw is near zero and this is a no-op; in March and April,
    # with the layer still shallow and the sun still up, it is not.
    Qsw_in     = QSW .* (1 .- solar_below.(H))
    nonsolar   = TFLUX .- QSW                # what the top boundary condition carries
    Fs         = SFLUX .- S0 .* FW           # the virtual salt flux

    b_nonsolar = g .* α .* nonsolar ./ (ρ₀ * cₚ)
    b_solar    = g .* α .* Qsw_in   ./ (ρ₀ * cₚ)
    b_salt     = .-g .* β .* Fs ./ ρ₀
    return b_nonsolar .+ b_solar .+ b_salt, b_nonsolar, b_solar, b_salt
end

# ── the column side: B-SOSE's own stratification ──────────────────────────────────────────────
function load3d(token, k)
    ds = Dataset(joinpath(DIR, filename(token)))
    v = varname(ds, token)
    a = Float64.(coalesce.(ds[v][IR, JR, 1:NZ, k], NaN))
    close(ds)
    return a
end

"""Deepest wet level of each column, in B-SOSE's surface-first ordering."""
kbot_native = [wet[i, j] ? something(findlast(k -> -RC[k] <= depth[i, j], 1:NZ), 0) : 0
               for i in axes(depth, 1), j in axes(depth, 2)]

"""
    deficit_profile(Tc, Sc, zs, h)

Buoyancy the mixed layer must remove to reach depth `h`, in m^2/s^2, for one column given
SURFACE-FIRST profiles. `NaN` when the column does not reach `h` or carries a gap.

Shared by the B-SOSE path and the model path so the two cannot drift apart; the trapezoid and
the surface-cell extension are pinned by the scratchpad test against an independent sum and
against the analytic N^2 h^2 / 2 for a linear profile.
"""
function deficit_profile(Tc, Sc, zs, h)
    kb = length(zs)
    kb >= 2 || return NaN
    (isfinite(h) && h > 0) || return NaN
    -zs[kb] >= h || return NaN                  # column shallower than the target depth
    kh = findfirst(k -> -zs[k] >= h, 1:kb)
    kh === nothing && return NaN

    bz(k) = (ρ = rho_full(Tc[k], Sc[k]); isfinite(ρ) ? -g * (ρ - ρ₀) / ρ₀ : NaN)
    bh = bz(kh)
    isfinite(bh) || return NaN

    # Trapezoid over the top kh levels, with the surface cell extended to z = 0.
    total = 0.0
    for k in 1:(kh - 1)
        b1, b2 = bz(k), bz(k + 1)
        (isfinite(b1) && isfinite(b2)) || return NaN
        total += 0.5 * ((b1 - bh) + (b2 - bh)) * (zs[k] - zs[k + 1])
    end
    total += (bz(1) - bh) * (0.0 - zs[1])
    return total
end

"""
    column_mld_and_deficit(T, S, i, j, h)

`(mld, deficit)` for one column. `mld` uses the shared threshold from `tools/mld.jl`, which
expects BOTTOM-FIRST ordering, so the profile is reversed into it. `deficit` is the buoyancy
integral the mixed layer has to remove to reach `h`, in m^2/s^2, and is `NaN` if `h` is deeper
than the column.
"""
function column_mld_and_deficit(T, S, i, j, h)
    kb = kbot_native[i, j]
    kb >= 2 || return (NaN, NaN)
    Tc, Sc = @view(T[i, j, 1:kb]), @view(S[i, j, 1:kb])
    zc = @view(RC[1:kb])

    hmld = mld(reverse(Tc), reverse(Sc), reverse(zc), 1)

    return (hmld, deficit_profile(Tc, Sc, zc, h))
end

band_mask(lo, hi) = [lo <= LATS[j] < hi for j in axes(depth, 2)]

function band_mean(A, lo, hi)
    num = den = 0.0
    for i in axes(A, 1), j in axes(A, 2)
        (lo <= LATS[j] < hi && isfinite(A[i, j])) || continue
        w = cosd(LATS[j]); num += w * A[i, j]; den += w
    end
    return den == 0 ? NaN : num / den
end

# ── run ───────────────────────────────────────────────────────────────────────────────────────
if !(have("Theta") && have("Salt"))
    @warn """$(filename("Theta")) or $(filename("Salt")) is not in $DIR, so the REQUIRED side \
             cannot be computed and only the supplied surface buoyancy flux is reported. Run \
             this where the 3-D monthly files live."""
end

months = [k for k in eachindex(stamps) if inyear(k, YEAR)]
isempty(months) && error("no B-SOSE record in $YEAR")

# Records of the FLUX axis inside YEAR. 12 in the monthly arm, 73 in the 5-day one.
fluxrec = [kf for kf in eachindex(fstamps) if year(fcenter(kf)) == YEAR]
isempty(fluxrec) && error("no surface flux record in $YEAR on the $CADENCE axis")
@info "cadence: $CADENCE -- $(length(fluxrec)) flux record(s) in $YEAR " *
      "($(Dates.format(fcenter(first(fluxrec)), "yyyy-mm-dd")) to " *
      "$(Dates.format(fcenter(last(fluxrec)), "yyyy-mm-dd")) window centres), " *
      "$(length(months)) monthly record(s) for the 3-D side"

const HAS3D = have("Theta") && have("Salt")

# One pass over the 3-D files per month. Three things need it: the surface T and S that set
# alpha and beta, the mixed layer depth that decides how much shortwave is absorbed above the
# layer base, and the season detection below. Without the 3-D files every mixed layer is NaN,
# `solar_below` returns 1, and the shortwave drops out entirely -- the surface-only fallback.
mldof = Dict{Int,Matrix{Float64}}()
Tsurf = Dict{Int,Matrix{Float64}}()
Ssurf = Dict{Int,Matrix{Float64}}()
for k in months
    if HAS3D
        @info "  $(Dates.format(monthof(k), "yyyy-mm")): mixed layer from Theta and Salt"
        T, S = load3d("Theta", k), load3d("Salt", k)
        Tsurf[k] = T[:, :, 1]
        Ssurf[k] = S[:, :, 1]
        M = fill(NaN, size(depth))
        for i in axes(depth, 1), j in axes(depth, 2)
            M[i, j] = column_mld_and_deficit(T, S, i, j, NaN)[1]
        end
        mldof[k] = M
    else
        Tsurf[k] = fill(0.5, size(depth))
        Ssurf[k] = fill(34.0, size(depth))
        mldof[k] = fill(NaN, size(depth))
    end
end

# Keyed on the FLUX axis. Each record takes alpha, beta and the mixed-layer depth from the
# monthly record containing it, which is the identity in the monthly arm.
B0 = Dict{Int,NTuple{4,Matrix{Float64}}}()
for kf in fluxrec
    m = monthly_for(kf)
    B0[kf] = buoyancy_flux(kf, Tsurf[m], Ssurf[m], mldof[m])
end

println()
println("="^108)
println("SURFACE BUOYANCY FLUX, $YEAR  (1e-8 m^2/s^3, cos-lat weighted; NEGATIVE drives convection)" *
        (FIVEDAY ? "\n  read from the 5-DAY files; columns are those records grouped by month, " *
                   "`most neg` is the single most convective window in each" : ""))
println("  heat = g*alpha*(TFLUX-oceQsw)/(rho0*cp)      salt = -g*beta*(SFLUX-$(S0)*oceFWflx)/rho0")
println("="^108)
# The table stays on MONTHLY columns even at 5-day cadence -- 73 of them does not fit on a
# terminal, and keeping the shape means the two arms can be read side by side. The 5-day records
# are grouped by the month containing them and averaged for display only; the supply integral
# below uses every record at full resolution.
const RECS_IN = Dict(k => [kf for kf in fluxrec if monthly_for(kf) == k] for k in months)
agg(k, idx, lo, hi) = (v = [band_mean(B0[kf][idx], lo, hi) for kf in RECS_IN[k]];
                       w = filter(isfinite, v); isempty(w) ? NaN : mean(w))
worst(k, lo, hi) = (v = [band_mean(B0[kf][1], lo, hi) for kf in RECS_IN[k]];
                    w = filter(isfinite, v); isempty(w) ? NaN : minimum(w))

for (lo, hi) in BANDS
    @printf("\n  %.0fS to %.0fS\n", -lo, -hi)
    @printf("  %-8s |", "")
    for k in months; @printf("%8s", Dates.format(monthof(k), "mm")); end
    println()
    for (tag, idx) in (("non-solar", 2), ("solar>ML", 3), ("salt", 4), ("total", 1))
        @printf("  %-9s|", tag)
        for k in months
            @printf("%8.2f", 1e8 * agg(k, idx, lo, hi))
        end
        println()
    end
    # The point of running at 5-day cadence: the most convective single window inside each
    # month. A month whose MEAN is stabilising can still contain a strongly destabilising
    # five days, and a monthly product cannot show that at all.
    if FIVEDAY
        @printf("  %-9s|", "most neg")
        for k in months
            @printf("%8.2f", 1e8 * worst(k, lo, hi))
        end
        println()
        @printf("  %-9s|", "n rec")
        for k in months; @printf("%8d", length(RECS_IN[k])); end
        println()
    end
end


# ── our model's own column ────────────────────────────────────────────────────────────────────
# No `for` loop with an assignment in it: at top level that makes a NEW LOCAL and leaves the
# global empty, silently, which is how this returned "--" the first time. `findfirst` keeps the
# assignment out of the loop body entirely.
hit = isempty(MODEL_RUN) ? nothing :
      findfirst(d -> isfile(joinpath(d, "$(MODEL_RUN)_eke.nc")), MODEL_DIRS)
model_path = hit === nothing ? "" : joinpath(MODEL_DIRS[hit], "$(MODEL_RUN)_eke.nc")

if !isempty(MODEL_RUN)
    isempty(model_path) ?
        @warn("no $(MODEL_RUN)_eke.nc under $(join(MODEL_DIRS, " or ")) -- skipping the model column") :
        @info "model column from $model_path"
end

# NetCDFWriter names its coordinates after the field's location, so the z of a Center field is
# `z_aac` rather than anything guessable. Try the spellings in use and say what is there on a miss.
function coordvar(ds, cands, what)
    for c in cands
        haskey(ds, c) && return Float64.(coalesce.(ds[c][:], NaN))
    end
    error("no $what coordinate in $(path(ds)); tried $(join(cands, ", ")). Keys: $(collect(keys(ds)))")
end

"""
    model_required(file, target_month, lo, hi, h)

`(required, offset_days)` for our own column: the buoyancy the mixed layer must remove to reach
`h` in band `lo..hi`, from the 30-day mean nearest `target_month`, cos-lat weighted.

Read against the B-SOSE value in the same row. B-SOSE's answers whether the forcing is adequate
in principle; this one answers whether our drifted column is still mixable by it, and the two
diverging is the whole signature of a state problem rather than a closure problem.
"""
function model_required(file, target_month, lo, hi, h)
    ds = Dataset(file)
    try
        zs_up = coordvar(ds, ["z_aac", "zC", "z_c", "z"], "z")              # bottom-to-top
        lats  = coordvar(ds, ["y_aca", "φ_aca", "φ_cca", "yC", "latitude", "lat", "y"], "latitude")
        tt    = Float64.(ds["time"][:])
        win   = haskey(ds.attrib, "averaging_window_seconds") ?
                Float64(ds.attrib["averaging_window_seconds"]) : 30 * 86400.0
        centres = [MODEL_START + Second(round(Int, x - win / 2)) for x in tt]
        gap, n = findmin(abs(Dates.value(c - target_month)) for c in centres)
        gapdays = gap / 86400_000
        if gapdays > 20
            @warn "nearest model window is $(round(gapdays, digits = 1)) days from " *
                  "$(Dates.format(target_month, "yyyy-mm")); not pairing it"
            return (NaN, NaN)
        end
        T = Float64.(coalesce.(ds["T"][:, :, :, n], NaN))
        S = Float64.(coalesce.(ds["S"][:, :, :, n], NaN))
        zs = reverse(zs_up)                                                 # surface-first

        num = den = 0.0
        for j in axes(T, 2)
            (lo <= lats[j] < hi) || continue
            w = cosd(lats[j])
            for i in axes(T, 1)
                Tc, Sc = reverse(T[i, j, :]), reverse(S[i, j, :])
                # Oceananigans writes immersed cells as EXACT zeros, so `S != 0` is the wet
                # test. `isfinite` is not, for the same reason it is not on the B-SOSE side.
                kb = something(findlast(k -> Sc[k] != 0 && isfinite(Sc[k]), eachindex(Sc)), 0)
                kb >= 2 || continue
                d = deficit_profile(Tc[1:kb], Sc[1:kb], zs[1:kb], h)
                isfinite(d) || continue
                num += w * d; den += w
            end
        end
        return (den == 0 ? NaN : num / den,
                Dates.value(centres[n] - target_month) / 86400_000)
    finally
        close(ds)
    end
end


"""
    deepening_start(kmin, ke, lo, hi)

First month in `kmin:ke` whose band-mean surface buoyancy flux is NEGATIVE -- the autumn sign
change, when the band actually begins losing buoyancy. Falls back to `kmin` with a warning if it
never does before its mixed layer maximum, which would otherwise leave the window empty.

Past the sign change the integral runs SIGNED through to `ke`: a single positive month inside the
deepening phase genuinely slows the layer down and is not something to skip over.
"""
function deepening_start(kmin, ke, lo, hi)
    # Scanned on the FLUX axis, so at CADENCE=fiveday the sign change is located to within 5
    # days instead of a month. The MONTH containing it is what comes back, because `required`
    # can only be read from a monthly column -- returning the 5-day date would anchor the two
    # sides up to 15 days apart. See the two-axis note in the header.
    span = [kf for kf in fluxrec if monthof(kmin) <= fcenter(kf) < monthof(ke)]
    function isneg(kf)
        b = band_mean(B0[kf][1], lo, hi)
        return isfinite(b) && b < 0
    end
    hit = findfirst(isneg, span)
    if hit === nothing
        @warn "$(@sprintf("%.0fS-%.0fS", -lo, -hi)): surface buoyancy flux never turns negative " *
              "before $(Dates.format(monthof(ke), "yyyy-mm")); keeping the mixed-layer minimum"
        return kmin
    end
    ks = monthly_for(span[hit])
    # A month whose MEAN is still positive can contain the first negative five days, so the
    # 5-day arm can anchor a month earlier than the monthly arm. That is a real difference in
    # the window, not noise, and it moves `required` with it -- say so rather than hide it.
    if FIVEDAY
        # One literal format string per `@sprintf` -- it takes the format at macroexpand time,
        # so a concatenation there fails to EXPAND, which neither check_syntax.jl nor
        # check_forward_refs.jl can see.
        band = @sprintf("%.0fS-%.0fS", -lo, -hi)
        @info "$band: buoyancy flux first negative in the window ending " *
              "$(Dates.format(DateTime(fstamps[span[hit]]), "yyyy-mm-dd")) -- " *
              "season anchored at $(Dates.format(monthof(ks), "yyyy-mm"))"
    end
    return ks
end
if HAS3D
    println()
    println("="^108)
    println("CAN THAT FORCING MAKE B-SOSE'S OWN MIXED LAYER?  season of $YEAR" *
            (FIVEDAY ? "   [surface fluxes: 5-day]" : ""))
    println("="^108)

    # SEASON. `ke` is the deepest B-SOSE mixed layer in Aug-Nov, which sets the target depth.
    # The START is the subtle end, and `WINDOW` chooses it.
    #
    # WHY THE DEFAULT IS NOT THE MIXED-LAYER MINIMUM. Anchoring at the shallowest mixed layer
    # puts January and February inside the integral, so the season is charged for the whole
    # summer restratification before any winter loss has begun. What the ratio then measures is
    # how strong that particular summer was, not whether the forcing can make the mixed layer:
    # 70S-65S came out at 0.91 for 2015 and 0.35 for 2016 off the same forcing product, and
    # 60S-55S went NEGATIVE in 2016 -- a net buoyancy GAIN across its own freezing season, which
    # is an artefact of the window and not a statement about anything physical. It is also
    # internally inconsistent: `required` is read off the January column while `supplied`
    # includes the summer flux that goes on to stratify that very column, so the two sides
    # describe different water. Anchoring both at the autumn sign change makes them agree.
    cand_start = [k for k in months if month(monthof(k)) in 1:4]
    cand_end   = [k for k in months if month(monthof(k)) in 8:11]
    RESULTS = NamedTuple[]
    (isempty(cand_start) || isempty(cand_end)) && error("$YEAR does not cover Jan-Apr and Aug-Nov")

    println("  window: $WINDOW -- the season starts at " *
            (WINDOW == "deepening" ? "the autumn buoyancy-flux sign change" :
                                     "the shallowest mixed layer in Jan-Apr") *
            ", and both `required` columns are read at that month." *
            (FIVEDAY ? "\n  The sign change is located on the 5-DAY axis, so this can anchor a " *
                       "month earlier than the monthly arm\n  on the same files -- see the log " *
                       "lines above. `required` moves with it, so compare arms at a PINNED\n  " *
                       "window before reading anything into a difference." : ""))

    @printf("\n  %-10s %-9s %-9s %9s %9s %11s %11s %11s %8s %8s\n",
            "band", "start", "end", "MLD start", "MLD end",
            "req BSOSE", "req model", "supplied", "ratio B", "ratio M")
    @printf("  %-10s %-9s %-9s %9s %9s %11s %11s %11s %8s %8s\n",
            "", "", "", "(m)", "(m)", "(m2/s2)", "(m2/s2)", "(m2/s2)", "", "")

    for (lo, hi) in BANDS
        kmin  = cand_start[argmin([band_mean(mldof[k], lo, hi) for k in cand_start])]
        ke    = cand_end[argmax([band_mean(mldof[k], lo, hi) for k in cand_end])]
        ks    = WINDOW == "mldmin" ? kmin : deepening_start(kmin, ke, lo, hi)
        h_end = band_mean(mldof[ke], lo, hi)

        T0, S0f = load3d("Theta", ks), load3d("Salt", ks)
        D = fill(NaN, size(depth))
        for i in axes(depth, 1), j in axes(depth, 2)
            D[i, j] = column_mld_and_deficit(T0, S0f, i, j, h_end)[2]
        end
        required = band_mean(D, lo, hi)

        # Supplied: integrate the surface buoyancy flux between the two month centres, on the
        # FLUX axis, so CADENCE=fiveday integrates every 5-day record rather than 12 monthly
        # ones. Trapezoid over the actual window centres, not a nominal step: the records are
        # MEANS, so the value sits at the centre and the interval between centres is what
        # separates them. Summing n records over an (n-1)-interval span would overcount by one.
        #
        # The endpoints stay the MONTH centres in both arms, so the two cadences integrate the
        # same span. They also give nearly the same answer -- the integral is linear in B0 and
        # the monthly means are averages of the 5-day records; see the header.
        span = [kf for kf in fluxrec if monthof(ks) <= fcenter(kf) <= monthof(ke)]
        supplied = 0.0
        for n in 1:(length(span) - 1)
            b1 = band_mean(B0[span[n]][1], lo, hi)
            b2 = band_mean(B0[span[n + 1]][1], lo, hi)
            dt = Dates.value(Millisecond(fcenter(span[n + 1]) - fcenter(span[n]))) / 1000
            supplied -= 0.5 * (b1 + b2) * dt
        end

        req_model = isempty(model_path) ? NaN :
                    first(model_required(model_path, monthof(ks), lo, hi, h_end))
        cellf(v) = isnan(v) ? lpad("--", 11) : @sprintf("%11.4f", v)
        ratf(v)  = isnan(v) ? lpad("--", 8)  : @sprintf("%8.2f", v)
        @printf("  %4.0fS-%2.0fS  %-9s %-9s %9.1f %9.1f %s %s %s %s %s\n",
                -lo, -hi, Dates.format(monthof(ks), "yyyy-mm"), Dates.format(monthof(ke), "yyyy-mm"),
                band_mean(mldof[ks], lo, hi), h_end,
                cellf(required), cellf(req_model), cellf(supplied),
                ratf(required == 0 ? NaN : supplied / required),
                ratf(isfinite(req_model) && req_model != 0 ? supplied / req_model : NaN))
        push!(RESULTS, (band = @sprintf("%.0fS-%.0fS", -lo, -hi), h0 = band_mean(mldof[ks], lo, hi),
                        h1 = h_end, required = required, req_model = req_model, supplied = supplied,
                        season = Dates.format(monthof(ks), "u") * "-" * Dates.format(monthof(ke), "u")))
    end

    println()
    println("  ratio >= 1 : the forcing supplies enough buoyancy loss to make that mixed layer,")
    println("               so a model that does not make it is not being starved of energy.")
    println("  ratio <  1 : buoyancy loss alone cannot make it. Wind stirring adds to the supply")
    println("               and eddy restratification adds to the requirement, and neither is")
    println("               counted here, so a ratio below 1 is expected and is not by itself")
    println("               evidence of a forcing deficit.")
    println()
    println("  THE ROBUST COMPARISON IS `req model` AGAINST `req BSOSE`, not either ratio. That")
    println("  pair never touches `supplied`, so it does not inherit the window sensitivity. It")
    println("  asks only whether our column is harder to mix than B-SOSE's, and a large excess")
    println("  in one band and not the others is a drifted state rather than a bad closure.")
end

# ── figures ───────────────────────────────────────────────────────────────────────────────────
mkpath(OUTDIR)
mlabels = [Dates.format(monthof(k), "u") for k in months]

# Figure 1 is drawn on the FLUX axis, so at 5-day cadence it shows the episodic structure that
# is the whole reason for running that arm. One tick per month either way: 73 dated ticks would
# be illegible, and the monthly arm then keeps exactly the ticks it had.
tickpos = filter(!isnothing,
                 [findfirst(n -> year(fcenter(fluxrec[n])) == year(monthof(k)) &&
                                 month(fcenter(fluxrec[n])) == month(monthof(k)),
                            eachindex(fluxrec)) for k in months])
ticklab = [Dates.format(fcenter(fluxrec[n]), "u") for n in tickpos]

# 1. the seasonal cycle of each buoyancy term, per band. The zero line is the thing to read
#    against: a month above it is being STABILISED by the surface forcing, whatever the mixed
#    layer is doing, and cannot be deepening on buoyancy alone.
fig = Figure(size = (320 * length(BANDS) + 160, 420))
Label(fig[0, 1:length(BANDS)],
      "Surface buoyancy flux as the model applies it — $YEAR" *
      (FIVEDAY ? "  (5-day records)" : "") *
      (HAS3D ? "" : "  (no 3-D files: shortwave term omitted)"),
      fontsize = 17, font = :bold)
ymax = maximum(abs, [1e8 * band_mean(B0[kf][i], lo, hi)
                     for kf in fluxrec, i in 1:4, (lo, hi) in BANDS
                     if isfinite(band_mean(B0[kf][i], lo, hi))]; init = 1.0)
for (c, (lo, hi)) in enumerate(BANDS)
    ax = Axis(fig[1, c], title = @sprintf("%.0fS to %.0fS", -lo, -hi), titlesize = 12,
              xlabel = "month", ylabel = c == 1 ? "1e-8 m² s⁻³" : "",
              xticks = (tickpos, ticklab), yticklabelsvisible = c == 1,
              # At 5-day cadence the month ticks land on the first record of each month, which
              # is not evenly spaced, so the labels crowd where a short month follows a long one.
              xticklabelsize = FIVEDAY ? 9 : 11)
    hlines!(ax, [0], color = :gray60, linewidth = 1)
    for (tag, idx, col, lw) in (("non-solar", 2, :firebrick, 1.5), ("solar above ML", 3, :goldenrod, 1.5),
                                ("salt", 4, :royalblue, 1.5), ("total", 1, :black, 2.5))
        ys = [1e8 * band_mean(B0[kf][idx], lo, hi) for kf in fluxrec]
        lines!(ax, 1:length(fluxrec), ys, color = col, linewidth = lw, label = tag)
    end
    ylims!(ax, -1.1ymax, 1.1ymax)
    c == length(BANDS) && axislegend(ax; position = :lt, framevisible = false, labelsize = 9)
end
Label(fig[2, 1:length(BANDS)],
      "Negative drives convection. Where `total` sits above zero through the freezing season, " *
      "the surface forcing is stabilising that band\nand whatever mixed layer forms there is " *
      "being made by the wind, which this budget does not count.",
      fontsize = 11, justification = :center)
f1 = joinpath(OUTDIR, "buoyancy_flux_$(YEAR)$(CADTAG).png"); save(f1, fig); @info "wrote $f1"

# 2. required against supplied. A bar chart because the question is a comparison of two
#    scalars per band, and the ratio is the whole answer.
if HAS3D && !isempty(RESULTS)
    fig = Figure(size = (900, 460))
    Label(fig[0, 1],
          "Can the applied surface forcing make B-SOSE's own mixed layer? — $YEAR" *
          (WINDOW == "deepening" ? "" : "  (window: $WINDOW)"),
          fontsize = 17, font = :bold)
    ax = Axis(fig[1, 1], xlabel = "", ylabel = "buoyancy  (m² s⁻²)",
              xticks = (1:length(RESULTS), [r.band for r in RESULTS]))
    n = length(RESULTS)
    hasmodel = any(r -> isfinite(r.req_model), RESULTS)
    off, wid = hasmodel ? (0.26, 0.24) : (0.19, 0.36)
    barplot!(ax, (1:n) .- off, [r.required for r in RESULTS], width = wid,
             color = :grey35, label = "required — B-SOSE column")
    hasmodel && barplot!(ax, 1:n, [isfinite(r.req_model) ? r.req_model : 0.0 for r in RESULTS],
                         width = wid, color = :darkorange3, label = "required — our column")
    barplot!(ax, (1:n) .+ off, [isfinite(r.supplied) ? max(r.supplied, 0.0) : 0.0 for r in RESULTS],
             width = wid, color = :seagreen, label = "supplied — surface buoyancy loss")
    for (i, r) in enumerate(RESULTS)
        ratio = r.required == 0 ? NaN : r.supplied / r.required
        rm = isfinite(r.req_model) && r.req_model != 0 ? r.supplied / r.req_model : NaN
        top = maximum(x -> isfinite(x) ? x : 0.0, (r.required, r.req_model, r.supplied))
        text!(ax, i, top * 1.06,
              text = isfinite(rm) ? @sprintf("B %.2f / M %.2f", ratio, rm) : @sprintf("%.2f", ratio),
              align = (:center, :bottom),
              fontsize = isfinite(rm) ? 11 : 13, font = :bold,
              color = (isfinite(rm) ? rm : ratio) >= 1 ? :seagreen : :firebrick)
        text!(ax, i, -0.02 * maximum(x -> max(x.required, isfinite(x.req_model) ? x.req_model : 0.0), RESULTS),
              text = @sprintf("%.0f→%.0f m\n%s", r.h0, r.h1, r.season),
              align = (:center, :top), fontsize = 9, color = :gray40)
    end
    # Room under the axis for the two-line annotation, which otherwise clips.
    ylims!(ax, -0.17 * maximum(x -> max(x.required, isfinite(x.req_model) ? x.req_model : 0.0), RESULTS), nothing)
    axislegend(ax; position = :lt, framevisible = false, labelsize = 10)
    Label(fig[2, 1],
          "Grey against orange is the robust comparison -- it never touches `supplied`. A large " *
          "excess of orange over grey in one band and not the others is a drifted\nstate there, " *
          "not a bad closure. The ratios do involve `supplied`, so they depend on where the " *
          "season is taken to start (window: $WINDOW). Wind is NOT counted.",
          fontsize = 11, justification = :center)
    f2 = joinpath(OUTDIR,
                  "mixing_energetics_$(YEAR)$(WINDOW == "deepening" ? "" : "_" * WINDOW)$(CADTAG).png")
    save(f2, fig); @info "wrote $f2"
end
