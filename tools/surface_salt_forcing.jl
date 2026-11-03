# ==============================================================================================
# surface_salt_forcing.jl -- the surface salinity forcing terms, in the units of the error.
#
#     julia --project=. tools/surface_salt_forcing.jl
#     YEARS=2014,2015,2016 H_REF=100 julia --project=. tools/surface_salt_forcing.jl
#     TRIM_SPONGE=0 julia --project=. tools/surface_salt_forcing.jl     # whole bands
#
# THE QUESTION. profile_drift.jl shows WHERE the salinity error lives (depth x time, per band).
# This shows what the surface is DOING over the same months and bands, so the two can be read
# against each other: does any forcing term carry the seasonality of the error, and does its
# accumulation have the size of the drift?
#
# EVERY TERM IS CONVERTED TO THE SAME UNITS -- implied g/kg per month in a reference layer of
# depth H_REF -- because that is the only way SFLUX and oceFWflx can be compared at all:
#
#     SFLUX      g/m^2/s    total SALT flux, >0 adds salt
#     oceFWflx   kg/m^2/s   net surface FRESHWATER flux into the ocean, >0 DILUTES
#
#     virtual salt flux   Fv    = -S0 * oceFWflx          [g/kg][kg/m^2/s] = g/m^2/s
#     what the model gets Fnet  = SFLUX - S0 * oceFWflx   (setup_bsose.jl applies exactly this)
#     implied tendency    dS/dt = F / (rho0 * H_REF)      (g/m^2/s)/(kg/m^2) = g/kg/s
#
# so a panel reading +0.05 means "this term alone would salinify the top H_REF metres by
# 0.05 g/kg during that month". POSITIVE = SALINIFYING throughout, for every term.
#
# WHY SFLUX AND oceFWflx ARE SEPARATE TERMS AND NOT A DOUBLE COUNT. MITgcm ran with a real
# freshwater flux, so P, E, runoff and ice melt move WATER: they change salinity by dilution
# without moving salt mass, and so never enter a salt-CONTENT flux. What is left in SFLUX is
# brine. tools/check_freshwater_flux.jl tested this rather than assuming it -- north of 65S,
# where there is no ice to reject brine, SFLUX is ~2 orders of magnitude below -S0*oceFWflx.
#
# VERIFICATION PANELS, included deliberately rather than trimmed away:
#   oceSflux   must be bit-for-bit SFLUX (checked, ratio exactly 0 in check_flux_identities)
#   oceFreez   is identically ZERO in i156 -- a panel of exact zeros is the expected result
# If either stops being true, the loader changed under us and the rest of the figure is suspect.
#
# THE WET MASK IS maskInC, NOT `isfinite && != 0`. oceFreez is identically zero over the whole
# domain, so the usual heuristic would classify every ocean cell in that field as land and the
# band means would come back NaN. The files ship a real 0/1 mask; use it.
#
# ENV
#   BSOSE_DIR    directory holding the monthly files   (default: data)
#   YEARS        comma-separated years to plot         (default: 2014,2015,2016)
#   H_REF        reference layer depth, m              (default: 100)
#   S0           reference salinity in the virtual flux(default: 34.0, matches model.jl)
#   TRIM_SPONGE  degrees of boundary sponge to exclude (default: 3.0; 0 = whole band)
#   OUTDIR       figure directory                      (default: plots)
#
# The sponge trim is applied so the band means cover the SAME WATER as profile_drift.jl's
# panels. These are B-SOSE's own fields and our sponge cannot touch them, so the trim is about
# comparability with the error figures, not about contamination.
#
# LOCAL / LOGIN-NODE SAFE: four 2-D monthly files, one domain subset resident at a time.
# ==============================================================================================

using NCDatasets, Dates, Printf, Statistics, CairoMakie

const DIR     = get(ENV, "BSOSE_DIR", "data")
const H_REF   = parse(Float64, get(ENV, "H_REF", "100"))
const S0      = parse(Float64, get(ENV, "S0", "34.0"))
const OUTDIR  = get(ENV, "OUTDIR", "plots")
const YEARS   = [parse(Int, strip(y)) for y in split(get(ENV, "YEARS", "2014,2015,2016"), ",")]
const YEARLIST = join(YEARS, ", ")
const YEARSPAN = join(YEARS, "-")
const CADENCE = get(ENV, "CADENCE", "fiveday")
CADENCE in ("monthly", "fiveday") ||
    error("CADENCE must be \"monthly\" or \"fiveday\", got \"$CADENCE\"")
const FIVEDAY = CADENCE == "fiveday"
const CADTAG  = FIVEDAY ? "_5day" : ""      # figure suffix, so the two cadences do not collide

const ρ₀   = 1026.0                 # matches tools/mixing_energetics.jl
const SPM  = 86400 * 30.44          # seconds per mean month: the RATE unit, both cadences
const LON  = (90.0, 150.0)
const LAT  = (-70.0, -45.0)
const BANDS = [(-70.0, -65.0), (-65.0, -60.0), (-60.0, -55.0), (-55.0, -50.0), (-50.0, -45.0)]

include(joinpath(@__DIR__, "sponge.jl"))
const KEEP = sponge_trimmer(LON, LAT)

# ── file naming and the two time axes ─────────────────────────────────────────────────────────
# Monthly is one file per variable spanning 2013-2024; 5-day is one file PER YEAR with 73
# records. Which tokens exist at which cadence is not uniform, and that is the whole reason the
# resolution below is per token rather than global:
#
#   surfSflx, oceFWflx   both cadences        the two terms the model actually applies
#   oceSflux, oceFreez   MONTHLY only         verification panels
#   SIArea,   SIHeff     5-DAY only           sea ice, the mechanism behind SFLUX
#
# So neither cadence can show everything, and a token that cannot match the chosen one is
# SKIPPED with a reason rather than quietly read at the other -- two series on different time
# axes in one panel would be a straightforward lie.
monthly_filename(token) = joinpath(DIR, "$(token)_bsoseI156_2013to2024_monthly.nc")
fiveday_filename(token, y) = joinpath(DIR, "$(token)_bsoseI156_$(y)_5day.nc")

has_fiveday(token) = all(y -> isfile(fiveday_filename(token, y)), YEARS)
has_monthly(token) = isfile(monthly_filename(token))

"""Files for one token at the active cadence, or `nothing` if it cannot be read there."""
function token_files(token)
    if FIVEDAY
        has_fiveday(token) || return nothing
        return [fiveday_filename(token, y) for y in YEARS]
    end
    has_monthly(token) || return nothing
    return [monthly_filename(token)]
end

"""B-SOSE stamps the END of the averaging window; the value belongs at the midpoint."""
function window_center(stamp)
    stop  = round(DateTime(stamp), Month)
    start = stop - Month(1)
    return start + Millisecond(Dates.value(Millisecond(stop - start)) ÷ 2)
end

# A 5-day stamp carries NO drift (spacing is exactly 5.0 days), so unlike the monthly stamps it
# must NOT be snapped to anything -- the window is simply the 5 days behind it. Same trap
# `bsose_averaging_window` warns about in setup_bsose.jl.
fiveday_center(stamp) = DateTime(stamp) - Millisecond(Dates.value(Millisecond(Day(5))) ÷ 2)

center_of(stamp) = FIVEDAY ? fiveday_center(stamp) : window_center(stamp)

"""Seconds the record ending at `stamp` averages over. Months are not all 30.44 days."""
function window_seconds(stamp)
    FIVEDAY && return 5 * 86400.0
    stop = round(DateTime(stamp), Month)
    return Dates.value(Millisecond(stop - (stop - Month(1)))) / 1000
end

"""The variable inside the file, which is not always the filename token."""
function varname(ds, token)
    cands = [token, uppercase(token), lowercase(token), titlecase(token)]
    token == "surfSflx" && push!(cands, "SFLUX")
    token == "SIArea"   && append!(cands, ["SIarea", "SIAREA"])
    token == "SIHeff"   && append!(cands, ["SIheff", "SIHEFF"])
    for c in cands
        haskey(ds, c) && return c
    end
    avail = join(keys(ds), ", ")
    error("no variable for $token in the file; saw $avail")
end

# ── geometry and the time axis, from the primary token ────────────────────────────────────────
# surfSflx defines both. Every other token is checked against this axis, never the reverse.
const PRIMARY = "surfSflx"
if token_files(PRIMARY) === nothing
    error(FIVEDAY ?
        "CADENCE=fiveday needs $(join([basename(fiveday_filename(PRIMARY, y)) for y in YEARS], ", ")) " *
        "in $DIR.\n  Fetch them, or re-run with CADENCE=monthly." :
        "no $(monthly_filename(PRIMARY)). Set BSOSE_DIR, or fetch with\n" *
        "  ./download_bsose_5day.sh --monthly --vars surfSflx,oceFWflx --dest $DIR")
end

probe = Dataset(first(token_files(PRIMARY)))
xc = Float64.(probe["XC"][:]); yc = Float64.(probe["YC"][:])
const IS = findall(x -> LON[1] <= x <= LON[2], xc)
const JS = findall(y -> LAT[1] <= y <= LAT[2], yc)
const XS, YS = xc[IS], yc[JS]
const MASK = Float64.(probe["maskInC"][IS, JS]) .> 0.5
close(probe)

"""Stamps, centres and window lengths across every file of the active cadence."""
function time_axis(token)
    cent = DateTime[]; secs = Float64[]; keep = Vector{Tuple{Int,Int}}()
    for (fi, f) in enumerate(token_files(token))
        ds = Dataset(f)
        try
            for (k, t) in enumerate(ds["time"][:])
                c = center_of(t)
                year(c) in YEARS || continue      # monthly file spans 2013-2024
                push!(cent, c); push!(secs, window_seconds(t)); push!(keep, (fi, k))
            end
        finally
            close(ds)
        end
    end
    return cent, secs, keep
end

const TS, DTS, KEEPIDX = time_axis(PRIMARY)
isempty(TS) && error("no records in $(YEARLIST) at CADENCE=$CADENCE")
const LABELS = [Dates.format(t, "yy-mm") for t in TS]
@info "$CADENCE: $(length(TS)) records, $(LABELS[1]) to $(LABELS[end])"

# KEEP has no time or depth dependence, so the column test is done ONCE here rather than inside
# every band average: a (nx, ny) Bool that already folds in the land mask and the sponge trim.
const USE = [MASK[i, j] && KEEP(XS[i], YS[j]) for i in eachindex(XS), j in eachindex(YS)]
const WEIGHT = [cosd(YS[j]) for _ in eachindex(XS), j in eachindex(YS)]
count(USE) == 0 && error("the trim removed every column; lower TRIM_SPONGE")

"""Read one token over the domain and the selected records, as (nx, ny, nt) with land NaN."""
function readterm(token)
    files = token_files(token)
    files === nothing && return nothing
    cent, _, keep = time_axis(token)
    if length(cent) != length(TS) || any(cent .!= TS)
        @info "$token has $(length(cent)) records against $PRIMARY's $(length(TS)) at " *
              "CADENCE=$CADENCE -- skipping rather than plotting two time axes together"
        return nothing
    end
    A = fill(NaN, length(IS), length(JS), length(TS))
    open_ds = Dict{Int,Any}()
    try
        for (n, (fi, k)) in enumerate(keep)
            ds = get!(open_ds, fi) do; Dataset(files[fi]); end
            v = varname(ds, token)
            A[:, :, n] = Float64.(coalesce.(ds[v][IS, JS, k], NaN))
        end
    finally
        for ds in values(open_ds); close(ds); end
    end
    for t in axes(A, 3), j in axes(A, 2), i in axes(A, 1)
        USE[i, j] || (A[i, j, t] = NaN)
    end
    return A
end

"""Longitude mean at each latitude and record -> (ny, nt), the latitude-time Hovmoller."""
function lat_time(A)
    ny, nt = size(A, 2), size(A, 3)
    out = fill(NaN, ny, nt)
    for t in 1:nt, j in 1:ny
        s = 0.0; w = 0.0
        for i in axes(A, 1)
            isfinite(A[i, j, t]) || continue
            s += A[i, j, t]; w += 1.0
        end
        w > 0 && (out[j, t] = s / w)
    end
    return out
end

"""cos-lat weighted mean over one band at each record -> (nt,)."""
function band_series(A, lo, hi)
    out = fill(NaN, size(A, 3))
    for t in axes(A, 3)
        num = 0.0; den = 0.0
        for j in axes(A, 2)
            (lo <= YS[j] < hi) || continue
            for i in axes(A, 1)
                isfinite(A[i, j, t]) || continue
                num += WEIGHT[i, j] * A[i, j, t]; den += WEIGHT[i, j]
            end
        end
        den > 0 && (out[t] = num / den)
    end
    return out
end

# ── the terms, all as implied g/kg per MONTH (a rate, so both cadences are comparable) ────────
tend(F) = F .* (SPM / (ρ₀ * H_REF))

@info "reading $PRIMARY and oceFWflx over $(length(TS)) records, $(length(IS))x$(length(JS)) columns"
SFLUX = readterm("surfSflx")
FW    = readterm("oceFWflx")
SFLUX === nothing && error("could not read surfSflx at CADENCE=$CADENCE")
FW === nothing && error(FIVEDAY ?
    "oceFWflx has no 5-day file for all of $YEARLIST in $DIR -- re-run with CADENCE=monthly" :
    "no $(monthly_filename("oceFWflx"))")

TERMS = Pair{String,Array{Float64,3}}[]
push!(TERMS, "SFLUX (brine)"            => tend(SFLUX))
push!(TERMS, "-S0*oceFWflx (dilution)"  => tend(.-S0 .* FW))
push!(TERMS, "net applied"              => tend(SFLUX .- S0 .* FW))

# Verification terms: present only at monthly, and expected to be boring. Their job is to fail.
for (tok, lab) in (("oceSflux", "oceSflux (= SFLUX?)"), ("oceFreez", "oceFreez (= 0?)"))
    A = readterm(tok)
    if A === nothing
        @info "$tok unavailable at CADENCE=$CADENCE (it is monthly-only) -- panel skipped"
    else
        push!(TERMS, lab => tend(A))
    end
end

# Sea ice is not a salt flux and gets its own figure, but it IS the mechanism behind SFLUX.
ICE = Pair{String,Array{Float64,3}}[]
for (tok, lab) in (("SIArea", "sea ice area fraction"), ("SIHeff", "sea ice thickness (m)"))
    A = readterm(tok)
    if A === nothing
        @info "$tok unavailable at CADENCE=$CADENCE" *
              (has_fiveday(tok) ? " (it is 5-day only -- re-run with CADENCE=fiveday)" : "")
    else
        push!(ICE, lab => A)
    end
end

# ── the table: is any term seasonal, and in phase with what? ──────────────────────────────────
println("\n" * "="^104)
println("SURFACE SALINITY FORCING -- implied g/kg per month in the top $(Int(H_REF)) m, $YEARLIST")
println("  cadence: $CADENCE ($(length(TS)) records); `accum` integrates the rate over each window.")
println("="^104)
println("\n  " * sponge_note(LON, LAT))
println("  POSITIVE = SALINIFYING. S0 = $S0, rho0 = $ρ₀.\n")

@printf("  %-24s %-9s %9s %9s %9s %9s %9s\n",
        "term", "band", "mean/mo", "accum", "amp", "peak mo", "min mo")
@printf("  %-24s %-9s %9s %9s %9s %9s %9s\n",
        "", "", "(g/kg)", "(g/kg)", "(g/kg)", "", "")
"""Integral of a per-month RATE over the record, in g/kg. Weighted by each window's own
length, so a 5-day series and a monthly series of the same forcing accumulate to the same total."""
accum(v) = sum(isfinite(v[k]) ? v[k] * DTS[k] / SPM : 0.0 for k in eachindex(v))

SERIES = Dict{Tuple{String,Tuple{Float64,Float64}},Vector{Float64}}()
for (lab, A) in TERMS
    for b in BANDS
        s = band_series(A, b...)
        SERIES[(lab, b)] = s
        f = filter(isfinite, s)
        isempty(f) && continue
        # Climatology by calendar month, so "seasonal amplitude" is a statement about the
        # seasonal cycle and not about the trend across the three years.
        clim = [mean(filter(isfinite, s[findall(t -> month(t) == m, TS)])) for m in 1:12]
        good = findall(isfinite, clim)
        isempty(good) && continue
        amp = maximum(clim[good]) - minimum(clim[good])
        @printf("  %-24s %4.0fS-%2.0fS %9.4f %9.4f %9.4f %9s %9s\n",
                lab, -b[1], -b[2], mean(f), accum(s), amp,
                Dates.format(DateTime(2015, good[argmax(clim[good])], 1), "u"),
                Dates.format(DateTime(2015, good[argmin(clim[good])], 1), "u"))
    end
    println()
end

println("  HOW TO READ THIS AGAINST THE ERROR HOVMOLLERS")
println("    `accum` is the whole-period sum: the drift this term alone would produce in the top")
println("    $(Int(H_REF)) m if nothing removed it. Compare its SIGN and SIZE to the 0-$(Int(H_REF)) m panel of")
println("    profile_drift.jl over the same months. A term that is large but opposite in sign to")
println("    the error is not the cause; a term whose `peak mo` is out of phase with the error's")
println("    seasonality is not the cause either.")
println("    `amp` is the seasonal range of the month-of-year climatology. A term with a large")
println("    `mean/mo` but a small `amp` cannot explain a SEASONAL signal, only a trend.")
println("    REMEMBER the error is model - B-SOSE while these are B-SOSE's own fluxes. They say")
println("    what the model was TOLD to do, not what it did. A term can be in phase and still")
println("    not be the cause if the model applied it correctly.")

# ── figures ───────────────────────────────────────────────────────────────────────────────────
mkpath(OUTDIR)
monthticks = let step = max(1, length(TS) ÷ 8)
    (1:step:length(TS), LABELS[1:step:end])
end

# 1. latitude-time Hovmoller, one panel per term -- the direct analogue of the error figures
let n = length(TERMS)
    fig = Figure(size = (420 * n, 460))
    Label(fig[0, 1:n], "Surface salinity forcing -- implied g/kg per month in the top $(Int(H_REF)) m",
          fontsize = 17, font = :bold)
    Label(fig[-1, 1:n], "", fontsize = 1)   # spacer so the subtitle below does not crowd the title
    H = [lat_time(A) for (_, A) in TERMS]
    # A robust limit, not the max. A handful of ice-edge cells at 67S run an order of magnitude
    # above everything else, and scaling to them flattens all five panels to white.
    allv = abs.(vcat([filter(isfinite, vec(h)) for h in H]...))
    lim  = isempty(allv) ? 1.0 : max(quantile(allv, 0.99), 1e-12)
    local hm = nothing
    for (k, (lab, _)) in enumerate(TERMS)
        ax = Axis(fig[1, k]; title = lab, titlesize = 12, xlabel = "month",
                  ylabel = k == 1 ? "latitude" : "", xticks = monthticks,
                  xticklabelrotation = π / 4, xticklabelsize = 9)
        hm = heatmap!(ax, 1:length(TS), YS, permutedims(H[k]);
                      # REVERSED RdBu so RED = positive = salinifying, matching the sign
                      # convention of profile_drift.jl's panels. These figures get read side by
                      # side and an inverted colour scale there would reverse every conclusion.
                      colormap = Reverse(:RdBu), colorrange = (-lim, lim))
    end
    Colorbar(fig[1, n + 1], hm, label = "implied g/kg per month")
    Label(fig[2, 1:n], @sprintf("colour limit is the 99th percentile (%.3f); ice-edge cells saturate", lim),
          fontsize = 10, color = :gray40)
    save(joinpath(OUTDIR, "surface_salt_forcing_hovmoller$(CADTAG).png"), fig; px_per_unit = 2)
    @info "wrote $(joinpath(OUTDIR, "surface_salt_forcing_hovmoller$(CADTAG).png"))"
end

# 2. per-band time series, every term overlaid -- phase comparison
let
    fig = Figure(size = (1180, 820))
    Label(fig[0, 1:2], "Surface salinity forcing by band -- implied g/kg per month",
          fontsize = 17, font = :bold)
    cols = Makie.wong_colors()
    for (n, b) in enumerate(BANDS)
        ax = Axis(fig[fld(n - 1, 2) + 1, mod(n - 1, 2) + 1];
                  title = @sprintf("%.0fS to %.0fS", -b[1], -b[2]), titlesize = 12,
                  xlabel = "month", ylabel = "g/kg per month",
                  xticks = monthticks, xticklabelrotation = π / 4, xticklabelsize = 9)
        for (k, (lab, _)) in enumerate(TERMS)
            s = SERIES[(lab, b)]
            any(isfinite, s) || continue
            lines!(ax, 1:length(TS), s; color = cols[mod1(k, length(cols))], linewidth = 2,
                   linestyle = occursin("?", lab) ? :dash : :solid, label = lab)
        end
        hlines!(ax, [0], color = :gray50, linewidth = 1)
        n == 1 && axislegend(ax; position = :lb, labelsize = 8, framevisible = false)
    end
    save(joinpath(OUTDIR, "surface_salt_forcing_bands$(CADTAG).png"), fig; px_per_unit = 2)
    @info "wrote $(joinpath(OUTDIR, "surface_salt_forcing_bands$(CADTAG).png"))"
end

# 3. month-of-year climatology -- the seasonality question, answered directly
let
    fig = Figure(size = (1180, 820))
    Label(fig[0, 1:2], "Seasonal cycle of the surface salinity forcing ($YEARSPAN climatology)",
          fontsize = 17, font = :bold)
    cols = Makie.wong_colors()
    mn = Dates.format.(DateTime.(2015, 1:12, 1), "u")
    for (n, b) in enumerate(BANDS)
        ax = Axis(fig[fld(n - 1, 2) + 1, mod(n - 1, 2) + 1];
                  title = @sprintf("%.0fS to %.0fS", -b[1], -b[2]), titlesize = 12,
                  xlabel = "month of year", ylabel = "g/kg per month",
                  xticks = (1:12, mn), xticklabelsize = 9)
        for (k, (lab, _)) in enumerate(TERMS)
            s = SERIES[(lab, b)]
            clim = [(v = filter(isfinite, s[findall(t -> month(t) == m, TS)]);
                     isempty(v) ? NaN : mean(v)) for m in 1:12]
            any(isfinite, clim) || continue
            lines!(ax, 1:12, clim; color = cols[mod1(k, length(cols))], linewidth = 2,
                   linestyle = occursin("?", lab) ? :dash : :solid, label = lab)
        end
        hlines!(ax, [0], color = :gray50, linewidth = 1)
        n == 1 && axislegend(ax; position = :lb, labelsize = 8, framevisible = false)
    end
    save(joinpath(OUTDIR, "surface_salt_forcing_seasonal$(CADTAG).png"), fig; px_per_unit = 2)
    @info "wrote $(joinpath(OUTDIR, "surface_salt_forcing_seasonal$(CADTAG).png"))"
end

# 4. cumulative accumulation -- the size test, directly comparable to a g/kg error panel
let
    fig = Figure(size = (1180, 820))
    Label(fig[0, 1:2], "Cumulative implied salinity change in the top $(Int(H_REF)) m",
          fontsize = 17, font = :bold)
    cols = Makie.wong_colors()
    for (n, b) in enumerate(BANDS)
        ax = Axis(fig[fld(n - 1, 2) + 1, mod(n - 1, 2) + 1];
                  title = @sprintf("%.0fS to %.0fS", -b[1], -b[2]), titlesize = 12,
                  xlabel = "month", ylabel = "g/kg accumulated",
                  xticks = monthticks, xticklabelrotation = π / 4, xticklabelsize = 9)
        for (k, (lab, _)) in enumerate(TERMS)
            s = SERIES[(lab, b)]
            any(isfinite, s) || continue
            c = cumsum([isfinite(s[k]) ? s[k] * DTS[k] / SPM : 0.0 for k in eachindex(s)])
            lines!(ax, 1:length(TS), c; color = cols[mod1(k, length(cols))], linewidth = 2,
                   linestyle = occursin("?", lab) ? :dash : :solid, label = lab)
        end
        hlines!(ax, [0], color = :gray50, linewidth = 1)
        n == 1 && axislegend(ax; position = :lb, labelsize = 8, framevisible = false)
    end
    save(joinpath(OUTDIR, "surface_salt_forcing_cumulative$(CADTAG).png"), fig; px_per_unit = 2)
    @info "wrote $(joinpath(OUTDIR, "surface_salt_forcing_cumulative$(CADTAG).png"))"
end

# 5. sea ice, only if it was downloaded -- the mechanism behind SFLUX
if !isempty(ICE)
    fig = Figure(size = (420 * length(ICE) + 120, 460))
    Label(fig[0, 1:length(ICE)], "Sea ice over the same months", fontsize = 17, font = :bold)
    for (k, (lab, A)) in enumerate(ICE)
        h = lat_time(A)
        ax = Axis(fig[1, k]; title = lab, titlesize = 12, xlabel = "month",
                  ylabel = k == 1 ? "latitude" : "", xticks = monthticks,
                  xticklabelrotation = π / 4, xticklabelsize = 9)
        hmk = heatmap!(ax, 1:length(TS), YS, permutedims(h); colormap = :ice)
        Colorbar(fig[1, length(ICE) + k], hmk)
    end
    save(joinpath(OUTDIR, "surface_salt_forcing_seaice$(CADTAG).png"), fig; px_per_unit = 2)
    @info "wrote $(joinpath(OUTDIR, "surface_salt_forcing_seaice$(CADTAG).png"))"
end

println("\n  figures in $OUTDIR")
