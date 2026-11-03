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

const DIR    = get(ENV, "BSOSE_DIR", "data")
const H_REF  = parse(Float64, get(ENV, "H_REF", "100"))
const S0     = parse(Float64, get(ENV, "S0", "34.0"))
const OUTDIR = get(ENV, "OUTDIR", "plots")
const YEARS  = [parse(Int, strip(y)) for y in split(get(ENV, "YEARS", "2014,2015,2016"), ",")]
const YEARLIST = join(YEARS, ", ")
const YEARSPAN = join(YEARS, "-")

const ρ₀   = 1026.0                 # matches tools/mixing_energetics.jl
const SPM  = 86400 * 30.44          # seconds per mean month, for the per-month tendency
const LON  = (90.0, 150.0)
const LAT  = (-70.0, -45.0)
const BANDS = [(-70.0, -65.0), (-65.0, -60.0), (-60.0, -55.0), (-55.0, -50.0), (-50.0, -45.0)]

include(joinpath(@__DIR__, "sponge.jl"))
const KEEP = sponge_trimmer(LON, LAT)

"""B-SOSE stamps the END of the averaging window; the value belongs at the window's midpoint."""
function window_center(stamp)
    stop  = round(DateTime(stamp), Month)
    start = stop - Month(1)
    return start + Millisecond(Dates.value(Millisecond(stop - start)) ÷ 2)
end

filename(token) = joinpath(DIR, "$(token)_bsoseI156_2013to2024_monthly.nc")

"""The variable inside the file, which is not always the filename token (surfSflx holds SFLUX)."""
function varname(ds, token)
    for c in (token, uppercase(token), lowercase(token),
              token == "surfSflx" ? "SFLUX" : token,
              token == "SIArea"   ? "SIarea" : token,
              token == "SIHeff"   ? "SIheff" : token)
        haskey(ds, c) && return c
    end
    avail = join(keys(ds), ", ")
    error("no variable for $token in the file; saw $avail")
end

# ── geometry, read once from whichever file is present ────────────────────────────────────────
isfile(filename("surfSflx")) ||
    error("no $(filename("surfSflx")). Set BSOSE_DIR, or fetch with\n" *
          "  ./download_bsose_5day.sh --monthly --vars surfSflx,oceFWflx --dest $DIR")

probe = Dataset(filename("surfSflx"))
xc = Float64.(probe["XC"][:]); yc = Float64.(probe["YC"][:])
const IS = findall(x -> LON[1] <= x <= LON[2], xc)
const JS = findall(y -> LAT[1] <= y <= LAT[2], yc)
const XS, YS = xc[IS], yc[JS]
const MASK = Float64.(probe["maskInC"][IS, JS]) .> 0.5
stamps = [window_center(t) for t in probe["time"][:]]
close(probe)

const KS = [k for k in eachindex(stamps) if year(stamps[k]) in YEARS]
isempty(KS) && error("no months in $(join(YEARS, ",")) -- the file covers " *
                     "$(Dates.format(first(stamps), "yyyy-mm")) to $(Dates.format(last(stamps), "yyyy-mm"))")
const TS     = stamps[KS]
const LABELS = [Dates.format(t, "yy-mm") for t in TS]

# KEEP has no time or depth dependence, so the column test is done ONCE here rather than inside
# every band average: a (nx, ny) Bool that already folds in the land mask and the sponge trim.
const USE = [MASK[i, j] && KEEP(XS[i], YS[j]) for i in eachindex(XS), j in eachindex(YS)]
const WEIGHT = [cosd(YS[j]) for _ in eachindex(XS), j in eachindex(YS)]

count(USE) == 0 && error("the trim removed every column; lower TRIM_SPONGE")

"""Read one token over the domain and the selected months, as (nx, ny, nt) with land NaN."""
function readterm(token)
    p = filename(token)
    isfile(p) || return nothing
    ds = Dataset(p)
    try
        v = varname(ds, token)
        A = Float64.(coalesce.(ds[v][IS, JS, KS], NaN))
        for t in axes(A, 3), j in axes(A, 2), i in axes(A, 1)
            USE[i, j] || (A[i, j, t] = NaN)
        end
        return A
    finally
        close(ds)
    end
end

"""Longitude mean at each latitude and month -> (ny, nt), the latitude-time Hovmoller."""
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

"""cos-lat weighted mean over one band at each month -> (nt,)."""
function band_series(A, lo, hi)
    nt = size(A, 3)
    out = fill(NaN, nt)
    for t in 1:nt
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

# ── the terms, all in implied g/kg per month ──────────────────────────────────────────────────
tend(F) = F .* (SPM / (ρ₀ * H_REF))     # g/m^2/s -> g/kg per month in H_REF metres

@info "reading surfSflx and oceFWflx over $(length(KS)) months, $(length(IS))x$(length(JS)) columns"
SFLUX = readterm("surfSflx")
FW    = readterm("oceFWflx")
SFLUX === nothing && error("missing $(filename("surfSflx"))")
FW    === nothing && error("missing $(filename("oceFWflx"))")

TERMS = Pair{String,Array{Float64,3}}[]
push!(TERMS, "SFLUX (brine)"            => tend(SFLUX))
push!(TERMS, "-S0*oceFWflx (dilution)"  => tend(.-S0 .* FW))
push!(TERMS, "net applied"              => tend(SFLUX .- S0 .* FW))

# Verification terms: present if downloaded, and expected to be boring. Their job is to fail.
for (tok, lab) in (("oceSflux", "oceSflux (= SFLUX?)"), ("oceFreez", "oceFreez (= 0?)"))
    A = readterm(tok)
    A === nothing ? @info("$tok not present -- skipping its panel") :
                    push!(TERMS, lab => tend(A))
end

# Sea ice is not a salt flux and gets its own figure, but it is the mechanism behind SFLUX.
ICE = Pair{String,Array{Float64,3}}[]
for (tok, lab) in (("SIArea", "sea ice area fraction"), ("SIHeff", "sea ice effective thickness (m)"))
    A = readterm(tok)
    A === nothing ? @info("$tok not downloaded -- skipping (./download_bsose_5day.sh --monthly --vars $tok)") :
                    push!(ICE, lab => A)
end

# ── the table: is any term seasonal, and in phase with what? ──────────────────────────────────
println("\n" * "="^104)
println("SURFACE SALINITY FORCING -- implied g/kg per month in the top $(Int(H_REF)) m, $YEARLIST")
println("="^104)
println("\n  " * sponge_note(LON, LAT))
println("  POSITIVE = SALINIFYING. S0 = $S0, rho0 = $ρ₀.\n")

@printf("  %-24s %-9s %9s %9s %9s %9s %9s\n",
        "term", "band", "mean/mo", "accum", "amp", "peak mo", "min mo")
@printf("  %-24s %-9s %9s %9s %9s %9s %9s\n",
        "", "", "(g/kg)", "(g/kg)", "(g/kg)", "", "")
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
                lab, -b[1], -b[2], mean(f), sum(f), amp,
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
    save(joinpath(OUTDIR, "surface_salt_forcing_hovmoller.png"), fig; px_per_unit = 2)
    @info "wrote $(joinpath(OUTDIR, "surface_salt_forcing_hovmoller.png"))"
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
    save(joinpath(OUTDIR, "surface_salt_forcing_bands.png"), fig; px_per_unit = 2)
    @info "wrote $(joinpath(OUTDIR, "surface_salt_forcing_bands.png"))"
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
    save(joinpath(OUTDIR, "surface_salt_forcing_seasonal.png"), fig; px_per_unit = 2)
    @info "wrote $(joinpath(OUTDIR, "surface_salt_forcing_seasonal.png"))"
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
            c = cumsum([isfinite(v) ? v : 0.0 for v in s])
            lines!(ax, 1:length(TS), c; color = cols[mod1(k, length(cols))], linewidth = 2,
                   linestyle = occursin("?", lab) ? :dash : :solid, label = lab)
        end
        hlines!(ax, [0], color = :gray50, linewidth = 1)
        n == 1 && axislegend(ax; position = :lb, labelsize = 8, framevisible = false)
    end
    save(joinpath(OUTDIR, "surface_salt_forcing_cumulative.png"), fig; px_per_unit = 2)
    @info "wrote $(joinpath(OUTDIR, "surface_salt_forcing_cumulative.png"))"
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
    save(joinpath(OUTDIR, "surface_salt_forcing_seaice.png"), fig; px_per_unit = 2)
    @info "wrote $(joinpath(OUTDIR, "surface_salt_forcing_seaice.png"))"
end

println("\n  figures in $OUTDIR")
