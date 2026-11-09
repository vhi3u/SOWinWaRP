# ==============================================================================
# model_surface_buoyancy.jl -- the surface buoyancy flux a nested run ACTUALLY received, against
#                              B-SOSE's, by band and on the shelf, with brine and melt kept apart.
#
#     julia --project=. tools/model_surface_buoyancy.jl 2Y_nested_seaice_fix5_flux
#     BOX=130,150,-70,-65 TRIM_SPONGE=0.67 julia --project=. tools/model_surface_buoyancy.jl <RUN>
#
# WHY. tools/mixing_energetics.jl's `supplied` is B-SOSE's own forcing (TFLUX, SFLUX, oceFWflx)
# -- what src/model.jl used to apply. A nested run is forced by ERA5 through the bulk formulae
# and by its own sea ice, so that number says nothing about it. This reads what the coupler
# handed the ocean (<RUN>_surface_fluxes.nc, daily means) and puts it beside B-SOSE's, record
# for record, in the same units as mixing_energetics' first table.
#
# WHAT THE RUN WRITES (NumericalEarth's net ocean fluxes, the run's TwoColorRadiation):
#     heat_flux        J_T, kinematic (K m/s), positive OUT: turbulent + longwave + the ice-ocean
#                      interface heat. NO shortwave -- with TwoColorRadiation all of it is put
#                      through the column, so J_T is the analogue of B-SOSE's TFLUX - oceQsw.
#     ice_salt_flux    J_S, kinematic (g/kg m/s), positive out: the sea-ice salt exchange only
#     freshwater_flux  J_w (m/s), positive INTO the ocean: rain, snow, evaporation, ice melt/growth
#     shortwave        W/m2, negative down, already (1 - ℵ)-weighted; it all penetrates
# The salinity boundary condition adds S_top J_w live, so the salt flux the ocean sees is
# J_S + S_top J_w (positive out). S_top is the window mean of <RUN>_surface.nc's S.
#
#     model   b_heat = -g α J_T                 b_salt = +g β (J_S + S_top J_w)
#     B-SOSE  b_heat = g α (TFLUX - oceQsw)/(ρ0 cp)   b_salt = -g β (SFLUX - S0 oceFWflx)/ρ0
# exactly mixing_energetics' heat and salt terms; α and β from the TEOS10 polynomial at each
# side's own surface T and S (tools/mld.jl). NEGATIVE drives convection.
#
# LIKE FOR LIKE. B-SOSE's 5-day files when every year the run spans has all four tokens, else
# the monthly files. Each B-SOSE record is a window mean, so the model's daily means are averaged
# over the SAME window first, and only then is anything split or compared.
#
# BRINE AND MELT APART. b_salt is split by sign per cell and per window BEFORE the region mean:
#     salt gain (b_salt < 0)   brine from freezing, and evaporation   -- "brine" below
#     freshening (b_salt > 0)  melt, precipitation, runoff            -- "fresh" below
# A cell freezing next to one that is melting cancels in a region-mean salt flux while both are
# really happening (the same reasoning as tools/ice_production.jl). Both datasets are split on the
# same window length, so the split is like for like; a shorter window would show more of both.
#
# THE SHORTWAVE IS LEFT OUT of `total` and shown on its own (W/m2, positive in). Which part of it
# buoys the mixed layer depends on the mixed layer depth (mixing_energetics' "solar>ML" row); in
# Apr-Sep south of 60S it is small, further north it is not. `total` = heat + brine + fresh.
#
# REGIONS: the latitude bands every tool uses, and BOX (default the eastern shelf, 130-150E south
# of 65S, where the Adélie/Mertz polynya should be), all outside the TRIM_SPONGE band (0.67 for
# a nested run). cos(lat)-weighted over wet cells.
#
# PRINTS per region and month: model / B-SOSE for heat, brine, fresh, total (1e-8 m2/s3) and the
#        shortwave (W/m2); then each freezing season (Apr-Oct) integrated, -∫B dt (m2/s2), which
#        is mixing_energetics' `supplied` for THIS run, less the solar term.
# WRITES (OUTDIR, default plots/):
#     <RUN>_buoyancy_series.png       monthly heat, brine, fresh and total, model red, B-SOSE black
#     <RUN>_buoyancy_winter_maps.png  Jun-Sep brine and heat, model / B-SOSE / difference, per winter
#     <RUN>_buoyancy_summer_maps.png  Dec-Mar freshening and heat, the same
#
# LOGIN-NODE SAFE: NCDatasets and SeawaterPolynomials only, one window at a time.
#
# ENV
#     DIR          where <RUN>_surface_fluxes.nc and <RUN>_surface.nc are   (default: .)
#     BSOSE_DIR    B-SOSE files                                             (default: data)
#     BOX          lon1,lon2,lat1,lat2 of the extra region                  (default: 130,150,-70,-65)
#     TRIM_SPONGE  boundary band left out, deg                              (default 3.0 via sponge.jl)
#     S0           B-SOSE's reference salinity for the virtual salt flux    (default: 34.0)
#     OUTDIR       figures                                                  (default: plots)
# ==============================================================================

using NCDatasets, Dates, Printf, Statistics, CairoMakie

include(joinpath(@__DIR__, "mld.jl"))      # rho_theta, MLD_EOS
include(joinpath(@__DIR__, "sponge.jl"))

const RUN    = isempty(ARGS) ? get(ENV, "RUN", "") : ARGS[1]
isempty(RUN) && error("give the run name: julia --project=. tools/model_surface_buoyancy.jl <RUN>")
const DIR    = get(ENV, "DIR", ".")
const BDIR   = get(ENV, "BSOSE_DIR", "data")
const OUTDIR = get(ENV, "OUTDIR", "plots")
const S0     = parse(Float64, get(ENV, "S0", "34.0"))
const BOX    = Tuple(parse.(Float64, split(get(ENV, "BOX", "130,150,-70,-65"), ",")))
mkpath(OUTDIR)

const LON   = (90.0, 150.0)
const LAT   = (-70.0, -45.0)
const BANDS = [(-70.0, -65.0), (-65.0, -60.0), (-60.0, -55.0), (-55.0, -50.0), (-50.0, -45.0)]
const KEEP  = sponge_trimmer(LON, LAT)
const ρ₀, cₚ, g = 1026.0, 3991.86795711963, 9.81       # B-SOSE's conversion, as mixing_energetics
const SCALE = 1e8                                      # print buoyancy in 1e-8 m2/s3

const REGIONS = [[(@sprintf("%.0fS-%.0fS", -lo, -hi), (λ, φ) -> lo <= φ < hi) for (lo, hi) in BANDS];
                 (@sprintf("box %.0f-%.0fE %.0f-%.0fS", BOX[1], BOX[2], -BOX[3], -BOX[4]),
                  (λ, φ) -> BOX[1] <= λ <= BOX[2] && BOX[3] <= φ <= BOX[4])]

f64(A) = Float64.(coalesce.(A, NaN))

"""α and β (per K, per g/kg) from the TEOS10 polynomial at (T, S), as mixing_energetics does."""
function alpha_beta(T, S)
    (isfinite(T) && isfinite(S)) || return NaN, NaN
    ρ = MLD_EOS.reference_density + rho_theta(T, S)
    α = -(rho_theta(T + 0.01, S) - rho_theta(T - 0.01, S)) / 0.02 / ρ
    β =  (rho_theta(T, S + 0.01) - rho_theta(T, S - 0.01)) / 0.02 / ρ
    return α, β
end

# ── the model ─────────────────────────────────────────────────────────────────────────────────
const FLUXFILE = joinpath(DIR, "$(RUN)_surface_fluxes.nc")
const SURFFILE = joinpath(DIR, "$(RUN)_surface.nc")
for f in (FLUXFILE, SURFFILE)
    isfile(f) || error("$f not found (set DIR to the run directory)")
end

const M = let
    ds = NCDataset(FLUXFILE)
    for v in ("heat_flux", "ice_salt_flux", "freshwater_flux", "shortwave")
        haskey(ds, v) || error("$FLUXFILE has no $v; the run wrote $(collect(keys(ds)))")
    end
    lon, lat = f64(ds["λ_caa"][:]), f64(ds["φ_aca"][:])
    tf = f64(ds["time"][:])
    win = haskey(ds.attrib, "averaging_window_seconds") ? Float64(ds.attrib["averaging_window_seconds"]) : 86400.0
    start = DateTime(string(ds.attrib["start_date"]))
    bh = haskey(ds, "bottom_height") ? f64(ds["bottom_height"][:, :]) : nothing
    close(ds)
    ss = NCDataset(SURFFILE)
    ts = f64(ss["time"][:])
    bh === nothing && (bh = f64(ss["bottom_height"][:, :]))
    close(ss)
    keep = findall(>(0), tf)                                  # t = 0 is the initial state, not a mean
    centres = [start + Millisecond(round(Int, 1000 * (x - win / 2))) for x in tf]
    (; lon, lat, wet = bh .< 0, tf, ts, start, win, keep, centres)
end
@info @sprintf("%s: %d daily flux means, %s to %s", basename(FLUXFILE), length(M.keep),
               Date(M.centres[M.keep[1]]), Date(M.centres[M.keep[end]]))

# Read a variable over records `n`, as (Nx, Ny, length(n)) whatever its dimensions
function readrecs(ds, v, n)
    A = ds[v]
    nd = ndims(A)
    X = nd == 3 ? A[:, :, n] : A[:, :, 1, n]
    return reshape(f64(X), size(X, 1), size(X, 2), length(n))
end

"""Model window means over [a, b): (b_heat, b_salt, sw, ndays), per cell."""
function model_window(a, b)
    n = [k for k in M.keep if a <= M.centres[k] < b]
    isempty(n) && return nothing
    ds = NCDataset(FLUXFILE)
    JT = dropdims(mean(readrecs(ds, "heat_flux", n), dims = 3), dims = 3)
    JS = readrecs(ds, "ice_salt_flux", n)
    Jw = readrecs(ds, "freshwater_flux", n)
    SW = dropdims(mean(readrecs(ds, "shortwave", n), dims = 3), dims = 3)
    close(ds)
    # S_top: the surface snapshots at the two ends of each day, averaged into the day
    ss = NCDataset(SURFFILE)
    ends = [M.tf[k] for k in n]
    snaps = unique(vcat([findmin(abs.(M.ts .- (e - M.win)))[2] for e in ends],
                        [findmin(abs.(M.ts .- e))[2] for e in ends]))
    sort!(snaps)
    Ss, Ts = readrecs(ss, "S", snaps), readrecs(ss, "T", snaps)
    close(ss)
    Sday = similar(JS)
    for (d, e) in enumerate(ends)
        k1 = searchsortedfirst(snaps, findmin(abs.(M.ts .- (e - M.win)))[2])
        k2 = searchsortedfirst(snaps, findmin(abs.(M.ts .- e))[2])
        @views Sday[:, :, d] .= 0.5 .* (Ss[:, :, k1] .+ Ss[:, :, k2])
    end
    Jsalt = dropdims(mean(JS .+ Sday .* Jw, dims = 3), dims = 3)   # positive out
    Tm = dropdims(mean(Ts, dims = 3), dims = 3)
    Sm = dropdims(mean(Ss, dims = 3), dims = 3)
    bh, bs = fill(NaN, size(JT)), fill(NaN, size(JT))
    for i in axes(JT, 1), j in axes(JT, 2)
        M.wet[i, j] || continue
        α, β = alpha_beta(Tm[i, j], Sm[i, j])
        bh[i, j] = -g * α * JT[i, j]
        bs[i, j] =  g * β * Jsalt[i, j]
    end
    sw = [M.wet[k] ? -SW[k] : NaN for k in eachindex(SW)]
    return bh, bs, reshape(sw, size(SW)), length(n)
end

# ── B-SOSE ────────────────────────────────────────────────────────────────────────────────────
const TOKENS = ("surfTflx", "surfSflx", "oceFWflx", "oceQsw")
monthly_file(token) = joinpath(BDIR, "$(token)_bsoseI156_2013to2024_monthly.nc")
fiveday_file(token, y) = joinpath(BDIR, "$(token)_bsoseI156_$(y)_5day.nc")

function varname(ds, token)
    for c in (token, uppercase(token), token == "surfTflx" ? "TFLUX" : token == "surfSflx" ? "SFLUX" : token,
              token == "Theta" ? "THETA" : token == "Salt" ? "SALT" : token)
        haskey(ds, c) && return c
    end
    error("no $token in $(path(ds)); keys are $(collect(keys(ds)))")
end

# A monthly stamp sits at (about) the end of the month it averages; a 5-day stamp exactly at the
# end of its 5 days (the two traps tools/mixing_energetics.jl documents).
monthly_window(stamp) = (stop = round(DateTime(stamp), Month); (stop - Month(1), stop))
fiveday_window(stamp) = (DateTime(stamp) - Day(5), DateTime(stamp))

const B = let
    for t in (TOKENS..., "Theta", "Salt")
        isfile(monthly_file(t)) || error("$(monthly_file(t)) not found (set BSOSE_DIR)")
    end
    ds = NCDataset(monthly_file("surfTflx"))
    xc, yc = f64(ds["XC"][:]), f64(ds["YC"][:])
    I = findall(x -> LON[1] <= x <= LON[2], xc); J = findall(y -> LAT[1] <= y <= LAT[2], yc)
    IR, JR = minimum(I):maximum(I), minimum(J):maximum(J)
    wet = f64(ds["Depth"][IR, JR]) .> 0
    mstamps = [DateTime(s) for s in ds["time"][:]]
    close(ds)
    first_day, last_day = M.centres[M.keep[1]], M.centres[M.keep[end]]
    years = year(first_day):year(last_day)
    fiveday = all(isfile(fiveday_file(t, y)) for t in TOKENS for y in years)
    recs = Tuple{String,Int,DateTime,DateTime}[]                  # (file key, record, a, b)
    if fiveday
        for y in years
            s = NCDataset(fiveday_file("surfTflx", y)); st = [DateTime(x) for x in s["time"][:]]; close(s)
            for (k, x) in enumerate(st)
                a, b = fiveday_window(x)
                a >= first_day - Hour(12) && b <= last_day + Hour(12) && push!(recs, ("5day:$y", k, a, b))
            end
        end
    else
        for (k, x) in enumerate(mstamps)
            a, b = monthly_window(x)
            a >= first_day - Day(1) && b <= last_day + Day(1) && push!(recs, ("monthly", k, a, b))
        end
    end
    @info "B-SOSE: $(fiveday ? "5-day" : "monthly (5-day files missing for some year)") fluxes, $(length(recs)) windows"
    (; lon = xc[IR], lat = yc[JR], IR, JR, wet, mstamps, recs, cadence = fiveday ? "5-day" : "monthly")
end

function bsose_2d(path, token, k)
    ds = NCDataset(path); a = f64(ds[varname(ds, token)][B.IR, B.JR, k]); close(ds)
    return a
end
function bsose_top(token, k)
    ds = NCDataset(monthly_file(token)); a = f64(ds[varname(ds, token)][B.IR, B.JR, 1, k]); close(ds)
    return a
end

"""B-SOSE record: (b_heat, b_salt, sw), per cell."""
function bsose_window(key, k, a, b)
    file(t) = key == "monthly" ? monthly_file(t) : fiveday_file(t, parse(Int, split(key, ":")[2]))
    TF, SF, FW, QS = (bsose_2d(file(t), t, k) for t in TOKENS)
    mid = a + (b - a) ÷ 2
    km = findfirst(s -> (w = monthly_window(s); w[1] <= mid < w[2]), B.mstamps)
    km === nothing && error("no monthly Theta/Salt record holds $mid")
    Tt, St = bsose_top("Theta", km), bsose_top("Salt", km)
    bh, bs = fill(NaN, size(TF)), fill(NaN, size(TF))
    for i in axes(TF, 1), j in axes(TF, 2)
        B.wet[i, j] || continue
        α, β = alpha_beta(Tt[i, j], St[i, j])
        bh[i, j] =  g * α * (TF[i, j] - QS[i, j]) / (ρ₀ * cₚ)
        bs[i, j] = -g * β * (SF[i, j] - S0 * FW[i, j]) / ρ₀
    end
    sw = [B.wet[q] ? QS[q] : NaN for q in eachindex(QS)]
    return bh, bs, reshape(sw, size(QS))
end

# ── region means ──────────────────────────────────────────────────────────────────────────────
masks(lon, lat, wet) = [[wet[i, j] && KEEP(lon[i], lat[j]) && r(lon[i], lat[j]) for i in eachindex(lon), j in eachindex(lat)]
                        for (_, r) in REGIONS]
const MASK_M = masks(M.lon, M.lat, M.wet)
const MASK_B = masks(B.lon, B.lat, B.wet)

function regional(A, mask, lat)
    num = den = 0.0
    for i in axes(A, 1), j in axes(A, 2)
        (mask[i, j] && isfinite(A[i, j])) || continue
        w = cosd(lat[j]); num += w * A[i, j]; den += w
    end
    return den == 0 ? NaN : num / den
end

stats(bh, bs, sw, masks_, lat) = [(heat = regional(bh, m, lat), brine = regional(min.(bs, 0.0), m, lat),
                                    fresh = regional(max.(bs, 0.0), m, lat),
                                    total = regional(bh .+ bs, m, lat), sw = regional(sw, m, lat))
                                   for m in masks_]

# Maps are accumulated per season and year: Jun-Sep (winter of year y) and Dec-Mar (summer that
# starts in December of year y). Keyed (season, y) => sums.
season_of(t) = month(t) in 6:9 ? (:winter, year(t)) : month(t) in (12, 1, 2, 3) ? (:summer, month(t) == 12 ? year(t) : year(t) - 1) : nothing
acc = Dict{Any,Any}()
function accumulate!(key, side, fields, w)
    d = get!(acc, (key, side), Dict{Symbol,Any}(:w => 0.0))
    for (name, A) in pairs(fields)
        haskey(d, name) || (d[name] = zeros(size(A)))
        d[name] .+= w .* A
    end
    d[:w] += w
end

# ── run ───────────────────────────────────────────────────────────────────────────────────────
rows = []
for (key, k, a, b) in B.recs
    mw = model_window(a, b)
    mw === nothing && continue
    bhm, bsm, swm, nd = mw
    bhb, bsb, swb = bsose_window(key, k, a, b)
    mid = a + (b - a) ÷ 2
    dt = Dates.value(b - a) / 1000
    push!(rows, (; mid, dt, nd, m = stats(bhm, bsm, swm, MASK_M, M.lat), b = stats(bhb, bsb, swb, MASK_B, B.lat)))
    s = season_of(mid)
    if s !== nothing
        accumulate!(s, :model, (heat = bhm, brine = min.(bsm, 0.0), fresh = max.(bsm, 0.0)), dt)
        accumulate!(s, :bsose, (heat = bhb, brine = min.(bsb, 0.0), fresh = max.(bsb, 0.0)), dt)
    end
end
isempty(rows) && error("no B-SOSE window overlaps the run")
@info "$(length(rows)) windows compared ($(B.cadence)); model days per window $(join(unique(r.nd for r in rows), "/"))"

# Monthly aggregation of the windows (by window midpoint), weighted by window length
months = unique([(year(r.mid), month(r.mid)) for r in rows])
function monthly(side, ireg, field)
    [(rs = [r for r in rows if (year(r.mid), month(r.mid)) == ym];
      w = [r.dt for r in rs]; v = [getfield(getfield(r, side)[ireg], field) for r in rs];
      ok = isfinite.(v); any(ok) ? sum(w[ok] .* v[ok]) / sum(w[ok]) : NaN) for ym in months]
end

# ── tables ────────────────────────────────────────────────────────────────────────────────────
println()
println("="^112)
println("SURFACE BUOYANCY FLUX the ocean received, $RUN (model) against B-SOSE ($(B.cadence) records)")
println("  1e-8 m^2/s^3, cos-lat weighted, outside the $(SPONGE_TRIM)-deg boundary band. NEGATIVE drives convection.")
println("  brine = the salt-gaining part of b_salt, fresh = the freshening part, split per cell per window;")
println("  total = heat + brine + fresh (no shortwave). SW = shortwave into the ocean, W/m^2, shown alone.")
println("="^112)
cell(v) = isnan(v) ? lpad("--", 6) : @sprintf("%6.2f", v)
for (ir, (name, _)) in enumerate(REGIONS)
    @printf("\n  %s\n", name)
    println("  month    |    heat M/B    |   brine M/B    |   fresh M/B    |   total M/B    |  SW (W/m2) M/B")
    q = Dict(f => (monthly(:m, ir, f), monthly(:b, ir, f)) for f in (:heat, :brine, :fresh, :total))
    sw = (monthly(:m, ir, :sw), monthly(:b, ir, :sw))
    for (n, (y, mo)) in enumerate(months)
        @printf("  %4d-%02d  |", y, mo)
        for f in (:heat, :brine, :fresh, :total)
            @printf(" %s %s |", cell(SCALE * q[f][1][n]), cell(SCALE * q[f][2][n]))
        end
        @printf(" %6.1f %6.1f\n", sw[1][n], sw[2][n])
    end
end

println("\n  FREEZING SEASON (Apr-Oct, by window midpoint), -∫B dt in m^2/s^2: positive = buoyancy LOST.")
println("  mixing_energetics' `supplied` for this run, without its solar term. Years the run does not")
println("  cover from April are marked with the month it starts.")
@printf("  %-22s %5s | %15s | %15s | %15s | %15s\n", "region", "year", "heat M/B", "brine M/B", "fresh M/B", "total M/B")
for (ir, (name, _)) in enumerate(REGIONS), y in unique(year(r.mid) for r in rows)
    rs = [r for r in rows if year(r.mid) == y && month(r.mid) in 4:10]
    isempty(rs) && continue
    first_m = minimum(month(r.mid) for r in rs)
    integ(side, f) = -sum(r.dt * getfield(getfield(r, side)[ir], f) for r in rs if isfinite(getfield(getfield(r, side)[ir], f)); init = 0.0)
    @printf("  %-22s %4d%s |", name, y, first_m > 4 ? "*" : " ")
    for f in (:heat, :brine, :fresh, :total)
        @printf(" %7.3f %7.3f |", integ(:m, f), integ(:b, f))
    end
    println(first_m > 4 ? "   *from $(Dates.monthabbr(first_m))" : "")
end

# ── figure: monthly series ────────────────────────────────────────────────────────────────────
let
    x = collect(1:length(months))
    lab = [@sprintf("%d-%02d", y, m) for (y, m) in months]
    step = max(1, cld(length(x), 8))
    nc = length(REGIONS)
    fig = Figure(size = (320 * nc + 80, 900))
    Label(fig[0, 1:nc], "$RUN — surface buoyancy flux the ocean received (red) against B-SOSE (black), 1e-8 m²/s³",
          fontsize = 17, font = :bold)
    for (ir, (name, _)) in enumerate(REGIONS), (rr, f) in enumerate((:heat, :brine, :fresh, :total))
        ax = Axis(fig[rr, ir]; title = rr == 1 ? name : "", ylabel = ir == 1 ? string(f) : "",
                  xticks = (x[1:step:end], lab[1:step:end]), xticklabelrotation = π / 4,
                  xticklabelsvisible = rr == 4)
        hlines!(ax, [0.0]; color = :gray70)
        lines!(ax, x, SCALE .* monthly(:b, ir, f); color = :black)
        lines!(ax, x, SCALE .* monthly(:m, ir, f); color = :firebrick)
    end
    Label(fig[5, 1:nc], "brine: the salt-gaining part of the salt term, fresh: the freshening part, split per cell " *
          "and window before averaging. total = heat + brine + fresh, without the shortwave. Negative drives convection.",
          fontsize = 11)
    f = joinpath(OUTDIR, "$(RUN)_buoyancy_series.png")
    save(f, fig); @info "wrote $f"
end

# ── figures: seasonal maps ────────────────────────────────────────────────────────────────────
# The difference is taken on the model grid by nearest B-SOSE cell: both are 1/6 deg over the
# same box, so this is a lookup, not an interpolation.
const NEAR_I = [argmin(abs.(B.lon .- x)) for x in M.lon]
const NEAR_J = [argmin(abs.(B.lat .- y)) for y in M.lat]
on_model(A) = [A[NEAR_I[i], NEAR_J[j]] for i in eachindex(M.lon), j in eachindex(M.lat)]

for (season, field, title) in ((:winter, :brine, "Jun-Sep brine (salt gain)"), (:summer, :fresh, "Dec-Mar freshening"))
    ys = sort([k[1][2] for k in keys(acc) if k[1][1] == season && k[2] == :model])
    isempty(ys) && continue
    fig = Figure(size = (1700, 330 * length(ys) + 90))
    Label(fig[0, 1:8], "$RUN — $title and heat, 1e-8 m²/s³ (negative drives convection)", fontsize = 17, font = :bold)
    for (row, y) in enumerate(ys)
        dm, db = acc[((season, y), :model)], acc[((season, y), :bsose)]
        Am = SCALE .* dm[field] ./ dm[:w]
        Ab = SCALE .* on_model(db[field] ./ db[:w])
        Hd = SCALE .* (dm[:heat] ./ dm[:w] .- on_model(db[:heat] ./ db[:w]))
        lim = max(1e-3, quantile(filter(isfinite, abs.([vec(Am); vec(Ab)])), 0.98))
        dl  = max(1e-3, quantile(filter(isfinite, abs.(vec(Am .- Ab))), 0.98))
        hl  = max(1e-3, quantile(filter(isfinite, abs.(vec(Hd))), 0.98))
        span = season == :winter ? "Jun-Sep $y" : "Dec $y-Mar $(y + 1)"
        panels = (("model, $span", Am, field == :brine ? Reverse(:Blues) : :Blues, field == :brine ? (-lim, 0.0) : (0.0, lim)),
                  ("B-SOSE, $span", Ab, field == :brine ? Reverse(:Blues) : :Blues, field == :brine ? (-lim, 0.0) : (0.0, lim)),
                  ("model - B-SOSE", Am .- Ab, :balance, (-dl, dl)),
                  ("heat: model - B-SOSE", Hd, :balance, (-hl, hl)))
        for (c, (t, A, cmap, lims)) in enumerate(panels)
            ax = Axis(fig[row, 2c - 1]; title = t, xlabel = row == length(ys) ? "longitude" : "",
                      ylabel = c == 1 ? "latitude" : "")
            hm = heatmap!(ax, M.lon, M.lat, A; colormap = cmap, colorrange = lims, nan_color = :gray80)
            lines!(ax, [BOX[1], BOX[2], BOX[2], BOX[1], BOX[1]], [BOX[3], BOX[3], BOX[4], BOX[4], BOX[3]];
                   color = :black, linestyle = :dash)
            Colorbar(fig[row, 2c], hm)
        end
    end
    Label(fig[length(ys) + 1, 1:8], "Window means of the model's daily fluxes against B-SOSE's $(B.cadence) records, split per " *
          "cell, then averaged over the season. Dashed: the BOX region. Grey: land.", fontsize = 11)
    f = joinpath(OUTDIR, "$(RUN)_buoyancy_$(season)_maps.png")
    save(f, fig); @info "wrote $f"
end
