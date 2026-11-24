# ==============================================================================================
# compare_eke.jl -- surface eddy kinetic energy of one or more runs against B-SOSE, from sea surface height.
#
#     BSOSE_DIR=/storage/scratch1/1/vnguyen480/SOWinWaRP/data julia --project=. tools/compare_eke.jl RUN [RUN2 ...]
#
# WHY SSH. The runs write daily η; B-SOSE has 5-day SSH. Its velocities are monthly (5-day 3-D velocities are ~18 GB
# per variable and year and not on disk), and a monthly mean has already averaged most eddies away. So every source is
# treated the same way: 5-day means of η, anomalies from that source's own mean over the record, and the geostrophic
#     u' = -(g/f) ∂η'/∂y,   v' = (g/f) ∂η'/∂x,   EKE = ½ ⟨u'² + v'²⟩
# on the run's 1/6° cells (centred differences; a cell needs all four neighbours). Altimetry's EKE is built the same
# way, so these numbers are comparable in kind to AVISO/DUACS maps (not loaded here).
#
# WINDOWS. B-SOSE's 5-day records, [stamp - 5 days, stamp), that the first run covers; a run's window is the mean of
# its daily records inside it (at least 4). Each window's mean over the compared cells is removed first (a run's
# domain-mean η drifts; B-SOSE's mean moves with its own volume), so only the spatial pattern of η counts.
# INVERSE BAROMETER (IB=1, default, as front_position.jl): B-SOSE's ETAN carries -(p - p̄)/(ρ₀ g) and the runs' η does
# not (tools/ib_check.jl); + (p - p̄)/(ρ₀ g) from ERA5 msl over the same 5 days is added back to B-SOSE. IB=0 skips it.
# THE RUN'S η FILE. Runs before _iceobc wrote η Δz / H; as in front_position.jl, if record 1 has no gradient, η is
# rebuilt as file × H / Δz, Δz fitted against B-SOSE's initial monthly ETAN (correlation must exceed 0.95).
# SEA-ICE LOAD (ICE_LOAD=1, default): MITgcm's sea ice presses on the ocean, so B-SOSE's ETAN sits lower under ice by
# its mass, (ρᵢ HEFF + ρₛ HSNOW)/ρ₀ (sIceLoad/rhoConst; ECCO's dynamic SSH adds it back). The runs' η has no such term
# (NumericalEarth's barotropic potential is the atmosphere's pressure only). 5-day swings of coastal ice thickness then
# read as 0.1-1 m of η: in a first pass, 65-70S EKE of 1773 cm²/s² in B-SOSE against 9 in the runs. So B-SOSE gets
# + (910 SIheff + 330 SIhsnow)/1027 back (5-day files; snow left out if there is no SIhsnow file, with a warning), and
# the tool prints how B-SOSE's ETAN follows -load across the windows where there is ice (slope 1 = full loading).
# B-SOSE's rows are interpolated linearly in latitude onto the run's (its spacing is 0.06-0.12°, the run's 1/6°;
# nearest rows gave uneven separations and striped EKE).
# COMPARED CELLS: wet in B-SOSE and in every run, SIDE_DEG (1°) away from the west, east and north faces (the
# relaxation band and the boundary forcing).
#
# WRITES OUTDIR/<RUN>_eke_maps.png (B-SOSE, each run, and log10 run/B-SOSE), OUTDIR/<RUN>_eke_profiles.png (zonal-mean
# EKE by latitude; band EKE window by window). STDOUT: band means (cm²/s²), run/B-SOSE, year 1 and year 2.
# ENV  DIR (.), OUTDIR (plots), BSOSE_DIR (required), ERA5_DIR ($BSOSE_DIR/era5), IB (1), SIDE_DEG (1.0),
#      BANDS (-70,-65,-60,-55,-50,-45), SSH_TOKEN (SSH), MIN_DAYS (4), ICE_LOAD (1), ICE_TOKEN (SIheff), SNOW_TOKEN (SIhsnow).
# ==============================================================================================

using NCDatasets, CairoMakie, Dates, Printf, Statistics

include(joinpath(@__DIR__, "model_grid.jl"))     # file_levels, file_kbot

ENV["HDF5_USE_FILE_LOCKING"] = "FALSE"

const RUNS      = isempty(ARGS) ? split(get(ENV, "RUN", ""), ',') : ARGS
(isempty(RUNS) || isempty(first(RUNS))) && error("usage: BSOSE_DIR=... julia --project=. tools/compare_eke.jl RUN [RUN2 ...]")
const BSOSE_DIR = get(ENV, "BSOSE_DIR", "")
isempty(BSOSE_DIR) && error("set BSOSE_DIR")
const ERA5_DIR  = get(ENV, "ERA5_DIR", joinpath(BSOSE_DIR, "era5"))
const DIR       = get(ENV, "DIR", ".")
const OUTDIR    = get(ENV, "OUTDIR", "plots")
const IB        = get(ENV, "IB", "1") != "0"
const SIDE_DEG  = parse(Float64, get(ENV, "SIDE_DEG", "1.0"))
const EDGES     = parse.(Float64, split(get(ENV, "BANDS", "-70,-65,-60,-55,-50,-45"), ','))
const TOKEN     = get(ENV, "SSH_TOKEN", "SSH")
const MIN_DAYS  = parse(Int, get(ENV, "MIN_DAYS", "4"))
const ICE_LOAD  = get(ENV, "ICE_LOAD", "1") != "0"
const ICE_TOKEN = get(ENV, "ICE_TOKEN", "SIheff")
const SNOW_TOKEN = get(ENV, "SNOW_TOKEN", "SIhsnow")
const ρᵢ, ρₛ, ρ₀ = 910.0, 330.0, 1027.0
const g, Ω, R   = 9.81, 7.2921e-5, 6.371e6
const ρ₀g       = 1027.0 * 9.81

f64(A) = Float64.(coalesce.(A, NaN))
nanmean(v) = (w = filter(isfinite, v); isempty(w) ? NaN : mean(w))

# ── the run's daily η, read one record at a time, rebuilt if the file holds η Δz / H ──────────────────
struct RunEta
    name::String
    ds::NCDataset
    days::Vector{DateTime}
    scale::Matrix{Float64}
    note::String
end

function bsose_monthly_etan(λ, φ, date)
    bs = NCDataset(joinpath(BSOSE_DIR, "SSH_bsoseI156_2013to2024_monthly.nc"))
    XC, YC = mod.(f64(bs["XC"][:]), 360), f64(bs["YC"][:])
    stamps = DateTime.(bs["time"][:])
    n = argmin(abs.(Dates.value.(stamps .- date)))
    ib = [argmin(abs.(XC .- mod(x, 360))) for x in λ]; jb = [argmin(abs.(YC .- y)) for y in φ]
    A = f64(bs["ETAN"][:, :, n])[ib, jb]
    A[f64(bs["Depth"][:, :])[ib, jb] .<= 0] .= NaN
    close(bs)
    return A
end

function open_run(name)
    path = joinpath(DIR, "$(name)_free_surface.nc")
    isfile(path) || error("no $path")
    ds = NCDataset(path)
    start = DateTime(string(ds.attrib["start_date"]))
    days = [start + Millisecond(round(Int, 1000s)) for s in f64(ds["time"][:])]
    λ, φ = f64(ds["λ_caa"][:]), f64(ds["φ_aca"][:])
    η1 = read_eta(ds, 1)
    B0 = bsose_monthly_etan(λ, φ, start)
    jn, js = argmin(abs.(φ .+ 47)), argmin(abs.(φ .+ 55))
    drop(H) = nanmean(H[:, jn]) - nanmean(H[:, js])
    scale = ones(length(λ), length(φ)); note = "η used as written"
    if abs(drop(η1)) < 0.05 * abs(drop(B0))
        gfile = ""
        for suf in ("_eke.nc", "_surface.nc"); f = joinpath(DIR, name * suf); isfile(f) && (gfile = f; break); end
        isempty(gfile) && error("η in $path is flat (η Δz / H?) and no _eke.nc or _surface.nc to read H from")
        Hc = NCDataset(gfile) do gd
            zc, zf, _ = file_levels(gd)
            if haskey(gd, "bottom_height")
                b = f64(gd["bottom_height"][:, :]); [b[i, j] < 0 ? min(-b[i, j], zf[end] - zf[1]) : NaN for i in axes(b, 1), j in axes(b, 2)]
            else
                kb = file_kbot(gd, zc, zf)[1]; [kb[i, j] === nothing ? NaN : zf[end] - zf[kb[i, j]] for i in axes(kb, 1), j in axes(kb, 2)]
            end
        end
        G = η1 .* Hc
        ok = isfinite.(G) .& isfinite.(B0)
        x = G[ok] .- mean(G[ok]); y = B0[ok] .- mean(B0[ok])
        s = sum(x .* y) / sum(abs2, x); r = cor(x, y)
        r < 0.95 && error(@sprintf("η in %s is flat and file × H / Δz does not match B-SOSE's initial ETAN (r = %.3f)", path, r))
        scale = s .* Hc
        note = @sprintf("η REBUILT as file × H / Δz (fitted Δz %.2f m, r = %.4f)", 1 / s, r)
    end
    return RunEta(name, ds, days, scale, note), λ, φ
end

read_eta(ds, n) = (A = ds["η"]; f64(ndims(A) == 4 ? A[:, :, 1, n] : A[:, :, n]))

"The run's mean η over [t0, t1), from its daily records (the t = 0 snapshot left out); nothing if fewer than MIN_DAYS."
function run_window(r::RunEta, t0, t1)
    ns = [n for n in 2:length(r.days) if t0 <= r.days[n] < t1]
    length(ns) < MIN_DAYS && return nothing
    acc = zeros(size(r.scale))
    for n in ns; acc .+= read_eta(r.ds, n); end
    H = (acc ./ length(ns)) .* r.scale
    H[.!isfinite.(H)] .= NaN
    return H
end

# ── B-SOSE 5-day ETAN on the run's cells, and ERA5 msl over the same window ──────────────────────────
function bsose_5day_records(years)
    recs = Tuple{String, String, Int, DateTime, DateTime}[]
    for y in years
        hits = filter(f -> lowercase(f) == lowercase("$(TOKEN)_bsoseI156_$(y)_5day.nc"), readdir(BSOSE_DIR))
        isempty(hits) && (@warn "no $(TOKEN)_bsoseI156_$(y)_5day.nc in $BSOSE_DIR: $y left out"; continue)
        p = joinpath(BSOSE_DIR, first(hits))
        NCDataset(p) do d
            v = first(k for k in ("ETAN", "SSH", "Eta") if haskey(d, k))
            for (n, s) in enumerate(d["time"][:])
                push!(recs, (p, v, n, DateTime(s) - Day(5), DateTime(s)))
            end
        end
    end
    isempty(recs) && error("no B-SOSE 5-day SSH ($(TOKEN)_bsoseI156_<year>_5day.nc) in $BSOSE_DIR for $(collect(years))")
    return recs
end

"Stamp => (path, variable, record) for a B-SOSE 5-day token, over `years`; empty if no file."
function bsose_5day_index(token, years)
    idx = Dict{DateTime, Tuple{String, String, Int}}()
    for y in years
        hits = filter(f -> lowercase(f) == lowercase("$(token)_bsoseI156_$(y)_5day.nc"), readdir(BSOSE_DIR))
        isempty(hits) && continue
        p = joinpath(BSOSE_DIR, first(hits))
        NCDataset(p) do d
            v = first(k for k in keys(d) if ndims(d[k]) >= 3)
            for (n, s) in enumerate(d["time"][:]); idx[DateTime(s)] = (p, v, n); end
        end
    end
    return idx
end

function msl_window(t0, t1, λ, φ)
    y = year(t0)
    fs = filter(f -> startswith(f, "mean_sea_level_pressure_$(y)_") && endswith(f, ".nc"), readdir(ERA5_DIR))
    isempty(fs) && return nothing
    NCDataset(joinpath(ERA5_DIR, first(fs))) do e
        lon, lat = mod.(f64(e["longitude"][:]), 360), f64(e["latitude"][:])
        t = e["valid_time"][:]
        sel = findall(d -> t0 <= d < t1, t)
        (isempty(sel) || t[last(sel)] - t[first(sel)] < Day(4)) && return nothing   # a window across New Year: skip IB
        P = dropdims(mean(f64(e["msl"].var[:, :, first(sel):last(sel)]); dims = 3); dims = 3)
        P[[argmin(abs.(lon .- mod(x, 360))) for x in λ], [argmin(abs.(lat .- yy)) for yy in φ]]
    end
end

# ── geostrophic EKE pieces on the run's grid ────────────────────────────────────────────────────────
"Geostrophic (u', v') at cell centres from an anomaly field η' (lon × lat, NaN on land); NaN where a neighbour is missing."
function geostrophic(ηp, λ, φ)
    nx, ny = size(ηp)
    u = fill(NaN, nx, ny); v = fill(NaN, nx, ny)
    Δλ, Δφ = deg2rad(λ[2] - λ[1]), deg2rad(φ[2] - φ[1])
    for j in 2:ny-1, i in 2:nx-1
        f = 2Ω * sind(φ[j])
        a, b, c, d = ηp[i+1, j], ηp[i-1, j], ηp[i, j+1], ηp[i, j-1]
        (isfinite(a) && isfinite(b) && isfinite(c) && isfinite(d)) || continue
        u[i, j] = -(g / f) * (c - d) / (2R * Δφ)
        v[i, j] =  (g / f) * (a - b) / (2R * cosd(φ[j]) * Δλ)
    end
    return u, v
end

function main()
    runs = RunEta[]; λ = φ = nothing
    for name in RUNS
        r, λr, φr = open_run(String(name))
        λ === nothing ? (λ = λr; φ = φr) : (size(λr) == size(λ) && size(φr) == size(φ) || error("$name is on a different grid"))
        push!(runs, r)
        @info "$(r.name): $(r.note); $(r.days[1]) to $(r.days[end])"
    end
    first_day, last_day = runs[1].days[2], runs[1].days[end]
    recs = filter(r -> r[4] >= first_day - Day(1) && r[5] <= last_day + Day(1), bsose_5day_records(year(first_day):year(last_day)))
    IB && !isdir(ERA5_DIR) && error("IB=1 needs ERA5 msl in $ERA5_DIR; IB=0 skips the inverse-barometer correction")

    # B-SOSE's columns: its longitudes are the run's cell centres; its rows interpolated linearly to the run's latitudes
    # (coordinates and seafloor from the monthly SSH file, the same 2160 x 588 grid as the 5-day files)
    XC, YC, Depth = NCDataset(joinpath(BSOSE_DIR, "SSH_bsoseI156_2013to2024_monthly.nc")) do d
        mod.(f64(d["XC"][:]), 360), f64(d["YC"][:]), f64(d["Depth"][:, :])
    end
    ib = [argmin(abs.(XC .- mod(x, 360))) for x in λ]
    j0 = [clamp(searchsortedlast(YC, y), 1, length(YC) - 1) for y in φ]
    wj = [clamp((φ[k] - YC[j0[k]]) / (YC[j0[k]+1] - YC[j0[k]]), 0.0, 1.0) for k in eachindex(φ)]
    jr = minimum(j0):maximum(j0) + 1
    onrun(S) = (a = S[ib, j0 .- first(jr) .+ 1]; b = S[ib, j0 .- first(jr) .+ 2]; a .* (1 .- wj') .+ b .* wj')   # S: rows jr
    wetB = (Depth[ib, j0] .> 0) .& (Depth[ib, j0 .+ 1] .> 0)
    years = year(first_day):year(last_day)
    ice_idx  = ICE_LOAD ? bsose_5day_index(ICE_TOKEN, years) : Dict{DateTime, Tuple{String, String, Int}}()
    snow_idx = ICE_LOAD ? bsose_5day_index(SNOW_TOKEN, years) : Dict{DateTime, Tuple{String, String, Int}}()
    ICE_LOAD && isempty(ice_idx) && error("ICE_LOAD=1 needs B-SOSE's 5-day $(ICE_TOKEN)_bsoseI156_<year>_5day.nc in $BSOSE_DIR (ICE_LOAD=0 skips it)")
    ICE_LOAD && isempty(snow_idx) && @warn "no $(SNOW_TOKEN)_bsoseI156_<year>_5day.nc: the sea-ice load added back to B-SOSE leaves out snow"
    read2d(path, var, n) = NCDataset(path) do d
        A = d[var]; onrun(f64(ndims(A) == 4 ? A[:, jr, 1, n] : A[:, jr, n]))
    end
    loads = Matrix{Float64}[]; etans = Matrix{Float64}[]; nload = 0

    keep = [x - λ[1] >= SIDE_DEG && λ[end] - x >= SIDE_DEG for x in λ] .* [y <= φ[end] - SIDE_DEG for y in φ]'
    nr = length(runs)
    srcs = vcat(["B-SOSE"], [r.name for r in runs])
    Hs = [Matrix{Float64}[] for _ in 1:nr+1]
    wins = Tuple{DateTime, DateTime}[]
    nib = 0
    for (p, var, n, t0, t1) in recs
        rw = [run_window(r, t0, t1) for r in runs]
        any(isnothing, rw) && continue
        B = read2d(p, var, n)
        B[.!wetB] .= NaN
        if IB
            P = msl_window(t0, t1, λ, φ)
            if P !== nothing
                X = P ./ ρ₀g; X[.!isfinite.(B)] .= NaN
                B .+= X .- nanmean(X); nib += 1
            end
        end
        if ICE_LOAD
            stamp = t1
            if haskey(ice_idx, stamp)
                L = ρᵢ .* coalesce.(read2d(ice_idx[stamp]...), 0.0)
                haskey(snow_idx, stamp) && (L .+= ρₛ .* coalesce.(read2d(snow_idx[stamp]...), 0.0))
                L = replace(L ./ ρ₀, NaN => 0.0)
                push!(etans, copy(B)); push!(loads, L)
                B .+= L; nload += 1
            end
        end
        push!(wins, (t0, t1)); push!(Hs[1], B)
        for k in 1:nr; push!(Hs[k+1], rw[k]); end
    end
    nw = length(wins)
    nw < 10 && error("only $nw 5-day windows in common between B-SOSE and the runs")
    @info "$nw 5-day windows, $(wins[1][1]) to $(wins[end][2]); inverse barometer removed from B-SOSE in $nib of them$(IB ? "" : " (IB=0)"), sea-ice load added back in $nload$(ICE_LOAD ? "" : " (ICE_LOAD=0)")"

    # compared cells: finite in every source and every window, away from the open faces
    mask = copy(keep)
    for H in Hs, A in H; mask .&= isfinite.(A); end
    @info "$(count(mask)) compared cells"

    # does B-SOSE's ETAN carry the ice load? Per cell with ice, ETAN' against -load' across the windows
    load_fit = nothing
    if length(loads) >= 10
        Lm = sum(loads) ./ length(loads)
        icy = mask .& (Lm .> 0.2 * ρᵢ / ρ₀)                            # mean ice thicker than 0.2 m
        dm(X) = (Y = copy(X); Y[.!mask] .= NaN; Y .- nanmean(Y))
        Ea = [dm(E) for E in etans]; La = [L .- nanmean(ifelse.(mask, L, NaN)) for L in loads]
        Em = sum(Ea) ./ length(Ea); Lam = sum(La) ./ length(La)
        slopes = Float64[]
        for c in findall(icy)
            x = [-(L[c] - Lam[c]) for L in La]; y = [E[c] - Em[c] for E in Ea]
            sum(abs2, x) > 0 && push!(slopes, sum(x .* y) / sum(abs2, x))
        end
        isempty(slopes) || (load_fit = (count(icy), quantile(slopes, 0.25), median(slopes), quantile(slopes, 0.75)))
    end

    # anomalies: each window's mean over the compared cells removed, then each source's time mean
    eke = Matrix{Float64}[]; series = Matrix{Float64}[]          # per source: EKE map; band × window EKE
    bands = [(EDGES[b], EDGES[b+1]) for b in 1:length(EDGES)-1]
    inband(b) = [bands[b][1] <= y < bands[b][2] for y in φ]
    wlat = cosd.(φ)
    for H in Hs
        A = [(X = copy(Hk); X[.!mask] .= NaN; X .- nanmean(X)) for Hk in H]
        M = sum(A) ./ nw
        E = zeros(size(M)); cnt = zeros(size(M)); S = fill(NaN, length(bands), nw)
        for (w, Ak) in enumerate(A)
            u, v = geostrophic(Ak .- M, λ, φ)
            e = 0.5 .* (u .^ 2 .+ v .^ 2)
            ok = isfinite.(e) .& mask
            E[ok] .+= e[ok]; cnt[ok] .+= 1
            for b in eachindex(bands)
                sel = ok .& inband(b)'
                ww = (sel .* wlat')
                S[b, w] = sum(ww) > 0 ? sum(e[sel] .* ww[sel]) / sum(ww[sel]) : NaN
            end
        end
        E = ifelse.(cnt .> 0.9nw, E ./ max.(cnt, 1), NaN)
        push!(eke, E .* 1e4); push!(series, S .* 1e4)                 # cm²/s²
    end
    close.(getfield.(runs, :ds))

    # ── stdout ───────────────────────────────────────────────────────────────────────────────────
    bandmean(E, b) = (sel = isfinite.(E) .& inband(b)'; w = (sel .* wlat')[sel]; isempty(w) ? NaN : sum(E[sel] .* w) / sum(w))
    y1 = [w for w in 1:nw if wins[w][1] < wins[1][1] + Day(365)]; y2 = setdiff(1:nw, y1)
    bar = "="^(30 + 26nr)
    println("\n", bar)
    println("SURFACE GEOSTROPHIC EKE from 5-day mean SSH, cm²/s², $(round(λ[1], digits=1))-$(round(λ[end], digits=1))E, $nw windows ",
            "($(Date(wins[1][1])) to $(Date(wins[end][2]))); $(count(mask)) cells, $(SIDE_DEG)° off the open faces")
    println("  B-SOSE: 5-day ETAN", IB ? " + (p - p̄)/(ρ₀ g) (inverse barometer removed)" : "",
            ICE_LOAD ? " + (ρᵢ SIheff + ρₛ SIhsnow)/ρ₀ (sea-ice load added back)" : "", IB || ICE_LOAD ? "" : " as is")
    load_fit === nothing || @printf("  ice load check: in %d cells with ice, B-SOSE's raw ETAN' against -load': median slope %.2f (quartiles %.2f to %.2f); 1 = ETAN carries the full load\n",
                                    load_fit[1], load_fit[3], load_fit[2], load_fit[4])
    println(bar)
    @printf("  %-9s | %9s", "band", "B-SOSE")
    for r in runs; @printf(" | %-22s", first(r.name, 22)); end
    println()
    @printf("  %-9s | %9s", "", "")
    for _ in runs; @printf(" | %9s %12s", "EKE", "run/B-SOSE"); end
    println()
    for b in eachindex(bands)
        eb = bandmean(eke[1], b)
        @printf("  %3.0f-%3.0fS | %9.1f", -bands[b][2], -bands[b][1], eb)
        for k in 1:nr; er = bandmean(eke[k+1], b); @printf(" | %9.1f %12.2f", er, er / eb); end
        println()
    end
    if !isempty(y2)
        println("\n  run/B-SOSE by year (band EKE, mean over the year's windows)")
        @printf("  %-9s", "band")
        for r in runs; @printf(" | %-22s", first(r.name, 22)); end
        println()
        for b in eachindex(bands)
            @printf("  %3.0f-%3.0fS", -bands[b][2], -bands[b][1])
            for k in 1:nr
                r1 = nanmean(series[k+1][b, y1]) / nanmean(series[1][b, y1]); r2 = nanmean(series[k+1][b, y2]) / nanmean(series[1][b, y2])
                @printf(" | yr1 %5.2f  yr2 %5.2f   ", r1, r2)
            end
            println()
        end
    end
    println("\n  (band EKE by window is per-window: a window's EKE uses each source's own mean over all windows)\n")

    # ── figures ──────────────────────────────────────────────────────────────────────────────────
    mkpath(OUTDIR)
    tag = runs[end].name
    vals = filter(isfinite, reduce(vcat, vec.(eke)))
    hi = quantile(vals, 0.99); lo = max(quantile(vals, 0.02), hi / 1000)
    fig = Figure(size = (1000, 300 * (2nr + 1) + 60), fontsize = 13)
    Label(fig[0, 1:2], "Surface geostrophic EKE from 5-day SSH (cm²/s², log scale) and run/B-SOSE"; font = :bold)
    row = 1
    for (k, E) in enumerate(eke)
        ax = Axis(fig[row, 1]; title = srcs[k], ylabel = "latitude")
        hm = heatmap!(ax, λ, φ, log10.(clamp.(E, lo, hi)); colormap = :thermal, colorrange = (log10(lo), log10(hi)),
                      nan_color = :gray85)
        k == 1 && Colorbar(fig[row, 2], hm; label = "log10 EKE (cm²/s²)")
        row += 1
    end
    for k in 1:nr
        ax = Axis(fig[row, 1]; title = "log10 ($(srcs[k+1]) / B-SOSE): +1 = 10x B-SOSE", ylabel = "latitude")
        Rt = log10.(eke[k+1] ./ eke[1])
        hm = heatmap!(ax, λ, φ, clamp.(Rt, -1, 1); colormap = :balance, colorrange = (-1, 1), nan_color = :gray85)
        k == 1 && Colorbar(fig[row, 2], hm; label = "log10 ratio")
        row += 1
    end
    out1 = joinpath(OUTDIR, "$(tag)_eke_maps.png"); save(out1, fig); @info "wrote $out1"

    fig2 = Figure(size = (1300, 380 + 220 * length(bands)), fontsize = 13)
    ax = Axis(fig2[1, 1:2]; title = "zonal-mean EKE, compared cells", xlabel = "latitude", ylabel = "EKE (cm²/s²)", yscale = log10)
    cols = vcat([:black], Makie.wong_colors()[1:nr])
    for (k, E) in enumerate(eke)
        z = [(s = isfinite.(E[:, j]); any(s) ? mean(E[s, j]) : NaN) for j in eachindex(φ)]
        ok = isfinite.(z) .& (z .> 0)
        lines!(ax, φ[ok], z[ok]; color = cols[k], linewidth = 2, linestyle = k == 1 ? :dash : :solid, label = srcs[k])
    end
    axislegend(ax; position = :lt)
    tx = [Dates.value(t0 + Hour(60) - wins[1][1]) / 86_400_000 for (t0, _) in wins]
    for b in eachindex(bands)
        axb = Axis(fig2[b+1, 1:2]; title = @sprintf("%.0f-%.0fS: band EKE, each 5-day window", -bands[b][2], -bands[b][1]),
                   ylabel = "cm²/s²", xlabel = b == length(bands) ? "days since $(Date(wins[1][1]))" : "")
        for k in eachindex(series)
            lines!(axb, tx, series[k][b, :]; color = cols[k], linewidth = 1.5, linestyle = k == 1 ? :dash : :solid, label = srcs[k])
        end
        b == 1 && axislegend(axb; position = :lt)
    end
    out2 = joinpath(OUTDIR, "$(tag)_eke_profiles.png"); save(out2, fig2); @info "wrote $out2"
end

main()
