# ==============================================================================================
# ib_check.jl -- is the run's η missing an inverse-barometer response that B-SOSE's ETAN carries?
#
#     BSOSE_DIR=/storage/scratch1/1/vnguyen480/SOWinWaRP/data julia --project=. tools/ib_check.jl RUN_NAME
#
# WHY. At NumericalEarth 768ed1d the atmosphere's pressure never reaches a NestedModel child, so the run's η has no
# inverse-barometer (IB) part. If B-SOSE was forced with atmospheric loading, its ETAN does:
#     η_IB = -(p - p̄)/(ρ₀ g)      about -1 cm per hPa, p̄ the domain mean
# Then run minus B-SOSE should follow +(p - p̄)/(ρ₀ g): slope 1 against X = (p - p̄)/(ρ₀ g), slope 0 if B-SOSE has no loading.
# In bsose_ggl_hj_pr207_icein_mitbulk_mar_iceobc2 the 47-55S drop Δη equals B-SOSE's at day 0 and is 15% larger
# from day 1 on, which is what dropping B-SOSE's IB part at the first step would do.
#
# THREE TESTS, monthly means, ERA5 msl averaged to calendar months and taken to the run's columns (nearest cell):
#  1. Δη: (run - B-SOSE) of the zonal-mean drop 47S minus 55S, against the same drop in X. IB predicts equal.
#  2. Zonal-mean profiles: per month, the run-minus-B-SOSE zonal mean against X's (domain means removed), 45-65S;
#     regression slope over all months and latitudes.
#  3. B-SOSE alone: per cell, ETAN against -X across the months (each month's domain mean removed); the median slope
#     over cells is ≈1 with loading, ≈0 without. The same for the run as a control (≈0 expected).
# The open faces' 1° (SIDE_DEG) are left out, as in front_position.jl.
#
# ENV  DIR (.), BSOSE_DIR (required), ERA5_DIR ($BSOSE_DIR/era5), SIDE_DEG (1.0), MIN_DAYS (20),
#      STRENGTH_S (-55), STRENGTH_N (-47), LAT_LO (-65), LAT_HI (-45.5) for test 2.
# Reads <ERA5_DIR>/mean_sea_level_pressure_<YYYY>_*.nc (tools/download_era5_ncar.jl: msl, Pa, hourly).
# ==============================================================================================

using NCDatasets, Dates, Printf, Statistics

const RUN        = isempty(ARGS) ? get(ENV, "RUN", "") : ARGS[1]
isempty(RUN) && error("usage: BSOSE_DIR=... julia --project=. tools/ib_check.jl RUN_NAME")
const BSOSE_DIR  = get(ENV, "BSOSE_DIR", "")
isempty(BSOSE_DIR) && error("set BSOSE_DIR")
const ERA5_DIR   = get(ENV, "ERA5_DIR", joinpath(BSOSE_DIR, "era5"))
const DIR        = get(ENV, "DIR", ".")
const SIDE_DEG   = parse(Float64, get(ENV, "SIDE_DEG", "1.0"))
const MIN_DAYS   = parse(Int, get(ENV, "MIN_DAYS", "20"))
const STRENGTH_S = parse(Float64, get(ENV, "STRENGTH_S", "-55"))
const STRENGTH_N = parse(Float64, get(ENV, "STRENGTH_N", "-47"))
const LAT_LO     = parse(Float64, get(ENV, "LAT_LO", "-65"))
const LAT_HI     = parse(Float64, get(ENV, "LAT_HI", "-45.5"))
const ρ₀g        = 1027.0 * 9.81                      # Pa per metre of water: 1 hPa ≈ 0.99 cm

ENV["HDF5_USE_FILE_LOCKING"] = "FALSE"

f64(A) = Float64.(coalesce.(A, NaN))
monthkey(d) = (year(d), month(d))
nanmean(v) = (w = filter(isfinite, v); isempty(w) ? NaN : mean(w))
nanmedian(v) = (w = filter(isfinite, v); isempty(w) ? NaN : median(w))
"Least-squares slope and correlation of y on x over the finite pairs."
function fit(x, y)
    ok = isfinite.(x) .& isfinite.(y)
    count(ok) < 3 && return (NaN, NaN, 0)
    a = x[ok] .- mean(x[ok]); b = y[ok] .- mean(y[ok])
    return (sum(a .* b) / sum(abs2, a), cor(a, b), count(ok))
end
"Zonal mean of each latitude row over the kept columns."
zmean(H) = [nanmean(view(H, :, j)) for j in axes(H, 2)]
demean(H) = H .- nanmean(H)

function main()
    # ── the run's daily η to calendar months (the t = 0 snapshot left out) ───────────────────────
    path = joinpath(DIR, "$(RUN)_free_surface.nc")
    isfile(path) || error("no $path")
    fs = NCDataset(path)
    λ, φ = f64(fs["λ_caa"][:]), f64(fs["φ_aca"][:])
    start = DateTime(string(fs.attrib["start_date"]))
    days = [start + Millisecond(round(Int, 1000s)) for s in f64(fs["time"][:])]
    rd(n) = (A = fs["η"]; f64(ndims(A) == 4 ? A[:, :, 1, n] : A[:, :, n]))
    keepi = findall(x -> x - λ[1] >= SIDE_DEG && λ[end] - x >= SIDE_DEG, λ)
    runm = Dict{Tuple{Int, Int}, Matrix{Float64}}()
    for mk in unique(monthkey.(days[2:end]))
        ns = [n for n in 2:length(days) if monthkey(days[n]) == mk]
        length(ns) < MIN_DAYS && continue
        acc = zeros(length(λ), length(φ))
        for n in ns; acc .+= rd(n); end
        runm[mk] = (acc ./ length(ns))[keepi, :]
    end
    close(fs)
    λk = λ[keepi]
    months = sort(collect(keys(runm)))
    isempty(months) && error("no calendar month with $MIN_DAYS days in $path")

    # ── B-SOSE monthly ETAN on the run's columns ───────────────────────────────────────────────
    bs = NCDataset(joinpath(BSOSE_DIR, "SSH_bsoseI156_2013to2024_monthly.nc"))
    XC, YC = mod.(f64(bs["XC"][:]), 360), f64(bs["YC"][:])
    depth = f64(bs["Depth"][:, :])
    bmk = [monthkey(DateTime(s) - Day(14)) for s in bs["time"][:]]
    ib = [argmin(abs.(XC .- mod(x, 360))) for x in λk]
    jb = [argmin(abs.(YC .- y)) for y in φ]; jr = minimum(jb):maximum(jb)
    wet = depth[ib, jb] .> 0
    bsose(mk) = (n = findfirst(==(mk), bmk); n === nothing ? nothing :
                 (A = f64(bs["ETAN"][:, jr, n])[ib, jb .- first(jr) .+ 1]; A[.!wet] .= NaN; A))

    # ── ERA5 msl, monthly means on the same columns ────────────────────────────────────────────
    function era5_month(mk)
        y, m = mk
        files = filter(f -> startswith(f, "mean_sea_level_pressure_$(y)_") && endswith(f, ".nc"), readdir(ERA5_DIR))
        isempty(files) && error("no mean_sea_level_pressure_$(y)_*.nc in $ERA5_DIR")
        NCDataset(joinpath(ERA5_DIR, files[1])) do e
            lon, lat = mod.(f64(e["longitude"][:]), 360), f64(e["latitude"][:])
            t = e["valid_time"][:]
            sel = findall(d -> year(d) == y && month(d) == m, t)
            isempty(sel) && error("$(files[1]) has no hours in $y-$m")
            P = dropdims(mean(f64(e["msl"].var[:, :, first(sel):last(sel)]); dims = 3); dims = 3)
            ie = [argmin(abs.(lon .- mod(x, 360))) for x in λk]
            je = [argmin(abs.(lat .- y)) for y in φ]
            P[ie, je]
        end
    end

    # ── month by month ─────────────────────────────────────────────────────────────────────────
    jn, js = argmin(abs.(φ .- STRENGTH_N)), argmin(abs.(φ .- STRENGTH_S))
    band = findall(y -> LAT_LO <= y <= LAT_HI, φ)
    rows = NamedTuple[]
    prof_x, prof_d = Float64[], Float64[]
    stackB, stackR, stackX = Matrix{Float64}[], Matrix{Float64}[], Matrix{Float64}[]
    for mk in months
        B = bsose(mk); B === nothing && continue
        R = runm[mk]
        both = isfinite.(B) .& isfinite.(R)
        B[.!both] .= NaN; R[.!both] .= NaN
        P = era5_month(mk); P[.!both] .= NaN
        X = demean(P ./ ρ₀g)                                  # (p - p̄)/(ρ₀ g), p̄ over the compared cells
        zB, zR, zX = zmean(B), zmean(R), zmean(X)
        drift = nanmean(R) - nanmean(B)
        push!(rows, (; mk, ΔB = zB[jn] - zB[js], ΔR = zR[jn] - zR[js], ΔX = zX[jn] - zX[js], drift, pbar = nanmean(P) / 100))
        d = (zR .- nanmean(R)) .- (zB .- nanmean(B))
        append!(prof_x, zX[band]); append!(prof_d, d[band])
        push!(stackB, demean(B)); push!(stackR, demean(R)); push!(stackX, X)
    end
    close(bs)
    isempty(rows) && error("no month in common between the run and B-SOSE")

    println()
    println("="^100)
    println("INVERSE BAROMETER CHECK: $RUN against B-SOSE ETAN and ERA5 msl, $(round(λk[1], digits=1))-$(round(λk[end], digits=1))E")
    println("  X = (p - p̄)/(ρ₀ g), p̄ the month's mean over the compared cells; 1 hPa = $(round(100 / ρ₀g * 100, digits=2)) cm")
    println("  If B-SOSE's ETAN carries the IB response and the run's η does not, run - B-SOSE ≈ +X (slope 1); without loading, slope 0.")
    println("="^100)
    println("\n  1. THE 47-55S DROP (zonal mean at $(STRENGTH_N) minus $(STRENGTH_S)), m")
    @printf("  %-8s | %7s %7s %8s | %8s | %8s  %9s\n", "month", "Δη B", "Δη run", "run - B", "ΔX (IB)", "residual", "p̄ (hPa)")
    for r in rows
        @printf("  %d-%02d  | %7.3f %7.3f %+8.3f | %+8.3f | %+8.3f  %9.1f\n", r.mk[1], r.mk[2], r.ΔB, r.ΔR, r.ΔR - r.ΔB, r.ΔX,
                r.ΔR - r.ΔB - r.ΔX, r.pbar)
    end
    dd = [r.ΔR - r.ΔB for r in rows]; dx = [r.ΔX for r in rows]
    s, c, _ = fit(dx, dd)
    @printf("  mean: run - B-SOSE %+.3f m, ΔX %+.3f m, so IB would explain %.0f%% of it; month-to-month slope %.2f (r = %.2f)\n",
            mean(dd), mean(dx), 100 * mean(dx) / mean(dd), s, c)
    sd, cd, _ = fit([r.pbar for r in rows], [r.drift for r in rows])
    @printf("  domain-mean drift (run - B-SOSE η) against the month's mean msl: %+.2f cm/hPa (r = %.2f; IB with loading in B-SOSE: about +1)\n",
            100 * sd, cd)

    s2, c2, n2 = fit(prof_x, prof_d)
    println("\n  2. ZONAL-MEAN PROFILES $(LAT_LO) to $(LAT_HI), run - B-SOSE against X, domain means removed")
    @printf("     slope %.2f, r = %.2f over %d month-latitudes (1 = the whole meridional difference is IB)\n", s2, c2, n2)

    println("\n  3. PER CELL, η against -X across $(length(stackX)) months (domain means removed); median slope over cells")
    nb = length(stackX)
    cellslope(S) = [begin
                        x = [-stackX[k][i, j] for k in 1:nb]; y = [S[k][i, j] for k in 1:nb]
                        fit(x, y)[1]
                    end for i in axes(stackX[1], 1), j in axes(stackX[1], 2)]
    sB, sR = cellslope(stackB), cellslope(stackR)
    q(v) = (w = filter(isfinite, vec(v)); isempty(w) ? (NaN, NaN, NaN) : (quantile(w, 0.25), median(w), quantile(w, 0.75)))
    @printf("     B-SOSE ETAN: median %.2f (quartiles %.2f to %.2f)   ≈1: B-SOSE carries the IB response\n", q(sB)[2], q(sB)[1], q(sB)[3])
    @printf("     run η:       median %.2f (quartiles %.2f to %.2f)   ≈0 expected: no pressure reaches the child\n", q(sR)[2], q(sR)[1], q(sR)[3])
    println("     Only $(nb) months: eddies add scatter, so read the medians, not single cells.")
end

main()
