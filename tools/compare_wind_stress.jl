# ==============================================================================================
# compare_wind_stress.jl -- Large & Yeager (2004) wind stress from the ERA5 the runs are forced with, against
# B-SOSE's own open-water stress (oceTAUX/Y), by latitude band and month, over B-SOSE's open water.
#
#     BSOSE_DIR=/storage/scratch1/1/vnguyen480/SOWinWaRP/data julia --project=. tools/compare_wind_stress.jl
#
# WHY. _icein_mitbulk (Large & Yeager over open water) cut the summer open-water mixed layer excess at 55-45S
# from ~1.5x B-SOSE's to ~1.3x, and model_surface_buoyancy shows its summer buoyancy loss there is SMALLER than
# B-SOSE's, so what is left is the stress. The run does not write its stress, so this rebuilds it offline from
# the same ERA5 hours: if it is still above oceTAU, the winds or the air density are the difference; if it
# matches, the difference is in the coupling (currents, ice) or in the run itself.
#
# THE STRESS, hourly, on ERA5's 0.25° grid: τ = ρa Cd(U10) U10 (u10, v10), with NumericalEarth's
# PolynomialNeutralDragCoefficient (L&Y: Cd = (2.7/U + 0.142 + U/13.09 - 3.14807e-10 U^6) 1e-3, U >= 0.5 m/s,
# 2.34e-3 above 33 m/s) -- NEUTRAL, no stability correction, and the absolute wind (no ocean current). Two air
# densities: ρa from ERA5's msl, t2m and d2m (ideal gas, virtual temperature), as NumericalEarth's coupler
# computes it, and a constant 1.2 kg/m³ (MITgcm EXF's atmrho default, as far as we know; i156's namelist is not
# public). Averaged over each of B-SOSE's 5-day records (the hours ending in (stamp - 5 days, stamp]), then each
# B-SOSE cell takes its nearest ERA5 cell.
#
# COMPARED: the magnitude of the 5-day mean stress vector, |<τ>|, both sides (B-SOSE's records are 5-day means
# of τx and τy, so this is the like-with-like quantity, not <|τ|>), cos-lat weighted over B-SOSE's wet cells with
# 5-day SIArea < ICE_MIN, by band; monthly = the mean of the records whose centre falls in the month. B-SOSE's
# τx sits on u-points, τy on v-points, half a cell off the centres: ignored for band means.
#
# PRINTS per band and month: open-water cells, B-SOSE |<τ>| (N/m²), and the ratios ERA5-L&Y / B-SOSE with ρa
# from ERA5 and with 1.2. WRITES OUTDIR/wind_stress_bsose_era5.png.
# LOGIN-NODE SAFE: NCDatasets and CairoMakie; ERA5 is read one 5-day window at a time (~70 MB).
# ENV: BSOSE_DIR (required: B-SOSE 5-day oceTAUX, oceTAUY, SIArea, Theta for the wet mask), ERA5_DIR (default
#      BSOSE_DIR/era5), START (2014-06-01), END (2016-06-01), BANDS (-70,-65,-60,-55,-50,-45), ICE_MIN (0.15),
#      OUTDIR (plots)
# ==============================================================================================

using NCDatasets, CairoMakie, Dates, Printf, Statistics

ENV["HDF5_USE_FILE_LOCKING"] = "FALSE"
ENV["SOURCE"] = "5day"                              # B-SOSE's stress is compared record by record, on its 5-day files
include(joinpath(@__DIR__, "bsose_windows.jl"))   # bsose_path, bsose_data_var

const BSOSE_DIR = get(ENV, "BSOSE_DIR", "")
isempty(BSOSE_DIR) && error("set BSOSE_DIR to the directory with B-SOSE's 5-day oceTAUX, oceTAUY, SIArea and Theta files")
const ERA5_DIR = get(ENV, "ERA5_DIR", joinpath(BSOSE_DIR, "era5"))
const START    = DateTime(get(ENV, "START", "2014-06-01"))
const STOP     = DateTime(get(ENV, "END", "2016-06-01"))
const EDGES    = parse.(Float64, split(get(ENV, "BANDS", "-70,-65,-60,-55,-50,-45"), ','))
const ICE_MIN  = parse(Float64, get(ENV, "ICE_MIN", "0.15"))
const OUTDIR   = get(ENV, "OUTDIR", "plots")
const BOX      = (90.0, 150.0, -70.0, -45.0)
const RHO_EXF  = 1.2

f64(A) = Float64.(coalesce.(A, NaN))
band_name(b) = @sprintf("%.0f-%.0fS", -EDGES[b], -EDGES[b+1])

# NumericalEarth's PolynomialNeutralDragCoefficient, defaults (768ed1d)
neutral_cd(U) = (U = max(U, 0.5); U < 33 ? (2.7 / U + 0.142 + U / 13.09 - 3.14807e-10 * U^6) / 1000 : 2.34e-3)

"Moist air density (kg/m³) from pressure (Pa), temperature and dewpoint (K)."
function air_density(p, T, Td)
    Tdc = Td - 273.15
    e = 611.2 * exp(17.67 * Tdc / (Tdc + 243.5))      # vapour pressure, Pa (Bolton 1980)
    q = 0.622 * e / (p - 0.378 * e)
    return p / (287.04 * T * (1 + 0.608 * q))
end

"The ERA5 file for `short` (u10, v10, t2m, d2m, msl) and year y in ERA5_DIR, found by opening the candidates."
function era5_file(short, y)
    for f in sort(readdir(ERA5_DIR; join = true))
        (endswith(f, ".nc") && occursin("_$(y)_", basename(f))) || continue
        NCDataset(f) do ds; haskey(ds, short) end && return f
    end
    error("no ERA5 file with $short for $y in $ERA5_DIR (tools/download_era5_ncar.jl writes them)")
end

function main()
    years = year(START):year(STOP - Millisecond(1))

    # ── B-SOSE: grid, box, wet mask, records ────────────────────────────────────────────────
    tpath = bsose_path(BSOSE_DIR, "Theta", first(years)); isempty(tpath) && error("no 5-day Theta for $(first(years)) in $BSOSE_DIR")
    XC, YC, wet0 = NCDataset(tpath) do ds
        xc = f64(ds["XC"][:]); yc = f64(ds["YC"][:])
        xc = ndims(xc) == 1 ? xc : xc[:, 1]; yc = ndims(yc) == 1 ? yc : yc[1, :]
        h = haskey(ds, "hFacC") ? f64(ds["hFacC"][:, :, 1]) : f64(ds["maskC"][:, :, 1])
        xc, yc, h
    end
    ir = findall(x -> BOX[1] <= x <= BOX[2], XC); jr = findall(y -> BOX[3] <= y <= BOX[4], YC)
    ir, jr = first(ir):last(ir), first(jr):last(jr)
    λ, φ = XC[ir], YC[jr]
    wet = wet0[ir, jr] .> 0
    nb = length(EDGES) - 1
    band_of = [findfirst(b -> EDGES[b] <= y < EDGES[b+1], 1:nb) for y in φ]
    wlat = cosd.(φ)

    recs = Dict(t => bsose_records(BSOSE_DIR, t, n, years) for (t, n) in
                (("oceTAUX", ("oceTAUX",)), ("oceTAUY", ("oceTAUY",)), ("SIArea", ("SIarea", "SIArea", "SIAREA"))))
    stamps = [r.t1 for r in recs["oceTAUX"] if START < r.t1 - Hour(60) <= STOP]
    @info @sprintf("B-SOSE box %d x %d (%.2f-%.2fE, %.2f-%.2fS); %d 5-day records %s to %s",
                   length(ir), length(jr), λ[1], λ[end], -φ[1], -φ[end], length(stamps), first(stamps), last(stamps))

    read2d(t, s) = (r = only(filter(r -> r.t1 == s, recs[t]));
                    NCDataset(r.path) do ds; v = ds[r.var]; f64(ndims(v) == 4 ? v[:, jr, 1, r.n] : v[:, jr, r.n])[ir, :] end)

    # ── ERA5: grid and the nearest ERA5 cell for each B-SOSE cell ───────────────────────────
    elon, elat = NCDataset(era5_file("u10", first(years))) do ds
        lon = f64(ds[haskey(ds, "longitude") ? "longitude" : "lon"][:]); lat = f64(ds[haskey(ds, "latitude") ? "latitude" : "lat"][:])
        lon, lat
    end
    elon = mod.(elon, 360)
    ie = [argmin(abs.(elon .- mod(x, 360))) for x in λ]; je = [argmin(abs.(elat .- y)) for y in φ]
    maximum(abs.(elon[ie] .- mod.(λ, 360))) < 0.2 && maximum(abs.(elat[je] .- φ)) < 0.2 ||
        error("ERA5's grid does not cover the box: nearest cells up to $(maximum(abs.(elat[je] .- φ)))° away")

    handles = Dict{Tuple{String, Int}, NCDataset}()
    era5(short, y) = get!(() -> NCDataset(era5_file(short, y)), handles, (short, y))
    etimes = Dict(y => DateTime.(era5("u10", y)["time"][:]) for y in years)
    for y in years
        d = dimnames(era5("u10", y)["u10"])
        length(d) == 3 && occursin("time", lowercase(d[3])) || error("ERA5 u10 for $y is $(d), expected (lon, lat, time)")
    end

    "ERA5's 5-day mean τx, τy (two air densities) over the hours ending in (s - 5 days, s], on ERA5's grid."
    function era5_stress(s)
        acc = nothing; n = 0
        for y in unique((year(s - Day(5) + Hour(1)), year(s)))
            y in years || continue
            idx = findall(t -> s - Day(5) < t <= s, etimes[y]); isempty(idx) && continue
            idx == first(idx):last(idx) || error("ERA5 $y hours are not consecutive around $s")
            idx = first(idx):last(idx)                 # one contiguous read, not hour by hour
            u = f64(era5("u10", y)["u10"][:, :, idx]); v = f64(era5("v10", y)["v10"][:, :, idx])
            ρ = air_density.(f64(era5("msl", y)["msl"][:, :, idx]), f64(era5("t2m", y)["t2m"][:, :, idx]),
                             f64(era5("d2m", y)["d2m"][:, :, idx]))
            U = sqrt.(u .^ 2 .+ v .^ 2); k = neutral_cd.(U) .* U
            terms = (dropdims(sum(ρ .* k .* u; dims = 3); dims = 3), dropdims(sum(ρ .* k .* v; dims = 3); dims = 3),
                     dropdims(sum(RHO_EXF .* k .* u; dims = 3); dims = 3), dropdims(sum(RHO_EXF .* k .* v; dims = 3); dims = 3),
                     dropdims(sum(ρ; dims = 3); dims = 3))
            acc = acc === nothing ? collect(terms) : acc .+ collect(terms); n += length(idx)
        end
        n < 100 && return nothing                     # 120 hours expected
        return map(a -> a ./ n, acc), n
    end

    # ── per record ──────────────────────────────────────────────────────────────────────────
    cols = (:cells, :bsose, :era5, :era5_12, :rho)
    rec = Dict(c => fill(NaN, nb, length(stamps)) for c in cols)
    for (w, s) in enumerate(stamps)
        e = era5_stress(s)
        e === nothing && (@warn "$(s): ERA5 hours missing, skipped"; continue)
        (τx, τy, τx12, τy12, ρm), _ = e
        bx, by, A = read2d("oceTAUX", s), read2d("oceTAUY", s), read2d("SIArea", s)
        for b in 1:nb
            cells = [(i, j) for i in eachindex(λ), j in eachindex(φ) if band_of[j] == b && wet[i, j] &&
                                                                       isfinite(A[i, j]) && A[i, j] < ICE_MIN &&
                                                                       isfinite(bx[i, j]) && isfinite(by[i, j])]
            isempty(cells) && continue
            W = [wlat[j] for (i, j) in cells]
            wm(f) = sum(W .* [f(i, j) for (i, j) in cells]) / sum(W)
            rec[:cells][b, w]   = length(cells)
            rec[:bsose][b, w]   = wm((i, j) -> hypot(bx[i, j], by[i, j]))
            rec[:era5][b, w]    = wm((i, j) -> hypot(τx[ie[i], je[j]], τy[ie[i], je[j]]))
            rec[:era5_12][b, w] = wm((i, j) -> hypot(τx12[ie[i], je[j]], τy12[ie[i], je[j]]))
            rec[:rho][b, w]     = wm((i, j) -> ρm[ie[i], je[j]])
        end
        w % 10 == 0 && @info "record $w/$(length(stamps)) ($(Date(s))) done"
    end
    foreach(close, values(handles))

    # ── monthly: records by centre month ────────────────────────────────────────────────────
    centre = [s - Hour(60) for s in stamps]
    months = unique([(year(c), month(c)) for c in centre])
    mon = Dict(c => fill(NaN, nb, length(months)) for c in cols)
    for (m, ym) in enumerate(months), b in 1:nb, c in cols
        v = [rec[c][b, w] for w in eachindex(stamps) if (year(centre[w]), month(centre[w])) == ym && isfinite(rec[c][b, w])]
        isempty(v) || (mon[c][b, m] = mean(v))
    end

    bar = "="^112
    println("\n", bar)
    println("OPEN-WATER WIND STRESS: Large & Yeager (neutral, absolute wind) from ERA5 against B-SOSE's oceTAU, $(Date(START)) to $(Date(STOP))")
    println("  |<τ>| of each 5-day record, cos-lat weighted over B-SOSE's wet cells with SIArea < $ICE_MIN, then the month's records averaged")
    println("  ratio ρ: ERA5 air density (msl, t2m, d2m); ratio 1.2: constant 1.2 kg/m³ (EXF atmrho)")
    println(bar)
    for b in 1:nb
        all(isnan, mon[:cells][b, :]) && continue
        @printf("\n  %s\n  month    |  cells | B-SOSE N/m² | ERA5 ρ N/m²  ratio ρ  ratio 1.2 | mean ρa\n", band_name(b))
        for (m, (y, mo)) in enumerate(months)
            isnan(mon[:cells][b, m]) && continue
            @printf("  %04d-%02d  | %6.0f |      %6.3f |      %6.3f    %5.2f      %5.2f |  %5.3f\n", y, mo,
                    mon[:cells][b, m], mon[:bsose][b, m], mon[:era5][b, m],
                    mon[:era5][b, m] / mon[:bsose][b, m], mon[:era5_12][b, m] / mon[:bsose][b, m], mon[:rho][b, m])
        end
        ok = filter(m -> isfinite(mon[:bsose][b, m]), eachindex(months))
        sm = filter(m -> months[m][2] in (12, 1, 2), ok); wi = filter(m -> months[m][2] in (6, 7, 8), ok)
        for (lab, sel) in (("Dec-Feb", sm), ("Jun-Aug", wi))
            isempty(sel) && continue
            @printf("  %s mean ratio: ρ %.2f, 1.2 %.2f\n", lab, mean(mon[:era5][b, sel] ./ mon[:bsose][b, sel]),
                    mean(mon[:era5_12][b, sel] ./ mon[:bsose][b, sel]))
        end
    end
    println()

    # ── figure ──────────────────────────────────────────────────────────────────────────────
    mkpath(OUTDIR)
    bands = [b for b in 1:nb if !all(isnan, mon[:cells][b, :])]
    x = [Dates.value(Date(y, mo, 15) - Date(START)) for (y, mo) in months]
    ticks = (x[1:3:end], [@sprintf("%04d-%02d", y, mo) for (y, mo) in months[1:3:end]])
    fig = Figure(size = (1100, 260 * length(bands) + 60), fontsize = 13)
    Label(fig[0, 1], "Open-water wind stress |<τ>|: ERA5 + Large & Yeager (neutral) against B-SOSE"; font = :bold, fontsize = 15)
    for (r, b) in enumerate(bands)
        ax = Axis(fig[r, 1]; title = band_name(b), ylabel = "N/m²", xticks = ticks, xticklabelrotation = π / 6)
        lines!(ax, x, mon[:bsose][b, :]; color = :black, linewidth = 2.5, label = "B-SOSE oceTAU")
        lines!(ax, x, mon[:era5][b, :]; color = :royalblue, linewidth = 2, label = "ERA5 L&Y, ρa from ERA5")
        lines!(ax, x, mon[:era5_12][b, :]; color = :royalblue, linewidth = 2, linestyle = :dash, label = "ERA5 L&Y, ρa = 1.2")
        r == 1 && axislegend(ax; position = :lt, backgroundcolor = (:white, 0.8))
    end
    out = joinpath(OUTDIR, "wind_stress_bsose_era5.png")
    save(out, fig)
    @info "wrote $out"
end

main()
