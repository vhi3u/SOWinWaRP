# ==============================================================================
# compare_winter_water.jl -- Antarctic Winter Water in B-SOSE and in a model run, detected the
#                            way Spira et al. (2024) detect it in observations.
#
#     MODEL_RUN=2Y_nested_seaice_fix5 TRIM_SPONGE=0.67 julia --project=. tools/compare_winter_water.jl
#     YEARS=2015 julia --project=. tools/compare_winter_water.jl            # B-SOSE alone
#     MODEL_RUN=... MAP_MONTHS=2,9 julia --project=. tools/compare_winter_water.jl
#
# The detection is tools/winter_water.jl, one column at a time: mixed-layer WW (WW_ML) or
# subsurface WW (WW_SS, a >= 20 m temperature inversion below the mixed layer), with no WW where
# the core is >= 2 °C, deeper than its own lower boundary, or where the core or lower boundary is
# deeper than 300 m. See that file for the rules and their settings.
#
# LIKE FOR LIKE. B-SOSE's Theta/Salt are monthly means, so the model side is its 30-day means
# (RUN_eke.nc), each paired with the B-SOSE month whose centre is within 20 days of its own. WW is
# detected on each dataset's OWN grid -- B-SOSE i156 and the nested run are both 1/6 deg on the
# same 52 levels -- so nothing is regridded, and this is NCDatasets only: LOGIN-NODE SAFE.
# A WW of the monthly-MEAN state is not the mean of daily WWs (the Jensen point compare_mld.jl
# makes about the mixed layer); the daily section figure below is there to show how far apart
# they are along 120E.
#
# PRINTS, for each latitude band and month, B-SOSE against the model, cos(lat)-weighted over
# ocean columns outside the boundary band (TRIM_SPONGE; 0.67 for a nested run):
#     ML%, SS%     share of columns with mixed-layer / subsurface WW
#     Tcore        mean core temperature of the columns with WW, °C
#     core, thick  mean core depth and layer thickness, m
# and, per month, the share of columns removed as too warm, too deep, or inverted.
#
# WRITES (OUTDIR, default plots/):
#     ww_<tag>_series.png         the table above as time series, one column per latitude band
#     ww_<tag>_map_<yyyy-mm>.png  WW type, core temperature and thickness, B-SOSE over model, for
#                                 each month in MAP_MONTHS (calendar months, default 2 and 9)
#     ww_<tag>_section.png        core temperature and depth along the RUN_midlon.nc meridian:
#                                 B-SOSE monthly against the model DAILY
#
# ENV
#     BSOSE_DIR    B-SOSE monthly files and grid.nc         (default: data)
#     MODEL_RUN    run to compare; without it, B-SOSE alone (default: none)
#     MODEL_DIR    where RUN_eke.nc / RUN_midlon.nc are     (default: .)
#     YEARS        years to analyse without a model run     (default: 2015)
#     MAP_MONTHS   calendar months to map                   (default: 2,9)
#     TRIM_SPONGE  boundary band to exclude, deg            (default: 3.0 via tools/sponge.jl)
#     ZREAD        depth read from both datasets, m         (default: 700)
#     OUTDIR       figures                                  (default: plots)
#   and WW_INVERSION, WW_TMAX, WW_ZMAX, WW_SEARCH, MLD_DT from tools/winter_water.jl
# ==============================================================================

using NCDatasets, Dates, Printf, Statistics, CairoMakie

include(joinpath(@__DIR__, "mld.jl"))
include(joinpath(@__DIR__, "winter_water.jl"))
include(joinpath(@__DIR__, "model_grid.jl"))
include(joinpath(@__DIR__, "sponge.jl"))

const BDIR       = get(ENV, "BSOSE_DIR", "data")
const MODEL_RUN  = get(ENV, "MODEL_RUN", "")
const MODEL_DIR  = get(ENV, "MODEL_DIR", ".")
const ZREAD      = parse(Float64, get(ENV, "ZREAD", "700"))
const MAP_MONTHS = parse.(Int, split(get(ENV, "MAP_MONTHS", "2,9"), ","))
const OUTDIR     = get(ENV, "OUTDIR", "plots")
mkpath(OUTDIR)

const LON   = (90.0, 150.0)
const LAT   = (-70.0, -45.0)
const BANDS = [(-70.0, -65.0), (-65.0, -60.0), (-60.0, -55.0), (-55.0, -50.0), (-50.0, -45.0)]
const KEEP  = sponge_trimmer(LON, LAT)

bfile(token) = joinpath(BDIR, "$(token)_bsoseI156_2013to2024_monthly.nc")

function varname(ds, token)
    for c in (token, uppercase(token))
        haskey(ds, c) && return c
    end
    error("no $token in $(path(ds)); keys are $(collect(keys(ds)))")
end

function coordvar(ds, cands)
    for c in cands
        haskey(ds, c) && return Float64.(coalesce.(ds[c][:], NaN))
    end
    error("none of $(join(cands, ", ")) in $(path(ds))")
end

# A monthly B-SOSE stamp sits at (roughly) the END of the month it averages
function window_center(stamp)
    stop = round(DateTime(stamp), Month)
    start = stop - Month(1)
    return start + Millisecond(Dates.value(Millisecond(stop - start)) ÷ 2)
end

# ── B-SOSE: geometry ──────────────────────────────────────────────────────────────────────────
# Depth and the horizontal grid from the surface heat flux file, the levels from grid.nc (RC and
# DRF, SURFACE FIRST), exactly as tools/mixing_energetics.jl reads them.
const B = let
    for f in (bfile("surfTflx"), bfile("Theta"), bfile("Salt"), joinpath(BDIR, "grid.nc"))
        isfile(f) || error("$f not found (set BSOSE_DIR)")
    end
    ds = Dataset(bfile("surfTflx"))
    xc, yc = coordvar(ds, ["XC"]), coordvar(ds, ["YC"])
    I = findall(x -> LON[1] <= x <= LON[2], xc)
    J = findall(y -> LAT[1] <= y <= LAT[2], yc)
    IR, JR = minimum(I):maximum(I), minimum(J):maximum(J)
    depth = Float64.(coalesce.(ds["Depth"][IR, JR], 0.0))
    stamps = [DateTime(t) for t in ds["time"][:]]
    close(ds)
    g = Dataset(joinpath(BDIR, "grid.nc"))
    RC, DRF = coordvar(g, ["RC"]), coordvar(g, ["DRF"])
    close(g)
    NZ = something(findlast(z -> -z <= ZREAD, RC), length(RC))
    z, dz = RC[1:NZ], DRF[1:NZ]
    kb = [depth[i, j] > 0 ? something(findlast(k -> -z[k] <= depth[i, j], 1:NZ), 0) : 0
          for i in axes(depth, 1), j in axes(depth, 2)]
    (; lon = xc[IR], lat = yc[JR], IR, JR, z, dz, NZ, kb, stamps,
       centres = window_center.(stamps))
end
@info @sprintf("B-SOSE: %dx%d columns over %.1f-%.1fE, %d levels to %.0f m",
               length(B.lon), length(B.lat), B.lon[1], B.lon[end], B.NZ, -B.z[end])

"""B-SOSE T and S (surface-first) for monthly record `k`."""
function bsose_month(k)
    out = map(("Theta", "Salt")) do token
        ds = Dataset(bfile(token))
        a = Float64.(coalesce.(ds[varname(ds, token)][B.IR, B.JR, 1:B.NZ, k], NaN))
        close(ds)
        a
    end
    return out[1], out[2]
end

# ── the model: geometry and the 30-day-mean records ───────────────────────────────────────────
const M = isempty(MODEL_RUN) ? nothing : let
    path = joinpath(MODEL_DIR, "$(MODEL_RUN)_eke.nc")
    isfile(path) || error("$path not found: the 30-day-mean full-depth output is what this compares")
    ds = Dataset(path)
    lv = file_levels(ds)
    lv === nothing && error("$path carries no z_aac/z_aaf")
    zc_up, zf_up, _ = lv
    fl = file_kbot(ds, zc_up, zf_up)
    fl === nothing && error("$path carries no inactive_nodes_ccc or bottom_height")
    kbf = fl[1]
    lon, lat = coordvar(ds, ["λ_caa", "x_caa"]), coordvar(ds, ["φ_aca", "y_aca"])
    t = Float64.(ds["time"][:])
    win = haskey(ds.attrib, "averaging_window_seconds") ? Float64(ds.attrib["averaging_window_seconds"]) : 30 * 86400.0
    start = haskey(ENV, "MODEL_START") ? DateTime(ENV["MODEL_START"]) :
            haskey(ds.attrib, "start_date") ? DateTime(string(ds.attrib["start_date"])) : DateTime(2014, 6, 1)
    close(ds)
    Nz = length(zc_up)
    z, dz = reverse(zc_up), reverse(diff(zf_up))                     # surface-first
    NZ = something(findlast(zz -> -zz <= ZREAD, z), Nz)
    kb = [kbf[i, j] === nothing ? 0 : min(Nz - kbf[i, j] + 1, NZ) for i in axes(kbf, 1), j in axes(kbf, 2)]
    centres = [start + Millisecond(round(Int, 1000 * (x - win / 2))) for x in t]
    (; path, lon, lat, z = z[1:NZ], dz = dz[1:NZ], NZ, Nz, kb, start, centres)
end

"""Model T and S (surface-first, top NZ levels) for 30-day record `n`."""
function model_month(n)
    ds = Dataset(M.path)
    T = Float64.(coalesce.(ds["T"][:, :, :, n], NaN))
    S = Float64.(coalesce.(ds["S"][:, :, :, n], NaN))
    close(ds)
    return reverse(T, dims = 3)[:, :, 1:M.NZ], reverse(S, dims = 3)[:, :, 1:M.NZ]
end

# ── which months ──────────────────────────────────────────────────────────────────────────────
# With a model: every B-SOSE month with a model 30-day mean centred within 20 days of it.
month_pairs = if M === nothing
    years = parse.(Int, split(get(ENV, "YEARS", "2015"), ","))
    [(k, 0) for k in eachindex(B.stamps) if year(B.centres[k]) in years]
else
    p = Tuple{Int,Int}[]
    for k in eachindex(B.stamps)
        gap, n = findmin(abs(Dates.value(c - B.centres[k])) for c in M.centres)
        gap <= 20 * 86400_000 && push!(p, (k, n))
    end
    p
end
isempty(month_pairs) && error("no B-SOSE month to analyse (model records too far from every B-SOSE month?)")
labels = [Dates.format(B.centres[k], "yyyy-mm") for (k, _) in month_pairs]
@info "months: $(first(labels)) to $(last(labels)) ($(length(month_pairs)))" *
      (M === nothing ? "" : ", model 30-day means from $(basename(M.path))")

# ── detection over a whole record ─────────────────────────────────────────────────────────────
# kind: -1 not analysed (land or boundary band), 0 no WW, 1 WW_ML, 2 WW_SS
function detect(T, S, z, dz, kb, lon, lat)
    nx, ny = size(kb)
    kind   = fill(Int8(-1), nx, ny)
    reason = fill(:land, nx, ny)
    Tc, core, thick = fill(NaN, nx, ny), fill(NaN, nx, ny), fill(NaN, nx, ny)
    for i in 1:nx, j in 1:ny
        (kb[i, j] >= 2 && KEEP(lon[i], lat[j])) || continue
        w = winter_water(view(T, i, j, :), view(S, i, j, :), z, dz, kb[i, j])
        reason[i, j] = w.reason
        kind[i, j] = w.kind === :ML ? 1 : w.kind === :SS ? 2 : 0
        if kind[i, j] > 0
            Tc[i, j], core[i, j], thick[i, j] = w.T, w.core, w.thickness
        end
    end
    return (; kind, reason, T = Tc, core, thick)
end

"""cos(lat)-weighted band statistics of one detected record."""
function band_stats(F, lat, lo, hi)
    wall = wml = wss = 0.0
    sT = sc = sh = ww = 0.0
    for i in axes(F.kind, 1), j in axes(F.kind, 2)
        (lo <= lat[j] < hi && F.kind[i, j] >= 0) || continue
        w = cosd(lat[j])
        wall += w
        F.kind[i, j] == 1 && (wml += w)
        F.kind[i, j] == 2 && (wss += w)
        if F.kind[i, j] > 0
            ww += w; sT += w * F.T[i, j]; sc += w * F.core[i, j]; sh += w * F.thick[i, j]
        end
    end
    wall == 0 && return (ml = NaN, ss = NaN, T = NaN, core = NaN, thick = NaN)
    return (ml = 100wml / wall, ss = 100wss / wall,
            T = ww > 0 ? sT / ww : NaN, core = ww > 0 ? sc / ww : NaN, thick = ww > 0 ? sh / ww : NaN)
end

removed_pct(F, r) = (n = count(>=(0), F.kind); n == 0 ? NaN : 100 * count(==(r), F.reason[F.kind .>= 0]) / n)

# ── run ───────────────────────────────────────────────────────────────────────────────────────
const SECT_FILE = isempty(MODEL_RUN) ? "" : joinpath(MODEL_DIR, "$(MODEL_RUN)_midlon.nc")
const MID = (M === nothing || !isfile(SECT_FILE)) ? 0 : section_column(SECT_FILE, length(M.lon))
const BSECT_I = MID == 0 ? 0 : argmin(abs.(B.lon .- M.lon[MID]))

results = map(month_pairs) do (k, n)
    Tb, Sb = bsose_month(k)
    Fb = detect(Tb, Sb, B.z, B.dz, B.kb, B.lon, B.lat)
    bsect = BSECT_I == 0 ? nothing :
            [winter_water(view(Tb, BSECT_I, j, :), view(Sb, BSECT_I, j, :), B.z, B.dz, B.kb[BSECT_I, j])
             for j in eachindex(B.lat)]
    Fm = nothing
    if M !== nothing
        Tm, Sm = model_month(n)
        Fm = detect(Tm, Sm, M.z, M.dz, M.kb, M.lon, M.lat)
    end
    @info "  $(Dates.format(B.centres[k], "yyyy-mm")) done"
    (; k, n, Fb, Fm, bsect)
end

# ── tables ────────────────────────────────────────────────────────────────────────────────────
println()
println("="^118)
@printf("WINTER WATER after Spira et al. (2024): inversion >= %.0f m, core < %.1f °C, core and lower boundary <= %.0f m, MLD Δρ = %.2f\n",
        WW_INVERSION, WW_TMAX, WW_ZMAX, MLD_DT)
println("  ML% / SS% = share of ocean columns with mixed-layer / subsurface WW; Tcore, core, thick over the columns with WW.")
println("  cos(lat)-weighted, outside the $(SPONGE_TRIM)-deg boundary band. B-SOSE monthly means" *
        (M === nothing ? "." : " against the model's 30-day means ($(MODEL_RUN))."))
println("="^118)
fmt(v, d) = isnan(v) ? lpad("--", 7) : d == 0 ? @sprintf("%7.0f", v) : d == 1 ? @sprintf("%7.1f", v) : @sprintf("%7.2f", v)
for (lo, hi) in BANDS
    @printf("\n  %.0fS-%.0fS\n", -lo, -hi)
    @printf("  %-8s |%s", "month", "    ML%    SS%  Tcore   core  thick")
    M === nothing || print(" |    ML%    SS%  Tcore   core  thick")
    println()
    for (r, lab) in zip(results, labels)
        b = band_stats(r.Fb, B.lat, lo, hi)
        @printf("  %-8s |%s%s%s%s%s", lab, fmt(b.ml, 0), fmt(b.ss, 0), fmt(b.T, 2), fmt(b.core, 0), fmt(b.thick, 0))
        if r.Fm !== nothing
            m = band_stats(r.Fm, M.lat, lo, hi)
            @printf(" |%s%s%s%s%s", fmt(m.ml, 0), fmt(m.ss, 0), fmt(m.T, 2), fmt(m.core, 0), fmt(m.thick, 0))
        end
        println()
    end
end
M === nothing || println("\n  left block B-SOSE, right block $(MODEL_RUN)")

println("\n  Columns REMOVED (% of ocean columns): too warm (core >= $(WW_TMAX) °C) / too deep (> $(WW_ZMAX) m) / inverted")
@printf("  %-8s | %22s", "month", "B-SOSE warm deep inv")
M === nothing || @printf(" | %22s", "model warm deep inv")
println()
for (r, lab) in zip(results, labels)
    @printf("  %-8s | %8.1f %6.1f %6.1f", lab, removed_pct(r.Fb, :warm), removed_pct(r.Fb, :deep), removed_pct(r.Fb, :inverted))
    r.Fm === nothing || @printf(" | %8.1f %6.1f %6.1f", removed_pct(r.Fm, :warm), removed_pct(r.Fm, :deep), removed_pct(r.Fm, :inverted))
    println()
end

tag = isempty(MODEL_RUN) ? "bsose_" * join(unique(year.(B.centres[[r.k for r in results]])), "-") : MODEL_RUN

# ── figure: time series by band ───────────────────────────────────────────────────────────────
let
    x = collect(1:length(results))
    step = max(1, cld(length(x), 8))
    ticks = (x[1:step:end], labels[1:step:end])
    fig = Figure(size = (330 * length(BANDS) + 80, 950))
    Label(fig[0, 1:length(BANDS)], "Winter Water after Spira et al. (2024) — black B-SOSE" *
          (M === nothing ? "" : ", red $(MODEL_RUN)"), fontsize = 18, font = :bold)
    rows = (("WW columns (%)", s -> s.ml + s.ss), ("core T (°C)", s -> s.T),
            ("core depth (m)", s -> s.core), ("thickness (m)", s -> s.thick))
    for (c, (lo, hi)) in enumerate(BANDS), (rr, (ylab, stat)) in enumerate(rows)
        ax = Axis(fig[rr, c]; title = rr == 1 ? @sprintf("%.0fS-%.0fS", -lo, -hi) : "",
                  ylabel = c == 1 ? ylab : "", xticks = ticks, xticklabelrotation = π / 4,
                  xticklabelsvisible = rr == length(rows), yreversed = rr == 3)
        sb = [band_stats(r.Fb, B.lat, lo, hi) for r in results]
        lines!(ax, x, [stat(s) for s in sb], color = :black)
        rr == 1 && lines!(ax, x, [s.ss for s in sb], color = :black, linestyle = :dash)
        if M !== nothing
            sm = [band_stats(r.Fm, M.lat, lo, hi) for r in results]
            lines!(ax, x, [stat(s) for s in sm], color = :firebrick)
            rr == 1 && lines!(ax, x, [s.ss for s in sm], color = :firebrick, linestyle = :dash)
        end
    end
    Label(fig[length(rows) + 1, 1:length(BANDS)],
          "Top row: solid = all WW (mixed-layer + subsurface), dashed = subsurface only. Core T, depth and thickness " *
          "are means over the columns with WW. Monthly means against 30-day means.", fontsize = 11)
    f = joinpath(OUTDIR, "ww_$(tag)_series.png")
    save(f, fig); @info "wrote $f"
end

# ── figures: maps for MAP_MONTHS ──────────────────────────────────────────────────────────────
for (r, lab) in zip(results, labels)
    month(B.centres[r.k]) in MAP_MONTHS || continue
    sets = M === nothing ? (("B-SOSE", r.Fb, B.lon, B.lat),) :
                           (("B-SOSE", r.Fb, B.lon, B.lat), (MODEL_RUN, r.Fm, M.lon, M.lat))
    fig = Figure(size = (1500, 330 * length(sets) + 90))
    Label(fig[0, 1:6], "Winter Water, $lab (Spira et al. 2024 detection)", fontsize = 18, font = :bold)
    for (row, (name, F, lon, lat)) in enumerate(sets)
        kindmap = [F.kind[i, j] < 0 ? NaN : Float64(F.kind[i, j]) for i in axes(F.kind, 1), j in axes(F.kind, 2)]
        panels = (("type: none / ML / subsurface", kindmap, cgrad([:gray97, :skyblue, :navy], 3, categorical = true), (-0.5, 2.5), ""),
                  ("core T", F.T, :thermal, (-2.0, WW_TMAX), "°C"),
                  ("thickness", F.thick, :dense, (0.0, WW_ZMAX), "m"))
        for (c, (title, A, cmap, lims, unit)) in enumerate(panels)
            ax = Axis(fig[row, 2c - 1]; title = "$name — $title", ylabel = c == 1 ? "latitude" : "",
                      xlabel = row == length(sets) ? "longitude" : "")
            hm = heatmap!(ax, lon, lat, A; colormap = cmap, colorrange = lims, nan_color = :gray75)
            c == 1 ? Colorbar(fig[row, 2c], hm, ticks = ([0, 1, 2], ["none", "ML", "subsurface"])) :
                     Colorbar(fig[row, 2c], hm, label = unit)
        end
    end
    Label(fig[length(sets) + 1, 1:6], "Dark grey: land, the boundary band, or (core T, thickness) no WW.", fontsize = 11)
    f = joinpath(OUTDIR, "ww_$(tag)_map_$(lab).png")
    save(f, fig); @info "wrote $f"
end

# ── figure: along the section, B-SOSE monthly against the model daily ─────────────────────────
if MID > 0
    ds = Dataset(SECT_FILE)
    lv = file_levels(ds)
    fl = lv === nothing ? nothing : file_kbot(ds, lv[1], lv[2])
    if fl === nothing
        @warn "$SECT_FILE carries no levels or sea floor: no section figure"
    else
        zc_up, zf_up, _ = lv
        Nz = length(zc_up)
        z, dz = reverse(zc_up), reverse(diff(zf_up))
        NZ = something(findlast(zz -> -zz <= ZREAD, z), Nz)
        kbf = vec(fl[1])
        kb = [k === nothing ? 0 : min(Nz - k + 1, NZ) for k in kbf]
        st = Float64.(ds["time"][:])
        lat = M.lat
        days = st ./ 86400
        CT, CD = fill(NaN, length(st), length(lat)), fill(NaN, length(st), length(lat))
        for n in eachindex(st)
            n == 1 && st[n] == 0 && continue
            T = reverse(Float64.(coalesce.(ds["T"][1, :, :, n], NaN)), dims = 2)
            S = reverse(Float64.(coalesce.(ds["S"][1, :, :, n], NaN)), dims = 2)
            for j in eachindex(lat)
                (kb[j] >= 2 && KEEP(M.lon[MID], lat[j])) || continue
                w = winter_water(view(T, j, 1:NZ), view(S, j, 1:NZ), z[1:NZ], dz[1:NZ], kb[j])
                w.kind === :none || (CT[n, j] = w.T; CD[n, j] = w.core)
            end
        end
        close(ds)
        bdays = [Dates.value(B.centres[r.k] - M.start) / 86400_000 for r in results]
        BT = [r.bsect[j].kind === :none ? NaN : r.bsect[j].T for r in results, j in eachindex(B.lat)]
        BD = [r.bsect[j].kind === :none ? NaN : r.bsect[j].core for r in results, j in eachindex(B.lat)]
        fig = Figure(size = (1500, 800))
        Label(fig[0, 1:3], @sprintf("Winter Water core along %.1fE: B-SOSE monthly (left) against %s daily (right)",
                                    M.lon[MID], MODEL_RUN), fontsize = 18, font = :bold)
        for (row, (A_b, A_m, title, cmap, lims, unit)) in enumerate(((BT, CT, "core T", :thermal, (-2.0, WW_TMAX), "°C"),
                                                                     (BD, CD, "core depth", Reverse(:deep), (0.0, WW_ZMAX), "m")))
            axb = Axis(fig[row, 1]; title = "B-SOSE $title", ylabel = "latitude", xlabel = row == 2 ? "days since $(Date(M.start))" : "")
            heatmap!(axb, bdays, B.lat, A_b; colormap = cmap, colorrange = lims, nan_color = :gray85)
            axm = Axis(fig[row, 2]; title = "model $title", xlabel = row == 2 ? "days since $(Date(M.start))" : "")
            hm = heatmap!(axm, days, lat, A_m; colormap = cmap, colorrange = lims, nan_color = :gray85)
            linkaxes!(axb, axm)
            Colorbar(fig[row, 3], hm, label = unit)
        end
        Label(fig[3, 1:3], "Grey: no WW (none detected, removed, land or boundary band). B-SOSE at its nearest " *
              "column, $(round(B.lon[BSECT_I], digits = 2))E.", fontsize = 11)
        f = joinpath(OUTDIR, "ww_$(tag)_section.png")
        save(f, fig); @info "wrote $f"
    end
end
