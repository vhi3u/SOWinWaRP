# ==============================================================================
# compare_winter_water.jl -- Antarctic Winter Water in B-SOSE and in a model run, detected the
#                            way Spira et al. (2024) detect it in observations, and placed
#                            against the ACC fronts.
#
#     MODEL_RUN=2Y_nested_seaice_fix5 TRIM_SPONGE=0.67 julia --project=. tools/compare_winter_water.jl
#     julia --project=. tools/compare_winter_water.jl                      # B-SOSE alone, 2014-2016
#     CIRCUMPOLAR=1 julia -t 8 --project=. tools/compare_winter_water.jl   # all the way round
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
# THE DOMAIN. By default the nest's box, 90-150E / 70-45S, with the boundary band (TRIM_SPONGE)
# left out of both datasets. CIRCUMPOLAR=1 is B-SOSE alone over every longitude from 80S to 35S,
# shelves included, with no trim; it adds a table by sector (south of the PF only) and draws the
# maps in a south-polar stereographic projection. It reads ~2160 x 300 columns a month, so give it
# threads (julia -t N): the detection loop is threaded over longitude.
#
# THE FRONTS (tools/fronts.jl). The CTOH weekly PF, SAF and NSAF (the -0.3, -0.1 and 0 m ADT
# contours), the median over each B-SOSE month's weeks, split every column into a zone: south of
# the PF, PF-SAF, SAF-NSAF, north of the NSAF. The model is zoned by the SAME observed fronts --
# it has no altimetry of its own -- so a model whose fronts have moved has its water counted in
# the observed zone, not its own.
#
# PRINTS, cos(lat)-weighted over the ocean columns analysed, for each month, B-SOSE against the
# model, by latitude band, by front zone and (CIRCUMPOLAR) by sector:
#     ML%, SS%     share of columns with mixed-layer / subsurface WW
#     Tcore        mean core temperature of the columns with WW, °C
#     core, thick  mean core depth and layer thickness, m
# then the share of columns removed as too warm, too deep, or inverted, and the mean latitude of
# each front over the domain.
#
# WRITES (OUTDIR, default plots/):
#     ww_<tag>_series.png         the band table as time series, one column per latitude band
#     ww_<tag>_zones.png          the same by front zone
#     ww_<tag>_sectors.png        (CIRCUMPOLAR) the same by sector, south of the PF
#     ww_<tag>_map_<yyyy-mm>.png  WW type, core temperature and thickness with the month's fronts,
#                                 B-SOSE over model, for each month in MAP_MONTHS (default 2 and 9)
#     ww_<tag>_section.png        core temperature and depth along the RUN_midlon.nc meridian:
#                                 B-SOSE monthly against the model DAILY, fronts overlaid
#
# ENV
#     BSOSE_DIR    B-SOSE monthly files and grid.nc         (default: data)
#     FRONTS_DIR   the CTOH front files                     (default: BSOSE_DIR)
#     MODEL_RUN    run to compare; without it, B-SOSE alone (default: none)
#     MODEL_DIR    where RUN_eke.nc / RUN_midlon.nc are     (default: .)
#     YEARS        years to analyse without a model run, "2014-2016" or "2014,2016" (default: 2014-2016)
#     CIRCUMPOLAR  1 for B-SOSE all the way round            (default: 0)
#     MAP_MONTHS   calendar months to map                   (default: 2,9)
#     MAP_STRIDE   plot every n-th column on the polar maps (default: 2)
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
include(joinpath(@__DIR__, "fronts.jl"))

const BDIR        = get(ENV, "BSOSE_DIR", "data")
const MODEL_RUN   = get(ENV, "MODEL_RUN", "")
const MODEL_DIR   = get(ENV, "MODEL_DIR", ".")
const ZREAD       = parse(Float64, get(ENV, "ZREAD", "700"))
const MAP_MONTHS  = parse.(Int, split(get(ENV, "MAP_MONTHS", "2,9"), ","))
const MAP_STRIDE  = parse(Int, get(ENV, "MAP_STRIDE", "2"))
const OUTDIR      = get(ENV, "OUTDIR", "plots")
const CIRCUMPOLAR = get(ENV, "CIRCUMPOLAR", "0") == "1"
mkpath(OUTDIR)
CIRCUMPOLAR && !isempty(MODEL_RUN) && error("CIRCUMPOLAR=1 is B-SOSE alone: unset MODEL_RUN")

const LON   = CIRCUMPOLAR ? (0.0, 360.0) : (90.0, 150.0)
const LAT   = CIRCUMPOLAR ? (-80.0, -35.0) : (-70.0, -45.0)
const BANDS = CIRCUMPOLAR ?
    [(-80.0, -70.0), (-70.0, -65.0), (-65.0, -60.0), (-60.0, -55.0), (-55.0, -50.0), (-50.0, -45.0), (-45.0, -35.0)] :
    [(-70.0, -65.0), (-65.0, -60.0), (-60.0, -55.0), (-55.0, -50.0), (-50.0, -45.0)]
const KEEP  = CIRCUMPOLAR ? ((λ, φ) -> true) : sponge_trimmer(LON, LAT)

# Sectors (CIRCUMPOLAR), degrees east; the Weddell one wraps through 0
const SECTORS = [("Weddell", 300.0, 20.0), ("Indian", 20.0, 90.0), ("Australian", 90.0, 160.0),
                 ("Ross", 160.0, 230.0), ("Amundsen-Bell.", 230.0, 300.0)]
in_sector(λ, (_, a, b)) = (x = mod(λ, 360); a < b ? a <= x < b : (x >= a || x < b))

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

# "2014-2016,2018" -> [2014, 2015, 2016, 2018]
parse_years(s) = reduce(vcat, [occursin('-', p) ? collect(range(parse.(Int, split(p, '-'))...)) : [parse(Int, p)]
                               for p in split(s, ",")])

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
@info @sprintf("B-SOSE: %dx%d columns over %.1f-%.1fE, %.1f-%.1fN, %d levels to %.0f m",
               length(B.lon), length(B.lat), B.lon[1], B.lon[end], B.lat[1], B.lat[end], B.NZ, -B.z[end])

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
    years = parse_years(get(ENV, "YEARS", "2014-2016"))
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
      (M === nothing ? "" : ", model 30-day means from $(basename(M.path))") *
      ", $(Threads.nthreads()) thread(s)"

# The fronts, for the whole period at once
const FR = let t = [B.centres[k] for (k, _) in month_pairs]
    load_fronts(; from = DateTime(firstdayofmonth(minimum(t))), to = DateTime(firstdayofmonth(maximum(t))) + Month(1))
end
@info "fronts: $(length(FR.PF.weeks)) CTOH weeks, $(FR.PF.weeks[1]) to $(FR.PF.weeks[end]), from $FRONTS_DIR"

# ── detection over a whole record ─────────────────────────────────────────────────────────────
# kind: -1 not analysed (land or boundary band), 0 no WW, 1 WW_ML, 2 WW_SS
function detect(T, S, z, dz, kb, lon, lat)
    nx, ny = size(kb)
    kind   = fill(Int8(-1), nx, ny)
    reason = fill(:land, nx, ny)
    Tc, core, thick = fill(NaN, nx, ny), fill(NaN, nx, ny), fill(NaN, nx, ny)
    Threads.@threads for i in 1:nx
        for j in 1:ny
            (kb[i, j] >= 2 && KEEP(lon[i], lat[j])) || continue
            w = winter_water(view(T, i, j, :), view(S, i, j, :), z, dz, kb[i, j])
            reason[i, j] = w.reason
            kind[i, j] = w.kind === :ML ? 1 : w.kind === :SS ? 2 : 0
            if kind[i, j] > 0
                Tc[i, j], core[i, j], thick[i, j] = w.T, w.core, w.thickness
            end
        end
    end
    return (; kind, reason, T = Tc, core, thick)
end

"""cos(lat)-weighted statistics of one detected record over the columns where `member(i, j)`."""
function group_stats(F, lat, member)
    wall = wml = wss = 0.0
    sT = sc = sh = ww = 0.0
    for i in axes(F.kind, 1), j in axes(F.kind, 2)
        (F.kind[i, j] >= 0 && member(i, j)) || continue
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

"""Every table's numbers for one record: by band, by front zone, by sector, and the removals."""
function summarize(F, lon, lat, fr)
    zone = [front_zone(lat[j], fr.PF[i], fr.SAF[i], fr.NSAF[i]) for i in eachindex(lon), j in eachindex(lat)]
    bands = [group_stats(F, lat, (i, j) -> lo <= lat[j] < hi) for (lo, hi) in BANDS]
    zones = [group_stats(F, lat, (i, j) -> zone[i, j] == z) for z in eachindex(ZONE_NAMES)]
    sectors = CIRCUMPOLAR ? [group_stats(F, lat, (i, j) -> zone[i, j] == 1 && in_sector(lon[i], s)) for s in SECTORS] :
                            nothing
    removed = (warm = removed_pct(F, :warm), deep = removed_pct(F, :deep), inverted = removed_pct(F, :inverted))
    return (; bands, zones, sectors, removed)
end

# ── run ───────────────────────────────────────────────────────────────────────────────────────
# Only the summaries are kept for every month; the full fields only for the months mapped, which
# is what keeps CIRCUMPOLAR inside a login node's memory.
const SECT_FILE = isempty(MODEL_RUN) ? "" : joinpath(MODEL_DIR, "$(MODEL_RUN)_midlon.nc")
const MID = (M === nothing || !isfile(SECT_FILE)) ? 0 : section_column(SECT_FILE, length(M.lon))
const BSECT_I = MID == 0 ? 0 : argmin(abs.(B.lon .- M.lon[MID]))

results = map(month_pairs) do (k, n)
    t = B.centres[k]
    mapped = month(t) in MAP_MONTHS
    frb = fronts_for_month(FR, B.lon, t)
    Tb, Sb = bsose_month(k)
    Fb = detect(Tb, Sb, B.z, B.dz, B.kb, B.lon, B.lat)
    sb = summarize(Fb, B.lon, B.lat, frb)
    bsect = BSECT_I == 0 ? nothing :
            [winter_water(view(Tb, BSECT_I, j, :), view(Sb, BSECT_I, j, :), B.z, B.dz, B.kb[BSECT_I, j])
             for j in eachindex(B.lat)]
    Fm = sm = frm = nothing
    if M !== nothing
        frm = fronts_for_month(FR, M.lon, t)                 # the B-SOSE month's observed fronts
        Tm, Sm = model_month(n)
        Fm = detect(Tm, Sm, M.z, M.dz, M.kb, M.lon, M.lat)
        sm = summarize(Fm, M.lon, M.lat, frm)
    end
    @info "  $(Dates.format(t, "yyyy-mm")) done"
    (; k, n, sb, sm, frb, frm, bsect, Fb = mapped ? Fb : nothing, Fm = mapped ? Fm : nothing)
end

# ── tables ────────────────────────────────────────────────────────────────────────────────────
fmt(v, d) = isnan(v) ? lpad("--", 7) : d == 0 ? @sprintf("%7.0f", v) : d == 1 ? @sprintf("%7.1f", v) : @sprintf("%7.2f", v)
fmtstats(s) = fmt(s.ml, 0) * fmt(s.ss, 0) * fmt(s.T, 2) * fmt(s.core, 0) * fmt(s.thick, 0)

function print_table(title, pick)
    @printf("\n  %s\n", title)
    @printf("  %-8s |%s", "month", "    ML%    SS%  Tcore   core  thick")
    M === nothing || print(" |    ML%    SS%  Tcore   core  thick")
    println()
    for (r, lab) in zip(results, labels)
        @printf("  %-8s |%s", lab, fmtstats(pick(r.sb)))
        r.sm === nothing || @printf(" |%s", fmtstats(pick(r.sm)))
        println()
    end
end

println()
println("="^118)
@printf("WINTER WATER after Spira et al. (2024): inversion >= %.0f m, core < %.1f °C, core and lower boundary <= %.0f m, MLD Δρ = %.2f\n",
        WW_INVERSION, WW_TMAX, WW_ZMAX, MLD_DT)
println("  ML% / SS% = share of ocean columns with mixed-layer / subsurface WW; Tcore, core, thick over the columns with WW.")
println("  cos(lat)-weighted" * (CIRCUMPOLAR ? ", circumpolar, " : ", outside the $(SPONGE_TRIM)-deg boundary band, ") *
        @sprintf("%.0f-%.0fS. B-SOSE monthly means", -LAT[1], -LAT[2]) *
        (M === nothing ? "." : " against the model's 30-day means ($(MODEL_RUN))."))
println("="^118)

println("\n  BY LATITUDE BAND")
for (b, (lo, hi)) in enumerate(BANDS)
    print_table(@sprintf("%.0fS-%.0fS", -lo, -hi), s -> s.bands[b])
end
println("\n  BY FRONT ZONE (CTOH PF -0.3 m, SAF -0.1 m, NSAF 0 m ADT; the model zoned by the observed fronts)")
for (z, name) in enumerate(ZONE_NAMES)
    print_table(name, s -> s.zones[z])
end
if CIRCUMPOLAR
    println("\n  BY SECTOR, SOUTH OF THE PF")
    for (c, (name, a, b)) in enumerate(SECTORS)
        print_table(@sprintf("%s (%.0f-%.0fE)", name, a, b), s -> s.sectors[c])
    end
end
M === nothing || println("\n  left block B-SOSE, right block $(MODEL_RUN)")

println("\n  Columns REMOVED (% of ocean columns): too warm (core >= $(WW_TMAX) °C) / too deep (> $(WW_ZMAX) m) / inverted")
@printf("  %-8s | %22s", "month", "B-SOSE warm deep inv")
M === nothing || @printf(" | %22s", "model warm deep inv")
println()
for (r, lab) in zip(results, labels)
    @printf("  %-8s | %8.1f %6.1f %6.1f", lab, r.sb.removed.warm, r.sb.removed.deep, r.sb.removed.inverted)
    r.sm === nothing || @printf(" | %8.1f %6.1f %6.1f", r.sm.removed.warm, r.sm.removed.deep, r.sm.removed.inverted)
    println()
end

println("\n  FRONTS: mean latitude over the domain's longitudes (CTOH, monthly median of weekly positions)")
@printf("  %-8s | %7s %7s %7s\n", "month", "PF", "SAF", "NSAF")
for (r, lab) in zip(results, labels)
    @printf("  %-8s | %7.2f %7.2f %7.2f\n", lab, mean(r.frb.PF), mean(r.frb.SAF), mean(r.frb.NSAF))
end

yrs = unique(year.(B.centres[[r.k for r in results]]))
tag = !isempty(MODEL_RUN) ? MODEL_RUN :
      "bsose_" * (CIRCUMPOLAR ? "circumpolar_" : "") * (length(yrs) == 1 ? "$(yrs[1])" : "$(first(yrs))-$(last(yrs))")

# ── figures: time series by band, by zone, by sector ──────────────────────────────────────────
function series_figure(titles, pick, name, note)
    x = collect(1:length(results))
    step = max(1, cld(length(x), 8))
    ticks = (x[1:step:end], labels[1:step:end])
    nc = length(titles)
    fig = Figure(size = (330 * nc + 80, 950))
    Label(fig[0, 1:nc], "Winter Water after Spira et al. (2024) — black B-SOSE" *
          (M === nothing ? "" : ", red $(MODEL_RUN)"), fontsize = 18, font = :bold)
    rows = (("WW columns (%)", s -> s.ml + s.ss), ("core T (°C)", s -> s.T),
            ("core depth (m)", s -> s.core), ("thickness (m)", s -> s.thick))
    for c in 1:nc, (rr, (ylab, stat)) in enumerate(rows)
        ax = Axis(fig[rr, c]; title = rr == 1 ? titles[c] : "",
                  ylabel = c == 1 ? ylab : "", xticks = ticks, xticklabelrotation = π / 4,
                  xticklabelsvisible = rr == length(rows), yreversed = rr == 3)
        sb = [pick(r.sb, c) for r in results]
        lines!(ax, x, [stat(s) for s in sb], color = :black)
        rr == 1 && lines!(ax, x, [s.ss for s in sb], color = :black, linestyle = :dash)
        if M !== nothing
            sm = [pick(r.sm, c) for r in results]
            lines!(ax, x, [stat(s) for s in sm], color = :firebrick)
            rr == 1 && lines!(ax, x, [s.ss for s in sm], color = :firebrick, linestyle = :dash)
        end
    end
    Label(fig[length(rows) + 1, 1:nc],
          "Top row: solid = all WW (mixed-layer + subsurface), dashed = subsurface only. Core T, depth and thickness " *
          "are means over the columns with WW. " * note, fontsize = 11)
    f = joinpath(OUTDIR, "ww_$(tag)_$(name).png")
    save(f, fig); @info "wrote $f"
end

series_figure([@sprintf("%.0fS-%.0fS", -lo, -hi) for (lo, hi) in BANDS], (s, c) -> s.bands[c], "series",
              "Monthly means" * (M === nothing ? "." : " against 30-day means."))
series_figure(collect(ZONE_NAMES), (s, c) -> s.zones[c], "zones",
              "Zones from the CTOH weekly fronts (PF -0.3 m, SAF -0.1 m, NSAF 0 m ADT), monthly medians.")
CIRCUMPOLAR && series_figure([s[1] for s in SECTORS], (s, c) -> s.sectors[c], "sectors",
                             "South of the CTOH Polar Front only.")

# ── figures: maps for MAP_MONTHS ──────────────────────────────────────────────────────────────
const FRONT_STYLE = ((:PF, :solid, "PF"), (:SAF, :dash, "SAF"), (:NSAF, :dot, "NSAF"))

# South-polar stereographic, Greenwich up and 90E to the right
polar_xy(λ, φ) = (ρ = tand((90 + φ) / 2); (ρ * sind(λ), ρ * cosd(λ)))

function map_panel!(ax, lon, lat, A, fr; kw...)
    if CIRCUMPOLAR
        I = [collect(1:MAP_STRIDE:length(lon)); 1]               # close the seam at 0E
        J = collect(1:MAP_STRIDE:length(lat))
        X = [polar_xy(lon[i], lat[j])[1] for i in I, j in J]
        Y = [polar_xy(lon[i], lat[j])[2] for i in I, j in J]
        hm = surface!(ax, X, Y, zeros(size(X)); color = A[I, J], shading = NoShading, kw...)
        for φ in -80:10:-40
            xy = polar_xy.(0:2:360, φ)
            lines!(ax, first.(xy), last.(xy), color = (:gray40, 0.5), linewidth = 0.5)
        end
        for (key, ls, _) in FRONT_STYLE
            xy = polar_xy.([lon; lon[1]], [fr[key]; fr[key][1]])
            lines!(ax, first.(xy), last.(xy), color = :crimson, linestyle = ls, linewidth = 1.2)
        end
        hidedecorations!(ax); hidespines!(ax)
    else
        hm = heatmap!(ax, lon, lat, A; kw...)
        for (key, ls, _) in FRONT_STYLE
            lines!(ax, lon, fr[key], color = :crimson, linestyle = ls, linewidth = 1.5)
        end
        ylims!(ax, lat[1], lat[end])
    end
    return hm
end

for (r, lab) in zip(results, labels)
    r.Fb === nothing && continue
    sets = M === nothing ? (("B-SOSE", r.Fb, B.lon, B.lat, r.frb),) :
                           (("B-SOSE", r.Fb, B.lon, B.lat, r.frb), (MODEL_RUN, r.Fm, M.lon, M.lat, r.frm))
    fig = CIRCUMPOLAR ? Figure(size = (1800, 640)) : Figure(size = (1500, 330 * length(sets) + 90))
    Label(fig[0, 1:6], "Winter Water, $lab (Spira et al. 2024 detection)", fontsize = 18, font = :bold)
    for (row, (name, F, lon, lat, fr)) in enumerate(sets)
        kindmap = [F.kind[i, j] < 0 ? NaN : Float64(F.kind[i, j]) for i in axes(F.kind, 1), j in axes(F.kind, 2)]
        panels = (("type: none / ML / subsurface", kindmap, cgrad([:gray97, :skyblue, :navy], 3, categorical = true), (-0.5, 2.5), ""),
                  ("core T", F.T, :thermal, (-2.0, WW_TMAX), "°C"),
                  ("thickness", F.thick, :dense, (0.0, WW_ZMAX), "m"))
        for (c, (title, A, cmap, lims, unit)) in enumerate(panels)
            ax = CIRCUMPOLAR ? Axis(fig[row, 2c - 1]; title = "$name — $title", aspect = DataAspect()) :
                               Axis(fig[row, 2c - 1]; title = "$name — $title", ylabel = c == 1 ? "latitude" : "",
                                    xlabel = row == length(sets) ? "longitude" : "")
            hm = map_panel!(ax, lon, lat, A, fr; colormap = cmap, colorrange = lims, nan_color = :gray75)
            c == 1 ? Colorbar(fig[row, 2c], hm, ticks = ([0, 1, 2], ["none", "ML", "subsurface"])) :
                     Colorbar(fig[row, 2c], hm, label = unit)
        end
    end
    Label(fig[length(sets) + 1, 1:6], "Dark grey: land" * (CIRCUMPOLAR ? "" : ", the boundary band") *
          ", or (core T, thickness) no WW. Red: CTOH fronts for the month — solid PF, dashed SAF, dotted NSAF." *
          (CIRCUMPOLAR ? " Grey circles every 10° from 80S to 40S; Greenwich up, 90E right." : ""), fontsize = 11)
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
            for (key, ls, _) in FRONT_STYLE
                lines!(axb, bdays, [r.frb[key][BSECT_I] for r in results], color = :crimson, linestyle = ls)
                lines!(axm, bdays, [r.frm[key][MID] for r in results], color = :crimson, linestyle = ls)
            end
            linkaxes!(axb, axm)
            ylims!(axb, lat[1], lat[end])
            Colorbar(fig[row, 3], hm, label = unit)
        end
        Label(fig[3, 1:3], "Grey: no WW (none detected, removed, land or boundary band). B-SOSE at its nearest " *
              "column, $(round(B.lon[BSECT_I], digits = 2))E. Red: CTOH monthly fronts at the section — " *
              "solid PF, dashed SAF, dotted NSAF.", fontsize = 11)
        f = joinpath(OUTDIR, "ww_$(tag)_section.png")
        save(f, fig); @info "wrote $f"
    end
end
