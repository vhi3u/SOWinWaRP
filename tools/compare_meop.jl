# ==============================================================================================
# compare_meop.jl -- seal CTD profiles (MEOP-CTD 2024-03-08) against B-SOSE and runs: T and S on the shelf, the
# slope and offshore, 90-150E, 70-60S, by season and layer.
#
#     BSOSE_DIR=/storage/scratch1/1/vnguyen480/SOWinWaRP/data julia --project=. tools/compare_meop.jl
#     BSOSE_DIR=... julia --project=. tools/compare_meop.jl <RUN> [<RUN> ...]       # + runs, over their span
#
# WHY. The coast is where the runs and B-SOSE disagree most (S too salty in the coupled runs, too fresh on B-SOSE's
# own fluxes), and B-SOSE's shelf is the least constrained part of it (no cavities, Nakayama 2024). Seals sample the
# shelf in autumn and winter, when ships do not.
#
# DATA. MEOP/list_profiles.csv (found under data/meop) picks the tags with profiles in the box; each tag's <COUNTRY>/DATA/<tag>_all_prof.nc
# is read: PRES/TEMP/PSAL_ADJUSTED, levels kept where the pressure and the variable both have QC flag 1 or 2,
# profiles dropped with POSITION_QC 3, 4 or 9. JULD is read raw (days since 1950-01-01, checked against the file's
# REFERENCE_DATE_TIME and the CSV). Pressure becomes depth by Saunders (1981). Each profile is put on B-SOSE's level
# centres by linear interpolation between its own levels (SRDL tags send 10-25 levels), never below its deepest
# level; the shallowest value is carried to the surface if it is within TOPGAP m of it.
# T is IN SITU against the models' potential T: above 1000 m in water near 0 °C the difference is under 0.03 °C.
#
# SAMPLING. Each profile takes its nearest WET cell (within SEARCH cells, else dropped) in B-SOSE and in each run.
#   A  YEARS: B-SOSE's MONTHLY MEAN of the profile's calendar month
#   B  the runs' common span: B-SOSE's 5-DAY record holding the profile's time, B-SOSE monthly, and each run's
#      30-day mean (_eke.nc) holding it, interpolated to B-SOSE's levels (the nested runs use them, so exact)
#   C  every profile of YEARS at its CALENDAR date: each run's windows holding that date (one per run year, averaged),
#      B-SOSE's monthly mean of its own month (Bmon) and B-SOSE's mean of that calendar month over the runs' years
#      (Bclim). Fills the sectors the runs' own years barely sample (2014-16: mostly 90-100E); Bclim - Bmon shows what
#      mixing years costs. Writes OUTDIR/meop_profiles_clim_<RUN>....png
# A profile is an instant and the models are means: read A's and B's means, not single profiles. In each table
# every column uses the SAME levels of the same profiles (where the obs and every model are finite).
# REGION by B-SOSE's depth at the profile's cell: shelf < ISOBATHS[1], slope < ISOBATHS[2], deep beyond.
#
# PRINTS: profiles in the box per year and 10° sector (from the CSV, all years), the deployments used, QC counts,
# then per part: region x season x layer, mean obs and mean (model - obs) for T and S; and per sector (shelf + slope,
# SECTOR_LAYER) by season.
# WRITES OUTDIR/meop_map.png (profiles, B-SOSE isobaths) and OUTDIR/meop_profiles[_<RUN>...].png (mean profiles by
# region and season, part B with runs, else part A).
# ENV: BSOSE_DIR (required), DIR (run files, .), OUTDIR (plots), LONS (90,150), LATS (-70,-60),
#      YEARS (2013:2023), ZMAX (1000), ISOBATHS (1000,3000), LAYERS (0,50,200,500,1000), SECTOR_LAYER (0,200),
#      TOPGAP (20), SEARCH (2)
# ==============================================================================================

using NCDatasets, CairoMakie, Dates, Printf, Statistics

ENV["HDF5_USE_FILE_LOCKING"] = "FALSE"
include(joinpath(@__DIR__, "bsose_windows.jl")) # bsose_path, bsose_data_var
include(joinpath(@__DIR__, "model_grid.jl"))    # file_levels, file_kbot

const RUNS      = copy(ARGS)
const BSOSE_DIR = get(ENV, "BSOSE_DIR", "")
isempty(BSOSE_DIR) && error("set BSOSE_DIR to the directory with B-SOSE's monthly and 5-day Theta and Salt")
# Fixed: wherever list_profiles.csv sits under data/meop (the zip unpacks to media/disk2/roquet/MEOP_public/...)
const MEOP_ROOT = "/storage/scratch1/1/vnguyen480/SOWinWaRP/data/meop"
const MEOP = let hit = nothing
    isdir(MEOP_ROOT) || error("$MEOP_ROOT not found")
    for (d, _, fs) in walkdir(MEOP_ROOT)
        "list_profiles.csv" in fs && (hit = d; break)
    end
    hit === nothing && error("no list_profiles.csv anywhere under $MEOP_ROOT")
    hit
end
const DIR    = get(ENV, "DIR", ".")
const OUTDIR = get(ENV, "OUTDIR", "plots")
envlist(k, d) = parse.(Float64, split(get(ENV, k, d), ','))
const LONS     = envlist("LONS", "90,150")
const LATS     = envlist("LATS", "-70,-60")
const YEARS    = eval(Meta.parse(get(ENV, "YEARS", "2013:2023")))
const ZMAX     = parse(Float64, get(ENV, "ZMAX", "1000"))
const ISOBATHS = envlist("ISOBATHS", "1000,3000")
const LAYERS   = envlist("LAYERS", "0,50,200,500,1000")
const SLAYER   = envlist("SECTOR_LAYER", "0,200")
const TOPGAP   = parse(Float64, get(ENV, "TOPGAP", "20"))
const SEARCH   = parse(Int, get(ENV, "SEARCH", "2"))

f64(A) = Float64.(coalesce.(A, NaN))
in_box(lon, lat) = LONS[1] <= lon <= LONS[2] && LATS[1] <= lat <= LATS[2]
season(m) = m in (12, 1, 2) ? 1 : m in (3, 4, 5) ? 2 : m in (6, 7, 8) ? 3 : 4
const SEASONS = ("DJF", "MAM", "JJA", "SON")
region(h) = h < ISOBATHS[1] ? 1 : h < ISOBATHS[2] ? 2 : 3
const REGIONS = (@sprintf("shelf <%.0f m", ISOBATHS[1]), @sprintf("slope %.0f-%.0f", ISOBATHS[1], ISOBATHS[2]),
                 @sprintf("deep >%.0f m", ISOBATHS[2]))
const NSEC = ceil(Int, (LONS[2] - LONS[1]) / 10)
sector(lon) = clamp(floor(Int, (lon - LONS[1]) / 10) + 1, 1, NSEC)
sector_name(s) = @sprintf("%.0f-%.0fE", LONS[1] + 10(s - 1), min(LONS[1] + 10s, LONS[2]))
good(q) = q == '1' || q == '2'
"Depth (m) from pressure (dbar), Saunders (1981)."
saunders(p, lat) = (c1 = (5.92 + 5.25 * sind(lat)^2) * 1e-3; (1 - c1) * p - 2.21e-6 * p^2)

"""
Linear interpolation of a top-down profile (`dsrc` strictly increasing depths) to the depths `dt`; NaN outside it,
except that targets shallower than its first level take that level's value if the first level is at most `topok` deep.
"""
function interp_to(dsrc, vsrc, dt, topok)
    out = fill(NaN, length(dt)); n = length(dsrc)
    n == 0 && return out
    for (k, d) in enumerate(dt)
        if d < dsrc[1]
            dsrc[1] <= topok && (out[k] = vsrc[1])
        elseif d <= dsrc[end]
            j = searchsortedlast(dsrc, d)
            if j == n
                out[k] = vsrc[n]
            else
                w = (d - dsrc[j]) / (dsrc[j+1] - dsrc[j]); out[k] = (1 - w) * vsrc[j] + w * vsrc[j+1]
            end
        end
    end
    return out
end

# ── the MEOP list and files ─────────────────────────────────────────────────────────────────────
"Every profile in the box from list_profiles.csv (all years): tag, deployment, cycle, time, position, has-S."
function read_list()
    path = joinpath(MEOP, "list_profiles.csv")
    isfile(path) || error("$path not found")
    lines = readlines(path)
    hdr = split(lines[1], ',')
    col(n) = something(findfirst(==(n), hdr), 0)
    ic, id, iy, ij, ila, ilo, ins = col.(("SMRU_PLATFORM_CODE", "DEPLOYMENT_CODE", "CYCLE_NUMBER", "JULD",
                                          "LATITUDE", "LONGITUDE", "N_PSAL"))
    any(==(0), (ic, id, iy, ij, ila, ilo, ins)) && error("$path: unexpected header $(lines[1])")
    L = (code = String[], dep = String[], cyc = Int[], time = DateTime[], lat = Float64[], lon = Float64[], hasS = Bool[])
    nbad = 0
    for ln in @view lines[2:end]
        f = split(ln, ',')
        length(f) < length(hdr) && (nbad += 1; continue)
        lat, lon = tryparse(Float64, f[ila]), tryparse(Float64, f[ilo])
        (lat === nothing || lon === nothing) && (nbad += 1; continue)
        in_box(lon, lat) || continue
        t = tryparse(DateTime, f[ij], dateformat"yyyy-mm-dd HH:MM:SS")
        t === nothing && (nbad += 1; continue)
        cy = tryparse(Float64, f[iy])
        ns = tryparse(Float64, f[ins])
        push!(L.code, f[ic]); push!(L.dep, f[id]); push!(L.cyc, cy === nothing ? -1 : round(Int, cy))
        push!(L.time, t); push!(L.lat, lat); push!(L.lon, lon); push!(L.hasS, ns !== nothing && ns > 0)
    end
    @info "list_profiles.csv: $(length(lines) - 1) profiles, $(length(L.time)) in the box, $nbad lines unreadable"
    return L
end

"tag => (path to <COUNTRY>/DATA/<tag>_all_prof.nc, country)"
function meop_index()
    idx = Dict{String, Tuple{String, String}}()
    for c in readdir(MEOP)
        d = joinpath(MEOP, c, "DATA"); isdir(d) || continue
        for f in readdir(d)
            endswith(f, "_all_prof.nc") && (idx[f[1:end-12]] = (joinpath(d, f), c))
        end
    end
    return idx
end

function rd(ds, v)
    a = haskey(ds, v * "_ADJUSTED") ? v * "_ADJUSTED" : v
    return f64(ds[a][:, :]), coalesce.(ds[a * "_QC"][:, :], '9')
end

"One variable of profile n on the depths dt; counts its levels (nlev[slot]) and QC rejections (nlev[slot+1])."
function profile_on(P, Pq, V, Vq, n, lat, dt, nlev, slot)
    ks = [k for k in axes(V, 1) if isfinite(P[k, n]) && isfinite(V[k, n])]
    nlev[slot] += length(ks)
    kg = filter(k -> good(Pq[k, n]) && good(Vq[k, n]), ks)
    nlev[slot+1] += length(ks) - length(kg)
    isempty(kg) && return fill(NaN, length(dt))
    d = saunders.(P[kg, n], lat); v = V[kg, n]
    o = sortperm(d); d = d[o]; v = v[o]
    u = [1; findall(diff(d) .> 0) .+ 1]
    return interp_to(d[u], v[u], dt, TOPGAP)
end

"The profiles of `codes` in the box and YEARS, on B-SOSE's level centres g.d (top-down)."
function read_meop(codes, idx, g, L)
    ref = Dict(zip(zip(L.code, L.cyc), L.time))
    lon, lat, tim, dep = Float64[], Float64[], DateTime[], String[]
    Tc, Sc = Vector{Vector{Float64}}(), Vector{Vector{Float64}}()
    nlev = zeros(Int, 4)                # T levels, T rejected, S levels, S rejected
    nmatch, dtmax, npos = Ref(0), Ref(0.0), Ref(0)
    for c in codes
        haskey(idx, c) || continue
        NCDataset(idx[c][1]) do ds
            refd = String(coalesce.(ds["REFERENCE_DATE_TIME"][:], ' '))
            startswith(refd, "19500101") || error("$(idx[c][1]): REFERENCE_DATE_TIME $refd, expected 1950-01-01")
            la, lo = f64(ds["LATITUDE"][:]), f64(ds["LONGITUDE"][:])
            jd = Float64.(NCDatasets.variable(ds, "JULD")[:])           # raw days, 99999 = fill
            pq = haskey(ds, "POSITION_QC") ? coalesce.(ds["POSITION_QC"][:], '9') : fill('1', length(la))
            cy = coalesce.(ds["CYCLE_NUMBER"][:], -1)
            P, Pq = rd(ds, "PRES"); T, Tq = rd(ds, "TEMP")
            S, Sq = haskey(ds, "PSAL") ? rd(ds, "PSAL") : (fill(NaN, size(T)), fill('9', size(T)))
            for n in eachindex(la)
                (isfinite(la[n]) && isfinite(lo[n]) && jd[n] < 9e4) || continue
                in_box(lo[n], la[n]) || continue
                pq[n] in ('3', '4', '9') && (npos[] += 1; continue)
                t = DateTime(1950) + Millisecond(round(Int64, jd[n] * 86_400_000))
                year(t) in YEARS || continue
                tt = get(ref, (c, cy[n]), nothing)
                tt === nothing || (nmatch[] += 1; dtmax[] = max(dtmax[], abs(Dates.value(tt - t)) / 60_000))
                pT = profile_on(P, Pq, T, Tq, n, la[n], g.d, nlev, 1)
                pS = profile_on(P, Pq, S, Sq, n, la[n], g.d, nlev, 3)
                (any(isfinite, pT) || any(isfinite, pS)) || continue
                push!(lon, lo[n]); push!(lat, la[n]); push!(tim, t); push!(dep, c); push!(Tc, pT); push!(Sc, pS)
            end
        end
    end
    @info @sprintf("MEOP read: %d profiles; levels T %d (%.1f%% rejected by QC), S %d (%.1f%% rejected); %d dropped for position QC",
                   length(tim), nlev[1], 100nlev[2] / max(nlev[1], 1), nlev[3], 100nlev[4] / max(nlev[3], 1), npos[])
    @info @sprintf("MEOP times against list_profiles.csv: %d matched by (tag, cycle), largest difference %.1f min",
                   nmatch[], dtmax[])
    nk = length(g.d)
    T = isempty(Tc) ? zeros(nk, 0) : reduce(hcat, Tc); S = isempty(Sc) ? zeros(nk, 0) : reduce(hcat, Sc)
    return (; lon, lat, time = tim, tag = dep, T, S)
end

# ── B-SOSE ──────────────────────────────────────────────────────────────────────────────────────
"B-SOSE's grid over the box rows: level centres g.d and thicknesses g.dr to ZMAX, and the column depth hb."
function bsose_grid()
    tp, sp = bsose_path(BSOSE_DIR, "Theta"), bsose_path(BSOSE_DIR, "Salt")
    (isempty(tp) || isempty(sp)) && error("no Theta/Salt_bsoseI156_2013to2024_monthly.nc in $BSOSE_DIR")
    xc, yc, rf = NCDataset(tp) do ds
        xc = f64(ds["XC"][:]); yc = f64(ds["YC"][:])
        xc = ndims(xc) == 1 ? xc : xc[:, 1]; yc = ndims(yc) == 1 ? yc : yc[1, :]
        rf = haskey(ds, "RF") ? f64(ds["RF"][:]) : [0.0; -cumsum(f64(ds[haskey(ds, "drF") ? "drF" : "DRF"][:]))]
        xc, yc, rf
    end
    dc = [-(rf[k] + rf[k+1]) / 2 for k in 1:length(rf)-1]
    nk = findlast(<=(ZMAX), dc)
    jr = findfirst(>=(LATS[1] - 0.5), yc):findlast(<=(LATS[2] + 0.5), yc)
    S1 = NCDataset(ds -> f64(ds[bsose_data_var(ds, ("SALT", "Salt"), sp)][:, jr, :, 1]), sp)
    hb = [(kb = findlast(s -> isfinite(s) && s != 0, view(S1, i, j, :))) === nothing ? 0.0 : -rf[kb+1]
          for i in axes(S1, 1), j in axes(S1, 2)]
    return (; xc, yc = yc[jr], jr, nk, d = dc[1:nk], dr = -diff(rf)[1:nk], hb)
end

"Nearest cell with iswet(i, j) within SEARCH cells of (x, y), by distance; nothing if none."
function nearest_wet(xs, ys, x, y, iswet)
    i0, j0 = argmin(abs.(xs .- x)), argmin(abs.(ys .- y))
    best, bd = nothing, Inf
    for j in max(1, j0 - SEARCH):min(length(ys), j0 + SEARCH), i in max(1, i0 - SEARCH):min(length(xs), i0 + SEARCH)
        iswet(i, j) || continue
        dd = ((xs[i] - x) * cosd(y))^2 + (ys[j] - y)^2
        dd < bd && (bd = dd; best = (i, j))
    end
    return best
end

"B-SOSE columns at profile cells: groups maps (Tpath, Spath, record) => [(profile, i, j)], j within g.jr."
function sample_bsose!(MT, MS, groups, g)
    tlast = time()
    for (q, ((tp, sp, n), items)) in enumerate(groups)
        if time() - tlast > 60
            @info "  B-SOSE: record $q of $(length(groups)) ($(basename(tp)))"; tlast = time()
        end
        Tb = NCDataset(ds -> f64(ds[bsose_data_var(ds, ("THETA", "Theta"), tp)][:, g.jr, 1:g.nk, n]), tp)
        Sb = NCDataset(ds -> f64(ds[bsose_data_var(ds, ("SALT", "Salt"), sp)][:, g.jr, 1:g.nk, n]), sp)
        for (p, i, j) in items, k in 1:g.nk
            s = Sb[i, j, k]
            (isfinite(s) && s != 0) || continue
            MT[k, p] = Tb[i, j, k]; MS[k, p] = s
        end
    end
    return MT, MS
end

# ── a run ───────────────────────────────────────────────────────────────────────────────────────
"The profile's time moved to year Y (Feb 29 -> Feb 28)."
in_year(t, Y) = DateTime(Y, month(t), min(day(t), daysinmonth(Y, month(t)))) + (t - floor(t, Day))

"Does the profile's calendar date and time of day fall in (w0, w1] in any of the window's years?"
in_window_doy(t, w0, w1) = any(Y -> w0 < in_year(t, Y) <= w1, year(w0):year(w1))

"""
A run's 30-day means at the profiles' cells, on B-SOSE's levels; and the run's span. clim = true: every profile, any
year, takes the mean of the run's windows that hold its CALENDAR date (one per run year).
"""
function sample_run(run, P, g; clim = false)
    np = length(P.time)
    MT, MS = fill(NaN, g.nk, np), fill(NaN, g.nk, np)
    sT, sS, nT, nS = zeros(g.nk, np), zeros(g.nk, np), zeros(Int, g.nk, np), zeros(Int, g.nk, np)
    path = joinpath(DIR, "$(run)_eke.nc")
    isfile(path) || error("$path not found (set DIR)")
    eke = NCDataset(path)
    λ, φ = f64(eke["λ_caa"][:]), f64(eke["φ_aca"][:])
    lv = file_levels(eke); lv === nothing && error("$path: no z_aac/z_aaf levels")
    zc, zf, _ = lv
    kbot = file_kbot(eke, zc, zf)[1]; Nz = length(zc)
    start = DateTime(string(eke.attrib["start_date"]))
    win = haskey(eke.attrib, "averaging_window_seconds") ? Float64(eke.attrib["averaging_window_seconds"]) : 30 * 86400.0
    tend = f64(eke["time"][:]); keepn = findall(t -> t >= 0.99win, tend)
    wins = [(start + Millisecond(round(Int, 1000 * (tend[n] - win))), start + Millisecond(round(Int, 1000 * tend[n])), n)
            for n in keepn]
    isempty(wins) && error("$path: no complete 30-day window")
    cells = [nearest_wet(λ, φ, P.lon[p], P.lat[p], (i, j) -> kbot[i, j] !== nothing) for p in 1:np]
    nw = 0
    for (w0, w1, n) in wins
        ps = clim ? [p for p in 1:np if cells[p] !== nothing && in_window_doy(P.time[p], w0, w1)] :
                    [p for p in 1:np if w0 < P.time[p] <= w1 && cells[p] !== nothing]
        isempty(ps) && continue
        nw += 1
        Tr = f64(eke["T"][:, :, :, n]); Sr = f64(eke["S"][:, :, :, n])
        for p in ps
            i, j = cells[p]; ks = Nz:-1:kbot[i, j]
            dsrc = -zc[ks]
            t = interp_to(dsrc, Tr[i, j, ks], g.d, Inf); s = interp_to(dsrc, Sr[i, j, ks], g.d, Inf)
            if clim
                for k in 1:g.nk
                    isfinite(t[k]) && (sT[k, p] += t[k]; nT[k, p] += 1)
                    isfinite(s[k]) && (sS[k, p] += s[k]; nS[k, p] += 1)
                end
            else
                MT[:, p] = t; MS[:, p] = s
            end
        end
    end
    close(eke)
    if clim
        MT = ifelse.(nT .> 0, sT ./ max.(nT, 1), NaN); MS = ifelse.(nS .> 0, sS ./ max.(nS, 1), NaN)
    end
    span = (wins[1][1], wins[end][2])
    @info @sprintf("%s%s: %d windows %s to %s, %d with profiles; sampled at %d profiles, %d with no wet cell nearby",
                   run, clim ? " (calendar-date climatology)" : "", length(wins), Date(span[1]), Date(span[2]), nw, count(p -> any(isfinite, view(MT, :, p)), 1:np),
                   count(isnothing, cells))
    return MT, MS, span
end

# ── statistics ──────────────────────────────────────────────────────────────────────────────────
layer_levels(g, a, b) = [k for k in 1:g.nk if a <= g.d[k] < b]

"dz-weighted layer means of the obs and of (model - obs) over the levels where the obs and every model are finite."
function layer_vals(p, O, Ms, lev, dr)
    w, o, d = 0.0, 0.0, zeros(length(Ms))
    for k in lev
        (isfinite(O[k, p]) && all(M -> isfinite(M[k, p]), Ms)) || continue
        w += dr[k]; o += dr[k] * O[k, p]
        for (q, M) in enumerate(Ms)
            d[q] += dr[k] * (M[k, p] - O[k, p])
        end
    end
    return w == 0 ? nothing : (o / w, d ./ w)
end

function agg(ps, O, Ms, lev, dr)
    n, o, d = 0, 0.0, zeros(length(Ms))
    for p in ps
        v = layer_vals(p, O, Ms, lev, dr); v === nothing && continue
        n += 1; o += v[1]; d .+= v[2]
    end
    return n, o / max(n, 1), d ./ max(n, 1)
end

cellT(n, o, d) = n == 0 ? " " * " "^5 * "     --" * " "^(7length(d)) :
    @sprintf(" %5d %6.2f", n, o) * join(@sprintf(" %+6.2f", x) for x in d)
cellS(n, o, d) = n == 0 ? " " * " "^5 * "      --" * " "^(7length(d)) :
    @sprintf(" %5d %7.3f", n, o) * join(@sprintf(" %+6.3f", x) for x in d)

"""
Tables for one part: `models` is [(label, MT, MS)], `sel` the profiles. Region x season x layer, then shelf + slope
by sector and season over SECTOR_LAYER.
"""
function report(title, models, sel, P, g)
    labs = [m[1] for m in models]; MTs = [m[2] for m in models]; MSs = [m[3] for m in models]
    hdrT = " " * lpad("n", 5) * lpad("obs T", 7) * join(lpad("Δ" * l, 7) for l in labs)
    hdrS = " " * lpad("n", 5) * lpad("obs S", 8) * join(lpad("Δ" * l, 7) for l in labs)
    println("\n  ", title)
    println("  Δ = model - obs, dz-weighted layer means per profile, then the mean over profiles; every column on the same levels")
    @printf("  %-17s %-3s %-9s |%s |%s\n", "region", "", "layer", hdrT, hdrS)
    for r in 1:3, s in 1:4
        ps = [p for p in sel if P.region[p] == r && P.season[p] == s]
        isempty(ps) && continue
        for l in 1:length(LAYERS)-1
            lev = layer_levels(g, LAYERS[l], LAYERS[l+1])
            nT, oT, dT = agg(ps, P.T, MTs, lev, g.dr); nS, oS, dS = agg(ps, P.S, MSs, lev, g.dr)
            nT + nS == 0 && continue
            @printf("  %-17s %-3s %-9s |%s |%s\n", REGIONS[r], SEASONS[s], @sprintf("%.0f-%.0f m", LAYERS[l], LAYERS[l+1]),
                    cellT(nT, oT, dT), cellS(nS, oS, dS))
        end
    end
    lev = layer_levels(g, SLAYER...)
    println(@sprintf("\n  shelf + slope by sector, %.0f-%.0f m", SLAYER[1], SLAYER[2]))
    @printf("  %-9s %-3s |%s |%s\n", "sector", "", hdrT, hdrS)
    for c in 1:NSEC, s in 1:4
        ps = [p for p in sel if P.region[p] in (1, 2) && P.sector[p] == c && P.season[p] == s]
        isempty(ps) && continue
        nT, oT, dT = agg(ps, P.T, MTs, lev, g.dr); nS, oS, dS = agg(ps, P.S, MSs, lev, g.dr)
        nT + nS == 0 && continue
        @printf("  %-9s %-3s |%s |%s\n", sector_name(c), SEASONS[s], cellT(nT, oT, dT), cellS(nS, oS, dS))
    end
    flush(stdout)
end

# ── figures ─────────────────────────────────────────────────────────────────────────────────────
const RCOL = ("#0072B2", "#009E73", "#CC79A7")
const RUNCOL = ("#0072B2", "#009E73", "#CC79A7", "#56B4E9")

function map_figure(P, g, sel_b)
    ib = findall(x -> LONS[1] - 1 <= x <= LONS[2] + 1, g.xc)
    panels = isempty(sel_b) ? [(1:length(P.time), "$(first(YEARS))-$(last(YEARS))")] :
             [(1:length(P.time), "$(first(YEARS))-$(last(YEARS))"), (sel_b, "runs' span")]
    fig = Figure(size = (900, 330 * length(panels) + 40), fontsize = 13)
    Label(fig[0, 1], "MEOP profiles by region (B-SOSE depth at the cell); contours: coast, $(Int(ISOBATHS[1])) m, $(Int(ISOBATHS[2])) m";
          font = :bold)
    for (r, (ps, name)) in enumerate(panels)
        ax = Axis(fig[r, 1]; title = "$name: $(length(ps)) profiles", xlabel = "°E", ylabel = "°N", aspect = DataAspect())
        contour!(ax, g.xc[ib], g.yc, g.hb[ib, :]; levels = [1.0], color = :black, linewidth = 1)
        contour!(ax, g.xc[ib], g.yc, g.hb[ib, :]; levels = ISOBATHS, color = :gray55, linewidth = 0.7)
        for k in 1:3
            q = [p for p in ps if P.region[p] == k]
            isempty(q) || scatter!(ax, P.lon[q], P.lat[q]; color = (RCOL[k], 0.5), markersize = 3, label = REGIONS[k])
        end
        xlims!(ax, LONS...); ylims!(ax, LATS[1] - 0.5, LATS[2])
        r == 1 && axislegend(ax; position = :rt, backgroundcolor = (:white, 0.85))
    end
    out = joinpath(OUTDIR, "meop_map.png"); save(out, fig); @info "wrote $out"
end

function profile_figure(models, sel, P, g, tag)
    styles = [(m[1], m[1] == "B5d" ? ("#E69F00", :dot) : m[1] == "Bmon" ? ("#D55E00", :dash) :
               (RUNCOL[mod1(i - count(x -> x[1] in ("B5d", "Bmon"), models), 4)], :solid)) for (i, m) in enumerate(models)]
    rows = [(r, v) for r in 1:3 for v in (:T, :S) if any(p -> P.region[p] == r, sel)]
    isempty(rows) && return
    fig = Figure(size = (1300, 300 * length(rows) + 80), fontsize = 12)
    Label(fig[0, 1:4], "Mean profiles at the MEOP profiles: obs (black) and models, same profiles at each level ($tag)"; font = :bold)
    for (ri, (r, v)) in enumerate(rows), s in 1:4
        ps = [p for p in sel if P.region[p] == r && P.season[p] == s]
        ax = Axis(fig[ri, s]; title = "$(REGIONS[r]), $(SEASONS[s]): $(length(ps))", yreversed = true,
                  xlabel = v == :T ? "T (°C)" : "S", ylabel = s == 1 ? "depth (m)" : "")
        O = v == :T ? P.T : P.S; Ms = [v == :T ? m[2] : m[3] for m in models]
        prof = [Float64[] for _ in 1:length(models) + 1]; dk = Float64[]
        for k in 1:g.nk
            q = [p for p in ps if isfinite(O[k, p]) && all(M -> isfinite(M[k, p]), Ms)]
            length(q) < 3 && continue
            push!(dk, g.d[k]); push!(prof[1], mean(O[k, q]))
            for (i, M) in enumerate(Ms)
                push!(prof[i+1], mean(M[k, q]))
            end
        end
        isempty(dk) && continue
        for (i, (lab, (c, ls))) in enumerate(styles)
            lines!(ax, prof[i+1], dk; color = c, linestyle = ls, linewidth = 1.8, label = lab)
        end
        lines!(ax, prof[1], dk; color = :black, linewidth = 2.2, label = "MEOP")
    end
    Legend(fig[length(rows) + 1, 1:4], [[LineElement(color = c, linestyle = ls, linewidth = 2) for (_, (c, ls)) in styles];
                LineElement(color = :black, linewidth = 2.2)], [[s[1] for s in styles]; "MEOP"]; orientation = :horizontal)
    out = joinpath(OUTDIR, "meop_profiles$(tag == "" ? "" : "_" * tag).png"); save(out, fig); @info "wrote $out"
end

# ── main ────────────────────────────────────────────────────────────────────────────────────────
function main()
    L = read_list()
    isempty(L.time) && error("no MEOP profiles in $(LONS)E $(LATS)N")
    idx = meop_index()
    @info "MEOP files: $(length(idx)) tags with a DATA/*_all_prof.nc"

    bar = "="^118
    println("\n", bar)
    println("MEOP-CTD profiles in $(Int(LONS[1]))-$(Int(LONS[2]))E, $(-Int(LATS[1]))-$(-Int(LATS[2]))S (list_profiles.csv, all years): T profiles / with S")
    println(bar)
    yrs = sort(unique(year.(L.time)))
    @printf("  %-5s |%s |   total\n", "year", join(lpad(sector_name(c), 12) for c in 1:NSEC))
    for y in yrs
        cnt(c) = (q = [p for p in eachindex(L.time) if year(L.time[p]) == y && sector(L.lon[p]) == c];
                  isempty(q) ? "-" : "$(length(q))/$(count(L.hasS[q]))")
        @printf("  %-5d |%s | %7d\n", y, join(lpad(cnt(c), 12) for c in 1:NSEC), count(t -> year(t) == y, L.time))
    end

    sel = [p for p in eachindex(L.time) if year(L.time[p]) in YEARS]
    codes = unique(L.code[sel])
    miss = filter(c -> !haskey(idx, c), codes)
    isempty(miss) || @warn "$(length(miss)) tags in the CSV have no DATA file: $(join(first(miss, 10), ", "))"
    println("\n  deployments with profiles in the box, $(first(YEARS))-$(last(YEARS))")
    println("  deployment  country        tags  profiles  first       last")
    for d in sort(unique(L.dep[sel]))
        q = [p for p in sel if L.dep[p] == d]
        cs = unique(L.code[q])
        ci = findfirst(c -> haskey(idx, c), cs); ctry = ci === nothing ? "?" : idx[cs[ci]][2]
        @printf("  %-10s  %-13s %5d %9d  %s  %s\n", d, ctry, length(cs), length(q), Date(minimum(L.time[q])), Date(maximum(L.time[q])))
    end
    flush(stdout)

    g = bsose_grid()
    P0 = read_meop(codes, idx, g, L)
    np = length(P0.time); np == 0 && error("no usable MEOP profiles in the box for $YEARS")
    cells = [nearest_wet(g.xc, g.yc, P0.lon[p], P0.lat[p], (i, j) -> g.hb[i, j] > 0) for p in 1:np]
    nodry = count(isnothing, cells)
    nodry > 0 && @warn "$nodry profiles have no wet B-SOSE cell within $SEARCH cells; dropped"
    keep = findall(!isnothing, cells)
    P = (; lon = P0.lon[keep], lat = P0.lat[keep], time = P0.time[keep], tag = P0.tag[keep], T = P0.T[:, keep], S = P0.S[:, keep],
         ib = [cells[p][1] for p in keep], jb = [cells[p][2] for p in keep],
         region = [region(g.hb[cells[p]...]) for p in keep], season = [season(month(P0.time[p])) for p in keep],
         sector = [sector(P0.lon[p]) for p in keep])
    np = length(P.time)
    println("\n  profiles read, by region and season ($(first(YEARS))-$(last(YEARS))): T / with S")
    @printf("  %-17s |%s\n", "region", join(lpad(s, 12) for s in SEASONS))
    for r in 1:3
        cnt(s) = (q = [p for p in 1:np if P.region[p] == r && P.season[p] == s];
                  "$(count(p -> any(isfinite, view(P.T, :, p)), q))/$(count(p -> any(isfinite, view(P.S, :, p)), q))")
        @printf("  %-17s |%s\n", REGIONS[r], join(lpad(cnt(s), 12) for s in 1:4))
    end
    flush(stdout)

    # A: B-SOSE monthly
    mT, mS = bsose_path(BSOSE_DIR, "Theta"), bsose_path(BSOSE_DIR, "Salt")
    mstamp = NCDataset(ds -> DateTime.(ds["time"][:]), mT)
    mkey = [(year(s - Day(14)), month(s - Day(14))) for s in mstamp]       # monthly stamps are month ends
    groups = Dict{Tuple{String, String, Int}, Vector{NTuple{3, Int}}}()
    for p in 1:np
        n = findfirst(==((year(P.time[p]), month(P.time[p]))), mkey); n === nothing && continue
        push!(get!(groups, (mT, mS, n), NTuple{3, Int}[]), (p, P.ib[p], P.jb[p]))
    end
    BmT, BmS = sample_bsose!(fill(NaN, g.nk, np), fill(NaN, g.nk, np), groups, g)
    @info "B-SOSE monthly sampled at $(count(p -> any(isfinite, view(BmS, :, p)), 1:np)) of $np profiles"

    println("\n", bar)
    println("MEOP against B-SOSE", isempty(RUNS) ? "" : " and $(join(RUNS, ", "))",
            ". T in situ (obs) vs potential (models); S practical. Obs on B-SOSE levels, surface fill <= $(TOPGAP) m")
    println(bar)
    report("A. $(first(YEARS))-$(last(YEARS)): B-SOSE MONTHLY MEAN of the profile's month (Bmon)", [("Bmon", BmT, BmS)], 1:np, P, g)

    mkpath(OUTDIR)
    sel_b = Int[]
    if isempty(RUNS)
        map_figure(P, g, sel_b)
        profile_figure([("Bmon", BmT, BmS)], 1:np, P, g, "")
        return
    end

    # B: the runs' common span
    rs = [sample_run(r, P, g) for r in RUNS]
    span = (maximum(r[3][1] for r in rs), minimum(r[3][2] for r in rs))
    sel_b = [p for p in 1:np if span[1] < P.time[p] <= span[2]]
    B5T, B5S = fill(NaN, g.nk, np), fill(NaN, g.nk, np)
    g5 = Dict{Tuple{String, String, Int}, Vector{NTuple{3, Int}}}()
    for y in year(span[1]):year(span[2])
        tp, sp = bsose_path(BSOSE_DIR, "Theta", y), bsose_path(BSOSE_DIR, "Salt", y)
        (isempty(tp) || isempty(sp)) && (@warn "no 5-day Theta/Salt for $y"; continue)
        st = NCDataset(ds -> DateTime.(ds["time"][:]), tp)
        for p in sel_b
            year(P.time[p]) == y || continue
            n = findfirst(s -> s - Day(5) < P.time[p] <= s, st); n === nothing && continue
            push!(get!(g5, (tp, sp, n), NTuple{3, Int}[]), (p, P.ib[p], P.jb[p]))
        end
    end
    sample_bsose!(B5T, B5S, g5, g)
    n5 = count(p -> any(isfinite, view(B5S, :, p)), sel_b)
    models = Tuple{String, Matrix{Float64}, Matrix{Float64}}[("Bmon", BmT, BmS)]
    n5 > 0 ? pushfirst!(models, ("B5d", B5T, B5S)) : @warn "no B-SOSE 5-day records for the runs' span: part B without B5d"
    for (i, r) in enumerate(RUNS)
        push!(models, ("run$i", rs[i][1], rs[i][2]))
    end
    println("\n  runs: ", join(("run$i = $r" for (i, r) in enumerate(RUNS)), ", "))
    report("B. $(Date(span[1])) to $(Date(span[2])) ($(length(sel_b)) profiles): B-SOSE 5-day (B5d), monthly (Bmon), runs' 30-day means",
           models, sel_b, P, g)
    println()
    map_figure(P, g, sel_b)
    profile_figure(models, sel_b, P, g, join(RUNS, "_"))

    # C: every profile of YEARS against the runs' calendar-date climatology (the mean of their windows holding the
    # profile's date, one per run year), and B-SOSE's monthly climatology over the runs' months (Bclim: the mean of the
    # profile's calendar month in each run year). Bclim - Bmon is what mixing years costs B-SOSE: read the runs' Δ
    # against Bclim's, not Bmon's.
    rc = [sample_run(r, P, g; clim = true) for r in RUNS]
    spanm = Set{Tuple{Int,Int}}()
    let d = DateTime(year(span[1]), month(span[1]))
        while d <= span[2]
            # a month counts if the runs cover at least half of it
            a, b = max(d, span[1]), min(d + Month(1), span[2])
            2 * Dates.value(b - a) >= Dates.value(d + Month(1) - d) && push!(spanm, (year(d), month(d)))
            d += Month(1)
        end
    end
    slots = Dict{Int, Vector{Int}}()                     # calendar month => the monthly records of it in the runs' span
    for (n, k) in enumerate(mkey)
        k in spanm && push!(get!(slots, k[2], Int[]), n)
    end
    nslot = maximum(length, values(slots))
    sTc, sSc, cT, cS = zeros(g.nk, np), zeros(g.nk, np), zeros(Int, g.nk, np), zeros(Int, g.nk, np)
    for q in 1:nslot
        gq = Dict{Tuple{String, String, Int}, Vector{NTuple{3, Int}}}()
        for p in 1:np
            ns = get(slots, month(P.time[p]), Int[]); q <= length(ns) || continue
            push!(get!(gq, (mT, mS, ns[q]), NTuple{3, Int}[]), (p, P.ib[p], P.jb[p]))
        end
        QT, QS = sample_bsose!(fill(NaN, g.nk, np), fill(NaN, g.nk, np), gq, g)
        f = isfinite.(QT); sTc[f] .+= QT[f]; cT .+= f
        f = isfinite.(QS); sSc[f] .+= QS[f]; cS .+= f
    end
    BcT = ifelse.(cT .> 0, sTc ./ max.(cT, 1), NaN); BcS = ifelse.(cS .> 0, sSc ./ max.(cS, 1), NaN)
    modelsC = Tuple{String, Matrix{Float64}, Matrix{Float64}}[("Bmon", BmT, BmS), ("Bclim", BcT, BcS)]
    for (i, r) in enumerate(RUNS)
        push!(modelsC, ("run$i", rc[i][1], rc[i][2]))
    end
    report("C. $(first(YEARS))-$(last(YEARS)), every profile at its CALENDAR date: B-SOSE monthly of its own month (Bmon), " *
           "B-SOSE's monthly climatology over the runs' months (Bclim), the runs' calendar-date climatology",
           modelsC, 1:np, P, g)
    println("  (Bclim - Bmon = the cost of mixing years, for B-SOSE; a run's Δ minus Bclim's Δ = the run against B-SOSE, same years)")
    profile_figure(modelsC, 1:np, P, g, "clim_" * join(RUNS, "_"))
end

main()
