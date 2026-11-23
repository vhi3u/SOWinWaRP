# ==============================================================================================
# compare_spira_mld.jl -- is B-SOSE's mixed layer right? Observed profiles (Spira et al. 2024, zenodo 10258138:
# Argo, MEOP, SOCCOM, WOD18 ship and glider CTDs, QC'd, 2 dbar) against B-SOSE, and optionally a run, sampled at
# each profile's cell and time, by latitude band and season over 90-150E.
#
#     BSOSE_DIR=/storage/scratch1/1/vnguyen480/SOWinWaRP/data julia --project=. tools/compare_spira_mld.jl
#     BSOSE_DIR=... julia --project=. tools/compare_spira_mld.jl bsose_ggl_hj_pr207_icein_mitbulk     # + the run
#
# WHY. Under hourly or 6-hourly ERA5 every closure we have mixes the summer open-water layer 1.4-1.6x deeper than
# B-SOSE (single column, 52S: 75-92 m against 55-57), while under B-SOSE's own 5-day stress CATKE matches it. B-SOSE
# fits Argo through its adjoint (winds, initial state), so before tuning anything to it: is its summer MLD what the
# floats see?
#
# MLD: tools/mld.jl on every side (potential density at 0 dbar, TEOS-10 polynomial fed the file's T and S as-is,
# 0.03 kg/m³ from the 10 m level, first level past it). Profiles: z = -pres (dbar ~ m), their own 2 dbar levels;
# the file's own `mld` is printed beside ours as a check. A profile with no crossing above its deepest valid level is
# dropped from the pairs (counted as "no-cross").
# SAMPLING: each profile's nearest B-SOSE cell, and
#   A  2013-2021 (B-SOSE monthly x Spira): the B-SOSE MONTHLY MEAN of the profile's calendar month
#   B  the run's span (with a run name): B-SOSE's 5-day record holding the profile's time, B-SOSE monthly as in A,
#      and the run's 30-day mean (_eke.nc) holding it, the run's nearest cell
# A profile is an instant; B-SOSE monthly and the run's 30-day means are MEAN STATES, whose MLD is shallower than the
# mean instantaneous MLD in a variable season (compare_mld's Jensen check: 6-9 m in winter). The 5-day records in B are
# the closest B-SOSE gets to an instant: read obs against B-SOSE 5-day, run against B-SOSE monthly.
#
# PRINTS per band and season (DJF, MAM, JJA, SON): profiles, no-cross %, medians of obs (ours, the file's) and of
# each model at the same profiles, and the median of model/obs per profile. Then DJF by month for part A.
# WRITES OUTDIR/spira_mld[_<RUN>].png: per band, monthly medians, obs against B-SOSE (and the run).
# LOGIN-NODE SAFE: profiles are read in runs of nearby indices, only those in the box; B-SOSE one record at a time.
# ENV: BSOSE_DIR (required), SPIRA (default BSOSE_DIR/spira/qc_profile_ds_2dbar.nc), DIR (run files, .), OUTDIR
#      (plots), BANDS (-70,-65,-60,-55,-50,-45), YEARS (2013:2021), ZMAX (1000 m, B-SOSE levels searched), MLD_DT
# ==============================================================================================

using NCDatasets, CairoMakie, Dates, Printf, Statistics

ENV["HDF5_USE_FILE_LOCKING"] = "FALSE"
include(joinpath(@__DIR__, "mld.jl"))           # mld, MLD_DT
include(joinpath(@__DIR__, "bsose_windows.jl")) # bsose_path, bsose_data_var
include(joinpath(@__DIR__, "model_grid.jl"))    # file_levels, file_kbot

const RUN       = isempty(ARGS) ? get(ENV, "RUN", "") : ARGS[1]
const BSOSE_DIR = get(ENV, "BSOSE_DIR", "")
isempty(BSOSE_DIR) && error("set BSOSE_DIR to the directory with B-SOSE's monthly and 5-day Theta and Salt")
const SPIRA  = get(ENV, "SPIRA", joinpath(BSOSE_DIR, "spira", "qc_profile_ds_2dbar.nc"))
const DIR    = get(ENV, "DIR", ".")
const OUTDIR = get(ENV, "OUTDIR", "plots")
const EDGES  = parse.(Float64, split(get(ENV, "BANDS", "-70,-65,-60,-55,-50,-45"), ','))
const YEARS  = eval(Meta.parse(get(ENV, "YEARS", "2013:2021")))
const ZMAX   = parse(Float64, get(ENV, "ZMAX", "1000"))
const BOX    = (90.0, 150.0, EDGES[1], EDGES[end])

f64(A) = Float64.(coalesce.(A, NaN))
band_name(b) = @sprintf("%.0f-%.0fS", -EDGES[b], -EDGES[b+1])
season(m) = m in (12, 1, 2) ? 1 : m in (3, 4, 5) ? 2 : m in (6, 7, 8) ? 3 : 4
const SEASONS = ("DJF", "MAM", "JJA", "SON")
firstkey(ds, names) = (k = findfirst(n -> haskey(ds, n), names); k === nothing ? error("none of $names in $(SPIRA)") : names[k])

# ── the profiles in the box ──────────────────────────────────────────────────────────────────────
function read_profiles()
    isfile(SPIRA) || error("$SPIRA not found (zenodo 10258138, qc_profile_ds_2dbar.nc); set SPIRA")
    NCDataset(SPIRA) do ds
        tv, sv = firstkey(ds, ("temp", "ctemp", "TEMP")), firstkey(ds, ("psal", "asal", "PSAL"))
        dn = dimnames(ds[tv]); length(dn) == 2 || error("$tv has dims $dn, expected (pres, n_prof)")
        presfirst = occursin("pres", lowercase(dn[1]))
        lon = mod.(f64(ds["lon"][:]), 360); lat = f64(ds["lat"][:]); tim = ds["time"][:]
        pres = f64(ds["pres"][:])
        sel = findall(n -> BOX[1] <= lon[n] <= BOX[2] && BOX[3] <= lat[n] < BOX[4] && !ismissing(tim[n]) &&
                           year(tim[n]) in YEARS, eachindex(lat))
        @info @sprintf("Spira: %d profiles, %d levels (%.0f-%.0f dbar); %d in 90-150E %.0f-%.0fS, %s",
                       length(lat), length(pres), pres[1], pres[end], length(sel), -BOX[4], -BOX[3], YEARS)
        T = fill(NaN, length(pres), length(sel)); S = similar(T)
        # read in runs of nearby indices (contiguous blocks), not profile by profile
        a = 1
        while a <= length(sel)
            b = a
            while b < length(sel) && sel[b+1] - sel[b] <= 64 && sel[b+1] - sel[a] < 4096; b += 1; end
            r = sel[a]:sel[b]; keep = sel[a:b] .- sel[a] .+ 1
            blkT = f64(presfirst ? ds[tv][:, r] : permutedims(ds[tv][r, :]))
            blkS = f64(presfirst ? ds[sv][:, r] : permutedims(ds[sv][r, :]))
            T[:, a:b] = blkT[:, keep]; S[:, a:b] = blkS[:, keep]
            a = b + 1
        end
        own = haskey(ds, "mld") ? f64(ds["mld"][sel]) : fill(NaN, length(sel))
        return (; lon = lon[sel], lat = lat[sel], time = DateTime.(tim[sel]), pres, T, S, own, tv, sv)
    end
end

"MLD of one top-down profile on pressure levels: tools/mld.jl, down to its deepest valid level."
function profile_mld(Tc, Sc, pres)
    kdeep = findlast(k -> isfinite(Tc[k]) && isfinite(Sc[k]), eachindex(pres))
    kdeep === nothing && return NaN
    n = kdeep
    return mld(reverse(Tc[1:n]), reverse(Sc[1:n]), -reverse(pres[1:n]), 1)
end

# ── B-SOSE sampled at cells, one record at a time ────────────────────────────────────────────────
struct BGrid; XC; YC; RF; nk; jr; end
function bsose_grid()
    p = bsose_path(BSOSE_DIR, "Theta"); isempty(p) && error("no Theta_bsoseI156_2013to2024_monthly.nc in $BSOSE_DIR")
    NCDataset(p) do ds
        xc = f64(ds["XC"][:]); yc = f64(ds["YC"][:]); xc = ndims(xc) == 1 ? xc : xc[:, 1]; yc = ndims(yc) == 1 ? yc : yc[1, :]
        rf = haskey(ds, "RF") ? f64(ds["RF"][:]) : [0.0; -cumsum(f64(ds[haskey(ds, "drF") ? "drF" : "DRF"][:]))]
        nk = something(findlast(k -> -rf[k+1] <= ZMAX, 1:length(rf)-1), 1)
        jr = findfirst(>=(BOX[3] - 0.5), yc):findlast(<=(BOX[4] + 0.5), yc)
        BGrid(xc, yc, rf, nk, jr)
    end
end

"""
MLD from B-SOSE at (i, j) cells for each record: `groups` maps (path, n) => [(profile, i, j)]. Each record is read
once, its box rows at full width (contiguous on disk). Land (S == 0) gives NaN.
"""
function bsose_mlds!(out, groups, g, tvar, svar)
    zs = [(g.RF[k] + g.RF[k+1]) / 2 for k in g.nk:-1:1]           # bottom-up centres, negative (RF is)
    for ((tpath, spath, n), items) in groups
        Tb = NCDataset(ds -> f64(ds[tvar(ds, tpath)][:, g.jr, 1:g.nk, n]), tpath)
        Sb = NCDataset(ds -> f64(ds[svar(ds, spath)][:, g.jr, 1:g.nk, n]), spath)
        for (p, i, j) in items
            jj = j - first(g.jr) + 1
            Sc = Sb[i, jj, :]; Tc = Tb[i, jj, :]
            (Sc[1] == 0 || !isfinite(Sc[1])) && continue
            kb = findlast(s -> s != 0 && isfinite(s), Sc)
            out[p] = mld(reverse(Tc[1:kb]), reverse(Sc[1:kb]), zs[end-kb+1:end], 1)
        end
    end
    return out
end

function main()
    P = read_profiles()
    np = length(P.lat)
    np == 0 && error("no Spira profiles in the box for $YEARS")
    obs = [profile_mld(P.T[:, p], P.S[:, p], P.pres) for p in 1:np]
    g = bsose_grid()
    ic = [argmin(abs.(g.XC .- x)) for x in P.lon]; jc = [argmin(abs.(g.YC .- y)) for y in P.lat]
    tvar(ds, path) = bsose_data_var(ds, ("THETA", "Theta"), path); svar(ds, path) = bsose_data_var(ds, ("SALT", "Salt"), path)

    # A: B-SOSE monthly, the profile's calendar month
    mT, mS = bsose_path(BSOSE_DIR, "Theta"), bsose_path(BSOSE_DIR, "Salt")
    mstamp = NCDataset(ds -> DateTime.(ds["time"][:]), mT)
    mkey = [(year(s - Day(14)), month(s - Day(14))) for s in mstamp]     # monthly stamps are month ends
    groups = Dict{Tuple{String, String, Int}, Vector{NTuple{3, Int}}}()
    for p in 1:np
        n = findfirst(==((year(P.time[p]), month(P.time[p]))), mkey); n === nothing && continue
        push!(get!(groups, (mT, mS, n), NTuple{3, Int}[]), (p, ic[p], jc[p]))
    end
    bmon = bsose_mlds!(fill(NaN, np), groups, g, tvar, svar)
    @info "B-SOSE monthly sampled at $(count(isfinite, bmon)) profiles"

    # B: the run's span
    have_run = !isempty(RUN)
    b5, runm = fill(NaN, np), fill(NaN, np)
    span = (DateTime(0), DateTime(0))
    if have_run
        eke = NCDataset(joinpath(DIR, "$(RUN)_eke.nc"))
        λ, φ = f64(eke["λ_caa"][:]), f64(eke["φ_aca"][:])
        zc, zf, _ = file_levels(eke); kbot = file_kbot(eke, zc, zf)[1]; Nz = length(zc)
        start = DateTime(string(eke.attrib["start_date"]))
        win = haskey(eke.attrib, "averaging_window_seconds") ? Float64(eke.attrib["averaging_window_seconds"]) : 30 * 86400.0
        tend = f64(eke["time"][:]); keepn = findall(t -> t >= 0.99win, tend)
        wins = [(start + Millisecond(round(Int, 1000 * (tend[n] - win))), start + Millisecond(round(Int, 1000 * tend[n])), n) for n in keepn]
        span = (wins[1][1], wins[end][2])
        lo = findfirst(k -> -zc[k] <= ZMAX, 1:Nz); zs = zc[lo:Nz]
        ir = [argmin(abs.(λ .- x)) for x in P.lon]; jrun = [argmin(abs.(φ .- y)) for y in P.lat]
        for (w0, w1, n) in wins
            ps = [p for p in 1:np if w0 < P.time[p] <= w1]; isempty(ps) && continue
            Tr = f64(eke["T"][:, :, lo:Nz, n]); Sr = f64(eke["S"][:, :, lo:Nz, n])
            for p in ps
                kb = kbot[ir[p], jrun[p]]; kb === nothing && continue
                runm[p] = mld(view(Tr, ir[p], jrun[p], :), view(Sr, ir[p], jrun[p], :), zs, max(kb - lo + 1, 1))
            end
        end
        close(eke)
        # B-SOSE 5-day record holding the profile's time
        g5 = Dict{Tuple{String, String, Int}, Vector{NTuple{3, Int}}}()
        for y in year(span[1]):year(span[2])
            tp, sp = bsose_path(BSOSE_DIR, "Theta", y), bsose_path(BSOSE_DIR, "Salt", y)
            (isempty(tp) || isempty(sp)) && (@warn "no 5-day Theta/Salt for $y"; continue)
            st = NCDataset(ds -> DateTime.(ds["time"][:]), tp)
            for p in 1:np
                (span[1] < P.time[p] <= span[2] && year(P.time[p]) == y) || continue
                n = findfirst(s -> s - Day(5) < P.time[p] <= s, st); n === nothing && continue
                push!(get!(g5, (tp, sp, n), NTuple{3, Int}[]), (p, ic[p], jc[p]))
            end
        end
        bsose_mlds!(b5, g5, g, tvar, svar)
        @info "$RUN: run sampled at $(count(isfinite, runm)), B-SOSE 5-day at $(count(isfinite, b5)) profiles in $(Date(span[1])) to $(Date(span[2]))"
    end

    # ── report ───────────────────────────────────────────────────────────────────────────────
    nb = length(EDGES) - 1
    band = [findfirst(b -> EDGES[b] <= y < EDGES[b+1], 1:nb) for y in P.lat]
    md(v) = isempty(v) ? NaN : median(v)
    function row(sel, models)
        o = obs[sel]; ok = isfinite.(o)
        nc = isempty(sel) ? NaN : 100 * count(!, ok) / length(sel)
        s = sel[ok]
        cols = [(md(filter(isfinite, P.own[s]))), md(obs[s])]
        for m in models
            pair = [p for p in s if isfinite(m[p])]
            push!(cols, md(m[pair])); push!(cols, md(m[pair] ./ obs[pair]))
        end
        return length(sel), nc, cols
    end
    fmt(x) = isfinite(x) ? @sprintf("%6.0f", x) : "    --"
    fr(x) = isfinite(x) ? @sprintf("%5.2f", x) : "   --"
    bar = "="^112
    println("\n", bar)
    println("MIXED LAYER DEPTH: Spira et al. profiles against B-SOSE", have_run ? " and $RUN" : "", " (tools/mld.jl, $(MLD_DT) kg/m³ from 10 m)")
    println("  medians over the profiles; model/obs = median of the per-profile ratio; no-cross % = obs profiles with no crossing (dropped)")
    println("  obs file = the dataset's own `mld` variable; obs = ours. Variables read: $(P.tv), $(P.sv)")
    println(bar)
    println("\n  A. $(first(YEARS))-$(last(YEARS)), B-SOSE MONTHLY MEAN of the profile's month (a mean state: reads shallower than an instant)")
    println("  band     season |  profiles no-cross% | obs file    obs | B-SOSE mon  ratio")
    for b in 1:nb, s in 1:4
        sel = [p for p in 1:np if band[p] == b && season(month(P.time[p])) == s]
        isempty(sel) && continue
        n, nc, c = row(sel, (bmon,))
        @printf("  %-8s %-6s | %9d %8.0f  | %s %s |   %s  %s\n", band_name(b), SEASONS[s], n, nc, fmt(c[1]), fmt(c[2]), fmt(c[3]), fr(c[4]))
    end
    println("\n  A, DJF by month")
    for b in 1:nb, m in (12, 1, 2)
        sel = [p for p in 1:np if band[p] == b && month(P.time[p]) == m]
        isempty(sel) && continue
        n, nc, c = row(sel, (bmon,))
        @printf("  %-8s month %2d | %9d %8.0f  | %s %s |   %s  %s\n", band_name(b), m, n, nc, fmt(c[1]), fmt(c[2]), fmt(c[3]), fr(c[4]))
    end
    if have_run
        println("\n  B. $(Date(span[1])) to $(Date(span[2])): B-SOSE 5-day (nearest to an instant), B-SOSE monthly, $RUN 30-day mean")
        println("  band     season |  profiles no-cross% | obs file    obs | B-SOSE 5d  ratio | B-SOSE mon  ratio |    run  ratio")
        for b in 1:nb, s in 1:4
            sel = [p for p in 1:np if band[p] == b && season(month(P.time[p])) == s && span[1] < P.time[p] <= span[2]]
            isempty(sel) && continue
            n, nc, c = row(sel, (b5, bmon, runm))
            @printf("  %-8s %-6s | %9d %8.0f  | %s %s |  %s  %s |   %s  %s | %s  %s\n", band_name(b), SEASONS[s], n, nc,
                    fmt(c[1]), fmt(c[2]), fmt(c[3]), fr(c[4]), fmt(c[5]), fr(c[6]), fmt(c[7]), fr(c[8]))
        end
    end
    println()

    # ── figure: monthly medians per band, part A ─────────────────────────────────────────────
    mkpath(OUTDIR)
    bands = [b for b in 1:nb if any(==(b), band)]
    fig = Figure(size = (1000, 240 * length(bands) + 60), fontsize = 13)
    Label(fig[0, 1], "Mixed layer depth: Spira profiles against B-SOSE monthly at the same profiles, $(first(YEARS))-$(last(YEARS))"; font = :bold)
    for (r, b) in enumerate(bands)
        ax = Axis(fig[r, 1]; title = band_name(b), ylabel = "m", xticks = (1:12, [monthabbr(m) for m in 1:12]), yreversed = true)
        for (v, lab, col, ls) in ((obs, "observed (Spira)", :black, :solid), (bmon, "B-SOSE monthly", :darkorange, :dash))
            y = [md([v[p] for p in 1:np if band[p] == b && month(P.time[p]) == m && isfinite(obs[p]) && isfinite(bmon[p])]) for m in 1:12]
            lines!(ax, 1:12, y; color = col, linestyle = ls, linewidth = 2, label = lab)
        end
        r == 1 && axislegend(ax; position = :lb, backgroundcolor = (:white, 0.8))
    end
    out = joinpath(OUTDIR, have_run ? "spira_mld_$(RUN).png" : "spira_mld.png")
    save(out, fig); @info "wrote $out"
end

main()
