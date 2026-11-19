# ==============================================================================================
# compare_mld_5day.jl -- the run's mixed layer depth against B-SOSE's, by latitude band, under
# B-SOSE's ice and in open water, from B-SOSE's 5-DAY T and S and its own MLD diagnostic.
#
#     BSOSE_DIR=/storage/scratch1/1/vnguyen480/SOWinWaRP/data julia --project=. tools/compare_mld_5day.jl 2Y_nested_bsose_ggl_hj_pr207_icein
#
# WHY. under_ice_heat.jl (_icein) found 70-65S keeping its deep heat down: B-SOSE moves ~140 MJ/m²
# out of 100-290 m each winter, the run ~15, with the same top-cell salinity. That reads as a winter
# mixed layer too shallow. compare_mld.jl measures B-SOSE's MLD from its MONTHLY-mean T and S, and
# the mixed layer of a mean state is shallower than the mean mixed layer; the 5-day records and
# B-SOSE's own MLD diagnostic avoid most of that.
#
# THREE NUMBERS per band, ice class and 30-day window:
#   run         mld.jl (density threshold MLD_DT, 10 m reference) of the run's 30-day mean T, S (<RUN>_eke.nc)
#   B-SOSE      the same function of B-SOSE's 5-day T, S averaged over the same window (like with like)
#   B-SOSE diag B-SOSE's own MLD (5-day files), averaged over the window: its criterion, its time
#               sampling. B-SOSE against B-SOSE diag says how far our criterion and the 30-day mean
#               state move the number; run against B-SOSE is the fair comparison.
# A column with no crossing above ZMAX (or its floor) counts at that depth, and the share of such
# columns is printed: in a deep winter mixed layer it is the result, not missing data.
# Cells: B-SOSE's nearest cell to each run cell (the grids coincide at 1/6°); ice = B-SOSE's window-
# mean SIArea >= ICE_MIN. Means are cos(latitude)-weighted.
#
# WRITES OUTDIR/<RUN>_mld_5day.png: per band, the three MLDs under ice and in open water.
# ENV: DIR, OUTDIR, BSOSE_DIR (required), BANDS (-70,-65,-60,-55,-50,-45), ICE_MIN (0.15),
#      ZMAX (1500 m), MIN_RECORDS (4), MLD_DT (0.03 kg/m³, read by mld.jl)
# ==============================================================================================

using NCDatasets, CairoMakie, Dates, Printf, Statistics

ENV["HDF5_USE_FILE_LOCKING"] = "FALSE"
include(joinpath(@__DIR__, "model_grid.jl"))   # file_levels, file_kbot
include(joinpath(@__DIR__, "mld.jl"))          # mld, MLD_DT

const RUN = isempty(ARGS) ? get(ENV, "RUN", "") : ARGS[1]
isempty(RUN) && error("give the run name: julia --project=. tools/compare_mld_5day.jl <RUN>")
const DIR         = get(ENV, "DIR", ".")
const OUTDIR      = get(ENV, "OUTDIR", "plots")
const BSOSE_DIR   = get(ENV, "BSOSE_DIR", "")
isempty(BSOSE_DIR) && error("set BSOSE_DIR to the directory with B-SOSE's 5-day Theta, Salt, SIArea and MLD files")
const EDGES       = parse.(Float64, split(get(ENV, "BANDS", "-70,-65,-60,-55,-50,-45"), ','))
const ICE_MIN     = parse(Float64, get(ENV, "ICE_MIN", "0.15"))
const ZMAX        = parse(Float64, get(ENV, "ZMAX", "1500"))
const MIN_RECORDS = parse(Int, get(ENV, "MIN_RECORDS", "4"))

f64(A) = Float64.(coalesce.(A, NaN))
coord(ds, name, dim) = (v = f64(ds[name][:]); ndims(v) == 1 ? v : dim == 1 ? v[:, 1] : v[1, :])
band_name(b) = @sprintf("%.0f-%.0fS", -EDGES[b], -EDGES[b+1])
label(d) = Dates.format(d, "yyyy-mm-dd")

function bsose_file(token, y)
    want = lowercase("$(token)_bsoseI156_$(y)_5day.nc")
    hits = filter(f -> lowercase(f) == want, readdir(BSOSE_DIR))
    isempty(hits) ? "" : joinpath(BSOSE_DIR, first(hits))
end
opens(p) = !isempty(p) && (try NCDataset(_ -> true, p) catch; false end)

"The data variable of a B-SOSE file: the first of `names`, else its only variable over (x, y, time) or (x, y, z, time)."
function data_var(ds, names, path)
    for n in names
        haskey(ds, n) && return n
    end
    cands = [k for k in keys(ds) if ndims(ds[k]) >= 3]
    length(cands) == 1 && return only(cands)
    error("none of $names in $path, and no single data variable ($(cands))")
end

function bsose_records(token, names, years)
    recs = Tuple{String, String, Int, DateTime}[]
    for y in years
        path = bsose_file(token, y)
        NCDataset(path) do ds
            var = data_var(ds, names, path)
            for (n, stamp) in enumerate(ds["time"][:])
                push!(recs, (path, var, n, DateTime(stamp) - Hour(60)))
            end
        end
    end
    return recs
end

function main()
    # ── the run ──────────────────────────────────────────────────────────────────────────────
    eke_path = joinpath(DIR, "$(RUN)_eke.nc")
    isfile(eke_path) || error("$eke_path not found; set DIR to the run directory")
    eke = NCDataset(eke_path)
    λ, φ = f64(eke["λ_caa"][:]), f64(eke["φ_aca"][:])
    Nx, Ny = length(λ), length(φ)
    lv = file_levels(eke); lv === nothing && error("$eke_path carries no vertical levels")
    zc, zf, _ = lv
    Nz = length(zc)
    fk = file_kbot(eke, zc, zf); fk === nothing && error("$eke_path carries no sea floor")
    kbot = fk[1]
    size(eke["T"], 3) == Nz || error("$eke_path holds $(size(eke["T"], 3)) levels of T, not the full $Nz: it was written without EKE_FULL_DEPTH?")
    start = haskey(ENV, "START") ? DateTime(ENV["START"]) : DateTime(string(eke.attrib["start_date"]))
    win = haskey(eke.attrib, "averaging_window_seconds") ? Float64(eke.attrib["averaging_window_seconds"]) : 30 * 86400.0
    tend = f64(eke["time"][:])
    keepn = findall(t -> t >= 0.99win, tend)
    windows = [(start + Millisecond(round(Int, 1000 * (tend[n] - win))), start + Millisecond(round(Int, 1000 * tend[n]))) for n in keepn]

    # levels down to ZMAX: run indices lo:Nz (bottom-up), nk of them
    lo = findfirst(k -> -zc[k] <= ZMAX, 1:Nz)
    nk = Nz - lo + 1
    zs = zc[lo:Nz]
    depth_top_down = [-zc[k] for k in Nz:-1:lo]

    # ── B-SOSE ──────────────────────────────────────────────────────────────────────────────
    years = filter(year(first(windows)[1]):year(last(windows)[2])) do y
        bad = [t for t in ("Theta", "Salt", "SIArea", "MLD") if !opens(bsose_file(t, y))]
        isempty(bad) || @warn "B-SOSE $y skipped: $(join(bad, ", ")) 5-day file missing or unreadable"
        isempty(bad)
    end
    isempty(years) && error("no year with B-SOSE's 5-day Theta, Salt, SIArea and MLD readable in $BSOSE_DIR")
    Trecs = bsose_records("Theta", ("THETA", "Theta"), years)
    Srecs = bsose_records("Salt", ("SALT", "Salt"), years)
    Arecs = bsose_records("SIArea", ("SIarea", "SIArea", "SIAREA"), years)
    Mrecs = bsose_records("MLD", ("MLD", "MXLDEPTH", "BLGMLD"), years)
    XC, YC, RF = NCDataset(Trecs[1][1]) do ds
        rf = haskey(ds, "RF") ? f64(ds["RF"][:]) : [0.0; -cumsum(f64(ds[haskey(ds, "drF") ? "drF" : "DRF"][:]))]
        coord(ds, "XC", 1), coord(ds, "YC", 2), rf
    end
    ix = [argmin(abs.(XC .- x)) for x in λ]; jy = [argmin(abs.(YC .- y)) for y in φ]
    ir, jr = minimum(ix):maximum(ix), minimum(jy):maximum(jy)
    il, jl = ix .- first(ir) .+ 1, jy .- first(jr) .+ 1
    dB = [-(RF[k] + RF[k+1]) / 2 for k in 1:nk]
    maximum(abs.(dB .- depth_top_down)) < 0.05 || error("B-SOSE's top $nk level centres differ from the run's")
    @info @sprintf("%s: %dx%d, %d levels to %.0f m; %d windows %s to %s; MLD_DT %.3f kg/m³", RUN, Nx, Ny, nk,
                   depth_top_down[end], length(windows), label(first(windows)[1]), label(last(windows)[2]), MLD_DT)

    function bsose_mean(recs, w0, w1, levels)
        sel = filter(r -> w0 < r[4] <= w1, recs)
        length(sel) < MIN_RECORDS && return nothing
        acc = nothing
        for (path, var, n, _) in sel
            A = NCDataset(path) do ds
                v = ds[var]
                f64(levels === nothing ? (ndims(v) == 4 ? v[ir, jr, 1, n] : v[ir, jr, n]) : v[ir, jr, 1:levels, n])
            end
            acc = acc === nothing ? A : acc .+ A
        end
        return acc ./ length(sel)
    end

    # ── per window ──────────────────────────────────────────────────────────────────────────
    nb, nw = length(EDGES) - 1, length(windows)
    band_of = [findfirst(b -> EDGES[b] <= φ[j] < EDGES[b+1], 1:nb) for j in 1:Ny]
    wlat = cosd.(φ)
    classes = (:ice, :open)
    res = Dict((c, q) => fill(NaN, nb, nw) for c in classes for q in (:cells, :run, :bsose, :diag, :deep_run, :deep_bsose))

    for (w, (w0, w1)) in enumerate(windows)
        Tb = bsose_mean(Trecs, w0, w1, nk); Sb = bsose_mean(Srecs, w0, w1, nk)
        Ab = bsose_mean(Arecs, w0, w1, nothing); Mb = bsose_mean(Mrecs, w0, w1, nothing)
        if any(isnothing, (Tb, Sb, Ab, Mb))
            @warn "$(label(w0)) to $(label(w1)): fewer than $MIN_RECORDS B-SOSE records, skipped"
            continue
        end
        Tb, Sb, Ab, Mb = Tb[il, jl, :], Sb[il, jl, :], Ab[il, jl], Mb[il, jl]
        n = keepn[w]
        Tr = f64(eke["T"][:, :, lo:Nz, n]); Sr = f64(eke["S"][:, :, lo:Nz, n])

        h_run = fill(NaN, Nx, Ny); h_b = fill(NaN, Nx, Ny); deep_r = falses(Nx, Ny); deep_b = falses(Nx, Ny)
        for j in 1:Ny, i in 1:Nx
            band_of[j] === nothing && continue
            kb = kbot[i, j]; kb === nothing && continue
            Sb[i, j, 1] == 0 && continue                                  # B-SOSE land
            # run: bottom-up, truncated at ZMAX
            kr = max(kb - lo + 1, 1)
            h = mld(view(Tr, i, j, :), view(Sr, i, j, :), zs, kr)
            isfinite(h) || (h = -zf[kr + lo - 1]; deep_r[i, j] = true)
            h_run[i, j] = h
            # B-SOSE: top-down -> bottom-up; its floor is the deepest level with S != 0
            Tc, Sc = reverse(Tb[i, j, :]), reverse(Sb[i, j, :])
            kbB = findfirst(!=(0), Sc); kbB === nothing && continue
            hb = mld(Tc, Sc, zs, kbB)
            isfinite(hb) || (hb = -zf[kbB + lo - 1]; deep_b[i, j] = true)
            h_b[i, j] = hb
        end

        for b in 1:nb, c in classes
            cells = [(i, j) for i in 1:Nx, j in 1:Ny if band_of[j] == b && isfinite(h_run[i, j]) && isfinite(h_b[i, j]) &&
                                                        isfinite(Mb[i, j]) && ((Ab[i, j] >= ICE_MIN) == (c === :ice))]
            isempty(cells) && continue
            W = [wlat[j] for (i, j) in cells]
            wm(f) = sum(W .* [f(i, j) for (i, j) in cells]) / sum(W)
            res[(c, :cells)][b, w] = length(cells)
            res[(c, :run)][b, w] = wm((i, j) -> h_run[i, j])
            res[(c, :bsose)][b, w] = wm((i, j) -> h_b[i, j])
            res[(c, :diag)][b, w] = wm((i, j) -> Mb[i, j])
            res[(c, :deep_run)][b, w] = 100 * count(((i, j),) -> deep_r[i, j], cells) / length(cells)
            res[(c, :deep_bsose)][b, w] = 100 * count(((i, j),) -> deep_b[i, j], cells) / length(cells)
        end
        @info "window $w/$nw ($(label(w1))) done"
    end
    close(eke)

    # ── report ───────────────────────────────────────────────────────────────────────────────
    centre(w) = windows[w][1] + (windows[w][2] - windows[w][1]) ÷ 2
    bar = "="^118
    println("\n", bar)
    println("MIXED LAYER DEPTH, $RUN against B-SOSE (m; density threshold $(MLD_DT) kg/m³ from 10 m, columns to $(Int(ZMAX)) m)")
    println("  run, B-SOSE: that criterion on each one's 30-day mean T, S; diag: B-SOSE's own MLD, averaged over the window")
    println("  no-cross %: columns with no crossing above the floor or $(Int(ZMAX)) m, counted at that depth")
    println("  ice: B-SOSE window-mean SIArea >= $ICE_MIN")
    println(bar)
    for b in 1:nb, c in classes
        all(isnan, res[(c, :cells)][b, :]) && continue
        @printf("\n  %s, %s\n", band_name(b), c === :ice ? "UNDER B-SOSE'S ICE" : "OPEN WATER")
        println("  window centre | cells |  run  B-SOSE  B-SOSE diag | run/B-SOSE | no-cross % run B-SOSE")
        for w in 1:nw
            isnan(res[(c, :cells)][b, w]) && continue
            @printf("  %s    | %5d | %5.0f  %6.0f  %11.0f |    %6.2f  |          %5.0f  %5.0f\n", label(centre(w)),
                    res[(c, :cells)][b, w], res[(c, :run)][b, w], res[(c, :bsose)][b, w], res[(c, :diag)][b, w],
                    res[(c, :run)][b, w] / res[(c, :bsose)][b, w], res[(c, :deep_run)][b, w], res[(c, :deep_bsose)][b, w])
        end
    end
    println()

    # ── figure ───────────────────────────────────────────────────────────────────────────────
    mkpath(OUTDIR)
    bands = [b for b in 1:nb if any(c -> !all(isnan, res[(c, :cells)][b, :]), classes)]
    isempty(bands) && return
    x = [Dates.value(centre(w) - start) / 86_400_000 for w in 1:nw]
    ticks = (x[1:3:end], [Dates.format(centre(w), "yyyy-mm") for w in 1:3:nw])
    fig = Figure(size = (1300, 300 * length(bands) + 80), fontsize = 13)
    Label(fig[0, 1:2], "$RUN: mixed layer depth against B-SOSE (Δσ = $(MLD_DT) kg/m³)"; font = :bold, fontsize = 16)
    for (r, b) in enumerate(bands), (col, c) in enumerate(classes)
        ax = Axis(fig[r, col]; title = "$(band_name(b)), $(c === :ice ? "under B-SOSE's ice" : "open water")", ylabel = "m",
                  xticks = ticks, xticklabelrotation = π / 6, yreversed = true)
        lines!(ax, x, res[(c, :run)][b, :]; color = :royalblue, linewidth = 2.5, label = "run")
        lines!(ax, x, res[(c, :bsose)][b, :]; color = :black, linewidth = 2, linestyle = :dash, label = "B-SOSE (same criterion)")
        lines!(ax, x, res[(c, :diag)][b, :]; color = :darkorange, linewidth = 2, label = "B-SOSE MLD diagnostic")
        r == 1 && col == 1 && axislegend(ax; position = :lb, backgroundcolor = (:white, 0.8))
    end
    out = joinpath(OUTDIR, "$(RUN)_mld_5day.png")
    save(out, fig)
    @info "wrote $out"
end

main()
