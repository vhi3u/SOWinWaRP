# ==============================================================================================
# transport_timeseries.jl -- the runs' shelf, slope and boundary transports month by month, against B-SOSE's.
#
#     BSOSE_DIR=/storage/scratch1/1/vnguyen480/SOWinWaRP/data julia --project=. tools/transport_timeseries.jl <RUN> [<RUN> ...]
#
# WHY. tools/barotropic_circulation.jl: over Mar 2014 - Feb 2016 the runs have no westward shelf/slope current (B-SOSE
# ~-1 Sv on the shelf all along the coast); theirs is eastward and grows to +9 Sv (shelf) / +13 (slope) by 150E, and they
# carry ~30 Sv more eastward than B-SOSE east of 105E. Two readings, told apart by WHEN it happens:
#   from the first window   -> the boundary / the run's barotropic dynamics (the east face never imposes B-SOSE's
#                              westward inflow; pressure never reaches the child)
#   builds over months      -> the interior state: the shelf getting saltier/warmer and the slope front collapsing
#
# QUANTITIES (Sv), per run window (30-day means of <RUN>_eke.nc, u, v at cell centres) and per B-SOSE month
# (UVEL x hFacW, VVEL x hFacS, x drF; grid.nc):
#   shelf@L, slope@L   eastward transport from the coast to the 1000 / 3000 m isobath, at each of LONS, the mean over the
#                      columns within ±HALF degrees (single columns are spiky where the isobath wiggles)
#   coast W, coast E   eastward, coast to 65S, at the domain's first / last column (the run's boundary columns)
#   total W, total E   eastward, full width 70-45S, same columns
#   north in           net SOUTHWARD (into the domain) through 45S
#   south across 65S   net SOUTHWARD across 65S, full depth (salt_budget: run ~19 Sv vs B-SOSE -3.3 over the span)
#
# WRITES OUTDIR/transport_timeseries_<runs>.png and a month-by-month table on stdout.
# ENV: DIR (.), OUTDIR (plots), BSOSE_DIR (required), LONS (95,120,145), HALF (0.5), ISOBATHS (1000,3000), EDGE (-65)
# ==============================================================================================

ENV["HDF5_USE_FILE_LOCKING"] = "FALSE"
using NCDatasets, CairoMakie, Dates, Printf, Statistics
include(joinpath(@__DIR__, "bsose_windows.jl"))   # bsose_path, bsose_data_var
include(joinpath(@__DIR__, "model_grid.jl"))      # file_levels

const RUNS = copy(ARGS)
isempty(RUNS) && error("give one or more run names: julia --project=. tools/transport_timeseries.jl <RUN> [<RUN> ...]")
const DIR       = get(ENV, "DIR", ".")
const OUTDIR    = get(ENV, "OUTDIR", "plots")
const BSOSE_DIR = get(ENV, "BSOSE_DIR", "")
isempty(BSOSE_DIR) && error("set BSOSE_DIR to the directory with B-SOSE's monthly Uvel, Vvel and grid.nc")
const LONS     = parse.(Float64, split(get(ENV, "LONS", "95,120,145"), ','))
const HALF     = parse(Float64, get(ENV, "HALF", "0.5"))
const ISOBATHS = parse.(Float64, split(get(ENV, "ISOBATHS", "1000,3000"), ','))
const EDGE     = parse(Float64, get(ENV, "EDGE", "-65"))
const R_EARTH  = 6.371e6
const NORTH    = -45.0
const λW, λE   = 90.0, 150.0
const Sv       = 1e6

f64(A) = Float64.(coalesce.(A, NaN))
fill0(A) = (B = f64(A); B[.!isfinite.(B)] .= 0; B)
coord(ds, name, dim) = (v = f64(ds[name][:]); ndims(v) == 1 ? v : dim == 1 ? v[:, 1] : v[1, :])
gvar(ds, names, path) = (for n in names; haskey(ds, n) && return n; end;
                         error("none of $names in $path; it has $(join(keys(ds), ", "))"))

# the quantity names, in table/plot order
const QNAMES = vcat([@sprintf("shelf@%.0fE", L) for L in LONS], [@sprintf("slope@%.0fE", L) for L in LONS],
                    ["coast W", "coast E", "total W", "total E", "north in", @sprintf("south across %.0fS", -EDGE)])

"Eastward transport (Sv) of one column of U (m²/s, rows south to north, heights dy) from the coast to depth `iso`."
function to_isobath(Ucol, dy, H, iso)
    jw = findfirst(>(0), H); jw === nothing && return NaN
    j = findfirst(h -> h >= iso, @view H[jw:end]); j === nothing && return NaN
    return sum(Ucol[1:jw+j-2] .* dy[1:jw+j-2]) / Sv
end

"Eastward transport (Sv) of one column from the coast to latitude `L` (rows with centres south of L)."
to_lat(Ucol, dy, y, L) = sum(Ucol[y .< L] .* dy[y .< L]) / Sv

"The shared quantities from U (x, y), the column longitudes x, row centres y, heights dy, depth H, and the two meridional
transports (Sv, northward): through the northern edge and across EDGE."
function quantities(U, x, y, dy, H, vnorth, vedge)
    q = Float64[]
    for iso in ISOBATHS, L in LONS
        cols = findall(c -> abs(c - L) <= HALF, x)
        push!(q, nanmean(filter(isfinite, [to_isobath(U[i, :], dy, H[i, :], iso) for i in cols])))
    end
    push!(q, to_lat(U[1, :], dy, y, EDGE), to_lat(U[end, :], dy, y, EDGE),
             to_lat(U[1, :], dy, y, NORTH + 1), to_lat(U[end, :], dy, y, NORTH + 1), -vnorth, -vedge)
    return q
end
nanmean(v) = isempty(v) ? NaN : mean(v)

# ── the runs ─────────────────────────────────────────────────────────────────────────────────────
function run_series(run)
    path = joinpath(DIR, "$(run)_eke.nc")
    isfile(path) || error("$path not found (set DIR to the run directory)")
    NCDataset(path) do eke
        λ, φ = f64(eke["λ_caa"][:]), f64(eke["φ_aca"][:])
        lv = file_levels(eke); lv === nothing && error("$path carries no vertical levels")
        zc, zf, _ = lv
        Nx, Ny, Nz = length(λ), length(φ), length(zc)
        hb = haskey(eke, "bottom_height") ? fill0(eke["bottom_height"][:, :]) :
             NCDataset(d -> fill0(d["bottom_height"][:, :]), joinpath(DIR, "$(run)_surface.nc"))
        H = max.(-hb, 0.0)
        dz = [hb[i, j] < 0 ? max(0.0, zf[k+1] - max(zf[k], hb[i, j])) : 0.0 for i in 1:Nx, j in 1:Ny, k in 1:Nz]
        start = DateTime(string(eke.attrib["start_date"]))
        window = haskey(eke.attrib, "averaging_window_seconds") ? Float64(eke.attrib["averaging_window_seconds"]) : 30 * 86400.0
        tend = f64(eke["time"][:])
        recs = findall(>(0), tend)
        Δλ, Δφ = deg2rad(λ[2] - λ[1]), deg2rad(φ[2] - φ[1])
        dy = fill(R_EARTH * Δφ, Ny)
        dx(φf) = R_EARTH * Δλ * cosd(φf)
        j1 = findlast(<(EDGE), φ); j2 = j1 + 1                          # the rows either side of EDGE
        times = DateTime[]; Q = Vector{Float64}[]
        for n in recs
            u = fill0(eke["u"][:, :, :, n])
            U = dropdims(sum(u .* dz, dims = 3), dims = 3)
            vN = fill0(eke["v"][:, Ny, :, n])                           # (x, z): the last row's centre, 1/12 deg south of 45S
            vnorth = sum(vN .* dz[:, Ny, :]) * dx(φ[Ny]) / Sv
            v1, v2 = fill0(eke["v"][:, j1, :, n]), fill0(eke["v"][:, j2, :, n])
            h = min.(dz[:, j1, :], dz[:, j2, :])
            vedge = sum((v1 .+ v2) ./ 2 .* h) * dx((φ[j1] + φ[j2]) / 2) / Sv
            push!(Q, quantities(U, λ, φ, dy, H, vnorth, vedge))
            push!(times, start + Millisecond(round(Int, 1000 * (tend[n] - window / 2))))
        end
        @info @sprintf("%s: %d windows from %s", run, length(times), times[1])
        (; name = run, times, Q = reduce(hcat, Q)')                     # (time, quantity)
    end
end

# ── B-SOSE ───────────────────────────────────────────────────────────────────────────────────────
function bsose_series(t0, t1)
    gpath = joinpath(BSOSE_DIR, "grid.nc")
    upath, vpath = bsose_path(BSOSE_DIR, "Uvel"), bsose_path(BSOSE_DIR, "Vvel")
    (isfile(gpath) && !isempty(upath) && !isempty(vpath)) || error("need grid.nc, Uvel and Vvel monthly files in $BSOSE_DIR")
    xc, yc = NCDataset(d -> (coord(d, "XC", 1), coord(d, "YC", 2)), bsose_path(BSOSE_DIR, "Salt"))
    I = findall(x -> λW < x < λE, xc); i0, i1 = first(I), last(I)
    iu = i0:i1+1                                                         # west faces + the face at 150E
    J = 1:findlast(<(NORTH), yc)
    ys = vcat(1.5yc[1] - 0.5yc[2], (yc[1:end-1] .+ yc[2:end]) ./ 2)     # south faces (YG in grid.nc is corrupt)
    jN = argmin(abs.(ys .- NORTH)); jE = argmin(abs.(ys .- EDGE))       # VVEL rows on the 45S and EDGE faces
    hW, hS, drF, depth = NCDataset(gpath) do g
        fill0(g[gvar(g, ("hFacW", "HFacW"), gpath)][iu, J, :]),
        (hs = g[gvar(g, ("hFacS", "HFacS"), gpath)]; cat(fill0(hs[I, jN:jN, :]), fill0(hs[I, jE:jE, :]), dims = 2)),
        f64(g[gvar(g, ("drF", "DRF"), gpath)][:]), fill0(g[gvar(g, ("Depth", "DEPTH"), gpath)][I, J])
    end
    dy = R_EARTH .* deg2rad.(ys[J.+1] .- ys[J])
    Δx = deg2rad(xc[2] - xc[1]) * R_EARTH
    xg = vcat(xc[I] .- (xc[2] - xc[1]) / 2, xc[i1] + (xc[2] - xc[1]) / 2)
    Hf = vcat(depth, depth[end:end, :])                                  # depth on the faces: the cell to the east
    times = DateTime[]; Q = Vector{Float64}[]
    du, dv = NCDataset(upath), NCDataset(vpath)
    try
        uvar, vvar = bsose_data_var(du, ("UVEL", "Uvel"), upath), bsose_data_var(dv, ("VVEL", "Vvel"), vpath)
        stamps = DateTime.(du["time"][:])
        for (n, s) in enumerate(stamps)
            d = s - Day(14); m0 = DateTime(year(d), month(d), 1)
            (m0 + Month(1) > t0 && m0 < t1) || continue
            u = fill0(du[uvar][:, J, :, n])[iu, :, :]
            U = dropdims(sum(u .* hW .* reshape(drF, 1, 1, :), dims = 3), dims = 3)
            vr = cat(fill0(dv[vvar][:, jN:jN, :, n]), fill0(dv[vvar][:, jE:jE, :, n]), dims = 2)[I, :, :]
            vnorth = sum(vr[:, 1, :] .* hS[:, 1, :] .* reshape(drF, 1, :)) * Δx * cosd(ys[jN]) / Sv
            vedge = sum(vr[:, 2, :] .* hS[:, 2, :] .* reshape(drF, 1, :)) * Δx * cosd(ys[jE]) / Sv
            push!(Q, quantities(U, xg, yc[J], dy, Hf, vnorth, vedge))
            push!(times, m0 + Day(daysinmonth(m0) ÷ 2))
        end
    finally
        close(du); close(dv)
    end
    @info @sprintf("B-SOSE: %d months, faces at %.2fS (north) and %.2fS (edge)", length(times), -ys[jN], -ys[jE])
    (; name = "B-SOSE", times, Q = reduce(hcat, Q)')
end

# ── main ─────────────────────────────────────────────────────────────────────────────────────────
runs = [run_series(r) for r in RUNS]
t0 = minimum(first(r.times) for r in runs) - Day(15); t1 = maximum(last(r.times) for r in runs) + Day(15)
B = bsose_series(t0, t1)
short(n) = replace(n, r"^bsose_ggl_hj_pr207_" => "", r"^bsose_ggl_" => "")

"B-SOSE at time t: the month containing it."
bsose_at(t, k) = (n = argmin(abs.(Dates.value.(B.times .- t))); B.Q[n, k])

println("\n", "="^120)
println("TRANSPORTS (Sv), each run window against the B-SOSE month nearest its centre; + east / + INTO the domain (north in) / + south")
for r in runs
    println("\n  ", r.name)
    @printf("  %-10s", "window"); for q in QNAMES; @printf(" %14s", q); end; println()
    for (n, t) in enumerate(r.times)
        @printf("  %-10s", Dates.format(t, "yyyy-mm-dd"))
        for k in eachindex(QNAMES); @printf(" %6.1f (%5.1f)", r.Q[n, k], bsose_at(t, k)); end
        println()
    end
end
println("  run (B-SOSE)")

colors = Makie.wong_colors()
nq = length(QNAMES); ncol = 3; nrow = cld(nq, ncol)
fig = Figure(size = (1500, 270 * nrow))
for (k, q) in enumerate(QNAMES)
    ax = Axis(fig[cld(k, ncol), mod1(k, ncol)], title = q, ylabel = "Sv", xticklabelrotation = π / 6)
    hlines!(ax, [0.0], color = :gray60, linewidth = 1)
    xs(ts) = [Dates.value(t - DateTime(2014)) / (86400e3 * 365.25) + 2014 for t in ts]
    lines!(ax, xs(B.times), B.Q[:, k], color = :black, linewidth = 2, label = "B-SOSE")
    for (c, r) in enumerate(runs)
        scatterlines!(ax, xs(r.times), r.Q[:, k], color = colors[c], linewidth = 2, markersize = 8, label = short(r.name))
    end
    k == 1 && axislegend(ax, position = :lb, framevisible = false)
end
Label(fig[0, :], "Transports month by month (shelf/slope: + east; north in: + into the domain; across $(Int(-EDGE))S: + south)",
      fontsize = 18)
mkpath(OUTDIR)
out = joinpath(OUTDIR, "transport_timeseries_$(join(short.(RUNS), "_")).png")
save(out, fig); @info "wrote $out"
