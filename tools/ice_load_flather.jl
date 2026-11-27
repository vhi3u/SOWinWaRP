# ==============================================================================================
# ice_load_flather.jl -- does B-SOSE's sea-ice load in the parent η explain the runs' boundary transport errors?
#
#     BSOSE_DIR=/storage/scratch1/1/vnguyen480/SOWinWaRP/data julia --project=. tools/ice_load_flather.jl <RUN> [<RUN> ...]
#
# WHY. tools/transport_timeseries.jl: at the east face the runs send the coastal band EAST (+5 to +20 Sv) where B-SOSE
# brings +3 to +10 Sv in westward, from the first 30-day window on; at the west face the coastal band goes -10 to -12
# Sv in winter (B-SOSE -1.5); 25-32 Sv flow south across 65S in winter. Identical with and without the run's sea ice.
#
# HYPOTHESIS. At NumericalEarth 768ed1d the depth-integrated flow through an open face is set only by Flather,
#     U_out = U_parent_out + √(g H) (η_child - η_parent)
# (the relaxation band nudges only the baroclinic u, v). B-SOSE's ETAN carries the sea-ice load, -(ρᵢ hᵢ + ρₛ hₛ)/ρ₀
# (compare_eke: slope 1.12 on -(910 SIheff)/1027 over icy cells); the child carries none, and the _ibobc/_bsoseflux
# hook removes only the inverse barometer. So wherever B-SOSE has ice at a face, η_child - η_parent ≈ +load and Flather
# drives an extra OUTWARD transport ≈ ∫ √(g H) load dy: westward at the west face, eastward at the east face, both fed
# from the north.
#
# WHAT. Per month, at B-SOSE's first and last cell columns of 90-150E: the load (m) from B-SOSE's monthly SIheff
# (and SIhsnow if there is a file; without snow the load is ~12% low), and the Flather transport it would drive,
# as EASTWARD Sv (west face: -∫, east face: +∫), in the bands coast-EDGE and EDGE-45S. Against it, each run's error
# in eastward transport at its boundary columns (run 30-day window minus the B-SOSE month nearest its centre; same
# U = ∫ u dz as transport_timeseries). The prediction is an upper bound: it assumes the child's η does not move toward
# the parent's, and the interior's adjustment takes some of it back. Read the SHAPE: does the error follow the load
# month by month (winter peak at the west, a March signal at the east if B-SOSE has ice at 150E then)?
#
# WRITES OUTDIR/ice_load_flather_<runs>.png and a table on stdout (with the correlation and the regression slope of
# each run's error on the prediction).
# ENV: DIR (.), OUTDIR (plots), BSOSE_DIR (required), EDGE (-65), ICE_TOKEN (SIheff), SNOW_TOKEN (SIhsnow)
# ==============================================================================================

ENV["HDF5_USE_FILE_LOCKING"] = "FALSE"
using NCDatasets, CairoMakie, Dates, Printf, Statistics
include(joinpath(@__DIR__, "bsose_windows.jl"))   # bsose_path, bsose_data_var
include(joinpath(@__DIR__, "model_grid.jl"))      # file_levels

const RUNS = copy(ARGS)
isempty(RUNS) && error("give one or more run names: julia --project=. tools/ice_load_flather.jl <RUN> [<RUN> ...]")
const DIR        = get(ENV, "DIR", ".")
const OUTDIR     = get(ENV, "OUTDIR", "plots")
const BSOSE_DIR  = get(ENV, "BSOSE_DIR", "")
isempty(BSOSE_DIR) && error("set BSOSE_DIR")
const EDGE       = parse(Float64, get(ENV, "EDGE", "-65"))
const ICE_TOKEN  = get(ENV, "ICE_TOKEN", "SIheff")
const SNOW_TOKEN = get(ENV, "SNOW_TOKEN", "SIhsnow")
const ρ₀, ρᵢ, ρₛ, g = 1027.0, 910.0, 330.0, 9.81
const R_EARTH = 6.371e6
const NORTH   = -45.0
const λW, λE  = 90.0, 150.0
const Sv      = 1e6

f64(A) = Float64.(coalesce.(A, NaN))
fill0(A) = (B = f64(A); B[.!isfinite.(B)] .= 0; B)
coord(ds, name, dim) = (v = f64(ds[name][:]); ndims(v) == 1 ? v : dim == 1 ? v[:, 1] : v[1, :])
gvar(ds, names, path) = (for n in names; haskey(ds, n) && return n; end;
                         error("none of $names in $path; it has $(join(keys(ds), ", "))"))
monthof(s) = (d = DateTime(s) - Day(14); (year(d), month(d)))    # monthly stamps sit at the month's end
const COLS = ("coast W", "coast E", "north W", "north E")         # eastward Sv; "north" = EDGE to 45S

# ── B-SOSE: geometry, load, transport at the boundary columns ───────────────────────────────────
const G = let
    sp = bsose_path(BSOSE_DIR, "Salt"); isempty(sp) && error("no Salt monthly file in $BSOSE_DIR")
    xc, yc = NCDataset(d -> (coord(d, "XC", 1), coord(d, "YC", 2)), sp)
    I = findall(x -> λW < x < λE, xc); i0, i1 = first(I), last(I)
    J = 1:findlast(<(NORTH), yc)
    ys = vcat(1.5yc[1] - 0.5yc[2], (yc[1:end-1] .+ yc[2:end]) ./ 2)
    dy = R_EARTH .* deg2rad.(ys[J.+1] .- ys[J])
    gpath = joinpath(BSOSE_DIR, "grid.nc")
    H, hW, drF = NCDataset(gpath) do gd
        dep = gd[gvar(gd, ("Depth", "DEPTH"), gpath)]; hw = gd[gvar(gd, ("hFacW", "HFacW"), gpath)]
        hcat(fill0(dep[i0, J]), fill0(dep[i1, J])),
        cat(fill0(hw[i0:i0, J, :]), fill0(hw[i1+1:i1+1, J, :]), dims = 1), f64(gd[gvar(gd, ("drF", "DRF"), gpath)][:])
    end
    (; xc, yc = yc[J], i0, i1, J, dy, H, hW, drF)
end
bands(y) = (y .< EDGE, (y .>= EDGE) .& (y .< NORTH + 1))

"Monthly (W column, E column) of a sea-ice token over rows J: from the monthly file if there is one, else the 5-day
files averaged by calendar month. Dict (year, month) => (Nj, 2); nothing if no file at all."
function ice_months(token, months)
    out = Dict{Tuple{Int,Int}, Matrix{Float64}}()
    mp = bsose_path(BSOSE_DIR, token)
    if !isempty(mp)
        NCDataset(mp) do ds
            v = ds[bsose_data_var(ds, (token,), mp)]
            for (n, s) in enumerate(ds["time"][:])
                k = monthof(s); k in months || continue
                out[k] = hcat(fill0(v[G.i0, G.J, n]), fill0(v[G.i1, G.J, n]))
            end
        end
        @info "$token: monthly file $(basename(mp))"
        return out
    end
    acc = Dict{Tuple{Int,Int}, Tuple{Matrix{Float64}, Int}}()
    for y in sort(unique(first.(months)))
        p = bsose_path(BSOSE_DIR, token, y)
        isempty(p) && (@warn "no $token 5-day file for $y"; continue)
        NCDataset(p) do ds
            v = ds[bsose_data_var(ds, (token,), p)]
            for (n, s) in enumerate(ds["time"][:])
                c = DateTime(s) - Hour(60); k = (year(c), month(c)); k in months || continue
                A = hcat(fill0(v[G.i0, G.J, n]), fill0(v[G.i1, G.J, n]))
                a, m = get(acc, k, (zeros(size(A)), 0)); acc[k] = (a .+ A, m + 1)
            end
        end
    end
    isempty(acc) && return nothing
    @info "$token: 5-day files averaged by month"
    for (k, (a, m)) in acc; out[k] = a ./ m; end
    return out
end

"B-SOSE's eastward transport (Sv) in the four COLS for one monthly record."
function bsose_transport(du, uvar, n)
    Q = zeros(4)
    for (c, i) in enumerate((G.i0, G.i1 + 1))
        u = fill0(du[uvar][i:i, G.J, :, n])[1, :, :]
        U = vec(sum(u .* G.hW[c, :, :] .* reshape(G.drF, 1, :), dims = 2))
        s, nb = bands(G.yc)
        Q[c] = sum(U[s] .* G.dy[s]) / Sv; Q[c+2] = sum(U[nb] .* G.dy[nb]) / Sv
    end
    return Q
end

# ── the runs: eastward transport at their boundary columns, per 30-day window ───────────────────
function run_transports(run)
    path = joinpath(DIR, "$(run)_eke.nc")
    isfile(path) || error("$path not found (set DIR)")
    NCDataset(path) do eke
        φ = f64(eke["φ_aca"][:]); λ = f64(eke["λ_caa"][:])
        zc, zf, _ = file_levels(eke)
        Nx, Ny, Nz = length(λ), length(φ), length(zc)
        hb = haskey(eke, "bottom_height") ? fill0(eke["bottom_height"][:, :]) :
             NCDataset(d -> fill0(d["bottom_height"][:, :]), joinpath(DIR, "$(run)_surface.nc"))
        dzc(i) = [hb[i, j] < 0 ? max(0.0, zf[k+1] - max(zf[k], hb[i, j])) : 0.0 for j in 1:Ny, k in 1:Nz]
        dzW, dzE = dzc(1), dzc(Nx)
        dy = R_EARTH * deg2rad(φ[2] - φ[1])
        start = DateTime(string(eke.attrib["start_date"]))
        win = haskey(eke.attrib, "averaging_window_seconds") ? Float64(eke.attrib["averaging_window_seconds"]) : 30 * 86400.0
        tend = f64(eke["time"][:]); recs = findall(>(0), tend)
        s, nb = bands(φ)
        times = DateTime[]; Q = Vector{Float64}[]
        for n in recs
            UW = vec(sum(fill0(eke["u"][1, :, :, n]) .* dzW, dims = 2))
            UE = vec(sum(fill0(eke["u"][Nx, :, :, n]) .* dzE, dims = 2))
            push!(Q, [sum(UW[s]) * dy / Sv, sum(UE[s]) * dy / Sv, sum(UW[nb]) * dy / Sv, sum(UE[nb]) * dy / Sv])
            push!(times, start + Millisecond(round(Int, 1000 * (tend[n] - win / 2))))
        end
        (; name = run, times, Q = reduce(hcat, Q)')
    end
end

# ── main ─────────────────────────────────────────────────────────────────────────────────────────
runs = [run_transports(r) for r in RUNS]
months = let m = Tuple{Int,Int}[], d = DateTime(year(minimum(first(r.times) for r in runs)), month(minimum(first(r.times) for r in runs)))
    while d <= maximum(last(r.times) for r in runs); push!(m, (year(d), month(d))); d += Month(1); end; m
end
hi = ice_months(ICE_TOKEN, months); hi === nothing && error("no $ICE_TOKEN monthly or 5-day files in $BSOSE_DIR")
hs = ice_months(SNOW_TOKEN, months)
hs === nothing && @warn "no $SNOW_TOKEN file: the load is ice only (~12% low)"

load = Dict(k => (ρᵢ .* hi[k] .+ (hs === nothing || !haskey(hs, k) ? 0.0 : ρₛ .* hs[k])) ./ ρ₀ for k in keys(hi))
"Predicted eastward Flather transport (Sv) in the four COLS from the load: the west face pushes west, the east east."
function predicted(k)
    L = load[k]; s, nb = bands(G.yc)
    T(c, sel) = sum(sqrt.(g .* G.H[sel, c]) .* L[sel, c] .* G.dy[sel] .* (G.H[sel, c] .> 0)) / Sv
    return [-T(1, s), T(2, s), -T(1, nb), T(2, nb)]
end
maxload(k, c) = maximum(load[k][:, c] .* (G.H[:, c] .> 0))

upath = bsose_path(BSOSE_DIR, "Uvel"); isempty(upath) && error("no Uvel monthly file in $BSOSE_DIR")
BQ = Dict{Tuple{Int,Int}, Vector{Float64}}()
NCDataset(upath) do du
    uvar = bsose_data_var(du, ("UVEL", "Uvel"), upath)
    for (n, s) in enumerate(du["time"][:])
        k = monthof(s); k in months && (BQ[k] = bsose_transport(du, uvar, n))
    end
end
mk = [k for k in months if haskey(load, k) && haskey(BQ, k)]
@info "$(length(mk)) months with B-SOSE ice and velocity"
nearest_month(t) = mk[argmin([abs(Dates.value(DateTime(k[1], k[2], 15) - t)) for k in mk])]

println("\n", "="^128)
println("FLATHER FROM B-SOSE's ICE LOAD vs THE RUNS' BOUNDARY ERRORS: eastward Sv; coast = coast-$(Int(-EDGE))S, north = $(Int(-EDGE))-45S")
println("  pred = √(gH) load integrated along the face (upper bound); err = run - B-SOSE (window vs nearest month)")
for r in runs
    println("\n  ", r.name)
    @printf("  %-10s %7s %7s |", "window", "maxL W", "maxL E")
    for c in COLS; @printf(" %16s", c * " pred/err"); end; println()
    P = zeros(length(r.times), 4); E = similar(P)
    for (n, t) in enumerate(r.times)
        k = nearest_month(t)
        P[n, :] = predicted(k); E[n, :] = r.Q[n, :] .- BQ[k]
        @printf("  %-10s %7.2f %7.2f |", Dates.format(t, "yyyy-mm-dd"), maxload(k, 1), maxload(k, 2))
        for c in 1:4; @printf("   %6.1f / %6.1f", P[n, c], E[n, c]); end; println()
    end
    @printf("  %-26s|", "corr(err, pred)"); for c in 1:4; @printf(" %16.2f", cor(P[:, c], E[:, c])); end; println()
    @printf("  %-26s|", "slope err on pred"); for c in 1:4
        x = P[:, c] .- mean(P[:, c]); @printf(" %16.2f", sum(x .* (E[:, c] .- mean(E[:, c]))) / max(sum(x .^ 2), eps()))
    end; println()
    @printf("  %-26s|", "mean err / mean pred"); for c in 1:4; @printf("  %6.1f / %6.1f ", mean(E[:, c]), mean(P[:, c])); end; println()
end
println("  maxL: the largest load (m of water) on the face's wet cells that month")

colors = Makie.wong_colors()
fig = Figure(size = (1300, 700))
xs(ts) = [Dates.value(t - DateTime(2014)) / (86400e3 * 365.25) + 2014 for t in ts]
for (c, name) in enumerate(COLS)
    ax = Axis(fig[cld(c, 2), mod1(c, 2)], title = "$name: run - B-SOSE (lines+dots) vs Flather from the ice load (black)",
              ylabel = "eastward Sv")
    hlines!(ax, [0.0], color = :gray60, linewidth = 1)
    tk = [DateTime(k[1], k[2], 15) for k in mk]
    lines!(ax, xs(tk), [predicted(k)[c] for k in mk], color = :black, linewidth = 2, label = "Flather from load")
    for (q, r) in enumerate(runs)
        scatterlines!(ax, xs(r.times), [r.Q[n, c] - BQ[nearest_month(t)][c] for (n, t) in enumerate(r.times)],
                      color = colors[q], markersize = 7, label = replace(r.name, r"^bsose_ggl_hj_pr207_" => ""))
    end
    c == 1 && axislegend(ax, position = :lb, framevisible = false)
end
mkpath(OUTDIR)
out = joinpath(OUTDIR, "ice_load_flather_$(join(replace.(RUNS, r"^bsose_ggl_hj_pr207_" => ""), "_")).png")
save(out, fig); @info "wrote $out"
