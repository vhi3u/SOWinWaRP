# ==============================================================================================
# barotropic_circulation.jl -- the depth-integrated flow of the runs against B-SOSE's, over the runs' common span.
#
#     BSOSE_DIR=/storage/scratch1/1/vnguyen480/SOWinWaRP/data julia --project=. tools/barotropic_circulation.jl <RUN> [<RUN> ...]
#
# WHY. tools/salt_budget.jl found the run carrying ~19 Sv SOUTHWARD across 65S (full depth) where B-SOSE carries
# 3.3 Sv northward, and the coastal band's net flow at 150E reversed (run -3.6 Sv eastward out in 0-210 m, B-SOSE
# +0.9 westward in). This shows where: a map of the barotropic streamfunction, and the coastal current along the coast.
#
# WHAT.
#   U = ∫ u dz (m²/s), from the run's 30-day means (<RUN>_eke.nc, u at cell centres) and B-SOSE's monthly UVEL x hFacW x
#   drF (grid.nc; UVEL on the cells' west faces). B-SOSE's mean is overlap-weighted over the same span, as
#   tools/bsose_windows.jl does for one window. r*'s (1 + η/H) is left out: ~1%, nothing at the scale of this map.
#   ψ(x, y) = -∫ U dy from the Antarctic coast (ψ = 0 on the continent), Sv; U = -∂ψ/∂y. In the Southern Hemisphere
#   flow runs along ψ contours with HIGH ψ on its left. ψ at the northern edge = the total eastward transport there.
#   Only U is used: a run whose depth-integrated flow is not divergence-free (its volumes do not close) still gets a
#   well-defined ψ, but ∂ψ/∂x is then not exactly its V.
#
# WRITES OUTDIR/barotropic_<runs>.png      ψ for B-SOSE and each run, and run - B-SOSE (on the run's grid)
#        OUTDIR/barotropic_coast_<runs>.png  against longitude: the transport between the coast and 65S (ψ there) and
#                                            between the coast and each ISOBATH (shelf + slope: the slope current)
#        stdout: the eastward transport through meridional SECTIONS in the bands coast-65S, 65-60S, 60S-north edge.
#
# ENV: DIR (.), OUTDIR (plots), BSOSE_DIR (required), SECTIONS (90.1,105,120,135,149.9), BANDS (-65,-60),
#      ISOBATHS (1000,3000), CLIM (Sv, the ψ colour limit; default from B-SOSE), DCLIM (the difference's)
# ==============================================================================================

ENV["HDF5_USE_FILE_LOCKING"] = "FALSE"
using NCDatasets, CairoMakie, Dates, Printf, Statistics
include(joinpath(@__DIR__, "bsose_windows.jl"))   # bsose_path, bsose_records, bsose_weights, bsose_window_mean
include(joinpath(@__DIR__, "model_grid.jl"))      # file_levels

const RUNS = copy(ARGS)
isempty(RUNS) && error("give one or more run names: julia --project=. tools/barotropic_circulation.jl <RUN> [<RUN> ...]")
const DIR       = get(ENV, "DIR", ".")
const OUTDIR    = get(ENV, "OUTDIR", "plots")
const BSOSE_DIR = get(ENV, "BSOSE_DIR", "")
isempty(BSOSE_DIR) && error("set BSOSE_DIR to the directory with B-SOSE's monthly Uvel and grid.nc")
const SECTIONS  = parse.(Float64, split(get(ENV, "SECTIONS", "90.1,105,120,135,149.9"), ','))
const EDGES     = parse.(Float64, split(get(ENV, "BANDS", "-65,-60"), ','))
const ISOBATHS  = parse.(Float64, split(get(ENV, "ISOBATHS", "1000,3000"), ','))
const R_EARTH   = 6.371e6
const SOUTH, NORTH = -70.0, -45.0
const λW, λE    = 90.0, 150.0
const Sv        = 1e6

f64(A) = Float64.(coalesce.(A, NaN))
fill0(A) = (B = f64(A); B[.!isfinite.(B)] .= 0; B)
coord(ds, name, dim) = (v = f64(ds[name][:]); ndims(v) == 1 ? v : dim == 1 ? v[:, 1] : v[1, :])
gvar(ds, names, path) = (for n in names; haskey(ds, n) && return n; end;
                         error("none of $names in $path; it has $(join(keys(ds), ", "))"))

"Linear interpolation of y(x) at xq, x increasing; NaN outside."
function interp1(x, y, xq)
    (xq < x[1] || xq > x[end]) && return NaN
    n = clamp(searchsortedlast(x, xq), 1, length(x) - 1)
    w = (xq - x[n]) / (x[n+1] - x[n])
    return (1 - w) * y[n] + w * y[n+1]
end

"ψ (Sv) at the rows' NORTH faces from U (m²/s) on rows of height dy (m): -∑ U dy from the south."
streamfunction(U, dy) = -cumsum(U .* reshape(dy, 1, :), dims = 2) ./ Sv

# ── the runs ─────────────────────────────────────────────────────────────────────────────────────
"A run's grid, bathymetry and 30-day windows (start, end) from its _eke.nc."
function run_meta(run)
    path = joinpath(DIR, "$(run)_eke.nc")
    isfile(path) || error("$path not found (set DIR to the run directory)")
    NCDataset(path) do eke
        λ, φ = f64(eke["λ_caa"][:]), f64(eke["φ_aca"][:])
        lv = file_levels(eke); lv === nothing && error("$path carries no vertical levels")
        zc, zf, _ = lv
        size(eke["u"], 3) == length(zc) || error("$path holds $(size(eke["u"], 3)) levels of u, not the full $(length(zc))")
        hb = haskey(eke, "bottom_height") ? fill0(eke["bottom_height"][:, :]) :
             NCDataset(d -> fill0(d["bottom_height"][:, :]), joinpath(DIR, "$(run)_surface.nc"))
        start = DateTime(string(eke.attrib["start_date"]))
        window = haskey(eke.attrib, "averaging_window_seconds") ? Float64(eke.attrib["averaging_window_seconds"]) : 30 * 86400.0
        tend = f64(eke["time"][:])
        recs = findall(>(0), tend)
        ms(s) = Millisecond(round(Int, 1000s))
        wins = [(start + ms(tend[n] - window), start + ms(tend[n])) for n in recs]
        (; name = run, path, λ, φ, zc, zf, hb, recs, wins)
    end
end

"The run's mean U (m²/s) over its windows inside [w0, w1]."
function run_U(m, w0, w1)
    sel = [q for (q, (a, b)) in enumerate(m.wins) if a >= w0 - Day(1) && b <= w1 + Day(1)]
    isempty(sel) && error("$(m.name): no 30-day window inside $w0 to $w1")
    Nx, Ny, Nz = length(m.λ), length(m.φ), length(m.zc)
    dz = [m.hb[i, j] < 0 ? max(0.0, m.zf[k+1] - max(m.zf[k], m.hb[i, j])) : 0.0 for i in 1:Nx, j in 1:Ny, k in 1:Nz]
    U = zeros(Nx, Ny)
    NCDataset(m.path) do eke
        for q in sel
            U .+= dropdims(sum(fill0(eke["u"][:, :, :, m.recs[q]]) .* dz, dims = 3), dims = 3)
        end
    end
    @info @sprintf("%s: %d windows, %s to %s", m.name, length(sel), m.wins[sel[1]][1], m.wins[sel[end]][2])
    return U ./ length(sel)
end

# ── B-SOSE ───────────────────────────────────────────────────────────────────────────────────────
"B-SOSE's mean U (m²/s) on the west faces of the cells in 90-150E (plus the face at 150E), rows from its southern edge to 45S."
function bsose_U(w0, w1)
    gpath = joinpath(BSOSE_DIR, "grid.nc")
    isfile(gpath) || error("$gpath not found")
    upath = bsose_path(BSOSE_DIR, "Uvel")
    isempty(upath) && error("Uvel_bsoseI156_2013to2024_monthly.nc not found in $BSOSE_DIR")
    xc, yc = NCDataset(upath) do ds
        # UVEL sits on XG; the cell centres come from a tracer file (grid.nc's XG/YG are corrupt in i156)
        haskey(ds, "XC") && haskey(ds, "YC") ? (coord(ds, "XC", 1), coord(ds, "YC", 2)) :
            NCDataset(d -> (coord(d, "XC", 1), coord(d, "YC", 2)), bsose_path(BSOSE_DIR, "Salt"))
    end
    I = findall(x -> λW < x < λE, xc); iu = first(I):last(I)+1                 # west faces, + the face at 150E
    J = 1:findlast(<(NORTH), yc)                                               # from B-SOSE's southern edge
    hW, drF, depth = NCDataset(gpath) do g
        fill0(g[gvar(g, ("hFacW", "HFacW"), gpath)][iu, J, :]), f64(g[gvar(g, ("drF", "DRF"), gpath)][:]),
        fill0(g[gvar(g, ("Depth", "DEPTH"), gpath)][I, J])
    end
    recs = bsose_records(BSOSE_DIR, "Uvel", ("UVEL", "Uvel"), nothing)
    u = bsose_window_mean(recs, w0, w1, iu, J, length(drF))
    u === nothing && error("B-SOSE's monthly Uvel does not cover $w0 to $w1")
    U = dropdims(sum(coalesce.(replace(u, NaN => 0.0), 0.0) .* hW .* reshape(drF, 1, 1, :), dims = 3), dims = 3)
    yf = vcat((yc[J[1:end-1]] .+ yc[J[2:end]]) ./ 2, 1.5yc[J[end]] - 0.5yc[J[end-1]])   # north faces
    ys = vcat(1.5yc[J[1]] - 0.5yc[J[2]], yf[1:end-1])                                   # south faces
    dy = R_EARTH .* deg2rad.(yf .- ys)
    Δx = xc[2] - xc[1]
    xg = vcat(xc[I] .- Δx / 2, xc[last(I)] + Δx / 2)
    @info @sprintf("B-SOSE: %d faces %.2f-%.2fE, %d rows to %.2fS", length(xg), xg[1], xg[end], length(J), -yf[end])
    return (; U, ψ = streamfunction(U, dy), xg, xc = xc[I], yc = yc[J], yf, depth)
end

# ── shared readings ──────────────────────────────────────────────────────────────────────────────
"ψ of a column at latitude L, from its rows' north faces yf; 0 at and south of the first row's south face (the wall)."
function ψ_at(ψcol, yf, L)
    y0 = yf[1] - (yf[2] - yf[1])
    return L <= y0 ? 0.0 : interp1(vcat(y0, yf), vcat(0.0, ψcol), L)
end

"ψ at the north face of the last row before a column first reaches depth `iso` north of the coast; NaN if it never does."
function ψ_isobath(ψcol, H, iso)
    jw = findfirst(>(0), H); jw === nothing && return NaN
    j = findfirst(h -> h >= iso, @view H[jw:end]); j === nothing && return NaN
    j = jw + j - 1
    return j == 1 ? 0.0 : ψcol[j-1]
end

# ── main ─────────────────────────────────────────────────────────────────────────────────────────
metas = [run_meta(r) for r in RUNS]
w0 = maximum(first(first(m.wins)) for m in metas)
w1 = minimum(last(last(m.wins)) for m in metas)
w1 > w0 || error("the runs share no span")
@info "common span $w0 to $w1"
B = bsose_U(w0, w1)

runs = map(metas) do m
    U = run_U(m, w0, w1)
    Δφ = m.φ[2] - m.φ[1]
    dy = fill(R_EARTH * deg2rad(Δφ), length(m.φ))
    yf = m.φ .+ Δφ / 2
    ψ = streamfunction(U, dy)
    # B-SOSE's ψ on this run's grid: in latitude per B-SOSE column, then in longitude (faces -> the run's centres)
    ψBy = reduce(hcat, [[ψ_at(B.ψ[i, :], B.yf, y) for i in axes(B.ψ, 1)] for y in yf])   # (B faces, run rows)
    ψB = [interp1(B.xg, ψBy[:, j], x) for x in m.λ, j in eachindex(yf)]
    (; m.name, λ = m.λ, φ = m.φ, yf, H = max.(-m.hb, 0.0), U, ψ, ψB)
end

short(n) = replace(n, r"^bsose_ggl_hj_pr207_" => "", r"^bsose_ggl_" => "")
tag = join(short.(RUNS), "_")
mkpath(OUTDIR)

# ── the table: eastward transport through meridional sections, by band ──────────────────────────
bands = vcat([("coast-$(Int(-EDGES[1]))S", SOUTH - 10, EDGES[1])],
             [("$(Int(-EDGES[i-1]))-$(Int(-EDGES[i]))S", EDGES[i-1], EDGES[i]) for i in 2:length(EDGES)],
             [("$(Int(-EDGES[end]))S-north", EDGES[end], NORTH + 1)])
band_ψ(ψcol, yf, lo, hi) = ψ_at(ψcol, yf, lo) - ψ_at(ψcol, yf, min(hi, yf[end]))   # eastward = ψ(south) - ψ(north)
println("\n", "="^110)
@printf("EASTWARD transport (Sv) through meridians, %s to %s; + east, - west. B-SOSE on its own faces.\n",
        Dates.format(w0, "yyyy-mm-dd"), Dates.format(w1, "yyyy-mm-dd"))
@printf("%-10s %-28s", "lon", "model"); for (b, _, _) in bands; @printf(" %12s", b); end; @printf(" %12s\n", "total")
for x in SECTIONS
    iB = argmin(abs.(B.xg .- x))
    rows = [("B-SOSE", B.ψ[iB, :], B.yf, B.xg[iB])]
    for r in runs
        i = argmin(abs.(r.λ .- x)); push!(rows, (short(r.name), r.ψ[i, :], r.yf, r.λ[i]))
    end
    for (name, ψc, yf, xa) in rows
        @printf("%-10s %-28s", @sprintf("%.2fE", xa), name)
        for (_, lo, hi) in bands; @printf(" %12.2f", band_ψ(ψc, yf, lo, hi)); end
        @printf(" %12.2f\n", -ψ_at(ψc, yf, min(NORTH, yf[end])))
    end
end
println("  (a band's number is ψ(south edge) - ψ(north edge) = ∫ U dy over the band; 'coast' starts at the continent)")

# ── figure 1: ψ maps ─────────────────────────────────────────────────────────────────────────────
clim = haskey(ENV, "CLIM") ? parse(Float64, ENV["CLIM"]) :
       ceil(maximum(abs, filter(isfinite, B.ψ[:, B.yf .>= SOUTH])) / 10) * 10
dclim = haskey(ENV, "DCLIM") ? parse(Float64, ENV["DCLIM"]) : clim / 2
levels = range(-clim, clim, length = 21)
nr = length(runs)
fig = Figure(size = (520 * (nr + 1), 760))
land(H) = [h > 0 ? NaN : 1.0 for h in H]
let ax = Axis(fig[1, 1], title = "B-SOSE ψ (Sv)", ylabel = "latitude")
    sel = B.yf .>= SOUTH - 0.5
    hm = heatmap!(ax, B.xg, B.yf[sel], B.ψ[:, sel], colormap = :balance, colorrange = (-clim, clim))
    contour!(ax, B.xg, B.yf[sel], B.ψ[:, sel], levels = levels, color = :black, linewidth = 0.5)
    heatmap!(ax, B.xc, B.yc[sel], land(B.depth[:, sel]), colormap = [:gray70, :gray70], colorrange = (0, 2))
    ylims!(ax, SOUTH, NORTH)
    Colorbar(fig[3, 1], hm, vertical = false, label = "ψ (Sv), U = -∂ψ/∂y, ψ = 0 on the continent", flipaxis = false)
end
for (c, r) in enumerate(runs)
    ax = Axis(fig[1, c+1], title = "$(short(r.name)) ψ (Sv)")
    heatmap!(ax, r.λ, r.yf, r.ψ, colormap = :balance, colorrange = (-clim, clim))
    contour!(ax, r.λ, r.yf, r.ψ, levels = levels, color = :black, linewidth = 0.5)
    heatmap!(ax, r.λ, r.φ, land(r.H), colormap = [:gray70, :gray70], colorrange = (0, 2))
    ylims!(ax, SOUTH, NORTH)
    ax2 = Axis(fig[2, c+1], title = "$(short(r.name)) - B-SOSE (Sv)", xlabel = "longitude")
    hd = heatmap!(ax2, r.λ, r.yf, r.ψ .- r.ψB, colormap = :balance, colorrange = (-dclim, dclim))
    contour!(ax2, r.λ, r.yf, r.ψB, levels = levels, color = (:black, 0.5), linewidth = 0.5)
    heatmap!(ax2, r.λ, r.φ, land(r.H), colormap = [:gray70, :gray70], colorrange = (0, 2))
    ylims!(ax2, SOUTH, NORTH)
    c == 1 && (ax2.ylabel = "latitude")
    c == nr && Colorbar(fig[3, c+1], hd, vertical = false, flipaxis = false, label = "run - B-SOSE (Sv); contours B-SOSE's ψ")
end
Label(fig[0, :], @sprintf("Depth-integrated flow, %s to %s", Dates.format(w0, "yyyy-mm-dd"), Dates.format(w1, "yyyy-mm-dd")),
      fontsize = 18)
out1 = joinpath(OUTDIR, "barotropic_$(tag).png")
save(out1, fig); @info "wrote $out1"

# ── figure 2: the coastal current along the coast ────────────────────────────────────────────────
colors = Makie.wong_colors()
targets = vcat([("coast to $(Int(-EDGES[1]))S", :lat, EDGES[1])], [("coast to the $(Int(iso)) m isobath", :iso, iso) for iso in ISOBATHS])
fig2 = Figure(size = (900, 300 * length(targets)))
for (p, (ttl, kind, v)) in enumerate(targets)
    ax = Axis(fig2[p, 1], title = "Eastward transport, $ttl (Sv; - = westward)", ylabel = "Sv",
              xlabel = p == length(targets) ? "longitude" : "")
    hlines!(ax, [0.0], color = :gray60, linewidth = 1)
    col(ψc, yf, H) = kind == :lat ? ψ_at(ψc, yf, v) : ψ_isobath(ψc, H, v)
    # the isobath search needs depth on the same columns as ψ: B-SOSE's faces take the cell to their east
    HB = vcat(B.depth, B.depth[end:end, :])
    yb = [-col(B.ψ[i, :], B.yf, HB[i, :]) for i in axes(B.ψ, 1)]
    lines!(ax, B.xg, yb, color = :black, linewidth = 2, label = "B-SOSE")
    for (c, r) in enumerate(runs)
        yr = [-col(r.ψ[i, :], r.yf, r.H[i, :]) for i in eachindex(r.λ)]
        lines!(ax, r.λ, yr, color = colors[c], linewidth = 2, label = short(r.name))
    end
    p == 1 && axislegend(ax, position = :lb, framevisible = false)
end
out2 = joinpath(OUTDIR, "barotropic_coast_$(tag).png")
save(out2, fig2); @info "wrote $out2"
