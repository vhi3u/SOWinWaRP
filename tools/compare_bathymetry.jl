# ==============================================================================================
# compare_bathymetry.jl -- the seafloor the model actually ran on, against B-SOSE's.
#
#     julia --project=. tools/compare_bathymetry.jl
#     MODEL_RUN=testbed/control julia --project=. tools/compare_bathymetry.jl
#     SOURCES=etopo,ibcso julia --project=. tools/compare_bathymetry.jl   # needs downloads
#
# NO DOWNLOAD, AND NO RECONSTRUCTION. The grid is not saved anywhere -- src/model.jl rebuilds it
# every run -- but the bathymetry is recoverable from any run's own output, because Oceananigans
# writes IMMERSED CELLS AS EXACT ZEROS. In a column of `RUN_NAME_eke.nc` the cells below the
# seafloor are identically 0 and the wet ones are not, so the deepest non-zero cell IS the
# bottom. That is the seafloor the model used -- after `regrid_bathymetry`, after
# `fix_boundary_topography!`, after the immersed boundary discretised it -- rather than a
# re-derivation that could differ from it. It also needs no ETOPO (0.44 GB) and no NumericalEarth.
#
# SALINITY, NOT TEMPERATURE, is the wet test. B-SOSE salinity is ~34 and never legitimately 0,
# while T passes through 0 degC constantly in this domain: a zero there is ambiguous between
# land and water at the freezing point.
#
# THE RECOVERED DEPTH IS CELL-RESOLUTION, and that is a real limit, not a rounding detail.
# `GridFittedBottom` immerses a cell when its CENTRE is below the bottom height, so the true
# bottom lies between the deepest wet centre and the dry centre beneath it. This reports the
# interface between them, so the answer carries about +/- half a cell -- which near the bottom
# of a stretched grid is tens of metres. Differences smaller than the local cell height are not
# resolved and should not be read as disagreement.
#
# OPTIONAL REGRID SOURCES, off by default because they download: SOURCES=etopo,ibcso,gebco
# pulls ETOPO2022 (0.44 GB) and/or IBCSOv2 (0.23 GB) through NumericalEarth and regrids them
# with src/model.jl's own knobs (model.jl:452: height_above_water=1, minimum_depth=10,
# interpolation_passes=5, and no major_basins, so 1). Note model.jl passes no `dataset` either,
# so the model is on ETOPO2022 -- we have never run on IBCSO.
# IBCSOv2 spans -90 to -50 ONLY and NumericalEarth ENFORCES it: `validate_dataset_coverage`
# throws when the grid reaches north of 50S. It gets its own 70-50S sub-grid here and is padded
# with NaN; every statistic is over the overlap. `IBCSOv2`/`GEBCO2026` are also not exported at
# top level in 0.8.2 and come from their submodules.
#
# COMMON CURRENCY IS DEPTH, metres, positive down, land = 0.
#
# ENV
#   MODEL_RUN   run whose output holds the grid   (default: the first *_eke.nc found)
#   MODEL_DIR   where to look for it              (default: .)
#   BSOSE_GRID  B-SOSE grid file                  (default: data/grid.nc)
#   SOURCES     extra regrid sources, needs net   (default: none)
#   NZ_VIEW     z-limit for the 3-D plots, m      (default: 5800, = Z_EXTENT)
#   OUTDIR      figures                           (default: plots)
#
# LOGIN-NODE SAFE in the default path: NCDatasets and CairoMakie only, one 3-D field resident.
# ==============================================================================================

using NCDatasets, Printf, Statistics, CairoMakie

const BGRID   = get(ENV, "BSOSE_GRID", "data/grid.nc")
const OUTDIR  = get(ENV, "OUTDIR", "plots")
const MDIR    = get(ENV, "MODEL_DIR", ".")
const NZVIEW  = parse(Float64, get(ENV, "NZ_VIEW", "5800"))
const SOURCES = [lowercase(strip(s)) for s in split(get(ENV, "SOURCES", ""), ",") if !isempty(strip(s))]

# model.jl's values, cited so a change there is traceable to here. Only used by SOURCES.
const PASSES, BASINS, MIN_DEP, HAW = 5, 1, 10.0, 1.0        # model.jl:453-455
const IBCSO_φ₂ = -50.0

"""The first name that is a variable in `ds`, or `nothing`."""
pick(ds, cands) = (h = findfirst(c -> haskey(ds, c), cands); h === nothing ? nothing : cands[h])

# ── the model's own grid, from its output ─────────────────────────────────────────────────────
function find_run()
    want = get(ENV, "MODEL_RUN", "")
    if !isempty(want)
        p = joinpath(MDIR, "$(want)_eke.nc")
        isfile(p) && return p
        error("no $p -- set MODEL_RUN/MODEL_DIR")
    end
    hits = String[]
    for sub in ("", "testbed", "surface_tests")
        d = isempty(sub) ? MDIR : joinpath(MDIR, sub)
        isdir(d) || continue
        append!(hits, [joinpath(d, f) for f in readdir(d) if endswith(f, "_eke.nc")])
    end
    isempty(hits) && error("no *_eke.nc under $MDIR (or its testbed/, surface_tests/) -- " *
                           "set MODEL_RUN, or run with SOURCES=etopo to regrid instead")
    return first(sort(hits))
end

"""
Seafloor depth per column, recovered from a run's own output.

Returns `(lons, lats, depth)`. A column that is zero at every level is land (depth 0).
"""
function model_depth(path)
    ds = Dataset(path)
    try
        sv = pick(ds, ["S", "salinity"])
        sv === nothing && error("$path has no S -- cannot find the seafloor without salinity")
        ndims(ds[sv]) == 4 ||
            error("$path holds $(ndims(ds[sv]))-D S, so it was written surface-only. " *
                  "The seafloor needs a full-depth field: set EKE_FULL_DEPTH=true in model.jl, " *
                  "or point MODEL_RUN at a run that had it on.")
        xn = pick(ds, ["x_caa", "λ_caa", "λ_fca", "xC", "longitude", "lon", "x"])
        yn = pick(ds, ["y_aca", "φ_aca", "φ_cca", "yC", "latitude", "lat", "y"])
        zn = pick(ds, ["z_aac", "zC", "z_c", "z"])
        (xn === nothing || yn === nothing || zn === nothing) &&
            error("$path is missing a coordinate: x=$xn y=$yn z=$zn")
        lons = Float64.(coalesce.(ds[xn][:], NaN))
        lats = Float64.(coalesce.(ds[yn][:], NaN))
        zc   = Float64.(coalesce.(ds[zn][:], NaN))      # increasing upward; zc[1] is deepest
        nt   = size(ds[sv], 4)
        S    = Float64.(coalesce.(ds[sv][:, :, :, nt], NaN))   # last record: fully spun up

        nx, ny, nz = size(S)
        # Cell interfaces from centres. The bottom of cell k sits midway to cell k-1; the
        # deepest cell's floor is extrapolated by half its own spacing.
        zf = similar(zc)
        for k in eachindex(zc)
            zf[k] = k == 1 ? zc[1] - (zc[2] - zc[1]) / 2 : (zc[k - 1] + zc[k]) / 2
        end

        D = zeros(nx, ny)
        for j in 1:ny, i in 1:nx
            kwet = 0
            for k in 1:nz                      # k = 1 is the deepest level
                v = S[i, j, k]
                if isfinite(v) && v != 0
                    kwet = k; break            # first non-zero from below = deepest wet cell
                end
            end
            D[i, j] = kwet == 0 ? 0.0 : -zf[kwet]
        end
        return lons, lats, D
    finally
        close(ds)
    end
end

const RUNPATH = isempty(SOURCES) || !isempty(get(ENV, "MODEL_RUN", "")) ? find_run() :
                (try find_run() catch; "" end)
@info "model grid from $RUNPATH"
const LONS, LATS, MODELD = model_depth(RUNPATH)
const Nx, Ny = length(LONS), length(LATS)
@info "$(Nx)x$(Ny) columns, $(count(>(0), MODELD)) wet; " *
      @sprintf("deepest %.0f m", maximum(MODELD))

# ── B-SOSE, interpolated onto that same grid ──────────────────────────────────────────────────
function bsose_depth()
    isfile(BGRID) || error("no $BGRID -- set BSOSE_GRID")
    ds = Dataset(BGRID)
    try
        # XC/YC are stored 2-D but the grid is separable (XC constant along dim 2, YC along
        # dim 1). XG/YG are CORRUPT in the i156 grid file -- never use them.
        bx = Float64.(ds["XC"][:, 1])
        by = Float64.(ds["YC"][1, :])
        D  = Float64.(coalesce.(ds["Depth"][:, :], 0.0))
        out = fill(NaN, Nx, Ny)
        for j in 1:Ny, i in 1:Nx
            λ, φ = LONS[i], LATS[j]
            (bx[1] <= λ <= bx[end] && by[1] <= φ <= by[end]) || continue
            i2 = searchsortedfirst(bx, λ); i1 = max(1, i2 - 1); i2 = min(length(bx), i2)
            j2 = searchsortedfirst(by, φ); j1 = max(1, j2 - 1); j2 = min(length(by), j2)
            tx = i2 == i1 ? 0.0 : (λ - bx[i1]) / (bx[i2] - bx[i1])
            ty = j2 == j1 ? 0.0 : (φ - by[j1]) / (by[j2] - by[j1])
            out[i, j] = (1 - tx) * (1 - ty) * D[i1, j1] + tx * (1 - ty) * D[i2, j1] +
                        (1 - tx) * ty       * D[i1, j2] + tx * ty       * D[i2, j2]
        end
        return max.(out, 0.0)
    finally
        close(ds)
    end
end

SRC = Pair{String,Matrix{Float64}}["model (as run)" => MODELD, "B-SOSE" => bsose_depth()]

# ── optional regrid sources ───────────────────────────────────────────────────────────────────
"""One NumericalEarth dataset regridded with model.jl's knobs, onto the model's own extent."""
function regrid_source(name)
    dλ = (LONS[2] - LONS[1]); dφ = (LATS[2] - LATS[1])
    λ₁, λ₂ = LONS[1] - dλ / 2, LONS[end] + dλ / 2
    φ₁, φ₂ = LATS[1] - dφ / 2, LATS[end] + dφ / 2
    mk(south, north, ny) = LatitudeLongitudeGrid(CPU(); size = (Nx, ny, 1), halo = (7, 7, 7),
                                                 longitude = (λ₁, λ₂), latitude = (south, north),
                                                 z = (-NZVIEW, 0))
    kw = (; height_above_water = HAW, minimum_depth = MIN_DEP,
            interpolation_passes = PASSES, major_basins = BASINS)
    if name == "ibcso"
        ny = round(Int, (IBCSO_φ₂ - φ₁) / dφ)
        h  = Array(interior(regrid_bathymetry(mk(φ₁, IBCSO_φ₂, ny); dataset = IBCSOv2(), kw...), :, :, 1))
        out = fill(NaN, Nx, Ny)
        out[:, 1:min(ny, Ny)] .= max.(0.0, .-h[:, 1:min(ny, Ny)])   # 50-45S stays NaN
        return "IBCSOv2 (70-50S)", out
    end
    ds = name == "etopo" ? ETOPO2022() : name == "gebco" ? GEBCO2026() :
         error("unknown source $name; use etopo, ibcso or gebco")
    h = Array(interior(regrid_bathymetry(mk(φ₁, φ₂, Ny); dataset = ds, kw...), :, :, 1))
    return (name == "etopo" ? "ETOPO2022 regrid" : "GEBCO2026 regrid"), max.(0.0, .-h)
end

# Loaded only if asked for: the default path must not pay Oceananigans' load time, and must not
# touch the network at all.
if !isempty(SOURCES)
    @info "SOURCES=$(join(SOURCES, ",")) -- loading NumericalEarth; first use DOWNLOADS " *
          "(ETOPO 0.44 GB, IBCSO 0.23 GB)"
    @eval using Oceananigans
    @eval using Oceananigans.Grids: λnodes, φnodes
    @eval using NumericalEarth
    @eval using NumericalEarth.DataWrangling.IBCSO: IBCSOv2
    @eval using NumericalEarth.DataWrangling.GEBCO: GEBCO2026
    for s in SOURCES
        try
            lab, D = Base.invokelatest(regrid_source, s)
            push!(SRC, lab => D)
        catch e
            @warn "$s failed -- skipping\n    $(first(split(sprint(showerror, e), '\n')))"
        end
    end
end

# ── statistics ────────────────────────────────────────────────────────────────────────────────
both(A, B) = [isfinite(A[i]) && isfinite(B[i]) for i in eachindex(A)]

"""Signed difference B-A where both have data, NaN elsewhere, SHAPED like A."""
diffmap(A, B) = [isfinite(A[i]) && isfinite(B[i]) ? B[i] - A[i] : NaN for i in CartesianIndices(A)]

println("\n" * "="^104)
println("BATHYMETRY -- depth in metres, positive down, land = 0")
println("="^104)
println("\n  model seafloor recovered from $(basename(RUNPATH)) (deepest non-zero S per column).")
println("  Cell-resolution: the true bottom is within half a cell of the interface reported,")
println("  so a difference smaller than the local cell height is not resolved.\n")

@printf("  %-30s %8s %9s %9s %9s %8s %8s\n",
        "pair", "n cells", "mean", "rms", "max |d|", "at lon", "at lat")
@printf("  %-30s %8s %9s %9s %9s %8s %8s\n", "", "", "(m)", "(m)", "(m)", "", "")
for a in 1:length(SRC), b in (a + 1):length(SRC)
    (la, A), (lb, B) = SRC[a], SRC[b]
    m = both(A, B); n = count(m)
    n == 0 && (@printf("  %-30s %8s\n", "$lb - $la", "no overlap"); continue)
    idx = findall(m); d = [B[i] - A[i] for i in idx]
    ci = CartesianIndices(A)[idx[argmax(abs.(d))]]
    @printf("  %-30s %8d %9.1f %9.1f %9.1f %8.2f %8.2f\n",
            "$lb - $la", n, mean(d), sqrt(mean(d .^ 2)), maximum(abs.(d)),
            LONS[ci[1]], LATS[ci[2]])
end

println("\n  WET/DRY DISAGREEMENT -- cells one calls ocean and the other land.")
println("  The one that changes a model: it moves the coastline and the open-boundary columns.")
for a in 1:length(SRC), b in (a + 1):length(SRC)
    (la, A), (lb, B) = SRC[a], SRC[b]
    m = both(A, B); n = count(m); n == 0 && continue
    aonly = count(i -> m[i] && A[i] > 0 && !(B[i] > 0), eachindex(A))
    bonly = count(i -> m[i] && !(A[i] > 0) && B[i] > 0, eachindex(A))
    @printf("    %-30s %5d of %6d (%.2f%%) -- %d wet only in %s, %d wet only in %s\n",
            "$lb vs $la", aonly + bonly, n, 100 * (aonly + bonly) / n, aonly, la, bonly, lb)
end

println("\n  OPEN-BOUNDARY ROWS ONLY -- mean |difference| on each face (m).")
println("  A boundary column at the wrong depth is what produced the west-face velocity pileup.")
@printf("    %-30s %9s %9s %9s %9s\n", "pair", "west", "east", "south", "north")
for a in 1:length(SRC), b in (a + 1):length(SRC)
    (la, A), (lb, B) = SRC[a], SRC[b]
    face(ii, jj) = (v = [abs(B[i, j] - A[i, j]) for i in ii, j in jj
                         if isfinite(A[i, j]) && isfinite(B[i, j])];
                    isempty(v) ? NaN : mean(v))
    f(v) = isfinite(v) ? @sprintf("%9.1f", v) : lpad("--", 9)
    @printf("    %-30s %s %s %s %s\n", "$lb - $la",
            f(face(1:1, 1:Ny)), f(face(Nx:Nx, 1:Ny)), f(face(1:Nx, 1:1)), f(face(1:Nx, Ny:Ny)))
end

# ── figures ───────────────────────────────────────────────────────────────────────────────────
mkpath(OUTDIR)
zlim = (-NZVIEW, 200.0)
surf(A) = [isfinite(v) ? -v : NaN for v in A]    # plot as bottom height: it looks like seafloor

let   # 1. THE ASK: every source in ONE Axis3, translucent so overlaps read
    fig = Figure(size = (1240, 900))
    Label(fig[0, 1], @sprintf("Seafloor, all sources overlaid — %.0f-%.0fE, %.0f-%.0fS",
                              LONS[1], LONS[end], -LATS[1], -LATS[end]),
          fontsize = 18, font = :bold)
    ax = Axis3(fig[1, 1]; xlabel = "longitude", ylabel = "latitude", zlabel = "bottom height (m)",
               azimuth = 0.62π, elevation = 0.17π, aspect = (2.2, 1.3, 0.85),
               limits = (nothing, nothing, zlim))
    cols = [:black, :dodgerblue, :orangered, :forestgreen]
    elems = Any[]
    for (k, (lab, A)) in enumerate(SRC)
        surface!(ax, LONS, LATS, surf(A); color = fill(cols[mod1(k, 4)], Nx, Ny),
                 alpha = k == 1 ? 0.92 : 0.42, shading = NoShading, transparency = true)
        push!(elems, PolyElement(color = (cols[mod1(k, 4)], k == 1 ? 0.92 : 0.42)))
    end
    Legend(fig[1, 2], elems, [l for (l, _) in SRC], framevisible = false)
    Label(fig[2, 1], "The model is the opaque black surface. Where a coloured surface shows " *
          "through BELOW it, that source is deeper there.", fontsize = 11, color = :gray40)
    save(joinpath(OUTDIR, "bathymetry_overlay3d.png"), fig; px_per_unit = 2)
    @info "wrote $(joinpath(OUTDIR, "bathymetry_overlay3d.png"))"
end

let n = length(SRC)   # 2. one surface per source, shared colour and z limits
    fig = Figure(size = (560 * n, 640))
    Label(fig[0, 1:n], "Seafloor by source — same colour and z limits", fontsize = 18, font = :bold)
    local sp = nothing
    for (k, (lab, A)) in enumerate(SRC)
        ax = Axis3(fig[1, k]; title = lab, titlesize = 13, xlabel = "lon", ylabel = "lat",
                   zlabel = k == 1 ? "bottom height (m)" : "", azimuth = 0.62π, elevation = 0.17π,
                   aspect = (2.2, 1.3, 0.85), limits = (nothing, nothing, zlim))
        sp = surface!(ax, LONS, LATS, surf(A); colormap = :deep, colorrange = (-NZVIEW, 0),
                      shading = NoShading, nan_color = :gray85)
    end
    Colorbar(fig[1, n + 1], sp, label = "bottom height (m)")
    save(joinpath(OUTDIR, "bathymetry_by_source3d.png"), fig; px_per_unit = 2)
    @info "wrote $(joinpath(OUTDIR, "bathymetry_by_source3d.png"))"
end

PAIRS = [(1, b) for b in 2:length(SRC)]
let   # 3. differences in 3-D: shape is the model's seafloor, COLOUR is the disagreement
    fig = Figure(size = (560 * length(PAIRS), 660))
    Label(fig[0, 1:length(PAIRS)],
          "Where they disagree — surface is the model, colour is the difference in depth",
          fontsize = 18, font = :bold)
    lim = 1.0
    for (a, b) in PAIRS
        A, B = SRC[a][2], SRC[b][2]
        d = filter(isfinite, abs.(diffmap(A, B)))
        isempty(d) || (lim = max(lim, quantile(d, 0.98)))
    end
    local hp = nothing
    for (k, (a, b)) in enumerate(PAIRS)
        (la, A), (lb, B) = SRC[a], SRC[b]
        d = diffmap(A, B)
        ax = Axis3(fig[1, k]; title = "$lb − $la", titlesize = 13, xlabel = "lon", ylabel = "lat",
                   zlabel = k == 1 ? "bottom height (m)" : "", azimuth = 0.62π, elevation = 0.17π,
                   aspect = (2.2, 1.3, 0.85), limits = (nothing, nothing, zlim))
        hp = surface!(ax, LONS, LATS, surf(A); color = d, colormap = Reverse(:RdBu),
                      colorrange = (-lim, lim), shading = NoShading, nan_color = :gray85)
    end
    Colorbar(fig[1, length(PAIRS) + 1], hp, label = "difference in depth (m), red = deeper")
    Label(fig[2, 1:length(PAIRS)],
          @sprintf("colour limit is the 98th percentile of |difference| (%.0f m); grey = no data there", lim),
          fontsize = 11, color = :gray40)
    save(joinpath(OUTDIR, "bathymetry_difference3d.png"), fig; px_per_unit = 2)
    @info "wrote $(joinpath(OUTDIR, "bathymetry_difference3d.png"))"
end

let ncol = length(PAIRS)   # 4. flat maps: easiest to localise a feature on
    fig = Figure(size = (540 * ncol, 780))
    Label(fig[0, 1:ncol], "Difference maps and coastline disagreement", fontsize = 18, font = :bold)
    for (k, (a, b)) in enumerate(PAIRS)
        (la, A), (lb, B) = SRC[a], SRC[b]
        d = diffmap(A, B)
        fin = filter(isfinite, d)
        lim = isempty(fin) ? 1.0 : max(quantile(abs.(fin), 0.98), 1.0)
        ax = Axis(fig[1, k]; title = "$lb − $la  (m, red = deeper)", titlesize = 12,
                  xlabel = "longitude", ylabel = k == 1 ? "latitude" : "")
        hm = heatmap!(ax, LONS, LATS, d; colormap = Reverse(:RdBu), colorrange = (-lim, lim),
                      nan_color = :gray85)
        Colorbar(fig[1, ncol + k], hm)
        w = [(!isfinite(A[i]) || !isfinite(B[i])) ? NaN :
             (A[i] > 0) == (B[i] > 0) ? 0.0 : (B[i] > 0 ? 1.0 : -1.0) for i in CartesianIndices(A)]
        ax2 = Axis(fig[2, k]; title = "coastline: red = wet only in $lb, blue = wet only in $la",
                   titlesize = 11, xlabel = "longitude", ylabel = k == 1 ? "latitude" : "")
        heatmap!(ax2, LONS, LATS, w; colormap = Reverse(:RdBu), colorrange = (-1, 1), nan_color = :gray85)
    end
    save(joinpath(OUTDIR, "bathymetry_difference_maps.png"), fig; px_per_unit = 2)
    @info "wrote $(joinpath(OUTDIR, "bathymetry_difference_maps.png"))"
end

println("\n  figures in $OUTDIR")
