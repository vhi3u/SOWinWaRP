# ==============================================================================================
# compare_bathymetry.jl -- the seafloor the model actually runs on, against B-SOSE's.
#
#     julia --project=. tools/compare_bathymetry.jl
#     SOURCES=model,nofix,ibcso julia --project=. tools/compare_bathymetry.jl
#     FIX_BOUNDARY=0 julia --project=. tools/compare_bathymetry.jl
#
# THE GRID IS NOT SAVED ANYWHERE, so this REBUILDS it with src/model.jl's own recipe, read from
# that file on 2026-11-03 (model.jl:452) and reproduced here line for line:
#
#     bottom_height = regrid_bathymetry(grid, height_above_water=1,
#                                       minimum_depth=10, interpolation_passes=5)
#     fix_boundary_topography!(bottom_height, grid; minimum_boundary_depth = 100)
#     grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom_height))
#
# Two things about that recipe are easy to miss and both matter here:
#   - there is NO `dataset` kwarg, so the model is on ETOPO2022 -- NumericalEarth's default.
#     We have never run on IBCSO despite it being the obvious Southern Ocean choice.
#   - there is no `major_basins` either, so it defaults to 1.
# If model.jl changes, these defaults must be changed with it; they are not imported.
#
# WHY IT MATTERS. The model is forced, initialised and boundary-fed from B-SOSE, so B-SOSE's
# seafloor is the one its transports were computed on. Where the two disagree, a boundary column
# is a different depth in the model than in the data feeding it -- which is how the west-face
# pileup happened. `fix_boundary_topography!` was the fix, and it is applied here BY DEFAULT so
# what you see is the real thing; SOURCES=model,nofix shows what that step changed.
#
# COVERAGE DIFFERS AND IS ENFORCED. IBCSOv2 spans -90 to -50 ONLY. NumericalEarth's
# `validate_dataset_coverage` ERRORS if the target grid reaches north of 50S, so IBCSO cannot be
# regridded on the full 70-45S domain at all: it gets its own 70-50S sub-grid here and is padded
# with NaN north of 50S. Every statistic is computed over the overlap only.
#
# IMPORT PATHS DIFFER TOO: `ETOPO2022` is exported by NumericalEarth; `IBCSOv2` and `GEBCO2026`
# are NOT at top level in 0.8.2 and come from their submodules. Verified, not assumed.
#
# COMMON CURRENCY IS DEPTH, metres, positive down, land = 0. NumericalEarth returns
# `bottom_height` (negative below sea level, positive land elevation); B-SOSE's grid.nc stores
# `Depth` positive-down with land 0. Both map to depth = max(0, -h), so a difference is a
# difference in water depth and nothing else.
#
# ENV
#   SOURCES     model,nofix,ibcso,gebco      (default: model,ibcso)
#   BSOSE_GRID  B-SOSE grid file             (default: data/grid.nc)
#   FIX_BOUNDARY  1/0, apply the boundary conditioning (default: 1, as model.jl)
#   MIN_BOUNDARY_DEPTH  m                    (default: 100, as model.jl)
#   PASSES / BASINS / MIN_DEPTH / HEIGHT_ABOVE_WATER   the regrid knobs (model.jl's values)
#   NZ_VIEW     z-limit for the 3-D plots, m (default: 5800, = Z_EXTENT)
#   OUTDIR      figures                      (default: plots)
#
# DOWNLOADS, by HTTP HEAD 2026-11-03: ETOPO2022 0.44 GB, IBCSOv2 0.23 GB, into NumericalEarth's
# own cache (not data/), reused across runs.
# ==============================================================================================

using Oceananigans, NumericalEarth, NCDatasets, Printf, Statistics, CairoMakie
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Grids: λnodes, φnodes
using NumericalEarth.DataWrangling.IBCSO: IBCSOv2
using NumericalEarth.DataWrangling.GEBCO: GEBCO2026

const BGRID   = get(ENV, "BSOSE_GRID", "data/grid.nc")
const OUTDIR  = get(ENV, "OUTDIR", "plots")
const NZVIEW  = parse(Float64, get(ENV, "NZ_VIEW", "5800"))
const SOURCES = [lowercase(strip(s)) for s in split(get(ENV, "SOURCES", "model,ibcso"), ",") if !isempty(strip(s))]

# ---- model.jl's values. Changed there => change here. ----
const PASSES   = parse(Int,     get(ENV, "PASSES", "5"))                  # model.jl:455
const BASINS   = parse(Int,     get(ENV, "BASINS", "1"))                  # the default; not set in model.jl
const MIN_DEP  = parse(Float64, get(ENV, "MIN_DEPTH", "10"))              # model.jl:454
const HAW      = parse(Float64, get(ENV, "HEIGHT_ABOVE_WATER", "1"))      # model.jl:453
const FIXBDY   = get(ENV, "FIX_BOUNDARY", "1") == "1"                     # model.jl:110
const MINBDY   = parse(Float64, get(ENV, "MIN_BOUNDARY_DEPTH", "100"))    # model.jl:111

const λ₁, λ₂ = 90.0, 150.0          # model.jl:414
const φ₁, φ₂ = -70.0, -45.0         # model.jl:415
const SCALING = 6                   # model.jl:81
const Nx, Ny  = Int(SCALING * (λ₂ - λ₁)), Int(SCALING * (φ₂ - φ₁))   # 360 x 150
const IBCSO_φ₂ = -50.0              # IBCSOv2's hard northern limit

# `fix_boundary_topography!` lives in setup_bsose.jl. Include rather than reimplement: a second
# copy of the boundary conditioning would be the one thing in this comparison guaranteed to
# drift away from what the model does.
const HAVE_FIX = try
    include(joinpath(@__DIR__, "..", "src", "setup_bsose.jl"))
    isdefined(Main, :fix_boundary_topography!)
catch e
    @warn "could not load src/setup_bsose.jl -- the boundary conditioning will be SKIPPED, so\n" *
          "    `model` here is the raw regrid and not what model.jl runs.\n    " *
          first(split(sprint(showerror, e), '\n'))
    false
end

"""The model's horizontal grid. Nz = 1: only the horizontal is used for bathymetry."""
hgrid(south, north, ny) =
    LatitudeLongitudeGrid(CPU(); size = (Nx, ny, 1), halo = (7, 7, 7),
                          longitude = (λ₁, λ₂), latitude = (south, north), z = (-NZVIEW, 0))

const GRID = hgrid(φ₁, φ₂, Ny)
# λnodes/φnodes, NOT xnodes/ynodes. On a LatitudeLongitudeGrid `xnodes(grid, ℓx, ℓy)` converts
# through `grid.radius` and returns Cartesian METRES; it would have given plausible-looking axes
# in entirely the wrong units. λnodes/φnodes are the ones that return degrees.
const LONS = Array(λnodes(GRID, Center()))
const LATS = Array(φnodes(GRID, Center()))

depth_of(h) = max(0.0, -h)      # bottom_height -> depth, positive down, land 0

# ── B-SOSE ────────────────────────────────────────────────────────────────────────────────────
"""B-SOSE's `Depth` bilinearly interpolated onto the model grid. Land (Depth<=0) stays 0."""
function bsose_depth()
    isfile(BGRID) || error("no $BGRID -- set BSOSE_GRID")
    ds = Dataset(BGRID)
    try
        # XC/YC are stored 2-D but the grid is separable (verified: XC constant along dim 2,
        # YC constant along dim 1). XG/YG are CORRUPT in the i156 grid file -- never use them.
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

# ── the model's recipe, applied to a chosen dataset ───────────────────────────────────────────
"""model.jl's bathymetry construction, verbatim, for `dataset` on `grid`. Returns depth."""
function model_recipe(grid, dataset; fix)
    bh = regrid_bathymetry(grid; dataset, height_above_water = HAW,
                           minimum_depth = MIN_DEP, interpolation_passes = PASSES,
                           major_basins = BASINS)
    if fix && HAVE_FIX
        fix_boundary_topography!(bh, grid; minimum_boundary_depth = MINBDY)
        fill_halo_regions!(bh)
    end
    return depth_of.(Array(interior(bh, :, :, 1)))
end

function source_depth(name)
    if name == "model"
        return "model (ETOPO2022)" => model_recipe(GRID, ETOPO2022(); fix = FIXBDY)
    elseif name == "nofix"
        return "ETOPO2022, no bdy fix" => model_recipe(GRID, ETOPO2022(); fix = false)
    elseif name == "gebco"
        return "GEBCO2026" => model_recipe(GRID, GEBCO2026(); fix = FIXBDY)
    elseif name == "ibcso"
        # The validator ERRORS north of 50S, so ask only for what IBCSO has and pad the rest.
        ny = Int(SCALING * (IBCSO_φ₂ - φ₁))
        d  = model_recipe(hgrid(φ₁, IBCSO_φ₂, ny), IBCSOv2(); fix = FIXBDY)
        out = fill(NaN, Nx, Ny)
        out[:, 1:ny] .= d          # 50-45S stays NaN: absent data, NOT zero depth
        return "IBCSOv2 (70-50S)" => out
    end
    error("unknown source $name; use model, nofix, ibcso or gebco")
end

@info "grid $(Nx)x$(Ny) over $(λ₁)-$(λ₂)E, $(φ₁)-$(φ₂)S"
@info "recipe: height_above_water=$HAW minimum_depth=$MIN_DEP interpolation_passes=$PASSES " *
      "major_basins=$BASINS boundary_fix=$(FIXBDY && HAVE_FIX ? "on @ $(MINBDY) m" : "OFF")"

SRC = Pair{String,Matrix{Float64}}["B-SOSE" => bsose_depth()]
for s in SOURCES
    @info "building $s (first run downloads: ETOPO 0.44 GB, IBCSO 0.23 GB)"
    try
        push!(SRC, source_depth(s))
    catch e
        @warn "$s failed -- skipping it\n    $(first(split(sprint(showerror, e), '\n')))"
    end
end
length(SRC) < 2 && error("only B-SOSE available; nothing to compare against")

# ── statistics over the common coverage ───────────────────────────────────────────────────────
both(A, B) = [isfinite(A[i]) && isfinite(B[i]) for i in eachindex(A)]

println("\n" * "="^104)
println("BATHYMETRY -- depth in metres, positive down, land = 0")
println("="^104)
@printf("\n  model.jl recipe: height_above_water=%.0f minimum_depth=%.0f passes=%d basins=%d\n",
        HAW, MIN_DEP, PASSES, BASINS)
println("  boundary conditioning: " *
        (FIXBDY && HAVE_FIX ? "ON, minimum_boundary_depth = $(Int(MINBDY)) m (as model.jl)" :
         FIXBDY ? "REQUESTED but setup_bsose.jl did not load -- NOT applied" : "off"))
println("  the model is on ETOPO2022: model.jl passes no `dataset`, and that is the default.\n")

@printf("  %-26s %8s %9s %9s %9s %8s %8s\n",
        "pair", "n cells", "mean", "rms", "max |d|", "at lon", "at lat")
@printf("  %-26s %8s %9s %9s %9s %8s %8s\n", "", "", "(m)", "(m)", "(m)", "", "")
for a in 1:length(SRC), b in (a + 1):length(SRC)
    (la, A), (lb, B) = SRC[a], SRC[b]
    m = both(A, B); n = count(m)
    n == 0 && (@printf("  %-26s %8s\n", "$lb - $la", "no overlap"); continue)
    idx = findall(m)
    d = [B[i] - A[i] for i in idx]
    ci = CartesianIndices(A)[idx[argmax(abs.(d))]]
    @printf("  %-26s %8d %9.1f %9.1f %9.1f %8.2f %8.2f\n",
            "$lb - $la", n, mean(d), sqrt(mean(d .^ 2)), maximum(abs.(d)),
            LONS[ci[1]], LATS[ci[2]])
end

println("\n  WET/DRY DISAGREEMENT -- cells one calls ocean and the other calls land.")
println("  The one that changes a model: it moves the coastline and the open-boundary columns.")
for a in 1:length(SRC), b in (a + 1):length(SRC)
    (la, A), (lb, B) = SRC[a], SRC[b]
    m = both(A, B); n = count(m); n == 0 && continue
    aonly = count(i -> m[i] && A[i] > 0 && !(B[i] > 0), eachindex(A))
    bonly = count(i -> m[i] && !(A[i] > 0) && B[i] > 0, eachindex(A))
    @printf("    %-26s %5d of %6d (%.2f%%) -- %d wet only in %s, %d wet only in %s\n",
            "$lb vs $la", aonly + bonly, n, 100 * (aonly + bonly) / n, aonly, la, bonly, lb)
end

# The open boundaries specifically: four rows/columns, and where the pileup came from.
println("\n  OPEN-BOUNDARY ROWS ONLY -- mean |difference| on each face (m)")
@printf("    %-26s %9s %9s %9s %9s\n", "pair", "west", "east", "south", "north")
for a in 1:length(SRC), b in (a + 1):length(SRC)
    (la, A), (lb, B) = SRC[a], SRC[b]
    face(ii, jj) = (v = [abs(B[i, j] - A[i, j]) for i in ii, j in jj
                         if isfinite(A[i, j]) && isfinite(B[i, j])];
                    isempty(v) ? NaN : mean(v))
    f(v) = isfinite(v) ? @sprintf("%9.1f", v) : lpad("--", 9)
    @printf("    %-26s %s %s %s %s\n", "$lb - $la",
            f(face(1:1, 1:Ny)), f(face(Nx:Nx, 1:Ny)), f(face(1:Nx, 1:1)), f(face(1:Nx, Ny:Ny)))
end

# ── figures ───────────────────────────────────────────────────────────────────────────────────
mkpath(OUTDIR)
zlim = (-NZVIEW, 200.0)
surf(A) = [isfinite(v) ? -v : NaN for v in A]    # plot as bottom height so it looks like seafloor

# 1. THE ASK: every source in ONE Axis3, translucent so the overlaps read.
let
    fig = Figure(size = (1240, 900))
    Label(fig[0, 1], "Seafloor, all sources overlaid — $(Int(λ₁))-$(Int(λ₂))E, $(Int(-φ₁))-$(Int(-φ₂))S",
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
    Label(fig[2, 1], "B-SOSE is the opaque black surface. Where a coloured surface shows through " *
          "BELOW it, that dataset is deeper there.", fontsize = 11, color = :gray40)
    save(joinpath(OUTDIR, "bathymetry_overlay3d.png"), fig; px_per_unit = 2)
    @info "wrote $(joinpath(OUTDIR, "bathymetry_overlay3d.png"))"
end

# 2. one surface per source, shared colour and z limits
let n = length(SRC)
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

# 3. the differences in 3-D: shape is B-SOSE's seafloor, COLOUR is the disagreement
PAIRS = [(1, b) for b in 2:length(SRC)]
let
    fig = Figure(size = (560 * length(PAIRS), 660))
    Label(fig[0, 1:length(PAIRS)],
          "Where they disagree — surface is B-SOSE, colour is the difference in depth",
          fontsize = 18, font = :bold)
    lim = 1.0
    for (a, b) in PAIRS
        A, B = SRC[a][2], SRC[b][2]
        d = filter(isfinite, [isfinite(A[i]) && isfinite(B[i]) ? abs(B[i] - A[i]) : NaN
                              for i in eachindex(A)])
        isempty(d) || (lim = max(lim, quantile(d, 0.98)))
    end
    local hp = nothing
    for (k, (a, b)) in enumerate(PAIRS)
        (la, A), (lb, B) = SRC[a], SRC[b]
        d = [isfinite(A[i]) && isfinite(B[i]) ? B[i] - A[i] : NaN for i in eachindex(A)]
        ax = Axis3(fig[1, k]; title = "$lb − $la", titlesize = 13, xlabel = "lon", ylabel = "lat",
                   zlabel = k == 1 ? "bottom height (m)" : "", azimuth = 0.62π, elevation = 0.17π,
                   aspect = (2.2, 1.3, 0.85), limits = (nothing, nothing, zlim))
        hp = surface!(ax, LONS, LATS, surf(A); color = d, colormap = Reverse(:RdBu),
                      colorrange = (-lim, lim), shading = NoShading, nan_color = :gray85)
    end
    Colorbar(fig[1, length(PAIRS) + 1], hp, label = "difference in depth (m), red = deeper")
    Label(fig[2, 1:length(PAIRS)],
          @sprintf("colour limit is the 98th percentile of |difference| (%.0f m); grey = no data in that source", lim),
          fontsize = 11, color = :gray40)
    save(joinpath(OUTDIR, "bathymetry_difference3d.png"), fig; px_per_unit = 2)
    @info "wrote $(joinpath(OUTDIR, "bathymetry_difference3d.png"))"
end

# 4. flat maps -- easiest to localise a feature on, plus the coastline disagreement
let ncol = length(PAIRS)
    fig = Figure(size = (540 * ncol, 780))
    Label(fig[0, 1:ncol], "Difference maps and coastline disagreement", fontsize = 18, font = :bold)
    for (k, (a, b)) in enumerate(PAIRS)
        (la, A), (lb, B) = SRC[a], SRC[b]
        d = [isfinite(A[i]) && isfinite(B[i]) ? B[i] - A[i] : NaN for i in eachindex(A)]
        fin = filter(isfinite, d)
        lim = isempty(fin) ? 1.0 : max(quantile(abs.(fin), 0.98), 1.0)
        ax = Axis(fig[1, k]; title = "$lb − $la  (m, red = deeper)", titlesize = 12,
                  xlabel = "longitude", ylabel = k == 1 ? "latitude" : "")
        hm = heatmap!(ax, LONS, LATS, d; colormap = Reverse(:RdBu), colorrange = (-lim, lim),
                      nan_color = :gray85)
        Colorbar(fig[1, ncol + k], hm)
        w = [(!isfinite(A[i]) || !isfinite(B[i])) ? NaN :
             (A[i] > 0) == (B[i] > 0) ? 0.0 : (B[i] > 0 ? 1.0 : -1.0) for i in eachindex(A)]
        ax2 = Axis(fig[2, k]; title = "coastline: red = wet only in $lb, blue = wet only in $la",
                   titlesize = 11, xlabel = "longitude", ylabel = k == 1 ? "latitude" : "")
        heatmap!(ax2, LONS, LATS, w; colormap = Reverse(:RdBu), colorrange = (-1, 1), nan_color = :gray85)
    end
    save(joinpath(OUTDIR, "bathymetry_difference_maps.png"), fig; px_per_unit = 2)
    @info "wrote $(joinpath(OUTDIR, "bathymetry_difference_maps.png"))"
end

println("\n  figures in $OUTDIR")
