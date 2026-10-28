# ==============================================================================
# bsose_masks.jl -- "is there ocean in BOTH grids here", for every model-vs-B-SOSE figure.
#
#     include(joinpath(@__DIR__, "bsose_masks.jl"))
#     wet = both_wet_surface(grid, dataset, λc, φc)
#     wetsec, kup = both_wet_section(grid, dataset, MID, φc, zc, Z_TOP)
#
# WHY THIS IS NOT `isfinite`. B-SOSE's T and S are FINITE below its own sea floor and under the
# ice shelves: `fill_bathymetry_gaps!` in setup_bsose.jl deliberately propagates the deepest
# valid water mass downward into every immersed cell, so no NaN survives for `nan_color` to
# catch. Our model output is no better -- Oceananigans writes immersed tracer cells as zeros,
# not NaN. Masking a difference on `isfinite` therefore masks nothing at all: the section
# figures drew three degrees of Antarctic continent as if it were ocean, and the panel averages
# were taken over rock.
#
# So the masks are built from each grid's OWN bathymetry and intersected. Ours is ETOPO through
# `regrid_bathymetry`; B-SOSE's is the `Depth` field carried in its own files. They disagree
# along the margin, which is the entire reason both are needed.
# ==============================================================================

using Oceananigans.ImmersedBoundaries: immersed_cell

"""
    bsose_depth(dataset, lons, lats) -> (length(lons), length(lats)) matrix

B-SOSE's own sea-floor depth in metres, positive down and 0 over land, nearest-neighboured onto
the given longitudes and latitudes. Read from `Depth` in the surface-salt-flux file, which is
the smallest B-SOSE file that carries the grid.
"""
function bsose_depth(dataset, lons, lats)
    file = joinpath(dataset.dir,
                    resolve_bsose_filename(dataset.dir, :surface_salt_flux, dataset.iteration))
    d = Dataset(file)
    try
        xc = Float64.(coalesce.(d["XC"][:], NaN))
        yc = Float64.(coalesce.(d["YC"][:], NaN))
        is = [argmin(abs.(xc .- λ)) for λ in lons]
        js = [argmin(abs.(yc .- φ)) for φ in lats]
        # One strided read of the bounding box, not length(lons)*length(lats) scalar reads: the
        # file is global (2160 x 588) and on a networked filesystem the difference is minutes.
        D  = Float64.(coalesce.(d["Depth"][minimum(is):maximum(is), minimum(js):maximum(js)], NaN))
        i0, j0 = minimum(is) - 1, minimum(js) - 1
        return [(v = D[is[a] - i0, js[b] - j0]; isfinite(v) ? max(v, 0.0) : 0.0)
                for a in eachindex(lons), b in eachindex(lats)]
    finally
        close(d)
    end
end

"""
    both_wet_surface(grid, dataset, λc, φc) -> BitMatrix (Nx, Ny)

True where the top cell is ocean in our bathymetry AND B-SOSE has water.
"""
function both_wet_surface(grid, dataset, λc, φc)
    Nx, Ny, Nz = size(grid)
    parent_depth = bsose_depth(dataset, λc, φc)
    return BitMatrix([!immersed_cell(i, j, Nz, grid) && parent_depth[i, j] > 0
                      for i in 1:Nx, j in 1:Ny])
end

"""
    both_wet_section(grid, dataset, i, φc, zc, z_top) -> (BitMatrix (Ny, Nz), kupper)

True where cell (j, k) on the meridian through column `i` is ocean in BOTH bathymetries and
lies in the upper `z_top` metres. Also returns the vertical indices of that upper layer.

`kbot_*[j]` is the DEEPEST WET index of column j, or `nothing` for a dry column; everything
below it is rock in that grid.
"""
function both_wet_section(grid, dataset, i, φc, zc, z_top)
    Nx, Ny, Nz = size(grid)
    lon = Array(λnodes(grid, Center()))[i]

    kbot_model  = [findfirst(k -> !immersed_cell(i, j, k, grid), 1:Nz) for j in 1:Ny]
    depth       = bsose_depth(dataset, [lon], φc)[1, :]
    kbot_parent = [d <= 0 ? nothing : findfirst(k -> -zc[k] <= d, 1:Nz) for d in depth]

    kupper = [k for k in 1:Nz if -zc[k] <= z_top]
    wet = BitMatrix(undef, Ny, Nz)
    fill!(wet, false)
    for j in 1:Ny, k in kupper
        kbot_model[j] === nothing && continue
        kbot_parent[j] === nothing && continue
        wet[j, k] = k >= kbot_model[j] && k >= kbot_parent[j]
    end

    # Interpolation, not @sprintf: its first argument has to be a LITERAL format string, and a
    # concatenated one parses fine and then dies at macro expansion with "First argument to
    # @sprintf must be a format string".
    @info "section $(round(lon, digits = 2))E: " *
          "$(count(!isnothing, kbot_model)) of $Ny columns wet in the model, " *
          "$(count(!isnothing, kbot_parent)) in B-SOSE; " *
          "$(count(wet)) of $(Ny * length(kupper)) upper cells wet in both"
    return wet, kupper
end
