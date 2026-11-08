# ==============================================================================
# model_grid.jl -- a run's vertical levels and sea floor, READ OFF ITS OWN OUTPUT.
#
#     include(joinpath(@__DIR__, "model_grid.jl"))
#     zc, zf, zname     = file_levels(ds)             # or nothing
#     kbot, source      = file_kbot(ds, zc, zf)       # or nothing
#     MID               = section_column(sect_path, Nx)
#
# Shared by tools/compare_mld.jl and tools/compare_bsose.jl so the two cannot drift apart. Both
# used to rebuild src/model.jl's grid (53 stretched levels, GridFittedBottom) for every run, which
# put a nested run's T and S (B-SOSE's 52 levels, partial cells, pocket fill) on the wrong depths
# with no error. Reading the levels and the floor from the file makes every run compared at its
# own depths: src/model.jl's static z, examples/sowinwarp_nested.jl's, a z-star run's reference
# levels.
# ==============================================================================

"""Cell centres and faces of the vertical grid, from `z_aac`/`z_aaf` (static z) or `r_aac`/`r_aaf` (z-star)."""
function file_levels(ds)
    for (c, f) in (("z_aac", "z_aaf"), ("r_aac", "r_aaf"))
        haskey(ds, c) && haskey(ds, f) && return Float64.(ds[c][:]), Float64.(ds[f][:]), c
    end
    return nothing
end

"""
Deepest wet index per column, from the model's own mask if the file has it, else from
`bottom_height`; `nothing` for a dry column. Shaped like the file's horizontal extent, so a
single-meridian section file gives a (1, Ny) matrix.
"""
function file_kbot(ds, zc, zf)
    if haskey(ds, "inactive_nodes_ccc")
        A = ds["inactive_nodes_ccc"][:, :, :]
        return [findfirst(k -> A[i, j, k] == 0, axes(A, 3)) for i in axes(A, 1), j in axes(A, 2)],
               "inactive_nodes_ccc"
    elseif haskey(ds, "bottom_height")
        b = Float64.(coalesce.(ds["bottom_height"][:, :], NaN))
        kind = try ds.group["grid_reconstruction_metadata"].attrib["immersed_boundary_type"] catch; "" end
        # PartialCellBottom keeps the cell the floor cuts through; GridFittedBottom drops it
        # unless its centre is above the floor.
        wet(k, h) = occursin("Partial", kind) ? zf[k + 1] > h : zc[k] > h
        return [isfinite(b[i, j]) && b[i, j] < 0 ? findfirst(k -> wet(k, b[i, j]), eachindex(zc)) : nothing
                for i in axes(b, 1), j in axes(b, 2)],
               "bottom_height ($(isempty(kind) ? "full cells assumed" : kind))"
    end
    return nothing
end

"""The section's column, from the writer's own `indices` attribute, e.g. "(180:180, 1:150, 1:52)"; Nx/2 if absent."""
function section_column(sect_path, Nx)
    isfile(sect_path) || return div(Nx, 2)
    d = Dataset(sect_path)
    s = haskey(d["T"].attrib, "indices") ? string(d["T"].attrib["indices"]) : ""
    close(d)
    m = match(r"^\(\s*(\d+)\s*:", s)
    return m === nothing ? div(Nx, 2) : parse(Int, m[1])
end
