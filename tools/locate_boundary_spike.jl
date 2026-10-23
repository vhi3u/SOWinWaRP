# ==============================================================================
# locate_boundary_spike.jl
#
# Where is the blowup?
#
# The 20D_glorys_pinned run holds its face transports to 0.02 Sv and its mean free surface
# flat, so the net-transport failure is fixed -- but max|η| reaches 9 m against a 1.76 m
# initial condition and |v|ₘₐₓ sits at 6.2 m/s. Both are maxima over healthy means, and both
# are quasi-steady rather than growing, which says a few cells are locked in a bad state
# rather than the solution diverging.
#
# This finds those cells instead of guessing at which face is responsible. For each output
# time it reports the largest |η| and |v|, where they sit, and how far they are from each
# open boundary -- because "two cells in from the north-west corner" and "mid-domain" call
# for completely different fixes.
#
#     julia --project=. tools/locate_boundary_spike.jl [RUN_NAME]
#
# Environment knobs:
#     RUN_NAME   defaults to 20D_glorys_pinned
#     TOPN       how many worst cells to list per field (5)
#     DIR        where the .nc files are (the working directory)
# ==============================================================================

using NCDatasets, Printf, Statistics

const RUN  = get(ENV, "RUN_NAME", length(ARGS) > 0 ? ARGS[1] : "20D_glorys_pinned")
const DIR  = get(ENV, "DIR", ".")
const TOPN = parse(Int, get(ENV, "TOPN", "5"))

hdr(s) = (println(); println("="^78); println(s); println("="^78))

"""Distance in cells from (i, j) to each open boundary, and the nearest one."""
function edge_report(i, j, Nx, Ny)
    d = (west = i - 1, east = Nx - i, south = j - 1, north = Ny - j)
    nearest = argmin(d)
    return d, nearest, d[nearest]
end

"""
    axis_values(ds, var)

The x and y coordinate vectors for `var`, read from its OWN dimension names.

Oceananigans names axes by field location -- `u` is on `x_faa` while `T` is on `x_caa` -- so
guessing from a fixed list of names silently returns `nothing` and prints NaN, which is what
the first version of this script did.
"""
function axis_values(ds, var)
    dn = dimnames(ds[var])
    get_or_nothing(n) = (n !== nothing && haskey(ds, n)) ? Float64.(ds[n][:]) : nothing
    xn = length(dn) >= 1 ? dn[1] : nothing
    yn = length(dn) >= 2 ? dn[2] : nothing
    return get_or_nothing(xn), get_or_nothing(yn), dn
end

"""
    slab(ds, var, t)

The horizontal field of `var` at output index `t`, whatever its rank.

The writers emit 2-D fields as (x, y, time) and level-sliced 3-D fields as (x, y, z, time)
with z of length one. Indexing the latter as if it were the former reads the z axis with a
time index -- a `BoundsError` past t = 1, and silently the wrong slab at t = 1.
"""
function slab(ds, var, t)
    v = ds[var]
    nd = ndims(v)
    A = nd == 4 ? v[:, :, 1, t] :
        nd == 3 ? v[:, :, t]    :
        error("unexpected rank $nd for `$var` with dims $(dimnames(v))")
    return Float64.(coalesce.(A, NaN))
end

"""The TOPN largest |A| cells of a 2-D slab, with position and distance to the edges."""
function report_slab(A, lon, lat, label)
    Nx, Ny = size(A)
    finite = [(abs(A[i, j]), i, j) for i in 1:Nx, j in 1:Ny if isfinite(A[i, j])]
    isempty(finite) && (println("  $label: all non-finite"); return)
    sort!(finite; by = first, rev = true)

    vals = [A[i, j] for i in 1:Nx, j in 1:Ny if isfinite(A[i, j])]
    @printf("  %s: max|.| %.3f, range %+.3f to %+.3f, mean %+.4f over %d finite cells\n",
            label, first(finite)[1], minimum(vals), maximum(vals), mean(vals), length(finite))
    for (v, i, j) in finite[1:min(TOPN, length(finite))]
        d, nearest, dist = edge_report(i, j, Nx, Ny)
        λ = lon === nothing ? NaN : lon[min(i, length(lon))]
        φ = lat === nothing ? NaN : lat[min(j, length(lat))]
        @printf("     %8.3f at (i=%3d, j=%3d)  λ=%7.3f φ=%7.3f  | W %3d E %3d S %3d N %3d cells -> nearest %s (%d)\n",
                v, i, j, λ, φ, d.west, d.east, d.south, d.north, nearest, dist)
    end
end

# ------------------------------------------------------------------ free surface
fs_path = joinpath(DIR, "$(RUN)_free_surface.nc")
if isfile(fs_path)
    hdr("Free surface: $(basename(fs_path))")
    ds = Dataset(fs_path)
    lon, lat, dn = axis_values(ds, "η")
    @info "η dims: $dn, size $(size(ds["η"]))"
    nt = size(ds["η"])[end]
    for t in 1:nt
        println("\n t = $t of $nt")
        report_slab(slab(ds, "η", t), lon, lat, "η (m)")
    end
    close(ds)
else
    @warn "no free-surface file at $fs_path"
end

# ------------------------------------------------------------------ surface velocities
sf_path = joinpath(DIR, "$(RUN)_surface.nc")
if isfile(sf_path)
    hdr("Surface fields: $(basename(sf_path))  (top level only -- a deeper max will not appear here)")
    ds = Dataset(sf_path)
    nt = size(ds["v"])[end]
    for name in ("u", "v")
        haskey(ds, name) && @info "$name dims: $(dimnames(ds[name])), size $(size(ds[name]))"
    end
    for t in 1:nt
        println("\n t = $t of $nt")
        for name in ("u", "v")
            haskey(ds, name) || continue
            lon, lat, _ = axis_values(ds, name)
            report_slab(slab(ds, name, t), lon, lat, "$name (m/s)")
        end
    end
    close(ds)
else
    @warn "no surface file at $sf_path"
end

println("""

Reading this:
  * A spike within 1-4 cells of a face is the open boundary itself -- and if it is within
    `GLORYS_N_MATCH` (4) cells, the bathymetry matching reshaped that column, so check its
    depth: `minimum_depth = 10 m` plus a modest transport is metres per second.
  * A spike at a CORNER (small distance to two faces at once) is the two conditions
    disagreeing, not either one being wrong on its own.
  * A spike well inside the domain is not a boundary problem at all.
  * If |v| looks healthy at the surface while the log reports 6.2 m/s, the maximum is at
    depth -- the surface file only holds level Nz.
""")
