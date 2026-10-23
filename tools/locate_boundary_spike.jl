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

function coords(ds)
    # Oceananigans' NetCDF writer names axes by location; accept whichever is present.
    lon = haskey(ds, "x_caa") ? ds["x_caa"][:] : haskey(ds, "xC") ? ds["xC"][:] : haskey(ds, "lon") ? ds["lon"][:] : nothing
    lat = haskey(ds, "y_aca") ? ds["y_aca"][:] : haskey(ds, "yC") ? ds["yC"][:] : haskey(ds, "lat") ? ds["lat"][:] : nothing
    return lon, lat
end

"""The TOPN largest |A| cells of a 2-D slab, with position and distance to the edges."""
function report_slab(A, lon, lat, label)
    Nx, Ny = size(A)
    finite = [(abs(A[i, j]), i, j) for i in 1:Nx, j in 1:Ny if isfinite(A[i, j])]
    isempty(finite) && (println("  $label: all non-finite"); return)
    sort!(finite; by = first, rev = true)

    @printf("  %s: max %.3f, mean %.4f over %d finite cells\n",
            label, first(finite)[1], mean(first.(finite)), length(finite))
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
    lon, lat = coords(ds)
    η = ds["η"]
    nt = size(η)[end]
    for t in 1:nt
        A = Float64.(coalesce.(η[:, :, t], NaN))
        ndims(A) == 3 && (A = A[:, :, 1])
        println("\n t = $t of $nt")
        report_slab(A, lon, lat, "|η| (m)")
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
    lon, lat = coords(ds)
    nt = size(ds["v"])[end]
    for t in 1:nt
        println("\n t = $t of $nt")
        for name in ("u", "v")
            haskey(ds, name) || continue
            A = Float64.(coalesce.(ds[name][:, :, t], NaN))
            ndims(A) == 3 && (A = A[:, :, 1])
            report_slab(A, lon, lat, "|$name| (m/s)")
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
