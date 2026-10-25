# ==============================================================================
# boundary_components.jl -- is the boundary feature NORMAL or TANGENTIAL?
#
#     julia --project=. tools/boundary_components.jl [RUN_NAME]
#
# `map_boundary_velocities.jl` plots speed, √(u²+v²), which cannot tell the two apart. That
# distinction decides which of two different bugs you have, with two different fixes:
#
#   NORMAL dominant   (u on west/east, v on south/north)
#       The water crossing the boundary is wrong. Suspects: Flather, the transport pin, or the
#       exterior transport Uᵉˣᵗ. The net through each face is already logged daily, so compare.
#
#   TANGENTIAL dominant   (v on west/east, u on south/north)
#       The flow running ALONG the boundary is wrong. Prime suspect: the baroclinic and
#       barotropic halves of the same face disagreeing. The split-explicit free surface gives
#       the barotropic tangential component a NoFlux (zero-gradient) condition at those faces
#       no matter what the 3-D field carries, so a prescribed or radiated Value on the 3-D
#       tangential leaves the two halves inconsistent; the mirrored halo v_halo = 2v̄ - v₁ then
#       drives an overshoot against the biharmonic viscosity. That mechanism predicts a jet
#       pinned ALONG the western edge, which is the symptom recorded for this model in 2026-09.
#       The recorded fix was TANGENTIAL_BC = :gradient, never confirmed at 1/6 degree.
#
# Output: a table per face (the part that matters) and a figure of u and v along each face.
#
# Environment knobs:
#     DIR     where the .nc files live (default: working directory)
#     DAYS    comma-separated output indices, or "first,mid,last" (default)
#     OUTDIR  where figures go (tools/boundary_maps)
#     TRIM_CORNERS  drop N cells from each end of every face line (default 0). Run it twice.
#                   A face's inward lines span the whole perpendicular extent, so a feature on
#                   a NEIGHBOURING wall is counted in this face's numbers too -- which is how
#                   the north face came to report its normal component RISING inward to 2.99 m/s
#                   on 2026-10-25 when the signal was the west wall's v at (i=1, j~130).
# ==============================================================================

using NCDatasets, CairoMakie, Printf, Statistics

const RUN    = get(ENV, "RUN_NAME", length(ARGS) > 0 ? ARGS[1] : "20D_bsose_v1")
const DIR    = get(ENV, "DIR", ".")
const OUTDIR = get(ENV, "OUTDIR", joinpath(@__DIR__, "boundary_maps"))
const TRIM   = parse(Int, get(ENV, "TRIM_CORNERS", "0"))
mkpath(OUTDIR)

sf = joinpath(DIR, "$(RUN)_surface.nc")
isfile(sf) || error("no surface file at $sf")
ds = Dataset(sf)

slab(var, t) = (v = ds[var]; A = ndims(v) == 4 ? v[:, :, 1, t] : v[:, :, t];
                Float64.(coalesce.(A, NaN)))

nt = size(ds["v"])[end]
days = let s = get(ENV, "DAYS", "first,mid,last")
    s == "first,mid,last" ? unique([1, cld(nt, 2), nt]) :
        [parse(Int, x) for x in split(s, ",") if !isempty(x)]
end

"""
Model `u` and `v` on a line `d` cells in from `side`, as (normal, tangential).

u lives on x-faces (Nx+1, Ny), v on y-faces (Nx, Ny+1), so the normal component is read on the
face itself and the tangential at the adjacent cell. West/east: normal = u, tangential = v.
South/north: normal = v, tangential = u.
"""
function face_lines(u, v, side, d)
    Nx, Ny = size(u, 1) - 1, size(v, 2) - 1
    n, t = if side === :west
        u[1 + d, 1:Ny],       v[1 + d, 1:Ny]
    elseif side === :east
        u[Nx + 1 - d, 1:Ny],  v[Nx - d, 1:Ny]
    elseif side === :south
        v[1:Nx, 1 + d],       u[1:Nx, 1 + d]
    else
        v[1:Nx, Ny + 1 - d],  u[1:Nx, Ny - d]
    end
    return trim_corners(n), trim_corners(t)
end

"""Drop `TRIM` cells from each end of a face line, leaving it untouched if that would empty it."""
function trim_corners(x)
    TRIM <= 0 && return x
    n = length(x)
    return 2TRIM >= n ? x : x[(TRIM + 1):(n - TRIM)]
end

finite(x) = filter(isfinite, x)
peak(x) = (f = finite(abs.(x)); isempty(f) ? NaN : maximum(f))
p99(x)  = (f = finite(abs.(x)); isempty(f) ? NaN : quantile(f, 0.99))

# ------------------------------------------------------------------ the table
println()
println(TRIM > 0 ?
        "corners trimmed: $TRIM cells off each end of every face line" :
        "corners NOT trimmed -- rerun with TRIM_CORNERS=30 to see which face owns each signal")
println()
@printf("%-6s %4s | %18s | %18s | %s\n", "face", "day", "NORMAL max (p99)", "TANGENTIAL max (p99)", "verdict at the edge")
println("-"^92)

for side in (:west, :east, :south, :north), t in days
    u, v = slab("u", t), slab("v", t)
    n0, g0 = face_lines(u, v, side, 0)
    pn, pg = peak(n0), peak(g0)
    verdict = if !isfinite(pn) || !isfinite(pg) || (pn == 0 && pg == 0)
        "face is dry"
    elseif pg > 1.5pn
        "TANGENTIAL dominates ($(round(pg/pn, digits=1))x)"
    elseif pn > 1.5pg
        "NORMAL dominates ($(round(pn/pg, digits=1))x)"
    else
        "comparable"
    end
    @printf("%-6s %4d | %8.3f (%6.3f) | %8.3f (%6.3f) | %s\n",
            side, t - 1, pn, p99(n0), pg, p99(g0), verdict)
end

# ------------------------------------------- decay away from the edge, per component
println("\nHow each component decays inward (max |.|, m/s), last output:")
@printf("%-6s %8s %8s %8s %8s %8s %8s\n", "face", "d=0", "d=1", "d=2", "d=4", "d=8", "d=16")
let t = last(days)
    u, v = slab("u", t), slab("v", t)
    for side in (:west, :east, :south, :north)
        nm = [peak(face_lines(u, v, side, d)[1]) for d in (0,1,2,4,8,16)]
        tg = [peak(face_lines(u, v, side, d)[2]) for d in (0,1,2,4,8,16)]
        @printf("%-6s %8s %8.3f %8.3f %8.3f %8.3f %8.3f   normal\n", side, @sprintf("%.3f", nm[1]), nm[2], nm[3], nm[4], nm[5], nm[6])
        @printf("%-6s %8s %8.3f %8.3f %8.3f %8.3f %8.3f   tangential\n", "", @sprintf("%.3f", tg[1]), tg[2], tg[3], tg[4], tg[5], tg[6])
    end
end

# ------------------------------------------------------------------ the figure
for t in days
    u, v = slab("u", t), slab("v", t)
    fig = Figure(size = (1400, 950))
    Label(fig[0, 1:2], "$(RUN) — day $(t-1): boundary velocity by component" *
                       (TRIM > 0 ? "  (corners trimmed $TRIM)" : ""),
          fontsize = 20, font = :bold)

    for (k, side) in enumerate((:west, :east, :south, :north))
        ax = Axis(fig[cld(k, 2), 2 - k % 2],
                  title = "$(side) face",
                  xlabel = side in (:west, :east) ? "j (south → north)" : "i (west → east)",
                  ylabel = "velocity (m/s)")
        for (d, α) in ((0, 1.0), (2, 0.55), (8, 0.28))
            n, g = face_lines(u, v, side, d)
            lines!(ax, 1:length(n), n, color = (:firebrick, α), linewidth = d == 0 ? 2 : 1,
                   label = d == 0 ? "normal (d=0)" : nothing)
            lines!(ax, 1:length(g), g, color = (:dodgerblue, α), linewidth = d == 0 ? 2 : 1,
                   label = d == 0 ? "tangential (d=0)" : nothing)
        end
        hlines!(ax, [0], color = :gray, linestyle = :dash)
        k == 1 && axislegend(ax, position = :lt, labelsize = 11)
    end

    f = joinpath(OUTDIR, "$(RUN)_components_day$(lpad(t-1, 2, '0')).png")
    save(f, fig)
    @info "wrote $f"
end

close(ds)

println("""

Reading this:
  * TANGENTIAL dominant on west/east => the 3-D tangential condition is fighting the
    barotropic solver's zero-gradient. Try TANGENTIAL_BC = :gradient, which matches what the
    barotropic half already does. This is the documented 2026-09 fix for the western jet,
    never confirmed at 1/6 degree.
  * NORMAL dominant => the water crossing the face is wrong, not the water running along it.
    Cross-check against the per-face transport log: if the Sv are on target but the velocity
    is not, the PROFILE along the face is wrong while its integral is right -- which is what
    `target_transport` guarantees by construction and cannot detect.
  * Both comparable, decaying within 1-2 cells => a halo artifact rather than a jet.
  * A component that RISES inward is not a boundary pileup. Rerun with TRIM_CORNERS=30: if it
    collapses, you were reading a feature on the adjoining wall, because every face's inward
    lines span the full perpendicular extent and `peak` maxes over all of it.
""")
