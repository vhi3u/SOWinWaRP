# ==============================================================================
# map_boundary_velocities.jl
#
# Where are the fast velocities, and is anything piling up against a boundary?
#
# `locate_boundary_spike.jl` answers "which cell" as a table. This answers "what does it look
# like" as a map, which is the question you cannot settle from a top-5 list: a single stuck
# cell, a jet hugging one edge, and a genuine current exiting the domain all produce similar
# numbers and completely different pictures.
#
# Three panels per selected day:
#
#   1. Surface speed √(u²+v²) with the boundary zone outlined, so a feature glued to an edge
#      is obvious at a glance.
#   2. The same field clipped to a percentile, so a 6 m/s cell cannot wash out a 0.5 m/s
#      field everywhere else -- the usual reason these maps look empty.
#   3. Distance-from-boundary profiles: for each of the four edges, the maximum and the 99th
#      percentile of speed as a function of how many cells in you are. A pileup is a spike at
#      distance 0 that decays within a few cells; a real current is flat.
#
# Plus a summary time series: max speed inside vs outside an exclusion band, which is the
# number that decides whether a boundary problem is contaminating the interior. Ito et al.
# (2026) exclude ~300 km from their open boundaries when evaluating the Gulf of Alaska model;
# EXCLUDE_KM applies the same convention here.
#
#     julia --project=. tools/map_boundary_velocities.jl [RUN_NAME]
#
# Environment knobs:
#     RUN_NAME     defaults to 20D_glorys_pinned
#     DIR          where the .nc files live (working directory)
#     DAYS         comma-separated output indices to map, or "first,mid,last" (default)
#     EXCLUDE_KM   width of the boundary exclusion band, km (300, after Ito et al.)
#     CLIP_PCT     percentile for the clipped panel (99.0)
#     TRIM_CORNERS cells dropped from each end of every edge line before the profile
#                  statistics (0). Set it to ~30 to ask whether a face is bad or whether a
#                  corner is setting the percentile for both faces that meet there.
#     OUTDIR       where figures go (tools/boundary_maps)
# ==============================================================================

using NCDatasets, CairoMakie, Printf, Statistics

const RUN        = get(ENV, "RUN_NAME", length(ARGS) > 0 ? ARGS[1] : "20D_glorys_pinned")
const DIR        = get(ENV, "DIR", ".")
const EXCLUDE_KM = parse(Float64, get(ENV, "EXCLUDE_KM", "300"))
const CLIP_PCT   = parse(Float64, get(ENV, "CLIP_PCT", "99.0"))
const TRIM       = parse(Int,     get(ENV, "TRIM_CORNERS", "0"))
const OUTDIR     = get(ENV, "OUTDIR", joinpath(@__DIR__, "boundary_maps"))

mkpath(OUTDIR)

"""Coordinate vectors for `var`, read from its own dimension names (u is on x_faa, T on x_caa)."""
function axis_values(ds, var)
    dn = dimnames(ds[var])
    grab(n) = (n !== nothing && haskey(ds, n)) ? Float64.(ds[n][:]) : nothing
    return grab(length(dn) >= 1 ? dn[1] : nothing), grab(length(dn) >= 2 ? dn[2] : nothing)
end

"""Horizontal slab at output index `t`, for a 3-D (x,y,t) or 4-D (x,y,z,t) variable."""
function slab(ds, var, t)
    v = ds[var]
    A = ndims(v) == 4 ? v[:, :, 1, t] : ndims(v) == 3 ? v[:, :, t] :
        error("unexpected rank $(ndims(v)) for `$var`")
    return Float64.(coalesce.(A, NaN))
end

"""u and v interpolated to cell centres so speed is defined on one grid."""
function centred_speed(ds, t)
    u = slab(ds, "u", t)        # (Nx+1, Ny) on faces in x
    v = slab(ds, "v", t)        # (Nx, Ny+1) on faces in y
    Nx, Ny = size(u, 1) - 1, size(v, 2) - 1
    uc = 0.5 .* (u[1:Nx, 1:Ny] .+ u[2:Nx+1, 1:Ny])
    vc = 0.5 .* (v[1:Nx, 1:Ny] .+ v[1:Nx, 2:Ny+1])
    return sqrt.(uc .^ 2 .+ vc .^ 2), uc, vc, Nx, Ny
end

"""Cells corresponding to `km`, from the actual grid spacing at mid-domain."""
function band_cells(lon, lat, km)
    Δx = lon === nothing ? 15.0 : 111.0 * abs(lon[2] - lon[1]) * cosd(mean(lat))
    Δy = lat === nothing ? 15.0 : 111.0 * abs(lat[2] - lat[1])
    return max(1, round(Int, km / Δx)), max(1, round(Int, km / Δy)), Δx, Δy
end

"""
Max and 99th percentile of `S` as a function of distance in cells from each edge.

`trim` drops that many cells from each END of every line before the statistics, which separates
"this face is bad" from "a corner is bad and it is setting the percentile for both faces that
meet there". A corner feature spanning ~10 cells is the top 3% of a 360-cell row, so it alone
can set that row's p99. If a face's profile collapses to the interior value under a modest trim,
the face is clean and the corner owns the signal.
"""
function edge_profiles(S; trim = 0)
    Nx, Ny = size(S)
    out = Dict{Symbol,Tuple{Vector{Float64},Vector{Float64}}}()
    for (side, n, sel) in ((:west,  Nx, d -> S[d + 1, :]),
                           (:east,  Nx, d -> S[Nx - d, :]),
                           (:south, Ny, d -> S[:, d + 1]),
                           (:north, Ny, d -> S[:, Ny - d]))
        dmax = min(60, n - 1)
        mx, p99 = Float64[], Float64[]
        for d in 0:dmax
            raw = sel(d)
            t = min(trim, (length(raw) - 1) ÷ 2)
            line = filter(isfinite, raw[1+t:end-t])
            push!(mx,  isempty(line) ? NaN : maximum(line))
            push!(p99, isempty(line) ? NaN : quantile(line, 0.99))
        end
        out[side] = (mx, p99)
    end
    return out
end

# ------------------------------------------------------------------------------
sf = joinpath(DIR, "$(RUN)_surface.nc")
isfile(sf) || error("no surface file at $sf")

ds = Dataset(sf)
lon, lat = axis_values(ds, "u")
latc = axis_values(ds, "v")[2]
nt = size(ds["v"])[end]

# Cell centres for plotting (u's x-axis is faces, so drop the last point).
λ = lon === nothing ? collect(1.0:size(slab(ds, "u", 1), 1)-1) : lon[1:end-1]
φ = lat === nothing ? collect(1.0:size(slab(ds, "v", 1), 2)-1) : lat

days = let s = get(ENV, "DAYS", "first,mid,last")
    s == "first,mid,last" ? unique([1, cld(nt, 2), nt]) :
        [parse(Int, x) for x in split(s, ",") if !isempty(x)]
end

bx, by, Δx, Δy = band_cells(lon, lat, EXCLUDE_KM)
@info @sprintf("grid spacing ~%.1f x %.1f km; %.0f km exclusion = %d cells in x, %d in y",
               Δx, Δy, EXCLUDE_KM, bx, by)

# ------------------------------------------------------------------- per-day maps
for t in days
    S, uc, vc, Nx, Ny = centred_speed(ds, t)
    finite = filter(isfinite, S)
    clip = quantile(finite, CLIP_PCT / 100)

    fig = Figure(size = (1500, 1050))
    Label(fig[0, 1:2], "$(RUN) — day $(t-1):  max speed $(round(maximum(finite), digits=2)) m/s",
          fontsize = 20, font = :bold)

    # full range
    ax1 = Axis(fig[1, 1], title = "surface speed (full range)", xlabel = "longitude", ylabel = "latitude")
    hm1 = heatmap!(ax1, λ, φ, S, colormap = :turbo, colorrange = (0, maximum(finite)))
    Colorbar(fig[1, 1][1, 2], hm1, label = "m/s")

    # clipped, so the field is visible rather than one cell
    ax2 = Axis(fig[1, 2], title = @sprintf("clipped at the %.0fth percentile (%.2f m/s)", CLIP_PCT, clip),
               xlabel = "longitude", ylabel = "latitude")
    hm2 = heatmap!(ax2, λ, φ, min.(S, clip), colormap = :turbo, colorrange = (0, clip))
    Colorbar(fig[1, 2][1, 2], hm2, label = "m/s")

    # the exclusion band, drawn on both
    for ax in (ax1, ax2)
        xi, xo = λ[min(bx + 1, length(λ))], λ[max(length(λ) - bx, 1)]
        yi, yo = φ[min(by + 1, length(φ))], φ[max(length(φ) - by, 1)]
        lines!(ax, [xi, xo, xo, xi, xi], [yi, yi, yo, yo, yi],
               color = :white, linestyle = :dash, linewidth = 2)
    end

    # distance-from-edge profiles
    prof = edge_profiles(S; trim = TRIM)
    tlab = TRIM > 0 ? "  (corners trimmed $(TRIM) cells)" : ""
    ax3 = Axis(fig[2, 1], title = "max speed vs distance from each boundary" * tlab,
               xlabel = "cells from boundary", ylabel = "max |u⃗| (m/s)")
    ax4 = Axis(fig[2, 2], title = "99th percentile vs distance (the field, not the outlier)" * tlab,
               xlabel = "cells from boundary", ylabel = "p99 |u⃗| (m/s)")
    for (side, c) in ((:west, :dodgerblue), (:east, :firebrick), (:south, :orange), (:north, :seagreen))
        mx, p99 = prof[side]
        lines!(ax3, 0:length(mx)-1, mx,  color = c, linewidth = 2, label = String(side))
        lines!(ax4, 0:length(p99)-1, p99, color = c, linewidth = 2, label = String(side))
    end
    vlines!(ax3, [by], color = :gray, linestyle = :dash)
    vlines!(ax4, [by], color = :gray, linestyle = :dash)
    axislegend(ax3, position = :rt); axislegend(ax4, position = :rt)

    f = joinpath(OUTDIR, "$(RUN)_day$(lpad(t-1, 2, '0'))$(TRIM > 0 ? "_trim$(TRIM)" : "").png")
    save(f, fig)
    @info "wrote $f"
end

# ------------------------------------------------- interior vs boundary time series
inner_max, outer_max, inner_p99 = Float64[], Float64[], Float64[]
for t in 1:nt
    S, _, _, Nx, Ny = centred_speed(ds, t)
    inner = S[bx+1:max(bx+1, Nx-bx), by+1:max(by+1, Ny-by)]
    mask = trues(size(S)); mask[bx+1:max(bx+1, Nx-bx), by+1:max(by+1, Ny-by)] .= false
    fi = filter(isfinite, inner); fo = filter(isfinite, S[mask])
    push!(inner_max, isempty(fi) ? NaN : maximum(fi))
    push!(inner_p99, isempty(fi) ? NaN : quantile(fi, 0.99))
    push!(outer_max, isempty(fo) ? NaN : maximum(fo))
end

fig = Figure(size = (1000, 460))
ax = Axis(fig[1, 1], title = "$(RUN): does the boundary contaminate the interior?",
          xlabel = "day", ylabel = "speed (m/s)")
lines!(ax, 0:nt-1, outer_max, color = :firebrick, linewidth = 2,
       label = "max, within $(Int(EXCLUDE_KM)) km of a boundary")
lines!(ax, 0:nt-1, inner_max, color = :dodgerblue, linewidth = 2, label = "max, interior")
lines!(ax, 0:nt-1, inner_p99, color = :dodgerblue, linewidth = 2, linestyle = :dash,
       label = "p99, interior")
axislegend(ax, position = :lt)
save(joinpath(OUTDIR, "$(RUN)_interior_vs_boundary.png"), fig)
close(ds)

println()
@printf("%4s  %22s  %16s  %16s\n", "day", "max in boundary band", "max interior", "p99 interior")
for t in 1:nt
    @printf("%4d  %22.3f  %16.3f  %16.3f\n", t - 1, outer_max[t], inner_max[t], inner_p99[t])
end

println("""

Reading this:
  * Boundary column climbing while the interior columns stay flat => the problem is confined
    to the band Ito et al. (2026) would have excluded anyway. Document the exclusion, move on.
  * Interior columns climbing too => it is leaking inward and must be fixed.
  * In the profile panels, a pileup is a spike at distance 0 decaying within a few cells. A
    real current crossing the boundary is flat with distance.
  * The p99 lines matter more than the max: one stuck cell moves the max and nothing else.
  * Run it twice, once with TRIM_CORNERS=30. A face whose edge value collapses toward the
    interior under the trim is clean -- the corner where it meets its neighbour owns the
    signal, and both faces were reporting it.
""")
