# ==============================================================================
# inspect_bathymetry.jl -- what does the sea floor actually look like, especially at the edges?
#
#     julia --project=. tools/inspect_bathymetry.jl                 # domain map + boundary report
#     julia --project=. tools/inspect_bathymetry.jl 1 21            # ...plus a table around (i,j)
#
# Environment knobs:
#     REGION   "i1:i2,j1:j2" index box to zoom, or "lon1:lon2,lat1:lat2" if the numbers look
#              like degrees. Example: REGION=1:30,5:40  or  REGION=90:95,-68:-64
#     RADIUS   half-width of the neighbourhood table printed around (i,j)   (default 4)
#     OUTDIR   where figures go                                 (tools/bathymetry)
#
# WHY THIS EXISTS. The model's fastest cell settled on (i=1, j=21, k=39): the BOTTOM cell of a
# 485 m column sitting on the west open boundary, in a domain averaging 3130 m. A boundary
# column far shallower than its neighbours turns a perfectly ordinary transport into an
# enormous velocity, because the barotropic corrector sets depth_mean(u) = U whatever the
# column depth. So the question is whether that column is a real shelf feature or a one-cell
# artifact of `regrid_bathymetry`, and the only way to tell is to look.
#
# WHAT IT CHECKS, beyond drawing maps:
#   * MITgcm's `OBCSfixtopo` condition -- a boundary column deeper than its inward neighbour.
#     MITgcm raises those unconditionally; we had a port of it in glorys_obc.jl and lost it
#     when that file was deleted.
#   * Isolated columns -- a boundary column whose depth is far from the median of its
#     along-face neighbours. These are regridding artifacts, not bathymetry.
#   * The thinnest wet columns on each face, which is where U/H blows up.
#
# IMPORTANT: this builds the 360x150x135 GPU grid explicitly, NOT the 1-degree CPU grid that
# `src/model.jl` would construct when `arch isa CPU`. Matching the production grid is what
# makes it hit the existing bathymetry cache and describe the field the model actually used.
# ==============================================================================

using Oceananigans, Oceananigans.Units, Oceananigans.Grids
using NumericalEarth
using CairoMakie, Printf, Statistics

const OUTDIR = get(ENV, "OUTDIR", joinpath(@__DIR__, "bathymetry"))
const RADIUS = parse(Int, get(ENV, "RADIUS", "4"))
mkpath(OUTDIR)

# ---- the production grid, hard-coded to match src/model.jl's GPU branch ----
const λ₁, λ₂ = 90.0, 150.0
const φ₁, φ₂ = -70.0, -45.0
const SCALING, DZ_SURFACE, DZ_BOTTOM = 6, 5, 200

z = ReferenceToStretchedDiscretization(; extent = 5800,
                                       constant_spacing = DZ_SURFACE,
                                       maximum_spacing = DZ_BOTTOM,
                                       constant_spacing_extent = 500,
                                       stretching = PowerLawStretching(1.12))
Nx, Ny, Nz = Int(SCALING * (λ₂ - λ₁)), Int(SCALING * (φ₂ - φ₁)), length(z)

@info "building the production grid ($Nx x $Ny x $Nz) so the bathymetry cache is reused..."
grid = LatitudeLongitudeGrid(CPU(); size = (Nx, Ny, Nz),
                             longitude = (λ₁, λ₂), latitude = (φ₁, φ₂), z, halo = (7, 7, 7))

bottom = regrid_bathymetry(grid; height_above_water = 1, minimum_depth = 10,
                           interpolation_passes = 5)
B = Array(interior(bottom))[:, :, 1]          # (Nx, Ny), negative below sea level, 0 on land

λc = collect(λnodes(grid, Center()))
φc = collect(φnodes(grid, Center()))

depth(i, j) = -B[i, j]                        # positive metres of water, 0 = land
wet(i, j)   = depth(i, j) > 0

# ------------------------------------------------------------------ summary
nwet = count(wet(i, j) for i in 1:Nx, j in 1:Ny)
wetd = [depth(i, j) for i in 1:Nx, j in 1:Ny if wet(i, j)]
println()
@printf("domain %d x %d, %d wet of %d cells (%.0f%%)\n", Nx, Ny, nwet, Nx*Ny, 100nwet/(Nx*Ny))
@printf("depth: mean %.0f m, median %.0f m, min %.0f m, max %.0f m\n\n",
        mean(wetd), median(wetd), minimum(wetd), maximum(wetd))

# ------------------------------------------------- per-face geometry and pathologies
"""Indices of the column `d` cells in from `side`, along the face."""
face_index(side, n, d) = side === :west  ? (1 + d, n)  :
                         side === :east  ? (Nx - d, n) :
                         side === :south ? (n, 1 + d)  : (n, Ny - d)

face_len(side) = side in (:west, :east) ? Ny : Nx

for side in (:west, :east, :south, :north)
    N = face_len(side)
    d0 = [depth(face_index(side, n, 0)...) for n in 1:N]   # the boundary column
    d1 = [depth(face_index(side, n, 1)...) for n in 1:N]   # one cell inward
    w  = findall(>(0), d0)

    println("="^92)
    @printf("%s face: %d/%d columns wet (%.0f%%)\n", uppercase(String(side)), length(w), N, 100length(w)/N)
    if isempty(w)
        println("   entirely land -- carries no transport, its boundary condition is inert.\n")
        continue
    end
    @printf("   depth: mean %.0f m, median %.0f m, shallowest %.0f m, deepest %.0f m\n",
            mean(d0[w]), median(d0[w]), minimum(d0[w]), maximum(d0[w]))

    # --- MITgcm OBCSfixtopo: boundary deeper than the cell inside it ---
    viol = [n for n in w if d1[n] > 0 && d0[n] > d1[n]]
    if isempty(viol)
        println("   OBCSfixtopo: no boundary column is deeper than its inward neighbour. OK.")
    else
        Δ = [d0[n] - d1[n] for n in viol]
        @printf("   OBCSfixtopo: %d columns are DEEPER than the cell inside them (mean %.0f m, worst %.0f m at n=%d)\n",
                length(viol), mean(Δ), maximum(Δ), viol[argmax(Δ)])
        println("                MITgcm raises these unconditionally; we lost that port with glorys_obc.jl.")
    end

    # --- isolated columns: far from the median of their along-face neighbours ---
    iso = NamedTuple[]
    for n in w
        lo, hi = max(1, n-3), min(N, n+3)
        nb = [d0[m] for m in lo:hi if m != n && d0[m] > 0]
        length(nb) < 3 && continue
        med = median(nb)
        r = d0[n] / med
        (r < 0.5 || r > 2.0) && push!(iso, (n = n, d = d0[n], med = med, ratio = r))
    end
    if isempty(iso)
        println("   isolation   : no column differs from its neighbours by more than 2x. OK.")
    else
        @printf("   isolation   : %d column(s) differ from their along-face neighbours by >2x:\n", length(iso))
        for x in first(sort(iso, by = y -> abs(log(y.ratio)), rev = true), 5)
            i, j = face_index(side, x.n, 0)
            @printf("                 n=%3d (i=%3d, j=%3d) %7.0f m vs neighbours ~%7.0f m  (%.2fx)\n",
                    x.n, i, j, x.d, x.med, x.ratio)
        end
    end

    # --- the thinnest wet columns: where U/H explodes ---
    order = sort(w, by = n -> d0[n])
    print("   thinnest    : ")
    for n in first(order, 5)
        i, j = face_index(side, n, 0)
        @printf("(%d,%d) %.0fm  ", i, j, d0[n])
    end
    println("\n")
end

# ------------------------------------------------------ neighbourhood table
if length(ARGS) >= 2
    i0, j0 = parse(Int, ARGS[1]), parse(Int, ARGS[2])
    println("="^92)
    @printf("DEPTH (m) around (i=%d, j=%d)  —  lon %.3f, lat %.3f   [0 = land]\n\n",
            i0, j0, λc[clamp(i0,1,Nx)], φc[clamp(j0,1,Ny)])
    is = max(1, i0-RADIUS):min(Nx, i0+RADIUS)
    js = max(1, j0-RADIUS):min(Ny, j0+RADIUS)
    print("        j="); for j in js; @printf("%8d", j); end; println()
    for i in reverse(collect(is))
        @printf("  i=%3d  ", i)
        for j in js
            d = depth(i, j)
            # Mark the requested cell with a TRAILING star: a leading one reads as though it
            # belonged to the column before it.
            print(i == i0 && j == j0 ? @sprintf("%7.0f*", d) :
                  d == 0             ? "    land" : @sprintf("%8.0f", d))
        end
        println()
    end
    @printf("\n(* = the requested cell: %.0f m)\n\n", depth(clamp(i0,1,Nx), clamp(j0,1,Ny)))
end

# ------------------------------------------------------------------ figures
fig = Figure(size = (1500, 950))
Label(fig[0, 1:2], "Model bathymetry — $(Nx)x$(Ny), 1/$(SCALING)°", fontsize = 20, font = :bold)

ax = Axis(fig[1, 1:2], title = "depth (m), land masked", xlabel = "longitude", ylabel = "latitude")
D = [wet(i, j) ? depth(i, j) : NaN for i in 1:Nx, j in 1:Ny]
hm = heatmap!(ax, λc, φc, D, colormap = Reverse(:deep), nan_color = :gray70)
Colorbar(fig[1, 3], hm, label = "m")
lines!(ax, [λc[1], λc[end], λc[end], λc[1], λc[1]],
           [φc[1], φc[1], φc[end], φc[end], φc[1]], color = :red, linestyle = :dash, linewidth = 2)

for (k, side) in enumerate((:west, :east, :south, :north))
    N = face_len(side)
    axp = Axis(fig[2, 1][1, k], title = "$(side)", xlabel = "along-face index",
               ylabel = k == 1 ? "depth (m)" : "", yreversed = true)
    for (d, c) in ((0, :firebrick), (1, :darkorange), (2, :goldenrod), (4, :gray60))
        prof = [depth(face_index(side, n, d)...) for n in 1:N]
        prof[prof .== 0] .= NaN
        lines!(axp, 1:N, prof, color = c, linewidth = d == 0 ? 2 : 1,
               label = k == 1 ? "d=$d" : nothing)
    end
    k == 1 && axislegend(axp, position = :rb, labelsize = 10)
end
Label(fig[2, 1][1, 1:4, Top()], "depth along each face, and 1/2/4 cells inward",
      fontsize = 14, padding = (0, 0, 24, 0))

f = joinpath(OUTDIR, "bathymetry_domain.png")
save(f, fig); @info "wrote $f"

# ---- optional zoom ----
if haskey(ENV, "REGION")
    spec = ENV["REGION"]
    a, b = split(spec, ",")
    p1, p2 = parse.(Float64, split(a, ":")), parse.(Float64, split(b, ":"))
    degrees = p1[1] > 60 || p2[1] < 0          # lon/lat rather than indices
    is, js = if degrees
        (findfirst(>=(p1[1]), λc):findlast(<=(p1[2]), λc),
         findfirst(>=(p2[1]), φc):findlast(<=(p2[2]), φc))
    else
        (Int(p1[1]):Int(p1[2]), Int(p2[1]):Int(p2[2]))
    end

    f2 = Figure(size = (900, 700))
    ax2 = Axis(f2[1, 1], title = "bathymetry, REGION=$spec", xlabel = "longitude", ylabel = "latitude")
    Dz = [wet(i, j) ? depth(i, j) : NaN for i in is, j in js]
    hm2 = heatmap!(ax2, λc[is], φc[js], Dz, colormap = Reverse(:deep), nan_color = :gray70)
    Colorbar(f2[1, 2], hm2, label = "m")
    g = joinpath(OUTDIR, "bathymetry_region.png")
    save(g, f2); @info "wrote $g"
end

println("""
Reading this:
  * A boundary column much shallower than its neighbours is the dangerous case. The barotropic
    corrector sets depth_mean(u) = U regardless of depth, so the same transport per unit width
    becomes U/H -- an order of magnitude larger in a thin column.
  * `OBCSfixtopo` violations are the ones MITgcm fixes by construction. Those are worth
    correcting whatever else is going on.
  * An isolated column flagged at >2x its neighbours is almost always a regridding artifact,
    not bathymetry. Raising it to the local median is defensible; smoothing the whole field is
    not, because it moves the shelf break.
""")
