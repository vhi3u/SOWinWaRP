# ==============================================================================
# locate_ice_pile.jl -- where a run's sea ice piles up and where it (and the surface ocean) runs
# unphysically fast, and whether those are the same places.
#
#     julia --project=. tools/locate_ice_pile.jl 2Y_nested_bsose_conv_hj
#
# WHY. 2Y_nested_bsose_conv_hj's progress lines show ice at 2-3.9 m/s and the surface ocean at 2-3.6 m/s
# (Antarctic ice rarely drifts faster than ~0.5-1 m/s), the daily ice maxima tracking the ocean's, and the
# largest h·ℵ climbing from 3 to 14 m in 45 days. Before changing the ice advection (ClimaSeaIce PR #207),
# find WHERE: one recurring spot, the coast, or the open sides where the nest relaxes and the ice enters.
#
# READS <RUN>_sea_ice.nc (h, ℵ, u, v, daily snapshots) and <RUN>_surface.nc (top-cell u, v, daily snapshots,
# and bottom_height), one day at a time. Face velocities are averaged onto cell centres.
#
# PRINTS
#   a daily line (every STRIDE days): the fastest ice (where ℵ > 0.15), the fastest surface ocean, and the
#     largest h·ℵ, each with its place;
#   the cells most often fast or thick, with their depth and their distance from the coast and from the
#     open sides (the nest's relaxation band is ~4 cells);
#   how the fast and thick cells split between the open-side band, the coast and the interior.
# WRITES OUTDIR/<RUN>_ice_pile.png: maps of the days with fast ice, the days with a fast surface ocean, and
#   the largest h·ℵ over the run.
#
# LOGIN-NODE SAFE: NCDatasets and CairoMakie, one record at a time.
#
# ENV
#     DIR, OUTDIR      (., plots)
#     STRIDE           days between printed lines (default 5)
#     ICE_FAST         ice speed counted as fast, m/s (default 1.0)
#     OCEAN_FAST       surface ocean speed counted as fast, m/s (default 1.5)
#     THICK            h·ℵ counted as piled, m (default 5)
#     BAND             open-side band width, cells (default 4)
# ==============================================================================

# HDF5 file locking off, so reading a run that is still writing cannot block its writer
ENV["HDF5_USE_FILE_LOCKING"] = "FALSE"

using NCDatasets, CairoMakie, Printf, Dates

const RUN = length(ARGS) > 0 ? ARGS[1] : error("give the run: julia --project=. tools/locate_ice_pile.jl <RUN>")
const DIR        = get(ENV, "DIR", ".")
const OUTDIR     = get(ENV, "OUTDIR", "plots")
const STRIDE     = parse(Int, get(ENV, "STRIDE", "5"))
const ICE_FAST   = parse(Float64, get(ENV, "ICE_FAST", "1.0"))
const OCEAN_FAST = parse(Float64, get(ENV, "OCEAN_FAST", "1.5"))
const THICK      = parse(Float64, get(ENV, "THICK", "5"))
const BAND       = parse(Int, get(ENV, "BAND", "4"))
mkpath(OUTDIR)

const ICE_FILE  = joinpath(DIR, "$(RUN)_sea_ice.nc")
const SURF_FILE = joinpath(DIR, "$(RUN)_surface.nc")
for f in (ICE_FILE, SURF_FILE)
    isfile(f) || error("no $f (set DIR)")
end

f64(A) = Float64.(coalesce.(A, NaN))

"A 2-D record `n` of `name`, dropping a singleton z, and averaged onto (Nx, Ny) cell centres if it is on faces."
function read2d(ds, name, n, Nx, Ny)
    v = ds[name]
    A = f64(ndims(v) == 4 ? v[:, :, 1, n] : v[:, :, n])
    size(A, 1) == Nx + 1 && (A = 0.5 .* (A[1:end-1, :] .+ A[2:end, :]))
    size(A, 2) == Ny + 1 && (A = 0.5 .* (A[:, 1:end-1] .+ A[:, 2:end]))
    size(A) == (Nx, Ny) || error("$name is $(size(A)), not ($Nx, $Ny)")
    return A
end

surf = Dataset(SURF_FILE)
ice  = Dataset(ICE_FILE)
λ, φ = f64(surf["λ_caa"][:]), f64(surf["φ_aca"][:])
Nx, Ny = length(λ), length(φ)
seafloor = f64(surf["bottom_height"][:, :])
wet = seafloor .< 0
start = DateTime(string(surf.attrib["start_date"]))
t_surf, t_ice = f64(surf["time"][:]), f64(ice["time"][:])
recs = [(n, argmin(abs.(t_surf .- t))) for (n, t) in enumerate(t_ice) if minimum(abs.(t_surf .- t)) < 3600]
isempty(recs) && error("no sea-ice record has a surface record within an hour")

# distance (cells, Chebyshev) from land, and from the open sides (W, E, N)
coast_cells = fill(typemax(Int), Nx, Ny)
let queue = [(i, j) for i in 1:Nx, j in 1:Ny if !wet[i, j]]
    foreach(c -> coast_cells[c...] = 0, queue)
    h = 1
    while h <= length(queue)
        i, j = queue[h]; h += 1
        for di in -1:1, dj in -1:1
            a, b = i + di, j + dj
            (1 <= a <= Nx && 1 <= b <= Ny && coast_cells[a, b] > coast_cells[i, j] + 1) || continue
            coast_cells[a, b] = coast_cells[i, j] + 1
            push!(queue, (a, b))
        end
    end
end
side_cells = [min(i - 1, Nx - i, Ny - j) for i in 1:Nx, j in 1:Ny]
place(I) = @sprintf("%.2fE %.2fS", λ[I[1]], -φ[I[2]])
where_kind(I) = side_cells[I] < BAND ? "open-side band" : coast_cells[I] <= 2 ? "coast" : "interior"

# ── day by day ────────────────────────────────────────────────────────────────────────────────
ice_fast_days, ocean_fast_days = zeros(Int, Nx, Ny), zeros(Int, Nx, Ny)
thick_days, hv_max = zeros(Int, Nx, Ny), zeros(Nx, Ny)
daily = (; day = Float64[], ice = Float64[], ocean = Float64[], hv = Float64[], same = Bool[])

println("\n", "="^120)
println("WHERE THE ICE IS FASTEST AND THICKEST, $RUN (daily snapshots; ice speed where ℵ > 0.15)")
println("="^120)
@printf("  %-10s | %-34s | %-34s | %-30s\n", "date", "fastest ice (m/s)", "fastest surface ocean (m/s)", "largest h·ℵ (m)")
for (m, (n, s)) in enumerate(recs)
    h, ℵ = read2d(ice, "h", n, Nx, Ny), read2d(ice, "ℵ", n, Nx, Ny)
    uᵢ, vᵢ = read2d(ice, "u", n, Nx, Ny), read2d(ice, "v", n, Nx, Ny)
    u, v = read2d(surf, "u", s, Nx, Ny), read2d(surf, "v", s, Nx, Ny)
    hv = ifelse.(wet, h .* ℵ, 0.0)
    sᵢ = ifelse.(wet .& (ℵ .> 0.15), hypot.(uᵢ, vᵢ), 0.0)
    sₒ = ifelse.(wet, hypot.(u, v), 0.0)
    replace!(x -> isfinite(x) ? x : 0.0, hv); replace!(x -> isfinite(x) ? x : 0.0, sᵢ); replace!(x -> isfinite(x) ? x : 0.0, sₒ)

    ice_fast_days   .+= sᵢ .> ICE_FAST
    ocean_fast_days .+= sₒ .> OCEAN_FAST
    thick_days      .+= hv .> THICK
    hv_max .= max.(hv_max, hv)

    Iᵢ, Iₒ, Iₕ = argmax(sᵢ), argmax(sₒ), argmax(hv)
    day = t_ice[n] / 86400
    push!(daily.day, day); push!(daily.ice, sᵢ[Iᵢ]); push!(daily.ocean, sₒ[Iₒ]); push!(daily.hv, hv[Iₕ])
    push!(daily.same, maximum(abs.(Tuple(Iᵢ) .- Tuple(Iₒ))) <= 2)
    (m == 1 || m % STRIDE == 0) &&
        @printf("  %-10s | %5.2f %-15s %-12s | %5.2f %-15s %-12s | %5.1f %-15s %-8s\n",
                Dates.format(start + Millisecond(round(Int, 1000 * t_ice[n])), "yyyy-mm-dd"),
                sᵢ[Iᵢ], place(Iᵢ), where_kind(Iᵢ), sₒ[Iₒ], place(Iₒ), where_kind(Iₒ), hv[Iₕ], place(Iₕ), where_kind(Iₕ))
end
close(ice); close(surf)
@printf("\n  On %d of %d days the fastest ice is within 2 cells of the fastest surface ocean.\n", count(daily.same), length(daily.same))

# ── the recurring cells ───────────────────────────────────────────────────────────────────────
function top_cells(title, A, unit; n = 12)
    order = sortperm(vec(A); rev = true)[1:n]
    println("\n  $title")
    @printf("    %-16s %8s %8s %10s %10s  %s\n", "place", unit, "depth m", "from coast", "from side", "kind")
    for k in order
        I = CartesianIndices(A)[k]
        A[I] > 0 || break
        @printf("    %-16s %8.1f %8.0f %10d %10d  %s\n", place(I), A[I], -seafloor[I], coast_cells[I], side_cells[I], where_kind(I))
    end
end
top_cells("CELLS MOST OFTEN WITH FAST ICE (> $(ICE_FAST) m/s, days)", ice_fast_days, "days")
top_cells("CELLS MOST OFTEN WITH A FAST SURFACE OCEAN (> $(OCEAN_FAST) m/s, days)", ocean_fast_days, "days")
top_cells("CELLS WITH THE MOST ICE (largest h·ℵ over the run, m)", hv_max, "m")

println("\n  WHERE THE FAST AND THICK CELL-DAYS ARE (share of cell-days)")
@printf("    %-26s %16s %16s %16s\n", "", "open-side band", "coast (<= 2)", "interior")
kinds = [where_kind(I) for I in CartesianIndices(wet)]
for (name, A) in (("fast ice", ice_fast_days), ("fast surface ocean", ocean_fast_days), ("h·ℵ > $(THICK) m", thick_days))
    total = sum(A)
    shares = [total == 0 ? 0.0 : 100 * sum(A[kinds .== k]) / total for k in ("open-side band", "coast", "interior")]
    @printf("    %-26s %15.0f%% %15.0f%% %15.0f%%   (%d cell-days)\n", name, shares..., total)
end
@printf("    %-26s %15.0f%% %15.0f%% %15.0f%%\n", "(share of wet cells)",
        [100 * count(wet .& (kinds .== k)) / count(wet) for k in ("open-side band", "coast", "interior")]...)

# ── maps and daily ───────────────────────────────────────────────────────────────────────────
fig = Figure(size = (1500, 1250))
for (row, (A, title, cmap)) in enumerate(((ice_fast_days, "days with ice faster than $(ICE_FAST) m/s", :amp),
                                          (ocean_fast_days, "days with surface ocean faster than $(OCEAN_FAST) m/s", :amp),
                                          (hv_max, "largest h·ℵ over the run (m)", :ice)))
    ax = Axis(fig[row, 1]; title, aspect = DataAspect(), xlabel = "longitude", ylabel = "latitude")
    B = ifelse.(wet, Float64.(A), NaN)
    hi = max(maximum(x -> isfinite(x) ? x : 0.0, B), 1.0)
    heatmap!(ax, λ, φ, B; colormap = cmap, colorrange = (0, hi), nan_color = :gray70)
    Colorbar(fig[row, 2]; colormap = cmap, colorrange = (0, hi))
end
ax = Axis(fig[4, 1]; xlabel = "day", ylabel = "m/s, m", title = "daily maxima")
lines!(ax, daily.day, daily.ice; color = :steelblue, label = "fastest ice (m/s)")
lines!(ax, daily.day, daily.ocean; color = :firebrick, label = "fastest surface ocean (m/s)")
lines!(ax, daily.day, daily.hv ./ 5; color = :black, label = "largest h·ℵ / 5 (m)")
axislegend(ax; position = :lt)
out = joinpath(OUTDIR, "$(RUN)_ice_pile.png")
save(out, fig)
@info "wrote $out"
