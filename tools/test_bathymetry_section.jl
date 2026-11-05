# ==============================================================================================
# test_bathymetry_section.jl -- why the model's seafloor on a meridional section differs from
# B-SOSE's: the source dataset, or the smoothing in `regrid_bathymetry`?
#
#     julia --project=. tools/test_bathymetry_section.jl          # the mid-longitude, 119.917E
#     LON=119.917 LAT_RANGE=-68:-64 julia --project=. tools/test_bathymetry_section.jl
#
# Along one model column it draws:
#
#   ETOPO2022 native     the 1-arcminute source, as the min-max envelope of the 10 native columns
#                        inside the model cell (and their mean). NumericalEarth downloads the
#                        SURFACE version: over ice shelves it is ice top, not seabed.
#   passes = 1           regrid_bathymetry with one interpolation pass
#   passes = 5           ... with five, the setting examples/sowinwarp_nested.jl and src/model.jl use
#   B-SOSE Depth         B-SOSE i156's own seafloor (ETOPO1-derived, with its partial cells)
#
# Reading it: if the native envelope already lacks a feature B-SOSE has, the difference is the
# SOURCE. If the native data has it and only passes = 5 loses it, it is the SMOOTHING, and fewer
# passes would bring it back. Every regridding uses the nested run's other knobs
# (height_above_water = 1, minimum_depth = 10, major_basins = 1), and is cached by NumericalEarth.
#
# Writes plots/bathymetry_section_<lon>E.png and prints the shallowest point of each curve in the
# latitude window. CPU only; ~15 GB for the native ETOPO2022 grid during regridding.
# ==============================================================================================

using Oceananigans
using NumericalEarth
using NumericalEarth.DataWrangling: metadata_path
using NCDatasets, CairoMakie, Printf

include(joinpath(@__DIR__, "..", "src", "setup_bsose.jl"))
include(joinpath(@__DIR__, "bsose_masks.jl"))

# MUST MATCH examples/sowinwarp_nested.jl
const λ₁, λ₂ = 90.0, 150.0
const φ₁, φ₂ = -70.0, -45.0
const Nx, Ny = 360, 150

BDIR    = get(ENV, "BSOSE_DIR", default_bsose_directory())
OUTDIR  = get(ENV, "OUTDIR", "plots")
lat_lim = Tuple(parse.(Float64, split(get(ENV, "LAT_RANGE", "-68:-63"), ':')))

# Horizontal only: the seafloor does not depend on the levels
grid = LatitudeLongitudeGrid(CPU(); size = (Nx, Ny, 1), z = (-6000, 0),
                             longitude = (λ₁, λ₂), latitude = (φ₁, φ₂), halo = (7, 7, 7))

λc = λnodes(grid, Center())
φc = φnodes(grid, Center())
i  = haskey(ENV, "LON") ? argmin(abs.(λc .- parse(Float64, ENV["LON"]))) : div(Nx, 2)
λ  = λc[i]
Δλ = (λ₂ - λ₁) / Nx
@info @sprintf("section at model column i = %d, λ = %.3fE (cell %.3f-%.3fE)", i, λ, λ - Δλ/2, λ + Δλ/2)

# ── ETOPO2022 at native resolution ────────────────────────────────────────────────────────────
etopo_path = metadata_path(Metadatum(:bottom_height; dataset = ETOPO2022()))
isfile(etopo_path) || error("$etopo_path not found; regrid_bathymetry downloads it on first use")

native_lat, native_lon, native_z = NCDataset(etopo_path) do ds
    haskey(ds, "lon") && haskey(ds, "lat") || error("$etopo_path has no lon/lat variables: $(keys(ds))")
    lon = Float64.(ds["lon"][:])
    lat = Float64.(ds["lat"][:])
    is  = findall(l -> λ - Δλ/2 <= l <= λ + Δλ/2, lon)
    js  = findall(p -> φ₁ <= p <= φ₂, lat)
    lat[js], lon[is], Float64.(coalesce.(ds["z"][first(is):last(is), first(js):last(js)], NaN))
end
@info "ETOPO2022 native: $(length(native_lon)) columns x $(length(native_lat)) rows inside the model cell"

native_min  = vec(minimum(native_z; dims = 1))
native_max  = vec(maximum(native_z; dims = 1))
native_mean = vec(sum(native_z; dims = 1)) ./ size(native_z, 1)

# ── regridded, as the nested run does it ──────────────────────────────────────────────────────
regridded(passes) = Array(interior(regrid_bathymetry(grid; dataset = ETOPO2022(),
                                                     height_above_water = 1, minimum_depth = 10,
                                                     interpolation_passes = passes, major_basins = 1)))[i, :, 1]
z1 = regridded(1)
z5 = regridded(5)

# ── B-SOSE's own ──────────────────────────────────────────────────────────────────────────────
dataset = BSOSEMonthly(dir = BDIR, iteration = 156)
zb = .-vec(bsose_depth(dataset, [λ], φc))

# Land as NaN, so lines stop at the coast instead of drawing the ice sheet
wet(z) = [h < 0 ? h : NaN for h in z]

# ── shallowest point of each in the window ────────────────────────────────────────────────────
function shallowest(lats, z)
    inside = findall(p -> lat_lim[1] <= p <= lat_lim[2], lats)
    k = [k for k in inside if z[k] < 0]
    isempty(k) && return (NaN, NaN)
    # the coast is the shallowest water of all, so look seaward of the first deep (> 500 m) cell
    deep = findfirst(kk -> z[kk] < -500, k)
    k = isnothing(deep) ? k : k[deep:end]
    m = k[argmax(z[k])]
    return (lats[m], z[m])
end

println()
println(@sprintf("Shallowest point seaward of the shelf, %.1f to %.1f N, at %.3fE:", lat_lim..., λ))
for (name, lats, z) in (("ETOPO2022 native max", native_lat, native_max),
                        ("ETOPO2022 native mean", native_lat, native_mean),
                        ("regridded, passes = 1", φc, z1),
                        ("regridded, passes = 5", φc, z5),
                        ("B-SOSE Depth", φc, zb))
    p, h = shallowest(lats, z)
    println(@sprintf("  %-24s %8.1f m at %7.3f", name, h, p))
end
println()

# ── figure ────────────────────────────────────────────────────────────────────────────────────
mkpath(OUTDIR)
fig = Figure(size = (1100, 900))

for (row, (title, limits)) in enumerate(((@sprintf("Seafloor along %.3f°E, %.0f to %.0f°N", λ, lat_lim...), lat_lim),
                                         ("Whole section", (φ₁, φ₂))))
    ax = Axis(fig[row, 1]; title, xlabel = "Latitude (°N)", ylabel = "Height (m)")
    band!(ax, native_lat, native_min, native_max; color = (:grey50, 0.35),
          label = "ETOPO2022 native, min–max across the cell")
    lines!(ax, native_lat, native_mean; color = :grey30, linewidth = 1, label = "ETOPO2022 native, mean")
    stairs!(ax, φc, wet(z1); step = :center, color = :royalblue, linewidth = 2, label = "regridded, passes = 1")
    stairs!(ax, φc, wet(z5); step = :center, color = :firebrick, linewidth = 2, label = "regridded, passes = 5 (the run)")
    stairs!(ax, φc, wet(zb); step = :center, color = :black, linewidth = 2, linestyle = :dash, label = "B-SOSE Depth")
    xlims!(ax, limits...)
    row == 1 && ylims!(ax, -4000, 100)
    row == 1 && axislegend(ax; position = :rb, framevisible = false)
end

Label(fig[3, 1], "ETOPO2022 is the surface version: over ice shelves it is the ice top. " *
                 "The regridded and B-SOSE curves are cell values; NaN (land) is not drawn.";
      fontsize = 12, tellwidth = false)

out = joinpath(OUTDIR, @sprintf("bathymetry_section_%.3fE.png", λ))
save(out, fig)
@info "wrote $out"
