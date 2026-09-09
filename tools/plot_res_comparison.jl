using Pkg
Pkg.activate(".")
using Oceananigans
using NumericalEarth
using NCDatasets
using CairoMakie
using Statistics: mean
using Printf

println("Generating 1/3° horizontal comparison...")

λ₁, λ₂ = 90.0, 150.0
φ₁, φ₂ = -70.0, -40.0
SCALING = 3 # 1/3 degree
Nx = Int(SCALING * (λ₂ - λ₁)) # 180
Ny = Int(SCALING * (φ₂ - φ₁)) # 90
DZ_SURFACE = 2
DZ_BOTTOM = 200

# ── 1. Model ETOPO at 1/3° (passes = 5, extent = 5800 m) ─────────────────────
z_model = ReferenceToStretchedDiscretization(; extent=5800,
    constant_spacing=DZ_SURFACE,
    maximum_spacing=DZ_BOTTOM,
    constant_spacing_extent=500,
    stretching=PowerLawStretching(1.15))

grid_model = LatitudeLongitudeGrid(CPU();
    size=(Nx, Ny, length(z_model)),
    latitude=(φ₁, φ₂),
    longitude=(λ₁, λ₂),
    z=z_model,
    halo=(7, 7, 7))

bh_model = regrid_bathymetry(grid_model, height_above_water=1, minimum_depth=10, interpolation_passes=5)
mid_lon_idx = div(Nx, 2) # 90
mid_lon_val = grid_model.λᶜᵃᵃ[mid_lon_idx] # 119.83°E
lats_1_3 = [grid_model.φᵃᶜᵃ[j] for j in 1:Ny]
etopo_mid_1_3 = Array(interior(bh_model))[mid_lon_idx, :]

# ── 2. BSOSE Subsampled/Averaged to 1/3° Grid ──────────────────────────────────
ds_bsose = Dataset("data/grid105.nc")
Nx_b, Ny_b = 1080, 294
bsose_xc = reshape(ds_bsose["XC"][:], Nx_b, Ny_b)[:, 1] # 1/6° lon
bsose_yc = reshape(ds_bsose["YC"][:], Nx_b, Ny_b)[1, :] # 1/6° lat
bsose_depth = ds_bsose["Depth"][:, :]                   # positive downwards
close(ds_bsose)

# At each of the 90 model latitude cell centers (1/3°), find the corresponding BSOSE bathymetry
# at longitude = 119.83°E
bsose_mid_lon_idx = argmin(abs.(bsose_xc .- mid_lon_val))
bsose_at_1_3 = zeros(Ny)
for j in 1:Ny
    lat = lats_1_3[j]
    near_lats = findall(y -> abs(y - lat) <= (0.333333 / 2.0), bsose_yc)
    if !isempty(near_lats)
        bsose_at_1_3[j] = -mean(bsose_depth[bsose_mid_lon_idx, near_lats])
    else
        closest = argmin(abs.(bsose_yc .- lat))
        bsose_at_1_3[j] = -bsose_depth[bsose_mid_lon_idx, closest]
    end
end

# Also keep native BSOSE line for reference
bsose_lat_indices = findall(y -> φ₁ <= y <= φ₂, bsose_yc)
bsose_native_lats = bsose_yc[bsose_lat_indices]
bsose_native_z = -bsose_depth[bsose_mid_lon_idx, bsose_lat_indices]

# ── 3. Plot ───────────────────────────────────────────────────────────────────
output_fig = "plots/bathymetry_comparison_both_third_deg.png"
mkpath(dirname(output_fig))

fig = Figure(size=(1400, 950), fontsize=14)

Label(fig[1, 1:2],
      @sprintf("Bathymetry Comparison at Matched 1/3° Horizontal Resolution (Mid-Longitude = %.2f°E)\nModel ETOPO2022 vs BSOSE Iteration 105", mid_lon_val),
      fontsize=20, font=:bold)

ax = Axis(fig[2, 1],
    title="Both Data Sources Sampled on the Same 1/3° Grid Cells (Ny = 90 cells)",
    xlabel="Latitude (°N)",
    ylabel="Depth / Elevation z (m)",
    limits=(φ₁, φ₂, -6000, 100),
    xticks=φ₁:5:φ₂,
    yticks=-6000:1000:0)

band!(ax, lats_1_3, -6000, bsose_at_1_3, color=(:skyblue, 0.15), label="BSOSE Ocean Basin (1/3°)")

# Lines
lines!(ax, bsose_native_lats, bsose_native_z, color=(:blue, 0.35), linewidth=1.5, label="BSOSE Native 1/6° Reference")
lines!(ax, lats_1_3, bsose_at_1_3, color=:blue, linewidth=2.8, label="BSOSE at 1/3° Resolution")
scatter!(ax, lats_1_3, bsose_at_1_3, color=:blue, markersize=6)

lines!(ax, lats_1_3, etopo_mid_1_3, color=:crimson, linewidth=2.8, linestyle=:dash, label="Model ETOPO2022 at 1/3° Resolution")
scatter!(ax, lats_1_3, etopo_mid_1_3, color=:crimson, markersize=6, marker=:rect)

hlines!(ax, [0.0], color=:black, linewidth=1, linestyle=:dot)

# Vertical grid levels
for z_face in collect(grid_model.z.cᵃᵃᶠ)
    if z_face >= -6000
        hlines!(ax, [z_face], color=(:gray70, 0.20), linewidth=0.3)
    end
end

axislegend(ax, position=:rb)

# Difference Subplot
ax_diff = Axis(fig[3, 1],
    title="Exact Cell-by-Cell Difference at 1/3° Resolution: Δz = z_ETOPO(1/3°) - z_BSOSE(1/3°)",
    xlabel="Latitude (°N)",
    ylabel="Δz = z_ETOPO - z_BSOSE (m)",
    limits=(φ₁, φ₂, -800, 800),
    xticks=φ₁:5:φ₂,
    yticks=-800:400:800)

diff_1_3 = etopo_mid_1_3 .- bsose_at_1_3
barplot!(ax_diff, lats_1_3, diff_1_3, color=[d < 0 ? :darkred : :navy for d in diff_1_3], width=0.28)
hlines!(ax_diff, [0.0], color=:black, linewidth=1)

rowsize!(fig.layout, 2, Relative(0.65))
rowsize!(fig.layout, 3, Relative(0.35))

save(output_fig, fig, px_per_unit=2)
println("Plot saved to: ", output_fig)
