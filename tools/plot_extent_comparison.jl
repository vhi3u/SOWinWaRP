using Pkg
Pkg.activate(".")
using Oceananigans
using NumericalEarth
using NCDatasets
using CairoMakie
using Printf

println("Setting up grids for extent comparison (BSOSE vs extent=5000 vs extent=5800)...")

λ₁, λ₂ = 90.0, 150.0
φ₁, φ₂ = -70.0, -40.0
SCALING = 3
DZ_SURFACE = 2 # m
DZ_BOTTOM = 200 # m

Nx = Int(SCALING * (λ₂ - λ₁)) # 180
Ny = Int(SCALING * (φ₂ - φ₁)) # 90

# ── 1. Model Grid with extent = 5000 m ───────────────────────────────────────
println("Generating extent = 5000 m grid...")
z_5000 = ReferenceToStretchedDiscretization(; extent=5000,
    constant_spacing=DZ_SURFACE,
    maximum_spacing=DZ_BOTTOM,
    constant_spacing_extent=500,
    stretching=PowerLawStretching(1.15))

grid_5000 = LatitudeLongitudeGrid(CPU();
    size=(Nx, Ny, length(z_5000)),
    latitude=(φ₁, φ₂),
    longitude=(λ₁, λ₂),
    z=z_5000,
    halo=(7, 7, 7))

bh_5000 = regrid_bathymetry(grid_5000, height_above_water=1, minimum_depth=10, interpolation_passes=5)
mid_lon_idx = div(Nx, 2)
mid_lon_val = grid_5000.λᶜᵃᵃ[mid_lon_idx]
model_lats = [grid_5000.φᵃᶜᵃ[j] for j in 1:Ny]
z_profile_5000 = Array(interior(bh_5000))[mid_lon_idx, :]

# ── 2. Model Grid with extent = 5800 m ───────────────────────────────────────
println("Generating extent = 5800 m grid...")
z_5800 = ReferenceToStretchedDiscretization(; extent=5800,
    constant_spacing=DZ_SURFACE,
    maximum_spacing=DZ_BOTTOM,
    constant_spacing_extent=500,
    stretching=PowerLawStretching(1.15))

grid_5800 = LatitudeLongitudeGrid(CPU();
    size=(Nx, Ny, length(z_5800)),
    latitude=(φ₁, φ₂),
    longitude=(λ₁, λ₂),
    z=z_5800,
    halo=(7, 7, 7))

bh_5800 = regrid_bathymetry(grid_5800, height_above_water=1, minimum_depth=10, interpolation_passes=5)
z_profile_5800 = Array(interior(bh_5800))[mid_lon_idx, :]

# ── 3. BSOSE Iteration 105 ───────────────────────────────────────────────────
println("Loading BSOSE iteration 105...")
ds_bsose = Dataset("data/grid105.nc")
Nx_b, Ny_b = 1080, 294
bsose_xc = reshape(ds_bsose["XC"][:], Nx_b, Ny_b)[:, 1]
bsose_yc = reshape(ds_bsose["YC"][:], Nx_b, Ny_b)[1, :]
bsose_depth = ds_bsose["Depth"][:, :]
close(ds_bsose)

bsose_mid_lon_idx = argmin(abs.(bsose_xc .- mid_lon_val))
bsose_lat_indices = findall(y -> φ₁ <= y <= φ₂, bsose_yc)
bsose_lats = bsose_yc[bsose_lat_indices]
bsose_mid_bathymetry = -bsose_depth[bsose_mid_lon_idx, bsose_lat_indices]

# ── 4. Plot Comparison ────────────────────────────────────────────────────────
println("Plotting 3-way comparison...")
output_fig = "plots/bathymetry_extent_comparison.png"
mkpath(dirname(output_fig))

fig = Figure(size=(1400, 950), fontsize=14)

Label(fig[1, 1:2],
      @sprintf("Bathymetry Comparison at Mid-Longitude (%.2f°E)\nBSOSE vs Model Grid (extent = 5000 m) vs Model Grid (extent = 5800 m)", mid_lon_val),
      fontsize=20, font=:bold)

ax = Axis(fig[2, 1],
    title="Mid-Longitude Transect Bathymetry Profile",
    xlabel="Latitude (°N)",
    ylabel="Depth / Elevation z (m)",
    limits=(φ₁, φ₂, -6000, 100),
    xticks=φ₁:5:φ₂,
    yticks=-6000:1000:0)

# Fill ocean basin for BSOSE
band!(ax, bsose_lats, -6000, bsose_mid_bathymetry, color=(:skyblue, 0.12), label="BSOSE Ocean Basin")

# Lines
lines!(ax, bsose_lats, bsose_mid_bathymetry, color=:blue, linewidth=3.0, label="BSOSE Iteration 105 (1/6°)")
scatter!(ax, bsose_lats, bsose_mid_bathymetry, color=:blue, markersize=5)

lines!(ax, model_lats, z_profile_5000, color=:darkorange, linewidth=2.5, linestyle=:dash,
       label=@sprintf("Model ETOPO (extent = 5000 m, %d levels)", length(z_5000)))
scatter!(ax, model_lats, z_profile_5000, color=:darkorange, markersize=5, marker=:rect)

lines!(ax, model_lats, z_profile_5800, color=:crimson, linewidth=2.0, linestyle=:dot,
       label=@sprintf("Model ETOPO (extent = 5800 m, %d levels)", length(z_5800)))
scatter!(ax, model_lats, z_profile_5800, color=:crimson, markersize=5, marker=:circle)

# Draw grid bottom boundary limits for both extents
hlines!(ax, [-5000.0], color=:darkorange, linewidth=1.2, linestyle=:dash, label="Grid Bottom limit (extent = 5000 m)")
hlines!(ax, [minimum(collect(grid_5800.z.cᵃᵃᶠ))], color=:crimson, linewidth=1.2, linestyle=:dot, label="Grid Bottom limit (extent = 5800 m)")
hlines!(ax, [0.0], color=:black, linewidth=1)

axislegend(ax, position=:rb)

# Difference Subplot
ax_diff = Axis(fig[3, 1],
    title="Bathymetry Difference from BSOSE (z_model - z_bsose)",
    xlabel="Latitude (°N)",
    ylabel="Δz (m)",
    limits=(φ₁, φ₂, -800, 800),
    xticks=φ₁:5:φ₂,
    yticks=-800:400:800)

bsose_interp = [bsose_mid_bathymetry[argmin(abs.(bsose_lats .- lat))] for lat in model_lats]

lines!(ax_diff, model_lats, z_profile_5000 .- bsose_interp, color=:darkorange, linewidth=2.5, linestyle=:dash, label="Δz (extent = 5000 m - BSOSE)")
lines!(ax_diff, model_lats, z_profile_5800 .- bsose_interp, color=:crimson, linewidth=2.0, linestyle=:dot, label="Δz (extent = 5800 m - BSOSE)")
hlines!(ax_diff, [0.0], color=:black, linewidth=1)

axislegend(ax_diff, position=:lb)

rowsize!(fig.layout, 2, Relative(0.65))
rowsize!(fig.layout, 3, Relative(0.35))

save(output_fig, fig, px_per_unit=2)
println("Plot saved successfully to: ", output_fig)
