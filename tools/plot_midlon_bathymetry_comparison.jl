using Pkg
Pkg.activate(".")
using Oceananigans
using NumericalEarth
using NCDatasets
using CairoMakie
using Printf

println("Setting up CPU model grid and bathymetry from model.jl...")

# ── 1. Setup Model Grid (CPU configuration from model.jl) ────────────────────
λ₁, λ₂ = 90.0, 150.0
φ₁, φ₂ = -70.0, -40.0

z = ReferenceToStretchedDiscretization(; extent=5000,
    constant_spacing=10,
    maximum_spacing=1000,
    constant_spacing_extent=500,
    stretching=PowerLawStretching(1.15))

Nx = Int(λ₂ - λ₁)
Ny = Int(φ₂ - φ₁)
Nz = length(z)

grid = LatitudeLongitudeGrid(CPU();
    size=(Nx, Ny, Nz),
    latitude=(φ₁, φ₂),
    longitude=(λ₁, λ₂),
    z,
    halo=(7, 7, 7))

# Mid-longitude index in model grid
mid_lon_idx = div(grid.Nx, 2) # div(60, 2) = 30
mid_lon_val = grid.λᶜᵃᵃ[mid_lon_idx]
println(@sprintf("Model grid mid_lon_idx = %d (lon = %.2f°E)", mid_lon_idx, mid_lon_val))

# Regrid bathymetry exactly as in model.jl
bottom_height = regrid_bathymetry(grid,
    height_above_water=1,
    minimum_depth=10,
    interpolation_passes=25)

model_etopo_z = Array(interior(bottom_height)) # 60 x 30
model_mid_bathymetry = model_etopo_z[mid_lon_idx, :] # 30 values along latitude
model_lats = [grid.φᵃᶜᵃ[j] for j in 1:Ny]

# Discrete cell depth interfaces for model
model_z_faces = grid.z.cᵃᵃᶠ

# ── 2. Load BSOSE Iteration 105 Grid Data ─────────────────────────────────────
println("Loading BSOSE iteration 105 grid...")
ds_bsose = Dataset("data/grid105.nc")
Nx_b, Ny_b = 1080, 294
bsose_xc = reshape(ds_bsose["XC"][:], Nx_b, Ny_b)[:, 1] # 1D longitudes
bsose_yc = reshape(ds_bsose["YC"][:], Nx_b, Ny_b)[1, :] # 1D latitudes
bsose_depth = ds_bsose["Depth"][:, :]                   # 1080 x 294, positive downwards

# Find BSOSE longitude index closest to mid_lon_val
bsose_mid_lon_idx = argmin(abs.(bsose_xc .- mid_lon_val))
bsose_actual_lon = bsose_xc[bsose_mid_lon_idx]
println(@sprintf("BSOSE closest lon_idx = %d (lon = %.2f°E)", bsose_mid_lon_idx, bsose_actual_lon))

# Extract BSOSE bathymetry along this longitude within domain latitudes [-70, -40]
bsose_lat_indices = findall(y -> φ₁ <= y <= φ₂, bsose_yc)
bsose_lats = bsose_yc[bsose_lat_indices]
# Depth in MITgcm is positive depth (m), convert to negative elevation z (m)
bsose_mid_bathymetry = -bsose_depth[bsose_mid_lon_idx, bsose_lat_indices]

close(ds_bsose)

# ── 3. Plot Comparison ────────────────────────────────────────────────────────
println("Generating comparison plot...")
output_fig = "plots/bathymetry_midlon_comparison.png"
mkpath(dirname(output_fig))

fig = Figure(size=(1200, 800), fontsize=14)

Label(fig[1, 1:2], @sprintf("Bathymetry Comparison at Mid-Longitude (%.1f°E)\nBSOSE Iteration 105 vs Model ETOPO2022 (CPU Grid)", mid_lon_val),
      fontsize=20, font=:bold)

ax = Axis(fig[2, 1],
    title="Mid-Longitude Transect Bathymetry Profile",
    xlabel="Latitude (°N)",
    ylabel="Depth / Elevation z (m)",
    limits=(φ₁, φ₂, -5500, 100),
    xticks=φ₁:5:φ₂,
    yticks=-5000:1000:0)

# Fill ocean areas
band!(ax, bsose_lats, -5500, bsose_mid_bathymetry, color=(:skyblue, 0.15), label="BSOSE Ocean Basin")

# Plot profile lines
lines!(ax, bsose_lats, bsose_mid_bathymetry, color=:blue, linewidth=2.5, label="BSOSE Iteration 105")
scatter!(ax, bsose_lats, bsose_mid_bathymetry, color=:blue, markersize=6)

lines!(ax, model_lats, model_mid_bathymetry, color=:crimson, linewidth=2.5, linestyle=:dash, label="Model ETOPO2022 (Regridded 25 passes)")
scatter!(ax, model_lats, model_mid_bathymetry, color=:crimson, markersize=8, marker=:rect)

# Draw sea surface reference line
hlines!(ax, [0.0], color=:black, linewidth=1, linestyle=:dot)

# Draw horizontal lines for the model's discrete vertical levels (CPU z grid)
for z_face in model_z_faces
    if z_face >= -5500
        hlines!(ax, [z_face], color=(:gray70, 0.35), linewidth=0.5)
    end
end

axislegend(ax, position=:rb)

# Difference Subplot
ax_diff = Axis(fig[3, 1],
    title="Bathymetry Discrepancy (Model ETOPO - BSOSE) along transect",
    xlabel="Latitude (°N)",
    ylabel="Δz = z_model - z_bsose (m)",
    limits=(φ₁, φ₂, -800, 800),
    xticks=φ₁:5:φ₂,
    yticks=-800:400:800)

# Interpolate BSOSE onto model latitudes to calculate difference
bsose_interp = [bsose_mid_bathymetry[argmin(abs.(bsose_lats .- lat))] for lat in model_lats]
diff_z = model_mid_bathymetry .- bsose_interp

barplot!(ax_diff, model_lats, diff_z, color=[d < 0 ? :darkred : :navy for d in diff_z], width=0.8)
hlines!(ax_diff, [0.0], color=:black, linewidth=1)

rowsize!(fig.layout, 2, Relative(0.65))
rowsize!(fig.layout, 3, Relative(0.35))

save(output_fig, fig, px_per_unit=2)
println("Plot saved successfully to: ", output_fig)
