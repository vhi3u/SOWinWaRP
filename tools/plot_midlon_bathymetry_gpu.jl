using Pkg
Pkg.activate(".")
using Oceananigans
using NumericalEarth
using NCDatasets
using CairoMakie
using Printf

println("Setting up GPU grid parameters (extent=5800m) from model.jl...")

# ── 1. Setup Model Grid with GPU parameters (extent = 5800) ──────────────────
λ₁, λ₂ = 90.0, 150.0
φ₁, φ₂ = -70.0, -40.0
SCALING = 3
DZ_SURFACE = 2 # m
DZ_BOTTOM = 200 # m

# GPU z discretization with extent = 5800 m
z_gpu = ReferenceToStretchedDiscretization(; extent=5800,
    constant_spacing=DZ_SURFACE,
    maximum_spacing=DZ_BOTTOM,
    constant_spacing_extent=500,
    stretching=PowerLawStretching(1.15))

Nx = Int(SCALING * (λ₂ - λ₁)) # 3 * 60 = 180 (1/3 degree)
Ny = Int(SCALING * (φ₂ - φ₁)) # 3 * 30 = 90  (1/3 degree)
Nz = length(z_gpu)

println(@sprintf("GPU grid resolution: %d x %d x %d (extent = 5800 m, dx = dy = %.2f°)", 
                 Nx, Ny, Nz, 1.0/SCALING))

# Construct grid
grid = LatitudeLongitudeGrid(CPU();
    size=(Nx, Ny, Nz),
    latitude=(φ₁, φ₂),
    longitude=(λ₁, λ₂),
    z=z_gpu,
    halo=(7, 7, 7))

mid_lon_idx = div(grid.Nx, 2) # 90
mid_lon_val = grid.λᶜᵃᵃ[mid_lon_idx]
println(@sprintf("GPU grid mid_lon_idx = %d (lon = %.2f°E)", mid_lon_idx, mid_lon_val))

# Regrid bathymetry with passes = 5
println("Regridding ETOPO2022 bathymetry (extent=5800m, passes=5)...")
bottom_height = regrid_bathymetry(grid,
    height_above_water=1,
    minimum_depth=10,
    interpolation_passes=5)

model_etopo_z = Array(interior(bottom_height)) # 180 x 90
model_mid_bathymetry = model_etopo_z[mid_lon_idx, :] # 90 values along latitude
model_lats = [grid.φᵃᶜᵃ[j] for j in 1:Ny]

model_z_faces = collect(grid.z.cᵃᵃᶠ)
deepest_z_face = minimum(model_z_faces)
println(@sprintf("Deepest vertical level face: %.2f m", deepest_z_face))

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

# Extract BSOSE bathymetry along this longitude within domain latitudes [-70, -40]
bsose_lat_indices = findall(y -> φ₁ <= y <= φ₂, bsose_yc)
bsose_lats = bsose_yc[bsose_lat_indices]
bsose_mid_bathymetry = -bsose_depth[bsose_mid_lon_idx, bsose_lat_indices]
close(ds_bsose)

# ── 3. Plot Comparison ────────────────────────────────────────────────────────
println("Generating updated GPU comparison plot...")
output_fig = "plots/bathymetry_midlon_gpu_comparison.png"
mkpath(dirname(output_fig))

fig = Figure(size=(1300, 920), fontsize=14)

Label(fig[1, 1:2], 
      @sprintf("Mid-Longitude Bathymetry Profile (%.2f°E) with extent = 5800 m\nBSOSE Iteration 105 vs Model ETOPO2022 (GPU Vertical Grid: %d levels)", 
               mid_lon_val, Nz),
      fontsize=20, font=:bold)

ax = Axis(fig[2, 1],
    title=@sprintf("Transect Profile (GPU Resolution: Δx=Δy=%.2f°, %d vertical levels, extent=5800m, passes=5)", 1.0/SCALING, Nz),
    xlabel="Latitude (°N)",
    ylabel="Depth / Elevation z (m)",
    limits=(φ₁, φ₂, -6000, 100),
    xticks=φ₁:5:φ₂,
    yticks=-6000:1000:0)

# Fill ocean basin for BSOSE
band!(ax, bsose_lats, -6000, bsose_mid_bathymetry, color=(:skyblue, 0.15), label="BSOSE Ocean Basin")

# Plot profile lines
lines!(ax, bsose_lats, bsose_mid_bathymetry, color=:blue, linewidth=2.5, label="BSOSE Iteration 105 (1/6°)")
scatter!(ax, bsose_lats, bsose_mid_bathymetry, color=:blue, markersize=5)

lines!(ax, model_lats, model_mid_bathymetry, color=:crimson, linewidth=2.5, linestyle=:dash, 
       label=@sprintf("Model ETOPO2022 (GPU grid: 1/3°, %d levels, extent=5800m)", Nz))
scatter!(ax, model_lats, model_mid_bathymetry, color=:crimson, markersize=5, marker=:rect)

# Draw sea surface reference line
hlines!(ax, [0.0], color=:black, linewidth=1, linestyle=:dot)

# Draw lines for the GPU model's discrete vertical levels
for z_face in model_z_faces
    if z_face >= -6000
        hlines!(ax, [z_face], color=(:gray60, 0.20), linewidth=0.35)
    end
end

axislegend(ax, position=:rb)

# Difference Subplot
ax_diff = Axis(fig[3, 1],
    title="Bathymetry Discrepancy (Model ETOPO - BSOSE) along GPU Transect",
    xlabel="Latitude (°N)",
    ylabel="Δz = z_model - z_bsose (m)",
    limits=(φ₁, φ₂, -800, 800),
    xticks=φ₁:5:φ₂,
    yticks=-800:400:800)

bsose_interp = [bsose_mid_bathymetry[argmin(abs.(bsose_lats .- lat))] for lat in model_lats]
diff_z = model_mid_bathymetry .- bsose_interp

barplot!(ax_diff, model_lats, diff_z, color=[d < 0 ? :darkred : :navy for d in diff_z], width=0.25)
hlines!(ax_diff, [0.0], color=:black, linewidth=1)

rowsize!(fig.layout, 2, Relative(0.65))
rowsize!(fig.layout, 3, Relative(0.35))

save(output_fig, fig, px_per_unit=2)
println("Plot saved successfully to: ", output_fig)
