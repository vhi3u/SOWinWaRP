using NCDatasets
using CairoMakie
using Statistics: mean

output = "plots/bsose_velocity_distribution.png"
mkpath(dirname(output))

println("Loading BSOSE velocity datasets...")
ds_u = Dataset("data/bsose_i105_2008to2012_monthly_Uvel.nc")
ds_v = Dataset("data/bsose_i105_2008to2012_monthly_Vvel.nc")

lons_u = Float64.(ds_u["XG"][:])
lats_u = Float64.(ds_u["YC"][:])

lons_v = Float64.(ds_v["XC"][:])
lats_v = Float64.(ds_v["YG"][:])

# Step 6 as chosen by user
u_raw = ds_u["UVEL"][:, :, 1, 6]
v_raw = ds_v["VVEL"][:, :, 1, 6]

clean(A) = [(ismissing(x) || x == 0.0 || isnan(x)) ? NaN32 : Float32(x) for x in A]
u_clean = clean(u_raw)
v_clean = clean(v_raw)

# Domain subset on cell centers (XC in [90, 150], YC in [-70, -40])
lons_c = Float64.(ds_v["XC"][:])
lats_c = Float64.(ds_u["YC"][:])

ix_dom = findall(x -> 90.0 <= x <= 150.0, lons_c)
iy_dom = findall(y -> -70.0 <= y <= -40.0, lats_c)

lon_dom = lons_c[ix_dom]
lat_dom = lats_c[iy_dom]

# Interpolate staggered u (at XG) to XC, and v (at YG) to YC on domain
# XG has same length as XC, periodic / shifted by 1/2 cell
Nx, Ny = size(u_clean)
u_c = 0.5f0 .* (u_clean[1:Nx, :] .+ u_clean[[2:Nx; 1], :])
v_c = 0.5f0 .* (v_clean[:, 1:Ny] .+ v_clean[:, [2:Ny; Ny]])

u_dom = u_c[ix_dom, iy_dom]
v_dom = v_c[ix_dom, iy_dom]

# Compute speed on common cell-center grid
speed_dom = sqrt.(u_dom .^ 2 .+ v_dom .^ 2)

close(ds_u)
close(ds_v)

# Find velocity extrema in domain
valid_mask = .!isnan.(speed_dom)
max_speed, idx_max = findmax(speed_dom[valid_mask])
indices = findall(valid_mask)
max_coord_idx = indices[idx_max]
max_lon = lon_dom[max_coord_idx[1]]
max_lat = lat_dom[max_coord_idx[2]]

# Max and min u in domain
u_valid_mask = .!isnan.(u_dom)
max_u_dom, idx_u_max = findmax(u_dom[u_valid_mask])
min_u_dom, idx_u_min = findmin(u_dom[u_valid_mask])
idx_u_max_coord = findall(u_valid_mask)[idx_u_max]
idx_u_min_coord = findall(u_valid_mask)[idx_u_min]
max_u_lon, max_u_lat = lon_dom[idx_u_max_coord[1]], lat_dom[idx_u_max_coord[2]]
min_u_lon, min_u_lat = lon_dom[idx_u_min_coord[1]], lat_dom[idx_u_min_coord[2]]

# Max and min v in domain
v_valid_mask = .!isnan.(v_dom)
max_v_dom, idx_v_max = findmax(v_dom[v_valid_mask])
min_v_dom, idx_v_min = findmin(v_dom[v_valid_mask])
idx_v_max_coord = findall(v_valid_mask)[idx_v_max]
idx_v_min_coord = findall(v_valid_mask)[idx_v_min]
max_v_lon, max_v_lat = lon_dom[idx_v_max_coord[1]], lat_dom[idx_v_max_coord[2]]
min_v_lon, min_v_lat = lon_dom[idx_v_min_coord[1]], lat_dom[idx_v_min_coord[2]]

println("=== Domain Velocity Extrema (90°E - 150°E, -70°N to -40°N) ===")
println("Max Surface Speed: $(round(max_speed, digits=3)) m/s at ($(round(max_lon, digits=2))°E, $(round(max_lat, digits=2))°N)")
println("Max Surface u vel: $(round(max_u_dom, digits=3)) m/s at ($(round(max_u_lon, digits=2))°E, $(round(max_u_lat, digits=2))°N)")
println("Min Surface u vel: $(round(min_u_dom, digits=3)) m/s at ($(round(min_u_lon, digits=2))°E, $(round(min_u_lat, digits=2))°N)")
println("Max Surface v vel: $(round(max_v_dom, digits=3)) m/s at ($(round(max_v_lon, digits=2))°E, $(round(max_v_lat, digits=2))°N)")
println("Min Surface v vel: $(round(min_v_dom, digits=3)) m/s at ($(round(min_v_lon, digits=2))°E, $(round(min_v_lat, digits=2))°N)")

# ── Figure Layout ────────────────────────────────────────────────────────────
println("Creating velocity figure...")
fig = Figure(size=(1600, 1100), fontsize=13)
Label(fig[0, 1:2], "BSOSE Reanalysis: Surface Velocity & Flow Distribution", fontsize=22, font=:bold)

# Panel 1: Full BSOSE Southern Ocean Surface Zonal Velocity (u)
g1 = fig[1, 1:2] = GridLayout()
ax1 = Axis(g1[1, 1], title="Full BSOSE Domain: Surface Zonal Velocity u (m/s)",
    xlabel="Longitude (°E)", ylabel="Latitude (°N)", limits=(0, 360, -78, -30),
    xticks=0:60:360, yticks=-75:15:-30)
hm1 = heatmap!(ax1, lons_u, lats_u, u_clean, colormap=:balance, colorrange=(-1.0, 1.0), nan_color=:gray30)
# Highlight model domain box
lines!(ax1, [90.0, 150.0, 150.0, 90.0, 90.0], [-70.0, -70.0, -40.0, -40.0, -70.0], color=:magenta, linewidth=2.5, label="Model Domain (90°-150°E, -70° to -40°N)")
axislegend(ax1, position=:lb)
Colorbar(g1[1, 2], hm1, label="m/s", width=14)

# Panel 2: Model Domain Zoom - Surface Speed
g2 = fig[2, 1:2] = GridLayout()
ax2 = Axis(g2[1, 1], title="Model Domain Zoom: Surface Speed √(u² + v²) (m/s)",
    xlabel="Longitude (°E)", ylabel="Latitude (°N)", limits=(90, 150, -70, -40),
    xticks=90:10:150, yticks=-70:5:-40, aspect=AxisAspect(2.0))
hm2 = heatmap!(ax2, lon_dom, lat_dom, speed_dom, colormap=:viridis, colorrange=(0.0, 0.6), nan_color=:gray30)

# Mark the maximum velocity hotspot
scatter!(ax2, [max_lon], [max_lat], color=:red, markersize=18, strokewidth=2, strokecolor=:white)
text!(ax2, max_lon - 2.0, max_lat - 2.5, text="Hotspot: $(round(max_speed, digits=2)) m/s\n($(round(max_lon, digits=1))°E, $(round(max_lat, digits=1))°N)", color=:white, fontsize=12, align=(:right, :center))
Colorbar(g2[1, 2], hm2, label="m/s", width=14)

# Panel 3: Velocity vs Latitude Profile (ACC Jet Core)
ax3 = Axis(g2[1, 3], title="Domain Zonal-Mean u",
    xlabel="Mean u (m/s)", ylabel="Latitude (°N)", yticks=-70:5:-40, xticks=-0.1:0.1:0.4)
lat_means_u = Float64[]
for j in 1:length(lat_dom)
    valid = filter(!isnan, u_dom[:, j])
    push!(lat_means_u, isempty(valid) ? NaN : mean(valid))
end
lines!(ax3, lat_means_u, lat_dom, color=:navy, linewidth=3)
vlines!(ax3, 0.0, color=:gray50, linestyle=:dash)
xlims!(ax3, -0.1, 0.35)
colsize!(g2, 3, Relative(0.28))

save(output, fig)
println("Saved plot to: ", output)
