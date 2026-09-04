using NCDatasets
using CairoMakie
using Dates

lon_bounds = (120.0, 180.0)
lat_bounds = (-70.0, -40.0)
lon_mid = 150.0

clean_data(arr) = Array{Float64}(map(x -> (ismissing(x) || x == 0.0) ? NaN : x, arr))

# Load UVEL (Surface and Slice)
println("Loading UVEL...")
ds_U = NCDataset("data/Uvel_bsoseI156_2013to2024_monthly.nc")
full_lon_U = ds_U["XG"][:]
full_lat_U = ds_U["YC"][:]
full_z_U = ds_U["Z"][:]

Nt = length(ds_U["time"])
t_indices = 1:Nt
time_vals = ds_U["time"][:]

ix_U = findall(x -> x >= lon_bounds[1] - 1.0 && x <= lon_bounds[2] + 1.0, full_lon_U)
iy_U = findall(y -> y >= lat_bounds[1] - 1.0 && y <= lat_bounds[2] + 1.0, full_lat_U)
ix_mid_U = argmin(abs.(full_lon_U .- lon_mid))

sose_lon_U = full_lon_U[ix_U]
sose_lat_U = full_lat_U[iy_U]

data_U_surf = Array{Union{Missing,Float32}}(undef, length(ix_U), length(iy_U), length(t_indices))
data_U_slice = Array{Union{Missing,Float32}}(undef, length(iy_U), length(full_z_U), length(t_indices))

for (i, t) in enumerate(t_indices)
    data_U_surf[:, :, i] = ds_U["UVEL"][ix_U, iy_U, 1, t]
    data_U_slice[:, :, i] = ds_U["UVEL"][ix_mid_U, iy_U, :, t]
end
close(ds_U)
data_U_surf = clean_data(data_U_surf)
data_U_slice = clean_data(data_U_slice)


# Load THETA (Surface and Slice)
println("Loading THETA...")
ds_T = NCDataset("data/Theta_bsoseI156_2013to2024_monthly.nc")
full_lon_T = ds_T["XC"][:]
full_lat_T = ds_T["YC"][:]
full_z_T = ds_T["Z"][:]

ix_T = findall(x -> x >= lon_bounds[1] - 1.0 && x <= lon_bounds[2] + 1.0, full_lon_T)
iy_T = findall(y -> y >= lat_bounds[1] - 1.0 && y <= lat_bounds[2] + 1.0, full_lat_T)
ix_mid_T = argmin(abs.(full_lon_T .- lon_mid))

sose_lon_T = full_lon_T[ix_T]
sose_lat_T = full_lat_T[iy_T]

data_T_surf = Array{Union{Missing,Float32}}(undef, length(ix_T), length(iy_T), length(t_indices))
data_T_slice = Array{Union{Missing,Float32}}(undef, length(iy_T), length(full_z_T), length(t_indices))

for (i, t) in enumerate(t_indices)
    data_T_surf[:, :, i] = ds_T["THETA"][ix_T, iy_T, 1, t]
    data_T_slice[:, :, i] = ds_T["THETA"][ix_mid_T, iy_T, :, t]
end
close(ds_T)
data_T_surf = clean_data(data_T_surf)
data_T_slice = clean_data(data_T_slice)


# Create Figure
fig = Figure(size=(1400, 1500))
t_idx = Observable(1)

surf_U = lift(t -> data_U_surf[:, :, t], t_idx)
slice_U = lift(t -> data_U_slice[:, :, t], t_idx)

surf_T = lift(t -> data_T_surf[:, :, t], t_idx)
slice_T = lift(t -> data_T_slice[:, :, t], t_idx)

date_str = lift(t -> begin
    t_val = time_vals[t]
    formatted = t_val isa Dates.AbstractTime ? Dates.format(t_val, "yyyy-mm") : string(t_val)
    "BSOSE Data: $formatted"
end, t_idx)
Label(fig[0, 1:4], date_str, fontsize=24, font=:bold)

# Top row: Regional Surface Fields
ax1 = Axis(fig[1, 1], title="Surface Zonal Velocity (u)", xlabel="Longitude (°E)", ylabel="Latitude (°N)", limits=(lon_bounds[1], lon_bounds[2], lat_bounds[1], lat_bounds[2]))
hm1 = heatmap!(ax1, sose_lon_U, sose_lat_U, surf_U, colormap=:balance, colorrange=(-0.5, 0.5))
Colorbar(fig[1, 2], hm1, label="m/s")

ax2 = Axis(fig[1, 3], title="Surface Temperature", xlabel="Longitude (°E)", ylabel="Latitude (°N)", limits=(lon_bounds[1], lon_bounds[2], lat_bounds[1], lat_bounds[2]))
hm2 = heatmap!(ax2, sose_lon_T, sose_lat_T, surf_T, colormap=:thermal, colorrange=(-2, 10))
Colorbar(fig[1, 4], hm2, label="°C")

# Middle row: Mid-longitude Slices (Full Depth)
ax3 = Axis(fig[2, 1], title="Zonal Velocity (Lat-Depth at Mid-Longitude $(Int(lon_mid))°E)", xlabel="Latitude (°N)", ylabel="Depth (m)", limits=(lat_bounds[1], lat_bounds[2], minimum(full_z_U), 0))
hm3 = heatmap!(ax3, sose_lat_U, full_z_U, slice_U, colormap=:balance, colorrange=(-0.1, 0.1))
Colorbar(fig[2, 2], hm3, label="m/s")

ax4 = Axis(fig[2, 3], title="Temperature (Lat-Depth at Mid-Longitude $(Int(lon_mid))°E)", xlabel="Latitude (°N)", ylabel="Depth (m)", limits=(lat_bounds[1], lat_bounds[2], minimum(full_z_T), 0))
hm4 = heatmap!(ax4, sose_lat_T, full_z_T, slice_T, colormap=:thermal, colorrange=(-2, 4))
Colorbar(fig[2, 4], hm4, label="°C")

# Bottom row: Mid-longitude Slices (Top 500m Zoom)
ax5 = Axis(fig[3, 1], title="Zonal Velocity (Lat-Depth at Mid-Longitude $(Int(lon_mid))°E, Top 500m)", xlabel="Latitude (°N)", ylabel="Depth (m)", limits=(lat_bounds[1], -60, -500, 0))
hm5 = heatmap!(ax5, sose_lat_U, full_z_U, slice_U, colormap=:balance, colorrange=(-0.1, 0.1))
Colorbar(fig[3, 2], hm5, label="m/s")

ax6 = Axis(fig[3, 3], title="Temperature (Lat-Depth at Mid-Longitude $(Int(lon_mid))°E, Top 500m)", xlabel="Latitude (°N)", ylabel="Depth (m)", limits=(lat_bounds[1], -60, -500, 0))
hm6 = heatmap!(ax6, sose_lat_T, full_z_T, slice_T, colormap=:thermal, colorrange=(-2, 4))
Colorbar(fig[3, 4], hm6, label="°C")

out_file = "bsose_6panel_2013to2024.gif"
println("Recording animation to $out_file...")
record(fig, out_file, 1:length(t_indices); framerate=12) do t
    t_idx[] = t
end
println("Done! Saved to $out_file")
