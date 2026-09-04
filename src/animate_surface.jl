using NCDatasets
using CairoMakie
using GeoMakie

println("Loading datasets...")
ds_surf = NCDataset("surface_fields.nc")
ds_slice = NCDataset("mid_lon.nc")

# Oceananigans geographic coordinate names
lon_c = ds_surf["λ_caa"][:]
lat_c = ds_surf["φ_aca"][:]
lon_f = ds_surf["λ_faa"][:]

# Depth coordinates
z_c = ds_slice["z_aac"][:]

times = ds_surf["time"][:]
Nt = length(times)

println("Found $Nt time steps.")

n = Observable(1)

# Surface fields (Top layer)
T_surf = @lift(ds_surf["T"][:, :, 1, $n])
u_surf = @lift(ds_surf["u"][:, :, 1, $n])

# Slice fields (lat-depth)
# indices=(Nx/2,:,:) extracts a single longitude slice, so we drop the first dim
T_slice = @lift(ds_slice["T"][1, :, :, $n])
u_slice = @lift(ds_slice["u"][1, :, :, $n])

time_str = @lift("Time: $(round(times[$n] / 86400, digits=1)) days")

# Set up the Figure
fig = Figure(size=(1400, 800))
Label(fig[0, :], time_str, fontsize=24, font=:bold)

# Top Left: Surface Temperature
ax1 = GeoAxis(fig[1, 1], dest="+proj=eqc", limits=((90, 150), (-70, -40)), title="Surface Temperature")
hm1 = surface!(ax1, lon_c, lat_c, T_surf, colormap=:thermal, colorrange=(0, 6))
poly!(ax1, GeoMakie.land(), color=:gray)
ax1.xgridstyle = :dash;
ax1.ygridstyle = :dash;

# Bottom Left: Surface Zonal Velocity
ax3 = GeoAxis(fig[2, 1], dest="+proj=eqc", limits=((90, 150), (-70, -40)), title="Surface Zonal Velocity (u)")
hm3 = surface!(ax3, lon_f, lat_c, u_surf, colormap=:balance, colorrange=(-0.5, 0.5))
poly!(ax3, GeoMakie.land(), color=:gray)
ax3.xgridstyle = :dash;
ax3.ygridstyle = :dash;

# Top Right: Lat-Z Temperature at mid-lon
ax2 = Axis(fig[1, 2], title="Temperature (Lat-Depth at Mid-Longitude)", xlabel="Latitude", ylabel="Depth (m)")
hm2 = heatmap!(ax2, lat_c, z_c, T_slice, colormap=:thermal, colorrange=(0, 6))

# Bottom Right: Lat-Z Zonal Velocity at mid-lon
ax4 = Axis(fig[2, 2], title="Zonal Velocity (Lat-Depth at Mid-Longitude)", xlabel="Latitude", ylabel="Depth (m)")
hm4 = heatmap!(ax4, lat_c, z_c, u_slice, colormap=:balance, colorrange=(-0.5, 0.5))

# Colorbars
Colorbar(fig[1, 3], hm1, label="T (°C)")
Colorbar(fig[2, 3], hm3, label="u (m/s)")

# Set column widths to perfectly match the data aspects!
# Column 1 width = 2.0 * Row 1 height (matches the 60x30 deg GeoAxis)
# Column 2 width = 1.0 * Row 1 height (makes the mid-lon slices perfect squares)
colsize!(fig.layout, 1, Aspect(1, 2.0))
colsize!(fig.layout, 2, Aspect(1, 1.0))

println("Recording animation to surface_and_slice_animation.mp4...")
record(fig, "surface_and_slice_animation.mp4", 1:Nt; framerate=10) do i
    n[] = i
end

close(ds_surf)
close(ds_slice)
println("Done!")
