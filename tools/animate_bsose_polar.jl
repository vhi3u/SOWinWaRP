using NCDatasets
using CairoMakie
using GeoMakie
using Dates

region_lon_bounds = (120.0, 180.0)
lat_bounds = (-70.0, -40.0)
depth_subsurface = -500.0

clean_data(arr) = Array{Float64}(map(x -> (ismissing(x) || x == 0.0) ? NaN : Float64(x), arr))

println("Loading BSOSE fields (UVEL, VVEL, THETA)...")
ds_U = NCDataset("data/Uvel_bsoseI156_2013to2024_monthly.nc")
ds_V = NCDataset("data/Vvel_bsoseI156_2013to2024_monthly.nc")
ds_T = NCDataset("data/Theta_bsoseI156_2013to2024_monthly.nc")

full_lon_U = ds_U["XG"][:]
full_lat_U = ds_U["YC"][:]
full_lon_V = ds_V["XC"][:]
full_lat_V = ds_V["YG"][:]
full_lon_T = ds_T["XC"][:]
full_lat_T = ds_T["YC"][:]
full_z_T = ds_T["Z"][:]

Nt = length(ds_T["time"])
t_indices = 1:Nt
time_vals = ds_T["time"][:]

iy_U = findall(y -> y >= lat_bounds[1] - 1.0 && y <= lat_bounds[2] + 1.0, full_lat_U)
iy_T = findall(y -> y >= lat_bounds[1] - 1.0 && y <= lat_bounds[2] + 1.0, full_lat_T)

k_depth = argmin(abs.(full_z_T .- depth_subsurface))
actual_depth = round(Int, abs(full_z_T[k_depth]))

sose_lon_U = full_lon_U
sose_lat_U = full_lat_U[iy_U]
sose_lon_T = full_lon_T
sose_lat_T = full_lat_T[iy_T]

Nx = length(full_lon_U)
Ny = length(iy_U)
lon_vort = full_lon_U
lat_vort = full_lat_V[iy_U[1:end-1]]

data_T_surf = Array{Float64}(undef, length(sose_lon_T), length(iy_T), length(t_indices))
data_T_depth = Array{Float64}(undef, length(sose_lon_T), length(iy_T), length(t_indices))
data_U_surf = Array{Float64}(undef, length(sose_lon_U), length(iy_U), length(t_indices))
data_vort = Array{Float64}(undef, Nx, Ny - 1, length(t_indices))

R = 6371000.0
deg2rad = π / 180.0
dlon = (full_lon_U[2] - full_lon_U[1]) * deg2rad

for (idx, t) in enumerate(t_indices)
    if idx % 12 == 1 || idx == length(t_indices)
        println("Processing month $idx/$(length(t_indices))...")
    end
    data_T_surf[:, :, idx] = clean_data(ds_T["THETA"][:, iy_T, 1, t])
    data_T_depth[:, :, idx] = clean_data(ds_T["THETA"][:, iy_T, k_depth, t])

    u_raw = ds_U["UVEL"][:, iy_U, 1, t]
    v_raw = ds_V["VVEL"][:, iy_U, 1, t]
    data_U_surf[:, :, idx] = clean_data(u_raw)

    u_clean = Array{Float64}(map(x -> (ismissing(x) || x == 0.0) ? 0.0 : Float64(x), u_raw))
    v_clean = Array{Float64}(map(x -> (ismissing(x) || x == 0.0) ? 0.0 : Float64(x), v_raw))

    for j in 2:Ny
        dy = R * (full_lat_U[iy_U[j]] - full_lat_U[iy_U[j-1]]) * deg2rad
        coslat = cos(full_lat_V[iy_U[j]] * deg2rad)
        dx = R * coslat * dlon
        for i in 1:Nx
            im1 = (i == 1) ? Nx : i - 1
            if ismissing(u_raw[i, j]) || ismissing(u_raw[i, j-1]) || ismissing(v_raw[i, j]) || ismissing(v_raw[im1, j]) ||
               u_raw[i, j] == 0.0 || u_raw[i, j-1] == 0.0 || v_raw[i, j] == 0.0 || v_raw[im1, j] == 0.0
                data_vort[i, j-1, idx] = NaN
            else
                dv_dx = (v_clean[i, j] - v_clean[im1, j]) / dx
                du_dy = (u_clean[i, j] - u_clean[i, j-1]) / dy
                data_vort[i, j-1, idx] = dv_dx - du_dy
            end
        end
    end
end

close(ds_U);
close(ds_V);
close(ds_T);

# Create 2-Panel Figure
fig = Figure(size=(1400, 700))
t_idx = Observable(1)

surf_T1 = lift(t -> data_T_surf[:, :, t], t_idx)
surf_Vort = lift(t -> data_vort[:, :, t], t_idx)

date_str = lift(t -> begin
    t_val = time_vals[t]
    formatted = t_val isa Dates.AbstractTime ? Dates.format(t_val, "yyyy-mm") : string(t_val)
    "BSOSE Polar Maps: $formatted"
end, t_idx)
Label(fig[0, 1:4], date_str, fontsize=24, font=:bold)

dest_proj = "+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=0 +datum=WGS84 +units=m"
polar_limits = ((-180, 180), (-90, -40))

lat_radial = range(lat_bounds[1], lat_bounds[2], length=100)
lon_arc = range(region_lon_bounds[1], region_lon_bounds[2], length=100)

function style_polar_axis!(ax)
    poly!(ax, GeoMakie.land(), color=:gray)
    lines!(ax, fill(region_lon_bounds[1], 100), lat_radial, color=:black, linewidth=2.5)
    lines!(ax, fill(region_lon_bounds[2], 100), lat_radial, color=:black, linewidth=2.5)
    lines!(ax, lon_arc, fill(lat_bounds[1], 100), color=:black, linewidth=2.5)
    lines!(ax, lon_arc, fill(lat_bounds[2], 100), color=:black, linewidth=2.5)
    ax.xgridstyle = :dash
    ax.ygridstyle = :dash
end

# 1. Surface Temperature
ax1 = GeoAxis(fig[1, 1], dest=dest_proj, limits=polar_limits, title="Surface Temperature")
hm1 = surface!(ax1, sose_lon_T, sose_lat_T, surf_T1, colormap=:thermal, colorrange=(-2, 10))
style_polar_axis!(ax1)
Colorbar(fig[1, 2], hm1, label="°C")

# 2. Surface Relative Vorticity
ax2 = GeoAxis(fig[1, 3], dest=dest_proj, limits=polar_limits, title="Surface Relative Vorticity (ζ)")
hm2 = surface!(ax2, lon_vort, lat_vort, surf_Vort, colormap=:balance, colorrange=(-5e-6, 5e-6))
style_polar_axis!(ax2)
Colorbar(fig[1, 4], hm2, label="s⁻¹")

out_file = "bsose_polar_2panel_2013to2024.gif"
println("Recording animation to $out_file...")
record(fig, out_file, 1:length(t_indices); framerate=12) do t
    t_idx[] = t
end
println("Done! Saved to $out_file")
