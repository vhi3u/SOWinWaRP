using NCDatasets
using CairoMakie
using Printf

surface_file = "model_surface_fields.nc"
output = "animations/bsose_simulation.mp4"
framerate = 8

mkpath(dirname(output))

println("Loading simulation surface fields from: ", surface_file)
ds = Dataset(surface_file)
lon = Float64.(ds["λ_caa"][:])
lat = Float64.(ds["φ_aca"][:])
times = Float64.(ds["time"][:])
Nt = length(times)

mask = ds["inactive_nodes_ccc"][:, :, 1] .!= 0

T_data = Array{Float32}(undef, length(lon), length(lat), Nt)
S_data = Array{Float32}(undef, length(lon), length(lat), Nt)
u_data = Array{Float32}(undef, length(lon), length(lat), Nt)
v_data = Array{Float32}(undef, length(lon), length(lat), Nt)

for t in 1:Nt
    Tt = ds["T"][:, :, 1, t]
    St = ds["S"][:, :, 1, t]
    u_raw = ds["u"][:, :, 1, t]
    v_raw = ds["v"][:, :, 1, t]

    uc = 0.5f0 .* (u_raw[1:end-1, :] .+ u_raw[2:end, :])
    vc = 0.5f0 .* (v_raw[:, 1:end-1] .+ v_raw[:, 2:end])

    Tt[mask] .= NaN32
    St[mask] .= NaN32
    uc[mask] .= NaN32
    vc[mask] .= NaN32

    T_data[:, :, t] = Tt
    S_data[:, :, t] = St
    u_data[:, :, t] = uc
    v_data[:, :, t] = vc
end
close(ds)

# Scan dynamic colorbar limits directly from valid (non-NaN) data
valid_T = filter(!isnan, T_data)
valid_S = filter(!isnan, S_data)
valid_u = filter(!isnan, u_data)
valid_v = filter(!isnan, v_data)

T_lims = isempty(valid_T) ? (-2.0, 20.0) : (floor(minimum(valid_T) * 10) / 10, ceil(maximum(valid_T) * 10) / 10)
S_lims = isempty(valid_S) ? (32.5, 35.5) : (floor(minimum(valid_S) * 10) / 10, ceil(maximum(valid_S) * 10) / 10)

# For velocities with diverging colormap (:balance), use symmetric bounds around zero
max_u = isempty(valid_u) ? 0.6 : ceil(maximum(abs, valid_u) * 10) / 10
max_v = isempty(valid_v) ? 0.4 : ceil(maximum(abs, valid_v) * 10) / 10
u_lims = (-max_u, max_u)
v_lims = (-max_v, max_v)

println("Dynamic Colorbar Limits Scanned from Data:")
println("  - Temperature (T) : $T_lims °C")
println("  - Salinity (S)    : $S_lims PSU")
println("  - Zonal vel (u)   : $u_lims m/s")
println("  - Merid vel (v)   : $v_lims m/s")

# ── Figure Layout: 2x2 grid following animate_bsose.jl style ────────────────
fig = Figure(size=(1400, 950), fontsize=14)
t_idx = Observable(1)

surf_T = @lift(T_data[:, :, $t_idx])
surf_S = @lift(S_data[:, :, $t_idx])
surf_u = @lift(u_data[:, :, $t_idx])
surf_v = @lift(v_data[:, :, $t_idx])

time_str = @lift(@sprintf("Model Simulation: Day %.1f / %.1f (Step %d / %d)",
    times[$t_idx] / 86400, times[end] / 86400, $t_idx, Nt))
Label(fig[0, 1:4], time_str, fontsize=22, font=:bold)

lon_bounds = (minimum(lon), maximum(lon))
lat_bounds = (minimum(lat), maximum(lat))

# Row 1: Surface Temperature & Surface Salinity
ax1 = Axis(fig[1, 1], title="Surface Temperature", xlabel="Longitude (°E)", ylabel="Latitude (°N)",
    limits=(lon_bounds[1], lon_bounds[2], lat_bounds[1], lat_bounds[2]))
hm1 = heatmap!(ax1, lon, lat, surf_T, colormap=:thermal, colorrange=T_lims, nan_color=:gray30)
Colorbar(fig[1, 2], hm1, label="°C")

ax2 = Axis(fig[1, 3], title="Surface Salinity", xlabel="Longitude (°E)", ylabel="Latitude (°N)",
    limits=(lon_bounds[1], lon_bounds[2], lat_bounds[1], lat_bounds[2]))
hm2 = heatmap!(ax2, lon, lat, surf_S, colormap=:haline, colorrange=S_lims, nan_color=:gray30)
Colorbar(fig[1, 4], hm2, label="PSU")

# Row 2: Surface Zonal Velocity (u) & Surface Meridional Velocity (v)
ax3 = Axis(fig[2, 1], title="Surface Zonal Velocity (u)", xlabel="Longitude (°E)", ylabel="Latitude (°N)",
    limits=(lon_bounds[1], lon_bounds[2], lat_bounds[1], lat_bounds[2]))
hm3 = heatmap!(ax3, lon, lat, surf_u, colormap=:balance, colorrange=u_lims, nan_color=:gray30)
Colorbar(fig[2, 2], hm3, label="m/s")

ax4 = Axis(fig[2, 3], title="Surface Meridional Velocity (v)", xlabel="Longitude (°E)", ylabel="Latitude (°N)",
    limits=(lon_bounds[1], lon_bounds[2], lat_bounds[1], lat_bounds[2]))
hm4 = heatmap!(ax4, lon, lat, surf_v, colormap=:balance, colorrange=v_lims, nan_color=:gray30)
Colorbar(fig[2, 4], hm4, label="m/s")

println("Recording animation to $output (framerate = $framerate fps)...")
record(fig, output, 1:Nt; framerate=framerate) do t
    t_idx[] = t
end
println("Done! Saved to $output")
