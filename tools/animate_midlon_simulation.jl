# ==============================================================================
# animate_midlon_simulation.jl
#
# Creates a multi-panel animated GIF of the mid-longitude vertical transect:
#   - Row 1: Temperature (T) & Salinity (S) along latitude vs depth
#   - Row 2: Zonal Velocity (u) & Meridional Velocity (v) along latitude vs depth
# Includes dynamic color ranges, bathymetry bottom line, and time labels.
# ==============================================================================

using NCDatasets
using CairoMakie
using Printf

midlon_file = get(ENV, "MIDLON_FILE", "model_mid_lon.nc")
output = get(ENV, "ANIMATION_OUTPUT", "animations/bsose_simulation_midlon.gif")
framerate = parse(Int, get(ENV, "FRAMERATE", "8"))

mkpath(dirname(output))

if !isfile(midlon_file)
    error("Mid-longitude slice file '$midlon_file' not found! Make sure the model output writer ran.")
end

println("Loading simulation mid-longitude fields from: ", midlon_file)
ds = Dataset(midlon_file)
lat = Float64.(ds["φ_aca"][:])
z = Float64.(ds["z_aac"][:])
times = Float64.(ds["time"][:])
mid_lon = haskey(ds, "λ_caa") ? ds["λ_caa"][1] : 120.0
Nt = length(times)
Ny = length(lat)
Nz = length(z)

has_bottom = haskey(ds, "bottom_height")
bottom_h = has_bottom ? Float64.(ds["bottom_height"][1, :]) : fill(-5000.0, Ny)
has_mask = haskey(ds, "inactive_nodes_ccc")
mask_2d = has_mask ? (ds["inactive_nodes_ccc"][1, :, :] .!= 0) : falses(Ny, Nz)

println(@sprintf("  - Slice Location: Mid-Longitude = %.2f°E", mid_lon))
println(@sprintf("  - Grid Dimensions: Ny = %d, Nz = %d, Time steps = %d", Ny, Nz, Nt))
println(@sprintf("  - Latitude Range: [%.2f°S, %.2f°S]", minimum(lat), maximum(lat)))
println(@sprintf("  - Depth Range   : [%.1f m, %.1f m]", minimum(z), maximum(z)))

T_data = Array{Float32}(undef, Ny, Nz, Nt)
S_data = Array{Float32}(undef, Ny, Nz, Nt)
u_data = Array{Float32}(undef, Ny, Nz, Nt)
v_data = Array{Float32}(undef, Ny, Nz, Nt)

for t in 1:Nt
    Tt = Float32.(ds["T"][1, :, :, t])
    St = Float32.(ds["S"][1, :, :, t])
    ut = Float32.(ds["u"][1, :, :, t])
    v_raw = Float32.(ds["v"][1, :, :, t])

    # Interpolate v from φ_afa faces to φ_aca centers
    vc = 0.5f0 .* (v_raw[1:end-1, :] .+ v_raw[2:end, :])

    Tt[mask_2d] .= NaN32
    St[mask_2d] .= NaN32
    ut[mask_2d] .= NaN32
    vc[mask_2d] .= NaN32

    T_data[:, :, t] = Tt
    S_data[:, :, t] = St
    u_data[:, :, t] = ut
    v_data[:, :, t] = vc
end
close(ds)

# Scan dynamic colorbar limits directly from valid (non-NaN) data
valid_T = filter(!isnan, T_data)
valid_S = filter(!isnan, S_data)
valid_u = filter(!isnan, u_data)
valid_v = filter(!isnan, v_data)

T_lims = isempty(valid_T) ? (-2.0, 15.0) : (floor(minimum(valid_T) * 10) / 10, ceil(maximum(valid_T) * 10) / 10)
S_lims = isempty(valid_S) ? (33.0, 35.5) : (floor(minimum(valid_S) * 10) / 10, ceil(maximum(valid_S) * 10) / 10)

max_u = isempty(valid_u) ? 0.5 : ceil(maximum(abs, valid_u) * 10) / 10
max_v = isempty(valid_v) ? 0.3 : ceil(maximum(abs, valid_v) * 10) / 10
u_lims = (-max_u, max_u)
v_lims = (-max_v, max_v)

println("Dynamic Colorbar Limits Scanned from Data:")
println("  - Temperature (T) : $T_lims °C")
println("  - Salinity (S)    : $S_lims PSU")
println("  - Zonal vel (u)   : $u_lims m/s")
println("  - Merid vel (v)   : $v_lims m/s")

# ── Figure Layout: 2x2 grid matching animate_simulation.jl style ─────────────
fig = Figure(size=(1400, 950), fontsize=14)
t_idx = Observable(1)

slice_T = @lift(T_data[:, :, $t_idx])
slice_S = @lift(S_data[:, :, $t_idx])
slice_u = @lift(u_data[:, :, $t_idx])
slice_v = @lift(v_data[:, :, $t_idx])

time_str = @lift(@sprintf("Model Simulation Mid-Longitude (λ = %.1f°E): Day %.1f / %.1f (Step %d / %d)",
    mid_lon, times[$t_idx] / 86400, times[end] / 86400, $t_idx, Nt))
Label(fig[0, 1:4], time_str, fontsize=22, font=:bold)

lat_bounds = (minimum(lat), maximum(lat))
z_bounds = (minimum(z), 0.0)

# Row 1: Temperature & Salinity
ax1 = Axis(fig[1, 1], title="Temperature (T)", xlabel="Latitude (°N)", ylabel="Depth (m)",
    limits=(lat_bounds[1], lat_bounds[2], z_bounds[1], z_bounds[2]))
hm1 = heatmap!(ax1, lat, z, slice_T, colormap=:thermal, colorrange=T_lims, nan_color=:gray30)
lines!(ax1, lat, bottom_h, color=:black, linewidth=1.5)
Colorbar(fig[1, 2], hm1, label="°C")

ax2 = Axis(fig[1, 3], title="Salinity (S)", xlabel="Latitude (°N)", ylabel="Depth (m)",
    limits=(lat_bounds[1], lat_bounds[2], z_bounds[1], z_bounds[2]))
hm2 = heatmap!(ax2, lat, z, slice_S, colormap=:haline, colorrange=S_lims, nan_color=:gray30)
lines!(ax2, lat, bottom_h, color=:black, linewidth=1.5)
Colorbar(fig[1, 4], hm2, label="PSU")

# Row 2: Zonal Velocity (u) & Meridional Velocity (v)
ax3 = Axis(fig[2, 1], title="Zonal Velocity (u)", xlabel="Latitude (°N)", ylabel="Depth (m)",
    limits=(lat_bounds[1], lat_bounds[2], z_bounds[1], z_bounds[2]))
hm3 = heatmap!(ax3, lat, z, slice_u, colormap=:balance, colorrange=u_lims, nan_color=:gray30)
lines!(ax3, lat, bottom_h, color=:black, linewidth=1.5)
Colorbar(fig[2, 2], hm3, label="m/s")

ax4 = Axis(fig[2, 3], title="Meridional Velocity (v)", xlabel="Latitude (°N)", ylabel="Depth (m)",
    limits=(lat_bounds[1], lat_bounds[2], z_bounds[1], z_bounds[2]))
hm4 = heatmap!(ax4, lat, z, slice_v, colormap=:balance, colorrange=v_lims, nan_color=:gray30)
lines!(ax4, lat, bottom_h, color=:black, linewidth=1.5)
Colorbar(fig[2, 4], hm4, label="m/s")

println("Recording full-depth animation to $output (framerate = $framerate fps)...")
record(fig, output, 1:Nt; framerate=framerate) do t
    t_idx[] = t
end
println("Done! Full-depth animation saved to $output")

# ==============================================================================
# ANIMATION 2: Focused Upper-Ocean Transect (0 to -500 m)
# ==============================================================================
output_focus = get(ENV, "ANIMATION_FOCUS_OUTPUT", "animations/bsose_simulation_midlon_focus.gif")
println("\nGenerating focused upper-ocean animation (0 to -500 m) -> $output_focus ...")

# Upper 500m depth mask & limits
z_focus_mask = z .>= -500.0
valid_T_focus = filter(!isnan, T_data[:, z_focus_mask, :])
valid_S_focus = filter(!isnan, S_data[:, z_focus_mask, :])
valid_u_focus = filter(!isnan, u_data[:, z_focus_mask, :])
valid_v_focus = filter(!isnan, v_data[:, z_focus_mask, :])

# T_lims_focus = isempty(valid_T_focus) ? T_lims : (floor(minimum(valid_T_focus) * 10) / 10, ceil(maximum(valid_T_focus) * 10) / 10)
T_lims_focus = (-2, 2)
S_lims_focus = isempty(valid_S_focus) ? S_lims : (floor(minimum(valid_S_focus) * 10) / 10, ceil(maximum(valid_S_focus) * 10) / 10)

max_u_focus = isempty(valid_u_focus) ? 0.5 : ceil(maximum(abs, valid_u_focus) * 10) / 10
max_v_focus = isempty(valid_v_focus) ? 0.3 : ceil(maximum(abs, valid_v_focus) * 10) / 10
u_lims_focus = (-max_u_focus, max_u_focus)
v_lims_focus = (-max_v_focus, max_v_focus)

println("Focused Upper-Ocean (0 to -500m) Colorbar Limits:")
println("  - Temperature (T) : $T_lims_focus °C")
println("  - Salinity (S)    : $S_lims_focus PSU")
println("  - Zonal vel (u)   : $u_lims_focus m/s")
println("  - Merid vel (v)   : $v_lims_focus m/s")

fig_focus = Figure(size=(1400, 950), fontsize=14)
t_focus_idx = Observable(1)

slice_T_focus = @lift(T_data[:, :, $t_focus_idx])
slice_S_focus = @lift(S_data[:, :, $t_focus_idx])
slice_u_focus = @lift(u_data[:, :, $t_focus_idx])
slice_v_focus = @lift(v_data[:, :, $t_focus_idx])

time_str_focus = @lift(@sprintf("Model Simulation Mid-Longitude Focus (λ = %.1f°E, Upper 500m): Day %.1f / %.1f (Step %d / %d)",
    mid_lon, times[$t_focus_idx] / 86400, times[end] / 86400, $t_focus_idx, Nt))
Label(fig_focus[0, 1:4], time_str_focus, fontsize=22, font=:bold)

z_focus_bounds = (-500.0, 0.0)

# Row 1: Temperature & Salinity (Upper 500m)
ax1_f = Axis(fig_focus[1, 1], title="Conservative Temperature (T, 0 to -500m)", xlabel="Latitude (°N)", ylabel="Depth (m)",
    limits=(lat_bounds[1], lat_bounds[2], z_focus_bounds[1], z_focus_bounds[2]))
hm1_f = heatmap!(ax1_f, lat, z, slice_T_focus, colormap=:thermal, colorrange=T_lims_focus, nan_color=:gray30)
lines!(ax1_f, lat, bottom_h, color=:black, linewidth=2.0)
Colorbar(fig_focus[1, 2], hm1_f, label="°C")

ax2_f = Axis(fig_focus[1, 3], title="Salinity (S, 0 to -500m)", xlabel="Latitude (°N)", ylabel="Depth (m)",
    limits=(lat_bounds[1], lat_bounds[2], z_focus_bounds[1], z_focus_bounds[2]))
hm2_f = heatmap!(ax2_f, lat, z, slice_S_focus, colormap=:haline, colorrange=S_lims_focus, nan_color=:gray30)
lines!(ax2_f, lat, bottom_h, color=:black, linewidth=2.0)
Colorbar(fig_focus[1, 4], hm2_f, label="PSU")

# Row 2: Zonal Velocity (u) & Meridional Velocity (v) (Upper 500m)
ax3_f = Axis(fig_focus[2, 1], title="Zonal Velocity (u, 0 to -500m)", xlabel="Latitude (°N)", ylabel="Depth (m)",
    limits=(lat_bounds[1], lat_bounds[2], z_focus_bounds[1], z_focus_bounds[2]))
hm3_f = heatmap!(ax3_f, lat, z, slice_u_focus, colormap=:balance, colorrange=u_lims_focus, nan_color=:gray30)
lines!(ax3_f, lat, bottom_h, color=:black, linewidth=2.0)
Colorbar(fig_focus[2, 2], hm3_f, label="m/s")

ax4_f = Axis(fig_focus[2, 3], title="Meridional Velocity (v, 0 to -500m)", xlabel="Latitude (°N)", ylabel="Depth (m)",
    limits=(lat_bounds[1], lat_bounds[2], z_focus_bounds[1], z_focus_bounds[2]))
hm4_f = heatmap!(ax4_f, lat, z, slice_v_focus, colormap=:balance, colorrange=v_lims_focus, nan_color=:gray30)
lines!(ax4_f, lat, bottom_h, color=:black, linewidth=2.0)
Colorbar(fig_focus[2, 4], hm4_f, label="m/s")

println("Recording focused upper-ocean animation to $output_focus (framerate = $framerate fps)...")
record(fig_focus, output_focus, 1:Nt; framerate=framerate) do t
    t_focus_idx[] = t
end
println("Done! Focused upper-ocean animation saved to $output_focus")

