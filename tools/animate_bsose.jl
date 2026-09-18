# ==============================================================================
# animate_bsose.jl
#
# Creates multi-panel animated GIFs of the BSOSE mid-longitude vertical transect:
#   - Row 1: Temperature (T) & Salinity (S) along latitude vs depth
#   - Row 2: Zonal Velocity (u) & Meridional Velocity (v) along latitude vs depth
#
# Produces two animations matching animate_midlon_simulation.jl:
#   1. Full-depth transect (down to -5800 m) -> animations/bsose_midlon.gif
#   2. Focused upper-ocean transect (0 to -500 m) with identical bounds to the
#      focused model panels -> animations/bsose_midlon_focus.gif
# ==============================================================================

using NCDatasets
using CairoMakie
using Dates
using Printf

# Configuration
data_dir = get(ENV, "BSOSE_DIR", "data")
lon_mid = parse(Float64, get(ENV, "LON_MID", "120.0"))
lat_bounds = (-70.0, -45.0)
framerate = parse(Int, get(ENV, "FRAMERATE", "8"))

output_full = get(ENV, "ANIMATION_OUTPUT", "animations/bsose_midlon.gif")
output_focus = get(ENV, "ANIMATION_FOCUS_OUTPUT", "animations/bsose_midlon_focus.gif")

mkpath(dirname(output_full))
mkpath(dirname(output_focus))

theta_file = joinpath(data_dir, "bsose_i105_2008to2012_monthly_Theta.nc")
salt_file  = joinpath(data_dir, "bsose_i105_2008to2012_monthly_Salt.nc")
uvel_file  = joinpath(data_dir, "bsose_i105_2008to2012_monthly_Uvel.nc")
vvel_file  = joinpath(data_dir, "bsose_i105_2008to2012_monthly_Vvel.nc")

for f in [theta_file, salt_file, uvel_file, vvel_file]
    if !isfile(f)
        error("Required BSOSE file '$f' not found!")
    end
end

println("Loading BSOSE coordinate grids and variables...")
ds_T = Dataset(theta_file)
ds_S = Dataset(salt_file)
ds_U = Dataset(uvel_file)
ds_V = Dataset(vvel_file)

xc = Float64.(ds_T["XC"][:])
yc = Float64.(ds_T["YC"][:])
z  = Float64.(ds_T["Z"][:])
time_vals = ds_T["time"][:]
Nt = length(time_vals)

# Find mid-longitude and latitude indices
ix_mid_c = argmin(abs.(xc .- lon_mid))
actual_mid_lon = xc[ix_mid_c]

xg = Float64.(ds_U["XG"][:])
ix_mid_u = argmin(abs.(xg .- lon_mid))

iy = findall(y -> y >= lat_bounds[1] && y <= lat_bounds[2], yc)
lat = yc[iy]
Ny = length(lat)
Nz = length(z)

println(@sprintf("  - Mid-Longitude: Requested = %.2f°E, Closest Grid = %.2f°E", lon_mid, actual_mid_lon))
println(@sprintf("  - Grid Dimensions: Ny = %d, Nz = %d, Months = %d", Ny, Nz, Nt))
println(@sprintf("  - Latitude Range: [%.2f°S, %.2f°S]", minimum(lat), maximum(lat)))
println(@sprintf("  - Depth Range   : [%.1f m, %.1f m]", minimum(z), maximum(z)))

# Bathymetry & Land Mask
depth = Float64.(ds_T["Depth"][ix_mid_c, iy])
bottom_h = -depth

hFac = ds_T["hFacC"][ix_mid_c, iy, :]
mask_2d = (hFac .== 0)

# Preallocate transect arrays (Ny, Nz, Nt)
T_data = Array{Float32}(undef, Ny, Nz, Nt)
S_data = Array{Float32}(undef, Ny, Nz, Nt)
u_data = Array{Float32}(undef, Ny, Nz, Nt)
v_data = Array{Float32}(undef, Ny, Nz, Nt)

# For V-velocity, extract boundary index for interpolation to cell centers
iy_v = iy[1]:(iy[end] + 1)

println("Extracting monthly transects...")
for t in 1:Nt
    Tt = Float32.(ds_T["THETA"][ix_mid_c, iy, :, t])
    St = Float32.(ds_S["SALT"][ix_mid_c, iy, :, t])
    ut = Float32.(ds_U["UVEL"][ix_mid_u, iy, :, t])
    v_raw = Float32.(ds_V["VVEL"][ix_mid_c, iy_v, :, t])

    # Interpolate v from YG faces to YC centers
    vt = 0.5f0 .* (v_raw[1:end-1, :] .+ v_raw[2:end, :])

    # Apply land mask and clean missing values
    Tt[mask_2d] .= NaN32
    St[mask_2d] .= NaN32
    ut[mask_2d] .= NaN32
    vt[mask_2d] .= NaN32

    # Handle any explicit missing values
    Tt[isnan.(Tt)] .= NaN32
    St[isnan.(St)] .= NaN32

    T_data[:, :, t] = Tt
    S_data[:, :, t] = St
    u_data[:, :, t] = ut
    v_data[:, :, t] = vt
end

close(ds_T)
close(ds_S)
close(ds_U)
close(ds_V)

# Full-depth dynamic colorbar limits
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

println("Full-Depth Colorbar Limits:")
println("  - Temperature (T) : $T_lims °C")
println("  - Salinity (S)    : $S_lims PSU")
println("  - Zonal vel (u)   : $u_lims m/s")
println("  - Merid vel (v)   : $v_lims m/s")

# ==============================================================================
# ANIMATION 1: Full-Depth Mid-Longitude Transect
# ==============================================================================
println("\nGenerating full-depth animation -> $output_full ...")
fig_full = Figure(size=(1400, 950), fontsize=14)
t_idx = Observable(1)

slice_T = @lift(T_data[:, :, $t_idx])
slice_S = @lift(S_data[:, :, $t_idx])
slice_u = @lift(u_data[:, :, $t_idx])
slice_v = @lift(v_data[:, :, $t_idx])

time_str = @lift(begin
    t_val = time_vals[$t_idx]
    formatted = t_val isa Dates.AbstractTime ? Dates.format(t_val, "yyyy-mm") : string(t_val)
    @sprintf("BSOSE Monthly Mid-Longitude (λ = %.1f°E): %s (Month %d / %d)", actual_mid_lon, formatted, $t_idx, Nt)
end)
Label(fig_full[0, 1:4], time_str, fontsize=22, font=:bold)

z_bounds = (minimum(z), 0.0)

# Row 1: Temperature & Salinity
ax1 = Axis(fig_full[1, 1], title="Temperature (T)", xlabel="Latitude (°N)", ylabel="Depth (m)",
    limits=(lat_bounds[1], lat_bounds[2], z_bounds[1], z_bounds[2]))
hm1 = heatmap!(ax1, lat, z, slice_T, colormap=:thermal, colorrange=T_lims, nan_color=:gray30)
lines!(ax1, lat, bottom_h, color=:black, linewidth=1.5)
Colorbar(fig_full[1, 2], hm1, label="°C")

ax2 = Axis(fig_full[1, 3], title="Salinity (S)", xlabel="Latitude (°N)", ylabel="Depth (m)",
    limits=(lat_bounds[1], lat_bounds[2], z_bounds[1], z_bounds[2]))
hm2 = heatmap!(ax2, lat, z, slice_S, colormap=:haline, colorrange=S_lims, nan_color=:gray30)
lines!(ax2, lat, bottom_h, color=:black, linewidth=1.5)
Colorbar(fig_full[1, 4], hm2, label="PSU")

# Row 2: Zonal Velocity (u) & Meridional Velocity (v)
ax3 = Axis(fig_full[2, 1], title="Zonal Velocity (u)", xlabel="Latitude (°N)", ylabel="Depth (m)",
    limits=(lat_bounds[1], lat_bounds[2], z_bounds[1], z_bounds[2]))
hm3 = heatmap!(ax3, lat, z, slice_u, colormap=:balance, colorrange=u_lims, nan_color=:gray30)
lines!(ax3, lat, bottom_h, color=:black, linewidth=1.5)
Colorbar(fig_full[2, 2], hm3, label="m/s")

ax4 = Axis(fig_full[2, 3], title="Meridional Velocity (v)", xlabel="Latitude (°N)", ylabel="Depth (m)",
    limits=(lat_bounds[1], lat_bounds[2], z_bounds[1], z_bounds[2]))
hm4 = heatmap!(ax4, lat, z, slice_v, colormap=:balance, colorrange=v_lims, nan_color=:gray30)
lines!(ax4, lat, bottom_h, color=:black, linewidth=1.5)
Colorbar(fig_full[2, 4], hm4, label="m/s")

println("Recording full-depth animation to $output_full (framerate = $framerate fps)...")
record(fig_full, output_full, 1:Nt; framerate=framerate) do t
    t_idx[] = t
end
println("Done! Saved to $output_full")

# ==============================================================================
# ANIMATION 2: Focused Upper-Ocean Transect (0 to -500 m)
# Uses identical bounds to animate_midlon_simulation.jl focused panels:
#   - Depth: 0 to -500 m
#   - Latitude: same bounds (-70.0 to -45.0)
#   - T_lims_focus: (-2, 2)
#   - S_lims_focus: (~33.0, 35.0)
#   - u_lims_focus: (-0.5, 0.5)
#   - v_lims_focus: (-0.3, 0.3)
# ==============================================================================
println("\nGenerating focused upper-ocean animation (0 to -500 m) -> $output_focus ...")

z_focus_mask = z .>= -500.0
valid_S_focus = filter(!isnan, S_data[:, z_focus_mask, :])

# Focused color limits matching animate_midlon_simulation.jl exactly
T_lims_focus = (-2.0, 2.0)
S_lims_focus = isempty(valid_S_focus) ? (33.0, 35.0) : (floor(minimum(valid_S_focus) * 10) / 10, ceil(maximum(valid_S_focus) * 10) / 10)
u_lims_focus = (-0.5, 0.5)
v_lims_focus = (-0.3, 0.3)

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

time_str_focus = @lift(begin
    t_val = time_vals[$t_focus_idx]
    formatted = t_val isa Dates.AbstractTime ? Dates.format(t_val, "yyyy-mm") : string(t_val)
    @sprintf("BSOSE Monthly Mid-Longitude Focus (λ = %.1f°E, Upper 500m): %s (Month %d / %d)", actual_mid_lon, formatted, $t_focus_idx, Nt)
end)
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
