using NCDatasets
using CairoMakie
using Statistics: mean
using Printf

output = "plots/model_temp_and_velocity_distribution.png"
mkpath(dirname(output))

println("Loading model simulation surface output...")
ds_s = Dataset("model_surface_fields.nc")
lons = Float64.(ds_s["λ_caa"][:])
lats = Float64.(ds_s["φ_aca"][:])
times = Float64.(ds_s["time"][:])
Nt = length(times)
mask = ds_s["inactive_nodes_ccc"][:, :, 1] .!= 0

# Compute time-mean fields over the full simulation
T_mean = zeros(Float64, length(lons), length(lats))
u_mean = zeros(Float64, length(lons), length(lats))
v_mean = zeros(Float64, length(lons), length(lats))
speed_mean = zeros(Float64, length(lons), length(lats))

# Also track global max over all time steps
max_T_val = -Inf; max_T_loc = (0.0, 0.0, 0.0)
max_spd_val = -Inf; max_spd_loc = (0.0, 0.0, 0.0)
max_u_val = -Inf; min_u_val = Inf; max_u_loc = (0.0, 0.0, 0.0)
max_v_val = -Inf; min_v_val = Inf; max_v_loc = (0.0, 0.0, 0.0)

for t in 1:Nt
    Tt = Float64.(ds_s["T"][:, :, 1, t])
    u_raw = Float64.(ds_s["u"][:, :, 1, t])
    v_raw = Float64.(ds_s["v"][:, :, 1, t])

    uc = 0.5 .* (u_raw[1:end-1, :] .+ u_raw[2:end, :])
    vc = 0.5 .* (v_raw[:, 1:end-1] .+ v_raw[:, 2:end])

    Tt[mask] .= NaN
    uc[mask] .= NaN
    vc[mask] .= NaN
    spdt = sqrt.(uc .^ 2 .+ vc .^ 2)

    T_mean .+= ifelse.(isnan.(Tt), 0.0, Tt)
    u_mean .+= ifelse.(isnan.(uc), 0.0, uc)
    v_mean .+= ifelse.(isnan.(vc), 0.0, vc)
    speed_mean .+= ifelse.(isnan.(spdt), 0.0, spdt)

    # Track max T
    val_T = .!isnan.(Tt)
    if any(val_T)
        vmax, idx = findmax(Tt[val_T])
        if vmax > max_T_val
            global max_T_val = vmax
            c = findall(val_T)[idx]
            global max_T_loc = (lons[c[1]], lats[c[2]], times[t] / 86400)
        end
    end

    # Track max Speed
    val_spd = .!isnan.(spdt)
    if any(val_spd)
        vmax, idx = findmax(spdt[val_spd])
        if vmax > max_spd_val
            global max_spd_val = vmax
            c = findall(val_spd)[idx]
            global max_spd_loc = (lons[c[1]], lats[c[2]], times[t] / 86400)
        end
    end

    # Track u
    val_u = .!isnan.(uc)
    if any(val_u)
        vmax, idx = findmax(uc[val_u])
        if vmax > max_u_val
            global max_u_val = vmax
            c = findall(val_u)[idx]
            global max_u_loc = (lons[c[1]], lats[c[2]], times[t] / 86400)
        end
        vmin = minimum(uc[val_u])
        if vmin < min_u_val; global min_u_val = vmin; end
    end

    # Track v
    val_v = .!isnan.(vc)
    if any(val_v)
        vmax, idx = findmax(vc[val_v])
        if vmax > max_v_val
            global max_v_val = vmax
            c = findall(val_v)[idx]
            global max_v_loc = (lons[c[1]], lats[c[2]], times[t] / 86400)
        end
        vmin = minimum(vc[val_v])
        if vmin < min_v_val; global min_v_val = vmin; end
    end
end
close(ds_s)

T_mean ./= Nt
u_mean ./= Nt
v_mean ./= Nt
speed_mean ./= Nt

T_mean[mask] .= NaN
u_mean[mask] .= NaN
v_mean[mask] .= NaN
speed_mean[mask] .= NaN

println("=== Model Output Extrema Summary ===")
@printf("Highest Temperature: %.3f °C at (%.2f°E, %.2f°N) on Day %.1f\n", max_T_val, max_T_loc[1], max_T_loc[2], max_T_loc[3])
@printf("Highest Surface Speed: %.3f m/s at (%.2f°E, %.2f°N) on Day %.1f\n", max_spd_val, max_spd_loc[1], max_spd_loc[2], max_spd_loc[3])
@printf("Highest Zonal Velocity u: %+.3f m/s at (%.2f°E, %.2f°N) on Day %.1f (Min u: %+.3f m/s)\n", max_u_val, max_u_loc[1], max_u_loc[2], max_u_loc[3], min_u_val)
@printf("Highest Meridional Velocity v: %+.3f m/s at (%.2f°E, %.2f°N) on Day %.1f (Min v: %+.3f m/s)\n", max_v_val, max_v_loc[1], max_v_loc[2], max_v_loc[3], min_v_val)

# ── Figure Layout: 4 Panels matching animate_simulation / bsose distribution ─
fig = Figure(size=(1600, 1100), fontsize=13)
Label(fig[0, 1:2], "Ocean Model Simulation: Time-Mean Fields & Extrema Locations", fontsize=22, font=:bold)

# Panel 1: Time-Mean Surface Temperature
g1 = fig[1, 1:2] = GridLayout()
ax1 = Axis(g1[1, 1], title="Time-Mean Surface Temperature (°C)",
    xlabel="Longitude (°E)", ylabel="Latitude (°N)", limits=(90, 150, -70, -40),
    xticks=90:10:150, yticks=-70:5:-40, aspect=AxisAspect(2.0))
hm1 = heatmap!(ax1, lons, lats, T_mean, colormap=:thermal, colorrange=(-2.0, 20.0), nan_color=:gray30)
# Mark highest temperature location
scatter!(ax1, [max_T_loc[1]], [max_T_loc[2]], color=:red, markersize=18, strokewidth=2, strokecolor=:white)
text!(ax1, max_T_loc[1] - 1.5, max_T_loc[2] - 2.0,
    text=@sprintf("Max T: %.1f°C\n(%.1f°E, %.1f°N)", max_T_val, max_T_loc[1], max_T_loc[2]),
    color=:white, fontsize=12, align=(:right, :center))
Colorbar(g1[1, 2], hm1, label="°C", width=14)

# Panel 2: Meridional Profile of Mean Temperature
ax2 = Axis(g1[1, 3], title="Zonal-Mean Temp vs Lat",
    xlabel="Mean Temp (°C)", ylabel="Latitude (°N)", yticks=-70:5:-40, xticks=0:5:20)
lat_means_T = [mean(filter(!isnan, T_mean[:, j])) for j in 1:length(lats)]
lines!(ax2, lat_means_T, lats, color=:crimson, linewidth=3)
xlims!(ax2, -2.0, 20.0)
colsize!(g1, 3, Relative(0.28))

# Panel 3: Time-Mean Surface Speed
g2 = fig[2, 1:2] = GridLayout()
ax3 = Axis(g2[1, 1], title="Time-Mean Surface Speed √(u² + v²) (m/s)",
    xlabel="Longitude (°E)", ylabel="Latitude (°N)", limits=(90, 150, -70, -40),
    xticks=90:10:150, yticks=-70:5:-40, aspect=AxisAspect(2.0))
hm3 = heatmap!(ax3, lons, lats, speed_mean, colormap=:viridis, colorrange=(0.0, 0.6), nan_color=:gray30)
# Mark highest speed location
scatter!(ax3, [max_spd_loc[1]], [max_spd_loc[2]], color=:red, markersize=18, strokewidth=2, strokecolor=:white)
text!(ax3, max_spd_loc[1] - 1.5, max_spd_loc[2] - 2.5,
    text=@sprintf("Max Speed: %.2f m/s\n(%.1f°E, %.1f°N)", max_spd_val, max_spd_loc[1], max_spd_loc[2]),
    color=:white, fontsize=12, align=(:right, :center))
Colorbar(g2[1, 2], hm3, label="m/s", width=14)

# Panel 4: Meridional Profile of Mean Zonal Velocity u
ax4 = Axis(g2[1, 3], title="Zonal-Mean u vs Lat",
    xlabel="Mean u (m/s)", ylabel="Latitude (°N)", yticks=-70:5:-40, xticks=-0.1:0.1:0.3)
lat_means_u = [mean(filter(!isnan, u_mean[:, j])) for j in 1:length(lats)]
lines!(ax4, lat_means_u, lats, color=:navy, linewidth=3)
vlines!(ax4, 0.0, color=:gray50, linestyle=:dash)
xlims!(ax4, -0.1, 0.3)
colsize!(g2, 3, Relative(0.28))

save(output, fig)
println("Saved plot to: ", output)
