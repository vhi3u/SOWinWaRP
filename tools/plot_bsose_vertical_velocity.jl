using NCDatasets
using CairoMakie
using Printf
using Statistics: mean

output = "plots/bsose_vertical_velocity_distribution.png"
mkpath(dirname(output))

println("Loading 3D BSOSE velocity datasets...")
ds_u = Dataset("data/bsose_i105_2008to2012_monthly_Uvel.nc")
ds_v = Dataset("data/bsose_i105_2008to2012_monthly_Vvel.nc")

lons_u = Float64.(ds_u["XG"][:])
lats_u = Float64.(ds_u["YC"][:])
depths = Float64.(ds_u["Z"][:])
Nz = length(depths)

# Focus on model domain: 90°E - 150°E, -70°N to -40°N
ix_dom = findall(x -> 90.0 <= x <= 150.0, lons_u)
iy_dom = findall(y -> -70.0 <= y <= -40.0, lats_u)

lon_dom = lons_u[ix_dom]
lat_dom = lats_u[iy_dom]

clean(A) = [(ismissing(x) || x == 0.0 || isnan(x)) ? NaN32 : Float32(x) for x in A]

println("Computing vertical profiles and 3D extrema...")
max_speed_z = zeros(Float64, Nz)
max_u_z = zeros(Float64, Nz)
min_u_z = zeros(Float64, Nz)
mean_speed_z = zeros(Float64, Nz)

# Store global 3D max locations
global_3d_max_speed = -Inf
loc_3d_max = (0.0, 0.0, 0.0)

for k in 1:Nz
    u_k = clean(ds_u["UVEL"][ix_dom, iy_dom, k, 6])
    v_k = clean(ds_v["VVEL"][ix_dom, iy_dom, k, 6])
    
    valid_u = .!isnan.(u_k)
    valid_v = .!isnan.(v_k)
    
    if any(valid_u)
        max_u_z[k] = maximum(u_k[valid_u])
        min_u_z[k] = minimum(u_k[valid_u])
        
        spd = sqrt.(u_k .^ 2 .+ v_k .^ 2)
        valid_spd_mask = .!isnan.(spd)
        if any(valid_spd_mask)
            vmax, idx = findmax(spd[valid_spd_mask])
            max_speed_z[k] = vmax
            mean_speed_z[k] = mean(spd[valid_spd_mask])
            
            if vmax > global_3d_max_speed
                global global_3d_max_speed = vmax
                coords = findall(valid_spd_mask)[idx]
                global loc_3d_max = (lon_dom[coords[1]], lat_dom[coords[2]], depths[k])
            end
        end
    else
        max_speed_z[k] = NaN
        max_u_z[k] = NaN
        min_u_z[k] = NaN
        mean_speed_z[k] = NaN
    end
end

# Mid-longitude vertical transect (around 146.5°E where the core jet sits)
ix_core = argmin(abs.(lon_dom .- 146.5))
core_lon = lon_dom[ix_core]
u_core_section = clean(ds_u["UVEL"][ix_dom[ix_core], iy_dom, :, 6]) # size: (Ny, Nz)

close(ds_u)
close(ds_v)

println("\n=== Vertical Velocity Extrema in Domain ===")
println(@sprintf("Overall 3D Max Speed: %.3f m/s at (%.2f°E, %.2f°N, depth = %.1f m)",
    global_3d_max_speed, loc_3d_max[1], loc_3d_max[2], loc_3d_max[3]))
println("Key Depth Levels:")
for d in [-2.1, -100.0, -200.0, -500.0, -1000.0, -2000.0, -3000.0, -4000.0]
    k = argmin(abs.(depths .- d))
    @printf("  Depth %7.1f m | Max Speed = %5.3f m/s | Max u = %+5.3f m/s | Mean Speed = %5.3f m/s\n",
        depths[k], max_speed_z[k], max_u_z[k], mean_speed_z[k])
end

# ── Create Figure ────────────────────────────────────────────────────────────
fig = Figure(size=(1600, 950), fontsize=13)
Label(fig[0, 1:3], "BSOSE Reanalysis: Vertical Extent of Velocity in Model Domain (90°-150°E)", fontsize=22, font=:bold)

# Panel 1: Vertical profile of Max and Mean Speed vs Depth
ax1 = Axis(fig[1, 1], title="Velocity Extrema vs Depth",
    xlabel="Speed / Velocity (m/s)", ylabel="Depth (m)", limits=(-0.3, 0.85, minimum(depths), 0),
    xticks=-0.2:0.2:0.8, yticks=-5000:1000:0)
lines!(ax1, max_speed_z, depths, color=:red, linewidth=3, label="Max Speed (domain)")
lines!(ax1, max_u_z, depths, color=:crimson, linewidth=2, linestyle=:dash, label="Max u (eastward)")
lines!(ax1, min_u_z, depths, color=:blue, linewidth=2, linestyle=:dash, label="Min u (westward)")
lines!(ax1, mean_speed_z, depths, color=:black, linewidth=2.5, label="Mean Speed")
vlines!(ax1, 0.0, color=:gray50, linestyle=:dot)
axislegend(ax1, position=:rb)

# Panel 2: Zoomed profile in the upper 1000 m
ax2 = Axis(fig[1, 2], title="Upper 1000 m Zoom",
    xlabel="Speed / Velocity (m/s)", ylabel="Depth (m)", limits=(-0.25, 0.85, -1000, 0),
    xticks=-0.2:0.2:0.8, yticks=-1000:200:0)
lines!(ax2, max_speed_z, depths, color=:red, linewidth=3, label="Max Speed")
lines!(ax2, max_u_z, depths, color=:crimson, linewidth=2, linestyle=:dash, label="Max u")
lines!(ax2, min_u_z, depths, color=:blue, linewidth=2, linestyle=:dash, label="Min u")
lines!(ax2, mean_speed_z, depths, color=:black, linewidth=2.5, label="Mean Speed")
vlines!(ax2, 0.0, color=:gray50, linestyle=:dot)
axislegend(ax2, position=:rb)

# Panel 3: Vertical Latitude-Depth Transect through the Jet Core (146.5°E)
# lat_dom has length Ny, depths has length Nz -> heatmap!(ax, x, y, matrix_Ny_Nz)
ax3 = Axis(fig[1, 3], title=@sprintf("Zonal Velocity u Section at %.1f°E (Core Jet)", core_lon),
    xlabel="Latitude (°N)", ylabel="Depth (m)", limits=(-70, -40, minimum(depths), 0),
    xticks=-70:10:-40, yticks=-5000:1000:0)
hm3 = heatmap!(ax3, lat_dom, depths, u_core_section, colormap=:balance, colorrange=(-0.4, 0.6), nan_color=:gray30)
Colorbar(fig[1, 4], hm3, label="m/s", width=14)

save(output, fig)
println("Saved vertical velocity plot to: ", output)
