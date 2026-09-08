using NCDatasets
using CairoMakie
using Statistics: mean

output = "plots/bsose_temperature_distribution.png"
mkpath(dirname(output))

println("Loading BSOSE Theta dataset...")
ds = Dataset("data/bsose_i105_2008to2012_monthly_Theta.nc")
lons = Float64.(ds["XC"][:])
lats = Float64.(ds["YC"][:])

# Step 15 corresponds to March 2009 (peak summer surface heating)
theta_summer = ds["THETA"][:, :, 1, 15]

# Clean missing / zero / land
clean(A) = [(ismissing(x) || x == 0.0 || isnan(x)) ? NaN32 : Float32(x) for x in A]
surf_summer = clean(theta_summer)

# Domain subset: Lon in [90, 150], Lat in [-70, -40]
ix_dom = findall(x -> 90.0 <= x <= 150.0, lons)
iy_dom = findall(y -> -70.0 <= y <= -40.0, lats)

lon_dom = lons[ix_dom]
lat_dom = lats[iy_dom]
surf_dom = surf_summer[ix_dom, iy_dom]
close(ds)

# ── Figure Layout ────────────────────────────────────────────────────────────
println("Creating figure...")
fig = Figure(size=(1600, 1100), fontsize=13)
Label(fig[0, 1:2], "BSOSE Reanalysis: Sea Surface Temperature & Warm Water Distribution", fontsize=22, font=:bold)

# Panel 1: Full BSOSE Southern Ocean Surface Temperature
g1 = fig[1, 1:2] = GridLayout()
ax1 = Axis(g1[1, 1], title="Full BSOSE Domain (Summer Surface Temperature, March 2009)",
    xlabel="Longitude (°E)", ylabel="Latitude (°N)", limits=(0, 360, -78, -30),
    xticks=0:60:360, yticks=-75:15:-30)
hm1 = heatmap!(ax1, lons, lats, surf_summer, colormap=:thermal, colorrange=(-2.0, 25.0), nan_color=:gray30)
# Highlight model domain box
lines!(ax1, [90.0, 150.0, 150.0, 90.0, 90.0], [-70.0, -70.0, -40.0, -40.0, -70.0], color=:cyan, linewidth=2.5, label="Model Domain (90°-150°E, -70° to -40°N)")
axislegend(ax1, position=:lb)
Colorbar(g1[1, 2], hm1, label="°C", width=14)

# Panel 2: Model Domain Zoom
g2 = fig[2, 1:2] = GridLayout()
ax2 = Axis(g2[1, 1], title="Model Domain Zoom (90°E - 150°E, -70°N to -40°N)",
    xlabel="Longitude (°E)", ylabel="Latitude (°N)", limits=(90, 150, -70, -40),
    xticks=90:10:150, yticks=-70:5:-40, aspect=AxisAspect(2.0))
hm2 = heatmap!(ax2, lon_dom, lat_dom, surf_dom, colormap=:thermal, colorrange=(-2.0, 22.0), nan_color=:gray30)

# Mark the maximum temperature hotspot (149.5°E, -40.15°N)
scatter!(ax2, [149.5], [-40.15], color=:red, markersize=18, strokewidth=2, strokecolor=:white)
text!(ax2, 147.0, -43.5, text="Hotspot: 22.65°C\n(149.5°E, -40.15°N)", color=:white, fontsize=12, align=(:right, :center))
Colorbar(g2[1, 2], hm2, label="°C", width=14)

# Panel 3: Temperature vs Latitude Profile
ax3 = Axis(g2[1, 3], title="Domain Zonal-Mean Temp",
    xlabel="Mean Temp (°C)", ylabel="Latitude (°N)", yticks=-70:5:-40, xticks=0:5:20)
lat_means = Float64[]
for j in 1:length(lat_dom)
    valid = filter(!isnan, surf_dom[:, j])
    push!(lat_means, isempty(valid) ? NaN : mean(valid))
end
lines!(ax3, lat_means, lat_dom, color=:crimson, linewidth=3)
xlims!(ax3, -2.0, 20.0)
colsize!(g2, 3, Relative(0.28))

save(output, fig)
println("Saved plot to: ", output)
