# ==============================================================================
# seasonal_midlon_profiles.jl
#
# Seasonal-mean mid-longitude vertical transects of temperature and salinity:
#   - Row 1: Temperature (T) — austral summer | austral winter | winter - summer
#   - Row 2: Salinity (S)    — austral summer | austral winter | winter - summer
#
# Writes two figures, in the same spirit as animate_midlon_simulation.jl:
#   1. the full-depth transect, and
#   2. the focused upper-ocean transect (0 to -FOCUS_DEPTH m).
#
# Snapshots are assigned to a season by the calendar month of START_DATE + time,
# so START_DATE must match the `start_date` the simulation was run with.
# ==============================================================================

using NCDatasets
using CairoMakie
using Printf
using Dates

midlon_file = get(ENV, "MIDLON_FILE", "model_mid_lon.nc")
output = get(ENV, "SEASONAL_OUTPUT", "plots/seasonal_midlon.png")
output_focus = get(ENV, "SEASONAL_FOCUS_OUTPUT", "plots/seasonal_midlon_focus.png")
start_date = DateTime(get(ENV, "START_DATE", "2014-01-01"))
focus_depth = parse(Float64, get(ENV, "FOCUS_DEPTH", "500"))

parse_months(s) = parse.(Int, strip.(split(s, ",")))
summer_months = parse_months(get(ENV, "SUMMER_MONTHS", "12,1,2"))  # austral summer (DJF)
winter_months = parse_months(get(ENV, "WINTER_MONTHS", "6,7,8"))   # austral winter (JJA)

mkpath(dirname(output))
mkpath(dirname(output_focus))

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

dates = start_date .+ Second.(round.(Int, times))

println(@sprintf("  - Slice Location: Mid-Longitude = %.2f°E", mid_lon))
println(@sprintf("  - Grid Dimensions: Ny = %d, Nz = %d, Time steps = %d", Ny, Nz, Nt))
println(@sprintf("  - Latitude Range: [%.2f°N, %.2f°N]", minimum(lat), maximum(lat)))
println(@sprintf("  - Depth Range   : [%.1f m, %.1f m]", minimum(z), maximum(z)))
println(@sprintf("  - Model time zero = %s, so the output spans %s to %s",
    Dates.format(start_date, "yyyy-mm-dd"),
    Dates.format(dates[1], "yyyy-mm-dd"), Dates.format(dates[end], "yyyy-mm-dd")))

# ── Season membership ────────────────────────────────────────────────────────
# Each snapshot covers one output interval, so an unweighted mean over the
# snapshots of a season is the seasonal mean whenever the writer schedule is uniform.
season_indices(months) = findall(d -> month(d) in months, dates)

summer_idx = season_indices(summer_months)
winter_idx = season_indices(winter_months)

for (name, months, idx) in (("Austral summer", summer_months, summer_idx),
    ("Austral winter", winter_months, winter_idx))
    isempty(idx) && error("$name (months $(join(months, ","))) has no output snapshots. Check " *
                          "START_DATE (currently $(Dates.format(start_date, "yyyy-mm-dd"))) " *
                          "and the months requested.")
    println(@sprintf("  - %-14s months %-8s : %3d snapshots, %s to %s",
        name, join(months, ","), length(idx),
        Dates.format(dates[first(idx)], "yyyy-mm-dd"), Dates.format(dates[last(idx)], "yyyy-mm-dd")))
end

# ── Seasonal means ───────────────────────────────────────────────────────────
# Accumulated straight out of the file, one snapshot at a time, so a long record
# never has to sit in memory all at once.
function seasonal_mean(ds, var, idx, mask)
    acc = zeros(Float64, Ny, Nz)
    for t in idx
        acc .+= Float64.(ds[var][1, :, :, t])
    end
    field = Float32.(acc ./ length(idx))
    field[mask] .= NaN32
    return field
end

println("Averaging T and S over each season...")
T_summer = seasonal_mean(ds, "T", summer_idx, mask_2d)
T_winter = seasonal_mean(ds, "T", winter_idx, mask_2d)
S_summer = seasonal_mean(ds, "S", summer_idx, mask_2d)
S_winter = seasonal_mean(ds, "S", winter_idx, mask_2d)
close(ds)

T_diff = T_winter .- T_summer
S_diff = S_winter .- S_summer

# ── Color limits, scanned over whatever depth range is being plotted ─────────
function shared_limits(fields, depth_mask; digits=1)
    vals = vcat((filter(!isnan, f[:, depth_mask]) for f in fields)...)
    isempty(vals) && return (0.0, 1.0)
    scale = 10.0^digits
    lo, hi = floor(minimum(vals) * scale) / scale, ceil(maximum(vals) * scale) / scale
    return lo == hi ? (lo - 1 / scale, hi + 1 / scale) : (lo, hi)
end

function symmetric_limits(field, depth_mask; digits=2)
    vals = filter(!isnan, field[:, depth_mask])
    isempty(vals) && return (-1.0, 1.0)
    scale = 10.0^digits
    m = ceil(maximum(abs, vals) * scale) / scale
    m == 0 && (m = 1 / scale)
    return (-m, m)
end

lat_bounds = (minimum(lat), maximum(lat))

function transect_figure(depth_mask, z_bounds, tag)
    T_lims = shared_limits((T_summer, T_winter), depth_mask)
    S_lims = shared_limits((S_summer, S_winter), depth_mask; digits=2)
    dT_lims = symmetric_limits(T_diff, depth_mask)
    dS_lims = symmetric_limits(S_diff, depth_mask; digits=3)

    println("Colorbar limits ($tag):")
    println("  - Temperature (T)    : $T_lims °C")
    println("  - Salinity (S)       : $S_lims PSU")
    println("  - ΔT (winter-summer) : $dT_lims °C")
    println("  - ΔS (winter-summer) : $dS_lims PSU")

    fig = Figure(size=(1900, 950), fontsize=14)
    Label(fig[0, 1:6],
        @sprintf("Seasonal Mean Mid-Longitude Transect (λ = %.1f°E, %s)", mid_lon, tag),
        fontsize=22, font=:bold)

    rows = ((1, "Temperature (T)", "°C", :thermal, T_summer, T_winter, T_diff, T_lims, dT_lims),
        (2, "Salinity (S)", "PSU", :haline, S_summer, S_winter, S_diff, S_lims, dS_lims))

    for (row, name, unit, cmap, summer, winter, diff, lims, dlims) in rows
        columns = (("Austral summer mean", summer, cmap, lims),
            ("Austral winter mean", winter, cmap, lims),
            ("Winter - summer", diff, :balance, dlims))

        for (col, (subtitle, field, colormap, colorrange)) in enumerate(columns)
            ax = Axis(fig[row, 2col-1], title="$name — $subtitle",
                xlabel="Latitude (°N)", ylabel="Depth (m)",
                limits=(lat_bounds[1], lat_bounds[2], z_bounds[1], z_bounds[2]))
            hm = heatmap!(ax, lat, z, field; colormap, colorrange, nan_color=:gray30)
            lines!(ax, lat, bottom_h, color=:black, linewidth=2.0)
            Colorbar(fig[row, 2col], hm, label=unit)
        end
    end

    return fig
end

# ── Figure 1: full depth ─────────────────────────────────────────────────────
fig_full = transect_figure(trues(Nz), (minimum(z), 0.0), "full depth")
save(output, fig_full)
println("Done! Full-depth seasonal transect saved to $output")

# ── Figure 2: focused upper ocean ────────────────────────────────────────────
println(@sprintf("\nGenerating focused upper-ocean figure (0 to -%.0f m) -> %s ...", focus_depth, output_focus))
fig_focus = transect_figure(z .>= -focus_depth, (-focus_depth, 0.0), @sprintf("upper %.0f m", focus_depth))
save(output_focus, fig_focus)
println("Done! Focused upper-ocean seasonal transect saved to $output_focus")
