using NCDatasets
using CairoMakie
using Printf

surface_file = get(ENV, "SURFACE_FILE", "model_surface_fields.nc")
output = get(ENV, "ANIMATION_OUTPUT", "animations/bsose_simulation.gif")
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

# ── Colour limits: explicit if given, otherwise scanned from the data ──────────
#
# Each field takes its limits from an environment variable if one is set, and
# scans the data if not. The literals further down are only fallbacks for a field
# that is entirely NaN.
#
#   U_LIMS=0.5          symmetric, becomes (-0.5, 0.5)
#   U_LIMS="-0.4,0.6"   explicit lo,hi pair
#   (same for V_LIMS, T_LIMS, S_LIMS; unset means scan)
#
# CLIM_QUANTILE affects the scanned limits only. The default 1.0 uses the true
# min/max, i.e. the full range of the run. Set it below 1 (e.g. 0.995) when a
# handful of extreme cells flatten everything else -- values outside the range are
# still drawn, in the colormap's end colours, so the extremes stay visible.
clim_q = parse(Float64, get(ENV, "CLIM_QUANTILE", "1.0"))

"""
    env_lims(key)

Colour limits read from `ENV[key]`, or `nothing` if it is unset or blank. Accepts
a single magnitude (`"0.5"` -> `(-0.5, 0.5)`) or a `"lo,hi"` pair.
"""
function env_lims(key)
    haskey(ENV, key) || return nothing
    raw = strip(ENV[key])
    isempty(raw) && return nothing

    parts = [parse(Float64, strip(p)) for p in split(raw, ',') if !isempty(strip(p))]

    if length(parts) == 1
        m = abs(parts[1])
        m > 0 || error("$key magnitude must be non-zero; got \"$raw\"")
        return (-m, m)
    elseif length(parts) == 2
        parts[1] == parts[2] && error("$key lo and hi must differ; got \"$raw\"")
        return (minimum(parts), maximum(parts))
    else
        error("$key must be a magnitude (\"0.5\") or a lo,hi pair (\"-0.4,0.6\"); got \"$raw\"")
    end
end

"Use the limits from `key` if it is set, otherwise the scanned ones."
function resolve_lims(key, scanned)
    given = env_lims(key)
    return given === nothing ? (scanned, "scanned") : (given, "set via $key")
end

valid_T = filter(!isnan, T_data)
valid_S = filter(!isnan, S_data)
valid_u = filter(!isnan, u_data)
valid_v = filter(!isnan, v_data)
valid_speed = sqrt.(valid_u .^ 2 .+ valid_v .^ 2)

quantile_of(v, p) = (s = sort(v); s[clamp(round(Int, p * length(s)), 1, length(s))])

"Limits spanning the data, rounded outward to a multiple of `step`."
function scan_lims(v, fallback; step=0.1)
    isempty(v) && return fallback
    lo = clim_q >= 1 ? minimum(v) : quantile_of(v, 1 - clim_q)
    hi = clim_q >= 1 ? maximum(v) : quantile_of(v, clim_q)
    lo == hi && return (lo - step, hi + step)
    return (floor(lo / step) * step, ceil(hi / step) * step)
end

"Limits symmetric about zero, so that the diverging :balance colormap stays centred."
function scan_sym_lims(v, fallback; step=0.1)
    isempty(v) && return fallback
    a = abs.(v)
    m = ceil((clim_q >= 1 ? maximum(a) : quantile_of(a, clim_q)) / step) * step
    return m <= 0 ? fallback : (-m, m)
end

T_lims, T_src = resolve_lims("T_LIMS", scan_lims(valid_T, (-2.0, 20.0)))
S_lims, S_src = resolve_lims("S_LIMS", scan_lims(valid_S, (32.5, 35.5)))
u_lims, u_src = resolve_lims("U_LIMS", scan_sym_lims(valid_u, (-0.5, 0.5)))
v_lims, v_src = resolve_lims("V_LIMS", scan_sym_lims(valid_v, (-0.3, 0.3)))

println("Colorbar limits",
    clim_q >= 1 ? " (scans use full min/max)" : @sprintf(" (scans use quantile %.4g)", clim_q), ":")
@printf("  - Temperature (T) : %8.2f .. %8.2f °C    [%s]\n", T_lims..., T_src)
@printf("  - Salinity (S)    : %8.2f .. %8.2f PSU   [%s]\n", S_lims..., S_src)
@printf("  - Zonal vel (u)   : %8.2f .. %8.2f m/s   [%s]\n", u_lims..., u_src)
@printf("  - Merid vel (v)   : %8.2f .. %8.2f m/s   [%s]\n", v_lims..., v_src)

# Where the fast water actually is. If max sits far above p99, the full-range
# colorbar will be dominated by a few cells and the rest of the field will look
# flat -- rerun with CLIM_QUANTILE=0.995 in that case.
println("Velocity magnitudes over all frames:")
for (name, v) in (("|u|", abs.(valid_u)), ("|v|", abs.(valid_v)), ("speed", valid_speed))
    isempty(v) && continue
    @printf("  - %-5s  max %6.3f   p99.9 %6.3f   p99 %6.3f   p95 %6.3f   median %6.3f m/s\n",
        name, maximum(v), quantile_of(v, 0.999), quantile_of(v, 0.99),
        quantile_of(v, 0.95), quantile_of(v, 0.5))
end

scanning_velocities = u_src == "scanned" || v_src == "scanned"

if scanning_velocities && !isempty(valid_speed) &&
   maximum(valid_speed) > 3 * quantile_of(valid_speed, 0.99)
    @printf("  note: peak speed %.2f m/s is %.1fx the 99th percentile (%.2f m/s).\n",
        maximum(valid_speed), maximum(valid_speed) / quantile_of(valid_speed, 0.99),
        quantile_of(valid_speed, 0.99))
    println("        Rerun with CLIM_QUANTILE=0.995 to bring out the broader field.")
end

# Values outside colorrange are drawn in the colormap's end colours rather than
# being silently saturated, so clipped extremes stay visible on the map.
clip_lo(cmap) = cgrad(cmap)[0.0]
clip_hi(cmap) = cgrad(cmap)[1.0]

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
hm1 = heatmap!(ax1, lon, lat, surf_T, colormap=:thermal, colorrange=T_lims, nan_color=:gray30,
    lowclip=clip_lo(:thermal), highclip=clip_hi(:thermal))
Colorbar(fig[1, 2], hm1, label="°C")

ax2 = Axis(fig[1, 3], title="Surface Salinity", xlabel="Longitude (°E)", ylabel="Latitude (°N)",
    limits=(lon_bounds[1], lon_bounds[2], lat_bounds[1], lat_bounds[2]))
hm2 = heatmap!(ax2, lon, lat, surf_S, colormap=:haline, colorrange=S_lims, nan_color=:gray30,
    lowclip=clip_lo(:haline), highclip=clip_hi(:haline))
Colorbar(fig[1, 4], hm2, label="PSU")

# Row 2: Surface Zonal Velocity (u) & Surface Meridional Velocity (v)
ax3 = Axis(fig[2, 1], title="Surface Zonal Velocity (u)", xlabel="Longitude (°E)", ylabel="Latitude (°N)",
    limits=(lon_bounds[1], lon_bounds[2], lat_bounds[1], lat_bounds[2]))
hm3 = heatmap!(ax3, lon, lat, surf_u, colormap=:balance, colorrange=u_lims, nan_color=:gray30,
    lowclip=clip_lo(:balance), highclip=clip_hi(:balance))
Colorbar(fig[2, 2], hm3, label="m/s")

ax4 = Axis(fig[2, 3], title="Surface Meridional Velocity (v)", xlabel="Longitude (°E)", ylabel="Latitude (°N)",
    limits=(lon_bounds[1], lon_bounds[2], lat_bounds[1], lat_bounds[2]))
hm4 = heatmap!(ax4, lon, lat, surf_v, colormap=:balance, colorrange=v_lims, nan_color=:gray30,
    lowclip=clip_lo(:balance), highclip=clip_hi(:balance))
Colorbar(fig[2, 4], hm4, label="m/s")

println("Recording animation to $output (framerate = $framerate fps)...")
record(fig, output, 1:Nt; framerate=framerate) do t
    t_idx[] = t
end
println("Done! Saved to $output")
