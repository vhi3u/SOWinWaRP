# ==============================================================================
# animate_simulation.jl
#
# Animates a model run, surface maps and mid-longitude transects together:
#
#   RUN_surface.mp4        surface T, S, u, v
#   RUN_midlon.mp4         mid-longitude T, S, u, v against latitude and depth
#   RUN_midlon_focus.mp4   the same transect over the upper FOCUS_DEPTH metres
#
# Reads `RUN_surface.nc` and `RUN_midlon.nc`, the files `RUN_NAME` in src/model.jl
# produces. Either may be missing; the animations that depend on it are skipped.
#
#     julia --project=.
#     julia> include("tools/animate_simulation.jl")
#     julia> animate_simulation("model")
#
# or, on a CPU node, `./run_animate_simulation model`.
#
# Frames are captioned with their calendar date when the run's start date can be
# resolved; see "Calendar dates" below for where that comes from and how to set it
# by hand for a file that predates it.
# ==============================================================================

using NCDatasets
using CairoMakie
using Printf
using Dates

# ── Colour limits: explicit if given, otherwise scanned from the data ──────────
#
# Each field takes its limits from an environment variable if one is set, and
# scans the data if not. The literals in the calls below are only fallbacks for a
# field that is entirely NaN.
#
#   U_LIMS=0.5          symmetric, becomes (-0.5, 0.5)
#   U_LIMS="-0.4,0.6"   explicit lo,hi pair
#   (same for V_LIMS, T_LIMS, S_LIMS; unset means scan)
#
# CLIM_QUANTILE affects the scanned limits only. The default 1.0 uses the true
# min/max, i.e. the full range of the run. Set it below 1 (e.g. 0.995) when a
# handful of extreme cells flatten everything else -- values outside the range are
# still drawn, in the colormap's end colours, so the extremes stay visible.

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

quantile_of(v, p) = (s = sort(v); s[clamp(round(Int, p * length(s)), 1, length(s))])

"Limits spanning the data, rounded outward to a multiple of `step`."
function scan_lims(v, fallback; step=0.1, clim_q=1.0)
    isempty(v) && return fallback
    lo = clim_q >= 1 ? minimum(v) : quantile_of(v, 1 - clim_q)
    hi = clim_q >= 1 ? maximum(v) : quantile_of(v, clim_q)
    lo == hi && return (lo - step, hi + step)
    return (floor(lo / step) * step, ceil(hi / step) * step)
end

"Limits symmetric about zero, so that the diverging :balance colormap stays centred."
function scan_sym_lims(v, fallback; step=0.1, clim_q=1.0)
    isempty(v) && return fallback
    a = abs.(v)
    m = ceil((clim_q >= 1 ? maximum(a) : quantile_of(a, clim_q)) / step) * step
    return m <= 0 ? fallback : (-m, m)
end

# Values outside colorrange are drawn in the colormap's end colours rather than
# being silently saturated, so clipped extremes stay visible.
clip_lo(cmap) = cgrad(cmap)[0.0]
clip_hi(cmap) = cgrad(cmap)[1.0]

"Report the chosen limits and where each came from."
function report_limits(pairs, clim_q)
    println("Colorbar limits",
        clim_q >= 1 ? " (scans use full min/max)" : @sprintf(" (scans use quantile %.4g)", clim_q), ":")
    for (name, unit, lims, src) in pairs
        @printf("  - %-17s : %8.2f .. %8.2f %-5s [%s]\n", name, lims..., unit, src)
    end
end

"""
    report_speeds(valid_u, valid_v, scanning)

Where the fast water actually is. If the maximum sits far above the 99th
percentile, a full-range colorbar is dominated by a few cells and the rest of the
field looks flat; say so, since `CLIM_QUANTILE` is the fix.
"""
function report_speeds(valid_u, valid_v, scanning)
    valid_speed = sqrt.(valid_u .^ 2 .+ valid_v .^ 2)
    println("Velocity magnitudes over all frames:")
    for (name, v) in (("|u|", abs.(valid_u)), ("|v|", abs.(valid_v)), ("speed", valid_speed))
        isempty(v) && continue
        @printf("  - %-5s  max %6.3f   p99.9 %6.3f   p99 %6.3f   p95 %6.3f   median %6.3f m/s\n",
            name, maximum(v), quantile_of(v, 0.999), quantile_of(v, 0.99),
            quantile_of(v, 0.95), quantile_of(v, 0.5))
    end

    if scanning && !isempty(valid_speed) &&
       maximum(valid_speed) > 3 * quantile_of(valid_speed, 0.99)
        @printf("  note: peak speed %.2f m/s is %.1fx the 99th percentile (%.2f m/s).\n",
            maximum(valid_speed), maximum(valid_speed) / quantile_of(valid_speed, 0.99),
            quantile_of(valid_speed, 0.99))
        println("        Rerun with CLIM_QUANTILE=0.995 to bring out the broader field.")
    end
end

# ── Calendar dates ────────────────────────────────────────────────────────────
#
# A model NetCDF records `time` as seconds from model time zero and carries no
# reference date of its own -- the only date NetCDF adds is when the file was
# written. Model time zero is `start_date` in src/model.jl, so a calendar caption
# needs that date from outside the time axis. In order of preference:
#
#   1. the `start_date` keyword argument,
#   2. the START_DATE environment variable,
#   3. the file's own `start_date` global attribute, written by src/model.jl,
#   4. the `start_date` currently set in src/model.jl.
#
# Only (3) is tied to the run being animated, so the source is always printed: a
# wrong reference mislabels every frame, and (4) in particular goes stale as soon
# as src/model.jl is edited for the next experiment. Files written before the
# attribute existed fall through to (4); pass `start_date` to pin them. When none
# of the four resolves, captions fall back to the elapsed-day counter alone.

"The run's start date from the file's global attributes, or `nothing`."
function attribute_start_date(ds)
    haskey(ds.attrib, "start_date") || return nothing
    raw = string(ds.attrib["start_date"])
    try
        return DateTime(raw)
    catch
        @warn "Ignoring unparseable start_date attribute \"$raw\"."
        return nothing
    end
end

"""
    model_jl_start_date(path)

The `start_date` assignment currently in src/model.jl, or `nothing` if the file is
missing or sets it some other way. The pattern is anchored at the start of a line so
the commented-out CPU-test dates just below it are not picked up.
"""
function model_jl_start_date(path=normpath(joinpath(@__DIR__, "..", "src", "model.jl")))
    isfile(path) || return nothing
    m = match(r"^[ \t]*start_date[ \t]*=[ \t]*DateTime\(\"([^\"]+)\"\)"m, read(path, String))
    m === nothing && return nothing
    try
        return DateTime(m.captures[1])
    catch
        return nothing
    end
end

"""
    resolve_start_date(ds, given)

`(start_date, source)` for the file open as `ds`, following the order above.
`start_date` is `nothing` when no source resolves.
"""
function resolve_start_date(ds, given)
    given !== nothing && return DateTime(given), "given as start_date"

    env = strip(get(ENV, "START_DATE", ""))
    if !isempty(env)
        return DateTime(env), "set via START_DATE"
    end

    from_file = attribute_start_date(ds)
    from_file !== nothing && return from_file, "from the file's start_date attribute"

    from_source = model_jl_start_date()
    from_source !== nothing &&
        return from_source, "parsed from src/model.jl — verify it is this run's start date"

    return nothing, "unresolved — captioning by elapsed day only"
end

"Report the reference date a set of captions will be built on."
function report_start_date(start_date, source)
    if start_date === nothing
        println("  - Calendar : ", source)
    else
        println("  - Calendar : model time zero = ",
            Dates.format(start_date, "yyyy-mm-dd"), " [", source, "]")
    end
end

"""
    time_caption(start_date, t, t_end, n, Nt)

The caption for model time `t` seconds: the calendar date followed by the elapsed-day
and step counters, or the counters alone when there is no reference date. Fractional
seconds are kept, so a sub-daily output interval still lands on the right day.
"""
function time_caption(start_date, t, t_end, n, Nt)
    counters = @sprintf("Day %.1f / %.1f (Step %d / %d)", t / 86400, t_end / 86400, n, Nt)
    start_date === nothing && return counters
    date = Dates.format(start_date + Millisecond(round(Int, 1000t)), "d U yyyy")
    return string(date, "  —  ", counters)
end

# ── Surface maps ──────────────────────────────────────────────────────────────

function animate_surface(surface_file, output; framerate, clim_q, start_date=nothing, video...)
    println("\nLoading surface fields from: ", surface_file)
    ds = Dataset(surface_file)
    lon = Float64.(ds["λ_caa"][:])
    lat = Float64.(ds["φ_aca"][:])
    times = Float64.(ds["time"][:])
    Nt = length(times)

    start_date, date_source = resolve_start_date(ds, start_date)
    report_start_date(start_date, date_source)

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

        # u and v live on faces; centre them on the tracer grid before plotting
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

    valid_T = filter(!isnan, T_data)
    valid_S = filter(!isnan, S_data)
    valid_u = filter(!isnan, u_data)
    valid_v = filter(!isnan, v_data)

    T_lims, T_src = resolve_lims("T_LIMS", scan_lims(valid_T, (-2.0, 20.0); clim_q))
    S_lims, S_src = resolve_lims("S_LIMS", scan_lims(valid_S, (32.5, 35.5); clim_q))
    u_lims, u_src = resolve_lims("U_LIMS", scan_sym_lims(valid_u, (-0.5, 0.5); clim_q))
    v_lims, v_src = resolve_lims("V_LIMS", scan_sym_lims(valid_v, (-0.3, 0.3); clim_q))

    report_limits((("Temperature (T)", "°C", T_lims, T_src),
            ("Salinity (S)", "PSU", S_lims, S_src),
            ("Zonal vel (u)", "m/s", u_lims, u_src),
            ("Merid vel (v)", "m/s", v_lims, v_src)), clim_q)
    report_speeds(valid_u, valid_v, u_src == "scanned" || v_src == "scanned")

    fig = Figure(size=(1400, 950), fontsize=14)
    t_idx = Observable(1)

    time_str = @lift(string("Model Simulation: ",
        time_caption(start_date, times[$t_idx], times[end], $t_idx, Nt)))
    Label(fig[0, 1:4], time_str, fontsize=22, font=:bold)

    bounds = (minimum(lon), maximum(lon), minimum(lat), maximum(lat))

    panels = (("Surface Temperature", T_data, :thermal, T_lims, "°C", 1, 1),
        ("Surface Salinity", S_data, :haline, S_lims, "PSU", 1, 3),
        ("Surface Zonal Velocity (u)", u_data, :balance, u_lims, "m/s", 2, 1),
        ("Surface Meridional Velocity (v)", v_data, :balance, v_lims, "m/s", 2, 3))

    for (title, data, cmap, lims, unit, row, col) in panels
        slice = @lift(data[:, :, $t_idx])
        ax = Axis(fig[row, col]; title, xlabel="Longitude (°E)", ylabel="Latitude (°N)", limits=bounds)
        hm = heatmap!(ax, lon, lat, slice, colormap=cmap, colorrange=lims, nan_color=:gray30,
            lowclip=clip_lo(cmap), highclip=clip_hi(cmap))
        Colorbar(fig[row, col+1], hm, label=unit)
    end

    println("Recording surface animation to $output (framerate = $framerate fps)...")
    record(fig, output, 1:Nt; framerate, video...) do t
        t_idx[] = t
    end
    println(@sprintf("Done! Saved to %s (%.1f MB)", output, filesize(output) / 1024^2))
end

# ── Mid-longitude transects ───────────────────────────────────────────────────

function animate_midlon(midlon_file, output, output_focus; framerate, clim_q, focus_depth,
    start_date=nothing, video...)
    println("\nLoading mid-longitude fields from: ", midlon_file)
    ds = Dataset(midlon_file)
    lat = Float64.(ds["φ_aca"][:])
    z = Float64.(ds["z_aac"][:])
    times = Float64.(ds["time"][:])
    mid_lon = haskey(ds, "λ_caa") ? ds["λ_caa"][1] : 120.0
    Nt, Ny, Nz = length(times), length(lat), length(z)

    start_date, date_source = resolve_start_date(ds, start_date)

    bottom_h = haskey(ds, "bottom_height") ? Float64.(ds["bottom_height"][1, :]) : fill(-5000.0, Ny)
    mask_2d = haskey(ds, "inactive_nodes_ccc") ? (ds["inactive_nodes_ccc"][1, :, :] .!= 0) : falses(Ny, Nz)

    println(@sprintf("  - Slice location: mid-longitude = %.2f°E", mid_lon))
    println(@sprintf("  - Grid: Ny = %d, Nz = %d, %d snapshots", Ny, Nz, Nt))
    println(@sprintf("  - Latitude range: [%.2f°N, %.2f°N]", minimum(lat), maximum(lat)))
    println(@sprintf("  - Depth range   : [%.1f m, %.1f m]", minimum(z), maximum(z)))
    report_start_date(start_date, date_source)

    T_data = Array{Float32}(undef, Ny, Nz, Nt)
    S_data = Array{Float32}(undef, Ny, Nz, Nt)
    u_data = Array{Float32}(undef, Ny, Nz, Nt)
    v_data = Array{Float32}(undef, Ny, Nz, Nt)

    for t in 1:Nt
        Tt = Float32.(ds["T"][1, :, :, t])
        St = Float32.(ds["S"][1, :, :, t])
        ut = Float32.(ds["u"][1, :, :, t])
        v_raw = Float32.(ds["v"][1, :, :, t])

        # v lives on φ_afa faces; centre it on φ_aca
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

    lat_bounds = (minimum(lat), maximum(lat))

    # One animation per depth range: the full water column, then the upper ocean with
    # its limits rescanned over just that band, where the seasonal signal lives.
    function transect_animation(out, depth_mask, z_bounds, tag)
        valid(d) = filter(!isnan, d[:, depth_mask, :])
        valid_T, valid_S = valid(T_data), valid(S_data)
        valid_u, valid_v = valid(u_data), valid(v_data)

        T_lims, T_src = resolve_lims("T_LIMS", scan_lims(valid_T, (-2.0, 15.0); clim_q))
        S_lims, S_src = resolve_lims("S_LIMS", scan_lims(valid_S, (33.0, 35.5); clim_q))
        u_lims, u_src = resolve_lims("U_LIMS", scan_sym_lims(valid_u, (-0.5, 0.5); clim_q))
        v_lims, v_src = resolve_lims("V_LIMS", scan_sym_lims(valid_v, (-0.3, 0.3); clim_q))

        println("\n$tag:")
        report_limits((("Temperature (T)", "°C", T_lims, T_src),
                ("Salinity (S)", "PSU", S_lims, S_src),
                ("Zonal vel (u)", "m/s", u_lims, u_src),
                ("Merid vel (v)", "m/s", v_lims, v_src)), clim_q)
        report_speeds(valid_u, valid_v, u_src == "scanned" || v_src == "scanned")

        fig = Figure(size=(1400, 950), fontsize=14)
        t_idx = Observable(1)

        time_str = @lift(string(@sprintf("Mid-Longitude (λ = %.1f°E, %s): ", mid_lon, tag),
            time_caption(start_date, times[$t_idx], times[end], $t_idx, Nt)))
        Label(fig[0, 1:4], time_str, fontsize=22, font=:bold)

        bounds = (lat_bounds[1], lat_bounds[2], z_bounds[1], z_bounds[2])
        panels = (("Temperature (T)", T_data, :thermal, T_lims, "°C", 1, 1),
            ("Salinity (S)", S_data, :haline, S_lims, "PSU", 1, 3),
            ("Zonal Velocity (u)", u_data, :balance, u_lims, "m/s", 2, 1),
            ("Meridional Velocity (v)", v_data, :balance, v_lims, "m/s", 2, 3))

        for (title, data, cmap, lims, unit, row, col) in panels
            slice = @lift(data[:, :, $t_idx])
            ax = Axis(fig[row, col]; title, xlabel="Latitude (°N)", ylabel="Depth (m)", limits=bounds)
            hm = heatmap!(ax, lat, z, slice, colormap=cmap, colorrange=lims, nan_color=:gray30,
                lowclip=clip_lo(cmap), highclip=clip_hi(cmap))
            lines!(ax, lat, bottom_h, color=:black, linewidth=2.0)
            Colorbar(fig[row, col+1], hm, label=unit)
        end

        println("Recording $tag animation to $out (framerate = $framerate fps)...")
        record(fig, out, 1:Nt; framerate, video...) do t
            t_idx[] = t
        end
        println(@sprintf("Done! Saved to %s (%.1f MB)", out, filesize(out) / 1024^2))
    end

    transect_animation(output, trues(Nz), (minimum(z), 0.0), "full depth")
    transect_animation(output_focus, z .>= -focus_depth, (-focus_depth, 0.0),
        @sprintf("upper %.0f m", focus_depth))
end

# ── Entry point ───────────────────────────────────────────────────────────────

"""
    animate_simulation(run_name; kwargs...)

Animate the run written under `run_name`, reading `\$(run_name)_surface.nc` and
`\$(run_name)_midlon.nc` and writing three videos into `output_dir`. A missing input
file is reported and its animations skipped, so this works on a run that wrote
only one of the two.

Keyword arguments (each defaults to an environment variable, then to a literal):
`surface_file`, `midlon_file`, `output_dir`, `framerate` (`FRAMERATE`),
`focus_depth` (`FOCUS_DEPTH`), `clim_quantile` (`CLIM_QUANTILE`),
`format` (`VIDEO_FORMAT`, "mp4" or "gif") and `compression` (`VIDEO_COMPRESSION`,
ffmpeg's -crf: lower is better quality and a bigger file, 20 by default, mp4 only).

`start_date` (`START_DATE`) is the calendar date of model time zero, used to caption
each frame with its date. It defaults to `nothing`, which lets each file resolve its
own — see "Calendar dates" above. Pass it only to override what the files say.
"""
function animate_simulation(run_name;
    surface_file=get(ENV, "SURFACE_FILE", "$(run_name)_surface.nc"),
    midlon_file=get(ENV, "MIDLON_FILE", "$(run_name)_midlon.nc"),
    output_dir=get(ENV, "ANIMATION_DIR", "animations"),
    framerate=parse(Int, get(ENV, "FRAMERATE", "8")),
    focus_depth=parse(Float64, get(ENV, "FOCUS_DEPTH", "500")),
    clim_quantile=parse(Float64, get(ENV, "CLIM_QUANTILE", "1.0")),
    start_date=nothing,
    format=get(ENV, "VIDEO_FORMAT", "mp4"),
    compression=parse(Int, get(ENV, "VIDEO_COMPRESSION", "20")))

    # `compression` is an ffmpeg -crf value and only applies to mp4; passing it for a gif
    # is harmless but pointless, so only send it where it does something.
    video = format == "mp4" ? (; compression) : (;)

    mkpath(output_dir)
    println("Animating run \"$run_name\" into $output_dir/")

    found = false

    if isfile(surface_file)
        animate_surface(surface_file, joinpath(output_dir, "$(run_name)_surface.$(format)");
            framerate, clim_q=clim_quantile, start_date, video...)
        found = true
    else
        println("\nSkipping surface animation: '$surface_file' not found.")
    end

    if isfile(midlon_file)
        animate_midlon(midlon_file,
            joinpath(output_dir, "$(run_name)_midlon.$(format)"),
            joinpath(output_dir, "$(run_name)_midlon_focus.$(format)");
            framerate, clim_q=clim_quantile, focus_depth, start_date, video...)
        found = true
    else
        println("\nSkipping mid-longitude animations: '$midlon_file' not found.")
    end

    found || error("Neither '$surface_file' nor '$midlon_file' exists — nothing to animate.")
    println("\nAll done.")
    return nothing
end
