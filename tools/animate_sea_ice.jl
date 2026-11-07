# ==============================================================================
# animate_sea_ice.jl
#
# Animates the sea ice of a sowinwarp_nested run, one frame per daily record:
#
#   RUN_sea_ice.mp4   concentration ℵ, volume per area h·ℵ (log scale), ice speed and snow
#                     thickness as maps, over the domain's ice area and volume with a marker
#                     at the frame's day
#
# Reads `RUN_sea_ice.nc` (h, ℵ, hs, u, v, bottom_height), the file
# examples/sowinwarp_nested.jl writes. Companion to animate_simulation.jl, whose helpers
# (calendar captions, colour limits, video settings) it reuses, so the same environment
# variables apply:
#
#     julia --project=. -e 'include("tools/animate_sea_ice.jl"); animate_sea_ice("2Y_nested_seaice_fix4")'
#
# or `./run_animate_simulation RUN`, which makes this movie too when RUN_sea_ice.nc exists.
#
# h·ℵ, not h, is the map of how much ice there is: h is the thickness of the ice-covered
# part and is meaningless where ℵ ≈ 0. Its colour scale is logarithmic, 1 cm to HV_MAX metres
# (default 10), with anything thicker drawn red, so a coastal pile-up stands out against
# ordinary ~1 m pack ice. The largest h·ℵ of each frame is marked with an x and printed.
#
# ENV, beyond animate_simulation.jl's (FRAMERATE, VIDEO_FORMAT, VIDEO_COMPRESSION,
# ANIMATION_DIR, START_DATE):
#     SEA_ICE_FILE   input              (default RUN_sea_ice.nc)
#     HV_MAX         top of the h·ℵ scale, m        (default 10)
#     ICE_SPEED_LIMS ice speed scale, m/s            (default: 99.5th percentile, scanned)
#     SNOW_LIMS      snow thickness scale, m         (default: 99.5th percentile, scanned)
# ==============================================================================

include(joinpath(@__DIR__, "animate_simulation.jl"))   # functions only: no side effects

"`name` from `ds` as Float32 (x, y, time), dropping a singleton z if the writer kept one."
function read_ice_field(ds, name)
    A = ds[name]
    data = ndims(A) == 4 ? A[:, :, 1, :] : A[:, :, :]
    return Float32.(coalesce.(data, NaN32))
end

"Centre a face field on the cell centres along `dim`, if it carries the extra face."
function centre_on_cells(A, n, dim)
    size(A, dim) == n && return A
    size(A, dim) == n + 1 || error("a face field of length $(size(A, dim)) on $n cells")
    return dim == 1 ? 0.5f0 .* (A[1:end-1, :, :] .+ A[2:end, :, :]) :
                      0.5f0 .* (A[:, 1:end-1, :] .+ A[:, 2:end, :])
end

function animate_sea_ice(run_name;
    sea_ice_file=get(ENV, "SEA_ICE_FILE", "$(run_name)_sea_ice.nc"),
    output_dir=get(ENV, "ANIMATION_DIR", "animations"),
    framerate=parse(Int, get(ENV, "FRAMERATE", "8")),
    start_date=nothing,
    format=get(ENV, "VIDEO_FORMAT", "mp4"),
    compression=parse(Int, get(ENV, "VIDEO_COMPRESSION", "20")),
    hv_max=parse(Float64, get(ENV, "HV_MAX", "10")))

    isfile(sea_ice_file) || error("'$sea_ice_file' not found: nothing to animate")
    video = format == "mp4" ? (; compression) : (;)
    mkpath(output_dir)
    output = joinpath(output_dir, "$(run_name)_sea_ice.$(format)")

    println("\nLoading sea ice from: ", sea_ice_file)
    ds = Dataset(sea_ice_file)
    lon = Float64.(ds["λ_caa"][:])
    lat = Float64.(ds["φ_aca"][:])
    times = Float64.(ds["time"][:])
    Nx, Ny, Nt = length(lon), length(lat), length(times)
    start_date, date_source = resolve_start_date(ds, start_date)
    report_start_date(start_date, date_source)

    wet = Float64.(coalesce.(ds["bottom_height"][:, :], 0.0)) .< 0
    area_m2 = haskey(ds, "Δx_cca") ?
              Float64.(ds["Δx_cca"][:, :]) .* Float64.(ds["Δy_cca"][:, :]) :
              [(6.371e6 * deg2rad(lon[2] - lon[1]))^2 * cosd(φ) for _ in lon, φ in lat]

    h  = read_ice_field(ds, "h")
    ℵ  = read_ice_field(ds, "ℵ")
    hs = haskey(ds, "hs") ? read_ice_field(ds, "hs") : nothing
    u  = haskey(ds, "u") ? centre_on_cells(read_ice_field(ds, "u"), Nx, 1) : nothing
    v  = haskey(ds, "v") ? centre_on_cells(read_ice_field(ds, "v"), Ny, 2) : nothing
    close(ds)

    # Land is NaN (drawn grey); open water is NaN in every ice field but ℵ, so only the ice shows
    open_water = .!(ℵ .> 0)
    for n in 1:Nt
        @views ℵ[:, :, n][.!wet] .= NaN32
    end
    V = h .* ℵ
    V[open_water .| isnan.(V)] .= NaN32
    ℵ[open_water] .= NaN32
    speed = (u === nothing || v === nothing) ? nothing : sqrt.(u .^ 2 .+ v .^ 2)
    speed === nothing || (speed[open_water .| isnan.(speed)] .= NaN32)
    hs === nothing || (hs[open_water .| isnan.(hs)] .= NaN32)

    # Domain totals for the bottom strip and the per-frame maximum
    total(A, n) = sum(x -> isnan(x[1]) ? 0.0 : Float64(x[1]) * x[2], zip(view(A, :, :, n), area_m2))
    area   = [total(ℵ, n) for n in 1:Nt] ./ 1e12   # 10⁶ km²
    volume = [total(V, n) for n in 1:Nt] ./ 1e9    # km³
    peak_at = [Tuple(argmax(replace(view(V, :, :, n), NaN32 => -Inf32))) for n in 1:Nt]
    peak    = [isnan(V[i, j, n]) ? 0f0 : V[i, j, n] for (n, (i, j)) in enumerate(peak_at)]

    scanned(A, fallback) = (A === nothing || !any(!isnan, A)) ? fallback :
                           (0.0, max(quantile_of(filter(!isnan, vec(A)), 0.995), 1e-3))
    speed_lims, speed_src = resolve_lims("ICE_SPEED_LIMS", scanned(speed, (0.0, 0.5)))
    snow_lims, snow_src   = resolve_lims("SNOW_LIMS", scanned(hs, (0.0, 0.5)))
    report_limits((("Concentration ℵ", "", (0.0, 1.0), "fixed"),
                   ("Volume h·ℵ (log)", "m", (0.01, hv_max), "HV_MAX"),
                   ("Ice speed", "m/s", speed_lims, speed_src),
                   ("Snow thickness", "m", snow_lims, snow_src)), 0.995)
    @printf("  Largest h·ℵ of the run: %.1f m on frame %d at %.1fE %.1fS\n",
            maximum(peak), argmax(peak), lon[peak_at[argmax(peak)][1]], -lat[peak_at[argmax(peak)][2]])

    # ── figure ────────────────────────────────────────────────────────────────
    fig = Figure(size=(1400, 1150), fontsize=14)
    t_idx = Observable(1)
    Label(fig[0, 1:4], @lift(string("Sea Ice: ", time_caption(start_date, times[$t_idx], times[end], $t_idx, Nt))),
          fontsize=22, font=:bold)
    bounds = (minimum(lon), maximum(lon), minimum(lat), maximum(lat))
    land = [wet[i, j] ? NaN : 1.0 for i in 1:Nx, j in 1:Ny]

    function map_panel(row, col, title, data, cmap, lims, unit; scale=identity, mark_peak=false)
        ax = Axis(fig[row, col]; title, xlabel="Longitude (°E)", ylabel="Latitude (°N)", limits=bounds,
                  backgroundcolor=:aliceblue)
        if data === nothing
            text!(ax, (bounds[1] + bounds[2]) / 2, (bounds[3] + bounds[4]) / 2; text="not in file", align=(:center, :center))
            return
        end
        hm = heatmap!(ax, lon, lat, @lift(data[:, :, $t_idx]); colormap=cmap, colorrange=lims,
                      colorscale=scale, nan_color=:transparent,
                      lowclip=clip_lo(cmap), highclip=scale === log10 ? :red : clip_hi(cmap))
        heatmap!(ax, lon, lat, land; colormap=[:gray30, :gray30], colorrange=(0, 2), nan_color=:transparent)
        if mark_peak
            scatter!(ax, @lift(Point2f(lon[peak_at[$t_idx][1]], lat[peak_at[$t_idx][2]]));
                     marker=:xcross, color=:red, markersize=14)
            text!(ax, bounds[1] + 1, bounds[4] - 1; text=@lift(@sprintf("max %.1f m", peak[$t_idx])),
                  align=(:left, :top), fontsize=13)
        end
        Colorbar(fig[row, col+1], hm; label=unit)
    end

    map_panel(1, 1, "Concentration ℵ", ℵ, :ice, (0.0, 1.0), "fraction")
    map_panel(1, 3, "Volume per area h·ℵ (red above $(hv_max) m)", V, :viridis, (0.01, hv_max), "m";
              scale=log10, mark_peak=true)
    map_panel(2, 1, "Ice speed", speed, :speed, speed_lims, "m/s")
    map_panel(2, 3, "Snow thickness (on the ice)", hs, :dense, snow_lims, "m")

    days = times ./ 86400
    ax_a = Axis(fig[3, 1:2]; title="Domain ice area", ylabel="10⁶ km²", xlabel="days since start", height=150)
    lines!(ax_a, days, area; color=:royalblue)
    vlines!(ax_a, @lift([days[$t_idx]]); color=:red)
    ax_v = Axis(fig[3, 3:4]; title="Domain ice volume", ylabel="km³", xlabel="days since start", height=150)
    lines!(ax_v, days, volume; color=:royalblue)
    vlines!(ax_v, @lift([days[$t_idx]]); color=:red)

    println("Recording sea-ice animation to $output (framerate = $framerate fps)...")
    record(fig, output, 1:Nt; framerate, video...) do t
        t_idx[] = t
    end
    println(@sprintf("Done! Saved to %s (%.1f MB)", output, filesize(output) / 1024^2))
    return output
end
