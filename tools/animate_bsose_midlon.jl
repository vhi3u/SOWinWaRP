# ==============================================================================
# animate_bsose_midlon.jl
#
# Animates a BSOSE transect at the model's mid-longitude, so the result can be put
# next to `RUN_midlon.mp4` and `RUN_midlon_focus.mp4` from animate_simulation.jl:
# the same four panels (T, S, u, v) against latitude and depth, once for the full
# water column and once for the upper FOCUS_DEPTH metres.
#
#     julia --project=. -e 'include("tools/animate_bsose_midlon.jl")'
#
# runs top to bottom and writes
#
#     animations/bsose_i156_midlon_201406_to_201605.mp4
#     animations/bsose_i156_midlon_201406_to_201605_focus.mp4
#
# Configuration is through the environment; bsose_animation_common.jl lists it.
# Two more settings apply here:
#
#     MID_LON            longitude of the transect, °E. Unset, it is the column
#                        src/model.jl writes: the middle of the domain.
#     FOCUS_DEPTH        depth of the upper-ocean animation, metres (500)
# ==============================================================================

# The configuration, the open BSOSE files, the domain subset and the records to
# animate.
include(joinpath(@__DIR__, "bsose_animation_common.jl"))

const FOCUS_DEPTH = parse(Float64, get(ENV, "FOCUS_DEPTH", "500"))

# ── Reading ───────────────────────────────────────────────────────────────────

# A face in rock has no flow, so a NaN there is a zero and not a hole: left as NaN it
# would blank the wet cell beside it when the faces are averaged onto cell centres.
read_face(ds, var, I...) = replace!(read2d(ds, var, I...), NaN32 => 0.0f0)

"""
    column_center_on_x(ds, var, i, iy, t, Nx)

`var` over the latitudes `iy` and every depth, averaged from the western faces of
column `i` and its eastern neighbour onto the column's tracer cells. The wrap at
the last column is the one `center_on_x` in animate_bsose_surface.jl makes.
"""
function column_center_on_x(ds, var, i, iy, t, Nx)
    i_next = i == Nx ? 1 : i + 1
    return 0.5f0 .* (read_face(ds, var, i, iy, :, t) .+ read_face(ds, var, i_next, iy, :, t))
end

"""
    column_center_on_y(ds, var, i, iy, t, Ny)

The same in latitude, from the southern faces of each row and the row above. As
in `center_on_y`, a subset reaching the last row keeps that row's face value.
"""
function column_center_on_y(ds, var, i, iy, t, Ny)
    inner = read_face(ds, var, i, iy, :, t)
    j_next = min(last(iy) + 1, Ny)
    outer = read_face(ds, var, i, j_next:j_next, :, t)
    return 0.5f0 .* (inner .+ vcat(inner[2:end, :], outer))
end

# ── Load ──────────────────────────────────────────────────────────────────────

# src/model.jl writes column div(Nx, 2) of its own grid.
i_mid = if haskey(ENV, "MID_LON")
    target = parse(Float64, ENV["MID_LON"])
    LON_BOUNDS[1] <= target <= LON_BOUNDS[2] || error("MID_LON ($target) is outside LON_BOUNDS $(LON_BOUNDS).")
    ix[argmin(abs.(lon .- target))]
else
    ix[div(Nx, 2)]
end
mid_lon = xc[i_mid]
Nz = length(z)

# Land mask, as in animate_bsose_surface.jl: hFac == 0 exactly, never hFac < 1. The
# fallback is zero salinity, the one thing no ocean cell ever has.
mask = if haskey(ds_T, "hFacC")
    Float32.(coalesce.(ds_T["hFacC"][i_mid, iy, :], 0.0f0)) .== 0
else
    @warn "No hFacC in $(basename(theta_path)); masking land where salinity is zero."
    S_first = read2d(ds_S, "SALT", i_mid, iy, :, first(it_S))
    (S_first .== 0) .| isnan.(S_first)
end

# BSOSE's Depth is positive and zero on land, so -Depth sits at the surface there.
bottom_h = haskey(ds_T, "Depth") ? -Float64.(coalesce.(ds_T["Depth"][i_mid, iy], 0.0)) : fill(-5000.0, Ny)

println(@sprintf("  - Slice  : %.2f°E (column %d)", mid_lon, i_mid))
println(@sprintf("  - Depth  : %.1f m to %.1f m (%d levels)", minimum(z), maximum(z), Nz))
println(@sprintf("  - Land   : %d of %d transect cells (%.1f%%)",
    count(mask), length(mask), 100 * count(mask) / length(mask)))

T_data = Array{Float32}(undef, Ny, Nz, Nt)
S_data = Array{Float32}(undef, Ny, Nz, Nt)
u_data = Array{Float32}(undef, Ny, Nz, Nt)
v_data = Array{Float32}(undef, Ny, Nz, Nt)

println("Extracting monthly transects...")
for (n, m) in enumerate(months)
    Tt = read2d(ds_T, "THETA", i_mid, iy, :, it_T[n])
    St = read2d(ds_S, "SALT", i_mid, iy, :, it_S[n])
    uc = column_center_on_x(ds_U, "UVEL", i_mid, iy, it_U[n], Nx_full)
    vc = column_center_on_y(ds_V, "VVEL", i_mid, iy, it_V[n], Ny_full)

    for f in (Tt, St, uc, vc)
        f[mask] .= NaN32
    end

    T_data[:, :, n] = Tt
    S_data[:, :, n] = St
    u_data[:, :, n] = uc
    v_data[:, :, n] = vc

    println(@sprintf("  %2d/%2d  %s", n, Nt, Dates.format(m, "yyyy-mm")))
end

close(ds_T)
close(ds_S)
close(ds_U)
close(ds_V)

# ── Animate ───────────────────────────────────────────────────────────────────

# One animation per depth range, as in animate_midlon: the full water column, then
# the upper ocean with its limits rescanned over just that band.
function transect_animation(out, depth_mask, z_bounds, tag)
    valid(d) = filter(!isnan, d[:, depth_mask, :])
    valid_T, valid_S = valid(T_data), valid(S_data)
    valid_u, valid_v = valid(u_data), valid(v_data)

    T_lims, T_src = resolve_lims("T_LIMS", scan_lims(valid_T, (-2.0, 15.0); clim_q=CLIM_QUANTILE))
    S_lims, S_src = resolve_lims("S_LIMS", scan_lims(valid_S, (33.0, 35.5); clim_q=CLIM_QUANTILE))
    u_lims, u_src = resolve_lims("U_LIMS", scan_sym_lims(valid_u, (-0.5, 0.5); clim_q=CLIM_QUANTILE))
    v_lims, v_src = resolve_lims("V_LIMS", scan_sym_lims(valid_v, (-0.3, 0.3); clim_q=CLIM_QUANTILE))

    println("\n$tag:")
    report_limits((("Temperature (T)", "°C", T_lims, T_src),
            ("Salinity (S)", "PSU", S_lims, S_src),
            ("Zonal vel (u)", "m/s", u_lims, u_src),
            ("Merid vel (v)", "m/s", v_lims, v_src)), CLIM_QUANTILE)
    report_speeds(valid_u, valid_v, false)

    fig = Figure(size=(1400, 950), fontsize=14)
    t_idx = Observable(1)

    # The month and year of the record on screen, which is the month BSOSE averaged
    # rather than the date the record is stamped with.
    time_str = @lift(@sprintf("BSOSE i%d Mid-Longitude (λ = %.1f°E, %s): %s (Month %d / %d)",
        BSOSE_ITERATION, mid_lon, tag, Dates.format(months[$t_idx], "U yyyy"), $t_idx, Nt))
    Label(fig[0, 1:4], time_str, fontsize=22, font=:bold)

    bounds = (minimum(lat), maximum(lat), z_bounds[1], z_bounds[2])
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

    println("Recording $tag animation to $out (framerate = $FRAMERATE fps)...")
    record(fig, out, 1:Nt; framerate=FRAMERATE, video...) do t
        t_idx[] = t
    end
    println(@sprintf("Done! Saved to %s (%.1f MB)", out, filesize(out) / 1024^2))
end

output = animation_output("midlon")
stem, ext = splitext(output)

transect_animation(output, trues(Nz), (minimum(z), 0.0), "full depth")
transect_animation("$(stem)_focus$(ext)", z .>= -FOCUS_DEPTH, (-FOCUS_DEPTH, 0.0),
    @sprintf("upper %.0f m", FOCUS_DEPTH))
