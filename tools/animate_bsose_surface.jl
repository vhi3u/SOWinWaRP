# ==============================================================================
# animate_bsose_surface.jl
#
# Animates BSOSE's own surface fields over the model's domain and period, so the
# result can be put next to `RUN_surface.mp4` from animate_simulation.jl and read
# as the same picture: the same four panels (T, S, u, v), the same colormaps, the
# same colour-limit rules.
#
#     julia --project=. -e 'include("tools/animate_bsose_surface.jl")'
#
# runs top to bottom like src/model.jl and writes
#
#     animations/bsose_i156_surface_201406_to_201605.mp4
#
# Configuration is through the environment; bsose_animation_common.jl lists it.
# ==============================================================================

# The configuration, the open BSOSE files, the domain subset and the records to
# animate.
include(joinpath(@__DIR__, "bsose_animation_common.jl"))

# ── Reading ───────────────────────────────────────────────────────────────────

"""
    center_on_x(ds, var, ix, iy, k, t, Nx)

`var` read over `ix × iy` and averaged onto tracer cell centres in longitude.
MITgcm puts `UVEL[i]` on the WESTERN face of tracer cell i, so centring cell i
needs face i+1 as well; the grid is periodic in longitude, so the face past the
subset's eastern edge wraps to 1 when the subset reaches the last column.

The face indices are the tracer indices — no `XG` is read. That is deliberate:
iteration 156's grid file has zero-padded `XG`/`YG` arrays, and index arithmetic
on the C-grid does not need the face coordinates at all.
"""
function center_on_x(ds, var, ix, iy, k, t, Nx)
    inner = read2d(ds, var, ix, iy, k, t)
    i_next = last(ix) == Nx ? 1 : last(ix) + 1
    outer = read2d(ds, var, i_next:i_next, iy, k, t)
    return 0.5f0 .* (inner .+ vcat(inner[2:end, :], outer))
end

"""
    center_on_y(ds, var, ix, iy, k, t, Ny)

The same in latitude, where `VVEL[j]` sits on the southern face of cell j. There
is no wrap in latitude; if the subset reaches the last row, that row keeps its
face value rather than being averaged with a row that does not exist.
"""
function center_on_y(ds, var, ix, iy, k, t, Ny)
    inner = read2d(ds, var, ix, iy, k, t)
    j_next = min(last(iy) + 1, Ny)
    outer = read2d(ds, var, ix, j_next:j_next, k, t)
    return 0.5f0 .* (inner .+ hcat(inner[:, 2:end], outer))
end

# ── Load ──────────────────────────────────────────────────────────────────────

# BSOSE indexes the vertical surface-to-bottom, so the surface is the level whose
# centre is closest to z = 0 — index 1, but found rather than assumed.
k_surface = argmin(abs.(z))

println(@sprintf("  - Surface: z = %.2f m (level %d of %d)", z[k_surface], k_surface, length(z)))

# Land mask. MITgcm's hFac is the wet FRACTION of a cell and BSOSE uses partial
# cells heavily, so land is hFac == 0 exactly, never hFac < 1. If the file carries
# no hFac, fall back to zero salinity, which no ocean cell ever has — a zero
# velocity or a zero temperature is real data and must not be read as land.
mask = if haskey(ds_T, "hFacC")
    Float32.(coalesce.(ds_T["hFacC"][ix, iy, k_surface], 0.0f0)) .== 0
else
    @warn "No hFacC in $(basename(theta_path)); masking land where surface salinity is zero."
    S_first = read2d(ds_S, "SALT", ix, iy, k_surface, first(it_S))
    (S_first .== 0) .| isnan.(S_first)
end
println(@sprintf("  - Land   : %d of %d surface cells (%.1f%%)",
    count(mask), length(mask), 100 * count(mask) / length(mask)))

T_data = Array{Float32}(undef, Nx, Ny, Nt)
S_data = Array{Float32}(undef, Nx, Ny, Nt)
u_data = Array{Float32}(undef, Nx, Ny, Nt)
v_data = Array{Float32}(undef, Nx, Ny, Nt)

println("Extracting monthly surface fields...")
for (n, m) in enumerate(months)
    Tt = read2d(ds_T, "THETA", ix, iy, k_surface, it_T[n])
    St = read2d(ds_S, "SALT", ix, iy, k_surface, it_S[n])
    uc = center_on_x(ds_U, "UVEL", ix, iy, k_surface, it_U[n], Nx_full)
    vc = center_on_y(ds_V, "VVEL", ix, iy, k_surface, it_V[n], Ny_full)

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

# ── Colour limits: scanned from BSOSE ─────────────────────────────────────────

valid_T = filter(!isnan, T_data)
valid_S = filter(!isnan, S_data)
valid_u = filter(!isnan, u_data)
valid_v = filter(!isnan, v_data)

T_lims, T_src = resolve_lims("T_LIMS", scan_lims(valid_T, (-2.0, 20.0); clim_q=CLIM_QUANTILE))
S_lims, S_src = resolve_lims("S_LIMS", scan_lims(valid_S, (32.5, 35.5); clim_q=CLIM_QUANTILE))
u_lims, u_src = resolve_lims("U_LIMS", scan_sym_lims(valid_u, (-0.5, 0.5); clim_q=CLIM_QUANTILE))
v_lims, v_src = resolve_lims("V_LIMS", scan_sym_lims(valid_v, (-0.3, 0.3); clim_q=CLIM_QUANTILE))

report_limits((("Temperature (T)", "°C", T_lims, T_src),
        ("Salinity (S)", "PSU", S_lims, S_src),
        ("Zonal vel (u)", "m/s", u_lims, u_src),
        ("Merid vel (v)", "m/s", v_lims, v_src)), CLIM_QUANTILE)

report_speeds(valid_u, valid_v, false)

# ── Animate ───────────────────────────────────────────────────────────────────

output = animation_output("surface")

fig = Figure(size=(1400, 950), fontsize=14)
t_idx = Observable(1)

# The month and year of the record on screen, which is the month BSOSE averaged
# rather than the date the record is stamped with.
time_str = @lift(@sprintf("BSOSE i%d Surface: %s (Month %d / %d)",
    BSOSE_ITERATION, Dates.format(months[$t_idx], "U yyyy"), $t_idx, Nt))
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

println("Recording surface animation to $output (framerate = $FRAMERATE fps)...")
record(fig, output, 1:Nt; framerate=FRAMERATE, video...) do t
    t_idx[] = t
end
println(@sprintf("Done! Saved to %s (%.1f MB)", output, filesize(output) / 1024^2))
