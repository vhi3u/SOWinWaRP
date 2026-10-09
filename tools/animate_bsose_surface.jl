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
# Everything is configurable through the environment; the defaults are the domain
# and the window src/model.jl is currently set to (90–150°E, 70–45°S, 2014-06-01
# to 2016-06-01):
#
#     BSOSE_DIR          directory holding the BSOSE NetCDF files
#     BSOSE_ITERATION    156 (2013–2024, 1/6°) or 105 (2008–2012, 1/3°)
#     START_DATE         "2014-06-01"
#     END_DATE           "2016-06-01"
#     LON_BOUNDS         "90,150"
#     LAT_BOUNDS         "-70,-45"
#     ANIMATION_OUTPUT   full output path, overriding the generated name
#     ANIMATION_DIR      directory for the generated name ("animations")
#     FRAMERATE          frames per second (4 — one frame is a whole month)
#     VIDEO_FORMAT       "mp4" or "gif"
#     VIDEO_COMPRESSION  ffmpeg -crf, mp4 only (20)
#     T_LIMS S_LIMS U_LIMS V_LIMS
#                        colour limits, read exactly as animate_simulation.jl
#                        reads them. Unset, the panels use the model animation's
#                        limits pinned below rather than BSOSE's own range.
#     CLIM_QUANTILE      affects only the BSOSE range reported for comparison;
#                        the panels themselves use the pinned limits.
#
# Iteration 156's monthly Theta/Salt/Uvel/Vvel files live on the cluster, not in
# the repo's data/, so this normally runs on a CPU node with BSOSE_DIR pointing at
# the scratch data directory.
# ==============================================================================

using Dates

# env_lims, resolve_lims, scan_lims, scan_sym_lims, clip_lo/clip_hi, report_limits
# and report_speeds, plus NCDatasets/CairoMakie/Printf. The file defines functions
# and runs nothing, so including it has no side effects.
include(joinpath(@__DIR__, "animate_simulation.jl"))

# ── Configuration ─────────────────────────────────────────────────────────────

"A `lo,hi` pair from `ENV[key]`, or `fallback` if it is unset or blank."
function env_pair(key, fallback)
    haskey(ENV, key) || return fallback
    raw = strip(ENV[key])
    isempty(raw) && return fallback
    parts = [parse(Float64, strip(p)) for p in split(raw, ',') if !isempty(strip(p))]
    length(parts) == 2 || error("$key must be a lo,hi pair; got \"$raw\"")
    parts[1] < parts[2] || error("$key must be increasing; got \"$raw\"")
    return (parts[1], parts[2])
end

const BSOSE_ITERATION = parse(Int, get(ENV, "BSOSE_ITERATION", "156"))
const START_DATE = DateTime(get(ENV, "START_DATE", "2014-06-01"))
const END_DATE = DateTime(get(ENV, "END_DATE", "2016-06-01"))
const LON_BOUNDS = env_pair("LON_BOUNDS", (90.0, 150.0))
const LAT_BOUNDS = env_pair("LAT_BOUNDS", (-70.0, -45.0))
const FRAMERATE = parse(Int, get(ENV, "FRAMERATE", "4"))
const CLIM_QUANTILE = parse(Float64, get(ENV, "CLIM_QUANTILE", "1.0"))
const VIDEO_FORMAT = get(ENV, "VIDEO_FORMAT", "mp4")
const VIDEO_COMPRESSION = parse(Int, get(ENV, "VIDEO_COMPRESSION", "20"))

END_DATE > START_DATE || error("END_DATE ($END_DATE) must be later than START_DATE ($START_DATE).")

# ── Where the data is ─────────────────────────────────────────────────────────

"""
    bsose_directory()

The directory holding the BSOSE files: `BSOSE_DIR` if it is set and exists, the
cluster's scratch data directory if that exists, otherwise the repo's `data/`.
Mirrors `default_bsose_directory` in src/setup_bsose.jl, without pulling in
Oceananigans and CUDA for a plotting script.
"""
function bsose_directory()
    haskey(ENV, "BSOSE_DIR") && isdir(ENV["BSOSE_DIR"]) && return abspath(ENV["BSOSE_DIR"])
    isdir("/storage/scratch1/1/vnguyen480/SOWinWaRP/data") && return "/storage/scratch1/1/vnguyen480/SOWinWaRP/data"
    repo_data = normpath(joinpath(@__DIR__, "..", "data"))
    isdir(repo_data) && return repo_data
    return abspath("data")
end

"""
    bsose_file(dir, shortname, iteration)

Path to the monthly file holding `shortname` ("Theta", "Salt", "Uvel", "Vvel") for
`iteration`. Tries the two products' naming conventions, then falls back to a
directory search for a file naming both the variable and the iteration — without
that preference a directory holding both iterations would quietly serve 1/3°
iteration-105 data to an iteration-156 request.
"""
function bsose_file(dir, shortname, iteration)
    cands = iteration == 156 ?
            ["$(shortname)_bsoseI156_2013to2024_monthly.nc",
             "bsose_i156_2013to2024_monthly_$(shortname).nc"] :
            ["bsose_i105_2008to2012_monthly_$(shortname).nc",
             "$(shortname)_bsoseI105_2008to2012_monthly.nc"]

    for cand in cands
        isfile(joinpath(dir, cand)) && return joinpath(dir, cand)
    end

    if isdir(dir)
        matches = filter(f -> endswith(f, ".nc") && occursin(lowercase(shortname), lowercase(f)), readdir(dir))
        for token in ("i$(iteration)", "$(iteration)")
            for f in matches
                occursin(token, lowercase(f)) && return joinpath(dir, f)
            end
        end
    end

    error("""No BSOSE iteration-$(iteration) $(shortname) file found in $dir.
             Expected one of: $(join(cands, ", ")).
             Set BSOSE_DIR to the directory holding the monthly BSOSE NetCDF files.""")
end

# ── Monthly records ───────────────────────────────────────────────────────────
#
# A BSOSE monthly record is stamped at the END of the month it averages, to within
# about a day: the stamps sit at uniform year/12 intervals and so drift either side
# of the month boundary (the February mean is stamped 03-01T20:00). Rounding the
# stamp to the nearest month boundary recovers the month averaged, which is both
# what the frame should be labelled with and what to test when choosing records —
# testing the stamps themselves would shift every label a month late.
# See `bsose_averaging_window` in src/setup_bsose.jl.

"The first instant of the month a BSOSE monthly record averages."
bsose_averaged_month(stamp) = round(DateTime(stamp), Month) - Month(1)

"The middle of that month, which is where the record's value effectively sits."
function bsose_window_center(stamp)
    window_start = bsose_averaged_month(stamp)
    window_stop = window_start + Month(1)
    return window_start + Millisecond(Dates.value(Millisecond(window_stop - window_start)) ÷ 2)
end

"""
    month_index_map(ds)

`month => record index` for every record in `ds`, keyed by the month averaged.
Used to line the Salt/Uvel/Vvel files up with the months chosen from Theta rather
than trusting the four files to be recorded in the same order.
"""
month_index_map(ds) = Dict(bsose_averaged_month(s) => i for (i, s) in enumerate(ds["time"][:]))

"The record in `ds` averaging `month`, or an error naming what is missing."
function record_index(ds, path, month)
    idx = get(month_index_map(ds), month, nothing)
    idx === nothing && error("$(basename(path)) holds no record for $(Dates.format(month, "yyyy-mm")).")
    return idx
end

# ── Reading ───────────────────────────────────────────────────────────────────
#
# BSOSE marks land with its _FillValue of NaN, which NCDatasets hands back as
# `missing`; `coalesce` turns that into the NaN the heatmaps draw as land.

read2d(ds, var, I...) = Float32.(coalesce.(ds[var][I...], NaN32))

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

data_dir = bsose_directory()
println("BSOSE iteration $BSOSE_ITERATION, reading from $data_dir")

theta_path = bsose_file(data_dir, "Theta", BSOSE_ITERATION)
salt_path = bsose_file(data_dir, "Salt", BSOSE_ITERATION)
uvel_path = bsose_file(data_dir, "Uvel", BSOSE_ITERATION)
vvel_path = bsose_file(data_dir, "Vvel", BSOSE_ITERATION)

for p in (theta_path, salt_path, uvel_path, vvel_path)
    println("  ", basename(p))
end

ds_T = Dataset(theta_path)
ds_S = Dataset(salt_path)
ds_U = Dataset(uvel_path)
ds_V = Dataset(vvel_path)

xc = Float64.(coalesce.(ds_T["XC"][:], NaN))
yc = Float64.(coalesce.(ds_T["YC"][:], NaN))
z = Float64.(coalesce.(ds_T["Z"][:], NaN))
Nx_full, Ny_full = length(xc), length(yc)

# Subset to the model domain. BSOSE is circumpolar, so the model's longitudes are
# an interior band; its latitudes are variably spaced (Mercator-like), so the
# number of rows in a 25° band is not 25 × resolution.
ix_all = findall(x -> LON_BOUNDS[1] <= x <= LON_BOUNDS[2], xc)
iy_all = findall(y -> LAT_BOUNDS[1] <= y <= LAT_BOUNDS[2], yc)

isempty(ix_all) && error("No BSOSE columns inside LON_BOUNDS $(LON_BOUNDS); XC spans $(extrema(xc)).")
isempty(iy_all) && error("No BSOSE rows inside LAT_BOUNDS $(LAT_BOUNDS); YC spans $(extrema(yc)).")

ix = first(ix_all):last(ix_all)
iy = first(iy_all):last(iy_all)
lon, lat = xc[ix], yc[iy]
Nx, Ny = length(lon), length(lat)

# BSOSE indexes the vertical surface-to-bottom, so the surface is the level whose
# centre is closest to z = 0 — index 1, but found rather than assumed.
k_surface = argmin(abs.(z))

# Months to animate: every record whose averaging window CENTRE falls inside the
# run, which is the same test src/model.jl uses to choose its forcing records.
stamps = ds_T["time"][:]
frames = findall(s -> START_DATE <= bsose_window_center(s) <= END_DATE, stamps)

if isempty(frames)
    error("""$(basename(theta_path)) holds no monthly record centred between $START_DATE and \
             $END_DATE. Its records average $(Dates.format(bsose_averaged_month(first(stamps)), "yyyy-mm")) \
             to $(Dates.format(bsose_averaged_month(last(stamps)), "yyyy-mm")).""")
end

months = [bsose_averaged_month(stamps[i]) for i in frames]
Nt = length(months)

it_T = frames
it_S = [record_index(ds_S, salt_path, m) for m in months]
it_U = [record_index(ds_U, uvel_path, m) for m in months]
it_V = [record_index(ds_V, vvel_path, m) for m in months]

println(@sprintf("  - Domain : %.2f–%.2f°E, %.2f–%.2f°N  (Nx = %d, Ny = %d)",
    minimum(lon), maximum(lon), minimum(lat), maximum(lat), Nx, Ny))
println(@sprintf("  - Surface: z = %.2f m (level %d of %d)", z[k_surface], k_surface, length(z)))
println(@sprintf("  - Period : %s to %s (%d monthly records)",
    Dates.format(first(months), "yyyy-mm"), Dates.format(last(months), "yyyy-mm"), Nt))

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

# ── Colour limits: pinned to the model animation ──────────────────────────────
#
# TEMPORARY, and the one place to edit. These are the limits the 2YS6_surface_flux
# surface animation scanned for itself, so the two movies can be read side by side
# on one scale instead of each picking its own.
#
# They were read off that movie's colorbars. The velocity ends are exact (its end
# ticks are labelled ±2.5 and ±3); the temperature and salinity ends are
# extrapolated from the tick spacing and are good to about a tenth. The exact
# numbers are printed in that run's animation log, under "Colorbar limits" —
# paste them in here.
#
# Note what pinning costs: the model's velocity range is roughly four times
# BSOSE's (BSOSE peaks near 0.8 m/s here), so the BSOSE u and v panels are nearly
# flat at this scale. That is the comparison rather than a fault in the plot — the
# model's extremes sit far outside anything BSOSE produces in this domain, as does
# a surface temperature below seawater's freezing point. BSOSE's own range is
# printed below for reference.
const MODEL_T_LIMS = (-6.1, 25.1)   # °C
const MODEL_S_LIMS = (32.4, 35.5)   # PSU
const MODEL_U_LIMS = (-2.5, 2.5)    # m/s
const MODEL_V_LIMS = (-3.0, 3.0)    # m/s

"The limits from `ENV[key]` if it is set, otherwise the model animation's."
function pinned_lims(key, model_lims)
    given = env_lims(key)
    return given === nothing ? (model_lims, "matched to the model animation") : (given, "set via $key")
end

valid_T = filter(!isnan, T_data)
valid_S = filter(!isnan, S_data)
valid_u = filter(!isnan, u_data)
valid_v = filter(!isnan, v_data)

T_lims, T_src = pinned_lims("T_LIMS", MODEL_T_LIMS)
S_lims, S_src = pinned_lims("S_LIMS", MODEL_S_LIMS)
u_lims, u_src = pinned_lims("U_LIMS", MODEL_U_LIMS)
v_lims, v_src = pinned_lims("V_LIMS", MODEL_V_LIMS)

report_limits((("Temperature (T)", "°C", T_lims, T_src),
        ("Salinity (S)", "PSU", S_lims, S_src),
        ("Zonal vel (u)", "m/s", u_lims, u_src),
        ("Merid vel (v)", "m/s", v_lims, v_src)), CLIM_QUANTILE)

# What BSOSE on its own would have asked for. Printed, not used: seeing how much
# narrower it is explains at a glance why the pinned panels look washed out.
println("BSOSE's own range over these months (not used for the panels):")
for (name, unit, lims) in (("Temperature (T)", "°C", scan_lims(valid_T, MODEL_T_LIMS; clim_q=CLIM_QUANTILE)),
    ("Salinity (S)", "PSU", scan_lims(valid_S, MODEL_S_LIMS; clim_q=CLIM_QUANTILE)),
    ("Zonal vel (u)", "m/s", scan_sym_lims(valid_u, MODEL_U_LIMS; clim_q=CLIM_QUANTILE)),
    ("Merid vel (v)", "m/s", scan_sym_lims(valid_v, MODEL_V_LIMS; clim_q=CLIM_QUANTILE)))
    @printf("  - %-17s : %8.2f .. %8.2f %s\n", name, lims..., unit)
end
report_speeds(valid_u, valid_v, false)

# ── Animate ───────────────────────────────────────────────────────────────────

output = get(ENV, "ANIMATION_OUTPUT") do
    dir = get(ENV, "ANIMATION_DIR", "animations")
    name = @sprintf("bsose_i%d_surface_%s_to_%s.%s", BSOSE_ITERATION,
        Dates.format(first(months), "yyyymm"), Dates.format(last(months), "yyyymm"), VIDEO_FORMAT)
    joinpath(dir, name)
end
mkpath(dirname(output))

# `compression` is an ffmpeg -crf value and means nothing for a gif.
video = VIDEO_FORMAT == "mp4" ? (; compression=VIDEO_COMPRESSION) : (;)

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
