#!/usr/bin/env julia
#
# Hydrographic plots from two CCHDO CTD cruise files in data/hydrographic/.
#
#   09AR20110104_ctd.nc  expocode 09AR1121_1, section SR03 (Aurora Australis, Jan 2011)
#       149 profiles from 44.0°S to 67.7°S along ~140-150°E. A true meridional repeat
#       section south of Tasmania, so it is plotted as a latitude-vs-depth section.
#
#   325020240221_ctd.nc  expocode 325020240221, section I08S (Mar 2024)
#       75 profiles from 28.3°S to 67.1°S along ~76-95°E, in the Indian sector. Also a
#       meridional GO-SHIP section, so it gets the same latitude-vs-depth treatment.
#
#   320620150324_ctd.nc  expocode 320620150324, section NBP1503 (N.B. Palmer, Mar-May 2015)
#       42 profiles, but 41 of them sit between 63.6°S and 65.7°S spread over 117-133°E
#       (a zonal shelf survey off Wilkes/Adelie Land) plus one transit cast near Hobart
#       at 44.0°S. It is NOT a meridional transect, so a latitude-vs-depth section would
#       collapse into a vertical smear. It is plotted as overlaid profiles coloured by
#       longitude instead.
#
# Run with:  julia --project=. src/plot_ctd_sections.jl
# Output:    plots/sr03_section_full.png
#            plots/sr03_section_upper500.png
#            plots/nbp1503_profiles.png

using NCDatasets
using CairoMakie
using Printf
using Dates

const ROOT    = normpath(joinpath(@__DIR__, ".."))
const DATADIR = joinpath(ROOT, "data", "hydrographic")
const OUTDIR  = joinpath(ROOT, "plots")

# WOCE CTD quality codes to keep: 2 = acceptable_measurement,
# 6 = interpolated_over_a_pressure_interval_larger_than_2_dbar.
# SR03 carries these flags; NBP1503 has no *_qc variables at all, so it falls back
# to the range checks below.
const GOOD_QC    = (2, 6)
const T_RANGE    = (-3.0, 40.0)     # degC, gross sanity check
const S_RANGE    = (2.0, 42.0)      # PSS-78, gross sanity check
const MIN_LEVELS = 20               # drop near-empty casts (NBP1503 has one with 6 levels)

# Stations closer together than this in latitude are merged into one section column.
const LAT_TOL = 0.02                # degrees
# Do not paint colour across station gaps wider than this (avoids inventing structure).
const MAX_LAT_GAP = 2.5             # degrees
const NLAT = 400                    # columns in the regular latitude grid
# CTD casts usually stop short of the seafloor. Carry the deepest good value down to
# the bottom by at most this much, so the section does not show white notches that
# are an artefact of cast depth rather than a real data gap.
const SEAFLOOR_FILL = 250.0         # m
# Colour limits are taken from these quantiles rather than the extremes, so that one
# fresh surface cast does not flatten the whole salinity panel. Values outside the
# range are still drawn, via lowclip/highclip.
const CLIM_Q = (0.005, 0.995)

# NBP1503: everything north of this is transit, not part of the shelf survey.
const NBP_SURVEY_LAT_MAX = -55.0

# ----------------------------------------------------------------------------------
# Pressure -> depth, UNESCO (Fofonoff & Millard 1983). Accurate to ~0.1 m, and avoids
# taking on GibbsSeaWater.jl as a dependency just for this one conversion.
# ----------------------------------------------------------------------------------
function depth_from_pressure(p, lat)
    x  = sind(lat)^2
    gr = 9.780318 * (1.0 + (5.2788e-3 + 2.36e-5 * x) * x) + 1.092e-6 * p
    d  = (((-1.82e-15 * p + 2.279e-10) * p - 2.2512e-5) * p + 9.72659) * p
    return d / gr
end

# Linear interpolation with no extrapolation: queries outside the data return NaN.
function interp1(x, y, xq)
    isempty(x) && return NaN
    (xq < first(x) || xq > last(x)) && return NaN
    i = searchsortedfirst(x, xq)
    i == 1 && return y[1]
    i > length(x) && return y[end]
    x[i] == xq && return y[i]
    w = (xq - x[i-1]) / (x[i] - x[i-1])
    return (1 - w) * y[i-1] + w * y[i]
end

nanmean(v) = begin
    s = 0.0; n = 0
    for x in v
        isfinite(x) && (s += x; n += 1)
    end
    n == 0 ? NaN : s / n
end

# ----------------------------------------------------------------------------------
# Reading
# ----------------------------------------------------------------------------------
struct CTDProfile
    lat::Float64
    lon::Float64
    time::DateTime
    btm_depth::Float64      # NaN where the file has no btm_depth variable
    z::Vector{Float64}      # depth (m, positive down), ascending
    T::Vector{Float64}      # in-situ temperature, degC (ITS-90)
    S::Vector{Float64}      # practical salinity, PSS-78
end

"""
    read_ctd(path)

Read a CCHDO CF-1.8 CTD file into a vector of `CTDProfile`, applying WOCE quality
flags when the file carries them and range checks always. Returns the profiles
along with the cruise's expocode and section id.
"""
function read_ctd(path)
    ds = NCDataset(path)

    lat  = Float64.(ds["latitude"][:])
    lon  = Float64.(ds["longitude"][:])
    time = ds["time"][:]

    P = ds["pressure"][:, :]
    T = ds["ctd_temperature"][:, :]
    S = ds["ctd_salinity"][:, :]

    # NBP1503 has neither QC flags nor bottom depth; guard both.
    Tq  = haskey(ds, "ctd_temperature_qc") ? ds["ctd_temperature_qc"][:, :] : nothing
    Sq  = haskey(ds, "ctd_salinity_qc")    ? ds["ctd_salinity_qc"][:, :]    : nothing
    btm = haskey(ds, "btm_depth")          ? ds["btm_depth"][:]             : nothing

    expocode = strip(join(ds["expocode"][:, 1]))
    section  = haskey(ds, "section_id") ? strip(join(ds["section_id"][:, 1])) : ""

    close(ds)

    profiles = CTDProfile[]
    n_dropped = 0

    for j in eachindex(lat)
        zz = Float64[]; tt = Float64[]; ss = Float64[]

        for k in axes(P, 1)
            p = P[k, j];  ismissing(p) && continue
            t = T[k, j];  ismissing(t) && continue
            s = S[k, j];  ismissing(s) && continue

            if Tq !== nothing
                tq = Tq[k, j]; sq = Sq[k, j]
                (ismissing(tq) || !(Int(tq) in GOOD_QC)) && continue
                (ismissing(sq) || !(Int(sq) in GOOD_QC)) && continue
            end

            (T_RANGE[1] <= t <= T_RANGE[2]) || continue
            (S_RANGE[1] <= s <= S_RANGE[2]) || continue

            push!(zz, depth_from_pressure(Float64(p), lat[j]))
            push!(tt, Float64(t))
            push!(ss, Float64(s))
        end

        if length(zz) < MIN_LEVELS
            n_dropped += 1
            continue
        end

        o = sortperm(zz)   # CCHDO files are downcast-ordered, but do not rely on it
        b = (btm === nothing || ismissing(btm[j])) ? NaN : Float64(btm[j])

        push!(profiles, CTDProfile(lat[j], lon[j], time[j], b, zz[o], tt[o], ss[o]))
    end

    @printf("%-24s expocode=%-14s section=%-8s %3d profiles kept, %d dropped (<%d good levels)\n",
            basename(path), expocode, section, length(profiles), n_dropped, MIN_LEVELS)

    return profiles, expocode, section
end

# ----------------------------------------------------------------------------------
# Gridding a section
# ----------------------------------------------------------------------------------
"""
    depth_grid(zmax)

Stretched depth grid: 5 m in the upper 300 m where the thermocline and Winter Water
layer live, coarsening to 50 m in the abyss.
"""
function depth_grid(zmax)
    zs = Float64[]
    append!(zs, 0.0:5.0:300.0)
    append!(zs, 310.0:10.0:1000.0)
    append!(zs, 1025.0:25.0:2000.0)
    append!(zs, 2050.0:50.0:(ceil(zmax / 50) * 50))
    return unique(zs)
end

"""
    grid_section(profiles, zg, latg)

Interpolate each profile onto the common depth grid `zg`, merge stations that share a
latitude, then interpolate across stations onto the regular latitude grid `latg`.
Cells further than `MAX_LAT_GAP` from a real station are left as NaN so that wide
station gaps stay visibly blank rather than being filled with invented structure.
"""
function grid_section(profiles, zg, latg)
    # 1. each profile onto the common depth grid
    ns = length(profiles)
    Tp = fill(NaN, length(zg), ns)
    Sp = fill(NaN, length(zg), ns)
    for (j, pr) in enumerate(profiles), (k, z) in enumerate(zg)
        Tp[k, j] = interp1(pr.z, pr.T, z)
        Sp[k, j] = interp1(pr.z, pr.S, z)
    end

    # 2. sort by latitude and merge stations at (effectively) the same latitude.
    #    SR03 repeats a few latitudes, which would otherwise break the x-axis.
    ord  = sortperm([pr.lat for pr in profiles])
    slat = Float64[]
    Tc   = Vector{Vector{Float64}}()
    Sc   = Vector{Vector{Float64}}()
    Bc   = Float64[]

    for j in ord
        lj = profiles[j].lat
        if !isempty(slat) && abs(lj - slat[end]) < LAT_TOL
            Tc[end] = [nanmean((Tc[end][k], Tp[k, j])) for k in eachindex(zg)]
            Sc[end] = [nanmean((Sc[end][k], Sp[k, j])) for k in eachindex(zg)]
            Bc[end] = nanmean((Bc[end], profiles[j].btm_depth))
        else
            push!(slat, lj)
            push!(Tc, Tp[:, j])
            push!(Sc, Sp[:, j])
            push!(Bc, profiles[j].btm_depth)
        end
    end

    # 3. across stations onto the regular latitude grid, level by level
    Tg = fill(NaN, length(zg), length(latg))
    Sg = fill(NaN, length(zg), length(latg))

    for k in eachindex(zg)
        xs = Float64[]; ts = Float64[]; ss = Float64[]
        for (j, l) in enumerate(slat)
            if isfinite(Tc[j][k]) && isfinite(Sc[j][k])
                push!(xs, l); push!(ts, Tc[j][k]); push!(ss, Sc[j][k])
            end
        end
        length(xs) < 2 && continue
        for (i, lq) in enumerate(latg)
            # blank out anything too far from an actual station at this depth
            minimum(abs.(xs .- lq)) > MAX_LAT_GAP && continue
            Tg[k, i] = interp1(xs, ts, lq)
            Sg[k, i] = interp1(xs, ss, lq)
        end
    end

    # bottom depth along the regular grid, for the seafloor mask
    bg = fill(NaN, length(latg))
    xs = [slat[j] for j in eachindex(slat) if isfinite(Bc[j])]
    bs = [Bc[j]   for j in eachindex(slat) if isfinite(Bc[j])]
    if length(xs) >= 2
        for (i, lq) in enumerate(latg)
            bg[i] = interp1(xs, bs, lq)
        end
    end

    # 4. reconcile the data with the seafloor.
    #    (a) blank anything interpolated below the bottom, which otherwise shows
    #        through the seafloor shading;
    #    (b) carry the deepest good value down to the bottom over a short distance,
    #        to close the white notch left by a cast that stopped above the seabed.
    for i in eachindex(latg)
        isfinite(bg[i]) || continue

        deepest = 0
        for k in eachindex(zg)
            if zg[k] > bg[i]
                Tg[k, i] = NaN
                Sg[k, i] = NaN
            elseif isfinite(Tg[k, i])
                deepest = k
            end
        end

        deepest == 0 && continue
        for k in (deepest+1):length(zg)
            zg[k] > bg[i] && break
            zg[k] - zg[deepest] > SEAFLOOR_FILL && break
            Tg[k, i] = Tg[deepest, i]
            Sg[k, i] = Sg[deepest, i]
        end
    end

    return Tg, Sg, slat, bg
end

"""
    robust_limits(F)

Colour limits from the `CLIM_Q` quantiles of the finite data, so that a single
outlying cast does not compress the colour scale for everything else.
"""
function robust_limits(F)
    v = sort!(filter(isfinite, vec(F)))
    isempty(v) && return (0.0, 1.0)
    n  = length(v)
    lo = v[clamp(round(Int, CLIM_Q[1] * n), 1, n)]
    hi = v[clamp(round(Int, CLIM_Q[2] * n), 1, n)]
    return lo == hi ? (lo - 0.5, hi + 0.5) : (lo, hi)
end

"""
    nice_levels(lo, hi; target = 10)

Contour levels on a 1/2/2.5/5 x 10^n step, giving roughly `target` lines across the
range whatever the field and depth window.
"""
function nice_levels(lo, hi; target = 10)
    span = hi - lo
    span <= 0 && return Float64[]
    raw  = span / target
    mag  = 10.0^floor(log10(raw))
    step = mag * argmin(x -> abs(log(x * mag / raw)), (1.0, 2.0, 2.5, 5.0, 10.0))
    return collect(ceil(lo / step) * step : step : hi)
end

"Format a latitude tick as e.g. 65°S."
latlabel(x) = string(abs(round(Int, x)), "°", x < 0 ? "S" : "N")

# ----------------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------------
"""
    plot_section(profiles, expocode, section, zmax, outfile)

Two-panel latitude-vs-depth section of in-situ temperature and practical salinity,
with station positions ticked along the top and the seafloor shaded where the file
reports bottom depth.
"""
function plot_section(profiles, expocode, section, zmax, outfile)
    zg   = depth_grid(zmax)
    lats = [pr.lat for pr in profiles]
    latg = collect(range(minimum(lats), maximum(lats); length = NLAT))

    Tg, Sg, slat, bg = grid_section(profiles, zg, latg)

    keep = zg .<= zmax
    zg   = zg[keep]
    Tg   = Tg[keep, :]
    Sg   = Sg[keep, :]

    t0 = minimum(pr.time for pr in profiles)
    t1 = maximum(pr.time for pr in profiles)
    when = Dates.format(t0, "u yyyy") == Dates.format(t1, "u yyyy") ?
           Dates.format(t0, "u yyyy") :
           string(Dates.format(t0, "u yyyy"), " - ", Dates.format(t1, "u yyyy"))

    fig = Figure(size = (1100, 900), fontsize = 14)

    Label(fig[0, 1:2],
          "$section ($expocode), $when  -  $(length(profiles)) CTD stations, "
          * @sprintf("%.1f-%.1f°E", minimum(pr.lon for pr in profiles),
                                    maximum(pr.lon for pr in profiles));
          fontsize = 17, font = :bold, padding = (0, 0, 8, 0))

    panels = ((1, Tg, "Temperature", "°C",     :thermal),
              (2, Sg, "Salinity",    "PSS-78", :haline))

    for (row, F, name, unit, cmap) in panels
        ax = Axis(fig[row, 1];
                  xlabel = row == 2 ? "Latitude" : "",
                  ylabel = "Depth (m)",
                  title  = "In-situ $(lowercase(name))",
                  yreversed = true,
                  xtickformat = xs -> latlabel.(xs),
                  xticklabelsvisible = row == 2)

        lo, hi = robust_limits(F)

        hm = heatmap!(ax, latg, zg, permutedims(F);
                      colormap = cmap, colorrange = (lo, hi),
                      lowclip = cgrad(cmap)[0.0], highclip = cgrad(cmap)[1.0])

        # a few contour lines to make water-mass boundaries readable
        levels = nice_levels(lo, hi)
        if length(levels) > 1
            contour!(ax, latg, zg, permutedims(F);
                     levels = levels, color = (:black, 0.35), linewidth = 0.6)
        end

        # seafloor, where the file reports it. Opaque, so nothing shows through.
        if any(isfinite, bg)
            b = [isfinite(x) ? min(x, zmax) : zmax for x in bg]
            band!(ax, latg, b, fill(zmax, length(latg)); color = :grey25)
            lines!(ax, latg, b; color = :black, linewidth = 1.0)
        end

        # station positions
        scatter!(ax, slat, fill(zmax * 0.012, length(slat));
                 marker = :dtriangle, markersize = 6, color = :black)

        xlims!(ax, minimum(latg), maximum(latg))
        ylims!(ax, zmax, 0)

        Colorbar(fig[row, 2], hm; label = "$name ($unit)", width = 14,
                 ticks = levels)
    end

    rowgap!(fig.layout, 12)
    mkpath(dirname(outfile))
    save(outfile, fig)
    println("  wrote ", relpath(outfile, ROOT))
    return fig
end

"""
    plot_profiles(profiles, expocode, section, outfile; lat_max)

Overlaid temperature and salinity profiles coloured by longitude. Suited to a station
cluster like NBP1503, where the stations spread zonally rather than along a transect.
"""
function plot_profiles(profiles, expocode, section, outfile; lat_max = NBP_SURVEY_LAT_MAX)
    survey  = filter(pr -> pr.lat <= lat_max, profiles)
    transit = filter(pr -> pr.lat >  lat_max, profiles)

    if !isempty(transit)
        @printf("  excluding %d transit cast(s) north of %.0f°S: %s\n",
                length(transit), -lat_max,
                join([@sprintf("%.2f°S/%.2f°E", -pr.lat, pr.lon) for pr in transit], ", "))
    end

    lons = [pr.lon for pr in survey]
    lo, hi = minimum(lons), maximum(lons)
    cmap = cgrad(:viridis)

    zmax = maximum(maximum(pr.z) for pr in survey)

    t0 = minimum(pr.time for pr in survey)
    t1 = maximum(pr.time for pr in survey)

    fig = Figure(size = (1000, 780), fontsize = 14)

    Label(fig[0, 1:3],
          "$section ($expocode), "
          * Dates.format(t0, "d u") * " - " * Dates.format(t1, "d u yyyy")
          * @sprintf("  -  %d stations, %.1f-%.1f°S, %.1f-%.1f°E",
                     length(survey),
                     -maximum(pr.lat for pr in survey), -minimum(pr.lat for pr in survey),
                     lo, hi);
          fontsize = 17, font = :bold, padding = (0, 0, 8, 0))

    axT = Axis(fig[1, 1]; xlabel = "In-situ temperature (°C)", ylabel = "Depth (m)",
               title = "Temperature", yreversed = true)
    axS = Axis(fig[1, 2]; xlabel = "Practical salinity (PSS-78)",
               title = "Salinity", yreversed = true)

    for pr in survey
        c = cmap[hi > lo ? (pr.lon - lo) / (hi - lo) : 0.5]
        lines!(axT, pr.T, pr.z; color = c, linewidth = 1.1)
        lines!(axS, pr.S, pr.z; color = c, linewidth = 1.1)
    end

    for ax in (axT, axS)
        ylims!(ax, zmax, 0)
    end
    linkyaxes!(axT, axS)
    hideydecorations!(axS; grid = false)

    Colorbar(fig[1, 3]; colormap = cmap, limits = (lo, hi),
             label = "Longitude (°E)", width = 14)

    mkpath(dirname(outfile))
    save(outfile, fig)
    println("  wrote ", relpath(outfile, ROOT))
    return fig
end

# ----------------------------------------------------------------------------------
"""
Each cruise, with how it should be drawn:

  :section  - a genuine meridional transect, drawn as latitude vs depth
  :profiles - a station cluster, drawn as overlaid profiles coloured by longitude
"""
const CRUISES = (
    (file = "09AR20110104_ctd.nc", stem = "sr03",    mode = :section),
    (file = "325020240221_ctd.nc", stem = "i08s",    mode = :section),
    (file = "320620150324_ctd.nc", stem = "nbp1503", mode = :profiles),
)

function main()
    CairoMakie.activate!(type = "png", px_per_unit = 2)

    written = String[]

    for c in CRUISES
        profiles, expocode, section = read_ctd(joinpath(DATADIR, c.file))

        if c.mode === :section
            zmax = ceil(maximum(maximum(pr.z) for pr in profiles) / 100) * 100
            for (tag, zlim) in (("full", zmax), ("upper500", 500.0))
                out = joinpath(OUTDIR, "$(c.stem)_section_$(tag).png")
                plot_section(profiles, expocode, section, zlim, out)
                push!(written, out)
            end
        else
            out = joinpath(OUTDIR, "$(c.stem)_profiles.png")
            plot_profiles(profiles, expocode, section, out)
            push!(written, out)
        end
    end

    println("\n", "="^72)
    println("DONE - $(length(written)) figure(s) written to $(OUTDIR)")
    println("="^72)
    for f in written
        sz = isfile(f) ? Base.format_bytes(filesize(f)) : "MISSING"
        @printf("  %-34s  %10s\n", basename(f), sz)
    end
    println()

    return written
end

# Run on both `julia src/plot_ctd_sections.jl` and `include(...)` from the REPL.
# Set ENV["CTD_PLOTS_NORUN"] = "1" beforehand if you only want the functions loaded.
if !haskey(ENV, "CTD_PLOTS_NORUN")
    main()
end
