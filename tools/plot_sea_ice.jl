# ==============================================================================================
# plot_sea_ice.jl -- how the sea ice of a sowinwarp_nested run evolves, and where it piles up.
#
#     julia --project=. tools/plot_sea_ice.jl 2Y_nested_seaice_fix2
#     CELL=97.1,-65.1 MAP_DAYS=200,236,238,261 julia --project=. tools/plot_sea_ice.jl 2Y_nested_seaice_fix2
#
# Reads <RUN>_sea_ice.nc (the run directory, or DIR=...) and writes to plots/ (or OUTDIR=...):
#
#   <RUN>_sea_ice_series.png   domain ice area and volume; the largest h·ℵ and h anywhere, on a log
#                              axis with the run's 50 m stop; and h, ℵ, h·ℵ at one CELL. CELL is
#                              "lon,lat"; by default it is the cell holding the largest h·ℵ of the run.
#   <RUN>_sea_ice_maps.png     h·ℵ maps on six days (MAP_DAYS, by default spread over the record),
#                              log colour scale, land grey, the day's largest h·ℵ marked with an x.
#
# Prints the days on which the largest h·ℵ jumped by more than JUMP metres (default 5) since the
# previous record, with where it was and that cell's h and ℵ: a pile that grows in bursts, as
# fix2's Shackleton corner did, shows up there.
#
# h·ℵ (volume per area), not h, is the measure of how much ice there is: h is the thickness of
# the ice-covered part, so it can be enormous where ℵ is tiny (fix1's 22 m at ℵ = 0.000).
# Daily records only: a jump between two records is the net of a day, not one time step.
# ==============================================================================================

using NCDatasets, CairoMakie, Dates, Printf

RUN    = isempty(ARGS) ? get(ENV, "RUN", "") : ARGS[1]
isempty(RUN) && error("give the run name: julia --project=. tools/plot_sea_ice.jl <RUN>")
DIR    = get(ENV, "DIR", ".")
OUTDIR = get(ENV, "OUTDIR", "plots")
JUMP   = parse(Float64, get(ENV, "JUMP", "5"))
const ICE_MAX_VOLUME = 50   # MUST MATCH examples/sowinwarp_nested.jl

path = joinpath(DIR, "$(RUN)_sea_ice.nc")
isfile(path) || error("$path not found; set DIR to the run directory")

g(A) = Float64.(coalesce.(A, NaN))

ds = NCDataset(path)
λ  = g(ds["λ_caa"][:])
φ  = g(ds["φ_aca"][:])
t  = g(ds["time"][:]) ./ 86400
h  = g(ds["h"][:, :, :])
ℵ  = g(ds["ℵ"][:, :, :])
Az = g(ds["Δx_cca"][:, :]) .* g(ds["Δy_cca"][:, :])
wet = g(ds["bottom_height"][:, :]) .< 0
start = haskey(ds.attrib, "start_date") ? DateTime(ds.attrib["start_date"]) : nothing
close(ds)

Nx, Ny, Nt = size(h)
V = h .* ℵ
for n in 1:Nt                          # land and missing values count as no ice
    @views V[:, :, n][.!wet] .= 0
    @views ℵ[:, :, n][.!wet] .= 0
end
replace!(V, NaN => 0); replace!(ℵ, NaN => 0)

day_label(d) = isnothing(start) ? @sprintf("day %.0f", d) :
                                  @sprintf("day %.0f (%s)", d, Dates.format(start + Second(round(Int, d * 86400)), "yyyy-mm-dd"))
where_is(i, j) = @sprintf("%.1fE %.1fS", λ[i], -φ[j])

# ── domain series ────────────────────────────────────────────────────────────────────────────
area   = [sum(ℵ[:, :, n] .* Az) for n in 1:Nt] ./ 1e12          # 10⁶ km²
volume = [sum(V[:, :, n] .* Az) for n in 1:Nt] ./ 1e9           # km³
imax   = [Tuple(argmax(V[:, :, n])) for n in 1:Nt]
maxV   = [V[i, j, n] for (n, (i, j)) in enumerate(imax)]
hice   = [ℵ[i, j, n] > 0 ? h[i, j, n] : NaN for i in 1:Nx, j in 1:Ny, n in 1:Nt]
maxh   = [maximum(x -> isnan(x) ? -Inf : x, view(hice, :, :, n)) for n in 1:Nt]

# ── the focus cell ───────────────────────────────────────────────────────────────────────────
if haskey(ENV, "CELL")
    lon, lat = parse.(Float64, split(ENV["CELL"], ','))
    ic, jc = argmin(abs.(λ .- lon)), argmin(abs.(φ .- lat))
else
    ic, jc = imax[argmax(maxV)]
end
cell = where_is(ic, jc)
cell_h, cell_ℵ, cell_V = h[ic, jc, :], ℵ[ic, jc, :], V[ic, jc, :]

# ── jumps ────────────────────────────────────────────────────────────────────────────────────
println()
println(@sprintf("%s: %d daily records, days %.0f-%.0f; largest h·ℵ %.1f m on %s at %s",
                 RUN, Nt, first(t), last(t), maximum(maxV), day_label(t[argmax(maxV)]), where_is(imax[argmax(maxV)]...)))
println(@sprintf("Days the largest h·ℵ grew by more than %.0f m since the previous record:", JUMP))
jumps = [n for n in 2:Nt if maxV[n] - maxV[n-1] > JUMP]
isempty(jumps) && println("  none")
for n in jumps
    i, j = imax[n]
    println(@sprintf("  %-26s %6.1f -> %6.1f m at %-14s  there: h %6.1f m, ℵ %.3f (previous record h %6.1f, ℵ %.3f)",
                     day_label(t[n]), maxV[n-1], maxV[n], where_is(i, j), h[i, j, n], ℵ[i, j, n], h[i, j, n-1], ℵ[i, j, n-1]))
end
println()

# ── figure 1: series ─────────────────────────────────────────────────────────────────────────
mkpath(OUTDIR)
positive(x) = [v > 0 ? v : NaN for v in x]

fig = Figure(fontsize = 14)
const SERIES_WIDTH, SERIES_HEIGHT = 850, 190
Label(fig[0, 1], "$RUN sea ice" * (isnothing(start) ? "" : ", day 0 = $(Date(start))"); fontsize = 16, font = :bold)

ax1 = Axis(fig[1, 1]; width = SERIES_WIDTH, height = SERIES_HEIGHT, ylabel = "area (10⁶ km²)", title = "Domain ice area, Σ ℵ·A")
lines!(ax1, t, area; color = :royalblue)

ax2 = Axis(fig[2, 1]; width = SERIES_WIDTH, height = SERIES_HEIGHT, ylabel = "volume (km³)", title = "Domain ice volume, Σ h·ℵ·A")
lines!(ax2, t, volume; color = :royalblue)

ax3 = Axis(fig[3, 1]; width = SERIES_WIDTH, height = SERIES_HEIGHT, ylabel = "m", yscale = log10, title = "Largest anywhere in the domain")
lines!(ax3, t, positive(maxV); color = :firebrick, linewidth = 2, label = "max h·ℵ")
lines!(ax3, t, positive(maxh); color = :grey40, label = "max h (where ℵ > 0)")
hlines!(ax3, [ICE_MAX_VOLUME]; color = :black, linestyle = :dash, label = "run stops (h·ℵ = $(ICE_MAX_VOLUME) m)")
axislegend(ax3; position = :lt, framevisible = false)

ax4 = Axis(fig[4, 1]; width = SERIES_WIDTH, height = SERIES_HEIGHT, ylabel = "m", title = "At $cell (i = $ic, j = $jc)")
lines!(ax4, t, cell_V; color = :firebrick, linewidth = 2, label = "h·ℵ")
lines!(ax4, t, cell_h; color = :grey40, label = "h")
axislegend(ax4; position = :lt, framevisible = false)

ax5 = Axis(fig[5, 1]; width = SERIES_WIDTH, height = SERIES_HEIGHT, ylabel = "ℵ", xlabel = "days since start", title = "Concentration at $cell")
lines!(ax5, t, cell_ℵ; color = :seagreen)
ylims!(ax5, -0.02, 1.02)

linkxaxes!(ax1, ax2, ax3, ax4, ax5)
for ax in (ax1, ax2, ax3, ax4); hidexdecorations!(ax; grid = false); end

resize_to_layout!(fig)
out1 = joinpath(OUTDIR, "$(RUN)_sea_ice_series.png")
save(out1, fig)
@info "wrote $out1"

# ── figure 2: maps ───────────────────────────────────────────────────────────────────────────
map_records = haskey(ENV, "MAP_DAYS") ?
    [argmin(abs.(t .- parse(Float64, d))) for d in split(ENV["MAP_DAYS"], ',')] :
    unique(round.(Int, range(1, Nt; length = 6)))

land = [wet[i, j] ? NaN : 1.0 for i in 1:Nx, j in 1:Ny]
aspect = (λ[end] - λ[1]) * cosd(sum(φ) / length(φ)) / (φ[end] - φ[1])
ncols = min(3, length(map_records))
nrows = cld(length(map_records), ncols)

fig = Figure(size = (420 * ncols + 120, 330 * nrows + 60), fontsize = 13)
Label(fig[0, 1:ncols], "$RUN: sea ice volume per area h·ℵ (x = the day's largest)"; fontsize = 15, font = :bold)
hm = nothing
for (k, n) in enumerate(map_records)
    row, col = fldmod1(k, ncols)
    ax = Axis(fig[row, col]; title = day_label(t[n]), aspect = AxisAspect(aspect),
              xlabel = row == nrows ? "Longitude (°E)" : "", ylabel = col == 1 ? "Latitude (°N)" : "")
    frame = [v > 0 ? v : NaN for v in V[:, :, n]]
    global hm = heatmap!(ax, λ, φ, frame; colormap = :viridis, colorscale = log10, colorrange = (0.01, ICE_MAX_VOLUME),
                         highclip = :red, nan_color = :white)
    heatmap!(ax, λ, φ, land; colormap = [:grey70, :grey70], colorrange = (0, 2), nan_color = :transparent)
    i, j = imax[n]
    maxV[n] > 0 && scatter!(ax, [λ[i]], [φ[j]]; marker = :xcross, color = :red, markersize = 12)
    text!(ax, λ[1] + 1, φ[end] - 1; text = @sprintf("max %.1f m", maxV[n]), align = (:left, :top), fontsize = 11)
    limits!(ax, λ[1], λ[end], φ[1], φ[end])
end
Colorbar(fig[1:nrows, ncols + 1], hm; label = "h·ℵ (m), red above $(ICE_MAX_VOLUME) m")

out2 = joinpath(OUTDIR, "$(RUN)_sea_ice_maps.png")
save(out2, fig)
@info "wrote $out2"
