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
#
# B-SOSE ALONGSIDE (BSOSE_DIR set and its 5-day ice files there): B-SOSE's SIarea and SIheff are
# drawn in black over the domain area and volume (summed over the same lon-lat box) and over the
# CELL panels (B-SOSE's nearest column), and the winter open-water fraction at the cell is
# printed for both, year by year. That is the polynya test: a coast B-SOSE opens in winter and the
# model keeps closed is a coast where the model makes no new ice and so rejects no brine.
# SIheff is MITgcm's effective thickness, already volume per area, so it compares with h·ℵ; B-SOSE's
# h is SIheff / SIarea. 5-day means against daily records: B-SOSE will look smoother.
#
#     BSOSE_DIR=/storage/scratch1/1/vnguyen480/SOWinWaRP/data CELL=145,-66.5 \
#         julia --project=. tools/plot_sea_ice.jl 2Y_nested_seaice_fix5_flux
# ==============================================================================================

using NCDatasets, CairoMakie, Dates, Printf, Statistics

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

# ── B-SOSE alongside (optional) ──────────────────────────────────────────────────────────────
# The 5-day files are one per year, each stamp at the END of the 5 days it averages. Geometry
# (XC, YC, rA, Depth) from the monthly surfTflx file, as tools/ice_production.jl reads it.
const BSOSE_DIR = get(ENV, "BSOSE_DIR", "")
bsose_ice_file(token, y) = joinpath(BSOSE_DIR, "$(token)_bsoseI156_$(y)_5day.nc")

B = let
    if isempty(BSOSE_DIR) || isnothing(start)
        isempty(BSOSE_DIR) || @warn "no start_date in $path: B-SOSE cannot be placed on the run's day axis"
        nothing
    else
        years = year(start):year(start + Second(round(Int, last(t) * 86400)))
        missing_files = [f for y in years for f in (bsose_ice_file("SIArea", y), bsose_ice_file("SIHeff", y)) if !isfile(f)]
        geom = joinpath(BSOSE_DIR, "surfTflx_bsoseI156_2013to2024_monthly.nc")
        if !isempty(missing_files) || !isfile(geom)
            @warn "B-SOSE ice not drawn: missing $(join(basename.([missing_files; isfile(geom) ? String[] : [geom]]), ", "))"
            nothing
        else
            gds = NCDataset(geom)
            xc, yc = g(gds["XC"][:]), g(gds["YC"][:])
            dλ, dφ = (λ[end] - λ[1]) / (Nx - 1), (φ[end] - φ[1]) / (Ny - 1)
            I = findall(x -> λ[1] - dλ / 2 <= x <= λ[end] + dλ / 2, xc)
            J = findall(y -> φ[1] - dφ / 2 <= y <= φ[end] + dφ / 2, yc)
            IR, JR = minimum(I):maximum(I), minimum(J):maximum(J)
            bwet = g(gds["Depth"][IR, JR]) .> 0
            rA = haskey(gds, "rA") ? g(gds["rA"][IR, JR]) :
                 [(6.371e6 * deg2rad(xc[2] - xc[1]))^2 * cosd(y) for _ in IR, y in yc[JR]]
            close(gds)
            ib, jb = argmin(abs.(xc[IR] .- λ[ic])), argmin(abs.(yc[JR] .- φ[jc]))
            days, barea, bvol, bcellA, bcellH = Float64[], Float64[], Float64[], Float64[], Float64[]
            for y in years
                a = NCDataset(bsose_ice_file("SIArea", y)); hf = NCDataset(bsose_ice_file("SIHeff", y))
                av = first(k for k in ("SIarea", "SIArea", "SIAREA") if haskey(a, k))
                hv = first(k for k in ("SIheff", "SIHeff", "SIHEFF") if haskey(hf, k))
                A3 = g(a[av][IR, JR, :]); H3 = g(hf[hv][IR, JR, :])
                stamps = [DateTime(s) for s in a["time"][:]]
                close(a); close(hf)
                for n in eachindex(stamps)
                    c = stamps[n] - Hour(60)                         # the 5-day window's centre
                    d = Dates.value(c - start) / 86_400_000
                    -2.5 <= d <= last(t) + 2.5 || continue
                    An, Hn = view(A3, :, :, n), view(H3, :, :, n)
                    push!(days, d)
                    push!(barea, sum(ifelse(bwet[k] && isfinite(An[k]), An[k] * rA[k], 0.0) for k in eachindex(An)) / 1e12)
                    push!(bvol,  sum(ifelse(bwet[k] && isfinite(Hn[k]), Hn[k] * rA[k], 0.0) for k in eachindex(Hn)) / 1e9)
                    push!(bcellA, An[ib, jb]); push!(bcellH, Hn[ib, jb])
                end
            end
            @info @sprintf("B-SOSE ice: %d 5-day records; cell %.2fE %.2fS (nearest to the model's %s)",
                           length(days), xc[IR][ib], -yc[JR][jb], cell)
            (; days, area = barea, volume = bvol, cellℵ = bcellA, cellV = bcellH,
               cellh = [a > 0.01 ? v / a : NaN for (a, v) in zip(bcellA, bcellH)],
               where = @sprintf("%.2fE %.2fS", xc[IR][ib], -yc[JR][jb]))
        end
    end
end

# Winter open water at the cell: the polynya test. Jun-Sep, mean of (1 - ℵ) and the share of
# records with ℵ < 0.85, model daily against B-SOSE 5-day.
if B !== nothing
    month_of(d) = month(start + Second(round(Int, d * 86400)))
    year_of(d)  = year(start + Second(round(Int, d * 86400)))
    println(@sprintf("Winter (Jun-Sep) open water at %s, model daily vs B-SOSE 5-day at %s:", cell, B.where))
    println("  year |  mean 1-ℵ  model / B-SOSE |  records with ℵ < 0.85  model / B-SOSE |  mean h·ℵ (m)  model / B-SOSE")
    for y in sort(unique(year_of.(t)))
        km = [n for n in eachindex(t) if year_of(t[n]) == y && month_of(t[n]) in 6:9]
        kb = [n for n in eachindex(B.days) if year_of(B.days[n]) == y && month_of(B.days[n]) in 6:9]
        (isempty(km) || isempty(kb)) && continue
        println(@sprintf("  %4d |      %5.3f / %5.3f       |            %5.1f%% / %5.1f%%           |      %5.2f / %5.2f",
                         y, mean(1 .- cell_ℵ[km]), mean(1 .- B.cellℵ[kb]),
                         100 * count(<(0.85), cell_ℵ[km]) / length(km), 100 * count(<(0.85), B.cellℵ[kb]) / length(kb),
                         mean(cell_V[km]), mean(B.cellV[kb])))
    end
    println()
end

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
lines!(ax1, t, area; color = :royalblue, label = "model")
B === nothing || (lines!(ax1, B.days, B.area; color = :black, linestyle = :dash, label = "B-SOSE, 5-day");
                  axislegend(ax1; position = :rt, framevisible = false))

ax2 = Axis(fig[2, 1]; width = SERIES_WIDTH, height = SERIES_HEIGHT, ylabel = "volume (km³)", title = "Domain ice volume, Σ h·ℵ·A")
lines!(ax2, t, volume; color = :royalblue)
B === nothing || lines!(ax2, B.days, B.volume; color = :black, linestyle = :dash)

ax3 = Axis(fig[3, 1]; width = SERIES_WIDTH, height = SERIES_HEIGHT, ylabel = "m", yscale = log10, title = "Largest anywhere in the domain")
lines!(ax3, t, positive(maxV); color = :firebrick, linewidth = 2, label = "max h·ℵ")
lines!(ax3, t, positive(maxh); color = :grey40, label = "max h (where ℵ > 0)")
hlines!(ax3, [ICE_MAX_VOLUME]; color = :black, linestyle = :dash, label = "run stops (h·ℵ = $(ICE_MAX_VOLUME) m)")
axislegend(ax3; position = :lt, framevisible = false)

ax4 = Axis(fig[4, 1]; width = SERIES_WIDTH, height = SERIES_HEIGHT, ylabel = "m", title = "At $cell (i = $ic, j = $jc)")
lines!(ax4, t, cell_V; color = :firebrick, linewidth = 2, label = "h·ℵ")
lines!(ax4, t, cell_h; color = :grey40, label = "h")
if B !== nothing
    lines!(ax4, B.days, B.cellV; color = :black, linewidth = 2, label = "B-SOSE SIheff (= h·ℵ)")
    lines!(ax4, B.days, B.cellh; color = :black, linestyle = :dot, label = "B-SOSE h")
end
axislegend(ax4; position = :lt, framevisible = false)

ax5 = Axis(fig[5, 1]; width = SERIES_WIDTH, height = SERIES_HEIGHT, ylabel = "ℵ", xlabel = "days since start", title = "Concentration at $cell")
lines!(ax5, t, cell_ℵ; color = :seagreen, label = "model")
B === nothing || (lines!(ax5, B.days, B.cellℵ; color = :black, linewidth = 2, label = "B-SOSE SIarea at $(B.where)");
                  axislegend(ax5; position = :lb, framevisible = false))
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
