# ==============================================================================================
# check_edge_bands.jl -- is the ice-deletion band along the open sides a salt source?
#
#     julia --project=. tools/check_edge_bands.jl 2Y_nested_seaice_fix3
#     EDGE=4 LAT_RANGE=-70:-60 julia --project=. tools/check_edge_bands.jl 2Y_nested_seaice_fix3
#
# THE SUSPICION. examples/sowinwarp_nested.jl deletes all ice in the EDGE cells along the open W,
# E and N sides every step (remove_edge_ice!). In winter that band is kept open water, so it keeps
# freezing: each freeze rejects brine into the ocean, the ice is deleted, and no meltwater ever
# comes back. A steady salt source would show as surface S rising through each winter in the
# band and the columns next to it, and as a mixed layer far deeper than the interior's.
# 2Y_nested_seaice_fix3's MLD map has exactly that at 90E, 62-67S, in winter 2015.
#
# Compares, over the ice zone (LAT_RANGE, ocean cells only), five groups of columns:
#   West edge   i = 1 .. EDGE                 ice deleted every step
#   West next   i = EDGE+1 .. EDGE+NEXT       the interior columns beside it
#   Interior    the middle INTERIOR columns   the reference
#   East next, East edge                      the same on the other side
#
# Reads <RUN>_surface.nc (daily surface T, S), <RUN>_sea_ice.nc (daily h, ℵ) and <RUN>_eke.nc
# (30-day-mean full-depth T, S, for the mixed layer; tools/mld.jl's threshold). Writes:
#   <RUN>_edge_bands.png     daily surface S and T, ice volume h·ℵ, and monthly MLD, per group
#   <RUN>_edge_hovmoller.png surface S against longitude and time, along the first and last
#                            HOV_WIDTH columns, averaged over HOV_LAT: whether salt spreads
#                            inward from an edge
# and prints each group's winter (Jun-Oct) mean surface S, its Jun -> Oct change, and the mean MLD.
# A west/east difference matters too: the coastal flow is westward, so ice drifts INTO the west
# band and is deleted there as well.
# ==============================================================================================

using NCDatasets, CairoMakie, Dates, Printf, Statistics

include(joinpath(@__DIR__, "mld.jl"))

RUN = isempty(ARGS) ? get(ENV, "RUN", "") : ARGS[1]
isempty(RUN) && error("give the run name: julia --project=. tools/check_edge_bands.jl <RUN>")
DIR        = get(ENV, "DIR", ".")
OUTDIR     = get(ENV, "OUTDIR", "plots")
EDGE       = parse(Int, get(ENV, "EDGE", "4"))          # MUST MATCH ICE_EDGE in examples/sowinwarp_nested.jl
NEXT       = parse(Int, get(ENV, "NEXT", "8"))
INTERIOR   = parse(Int, get(ENV, "INTERIOR", "160"))
HOV_WIDTH  = parse(Int, get(ENV, "HOV_WIDTH", "40"))
range_env(key, default) = Tuple(parse.(Float64, split(get(ENV, key, default), ':')))
LAT_RANGE  = range_env("LAT_RANGE", "-70:-60")
HOV_LAT    = range_env("HOV_LAT", "-67:-62")
WINTER     = 6:10

file(kind) = (p = joinpath(DIR, "$(RUN)_$(kind).nc"); isfile(p) || error("$p not found; set DIR"); p)
g(A) = Float64.(coalesce.(A, NaN))
surface_slice(v) = ndims(v) == 4 ? g(v[:, :, 1, :]) : g(v[:, :, :])

# ── surface, daily ───────────────────────────────────────────────────────────────────────────
ds = NCDataset(file("surface"))
λ, φ = g(ds["λ_caa"][:]), g(ds["φ_aca"][:])
t    = g(ds["time"][:]) ./ 86400
start = haskey(ds.attrib, "start_date") ? DateTime(ds.attrib["start_date"]) : DateTime(2014, 6, 1)
wet  = g(ds["bottom_height"][:, :]) .< 0
Ts, Ss = surface_slice(ds["T"]), surface_slice(ds["S"])
close(ds)
Nx, Ny, Nt = size(Ss)
dates = [start + Second(round(Int, d * 86400)) for d in t]

groups = ["West edge" => 1:EDGE,
          "West next" => (EDGE + 1):(EDGE + NEXT),
          "Interior"  => (div(Nx, 2) - div(INTERIOR, 2) + 1):(div(Nx, 2) + div(INTERIOR, 2)),
          "East next" => (Nx - EDGE - NEXT + 1):(Nx - EDGE),
          "East edge" => (Nx - EDGE + 1):Nx]
colours = [:firebrick, :orange, :black, :deepskyblue, :royalblue]
lats = findall(p -> LAT_RANGE[1] <= p <= LAT_RANGE[2], φ)

"""cos(lat)-weighted mean of a (Nx, Ny) map over columns `is`, the ice-zone rows, ocean only."""
function group_mean(A, is)
    num = den = 0.0
    for i in is, j in lats
        (wet[i, j] && isfinite(A[i, j])) || continue
        w = cosd(φ[j]); num += w * A[i, j]; den += w
    end
    return den == 0 ? NaN : num / den
end
series(F, is) = [group_mean(view(F, :, :, n), is) for n in 1:size(F, 3)]

S_series = [series(Ss, is) for (_, is) in groups]
T_series = [series(Ts, is) for (_, is) in groups]

# ── sea ice, daily ───────────────────────────────────────────────────────────────────────────
ds = NCDataset(file("sea_ice"))
V  = g(ds["h"][:, :, :]) .* g(ds["ℵ"][:, :, :])
ti = g(ds["time"][:]) ./ 86400
close(ds)
replace!(V, NaN => 0)
V_series = [series(V, is) for (_, is) in groups]

# ── mixed layer, from the 30-day means ───────────────────────────────────────────────────────
ds  = NCDataset(file("eke"))
zc  = g(ds["z_aac"][:])
Nz  = length(zc)
inactive = ds["inactive_nodes_ccc"][:, :, :]
kbot = [findfirst(k -> inactive[i, j, k] == 0, 1:Nz) for i in 1:Nx, j in 1:Ny]
win  = haskey(ds.attrib, "averaging_window_seconds") ? Float64(ds.attrib["averaging_window_seconds"]) : 30 * 86400.0
te   = g(ds["time"][:])
tm   = (te .- win / 2) ./ 86400                           # window centres, days
columns = sort(unique(vcat([collect(is) for (_, is) in groups]...)))
H = fill(NaN, Nx, Ny, length(te))
for n in eachindex(te)
    T3 = g(ds["T"][columns, lats, :, n]); S3 = g(ds["S"][columns, lats, :, n])
    for (a, i) in enumerate(columns), (b, j) in enumerate(lats)
        (wet[i, j] && kbot[i, j] !== nothing) || continue
        H[i, j, n] = mld(view(T3, a, b, :), view(S3, a, b, :), zc, kbot[i, j])
    end
end
close(ds)
H_series = [series(H, is) for (_, is) in groups]

# ── table ────────────────────────────────────────────────────────────────────────────────────
println()
println(@sprintf("%s: ice zone %.0f-%.0fS, edge band %d columns. Winter = Jun-Oct.", RUN, -LAT_RANGE[2], -LAT_RANGE[1], EDGE))
println("  group        columns  | winter | mean S   S Oct-Jun |  mean MLD | mean h·ℵ")
for (k, (name, is)) in enumerate(groups)
    for y in unique(year.(dates))
        sel = [n for n in eachindex(dates) if year(dates[n]) == y && month(dates[n]) in WINTER]
        length(sel) < 60 && continue
        jun = [n for n in sel if month(dates[n]) == first(WINTER)]
        oct = [n for n in sel if month(dates[n]) == last(WINTER)]
        dS  = isempty(jun) || isempty(oct) ? NaN : mean(S_series[k][oct]) - mean(S_series[k][jun])
        hsel = [n for n in eachindex(tm) if year(start + Second(round(Int, tm[n] * 86400))) == y &&
                                            month(start + Second(round(Int, tm[n] * 86400))) in WINTER]
        isel = [n for n in eachindex(ti) if year(start + Second(round(Int, ti[n] * 86400))) == y &&
                                            month(start + Second(round(Int, ti[n] * 86400))) in WINTER]
        mH = isempty(hsel) ? NaN : mean(filter(isfinite, H_series[k][hsel]))
        mV = isempty(isel) ? NaN : mean(V_series[k][isel])
        println(@sprintf("  %-10s %4d-%-4d |  %d  | %7.3f   %+7.3f   | %7.1f m | %6.2f m",
                         name, first(is), last(is), y, mean(S_series[k][sel]), dS, mH, mV))
    end
end
println()

# ── figure 1: the groups over time ───────────────────────────────────────────────────────────
mkpath(OUTDIR)
fig = Figure(fontsize = 14)
Label(fig[0, 1], @sprintf("%s: edge bands vs interior, %.0f-%.0fS (edge band = %d columns)",
                          RUN, -LAT_RANGE[2], -LAT_RANGE[1], EDGE); fontsize = 16, font = :bold)
panels = [("surface S (g/kg)", t, S_series), ("surface T (°C)", t, T_series),
          ("ice h·ℵ (m)", ti, V_series), ("MLD (m), 30-day means", tm, H_series)]
axes_ = Axis[]
for (row, (label, x, ys)) in enumerate(panels)
    ax = Axis(fig[row, 1]; ylabel = label, width = 900, height = 190,
              yreversed = startswith(label, "MLD"), xlabel = row == length(panels) ? "days since start" : "")
    for (k, (name, is)) in enumerate(groups)
        lines!(ax, x, ys[k]; color = colours[k], linewidth = name == "Interior" ? 2.5 : 1.5,
               label = @sprintf("%s (i %d-%d)", name, first(is), last(is)))
    end
    row == 1 && axislegend(ax; position = :lt, framevisible = false, labelsize = 11, nbanks = 2)
    push!(axes_, ax)
end
linkxaxes!(axes_...)
for ax in axes_[1:end-1]; hidexdecorations!(ax; grid = false); end
resize_to_layout!(fig)
out1 = joinpath(OUTDIR, "$(RUN)_edge_bands.png"); save(out1, fig); @info "wrote $out1"

# ── figure 2: Hovmöller along each edge ──────────────────────────────────────────────────────
hov_rows = findall(p -> HOV_LAT[1] <= p <= HOV_LAT[2], φ)
function hovmoller(is)
    A = fill(NaN, length(is), Nt)
    for (a, i) in enumerate(is), n in 1:Nt
        v = [Ss[i, j, n] for j in hov_rows if wet[i, j] && isfinite(Ss[i, j, n])]
        isempty(v) || (A[a, n] = mean(v))
    end
    return A
end
west, east = 1:HOV_WIDTH, (Nx - HOV_WIDTH + 1):Nx
Aw, Ae = hovmoller(west), hovmoller(east)
# anomaly from each column's own first-month mean, so the change over time is what shows
anomaly(A) = A .- [mean(filter(isfinite, A[a, 1:min(30, Nt)])) for a in axes(A, 1)]
Aw, Ae = anomaly(Aw), anomaly(Ae)
lim = max(quantile(filter(isfinite, abs.(vcat(vec(Aw), vec(Ae)))), 0.99), 1e-3)

fig = Figure(fontsize = 14)
Label(fig[0, 1:3], @sprintf("%s: surface S change since the first month, %.0f-%.0fS mean", RUN, -HOV_LAT[2], -HOV_LAT[1]);
      fontsize = 16, font = :bold)
hm = nothing
for (col, (title, is, A)) in enumerate((("West edge", west, Aw), ("East edge", east, Ae)))
    ax = Axis(fig[1, col]; title, xlabel = "Longitude (°E)", ylabel = col == 1 ? "days since start" : "",
              width = 420, height = 520)
    global hm = heatmap!(ax, λ[is], t, A; colormap = :balance, colorrange = (-lim, lim))
    vlines!(ax, col == 1 ? [λ[EDGE] + (λ[2] - λ[1]) / 2] : [λ[Nx - EDGE] + (λ[2] - λ[1]) / 2];
            color = :black, linestyle = :dash)
end
Colorbar(fig[1, 3], hm; label = "ΔS (g/kg); dashed: inner edge of the ice-deletion band")
resize_to_layout!(fig)
out2 = joinpath(OUTDIR, "$(RUN)_edge_hovmoller.png"); save(out2, fig); @info "wrote $out2"
