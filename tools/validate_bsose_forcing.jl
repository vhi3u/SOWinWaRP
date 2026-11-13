# ==============================================================================
# validate_bsose_forcing.jl -- is a B-SOSE-nested run (examples/sowinwarp_nested.jl) forced with B-SOSE at
# the right TIME? Checked on what the run actually did, not on the code that should do it.
#
#     julia --project=. tools/validate_bsose_forcing.jl 2Y_nested_seaice_fix9
#
# WHY IT CAN BE WRONG. B-SOSE stamps each monthly mean at the END of the month it averages (the stamps
# drift ~10 h a record either side of the boundary). src/setup_bsose.jl gives NumericalEarth each record's
# averaging window, so a parent series should place the value at the MIDDLE of its month. Placing it at the
# stamp delays every boundary value by half a month; the stamp-resolution bug fixed on 2026-10-28 shifted
# records by a whole month.
#
# THE TEST. In the outermost BAND cells of the open sides (west, east, north) the nest relaxes the child to
# the parent every 6 hours, so the run's daily top-cell T and S there should follow B-SOSE's top level
# (0-4.2 m, the same layer) with a lag of hours. B-SOSE's monthly means, placed at their window centres and
# interpolated linearly in time, are compared with the run's daily snapshots shifted by a lag L:
#     L ≈ 0      the run is forced on time
#     L ≈ +15 d  each record was placed at its stamp, the END of its month (the run lags B-SOSE)
#     L ≈ ±30 d  a month off
# The lag is scanned over ±45 days for the smallest RMS difference, per side and in all; months where the
# run's monthly mean in the band is far from B-SOSE's own record (a skipped or doubled month) are listed.
# The seasonal cycle in the top cell is what times it, so the lag is sharpest where that cycle is large.
#
# AND THE START. The run's t = 0 snapshot (2014-06-01 12:00 for sowinwarp_nested.jl) against B-SOSE
# interpolated to that date, and against the bare May and June records, over every wet cell.
#
# The coastal freshwater is constant in time and ERA5 is not B-SOSE: neither is checked here.
#
# WRITES (OUTDIR, default plots/): <RUN>_bsose_timing.png, the band-mean T and S per side: the run daily
# (red), B-SOSE's records at their window centres (black dots, the bar spanning the month averaged) and the
# interpolated series (black line). On stdout: the lag scan, the monthly check and the initial state.
#
# LOGIN-NODE SAFE: NCDatasets, the top level of B-SOSE over the run's box only.
#
# ENV
#     DIR, OUTDIR, BSOSE_DIR   (., plots, default_bsose_directory())
#     BAND                     open-side cells compared (default 2, inside the nest's ~4-cell band)
# ==============================================================================

using NCDatasets, Dates, Printf, Statistics, CairoMakie

include(joinpath(@__DIR__, "..", "src", "setup_bsose.jl"))   # default_bsose_directory, resolve_bsose_filename, bsose_window_center

const RUN = length(ARGS) > 0 ? ARGS[1] : error("give the run: julia --project=. tools/validate_bsose_forcing.jl <RUN>")
const DIR    = get(ENV, "DIR", ".")
const OUTDIR = get(ENV, "OUTDIR", "plots")
const BDIR   = get(ENV, "BSOSE_DIR", default_bsose_directory())
const BAND   = parse(Int, get(ENV, "BAND", "2"))
const LAGS   = -45:45
mkpath(OUTDIR)

f64(A) = Float64.(coalesce.(A, NaN))

# ── the run: grid, open sides, daily top-cell T and S ──────────────────────────────────────────
const SURF = joinpath(DIR, "$(RUN)_surface.nc")
isfile(SURF) || error("no $SURF (set DIR)")
sd = Dataset(SURF)
λc, φc = f64(sd["λ_caa"][:]), f64(sd["φ_aca"][:])
Nx, Ny = length(λc), length(φc)
start = DateTime(string(sd.attrib["start_date"]))
wet = f64(sd["bottom_height"][:, :]) .< 0
times = f64(sd["time"][:])
dates = [start + Millisecond(round(Int, 1000t)) for t in times]
readsurf(name, n) = (v = sd[name]; f64(ndims(v) == 4 ? v[:, :, 1, n] : v[:, :, n]))

side_masks = [("west",  [i <= BAND for i in 1:Nx, _ in 1:Ny]),
              ("east",  [i > Nx - BAND for i in 1:Nx, _ in 1:Ny]),
              ("north", [j > Ny - BAND for _ in 1:Nx, j in 1:Ny])]
SIDES = [(name, m .& wet) for (name, m) in side_masks if any(m .& wet)]
@info "$RUN: $(Nx)x$(Ny), $(length(times)) surface snapshots from $(first(dates)) to $(last(dates)); " *
      "the outer $BAND cells of " * join(["$(name) ($(count(m)) cells)" for (name, m) in SIDES], ", ")

# ── B-SOSE: the top level over the run's box, the records the run spans ────────────────────────
function bsose_top(var_name)
    path = joinpath(BDIR, resolve_bsose_filename(BDIR, var_name, 156))
    Dataset(path) do d
        names = var_name == :temperature ? ("THETA", "Theta") : ("SALT", "Salt")
        key = something(findfirst(in(keys(d)), names) |> i -> i === nothing ? nothing : names[i],
                        first(k for k in keys(d) if ndims(d[k]) == 4))
        xc, yc = f64(d["XC"][:]), f64(d["YC"][:])
        ii = [argmin(abs.(xc .- λ)) for λ in λc]
        jj = [argmin(abs.(yc .- φ)) for φ in φc]
        off = max(maximum(abs.(xc[ii] .- λc)), maximum(abs.(yc[jj] .- φc)))
        off > 1 / 24 && @warn @sprintf("B-SOSE's cells are up to %.3f degrees from the run's: nearest cell used", off)
        stamps = DateTime.(d["time"][:])
        centres = bsose_window_center.(stamps)
        keep = let k = findall(c -> first(dates) - Day(62) <= c <= last(dates) + Day(62), centres); first(k):last(k) end
        raw = f64(d[key][minimum(ii):maximum(ii), minimum(jj):maximum(jj), 1, keep])
        values = [raw[ii[i] - minimum(ii) + 1, jj[j] - minimum(jj) + 1, n] for i in 1:Nx, j in 1:Ny, n in eachindex(keep)]
        values[values .== 0] .= NaN                            # B-SOSE's land is zero
        (; values, stamps = stamps[keep], centres = centres[keep], key)
    end
end
B = Dict(:T => bsose_top(:temperature), :S => bsose_top(:salinity))
@info "B-SOSE $(B[:T].key)/$(B[:S].key): $(length(B[:T].centres)) monthly records, centres $(Date(first(B[:T].centres)))..$(Date(last(B[:T].centres)))"

"""B-SOSE at `date`, linear between window centres; NaN outside them."""
function bsose_at(b, date)
    n = searchsortedlast(b.centres, date)
    (1 <= n < length(b.centres)) || return fill(NaN, Nx, Ny)
    w = (date - b.centres[n]) / (b.centres[n + 1] - b.centres[n])
    return (1 - w) .* b.values[:, :, n] .+ w .* b.values[:, :, n + 1]
end
masked_mean(A, m) = (v = [A[k] for k in eachindex(A) if m[k] && isfinite(A[k])]; isempty(v) ? NaN : mean(v))

# Band means per day: the run, and B-SOSE's interpolated series kept as a function of date for the lag scan
snaps = findall(>(0), times)
run_band = Dict((v, s) => [masked_mean(readsurf(String(v), n), m) for n in snaps] for v in (:T, :S) for (s, m) in SIDES)

# ── the lag scan ──────────────────────────────────────────────────────────────────────────────
# L > 0: the run lags B-SOSE by L days, i.e. run(t) matches B-SOSE(t - L)
function lag_rms(v, sidelist, L)
    se = 0.0; n = 0
    for (s, m) in sidelist, (q, k) in enumerate(snaps)
        r = run_band[(v, s)][q]
        b = masked_mean(bsose_at(B[v], dates[k] - Day(L)), m)
        (isfinite(r) && isfinite(b)) || continue
        se += (r - b)^2; n += 1
    end
    return n == 0 ? NaN : sqrt(se / n)
end

println("\n", "="^96)
println("B-SOSE FORCING TIMING, $RUN: the outer $BAND cells of the open sides against B-SOSE's top level")
println("  L > 0: the run lags B-SOSE by L days. On time: L ≈ 0. Records at their stamps: L ≈ +15. A month off: ±30.")
println("="^96)
@printf("\n  %-8s %-6s | %7s %9s | %9s %9s %9s\n", "side", "var", "best L", "rms at L", "rms L=0", "rms L=15", "rms L=-15")
for (s, m) in vcat(SIDES, [("all", nothing)]), v in (:T, :S)
    sl = s == "all" ? SIDES : [(s, m)]
    r = [lag_rms(v, sl, L) for L in LAGS]
    best = LAGS[argmin(map(x -> isnan(x) ? Inf : x, r))]
    @printf("  %-8s %-6s | %+5d d %9.4f | %9.4f %9.4f %9.4f\n", s, v, best, minimum(filter(isfinite, r)),
            lag_rms(v, sl, 0), lag_rms(v, sl, 15), lag_rms(v, sl, -15))
end

# ── month by month: the run's band mean over each B-SOSE averaging window against that record ─
println("\n  MONTH BY MONTH: the run's band mean over the month each B-SOSE record averages, minus the record")
println("  (a month skipped or doubled shows as one large value among small ones; a whole-series lag as a")
println("  seasonal pattern of signs)")
@printf("  %-9s |", "record")
for (s, _) in SIDES; @printf(" %8s %8s", "ΔT $s", "ΔS $s"); end
println()
for (n, c) in enumerate(B[:T].centres)
    a, b = bsose_averaging_window(B[:T].stamps[n])
    q = [i for (i, k) in enumerate(snaps) if a <= dates[k] < b]
    length(q) < 20 && continue
    @printf("  %-9s |", Dates.format(c, "yyyy-mm"))
    for (s, m) in SIDES
        @printf(" %+8.3f %+8.3f", mean(run_band[(:T, s)][q]) - masked_mean(B[:T].values[:, :, n], m),
                mean(run_band[(:S, s)][q]) - masked_mean(B[:S].values[:, :, n], m))
    end
    println()
end

# ── the initial state ─────────────────────────────────────────────────────────────────────────
k0 = findfirst(==(0), times)
if k0 === nothing
    println("\n  INITIAL STATE: the surface file has no t = 0 snapshot; not checked")
else
    println("\n  INITIAL STATE, $(dates[k0]): the run's t = 0 top cell minus B-SOSE, over every wet cell")
    @printf("  %-34s | %9s %9s | %9s %9s\n", "B-SOSE taken as", "mean ΔT", "rms ΔT", "mean ΔS", "rms ΔS")
    nprev = searchsortedlast(B[:T].centres, dates[k0])
    for (label, f) in (("interpolated to $(Date(dates[k0]))", b -> bsose_at(b, dates[k0])),
                       ("record centred $(Date(B[:T].centres[nprev]))", b -> b.values[:, :, nprev]),
                       ("record centred $(Date(B[:T].centres[nprev + 1]))", b -> b.values[:, :, nprev + 1]))
        dT = readsurf("T", k0) .- f(B[:T]); dS = readsurf("S", k0) .- f(B[:S])
        ok = wet .& isfinite.(dT) .& isfinite.(dS)
        @printf("  %-34s | %+9.4f %9.4f | %+9.4f %9.4f\n", label, mean(dT[ok]), sqrt(mean(dT[ok] .^ 2)),
                mean(dS[ok]), sqrt(mean(dS[ok] .^ 2)))
    end
end
close(sd)

# ── plot ──────────────────────────────────────────────────────────────────────────────────────
fig = Figure(size = (1300, 300 * length(SIDES)))
xday(d) = Dates.value(d - start) / 86400e3
for (r, (s, m)) in enumerate(SIDES), (c, v) in enumerate((:T, :S))
    ax = Axis(fig[r, c]; title = "$s side, outer $BAND cells: $(v == :T ? "T (°C)" : "S (g/kg)"), top cell",
              xlabel = r == length(SIDES) ? "days since $(Date(start))" : "")
    lines!(ax, xday.(dates[snaps]), run_band[(v, s)]; color = :firebrick, label = "$RUN, daily")
    bx = xday.(B[v].centres)
    bv = [masked_mean(B[v].values[:, :, n], m) for n in eachindex(B[v].centres)]
    lines!(ax, bx, bv; color = :black, label = "B-SOSE, at window centres")
    for n in eachindex(bx)
        a, b = bsose_averaging_window(B[v].stamps[n])
        lines!(ax, [xday(a), xday(b)], [bv[n], bv[n]]; color = (:black, 0.4), linewidth = 3)
    end
    scatter!(ax, bx, bv; color = :black, markersize = 7)
    xlims!(ax, -10, xday(last(dates)) + 10)
    r == 1 && c == 1 && axislegend(ax; position = :lb)
end
out = joinpath(OUTDIR, "$(RUN)_bsose_timing.png")
save(out, fig)
@info "wrote $out"
