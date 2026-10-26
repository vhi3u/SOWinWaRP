# ==============================================================================
# compare_bsose.jl -- model against the B-SOSE parent it is nudged toward.
#
#     julia --project=. tools/compare_bsose.jl 8M_bsose_v2
#     DAYS=first,mid,last julia --project=. tools/compare_bsose.jl 8M_bsose_v2
#
# FOUR QUESTIONS, in order of how cheaply they can be wrong:
#
#   1. IS THE SPONGE ACTUALLY NUDGING? The test is NOT a domain-mean bias -- it is bias as a
#      function of distance from the boundary. The sponge restores T, S, u, v over the outer
#      SPONGE_WIDTH degrees (3 deg = 18 cells) at SPONGE_TIMESCALE (1 day). So the bias should be
#      near zero inside 18 cells and grow inward. A KNEE AT 18 CELLS is the signature that it
#      works; a flat profile means it is not doing anything, whatever its settings say.
#
#   2. DID THE SHORTWAVE SPLIT FIX THE MIXED LAYER? The recorded failure was a 3.1 m model mixed
#      layer where B-SOSE had 21.8 m, with summer SST +7 to +9 C too warm, caused by depositing
#      100% of TFLUX in the top cell (section 3.3 of Model Architecture.md). MLD and SST together
#      are the test of PENETRATING_SHORTWAVE, and this is the first run with it on.
#
#   3. IS WINTER WATER THERE? WW is a SUBSURFACE TEMPERATURE MINIMUM, the remnant of the winter
#      mixed layer, typically 50-200 m. So the diagnostic is not a temperature bias but whether
#      a genuine Tmin exists below the mixed layer, at what depth, and with what value -- against
#      B-SOSE's. This is the science target, and it is the reason the vertical grid keeps 8 m
#      spacing through that band.
#
#   4. IS THE DOMAIN DRIFTING? Surface T and S means against B-SOSE over time.
#
# WHAT THIS RUN CAN AND CANNOT SHOW. `src/model.jl` writes the SURFACE (`indices=(:,:,Nz)`) and
# ONE LONGITUDE SECTION (`indices=(mid_lon_idx,:,:)`, mid_lon_idx = Nx/2 = 180, about 120 E).
# There is no 3-D output, so:
#   * questions 1 and 4 are answered over the whole surface;
#   * questions 2 and 3 are answered ALONG THE SECTION ONLY.
# For WW that is not much of a loss -- a meridional transect at 120 E crosses the shelf, the
# slope and the ACC, which is where the layer's structure lives. A domain-wide MLD map would need
# a 3-D T/S output added to model.jl.
#
# CONVENTIONS, and a caveat that is NOT bookkeeping. The model runs TEOS10, so its tracers are
# CONSERVATIVE temperature and ABSOLUTE salinity. B-SOSE ships MITgcm `Theta` (POTENTIAL
# temperature) and `Salt` (PRACTICAL salinity), and nothing in setup_bsose.jl converts between
# them -- we initialise and nudge one with the other. At Winter Water temperatures the Theta/Theta_c
# difference is a few hundredths of a degree, but S_A - S_P is about 0.165 g/kg, worth ~0.13 kg/m3
# of density against an MLD threshold of 0.03. It largely cancels here because MLD is a
# THRESHOLD relative to 10 m and both sides go through the same equation of state below -- but do
# not read absolute densities off this, and do not dismiss the conversion when interpreting a
# Tmin difference of a few hundredths.
#
# Environment knobs:
#     DIR     where the model .nc files live          (default: working directory)
#     DAYS    output indices, or "first,mid,last"     (default: first,mid,last)
#     OUTDIR  where figures go                        (tools/bsose_comparison)
#     MLD_DT  density threshold, kg/m3                (default 0.03, de Boyer Montegut)
# ==============================================================================

using Oceananigans, Oceananigans.Units, Oceananigans.Grids

# `Time` is exported by BOTH Dates and Oceananigans.Units, and this script needs both modules, so
# an unqualified `Time` is ambiguous and errors at the first `fts[Time(t)]`. The one wanted here
# is Oceananigans', which is how a FieldTimeSeries is indexed in time.
using Oceananigans.Units: Time
using NumericalEarth, NumericalEarth.ECCO
using NCDatasets, CairoMakie, Printf, Statistics, Dates
using SeawaterPolynomials
using SeawaterPolynomials.TEOS10: TEOS10EquationOfState

include(joinpath(@__DIR__, "..", "src", "setup_bsose.jl"))

const RUN    = length(ARGS) > 0 ? ARGS[1] : get(ENV, "RUN_NAME", "8M_bsose_v2")
const DIR    = get(ENV, "DIR", ".")
const OUTDIR = get(ENV, "OUTDIR", joinpath(@__DIR__, "bsose_comparison"))
const MLD_DT = parse(Float64, get(ENV, "MLD_DT", "0.03"))

# Which months get a surface difference map. Calendar month NUMBERS, not year-months, so the
# selection survives a change of run dates. Default is June, August, October, December, January:
# a seasonal progression from mid-winter to late summer, which is the Winter Water
# formation-and-capping cycle.
#
# JANUARY RATHER THAN FEBRUARY on purpose. A June-February run (END_DATE = 2015-02-01) gives
# February exactly ONE output day, so its "monthly mean" would be a snapshot subtracted from a
# month -- mostly a mismatch of averaging windows rather than model error, sitting next to four
# genuine monthly means as though it were comparable. January gets 30 of its 31 days and is a
# real mean. Add 2 back only if END_DATE is extended to 2015-03-01.
const MONTHS = [parse(Int, x) for x in split(get(ENV, "MONTHS", "6,8,10,12,1"), ",") if !isempty(x)]
mkpath(OUTDIR)

const SPONGE_CELLS = 18      # 3 degrees at 1/6 deg; keep in step with SPONGE_WIDTH in model.jl
const EOS = TEOS10EquationOfState()

surf_file = joinpath(DIR, "$(RUN)_surface.nc")
sect_file = joinpath(DIR, "$(RUN)_midlon.nc")
isfile(surf_file) || error("no $surf_file")

# ------------------------------------------------------------------ the model's own grid
# Rebuilt rather than read, because the distance-to-boundary and z metrics have to match what the
# run used. Same hard-coded production grid as inspect_bathymetry.jl, and the same reasoning.
const λ₁, λ₂, φ₁, φ₂ = 90.0, 150.0, -70.0, -45.0
const SCALING = 6
const Z_EXTENT, DZ_SURFACE, DZ_BOTTOM, Z_CONST = 5800, 8, 400, 250

z = ReferenceToStretchedDiscretization(; extent = Z_EXTENT, constant_spacing = DZ_SURFACE,
                                       maximum_spacing = DZ_BOTTOM,
                                       constant_spacing_extent = Z_CONST,
                                       stretching = PowerLawStretching(1.12))
Nx, Ny, Nz = Int(SCALING * (λ₂ - λ₁)), Int(SCALING * (φ₂ - φ₁)), length(z)
@info "model grid $(Nx)x$(Ny)x$(Nz) (Nz from DZ_SURFACE=$DZ_SURFACE, DZ_BOTTOM=$DZ_BOTTOM, const=$Z_CONST)"

grid = LatitudeLongitudeGrid(CPU(); size = (Nx, Ny, Nz), longitude = (λ₁, λ₂),
                             latitude = (φ₁, φ₂), z, halo = (7, 7, 7))
bottom = regrid_bathymetry(grid; height_above_water = 1, minimum_depth = 10,
                           interpolation_passes = 5)
fix_boundary_topography!(bottom, grid; minimum_boundary_depth = 100)
grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom))

zc  = Array(znodes(grid, Center()))
φc  = Array(φnodes(grid, Center()))
MID = div(Nx, 2)

# ------------------------------------------------------------------ the B-SOSE parent
ds = Dataset(surf_file)
model_times = Float64.(ds["time"][:])
nt = length(model_times)
days = let s = get(ENV, "DAYS", "first,mid,last")
    s == "first,mid,last" ? unique([1, cld(nt, 2), nt]) :
        [parse(Int, x) for x in split(s, ",") if !isempty(x)]
end

start_date = DateTime("2014-06-01")
dataset = BSOSEMonthly(iteration = 156)
available = all_dates(dataset, :temperature)
run_end = start_date + Second(round(Int, maximum(model_times)))
inside = findall(d -> start_date <= bsose_window_center(d) <= run_end, available)
lo = max(firstindex(available), (isempty(inside) ? 1 : first(inside)) - 1)
hi = min(lastindex(available), (isempty(inside) ? length(available) : last(inside)) + 1)
bc_dates = available[lo:hi]
@info "B-SOSE records: $(length(bc_dates)), $(first(bc_dates)) to $(last(bc_dates))"

region = bsose_region(grid)
# time_indices_in_memory = the full length ON PURPOSE. The default is 2, and with a long record
# that reloads from disk and has been measured to hand back stale, duplicated snapshots
# (tools/face_transports.jl, 12 records requested and 9 distinct returned). For an analysis we
# want every record resident.
Tb = FieldTimeSeries(Metadata(:temperature; dataset, dates = bc_dates, region), grid;
                     time_indices_in_memory = length(bc_dates))
Sb = FieldTimeSeries(Metadata(:salinity;    dataset, dates = bc_dates, region), grid;
                     time_indices_in_memory = length(bc_dates))
ub = FieldTimeSeries(Metadata(:u_velocity;  dataset, dates = bc_dates, region), grid;
                     time_indices_in_memory = length(bc_dates))
vb = FieldTimeSeries(Metadata(:v_velocity;  dataset, dates = bc_dates, region), grid;
                     time_indices_in_memory = length(bc_dates))
offset = model_clock_offset(Metadata(:temperature; dataset, dates = bc_dates, region), start_date)

slab(d, var, t) = (v = d[var]; A = ndims(v) == 4 ? v[:, :, 1, t] : v[:, :, t];
                   Float64.(coalesce.(A, NaN)))
wet3(i, j, k) = !immersed_cell(i, j, k, grid)

"""Potential density anomaly at 0 dbar, same equation of state for model and parent."""
ρθ(T, S) = (isfinite(T) && isfinite(S)) ?
           SeawaterPolynomials.ρ′(T, S, 0.0, EOS) : NaN

"""
Mixed layer depth by the de Boyer Montegut density threshold: the shallowest depth at which
potential density exceeds its 10 m value by `MLD_DT` kg/m3. `T`, `S` are columns ordered
bottom-to-top as Oceananigans stores them; `zs` is the matching cell-centre depth (negative).
"""
function mld(T, S, zs; Δρ = MLD_DT)
    k10 = findlast(z -> z <= -10, zs)          # deepest cell at or above 10 m
    k10 === nothing && (k10 = length(zs))
    ref = ρθ(T[k10], S[k10])
    isfinite(ref) || return NaN
    for k in (k10 - 1):-1:1                     # Oceananigans orders z bottom-to-top, so a
                                                # SMALLER index is DEEPER
        ρ = ρθ(T[k], S[k])
        isfinite(ρ) || return -zs[k + 1]
        if ρ - ref >= Δρ
            return -zs[k]
        end
    end
    return -zs[1]                               # unstratified to the floor
end

"""
Subsurface temperature minimum -- the Winter Water core -- as (value, depth), or `nothing`.

Only a minimum that is genuinely INTERIOR counts: it must sit below the mixed layer and have
warmer water both above and below it. A monotonic winter column has no WW signature and should
report nothing rather than its coldest cell.
"""
function winter_water(T, zs, mld_m)
    kk = [k for k in eachindex(zs) if isfinite(T[k]) && -zs[k] > max(mld_m, 20)]
    length(kk) < 3 && return nothing
    best = nothing
    for n in 2:(length(kk) - 1)
        k, kup, kdn = kk[n], kk[n + 1], kk[n - 1]   # kup is shallower (larger index)
        if T[k] < T[kup] && T[k] < T[kdn]
            (best === nothing || T[k] < best[1]) && (best = (T[k], -zs[k]))
        end
    end
    return best
end

# =============================================================== 1. monthly differences
# MONTHLY MEAN MODEL minus the MONTHLY B-SOSE RECORD, which is the only apples-to-apples
# comparison available: the model writes daily, B-SOSE is a monthly mean. An earlier version of
# this script interpolated B-SOSE to each daily output time instead, which compares a snapshot
# against a month and attributes the difference to the model.
#
# HOW TO READ THE MAPS, and this is the part that is easy to get backwards. Only the outer
# SPONGE_CELLS cells are restored toward B-SOSE, and the sponge mask is LATERAL ONLY -- it takes
# (λ, φ, z) and never uses z. There is no surface restoring anywhere in this model, and never has
# been; the surface coupling is a prescribed FLUX (TFLUX/SFLUX) plus wind stress. So:
#
#     inside the band   -> a difference near zero is the sponge WORKING
#     in the interior   -> a difference is the MODEL, not an error
#
# A map that is uniformly near zero would mean the interior is over-constrained. A map with no
# knee at the band edge would mean the sponge is not doing anything.

"""Model output indices grouped by the B-SOSE record whose month contains them."""
function month_groups(times, dates, t0)
    groups = [Int[] for _ in dates]
    for (t, sec) in enumerate(times)
        d = t0 + Second(round(Int, sec))
        _, best = findmin(abs(Dates.value(bsose_window_center(dd) - d)) for dd in dates)
        push!(groups[best], t)
    end
    return groups
end

groups = month_groups(model_times, bc_dates, start_date)
bins = [(0, 2), (3, 5), (6, 11), (12, 17), (18, 23), (24, 35), (36, 59), (60, 200)]

"""
Distance-to-nearest-boundary and a wet mask SHAPED TO THE FIELD, not to the tracer grid.

`u` is Face-located in x so it is (Nx+1, Ny); `v` is (Nx, Ny+1); `T` and `S` are (Nx, Ny). One
mask for all four throws a BoundsError on the velocities, which is exactly what happened on the
first real run of this script. The distance rule is `min(index-1, extent-index)` using THAT
field's own extent per axis, so index 1 and index `extent` both come out 0 -- the same rule
`tools/compare_obc_logs.jl` uses for the velocity argmax.
"""
function field_metrics(sz)
    nx, ny = sz
    dist = [min(i - 1, nx - i, j - 1, ny - j) for i in 1:nx, j in 1:ny]
    wet  = [wet3(clamp(i, 1, Nx), clamp(j, 1, Ny), Nz) for i in 1:nx, j in 1:ny]
    return dist, wet
end

println()
println("="^100)
println("1. MONTHLY MEAN MODEL minus MONTHLY B-SOSE, at the surface")
println("   The sponge restores only the outer $SPONGE_CELLS cells, laterally, at all depths.")
println("   Near-zero INSIDE the band = sponge working. A difference in the INTERIOR is the model.")
println("="^100)

FIELDS = (("T", "°C", :balance), ("S", "g/kg", :delta), ("u", "m/s", :balance), ("v", "m/s", :balance))

for (n, dd) in enumerate(bc_dates)
    isempty(groups[n]) && continue
    centre = bsose_window_center(dd)
    month(centre) in MONTHS || continue
    label = Dates.format(centre, "yyyy-mm")
    ndays = length(groups[n])

    # A "monthly mean" built from a handful of days is a snapshot wearing a month's label. Kept
    # as a guard rather than removed with February: the first and last months of ANY run window
    # are partial, so this fires again the moment the dates change.
    partial = ndays < 15
    partial && @warn "$label has only $ndays model output(s): a SNAPSHOT, not a monthly mean. It is a partial month at the edge of the run window -- extend END_DATE, or drop this month from MONTHS."

    # monthly mean of the model, and the parent record at its own time (no interpolation)
    diffs = Dict{String,Matrix{Float64}}()
    for (f, _, _) in FIELDS
        acc = nothing
        for t in groups[n]
            A = slab(ds, f, t)
            acc = acc === nothing ? copy(A) : (acc .+= A)
        end
        model_month = acc ./ length(groups[n])
        parent = Array(interior(getfield((T = Tb, S = Sb, u = ub, v = vb), Symbol(f))[n]))
        diffs[f] = model_month .- parent[:, :, Nz]
    end

    @printf("\n%s  (%d model output%s, %s)\n", label, ndays, ndays == 1 ? "" : "s",
            partial ? "PARTIAL MONTH - snapshot" : "monthly mean")
    @printf("%-11s |", "cells from")
    for (f, un, _) in FIELDS; @printf(" %14s", "mean Δ$f ($un)"); end
    println()
    println("-"^(13 + 15 * length(FIELDS)))
    metrics = Dict(f => field_metrics(size(diffs[f])) for (f, _, _) in FIELDS)
    for (lo_, hi_) in bins
        any_row = false
        vals = String[]
        for (f, _, _) in FIELDS
            dist, wet = metrics[f]
            m = (dist .>= lo_) .& (dist .<= hi_) .& wet
            v = any(m) ? filter(isfinite, diffs[f][m]) : Float64[]
            push!(vals, isempty(v) ? "           ---" : @sprintf("%14.5f", mean(abs, v)))
            any_row |= !isempty(v)
        end
        any_row || continue
        @printf("%4d-%-6d |%s", lo_, hi_, join(vals))
        println(hi_ <= SPONGE_CELLS - 1 ? "   in sponge" : lo_ >= SPONGE_CELLS ? "   free" : "   band edge")
    end

    # ---- the map: one panel per field ----
    fig = Figure(size = (1350, 760))
    Label(fig[0, 1:4], "$(RUN) — $(label): monthly mean model − monthly B-SOSE (surface)",
          fontsize = 19, font = :bold)
    λc = Array(λnodes(grid, Center()))
    for (k, (f, un, cmap)) in enumerate(FIELDS)
        D = copy(diffs[f])
        _, wet = field_metrics(size(D))
        D[.!wet] .= NaN
        fin = filter(isfinite, D)
        lim = isempty(fin) ? 1.0 : max(quantile(abs.(fin), 0.99), 1e-6)
        ax = Axis(fig[1 + div(k - 1, 2), 1 + 2 * ((k - 1) % 2)],
                  title = "Δ$f  ($un)", xlabel = "longitude", ylabel = "latitude")
        hm = heatmap!(ax, 1:size(D, 1), 1:size(D, 2), D, colormap = :balance,
                      colorrange = (-lim, lim), nan_color = :gray85)
        # the sponge band edge
        vlines!(ax, [SPONGE_CELLS, Nx - SPONGE_CELLS], color = :black, linestyle = :dot)
        hlines!(ax, [SPONGE_CELLS, Ny - SPONGE_CELLS], color = :black, linestyle = :dot)
        Colorbar(fig[1 + div(k - 1, 2), 2 + 2 * ((k - 1) % 2)], hm)
    end
    Label(fig[3, 1:4], "dotted lines: edge of the $(SPONGE_CELLS)-cell sponge band. " *
          "Inside it a small difference means the sponge works; outside, the difference is the model.",
          fontsize = 12)
    fo = joinpath(OUTDIR, "$(RUN)_diff_$(label).png")
    save(fo, fig); @info "wrote $fo"
end

# =============================================================== 2 & 3. MLD and Winter Water
if isfile(sect_file)
    dsec = Dataset(sect_file)
    println("="^96)
    println("2 & 3. MIXED LAYER and WINTER WATER along the $(round(λ₁ + (MID - 0.5)/SCALING, digits=1))E section")
    println("   MLD: de Boyer Montegut, Δρ = $MLD_DT kg/m3 from 10 m, SAME equation of state both sides.")
    println("   The recorded failure to beat: model 3.1 m where B-SOSE had 21.8 m (section 3.3).")
    println("="^96)
    @printf("\n%6s | %-26s | %-26s | %s\n", "day", "MLD  model / B-SOSE (m)",
            "Tmin  model / B-SOSE (C)", "Tmin depth  model / B-SOSE (m)")
    println("-"^104)

    for t in days
        Ts = Float64.(coalesce.(dsec["T"][1, :, :, t], NaN))   # (Ny, Nz)
        Ss = Float64.(coalesce.(dsec["S"][1, :, :, t], NaN))
        tb = model_times[t] + offset
        Tps = Array(interior(Tb[Time(tb)]))[MID, :, :]
        Sps = Array(interior(Sb[Time(tb)]))[MID, :, :]

        mm, mp, wm, wp = Float64[], Float64[], Float64[], Float64[]
        dm, dp = Float64[], Float64[]
        for j in 1:Ny
            wet3(MID, j, Nz) || continue
            a = mld(Ts[j, :], Ss[j, :], zc); b = mld(Tps[j, :], Sps[j, :], zc)
            isfinite(a) && push!(mm, a); isfinite(b) && push!(mp, b)
            wa = winter_water(Ts[j, :], zc, isfinite(a) ? a : 0.0)
            wb = winter_water(Tps[j, :], zc, isfinite(b) ? b : 0.0)
            wa === nothing || (push!(wm, wa[1]); push!(dm, wa[2]))
            wb === nothing || (push!(wp, wb[1]); push!(dp, wb[2]))
        end
        mean_or(v) = isempty(v) ? NaN : mean(v)
        @printf("%6d | %11.1f / %-12.1f | %11.3f / %-12.3f | %9.0f / %-9.0f  (WW in %d/%d cols model, %d parent)\n",
                t - 1, mean_or(mm), mean_or(mp), mean_or(wm), mean_or(wp),
                mean_or(dm), mean_or(dp), length(wm), count(j -> wet3(MID, j, Nz), 1:Ny), length(wp))
    end

    # ---- figures: MLD and the WW core along the section ----
    let t = last(days)
        Ts = Float64.(coalesce.(dsec["T"][1, :, :, t], NaN))
        Ss = Float64.(coalesce.(dsec["S"][1, :, :, t], NaN))
        tb = model_times[t] + offset
        Tps = Array(interior(Tb[Time(tb)]))[MID, :, :]
        Sps = Array(interior(Sb[Time(tb)]))[MID, :, :]

        fig = Figure(size = (1250, 820))
        Label(fig[0, 1:2], "$(RUN) — day $(t-1) at $(round(λ₁ + (MID-0.5)/SCALING, digits=1))°E: model vs B-SOSE",
              fontsize = 19, font = :bold)

        ax1 = Axis(fig[1, 1:2], title = "mixed layer depth", xlabel = "latitude",
                   ylabel = "MLD (m)", yreversed = true)
        jm = [j for j in 1:Ny if wet3(MID, j, Nz)]
        lines!(ax1, φc[jm], [mld(Ts[j, :], Ss[j, :], zc) for j in jm],
               color = :firebrick, linewidth = 2, label = "model")
        lines!(ax1, φc[jm], [mld(Tps[j, :], Sps[j, :], zc) for j in jm],
               color = :black, linestyle = :dash, linewidth = 2, label = "B-SOSE")
        axislegend(ax1, position = :lb)

        for (col, (A, ttl)) in enumerate(((Ts, "model T"), (Tps, "B-SOSE T")))
            ax = Axis(fig[2, col], title = ttl, xlabel = "latitude", ylabel = "z (m)")
            hm = heatmap!(ax, φc, zc, A, colormap = :thermal,
                          colorrange = (-2, 8), nan_color = :gray85)
            ylims!(ax, -700, 0)
            col == 2 && Colorbar(fig[2, 3], hm, label = "°C")
            # the WW core, where one exists
            for j in jm
                w = winter_water(A, zc, mld(A[j, :], (col == 1 ? Ss : Sps)[j, :], zc))
                w === nothing || scatter!(ax, [φc[j]], [-w[2]], color = :cyan, markersize = 4)
            end
        end
        Label(fig[3, 1:2], "cyan dots: subsurface temperature minimum (Winter Water core) where one exists",
              fontsize = 12)
        f = joinpath(OUTDIR, "$(RUN)_section_day$(lpad(last(days)-1, 3, '0')).png")
        save(f, fig); @info "wrote $f"
    end
    close(dsec)
else
    @warn "no $sect_file -- skipping MLD and Winter Water. They need the midlon section."
end

# =============================================================== 4. drift
println()
println("="^96)
println("4. SURFACE DRIFT: wet-mean model minus B-SOSE, whole domain, over time")
println("="^96)
@printf("\n%6s | %10s %10s | %10s %10s | %9s %9s\n",
        "day", "T model", "T parent", "S model", "S parent", "ΔT", "ΔS")
println("-"^78)
for t in 1:nt
    (t == 1 || t == nt || t % max(1, cld(nt, 12)) == 0) || continue
    Tm, Sm = slab(ds, "T", t), slab(ds, "S", t)
    tb = model_times[t] + offset
    Tp = Array(interior(Tb[Time(tb)]))[:, :, Nz]
    Sp = Array(interior(Sb[Time(tb)]))[:, :, Nz]
    m = [wet3(i, j, Nz) for i in 1:Nx, j in 1:Ny]
    mo(A) = (f = filter(isfinite, A[m]); isempty(f) ? NaN : mean(f))
    @printf("%6d | %10.3f %10.3f | %10.4f %10.4f | %+9.3f %+9.4f\n",
            t - 1, mo(Tm), mo(Tp), mo(Sm), mo(Sp), mo(Tm) - mo(Tp), mo(Sm) - mo(Sp))
end
close(ds)

println("""

Reading this:
  * Section 1 is the only test of the sponge. A knee near $SPONGE_CELLS cells means it is working.
    No knee -- the same error at 2 cells as at 60 -- means it is not, and nothing downstream of
    the boundary conditions can be interpreted.
  * Section 2: if the model MLD is a few metres where B-SOSE is tens, PENETRATING_SHORTWAVE is
    not doing its job and the warm-bias failure of section 3.3 has returned. That is the single
    most important number in this report for the science.
  * Section 3: a WW column count far below the parent's means the model is not forming or is not
    preserving the temperature minimum. Check the Tmin DEPTH as well as its value -- too deep is
    as wrong as too warm, and it is the vertical grid's job to resolve it.
  * Section 4: a steady drift in the domain mean with no boundary imbalance points at surface
    forcing, not at the boundaries.

Caveats that are not this script's fault:
  * NO SEA ICE. B-SOSE's TFLUX includes the effect of an ice cover this model does not have, and
    a June-February run at 70-45S is ice-covered over much of its southern half in winter. Expect
    the largest surface discrepancies there, and do not read them as a boundary problem.
  * TEOS10 vs MITgcm conventions -- see the header. Threshold MLD is robust to it; absolute
    densities and few-hundredths-of-a-degree Tmin differences are not.
""")
