# ==============================================================================
# compare_offline.jl -- monthly model mean MINUS monthly B-SOSE mean, for every arm, offline.
#
#     julia --project=. tools/compare_offline.jl
#     RUNS=none,wind MONTHS=8,9,10 julia --project=. tools/compare_offline.jl
#
# WHY THIS IS A SEPARATE SCRIPT. The comparison used to run inside each simulation job, right
# after the run finished. That coupled two things that fail for unrelated reasons: an arm that
# completed could still show no figures because the plotting died, and an arm that died early
# produced nothing at all even though the months it DID reach were perfectly good. It also meant
# the figures could only ever be made once, on a compute node, at a moment nobody was watching.
#
# This runs on the login node, over whatever is on disk, as many times as you like. It builds
# the grid and loads B-SOSE ONCE and then writes every figure, so re-running after another arm
# lands costs one grid build, not one per arm.
#
# WHAT IT WRITES, into OUTDIR:
#     <arm>_surface_diff.png   2 rows (ΔT, ΔS) x one column per month, model minus B-SOSE
#     <arm>_section_diff.png   the same through the upper Z_TOP m at the mid-domain meridian
#     ladder_surface_T.png     every arm on one set of axes: B-SOSE's own field on top,
#     ladder_surface_S.png     each arm minus B-SOSE below it, one shared diverging scale
#     ladder_section_T.png
#     ladder_section_S.png
# plus a COVERAGE table and a bias table on stdout. Read the coverage table first: it is the
# answer to "why is that panel grey", and it is printed before anything slow happens.
#
# TWO BATHYMETRIES, NOT ONE. Ours is ETOPO, B-SOSE's is its own, and they disagree along the
# Antarctic margin. B-SOSE's T and S are finite BELOW its sea floor -- `fill_bathymetry_gaps!`
# propagates the deepest water mass downward into every immersed cell -- so `isfinite` is not a
# test for "there is ocean here". Masking on it draws the inpainting pipeline as if it were the
# state estimate. Every mask here is built from both grids' own floors, never from NaN.
#
# ENV
#     RUNS    comma-separated arm names   (default: the seven run_surface_tests submits)
#     DIR     where the .nc files live    (default: surface_tests)
#     MONTHS  calendar months to show     (default: 6,7,8,9,10,11,12,1 -- the whole run)
#     OUTDIR  figures                     (default: surface_tests/figs)
#     START   first day of the run        (default: 2014-06-01)
#     Z_TOP   section depth, m            (default: 200)
#     SCALE   shared | panel               (default: shared -- one colour range across months;
#                                           `panel` gives each month its own, for when one
#                                           month is so far off that it whites out the rest)
# ==============================================================================

using Oceananigans, Oceananigans.Units, Oceananigans.Grids
using NumericalEarth, NumericalEarth.ECCO
using NCDatasets, CairoMakie, Printf, Statistics, Dates

include(joinpath(@__DIR__, "..", "src", "setup_bsose.jl"))
include(joinpath(@__DIR__, "bsose_masks.jl"))

# The production grid, same literals as src/model.jl. Repeated rather than imported because
# model.jl does not export them; the Nx/Ny guard below is what catches them going stale.
const λ₁, λ₂, φ₁, φ₂ = 90.0, 150.0, -70.0, -45.0
const Z_EXTENT, DZ_SURFACE, DZ_BOTTOM, Z_CONST = 5800, 8, 400, 250

const RUNS = split(get(ENV, "RUNS",
                       "none,wind,wind_tsflux,wind_fw,wind_fw_sw,wind_qsw_fw,wind_fw_sw_freeze"), ",")
const DIR    = get(ENV, "DIR", "surface_tests")
const OUTDIR = get(ENV, "OUTDIR", joinpath(DIR, "figs"))
const MONTHS = parse.(Int, split(get(ENV, "MONTHS", "6,7,8,9,10,11,12,1"), ","))
const START  = DateTime(get(ENV, "START", "2014-06-01"))
const Z_TOP  = parse(Float64, get(ENV, "Z_TOP", "200"))
const SCALE  = get(ENV, "SCALE", "shared")
SCALE in ("shared", "panel") || error("SCALE must be \"shared\" or \"panel\", got \"$SCALE\"")
mkpath(OUTDIR)

# ── which arms are on disk, and how far each one got ──────────────────────────────────────────
surface_path(r) = joinpath(DIR, "$(r)_surface.nc")
section_path(r) = joinpath(DIR, "$(r)_midlon.nc")

function record_count(path)
    isfile(path) || return 0
    ds = Dataset(path)
    n = length(ds["time"])
    close(ds)
    return n
end

present = [r for r in RUNS if isfile(surface_path(r))]
isempty(present) && error("no $(DIR)/<arm>_surface.nc for any of $(join(RUNS, ", "))")
for r in RUNS
    r in present || @warn "$r has no $(surface_path(r)) -- it is not in this comparison"
end

n_surface = Dict(r => record_count(surface_path(r)) for r in present)
n_section = Dict(r => record_count(section_path(r)) for r in present)

# The reference arm defines the time axis every arm is grouped against, so it has to be the
# LONGEST one. Keying off the first arm instead would let a single arm that died in month 3
# silently truncate the whole figure to three columns and look like a deliberate choice.
refarm = present[argmax([n_surface[r] for r in present])]

@info "arms on disk: " * join(("$r=$(n_surface[r])" for r in present), ", ") *
      " surface records (time axis from $refarm)"

# ── the model grid, rebuilt exactly as model.jl builds it ─────────────────────────────────────
ds0 = Dataset(surface_path(refarm))
nxf = size(ds0["T"], 1)
SCALING = Int(nxf / (λ₂ - λ₁))
z = ReferenceToStretchedDiscretization(; extent = Z_EXTENT, constant_spacing = DZ_SURFACE,
                                       maximum_spacing = DZ_BOTTOM,
                                       constant_spacing_extent = Z_CONST,
                                       stretching = PowerLawStretching(1.12))
Nx, Ny, Nz = Int(SCALING * (λ₂ - λ₁)), Int(SCALING * (φ₂ - φ₁)), length(z)
(nxf == Nx && size(ds0["T"], 2) == Ny) || error(
    "grid mismatch: the output is $(nxf)x$(size(ds0["T"], 2)) but a 1/$(SCALING) deg grid over " *
    "$(λ₁)-$(λ₂)E, $(φ₁)-$(φ₂)N is $(Nx)x$(Ny). The grid constants at the top of this file are stale.")

@info "grid $(Nx)x$(Ny)x$(Nz) at 1/$(SCALING) deg"

grid = LatitudeLongitudeGrid(CPU(); size = (Nx, Ny, Nz), longitude = (λ₁, λ₂),
                             latitude = (φ₁, φ₂), z, halo = (7, 7, 7))
bottom = regrid_bathymetry(grid; height_above_water = 1, minimum_depth = 10,
                           interpolation_passes = 5)
fix_boundary_topography!(bottom, grid; minimum_boundary_depth = 100)
grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom))

λc = Array(λnodes(grid, Center())); φc = Array(φnodes(grid, Center()))
zc = Array(znodes(grid, Center())); MID = div(Nx, 2)
const SECT_LON = λ₁ + (MID - 0.5) / SCALING

# ── which B-SOSE records, and which model outputs belong to each ──────────────────────────────
model_times = Float64.(ds0["time"][:])
run_end     = START + Second(round(Int, maximum(model_times)))
dataset     = BSOSEMonthly(iteration = 156)
available   = all_dates(dataset, :temperature)
inside      = findall(d -> START <= bsose_window_center(d) <= run_end, available)
lo = max(firstindex(available), (isempty(inside) ? 1 : first(inside)) - 1)
hi = min(lastindex(available), (isempty(inside) ? length(available) : last(inside)) + 1)
bc_dates = available[lo:hi]

"""Model output indices belonging to each B-SOSE record, by nearest window centre."""
function month_groups(times, dates, t0)
    groups = [Int[] for _ in dates]
    for (t, s) in enumerate(times)
        d = t0 + Second(round(Int, s))
        _, best = findmin(abs(Dates.value(bsose_window_center(dd) - d)) for dd in dates)
        push!(groups[best], t)
    end
    return groups
end

groups = month_groups(model_times, bc_dates, START)
sel = [n for (n, dd) in enumerate(bc_dates)
       if !isempty(groups[n]) && month(bsose_window_center(dd)) in MONTHS]
isempty(sel) && error("no B-SOSE record matches month(s) $(join(MONTHS, ", ")) inside this run")
sel_dates = bc_dates[sel]
pos = Dict(n => k for (k, n) in enumerate(sel))
mlabels = [Dates.format(bsose_window_center(d), "yyyy-mm") for d in sel_dates]
nm = length(sel)

# How many of each arm's own outputs fall in each selected month. `groups` is built from the
# reference arm, so an arm that stopped early has indices in it that point past its own end --
# the same BoundsError in a different costume. Trim here, once, and every consumer below reads
# a count that is true for THAT arm.
take(r, n, nrec) = [t for t in groups[n] if t <= nrec]
surf_idx = Dict(r => [take(r, n, n_surface[r]) for n in sel] for r in present)
sect_idx = Dict(r => [take(r, n, n_section[r]) for n in sel] for r in present)

# ── COVERAGE, before anything slow ────────────────────────────────────────────────────────────
println()
println("="^100)
println("COVERAGE -- model outputs per arm per month, after trimming each arm to its own time axis")
println("  A number below 15 is a partial month: that panel is a snapshot, not a monthly mean.")
println("  A dash is a month the arm never reached. Its panel is drawn grey, not as a zero bias.")
println("="^100)
@printf("  %-20s |", "arm")
for l in mlabels; @printf("%9s", l); end
@printf("%10s\n", "section?")
println("  " * "-"^(23 + 9 * nm + 10))
for r in present
    @printf("  %-20s |", r)
    for k in 1:nm
        nd = length(surf_idx[r][k])
        print(nd == 0 ? lpad("--", 9) : lpad(string(nd) * (nd < 15 ? "*" : ""), 9))
    end
    @printf("%10s\n", n_section[r] == 0 ? "no" : "yes")
end
println("  " * "-"^(23 + 9 * nm + 10))
println("  * = partial month")
flush(stdout)

# ── the two sea floors ────────────────────────────────────────────────────────────────────────
# Both masks come from tools/bsose_masks.jl, which is also what the ladder script uses -- the
# "B-SOSE is finite below its own floor" trap is subtle enough to be worth having in one place.
wet_surface = both_wet_surface(grid, dataset, λc, φc)
wet_section, _ = both_wet_section(grid, dataset, MID, φc, zc, Z_TOP)
@info @sprintf("surface mask: %d of %d cells wet in both grids", count(wet_surface), Nx * Ny)

# ── B-SOSE ────────────────────────────────────────────────────────────────────────────────────
region = bsose_region(grid)
resident(v) = FieldTimeSeries(Metadata(v; dataset, dates = sel_dates, region), grid;
                              time_indices_in_memory = length(sel_dates))
@info "loading $(nm) B-SOSE records for T and S (~$(round(2 * nm * 33e6 / 1e9, digits = 1)) GB)"
Tb, Sb = resident(:temperature), resident(:salinity)

parT    = [Array(interior(Tb[pos[n]]))[:, :, Nz] for n in sel]
parS    = [Array(interior(Sb[pos[n]]))[:, :, Nz] for n in sel]
parTsec = [Array(interior(Tb[pos[n]]))[MID, :, :] for n in sel]
parSsec = [Array(interior(Sb[pos[n]]))[MID, :, :] for n in sel]

# ── per-arm monthly means ─────────────────────────────────────────────────────────────────────
#
# The surface writer is `indices=(:, :, grid.Nz)` and NetCDFWriter KEEPS that collapsed axis as
# a singleton, so the variable is (x, y, 1, time), not (x, y, time): `v[:, :, t]` would index z
# and throw on the first month. The section writer is `indices=(mid, :, :)`, always 4-D.
slab(ds, v, t) = (va = ds[v]; A = ndims(va) == 4 ? va[:, :, 1, t] : va[:, :, t];
                  Float64.(coalesce.(A, NaN)))
sect(ds, v, t) = Float64.(coalesce.(ds[v][1, :, :, t], NaN))

blank(dims) = fill(NaN, dims)

surfT = Dict{String,Vector{Matrix{Float64}}}()
surfS = Dict{String,Vector{Matrix{Float64}}}()
sectT = Dict{String,Vector{Matrix{Float64}}}()
sectS = Dict{String,Vector{Matrix{Float64}}}()

for r in present
    surfT[r] = Matrix{Float64}[]; surfS[r] = Matrix{Float64}[]
    sectT[r] = Matrix{Float64}[]; sectS[r] = Matrix{Float64}[]

    dsu = Dataset(surface_path(r))
    for k in 1:nm
        ts = surf_idx[r][k]
        if isempty(ts)
            push!(surfT[r], blank((Nx, Ny))); push!(surfS[r], blank((Nx, Ny)))
        else
            push!(surfT[r], mean(slab(dsu, "T", t) for t in ts))
            push!(surfS[r], mean(slab(dsu, "S", t) for t in ts))
        end
    end
    close(dsu)

    if n_section[r] > 0
        dss = Dataset(section_path(r))
        for k in 1:nm
            ts = sect_idx[r][k]
            if isempty(ts)
                push!(sectT[r], blank((Ny, Nz))); push!(sectS[r], blank((Ny, Nz)))
            else
                push!(sectT[r], mean(sect(dss, "T", t) for t in ts))
                push!(sectS[r], mean(sect(dss, "S", t) for t in ts))
            end
        end
        close(dss)
    end
end

sect_arms = [r for r in present if !isempty(sectT[r])]

# ── figures ───────────────────────────────────────────────────────────────────────────────────
masked(A, m) = (B = fill(NaN, size(A)); B[m] .= A[m]; B)

"""Symmetric range from the 99th percentile of |x|, so one outlier cannot flatten the rest."""
function sym_range(mats)
    fin = filter(isfinite, vcat(vec.(mats)...))
    isempty(fin) && return 1.0
    return max(quantile(abs.(fin), 0.99), 1e-12)
end

"""
    diff_figure(arm, dT, dS, xs, ys, xlab, ylab, fname; ylim)

One arm, two rows (ΔT then ΔS), one column per month, model monthly mean minus the B-SOSE
monthly record.

SCALE decides the colour range, and it is the difference between a readable figure and a row of
blank panels. With one range shared across months (SCALE=shared, the default) a single runaway
month sets the range for all of them: in the 8-month sweep `wind_qsw_fw` ends January about
10 degC warm, which renders its June, July and August panels -- all of them carrying real
signal of a few tenths -- as featureless white. That is not missing data, and it looked exactly
like missing data. So every panel is now titled with its OWN 99th-percentile magnitude, and
SCALE=panel gives each panel its own range so the shape is visible even when the amplitude is
two orders of magnitude off its neighbour.
"""
function diff_figure(arm, dT, dS, xs, ys, xlab, ylab, fname; ylim = nothing, where = "surface")
    fig = Figure(size = (300 * nm + 170, 2 * 290 + 150))
    Label(fig[0, 1:(nm + 1)],
          "$arm — monthly mean model − monthly B-SOSE, $where",
          fontsize = 18, font = :bold)

    for (row, (D, name, unit)) in enumerate(((dT, "ΔT", "°C"), (dS, "ΔS", "g/kg")))
        shared  = sym_range(D)
        perpan  = [sym_range([D[k]]) for k in 1:nm]
        local hm
        for k in 1:nm
            last_row = row == 2
            empty_k  = !any(isfinite, D[k])
            lim      = SCALE == "panel" ? perpan[k] : shared
            ax = Axis(fig[row, k],
                      title = empty_k ? "$(mlabels[k])  (no output)" :
                              @sprintf("%s  ±%.3g", mlabels[k], perpan[k]),
                      titlesize = 12,
                      ylabel = k == 1 ? "$name ($unit)" : "",
                      xlabel = last_row ? xlab : "",
                      xticksvisible = last_row, xticklabelsvisible = last_row,
                      yticklabelsvisible = k == 1)
            hm = heatmap!(ax, xs, ys, D[k], colormap = :balance,
                          colorrange = (-lim, lim), nan_color = :gray85)
            ylim === nothing || ylims!(ax, ylim...)
        end
        SCALE == "shared" && Colorbar(fig[row, nm + 1], hm, label = "Δ $unit")
    end

    # Two lines, explicitly. One long line is silently clipped at the figure edge, which is how
    # the note explaining the grey ended up half off the page.
    Label(fig[3, 1:(nm + 1)],
          "Grey: dry in our bathymetry or in B-SOSE's, or a month this arm never reached.\n" *
          "Panel titles are that panel's own 99th-percentile |Δ|, so a pale panel reads as a " *
          "small difference rather than as no data. " *
          (SCALE == "panel" ? "Colour range is PER PANEL — compare shapes, not amplitudes." :
                              "Colour range is shared across months (SCALE=panel to unshare)."),
          fontsize = 11, justification = :center)
    f = joinpath(OUTDIR, fname); save(f, fig); @info "wrote $f"
end

for r in present
    diff_figure(r, [masked(surfT[r][k] .- parT[k], wet_surface) for k in 1:nm],
                   [masked(surfS[r][k] .- parS[k], wet_surface) for k in 1:nm],
                λc, φc, "longitude (°E)", "latitude (°N)", "$(r)_surface_diff.png")
end
for r in sect_arms
    diff_figure(r, [masked(sectT[r][k] .- parTsec[k], wet_section) for k in 1:nm],
                   [masked(sectS[r][k] .- parSsec[k], wet_section) for k in 1:nm],
                φc, zc, "latitude (°N)", "z (m)", "$(r)_section_diff.png";
                ylim = (-Z_TOP, 0),
                where = @sprintf("upper %.0f m at %.1f°E", Z_TOP, SECT_LON))
end

"""
    ladder_figure(ref, perarm, arms, mask, name, unit, refcmap, xs, ys, xlab, ylab, fname; ylim)

Every arm on one set of axes. Row 1 is B-SOSE's own monthly field, on its own colour range, to
say what the water actually looks like; rows 2+ are each arm minus B-SOSE, all sharing ONE
symmetric range so the arms are comparable to each other and not just to the parent.
"""
function ladder_figure(ref, perarm, arms, mask, name, unit, refcmap, xs, ys, xlab, ylab, fname;
                       ylim = nothing)
    isempty(arms) && (@warn "no arm has data for $fname -- skipping"; return)

    R = [masked(ref[k], mask) for k in 1:nm]
    D = Dict(r => [masked(perarm[r][k] .- ref[k], mask) for k in 1:nm] for r in arms)

    rfin = filter(isfinite, vcat(vec.(R)...))
    rrange = isempty(rfin) ? (0.0, 1.0) : (quantile(rfin, 0.01), quantile(rfin, 0.99))
    lim = sym_range([D[r][k] for r in arms for k in 1:nm])

    nrow = length(arms) + 1
    fig = Figure(size = (230 * nm + 190, 195 * nrow + 150))
    Label(fig[0, 1:(nm + 1)],
          "$name — top row is B-SOSE's own monthly mean; below, each arm minus B-SOSE",
          fontsize = 18, font = :bold)

    local hmr, hmd
    for k in 1:nm
        ax = Axis(fig[1, k], title = mlabels[k], ylabel = k == 1 ? "B-SOSE" : "",
                  xticksvisible = false, xticklabelsvisible = false,
                  yticklabelsvisible = k == 1)
        hmr = heatmap!(ax, xs, ys, R[k], colormap = refcmap, colorrange = rrange,
                       nan_color = :gray85)
        ylim === nothing || ylims!(ax, ylim...)
    end
    Colorbar(fig[1, nm + 1], hmr, label = unit)

    for (i, r) in enumerate(arms), k in 1:nm
        last_row = i == length(arms)
        ax = Axis(fig[i + 1, k], ylabel = k == 1 ? r : "",
                  xlabel = last_row ? xlab : "",
                  xticksvisible = last_row, xticklabelsvisible = last_row,
                  yticklabelsvisible = k == 1)
        hmd = heatmap!(ax, xs, ys, D[r][k], colormap = :balance, colorrange = (-lim, lim),
                       nan_color = :gray85)
        ylim === nothing || ylims!(ax, ylim...)
    end
    Colorbar(fig[2:nrow, nm + 1], hmd, label = "Δ $unit")

    f = joinpath(OUTDIR, fname); save(f, fig); @info "wrote $f"
end

ladder_figure(parT, surfT, present, wet_surface, "T (°C), surface", "°C", :thermal,
              λc, φc, "longitude (°E)", "latitude (°N)", "ladder_surface_T.png")
ladder_figure(parS, surfS, present, wet_surface, "S (g/kg), surface", "g/kg", :haline,
              λc, φc, "longitude (°E)", "latitude (°N)", "ladder_surface_S.png")
ladder_figure(parTsec, sectT, sect_arms, wet_section,
              @sprintf("T (°C) at %.1f°E", SECT_LON), "°C", :thermal,
              φc, zc, "latitude (°N)", "z (m)", "ladder_section_T.png"; ylim = (-Z_TOP, 0))
ladder_figure(parSsec, sectS, sect_arms, wet_section,
              @sprintf("S (g/kg) at %.1f°E", SECT_LON), "g/kg", :haline,
              φc, zc, "latitude (°N)", "z (m)", "ladder_section_S.png"; ylim = (-Z_TOP, 0))

# ── bias table ────────────────────────────────────────────────────────────────────────────────
"""cos(lat)-weighted mean and rms of `D` over `m`; NaN when nothing is left."""
function stats(D, m, weights)
    num = den = sq = 0.0
    for i in eachindex(D)
        (m[i] && isfinite(D[i])) || continue
        w = weights[i]
        num += w * D[i]
        sq  += w * D[i]^2
        den += w
    end
    return den == 0 ? (NaN, NaN) : (num / den, sqrt(sq / den))
end

wsurf = [cosd(φc[j]) for _ in 1:Nx, j in 1:Ny]

println()
println("="^110)
println("BIAS BY ARM AND MONTH -- monthly mean model minus the monthly B-SOSE record, surface")
println("  cos(lat)-weighted over cells wet in BOTH bathymetries.")
println("  mean = is it biased.  rms = is it merely in the wrong place.")
println("  An arm with mean near zero and a large rms has the right forcing and the wrong eddies.")
println("="^110)
for (name, unit, surf, par) in (("ΔT", "°C", surfT, parT), ("ΔS", "g/kg", surfS, parS))
    @printf("\n  %s (%s)\n", name, unit)
    @printf("  %-20s |", "arm")
    for l in mlabels; @printf("%17s", l); end
    println()
    @printf("  %-20s |", "")
    for _ in mlabels; @printf("%9s%8s", "mean", "rms"); end
    println()
    println("  " * "-"^(23 + 17 * nm))
    for r in present
        @printf("  %-20s |", r)
        for k in 1:nm
            mu, rms = stats(surf[r][k] .- par[k], wet_surface, wsurf)
            isnan(mu) ? @printf("%9s%8s", "--", "--") : @printf("%9.3f%8.3f", mu, rms)
        end
        println()
    end
end

println()
println("Reading this:")
println("  * `none` is the control -- no surface forcing at all. Whatever drift it shows is the")
println("    boundaries and the sponge, not the surface. Subtract it mentally from every arm.")
println("  * Each arm adds ONE term to the one above it. The row where a feature first appears")
println("    is the term that causes it.")
println("  * A `--` is a month that arm never reached, not a zero bias. See the coverage table.")
