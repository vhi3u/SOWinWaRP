# ==============================================================================
# SUPERSEDED by tools/compare_offline.jl, which does everything this does plus the per-arm
# figures and a coverage table. Both share tools/bsose_masks.jl, so the four figures here are
# masked against both bathymetries exactly as the ones there are.
#
# compare_surface_tests.jl -- the surface-forcing ladder, all arms on one set of axes.
#
#     julia --project=. tools/compare_surface_tests.jl
#     RUNS=none,wind,wind_fw julia --project=. tools/compare_surface_tests.jl
#
# `run_surface_tests` submits five arms that differ by exactly one forcing term each. This puts
# them side by side so the question "what does THIS term do" is answered by reading across a row
# instead of flipping between files.
#
#   surface_tests_T.png          Top row: B-SOSE's own monthly mean. Every row below: that arm
#   surface_tests_S.png          minus B-SOSE. Columns are months, one shared diverging scale
#   surface_tests_midlon_T.png   across all the difference rows so arms are comparable to each
#   surface_tests_midlon_S.png   other. Surface and the 119.9E section, for T and for S.
#
# Plus a table of domain-mean and RMS bias per arm per month. Read the table first: the mean says
# whether an arm is biased, the RMS says whether it is merely in the wrong place. An arm whose
# mean is near zero and whose RMS is large has the right forcing and the wrong eddies.
#
# ENV
#     RUNS    comma-separated arm names   (default: the five run_surface_tests submits)
#     DIR     where the .nc files live    (default: surface_tests)
#     MONTHS  calendar months to show     (default: 6,7,8,9,10,11,12,1 -- the whole run)
#     OUTDIR  figures                     (default: tools/bsose_comparison)
# ==============================================================================

using Oceananigans, Oceananigans.Units, Oceananigans.Grids
using Oceananigans.Units: Time
using NumericalEarth, NumericalEarth.ECCO
using NCDatasets, CairoMakie, Printf, Statistics, Dates

include(joinpath(@__DIR__, "..", "src", "setup_bsose.jl"))
include(joinpath(@__DIR__, "bsose_masks.jl"))

# The production grid, same literals as compare_bsose.jl:153-154 and src/model.jl. Repeated
# rather than imported because neither file exports them; if model.jl's grid changes, these
# change too, and the Nx/Ny guard below is what catches it.
const λ₁, λ₂, φ₁, φ₂ = 90.0, 150.0, -70.0, -45.0
const Z_EXTENT, DZ_SURFACE, DZ_BOTTOM, Z_CONST = 5800, 8, 400, 250

const RUNS = split(get(ENV, "RUNS", "none,wind,wind_tsflux,wind_fw,wind_fw_sw,wind_qsw_fw,wind_fw_sw_freeze"), ",")
const DIR    = get(ENV, "DIR", "surface_tests")
const OUTDIR = get(ENV, "OUTDIR", joinpath(@__DIR__, "bsose_comparison"))
const MONTHS = parse.(Int, split(get(ENV, "MONTHS", "6,7,8,9,10,11,12,1"), ","))
mkpath(OUTDIR)

# ── the model grid, rebuilt exactly as compare_bsose.jl rebuilds it ───────────────────────────
present = [r for r in RUNS if isfile(joinpath(DIR, "$(r)_surface.nc"))]
isempty(present) && error("no $(DIR)/<run>_surface.nc for any of $(join(RUNS, ", "))")
for r in RUNS
    r in present || @warn "$r has no $(DIR)/$(r)_surface.nc -- skipping (still running, or it failed)"
end

# The reference arm defines the time axis every arm is grouped against, so it has to be the
# LONGEST one. `first(present)` would let a single arm that died in month 3 silently truncate the
# whole figure to three columns and look like a deliberate choice.
function ntime(r)
    ds = Dataset(joinpath(DIR, "$(r)_surface.nc"))
    n = length(ds["time"])
    close(ds)
    return n
end
nt = Dict(r => ntime(r) for r in present)
refarm = present[argmax([nt[r] for r in present])]
@info "output records per arm: " * join(("$r=$(nt[r])" for r in present), ", ") *
      " (time axis from $refarm)"
for r in present
    nt[r] == nt[refarm] ||
        @warn "$r stopped at output $(nt[r]) of $(nt[refarm]) -- its later months will be blank"
end

ds0 = Dataset(joinpath(DIR, "$(refarm)_surface.nc"))
nxf = size(ds0["T"], 1)
SCALING = Int(nxf / (λ₂ - λ₁))
z = ReferenceToStretchedDiscretization(; extent = Z_EXTENT, constant_spacing = DZ_SURFACE,
                                       maximum_spacing = DZ_BOTTOM,
                                       constant_spacing_extent = Z_CONST,
                                       stretching = PowerLawStretching(1.12))
Nx, Ny, Nz = Int(SCALING * (λ₂ - λ₁)), Int(SCALING * (φ₂ - φ₁)), length(z)
@info "grid $(Nx)x$(Ny)x$(Nz) at 1/$(SCALING) deg; arms present: $(join(present, ", "))"
(nxf == Nx && size(ds0["T"], 2) == Ny) || error(
    "grid mismatch: the output is $(nxf)x$(size(ds0["T"], 2)) but a 1/$(SCALING) deg grid over " *
    "$(λ₁)-$(λ₂)E, $(φ₁)-$(φ₂)N is $(Nx)x$(Ny). The grid constants at the top of this file are stale.")

grid = LatitudeLongitudeGrid(CPU(); size = (Nx, Ny, Nz), longitude = (λ₁, λ₂),
                             latitude = (φ₁, φ₂), z, halo = (7, 7, 7))
bottom = regrid_bathymetry(grid; height_above_water = 1, minimum_depth = 10,
                           interpolation_passes = 5)
fix_boundary_topography!(bottom, grid; minimum_boundary_depth = 100)
grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom))

λc = Array(λnodes(grid, Center())); φc = Array(φnodes(grid, Center()))
zc = Array(znodes(grid, Center())); MID = div(Nx, 2)

# ── which B-SOSE records, and which model outputs belong to each ──────────────────────────────
start_date  = DateTime("2014-06-01")
model_times = Float64.(ds0["time"][:])
run_end     = start_date + Second(round(Int, maximum(model_times)))
dataset     = BSOSEMonthly(iteration = 156)
available   = all_dates(dataset, :temperature)
inside      = findall(d -> start_date <= bsose_window_center(d) <= run_end, available)
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

groups = month_groups(model_times, bc_dates, start_date)
sel = [n for (n, dd) in enumerate(bc_dates)
       if !isempty(groups[n]) && month(bsose_window_center(dd)) in MONTHS]
isempty(sel) && error("no B-SOSE record matches month(s) $(join(MONTHS, ", ")) inside this run")
sel_dates = bc_dates[sel]
pos = Dict(n => k for (k, n) in enumerate(sel))
mlabels = [Dates.format(bsose_window_center(d), "yyyy-mm") for d in sel_dates]
@info "months: $(join(mlabels, ", "))"

region = bsose_region(grid)
resident(v) = FieldTimeSeries(Metadata(v; dataset, dates = sel_dates, region), grid;
                              time_indices_in_memory = length(sel_dates))
@info "loading $(length(sel)) B-SOSE records for T and S (~$(round(2 * length(sel) * 33e6 / 1e9, digits = 1)) GB)"
Tb, Sb = resident(:temperature), resident(:salinity)

# ── per-arm monthly means ─────────────────────────────────────────────────────────────────────
nm = length(sel)
# The surface writer is `indices=(:, :, grid.Nz)` (src/model.jl:1124), and NetCDFWriter KEEPS
# that collapsed axis as a singleton: the variable is (x, y, 1, time), not (x, y, time). So
# `v[:, :, t]` indexes z, not time, and throws a BoundsError on the very first month. Same `ndims`
# dispatch as compare_bsose.jl:240 -- that script reads these identical files without complaint,
# and copying this helper without its dispatch is what cost the first summary job.
slab(ds, v, t) = (va = ds[v]; A = ndims(va) == 4 ? va[:, :, 1, t] : va[:, :, t];
                  Float64.(coalesce.(A, NaN)))
sect(ds, v, t) = Float64.(coalesce.(ds[v][1, :, :, t], NaN))   # (1, y, z, time), always 4-D

surfT = Dict{String,Vector{Matrix{Float64}}}()   # arm -> per-month (Nx, Ny)
surfS = Dict{String,Vector{Matrix{Float64}}}()
sectT = Dict{String,Vector{Matrix{Float64}}}()   # arm -> per-month (Ny, Nz)
sectS = Dict{String,Vector{Matrix{Float64}}}()

for r in present
    dsu = Dataset(joinpath(DIR, "$(r)_surface.nc"))
    sect_path = joinpath(DIR, "$(r)_midlon.nc")
    dss = isfile(sect_path) ? Dataset(sect_path) : nothing
    dss === nothing && @warn "$r has no _midlon.nc -- it drops out of the section figures"

    surfT[r] = Matrix{Float64}[]; surfS[r] = Matrix{Float64}[]
    sectT[r] = Matrix{Float64}[]; sectS[r] = Matrix{Float64}[]

    # `groups` was built from the longest arm, so an arm that died early has indices pointing
    # past its own end -- the same BoundsError in a different costume. Trim to this arm's axis;
    # a month with nothing left becomes an all-NaN panel, drawn gray, which reads as "this arm
    # never got here" instead of as a bias of zero.
    ntu = length(dsu["time"])
    nts = dss === nothing ? 0 : length(dss["time"])
    blank(dims) = fill(NaN, dims)

    for n in sel
        mlab = Dates.format(bsose_window_center(bc_dates[n]), "yyyy-mm")
        ts = [t for t in groups[n] if t <= ntu]
        if isempty(ts)
            @warn "$r $mlab: no output at all -- that panel is blank"
            push!(surfT[r], blank((Nx, Ny))); push!(surfS[r], blank((Nx, Ny)))
        else
            length(ts) < 15 && @warn "$r $mlab has only $(length(ts)) output(s): " *
                                     "that panel is a snapshot, not a mean"
            push!(surfT[r], mean(slab(dsu, "T", t) for t in ts))
            push!(surfS[r], mean(slab(dsu, "S", t) for t in ts))
        end
        if dss !== nothing
            tss = [t for t in groups[n] if t <= nts]
            if isempty(tss)
                push!(sectT[r], blank((Ny, Nz))); push!(sectS[r], blank((Ny, Nz)))
            else
                push!(sectT[r], mean(sect(dss, "T", t) for t in tss))
                push!(sectS[r], mean(sect(dss, "S", t) for t in tss))
            end
        end
    end
    close(dsu); dss === nothing || close(dss)
end

# B-SOSE on the same grid
parT = [Array(interior(Tb[pos[n]]))[:, :, Nz] for n in sel]
parS = [Array(interior(Sb[pos[n]]))[:, :, Nz] for n in sel]
parTsec = [Array(interior(Tb[pos[n]]))[MID, :, :] for n in sel]
parSsec = [Array(interior(Sb[pos[n]]))[MID, :, :] for n in sel]

# BATHYMETRY, NOT `isfinite`. These panels used to be masked on "both sides are finite", which
# masks nothing: B-SOSE's T and S are finite below its own sea floor (fill_bathymetry_gaps!
# propagates the deepest water mass downward) and Oceananigans writes immersed tracer cells as
# zeros, not NaN. The section panels therefore drew the Antarctic margin and everything under
# the sea floor as though it were ocean. Both masks now come from the two grids' own
# bathymetries -- see tools/bsose_masks.jl, which compare_offline.jl shares.
#
# The mask is the same every month; it is held per month only because `ladder_figure` indexes
# it that way. A month an arm never reached is still blank, because its data is NaN and
# `nan_color` catches it.
bathy_surface = both_wet_surface(grid, dataset, λc, φc)
wetmask = [bathy_surface for _ in 1:nm]
@info @sprintf("surface mask: %d of %d cells wet in both grids", count(bathy_surface), Nx * Ny)

# ── table ─────────────────────────────────────────────────────────────────────────────────────
function stats(D, m)
    v = filter(isfinite, D[m])
    isempty(v) ? (NaN, NaN) : (mean(v), sqrt(mean(abs2, v)))
end

println()
println("="^110)
println("SURFACE BIAS BY ARM AND MONTH -- monthly mean model minus the monthly B-SOSE record")
println("  mean = is it biased.  rms = is it merely in the wrong place.")
println("  An arm with mean near zero and a large rms has the right forcing and the wrong eddies.")
println("="^110)
for (fld, name, unit, surf, par) in (("T", "ΔT", "°C", surfT, parT),
                                     ("S", "ΔS", "g/kg", surfS, parS))
    @printf("\n  %s (%s)\n", name, unit)
    @printf("  %-14s |", "arm")
    for l in mlabels; @printf("%17s", l); end
    println()
    @printf("  %-14s |", "")
    for _ in mlabels; @printf("%9s%8s", "mean", "rms"); end
    println()
    println("  " * "-"^(17 + 17 * nm))
    for r in present
        @printf("  %-14s |", r)
        for k in 1:nm
            mu, rms = stats(surf[r][k] .- par[k], wetmask[k])
            @printf("%9.3f%8.3f", mu, rms)
        end
        println()
    end
end

# ── figures ───────────────────────────────────────────────────────────────────────────────────
#
# ONE REFERENCE ROW, THEN ONE DIFFERENCE ROW PER ARM. Showing every arm's absolute field as well
# would double the panel count for nothing: at any sensible colour scale the arms are visually
# identical to each other and to B-SOSE -- the whole signal is in the differences. So B-SOSE's
# own field goes on top, once, to say what the water actually looks like, and everything below
# it is model minus B-SOSE on a single shared diverging scale so the rows are comparable.
#
# Same function for the surface and the section; only the axes and the masks differ.

"""
    ladder_figure(ref, perarm, arms, mask, name, unit, refcmap, xs, ys, xlab, ylab, fname; ylim)

`ref` is B-SOSE's own monthly field, `perarm[arm]` the model's. Both are vectors of one matrix
per month. Row 1 is `ref` on its own colour range; rows 2+ are `perarm[arm] - ref`, all sharing
one symmetric range taken from the 99th percentile so a single outlier cannot flatten the rest.
"""
function ladder_figure(ref, perarm, arms, mask, name, unit, refcmap, xs, ys, xlab, ylab, fname;
                       ylim = nothing)
    isempty(arms) && (@warn "no arm has data for $fname -- skipping"; return)

    masked(A, k) = (B = copy(A); B[.!mask[k]] .= NaN; B)
    R = [masked(ref[k], k) for k in 1:nm]
    D = Dict(r => [masked(perarm[r][k] .- ref[k], k) for k in 1:nm] for r in arms)

    rfin = filter(isfinite, vcat(vec.(R)...))
    rrange = isempty(rfin) ? (0.0, 1.0) : (quantile(rfin, 0.01), quantile(rfin, 0.99))
    dfin = filter(isfinite, vcat([vcat(vec.(D[r])...) for r in arms]...))
    lim = isempty(dfin) ? 1.0 : max(quantile(abs.(dfin), 0.99), 1e-12)

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

# The section needs its own mask: different shape, and the two bathymetries disagree with depth
# as well as in plan. SECT_DEPTH matches the `ylim` the section panels are drawn with, so the
# mask covers exactly what is on screen and no more.
const SECT_DEPTH = 700.0
sect_arms = [r for r in present if !isempty(sectT[r])]
bathy_section, _ = both_wet_section(grid, dataset, MID, φc, zc, SECT_DEPTH)
sectmask = isempty(sect_arms) ? Vector{BitMatrix}() : [bathy_section for _ in 1:nm]

ladder_figure(parT, surfT, present, wetmask, "T (°C), surface", "°C", :thermal,
              λc, φc, "longitude (°E)", "latitude (°N)", "surface_tests_T.png")
ladder_figure(parS, surfS, present, wetmask, "S (g/kg), surface", "g/kg", :haline,
              λc, φc, "longitude (°E)", "latitude (°N)", "surface_tests_S.png")
ladder_figure(parTsec, sectT, sect_arms, sectmask,
              "T (°C) at $(round(λ₁ + (MID - 0.5) / SCALING, digits = 1))°E", "°C", :thermal,
              φc, zc, "latitude (°N)", "z (m)", "surface_tests_midlon_T.png";
              ylim = (-SECT_DEPTH, 0))
ladder_figure(parSsec, sectS, sect_arms, sectmask,
              "S (g/kg) at $(round(λ₁ + (MID - 0.5) / SCALING, digits = 1))°E", "g/kg", :haline,
              φc, zc, "latitude (°N)", "z (m)", "surface_tests_midlon_S.png";
              ylim = (-SECT_DEPTH, 0))

println("\nReading this:")
println("  * `none` is the control. Any drift it shows is the boundaries and the sponge, not the")
println("    surface -- subtract it mentally from every other arm.")
println("  * Each arm adds ONE term to the one above. The row where a feature first appears is")
println("    the term that causes it.")
println("  * Mean vs rms in the table separates a bias from a displaced eddy field.")
