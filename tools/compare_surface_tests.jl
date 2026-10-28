# ==============================================================================
# compare_surface_tests.jl -- the surface-forcing ladder, all arms on one set of axes.
#
#     julia --project=. tools/compare_surface_tests.jl
#     RUNS=none,wind,wind_fw julia --project=. tools/compare_surface_tests.jl
#
# `run_surface_tests` submits five arms that differ by exactly one forcing term each. This puts
# them side by side so the question "what does THIS term do" is answered by reading across a row
# instead of flipping between files.
#
#   surface_tests_dT.png     rows = arms, cols = months. Monthly mean model T minus the monthly
#   surface_tests_dS.png     B-SOSE record, at the surface. One colour range for the whole grid.
#   surface_tests_midlon_T.png   rows = arms PLUS B-SOSE, cols = months. The 119.9E section
#   surface_tests_midlon_S.png   itself, not a difference, so the parent is directly comparable.
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

# The production grid, same literals as compare_bsose.jl:153-154 and src/model.jl. Repeated
# rather than imported because neither file exports them; if model.jl's grid changes, these
# change too, and the Nx/Ny guard below is what catches it.
const λ₁, λ₂, φ₁, φ₂ = 90.0, 150.0, -70.0, -45.0
const Z_EXTENT, DZ_SURFACE, DZ_BOTTOM, Z_CONST = 5800, 8, 400, 250

const RUNS = split(get(ENV, "RUNS", "none,wind,wind_tsflux,wind_fw,wind_fw_sw"), ",")
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

ds0 = Dataset(joinpath(DIR, "$(first(present))_surface.nc"))
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
slab(ds, v, t) = Float64.(coalesce.(ds[v][:, :, t], NaN))
sect(ds, v, t) = Float64.(coalesce.(ds[v][1, :, :, t], NaN))

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

    for n in sel
        ts = groups[n]
        length(ts) < 15 && @warn "$r $(Dates.format(bsose_window_center(bc_dates[n]), "yyyy-mm")) " *
                                 "has only $(length(ts)) output(s): that panel is a snapshot, not a mean"
        push!(surfT[r], mean(slab(dsu, "T", t) for t in ts))
        push!(surfS[r], mean(slab(dsu, "S", t) for t in ts))
        if dss !== nothing
            push!(sectT[r], mean(sect(dss, "T", t) for t in ts))
            push!(sectS[r], mean(sect(dss, "S", t) for t in ts))
        end
    end
    close(dsu); dss === nothing || close(dss)
end

# B-SOSE on the same grid
parT = [Array(interior(Tb[pos[n]]))[:, :, Nz] for n in sel]
parS = [Array(interior(Sb[pos[n]]))[:, :, Nz] for n in sel]
parTsec = [Array(interior(Tb[pos[n]]))[MID, :, :] for n in sel]
parSsec = [Array(interior(Sb[pos[n]]))[MID, :, :] for n in sel]

# wet where BOTH have data, per cell -- the two bathymetries disagree
wetmask = [isfinite.(parT[k]) .& isfinite.(surfT[first(present)][k]) for k in 1:nm]

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
"""Grid of difference maps: one row per arm, one column per month, one shared colour range."""
function diff_figure(surf, par, name, unit, fname)
    Ds = Dict(r => [let D = surf[r][k] .- par[k]; D[.!wetmask[k]] .= NaN; D end for k in 1:nm]
              for r in present)
    fin = filter(isfinite, vcat([vcat(vec.(Ds[r])...) for r in present]...))
    lim = isempty(fin) ? 1.0 : quantile(abs.(fin), 0.99)

    fig = Figure(size = (230 * nm + 140, 210 * length(present) + 90))
    Label(fig[0, 1:(nm + 1)], "$name ($unit): monthly mean model − monthly B-SOSE, surface",
          fontsize = 18, font = :bold)
    local hm
    for (i, r) in enumerate(present), k in 1:nm
        ax = Axis(fig[i, k], title = i == 1 ? mlabels[k] : "",
                  ylabel = k == 1 ? r : "", xticksvisible = false, yticksvisible = false,
                  xticklabelsvisible = false, yticklabelsvisible = false)
        hm = heatmap!(ax, λc, φc, Ds[r][k], colormap = :balance,
                      colorrange = (-lim, lim), nan_color = :gray85)
    end
    Colorbar(fig[1:length(present), nm + 1], hm, label = unit)
    f = joinpath(OUTDIR, fname); save(f, fig); @info "wrote $f"
end

diff_figure(surfT, parT, "ΔT", "°C",   "surface_tests_dT.png")
diff_figure(surfS, parS, "ΔS", "g/kg", "surface_tests_dS.png")

"""Section grid: one row per arm PLUS a B-SOSE row, one column per month. Values, not differences."""
function section_figure(sect, parsec, name, unit, cmap, fname)
    rows = [r for r in present if !isempty(sect[r])]
    isempty(rows) && (@warn "no _midlon.nc for any arm -- skipping $fname"; return)
    all_ = vcat([vcat(vec.(sect[r])...) for r in rows]..., vcat(vec.(parsec)...))
    fin = filter(isfinite, all_)
    cr = isempty(fin) ? (0.0, 1.0) : (quantile(fin, 0.01), quantile(fin, 0.99))

    nr = length(rows) + 1
    fig = Figure(size = (230 * nm + 140, 200 * nr + 90))
    Label(fig[0, 1:(nm + 1)], "$name ($unit) at $(round(λ₁ + (MID - 0.5) / SCALING, digits = 1))°E, " *
          "monthly mean; bottom row is B-SOSE", fontsize = 18, font = :bold)
    local hm
    for (i, r) in enumerate(vcat(rows, "B-SOSE")), k in 1:nm
        A = r == "B-SOSE" ? parsec[k] : sect[r][k]
        ax = Axis(fig[i, k], title = i == 1 ? mlabels[k] : "",
                  ylabel = k == 1 ? r : "", xticksvisible = false,
                  xticklabelsvisible = false, yticklabelsvisible = k == 1)
        hm = heatmap!(ax, φc, zc, A, colormap = cmap, colorrange = cr, nan_color = :gray85)
        ylims!(ax, -700, 0)
    end
    Colorbar(fig[1:nr, nm + 1], hm, label = unit)
    f = joinpath(OUTDIR, fname); save(f, fig); @info "wrote $f"
end

section_figure(sectT, parTsec, "T", "°C",   :thermal, "surface_tests_midlon_T.png")
section_figure(sectS, parSsec, "S", "g/kg", :haline,  "surface_tests_midlon_S.png")

println("\nReading this:")
println("  * `none` is the control. Any drift it shows is the boundaries and the sponge, not the")
println("    surface -- subtract it mentally from every other arm.")
println("  * Each arm adds ONE term to the one above. The row where a feature first appears is")
println("    the term that causes it.")
println("  * Mean vs rms in the table separates a bias from a displaced eddy field.")
