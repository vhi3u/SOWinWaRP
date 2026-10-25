# ==============================================================================
# face_transports.jl -- is the pinned transport target representative, or is it holding a
#                       face against its own seasonal sign reversal?
#
#     julia --project=. tools/face_transports.jl
#     DATES=2013-01-01,2016-01-01 julia --project=. tools/face_transports.jl
#
# WHY THIS EXISTS. `BAROTROPIC_SCHEME = :pinned` hands Flather a `target_transport`, and
# `setup_bsose.jl` is explicit about the compromise: the target is materialised ONCE when the
# free surface is built, so it is a single number per face for the whole run. Its own comment
# records what that costs -- "we pin the time mean and lose the seasonal swing of the net
# (west +183 to +212, north -31 to +16)".
#
# **North reverses sign over a year.** If the run window sits on the far side of that reversal
# from the mean, the pin is not a harmless simplification: it holds a constant inflow through a
# face that should be flowing out, every barotropic substep, for the whole run. On the
# 2026-10-25 `20D_bsose_tgrad_eta` run every one of the three velocity/SSH maxima sat on or
# beside the NORTH face -- max|eta| on row j=150 on twelve consecutive days, max|u| on the same
# row every one of those days, max|v| at i=1 in the rows adjoining it -- which is what made
# this worth measuring rather than assuming.
#
# WHAT IT PRINTS
#   * every monthly record's net transport through each face, in Sv, with the records the run
#     would actually select marked;
#   * the pinned target for each face, which is the mean over exactly those records;
#   * whether each face changes sign inside the window, and whether the target even lies
#     within the window's own range.
#
# It calls the model's own `face_transport_series`, which `mean_face_transport` -- and hence the
# pin -- is defined in terms of. A tool that re-derived that sum could drift from what the model
# pins, and then this would be measuring something the model does not do.
#
# Environment knobs:
#     DATES    "start,end" span to sweep              (default 2014-01-01,2014-12-31)
#              i156 holds 2013-2024 monthly, so up to 144 records are available -- but EVERY
#              record means loading and regridding a 2160x588x52 u and v field onto our
#              360x150x135 grid. The model's own boundary setup loads 3 records x 4 fields;
#              one year here is 12 x 2, so about 2x that. Three years is 6x. Default to one
#              calendar year: it puts a June run in the middle of its own seasonal cycle,
#              which is what decides whether the pinned number is representative. Widen only
#              if that year looks atypical.
#     RUN      "start,end" the run window to mark  (default 2014-06-01,2014-06-21, = model.jl)
#     BALANCE  apply `balance_barotropic_transports!` as the model does   (default true)
#     OUTDIR   where the figure goes                              (tools/boundary_maps)
#
# IMPORTANT: builds the 360x150x135 production grid explicitly, including
# `fix_boundary_topography!`, so it reuses the bathymetry cache and describes the geometry the
# model actually ran on. Same reasoning as the header of `inspect_bathymetry.jl`.
# ==============================================================================

using Oceananigans, Oceananigans.Units, Oceananigans.Grids
using NumericalEarth, NumericalEarth.ECCO
using NumericalEarth.DataWrangling: all_dates
using Dates, Printf, Statistics
using CairoMakie

include(joinpath(@__DIR__, "..", "src", "setup_bsose.jl"))

const OUTDIR = get(ENV, "OUTDIR", joinpath(@__DIR__, "boundary_maps"))
const BALANCE = lowercase(get(ENV, "BALANCE", "true")) in ("true", "1", "yes")
mkpath(OUTDIR)

parse_span(s, default) = let parts = split(get(ENV, s, default), ",")
    (DateTime(strip(parts[1])), DateTime(strip(parts[2])))
end
const SWEEP_START, SWEEP_END = parse_span("DATES", "2014-01-01,2014-12-31")
const RUN_START, RUN_END = parse_span("RUN", "2014-06-01,2014-06-21")

const SIDES = (:west, :east, :south, :north)
const OUTWARD = (west = -1.0, east = +1.0, south = -1.0, north = +1.0)

# ---- the production grid, matching src/model.jl's GPU branch ----
const λ₁, λ₂ = 90.0, 150.0
const φ₁, φ₂ = -70.0, -45.0
const SCALING, DZ_SURFACE, DZ_BOTTOM = 6, 5, 200
const BSOSE_ITERATION = 156

z = ReferenceToStretchedDiscretization(; extent = 5800,
                                       constant_spacing = DZ_SURFACE,
                                       maximum_spacing = DZ_BOTTOM,
                                       constant_spacing_extent = 500,
                                       stretching = PowerLawStretching(1.12))
Nx, Ny, Nz = Int(SCALING * (λ₂ - λ₁)), Int(SCALING * (φ₂ - φ₁)), length(z)

@info "building the production grid ($Nx x $Ny x $Nz) so the bathymetry cache is reused..."
grid = LatitudeLongitudeGrid(CPU(); size = (Nx, Ny, Nz),
                             longitude = (λ₁, λ₂), latitude = (φ₁, φ₂), z, halo = (7, 7, 7))

bottom_height = regrid_bathymetry(grid; height_above_water = 1, minimum_depth = 10,
                                  interpolation_passes = 5)
fix_boundary_topography!(bottom_height, grid; minimum_boundary_depth = 100)
grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom_height))

# ---- the records, and which ones the run would select ----
dataset = BSOSEMonthly(iteration = BSOSE_ITERATION)
available = all_dates(dataset, :temperature)

sweep_idx = findall(d -> SWEEP_START <= bsose_window_center(d) <= SWEEP_END, available)
isempty(sweep_idx) && error("no BSOSE records between $SWEEP_START and $SWEEP_END")
sweep_dates = available[sweep_idx]

# Mirror model.jl's selection: records whose WINDOW CENTRE falls in the run, extended by one at
# each end. Cross-check against the model's own startup log, which prints the same span.
inside = findall(d -> RUN_START <= bsose_window_center(d) <= RUN_END, available)
lo, hi = if isempty(inside)
    (something(findlast(d -> bsose_window_center(d) <= RUN_START, available), firstindex(available)),
     something(findfirst(d -> bsose_window_center(d) >= RUN_END, available), lastindex(available)))
else
    (first(inside), last(inside))
end
lo = max(firstindex(available), lo - 1)
hi = min(lastindex(available), hi + 1)
run_dates = available[lo:hi]
in_run(d) = d in run_dates

@info "sweeping $(length(sweep_dates)) records; the run window selects $(length(run_dates)): $(first(run_dates)) to $(last(run_dates))"

"""Per-face transport series over `dates`, as the model builds and (optionally) balances it."""
function transports_over(dates)
    region = bsose_region(grid)
    u_fts = FieldTimeSeries(Metadata(:u_velocity; dataset, dates, region), grid)
    v_fts = FieldTimeSeries(Metadata(:v_velocity; dataset, dates, region), grid)

    slices = Dict(:west  => boundary_slice_time_series(u_fts, :west),
                  :east  => boundary_slice_time_series(u_fts, :east),
                  :south => boundary_slice_time_series(v_fts, :south),
                  :north => boundary_slice_time_series(v_fts, :north))

    Us = Dict(s => bsose_barotropic_transport(slices[s], s, grid) for s in SIDES)

    if BALANCE
        open_faces = (west = Us[:west], east = Us[:east], south = Us[:south], north = Us[:north])
        balanced = balance_barotropic_transports!(open_faces, grid)
        for s in SIDES
            Us[s] = get(balanced, s, Us[s])
        end
    end

    return Dict(s => face_transport_series(Us[s], s, grid) ./ 1e6 for s in SIDES)   # Sv
end

series = transports_over(sweep_dates)

# ------------------------------------------------------------------ the table
println()
println(BALANCE ? "transports AFTER balance_barotropic_transports!, as the pin sees them" :
                  "RAW transports -- BALANCE=false, so these are not what the pin sees")
println()
@printf("%-12s %3s | %9s %9s %9s %9s | %9s\n", "record", "", "west", "east", "south", "north", "net out")
println("-"^72)
for (t, d) in enumerate(sweep_dates)
    net = sum(OUTWARD[s] * series[s][t] for s in SIDES)
    @printf("%-12s %3s | %+9.2f %+9.2f %+9.2f %+9.2f | %+9.3f\n",
            Dates.format(bsose_window_center(d), "yyyy-mm"), in_run(d) ? "RUN" : "",
            (series[s][t] for s in SIDES)..., net)
end

# ------------------------------------------------------- the verdict, per face
run_mask = [in_run(d) for d in sweep_dates]
any(run_mask) || error("the run window's records are outside DATES; widen DATES")

println()
@printf("%-6s | %11s | %19s | %19s | %s\n",
        "face", "PIN TARGET", "run-window range", "full-sweep range", "verdict")
println("-"^104)

for s in SIDES
    all_v = series[s]
    run_v = all_v[run_mask]
    target = sum(run_v) / length(run_v)          # exactly what mean_face_transport returns

    reverses_run = minimum(run_v) < 0 < maximum(run_v)
    reverses_full = minimum(all_v) < 0 < maximum(all_v)
    spread = maximum(run_v) - minimum(run_v)

    verdict = if all(iszero, all_v)
        "face is dry"
    elseif reverses_run
        "SIGN REVERSES INSIDE THE RUN -- the pin cannot represent it"
    elseif reverses_full
        @sprintf("reverses over the year but not in-window (spread %.1f Sv)", spread)
    elseif spread > 0.25abs(target)
        @sprintf("in-window spread %.1f Sv is %.0f%% of the target", spread, 100spread / abs(target))
    else
        @sprintf("steady in-window (spread %.1f Sv)", spread)
    end

    @printf("%-6s | %+11.2f | %+8.2f to %+8.2f | %+8.2f to %+8.2f | %s\n",
            s, target, minimum(run_v), maximum(run_v),
            minimum(all_v), maximum(all_v), verdict)
end

# ------------------------------------------------------------------ the figure
let fig = Figure(size = (1100, 700))
    Label(fig[0, 1:2], "B-SOSE i$(BSOSE_ITERATION) net transport per face — the pin holds the run-window mean",
          fontsize = 19, font = :bold)
    x = 1:length(sweep_dates)
    run_v = [in_run(d) for d in sweep_dates]

    for (k, s) in enumerate(SIDES)
        ax = Axis(fig[cld(k, 2), 2 - k % 2], title = "$(s) face",
                  xlabel = "record", ylabel = "transport (Sv)")
        lo_i, hi_i = findfirst(run_v), findlast(run_v)
        isnothing(lo_i) || vspan!(ax, lo_i - 0.5, hi_i + 0.5, color = (:orange, 0.18))
        lines!(ax, x, series[s], color = :steelblue, linewidth = 2)
        scatter!(ax, x[run_v], series[s][run_v], color = :orangered, markersize = 9)
        target = sum(series[s][run_v]) / count(run_v)
        hlines!(ax, [target], color = :orangered, linestyle = :dash, linewidth = 2)
        hlines!(ax, [0], color = :gray, linestyle = :dot)
    end

    f = joinpath(OUTDIR, "face_transports_i$(BSOSE_ITERATION).png")
    save(f, fig)
    @info "wrote $f"
end

println("""

Reading this:
  * A face whose sign reverses INSIDE the run window cannot be represented by one pinned
    number, and the pin will hold it against its own flow for the whole run. That is a
    mechanism for a boundary-locked anomaly that grows steadily rather than adjusting.
  * A face whose in-window spread is a large fraction of its target has the same problem in
    milder form: the pin is right on average and wrong most of the time.
  * The pinned target here is the mean over the RUN-marked records only. Compare it against
    the model's own startup line, which prints the same number:
        -> barotropic scheme: Flather + pinned transport. Targets (Sv): W ..., N ...
    If they disagree, this script's record selection has drifted from model.jl's and the
    selection block above is what to check first.
  * `net out` should be ~0 for every record when BALANCE=true. If it is not, the balance step
    is not doing its job and nothing downstream of it can be trusted.
  * THE FLAW SCALES WITH RUN LENGTH. `bc_dates` is the run window plus one record each side, so
    a 20-day June run pins the May/June/July mean -- a three-month window, fairly benign. A
    one-year run pins the ANNUAL mean, and on a face that swings -31 to +16 Sv that is wrong
    for most of the run. A multi-year run is worse still. So the in-window spread this prints
    for a 12-month DATES span is a preview of the eventual science run, not a diagnosis of the
    20-day test.
  * If a face does reverse in-window, widening the window is not the fix -- it makes it worse.
    `target_transport` is materialised once by the free-surface constructor and cannot vary in
    time without an upstream change, so the options are shorter segments with restarts (each
    getting its own target), or removing the need for the pin by giving Flather a correct
    eta_ext instead of 0.
""")
