# ==============================================================================
# compare_mld.jl -- model mixed layer depth against B-SOSE's, month by month.
#
#     RUN=wind_fw_sw julia --project=. tools/compare_mld.jl
#     RUN=2Y_wind_fw_sw MONTHS=8,9,10 julia --project=. tools/compare_mld.jl
#
# WHY. The summer salinity bias and the pre-shortwave-split warm band both sit in the seasonal
# ice zone, and both are what you would expect from a mixed layer that is too shallow: the same
# surface flux spread over less water makes a bigger anomaly. This measures the mixed layer
# directly instead of inferring it.
#
# WHAT IT NEEDS, from a completed run:
#     <RUN>_eke.nc      full-depth T and S, 30-day means -- the only full-depth output the
#                       model writes. Gives the MLD MAP.
#     <RUN>_midlon.nc   daily, full depth, one meridian. Gives the MLD SECTION, and the
#                       Jensen check below.
# It will run with either alone and says which figures it is skipping.
#
# MLD OF A MEAN IS NOT A MEAN OF MLDs. B-SOSE's monthly record is a monthly MEAN of T and S,
# so `mld(B-SOSE monthly)` is "the mixed layer of the monthly-mean state". To compare like with
# like, the model side has to be computed the same way -- which is exactly what the 30-day-mean
# EKE output gives. The daily section is then used to measure how large that distinction is:
# it reports both `mld(mean state)` and `mean(daily mld)`, and if those two differ by more than
# the model-minus-B-SOSE difference, the map is not measuring what it looks like it measures.
#
# ENV
#     RUN     run name prefix                 (default: wind_fw_sw)
#     DIR     where the .nc files live        (default: auto, "." then surface_tests)
#     OUTDIR  figures                         (default: surface_tests/figs, else plots)
#     MONTHS  calendar months                 (default: 6,7,8,9,10,11,12,1)
#     START   first day of the run            (default: 2014-06-01)
#     MLD_DT  density threshold, kg/m3        (default: 0.03, de Boyer Montegut)
#     MLD_MAX colour ceiling for the maps, m  (default: 400)
# ==============================================================================

using Oceananigans, Oceananigans.Units, Oceananigans.Grids
using NumericalEarth, NumericalEarth.ECCO
using NCDatasets, CairoMakie, Printf, Statistics, Dates

include(joinpath(@__DIR__, "..", "src", "setup_bsose.jl"))
include(joinpath(@__DIR__, "bsose_masks.jl"))
include(joinpath(@__DIR__, "mld.jl"))

const λ₁, λ₂, φ₁, φ₂ = 90.0, 150.0, -70.0, -45.0
const Z_EXTENT, DZ_SURFACE, DZ_BOTTOM, Z_CONST = 5800, 8, 400, 250

const RUN     = get(ENV, "RUN", "wind_fw_sw")

# WHERE THE OUTPUT IS. `src/model.jl` writes bare filenames, so a single run lands in whatever
# directory the job ran from, which is the repo root. The sweep is the exception:
# `run_surface_tests` puts its seven arms under `surface_tests/`. Defaulting to either one makes
# the other fail on a path nobody passed, so probe both and say which was used. An explicit DIR
# is honoured as given, including one that turns out to be wrong -- the check below then names
# the full paths it tried.
const DIR_CANDIDATES = [".", "surface_tests"]

function resolve_dir(run)
    haskey(ENV, "DIR") && return ENV["DIR"]
    for d in DIR_CANDIDATES
        isfile(joinpath(d, "$(run)_eke.nc")) && return d
        isfile(joinpath(d, "$(run)_midlon.nc")) && return d
    end
    error("found no $(run)_eke.nc or $(run)_midlon.nc under " *
          join(map(d -> "\"$d\"", DIR_CANDIDATES), " or ") *
          " -- pass DIR=<directory> if the run wrote somewhere else")
end

const DIR     = resolve_dir(RUN)
# Figures live beside the sweep when that is what is being read, because `surface_tests/figs`
# is where the seven arms' output already is. Anything else -- a single run out of the repo
# root -- puts them in `plots/` with the rest, rather than scattering a `figs/` next to it.
const OUTDIR  = get(ENV, "OUTDIR",
                    DIR == "surface_tests" ? joinpath(DIR, "figs") : "plots")
const MONTHS  = parse.(Int, split(get(ENV, "MONTHS", "6,7,8,9,10,11,12,1"), ","))
const START   = DateTime(get(ENV, "START", "2014-06-01"))
const MLD_MAX = parse(Float64, get(ENV, "MLD_MAX", "400"))
mkpath(OUTDIR)

eke_path  = joinpath(DIR, "$(RUN)_eke.nc")
sect_path = joinpath(DIR, "$(RUN)_midlon.nc")
has_eke   = isfile(eke_path)
has_sect  = isfile(sect_path)
has_eke || has_sect || error("neither $eke_path nor $sect_path exists -- nothing to compare")
@info "reading $RUN from $(abspath(DIR))" *
      (haskey(ENV, "DIR") ? "" : " (found by probing $(join(DIR_CANDIDATES, ", ")))")
has_eke  || @warn "$eke_path is missing: no MLD map. Was EKE_DIAGNOSTICS/EKE_FULL_DEPTH on?"
has_sect || @warn "$sect_path is missing: no MLD section, and no check on the mean-of-MLD gap"

@info "MLD threshold Δρ = $MLD_DT kg/m3 from 10 m, TEOS10 both sides"

# ── grid ──────────────────────────────────────────────────────────────────────────────────────
probe = Dataset(has_eke ? eke_path : sect_path)
nxf = has_eke ? size(probe["T"], 1) : nothing
run_seconds = maximum(Float64.(probe["time"][:]))
close(probe)
if has_eke && has_sect     # the section is daily, so it reaches further than the last 30-day mean
    d = Dataset(sect_path); run_seconds = max(run_seconds, maximum(Float64.(d["time"][:]))); close(d)
end
const RUN_END = START + Second(round(Int, run_seconds))
@info "run window: $START to $RUN_END"

SCALING = 6
z = ReferenceToStretchedDiscretization(; extent = Z_EXTENT, constant_spacing = DZ_SURFACE,
                                       maximum_spacing = DZ_BOTTOM,
                                       constant_spacing_extent = Z_CONST,
                                       stretching = PowerLawStretching(1.12))
Nx, Ny, Nz = Int(SCALING * (λ₂ - λ₁)), Int(SCALING * (φ₂ - φ₁)), length(z)
nxf === nothing || nxf == Nx ||
    error("grid mismatch: $(RUN)_eke.nc is $(nxf) wide but a 1/$(SCALING) deg grid is $Nx")

grid = LatitudeLongitudeGrid(CPU(); size = (Nx, Ny, Nz), longitude = (λ₁, λ₂),
                             latitude = (φ₁, φ₂), z, halo = (7, 7, 7))
bottom = regrid_bathymetry(grid; height_above_water = 1, minimum_depth = 10,
                           interpolation_passes = 5)
fix_boundary_topography!(bottom, grid; minimum_boundary_depth = 100)
grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom))

λc = Array(λnodes(grid, Center())); φc = Array(φnodes(grid, Center()))
zc = Array(znodes(grid, Center())); MID = div(Nx, 2)
const SECT_LON = λ₁ + (MID - 0.5) / SCALING

# Deepest wet index per column, in each grid. Everything below it is rock, and `mld` must not
# search into it -- see the note in tools/mld.jl.
kbot_model = [findfirst(k -> !immersed_cell(i, j, k, grid), 1:Nz) for i in 1:Nx, j in 1:Ny]
dataset    = BSOSEMonthly(iteration = 156)
pdepth     = bsose_depth(dataset, λc, φc)
kbot_par   = [d <= 0 ? nothing : findfirst(k -> -zc[k] <= d, 1:Nz) for d in pdepth]
wet_map    = BitMatrix([kbot_model[i, j] !== nothing && kbot_par[i, j] !== nothing
                        for i in 1:Nx, j in 1:Ny])
@info "$(count(wet_map)) of $(Nx * Ny) columns wet in both grids"

# ── B-SOSE records ────────────────────────────────────────────────────────────────────────────
available = all_dates(dataset, :temperature)
sel_dates = [d for d in available
             if month(bsose_window_center(d)) in MONTHS &&
                START <= bsose_window_center(d) <= RUN_END]
isempty(sel_dates) && error("no B-SOSE record matches month(s) $(join(MONTHS, ", "))")
mlabels = [Dates.format(bsose_window_center(d), "yyyy-mm") for d in sel_dates]
nm = length(sel_dates)
@info "B-SOSE months: $(join(mlabels, ", "))"

region = bsose_region(grid)
resident(v) = FieldTimeSeries(Metadata(v; dataset, dates = sel_dates, region), grid;
                              time_indices_in_memory = nm)
Tb, Sb = resident(:temperature), resident(:salinity)

"""MLD over a whole (Nx, Ny, Nz) field pair, masked to columns wet in both grids."""
function mld_map(T, S, kbot)
    H = fill(NaN, size(T, 1), size(T, 2))
    for i in axes(T, 1), j in axes(T, 2)
        (wet_map[i, j] && kbot[i, j] !== nothing) || continue
        H[i, j] = mld(view(T, i, j, :), view(S, i, j, :), zc, kbot[i, j])
    end
    return H
end

par_mld = [mld_map(Array(interior(Tb[k])), Array(interior(Sb[k])), kbot_par) for k in 1:nm]

# ── the model, from the 30-day-mean full-depth output ─────────────────────────────────────────
mod_mld = Vector{Matrix{Float64}}()
paired  = String[]

if has_eke
    ds = Dataset(eke_path)
    t  = Float64.(ds["time"][:])
    # The window length is carried on the file precisely so this does not have to be guessed;
    # an AveragedTimeInterval record is stamped at the END of its window.
    win = haskey(ds.attrib, "averaging_window_seconds") ?
          Float64(ds.attrib["averaging_window_seconds"]) : 30 * 86400.0
    centres = [START + Second(round(Int, s - win / 2)) for s in t]
    @info @sprintf("%s_eke.nc: %d records, %.0f-day window, centres %s to %s",
                   RUN, length(t), win / 86400,
                   Dates.format(first(centres), "yyyy-mm-dd"),
                   Dates.format(last(centres), "yyyy-mm-dd"))

    for k in 1:nm
        target = bsose_window_center(sel_dates[k])
        gap, n = findmin(abs(Dates.value(c - target)) for c in centres)
        gapdays = gap / 86400_000
        if gapdays > 20
            @warn "$(mlabels[k]): nearest model window centre is $(round(gapdays, digits=1)) " *
                  "days away -- skipping rather than pairing it to the wrong month"
            push!(mod_mld, fill(NaN, Nx, Ny)); push!(paired, "--")
            continue
        end
        T = Float64.(coalesce.(ds["T"][:, :, :, n], NaN))
        S = Float64.(coalesce.(ds["S"][:, :, :, n], NaN))
        push!(mod_mld, mld_map(T, S, kbot_model))
        push!(paired, @sprintf("%+.0fd", Dates.value(centres[n] - target) / 86400_000))
    end
    close(ds)
end

# ── the section: daily, so both orderings of mean and MLD are available ───────────────────────
sec_of_mean = Float64[]      # mld of the monthly-mean column
sec_mean_of = Float64[]      # monthly mean of the daily mld
sec_par     = Float64[]
sec_lat     = Float64[]

if has_sect
    ds = Dataset(sect_path)
    t  = Float64.(ds["time"][:])
    dts = [START + Second(round(Int, s)) for s in t]
    kb_sec  = [kbot_model[MID, j] for j in 1:Ny]
    kbp_sec = [kbot_par[MID, j] for j in 1:Ny]

    sec_of_mean = fill(NaN, Ny, nm); sec_mean_of = fill(NaN, Ny, nm); sec_par = fill(NaN, Ny, nm)
    for k in 1:nm
        target = bsose_window_center(sel_dates[k])
        idx = [n for (n, d) in enumerate(dts) if year(d) == year(target) && month(d) == month(target)]
        isempty(idx) && continue

        Tacc = zeros(Ny, Nz); Sacc = zeros(Ny, Nz)
        daily = [Float64[] for _ in 1:Ny]
        for n in idx
            T = Float64.(coalesce.(ds["T"][1, :, :, n], NaN))
            S = Float64.(coalesce.(ds["S"][1, :, :, n], NaN))
            Tacc .+= T; Sacc .+= S
            for j in 1:Ny
                kb_sec[j] === nothing && continue
                h = mld(view(T, j, :), view(S, j, :), zc, kb_sec[j])
                isfinite(h) && push!(daily[j], h)
            end
        end
        Tacc ./= length(idx); Sacc ./= length(idx)

        Tp = Array(interior(Tb[k]))[MID, :, :]; Sp = Array(interior(Sb[k]))[MID, :, :]
        for j in 1:Ny
            kb_sec[j] === nothing && continue
            sec_of_mean[j, k] = mld(view(Tacc, j, :), view(Sacc, j, :), zc, kb_sec[j])
            isempty(daily[j]) || (sec_mean_of[j, k] = mean(daily[j]))
            kbp_sec[j] === nothing ||
                (sec_par[j, k] = mld(view(Tp, j, :), view(Sp, j, :), zc, kbp_sec[j]))
        end
    end
    close(ds)
    sec_lat = φc
end

# ── table ─────────────────────────────────────────────────────────────────────────────────────
bands = [(-70.0, -65.0), (-65.0, -60.0), (-60.0, -55.0), (-55.0, -50.0), (-50.0, -45.0)]

"""cos(lat)-weighted mean of a (Nx, Ny) field over a latitude band, ignoring NaN."""
function band_mean(H, lo, hi)
    num = den = 0.0
    for i in 1:Nx, j in 1:Ny
        (lo <= φc[j] < hi && isfinite(H[i, j])) || continue
        w = cosd(φc[j]); num += w * H[i, j]; den += w
    end
    return den == 0 ? NaN : num / den
end

cell(v) = isnan(v) ? lpad("--", 8) : @sprintf("%8.1f", v)

if has_eke
    println()
    println("="^104)
    println("MIXED LAYER DEPTH (m), cos(lat)-weighted by latitude band -- MLD of the monthly-mean state")
    println("  B / M / Δ  =  B-SOSE / model / model minus B-SOSE.  Negative Δ = the model's")
    println("  mixed layer is TOO SHALLOW, which is the suspect for the summer surface biases.")
    println("="^104)
    # How well each model window lines up with the BSOSE month it is being compared against.
    # The EKE blocks are fixed 30-day steps from model time zero, NOT calendar months
    # (model.jl:379), so over a 2-year run they drift about ten days against the month centres.
    # In Sep and Oct, when the mixed layer moves fastest, that offset is worth reading with the
    # numbers rather than discovering afterwards.
    @printf("\n  %-8s |", "offset")
    for k in 1:nm; @printf("%9s", paired[k]); end
    println()

    for (lo, hi) in bands
        @printf("\n  %.0fS to %.0fS\n", -lo, -hi)
        @printf("  %-8s |", "")
        for l in mlabels; @printf("%9s", l); end
        println()
        for (tag, H) in (("B-SOSE", par_mld), ("model", mod_mld), ("delta", nothing))
            @printf("  %-8s |", tag)
            for k in 1:nm
                v = H === nothing ?
                    band_mean(mod_mld[k], lo, hi) - band_mean(par_mld[k], lo, hi) :
                    band_mean(H[k], lo, hi)
                print(cell(v) * " ")
            end
            println()
        end
    end
end

# The Jensen check: if these two differ by more than the model-minus-B-SOSE signal, the map
# above is measuring the averaging order, not the physics.
if has_sect
    println()
    println("="^104)
    println("MLD OF THE MEAN vs MEAN OF THE DAILY MLD, along the $(round(SECT_LON, digits=1))E section")
    println("  B-SOSE only publishes monthly means, so the map above necessarily uses the first.")
    println("  If |of-mean - mean-of| is comparable to |model - B-SOSE|, the map is not a clean")
    println("  measurement of the model's mixed layer and the section is the one to trust.")
    println("="^104)
    @printf("  %-14s |", "")
    for l in mlabels; @printf("%9s", l); end
    println()
    for (tag, A) in (("of-mean", sec_of_mean), ("mean-of", sec_mean_of), ("B-SOSE", sec_par))
        @printf("  %-14s |", tag)
        for k in 1:nm
            v = filter(isfinite, A[:, k])
            print(isempty(v) ? lpad("--", 8) * " " : @sprintf("%8.1f ", mean(v)))
        end
        println()
    end
    @printf("  %-14s |", "|gap|")
    for k in 1:nm
        d = filter(isfinite, abs.(sec_of_mean[:, k] .- sec_mean_of[:, k]))
        print(isempty(d) ? lpad("--", 8) * " " : @sprintf("%8.1f ", mean(d)))
    end
    println()
    if has_eke
        @printf("  %-14s |", "|model-BSOSE|")
        for k in 1:nm
            d = filter(isfinite, abs.(sec_of_mean[:, k] .- sec_par[:, k]))
            print(isempty(d) ? lpad("--", 8) * " " : @sprintf("%8.1f ", mean(d)))
        end
        println()
    end
end

# ── figures ───────────────────────────────────────────────────────────────────────────────────
if has_eke
    fig = Figure(size = (250 * nm + 200, 3 * 210 + 170))
    Label(fig[0, 1:(nm + 1)], "$RUN — mixed layer depth, MLD of the monthly-mean state",
          fontsize = 18, font = :bold)

    dlim = max(quantile(abs.(filter(isfinite, vcat([vec(mod_mld[k] .- par_mld[k]) for k in 1:nm]...))),
                        0.99), 1e-6)
    local hmA, hmD
    for (row, (tag, F)) in enumerate((("B-SOSE", par_mld), ("$RUN", mod_mld)))
        for k in 1:nm
            ax = Axis(fig[row, k], title = row == 1 ? mlabels[k] : "",
                      ylabel = k == 1 ? tag : "", titlesize = 12,
                      xticksvisible = false, xticklabelsvisible = false,
                      yticklabelsvisible = k == 1)
            hmA = heatmap!(ax, λc, φc, F[k], colormap = Reverse(:deep),
                           colorrange = (0, MLD_MAX), nan_color = :gray85)
        end
        Colorbar(fig[row, nm + 1], hmA, label = "MLD (m)")
    end
    for k in 1:nm
        ax = Axis(fig[3, k], ylabel = k == 1 ? "model − B-SOSE" : "",
                  xlabel = "longitude (°E)", yticklabelsvisible = k == 1)
        hmD = heatmap!(ax, λc, φc, mod_mld[k] .- par_mld[k], colormap = :balance,
                       colorrange = (-dlim, dlim), nan_color = :gray85)
    end
    Colorbar(fig[3, nm + 1], hmD, label = "Δ MLD (m)")

    Label(fig[4, 1:(nm + 1)],
          "Blue in the bottom row: the model's mixed layer is SHALLOWER than B-SOSE's.\n" *
          "Grey: dry in either bathymetry, or no $(MLD_DT) kg/m³ crossing above the sea floor " *
          "(a real result — that column is unstratified to the bottom).",
          fontsize = 11, justification = :center)

    f = joinpath(OUTDIR, "$(RUN)_mld_map.png"); save(f, fig); @info "wrote $f"
end

if has_sect
    fig = Figure(size = (250 * nm + 120, 420))
    Label(fig[0, 1:nm], "$RUN — MLD at $(round(SECT_LON, digits = 1))°E",
          fontsize = 18, font = :bold)
    for k in 1:nm
        ax = Axis(fig[1, k], title = mlabels[k], titlesize = 12,
                  xlabel = "latitude (°N)", ylabel = k == 1 ? "MLD (m)" : "",
                  yreversed = true, yticklabelsvisible = k == 1)
        lines!(ax, sec_lat, sec_par[:, k], color = :black, linewidth = 2, label = "B-SOSE")
        lines!(ax, sec_lat, sec_of_mean[:, k], color = :firebrick, linewidth = 2,
               label = "model, MLD of mean")
        lines!(ax, sec_lat, sec_mean_of[:, k], color = :firebrick, linewidth = 1.2,
               linestyle = :dash, label = "model, mean of daily MLD")
        ylims!(ax, MLD_MAX, 0)
        k == nm && axislegend(ax; position = :lb, framevisible = false, labelsize = 9)
    end
    Label(fig[2, 1:nm],
          "The two red curves are the same data averaged in the other order. Where they " *
          "separate, a monthly\nmean hides the mixing — and B-SOSE only publishes the mean, " *
          "so the map can only ever show the solid one.",
          fontsize = 11, justification = :center)
    f = joinpath(OUTDIR, "$(RUN)_mld_section.png"); save(f, fig); @info "wrote $f"
end
