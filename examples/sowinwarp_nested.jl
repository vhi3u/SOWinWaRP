# ==============================================================================================
# sowinwarp_nested.jl -- SOWinWaRP as a one-way nest in B-SOSE, on NumericalEarth's
# ss/nested-ocean-model branch. Modelled on examples/bouvet_nested_ocean_simulation.jl.
#
#     SMOKE=1 julia --project=. examples/sowinwarp_nested.jl     # 1.5 days, proves it starts
#     julia --project=. examples/sowinwarp_nested.jl             # Jun-Aug 2014
#     END_DATE=2016-06-01T12:00 julia --project=. examples/sowinwarp_nested.jl   # 2 years
#
# END_DATE can be any time up to 2016-12-31T22:00 (the ERA5 converted so far covers 2014-2016).
# OUTPUT follows src/model.jl, so `./run_animate_simulation RUN_NAME` works on it unchanged:
# RUN_NAME_surface.nc, _midlon.nc, _free_surface.nc and _eke.nc in the working directory, plus
# _sea_ice.nc (h, ℵ, snow, ice u/v daily).
# RUN_NAME defaults to nested_smoke or nested_<start>_<end>, so runs do not overwrite each other.
#
# RUN examples/nested_probe.jl FIRST. This script needs BSOSEMonthly to satisfy the parent
# dataset interface; the probe says whether it does and what is missing. Starting here instead
# means discovering the same gaps one GPU job at a time.
#
# WHAT THIS REPLACES from src/model.jl. `nested_ocean_simulation` builds, in one call, what
# bsose_open_boundary_conditions + bsose_sponge_layer_forcing + the transport pin do now:
#
#   open boundaries      ocean_open_boundary_conditions from the parent's own strips
#   the sponge           ocean_boundary_relaxation, rate 1/6hours over ~8/12 deg by default --
#                        note that is 2x STRONGER than our SPONGE_TIMESCALE = 12hours, and a
#                        band rather than a raised-cosine taper. Set `relaxation_rate` and
#                        `relaxation_width` explicitly before comparing against the old runs.
#   the barotropic mode  sample_parent_depth_mean! computes the Flather exterior from the
#                        parent every step. That retires `target_transport`, which is
#                        materialised ONCE in the current model and is the live suspect for the
#                        north-face anomaly.
#   boundary bathymetry  blend_parent_bathymetry tapers the child's seafloor onto the parent's
#                        over ~4/12 deg. That is the thin/stepped boundary column problem that
#                        caused the west-face pileup, handled by construction.
#
# TWO THINGS TO DECIDE BEFORE READING ANY RESULT:
#
#   1. SEA ICE, STARTED FROM NONE. `OceanSeaIceModel` with ClimaSeaIce through NumericalEarth's
#      `sea_ice_simulation`, on the child's grid, all defaults: slab ice + snow, ice salinity 4 g/kg
#      (B-SOSE implies 4.1), EVP dynamics, no-slip coasts, three-equation ice-ocean heat flux. The
#      nest's ocean coupling forwards to the child (NestedModels/nested_component.jl), so the ice
#      sees the child ocean. B-SOSE's SIarea/SIheff are not wired into setup_bsose.jl, so the ice
#      starts at zero on 1 June and forms over the first days from near-freezing coastal water;
#      read the first week of the coastal band with that in mind. The ice model's sides are walls,
#      so ice is deleted in the 4-cell relaxation band along the open W, E and N sides every step
#      (it leaves, as it would); none enters from outside. The wizard also limits Δt on the ICE
#      speed, and a run stops if any ice passes 50 m. 2Y_nested_seaice_full had none of these
#      three: its ice piled against the east edge from day 110 and grew to 10⁶ m.
#      The ocean-only runs (2Y_nested_test, 2Y_nested_bsosez) had none of this. Their coast went
#      warm and salty every summer, with nothing melting into it: the missing sea-ice cycle.
#
#   2. STATIC Z ON B-SOSE'S OWN LEVELS. The 52 levels of B-SOSE iteration 156 (4.2 m at the
#      surface to 400 m, 6000 m), not mutable. The parent's state then lands on the child level
#      for level, with no vertical interpolation at the boundaries, and the top cell is the same
#      4.2 m layer B-SOSE's surface fields describe. Static z as in src/model.jl, whose own grid
#      (8 m to 250 m, then stretched, 53 levels) is close but not identical. The first 2-year run
#      (2Y_nested_test) used a 40-level z-star ExponentialDiscretization instead: 11 m at the top,
#      11 levels in the upper 200 m. Its output names the vertical coordinate r_aac rather than
#      z_aac; tools/compare_surface_bsose.jl reads either.
# ==============================================================================================

using CUDA            # loads Oceananigans' CUDA extension, which is what defines `GPU()`
using CopernicusClimateDataStore    # an ERA5 backend: the ERA5 constructors always call `download`, and
                                    # without one that errors even when every file is on disk
using NumericalEarth
using NumericalEarth.DataWrangling: metadata_path
using Oceananigans
using Oceananigans.Units
using Oceananigans.Diagnostics: NaNChecker
using Oceanostics.ProgressMessengers: TimedMessenger
using NCDatasets      # loads Oceananigans' NetCDFWriter
using Dates, Printf

include(joinpath(@__DIR__, "..", "src", "setup_bsose.jl"))   # BSOSEMonthly and its file map

const SMOKE = get(ENV, "SMOKE", "0") == "1"
const ARCH  = get(ENV, "ARCH", "GPU") == "CPU" ? CPU() : GPU()
ARCH isa GPU && !CUDA.functional() &&
    (CUDA.functional(true); error("CUDA is not functional on this node. Check the GPU allocation, or set ARCH=CPU."))
ARCH isa GPU && @info "Running on GPU: $(CUDA.name(CUDA.device()))"
const BDIR  = get(ENV, "BSOSE_DIR", default_bsose_directory())

# ── domain: the production SOWinWaRP box at 1/6 deg ───────────────────────────────────────────
const λ₁, λ₂ = 90.0, 150.0
const φ₁, φ₂ = -70.0, -45.0
const SCALING = 6
const Nx, Ny = Int(SCALING * (λ₂ - λ₁)), Int(SCALING * (φ₂ - φ₁))   # 360 x 150

# B-SOSE iteration 156's level thicknesses, top down: `DRF` in its grid.nc (identical in i105's
# Theta/Salt files). 52 levels, 4.2 m to 400 m, summing to 6000 m.
const BSOSE_DRF = [4.2, 5.0, 5.9, 6.9, 8.5, 9.5, 10.0, 10.0, 10.0, 10.0,
                   10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 13.0, 17.0, 20.0, 20.0,
                   20.0, 20.0, 20.0, 20.0, 22.0, 30.0, 38.0, 45.0, 50.0, 50.0,
                   53.0, 72.0, 100.0, 100.0, 100.0, 100.0, 100.0, 150.0, 200.0, 200.0,
                   200.0, 220.0, 300.0, 380.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0,
                   400.0, 400.0]

# Faces bottom to top, as LatitudeLongitudeGrid wants them: -6000 m ... 0
z = reverse(-vcat(0.0, cumsum(BSOSE_DRF))) .+ 0.0     # .+ 0.0 turns the top face's -0.0 into 0.0
const Nz = length(z) - 1

# Must be a NATIVE B-SOSE stamp (records sit at odd times: 2014-05-02T02:00, 2014-06-01T12:00, ...).
# The initial snapshot goes through `bracketing_dates`, which looks `[start_date]` up in
# `all_dates` by exact match and crashes on `nothing` when it is not there. Workaround until
# that is fixed upstream on ss/nested-ocean-model.
start_date = DateTime(2014, 6, 1, 12)
end_date   = SMOKE ? DateTime(2014, 6, 3) : DateTime(get(ENV, "END_DATE", "2014-08-01T00:00"))
const RUN_NAME = get(ENV, "RUN_NAME", SMOKE ? "nested_smoke" :
                     "nested_" * Dates.format(start_date, "yyyymmdd") * "_" * Dates.format(end_date, "yyyymmdd"))

@info "SOWinWaRP nested: $(Nx)x$(Ny)x$(Nz) over $(λ₁)-$(λ₂)E, $(φ₁)-$(φ₂)S, $start_date to $end_date"

underlying_grid = LatitudeLongitudeGrid(ARCH;
                                        size = (Nx, Ny, Nz),
                                        halo = (7, 7, 7),
                                        z,
                                        longitude = (λ₁, λ₂),
                                        latitude = (φ₁, φ₂))

# The knobs match src/model.jl:452 so the nested run differs from the current model in the
# NESTING, not in the seafloor. (model.jl passes no `major_basins`, so it is the default 1.)
bottom_height = regrid_bathymetry(underlying_grid;
                                  dataset = ETOPO2022(),
                                  height_above_water = 1,
                                  minimum_depth = 10,
                                  interpolation_passes = 5,
                                  major_basins = 1)

grid = ImmersedBoundaryGrid(underlying_grid,
                            PartialCellBottom(bottom_height; minimum_fractional_cell_height = 0.2);
                            active_cells_map = true)

# ── the nest ──────────────────────────────────────────────────────────────────────────────────
# `dates` is a (start, end) tuple expanded to B-SOSE's native cadence. open_sides defaults to
# every bounded side of the grid, which for this box is all four. The south edge has no wet
# columns (Antarctica), so in practice it is the same three src/model.jl opens: W, E, N.
dataset = BSOSEMonthly(dir = BDIR, iteration = 156)

# EVERY parent record resident, as src/model.jl's `resident_fts` does, not the branch's default
# of 2 streamed from disk. Loading (and inpainting) then all happens here, before the run, and the
# stale-reload trap the `resident_fts` docstring measured cannot occur. The parent is only strips
# along the open sides, so even ~26 monthly records of 5 variables is a small fraction of the GPU.
parent_records = length(NumericalEarth.DataWrangling.bracketing_dates(dataset, :temperature, (start_date, end_date)))
@info "B-SOSE parent: $parent_records monthly records held in memory"

ocean = nested_ocean_simulation(grid, dataset;
                                dates = (start_date, end_date),
                                dir = BDIR,
                                parent_time_indices_in_memory = parent_records)

child = ocean.model.child

# ── atmosphere ────────────────────────────────────────────────────────────────────────────────
# Yearly files cropped to the region, converted from the Globus copy of NCAR's ERA5 by
# tools/download_era5_ncar.jl -- not downloaded from the CDS, whose queue made that impractical.
# Same region and directory as that script, which names the files after them.
#
# Mean sea level pressure, not surface pressure: interpolated onto coastal ocean cells, the
# surface pressure over the Antarctic ice sheet (tens of kPa lower) acts like a sea level step of
# several meters. NumericalEarth's docstring recommends this for ocean runs.
era5_dir = get(ENV, "ERA5_DIR", joinpath(BDIR, "era5"))
era5_region = BoundingBox(longitude = (λ₁ - 1.0, λ₂ + 1.0), latitude = (φ₁ - 1.0, φ₂ + 1.0))
era5_dataset = ERA5YearlySingleLevel()
era5_pressure = :mean_sea_level_pressure

# One extra hour at each end: ERA5 values are means over the hour ending at their stamp, so
# NumericalEarth places them half an hour earlier, and a series on exactly [start, end] leaves the
# first and last 30 minutes to `Cyclical()` wrap-around from the other end of the record.
era5_start = start_date - Hour(1)
era5_end   = end_date + Hour(1)

# Fail here rather than let the backend start a year-long CDS request for a missing file
era5_names = (:eastward_velocity, :northward_velocity, :temperature, :dewpoint_temperature, era5_pressure,
              :total_precipitation, :downwelling_shortwave_radiation, :downwelling_longwave_radiation)
missing_era5 = [path for name in era5_names
                for path in unique(metadata_path(Metadata(name; dataset = era5_dataset,
                                                          start_date = era5_start, end_date = era5_end,
                                                          dir = era5_dir, region = era5_region)))
                if !isfile(path)]
isempty(missing_era5) ||
    error("Missing ERA5 files (run tools/download_era5_ncar.jl):\n  " * join(missing_era5, "\n  "))

atmosphere = ERA5PrescribedAtmosphere(ARCH; dataset = era5_dataset, start_date = era5_start, end_date = era5_end,
                                      region = era5_region, dir = era5_dir, pressure = era5_pressure)
radiation  = ERA5PrescribedRadiation(ARCH; dataset = era5_dataset, start_date = era5_start, end_date = era5_end,
                                     region = era5_region, dir = era5_dir)

# ── sea ice: see note 1 in the header ─────────────────────────────────────────────────────────
# On the child's grid, so ice and ocean share cells and the exchange grid is the child's own.
sea_ice = sea_ice_simulation(child.grid, ocean; advection = WENO(order = 5))

coupled_model = OceanSeaIceModel(ocean, sea_ice; atmosphere, radiation)

# ── stepping and progress, as src/model.jl ────────────────────────────────────────────────────
# Start with a very small Δt so the first CATKE evaluation is numerically gentle;
# the wizard will ramp this up within a few iterations.
simulation = Simulation(coupled_model; Δt = 1seconds,
                        stop_time = (end_date - start_date).value / 1000)

# Adaptive time step on the advective CFL, with src/model.jl's settings. The wizard asks the
# simulation's model for its advective timescale, and the model here is the coupled
# EarthSystemModel, which has no such method for an ocean-only nest (NumericalEarth forwards it
# only for NestedModel itself and for nested ATMOSPHERES). So hand it the child ocean's directly.
#
# The ICE is advected on the same Δt and is faster: 2Y_nested_seaice_full reached 1-4 m/s ice
# against Δt of 20-40 min, i.e. ice CFL above 1 on these 6-8 km cells, and the thickness grew
# without bound (8 m on day 105, 1 km by day 180, hundreds of km after). Oceananigans'
# cell_advection_timescale wants 3-D velocities, so the ice gets a conservative bound instead:
# its fastest u and v over the smallest Δx and Δy anywhere on the grid. The wizard takes the
# shorter of the two, so Δt only drops when fast ice demands it; it is not a fixed cap.
const ICE_Δx_MIN = minimum_xspacing(child.grid.underlying_grid)
const ICE_Δy_MIN = minimum_yspacing(child.grid.underlying_grid)

function ice_advection_timescale()
    uᵢ, vᵢ = sea_ice.model.velocities
    inverse = maximum(abs, uᵢ) / ICE_Δx_MIN + maximum(abs, vᵢ) / ICE_Δy_MIN
    return inverse > 0 ? 1 / inverse : Inf
end

coupled_advection_timescale(_) = min(Oceananigans.Advection.cell_advection_timescale(child),
                                     ice_advection_timescale())
wizard = TimeStepWizard(cfl = 0.6, max_change = 1.1, min_Δt = 0.1,
                        cell_advection_timescale = coupled_advection_timescale)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

# Fail loudly and early on NaN: every field, every step, and throw (see src/model.jl).
simulation.callbacks[:nan_checker] =
    Callback(NaNChecker(fields = merge(child.velocities, child.tracers,
                                       (; h = sea_ice.model.ice_thickness, ℵ = sea_ice.model.ice_concentration)),
                        erroring = true),
             IterationInterval(1))

# ── ice leaves through the open sides ─────────────────────────────────────────────────────────
# The ocean's W, E and N sides are open, but the ice model's are walls: ice drifting west along
# the coast piled against the west edge, ice carried east by the ACC against the east edge, and
# the first runaway thickness in 2Y_nested_seaice_full started at the east edge (i = 360, day 110).
# So the ice is removed in the band where the nest relaxes the ocean to B-SOSE (8/12 deg, 4 cells
# here), every step: ice reaching an open side leaves the domain, as it would. Inside that band
# the ocean is restored to B-SOSE on 6 hours, which absorbs the freshwater the deleted ice never
# returns. No ice enters from outside either: B-SOSE's SIarea/SIheff are not wired in.
const ICE_EDGE = 4
const ICE_FIELDS = (sea_ice.model.ice_thickness, sea_ice.model.ice_concentration, sea_ice.model.snow_thickness)

function remove_edge_ice!(_)
    for field in ICE_FIELDS
        data = interior(field)
        data[1:ICE_EDGE, :, :] .= 0                # west
        data[Nx-ICE_EDGE+1:Nx, :, :] .= 0          # east
        data[:, Ny-ICE_EDGE+1:Ny, :] .= 0          # north
    end
    return nothing
end
simulation.callbacks[:remove_edge_ice] = Callback(remove_edge_ice!, IterationInterval(1))

# Thickness can run away without ever going NaN (the NaN checker did not fire on 2Y_nested_seaice_full,
# whose ice reached 10⁶ m). No real Antarctic ice is 50 m thick, so stop there.
const ICE_MAX_THICKNESS = 50
function check_ice_thickness(sim)
    h_max = maximum(sea_ice.model.ice_thickness)
    h_max > ICE_MAX_THICKNESS &&
        error(@sprintf("sea ice %.1f m thick at t = %.2f days, iteration %d: the ice is blowing up",
                       h_max, time(sim) / 86400, iteration(sim)))
    return nothing
end
simulation.callbacks[:ice_thickness] = Callback(check_ice_thickness, IterationInterval(10))

# Oceanostics' TimedMessenger, as src/model.jl's BSOSE path uses. It reads velocities, the closure
# and the CFL numbers off `simulation.model`, and the coupled model has none of them -- which is
# why src/model.jl swaps it out on its coupled path. This view hands it the child ocean as the
# model and the coupled simulation for everything else (Δt, time, iteration, stop time).
struct ChildView{S}
    simulation :: S
end
Base.getproperty(view::ChildView, name::Symbol) =
    name === :model ? child : getproperty(getfield(view, :simulation), name)
Oceananigans.Simulations.iteration(view::ChildView) = iteration(getfield(view, :simulation))   # the one Oceanostics calls
Base.time(view::ChildView) = time(getfield(view, :simulation))

output_interval = SMOKE ? 6hours : 1days

progress = TimedMessenger()
simulation.callbacks[:progress] = Callback(sim -> progress(ChildView(sim)), TimeInterval(output_interval))

# ── health check: one line per output interval, a @warn when something looks wrong ─────────────
# 2Y_nested_seaice_full went wrong for months without a NaN, and the surface only showed it ~130
# days after the ice did. This prints, once per output interval, the numbers that would have shown
# it: surface S and T ranges, the largest change since the last check, the domain-mean drift since
# the start, and the ice. It never stops the run (the NaN checker and the 50 m ice guard do that).
#
# Cheap by construction: the top level and the ice are copied to the CPU (about 2 MB) once per
# interval and summarised there, a few ms against the tens of seconds a model day takes.
#
# The thresholds are deliberately loose. The top cell is 4.2 m, so freezing 0.1 m of ice adds
# ~0.8 g/kg to it before mixing, melting 0.5 m removes ~3.5 g/kg, and a front crossing a cell can
# move it ~1 g/kg in a day: a single cell over the limit is weather, many at once are not.
const HEALTH_S_RANGE   = (30.0, 36.0)   # g/kg, surface
const HEALTH_T_RANGE   = (-2.5, 25.0)   # degC, surface
const HEALTH_ΔS_CELL   = 1.0            # g/kg per check: a cell counts as "jumping" past this
const HEALTH_ΔT_CELL   = 2.0            # degC per check
const HEALTH_JUMPERS   = 100            # warn when more cells than this jump at once (of ~46,000 wet)
const HEALTH_DRIFT_S   = 0.25           # g/kg, |domain-mean surface S - its starting value|
const HEALTH_DRIFT_T   = 1.0            # degC
const HEALTH_ICE_SOUTH = -55.0          # warn on ice north of this latitude...
const HEALTH_ICE_ℵ     = 0.15           # ...at more than this concentration

# The child's own seafloor (after the nest blended it onto the parent's): stored as a Field or as
# its bare offset array depending on how the grid was built, and both index the interior as 1:N
health_wet = Array(child.grid.immersed_boundary.bottom_height[1:Nx, 1:Ny, 1]) .< 0
health_λ   = Array(λnodes(child.grid, Center()))
health_φ   = Array(φnodes(child.grid, Center()))
health_w   = [health_wet[i, j] * cosd(health_φ[j]) for i in 1:Nx, j in 1:Ny]   # area weights, 0 on land
health_north = [φ > HEALTH_ICE_SOUTH for _ in 1:Nx, φ in health_φ]
const HEALTH_STATE = Dict{Symbol, Any}()

surface_slice(field) = Array(view(interior(field), :, :, Nz:Nz))[:, :, 1]   # copies the top level only
ice_slice(field)     = Array(interior(field))[:, :, 1]
weighted_mean(X, w)  = sum(X .* w) / sum(w)
where_is(X, k)       = @sprintf("%.2f at %.1fE %.1fS", X[k], health_λ[k[1]], -health_φ[k[2]])

function health_check(sim)
    S, T = surface_slice(child.tracers.S), surface_slice(child.tracers.T)
    h, ℵ = ice_slice(sea_ice.model.ice_thickness), ice_slice(sea_ice.model.ice_concentration)
    day  = time(sim) / 86400
    S̄, T̄ = weighted_mean(S, health_w), weighted_mean(T, health_w)

    if isempty(HEALTH_STATE)
        HEALTH_STATE[:S₀], HEALTH_STATE[:T₀] = S̄, T̄
        HEALTH_STATE[:S], HEALTH_STATE[:T] = S, T
        return nothing
    end

    ΔS = abs.(S .- HEALTH_STATE[:S]) .* health_wet
    ΔT = abs.(T .- HEALTH_STATE[:T]) .* health_wet
    HEALTH_STATE[:S], HEALTH_STATE[:T] = S, T

    Sw = ifelse.(health_wet, S, NaN); Tw = ifelse.(health_wet, T, NaN)
    lo(X) = argmin(map(x -> isnan(x) ? Inf : x, X))     # CartesianIndex of the wet minimum
    hi(X) = argmax(map(x -> isnan(x) ? -Inf : x, X))
    kSmin, kSmax, kTmin, kTmax = lo(Sw), hi(Sw), lo(Tw), hi(Tw)
    kΔS, kΔT, kh = argmax(ΔS), argmax(ΔT), argmax(h)
    nΔS, nΔT     = count(>(HEALTH_ΔS_CELL), ΔS), count(>(HEALTH_ΔT_CELL), ΔT)
    ice_area     = weighted_mean(ℵ, health_w)
    ice_volume   = weighted_mean(h .* ℵ, health_w)                       # m of ice, domain mean
    ice_north    = sum(ℵ .* health_w .* health_north) / sum(health_w .* health_north)

    @info @sprintf("health day %.1f, Δt %.1f min | S %.2f-%.2f, mean %+.3f since start, max Δ %.2f (%d cells > %.1f) | T %.2f-%.2f, mean %+.2f, max Δ %.2f (%d > %.1f) | ice: area %.3f, volume %.3f m, max h %.1f m, north of %.0fS %.3f",
                   day, sim.Δt / 60, S[kSmin], S[kSmax], S̄ - HEALTH_STATE[:S₀], ΔS[kΔS], nΔS, HEALTH_ΔS_CELL,
                   T[kTmin], T[kTmax], T̄ - HEALTH_STATE[:T₀], ΔT[kΔT], nΔT, HEALTH_ΔT_CELL,
                   ice_area, ice_volume, h[kh], -HEALTH_ICE_SOUTH, ice_north)

    S[kSmin] < HEALTH_S_RANGE[1] && @warn "health: surface S below $(HEALTH_S_RANGE[1]) g/kg: $(where_is(S, kSmin))"
    S[kSmax] > HEALTH_S_RANGE[2] && @warn "health: surface S above $(HEALTH_S_RANGE[2]) g/kg: $(where_is(S, kSmax))"
    T[kTmin] < HEALTH_T_RANGE[1] && @warn "health: surface T below $(HEALTH_T_RANGE[1]) degC: $(where_is(T, kTmin))"
    T[kTmax] > HEALTH_T_RANGE[2] && @warn "health: surface T above $(HEALTH_T_RANGE[2]) degC: $(where_is(T, kTmax))"
    nΔS > HEALTH_JUMPERS && @warn "health: $nΔS surface cells changed S by > $(HEALTH_ΔS_CELL) g/kg since the last check; largest $(where_is(ΔS, kΔS))"
    nΔT > HEALTH_JUMPERS && @warn "health: $nΔT surface cells changed T by > $(HEALTH_ΔT_CELL) degC since the last check; largest $(where_is(ΔT, kΔT))"
    abs(S̄ - HEALTH_STATE[:S₀]) > HEALTH_DRIFT_S && @warn @sprintf("health: domain-mean surface S has drifted %+.3f g/kg since the start", S̄ - HEALTH_STATE[:S₀])
    abs(T̄ - HEALTH_STATE[:T₀]) > HEALTH_DRIFT_T && @warn @sprintf("health: domain-mean surface T has drifted %+.2f degC since the start", T̄ - HEALTH_STATE[:T₀])
    ice_north > HEALTH_ICE_ℵ && @warn @sprintf("health: mean ice concentration north of %.0fS is %.3f; there should be almost none", -HEALTH_ICE_SOUTH, ice_north)
    h[kh] > 10 && @warn "health: thickest ice is $(where_is(h, kh)) (m); the run stops at $(ICE_MAX_THICKNESS) m"
    return nothing
end
simulation.callbacks[:health] = Callback(health_check, TimeInterval(output_interval))

# ── output ────────────────────────────────────────────────────────────────────────────────────
# The same four files src/model.jl writes, with the same names, fields, cadences and global
# attributes, so tools/animate_simulation.jl and the analysis tools read both runs alike:
#
#   RUN_NAME_surface.nc       u, v, T, S at k = Nz, daily
#   RUN_NAME_midlon.nc        u, v, T, S on the mid-longitude section, daily
#   RUN_NAME_free_surface.nc  η, daily
#   RUN_NAME_eke.nc           30-day means of u, v, T, S, uu, vv at cell centres, full depth
#
# Written off `child`, NOT off the NestedModel: the nest's prognostic state is the child.
const EKE_WINDOW = 30days
const EKE_STRIDE = 1

u, v, w = child.velocities
T, S = child.tracers.T, child.tracers.S

# Stamped on every file so the animation captions frames with calendar dates
output_metadata = Dict("run_name" => RUN_NAME,
    "start_date" => string(start_date),
    "end_date" => string(end_date),
    "dataset" => "BSOSE",
    "boundaries" => "one-way nest in B-SOSE (NumericalEarth nested_ocean_simulation)",
    "atmosphere" => "ERA5 (NCAR ds633.0), $(era5_pressure)",
    "sea_ice" => "ClimaSeaIce via sea_ice_simulation defaults, started ice-free",
    "vertical_coordinate" => "static z, B-SOSE i156's 52 levels")

ocean.output_writers[:surface] = NetCDFWriter(
    child,
    (; u, v, T, S),
    filename="$(RUN_NAME)_surface.nc",
    schedule=TimeInterval(output_interval),
    indices=(:, :, Nz),
    global_attributes=output_metadata,
    overwrite_files=true
)

mid_lon_idx = div(Nx, 2)
ocean.output_writers[:mid_lon] = NetCDFWriter(
    child,
    (; u, v, T, S),
    filename="$(RUN_NAME)_midlon.nc",
    schedule=TimeInterval(output_interval),
    indices=(mid_lon_idx, :, :),
    global_attributes=output_metadata,
    overwrite_files=true
)

# η lives at (Center, Center, Face) on a single level, which NetCDFWriter cannot write
# directly; reducing over the (length-one) vertical gives an ordinary 2D field.
η_2d = Field(Average(child.free_surface.displacement, dims=3))

ocean.output_writers[:free_surface] = NetCDFWriter(
    child,
    (; η=η_2d),
    filename="$(RUN_NAME)_free_surface.nc",
    schedule=TimeInterval(output_interval),
    global_attributes=output_metadata,
    overwrite_files=true
)

# Co-located velocities, squared after interpolating (see src/model.jl's EKE block). T and S
# ride along: this is the only full-depth output.
u_cc = @at (Center, Center, Center) u
v_cc = @at (Center, Center, Center) v
eke_outputs = (; u=u_cc, v=v_cc, T, S, uu=u_cc^2, vv=v_cc^2)

eke_metadata = merge(output_metadata, Dict(
    "averaging_window_seconds" => Float64(EKE_WINDOW),
    "averaging_stride" => EKE_STRIDE,
    "moment_location" => "Center, Center, Center",
    "eke_recipe" => "EKE = 0.5*((uu - u^2) + (vv - v^2)); MKE = 0.5*(u^2 + v^2)"))

simulation_seconds = (end_date - start_date).value / 1000
if EKE_WINDOW > simulation_seconds
    @warn "EKE_WINDOW ($(EKE_WINDOW / 1days) days) is longer than the run: $(RUN_NAME)_eke.nc will be created and left empty."
end

ocean.output_writers[:eke] = NetCDFWriter(
    child,
    eke_outputs,
    filename="$(RUN_NAME)_eke.nc",
    schedule=AveragedTimeInterval(EKE_WINDOW, window=EKE_WINDOW, stride=EKE_STRIDE),
    indices=(:, :, :),
    global_attributes=eke_metadata,
    overwrite_files=true
)

# Sea ice, daily: thickness, concentration, snow and the ice velocities
sea_ice_outputs = merge((; h = sea_ice.model.ice_thickness,
                           ℵ = sea_ice.model.ice_concentration,
                           hs = sea_ice.model.snow_thickness),
                        sea_ice.model.velocities)

sea_ice.output_writers[:sea_ice] = NetCDFWriter(
    sea_ice.model,
    sea_ice_outputs,
    filename="$(RUN_NAME)_sea_ice.nc",
    schedule=TimeInterval(output_interval),
    global_attributes=output_metadata,
    overwrite_files=true
)

@info "output: $(RUN_NAME)_{surface,midlon,free_surface,eke,sea_ice}.nc in $(pwd())"

@info "starting"
run!(simulation)
@info "done"
