# ==============================================================================================
# sowinwarp_glorys_baseline.jl -- the SOWinWaRP box nested in GLORYS instead of B-SOSE, as a
# baseline. After NumericalEarth's examples/arctic_nested_ocean_sea_ice.jl: GLORYS MONTHLY means for the
# ocean AND the sea ice parent (nested_ocean_simulation + nested_sea_ice_simulation), ERA5 for the
# atmosphere, ETOPO2022 bathymetry, and the nest's own defaults for the boundary relaxation.
#
#     julia --project=. examples/sowinwarp_glorys_baseline.jl
#
# EVERY SETTING IS HARD-CODED in "run settings" below -- RUN_NAME, SMOKE, PICKUP, END_DATE, CHECKPOINT_DAYS, the
# GPU and the data directories -- and nothing is read from the environment except the Copernicus credentials.
# So variables exported for sowinwarp_nested.jl (RUN_NAME, END_DATE, ...) cannot reach this run: they once
# did, and it wrote its output over 2Y_nested_seaice_fix9's. To smoke-test, set SMOKE = true below.
#
# WHAT IT KEEPS from examples/sowinwarp_nested.jl, so that the parent is what differs from fix7:
#   the domain and grid (1/6 deg, B-SOSE i156's 52 static levels), ETOPO2022 with the same knobs and
#   the dead-end pocket fill, ERA5 from the same files, the penetrating-shortwave forwarding,
#   IncrementalRemapping ice advection, free-slip ice coasts, grounding off, the ice CFL in the
#   wizard, the 50 m ice-volume stop, the health line, and the same output files, so every tool in
#   tools/ reads this run as it reads the B-SOSE ones.
#   Like it, the parent is monthly and every record is held in memory (~0.6 GB for the 2 years).
# WHAT IT DROPS: B-SOSE's coastal freshwater (derived from B-SOSE).
# WHAT CHANGES: the sea ice crosses the open sides as the Arctic example's does (see "sea ice"): GLORYS
#   ice thickness and concentration are its boundary values, its normal velocity lets ice out but never
#   pulls it in, inflowing ice carries no snow, and it starts from GLORYS' ice on start_date. That
#   replaces our 4-cell band relaxation, and sowinwarp_nested.jl's ICE_BOUNDARY = "open" does the same
#   with B-SOSE's ice, so the two runs differ in their parent, not in how ice meets it.
#
# GLORYS MONTHLY, DOWNLOADED BY NUMERICALEARTH. Every GLORYS file (cmems_mod_glo_phy_my_0.083deg_P1M-m: the
# strips, the starting state, the ice; and the static deptho for the seafloor) is downloaded from Copernicus
# Marine by NumericalEarth's CopernicusMarine extension the first time this runs, into GLORYS_DIR
# (<data>/glorys_monthly), and inpainted on first read, with the inpainted copy cached beside it. Later runs
# find both and download nothing. It needs COPERNICUSMARINE_SERVICE_USERNAME and _PASSWORD (create_job.sh
# sources the gitignored .copernicus_env), or a stored `copernicusmarine login`, and a node that can reach
# Copernicus. The daily path (tools/prefetch_glorys.jl, src/glorys_ncar.jl, <data>/glorys) is not used here.
# ==============================================================================================

using CUDA            # loads Oceananigans' CUDA extension, which is what defines `GPU()`
using CopernicusClimateDataStore    # an ERA5 backend: the ERA5 constructors always call `download`
using CopernicusMarine              # the GLORYS backend: NumericalEarth downloads every GLORYS file through it
using NumericalEarth
using NumericalEarth.DataWrangling: metadata_path, BoundingBox, bracketing_dates
using NumericalEarth.DataWrangling.GLORYS: GLORYSMonthly
using Oceananigans
using Oceananigans.Units
using Oceananigans.Diagnostics: NaNChecker
using Oceananigans.BoundaryConditions: fill_halo_regions!
using NumericalEarth.SeaIces.ClimaSeaIce: IncrementalRemapping
using Oceanostics.ProgressMessengers: TimedMessenger
using NCDatasets      # loads Oceananigans' NetCDFWriter
using Dates, Printf, Logging

# ── run settings: ALL hard-coded; the environment is not read (see the top) ──────────────────
const SMOKE           = false     # true: 2 days, 6-hourly output, a checkpoint every day, as "glorys_smoke"
const PICKUP          = false     # true: continue RUN_NAME from its latest checkpoint
const RUN_NAME        = SMOKE ? "glorys_smoke" : "2Y_nested_glorys_baseline"
const CHECKPOINT_DAYS = SMOKE ? 1 : 30
const ARCH            = GPU()

for name in ("RUN_NAME", "SMOKE", "PICKUP", "END_DATE", "CHECKPOINT_DAYS", "ARCH", "GLORYS_DIR", "ERA5_DIR", "BSOSE_DIR")
    haskey(ENV, name) && @warn "$name = $(ENV[name]) is set in the environment; this script ignores it (all its settings are hard-coded)"
end
@info "run settings: RUN_NAME = $RUN_NAME, SMOKE = $SMOKE, PICKUP = $PICKUP, CHECKPOINT_DAYS = $CHECKPOINT_DAYS"

ARCH isa GPU && !CUDA.functional() &&
    (CUDA.functional(true); error("CUDA is not functional on this node. Check the GPU allocation."))
ARCH isa GPU && @info "Running on GPU: $(CUDA.name(CUDA.device()))"

const DATA_DIR   = "/storage/scratch1/1/vnguyen480/SOWinWaRP/data"   # PACE; ERA5 is in era5/ here
isdir(DATA_DIR) || error("$DATA_DIR not found: this run is set up for PACE (DATA_DIR is hard-coded)")
const GLORYS_DIR = joinpath(DATA_DIR, "glorys_monthly")
mkpath(GLORYS_DIR)

@info "GLORYS monthly in $GLORYS_DIR; Copernicus credentials " *
      (haskey(ENV, "COPERNICUSMARINE_SERVICE_USERNAME") && haskey(ENV, "COPERNICUSMARINE_SERVICE_PASSWORD") ?
       "from the environment" : "NOT in the environment: any missing file needs a stored `copernicusmarine login`")

# ── experiment switches: set HERE, as in sowinwarp_nested.jl ──────────────────────────────────
const PENETRATING_SHORTWAVE = true    # forward get_radiative_forcing to the child (see sowinwarp_nested.jl)
const ICE_GROUNDING         = false   # LandfastBasalStress off, as fix7

@info "switches: parent = GLORYS monthly (ocean and sea ice), bathymetry = ETOPO2022, " *
      "PENETRATING_SHORTWAVE = $PENETRATING_SHORTWAVE, ICE_GROUNDING = $ICE_GROUNDING, ice at the open sides = open boundaries"

if PENETRATING_SHORTWAVE &&
   Base.unwrap_unionall(which(NumericalEarth.Oceans.get_radiative_forcing, (NumericalEarth.NestedModel,)).sig).parameters[2] === Any
    NumericalEarth.Oceans.get_radiative_forcing(nm::NumericalEarth.NestedModel) =
        NumericalEarth.Oceans.get_radiative_forcing(nm.child)
end

# ── domain and grid: as sowinwarp_nested.jl ───────────────────────────────────────────────────
const λ₁, λ₂ = 90.0, 150.0
const φ₁, φ₂ = -70.0, -45.0
const SCALING = 6
const Nx, Ny = Int(SCALING * (λ₂ - λ₁)), Int(SCALING * (φ₂ - φ₁))   # 360 x 150

const BSOSE_DRF = [4.2, 5.0, 5.9, 6.9, 8.5, 9.5, 10.0, 10.0, 10.0, 10.0,
                   10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 13.0, 17.0, 20.0, 20.0,
                   20.0, 20.0, 20.0, 20.0, 22.0, 30.0, 38.0, 45.0, 50.0, 50.0,
                   53.0, 72.0, 100.0, 100.0, 100.0, 100.0, 100.0, 150.0, 200.0, 200.0,
                   200.0, 220.0, 300.0, 380.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0,
                   400.0, 400.0]
z = reverse(-vcat(0.0, cumsum(BSOSE_DRF))) .+ 0.0
const Nz = length(z) - 1

# GLORYS monthly records are stamped at 00:00 on the 1st (the start of their month), so 1 June is native.
# Noon on 1 June would match the B-SOSE runs to the hour, but would not be a GLORYS stamp.
start_date = DateTime(2014, 6, 1)
const END_DATE = DateTime(2016, 6, 1, 12)   # 2 years, as the B-SOSE runs
end_date   = SMOKE ? DateTime(2014, 6, 3) : END_DATE

@info "SOWinWaRP GLORYS baseline: $(Nx)x$(Ny)x$(Nz) over $(λ₁)-$(λ₂)E, $(φ₁)-$(φ₂)S, $start_date to $end_date"

underlying_grid = LatitudeLongitudeGrid(ARCH;
                                        size = (Nx, Ny, Nz),
                                        halo = (7, 7, 7),
                                        z,
                                        longitude = (λ₁, λ₂),
                                        latitude = (φ₁, φ₂))

bottom_height = regrid_bathymetry(underlying_grid;
                                  dataset = ETOPO2022(),
                                  height_above_water = 1,
                                  minimum_depth = 10,
                                  interpolation_passes = 5,
                                  major_basins = 1)

# Dead-end pockets become land, as in sowinwarp_nested.jl (2Y_nested_seaice_fix1's 46 g/kg inlet)
function fill_dead_end_pockets!(bottom_height)
    z = Array(interior(bottom_height))[:, :, 1]
    nx, ny = size(z)
    land(i, j) = 1 <= i <= nx && 1 <= j <= ny && z[i, j] >= 0
    filled = 0
    while true
        pockets = [(i, j) for i in 1:nx, j in 1:ny
                   if z[i, j] < 0 && count((land(i-1, j), land(i+1, j), land(i, j-1), land(i, j+1))) >= 3]
        isempty(pockets) && break
        for (i, j) in pockets
            z[i, j] = 0
        end
        filled += length(pockets)
    end
    set!(bottom_height, reshape(z, nx, ny, 1))
    fill_halo_regions!(bottom_height)
    return filled
end

@info "bathymetry: ETOPO2022, $(fill_dead_end_pockets!(bottom_height)) dead-end pocket cells filled"

grid = ImmersedBoundaryGrid(underlying_grid,
                            PartialCellBottom(bottom_height; minimum_fractional_cell_height = 0.2);
                            active_cells_map = true)

# Ice area and volume over the domain, for the startup line
function ice_totals(ℵ, V)
    A = [(6.371e6 * deg2rad(1 / SCALING))^2 * cosd(φ₁ + (j - 0.5) / SCALING) for i in 1:Nx, j in 1:Ny]
    return sum(ℵ .* A) / 1e12, sum(V .* A) / 1e9   # 10⁶ km², km³
end

# ── the nest ──────────────────────────────────────────────────────────────────────────────────
# As the Bouvet example: all open sides (the south has no wet columns), the default relaxation band and
# rate, the seafloor blended toward GLORYS' own (its static deptho, which NumericalEarth downloads: no
# parent_bottom_height). Unlike it, and like sowinwarp_nested.jl's B-SOSE parent, the parent is monthly and
# every record of the strips is held in memory, loaded before the first step (~25 MB a record).
const PARENT_RECORDS = length(bracketing_dates(GLORYSMonthly(), :temperature, (start_date, end_date)))
@info "GLORYS parent: $PARENT_RECORDS monthly records, all held in memory; downloaded and inpainted first if missing"

const PARENT_LOAD_T₀ = time()
ocean = nested_ocean_simulation(grid, GLORYSMonthly();
                                dates = (start_date, end_date),
                                dir = GLORYS_DIR,
                                parent_time_indices_in_memory = PARENT_RECORDS)
@info @sprintf("GLORYS parent: loaded in %.1f min", (time() - PARENT_LOAD_T₀) / 60)

child = ocean.model.child

# ── atmosphere: the same ERA5 files as sowinwarp_nested.jl ────────────────────────────────────
era5_dir = joinpath(DATA_DIR, "era5")
era5_region = BoundingBox(longitude = (λ₁ - 1.0, λ₂ + 1.0), latitude = (φ₁ - 1.0, φ₂ + 1.0))
era5_dataset = ERA5YearlySingleLevel()
era5_pressure = :mean_sea_level_pressure
era5_start = start_date - Hour(1)
era5_end   = end_date + Hour(1)

era5_names = (:eastward_velocity, :northward_velocity, :temperature, :dewpoint_temperature, era5_pressure,
              :total_precipitation, :snowfall,   # snowfall: NumericalEarth reads it since d59fa0a (rain = total - snow)
              :downwelling_shortwave_radiation, :downwelling_longwave_radiation)
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

# ── sea ice: through the open sides, as the Arctic example ────────────────────────────────────
# nested_sea_ice_simulation (NestedModels/sea_ice_open_boundaries.jl) reads GLORYS' sithick, siconc, usi and
# vsi over the domain onto the child grid, starts the ice from them on start_date, and gives the ice the
# ocean's open sides: h and ℵ take GLORYS' values there, inflowing ice carries no snow, and the normal ice
# velocity lets ice leave but not be pulled in (where GLORYS has no ice velocity, the ocean's is used).
# sithick is floe thickness, ClimaSeaIce's h, as before. Ours on top of the example's defaults: remapping
# advection, free-slip coasts and grounding off (see sowinwarp_nested.jl, "sea ice"), and every monthly
# record held in memory, as for the ocean.
ice_dynamics = ICE_GROUNDING ? NumericalEarth.SeaIces.sea_ice_dynamics(child.grid, ocean) :
                               NumericalEarth.SeaIces.sea_ice_dynamics(child.grid, ocean; basal_stress = nothing)
sea_ice = nested_sea_ice_simulation(ocean, GLORYSMonthly();
                                    dates = (start_date, end_date),
                                    dir = GLORYS_DIR,
                                    time_indices_in_memory = PARENT_RECORDS,
                                    advection = IncrementalRemapping(),
                                    lateral_boundary_condition = :free_slip,
                                    dynamics = ice_dynamics)
@info "sea-ice grounding: $(ICE_GROUNDING ? "ON (LandfastBasalStress)" : "OFF")" *
      " -- basal_stress is $(nameof(typeof(sea_ice.model.dynamics.basal_stress)))"

if !PICKUP
    h₀ = Array(interior(sea_ice.model.ice_thickness))[:, :, 1]
    ℵ₀ = Array(interior(sea_ice.model.ice_concentration))[:, :, 1]
    area₀, volume₀ = ice_totals(ℵ₀, h₀ .* ℵ₀)
    @info @sprintf("sea ice from GLORYS on %s: area %.2f 10⁶ km², volume %.0f km³ (B-SOSE ~650), h up to %.1f m",
                   start_date, area₀, volume₀, maximum(h₀))
end

coupled_model = OceanSeaIceModel(ocean, sea_ice; atmosphere, radiation)

let R = NumericalEarth.Oceans.get_radiative_forcing(ocean)
    if PENETRATING_SHORTWAVE
        R isa NumericalEarth.Oceans.TwoColorRadiation ||
            error("PENETRATING_SHORTWAVE = true, but the coupler finds $(typeof(R)) for this ocean, not the child's TwoColorRadiation")
        @info "penetrating shortwave: ON (the coupler feeds the child's TwoColorRadiation)"
    end
end

# ── stepping: as sowinwarp_nested.jl (ice CFL in the wizard, NaN checker) ─────────────────────
simulation = Simulation(coupled_model; Δt = 1seconds,
                        stop_time = (end_date - start_date).value / 1000)

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

simulation.callbacks[:nan_checker] =
    Callback(NaNChecker(fields = merge(child.velocities, child.tracers,
                                       (; h = sea_ice.model.ice_thickness, ℵ = sea_ice.model.ice_concentration)),
                        erroring = true),
             IterationInterval(1))

const ICE_MAX_VOLUME = 50
function check_ice_volume(sim)
    v_max = maximum(interior(sea_ice.model.ice_thickness) .* interior(sea_ice.model.ice_concentration))
    v_max > ICE_MAX_VOLUME &&
        error(@sprintf("sea ice volume %.1f m (h·ℵ) at t = %.2f days, iteration %d: the ice is blowing up",
                       v_max, time(sim) / 86400, iteration(sim)))
    return nothing
end
simulation.callbacks[:ice_volume] = Callback(check_ice_volume, IterationInterval(10))

# ── progress and the health line: as sowinwarp_nested.jl ─────────────────────────────────────
struct ChildView{S}
    simulation :: S
end
Base.getproperty(view::ChildView, name::Symbol) =
    name === :model ? child : getproperty(getfield(view, :simulation), name)
Oceananigans.Simulations.iteration(view::ChildView) = iteration(getfield(view, :simulation))
Base.time(view::ChildView) = time(getfield(view, :simulation))

output_interval = SMOKE ? 6hours : 1days

progress = TimedMessenger()
simulation.callbacks[:progress] = Callback(sim -> progress(ChildView(sim)), TimeInterval(output_interval))

const HEALTH_S_RANGE   = (30.0, 36.0)
const HEALTH_T_RANGE   = (-2.5, 25.0)
const HEALTH_ΔS_CELL   = 1.0
const HEALTH_ΔT_CELL   = 2.0
const HEALTH_JUMPERS   = 100
const HEALTH_DRIFT_S   = 0.25
const HEALTH_DRIFT_T   = 1.0
const HEALTH_ICE_SOUTH = -55.0
const HEALTH_ICE_ℵ     = 0.15

health_wet = Array(child.grid.immersed_boundary.bottom_height[1:Nx, 1:Ny, 1]) .< 0
health_λ   = Array(λnodes(child.grid, Center()))
health_φ   = Array(φnodes(child.grid, Center()))
health_w   = [health_wet[i, j] * cosd(health_φ[j]) for i in 1:Nx, j in 1:Ny]
health_north = [φ > HEALTH_ICE_SOUTH for _ in 1:Nx, φ in health_φ]
const HEALTH_STATE = Dict{Symbol, Any}()

surface_slice(field) = Array(view(interior(field), :, :, Nz:Nz))[:, :, 1]
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
    lo(X) = argmin(map(x -> isnan(x) ? Inf : x, X))
    hi(X) = argmax(map(x -> isnan(x) ? -Inf : x, X))
    kSmin, kSmax, kTmin, kTmax = lo(Sw), hi(Sw), lo(Tw), hi(Tw)
    hℵ = h .* ℵ
    kΔS, kΔT, kv = argmax(ΔS), argmax(ΔT), argmax(hℵ)
    nΔS, nΔT     = count(>(HEALTH_ΔS_CELL), ΔS), count(>(HEALTH_ΔT_CELL), ΔT)
    ice_area     = weighted_mean(ℵ, health_w)
    ice_volume   = weighted_mean(hℵ, health_w)
    ice_north    = sum(ℵ .* health_w .* health_north) / sum(health_w .* health_north)

    @info @sprintf("health day %.1f, Δt %.1f min | S %.2f-%.2f, mean %+.3f since start, max Δ %.2f (%d cells > %.1f) | T %.2f-%.2f, mean %+.2f, max Δ %.2f (%d > %.1f) | ice: area %.3f, volume %.3f m, max h·ℵ %s m, north of %.0fS %.3f",
                   day, sim.Δt / 60, S[kSmin], S[kSmax], S̄ - HEALTH_STATE[:S₀], ΔS[kΔS], nΔS, HEALTH_ΔS_CELL,
                   T[kTmin], T[kTmax], T̄ - HEALTH_STATE[:T₀], ΔT[kΔT], nΔT, HEALTH_ΔT_CELL,
                   ice_area, ice_volume, where_is(hℵ, kv), -HEALTH_ICE_SOUTH, ice_north)

    S[kSmin] < HEALTH_S_RANGE[1] && @warn "health: surface S below $(HEALTH_S_RANGE[1]) g/kg: $(where_is(S, kSmin))"
    S[kSmax] > HEALTH_S_RANGE[2] && @warn "health: surface S above $(HEALTH_S_RANGE[2]) g/kg: $(where_is(S, kSmax))"
    T[kTmin] < HEALTH_T_RANGE[1] && @warn "health: surface T below $(HEALTH_T_RANGE[1]) degC: $(where_is(T, kTmin))"
    T[kTmax] > HEALTH_T_RANGE[2] && @warn "health: surface T above $(HEALTH_T_RANGE[2]) degC: $(where_is(T, kTmax))"
    nΔS > HEALTH_JUMPERS && @warn "health: $nΔS surface cells changed S by > $(HEALTH_ΔS_CELL) g/kg since the last check; largest $(where_is(ΔS, kΔS))"
    nΔT > HEALTH_JUMPERS && @warn "health: $nΔT surface cells changed T by > $(HEALTH_ΔT_CELL) degC since the last check; largest $(where_is(ΔT, kΔT))"
    abs(S̄ - HEALTH_STATE[:S₀]) > HEALTH_DRIFT_S && @warn @sprintf("health: domain-mean surface S has drifted %+.3f g/kg since the start", S̄ - HEALTH_STATE[:S₀])
    abs(T̄ - HEALTH_STATE[:T₀]) > HEALTH_DRIFT_T && @warn @sprintf("health: domain-mean surface T has drifted %+.2f degC since the start", T̄ - HEALTH_STATE[:T₀])
    ice_north > HEALTH_ICE_ℵ && @warn @sprintf("health: mean ice concentration north of %.0fS is %.3f; there should be almost none", -HEALTH_ICE_SOUTH, ice_north)
    hℵ[kv] > 10 && @warn "health: most ice is $(where_is(hℵ, kv)) (m of h·ℵ); the run stops at $(ICE_MAX_VOLUME) m"
    return nothing
end
simulation.callbacks[:health] = Callback(health_check, TimeInterval(output_interval))

# ── output: the same files as sowinwarp_nested.jl ─────────────────────────────────────────────
const EKE_WINDOW = 30days
const EKE_STRIDE = 1

u, v, w = child.velocities
T, S = child.tracers.T, child.tracers.S

output_metadata = Dict("run_name" => RUN_NAME,
    "start_date" => string(start_date),
    "end_date" => string(end_date),
    "dataset" => "GLORYS",
    "boundaries" => "one-way nest in GLORYS monthly (NumericalEarth nested_ocean_simulation, defaults)",
    "atmosphere" => "ERA5 (NCAR ds633.0), $(era5_pressure)",
    "sea_ice" => "ClimaSeaIce via nested_sea_ice_simulation, IncrementalRemapping advection, free-slip coasts, " *
                 "started from GLORYS monthly siconc/sithick, open sides take GLORYS monthly ice (open boundaries)",
    "coastal_freshwater" => "none",
    "ice_grounding" => ICE_GROUNDING ? "LandfastBasalStress (NumericalEarth default)" : "off (basal_stress = nothing)",
    "bathymetry" => "ETOPO2022",
    "penetrating_shortwave" => PENETRATING_SHORTWAVE ? "yes: TwoColorRadiation fed by the coupler" :
                                                       "no: all shortwave in the top cell's heat flux",
    "vertical_coordinate" => "static z, B-SOSE i156's 52 levels")

ocean.output_writers[:surface] = NetCDFWriter(child, (; u, v, T, S);
    filename = "$(RUN_NAME)_surface.nc", schedule = TimeInterval(output_interval),
    indices = (:, :, Nz), global_attributes = output_metadata, overwrite_files = !PICKUP)

mid_lon_idx = div(Nx, 2)
ocean.output_writers[:mid_lon] = NetCDFWriter(child, (; u, v, T, S);
    filename = "$(RUN_NAME)_midlon.nc", schedule = TimeInterval(output_interval),
    indices = (mid_lon_idx, :, :), global_attributes = output_metadata, overwrite_files = !PICKUP)

η_2d = Field(Average(child.free_surface.displacement, dims=3))
ocean.output_writers[:free_surface] = NetCDFWriter(child, (; η = η_2d);
    filename = "$(RUN_NAME)_free_surface.nc", schedule = TimeInterval(output_interval),
    global_attributes = output_metadata, overwrite_files = !PICKUP)

# What the coupler hands the ocean at its surface, daily means (see sowinwarp_nested.jl for the signs)
function surface_flux_outputs()
    candidates = (
        (:heat_flux,       () -> coupled_model.interfaces.net_fluxes.ocean.T),
        (:ice_salt_flux,   () -> coupled_model.interfaces.net_fluxes.ocean.S),
        (:freshwater_flux, () -> coupled_model.interfaces.net_fluxes.ocean.η),
        (:shortwave,       () -> coupled_model.radiation.interface_fluxes.ocean.downwelling_shortwave))
    found = Pair{Symbol,Any}[]
    for (name, get_field) in candidates
        f = try get_field() catch err; err end
        if !(f isa Oceananigans.Fields.AbstractField)
            @warn "surface fluxes: no $name ($(f isa Exception ? sprint(showerror, f) : typeof(f))); not written"
        elseif size(f, 1) != Nx || size(f, 2) != Ny
            @warn "surface fluxes: $name is $(size(f)), not on the child's $(Nx)x$(Ny) grid; not written"
        else
            push!(found, name => f)
        end
    end
    @info "surface fluxes: writing $(isempty(found) ? "nothing" : join(first.(found), ", ")) as daily means"
    return NamedTuple(found)
end

surface_fluxes = surface_flux_outputs()
isempty(surface_fluxes) || (ocean.output_writers[:surface_fluxes] = NetCDFWriter(child, surface_fluxes;
    filename = "$(RUN_NAME)_surface_fluxes.nc",
    schedule = AveragedTimeInterval(output_interval, window = output_interval),
    global_attributes = merge(output_metadata, Dict(
        "flux_convention" => "kinematic, positive OUT of the ocean, except freshwater_flux (+ into the ocean) and shortwave (W/m2, negative down)",
        "shortwave_in_heat_flux" => PENETRATING_SHORTWAVE ? "no" : "yes",
        "salt_flux_seen_by_ocean" => "ice_salt_flux + S_top * freshwater_flux, S_top from the surface file",
        "averaging_window_seconds" => Float64(output_interval))),
    overwrite_files = !PICKUP))

u_cc = @at (Center, Center, Center) u
v_cc = @at (Center, Center, Center) v
eke_outputs = (; u = u_cc, v = v_cc, T, S, uu = u_cc^2, vv = v_cc^2)
eke_metadata = merge(output_metadata, Dict(
    "averaging_window_seconds" => Float64(EKE_WINDOW),
    "averaging_stride" => EKE_STRIDE,
    "moment_location" => "Center, Center, Center",
    "eke_recipe" => "EKE = 0.5*((uu - u^2) + (vv - v^2)); MKE = 0.5*(u^2 + v^2)"))

ocean.output_writers[:eke] = NetCDFWriter(child, eke_outputs;
    filename = "$(RUN_NAME)_eke.nc",
    schedule = AveragedTimeInterval(EKE_WINDOW, window = EKE_WINDOW, stride = EKE_STRIDE),
    indices = (:, :, :), global_attributes = eke_metadata, overwrite_files = !PICKUP)

sea_ice_outputs = merge((; h = sea_ice.model.ice_thickness,
                           ℵ = sea_ice.model.ice_concentration,
                           hs = sea_ice.model.snow_thickness),
                        sea_ice.model.velocities)
sea_ice.output_writers[:sea_ice] = NetCDFWriter(sea_ice.model, sea_ice_outputs;
    filename = "$(RUN_NAME)_sea_ice.nc", schedule = TimeInterval(output_interval),
    global_attributes = output_metadata, overwrite_files = !PICKUP)

@info "output: $(RUN_NAME)_{surface,midlon,free_surface,surface_fluxes,eke,sea_ice}.nc in $(pwd())"

# ── checkpoints: as sowinwarp_nested.jl ───────────────────────────────────────────────────────
simulation.output_writers[:checkpointer] = Checkpointer(coupled_model;
                                                        schedule = TimeInterval(CHECKPOINT_DAYS * days),
                                                        prefix = "$(RUN_NAME)_checkpoint")

# NumericalEarth's DataWrangling info lines (reading, inpainting parent data) are dropped from here on: the
# GLORYS parent is all in memory, downloaded and inpainted during setup, so they would carry nothing, and
# the log is the model's. Its warnings, and every other module's lines, still show.
struct QuietParentData{L <: Logging.AbstractLogger} <: Logging.AbstractLogger
    logger :: L
end
is_data_wrangling(m::Module) = m === NumericalEarth.DataWrangling ||
                               (parentmodule(m) !== m && is_data_wrangling(parentmodule(m)))
Logging.min_enabled_level(q::QuietParentData) = Logging.min_enabled_level(q.logger)
Logging.catch_exceptions(q::QuietParentData)  = Logging.catch_exceptions(q.logger)
Logging.shouldlog(q::QuietParentData, level, _module, group, id) =
    (level >= Logging.Warn || !is_data_wrangling(_module)) && Logging.shouldlog(q.logger, level, _module, group, id)
Logging.handle_message(q::QuietParentData, args...; kwargs...) = Logging.handle_message(q.logger, args...; kwargs...)
global_logger(QuietParentData(global_logger()))

PICKUP && @info "picking up $RUN_NAME from its latest checkpoint"
@info "starting"
run!(simulation; pickup = PICKUP, checkpoint_at_end = true)
@info "done"
