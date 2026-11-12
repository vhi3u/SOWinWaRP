# nested_glorys_mini.jl -- the SOWinWaRP box nested in GLORYS, as simply as NumericalEarth's
# examples/bouvet_nested_ocean_simulation.jl does it: `nested_ocean_model` with all its defaults, ERA5,
# an ocean-only coupled model, and the example's 40 levels. Our domain, resolution, bathymetry and two
# years; GLORYS MONTHLY; the time-step wizard of our runs.
# No sea ice, no coastal freshwater, no penetrating shortwave, no extras: a reference for what the nest
# does on its own. The levels are static (z), not the example's z★: with z★ this run slowed to minutes a step.
#
#     source .copernicus_env          # GLORYS files already downloaded by the GLORYS baseline are reused
#     julia --project=. examples/nested_glorys_mini.jl
#
# Everything is hard-coded below; nothing is read from the environment but the Copernicus credentials.
# Writes nested_glorys_mini_surface.nc and _midlon.nc (daily), which tools/compare_glorys.jl reads:
#     julia --project=. tools/compare_glorys.jl nested_glorys_mini

using NumericalEarth
using NumericalEarth.DataWrangling: bracketing_dates
using NumericalEarth.DataWrangling.GLORYS: GLORYSMonthly
using Oceananigans
using Oceananigans.Units
using CopernicusMarine
using CopernicusClimateDataStore
using NCDatasets
using Dates
using Printf
using CUDA
using Oceanostics.ProgressMessengers: TimedMessenger

const RUN_NAME = "nested_glorys_mini"
const DATA_DIR = "/storage/scratch1/1/vnguyen480/SOWinWaRP/data"

arch = GPU()

# ### Grid and bathymetry: 90-150E, 70-45S at 1/6ᵒ, the Bouvet example's 40 levels (static), ETOPO2022
#
# The levels stretch exponentially from ~10 m at the surface (11 levels in the top 200 m) to ~510 m at
# 5300 m. Static (mutable = false): with the example's z★ (mutable = true) the run slowed from ~1 s a step
# on day 1 to ~2.4 minutes on day 4. Seafloor deeper than 5300 m is cut off there.

λ₁, λ₂ = 90, 150
φ₁, φ₂ = -70, -45

Nx = 6 * (λ₂ - λ₁)
Ny = 6 * (φ₂ - φ₁)

Nz = 40
depth = 5300meters
z = ExponentialDiscretization(Nz, -depth, 0; scale = depth/4, mutable = false)

underlying_grid = LatitudeLongitudeGrid(arch;
                                        size = (Nx, Ny, Nz),
                                        halo = (7, 7, 7),
                                        z,
                                        longitude = (λ₁, λ₂),
                                        latitude = (φ₁, φ₂))

bottom_height = regrid_bathymetry(underlying_grid;
                                  dataset = ETOPO2022(),
                                  minimum_depth = 10,
                                  interpolation_passes = 5,
                                  major_basins = 1)

grid = ImmersedBoundaryGrid(underlying_grid, PartialCellBottom(bottom_height; minimum_fractional_cell_height = 0.2);
                            active_cells_map = true)

# ### The nested ocean: GLORYS monthly, June 2014 to June 2016, every nest default

start_date = DateTime(2014, 6, 1)
end_date   = DateTime(2016, 6, 1, 12)
dates = (start_date, end_date)

# Unlike the Bouvet example (15 days of daily records), every monthly record is held in memory, as in the
# GLORYS baseline (27 for the two years, ~0.6 GB): with the default 2 the parent swaps records in from disk
# as time moves on, and this run slowed from ~0.35 s to ~30 s a step within three days.
parent_records = length(bracketing_dates(GLORYSMonthly(), :temperature, dates))

nested = nested_ocean_model(grid, GLORYSMonthly(); dates, dir = joinpath(DATA_DIR, "glorys_monthly"),
                            parent_time_indices_in_memory = parent_records)
child = nested.child

# ### Atmosphere and coupled model: ERA5 from our files (tools/download_era5_ncar.jl)

region = BoundingBox(longitude = (λ₁ - 1, λ₂ + 1), latitude = (φ₁ - 1, φ₂ + 1))
era5 = (; dataset = ERA5YearlySingleLevel(), start_date = start_date - Hour(1), end_date = end_date + Hour(1),
          region, dir = joinpath(DATA_DIR, "era5"))

atmosphere = ERA5PrescribedAtmosphere(arch; era5..., pressure = :mean_sea_level_pressure)
radiation  = ERA5PrescribedRadiation(arch; era5...)

ocean = Simulation(nested; Δt = 1seconds)
coupled_model = OceanOnlyModel(ocean; atmosphere, radiation)
simulation = Simulation(coupled_model; Δt = 1seconds, stop_time = Dates.value(end_date - start_date) / 1000)

# The time-step wizard of our runs (the ice term dropped: there is no ice): from 1 s, at most 10% longer
# each time, keeping the child's advective CFL at 0.6
wizard = TimeStepWizard(cfl = 0.6, max_change = 1.1, min_Δt = 0.1,
                        cell_advection_timescale = _ -> Oceananigans.Advection.cell_advection_timescale(child))
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

# The progress line of our other runs (Oceanostics' TimedMessenger: iteration, time, Δt, wall time per step,
# max |u|, advective CFL), daily. The messenger reads `sim.model`, which here is the coupled model, so it is
# handed a view whose `model` is the child ocean.
struct ChildView{S}
    simulation :: S
end
Base.getproperty(view::ChildView, name::Symbol) =
    name === :model ? child : getproperty(getfield(view, :simulation), name)
Oceananigans.Simulations.iteration(view::ChildView) = iteration(getfield(view, :simulation))
Base.time(view::ChildView) = time(getfield(view, :simulation))

progress = TimedMessenger()
simulation.callbacks[:progress] = Callback(sim -> progress(ChildView(sim)), TimeInterval(1days))

# ### Output: daily surface and mid-longitude section, as tools/compare_glorys.jl reads them

outputs = (; u = child.velocities.u, v = child.velocities.v, T = child.tracers.T, S = child.tracers.S)
attributes = Dict("run_name" => RUN_NAME, "start_date" => string(start_date), "end_date" => string(end_date),
                  "dataset" => "GLORYS monthly", "setup" => "nested_glorys_mini.jl: nested_ocean_model defaults, ERA5, ocean only",
                  "vertical_coordinate" => "static z, 40 exponential levels to 5300 m (the Bouvet example's, without its z-star)")

ocean.output_writers[:surface] = NetCDFWriter(child, outputs;
                                              filename = "$(RUN_NAME)_surface.nc",
                                              schedule = TimeInterval(1days),
                                              indices = (:, :, Nz),
                                              global_attributes = attributes,
                                              overwrite_files = true)

ocean.output_writers[:midlon] = NetCDFWriter(child, outputs;
                                             filename = "$(RUN_NAME)_midlon.nc",
                                             schedule = TimeInterval(1days),
                                             indices = (div(Nx, 2), :, :),
                                             global_attributes = attributes,
                                             overwrite_files = true)

run!(simulation)
