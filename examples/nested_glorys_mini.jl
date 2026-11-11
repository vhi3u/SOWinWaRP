# nested_glorys_mini.jl -- the SOWinWaRP box nested in GLORYS, as simply as NumericalEarth's
# examples/bouvet_nested_ocean_simulation.jl does it: `nested_ocean_model` with all its defaults, ERA5,
# an ocean-only coupled model. Our domain, resolution, levels, bathymetry and two years; GLORYS MONTHLY.
# No sea ice, no coastal freshwater, no penetrating shortwave, no extras: a reference for what the nest
# does on its own.
#
#     source .copernicus_env          # GLORYS files already downloaded by the GLORYS baseline are reused
#     julia --project=. examples/nested_glorys_mini.jl
#
# Everything is hard-coded below; nothing is read from the environment but the Copernicus credentials.
# Writes nested_glorys_mini_surface.nc and _midlon.nc (daily), which tools/compare_glorys.jl reads:
#     julia --project=. tools/compare_glorys.jl nested_glorys_mini

using NumericalEarth
using NumericalEarth.DataWrangling.GLORYS: GLORYSMonthly
using Oceananigans
using Oceananigans.Units
using CopernicusMarine
using CopernicusClimateDataStore
using NCDatasets
using Dates
using Printf
using CUDA

const RUN_NAME = "nested_glorys_mini"
const DATA_DIR = "/storage/scratch1/1/vnguyen480/SOWinWaRP/data"

arch = GPU()

# ### Grid and bathymetry: 90-150E, 70-45S at 1/6ᵒ, B-SOSE i156's 52 static levels, ETOPO2022

λ₁, λ₂ = 90, 150
φ₁, φ₂ = -70, -45

Nx = 6 * (λ₂ - λ₁)
Ny = 6 * (φ₂ - φ₁)

Δz = [4.2, 5.0, 5.9, 6.9, 8.5, 9.5, 10.0, 10.0, 10.0, 10.0,
      10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 13.0, 17.0, 20.0, 20.0,
      20.0, 20.0, 20.0, 20.0, 22.0, 30.0, 38.0, 45.0, 50.0, 50.0,
      53.0, 72.0, 100.0, 100.0, 100.0, 100.0, 100.0, 150.0, 200.0, 200.0,
      200.0, 220.0, 300.0, 380.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0,
      400.0, 400.0]
z = reverse(-vcat(0.0, cumsum(Δz)))
Nz = length(Δz)

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

nested = nested_ocean_model(grid, GLORYSMonthly(); dates, dir = joinpath(DATA_DIR, "glorys_monthly"))
child = nested.child

# ### Atmosphere and coupled model: ERA5 from our files (tools/download_era5_ncar.jl)

region = BoundingBox(longitude = (λ₁ - 1, λ₂ + 1), latitude = (φ₁ - 1, φ₂ + 1))
era5 = (; dataset = ERA5YearlySingleLevel(), start_date = start_date - Hour(1), end_date = end_date + Hour(1),
          region, dir = joinpath(DATA_DIR, "era5"))

atmosphere = ERA5PrescribedAtmosphere(arch; era5..., pressure = :mean_sea_level_pressure)
radiation  = ERA5PrescribedRadiation(arch; era5...)

Δt = 10minutes
ocean = Simulation(nested; Δt)
coupled_model = OceanOnlyModel(ocean; atmosphere, radiation)
simulation = Simulation(coupled_model; Δt, stop_time = Dates.value(end_date - start_date) / 1000)

wall_time = Ref(time_ns())

function progress(sim)
    u, v, w = child.velocities
    T, S = child.tracers.T, child.tracers.S
    @info @sprintf("time: %s, iter: %d, max|u|: (%.1e, %.1e, %.1e) m s⁻¹, T: (%.2f, %.2f) ᵒC, S: (%.2f, %.2f), wall time: %s",
                   prettytime(sim), iteration(sim), maximum(abs, u), maximum(abs, v), maximum(abs, w),
                   minimum(T), maximum(T), minimum(S), maximum(S), prettytime(1e-9 * (time_ns() - wall_time[])))
    wall_time[] = time_ns()
    return nothing
end

add_callback!(simulation, progress, TimeInterval(1days))

# ### Output: daily surface and mid-longitude section, as tools/compare_glorys.jl reads them

outputs = (; u = child.velocities.u, v = child.velocities.v, T = child.tracers.T, S = child.tracers.S)
attributes = Dict("run_name" => RUN_NAME, "start_date" => string(start_date), "end_date" => string(end_date),
                  "dataset" => "GLORYS monthly", "setup" => "nested_glorys_mini.jl: nested_ocean_model defaults, ERA5, ocean only")

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
