# ==============================================================================
# glorys_era5_gulf_stream.jl -- a small regional run on the OTHER forcing stack.
#
#     julia --project=. examples/glorys_era5_gulf_stream.jl
#     DOWNLOAD_ONLY=1 julia --project=. examples/glorys_era5_gulf_stream.jl   # fetch, then exit
#
# Ocean state from GLORYS (Copernicus Marine) instead of B-SOSE, and surface fluxes from ERA5
# through bulk formulae instead of prescribed B-SOSE flux fields. Gulf Stream separation region,
# 1/8 deg, 10 days. Nothing here is tuned; it exists to be read and to prove the plumbing.
#
# WHY IT IS WORTH HAVING. `src/model.jl` has an `ATMOSPHERE = :era5` branch that has never run.
# This is the same construction in 120 lines instead of 1200, on a domain small enough to debug
# in one sitting, so that when the real run is switched over there is a known-good reference to
# diff against. The two things it proves are the two things that break: that the credentials
# work, and that `OceanOnlyModel` steps.
#
# THE ONE THING IT DELIBERATELY LEAVES OUT IS OPEN BOUNDARIES. The four walls here are CLOSED.
# That is wrong for a regional domain -- the Gulf Stream will pile up against the eastern wall
# and the interior will start to spin down -- and over 10 days it does not matter for what this
# script is for. Upstream now has `nested_ocean_model(grid, GLORYSDaily(); dates)`, which builds
# the boundary strips, the open boundary conditions, the relaxation band and the blended
# bathymetry in one call, but it does NOT exist in the NumericalEarth v0.8.2 this project is
# pinned to (checked, not assumed). Getting it means upgrading, and upgrading means re-verifying
# every B-SOSE path in setup_bsose.jl. Not a side effect to take on by accident.
#
# ------------------------------------------------------------------------------------------
# CREDENTIALS. Two different accounts. This is the part that fails, and it fails silently-ish.
#
# 1. ERA5 -- Copernicus Climate Data Store.
#    a. Register at https://cds.climate.copernicus.eu/
#    b. Open the dataset page for "ERA5 hourly data on single levels from 1940 to present" and
#       ACCEPT ITS LICENCE. Licences are per dataset and per account; without this every request
#       returns 403 no matter how correct the key is. This is the most common failure.
#    c. Copy the Personal Access Token from https://cds.climate.copernicus.eu/profile
#    d. Write ~/.cdsapirc:
#
#           url: https://cds.climate.copernicus.eu/api
#           key: <your token>
#
#       NOTE THE URL. The retired CDS used `https://cds.climate.copernicus.eu/api/v2` and a
#       `<UID>:<KEY>` key format. Both still appear all over the internet and in older configs,
#       and both now fail. If ERA5 was tried before and did not work, check this line first.
#       Then `chmod 600 ~/.cdsapirc`.
#
#    On a batch system, prefer the environment instead -- it takes priority over the file and
#    does not need a home directory on the compute node:
#
#           export CDSAPI_URL=https://cds.climate.copernicus.eu/api
#           export CDSAPI_KEY=<your token>
#
#    No Python, conda or `era5cli` binary is involved: CopernicusClimateDataStore.jl v0.2.2
#    talks to the CDS REST API over HTTP directly.
#
# 2. GLORYS -- Copernicus Marine Service. A SEPARATE account from the one above.
#    a. Register at https://marine.copernicus.eu/
#    b. Either
#           export COPERNICUSMARINE_SERVICE_USERNAME=<user>
#           export COPERNICUSMARINE_SERVICE_PASSWORD=<pass>
#       or run `CopernicusMarine.login(username="...", password="...")` once.
#    CopernicusMarine.jl downloads a standalone `copernicusmarine` executable on first use, so
#    again no Python -- but the Linux builds are glibc 2.35/2.39, which is worth checking against
#    the compute nodes before assuming it runs there.
#
# COMPUTE NODES OFTEN HAVE NO INTERNET. Run `DOWNLOAD_ONLY=1 julia --project=. <this file>` on a
# login node first; it populates the caches below and exits before building the model. The real
# run then finds every file present and never reaches for the network.
# ==============================================================================

using Oceananigans
using Oceananigans.Units
using NumericalEarth
using CopernicusMarine              # activates the GLORYS download backend
using CopernicusClimateDataStore    # activates the ERA5 download backend
using CUDA: has_cuda_gpu
using Dates
using Printf

const DOWNLOAD_ONLY = get(ENV, "DOWNLOAD_ONLY", "0") in ("1", "true", "yes")
const DATA_DIR = get(ENV, "DATA_DIR", joinpath(@__DIR__, "..", "data", "example"))
mkpath(DATA_DIR)

arch = has_cuda_gpu() ? GPU() : CPU()
@info "Architecture: $arch"

# ---- domain ----------------------------------------------------------------------------------
# Gulf Stream separation, where the current leaves Hatteras and turns east. Chosen because the
# surface signal is large enough that a 10-day run visibly does something, and because it is
# nowhere near the Southern Ocean -- if something here only works at 60S, this will say so.

λ₁, λ₂ = -75.0, -60.0
φ₁, φ₂ =  32.0,  42.0

Nx, Ny, Nz = 8 * Int(λ₂ - λ₁), 8 * Int(φ₂ - φ₁), 30      # 1/8 degree
depth = 4000meters

# z★: cells stretch with the free surface. `mutable = true` is what makes that legal.
z = ExponentialDiscretization(Nz, -depth, 0; scale = depth / 4, mutable = true)

underlying_grid = LatitudeLongitudeGrid(arch;
                                        size = (Nx, Ny, Nz),
                                        halo = (7, 7, 7),
                                        longitude = (λ₁, λ₂),
                                        latitude = (φ₁, φ₂),
                                        z)

# ETOPO by default -- IBCSO, which the upstream Bouvet example uses, only covers south of 50S.
bottom_height = regrid_bathymetry(underlying_grid;
                                  minimum_depth = 10,
                                  interpolation_passes = 5,
                                  major_basins = 1)

grid = ImmersedBoundaryGrid(underlying_grid,
                            PartialCellBottom(bottom_height; minimum_fractional_cell_height = 0.2))

@info @sprintf("Grid %d x %d x %d over %.0f-%.0fE, %.0f-%.0fN", Nx, Ny, Nz, λ₁, λ₂, φ₁, φ₂)

# ---- dates -----------------------------------------------------------------------------------
# GLORYS daily runs 1993-01-01 to 2021-06-30. Ten days in January: winter, so the air-sea heat
# flux is large and an ERA5 coupling that is doing nothing is obvious rather than subtle.

start_date = DateTime(2010, 1, 1)
end_date   = DateTime(2010, 1, 11)

# One degree of padding on every side: the regridding interpolates, and interpolation at the
# edge of a dataset that stops exactly at the domain edge reads nothing.
region = BoundingBox(longitude = (λ₁ - 1, λ₂ + 1), latitude = (φ₁ - 1, φ₂ + 1))

# ---- fetch everything up front ---------------------------------------------------------------
# Both constructors below download on demand, so this block is not strictly necessary -- it is
# here so that DOWNLOAD_ONLY has something to do, and so a credential failure happens in the
# first ten seconds rather than after the grid and bathymetry have been built.

glorys = GLORYSDaily()
glorys_dir = joinpath(DATA_DIR, "glorys")
era5_dir = joinpath(DATA_DIR, "era5")
mkpath(glorys_dir)
mkpath(era5_dir)

@info "Fetching GLORYS initial state for $start_date ..."
ic_metadata = (T = Metadatum(:temperature; dataset = glorys, date = start_date, dir = glorys_dir, region),
               S = Metadatum(:salinity;    dataset = glorys, date = start_date, dir = glorys_dir, region),
               u = Metadatum(:u_velocity;  dataset = glorys, date = start_date, dir = glorys_dir, region),
               v = Metadatum(:v_velocity;  dataset = glorys, date = start_date, dir = glorys_dir, region))

@info "Building ERA5 atmosphere and radiation ($start_date to $end_date) ..."
atmosphere = ERA5PrescribedAtmosphere(arch; start_date, end_date, region, dir = era5_dir)
radiation  = ERA5PrescribedRadiation(arch;  start_date, end_date, region, dir = era5_dir)

if DOWNLOAD_ONLY
    @info "DOWNLOAD_ONLY set: data is cached under $DATA_DIR. Exiting before the model is built."
    exit(0)
end

# ---- ocean -----------------------------------------------------------------------------------
# Defaults throughout: whatever closure and free surface `ocean_simulation` picks. The point of
# the example is the forcing stack, not the physics.

ocean = ocean_simulation(grid)

@info "Setting the initial state from GLORYS ..."
set!(ocean.model;
     T = Field(ic_metadata.T, grid),
     S = Field(ic_metadata.S, grid),
     u = Field(ic_metadata.u, grid),
     v = Field(ic_metadata.v, grid))

# Note what is NOT set: the free surface. GLORYS ships `zos` (`:free_surface`) and starting from
# a flat η while the velocities carry GLORYS' geostrophic balance is internally inconsistent --
# the same trap documented in src/model.jl. Over 10 days in a closed box it is noise next to the
# closed walls; on anything longer, set it.

# ---- couple to ERA5 --------------------------------------------------------------------------
# `OceanOnlyModel` wraps the ocean SIMULATION and the outer Simulation steps the coupled model,
# so from here on anything reaching into the ocean goes through `ocean.model`, not `sim.model`.
# Its `sea_ice` defaults to `FreezingLimitedOceanTemperature`, a freezing-point clip -- which is
# a side effect worth knowing about, and the reason this path has a thermostat that the
# prescribed-flux B-SOSE path does not.

Δt = 5minutes
coupled_model = OceanOnlyModel(ocean; atmosphere, radiation)
simulation = Simulation(coupled_model; Δt, stop_time = end_date - start_date)

@info "Coupled model built: surface fluxes now come from ERA5 bulk formulae."

wall = Ref(time_ns())

function progress(sim)
    u, v, w = ocean.model.velocities
    T = ocean.model.tracers.T
    step_time = 1e-9 * (time_ns() - wall[])
    @info @sprintf("%s  iter %d | max|u| (%.2f, %.2f, %.1e) m/s | T %.2f to %.2f C | %s",
                   prettytime(sim), iteration(sim),
                   maximum(abs, u), maximum(abs, v), maximum(abs, w),
                   minimum(T), maximum(T), prettytime(step_time))
    flush(stdout)
    wall[] = time_ns()
    return nothing
end

add_callback!(simulation, progress, TimeInterval(12hours))

# ---- output ----------------------------------------------------------------------------------

u, v, w = ocean.model.velocities
T, S = ocean.model.tracers

simulation.output_writers[:surface] = NetCDFWriter(ocean.model, (; u, v, T, S),
    filename = joinpath(DATA_DIR, "gulf_stream_surface.nc"),
    schedule = TimeInterval(6hours),
    indices = (:, :, grid.Nz),
    global_attributes = Dict("start_date" => string(start_date),
                             "end_date" => string(end_date),
                             "ocean" => "GLORYSDaily",
                             "atmosphere" => "ERA5HourlySingleLevel"),
    overwrite_files = true)

@info "Running ..."
run!(simulation)
@info "Done. Surface output: $(joinpath(DATA_DIR, "gulf_stream_surface.nc"))"
