# ==============================================================================================
# sowinwarp_glorys_baseline.jl -- the SOWinWaRP box nested in GLORYS instead of B-SOSE, as a
# baseline. After NumericalEarth's examples/bouvet_nested_ocean_simulation.jl: GLORYS daily for the
# ocean parent, ERA5 for the atmosphere, ClimaSeaIce for the ice, ETOPO2022 bathymetry, and the
# nest's own defaults for the boundary relaxation.
#
#     julia --project=. tools/prefetch_glorys.jl                            # FIRST, on CPU: makes all the GLORYS ready
#     julia --project=. examples/sowinwarp_glorys_baseline.jl               # GPU: the run itself
#     SMOKE=1 julia --project=. examples/sowinwarp_glorys_baseline.jl       # 2 days, proves it starts
#
# RUN_NAME defaults to 2Y_nested_glorys_baseline; END_DATE is hard-coded below to 2016-06-01T12:00 (the same span
# as the 2-year B-SOSE runs; an END_DATE in the environment is ignored), so a plain sbatch of this script is the baseline.
#
# WHAT IT KEEPS from examples/sowinwarp_nested.jl, so that the parent is what differs from fix7:
#   the domain and grid (1/6 deg, B-SOSE i156's 52 static levels), ETOPO2022 with the same knobs and
#   the dead-end pocket fill, ERA5 from the same files, the penetrating-shortwave forwarding,
#   IncrementalRemapping ice advection, free-slip ice coasts, grounding off, the ice CFL in the
#   wizard, the 50 m ice-volume stop, the health line, and the same output files, so every tool in
#   tools/ reads this run as it reads the B-SOSE ones.
#   Like it, EVERY parent record is held in memory, loaded at startup (see "the nest").
# WHAT IT DROPS: B-SOSE's coastal freshwater (derived from B-SOSE).
# WHAT CHANGES: the ice starts from GLORYS siconc/sithick on 2014-06-01, and the 4-cell open-side
#   bands are relaxed to GLORYS ice, in place of B-SOSE's (see "GLORYS sea ice").
#
# GLORYS, AS ERA5: PREFETCHED, THEN LOADED. tools/prefetch_glorys.jl, one CPU job, makes every GLORYS file
# this run reads -- cut from the Globus copy of NSF NCAR's GLORYS12V1 (GDEX d010049) under NumericalEarth's
# own names, and deptho from Copernicus if the credentials are set -- and ends with "GLORYS READY". This run
# reads only those files: it checks they are all there at startup, as it checks the ERA5 files, and stops
# if not. Then it loads every daily record of the boundary strips into memory before the first step,
# inpainting each on the GPU as it goes (~0.25 s a day; NumericalEarth also keeps each as
# <file>_inpainted.jld2, so a restart only loads). Nothing is read, downloaded or inpainted while it runs,
# so its log is the model's.
# ==============================================================================================

using CUDA            # loads Oceananigans' CUDA extension, which is what defines `GPU()`
using CopernicusClimateDataStore    # an ERA5 backend: the ERA5 constructors always call `download`
using NumericalEarth
using NumericalEarth.DataWrangling: metadata_path, Metadatum, BoundingBox, bracketing_dates, dataset_variable_name
using NumericalEarth.DataWrangling.GLORYS: GLORYSDaily, GLORYSStatic
using Oceananigans
using Oceananigans.Units
using Oceananigans.Diagnostics: NaNChecker
using Oceananigans.BoundaryConditions: fill_halo_regions!
using NumericalEarth.SeaIces.ClimaSeaIce: IncrementalRemapping
using Oceanostics.ProgressMessengers: TimedMessenger
using NCDatasets      # loads Oceananigans' NetCDFWriter
using Dates, Printf, Logging

include(joinpath(@__DIR__, "..", "src", "setup_bsose.jl"))   # only for default_bsose_directory(), where ERA5 lives
include(joinpath(@__DIR__, "..", "src", "glorys_ncar.jl"))   # the GLORYS files this run reads; the ice days and region

const SMOKE    = get(ENV, "SMOKE", "0") == "1"
const PICKUP   = get(ENV, "PICKUP", "0") == "1"
const ARCH     = get(ENV, "ARCH", "GPU") == "CPU" ? CPU() : GPU()
ARCH isa GPU && !CUDA.functional() &&
    (CUDA.functional(true); error("CUDA is not functional on this node. Check the GPU allocation, or set ARCH=CPU."))
ARCH isa GPU && @info "Running on GPU: $(CUDA.name(CUDA.device()))"

const DATA_DIR   = default_bsose_directory()
const GLORYS_DIR = get(ENV, "GLORYS_DIR", joinpath(DATA_DIR, "glorys"))
mkpath(GLORYS_DIR)

# Every GLORYS file is on disk before the run (tools/prefetch_glorys.jl): "downloading" one is finding it.
# More specific than any backend's method, so nothing here ever goes to the network.
function NumericalEarth.DataWrangling.Downloads.download(meta::Metadatum{<:Union{GLORYSDaily, GLORYSStatic}}; kwargs...)
    path = joinpath(meta.dir, meta.filename)
    isfile(path) || error("$path is missing: run tools/prefetch_glorys.jl")
    return path
end

# ── experiment switches: set HERE, as in sowinwarp_nested.jl ──────────────────────────────────
const PENETRATING_SHORTWAVE = true    # forward get_radiative_forcing to the child (see sowinwarp_nested.jl)
const ICE_GROUNDING         = false   # LandfastBasalStress off, as fix7
const ICE_RELAXATION_HOURS  = 6       # open-side band ice relaxed to GLORYS's on this timescale
const ICE_RECORD_DAYS       = GLORYS_ICE_RECORD_DAYS   # 5: GLORYS ice sampled every 5 days (set in src/glorys_ncar.jl, which tools/prefetch_glorys.jl shares)
const PARENT_IN_MEMORY      = true    # every GLORYS strip record held in memory, loaded at startup; false streams 2 from disk

@info "switches: parent = GLORYS daily, bathymetry = ETOPO2022, PENETRATING_SHORTWAVE = $PENETRATING_SHORTWAVE, " *
      "ICE_GROUNDING = $ICE_GROUNDING, ICE_RELAXATION_HOURS = $ICE_RELAXATION_HOURS, ICE_RECORD_DAYS = $ICE_RECORD_DAYS"

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

# GLORYS daily records are stamped at 00:00 (the start of their day), so any midnight is native.
# Noon on 1 June would match the B-SOSE runs to the hour, but would not be a GLORYS stamp.
start_date = DateTime(2014, 6, 1)
const END_DATE = DateTime(2016, 6, 1, 12)   # hard-coded: 2 years (not read from ENV); tools/prefetch_glorys.jl's RUN_END must match
end_date   = SMOKE ? DateTime(2014, 6, 3) : END_DATE
const RUN_NAME = get(ENV, "RUN_NAME", SMOKE ? "glorys_smoke" : "2Y_nested_glorys_baseline")

@info "SOWinWaRP GLORYS baseline: $(Nx)x$(Ny)x$(Nz) over $(λ₁)-$(λ₂)E, $(φ₁)-$(φ₂)S, $start_date to $end_date"

# ── GLORYS: all there? (as the ERA5 check below) ──────────────────────────────────────────────
# Every GLORYS file this run reads (src/glorys_ncar.jl, glorys_run_files: the strips for every day, the
# starting state, the ice days). Stop here, before any of the setup, if tools/prefetch_glorys.jl has not
# made them all.
const GLORYS_FILES = glorys_run_files(GLORYS_DIR, start_date, end_date; λ₁, λ₂, φ₁, φ₂, Nx, Ny)
let missing_files = [basename(metadata_path(m)) for m in GLORYS_FILES.metadata if !isfile(metadata_path(m))]
    isempty(missing_files) ||
        error("GLORYS is not ready in $GLORYS_DIR: $(length(missing_files)) files missing. " *
              "Run tools/prefetch_glorys.jl. The first ones:\n  " * join(first(missing_files, 10), "\n  "))
    @info "GLORYS: all $(length(GLORYS_FILES.metadata)) files this run reads are in $GLORYS_DIR; seafloor " *
          (isfile(metadata_path(GLORYS_FILES.deptho)) ? "from deptho" : "from GLORYS' first-day temperature (no deptho)")
end

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

# ── GLORYS sea ice ────────────────────────────────────────────────────────────────────────────
# siconc (ℵ) and sithick, every ICE_RECORD_DAYS days, over the domain plus a degree, aggregated
# onto the child grid: each child cell takes the mean of the GLORYS ocean cells (1/12 deg, so ~4)
# whose centres fall inside it; a child ocean cell with none (our coast is not GLORYS's) takes the
# nearest child cell that has one, within ICE_FILL_CELLS. GLORYS land is missing in siconc.
#
# sithick is the thickness of the ice that is there (floe thickness), not volume per cell area, so
# the volume per cell area is sithick·siconc. Checked on NCAR's 2014-06-12 file over 90-150E,
# 70-55S: median sithick is 0.27 m where siconc < 0.15 and 0.99 m where siconc > 0.9; as volume per
# cell area, the edge floes would be 4-7 m thick. The startup line still prints the volume both ways.
const ICE_ℵ_MIN = 0.01
const ICE_FILL_CELLS = 6
const ICE_REGION = glorys_ice_region(λ₁, λ₂, φ₁, φ₂)

# The days sampled for the ice (src/glorys_ncar.jl): the day before the start, every 5 days, the day after the end
ice_record_starts(first_date, last_date) = glorys_ice_record_starts(first_date, last_date; every = ICE_RECORD_DAYS)

glorys_ice_file(name, date) =
    NumericalEarth.DataWrangling.Downloads.download(Metadatum(name; dataset = GLORYSDaily(), date,
                                                              region = ICE_REGION, dir = GLORYS_DIR))

function read_glorys_2d(path, name)
    ds = NCDataset(path)
    lon, lat = Float64.(ds["longitude"][:]), Float64.(ds["latitude"][:])
    A = Array(ds[name])                         # (longitude, latitude[, time = 1])
    close(ds)
    return lon, lat, Float64.(coalesce.(ndims(A) == 3 ? A[:, :, 1] : A, NaN))
end

function load_glorys_ice(first_date, last_date)
    starts = ice_record_starts(first_date, last_date)
    wet = Array(interior(bottom_height))[:, :, 1] .< 0
    ℵs, Vs = Matrix{Float64}[], Matrix{Float64}[]
    for (n, d) in enumerate(starts)
        lon, lat, c = read_glorys_2d(glorys_ice_file(:sea_ice_concentration, d), "siconc")
        _, _, h     = read_glorys_2d(glorys_ice_file(:sea_ice_thickness, d), "sithick")
        sum_ℵ, sum_V, cells = zeros(Nx, Ny), zeros(Nx, Ny), zeros(Int, Nx, Ny)
        for jb in eachindex(lat), ib in eachindex(lon)
            isnan(c[ib, jb]) && continue                         # GLORYS land
            i = floor(Int, (lon[ib] - λ₁) * SCALING) + 1
            j = floor(Int, (lat[jb] - φ₁) * SCALING) + 1
            (1 <= i <= Nx && 1 <= j <= Ny) || continue
            sum_ℵ[i, j] += c[ib, jb]
            sum_V[i, j] += c[ib, jb] * (isnan(h[ib, jb]) ? 0.0 : h[ib, jb])
            cells[i, j] += 1
        end
        ℵ = [cells[i, j] > 0 ? sum_ℵ[i, j] / cells[i, j] : NaN for i in 1:Nx, j in 1:Ny]
        V = [cells[i, j] > 0 ? sum_V[i, j] / cells[i, j] : NaN for i in 1:Nx, j in 1:Ny]
        # Child ocean cells with no GLORYS ocean inside: nearest child cell that has one
        unreached = 0
        for i in 1:Nx, j in 1:Ny
            (wet[i, j] && cells[i, j] == 0) || continue
            best, distance = nothing, Inf
            for di in -ICE_FILL_CELLS:ICE_FILL_CELLS, dj in -ICE_FILL_CELLS:ICE_FILL_CELLS
                a, b = i + di, j + dj
                (1 <= a <= Nx && 1 <= b <= Ny && cells[a, b] > 0 && di^2 + dj^2 < distance) || continue
                best, distance = (a, b), di^2 + dj^2
            end
            isnothing(best) ? (unreached += 1) : (ℵ[i, j] = ℵ[best...]; V[i, j] = V[best...])
        end
        n == 1 && unreached > 0 &&
            @warn "$unreached child ocean cells are over $ICE_FILL_CELLS cells from GLORYS ocean: no GLORYS ice there"
        for k in eachindex(ℵ)
            open_water = !wet[k] || isnan(ℵ[k]) || ℵ[k] < ICE_ℵ_MIN
            ℵ[k] = open_water ? 0.0 : min(ℵ[k], 1.0)
            V[k] = open_water ? 0.0 : max(V[k], 0.0)
        end
        push!(ℵs, ℵ); push!(Vs, V)
    end
    # A daily GLORYS record is the mean over its day: place it at noon
    return (; dates = starts .+ Hour(12), ℵ = stack(ℵs), V = stack(Vs))
end

function glorys_ice_at(ice, date)
    n = searchsortedlast(ice.dates, date)
    1 <= n < length(ice.dates) || error("$date is outside the GLORYS ice records")
    w = (date - ice.dates[n]) / (ice.dates[n+1] - ice.dates[n])
    return (1 - w) .* ice.ℵ[:, :, n] .+ w .* ice.ℵ[:, :, n+1],
           (1 - w) .* ice.V[:, :, n] .+ w .* ice.V[:, :, n+1]
end

function ice_totals(ℵ, V)
    A = [(6.371e6 * deg2rad(1 / SCALING))^2 * cosd(φ₁ + (j - 0.5) / SCALING) for i in 1:Nx, j in 1:Ny]
    return sum(ℵ .* A) / 1e12, sum(V .* A) / 1e9   # 10⁶ km², km³
end

# GLORYS' seafloor for the nest's bathymetry blend and Flather depth mean: NumericalEarth's default
# (deptho) when tools/prefetch_glorys.jl fetched it, else the seafloor GLORYS' temperature marks by its
# missing values on the first day, which is NumericalEarth's own fallback for a dataset without a static
# depth (nested_ocean_model.jl). Same region as the nest's own snapshot.
parent_seafloor(first_date) =
    isfile(metadata_path(GLORYS_FILES.deptho)) ?
        nothing :
        regrid_bathymetry(underlying_grid, Metadatum(:temperature; dataset = GLORYSDaily(), date = first_date,
                                                     region = BoundingBox(grid; padding = 1/6), dir = GLORYS_DIR);
                          major_basins = Inf)

# ── the nest ──────────────────────────────────────────────────────────────────────────────────
# As the Bouvet example: GLORYS daily, all open sides (the south has no wet columns), the default
# relaxation band and rate, the seafloor blended toward GLORYS' own. Unlike it, and like
# sowinwarp_nested.jl's B-SOSE parent, EVERY daily record of the strips is held in memory
# (PARENT_IN_MEMORY): nested_ocean_simulation loads them all here before the first step, inpainting each
# on the GPU (or loading NumericalEarth's cache of it from an earlier start), and the run never touches
# GLORYS on disk again. The strips sit on the GPU with the model; their size is estimated from the files
# first, and the run stops here if they would not fit.
const PARENT_RECORDS = length(bracketing_dates(GLORYSDaily(), :temperature, (start_date, end_date)))
parent_records_in_memory = PARENT_IN_MEMORY ? PARENT_RECORDS : 2

# A record's size: each strip variable's cut (longitude, latitude[, depth]), plus 3 halo cells a side, in Float64
record_bytes(m) = NCDataset(metadata_path(m)) do ds
    8 * prod(n -> n + 6, size(ds[dataset_variable_name(m)])[1:end-1])   # the last dimension is time (1)
end
let bytes_per_record = sum(record_bytes, GLORYS_FILES.first_strip_records),
    need = parent_records_in_memory * bytes_per_record
    if ARCH isa GPU
        free = CUDA.available_memory()
        reserve = 10e9   # the model, sea ice, coupler and output buffers on top
        need > free - reserve &&
            error(@sprintf("the GLORYS parent needs %.1f GB of GPU memory for %d daily records, and %.1f GB of this GPU's %.1f GB free is left after a %.0f GB reserve for the model. Use a GPU with more memory, or set PARENT_IN_MEMORY = false (streams 2 records from disk).", need / 1e9, parent_records_in_memory, max(free - reserve, 0) / 1e9, free / 1e9, reserve / 1e9))
        @info @sprintf("GLORYS parent: loading %d daily records into memory, %.1f GB of the GPU's %.1f GB free", parent_records_in_memory, need / 1e9, free / 1e9)
    else
        @info @sprintf("GLORYS parent: loading %d daily records into memory, %.1f GB", parent_records_in_memory, need / 1e9)
    end
end

# NumericalEarth's DataWrangling info lines are dropped from here on: two per GLORYS record inpainted while
# the parent loads (~22,000), then nothing worth reading, as the parent is all in memory. The load gets one
# summary line instead, and the log is the model's. Its warnings, and every other module's lines, still show.
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

const PARENT_LOAD_T₀ = time()
const PARENT_TO_INPAINT = count(needs_inpainting, GLORYS_FILES.metadata)
ocean = nested_ocean_simulation(grid, GLORYSDaily();
                                dates = (start_date, end_date),
                                dir = GLORYS_DIR,
                                parent_time_indices_in_memory = parent_records_in_memory,
                                parent_bottom_height = parent_seafloor(start_date))
@info @sprintf("GLORYS parent: loaded in %.1f min, %d records inpainted on the way (the rest from earlier caches)", (time() - PARENT_LOAD_T₀) / 60, PARENT_TO_INPAINT - count(needs_inpainting, GLORYS_FILES.metadata))

child = ocean.model.child

# ── atmosphere: the same ERA5 files as sowinwarp_nested.jl ────────────────────────────────────
era5_dir = get(ENV, "ERA5_DIR", joinpath(DATA_DIR, "era5"))
era5_region = BoundingBox(longitude = (λ₁ - 1.0, λ₂ + 1.0), latitude = (φ₁ - 1.0, φ₂ + 1.0))
era5_dataset = ERA5YearlySingleLevel()
era5_pressure = :mean_sea_level_pressure
era5_start = start_date - Hour(1)
era5_end   = end_date + Hour(1)

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

# ── sea ice: as sowinwarp_nested.jl (remapping, free-slip, grounding switch) ──────────────────
ice_dynamics = ICE_GROUNDING ? NumericalEarth.SeaIces.sea_ice_dynamics(child.grid, ocean) :
                               NumericalEarth.SeaIces.sea_ice_dynamics(child.grid, ocean; basal_stress = nothing)
sea_ice = sea_ice_simulation(child.grid, ocean; advection = IncrementalRemapping(),
                             lateral_boundary_condition = :free_slip,
                             dynamics = ice_dynamics)
@info "sea-ice grounding: $(ICE_GROUNDING ? "ON (LandfastBasalStress)" : "OFF")" *
      " -- basal_stress is $(nameof(typeof(sea_ice.model.dynamics.basal_stress)))"

const GLORYS_ICE = load_glorys_ice(start_date, end_date)

if !PICKUP
    ℵ₀, V₀ = glorys_ice_at(GLORYS_ICE, start_date)
    h₀ = map((a, v) -> a >= ICE_ℵ_MIN ? v / a : 0.0, ℵ₀, V₀)
    ℵ₀[h₀ .== 0] .= 0
    set!(sea_ice.model.ice_concentration, reshape(ℵ₀, Nx, Ny, 1))
    set!(sea_ice.model.ice_thickness, reshape(h₀, Nx, Ny, 1))
    fill_halo_regions!(sea_ice.model.ice_concentration)
    fill_halo_regions!(sea_ice.model.ice_thickness)
    area₀, volume₀ = ice_totals(ℵ₀, V₀)
    _, volume_if_cell_mean = ice_totals(ℵ₀, h₀)    # h₀ = V₀/ℵ₀ is sithick itself
    @info @sprintf("sea ice from GLORYS on %s: area %.2f 10⁶ km², volume %.0f km³ taking sithick as floe thickness (%.0f km³ if it were volume per cell area; B-SOSE ~650), h up to %.1f m",
                   
                   start_date, area₀, volume₀, volume_if_cell_mean, maximum(h₀))
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

# ── ice at the open sides: relaxed to GLORYS's in the 4-cell band (sowinwarp_nested.jl's "bsose") ─
const ICE_EDGE = 4
const ICE_FIELDS = (sea_ice.model.ice_thickness, sea_ice.model.ice_concentration, sea_ice.model.snow_thickness)
const ICE_RELAXATION = ICE_RELAXATION_HOURS * hours
const ICE_TARGET_EVERY = 1hours
const ICE_BAND = on_architecture(ARCH, [i <= ICE_EDGE || i > Nx - ICE_EDGE || j > Ny - ICE_EDGE
                                        for i in 1:Nx, j in 1:Ny])
const ICE_TARGET_ℵ = on_architecture(ARCH, zeros(Nx, Ny))
const ICE_TARGET_V = on_architecture(ARCH, zeros(Nx, Ny))
const ICE_TARGET_TIME = Ref(-Inf)

function refresh_ice_target!(t)
    t - ICE_TARGET_TIME[] < ICE_TARGET_EVERY && return nothing
    ℵ★, V★ = glorys_ice_at(GLORYS_ICE, start_date + Millisecond(round(Int, 1000t)))
    copyto!(ICE_TARGET_ℵ, ℵ★)
    copyto!(ICE_TARGET_V, V★)
    ICE_TARGET_TIME[] = t
    return nothing
end

@inline relaxed_area(ℵ, ℵ★, r) = (1 - r) * ℵ + r * ℵ★
@inline function relaxed_thickness(h, ℵ, ℵ★, V★, r)
    a = relaxed_area(ℵ, ℵ★, r)
    return ifelse(a < ICE_ℵ_MIN, zero(h), ((1 - r) * h * ℵ + r * V★) / max(a, ICE_ℵ_MIN))
end

function relax_edge_ice!(sim)
    refresh_ice_target!(time(sim))
    r = min(1.0, sim.Δt / ICE_RELAXATION)
    h, ℵ, hs = (view(interior(f), :, :, 1) for f in ICE_FIELDS)
    ℵ★, V★, band = ICE_TARGET_ℵ, ICE_TARGET_V, ICE_BAND
    @. hs = ifelse(band & (relaxed_area(ℵ, ℵ★, r) < ICE_ℵ_MIN), zero(hs), hs)
    @. h  = ifelse(band, relaxed_thickness(h, ℵ, ℵ★, V★, r), h)
    @. ℵ  = ifelse(band, ifelse(relaxed_area(ℵ, ℵ★, r) < ICE_ℵ_MIN, zero(ℵ), relaxed_area(ℵ, ℵ★, r)), ℵ)
    return nothing
end

simulation.callbacks[:edge_ice] = Callback(relax_edge_ice!, IterationInterval(1))

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
    "boundaries" => "one-way nest in GLORYS daily (NumericalEarth nested_ocean_simulation, defaults)",
    "atmosphere" => "ERA5 (NCAR ds633.0), $(era5_pressure)",
    "sea_ice" => "ClimaSeaIce via sea_ice_simulation defaults, IncrementalRemapping advection, free-slip coasts, " *
                 "started from GLORYS siconc/sithick, open-side bands relaxed to GLORYS ice on $(ICE_RELAXATION / hours) h",
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
const CHECKPOINT_DAYS = parse(Float64, get(ENV, "CHECKPOINT_DAYS", SMOKE ? "1" : "365"))
simulation.output_writers[:checkpointer] = Checkpointer(coupled_model;
                                                        schedule = TimeInterval(CHECKPOINT_DAYS * days),
                                                        prefix = "$(RUN_NAME)_checkpoint")


PICKUP && @info "picking up $RUN_NAME from its latest checkpoint"
@info "starting"
run!(simulation; pickup = PICKUP, checkpoint_at_end = true)
@info "done"
