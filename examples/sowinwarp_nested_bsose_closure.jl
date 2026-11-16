# SOWinWaRP (90-150E, 70-45S, 1/6°) nested in B-SOSE, with B-SOSE's closure in place of CATKE.
#
#     julia --project=. examples/sowinwarp_nested_bsose_closure.jl
#
# A fixed configuration: every setting is written below, nothing is read from the environment. Two years from
# 2014-06-01T12:00 as 2Y_nested_bsose_conv_hj, on B-SOSE's own bathymetry and levels, with Hammond & Jones
# (2016) Antarctic fresh water. Otherwise as examples/sowinwarp_nested.jl (fix9): open-boundary sea ice from
# B-SOSE, ERA5, penetrating shortwave, no ice grounding. Writes the same NetCDF files, so every tool in tools/
# reads it: <RUN_NAME>_{surface,midlon,free_surface,surface_fluxes,eke,sea_ice}.nc.

using CUDA                          # defines GPU()
using CopernicusClimateDataStore    # an ERA5 backend: the ERA5 constructors call `download` even when the files exist
using NumericalEarth
using NumericalEarth.DataWrangling: metadata_path, bracketing_dates
using NumericalEarth.SeaIces.ClimaSeaIce: IncrementalRemapping
using Oceananigans
using Oceananigans.Units
using Oceananigans.Diagnostics: NaNChecker
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.OutputReaders: Clamp
using Oceananigans.TurbulenceClosures: HorizontalScalarDiffusivity, ConvectiveAdjustmentVerticalDiffusivity
using Oceananigans.TimeSteppers: VerticallyImplicitTimeDiscretization
using NCDatasets                    # loads Oceananigans' NetCDFWriter
using Dates, Printf

include(joinpath(@__DIR__, "..", "src", "setup_bsose.jl"))          # BSOSEMonthly, resolve_bsose_filename, load_coord
include(joinpath(@__DIR__, "..", "src", "coastal_freshwater.jl"))   # coastal_freshwater_forcing

# ## Run settings

const SMOKE    = false      # true: 2 days, 6-hourly output, a checkpoint every day
const PICKUP   = false      # true: continue RUN_NAME from its latest checkpoint
const RUN_NAME = SMOKE ? "bsose_conv_smoke" : "2Y_nested_bsose_conv_hj"
const BDIR     = "/storage/scratch1/1/vnguyen480/SOWinWaRP/data"   # B-SOSE, its grid.nc, the 5-day ice, era5/
const ARCH     = GPU()

start_date = DateTime(2014, 6, 1, 12)      # a native B-SOSE stamp: the nest looks the initial state up by exact date
end_date   = SMOKE ? DateTime(2014, 6, 3, 12) : DateTime(2016, 6, 1, 12)
output_interval = SMOKE ? 6hours : 1days
checkpoint_interval = SMOKE ? 1days : 365days

for name in ("RUN_NAME", "END_DATE", "SMOKE", "PICKUP", "ARCH", "BSOSE_DIR", "ERA5_DIR", "CHECKPOINT_DAYS")
    haskey(ENV, name) && @warn "$name = $(ENV[name]) is set in the environment; this script ignores it"
end
ARCH isa GPU && !CUDA.functional() && error("CUDA is not functional on this node. Check the GPU allocation.")
@info "$RUN_NAME: $start_date to $end_date, PICKUP = $PICKUP"

# ## Grid
#
# 1/6° over 90-150°E, 70-45°S, on B-SOSE iteration 156's 52 static levels (4.2 m at the surface to 400 m, 6000 m),
# so the parent lands on the child level for level and the top cell is the layer B-SOSE's surface fields describe.

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
z = reverse(-vcat(0.0, cumsum(BSOSE_DRF))) .+ 0.0     # faces bottom to top; .+ 0.0 turns -0.0 into 0.0
const Nz = length(z) - 1

underlying_grid = LatitudeLongitudeGrid(ARCH; size = (Nx, Ny, Nz), halo = (7, 7, 7), z,
                                        longitude = (λ₁, λ₂), latitude = (φ₁, φ₂))

# ## Bathymetry: B-SOSE's own
#
# `Depth` in B-SOSE's grid.nc (partial cells included). Its longitudes are the nest's cell centres, so each nest
# column takes one B-SOSE column; its latitudes are Mercator-like (0.057° at 70°S to 0.117° at 45°S), so each nest
# cell gathers the 1-3 B-SOSE cells whose centres fall inside it, weighted by latitude span × cos(lat). A cell is
# ocean when at least half of that area is wet, at the mean depth of the wet part, and at least 10 m deep. Only the
# ocean connected to the open sides is kept. (XG/YG in that grid.nc are corrupt and RAC is missing over land, so
# neither is read.)

function bsose_bottom_height(path)
    ds = NCDataset(path)
    X = Float64.(ds["XC"][:, 1])
    Y = Float64.(ds["YC"][1, :])
    D = Float64.(coalesce.(ds["Depth"][:, :], 0.0))
    close(ds)

    edges = vcat(1.5Y[1] - 0.5Y[2], (Y[1:end-1] .+ Y[2:end]) ./ 2, 1.5Y[end] - 0.5Y[end-1])
    A = diff(edges) .* cosd.(Y)
    columns = findall(x -> λ₁ < x < λ₂, X)
    length(columns) == Nx || error("$path has $(length(columns)) longitudes in $(λ₁)-$(λ₂)E, the nest $Nx")

    area, wet, volume = zeros(Nx, Ny), zeros(Nx, Ny), zeros(Nx, Ny)
    for jb in eachindex(Y), ib in columns
        i = floor(Int, (X[ib] - λ₁) * SCALING) + 1
        j = floor(Int, (Y[jb] - φ₁) * SCALING) + 1
        1 <= j <= Ny || continue
        area[i, j] += A[jb]
        D[ib, jb] > 0 && (wet[i, j] += A[jb]; volume[i, j] += A[jb] * D[ib, jb])
    end
    zb = [wet[i, j] >= 0.5 * area[i, j] ? -max(volume[i, j] / wet[i, j], 10.0) : 0.0 for i in 1:Nx, j in 1:Ny]

    # one basin: the ocean reachable from the open sides (W, E, N)
    reached = falses(Nx, Ny)
    queue = [(i, j) for i in 1:Nx, j in 1:Ny if (i == 1 || i == Nx || j == Ny) && zb[i, j] < 0]
    foreach(c -> reached[c...] = true, queue)
    while !isempty(queue)
        i, j = pop!(queue)
        for (a, b) in ((i-1, j), (i+1, j), (i, j-1), (i, j+1))
            (1 <= a <= Nx && 1 <= b <= Ny && !reached[a, b] && zb[a, b] < 0) || continue
            reached[a, b] = true
            push!(queue, (a, b))
        end
    end
    zb[.!reached] .= 0.0
    return zb
end

# Dead-end pockets (ocean with land on 3 or 4 sides) become land, repeatedly, so one-cell inlets close from
# their dead end outward. 2Y_nested_seaice_fix1's 10 m inlet under a polynya went from 34.5 to 46 g/kg in 26 days.
function fill_dead_end_pockets!(zb)
    land(i, j) = 1 <= i <= Nx && 1 <= j <= Ny && zb[i, j] >= 0   # off the grid counts as ocean
    filled = 0
    while true
        pockets = [(i, j) for i in 1:Nx, j in 1:Ny
                   if zb[i, j] < 0 && count((land(i-1, j), land(i+1, j), land(i, j-1), land(i, j+1))) >= 3]
        isempty(pockets) && return filled
        foreach(c -> zb[c...] = 0, pockets)
        filled += length(pockets)
    end
end

zb = bsose_bottom_height(joinpath(BDIR, "grid.nc"))
filled = fill_dead_end_pockets!(zb)
bottom_height = Field{Center, Center, Nothing}(underlying_grid)
set!(bottom_height, reshape(zb, Nx, Ny, 1))
fill_halo_regions!(bottom_height)
@info "bathymetry: B-SOSE i156 Depth, $(count(zb .< 0)) wet columns, $filled dead-end pocket cells filled"

grid = ImmersedBoundaryGrid(underlying_grid, PartialCellBottom(bottom_height; minimum_fractional_cell_height = 0.2);
                            active_cells_map = true)

# ## Ocean, nested in B-SOSE
#
# nested_ocean_simulation reads B-SOSE monthly T, S, u and v on strips along the open sides (W, E, N; the south
# is Antarctica) and starts the ocean from B-SOSE on start_date. Every parent record is held in memory, loaded
# before the first step. NumericalEarth's defaults for the open boundaries, the Flather barotropic exterior, the
# bathymetry blend toward B-SOSE's seafloor and the 1/6-hour relaxation band.
#
# CLOSURE: B-SOSE's, as far as Oceananigans can. MITgcm runs B-SOSE with KPP, which Oceananigans does not have,
# over Laplacian horizontal mixing and constant vertical backgrounds. Here those backgrounds (ν, κ = 10 m²/s
# horizontal; νz = 1e-3, κz = 1e-4 m²/s) and convective adjustment (1 m²/s where the column is statically
# unstable), no CATKE: a bracket on KPP, not a copy. It has no bulk Richardson number, no non-local flux and no
# wind-driven entrainment, so the wind deepens the mixed layer only through resolved shear. Vertically implicit.
#
# COASTAL FRESHWATER: Hammond & Jones (2016), the Antarctic ice-shelf melt, iceberg melt and runoff B-SOSE
# prescribes (341 Gt/yr in the box, at most 2.7 m/yr), constant in time, as a virtual salt flux in the top cell
# added to the nest's relaxation. tools/hammond_jones_freshwater.jl writes coastal_freshwater_hj.nc; what falls on
# our land is spread over the wet cells within 6 cells. Not the B-SOSE residual of fix5-fix9, whose polynya
# hotspots (20-44 m/yr) follow the winter ice term, not melt.

dataset = BSOSEMonthly(dir = BDIR, iteration = 156)
parent_records = length(bracketing_dates(dataset, :temperature, (start_date, end_date)))

closure = (HorizontalScalarDiffusivity(ν = 10.0, κ = 10.0),
           ConvectiveAdjustmentVerticalDiffusivity(VerticallyImplicitTimeDiscretization();
                                                   convective_κz = 1.0, convective_νz = 1.0,
                                                   background_κz = 1e-4, background_νz = 1e-3))

coastal_freshwater = coastal_freshwater_forcing("coastal_freshwater_hj.nc", bottom_height;
                                                λ₁, φ₁, scaling = SCALING, arch = ARCH)

ocean = nested_ocean_simulation(grid, dataset;
                                dates = (start_date, end_date),
                                dir = BDIR,
                                parent_time_indices_in_memory = parent_records,
                                forcing = coastal_freshwater,
                                closure)

child = ocean.model.child   # the regional ocean; ocean.model also holds the B-SOSE parent

# Penetrating shortwave. The coupler finds the ocean's TwoColorRadiation with get_radiative_forcing, which the
# branch defines for a HydrostaticFreeSurfaceModel but not for a NestedModel, so all the shortwave went into the
# top cell's heat flux up to fix5_flux. Forward it to the child, unless the package already does.
if Base.unwrap_unionall(which(NumericalEarth.Oceans.get_radiative_forcing, (NumericalEarth.NestedModel,)).sig).parameters[2] === Any
    NumericalEarth.Oceans.get_radiative_forcing(nm::NumericalEarth.NestedModel) = NumericalEarth.Oceans.get_radiative_forcing(nm.child)
end

# ## Sea ice, nested in B-SOSE's
#
# ClimaSeaIce on the child grid. Ice crosses the open sides as in NumericalEarth's Arctic example: B-SOSE's
# thickness and concentration on inflow, ice let out but never pulled in, no snow brought in; it starts from
# B-SOSE's ice on start_date. B-SOSE's ice is its 5-day SIarea (ℵ) and SIheff (volume h·ℵ per cell area), placed at
# each window's centre (stamp − 2.5 days), regridded once: B-SOSE land takes its nearest B-SOSE ocean value within
# 6 cells (the coastlines differ), then bilinear. h = SIheff/SIarea; ℵ below 0.01 is open water. The parent's ice
# velocity is zero, so at the boundary the ice drifts with the nest's surface ocean.
#
# Incremental remapping moves ℵ and volume together, so h stays bounded where ℵ is small (WENO inflated h at the
# 97E corner in fix2). Free-slip coasts, MITgcm's default. No grounding: NumericalEarth turns on ClimaSeaIce's
# LandfastBasalStress, which pinned a 47 m pile in a shallow pocket up to fix6; MITgcm has it off.

ice_file(token, year) = joinpath(BDIR, "$(token)_bsoseI156_$(year)_5day.nc")

function bsose_ice(first_date, last_date)
    ds = NCDataset(joinpath(BDIR, resolve_bsose_filename(BDIR, :surface_heat_flux, 156)))
    xc, yc = load_coord(ds, "XC", 1), load_coord(ds, "YC", 2)
    iB = let r = findall(x -> λ₁ - 1 <= x <= λ₂ + 1, xc); minimum(r):maximum(r) end
    jB = let r = findall(y -> φ₁ - 1 <= y <= φ₂ + 1, yc); minimum(r):maximum(r) end
    wetB = Float64.(coalesce.(ds["Depth"][iB, jB], 0.0)) .> 0
    close(ds)
    xs, ys = xc[iB], yc[jB]

    lin = LinearIndices(wetB)
    source = map(CartesianIndices(wetB)) do I
        best, d² = 0, Inf
        for di in -6:6, dj in -6:6
            J = I + CartesianIndex(di, dj)
            (checkbounds(Bool, wetB, J) && wetB[J] && di^2 + dj^2 < d²) || continue
            best, d² = lin[J], di^2 + dj^2
        end
        best
    end

    λc = λ₁ .+ ((1:Nx) .- 0.5) ./ SCALING
    φc = φ₁ .+ ((1:Ny) .- 0.5) ./ SCALING
    bracket(s, x) = (a = clamp(searchsortedlast(s, x), 1, length(s) - 1); (a, clamp((x - s[a]) / (s[a+1] - s[a]), 0, 1)))
    function stencil(A, a, wx, b, wy)
        num = den = 0.0
        for (di, w₁) in ((0, 1 - wx), (1, wx)), (dj, w₂) in ((0, 1 - wy), (1, wy))
            s = source[a+di, b+dj]
            s > 0 && (num += w₁ * w₂ * A[s]; den += w₁ * w₂)
        end
        return den > 0 ? num / den : 0.0
    end
    wet = zb .< 0
    regrid(A) = [wet[i, j] ? stencil(A, bracket(xs, λc[i])..., bracket(ys, φc[j])...) : 0.0 for i in 1:Nx, j in 1:Ny]

    # the run's years, and a neighbour year on disk so a run near New Year is still bracketed
    years = [y for y in year(first_date)-1:year(last_date)+1
             if year(first_date) <= y <= year(last_date) || isfile(ice_file("SIArea", y))]
    dates, ℵs, Vs = DateTime[], Matrix{Float64}[], Matrix{Float64}[]
    for y in years, (token, names, out) in (("SIArea", ("SIarea", "SIArea", "SIAREA"), ℵs),
                                             ("SIHeff", ("SIheff", "SIHeff", "SIHEFF"), Vs))
        ds = NCDataset(ice_file(token, y))
        name = first(filter(n -> haskey(ds, n), names))
        A = Float64.(coalesce.(ds[name][iB, jB, :], 0.0))
        token == "SIArea" && append!(dates, DateTime.(ds["time"][:]) .- Hour(60))
        close(ds)
        append!(out, [regrid(A[:, :, n]) for n in axes(A, 3)])
    end
    (first(dates) <= first_date && last(dates) >= last_date) ||
        error("B-SOSE ice covers $(first(dates)) to $(last(dates)), not $first_date to $last_date")

    ℵ, V = stack(ℵs), stack(Vs)
    open_water = .!(ℵ .>= 0.01)
    ℵ[open_water] .= 0; V[open_water] .= 0
    clamp!(ℵ, 0, 1); V .= max.(V, 0)
    return (; dates, ℵ, V)
end

function parent_ice(ice)
    times = [Dates.value(d - start_date) / 1000 for d in ice.dates]       # seconds since start_date
    series(t) = FieldTimeSeries{Center, Center, Nothing}(child.grid, t; time_indexing = Clamp())
    h, ℵ = series(times), series(times)
    for n in eachindex(times)
        ℵⁿ, Vⁿ = ice.ℵ[:, :, n], ice.V[:, :, n]
        set!(ℵ[n], reshape(ℵⁿ, Nx, Ny, 1))
        set!(h[n], reshape(map((a, v) -> a >= 0.01 ? v / a : 0.0, ℵⁿ, Vⁿ), Nx, Ny, 1))
    end
    still = series([first(times), last(times)])
    return (; h, ℵ, u = still, v = still)
end

ice_dynamics = NumericalEarth.SeaIces.sea_ice_dynamics(child.grid, ocean; basal_stress = nothing)
sea_ice = nested_sea_ice_simulation(ocean, parent_ice(bsose_ice(start_date, end_date));
                                    advection = IncrementalRemapping(),
                                    lateral_boundary_condition = :free_slip,
                                    dynamics = ice_dynamics)

cell_area = [(6.371e6 * deg2rad(1 / SCALING))^2 * cosd(φ₁ + (j - 0.5) / SCALING) for i in 1:Nx, j in 1:Ny]
let h = Array(interior(sea_ice.model.ice_thickness))[:, :, 1], ℵ = Array(interior(sea_ice.model.ice_concentration))[:, :, 1]
    @info @sprintf("sea ice from B-SOSE: area %.2f 10⁶ km², volume %.0f km³, h up to %.1f m",
                   sum(ℵ .* cell_area) / 1e12, sum(h .* ℵ .* cell_area) / 1e9, maximum(h))
end

# ## Atmosphere and coupled model
#
# ERA5 yearly single-level files, cropped to the box + 1° (tools/download_era5_ncar.jl), from an hour before
# the start to an hour after the end: ERA5 values are means over the hour ending at their stamp. Mean sea level
# pressure, not surface pressure, which over the ice sheet acts like a sea level step of several metres.

era5 = (; dataset = ERA5YearlySingleLevel(), start_date = start_date - Hour(1), end_date = end_date + Hour(1),
          region = BoundingBox(longitude = (λ₁ - 1, λ₂ + 1), latitude = (φ₁ - 1, φ₂ + 1)),
          dir = joinpath(BDIR, "era5"))

missing_era5 = [path for name in (:eastward_velocity, :northward_velocity, :temperature, :dewpoint_temperature,
                                  :mean_sea_level_pressure, :total_precipitation, :snowfall,
                                  :downwelling_shortwave_radiation, :downwelling_longwave_radiation)
                for path in unique(metadata_path(Metadata(name; era5...))) if !isfile(path)]
isempty(missing_era5) || error("Missing ERA5 files (run tools/download_era5_ncar.jl):\n  " * join(missing_era5, "\n  "))

atmosphere = ERA5PrescribedAtmosphere(ARCH; era5..., pressure = :mean_sea_level_pressure)
radiation  = ERA5PrescribedRadiation(ARCH; era5...)

coupled_model = OceanSeaIceModel(ocean, sea_ice; atmosphere, radiation)

NumericalEarth.Oceans.get_radiative_forcing(ocean) isa NumericalEarth.Oceans.TwoColorRadiation ||
    error("the coupler does not see the child's TwoColorRadiation: the shortwave would all go into the top cell")

# ## Simulation
#
# From Δt = 1 s, the wizard lengthens Δt by at most 10% every 10 iterations, holding the advective CFL at 0.6 for
# the faster of the ocean and the ice (the ice's fastest u and v over the smallest cell: 2Y_nested_seaice_full's
# ice outran Δt and grew to 10⁶ m). A NaN anywhere stops the run, and so does ice volume h·ℵ above 50 m.

simulation = Simulation(coupled_model; Δt = 1seconds, stop_time = Dates.value(end_date - start_date) / 1000)

Δx_min = minimum_xspacing(child.grid.underlying_grid)
Δy_min = minimum_yspacing(child.grid.underlying_grid)
function ice_advection_timescale()
    uᵢ, vᵢ = sea_ice.model.velocities
    inverse = maximum(abs, uᵢ) / Δx_min + maximum(abs, vᵢ) / Δy_min
    return inverse > 0 ? 1 / inverse : Inf
end
wizard = TimeStepWizard(cfl = 0.6, max_change = 1.1, min_Δt = 0.1,
                        cell_advection_timescale = _ -> min(Oceananigans.Advection.cell_advection_timescale(child),
                                                            ice_advection_timescale()))
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

simulation.callbacks[:nan_checker] =
    Callback(NaNChecker(fields = merge(child.velocities, child.tracers,
                                       (; h = sea_ice.model.ice_thickness, ℵ = sea_ice.model.ice_concentration)),
                        erroring = true), IterationInterval(1))

function check_ice_volume(sim)
    v = maximum(interior(sea_ice.model.ice_thickness) .* interior(sea_ice.model.ice_concentration))
    v > 50 && error(@sprintf("sea ice volume %.1f m (h·ℵ) on day %.1f: the ice is blowing up", v, time(sim) / 86400))
    return nothing
end
simulation.callbacks[:ice_volume] = Callback(check_ice_volume, IterationInterval(10))

wall_time = Ref(time_ns())
function progress(sim)
    S, T = view(interior(child.tracers.S), :, :, Nz), view(interior(child.tracers.T), :, :, Nz)
    u, v, w = child.velocities
    hℵ = interior(sea_ice.model.ice_thickness) .* interior(sea_ice.model.ice_concentration)
    @info @sprintf("day %.1f, iter %d, Δt %.1f min | surface S (%.2f, %.2f), T (%.2f, %.2f) | max |u| %.2f, |v| %.2f, |w| %.1e m/s | max h·ℵ %.2f m | wall %s",
                   time(sim) / 86400, iteration(sim), sim.Δt / 60, extrema(S)..., extrema(T)...,
                   maximum(abs, u), maximum(abs, v), maximum(abs, w), maximum(hℵ),
                   prettytime(1e-9 * (time_ns() - wall_time[])))
    wall_time[] = time_ns()
    return nothing
end
simulation.callbacks[:progress] = Callback(progress, TimeInterval(output_interval))

# ## Output
#
# The files and attributes of sowinwarp_nested.jl, written off the child (the nest's prognostic state):
#   _surface.nc        u, v, T, S at the top level, daily
#   _midlon.nc         u, v, T, S on the 120°E section, daily
#   _free_surface.nc   η, daily
#   _surface_fluxes.nc what the coupler gives the ocean, daily means: heat_flux (°C m/s), ice_salt_flux
#                      (g/kg m/s), both + out of the ocean; freshwater_flux (m/s, + into the ocean); shortwave
#                      (W/m², negative down). The Hammond & Jones forcing is not in them.
#   _eke.nc            30-day means of u, v, T, S, uu, vv at cell centres, full depth
#   _sea_ice.nc        h, ℵ, snow, ice u and v, daily
# On a pickup the files are appended to.

u, v, w = child.velocities
T, S = child.tracers.T, child.tracers.S

output_metadata = Dict("run_name" => RUN_NAME, "start_date" => string(start_date), "end_date" => string(end_date),
                "dataset" => "BSOSE",
                "setup" => "examples/sowinwarp_nested_bsose_closure.jl",
                "boundaries" => "one-way nest in B-SOSE monthly (NumericalEarth nested_ocean_simulation)",
                "atmosphere" => "ERA5 (NCAR ds633.0), mean_sea_level_pressure",
                "sea_ice" => "ClimaSeaIce, IncrementalRemapping, free-slip coasts, started from B-SOSE SIarea/SIheff, " *
                             "open sides take B-SOSE 5-day ice (nested_sea_ice_simulation open boundaries)",
                "coastal_freshwater" => "constant, from coastal_freshwater_hj.nc (Hammond & Jones 2016)",
                "closure" => "bsose_conv: Laplacian 10 m2/s horizontal, nu_z 1e-3 / kappa_z 1e-4, convective adjustment 1 m2/s, no CATKE",
                "ice_grounding" => "off (basal_stress = nothing)",
                "bathymetry" => "B-SOSE i156 Depth (grid.nc), area-aggregated",
                "penetrating_shortwave" => "yes: TwoColorRadiation fed by the coupler",
                "vertical_coordinate" => "static z, B-SOSE i156's 52 levels")

ocean.output_writers[:surface] = NetCDFWriter(child, (; u, v, T, S);
                                              filename = "$(RUN_NAME)_surface.nc",
                                              schedule = TimeInterval(output_interval),
                                              indices = (:, :, Nz),
                                              global_attributes = output_metadata, overwrite_files = !PICKUP)

ocean.output_writers[:mid_lon] = NetCDFWriter(child, (; u, v, T, S);
                                              filename = "$(RUN_NAME)_midlon.nc",
                                              schedule = TimeInterval(output_interval),
                                              indices = (div(Nx, 2), :, :),
                                              global_attributes = output_metadata, overwrite_files = !PICKUP)

# η sits at (Center, Center, Face) on one level, which NetCDFWriter cannot write; averaging over that one level can
ocean.output_writers[:free_surface] = NetCDFWriter(child, (; η = Field(Average(child.free_surface.displacement, dims = 3)));
                                                   filename = "$(RUN_NAME)_free_surface.nc",
                                                   schedule = TimeInterval(output_interval),
                                                   global_attributes = output_metadata, overwrite_files = !PICKUP)

fluxes = coupled_model.interfaces.net_fluxes.ocean
surface_fluxes = (; heat_flux = fluxes.T, ice_salt_flux = fluxes.S, freshwater_flux = fluxes.η,
                    shortwave = coupled_model.radiation.interface_fluxes.ocean.downwelling_shortwave)
ocean.output_writers[:surface_fluxes] =
    NetCDFWriter(child, surface_fluxes;
                 filename = "$(RUN_NAME)_surface_fluxes.nc",
                 schedule = AveragedTimeInterval(output_interval, window = output_interval),
                 global_attributes = merge(output_metadata, Dict(
                     "flux_convention" => "kinematic, positive OUT of the ocean, except freshwater_flux (+ into the ocean) and shortwave (W/m2, negative down)",
                     "shortwave_in_heat_flux" => "no",
                     "salt_flux_seen_by_ocean" => "ice_salt_flux + S_top * freshwater_flux, S_top from the surface file",
                     "averaging_window_seconds" => Float64(output_interval))),
                 overwrite_files = !PICKUP)

u_cc = @at (Center, Center, Center) u
v_cc = @at (Center, Center, Center) v
ocean.output_writers[:eke] = NetCDFWriter(child, (; u = u_cc, v = v_cc, T, S, uu = u_cc^2, vv = v_cc^2);
                                          filename = "$(RUN_NAME)_eke.nc",
                                          schedule = AveragedTimeInterval(30days, window = 30days, stride = 1),
                                          indices = (:, :, :),
                                          global_attributes = merge(output_metadata, Dict(
                                              "averaging_window_seconds" => Float64(30days),
                                              "averaging_stride" => 1,
                                              "moment_location" => "Center, Center, Center",
                                              "eke_recipe" => "EKE = 0.5*((uu - u^2) + (vv - v^2)); MKE = 0.5*(u^2 + v^2)")),
                                          overwrite_files = !PICKUP)

sea_ice.output_writers[:sea_ice] = NetCDFWriter(sea_ice.model,
                                                merge((; h = sea_ice.model.ice_thickness,
                                                         ℵ = sea_ice.model.ice_concentration,
                                                         hs = sea_ice.model.snow_thickness),
                                                      sea_ice.model.velocities);
                                                filename = "$(RUN_NAME)_sea_ice.nc",
                                                schedule = TimeInterval(output_interval),
                                                global_attributes = output_metadata, overwrite_files = !PICKUP)

# ## Checkpoints and run
#
# The coupled state (the child ocean, the ice, the clocks, every writer's schedule) every checkpoint_interval
# and at the end, as <RUN_NAME>_checkpoint_iteration<N>.jld2. Set PICKUP = true to continue from the latest.

simulation.output_writers[:checkpointer] = Checkpointer(coupled_model;
                                                        schedule = TimeInterval(checkpoint_interval),
                                                        prefix = "$(RUN_NAME)_checkpoint")

@info "output: $(RUN_NAME)_{surface,midlon,free_surface,surface_fluxes,eke,sea_ice}.nc in $(pwd())"
run!(simulation; pickup = PICKUP, checkpoint_at_end = true)
