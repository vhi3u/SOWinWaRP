# Do this first if this is your first time running the model on HPC:
# using Pkg
Pkg.instantiate()

using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids
using NumericalEarth
using NumericalEarth.ECCO
using CUDA: has_cuda_gpu, allowscalar
using NCDatasets
using CopernicusMarine
using Dates
using Oceananigans.TurbulenceClosures

# add switches here 

if has_cuda_gpu()
    arch = GPU()
else
    arch = CPU()
end

# bathymetry: use ETOPO 1 which is already in NumericalEarth

λ₁, λ₂ = (90, 150)
φ₁, φ₂ = (-70, -40)

z = ReferenceToStretchedDiscretization(; extent=5000,
    constant_spacing=5.0,
    constant_spacing_extent=50,
    stretching=PowerLawStretching(1.15))

Nx = 1 * Int(λ₂ - λ₁) # 1/2 th of a degree resolution
Ny = 1 * Int(φ₂ - φ₁) # 1/2 th of a degree resolution
Nz = length(z)

grid = LatitudeLongitudeGrid(arch;
    size=(Nx, Ny, Nz),
    latitude=(φ₁, φ₂),
    longitude=(λ₁, λ₂),
    z,
    halo=(7, 7, 7))

# bathymetry: automatically regrid from NumericalEarth
bottom_height = regrid_bathymetry(grid,
    height_above_water=1,
    minimum_depth=10,
    interpolation_passes=25)

grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom_height))

# ==============================================================================
# BSOSE 3D Body Forcing (Temperature, Salinity)
# ==============================================================================
using Interpolations

function get_nodes(grid, loc)
    g = grid isa ImmersedBoundaryGrid ? grid.underlying_grid : grid
    λ_nodes = loc[1] === Center ? g.λᶜᵃᵃ[1:g.Nx] : g.λᶠᵃᵃ[1:g.Nx]
    φ_nodes = loc[2] === Center ? g.φᵃᶜᵃ[1:g.Ny] : g.φᵃᶠᵃ[1:g.Ny]

    if loc[3] === Nothing
        z_nodes = [0.0]
    else
        z_nodes = loc[3] === Center ? g.z.cᵃᵃᶜ[1:g.Nz] : g.z.cᵃᵃᶠ[1:g.Nz]
    end

    return λ_nodes, φ_nodes, z_nodes
end

function load_bsose_field_time_series(filename, var_name, grid, time_indices; loc=(Center, Center, Center), missing_val=0.0)
    @info "Interpolating $var_name from $filename onto model grid..."
    ds = NCDataset(filename)

    lon_name = haskey(ds, "XC") ? "XC" : "XG"
    lat_name = haskey(ds, "YC") ? "YC" : "YG"

    lon_raw = coalesce.(ds[lon_name][:], 0.0)
    lat_raw = coalesce.(ds[lat_name][:], 0.0)
    z_raw = coalesce.(ds["Z"][:], 0.0)

    # Reverse Z to be strictly increasing for Interpolations.jl
    z_rev = reverse(z_raw)
    x_nodes, y_nodes, z_nodes = get_nodes(grid, loc)

    # Assume 30 days per time index for forcing interpolation
    times = Float64[(i - 1) * 30days for i in 1:length(time_indices)]
    fts = FieldTimeSeries{loc[1],loc[2],loc[3]}(grid, times)

    for (i, t_idx) in enumerate(time_indices)
        data = ds[var_name][:, :, :, t_idx]
        data_clean = map(x -> (ismissing(x) || x == 0.0) ? missing_val : x, data)
        data_rev = reverse(data_clean, dims=3)

        itp = linear_interpolation((lon_raw, lat_raw, z_rev), data_rev, extrapolation_bc=Interpolations.Flat())

        for k in 1:grid.Nz, j in 1:grid.Ny, i_idx in 1:grid.Nx
            fts[i][i_idx, j, k] = itp(x_nodes[i_idx], y_nodes[j], z_nodes[k])
        end
    end

    close(ds)
    return fts
end

function load_bsose_2d_field_time_series(filename, var_name, grid, time_indices; loc=(Center, Center, Nothing), scaling_factor=1.0, missing_val=0.0)
    @info "Interpolating $var_name from $filename onto model grid (2D)..."
    ds = NCDataset(filename)

    lon_name = haskey(ds, "XC") ? "XC" : "XG"
    lat_name = haskey(ds, "YC") ? "YC" : "YG"

    lon_raw = coalesce.(ds[lon_name][:], 0.0)
    lat_raw = coalesce.(ds[lat_name][:], 0.0)

    x_nodes, y_nodes, _ = get_nodes(grid, loc)

    times = Float64[(i - 1) * 30days for i in 1:length(time_indices)]
    fts = FieldTimeSeries{loc[1],loc[2],loc[3]}(grid, times)

    for (i, t_idx) in enumerate(time_indices)
        data = ds[var_name][:, :, t_idx]
        data_clean = map(x -> (ismissing(x) || x == 0.0) ? missing_val : x, data) .* scaling_factor

        itp = linear_interpolation((lon_raw, lat_raw), data_clean, extrapolation_bc=Interpolations.Flat())

        for j in 1:grid.Ny, i_idx in 1:grid.Nx
            fts[i][i_idx, j, 1] = itp(x_nodes[i_idx], y_nodes[j])
        end
    end

    close(ds)
    return fts
end

# Sponge layer / mask
@inline function sponge_mask(x, y, z)
    # Surface nudging (top 50m)
    surface_mask = z > -50.0 ? 1.0 : 0.0

    # Edge nudging (within 2 degrees of edges)
    # This is CRITICALLY required to keep the deep ACC jet flowing!
    east_mask = x > 148.0 ? (x - 148.0) / 2.0 : 0.0
    west_mask = x < 92.0 ? (92.0 - x) / 2.0 : 0.0
    north_mask = y > -42.0 ? (y + 42.0) / 2.0 : 0.0
    south_mask = y < -68.0 ? (-68.0 - y) / 2.0 : 0.0

    edge_mask = clamp(east_mask + west_mask + north_mask + south_mask, 0.0, 1.0)

    return clamp(surface_mask + edge_mask, 0.0, 1.0)
end

# Load first 12 months for 1 year of forcing
time_indices = 1:12
T_bsose = load_bsose_field_time_series("data/bsose_i105_2008to2012_monthly_Theta.nc", "THETA", grid, time_indices, loc=(Center, Center, Center), missing_val=4.0)
S_bsose = load_bsose_field_time_series("data/bsose_i105_2008to2012_monthly_Salt.nc", "SALT", grid, time_indices, loc=(Center, Center, Center), missing_val=35.0)
U_bsose = load_bsose_field_time_series("data/bsose_i105_2008to2012_monthly_Uvel.nc", "UVEL", grid, time_indices, loc=(Face, Center, Center), missing_val=0.0)
V_bsose = load_bsose_field_time_series("data/bsose_i105_2008to2012_monthly_Vvel.nc", "VVEL", grid, time_indices, loc=(Center, Face, Center), missing_val=0.0)

ρ₀ = 1026.0
cₚ = 3991.0

taux_fts = load_bsose_2d_field_time_series("data/bsose_i105_2008to2012_monthly_oceTAUX.nc", "oceTAUX", grid, time_indices, loc=(Face, Center, Nothing), scaling_factor=1 / ρ₀)
tauy_fts = load_bsose_2d_field_time_series("data/bsose_i105_2008to2012_monthly_oceTAUY.nc", "oceTAUY", grid, time_indices, loc=(Center, Face, Nothing), scaling_factor=1 / ρ₀)
tflux_fts = load_bsose_2d_field_time_series("data/bsose_i105_2008to2012_monthly_surfTflx.nc", "TFLUX", grid, time_indices, loc=(Center, Center, Nothing), scaling_factor=1 / (ρ₀ * cₚ))

# Construct 3D time-varying Relaxation forcing with a 30-day timescale
rate = 1 / 30days
bsose_forcings = (
    u=Relaxation(rate=rate, mask=sponge_mask, target=U_bsose),
    v=Relaxation(rate=rate, mask=sponge_mask, target=V_bsose),
    T=Relaxation(rate=rate, mask=sponge_mask, target=T_bsose),
    S=Relaxation(rate=rate, mask=sponge_mask, target=S_bsose)
)

u_bc = FluxBoundaryCondition(taux_fts)
v_bc = FluxBoundaryCondition(tauy_fts)
T_bc = FluxBoundaryCondition(tflux_fts)

u_bcs = FieldBoundaryConditions(
    top=u_bc,
    east=NormalFlowBoundaryCondition(U_bsose),
    west=NormalFlowBoundaryCondition(U_bsose),
    north=ValueBoundaryCondition(U_bsose),
    south=ValueBoundaryCondition(U_bsose)
)
v_bcs = FieldBoundaryConditions(
    top=v_bc,
    east=ValueBoundaryCondition(V_bsose),
    west=ValueBoundaryCondition(V_bsose),
    north=NormalFlowBoundaryCondition(V_bsose),
    south=NormalFlowBoundaryCondition(V_bsose)
)
T_bcs = FieldBoundaryConditions(
    top=T_bc,
    east=ValueBoundaryCondition(T_bsose),
    west=ValueBoundaryCondition(T_bsose),
    north=ValueBoundaryCondition(T_bsose),
    south=ValueBoundaryCondition(T_bsose)
)
S_bcs = FieldBoundaryConditions(
    east=ValueBoundaryCondition(S_bsose),
    west=ValueBoundaryCondition(S_bsose),
    north=ValueBoundaryCondition(S_bsose),
    south=ValueBoundaryCondition(S_bsose)
)

boundary_conditions = (u=u_bcs, v=v_bcs, T=T_bcs, S=S_bcs)

# Add horizontal viscosity/diffusivity to prevent grid-scale noise
vertical_closure = NumericalEarth.Oceans.default_ocean_closure()
eddy_closure = IsopycnalSkewSymmetricDiffusivity(κ_skew=1e3, κ_symmetric=1e3)
closures = (vertical_closure, eddy_closure)

# construct model
ocean = ocean_simulation(grid; boundary_conditions=boundary_conditions, forcing=bsose_forcings, closure=closures)

# Initialize the model's tracers at t=0 using the first time step of BSOSE data
set!(ocean.model,
    u=U_bsose[1],
    v=V_bsose[1],
    T=T_bsose[1],
    S=S_bsose[1])
# ==============================================================================
# Print Model Info
# ==============================================================================
@info "--- Model Setup Complete ---"
@info "Grid Resolution: Nx=$(grid.Nx), Ny=$(grid.Ny), Nz=$(grid.Nz)"
@info "Grid Details:" grid
@info "" ocean.model

# ==============================================================================
# Simulation Setup & Progress
# ==============================================================================

# NOTE: To run a 1-year simulation, simply change `stop_time = 365days`
simulation = Simulation(ocean.model, Δt=10minutes, stop_time=365days)

# Adjust the timestep dynamically based on CFL (allows up to 2 hours to prevent late-stage blowup)
wizard = TimeStepWizard(cfl=0.3, max_change=1.1, max_Δt=2hours)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

# Add a simple custom progress messenger to bypass Oceanostics' DiffusiveCFL crashing
progress(sim) = @info "Time: $(prettytime(sim.model.clock.time)), Iteration: $(sim.model.clock.iteration)"
simulation.callbacks[:progress] = Callback(progress, TimeInterval(1days))

# ==============================================================================
# Output Writers
# ==============================================================================
u, v, w = ocean.model.velocities
T, S = ocean.model.tracers

simulation.output_writers[:surface] = NetCDFWriter(
    ocean.model,
    (; u, v, T, S),
    filename="surface_fields.nc",
    schedule=TimeInterval(1days),
    indices=(:, :, grid.Nz), # Extract the top layer only
    overwrite_existing=true
)

# Mid-longitude slice
mid_lon_idx = div(grid.Nx, 2)
simulation.output_writers[:mid_lon] = NetCDFWriter(
    ocean.model,
    (; u, v, T, S),
    filename="mid_lon.nc",
    schedule=TimeInterval(1days),
    indices=(mid_lon_idx, :, :),
    overwrite_existing=true
)

@info "Starting simulation..."
run!(simulation)
@info "Simulation complete!"
