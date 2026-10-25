using Pkg
Pkg.instantiate()

using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids
using NumericalEarth
using NumericalEarth.ECCO
using CopernicusClimateDataStore    # activates the ERA5 download backend (needs ~/.cdsapirc)
using NumericalEarth.DataWrangling: all_dates
using CUDA: has_cuda_gpu, allowscalar
using NCDatasets
using Dates
using Printf
using Statistics: mean
using SeawaterPolynomials
using Oceananigans.TurbulenceClosures
using Oceananigans.Grids: φnode
using Oceananigans.Operators: Azᶜᶜᶜ, Ax_qᶠᶜᶜ, Ay_qᶜᶠᶜ
using Oceananigans.BoundaryConditions: PerturbationAdvection, NormalRadiation, ObliqueRadiation,
                                       TracerReservoir
using Oceananigans.Diagnostics: NaNChecker
using Oceanostics.ProgressMessengers: TimedMessenger

# B-SOSE helpers: dataset, boundaries, forcing, initial conditions.
include("setup_bsose.jl")

import CUDA

if CUDA.functional()
    arch = GPU()
    @info "Running on GPU: $(CUDA.name(CUDA.device()))"
elseif haskey(ENV, "FORCE_CPU")
    arch = CPU()
    @warn "Running on CPU (forced by FORCE_CPU environment variable)"
else
    # Call functional(true) to output the exact reason why CUDA failed to initialize
    CUDA.functional(true)
    error("CUDA is not functional on this node! If you are running via sbatch, check your GPU allocation or logs. To force CPU testing, export FORCE_CPU=1.")
end

# ==============================================================================
# CONFIGURATION -- everything you change between runs lives here and nowhere else.
#
# The reasoning behind these values is NOT repeated here. It is in the architecture note,
# vhieuvault/Research (GATO)/SoWinWaRP/Model Architecture.md, which carries the measurements,
# the disproven theories and the traps. Keep that file current when you change this one.
# ==============================================================================

# ---- run identity -------------------------------------------------------------------------
# Prefixes every output file: RUN_NAME_surface.nc, _midlon.nc, _free_surface.nc.
# Change it per experiment or runs overwrite each other. Animate: ./run_animate_simulation NAME
const RUN_NAME = "20D_bsose_bathyfix"

# ---- period and domain --------------------------------------------------------------------
const START_DATE = DateTime("2014-06-01")
const END_DATE   = DateTime("2014-06-21")   # 20 days: a boundary test, not a science run

const SCALING    = 6        # horizontal resolution = 1/SCALING degrees (1/6 ~ 15 km)
const DZ_SURFACE = 5        # m, upper-ocean spacing -> ~135 levels, 100 of them above 500 m
const DZ_BOTTOM  = 200      # m, deep spacing
const CIRCUMPOLAR = false   # true = 0-360 degrees, Periodic in x (west/east BCs then unused)

const BSOSE_ITERATION = 156 # 156 covers 2013-2024, 105 covers 2008-2012. No fallback: a
                            # missing file is an error. Filenames: BSOSE_FILES in setup_bsose.jl

# Condition the sea floor on the open boundaries (see `fix_boundary_topography!`).
#
# The model's fastest cell has been the BOTTOM cell of a column on an open boundary in every
# run: (1,21,39) in a 483 m column, growing ~0.3 m/s/day. Three scheme hypotheses have been
# tested and eliminated -- missing bottom drag (no, it is active), the radiation stencil at
# immersed-adjacent cells (no, Dirichlet diverges in 3 days), and the transport pin amplifying
# ΔU/H (no, identical growth without it). The bathymetry is what is left, and the east face
# carries a 42 m column at (360,10) against neighbours of ~438 m: a 75x velocity amplifier.
const FIX_BOUNDARY_TOPOGRAPHY = true
const MINIMUM_BOUNDARY_DEPTH  = 100   # m; masks (360,10) at 42 m and nothing else

# ---- open boundaries ----------------------------------------------------------------------
const OBCS = true                           # false = whole-region restoring instead

const OBC_SCHEME    = "NormalRadiation"     # normal u/v: NormalRadiation | PerturbationAdvection
                                            #             | ObliqueRadiation | clamped
const TANGENTIAL_BC = :radiation            # tangential u/v: :radiation | :clamped | :gradient
                                            #   | :oblique | :perturbation_advection | ...
const RADIATION_TAU_IN = 6hours             # inflow nudging; outflow is Inf throughout
const OBLIQUE_TAU_IN   = 3days              # ObliqueRadiation only
const OBLIQUE_TAU_OUT  = 360days

# Tracers. :same gives T and S whatever OBC_SCHEME the normal flow uses.
const TRACER_SCHEME = TracerReservoir(inflow_length_scale = 5000, outflow_length_scale = 0)

# DO NOT SET OBC_SCHEME = "clamped" HERE. Tested 2026-10-24 and it DIVERGES: NaN in u at
# day 3.1, with Δt collapsing 5.6 -> 1.2 min and max|η| quadrupling 1.22 -> 4.86 m, against a
# NormalRadiation run that reached day 20. A hard Dirichlet over-specifies an OUTFLOW boundary,
# and the ACC leaves through the east face at +215 Sv; the eddies arriving there had nowhere to
# go. Radiation with outflow_timescale = Inf is what lets them out.
#
# The divergence started at (i=355, j=11) and (i=359, j=87) -- the thin-column cluster on the
# east face, where (360,10) is 42 m against neighbours of ~438 m. That is evidence the thin
# columns GENERATE the problem and radiation has been letting it escape, not causing it.

# Do not set ObliqueRadiation here, for two independent reasons.
#
# 1. It buys nothing. NestingValidation's truncation experiment puts oblique and normal within
#    noise of each other (enstrophy 0.184 vs 0.187); the whole gain, +32%, comes from having a
#    TANGENTIAL condition at all, which TANGENTIAL_BC provides either way.
#
# 2. It diverges next to land, and most of our faces have some. West and east are genuine
#    open-ocean ACC boundaries -- about 132 of 150 rows wet -- but the domain runs to 70S and
#    the Wilkes Land coast is at 66-68S, so the southern ~18 rows of every zonal face are
#    continent, and the south face is land across its whole length. The criterion is not "is
#    this face mostly ocean" but "does any wet cell on it have a land neighbour along the face":
#    six land cells out of 575 diverged a NestingValidation run, and oblique is what
#    destabilised this model once already. North, at 45S, is probably the one clean face.
#
# The startup log prints each face's wet fraction, which is the measurement, not an inference.

# ---- barotropic mode and the free surface -------------------------------------------------
# Not optional. The split-explicit corrector overwrites the depth mean of u/v at every face
# with the barotropic solution, so without these the net transport is driven to zero whatever
# the 3-D conditions say.
const BAROTROPIC_OBC    = true
const BAROTROPIC_SCHEME = :pinned   # :pinned (Flather + target_transport) | :flather
#
# DO NOT SET :flather WITHOUT A CORRECT eta_ext. Tested 2026-10-25 and the circulation INVERTS:
#
#     face      :pinned     :flather
#     west       +186.8      +234.3
#     east       +215.4       -60.3   <- sign flipped, ACC enters through BOTH zonal faces
#     north       -28.6      +295.2   <- and exits north
#
# with mean eta drifting -0.44 -> -0.53 m over ten days and still going. The cause is
# EXTERIOR_SSH = false: Flather's reference is a flat zero sea surface while the model sits near
# -0.5 m, and at the measured ~5 Sv/cm that is ~250 Sv of spurious forcing. The pin has been
# absorbing it. NestingValidation removed this same correction with no effect, but their own
# caveat applies -- they tested a 719 km^2 section carrying 24-45 Sv, not a basin carrying 200.
#
# The test was worth running anyway: |u| at (1,21,39) grew at 0.30 m/s/day WITHOUT the pin
# against 0.32 WITH it. Same cell, same rate. The pin is not the amplifier.
#
const BAROTROPIC_FACES  = :all      # :all | :zonal -- keep :all, the :zonal theory was
                                    # disproven (north carries -31 Sv, 18% of the zonal flow)
const EXTERIOR_SSH      = false     # eta_ext and the initial eta from B-SOSE ETAN.
                                    # OFF (2026-10-24). `bsose_exterior_free_surface` aborts the
                                    # process on the GPU -- SIGBUS / SIGABRT / corrupted heap, at
                                    # the same point every run, surviving a full depot wipe. This
                                    # skips the function entirely. It also takes eta_ext out of
                                    # the loop, which is worth doing on its own: its residual
                                    # range is -3.44 to +1.30 m, a 4.74 m span where the real
                                    # SSH drop across this domain is ~2 m. With eta_ext = 0 and
                                    # BAROTROPIC_SCHEME = :pinned the net transport is still
                                    # fixed, so only the along-face shape is given up.

# ---- surface forcing ----------------------------------------------------------------------
const WINDS                = false  # BSOSE oceTAUX / oceTAUY
const SURFACE_FLUXES       = false  # BSOSE TFLUX / SFLUX
const PENETRATING_SHORTWAVE = false # split oceQsw out of TFLUX into TwoColorRadiation
const SPONGE_LAYERS        = false  # interior restoring near the open edges
const ATMOSPHERE           = :none  # :none | :era5 (bulk formulae; needs ~/.cdsapirc)
const ERA5_THREADS         = 1      # CDS rejects concurrent queued requests

# Derived: BSOSE surface forcing only applies when no coupled atmosphere is computing it.
const BSOSE_WINDS     = WINDS && ATMOSPHERE === :none
const BSOSE_FLUXES    = SURFACE_FLUXES && ATMOSPHERE === :none
const BSOSE_SHORTWAVE = PENETRATING_SHORTWAVE && ATMOSPHERE === :none

# ---- physics and run control ---------------------------------------------------------------
const CLOSURE_CONFIG = "ito"   # ito | bsose | henyey_gm | biharmonic | none (built below)
const STOP_ITERATION = Inf     # cut the run short for a smoke test
const CHECKPOINTS    = false   # write restart checkpoints every 30 days
const BOUNDARY_DIAGNOSTICS = true  # free-surface file, per-face transport log, 30-min eta watch

# ==============================================================================
# End of configuration. Nothing below needs editing for a routine run.
# ==============================================================================

# Domain setup

# horizontal domain extent
if CIRCUMPOLAR
    λ₁, λ₂ = (0, 360)
    φ₁, φ₂ = (-78, 30)
else
    λ₁, λ₂ = (90, 150)
    φ₁, φ₂ = (-70, -45)
end

# z stretching so that the upper 500 meters of the ocean has dz = 5 meters, and the deeper ocean will gradually stretch to 200 m vertical resolution. 
if arch isa CPU
    z = ReferenceToStretchedDiscretization(; extent=5800,
        constant_spacing=10,
        maximum_spacing=1000,
        constant_spacing_extent=500,
        stretching=PowerLawStretching(1.15))
else
    z = ReferenceToStretchedDiscretization(; extent=5800,
        constant_spacing=DZ_SURFACE,
        maximum_spacing=DZ_BOTTOM,
        constant_spacing_extent=500,
        stretching=PowerLawStretching(1.12))
end

# grid
# if running on CPU, do a 1 degree horizontal resolution run:
if arch isa CPU
    Nx = Int((λ₂ - λ₁))
    Ny = Int((φ₂ - φ₁))
else
    Nx = Int(SCALING * (λ₂ - λ₁))
    Ny = Int(SCALING * (φ₂ - φ₁))
end
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
    interpolation_passes=5)

# NOTE: `match_boundary_bathymetry!` went with glorys_obc.jl. It blended the child bottom
# toward the GLORYS parent over a 4-cell band at the open edges, and also carried our port of
# MITgcm's `OBCSfixtopo` (a boundary column is never deeper than its inward neighbour). The
# blend was GLORYS-specific and does not apply here, but OBCSfixtopo is not -- MITgcm applies it
# whatever the parent. It is recoverable from git at 142f2c1 if we want it back for B-SOSE.

# Condition the sea floor along the open boundaries before the immersed grid freezes it.
# Masks columns too thin to carry transport and applies MITgcm's OBCSfixtopo; see
# `fix_boundary_topography!` for what each does and the numbers behind the threshold.
if FIX_BOUNDARY_TOPOGRAPHY
    fix_boundary_topography!(bottom_height, grid;
                             minimum_boundary_depth = MINIMUM_BOUNDARY_DEPTH)
end

grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom_height))

@info grid

# Wet fraction of each open face.
#
# NestingValidation's rule is that ObliqueRadiation needs a side that does not terminate in
# coast: six land cells out of 575 was enough to diverge their run, and they fall back to
# NormalRadiation on their southern boundary precisely because it is only 58% wet. Our own
# history agrees -- ObliqueRadiation is what destabilised this model once already.
#
# So print it, and treat anything short of 100% as disqualifying for oblique on that side.
# A 0% face is entirely land: it carries no transport, contributes no area to the balance,
# and its boundary condition is inert.
for side in (:west, :east, :south, :north)
    widths, depths = face_geometry(side, grid)
    nwet = count(>(0), depths)
    @info @sprintf(" -> %-5s face: %3d/%3d columns wet (%3.0f%%), wet area %.3e m^2",
                   side, nwet, length(depths), 100 * nwet / max(length(depths), 1),
                   sum(widths .* depths))
end

# dataset conditonal

dataset = BSOSEMonthly(iteration=BSOSE_ITERATION)

@info "Dataset: $(summary(dataset)) -- boundaries, forcing and initial conditions"

# The run window comes from START_DATE / END_DATE in the configuration block above. Everything
# downstream follows from them: the stop time, the monthly B-SOSE records selected for the
# boundaries and forcing, and the date the initial condition is read from.
start_date = START_DATE
end_date   = END_DATE

end_date > start_date || error("end_date ($end_date) must be later than start_date ($start_date).")

dates = (start_date, end_date)
simulation_seconds = Dates.value(Second(end_date - start_date))
simulation_days = simulation_seconds / 86400

# Monthly B-SOSE records for the boundary conditions, winds and sponge nudging.
all_available_dates = all_dates(dataset, :temperature)

if end_date < first(all_available_dates) || start_date > last(all_available_dates)
    error("""$(summary(dataset)) holds no data for $start_date to $end_date. Its monthly records \
             run from $(first(all_available_dates)) to $(last(all_available_dates)); set start_date \
             and end_date above to a period inside that range.""")
end

# Take the records whose data falls inside the run and extend by one at each end. Taking the
# whole window rather than a fixed twelve is what makes a multi-year run follow the real
# BSOSE years instead of replaying one year cyclically; the extra record at each end lets
# the series interpolate across the first and last half-month rather than wrapping around
# to the other end of the record.
#
# The test is on each record's window CENTRE, not on its stamp. A record's value sits at the
# middle of the month it averages while its stamp sits at the end of that month, so
# selecting on stamps pads the head by two records and leaves the tail half a month short --
# which silently hands the last two weeks of the run to the cyclical wrap.
inside = findall(d -> start_date <= bsose_window_center(d) <= end_date, all_available_dates)

lo, hi = if isempty(inside)   # a run shorter than the gap between records: bracket it
    (something(findlast(d -> bsose_window_center(d) <= start_date, all_available_dates), firstindex(all_available_dates)),
        something(findfirst(d -> bsose_window_center(d) >= end_date, all_available_dates), lastindex(all_available_dates)))
else
    (first(inside), last(inside))
end

lo = max(firstindex(all_available_dates), lo - 1)
hi = min(lastindex(all_available_dates), hi + 1)
bc_dates = all_available_dates[lo:hi]

if bsose_window_center(last(bc_dates)) < end_date
    @warn """$(summary(dataset)) records stop at $(last(bc_dates)), before the run ends at $end_date. \
             The forcing will wrap cyclically for the remainder of the run."""
end

@info @sprintf("Simulating %s to %s (%.1f days)", start_date, end_date, simulation_days)
@info "BSOSE records: $(length(bc_dates)) monthly, $(bc_dates[1]) to $(bc_dates[end])"
@info "Model time zero corresponds to $start_date; all forcing is stepped on that clock."

# ==============================================================================
# Boundary Conditions & Forcing
# ==============================================================================

# Assigned below when a Flather condition is built; the `!OBCS` branch leaves it at 0, which
# the initial-condition block tests against.
exterior_η = 0


if OBCS
    @info "Configuring BSOSE Open Boundary Conditions with radiation scheme..."
    # In an open-ocean regional domain with strong outflow (ACC carrying ~130 Sv eastward through 150°E),
    # rigid Dirichlet boundaries reflect outgoing eddies and waves, leading to tracer blowup at the boundary.
    # PerturbationAdvection / NormalRadiation with outflow_timescale=Inf allows internal waves/eddies to freely
    # radiate out of the domain, while inflow_timescale=1days nudges incoming boundary flow (West) smoothly to BSOSE.
    obc_scheme = if OBC_SCHEME == "ObliqueRadiation"
        ObliqueRadiation(inflow_timescale=OBLIQUE_TAU_IN, outflow_timescale=OBLIQUE_TAU_OUT)
    elseif OBC_SCHEME == "PerturbationAdvection"
        PerturbationAdvection(inflow_timescale=1days, outflow_timescale=Inf)
    elseif OBC_SCHEME == "NormalRadiation"
        NormalRadiation(inflow_timescale=RADIATION_TAU_IN, outflow_timescale=Inf)
    elseif OBC_SCHEME == "clamped" || OBC_SCHEME == "none"
        nothing
    else
        error("Unknown OBC_SCHEME: $OBC_SCHEME. Choose 'ObliqueRadiation', 'PerturbationAdvection', 'NormalRadiation', or 'clamped'.")
    end
    @info "  -> OBC Scheme: $(obc_scheme === nothing ? "Clamped Dirichlet" : summary(obc_scheme))"
    @info "  -> Tangential velocity condition: $TANGENTIAL_BC"

    # Exterior sea surface for the Flather condition. Without it ηᵉˣᵗ is 0, which tells every
    # open face the sea surface is flat -- inconsistent with the 200 Sv of geostrophic
    # transport the same condition prescribes, and the reason the east face delivered -176 Sv
    # of inflow where +203 Sv of outflow was asked for.
    exterior_η = BAROTROPIC_OBC && EXTERIOR_SSH ?
                 bsose_exterior_free_surface(grid; dataset=dataset, dates=bc_dates,
                     reference_date=start_date) : 0

    boundary_conditions = bsose_open_boundary_conditions(grid; dataset=dataset, dates=bc_dates,
        exterior_free_surface=exterior_η,
        winds=WINDS, surface_fluxes=SURFACE_FLUXES,
        penetrating_shortwave=SURFACE_FLUXES && PENETRATING_SHORTWAVE, scheme=obc_scheme,
        tangential_bc_kind=TANGENTIAL_BC, tracer_scheme=TRACER_SCHEME,
        radiation_inflow_timescale=RADIATION_TAU_IN,
        oblique_inflow_timescale=OBLIQUE_TAU_IN,
        oblique_outflow_timescale=OBLIQUE_TAU_OUT, reference_date=start_date,
        barotropic=BAROTROPIC_OBC, barotropic_faces=BAROTROPIC_FACES,
        barotropic_scheme=BAROTROPIC_SCHEME)

    if SPONGE_LAYERS
        sponge_timescale = 30days
        @info "Configuring boundary edge sponge layers (3.0° width, $(sponge_timescale) restoring timescale)..."
        forcings = bsose_sponge_layer_forcing(grid; dataset=dataset, dates=bc_dates, sponge_width=3.0, timescale=sponge_timescale, restore_velocities=true, reference_date=start_date)
    else
        forcings = NamedTuple()
    end
elseif !OBCS
    @info "Configuring whole-region DatasetRestoring (30-day restoring for T and S from BSOSE)..."
    wind_bcs = BSOSE_WINDS ? bsose_surface_wind_stress(grid; dataset=dataset, dates=bc_dates, reference_date=start_date) : nothing
    boundary_conditions = wind_bcs !== nothing ? (u=FieldBoundaryConditions(top=wind_bcs.u), v=FieldBoundaryConditions(top=wind_bcs.v)) : NamedTuple()

    restoring_region = bsose_region(grid)
    temperature_metadata = Metadata(:temperature; dataset, start_date, end_date, region=restoring_region)
    salinity_metadata = Metadata(:salinity; dataset, start_date, end_date, region=restoring_region)

    FT = DatasetRestoring(temperature_metadata, grid; rate=1 / 30days)
    FS = DatasetRestoring(salinity_metadata, grid; rate=1 / 30days)

    restore_on_model_clock!(FT, start_date)
    restore_on_model_clock!(FS, start_date)

    forcings = (T=FT, S=FS)
else
    boundary_conditions = NamedTuple()
    forcings = NamedTuple()
end

# Turbulence closures:
# 1. Vertical: CATKE / default NumericalEarth vertical closure
# ------------------------------------------------------------------------------
# Turbulence Closures & Eddy Parameterizations
# ------------------------------------------------------------------------------
# 1. Base vertical mixing: CATKE (TKE-based boundary layer and interior mixing)
catke_closure = NumericalEarth.Oceans.default_ocean_closure()

# 2. Closure configuration switch (set by CLOSURE_CONFIG at the top of this file):
#   - "ito" (Default): Ito et al. (2026) MITgcm baseline configuration
#         * Horizontal: Biharmonic viscosity & diffusivity Ah = 3×10⁹ m⁴/s (momentum & tracers, Laplacian K=0)
#         * Vertical: CATKE + background vertical diffusivity & viscosity (νz = 10⁻⁵ m²/s, κz = 10⁻⁵ m²/s)
#         * Specifically chosen for 1/6° to permit an active mesoscale eddy field while damping 2Δx noise.
#   - "bsose": BSOSE / ECCO standard Laplacian parameterization
#         * Horizontal: Laplacian viscosity νh = 10 m²/s, diffusivity κh = 10 m²/s
#         * Vertical: CATKE + Laplacian viscosity νz = 10⁻³ m²/s, diffusivity κz = 10⁻⁴ m²/s
#   - "henyey_gm": Henyey et al. (1986) + Gent-McWilliams (GM) isopycnal eddy closure
#         * Horizontal: Biharmonic viscosity with timescale of 15 days (ν = Az² / 15days)
#         * Vertical: CATKE + Henyey latitude-dependent vertical diffusivity κz(φ) = max(2e-6, 3e-5 * |sin(φ)|), νz = 10⁻⁵ m²/s
#         * Eddy: IsopycnalSkewSymmetricDiffusivity (κ_skew = 500, κ_symmetric = 200)
#   - "biharmonic": Constant biharmonic (ν = 10¹⁰ m⁴/s, κ = 10¹⁰ m⁴/s) + CATKE
#   - "none": CATKE vertical mixing only
closures = if CLOSURE_CONFIG == "ito"
    # Ito et al. (2026): biharmonic Ah = 3e9 m⁴/s, background νz,κz = 1e-5 m²/s
    horizontal_viscosity = HorizontalScalarBiharmonicDiffusivity(ν=3e9, κ=3e9)
    vertical_diffusivity = VerticalScalarDiffusivity(ν=1e-5, κ=1e-5)
    (catke_closure, horizontal_viscosity, vertical_diffusivity)

elseif CLOSURE_CONFIG == "bsose"
    # BSOSE standard values:
    # Horizontal viscosity 10 m² s⁻¹, Horizontal diffusivity 10 m² s⁻¹
    # Vertical viscosity 1e-3 m² s⁻¹, Vertical diffusivity 1e-4 m² s⁻¹
    horizontal_diffusivity = HorizontalScalarDiffusivity(ν=10.0, κ=10.0)
    vertical_diffusivity = VerticalScalarDiffusivity(ν=1e-3, κ=1e-4)
    (catke_closure, horizontal_diffusivity, vertical_diffusivity)

elseif CLOSURE_CONFIG == "henyey_gm"
    # Biharmonic horizontal viscosity with timescale of 15 days
    @inline νhb(i, j, k, grid, timescale) = Azᶜᶜᶜ(i, j, k, grid)^2 / timescale
    ν_field = Field{Center,Center,Center}(grid)
    set!(ν_field, KernelFunctionOperation{Center,Center,Center}(νhb, grid, 15days))
    horizontal_viscosity = HorizontalScalarBiharmonicDiffusivity(ν=ν_field, κ=ν_field)

    # Background vertical diffusivity following Henyey et al. (1986)
    @inline henyey_diffusivity(i, j, k, grid) = max(2e-6, 3e-5 * abs(sind(φnode(i, j, k, grid, Center(), Center(), Center()))))
    κz_field = Field{Center,Center,Center}(grid)
    set!(κz_field, KernelFunctionOperation{Center,Center,Center}(henyey_diffusivity, grid))
    vertical_diffusivity = VerticalScalarDiffusivity(ν=1e-5, κ=κz_field)

    # Gent-McWilliams & Redi isopycnal eddy closure
    eddy_closure = IsopycnalSkewSymmetricDiffusivity(κ_skew=500, κ_symmetric=200)

    (catke_closure, eddy_closure, horizontal_viscosity, vertical_diffusivity)

elseif CLOSURE_CONFIG == "biharmonic"
    horizontal_viscosity = HorizontalScalarBiharmonicDiffusivity(ν=1e10, κ=1e10)
    (catke_closure, horizontal_viscosity)

elseif CLOSURE_CONFIG == "none"
    catke_closure
else
    error("Unknown CLOSURE_CONFIG: $CLOSURE_CONFIG. Choose 'ito', 'bsose', 'henyey_gm', 'biharmonic', or 'none'.")
end

@info "Selected closure configuration: $CLOSURE_CONFIG"
@info "Turbulence closures: $(summary(closures))"


# build the ocean model

# Penetrating solar radiation.
#
# `ocean_simulation` builds a TwoColorRadiation by default, so passing one changes nothing on
# its own -- what matters is whether its surface flux is ever filled. Outside a coupled
# EarthSystemModel nothing fills it, so the default scheme sits at zero and does nothing.
# `bsose_penetrating_shortwave` returns one together with the callback that feeds it oceQsw,
# and the callback is registered on the simulation below.
radiation = TwoColorRadiation(grid)
update_shortwave! = nothing

if PENETRATING_SHORTWAVE && ATMOSPHERE !== :none
    @info """Penetrating shortwave is handled by the $(ATMOSPHERE) coupling: TwoColorRadiation's \
             surface flux is filled by the coupled model, which is what the BSOSE oceQsw \
             callback was standing in for. Not registering that callback."""
elseif PENETRATING_SHORTWAVE
    if !SURFACE_FLUXES
        @warn """PENETRATING_SHORTWAVE has no effect while SURFACE_FLUXES is false: with no \
                 surface heat flux there is no solar part to separate out."""
    else
        radiation, update_shortwave! = bsose_penetrating_shortwave(grid; dataset=dataset,
            dates=bc_dates, reference_date=start_date)
    end
end

@info "Constructing ocean simulation model..."
ocean = ocean_simulation(grid;
    boundary_conditions=boundary_conditions,
    forcing=forcings,
    closure=closures,
    radiative_forcing=radiation
)

# initial conditions based on either BSOSE or the other datasets

bsose_initial_conditions!(ocean.model; dataset=dataset, date=start_date, velocities=true)

# Initialise the free surface too. Setting u, v, T, S from BSOSE while leaving η flat is
# internally inconsistent: BSOSE's velocities are in geostrophic balance with a sea
# surface that drops about 2 m from 45S to the Antarctic margin, and starting without it
# leaves the model to build that tilt from scratch. Every run so far has done this --
# max|η| jumps from 0 to about 2.5 m on day 1 in all of them, which is the model
# manufacturing the sea surface it should have been handed.
#
# With Flather it is worse than a spin-up transient: ηᵉˣᵗ carries the tilt while ηᵇ starts
# at zero, so the condition sees a 0.7 m error on day one and answers it with hundreds of
# Sverdrups. The 2D field below shares the datum removed from the face values, so the
# interior and the boundary agree at t = 0 by construction.
if exterior_η isa NamedTuple
    set!(ocean.model.free_surface.displacement, exterior_η.initial)
    fill_halo_regions!(ocean.model.free_surface.displacement)
    # `mean` over a GPU view calls `first(A)`, which is scalar indexing. Julia's logging
    # swallows the exception, so the line silently vanishes instead of failing -- the
    # third time this has bitten in this file. Copy to the host first.
    local η₀ = Array(interior(ocean.model.free_surface.displacement))
    @info @sprintf(" -> free surface initialised from BSOSE ETAN: mean %+.3f m, range %+.3f to %+.3f m",
        mean(η₀), minimum(η₀), maximum(η₀))
end

# ── TEOS10 sqrt guard ─────────────────────────────────────────────────────────
# The TEOS10 coordinate s(Sᴬ) = √((Sᴬ + ΔS) / Sₐᵤ) crashes for Sᴬ < -32.
# Guard with max(..., 0) so that any residual bad values from immersed/halo cells
# don't crash the run — they will return zero (freshwater) density rather than NaN.
import SeawaterPolynomials.TEOS10 as TEOS10_mod
@eval SeawaterPolynomials.TEOS10 begin
    @inline s(Sᴬ::FT) where FT = √(max((Sᴬ + FT(ΔS)) / FT(Sₐᵤ), zero(FT)))
end

# ==============================================================================
# Atmosphere
#
# A prescribed atmosphere and downwelling radiation,
# coupled through `OceanOnlyModel`, so the stress and the heat and freshwater fluxes come from
# bulk formulae against real atmospheric state rather than being prescribed fields.
#
# `OceanOnlyModel` wraps the ocean SIMULATION, and the simulation then steps the coupled model
# rather than the ocean model directly. Everything that reaches into the model has to go
# through `.ocean.model` from here on -- which is why the callbacks below close over
# `ocean.model` instead of taking it from `sim.model`.
# ==============================================================================
stepped_model = ocean.model     # what Simulation steps; replaced below if there is an atmosphere

if ATMOSPHERE === :era5
    era5_dir = get(ENV, "ERA5_DIR", joinpath(default_bsose_directory(), "era5"))
    mkpath(era5_dir)
    era5_region = BoundingBox(longitude = (λ₁ - 1.0, λ₂ + 1.0), latitude = (φ₁ - 1.0, φ₂ + 1.0))

    @info "Building ERA5 prescribed atmosphere and radiation ($(start_date) to $(end_date))..."
    @info " -> ERA5 directory: $era5_dir"
    @info """ -> ERA5 needs the CopernicusClimateDataStore backend and a CDS API key in \
              ~/.cdsapirc. That is a different account from the Copernicus Marine one \
              uses; without it the download below will fail."""

    # Serially first -- see `prefetch_era5!`. The constructors below download with one thread
    # per variable, which CDS rejects outright.
    prefetch_era5!(start_date = start_date, end_date = end_date,
                   region = era5_region, dir = era5_dir, threads = ERA5_THREADS)

    atmosphere = ERA5PrescribedAtmosphere(arch;
        start_date = start_date, end_date = end_date,
        region = era5_region, dir = era5_dir)

    surface_radiation = ERA5PrescribedRadiation(arch;
        start_date = start_date, end_date = end_date,
        region = era5_region, dir = era5_dir)

    stepped_model = OceanOnlyModel(ocean; atmosphere, radiation = surface_radiation)
    @info "Coupled OceanOnlyModel built: surface fluxes now come from ERA5 bulk formulae."
elseif ATMOSPHERE !== :none
    error("Unknown ATMOSPHERE: $ATMOSPHERE. Choose :era5 or :none.")
else
    @info @sprintf("No atmosphere. Surface forcing: winds %s, heat/salt fluxes %s, penetrating shortwave %s.",
                   BSOSE_WINDS ? "BSOSE" : "OFF", BSOSE_FLUXES ? "BSOSE" : "OFF",
                   BSOSE_SHORTWAVE ? "ON" : "OFF")
end

# some housekeeping to set up the simulation (progress checks and adaptive timestep)

# The run length comes from start_date and end_date; STOP_ITERATION can cut it short.
# Start with a very small Δt so the first CATKE evaluation is numerically gentle;
# the wizard will ramp this up within a few iterations.
simulation = Simulation(stepped_model, Δt=1seconds,
    stop_time=Float64(simulation_seconds),
    stop_iteration=STOP_ITERATION)

# adaptive timestep wizard based on CFL (following mediterranean.jl: cfl=0.2, max_change=1.1)
wizard = TimeStepWizard(cfl=0.6, max_change=1.1, min_Δt=0.1)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

# Fail loudly and early on NaN. Oceananigans attaches its own checker, but it watches only the
# first prognostic field, runs every 100 iterations and does not throw -- at the Δt a blow-up
# produces that is 25 minutes of model time, and the last run outran it. The NaN then reached
# the wizard, and the run died as `InexactError: Int64(NaN)` inside the free-surface substep
# calculation, which names neither the field nor the time. This one checks every field every
# step and throws, so the error says what went wrong.
simulation.callbacks[:nan_checker] =
    Callback(NaNChecker(fields=merge(ocean.model.velocities, ocean.model.tracers), erroring=true),
        IterationInterval(1))

# Feed oceQsw to the radiation scheme. Without this callback the TwoColorRadiation attached to
# the model above keeps a zero surface flux and contributes nothing, which is the bug being
# fixed -- so register it whenever one was built.
#
# The callback fires after each time step, so the radiation runs one Δt behind the clock. Δt
# here is minutes and the series it interpolates is monthly, which makes the lag about four
# parts in 10⁴ of one month-to-month change. The one step that would matter is the first,
# before any callback has run, so prime the field now rather than let it start from zero.
if update_shortwave! !== nothing
    update_shortwave!(simulation)
    simulation.callbacks[:shortwave] = Callback(update_shortwave!, IterationInterval(1))
    @info "Penetrating shortwave is ON: oceQsw drives TwoColorRadiation, top T flux is TFLUX - oceQsw."
else
    @info "Penetrating shortwave is OFF: the whole surface heat flux lands in the top cell."
end

# Safety callback: clamp salinity/temperature to physically valid ranges before each time step.
# Operates on parent() to cover all cells including halos and immersed cells.
# Prevents DomainError in TEOS10's sqrt(S) from any numerical noise that accumulates
# in boundary-adjacent or immersed cells during time stepping.
# function clamp_salinity!(sim)
#     clamp!(parent(sim.model.tracers.S), S_MIN_PHYSICAL, S_MAX_PHYSICAL)
#     clamp!(parent(sim.model.tracers.T), T_MIN_PHYSICAL, T_MAX_PHYSICAL)
# end
# simulation.callbacks[:clamp_S] = Callback(clamp_salinity!, IterationInterval(1))


# output and progress interval (daily)
output_interval = 1days

# Wall-clock reference for the coupled progress messenger below (hoisted out of the branch:
# a `const` inside a conditional is asking for trouble).
const PROGRESS_WALL = Ref(time_ns())

# progress logger. Oceanostics' TimedMessenger reads `sim.model.velocities`, which on the
# coupled path is the EarthSystemModel and has none -- so there it is replaced by a messenger
# that closes over the ocean model. The BSOSE path keeps TimedMessenger so its logs stay
# comparable with every earlier run.
if ATMOSPHERE === :none
    progress = TimedMessenger()
    simulation.callbacks[:progress] = Callback(progress, TimeInterval(output_interval))
else
    function coupled_progress(sim)
        m = ocean.model
        u, v, w = m.velocities
        elapsed = (time_ns() - PROGRESS_WALL[]) / 1e9
        PROGRESS_WALL[] = time_ns()
        @info @sprintf("iter = %6d, time = %s, Δt = %s, walltime = %.1f s | |u⃗|ₘₐₓ = [%.2e, %.2e, %.2e] m/s",
            iteration(sim), prettytime(m.clock.time), prettytime(sim.Δt), elapsed,
            maximum(abs, u), maximum(abs, v), maximum(abs, w))
        flush(stdout)
    end
    simulation.callbacks[:progress] = Callback(coupled_progress, TimeInterval(output_interval))
end

# With Dirichlet open boundaries there is no wave radiation, so any net
# volume-flux imbalance across the four faces accumulates in the free surface.
# TimedMessenger does not report it, so track it separately.
# function log_free_surface(sim)
#     eta = sim.model.free_surface.displacement
#     @info @sprintf("        max|eta| = %.4f m, mean(eta) = %+.4f m",
#         maximum(abs, eta), mean(eta))
#     flush(stdout)
# end
# simulation.callbacks[:free_surface] = Callback(log_free_surface, TimeInterval(callback_interval))

# The free-surface watch.
#
# `maximum(abs, η)` includes immersed cells and is not a physical number -- it has read 9 m
# while the ocean itself peaked at 1.19 m -- so read the MEAN. A drifting mean η is the direct
# signature of a net volume imbalance the boundaries are not absorbing; a steady or decaying
# one means the barotropic condition is doing its job. Half-hourly, because the barotropic mode
# substeps at √(gH) ≈ 198 m/s and anything it does is invisible at daily output.
if BAROTROPIC_OBC
    function log_eta_watch(sim)
        η = ocean.model.free_surface.displacement
        @info @sprintf("        [eta] t = %8.3f d   mean η = %+.5f m",
                       ocean.model.clock.time / 86400, mean(η))
        flush(stdout)
    end
    simulation.callbacks[:eta_watch] = Callback(log_eta_watch, TimeInterval(30minutes))
end

# WHERE is the fastest cell?
#
# The progress messenger prints |u⃗|ₘₐₓ over the whole 3-D field but never says where it is.
# On 20D_bsose_v1 that number reached 32.6 m/s and grew linearly at ~1.2 m/s/day while every
# SURFACE diagnostic we had -- the maps, the edge profiles, the interior/boundary series --
# peaked at 2.84 m/s. The feature was at depth and we were blind to it for three runs.
#
# Worse, the per-face transport log read EXACTLY on target the whole time, because
# `target_transport` pins ∮U·dl by construction: a face can carry 30 m/s in one thin column,
# compensate elsewhere, and still integrate perfectly. A uniform transport correction ΔU
# becomes a velocity ΔU/H, so a shallow column turns a modest transport error into an enormous
# speed. That is why this prints the column DEPTH alongside the location.
#
# Read it as: if the fastest cell sits in shallow water within a cell or two of a face, the
# boundary is pushing transport through a column too thin to carry it.
# `bottom_height` is a bare 3-D OffsetArray with a singleton third axis, and on GPU it wraps a
# CuArray -- so copy `parent` to the host once and index through the offset, rather than
# touching it element-by-element from the callback.
let zc = Array(znodes(grid, Center())),
    B  = grid.immersed_boundary.bottom_height,
    Bh = Array(parent(grid.immersed_boundary.bottom_height))

    ox = 1 - first(axes(B, 1))
    oy = 1 - first(axes(B, 2))
    seafloor(i, j) = Bh[clamp(i, 1, grid.Nx) + ox, clamp(j, 1, grid.Ny) + oy, 1]

    function log_velocity_argmax(sim)
        for (name, f) in (("u", ocean.model.velocities.u), ("v", ocean.model.velocities.v))
            val, idx = findmax(abs, Array(interior(f)))
            i, j, k = Tuple(idx)
            @info @sprintf("        [max|%s|] %7.3f m/s at (i=%3d, j=%3d, k=%3d)  z = %8.1f m  seafloor = %8.1f m  |  %3d cells from west/east, %3d from south/north",
                           name, val, i, j, k, zc[clamp(k, 1, length(zc))], seafloor(i, j),
                           # u is Face-located in x (i runs to Nx+1), v in y (j to Ny+1), so
                           # clamp: index Nx+1 IS the east face, distance 0, not -1.
                           max(0, min(i - 1, grid.Nx + 1 - i)),
                           max(0, min(j - 1, grid.Ny + 1 - j)))
        end
        flush(stdout)
    end
    simulation.callbacks[:velocity_argmax] = Callback(log_velocity_argmax, TimeInterval(output_interval))
end

# output writers

u, v, w = ocean.model.velocities
T, S = ocean.model.tracers

# Every writer below records a `time` axis in seconds from model time zero and nothing
# else, so the file itself cannot say which calendar date time zero is -- the only date
# NetCDF adds on its own is when the file was written. Stamping the run's dates on as
# global attributes is what lets tools/animate_simulation.jl caption a frame with a real
# date instead of an elapsed day count.
output_metadata = Dict("run_name" => RUN_NAME,
    "start_date" => string(start_date),
    "end_date" => string(end_date),
    "dataset" => "BSOSE",
    "boundaries" => "$(OBC_SCHEME) normal / $(TANGENTIAL_BC) tangential, barotropic $(BAROTROPIC_SCHEME)")

simulation.output_writers[:surface] = NetCDFWriter(
    ocean.model,
    (; u, v, T, S),
    filename="$(RUN_NAME)_surface.nc",
    schedule=TimeInterval(output_interval),
    indices=(:, :, grid.Nz),
    global_attributes=output_metadata,
    overwrite_files=true
)

mid_lon_idx = div(grid.Nx, 2)
simulation.output_writers[:mid_lon] = NetCDFWriter(
    ocean.model,
    (; u, v, T, S),
    filename="$(RUN_NAME)_midlon.nc",
    schedule=TimeInterval(output_interval),
    indices=(mid_lon_idx, :, :),
    global_attributes=output_metadata,
    overwrite_files=true
)

# ==============================================================================
# Open boundary diagnostics
#
# Three things are needed to tell an open boundary from a bounded box, and none of
# them are in the surface or mid-longitude output:
#
#   1. Free-surface height. Water that cannot leave piles up in η, so a drifting
#      domain-mean η is the direct signature of a net volume imbalance.
#   2. Volume transport through each of the four faces. Net outflow should be a small
#      residual of the gross transport (the ACC alone carries ~100 Sv through this
#      domain) and should balance the rate of change of volume in the free surface.
#
# A third diagnostic -- full-depth slices at each face, written to RUN_NAME_face_*.nc -- used
# to live here. It was removed: four files per run cluttering the project directory, for
# something the transport log below already answers.
# ==============================================================================

if BOUNDARY_DIAGNOSTICS
    # η lives at (Center, Center, Face) on a single level, which NetCDFWriter cannot write
    # directly; reducing over the (length-one) vertical gives an ordinary 2D field.
    η_2d = Field(Average(ocean.model.free_surface.displacement, dims=3))

    simulation.output_writers[:free_surface] = NetCDFWriter(
        ocean.model,
        (; η=η_2d),
        filename="$(RUN_NAME)_free_surface.nc",
        schedule=TimeInterval(output_interval),
        global_attributes=output_metadata,
        overwrite_files=true
    )

    # Volume transport through each boundary face: ∫∫ u·n dA over the face.
    u_transport = KernelFunctionOperation{Face,Center,Center}(Ax_qᶠᶜᶜ, grid, u)
    v_transport = KernelFunctionOperation{Center,Face,Center}(Ay_qᶜᶠᶜ, grid, v)

    west_transport = Field(u_transport; indices=(1, :, :))
    east_transport = Field(u_transport; indices=(grid.Nx + 1, :, :))
    south_transport = Field(v_transport; indices=(:, 1, :))
    north_transport = Field(v_transport; indices=(:, grid.Ny + 1, :))

    function log_open_boundaries(sim)
        Sv = 1e6
        west = sum(compute!(west_transport)) / Sv
        east = sum(compute!(east_transport)) / Sv
        south = sum(compute!(south_transport)) / Sv
        north = sum(compute!(north_transport)) / Sv

        # Outward-positive, so `net` is the volume leaving the domain per unit time.
        net = -west + east - south + north
        gross = abs(west) + abs(east) + abs(south) + abs(north)
        residual = gross == 0 ? 0.0 : 100 * abs(net) / gross

        η = ocean.model.free_surface.displacement
        @info @sprintf("        transport (Sv): W %+8.2f | E %+8.2f | S %+8.2f | N %+8.2f || net out %+8.3f (%.2f%% of gross)",
            west, east, south, north, net, residual)
        @info @sprintf("        free surface  : mean η = %+.4f m, max|η| = %.4f m",
            mean(η), maximum(abs, η))
        flush(stdout)
    end

    simulation.callbacks[:open_boundaries] = Callback(log_open_boundaries, TimeInterval(output_interval))
end

# checkpointing

if CHECKPOINTS
    simulation.output_writers[:checkpointer] = Checkpointer(
        ocean.model,
        schedule=TimeInterval(30days),
        prefix="checkpoint",
        overwrite_files=true
    )
end

@info "--- Model Setup Complete ---"
@info "Grid Resolution: Nx=$(grid.Nx), Ny=$(grid.Ny), Nz=$(grid.Nz)"
@info "Dataset: B-SOSE iteration $BSOSE_ITERATION"
@info "Boundaries: normal/tracers $(OBC_SCHEME), tangential $(TANGENTIAL_BC), " *
      "barotropic $(BAROTROPIC_OBC ? String(BAROTROPIC_SCHEME) : "none") on $BAROTROPIC_FACES faces"
@info "Surface forcing: winds $BSOSE_WINDS | fluxes $BSOSE_FLUXES | " *
      "shortwave $(update_shortwave! !== nothing) | sponge $SPONGE_LAYERS | atmosphere $ATMOSPHERE"

# red button 

@info "Starting simulation..."
run!(simulation)
@info "Simulation completed successfully!"
