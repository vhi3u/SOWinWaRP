using Pkg
Pkg.instantiate()

using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids
using NumericalEarth
using NumericalEarth.ECCO
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
using Oceananigans.BoundaryConditions: PerturbationAdvection, NormalRadiation, ObliqueRadiation
using Oceananigans.Diagnostics: NaNChecker
using Oceanostics.ProgressMessengers: TimedMessenger

# Include our BSOSE helper module
include("setup_bsose.jl")
include("glorys_obc.jl")

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
# Run name
#
# Every file this run writes is prefixed with it:
#
#     RUN_NAME_surface.nc        surface fields, daily
#     RUN_NAME_midlon.nc         mid-longitude transect, daily
#     RUN_NAME_free_surface.nc   free-surface height, daily          (BOUNDARY_DIAGNOSTICS)
#     RUN_NAME_face_west.nc  …   the four boundary faces, 5-daily    (BOUNDARY_DIAGNOSTICS)
#
# Change it for each experiment so runs do not overwrite one another, then animate
# the result with `./run_animate_simulation RUN_NAME`.
# ==============================================================================
const RUN_NAME = "8M_glorys_obc"

# flags

const OBCS = true # open boundary conditions (NormalFlow with PerturbationAdvection)
# 02_mab_glorys_obc.jl removes the interior restoring deliberately -- "the boundary condition
# is the only thing communicating with the outside" -- so OBC_SOURCE = :glorys turns the sponge
# off to match. 05 keeps one as an option, default off. Set back to `true` to A/B it.
const SPONGE_LAYERS = false # sponge layer restoring (DatasetRestoring) near open boundary edges
const WINDS = true # time-varying surface wind forcing from BSOSE data (oceTAUX and oceTAUY)
# Surface heat and salt fluxes from BSOSE (TFLUX and SFLUX). Without these the ocean exchanges
# nothing with the atmosphere but momentum: it cannot cool in winter or warm in summer, and its
# surface temperature drifts away from BSOSE as water crosses the domain.
const SURFACE_FLUXES = true

# Distribute the solar part of the surface heat flux through the water column by Beer's law
# instead of depositing all of it in the top cell.
#
# BSOSE's TFLUX is the TOTAL heat flux, sunlight included. Sunlight is not absorbed at the
# surface: 58% of it goes within 0.35 m but the blue-green remainder decays over 23 m, so for
# the 5 m top cell here a third of it belongs deeper. Applying TFLUX whole as a top flux puts
# all of it in that one cell. Over Nov-Feb at 60-65S that is +344 W m⁻² into the top 5 m where
# it should be +97, and in February it heats a cell that ought to be cooling. The run that
# produced 2YS6_surface_flux did exactly this: a summer SST 7-9 C too warm over a mixed layer
# 3 m deep against BSOSE's 22 m, with the heat missing from 25-120 m underneath.
#
# `true` splits TFLUX into its solar and non-solar parts: oceQsw goes to a TwoColorRadiation
# forcing and TFLUX - oceQsw stays on the top boundary. Needs oceQsw in data/.
# ON, because the run now reaches December-January and the point is to see whether the warm
# bloom is gone. The |w| tripwire at day 1 keeps this from being a confounded test: with the
# split on and everything else at baseline, |w| grew 6x over day 1 (0.012 -> 0.073) where the
# baseline decayed it to 0.0076. That number arrives on day 1, long before the bloom does.
const PENETRATING_SHORTWAVE = true

const CHECKPOINTS = false # save state and restart if the model crashes. If false, the model will start from scratch.
const BOUNDARY_DIAGNOSTICS = true # write free-surface height, boundary-face slices, and a per-face volume transport log

# Barotropic (depth-integrated) open boundary conditions: Flather on U and V, fed the BSOSE
# depth-integrated transport, with the companion Chapman condition added to η automatically.
#
# Without this the split-explicit corrector overwrites the depth mean of the prescribed u/v at
# every face with the default impenetrable barotropic solution, so no net transport can cross
# the boundary no matter which scheme OBC_SCHEME selects — which is why every face transport
# reads 0.00 Sv and a jet sits on the western edge.
#
# ηᵉˣᵗ is still 0, so the Flather condition carries transport only. That is a placeholder, not
# a neutral choice: the real sea surface drops of order a metre between 45S and the Antarctic
# margin, and a zero tells the boundary it is flat. SSH_bsoseI156_2013to2024_monthly.nc is the
# fix and is not yet wired in. Read the face transports the barotropic setup now logs before
# reading anything into the run: they should be west +180, east +205, north -25 Sv.
const BAROTROPIC_OBC = true

# ηᵉˣᵗ for the Flather condition: BSOSE's sea surface, or zero.
#
# `false` here is deliberate. The idealised sandbox (obc_tests/flather_sandbox.jl) measures
# the boundary's response as
#
#     transport = 0.815 x Uᵉˣᵗ  +  60.8 Sv per metre of ηᵉˣᵗ tilt
#
# linear to three digits. BSOSE's ETAN drops 2.05 m across this domain, which on that face is
# ~125 Sv of tilt-driven transport on top of the ~197 Sv being prescribed. That only cancels
# if the model's own η sustains the same tilt, and it does not: supplying the real SSH made
# the transports worse, not better. A uniform ηᵉˣᵗ, by contrast, only moves the datum --
# 0 to 0.04 m changed the transport by 0.3%.
#
# So ηᵉˣᵗ = 0 is the conservative setting until the interior can be shown to hold the tilt.
# Set `true` to use the SSH field, which also initialises η to match (the two must agree, or
# the boundary sees a 2 m error on the first step).
# ηᵉˣᵗ, used only by the Flather barotropic scheme. With BAROTROPIC_SCHEME = :prescribed
# nothing reads it, so `false` also skips loading ETAN and skips initialising the free
# surface from it.
#
# Leaving the η initialisation off is deliberate. It existed to stop Flather seeing a 0.7 m
# error on day one. Without Flather there is nothing to protect, and initialising from BSOSE
# only buys a day-one shock: max|η| fell 3.40 -> 1.32 m in 24 h as the field radiated into
# balance with the model's own dynamics. Starting flat lets the surface build its own tilt
# from the prescribed transport and the wind, which is the tilt the interior can actually
# hold. Set this back to `true` only alongside BAROTROPIC_SCHEME = :flather.
const EXTERIOR_SSH = false

# How the barotropic mode is constrained: `:pinned` or `:flather`. See the long note in
# `bsose_open_boundary_conditions` for the measurements behind this.
#
# Flather alone prescribes Uᵇ - √(gH)ηᵇ, not transport, at 5.1 Sv/cm on west/east and 9.5 Sv/cm
# on north. Two runs bracket zero -- ηᵉˣᵗ = 0 gave north +213 to +219 Sv, ηᵉˣᵗ = BSOSE gave -393
# to -437, against a prescribed -31. Same magnitude, opposite sign: too stiff, not mis-signed.
#
# `:pinned` keeps Flather for the shape and hands it Oceananigans' `target_transport`, which
# rescales the face uniformly after every barotropic substep so the integral matches. The free
# surface can then redistribute transport along a face but cannot change the net. The target is
# materialised once, so it is the time mean: the seasonal swing of the net is given up, the
# shape and all baroclinic structure are not.
const BAROTROPIC_SCHEME = :pinned

# Where the open boundaries get their data, and how they are built.
#
#   :glorys  NumericalEarth's 02_mab_glorys_obc.jl, in src/glorys_obc.jl: GLORYS read through
#            `Interpolated` on a source grid that brackets the model. Normal u/v get
#            PerturbationAdvection; tangential u/v AND tracers get NormalRadiation, all fed
#            real exterior values. Barotropic: Flather on the true depth mean over GLORYS's
#            own 50 levels times the model's wet depth. η: Chapman, explicit on all four faces.
#   :bsose   everything this file built before -- cached boundary slices, the hand-rolled
#            barotropic transport, and BAROTROPIC_SCHEME / BAROTROPIC_FACES / EXTERIOR_SSH.
#
# `:glorys` ignores OBC_SCHEME, TANGENTIAL_BC, BAROTROPIC_SCHEME, BAROTROPIC_FACES and
# EXTERIOR_SSH -- it makes every one of those choices itself, as that script does.
const OBC_SOURCE = :glorys

# Which GLORYS product drives the boundaries. 02 uses `GLORYSDaily()` over a 12x8 degree box
# for ten days. This domain is 60x25 degrees for eight months: daily would be ~250 records and
# tens of GB, monthly is 11 records. Swap to `GLORYSDaily()` for eddy-resolving boundary
# forcing when the storage is there.
const GLORYS_DATASET = GLORYSMonthly()

# 02's MAB_UEXT. `:native` is its default and the mode it validated; `:masked` and `:legacy`
# exist for the A/B that `report_uext` prints at startup.
const GLORYS_UEXT_MODE = :native

# 02's MAB_TANGENTIAL: `:radiated` (its default) or `:prescribed`.
const GLORYS_TANGENTIAL = :radiated

# 02's MAB_TAU_IN: the inflow nudging timescale shared by the normal, tangential and tracer
# schemes. Outflow is Inf throughout, so outgoing features leave freely.
const GLORYS_TAU_IN = 1days

# 02's MAB_MATCH_BATHY / MAB_N_MATCH. The only thing here that changes the bathymetry, and it
# touches a 4-cell band at the open edges only. U = ∫u dz, so a depth mismatch at the boundary
# is a transport error no matter how carefully Uᵉˣᵗ is built.
const GLORYS_MATCH_BATHY = true
const GLORYS_N_MATCH = 4

# Initialise the interior from GLORYS as well, as 02 does. Driving the boundaries from GLORYS
# while starting the interior from BSOSE puts a discontinuity at the boundary on day one.
const GLORYS_INITIAL_CONDITIONS = true

# Which faces get a Flather condition: `:all` (west, east, north, south) or `:zonal`
# (west and east only, north/south depth-integrated flux held at zero).
#
# `:zonal` was added on the theory that the north face was a pure error amplifier: large
# area, almost no prescribed signal, so any (ηᵇ - ηᵉˣᵗ) error there becomes huge transport.
# tools/inspect_bsose_boundaries.jl measured that theory against BSOSE directly and it does
# not hold. On iteration 105, 2008-06 to 2009-01:
#
#     west +173.0   east +204.0   south +0.0   north -31.1 Sv     (monthly means)
#     four-face net -0.007 Sv    west+east-only net +31.1 Sv
#
# The four faces balance to 0.007 Sv out of ~200 Sv gross, so BSOSE does conserve volume
# here and the integral is trustworthy. But the north face carries -31 Sv, 18% of the zonal
# transport, not the 1.3% previously assumed -- +31 Sv genuinely enters from the north and
# leaves through the east, which is why east exceeds west by that much. Closing it forces
# that 31 Sv to zero, a 12% distortion of the face-mean velocity.
#
# The area ratio was also wrong: north/(west+east) is 1.10x on BSOSE's mask, not 1.9x. The
# 1.9x compared the north face to a single zonal face rather than to the pair.
#
# So `:all` stays the default. The sandbox result that motivated `:zonal` (a third of the
# throughflow leaking away when north/south are opened) was measured with Vᵉˣᵗ = 0, where no
# meridional flux should exist at all -- it is not the configuration the model runs, and it
# does not justify closing a face that carries 31 Sv. The real failure to explain is north
# reporting +213 to +219 Sv against a prescribed -31: an error of ~250 Sv, which points at
# ηᵉˣᵗ on that face rather than at the face being open.
#
# Re-run the tool on iteration 156 over the actual window before trusting these numbers.
const BAROTROPIC_FACES = :all

# Open boundary matching scheme for the NORMAL flow: "ObliqueRadiation",
# "PerturbationAdvection", "NormalRadiation", or "clamped".
const OBC_SCHEME = "PerturbationAdvection"

# Condition for the TANGENTIAL velocity (u on south/north, v on west/east). One of :oblique,
# :oblique_boundary_velocity, :gradient, :clamped, :radiation, :radiation_boundary_velocity,
# :perturbation_advection or :perturbation_advection_gravity; see
# `bsose_tangential_boundary_condition`.
#
# :oblique with OBC_SCHEME = "ObliqueRadiation" is the configuration the MAB sandbox runs —
# the same two-dimensional radiation on the normal flow and the tangential component alike.
const TANGENTIAL_BC = :gradient

# Nudging timescales for ObliqueRadiation, on inflow and on outflow (MAB's values). The
# other schemes below use 1 day and Inf; the difference is deliberate.
const OBLIQUE_TAU_IN = 3days
const OBLIQUE_TAU_OUT = 360days

# Turbulence closure set: "ito", "bsose", "henyey_gm", "biharmonic", or "none". Each is
# described where the closures are built below.
const CLOSURE_CONFIG = "ito"

# Stop the run early after this many time steps, for a smoke test. Inf runs the full period.
const STOP_ITERATION = Inf

# domain related parameters
const SCALING = 6 # horizontal resolution = 1 / SCALING degrees. 1/6 = ~15km, 1/2 = 50km
const DZ_SURFACE = 5 # m (gives ~135 vertical levels total, with 100 levels in upper 500m)
const DZ_BOTTOM = 200 # m
const CIRCUMPOLAR = false # if true, we will use a circumpolar domain

const DATASET = "BSOSE" # by default, our forcings and BCs will be BSOSE, but if we want to test GLORYS or ECCO, we should be able to do that. 

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

# 02's MAB_MATCH_BATHY. Must happen before the immersed grid is built, since that is what
# freezes the wet column the barotropic transport is computed over.
if OBC_SOURCE === :glorys && GLORYS_MATCH_BATHY
    match_boundary_bathymetry!(bottom_height, grid;
        region = BoundingBox(longitude = (λ₁ - 1.0, λ₂ + 1.0), latitude = (φ₁ - 1.0, φ₂ + 1.0)),
        n_match = GLORYS_N_MATCH, minimum_depth = 10)
end

grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom_height))

@info grid

# dataset conditonal

if DATASET == "BSOSE"
    dataset = BSOSEMonthly()
elseif DATASET == "GLORYS"
    dataset = GLORYSMonthly()
elseif DATASET == "ECCO"
    dataset = ECCO4Monthly()
else
    error("Unknown dataset: $DATASET. Choose 'BSOSE', 'GLORYS', or 'ECCO'.")
end

@info "Selected dataset: $(summary(dataset))"

# ==============================================================================
# Simulation window
#
# The period the model simulates: it starts at start_date and stops at end_date. These
# two lines are the only place the period is set — the stop time, the monthly records
# loaded for the boundary conditions, winds and sponges, and the date the initial
# condition is read from all follow from them.
#
# BSOSE iteration 156 (HPC) covers 2013-2024; iteration 105 (local CPU) covers
# 2008-2012, so a local test run needs dates inside that earlier range.
# ==============================================================================
# Eight months, to December-January, so the run carries a full seasonal cycle into the warm
# bloom. PENETRATING_SHORTWAVE stays OFF: this run is the barotropic/balance test, and leaving
# the shortwave out makes it something we have never had -- the bloom reproduced on top of a
# boundary that actually passes the ACC. That is the clean control the shortwave fix has been
# missing, and it costs nothing extra to collect it here.
#
# (Previous header, from the ninety-day version:)
# Ninety days. The fourteen-day A/B answered its question -- PerturbationAdvection on the
# normal flow with GradientBoundaryCondition(0) on the tangential reproduces 54021db's log
# almost exactly on current packages (day 13: |u| 0.83, |v| 1.17, |w| 0.0175, Δt 17.2 min
# against 0.83 / 1.17 / 0.018 / 17.16), so the package bump was never the problem and the
# oblique schemes were.
#
# This run adds the one thing that configuration lacks: a barotropic boundary, so the ACC's
# ~180 Sv can actually cross instead of turning and running along the edges. Every previous
# barotropic attempt paired it with the oblique schemes now known to be unstable, so Flather
# with PA + Gradient(0) has never been tried.
#
# Ninety days because the western boundary jet builds over a ~20-day cycle; at 14 days every
# configuration still looks acceptable. About 2.4 hours at the rate the A/B ran.
#
# That commit ran PerturbationAdvection on the normal flow, GradientBoundaryCondition(0) on the
# tangential, no barotropic condition, on Oceananigans 0.111.0 / NumericalEarth 0.7.0. Current
# main had changed all three settings AND both packages, and every comparison since has been
# treating the 54021db run as "the same thing minus the shortwave", which it never was.
#
# The flags below reproduce 54021db's boundary configuration on today's packages. Compare
# |u|, |v|, |w| and Δt day-for-day against its log:
#
#     day  1:  |u| 0.91  |v| 0.91  |w| 0.0076   Δt  7.56 min
#     day  6:  |u| 0.77  |v| 1.01  |w| 0.013    Δt  6.24 min
#     day 13:  |u| 0.83  |v| 1.17  |w| 0.018    Δt 17.16 min
#
# Matching those means the schemes are what broke it and the package bump is innocent.
# Still slow means the bump is implicated and the schemes are not the story.
#
# (Original eight-month header, for when this goes back to being a science run:)
# Eight months, June 2014 to February 2015. The start matches 2YS6_surface_flux so that |u|,
# |v|, |w| and Δt compare day-for-day against the log of a run that stayed stable, and the end
# reaches past the warm bloom: January 2015 is where the baseline ran SST +7.02 C too warm
# over a 3.1 m mixed layer against BSOSE's 21.8 m. Three questions, one run -- face transports
# at startup, the |w| tripwire on day 1, the bloom around day 200.
start_date = DateTime("2014-06-01")
end_date = DateTime("2015-02-01")

# option if need to do tests on CPU
# start_date = DateTime("2009-06-01")
# end_date = DateTime("2011-06-01")

end_date > start_date || error("end_date ($end_date) must be later than start_date ($start_date).")

dates = (start_date, end_date)
simulation_seconds = Dates.value(Second(end_date - start_date))
simulation_days = simulation_seconds / 86400

# Monthly records for the boundary conditions, winds and sponge nudging.
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
# selecting on stamps pads the head by two records and leaves the tail half a month short —
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
@info "Boundary condition dates: $(length(bc_dates)) monthly records from $(bc_dates[1]) to $(bc_dates[end])"
@info "Model time zero corresponds to $start_date; all forcing is stepped on that clock."

# ==============================================================================
# Boundary Conditions & Forcing
# ==============================================================================

if OBCS && DATASET == "BSOSE"
    @info "Configuring BSOSE Open Boundary Conditions with radiation scheme..."
    # In an open-ocean regional domain with strong outflow (ACC carrying ~130 Sv eastward through 150°E),
    # rigid Dirichlet boundaries reflect outgoing eddies and waves, leading to tracer blowup at the boundary.
    # PerturbationAdvection / NormalRadiation with outflow_timescale=Inf allows internal waves/eddies to freely
    # radiate out of the domain, while inflow_timescale=1days nudges incoming boundary flow (West) smoothly to BSOSE.
    if OBC_SOURCE === :glorys
        # 02_mab_glorys_obc.jl's construction. It makes its own choices for the normal,
        # tangential, tracer, barotropic and η conditions, so OBC_SCHEME, TANGENTIAL_BC,
        # BAROTROPIC_SCHEME, BAROTROPIC_FACES and EXTERIOR_SSH are unused on this path.
        # Our surface forcing is not part of that construction, so it is passed in.
        wind_bcs = WINDS ? bsose_surface_wind_stress(grid; dataset=dataset, dates=bc_dates,
                                                     reference_date=start_date) : nothing
        flux_bcs = SURFACE_FLUXES ?
            bsose_surface_tracer_fluxes(grid; dataset=dataset, dates=bc_dates,
                reference_date=start_date,
                subtract_shortwave=PENETRATING_SHORTWAVE) : nothing

        # 02 writes `start_date : Day(1) : stop_date` because GLORYSDaily has a record on
        # every one of those days. A monthly product does not, so take its own stamps.
        glorys_dates = glorys_boundary_dates(start_date, end_date; dataset = GLORYS_DATASET)
        @info "GLORYS boundary dates: $(length(glorys_dates)) records from $(first(glorys_dates)) to $(last(glorys_dates))"

        glorys_obc = glorys_open_boundary_conditions(grid;
            dates = glorys_dates,
            dataset = GLORYS_DATASET,
            tau_in = GLORYS_TAU_IN,
            uext_mode = GLORYS_UEXT_MODE,
            tangential = GLORYS_TANGENTIAL,
            top_u = wind_bcs === nothing ? nothing : wind_bcs.u,
            top_v = wind_bcs === nothing ? nothing : wind_bcs.v,
            top_T = flux_bcs === nothing ? nothing : flux_bcs.T,
            top_S = flux_bcs === nothing ? nothing : flux_bcs.S)

        boundary_conditions = glorys_obc.boundary_conditions
        exterior_η = 0            # the GLORYS path initialises η in its own block below
    else
        obc_scheme = if OBC_SCHEME == "ObliqueRadiation"
            ObliqueRadiation(inflow_timescale=OBLIQUE_TAU_IN, outflow_timescale=OBLIQUE_TAU_OUT)
        elseif OBC_SCHEME == "PerturbationAdvection"
            PerturbationAdvection(inflow_timescale=1days, outflow_timescale=Inf)
        elseif OBC_SCHEME == "NormalRadiation"
            NormalRadiation(inflow_timescale=1days, outflow_timescale=Inf)
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
        exterior_η = BAROTROPIC_OBC && EXTERIOR_SSH && DATASET == "BSOSE" ?
                     bsose_exterior_free_surface(grid; dataset=dataset, dates=bc_dates,
                         reference_date=start_date) : 0

        boundary_conditions = bsose_open_boundary_conditions(grid; dataset=dataset, dates=bc_dates,
            exterior_free_surface=exterior_η,
            winds=WINDS, surface_fluxes=SURFACE_FLUXES,
            penetrating_shortwave=SURFACE_FLUXES && PENETRATING_SHORTWAVE, scheme=obc_scheme,
            tangential_bc_kind=TANGENTIAL_BC, oblique_inflow_timescale=OBLIQUE_TAU_IN,
            oblique_outflow_timescale=OBLIQUE_TAU_OUT, reference_date=start_date,
            barotropic=BAROTROPIC_OBC, barotropic_faces=BAROTROPIC_FACES,
            barotropic_scheme=BAROTROPIC_SCHEME)
    end

    if SPONGE_LAYERS
        sponge_timescale = 30days
        @info "Configuring boundary edge sponge layers (3.0° width, $(sponge_timescale) restoring timescale)..."
        forcings = bsose_sponge_layer_forcing(grid; dataset=dataset, dates=bc_dates, sponge_width=3.0, timescale=sponge_timescale, restore_velocities=true, reference_date=start_date)
    else
        forcings = NamedTuple()
    end
elseif !OBCS
    @info "Configuring whole-region DatasetRestoring (30-day restoring for T and S from $DATASET)..."
    wind_bcs = (WINDS && DATASET == "BSOSE") ? bsose_surface_wind_stress(grid; dataset=dataset, dates=bc_dates, reference_date=start_date) : nothing
    boundary_conditions = wind_bcs !== nothing ? (u=FieldBoundaryConditions(top=wind_bcs.u), v=FieldBoundaryConditions(top=wind_bcs.v)) : NamedTuple()

    restoring_region = DATASET == "BSOSE" ? bsose_region(grid) : nothing
    temperature_metadata = Metadata(:temperature; dataset, start_date, end_date, region=restoring_region)
    salinity_metadata = Metadata(:salinity; dataset, start_date, end_date, region=restoring_region)

    FT = DatasetRestoring(temperature_metadata, grid; rate=1 / 30days)
    FS = DatasetRestoring(salinity_metadata, grid; rate=1 / 30days)

    if DATASET == "BSOSE"
        restore_on_model_clock!(FT, start_date)
        restore_on_model_clock!(FS, start_date)
    end

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

if PENETRATING_SHORTWAVE
    if DATASET != "BSOSE"
        @warn "PENETRATING_SHORTWAVE is only wired up for BSOSE; ignoring it for $DATASET."
    elseif !SURFACE_FLUXES
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

if OBC_SOURCE === :glorys && GLORYS_INITIAL_CONDITIONS
    # 02 initialises the interior from the same dataset that drives the boundaries. Mixing
    # them -- GLORYS at the edges, BSOSE inside -- puts a discontinuity on the boundary at
    # t = 0 that the open conditions then have to argue with.
    @info "Initializing model state from GLORYS at $start_date (02's initial conditions)..."
    glorys_initial_conditions!(ocean.model; date=start_date, dataset=GLORYS_DATASET,
                               region=glorys_obc.region)
    fill_bathymetry_gaps!(parent(ocean.model.tracers.S), S_MIN_PHYSICAL, S_MAX_PHYSICAL)
    fill_bathymetry_gaps!(parent(ocean.model.tracers.T), T_MIN_PHYSICAL, T_MAX_PHYSICAL)
    @info "Model successfully initialized from GLORYS and filled bathymetric gaps."
elseif DATASET == "BSOSE"
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
else
    set!(ocean.model,
        MetadataSet(:temperature; dataset=dataset, date=start_date),
        MetadataSet(:salinity; dataset=dataset, date=start_date)
    )
    fill_bathymetry_gaps!(parent(ocean.model.tracers.S), S_MIN_PHYSICAL, S_MAX_PHYSICAL)
    fill_bathymetry_gaps!(parent(ocean.model.tracers.T), T_MIN_PHYSICAL, T_MAX_PHYSICAL)
    fill_halo_regions!(ocean.model.tracers.T, ocean.model.clock, fields(ocean.model))
    fill_halo_regions!(ocean.model.tracers.S, ocean.model.clock, fields(ocean.model))
    fill_bathymetry_gaps!(parent(ocean.model.tracers.S), S_MIN_PHYSICAL, S_MAX_PHYSICAL)
    fill_bathymetry_gaps!(parent(ocean.model.tracers.T), T_MIN_PHYSICAL, T_MAX_PHYSICAL)
end

# ── TEOS10 sqrt guard ─────────────────────────────────────────────────────────
# The TEOS10 coordinate s(Sᴬ) = √((Sᴬ + ΔS) / Sₐᵤ) crashes for Sᴬ < -32.
# Guard with max(..., 0) so that any residual bad values from immersed/halo cells
# don't crash the run — they will return zero (freshwater) density rather than NaN.
import SeawaterPolynomials.TEOS10 as TEOS10_mod
@eval SeawaterPolynomials.TEOS10 begin
    @inline s(Sᴬ::FT) where FT = √(max((Sᴬ + FT(ΔS)) / FT(Sₐᵤ), zero(FT)))
end

# some housekeeping to set up the simulation (progress checks and adaptive timestep)

# The run length comes from start_date and end_date; STOP_ITERATION can cut it short.
# Start with a very small Δt so the first CATKE evaluation is numerically gentle;
# the wizard will ramp this up within a few iterations.
simulation = Simulation(ocean.model, Δt=1seconds,
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

# progress logger: Oceanostics TimedMessenger reports wall-clock timing,
# max velocities and CFL/diffusive stability numbers each interval.
progress = TimedMessenger()
simulation.callbacks[:progress] = Callback(progress, TimeInterval(output_interval))

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
    "dataset" => DATASET)

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
#   3. Full-depth slices at each face, to see whether flow and tracers cross the
#      boundary at every depth or only near the surface, and to compare against the
#      BSOSE values the boundary conditions prescribe.
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

    # Outermost cell of each face, full depth. The writer takes a single `indices` tuple,
    # so on the east and north faces the normal velocity written is the inner face of that
    # cell; the exact boundary-face transport is what the log below integrates.
    boundary_interval = 5days
    face_indices = (west=(1, :, :),
        east=(grid.Nx, :, :),
        south=(:, 1, :),
        north=(:, grid.Ny, :))

    for (face, indices) in pairs(face_indices)
        simulation.output_writers[Symbol(:face_, face)] = NetCDFWriter(
            ocean.model,
            (; u, v, T, S),
            filename="$(RUN_NAME)_face_$(face).nc",
            schedule=TimeInterval(boundary_interval),
            indices=indices,
            global_attributes=output_metadata,
            overwrite_files=true
        )
    end

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

        η = sim.model.free_surface.displacement
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
@info "OBCS: $OBCS | WINDS: $WINDS | SURFACE_FLUXES: $SURFACE_FLUXES | DATASET: $DATASET"
@info "PENETRATING_SHORTWAVE: $PENETRATING_SHORTWAVE (active: $(update_shortwave! !== nothing))"

# red button 

@info "Starting simulation..."
run!(simulation)
@info "Simulation completed successfully!"
