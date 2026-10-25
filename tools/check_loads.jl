# ==============================================================================
# check_loads.jl -- will the model files actually LOAD?
#
#     julia --project=. tools/check_loads.jl
#
# Run this before every submission. It takes a couple of minutes and has already caught two
# failures that reached the cluster:
#
#   1. A stray `end` left by deleting an `if`/`elseif` head.          (parse error)
#   2. A docstring inserted directly above another docstring, so the  (LOWERING error)
#      first one documented a string literal instead of a function:
#          ERROR: cannot document the following expression
#
# `tools/check_syntax.jl` catches (1) and is structurally blind to (2) -- a misplaced docstring
# is valid syntax. Only evaluating the file catches it.
#
# `include` evaluates top-level expressions one at a time, so a failure anywhere means
# everything above it already ran. On the cluster that is forty minutes of grid construction,
# boundary conditions and initial conditions before the error appears. Two minutes here.
#
# This does NOT run the model. It loads `setup_bsose.jl` (definitions only, no data access) and
# then checks that every name `model.jl` references at top level resolves. It cannot catch
# errors in `model.jl`'s own executable body, which needs data.
# ==============================================================================

using Oceananigans, Oceananigans.Units, Oceananigans.Grids
using NumericalEarth, NumericalEarth.ECCO
using NumericalEarth.DataWrangling: all_dates
using NCDatasets, Dates, Printf, SeawaterPolynomials
using Oceananigans.TurbulenceClosures
using Oceananigans.Grids: φnode
using Oceananigans.Operators: Azᶜᶜᶜ, Ax_qᶠᶜᶜ, Ay_qᶜᶠᶜ
using Oceananigans.BoundaryConditions: PerturbationAdvection, NormalRadiation,
                                       ObliqueRadiation, TracerReservoir
using Oceananigans.Diagnostics: NaNChecker
using Oceanostics.ProgressMessengers: TimedMessenger

const SRC = joinpath(@__DIR__, "..", "src")
failures = 0

# ---- 1. does setup_bsose.jl load? ----
print("loading setup_bsose.jl ... "); flush(stdout)
try
    include(joinpath(SRC, "setup_bsose.jl"))
    println("OK")
catch e
    global failures += 1
    println("FAILED")
    showerror(stdout, e); println()
end

# ---- 2. does model.jl parse, and do the names it needs exist? ----
print("parsing model.jl ... "); flush(stdout)
let src = read(joinpath(SRC, "model.jl"), String)
    errs = String[]
    ex = try
        Meta.parseall(src; filename = "model.jl")
    catch e
        push!(errs, sprint(showerror, e)); nothing
    end
    walk(e) = if e isa Expr
        (e.head === :error || e.head === :incomplete) && push!(errs, string(e.args))
        foreach(walk, e.args)
    end
    ex === nothing || walk(ex)
    if isempty(errs)
        println("OK")
    else
        global failures += 1
        println("FAILED"); foreach(e -> println("   ", first(e, 300)), errs)
    end
end

print("checking the names model.jl needs ... "); flush(stdout)
needed = [:bsose_open_boundary_conditions, :bsose_surface_wind_stress, :bsose_surface_tracer_fluxes,
          :bsose_penetrating_shortwave, :bsose_sponge_layer_forcing, :bsose_initial_conditions!,
          :bsose_exterior_free_surface, :bsose_region, :bsose_window_center, :face_geometry,
          :fill_bathymetry_gaps!, :prefetch_era5!, :restore_on_model_clock!,
          :S_MIN_PHYSICAL, :S_MAX_PHYSICAL, :T_MIN_PHYSICAL, :T_MAX_PHYSICAL,
          :default_bsose_directory, :BoundingBox, :Metadata, :MetadataSet, :BSOSEMonthly,
          :DatasetRestoring, :regrid_bathymetry, :ocean_simulation, :TwoColorRadiation,
          :OceanOnlyModel, :ERA5PrescribedAtmosphere, :ERA5PrescribedRadiation,
          :ReferenceToStretchedDiscretization, :PowerLawStretching, :GridFittedBottom]
missing_names = filter(n -> !isdefined(Main, n), needed)
if isempty(missing_names)
    println("OK  ($(length(needed)) names)")
else
    global failures += 1
    println("FAILED"); println("   missing: ", join(missing_names, ", "))
end

println()
println(failures == 0 ? "All checks passed — safe to submit." : "$failures check(s) FAILED — do not submit.")
exit(failures == 0 ? 0 : 1)
