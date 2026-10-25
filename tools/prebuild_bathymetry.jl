# ==============================================================================
# prebuild_bathymetry.jl -- build (or confirm) the bathymetry cache for one vertical grid.
#
#     Z_EXTENT=1200 DZ_SURFACE=10 DZ_BOTTOM=50 julia --project=. tools/prebuild_bathymetry.jl
#
# WHY THIS EXISTS. `regrid_bathymetry` caches its result under a key that includes Nz:
#
#     field_bottom_height_360x150x135_8ffd3b13.jld2
#
# So two `run_obc_tests` arms that differ only in a boundary scheme share a cache and are
# harmless, but two arms with the same NEW vertical grid -- `flat_upper` and
# `flat_upper_radiation`, say -- would both find the key missing and both start writing it.
# Concurrent writers to one JLD2 file can corrupt it and lose both jobs.
#
# `run_obc_tests` therefore submits one of these per distinct vertical grid and makes the real
# jobs depend on it. If the cache already exists this loads it and exits in seconds, so it is
# cheap enough to run unconditionally rather than trying to guess the key.
#
# CPU ONLY, and deliberately so: the cache key does not include the architecture, so a CPU
# build populates the cache a GPU job will read, without spending a GPU allocation. Same
# reasoning as the header of `inspect_bathymetry.jl`.
#
# IMPORTANT: this reproduces src/model.jl's GPU branch, not its CPU branch -- the CPU branch
# drops to a 1 degree grid and different z parameters, which would populate a different and
# useless key. The parameters below must stay in step with `src/model.jl`; the check at the end
# of this file is what catches it if they drift.
# ==============================================================================

using Oceananigans, Oceananigans.Units, Oceananigans.Grids
using NumericalEarth
using Printf

# ---- the production grid, matching src/model.jl's GPU branch ----
const λ₁, λ₂ = 90.0, 150.0
const φ₁, φ₂ = -70.0, -45.0
const SCALING = 6

const Z_EXTENT   = parse(Float64, get(ENV, "Z_EXTENT",   "5800"))
const DZ_SURFACE = parse(Float64, get(ENV, "DZ_SURFACE", "5"))
const DZ_BOTTOM  = parse(Float64, get(ENV, "DZ_BOTTOM",  "200"))

z = ReferenceToStretchedDiscretization(; extent = Z_EXTENT,
                                       constant_spacing = DZ_SURFACE,
                                       maximum_spacing = DZ_BOTTOM,
                                       constant_spacing_extent = 500,
                                       stretching = PowerLawStretching(1.12))

Nx, Ny, Nz = Int(SCALING * (λ₂ - λ₁)), Int(SCALING * (φ₂ - φ₁)), length(z)

@info @sprintf("prebuilding bathymetry for Z_EXTENT=%.0f DZ_SURFACE=%.0f DZ_BOTTOM=%.0f -> %dx%dx%d",
               Z_EXTENT, DZ_SURFACE, DZ_BOTTOM, Nx, Ny, Nz)

grid = LatitudeLongitudeGrid(CPU(); size = (Nx, Ny, Nz),
                             latitude = (φ₁, φ₂), longitude = (λ₁, λ₂), z,
                             halo = (7, 7, 7))

bottom_height = regrid_bathymetry(grid,
                                  height_above_water = 1,
                                  minimum_depth = 10,
                                  interpolation_passes = 5)

# Confirm it is a usable field rather than trusting that the call returned. A wet-column count
# of zero would mean the regrid produced land everywhere, which has happened before when a
# truncated ETOPO download was left in the cache.
B = Array(interior(bottom_height))[:, :, 1]
nwet = count(<(0), B)

if nwet == 0
    @error "regrid_bathymetry produced NO wet columns -- the ETOPO cache is probably truncated. \
            Delete it and rerun rather than letting the dependent jobs start."
    exit(1)
end

@info @sprintf(" -> OK: %d of %d columns wet (%.0f%%), deepest %.0f m. Cache is ready for Nz=%d.",
               nwet, Nx * Ny, 100nwet / (Nx * Ny), -minimum(B), Nz)
