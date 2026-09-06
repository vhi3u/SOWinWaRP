using Pkg
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

if has_cuda_gpu()
    arch = GPU()
else
    arch = CPU()
end

# flags

const OBCS = true # open boundary conditions; if false, we will use sponge layers instead? 
const WINDS = false # time-varying surface wind forcing from BSOSE data (oceTAUX and oceTAUY)
const TEOS = true # use TEOS because the nonlinearity from the full equation of state allows for AAIW formation ; if false uses linear equation of state
const CHECKPOINTS = false # save state and restart if the model crashes. If false, the model will start from scratch. 

# domain related parameters
const SCALING = 6 # horizontal resolution = 1 / SCALING degrees. 1/6 = ~15km, 1/2 = 50km
const DZ_SURFACE = 2 # m 
const DZ_BOTTOM = 200 # m
const CIRCUMPOLAR = false # if true, we will use a circumpolar domain

const DATASET = "BSOSE" # by default, our forcings and BCs will be BSOSE, but if we want to test GLORYS or ECCO, we should be able to do that. 

# Domain setup

# horizontal domain extent
if CIRCUMPOLAR
    λ₁, λ₂ = (0, 360)
    ϕ₁, ϕ₂ = (-78, 30)
else
    λ₁, λ₂ = (90, 150)
    ϕ₁, ϕ₂ = (-70, -40)
end

# z stretching so that the upper 500 meters of the ocean has dz = 2 meters, and the deeper ocean has dz = 200 meters. 
z = ReferenceToStretchedDiscretization(; extent=5000,
    constant_spacing=DZ_SURFACE,
    constant_spacing_extent=DZ_BOTTOM,
    stretching=PowerLawStretching(1.15))

# horizontal grid
Nx = Int(SCALING * (λ₂ - λ₁))
Ny = Int(SCALING * (φ₂ - φ₁))

# horizontal grid
Nx = 1 * Int(λ₂ - λ₁) # 1/2 th of a degree resolution
Ny = 1 * Int(φ₂ - φ₁) # 1/2 th of a degree resolution