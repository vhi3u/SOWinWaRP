# ==============================================================================
# mld.jl -- mixed layer depth by the de Boyer Montegut density threshold, in one place.
#
#     include(joinpath(@__DIR__, "mld.jl"))
#     h = mld(Tcol, Scol, zc, kbot)          # metres, positive down; NaN if no crossing
#
# Shared by tools/compare_bsose.jl and tools/compare_mld.jl so the two cannot drift apart.
# `MLD_DT` is read from the environment (default 0.03 kg/m3) by whichever script includes this.
#
# THE TRAP, kept from compare_bsose.jl where it was found and fixed:
#
# > `kbot` is not optional bookkeeping, it is the whole correctness of this function.
# >
# > It used to search to `k = 1` and, finding nothing, return `-zs[1]` -- the deepest cell
# > centre of the GRID, about 5.6 km, with no relation to the water depth. It relied on hitting
# > a NaN below the sea floor to stop, and there is none: `fill_bathymetry_gaps!` in
# > setup_bsose.jl deliberately propagates the deepest valid water mass DOWNWARD into every
# > immersed cell, so T and S below the floor are finite and constant.
#
# The model's own output has the same property for a different reason -- Oceananigans writes
# immersed cells as zeros, not NaN -- so `kbot` is required on both sides.
#
# TEOS10 vs MITgcm: the model carries conservative temperature and absolute salinity, B-SOSE
# carries potential temperature and practical salinity. A THRESHOLD MLD is robust to that
# (it is a difference of densities within one column, both evaluated the same way); an
# absolute density is not. Do not reuse `rho_theta` here for anything that compares absolute
# values across the two.
# ==============================================================================

using SeawaterPolynomials
using SeawaterPolynomials.TEOS10: TEOS10EquationOfState

const MLD_EOS = TEOS10EquationOfState()
const MLD_DT  = parse(Float64, get(ENV, "MLD_DT", "0.03"))

"""Potential density anomaly at 0 dbar. Same equation of state for model and parent."""
rho_theta(T, S) = (isfinite(T) && isfinite(S)) ?
                  SeawaterPolynomials.ρ′(T, S, 0.0, MLD_EOS) : NaN

"""
    mld(T, S, zs, kbot = 1; Δρ = MLD_DT)

Shallowest depth at which potential density exceeds its 10 m value by `Δρ` kg/m3.

`T`, `S` are columns ordered BOTTOM TO TOP, as Oceananigans stores them, so a smaller index is
deeper; `zs` is the matching cell-centre depth (negative). `kbot` is the deepest wet index of
the column -- the search stops there and must not be left at its default for a real grid.

Returns `NaN` when no crossing exists between 10 m and the sea floor. That is a real result
("this column is unstratified to the bottom"), not a failure, and callers should count it
rather than treat it as missing data.
"""
function mld(T, S, zs, kbot = 1; Δρ = MLD_DT)
    k10 = findlast(z -> z <= -10, zs)          # deepest cell at or above 10 m
    k10 === nothing && (k10 = length(zs))
    k10 <= kbot && return NaN                   # column too shallow for a 10 m reference
    ref = rho_theta(T[k10], S[k10])
    isfinite(ref) || return NaN
    for k in (k10 - 1):-1:kbot
        ρ = rho_theta(T[k], S[k])
        isfinite(ρ) || return NaN
        ρ - ref >= Δρ && return -zs[k]
    end
    return NaN                                  # unstratified through the whole water column
end
