# ==============================================================================
# winter_water.jl -- Antarctic Winter Water (WW) in one water column, after Spira et al. (2024),
# "The observed spatiotemporal variability of Antarctic Winter Water", JGR Oceans 129,
# e2024JC021017, doi:10.1029/2024JC021017 (their WW detection workflow figure).
#
#     include(joinpath(@__DIR__, "winter_water.jl"))
#     w = winter_water(T, S, z, dz, kb)      # SURFACE-FIRST columns
#     w.kind      :ML, :SS or :none
#     w.reason    why: :ML, :SS, :warm, :deep, :inverted, :dry
#     w.T, w.S, w.σ, w.core, w.top, w.bottom, w.thickness, w.mld   (depths in m, positive down)
#
# THE WORKFLOW, for each column:
#
#   1. Mixed layer depth by the de Boyer Montegut threshold, 0.03 kg/m3 from 10 m (tools/mld.jl,
#      MLD_DT). A column with no crossing down to its floor is mixed to the floor.
#   2. Is there a temperature inversion below the mixed layer of at least WW_INVERSION (20 m)?
#      Taken as: the coldest level below the mixed layer (down to WW_SEARCH) is colder than the
#      mixed layer, and temperature rises with depth beneath it for at least 20 m.
#        yes -> SUBSURFACE WW (WW_SS): core T, S, σ and depth at that temperature minimum.
#               upper boundary  the depth of max |dT/dz| between the mixed layer base and the
#                               core, or the 2 °C isotherm if the water there is warmer than 2 °C
#               lower boundary  the depth of max dT/dz below the core (within WW_SEARCH), or the
#                               2 °C isotherm if the water there is warmer than 2 °C
#        no  -> MIXED-LAYER WW (WW_ML): core T, S, σ the mixed-layer mean, core depth half the
#               mixed layer, boundaries the surface and the mixed layer base.
#   3. NO WW in the column when (Spira et al.):
#        the core is at or warmer than WW_TMAX (2 °C)                      reason :warm
#        the core is deeper than its own lower boundary                     reason :inverted
#        the core or the lower boundary is deeper than WW_ZMAX (300 m)      reason :deep
#      The last removes the deep convection of polynyas and dense water formation, which Spira
#      et al. leave out of scope ("deep mixing profiles removed").
#
# Why the core must be colder than the mixed layer: in winter the mixed layer IS the Winter Water,
# and the level just below it is the warmer top of the thermocline. Without that test that level
# would read as a "subsurface minimum" with warming beneath it, and every winter column would be
# misfiled as WW_SS.
#
# Depths are taken at cell centres, and the boundaries at the midpoint between the two centres the
# gradient spans; the 2 °C isotherm is linearly interpolated. On the 52 B-SOSE levels (10 m apart
# from 25 to 180 m) the 20 m inversion is two level spacings.
#
# Model vs B-SOSE: the model's T is conservative temperature, B-SOSE's potential temperature; near
# 2 °C they differ by a few hundredths, so the 2 °C test is safe. σ comes from TEOS10 for both, with
# B-SOSE's practical salinity fed in as if absolute, so compare σ within a dataset, not across.
#
# ENV
#     WW_INVERSION   minimum inversion thickness below the core, m   (default 20)
#     WW_TMAX        warmest core still counted as WW, °C            (default 2.0)
#     WW_ZMAX        deepest core or lower boundary, m               (default 300)
#     WW_SEARCH      deepest level searched for the core and the lower boundary, m (default 600)
#     MLD_DT         density threshold, kg/m3                        (default 0.03, via mld.jl)
# ==============================================================================

@isdefined(mld) || include(joinpath(@__DIR__, "mld.jl"))

const WW_INVERSION = parse(Float64, get(ENV, "WW_INVERSION", "20"))
const WW_TMAX      = parse(Float64, get(ENV, "WW_TMAX", "2.0"))
const WW_ZMAX      = parse(Float64, get(ENV, "WW_ZMAX", "300"))
const WW_SEARCH    = parse(Float64, get(ENV, "WW_SEARCH", "600"))

"""Potential density anomaly σ₀ (kg/m3), TEOS10 as everywhere else in tools/."""
sigma0(T, S) = (isfinite(T) && isfinite(S)) ?
               MLD_EOS.reference_density + rho_theta(T, S) - 1000 : NaN

# Depth (positive) where T crosses `Tx` between levels k and k+1, linearly
crossing(T, z, k, Tx) = -(z[k] + (Tx - T[k]) / (T[k+1] - T[k]) * (z[k+1] - z[k]))

function ww_result(kind, reason; mld = NaN, T = NaN, S = NaN, core = NaN, top = NaN, bottom = NaN)
    return (; kind, reason, mld, T, S, σ = sigma0(T, S), core, top, bottom, thickness = bottom - top)
end

"""
    winter_water(T, S, z, dz, kb)

Winter Water in one column after Spira et al. (2024); see the header. `T`, `S`, `z` (cell centres,
negative) and `dz` (cell thicknesses) are SURFACE-FIRST, the opposite of Oceananigans' order, and
`kb` is the deepest wet level (0 for land).
"""
function winter_water(T, S, z, dz, kb)
    kb >= 2 || return ww_result(:none, :dry)
    all(k -> isfinite(T[k]) && isfinite(S[k]), 1:kb) || return ww_result(:none, :dry)

    # 1. the mixed layer (mld.jl wants bottom-first columns)
    h = mld(reverse(T[1:kb]), reverse(S[1:kb]), reverse(z[1:kb]), 1)
    isfinite(h) || (h = -z[kb] + dz[kb] / 2)               # mixed to the floor
    kml = something(findlast(k -> -z[k] <= h, 1:kb), 1)    # deepest level inside the mixed layer
    w = dz[1:kml]
    Tml, Sml = sum(T[1:kml] .* w) / sum(w), sum(S[1:kml] .* w) / sum(w)

    # 2. a subsurface minimum colder than the mixed layer, with >= WW_INVERSION m of warming below it
    below = [k for k in (kml + 1):kb if -z[k] <= WW_SEARCH]
    kc = isempty(below) ? 0 : below[argmin(T[below])]
    kend = kc
    if kc > 0
        while kend < kb && T[kend+1] > T[kend]
            kend += 1
        end
    end
    subsurface = kc > 0 && T[kc] < Tml && (z[kc] - z[kend]) >= WW_INVERSION - 1e-6

    if subsurface
        # upper boundary: steepest |dT/dz| between the mixed layer base and the core
        ku = argmax(k -> abs((T[k] - T[k+1]) / (z[k] - z[k+1])), kml:(kc - 1))
        top = -(z[ku] + z[ku+1]) / 2
        if (T[ku] + T[ku+1]) / 2 > WW_TMAX
            kx = findlast(k -> T[k] > WW_TMAX >= T[k+1], kml:(kc - 1))
            kx === nothing || (top = crossing(T, z, (kml:(kc - 1))[kx], WW_TMAX))
        end
        # lower boundary: steepest warming with depth below the core, within the search depth
        span = [k for k in kc:(kb - 1) if -z[k+1] <= WW_SEARCH]
        if isempty(span)
            bottom = -z[kc]
        else
            kl = argmax(k -> (T[k+1] - T[k]) / (z[k] - z[k+1]), span)
            bottom = -(z[kl] + z[kl+1]) / 2
            if (T[kl] + T[kl+1]) / 2 > WW_TMAX
                kx = findfirst(k -> T[k] < WW_TMAX <= T[k+1], span)
                kx === nothing || (bottom = crossing(T, z, span[kx], WW_TMAX))
            end
        end
        kind, Tc, Sc, core = :SS, T[kc], S[kc], -z[kc]
    else
        kind, Tc, Sc, core, top, bottom = :ML, Tml, Sml, h / 2, 0.0, h
    end

    # 3. the removals
    reason = Tc >= WW_TMAX                      ? :warm :
             core > bottom                      ? :inverted :
             (core > WW_ZMAX || bottom > WW_ZMAX) ? :deep : kind
    reason == kind || return ww_result(:none, reason; mld = h, T = Tc, S = Sc, core, top, bottom)
    return ww_result(kind, reason; mld = h, T = Tc, S = Sc, core, top, bottom)
end
