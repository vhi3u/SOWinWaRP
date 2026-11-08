# ==============================================================================
# sponge.jl -- keep the boundary sponge out of every band average, in one place.
#
#     include(joinpath(@__DIR__, "sponge.jl"))   # needs Printf
#     keep = sponge_trimmer(LON, LAT)            # (λ, φ) -> Bool, true where the water is FREE
#     span = trimmed_band(LAT, lo, hi)           # what a band actually covers after trimming
#
# WHY. `src/model.jl` opens ALL FOUR faces (`for side in (:west, :east, :south, :north)`) and
# `bsose_sponge_layer_forcing` in src/setup_bsose.jl tapers a restoring mask inward from every
# one of them -- its `d_edge` is the MINIMUM of the distances to all four:
#
#     const SPONGE_LAYERS    = true
#     const SPONGE_WIDTH     = 3.0      # degrees; 18 cells at 1/6 deg
#     const SPONGE_TIMESCALE = 12hours  # T, S, u and v restored to B-SOSE
#
# Over 90-150E / 70-45S that puts rows 1-18 (70S to 67S) inside the southern sponge -- 18 of the
# 30 rows of the 70-65S band, which is the one band where the stratification excess lives. The
# raised cosine 0.5(1+cos(pi*d/w)) averages to 0.5 across the taper, so roughly 30% of that
# band's area is being pulled onto B-SOSE on a 12-24 hour timescale. The 90-93E and 147-150E
# strips take 10% of the longitude out of every band for the same reason.
#
# A band mean taken over that water is NOT a measure of how far the model has drifted from
# B-SOSE. It is partly a measure of how hard we are holding it there. The bias is
# one-directional -- restoring can only make the model look MORE like B-SOSE -- so an untrimmed
# band mean UNDERSTATES every anomaly, and understates it most in the most sponged band.
#
# This is the caveat recorded in the sponge-timescale note as "the sponged band is no longer a
# free model solution and must be excluded from analysis", finally applied.
#
# TRIM_SPONGE=0 restores whole-band behaviour so a trimmed and untrimmed pair can be compared.
# It is not the default, because the default should be the measurement that means something.
# ==============================================================================

const SPONGE_TRIM = parse(Float64, get(ENV, "TRIM_SPONGE", "3.0"))

"""
    sponge_trimmer(lon, lat; trim = SPONGE_TRIM) -> (λ, φ) -> Bool

`true` where a cell is OUTSIDE the sponge, i.e. where the model is a free solution. `lon` and
`lat` are the domain extents as (min, max) tuples, in degrees, with southern latitudes negative.
Every face is open, so all four edges are trimmed. `trim <= 0` keeps everything.
"""
function sponge_trimmer(lon, lat; trim = SPONGE_TRIM)
    trim <= 0 && return (λ, φ) -> true
    return (λ, φ) -> (λ - lon[1] >= trim && lon[2] - λ >= trim &&
                      φ - lat[1] >= trim && lat[2] - φ >= trim)
end

"""
    trimmed_band(lat, lo, hi; trim = SPONGE_TRIM) -> (lo, hi) or nothing

The latitude span a band actually covers once the sponge is removed, so a label like "70S-65S"
can be printed alongside what it now means rather than what it used to. `nothing` when the trim
eats the band whole -- which is a real possibility for a band narrower than SPONGE_WIDTH, and is
the caller's cue to stop rather than report an average over no water.
"""
function trimmed_band(lat, lo, hi; trim = SPONGE_TRIM)
    trim <= 0 && return (lo, hi)
    l = max(lo, lat[1] + trim)
    h = min(hi, lat[2] - trim)
    return l < h ? (l, h) : nothing
end

"""One line naming what is excluded, for the top of a tool's output."""
function sponge_note(lon, lat; trim = SPONGE_TRIM)
    trim <= 0 && return "TRIM_SPONGE=0 -- bands span the FULL domain, sponge included. " *
                        "Anomalies are damped toward B-SOSE and are UNDERSTATED."
    # @sprintf needs ONE literal format string -- a `*` concatenation of literals is not one,
    # and the error it throws names the argument rather than the concatenation.
    return @sprintf("TRIM_SPONGE=%.1f deg -- excluding %.0f-%.0fE, %.0f-%.0fE, %.0f-%.0fS and %.0f-%.0fS.\n  SPONGE_WIDTH in src/model.jl is 3.0 deg at a 12 h timescale; water inside it is\n  restored to B-SOSE and is not a free model solution.",
                    trim, lon[1], lon[1] + trim, lon[2] - trim, lon[2],
                    -lat[1], -(lat[1] + trim), -(lat[2] - trim), -lat[2])
end
