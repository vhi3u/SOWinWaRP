# ==============================================================================================
# single_column_bsose.jl -- vertical mixing schemes in ONE column under B-SOSE's own surface fluxes,
# against B-SOSE's own evolution of that column.
#
#     BSOSE_DIR=/storage/scratch1/1/vnguyen480/SOWinWaRP/data julia --project=. tools/single_column_bsose.jl
#     BSOSE_DIR=... START=2015-04-01 END=2015-11-01 julia --project=. tools/single_column_bsose.jl
#
# WHY. 2Y_nested_bsose_ggl_hj_pr207_icein mixes too SHALLOW under the ice (66-64S at 119.9E: B-SOSE
# 250-360 m, run 100-180 m) with the right brine and the right top-cell salinity, and too DEEP in
# open water (55-50S, 40-100 m in 2015). Both could be CATKE, or the run's state and surface fluxes.
# Here each column gets exactly B-SOSE's fluxes and starts from exactly B-SOSE's profile, so the
# only thing left to differ is the mixing scheme (and B-SOSE's lateral supply, measured below).
#
# WHAT IT DOES, per column (a small lon-lat box, box-mean over wet B-SOSE cells):
#   grid      B-SOSE's own levels (RF of the Theta file) down to ZMAX or the box's floor
#   start     B-SOSE's 5-day T, S of the first record at or after START, at rest (u = v = 0)
#   forcing   B-SOSE's 5-day records, held constant over each 5-day window (the window means):
#               heat   TFLUX - oceQsw at the top; oceQsw absorbed with MITgcm's swfrac jwtype 2
#                      (0.62, 0.6 m, 20 m), the rest of it in the bottom cell
#               salt   SFLUX, plus -S * oceFWflx (virtual salt flux, S the column's own top cell) ONLY when
#                      LATERAL=none: B-SOSE's monthly snapshots show its salinity tendency (TOTSTEND) closes with
#                      the advective + diffusive terms + SFLUX alone, i.e. its fresh water and brine reach the
#                      salinity THROUGH the advective terms (the inflow replacing water removed at the surface).
#                      With those terms applied, adding -S*oceFWflx counts it twice (86c2b9c did: 65S too salty,
#                      52S too fresh).
#               stress oceTAUX, oceTAUY (under ice: B-SOSE's ice-ocean + air-ocean stress), or with STRESS:
#                      era5_5day    ERA5 Large & Yeager stress (tools/era5_stress.jl: neutral, absolute wind, ρa
#                                   from ERA5, as the run's coupler) averaged over each 5-day window
#                      era5_hourly  the same stress hour by hour, linear in time between ERA5 hours
#                      era5_6hourly ERA5's wind, T, dewpoint and pressure at 00/06/12/18 only, linear in time to every
#                                   hour, then the stress (as EXF does with 6-hourly forcing, e.g. B-SOSE's ERA-Interim)
#                      era5_hourly_relative  hourly ERA5 wind MINUS the column's own surface current at every step,
#                                   τ = ρa Cd(|ΔU|) |ΔU| ΔU from the box-mean wind: what the run's coupler does
#                                   (RelativeVelocity), against B-SOSE's absolute wind; it damps inertial currents
#                      Both are OPEN-WATER stress over the box's ERA5 cells: meaningful only where B-SOSE has no ice
#                      (a warning names windows with box SIArea > 0.15). era5_hourly against era5_5day isolates
#                      what sub-5-day wind variability does to the mixed layer (the run is forced hourly);
#                      era5_5day against bsose, the mean stress (~1.07x B-SOSE's north of 65S, compare_wind_stress).
#             as the run: flux = -X / rho0 (tracers also / cp), rho0 1020, cp 3991.868, TEOS-10
#   Coriolis  f-plane at the box's centre latitude
#   schemes   CLOSURES, from
#               catke       the run's "ggl": NumericalEarth's CATKE (Cᵇ 0.01, Cᵂϵ 1.0) + νz 1e-3, κz KAPPA_Z (1e-5)
#               catke_bare  that CATKE alone (the _catke runs)
#               kepsilon    Oceananigans' k-ε (TKEDissipationVerticalDiffusivity), WITHOUT the backgrounds:
#                           in 0.117.2 its TKE top boundary condition fails on a closure tuple
#               ri          Oceananigans' RiBasedVerticalDiffusivity + the same backgrounds
#             k-ε and Ri are CONTROLS, not candidates: Oceananigans has no GGL90. If every scheme misses
#             B-SOSE the same way, the column (not the scheme) is missing something -- read the budget.
#
# MLD. tools/mld.jl (0.03 kg/m³ from 10 m) of the 5-DAY-MEAN state on both sides: the model's hourly T, S
# averaged over each B-SOSE window, against B-SOSE's window mean. A column with no crossing counts at its
# floor and is marked '*'. B-SOSE's own MLD diagnostic (box mean) is printed for reference only: it is
# shallower by its definition (compare_mld_5day).
#
# THE BUDGET IS THE CAVEAT. B-SOSE's column also changes by lateral advection, eddies and its adjoint's
# adjustments; a 1-D column cannot. Printed per month: B-SOSE's change in column heat and salt since the
# start against the surface forcing integrated over the same time, and the remainder ("lateral"). A
# remainder comparable to the forcing means B-SOSE's MLD there is not a 1-D target, and the comparison
# is between schemes only. The model's own remainder should be ~0 (a conservation check).
# (First run, 2014: at 52.5-53S the lateral salt was 14-21 kg/m² against 3-11 from the surface; at 65S every
# scheme convected to 700-800 m against B-SOSE's 200-240.)
#
# LATERAL=monthly (default) PUTS THAT SUPPLY BACK, from B-SOSE's own MONTHLY budget diagnostics (the 5-day ones
# are listed on the server but are dead links). Per level of the box, per calendar month:
#   horizontal  ADVx/ADVy + DFxE/DFyE through the box's four sides (MITgcm: x flux on the cell's west face, y on
#               its south face, positive east/north, tracer·m³/s)
#   vertical    ADVr through the level faces inside the column and at its floor, DFrI at its floor (positive up,
#               on each level's TOP face); ADVr at the sea surface is NOT applied (the column has its own surface
#               forcing) but is printed
# divided by the box's B-SOSE volume and added as a tendency, constant through each month. What is left to the
# column is its vertical mixing and the surface. It is B-SOSE's supply, so it does not respond if the column
# drifts from B-SOSE: fine for months, not for years.
# Checked first, per month and over the column: TOTTTEND/TOTSTEND against the sum of the terms (CLOSURE table;
# the salt surface term is shown with and without -S*oceFWflx, since B-SOSE's real freshwater flux makes that
# split ambiguous). A large residual means the supply is not B-SOSE's whole story.
# Then B-SOSE's own vertical mixing flux DFrI (GGL90 + backgrounds, implicit) against each scheme's, the scheme's
# from its 5-day snapshots: flux at each face = what its tendency needs beyond the sources below it.
# Files: <token>_bsoseI156_2013to2024_monthly.nc from BSOSE_DIR if present, else read over the network from
# BUDGET_URL with NetCDF byte ranges (~8 s per term per box; nothing is downloaded). Cached per box and months in
# OUTDIR/.single_column_budget_*.jls.
#
# WRITES OUTDIR/single_column_<column>.png: MLD against time (B-SOSE, B-SOSE's diagnostic, each scheme),
# B-SOSE's ice concentration, and S and T profiles at four times; with LATERAL=monthly also
# OUTDIR/single_column_<column>_mixing_flux.png, B-SOSE's DFrI against each scheme's mixing flux, by month.
#
# ENV
#   BSOSE_DIR  (required) B-SOSE's 5-day files: Theta, Salt, surfTflx, surfSflx, oceFWflx, oceQsw, oceTAUX,
#              oceTAUY (required); SIArea, MLD (optional)
#   BOXES      name:lon0:lon1:lat0:lat1,...  (default ice65S:119.4:120.4:-65.0:-64.5,open52S:119.4:120.4:-53.0:-52.5)
#   START, END first and last 5-day record stamps used (default 2014-06-01, 2014-11-01)
#   CLOSURES   (default catke,catke_bare,kepsilon; ri still available, dropped from the default 2026-11-22)
#   ZMAX       deepest level, m (default 1000)    DT  time step, s (default 600)
#   OUTDIR     (default plots)                    MLD_DT (default 0.03 kg/m³, read by mld.jl)
#   LATERAL    monthly | none                      (default monthly)
#   BUDGET_URL (default https://sose.ucsd.edu/SO6/ITER156/MONTHLY/)
#   RHO_BSOSE  B-SOSE's reference density for its own budget check, kg/m³ (default 1035; cp 3994)
#   STRESS     bsose | era5_5day | era5_hourly | era5_6hourly | era5_hourly_relative (default bsose); ERA5_DIR
#              (default BSOSE_DIR/era5)
#   KAPPA_Z    background κz of catke and ri, m²/s (default 1e-5 since 2026-11-22: B-SOSE i156's, and the runs' since
#              _k1e5; every single-column result before that used 1e-4)
#   Figures of a non-default STRESS or KAPPA_Z are named single_column_<column>_<STRESS>[_k<KAPPA_Z>]*.png
# ==============================================================================================

using Oceananigans
using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities: CATKEMixingLength, CATKEEquation
using SeawaterPolynomials.TEOS10: TEOS10EquationOfState
using NCDatasets, CairoMakie, Dates, Printf, Statistics, Serialization

ENV["HDF5_USE_FILE_LOCKING"] = "FALSE"
include(joinpath(@__DIR__, "mld.jl"))          # mld, MLD_DT
include(joinpath(@__DIR__, "era5_stress.jl"))  # era5_box_stress

const BSOSE_DIR = get(ENV, "BSOSE_DIR", "")
isempty(BSOSE_DIR) && error("set BSOSE_DIR to the directory with B-SOSE's 5-day files")
const OUTDIR   = get(ENV, "OUTDIR", "plots")
const START    = DateTime(get(ENV, "START", "2014-06-01"))
const STOP     = DateTime(get(ENV, "END", "2014-11-01"))
const ZMAX     = parse(Float64, get(ENV, "ZMAX", "1000"))
const DT       = parse(Float64, get(ENV, "DT", "600"))
const CLOSURES = String.(split(get(ENV, "CLOSURES", "catke,catke_bare,kepsilon"), ','))
const ρ₀, cₚ   = 1020.0, 3991.86795711963      # the run's (NumericalEarth's) reference density, heat capacity
const SWR, SWζ1, SWζ2 = 0.62, 0.6, 20.0        # MITgcm swfrac, jwtype 2
const WINDOW   = Day(5)
const LATERAL  = get(ENV, "LATERAL", "monthly")
LATERAL in ("monthly", "none") || error("LATERAL must be monthly or none, got $LATERAL")
const BUDGET_URL = get(ENV, "BUDGET_URL", "https://sose.ucsd.edu/SO6/ITER156/MONTHLY/")
const ρB       = parse(Float64, get(ENV, "RHO_BSOSE", "1035"))
const STRESS   = get(ENV, "STRESS", "bsose")
const STRESSES = ("bsose", "era5_5day", "era5_hourly", "era5_6hourly", "era5_hourly_relative")
STRESS in STRESSES || error("STRESS must be one of $(join(STRESSES, ", ")), got $STRESS")
const KAPPA_Z  = parse(Float64, get(ENV, "KAPPA_Z", "1e-5"))
const ERA5_DIR = get(ENV, "ERA5_DIR", joinpath(get(ENV, "BSOSE_DIR", ""), "era5"))
const TAG      = (STRESS == "bsose" ? "" : "_$STRESS") * (KAPPA_Z == 1e-5 ? "" : "_k$(KAPPA_Z)")
const cpB      = 3994.0

struct Column
    name::String
    lon::Tuple{Float64, Float64}
    lat::Tuple{Float64, Float64}
end

function parse_columns(s)
    map(split(s, ',')) do c
        p = split(c, ':')
        length(p) == 5 || error("BOXES entry '$c' is not name:lon0:lon1:lat0:lat1")
        v = parse.(Float64, p[2:5])
        Column(String(p[1]), (min(v[1], v[2]), max(v[1], v[2])), (min(v[3], v[4]), max(v[3], v[4])))
    end
end
const COLUMNS = parse_columns(get(ENV, "BOXES",
                    "ice65S:119.4:120.4:-65.0:-64.5,open52S:119.4:120.4:-53.0:-52.5"))

# ── B-SOSE files (same conventions as compare_mld_5day.jl) ────────────────────────────────────────
f64(A) = Float64.(coalesce.(A, NaN))
coord(ds, name, dim) = (v = f64(ds[name][:]); ndims(v) == 1 ? v : dim == 1 ? v[:, 1] : v[1, :])

function bsose_file(token, y)
    want = lowercase("$(token)_bsoseI156_$(y)_5day.nc")
    hits = filter(f -> lowercase(f) == want, readdir(BSOSE_DIR))
    isempty(hits) ? "" : joinpath(BSOSE_DIR, first(hits))
end

function data_var(ds, names, path)
    for n in names
        haskey(ds, n) && return n
    end
    cands = [k for k in keys(ds) if ndims(ds[k]) >= 3]
    length(cands) == 1 && return only(cands)
    error("none of $names in $path, and no single data variable ($(cands))")
end

const TOKENS = Dict(                  # file token => variable names inside
    "Theta" => ("THETA", "Theta"), "Salt" => ("SALT", "Salt"),
    "surfTflx" => ("TFLUX",), "surfSflx" => ("SFLUX",), "oceFWflx" => ("oceFWflx",),
    "oceQsw" => ("oceQsw",), "oceTAUX" => ("oceTAUX",), "oceTAUY" => ("oceTAUY",),
    "SIArea" => ("SIarea", "SIArea", "SIAREA"), "MLD" => ("MLD", "MXLDEPTH", "BLGMLD"))
const REQUIRED = ("Theta", "Salt", "surfTflx", "surfSflx", "oceFWflx", "oceQsw", "oceTAUX", "oceTAUY")

"stamp => (path, variable, record) for one token; stamps are the END of each 5-day window."
function records(token, years)
    out = Dict{DateTime, Tuple{String, String, Int}}()
    for y in years
        path = bsose_file(token, y)
        isempty(path) && continue
        NCDataset(path) do ds
            var = data_var(ds, TOKENS[token], path)
            for (n, s) in enumerate(ds["time"][:])
                out[DateTime(s)] = (path, var, n)
            end
        end
    end
    return out
end

# ── B-SOSE's monthly budget, on the box (LATERAL=monthly) ─────────────────────────────────────────────
const MONTHLY_SPAN = "2013to2024_monthly"
const SURFACE_NAMES = Dict("surfTflx" => ("TFLUX",), "oceQsw" => ("oceQsw",), "surfSflx" => ("SFLUX",),
                           "oceFWflx" => ("oceFWflx",), "Salt" => ("SALT", "Salt"))

function monthly_path(token)
    want = lowercase("$(token)_bsoseI156_$(MONTHLY_SPAN).nc")
    hits = filter(f -> lowercase(f) == want, readdir(BSOSE_DIR))
    isempty(hits) || return joinpath(BSOSE_DIR, first(hits))
    return BUDGET_URL * "$(token)_bsoseI156_$(MONTHLY_SPAN).nc#mode=bytes"
end

"(year, month) a monthly record averages: its stamp is the END of the month and drifts by hours."
monthkey(s) = (d = DateTime(s) - Day(14); (Dates.year(d), Dates.month(d)))
fill0(A) = (B = f64(A); B[.!isfinite.(B)] .= 0; B)

"NCDataset(f, path), retried for a remote file: one stalled byte-range read must not end a run."
function open_monthly(f, path)
    for attempt in 1:3
        try
            return NCDataset(f, path)
        catch e
            (attempt == 3 || !endswith(path, "#mode=bytes")) && rethrow()
            @warn "reading $path failed (attempt $attempt of 3); retrying in 30 s" exception = e
            sleep(30)
        end
    end
end

function month_range(ds, months, path)
    have = monthkey.(ds["time"][:])
    idx = [findfirst(==(m), have) for m in months]
    any(isnothing, idx) && error("$path has no record for $(months[findfirst(isnothing, idx)])")
    idx == collect(first(idx):last(idx)) || error("$path: the monthly records are not consecutive")
    return first(idx):last(idx)
end

"Sum over a face (or the box) of a monthly 3-D term: levels × months."
function face_sum(token, is, js, levels, months)
    path = monthly_path(token)
    open_monthly(path) do ds
        m = month_range(ds, months, path)
        X = fill0(ds[token][is, js, levels, m])
        return dropdims(sum(X, dims = (1, 2)), dims = (1, 2))
    end
end

"""
B-SOSE's monthly heat and salt budget of the box `I × J`, levels 1:nk (top down), for the (year, month) keys
`months`. Every 3-D term is a box total in tracer·m³/s.
"""
function read_budget(I, J, nk, months, XCref, YCref)
    nk + 1 <= 52 || error("the budget needs the face below level $nk; lower ZMAX")
    i0, i1, j0, j1 = first(I), last(I), first(J), last(J)
    t0 = time()
    rA, h, drF = open_monthly(monthly_path("TOTTTEND")) do ds
        xc, yc = coord(ds, "XC", 1), coord(ds, "YC", 2)
        (abs(xc[i0] - XCref[i0]) < 1e-3 && abs(yc[j0] - YCref[j0]) < 1e-3) ||
            error("the monthly budget files are not on the 5-day files' grid")
        fill0(ds["rA"][I, J]), fill0(ds["hFacC"][I, J, 1:nk]), f64(ds["drF"][1:nk])
    end
    cellV = rA .* h .* reshape(drF, 1, 1, nk)
    V = [sum(cellV[:, :, k]) for k in 1:nk]
    A = [sum(rA .* (h[:, :, k] .> 0)) for k in 1:nk]
    surf = h[:, :, 1] .> 0

    tracer = Dict{String, Any}()
    for (tr, tot) in (("TH", "TOTTTEND"), ("SLT", "TOTSTEND"))
        path = monthly_path(tot)
        tend = open_monthly(path) do ds
            m = month_range(ds, months, path)
            X = fill0(ds[tot][I, J, 1:nk, m])
            [sum(X[:, :, k, n] .* cellV[:, :, k]) / 86400 for k in 1:nk, n in 1:length(m)]
        end
        hadv = face_sum("ADVx_$tr", i0:i0, J, 1:nk, months) .- face_sum("ADVx_$tr", i1+1:i1+1, J, 1:nk, months) .+
               face_sum("ADVy_$tr", I, j0:j0, 1:nk, months) .- face_sum("ADVy_$tr", I, j1+1:j1+1, 1:nk, months)
        hdif = face_sum("DFxE_$tr", i0:i0, J, 1:nk, months) .- face_sum("DFxE_$tr", i1+1:i1+1, J, 1:nk, months) .+
               face_sum("DFyE_$tr", I, j0:j0, 1:nk, months) .- face_sum("DFyE_$tr", I, j1+1:j1+1, 1:nk, months)
        R = face_sum("ADVr_$tr", I, J, 1:nk+1, months)       # on each level's top face, positive up
        D = face_sum("DFrI_$tr", I, J, 1:nk+1, months)
        tracer[tr] = (; tend, hadv, hdif, R, D)
    end
    sfc = Dict{String, Vector{Float64}}()
    for tok in ("surfTflx", "oceQsw", "surfSflx", "oceFWflx", "Salt")
        path = monthly_path(tok)
        sfc[tok] = open_monthly(path) do ds
            m = month_range(ds, months, path)
            v = ds[data_var(ds, SURFACE_NAMES[tok], path)]
            X = fill0(ndims(v) == 4 ? v[I, J, 1, m] : v[I, J, m])
            tok == "Salt" ? [mean(X[:, :, n][surf]) for n in 1:length(m)] :      # box-mean top S
                            [sum(X[:, :, n] .* rA .* surf) for n in 1:length(m)]  # box total
        end
    end
    @info @sprintf("B-SOSE monthly budget for the box read in %.0f s", time() - t0)
    return (; months, V, A, TH = tracer["TH"], SLT = tracer["SLT"], sfc)
end

"Box budget, cached in OUTDIR so a re-run (other schemes, other dates inside the months) is not another download."
function box_budget(name, I, J, nk, months, XC, YC)
    mkpath(OUTDIR)
    key = @sprintf("%s_i%d-%d_j%d-%d_k%d_%04d%02d-%04d%02d", name, first(I), last(I), first(J), last(J), nk,
                   months[1]..., months[end]...)
    path = joinpath(OUTDIR, ".single_column_budget_$key.jls")
    isfile(path) && (@info "budget from cache $path"; return deserialize(path))
    b = read_budget(I, J, nk, months, XC, YC)
    serialize(path, b)
    return b
end

"∫ₐᵇ of a piecewise-constant series (segment n from starts[n]; the first extends back, the last forward)."
function piecewise_integral(starts, vals::AbstractMatrix, a, b)
    out = zeros(size(vals, 1))
    for n in eachindex(starts)
        lo = n == 1 ? -Inf : starts[n]
        hi = n == length(starts) ? Inf : starts[n+1]
        dt = max(0.0, min(b, hi) - max(a, lo))
        dt > 0 && (out .+= dt .* view(vals, :, n))
    end
    return out
end

# ── single-column model pieces ────────────────────────────────────────────────────────────────────
@inline window_index(p, t) = clamp(searchsortedlast(p.ts, t), 1, length(p.ts))
@inline Jᵀ(i, j, grid, clock, f, p) = (n = window_index(p, clock.time); -(p.Qtot[n] - p.Qsw[n]) / (p.ρ₀ * p.cₚ))
@inline Jˢ(i, j, grid, clock, f, p) = (n = window_index(p, clock.time);
                                       -(p.SF[n] - p.dilute * @inbounds(f.S[i, j, grid.Nz]) * p.FW[n]) / p.ρ₀)
# hourly stress (STRESS=era5_hourly): linear between ERA5 hours p.th (model seconds), held at the ends
@inline function hourly(th, τ, t)
    k = clamp(searchsortedlast(th, t), 1, length(th) - 1)
    w = clamp((t - th[k]) / (th[k+1] - th[k]), 0.0, 1.0)
    return (1 - w) * τ[k] + w * τ[k+1]
end
# relative wind (STRESS=era5_hourly_relative): the hourly box-mean ERA5 wind minus the top cell's current
@inline function relative_stress(i, j, grid, clock, f, p)
    t = clock.time
    Δu = hourly(p.th, p.uh, t) - @inbounds(f.u[i, j, grid.Nz])
    Δv = hourly(p.th, p.vh, t) - @inbounds(f.v[i, j, grid.Nz])
    ΔU = sqrt(Δu^2 + Δv^2)
    k = hourly(p.th, p.ρh, t) * neutral_cd(ΔU) * ΔU
    return k * Δu, k * Δv
end
@inline Jᵘ(i, j, grid, clock, f, p) = p.relative ? -relative_stress(i, j, grid, clock, f, p)[1] / p.ρ₀ :
                                      p.hourly ? -hourly(p.th, p.τxh, clock.time) / p.ρ₀ :
                                                 (n = window_index(p, clock.time); -p.τx[n] / p.ρ₀)
@inline Jᵛ(i, j, grid, clock, f, p) = p.relative ? -relative_stress(i, j, grid, clock, f, p)[2] / p.ρ₀ :
                                      p.hourly ? -hourly(p.th, p.τyh, clock.time) / p.ρ₀ :
                                                 (n = window_index(p, clock.time); -p.τy[n] / p.ρ₀)
@inline swfrac(z, p) = p.R * exp(z / p.ζ1) + (1 - p.R) * exp(z / p.ζ2)
@inline function shortwave_heating(i, j, k, grid, clock, f, p)
    n = window_index(p, clock.time)
    @inbounds ztop, zbot = p.zf[k+1], p.zf[k]
    below = k == 1 ? zero(ztop) : swfrac(zbot, p)          # the bottom cell keeps what reaches it
    return p.Qsw[n] * (swfrac(ztop, p) - below) / (p.ρ₀ * p.cₚ * (ztop - zbot))
end
# B-SOSE's lateral supply: p.LT, p.LS are levels (TOP DOWN) × months, constant from each p.tl (month start)
@inline month_index(p, t) = clamp(searchsortedlast(p.tl, t), 1, length(p.tl))
@inline lateral_S(i, j, k, grid, clock, f, p) = @inbounds p.LS[grid.Nz - k + 1, month_index(p, clock.time)]
@inline T_sources(i, j, k, grid, clock, f, p) =
    shortwave_heating(i, j, k, grid, clock, f, p) + @inbounds(p.LT[grid.Nz - k + 1, month_index(p, clock.time)])

function build_closure(name)
    background = VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(); ν = 1e-3, κ = KAPPA_Z)
    catke() = CATKEVerticalDiffusivity(VerticallyImplicitTimeDiscretization(), Float64;
                                       mixing_length = CATKEMixingLength(Cᵇ = 0.01),
                                       turbulent_kinetic_energy_equation = CATKEEquation(Cᵂϵ = 1.0))
    name == "catke"      && return (catke(), background)
    name == "catke_bare" && return catke()
    name == "kepsilon"   && return TKEDissipationVerticalDiffusivity()     # alone: its TKE top BC rejects a closure tuple (0.117.2)
    name == "ri"         && return (RiBasedVerticalDiffusivity(), background)
    error("unknown closure '$name' (catke, catke_bare, kepsilon, ri)")
end

"""
Run one scheme; return its window-mean T and S (top-down, nk × nr; record 1 is the start) and its T and S
snapshots at the run's edges (nk × nr+1: the start, then the end of each window).
"""
function run_column(name, zf, Tinit, Sinit, φc, p, edges)
    nk, nr = length(Tinit), length(edges) - 1
    grid = RectilinearGrid(CPU(); size = nk, z = zf, topology = (Flat, Flat, Bounded))
    bcs = (T = FieldBoundaryConditions(top = FluxBoundaryCondition(Jᵀ; discrete_form = true, parameters = p)),
           S = FieldBoundaryConditions(top = FluxBoundaryCondition(Jˢ; discrete_form = true, parameters = p)),
           u = FieldBoundaryConditions(top = FluxBoundaryCondition(Jᵘ; discrete_form = true, parameters = p)),
           v = FieldBoundaryConditions(top = FluxBoundaryCondition(Jᵛ; discrete_form = true, parameters = p)))
    model = HydrostaticFreeSurfaceModel(grid;
                buoyancy = SeawaterBuoyancy(equation_of_state = TEOS10EquationOfState(reference_density = ρ₀)),
                coriolis = FPlane(latitude = φc), closure = build_closure(name), tracers = (:T, :S),
                boundary_conditions = bcs,
                forcing = (T = Forcing(T_sources; discrete_form = true, parameters = p),
                           S = Forcing(lateral_S; discrete_form = true, parameters = p)))
    set!(model, T = reshape(reverse(Tinit), 1, 1, nk), S = reshape(reverse(Sinit), 1, 1, nk))
    haskey(model.tracers, :e) && set!(model, e = 1e-6)
    haskey(model.tracers, :ϵ) && set!(model, ϵ = 1e-9)

    Tm, Sm = fill(NaN, nk, nr), fill(NaN, nk, nr)
    Tm[:, 1], Sm[:, 1] = Tinit, Sinit
    Ts, Ss = fill(NaN, nk, nr + 1), fill(NaN, nk, nr + 1)
    snap!(c) = (Ts[:, c] = reverse(vec(interior(model.tracers.T))); Ss[:, c] = reverse(vec(interior(model.tracers.S))))
    snap!(1)
    accT, accS, count = zeros(nk), zeros(nk), Ref(0)
    function accumulate!(sim)
        accT .+= reverse(vec(interior(sim.model.tracers.T)))
        accS .+= reverse(vec(interior(sim.model.tracers.S)))
        count[] += 1
    end
    sim = Simulation(model; Δt = DT, stop_time = edges[2])
    add_callback!(sim, accumulate!, TimeInterval(3600.0))
    t0 = time()
    run!(sim)                                   # centre of record 1 to the end of its window
    snap!(2)
    for r in 2:nr
        accT .= 0; accS .= 0; count[] = 0
        sim.stop_time = edges[r+1]
        run!(sim)
        if count[] == 0 || !all(isfinite, accT) || !all(isfinite, accS)
            @warn "$name: non-finite T or S (or no samples) in window $r; stopping this scheme there"
            break
        end
        Tm[:, r], Sm[:, r] = accT ./ count[], accS ./ count[]
        snap!(r + 1)
    end
    @info @sprintf("%s: %d steps, %.0f s wall", name, iteration(sim), time() - t0)
    return Tm, Sm, Ts, Ss
end

# ── one column ────────────────────────────────────────────────────────────────────────────────────
function do_column(col, recs, stamps, XC, YC, RF)
    ir = findall(x -> col.lon[1] <= x <= col.lon[2], XC)
    jr = findall(y -> col.lat[1] <= y <= col.lat[2], YC)
    (isempty(ir) || isempty(jr)) && error("$(col.name): no B-SOSE cell in $(col.lon) x $(col.lat)")
    I, J = first(ir):last(ir), first(jr):last(jr)
    φc = mean(YC[J])

    read2(tok, s) = (r = recs[tok][s]; NCDataset(ds -> (v = ds[r[2]]; f64(ndims(v) == 4 ? v[I, J, 1, r[3]] : v[I, J, r[3]])), r[1]))
    read3(tok, s, nk) = (r = recs[tok][s]; NCDataset(ds -> f64(ds[r[2]][I, J, 1:nk, r[3]]), r[1]))

    # geometry from the first record: a level counts if at least half the box is wet there
    S1 = read3("Salt", stamps[1], length(RF) - 1)
    wet = (S1 .!= 0) .& isfinite.(S1)
    ncell = size(wet, 1) * size(wet, 2)
    kfloor = findlast(k -> count(view(wet, :, :, k)) >= ncell / 2, 1:size(wet, 3))
    kfloor === nothing && error("$(col.name): the box is land")
    nk = min(kfloor, something(findlast(k -> -RF[k+1] <= ZMAX, 1:length(RF)-1), 1))
    surf = wet[:, :, 1]
    zc = [(RF[k] + RF[k+1]) / 2 for k in 1:nk]          # top-down, negative
    Δz = [RF[k] - RF[k+1] for k in 1:nk]
    H = -RF[nk+1]
    @info @sprintf("%s: %d x %d B-SOSE cells (%.2f-%.2fE, %.2f-%.2fS), %d levels to %.0f m, f-plane at %.2f",
                   col.name, length(I), length(J), XC[first(I)], XC[last(I)], -YC[last(J)], -YC[first(J)],
                   nk, H, φc)

    boxmean2(A) = mean(A[surf])
    boxmean3(A) = [(m = view(wet, :, :, k); any(m) ? mean(view(A, :, :, k)[m]) : NaN) for k in 1:nk]

    nr = length(stamps)
    Tb, Sb = zeros(nk, nr), zeros(nk, nr)
    Qtot, Qsw, SF, FW, τx, τy, ice, mdiag = (fill(NaN, nr) for _ in 1:8)
    for (r, s) in enumerate(stamps)
        Tb[:, r] = boxmean3(read3("Theta", s, nk)); Sb[:, r] = boxmean3(read3("Salt", s, nk))
        Qtot[r] = boxmean2(read2("surfTflx", s)); Qsw[r] = boxmean2(read2("oceQsw", s))
        SF[r] = boxmean2(read2("surfSflx", s));   FW[r] = boxmean2(read2("oceFWflx", s))
        τx[r] = boxmean2(read2("oceTAUX", s));    τy[r] = boxmean2(read2("oceTAUY", s))
        haskey(recs, "SIArea") && haskey(recs["SIArea"], s) && (ice[r] = boxmean2(read2("SIArea", s)))
        haskey(recs, "MLD") && haskey(recs["MLD"], s) && (mdiag[r] = boxmean2(read2("MLD", s)))
    end
    any(!isfinite, Tb) && error("$(col.name): B-SOSE T has gaps in the column (a level with no wet cell?)")

    # ── STRESS=era5_*: ERA5 Large & Yeager stress over the box replaces B-SOSE's ──
    th, τxh, τyh, uh, vh, ρh = Float64[], Float64[], Float64[], Float64[], Float64[], Float64[]
    if STRESS != "bsose"
        eh, ex, ey, eρ, eu, ev = era5_box_stress(ERA5_DIR, col.lon, col.lat, stamps[1] - WINDOW, stamps[end];
                                                 every = STRESS == "era5_6hourly" ? 6 : 1)
        τb = copy(hypot.(τx, τy))
        for (r, s) in enumerate(stamps)
            sel = findall(t -> s - WINDOW < t <= s, eh)
            isempty(sel) && error("$(col.name): no ERA5 hours in the window ending $s")
            τx[r], τy[r] = mean(ex[sel]), mean(ey[sel])
        end
        iced = [Dates.format(stamps[r], "yyyy-mm-dd") for r in 1:nr if isfinite(ice[r]) && ice[r] > 0.15]
        isempty(iced) || @warn "$(col.name): B-SOSE ice > 0.15 in $(length(iced)) windows ($(first(iced))...): ERA5 stress is open-water stress"
        @info @sprintf("%s: STRESS=%s, %d ERA5 hours; 5-day |<τ>| ERA5 / B-SOSE = %.2f (mean over windows)",
                       col.name, STRESS, length(eh), mean(hypot.(τx, τy) ./ τb))
        if STRESS != "era5_5day"
            th = [Dates.value(t - (stamps[1] - Hour(60))) / 1000 for t in eh]
            τxh, τyh, uh, vh, ρh = ex, ey, eu, ev, eρ
        end
    end

    # model time 0 = centre of record 1; window r is (stamp_r - 5 d, stamp_r]
    t0 = stamps[1] - Hour(60)                     # half of the 5-day window
    secs(d) = Dates.value(d - t0) / 1000
    ts = [secs(s - WINDOW) for s in stamps]                 # forcing window starts (ts[1] < 0)
    edges = [0.0; [secs(s) for s in stamps]]                # run 0 -> end of window 1, then window by window
    zf = reverse(RF[1:nk+1]) .+ 0.0                         # bottom to top

    # ── B-SOSE's lateral supply, by calendar month ──
    first_month, last_month = Date(stamps[1] - WINDOW), Date(stamps[end])
    months = Tuple{Int, Int}[]
    d = Date(Dates.year(first_month), Dates.month(first_month), 1)
    while d <= last_month
        push!(months, (Dates.year(d), Dates.month(d))); d += Month(1)
    end
    tl = [secs(DateTime(y, m, 1)) for (y, m) in months]
    nm = length(months)
    bud = LATERAL == "monthly" ? box_budget(col.name, I, J, nk, months, XC, YC) : nothing
    function lateral_source(b)       # tracer/s per level (top down) × month: what the column gets from outside
        L = zeros(nk, nm)
        for n in 1:nm, k in 1:nk
            vert = b.R[k+1, n] - (k == 1 ? 0.0 : b.R[k, n]) + (k == nk ? b.D[nk+1, n] : 0.0)
            L[k, n] = (b.hadv[k, n] + b.hdif[k, n] + vert) / bud.V[k]
        end
        return L
    end
    LT = bud === nothing ? zeros(nk, nm) : lateral_source(bud.TH)
    LS = bud === nothing ? zeros(nk, nm) : lateral_source(bud.SLT)
    dilute = LATERAL == "none" ? 1.0 : 0.0         # see the header: B-SOSE's advective terms already carry the dilution
    p = (; ts, Qtot, Qsw, SF, FW, τx, τy, zf, ρ₀, cₚ, R = SWR, ζ1 = SWζ1, ζ2 = SWζ2, tl, LT, LS, dilute,
           hourly = STRESS in ("era5_hourly", "era5_6hourly"), relative = STRESS == "era5_hourly_relative",
           th, τxh, τyh, uh, vh, ρh)

    mlabel(n) = @sprintf("%04d-%02d", months[n]...)
    if bud !== nothing
        A1 = bud.A[1]
        println("\nB-SOSE'S OWN BUDGET OF THE BOX, 0-$(round(Int, H)) m, monthly (does it close?)")
        println("  heat, W/m² of box surface (ρ cp = $(ρB) x $(cpB)); + = into the column")
        @printf("  %-8s %9s %9s %9s %9s %9s %9s %9s | %9s\n", "month", "TOTTTEND", "h-adv", "h-diff", "ADVr@flr",
                "DFrI@flr", "surface", "residual", "ADVr@sfc")
        s = ρB * cpB / A1
        for n in 1:nm
            b = bud.TH
            tot, ha, hd = sum(b.tend[:, n]), sum(b.hadv[:, n]), sum(b.hdif[:, n])
            va, df, sf = b.R[nk+1, n], b.D[nk+1, n] - b.D[1, n], bud.sfc["surfTflx"][n] / (ρB * cpB)
            @printf("  %-8s %9.1f %9.1f %9.1f %9.1f %9.1f %9.1f %9.1f | %9.1f\n", mlabel(n), s * tot, s * ha, s * hd,
                    s * va, s * df, s * sf, s * (tot - ha - hd - va - df - sf + b.R[1, n]), -s * b.R[1, n])
        end
        println("  salt, g m⁻² day⁻¹; surface = SFLUX, dilution = -S·oceFWflx. B-SOSE closes WITHOUT the dilution (res, not res+dil;\n  checked against its monthly snapshots, Jun-Oct 2014): its advective terms already carry it, so the column omits it.")
        @printf("  %-8s %9s %9s %9s %9s %9s %9s %9s %9s %9s | %9s\n", "month", "TOTSTEND", "h-adv", "h-diff", "ADVr@flr",
                "DFrI@flr", "surface", "dilution", "res", "res+dil", "ADVr@sfc")
        s = ρB * 86400 / A1
        for n in 1:nm
            b = bud.SLT
            tot, ha, hd = sum(b.tend[:, n]), sum(b.hadv[:, n]), sum(b.hdif[:, n])
            va, df = b.R[nk+1, n], b.D[nk+1, n] - b.D[1, n]
            sf, dil = bud.sfc["surfSflx"][n] / ρB, -bud.sfc["Salt"][n] * bud.sfc["oceFWflx"][n] / ρB
            res = tot - ha - hd - va - df - sf + b.R[1, n]
            @printf("  %-8s %9.2f %9.2f %9.2f %9.2f %9.2f %9.2f %9.2f %9.2f %9.2f | %9.2f\n", mlabel(n), s * tot,
                    s * ha, s * hd, s * va, s * df, s * sf, s * dil, s * res, s * (res - dil), -s * b.R[1, n])
        end
        println("  residual = TOTTEND minus every other term. ADVr@sfc (vertical advection through the sea surface) is in ",
                "the residual's\n  sum but NOT applied to the column, which has its own surface forcing. The column gets ",
                "h-adv + h-diff\n  + ADVr through every face below the surface + DFrI at its floor, level by level.")
    end

    results = Dict{String, NTuple{4, Matrix{Float64}}}()
    for name in CLOSURES
        results[name] = run_column(name, zf, Tb[:, 1], Sb[:, 1], φc, p, edges)
    end

    # ── MLD ──
    zbu = reverse(zc)
    mld_of(T, S) = (h = mld(reverse(T), reverse(S), zbu, 1); isfinite(h) ? (h, false) : (H, true))
    hb = [mld_of(Tb[:, r], Sb[:, r]) for r in 1:nr]
    hm = Dict(n => [all(isfinite, results[n][1][:, r]) ? mld_of(results[n][1][:, r], results[n][2][:, r]) : (NaN, false)
                    for r in 1:nr] for n in CLOSURES)

    bar = "="^(52 + 11 * length(CLOSURES))
    println("\n", bar)
    @printf("%s  %.2f-%.2fE %.2f-%.2fS, column %.0f m; MLD (m) of the 5-day-mean state, * = mixed to the floor\n",
            col.name, XC[first(I)], XC[last(I)], -YC[last(J)], -YC[first(J)], H)
    println(bar)
    @printf("%-11s %5s %6s %6s %8s %7s", "window end", "ice", "τ N/m²", "Qnet", "B-SOSE", "diag")
    foreach(n -> @printf(" %10s", n), CLOSURES); println()
    fmt((h, deep)) = isfinite(h) ? @sprintf("%.0f%s", h, deep ? "*" : "") : "--"
    for r in 1:nr
        @printf("%-11s %5s %6.3f %6.0f %8s %7s", Dates.format(stamps[r], "yyyy-mm-dd"),
                isfinite(ice[r]) ? @sprintf("%.2f", ice[r]) : "--", hypot(τx[r], τy[r]), Qtot[r],
                fmt(hb[r]), isfinite(mdiag[r]) ? @sprintf("%.0f", mdiag[r]) : "--")
        foreach(n -> @printf(" %10s", fmt(hm[n][r])), CLOSURES); println()
    end

    # ── budget: change since record 1 against the forcing between the two window centres ──
    fwd(x, r) = r == 1 ? 0.0 : 5 * 86400 * (x[1] / 2 + sum(x[2:r-1]; init = 0.0) + x[r] / 2)
    salt_in = SF .- dilute .* Sb[1, :] .* FW                                  # g m⁻² s⁻¹, as the column applies it
    HC(T) = ρ₀ * cₚ * sum(T .* Δz) / 1e6                            # MJ m⁻²
    SC(S) = ρ₀ * sum(S .* Δz) / 1e3                                 # kg salt m⁻²
    ref = first(CLOSURES)
    println("\nCOLUMN BUDGET since $(Dates.format(stamps[1], "yyyy-mm-dd")) (0-$(round(Int, H)) m). heat MJ/m², salt kg/m² ",
            "(1 kg/m² = $(@sprintf("%.4f", 1000 / (ρ₀ * H))) g/kg over the column)")
    @printf("%-11s | %8s %8s %8s %8s %9s | %8s %8s %8s %8s %9s\n", "window end", "B-SOSE", "forcing", "lateral",
            "applied", "$ref-chk", "B-SOSE", "forcing", "lateral", "applied", "$ref-chk")
    centre(r) = secs(stamps[r] - Hour(60))
    for r in 2:nr
        (Dates.month(stamps[r]) != Dates.month(stamps[r-1]) || r == nr) || continue
        dHb, dSb = HC(Tb[:, r]) - HC(Tb[:, 1]), SC(Sb[:, r]) - SC(Sb[:, 1])
        Fh, Fs = fwd(Qtot, r) / 1e6, fwd(salt_in, r) / 1e3
        Ah = ρ₀ * cₚ * sum(piecewise_integral(tl, LT, centre(1), centre(r)) .* Δz) / 1e6
        As = ρ₀ * sum(piecewise_integral(tl, LS, centre(1), centre(r)) .* Δz) / 1e3
        Tm, Sm = results[ref]
        dHm = all(isfinite, Tm[:, r]) ? HC(Tm[:, r]) - HC(Tm[:, 1]) : NaN
        dSm = all(isfinite, Sm[:, r]) ? SC(Sm[:, r]) - SC(Sm[:, 1]) : NaN
        @printf("%-11s | %8.1f %8.1f %8.1f %8.1f %9.1f | %8.3f %8.3f %8.3f %8.3f %9.3f\n",
                Dates.format(stamps[r], "yyyy-mm-dd"), dHb, Fh, dHb - Fh, Ah, dHm - Fh - Ah, dSb, Fs, dSb - Fs, As, dSm - Fs - As)
    end
    println("  lateral = B-SOSE's change minus its surface forcing (what a 1-D column lacks); applied = B-SOSE's monthly ",
            "supply the column\n  was given (LATERAL=$LATERAL); $ref-chk = the column's change minus forcing minus applied, ",
            "~0 (salt uses the column's own top S).")

    # ── end-state layer means against B-SOSE ──
    layers = ((0, 100), (100, 300), (300, H))
    lmean(x, (a, b)) = (m = [a <= -z < b for z in zc]; any(m) ? sum(x[m] .* Δz[m]) / sum(Δz[m]) : NaN)
    println("\nLAST WINDOW, layer means minus B-SOSE (T °C, S g/kg); B-SOSE row = its change since the start")
    @printf("%-12s", ""); foreach(l -> @printf(" %15s", @sprintf("ΔT %d-%d", l[1], l[2])), layers)
    foreach(l -> @printf(" %15s", @sprintf("ΔS %d-%d", l[1], l[2])), layers); println()
    @printf("%-12s", "B-SOSE")
    foreach(l -> @printf(" %15.3f", lmean(Tb[:, nr] .- Tb[:, 1], l)), layers)
    foreach(l -> @printf(" %15.4f", lmean(Sb[:, nr] .- Sb[:, 1], l)), layers); println()
    for n in CLOSURES
        Tm, Sm = results[n]
        rl = findlast(r -> all(isfinite, Tm[:, r]), 1:nr)
        @printf("%-12s", n * (rl == nr ? "" : "†"))
        foreach(l -> @printf(" %15.3f", lmean(Tm[:, rl] .- Tb[:, rl], l)), layers)
        foreach(l -> @printf(" %15.4f", lmean(Sm[:, rl] .- Sb[:, rl], l)), layers); println()
    end
    any(n -> !all(isfinite, results[n][1][:, nr]), CLOSURES) && println("  † stopped early: its last finite window")

    # ── compact summary, collected per box and printed again at the end of the run ──
    sm = String[]
    push!(sm, @sprintf("[%s] %.2f-%.2fE %.2f-%.2fS, %d B-SOSE cells, column %.0f m, LATERAL=%s", col.name, XC[first(I)],
                       XC[last(I)], -YC[last(J)], -YC[first(J)], length(I) * length(J), H, LATERAL))
    if bud !== nothing
        closure_res(t, n, sfc) = sum(t.tend[:, n]) - sum(t.hadv[:, n]) - sum(t.hdif[:, n]) - (t.R[nk+1, n] - t.R[1, n]) -
                                 (t.D[nk+1, n] - t.D[1, n]) - sfc
        sT, sS = ρB * cpB / bud.A[1], ρB * 86400 / bud.A[1]
        push!(sm, @sprintf("B-SOSE closure, worst month: heat |res| %.1f W/m² (|TOTTTEND| up to %.0f); salt |res| %.1f g/m²/day (|TOTSTEND| up to %.0f)",
                           maximum(n -> abs(sT * closure_res(bud.TH, n, bud.sfc["surfTflx"][n] / (ρB * cpB))), 1:nm),
                           maximum(n -> abs(sT * sum(bud.TH.tend[:, n])), 1:nm),
                           maximum(n -> abs(sS * closure_res(bud.SLT, n, bud.sfc["surfSflx"][n] / ρB)), 1:nm),
                           maximum(n -> abs(sS * sum(bud.SLT.tend[:, n])), 1:nm)))
    end
    ymc(r) = (c = stamps[r] - Hour(60); (Dates.year(c), Dates.month(c)))
    push!(sm, "MLD (m), monthly mean of the 5-day values; [n] = windows mixed to the column floor")
    push!(sm, @sprintf("  %-8s %5s %7s %6s", "month", "ice", "B-SOSE", "diag") * join(@sprintf(" %10s", n) for n in CLOSURES))
    for (y, m) in unique(ymc.(2:nr))
        rs = [r for r in 2:nr if ymc(r) == (y, m)]
        mm(v) = (x = filter(isfinite, first.(v[rs])); nfl = count(last.(v[rs]));
                 isempty(x) ? "--" : @sprintf("%.0f", mean(x)) * (nfl > 0 ? "[$nfl]" : ""))
        icem, dm = filter(isfinite, ice[rs]), filter(isfinite, mdiag[rs])
        push!(sm, @sprintf("  %04d-%02d %5s %7s %6s", y, m, isempty(icem) ? "--" : @sprintf("%.2f", mean(icem)), mm(hb),
                           isempty(dm) ? "--" : @sprintf("%.0f", mean(dm))) * join(@sprintf(" %10s", mm(hm[n])) for n in CLOSURES))
    end
    let r = nr
        dHb, dSb = HC(Tb[:, r]) - HC(Tb[:, 1]), SC(Sb[:, r]) - SC(Sb[:, 1])
        Fh, Fs = fwd(Qtot, r) / 1e6, fwd(salt_in, r) / 1e3
        Ah = ρ₀ * cₚ * sum(piecewise_integral(tl, LT, centre(1), centre(r)) .* Δz) / 1e6
        As = ρ₀ * sum(piecewise_integral(tl, LS, centre(1), centre(r)) .* Δz) / 1e3
        Tr, Sr = results[ref]
        push!(sm, @sprintf("column budget to %s: heat MJ/m² B-SOSE %.1f, forcing %.1f, lateral %.1f, applied %.1f, %s-chk %.1f",
                           Dates.format(stamps[r], "yyyy-mm-dd"), dHb, Fh, dHb - Fh, Ah, ref, HC(Tr[:, r]) - HC(Tr[:, 1]) - Fh - Ah))
        push!(sm, @sprintf("                              salt kg/m² B-SOSE %.2f, forcing %.2f, lateral %.2f, applied %.2f, %s-chk %.3f",
                           dSb, Fs, dSb - Fs, As, ref, SC(Sr[:, r]) - SC(Sr[:, 1]) - Fs - As))
    end
    push!(sm, "last window minus B-SOSE (B-SOSE row: its own change): ΔT °C 0-100 | 100-300 | 300-floor || ΔS g/kg same layers")
    lrow(T, S) = join(@sprintf(" %7.3f", lmean(T, l)) for l in layers) * " ||" * join(@sprintf(" %7.4f", lmean(S, l)) for l in layers)
    push!(sm, @sprintf("  %-11s", "B-SOSE") * lrow(Tb[:, nr] .- Tb[:, 1], Sb[:, nr] .- Sb[:, 1]))
    for n in CLOSURES
        Tr, Sr = results[n]
        rl = findlast(r -> all(isfinite, Tr[:, r]), 1:nr)
        push!(sm, @sprintf("  %-11s", n * (rl == nr ? "" : "†")) * lrow(Tr[:, rl] .- Tb[:, rl], Sr[:, rl] .- Sb[:, rl]))
    end

    # ── figure ──
    colors =Dict(zip(CLOSURES, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
    days = [Dates.value(s - stamps[1]) / 86_400_000 for s in stamps]
    fig = Figure(size = (1400, 1150))
    Label(fig[0, 1:4], @sprintf("%s: one column under B-SOSE's surface fluxes (%.1f-%.1fE, %.1f-%.1fS, 0-%.0f m)",
                                col.name, XC[first(I)], XC[last(I)], -YC[last(J)], -YC[first(J)], H),
          fontsize = 18, font = :bold)
    ticks = (days[[1; findall(r -> Dates.month(stamps[r]) != Dates.month(stamps[r-1]), 2:nr) .+ 1]],
             Dates.format.(stamps[[1; findall(r -> Dates.month(stamps[r]) != Dates.month(stamps[r-1]), 2:nr) .+ 1]], "u yyyy"))
    ax = Axis(fig[1, 1:4], ylabel = "MLD (m)", yreversed = true, xticks = ticks,
              title = "mixed layer depth of the 5-day-mean state (open circles: mixed to the column floor)")
    lines!(ax, days, first.(hb), color = :black, linewidth = 2.5, label = "B-SOSE")
    any(isfinite, mdiag) && lines!(ax, days, mdiag, color = :gray55, linewidth = 2, linestyle = :dash,
                                   label = "B-SOSE MLD diagnostic")
    for n in CLOSURES
        lines!(ax, days, first.(hm[n]), color = colors[n], linewidth = 2, label = n)
        deep = findall(last.(hm[n]))
        isempty(deep) || scatter!(ax, days[deep], first.(hm[n])[deep], color = :white, strokecolor = colors[n],
                                  strokewidth = 2, markersize = 9)
    end
    deepb = findall(last.(hb))
    isempty(deepb) || scatter!(ax, days[deepb], first.(hb)[deepb], color = :white, strokecolor = :black,
                               strokewidth = 2, markersize = 9)
    axislegend(ax, position = :lb, framevisible = false)
    axi = Axis(fig[2, 1:4], ylabel = "ice conc.", xticks = ticks, limits = (nothing, (0, 1)))
    any(isfinite, ice) ? lines!(axi, days, ice, color = :gray30, linewidth = 2) :
                         text!(axi, 0.5, 0.5, text = "no SIArea file", space = :relative, align = (:center, :center))
    rowsize!(fig.layout, 2, Relative(0.08))

    zlim = min(H, 500.0)
    picks = unique(round.(Int, range(2, nr, length = 4)))
    for (c, r) in enumerate(picks), (row, (X, lab)) in enumerate(((Sb, "S (g/kg)"), (Tb, "T (°C)")))
        a = Axis(fig[2 + row, c], xlabel = lab, ylabel = c == 1 ? "depth (m)" : "",
                 title = row == 1 ? Dates.format(stamps[r], "yyyy-mm-dd") : "", limits = (nothing, (-zlim, 0)))
        lines!(a, X[:, 1], zc, color = :gray55, linestyle = :dot, linewidth = 1.5)
        lines!(a, X[:, r], zc, color = :black, linewidth = 2.5)
        for n in CLOSURES
            Y = row == 1 ? results[n][2] : results[n][1]
            all(isfinite, Y[:, r]) && lines!(a, Y[:, r], zc, color = colors[n], linewidth = 2)
        end
    end
    Label(fig[5, 1:4], "Profiles: black B-SOSE, dotted grey the start ($(Dates.format(stamps[1], "yyyy-mm-dd"))), " *
                       "colours as above. B-SOSE also changes by lateral supply; see the printed budget.",
          fontsize = 12)
    mkpath(OUTDIR)
    out = joinpath(OUTDIR, "single_column_$(col.name)$(TAG).png")
    save(out, fig); @info "wrote $out"

    bud === nothing && return sm
    # ── vertical mixing flux: B-SOSE's DFrI against each scheme's ──
    # A scheme's flux on the top face of level k (positive up), over window r, is what its tendency needs beyond
    # the sources below that face: F(k) = F(k+1) + Δz (source - dT/dt), from F = 0 on the column floor. The top
    # face should come out ~0 (the surface flux is a source of level 1): printed as a check.
    frac(z) = SWR * exp(z / SWζ1) + (1 - SWR) * exp(z / SWζ2)
    swk = [frac(RF[k]) - (k == nk ? 0.0 : frac(RF[k+1])) for k in 1:nk]
    function scheme_flux(Xs, which, r)
        Δt = edges[r+1] - edges[r]
        L = piecewise_integral(tl, which === :T ? LT : LS, edges[r], edges[r+1]) ./ Δt
        src = copy(L)
        if which === :T
            src .+= Qsw[r] .* swk ./ (ρ₀ * cₚ .* Δz)
            src[1] += (Qtot[r] - Qsw[r]) / (ρ₀ * cₚ * Δz[1])
        else
            Stop = (Xs[1, r] + Xs[1, r+1]) / 2
            src[1] += (SF[r] - dilute * Stop * FW[r]) / (ρ₀ * Δz[1])
        end
        dXdt = (Xs[:, r+1] .- Xs[:, r]) ./ Δt
        F = zeros(nk + 1)
        for k in nk:-1:1
            F[k] = F[k+1] + Δz[k] * (src[k] - dXdt[k])
        end
        return F[1:nk]
    end
    inmonth(r, n) = (c = stamps[r] - Hour(60); (Dates.year(c), Dates.month(c)) == months[n])   # window centre in month n
    shown = [n for n in 1:nm if count(r -> r >= 2 && inmonth(r, n), 1:nr) >= 3][1:min(end, 6)]
    isempty(shown) && return sm
    zface = RF[1:nk]                                                            # top faces, top down
    Hscale, Sscale = ρ₀ * cₚ, ρ₀ * 86400                                        # → W/m², g m⁻² day⁻¹
    println("\nVERTICAL MIXING FLUX, column mean over 0-$(round(Int, min(H, 300))) m, W/m² (heat) and g m⁻² day⁻¹ (salt), + = up")
    @printf("  %-8s %10s", "month", "B-SOSE")
    foreach(n -> @printf(" %10s", n), CLOSURES); @printf(" | %10s", "B-SOSE"); foreach(n -> @printf(" %10s", n), CLOSURES)
    println("   (top-face check, W/m²)")
    flux = Dict{Tuple{String, Symbol, Int}, Vector{Float64}}()
    for n in shown, nme in CLOSURES, which in (:T, :S)
        Xs = which === :T ? results[nme][3] : results[nme][4]
        rs = [r for r in 2:nr if inmonth(r, n) && all(isfinite, Xs[:, r+1])]
        flux[(nme, which, n)] = isempty(rs) ? fill(NaN, nk) : mean(scheme_flux(Xs, which, r) for r in rs)
    end
    upper = [2 <= k <= nk && -zface[k] <= 300 for k in 1:nk]
    bsose_flux(which, n) = [k == 1 ? 0.0 : (which === :T ? bud.TH : bud.SLT).D[k, n] / bud.A[k] for k in 1:nk]
    push!(sm, "vertical mixing flux, mean over faces 0-300 m, + = up: heat W/m² | salt g m⁻² day⁻¹")
    push!(sm, @sprintf("  %-8s %8s", "month", "B-SOSE") * join(@sprintf(" %8s", n) for n in CLOSURES) *
              @sprintf(" | %8s", "B-SOSE") * join(@sprintf(" %8s", n) for n in CLOSURES))
    for n in shown
        @printf("  %-8s %10.1f", mlabel(n), Hscale * mean(bsose_flux(:T, n)[upper]))
        foreach(nme -> @printf(" %10.1f", Hscale * mean(flux[(nme, :T, n)][upper])), CLOSURES)
        @printf(" | %10.2f", Sscale * mean(bsose_flux(:S, n)[upper]))
        foreach(nme -> @printf(" %10.2f", Sscale * mean(flux[(nme, :S, n)][upper])), CLOSURES)
        @printf("   (%s)\n", join([@sprintf("%.1f", Hscale * flux[(nme, :T, n)][1]) for nme in CLOSURES], ", "))
        push!(sm, @sprintf("  %-8s %8.1f", mlabel(n), Hscale * mean(bsose_flux(:T, n)[upper])) *
                  join(@sprintf(" %8.1f", Hscale * mean(flux[(nme, :T, n)][upper])) for nme in CLOSURES) *
                  @sprintf(" | %8.2f", Sscale * mean(bsose_flux(:S, n)[upper])) *
                  join(@sprintf(" %8.2f", Sscale * mean(flux[(nme, :S, n)][upper])) for nme in CLOSURES))
    end

    # ── implied vertical diffusivity BELOW the mixed layer: κ = -F / (∂X/∂z) ──
    # F the mixing flux on a level's top face (positive up), ∂X/∂z the monthly-mean gradient across that face (z up).
    # Faces from max(B-SOSE's monthly MLD + 50 m, 200 m) to min(600 m, floor), where the gradient is not tiny; the
    # median over those faces. B-SOSE's from its DFrI and its own profile; each scheme's from its diagnosed flux and
    # its own profile. Our backgrounds are κz = KAPPA_Z (catke, ri); catke_bare and kepsilon have none.
    gmin = (S = 2e-4, T = 1e-3)                                  # g/kg per m, °C per m
    function implied_kappa(F, X, hml, which)
        ks = Float64[]
        for k in 2:nk
            d = -RF[k]
            (max(hml + 50, 200.0) <= d <= min(600.0, H)) || continue
            g = (X[k-1] - X[k]) / (zc[k-1] - zc[k])
            abs(g) < (which === :S ? gmin.S : gmin.T) && continue
            push!(ks, -F[k] / g)
        end
        return isempty(ks) ? (NaN, 0) : (median(ks), length(ks))
    end
    fk(x) = isfinite(x[1]) ? @sprintf("%.1f", 1e5 * x[1]) : "--"
    println("\nIMPLIED κz BELOW THE MIXED LAYER (1e-5 m²/s; median over faces from max(MLD+50, 200) m to 600 m) " *
            "-- from salt | from heat. Our background is $(round(KAPPA_Z / 1e-5, digits = 1)) ($(KAPPA_Z)).")
    hdr = @sprintf("  %-8s %10s", "month", "B-SOSE(n)") * join(@sprintf(" %10s", n) for n in CLOSURES)
    println(hdr, " |", hdr[11:end])
    push!(sm, "implied κz below the mixed layer, 1e-5 m²/s (median, faces max(MLD+50,200)-600 m): salt | heat; ours = $(round(KAPPA_Z / 1e-5, digits = 1))")
    push!(sm, hdr * " |" * hdr[11:end])
    for n in shown
        rs = [r for r in 2:nr if inmonth(r, n)]
        hml = mean(first.(hb[rs]))
        row = @sprintf("  %-8s", mlabel(n))
        for (which, Xb, idx) in ((:S, Sb, 2), (:T, Tb, 1))
            kb = implied_kappa(bsose_flux(which, n), vec(mean(Xb[:, rs], dims = 2)), hml, which)
            which === :T && (row *= " |")
            row *= @sprintf(" %10s", fk(kb) * "($(kb[2]))")
            for nme in CLOSURES
                Xm = results[nme][idx]
                ok = [r for r in rs if all(isfinite, Xm[:, r])]
                row *= @sprintf(" %10s", isempty(ok) ? "--" :
                                fk(implied_kappa(flux[(nme, which, n)], vec(mean(Xm[:, ok], dims = 2)), hml, which)))
            end
        end
        println(row); push!(sm, row)
    end

    zlim = min(H, 600.0)
    figf = Figure(size = (300 * length(shown) + 120, 900))
    Label(figf[0, 1:length(shown)], "$(col.name): vertical mixing flux, B-SOSE's DFrI against each scheme " *
                                    "(monthly mean, positive up)", fontsize = 18, font = :bold)
    for (c, n) in enumerate(shown), (row, which, lab, sc) in ((1, :T, "heat (W/m²)", Hscale), (2, :S, "salt (g m⁻² day⁻¹)", Sscale))
        a = Axis(figf[row, c], xlabel = lab, ylabel = c == 1 ? "depth (m)" : "", title = row == 1 ? mlabel(n) : "",
                 limits = (nothing, (-zlim, 0)))
        vlines!(a, 0, color = :gray70, linewidth = 1)
        lines!(a, sc .* bsose_flux(which, n)[2:nk], zface[2:nk], color = :black, linewidth = 2.5,
               label = "B-SOSE DFrI")
        for nme in CLOSURES
            F = flux[(nme, which, n)]
            all(isfinite, F) && lines!(a, sc .* F[2:nk], zface[2:nk], color = colors[nme], linewidth = 2, label = nme)
        end
        row == 1 && c == 1 && axislegend(a, position = :rb, framevisible = false)
    end
    outf = joinpath(OUTDIR, "single_column_$(col.name)$(TAG)_mixing_flux.png")
    save(outf, figf); @info "wrote $outf"
    return sm
end

function main()
    years = Dates.year(START - WINDOW):Dates.year(STOP)
    recs = Dict(t => records(t, years) for t in keys(TOKENS))
    missing_tokens = [t for t in REQUIRED if isempty(recs[t])]
    isempty(missing_tokens) || error("no 5-day files for $(join(missing_tokens, ", ")) in $BSOSE_DIR for $years")
    stamps = sort([s for s in keys(recs["Theta"]) if START <= s <= STOP && all(t -> haskey(recs[t], s), REQUIRED)])
    length(stamps) < 3 && error("fewer than 3 5-day records with every required field between $START and $STOP")
    gaps = diff(stamps)
    all(==(Millisecond(WINDOW)), gaps) || @warn "records are not every 5 days between $START and $STOP: $(unique(gaps))"
    XC, YC, RF = NCDataset(recs["Theta"][stamps[1]][1]) do ds
        rf = haskey(ds, "RF") ? f64(ds["RF"][:]) : [0.0; -cumsum(f64(ds[haskey(ds, "drF") ? "drF" : "DRF"][:]))]
        coord(ds, "XC", 1), coord(ds, "YC", 2), rf
    end
    @info "$(length(stamps)) 5-day records, $(Dates.format(stamps[1], "yyyy-mm-dd")) to " *
          "$(Dates.format(stamps[end], "yyyy-mm-dd")); schemes $(join(CLOSURES, ", ")); Δt $(DT) s; MLD_DT $MLD_DT"
    summary = String[]
    for col in COLUMNS
        append!(summary, do_column(col, recs, stamps, XC, YC, RF)); push!(summary, "")
    end
    commit = try readchomp(`git -C $(@__DIR__) rev-parse --short HEAD`) catch; "?" end
    println("\n", "#"^34, " SUMMARY: copy from here ", "#"^34)
    println("single_column_bsose $commit | ", Dates.format(stamps[1], "yyyy-mm-dd"), " to ", Dates.format(stamps[end], "yyyy-mm-dd"),
            " | LATERAL=$LATERAL | STRESS=$STRESS | KAPPA_Z=$KAPPA_Z | ZMAX=$ZMAX | DT=$DT | MLD_DT=$MLD_DT")
    foreach(println, summary)
    println("#"^36, " END SUMMARY ", "#"^45)
end

main()
