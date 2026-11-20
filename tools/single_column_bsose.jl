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
#               salt   SFLUX - S * oceFWflx (virtual salt flux, S the column's own top cell)
#               stress oceTAUX, oceTAUY (under ice: B-SOSE's ice-ocean + air-ocean stress)
#             as the run: flux = -X / rho0 (tracers also / cp), rho0 1020, cp 3991.868, TEOS-10
#   Coriolis  f-plane at the box's centre latitude
#   schemes   CLOSURES, from
#               catke       the run's "ggl": NumericalEarth's CATKE (Cᵇ 0.01, Cᵂϵ 1.0) + νz 1e-3, κz 1e-4
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
#
# WRITES OUTDIR/single_column_<column>.png: MLD against time (B-SOSE, B-SOSE's diagnostic, each scheme),
# B-SOSE's ice concentration, and S and T profiles at four times.
#
# ENV
#   BSOSE_DIR  (required) B-SOSE's 5-day files: Theta, Salt, surfTflx, surfSflx, oceFWflx, oceQsw, oceTAUX,
#              oceTAUY (required); SIArea, MLD (optional)
#   BOXES      name:lon0:lon1:lat0:lat1,...  (default ice65S:119.4:120.4:-65.0:-64.5,open52S:119.4:120.4:-53.0:-52.5)
#   START, END first and last 5-day record stamps used (default 2014-06-01, 2014-11-01)
#   CLOSURES   (default catke,catke_bare,kepsilon,ri)
#   ZMAX       deepest level, m (default 1000)    DT  time step, s (default 600)
#   OUTDIR     (default plots)                    MLD_DT (default 0.03 kg/m³, read by mld.jl)
# ==============================================================================================

using Oceananigans
using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities: CATKEMixingLength, CATKEEquation
using SeawaterPolynomials.TEOS10: TEOS10EquationOfState
using NCDatasets, CairoMakie, Dates, Printf, Statistics

ENV["HDF5_USE_FILE_LOCKING"] = "FALSE"
include(joinpath(@__DIR__, "mld.jl"))          # mld, MLD_DT

const BSOSE_DIR = get(ENV, "BSOSE_DIR", "")
isempty(BSOSE_DIR) && error("set BSOSE_DIR to the directory with B-SOSE's 5-day files")
const OUTDIR   = get(ENV, "OUTDIR", "plots")
const START    = DateTime(get(ENV, "START", "2014-06-01"))
const STOP     = DateTime(get(ENV, "END", "2014-11-01"))
const ZMAX     = parse(Float64, get(ENV, "ZMAX", "1000"))
const DT       = parse(Float64, get(ENV, "DT", "600"))
const CLOSURES = String.(split(get(ENV, "CLOSURES", "catke,catke_bare,kepsilon,ri"), ','))
const ρ₀, cₚ   = 1020.0, 3991.86795711963      # the run's (NumericalEarth's) reference density, heat capacity
const SWR, SWζ1, SWζ2 = 0.62, 0.6, 20.0        # MITgcm swfrac, jwtype 2
const WINDOW   = Day(5)

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

# ── single-column model pieces ────────────────────────────────────────────────────────────────────
@inline window_index(p, t) = clamp(searchsortedlast(p.ts, t), 1, length(p.ts))
@inline Jᵀ(i, j, grid, clock, f, p) = (n = window_index(p, clock.time); -(p.Qtot[n] - p.Qsw[n]) / (p.ρ₀ * p.cₚ))
@inline Jˢ(i, j, grid, clock, f, p) = (n = window_index(p, clock.time);
                                       -(p.SF[n] - @inbounds(f.S[i, j, grid.Nz]) * p.FW[n]) / p.ρ₀)
@inline Jᵘ(i, j, grid, clock, f, p) = (n = window_index(p, clock.time); -p.τx[n] / p.ρ₀)
@inline Jᵛ(i, j, grid, clock, f, p) = (n = window_index(p, clock.time); -p.τy[n] / p.ρ₀)
@inline swfrac(z, p) = p.R * exp(z / p.ζ1) + (1 - p.R) * exp(z / p.ζ2)
@inline function shortwave_heating(i, j, k, grid, clock, f, p)
    n = window_index(p, clock.time)
    @inbounds ztop, zbot = p.zf[k+1], p.zf[k]
    below = k == 1 ? zero(ztop) : swfrac(zbot, p)          # the bottom cell keeps what reaches it
    return p.Qsw[n] * (swfrac(ztop, p) - below) / (p.ρ₀ * p.cₚ * (ztop - zbot))
end

function build_closure(name)
    background = VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(); ν = 1e-3, κ = 1e-4)
    catke() = CATKEVerticalDiffusivity(VerticallyImplicitTimeDiscretization(), Float64;
                                       mixing_length = CATKEMixingLength(Cᵇ = 0.01),
                                       turbulent_kinetic_energy_equation = CATKEEquation(Cᵂϵ = 1.0))
    name == "catke"      && return (catke(), background)
    name == "catke_bare" && return catke()
    name == "kepsilon"   && return TKEDissipationVerticalDiffusivity()     # alone: its TKE top BC rejects a closure tuple (0.117.2)
    name == "ri"         && return (RiBasedVerticalDiffusivity(), background)
    error("unknown closure '$name' (catke, catke_bare, kepsilon, ri)")
end

"""Run one scheme; return its window-mean T and S (top-down, nk × nr; record 1 is the start)."""
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
                forcing = (; T = Forcing(shortwave_heating; discrete_form = true, parameters = p)))
    set!(model, T = reshape(reverse(Tinit), 1, 1, nk), S = reshape(reverse(Sinit), 1, 1, nk))
    haskey(model.tracers, :e) && set!(model, e = 1e-6)
    haskey(model.tracers, :ϵ) && set!(model, ϵ = 1e-9)

    Tm, Sm = fill(NaN, nk, nr), fill(NaN, nk, nr)
    Tm[:, 1], Sm[:, 1] = Tinit, Sinit
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
    for r in 2:nr
        accT .= 0; accS .= 0; count[] = 0
        sim.stop_time = edges[r+1]
        run!(sim)
        if count[] == 0 || !all(isfinite, accT) || !all(isfinite, accS)
            @warn "$name: non-finite T or S (or no samples) in window $r; stopping this scheme there"
            break
        end
        Tm[:, r], Sm[:, r] = accT ./ count[], accS ./ count[]
    end
    @info @sprintf("%s: %d steps, %.0f s wall", name, iteration(sim), time() - t0)
    return Tm, Sm
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

    # model time 0 = centre of record 1; window r is (stamp_r - 5 d, stamp_r]
    t0 = stamps[1] - Hour(60)                     # half of the 5-day window
    secs(d) = Dates.value(d - t0) / 1000
    ts = [secs(s - WINDOW) for s in stamps]                 # forcing window starts (ts[1] < 0)
    edges = [0.0; [secs(s) for s in stamps]]                # run 0 -> end of window 1, then window by window
    zf = reverse(RF[1:nk+1]) .+ 0.0                         # bottom to top
    p = (; ts, Qtot, Qsw, SF, FW, τx, τy, zf, ρ₀, cₚ, R = SWR, ζ1 = SWζ1, ζ2 = SWζ2)

    results = Dict{String, Tuple{Matrix{Float64}, Matrix{Float64}}}()
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
    salt_in = SF .- Sb[1, :] .* FW                                  # g m⁻² s⁻¹, B-SOSE's top-cell S
    HC(T) = ρ₀ * cₚ * sum(T .* Δz) / 1e6                            # MJ m⁻²
    SC(S) = ρ₀ * sum(S .* Δz) / 1e3                                 # kg salt m⁻²
    ref = first(CLOSURES)
    println("\nCOLUMN BUDGET since $(Dates.format(stamps[1], "yyyy-mm-dd")) (0-$(round(Int, H)) m). heat MJ/m², salt kg/m² ",
            "(1 kg/m² = $(@sprintf("%.4f", 1000 / (ρ₀ * H))) g/kg over the column)")
    @printf("%-11s | %8s %8s %8s %9s | %8s %8s %8s %9s\n", "window end", "B-SOSE", "forcing", "lateral",
            "$ref-frc", "B-SOSE", "forcing", "lateral", "$ref-frc")
    for r in 2:nr
        (Dates.month(stamps[r]) != Dates.month(stamps[r-1]) || r == nr) || continue
        dHb, dSb = HC(Tb[:, r]) - HC(Tb[:, 1]), SC(Sb[:, r]) - SC(Sb[:, 1])
        Fh, Fs = fwd(Qtot, r) / 1e6, fwd(salt_in, r) / 1e3
        Tm, Sm = results[ref]
        dHm = all(isfinite, Tm[:, r]) ? HC(Tm[:, r]) - HC(Tm[:, 1]) : NaN
        dSm = all(isfinite, Sm[:, r]) ? SC(Sm[:, r]) - SC(Sm[:, 1]) : NaN
        @printf("%-11s | %8.1f %8.1f %8.1f %9.1f | %8.3f %8.3f %8.3f %9.3f\n", Dates.format(stamps[r], "yyyy-mm-dd"),
                dHb, Fh, dHb - Fh, dHm - Fh, dSb, Fs, dSb - Fs, dSm - Fs)
    end
    println("  lateral = B-SOSE's change minus its surface forcing: what a 1-D column cannot have. ",
            "$ref-frc should be ~0 (the column conserves; salt uses the model's own top S, so not exactly).")

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

    # ── figure ──
    colors = Dict(zip(CLOSURES, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
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
    out = joinpath(OUTDIR, "single_column_$(col.name).png")
    save(out, fig); @info "wrote $out"
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
    for col in COLUMNS
        do_column(col, recs, stamps, XC, YC, RF)
    end
end

main()
