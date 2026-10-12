# ==============================================================================
# inspect_bsose_boundaries.jl
#
# What does BSOSE iteration 156 actually prescribe at the four open boundaries?
#
# Every transport number quoted in the OBC work so far came from one of two places: BSOSE
# iteration *105* (a different solution), or the model's own iteration-0 diagnostic (which
# has already been through regridding, inpainting and remasking onto an ETOPO bathymetry
# that is not BSOSE's). Neither is the prescribed boundary data. This script reads the
# iteration 156 files directly and answers seven questions, in priority order:
#
#   Q1  With `BAROTROPIC_FACES = :zonal`, west and east are forced to balance each other.
#       What is BSOSE's own west+east net? That number IS the distortion :zonal imposes.
#       A few Sv and :zonal is nearly free; 20+ Sv and it is bending the ACC.
#   Q2  Is the north face really -2.6 Sv in the mean, ranging -31 to +16 Sv? That is what
#       :zonal gives up, and it was quoted from the model diagnostic, not from BSOSE.
#   Q3  How peaked is Uᵉˣᵗ? Feeds the sandbox JET rung, and distinguishes "the BC is wrong"
#       from "a 1/6 deg grid cannot hold this jet".
#   Q4  Is the prescribed inflow temperature warm in the 60-65S band where the bias lives?
#       The bias was diagnosed as redistribution, not import -- this is the check on that.
#   Q5  Is the prescribed boundary salinity consistent with the +0.5-0.7 PSU model bias?
#   Q6  Where do BSOSE's land mask and the model's ETOPO bathymetry disagree along the
#       boundaries? This is the mechanistic source of the ~+3 Sv residual the balance
#       routine removes -- worth knowing how big it is before trusting the balance.
#   Q7  Sanity: NaNs in the raw slices, and does BSOSE carry nonzero velocity below its
#       own sea floor (the inpainting trap `bsose_barotropic_transport` warns about)?
#   Q8  How far does BSOSE's sea surface sit from the domain mean at each face, and what is
#       that worth in Sv? Flather prescribes the Riemann invariant, so an error in
#       (ηᵇ - ηᵉˣᵗ) is transport at √(gH) x face width. This is the question that turned
#       out to matter: with EXTERIOR_SSH = false every face is told ηᵉˣᵗ = 0.
#
# Runs on CPU only and needs no GPU -- it is NetCDF reads and column integrals. CPU is
# deliberate: this codebase has already lost two debugging cycles to GPU scalar indexing
# inside @info, and nothing here benefits from a device. Run it on a compute or login node
# wherever the iteration 156 data lives.
#
#     julia --project=. tools/inspect_bsose_boundaries.jl
#
# Environment knobs:
#     BSOSE_DIR     where the iteration 156 files are (default: the usual search path)
#     ITERATION     BSOSE iteration (156); 105 reads the older solution for comparison
#     START, END    run window, ISO dates (2014-06-01, 2015-02-01)
#     ALL_MONTHS=1  ignore START/END and report every month in the files
#     MODEL_MASK=0  skip Q6; avoids building the grid and calling regrid_bathymetry
#     BAND          latitude band for the T/S focus, "lo,hi" (-65,-60)
#     FIGURES=0     skip plotting
#     OUTDIR        where figures and the CSV go (tools/bsose_boundary_report)
#
# Writes: a printed report, <OUTDIR>/face_transports.csv, and four figures.
# ==============================================================================

using NCDatasets
using Printf
using Statistics
using Dates

include(joinpath(@__DIR__, "..", "src", "setup_bsose.jl"))

const R_EARTH = 6371e3

getenvf(k, d) = haskey(ENV, k) ? parse(Float64, ENV[k]) : d
getenvi(k, d) = haskey(ENV, k) ? parse(Int, ENV[k]) : d

# Model domain, matching src/model.jl.
const λ₁, λ₂ = 90.0, 150.0
const φ₁, φ₂ = -70.0, -45.0

const START_DATE = DateTime(get(ENV, "START", "2014-06-01"))
const END_DATE   = DateTime(get(ENV, "END",   "2015-02-01"))
const ALL_MONTHS = getenvi("ALL_MONTHS", 0) == 1
const DO_MODEL_MASK = getenvi("MODEL_MASK", 1) == 1
const DO_FIGURES = getenvi("FIGURES", 1) == 1
const OUTDIR = get(ENV, "OUTDIR", joinpath(@__DIR__, "bsose_boundary_report"))
const BAND = let s = split(get(ENV, "BAND", "-65,-60"), ","); (parse(Float64, s[1]), parse(Float64, s[2])) end

mkpath(OUTDIR)

hdr(s) = (println(); println("="^78); println(s); println("="^78))

# `mean` has no `init`, and an all-land face is a real case here: the southern boundary has
# zero wet columns, which is what poisoned the free-surface datum earlier in this work.
safemean(v) = (w = filter(isfinite, v); isempty(w) ? NaN : mean(w))

# ------------------------------------------------------------------------------
# Geometry. `bsose_geometry` already works around the corrupt XG/YG in the
# iteration 156 grid.nc by rebuilding the longitude faces from the centers.
# ------------------------------------------------------------------------------
dataset = BSOSEMonthly(iteration = getenvi("ITERATION", 156))
@info "BSOSE directory: $(dataset.dir)"

geom = bsose_geometry(dataset)
λc, λf, φc, φf = geom.λc, geom.λf, geom.φc, geom.φf

grid_file = bsose_grid_file(dataset)
grid_file === nothing && error("No grid.nc found in $(dataset.dir); hFac masks are required.")

gds = Dataset(grid_file)
RF = Float64.(gds["RF"][:])                 # 0 at surface, descending
drF = abs.(diff(RF))                        # k = 1 is the top cell
zc = 0.5 .* (RF[1:end-1] .+ RF[2:end])
Nz_b = length(drF)
Depth_full = Float64.(coalesce.(gds["Depth"][:, :], 0.0))

# Index ranges of the model domain in BSOSE's grid. Cell `i` spans [λf[i], λf[i+1]), so the
# domain's western boundary is the western face of the first cell inside it, and its eastern
# boundary is the western face of the first cell outside -- standard C-grid staggering, and
# the easiest thing to get off by one.
i₁ = findfirst(>=(λ₁), λf);  i₂ = findlast(<(λ₂), λf)
j₁ = findfirst(>=(φ₁), φf);  j₂ = findlast(<(φ₂), φf)
iW, iE = i₁, i₂ + 1
jS, jN = j₁, j₂ + 1
irange, jrange = i₁:i₂, j₁:j₂

@info @sprintf("domain cells i %d:%d (%.4f to %.4f E), j %d:%d (%.4f to %.4f N)",
               i₁, i₂, λf[i₁], λf[i₂+1], j₁, j₂, φf[j₁], φf[j₂+1])
@info @sprintf("faces: west i=%d (%.4f E), east i=%d (%.4f E), south j=%d (%.4f N), north j=%d (%.4f N)",
               iW, λf[iW], iE, λf[iE], jS, φf[jS], jN, φf[jN])

# Face widths. dxG/dyG/rA are all absent from this grid.nc, so they come from the
# coordinates: a meridional face has width R*Δφ, a zonal face R*cos(φ)*Δλ.
dφ = diff(φf)
dλ = diff(λf)
dy_col = [R_EARTH * deg2rad(dφ[j]) for j in jrange]                                  # west/east
dx_col_S = [R_EARTH * cosd(φf[jS]) * deg2rad(dλ[i]) for i in irange]                 # south
dx_col_N = [R_EARTH * cosd(φf[jN]) * deg2rad(dλ[i]) for i in irange]                 # north

# hFac at the faces. hFacW[i] is the western face of cell i, hFacS[j] the southern face of
# cell j, so these indices line up with the velocity points by construction.
hW = Float64.(coalesce.(gds["hFacW"][iW, jrange, :], 0.0))
hE = Float64.(coalesce.(gds["hFacW"][iE, jrange, :], 0.0))
hS = Float64.(coalesce.(gds["hFacS"][irange, jS, :], 0.0))
hN = Float64.(coalesce.(gds["hFacS"][irange, jN, :], 0.0))
hC_W = Float64.(coalesce.(gds["hFacC"][i₁, jrange, :], 0.0))    # for T/S at the west column
close(gds)

wet_depth(h) = [sum(h[n, k] * drF[k] for k in 1:Nz_b) for n in 1:size(h, 1)]
HW, HE, HS, HN = wet_depth(hW), wet_depth(hE), wet_depth(hS), wet_depth(hN)

area(H, w) = sum(H .* w)
AW, AE = area(HW, dy_col), area(HE, dy_col)
AS, AN = area(HS, dx_col_S), area(HN, dx_col_N)

hdr("Face geometry, from BSOSE's own hFac mask")
@printf("%-7s %5d wet cols  mean H %7.1f m  max H %7.1f m  area %10.3e m2\n",
        "west", count(>(0), HW), safemean(HW[HW.>0]), maximum(HW), AW)
@printf("%-7s %5d wet cols  mean H %7.1f m  max H %7.1f m  area %10.3e m2\n",
        "east", count(>(0), HE), safemean(HE[HE.>0]), maximum(HE), AE)
@printf("%-7s %5d wet cols  mean H %7.1f m  max H %7.1f m  area %10.3e m2\n",
        "south", count(>(0), HS), safemean(HS[HS.>0]), maximum(HS), AS)
@printf("%-7s %5d wet cols  mean H %7.1f m  max H %7.1f m  area %10.3e m2\n",
        "north", count(>(0), HN), safemean(HN[HN.>0]), maximum(HN), AN)
@printf("\nwest+east area %.3e m2,  north+south area %.3e m2  (ratio %.2fx)\n",
        AW + AE, AS + AN, (AS + AN) / (AW + AE))

# ------------------------------------------------------------------------------
# Time selection.
# ------------------------------------------------------------------------------
# `resolve_bsose_filename` falls back through a list of candidate patterns, so a missing
# iteration 156 file resolves to the iteration 105 one sitting next to it and the script
# reports 2008-2012 data under a 156 heading. That is exactly the kind of silent
# substitution this whole exercise is trying to escape, so check the filename.
const ALLOW_MIX = getenvi("ALLOW_ITERATION_MISMATCH", 0) == 1

function open_var(key)
    name = BSOSE_VARIABLE_NAMES[key]
    fname = resolve_bsose_filename(dataset.dir, key, dataset.iteration)
    path = joinpath(dataset.dir, fname)
    isfile(path) || error("missing file for $key: $path")
    if !occursin(string(dataset.iteration), basename(path))
        msg = "$key resolved to $(basename(path)), which does not look like iteration " *
              "$(dataset.iteration). Reporting this as i$(dataset.iteration) data would be wrong."
        ALLOW_MIX ? @warn(msg) : error(msg * " Set ALLOW_ITERATION_MISMATCH=1 to override.")
    end
    return Dataset(path), name
end

uds, uname = open_var(:u_velocity)
times = uds["time"][:]
tsel = ALL_MONTHS ? collect(eachindex(times)) :
       findall(t -> START_DATE <= t <= END_DATE, times)
isempty(tsel) && error("no records in [$START_DATE, $END_DATE]; file spans $(times[1]) to $(times[end])")
@info @sprintf("%d monthly records selected, %s to %s", length(tsel), times[tsel[1]], times[tsel[end]])

# ------------------------------------------------------------------------------
# Q1, Q2, Q7 -- face transports, month by month.
#
# Transport = sum over the face of velocity * hFac * drF * width. hFac is the fractional
# open height of the cell, so it must multiply drF rather than being thresholded to a
# boolean: thresholding a 0.1 partial cell to 1.0 inflates that cell's contribution tenfold.
# ------------------------------------------------------------------------------
vds, vname = open_var(:v_velocity)

nt = length(tsel)
Tr = Dict(s => zeros(nt) for s in (:west, :east, :south, :north))
colTr = Dict(s => Float64[] for s in (:west, :east, :south, :north))
nan_count = Dict(s => 0 for s in (:west, :east, :south, :north))
dry_signal = Dict(s => 0.0 for s in (:west, :east, :south, :north))
peak = Dict(s => (0.0, 0.0, 0.0) for s in (:west, :east, :south, :north))

specs = (
    (:west,  uds, uname, iW, jrange, hW, dy_col,   :y),
    (:east,  uds, uname, iE, jrange, hE, dy_col,   :y),
    (:south, vds, vname, jS, irange, hS, dx_col_S, :x),
    (:north, vds, vname, jN, irange, hN, dx_col_N, :x),
)

u_sections = Dict{Symbol,Matrix{Float64}}()   # for figures: column x depth, time mean

for (side, ds, name, fixed, rng, h, widths, along) in specs
    N = length(rng)
    sec = zeros(N, Nz_b)
    for (ti, t) in enumerate(tsel)
        raw = along === :x ? ds[name][rng, fixed, :, t] : ds[name][fixed, rng, :, t]
        V = Float64.(coalesce.(reshape(raw, N, Nz_b), NaN))
        nan_count[side] += count(isnan, V)

        # Velocity sitting in a cell BSOSE itself calls land. BSOSE writes 0.0 there, so a
        # nonzero value means the file has been inpainted, and integrating it is how a
        # boundary comes to demand metres per second of depth-mean inflow.
        dry_signal[side] += sum(abs, filter(isfinite, V[h .== 0.0]))

        Vc = replace(V, NaN => 0.0)
        colT = [sum(Vc[n, k] * h[n, k] * drF[k] for k in 1:Nz_b) * widths[n] for n in 1:N]
        Tr[side][ti] = sum(colT) / 1e6
        ti == 1 && (colTr[side] = colT ./ 1e6)
        sec .+= Vc .* (h .> 0)

        # Peakedness: the largest depth-mean speed on the face against the face mean.
        H = wet_depth(h)
        dmean = [H[n] > 0 ? sum(Vc[n, k] * h[n, k] * drF[k] for k in 1:Nz_b) / H[n] : 0.0 for n in 1:N]
        mx = maximum(abs, dmean)
        fm = abs(Tr[side][ti]) * 1e6 / max(area(H, widths), eps())
        peak[side] = max(peak[side], (mx, fm, fm > 0 ? mx / fm : 0.0))
    end
    u_sections[side] = sec ./ nt
end

hdr("Q1/Q2 -- prescribed transport through each face, monthly (Sv, positive = +x/+y)")
println(" date            west      east     south     north |  4-face net  W/E-only net")
println("-"^78)
net4 = zeros(nt); netWE = zeros(nt)
for ti in 1:nt
    w, e, s, n = Tr[:west][ti], Tr[:east][ti], Tr[:south][ti], Tr[:north][ti]
    net4[ti]  = -w + e - s + n          # outward-positive sum over all four faces
    netWE[ti] = -w + e                  # what :zonal must force to zero
    @printf(" %s  %+8.2f  %+8.2f  %+8.2f  %+8.2f |  %+9.3f   %+9.3f\n",
            Dates.format(times[tsel[ti]], "yyyy-mm-dd"), w, e, s, n, net4[ti], netWE[ti])
end
println("-"^78)
@printf(" mean          %+8.2f  %+8.2f  %+8.2f  %+8.2f |  %+9.3f   %+9.3f\n",
        mean(Tr[:west]), mean(Tr[:east]), mean(Tr[:south]), mean(Tr[:north]), mean(net4), mean(netWE))
@printf(" min           %+8.2f  %+8.2f  %+8.2f  %+8.2f |  %+9.3f   %+9.3f\n",
        minimum(Tr[:west]), minimum(Tr[:east]), minimum(Tr[:south]), minimum(Tr[:north]),
        minimum(net4), minimum(netWE))
@printf(" max           %+8.2f  %+8.2f  %+8.2f  %+8.2f |  %+9.3f   %+9.3f\n",
        maximum(Tr[:west]), maximum(Tr[:east]), maximum(Tr[:south]), maximum(Tr[:north]),
        maximum(net4), maximum(netWE))

hdr("Q1 -- the verdict on BAROTROPIC_FACES = :zonal")
@printf("""
BSOSE's own west+east net is %+.3f Sv in the mean, ranging %+.3f to %+.3f Sv.
:zonal forces this to zero, so that range is the distortion it imposes on the ACC.
As a uniform velocity over the west+east area (%.3e m2) the correction is %.2e m/s,
against a face-mean speed of %.3e m/s -- that is %.2f%% of the signal.

The north face carries %+.2f Sv in the mean (%.1f%% of the %.1f Sv gross zonal
transport), ranging %+.2f to %+.2f Sv. That is what :zonal gives up.
""",
mean(netWE), minimum(netWE), maximum(netWE),
AW + AE, maximum(abs.(netWE)) * 1e6 / (AW + AE),
mean(abs.(Tr[:west])) * 1e6 / AW,
100 * (maximum(abs.(netWE)) * 1e6 / (AW + AE)) / max(mean(abs.(Tr[:west])) * 1e6 / AW, eps()),
mean(Tr[:north]), 100 * abs(mean(Tr[:north])) / max(mean(abs.(Tr[:west])), eps()),
mean(abs.(Tr[:west])), minimum(Tr[:north]), maximum(Tr[:north]))

if abs(mean(netWE)) > 10 || maximum(abs.(netWE)) > 20
    println(">>> west+east do NOT balance on their own. :zonal is bending the ACC by this")
    println(">>> much, and the north face is carrying real volume. Reconsider.")
else
    println(">>> west+east nearly balance on their own. :zonal costs little.")
end

hdr("Q3 -- structure of the prescribed normal flow (peak over the window)")
println("        max |depth-mean V|   face-mean    peakedness")
for s in (:west, :east, :south, :north)
    mx, fm, r = peak[s]
    @printf(" %-6s  %10.4f m/s     %9.4f m/s   %6.1fx\n", s, mx, fm, r)
end
println("""
 Peakedness is max|depth-mean| / face-mean. The sandbox JET knob should be set so its
 profile matches the west/east figure here; a uniform Uᵉˣᵗ in the sandbox is not testing
 the same boundary the model sees.""")

hdr("Q7 -- raw-slice sanity")
for s in (:west, :east, :south, :north)
    @printf(" %-6s  NaNs in slab: %8d   |velocity| summed over BSOSE-dry cells: %.4e m/s\n",
            s, nan_count[s], dry_signal[s])
end
println("""
 Nonzero velocity in BSOSE-dry cells means the downloaded file is already inpainted.
 It is harmless here (hFac zeroes it) but NOT harmless in the model, which regrids onto a
 deeper ETOPO floor and can pick those values up -- see Q6.""")

# ------------------------------------------------------------------------------
# Q4, Q5 -- prescribed T and S at the inflow face.
# ------------------------------------------------------------------------------
tds, tname = open_var(:temperature)
sds, sname = open_var(:salinity)

φ_b = φc[jrange]
band = findall(φ -> BAND[1] <= φ <= BAND[2], φ_b)
k200 = findlast(k -> abs(zc[k]) <= 200, 1:Nz_b)

hdr("Q4/Q5 -- T and S prescribed at the western inflow column (i = $i₁)")
@printf(" band for the focus column: %.1f to %.1f N (%d cells), upper 200 m = k 1:%d\n\n",
        BAND[1], BAND[2], length(band), k200)
println(" date          surf T   0-200m T   full T  |  surf S   0-200m S   full S  | band 0-200m T   band S")
println("-"^104)
bandT = zeros(nt); bandS = zeros(nt)
for (ti, t) in enumerate(tsel)
    T = Float64.(coalesce.(reshape(tds[tname][i₁, jrange, :, t], length(jrange), Nz_b), NaN))
    S = Float64.(coalesce.(reshape(sds[sname][i₁, jrange, :, t], length(jrange), Nz_b), NaN))
    wet = hC_W .> 0
    msk(A, kk) = (v = A[:, kk][wet[:, kk]]; isempty(v) ? NaN : mean(filter(isfinite, v)))
    bT = let w = wet[band, 1:k200], A = T[band, 1:k200]; v = A[w]; isempty(v) ? NaN : mean(filter(isfinite, v)) end
    bS = let w = wet[band, 1:k200], A = S[band, 1:k200]; v = A[w]; isempty(v) ? NaN : mean(filter(isfinite, v)) end
    bandT[ti] = bT; bandS[ti] = bS
    @printf(" %s  %6.2f    %7.2f  %7.2f  |  %6.3f    %7.3f  %7.3f  |     %7.2f   %7.3f\n",
            Dates.format(times[t], "yyyy-mm-dd"),
            msk(T, 1:1), msk(T, 1:k200), msk(T, 1:Nz_b),
            msk(S, 1:1), msk(S, 1:k200), msk(S, 1:Nz_b), bT, bS)
end
println("-"^104)
@printf(" The %.0f-%.0fS band inflow is %.2f to %.2f C (mean %.2f) and %.3f to %.3f PSU (mean %.3f).\n",
        -BAND[2], -BAND[1], minimum(filter(isfinite, bandT)), maximum(filter(isfinite, bandT)),
        mean(filter(isfinite, bandT)), minimum(filter(isfinite, bandS)),
        maximum(filter(isfinite, bandS)), mean(filter(isfinite, bandS)))
println(""" If the band inflow T stays near freezing while the model develops +7-9 C there, the
 boundary is not importing the warm bias and the redistribution diagnosis stands. If the
 prescribed inflow is itself several degrees warm, it is a second source.""")

# ------------------------------------------------------------------------------
# Q8 -- where BSOSE's sea surface actually sits at each face.
#
# `bsose_exterior_free_surface` removes a single domain-wide datum, which lines BSOSE's
# reference up with the model's (whose η starts at zero domain mean) while preserving the
# spatial structure. What is left is the per-face offset the Flather condition must be told
# about. Setting EXTERIOR_SSH = false replaces all of it with zero.
# ------------------------------------------------------------------------------
hdr("Q8 -- BSOSE sea surface at each face, relative to the domain datum")

eds, ename = open_var(:sea_surface_height)
gds2 = Dataset(grid_file)
Depth_dom = Float64.(coalesce.(gds2["Depth"][irange, jrange], 0.0))
close(gds2)
wet_dom = Depth_dom .> 0

# An eta offset of Δη on a face is worth Δη * √(gH) * Σ(width) of transport.
Σw_we = sum(dy_col)
Σw_N  = sum(dx_col_N)
HN_mean = safemean(HN[HN .> 0])
HW_mean = safemean(HW[HW .> 0])
sens_we = 0.01 * sqrt(9.81 * HW_mean) * Σw_we / 1e6
sens_N  = 0.01 * sqrt(9.81 * HN_mean) * Σw_N  / 1e6

println(" date           datum     west-d    east-d   north-d |  north offset in Sv")
println("-"^74)
offW, offE, offN = Float64[], Float64[], Float64[]
for t in tsel
    η = Float64.(coalesce.(eds[ename][irange, jrange, t], 0.0))   # land is 0.0, never NaN
    datum = mean(η[wet_dom])
    fw = safemean(η[1, :][wet_dom[1, :]])
    fe = safemean(η[end, :][wet_dom[end, :]])
    fn = safemean(η[:, end][wet_dom[:, end]])
    push!(offW, fw - datum); push!(offE, fe - datum); push!(offN, fn - datum)
    @printf(" %s %+9.4f %+9.4f %+9.4f %+9.4f | %+10.1f Sv\n",
            Dates.format(times[t], "yyyy-mm-dd"), datum, fw - datum, fe - datum, fn - datum,
            (fn - datum) * sqrt(9.81 * HN_mean) * Σw_N / 1e6)
end
close(eds)
println("-"^74)
@printf(" mean offsets: west %+.4f m, east %+.4f m, north %+.4f m\n",
        mean(offW), mean(offE), mean(offN))
@printf("\n sensitivity: 1 cm of (etaB - etaExt) error is worth %.1f Sv on west/east, %.1f Sv on north.\n",
        sens_we, sens_N)
@printf(" The north face sits %+.3f m off the domain datum. Told etaExt = 0, that is %+.0f Sv.\n",
        mean(offN), mean(offN) * sqrt(9.81 * HN_mean) * Σw_N / 1e6)
println("""
 The model never shows that number because the free surface drifts down until the error
 partially cancels -- which is the observed 'mean eta walking down to -0.65 m'. What
 survives the drift is the few-hundred-Sv residual the failed runs reported.

 There is no scalar etaExt that is right for more than one face: the structure spans more
 than a metre across this domain. EXTERIOR_SSH = false is not a simplification, it is a
 different and wrong boundary condition.""")

# ------------------------------------------------------------------------------
# Q6 -- BSOSE's mask against the model's ETOPO bathymetry.
# ------------------------------------------------------------------------------
if DO_MODEL_MASK
    hdr("Q6 -- BSOSE hFac mask vs the model's regridded ETOPO bathymetry")
    using Oceananigans
    using NumericalEarth

    SCALING = 6
    Nxm, Nym = Int(SCALING * (λ₂ - λ₁)), Int(SCALING * (φ₂ - φ₁))
    zm = ReferenceToStretchedDiscretization(; extent = 5800, constant_spacing = 5,
             maximum_spacing = 200, constant_spacing_extent = 500,
             stretching = PowerLawStretching(1.12))
    mgrid = LatitudeLongitudeGrid(CPU(); size = (Nxm, Nym, length(zm)),
                latitude = (φ₁, φ₂), longitude = (λ₁, λ₂), z = zm, halo = (7, 7, 7))
    bh = regrid_bathymetry(mgrid, height_above_water = 1, minimum_depth = 10,
                           interpolation_passes = 5)
    Hm = -Array(interior(bh, :, :, 1))     # positive depth, land <= 0

    φm = φnodes(mgrid, Center())
    λm = λnodes(mgrid, Center())

    # Interpolate BSOSE's wet depth onto the model's boundary columns, nearest neighbour.
    nearest(x, v) = argmin(abs.(v .- x))
    model_area = Dict{Symbol,Float64}()
    for (side, Hb, coord_b, Hmodel, coord_m) in (
            (:west,  HW, φ_b, Hm[1, :],    φm),
            (:east,  HE, φ_b, Hm[end, :],  φm),
            (:south, HS, λc[irange], Hm[:, 1],   λm),
            (:north, HN, λc[irange], Hm[:, end], λm))
        Hb_on_m = [Hb[nearest(c, coord_b)] for c in coord_m]
        bwet, mwet = Hb_on_m .> 0, Hmodel .> 0
        both = bwet .& mwet
        # Face area on the model's own mask -- this, not BSOSE's, is what sets how much
        # transport a given (ηᵇ - ηᵉˣᵗ) error buys on each face.
        wm = side in (:south, :north) ?
             [R_EARTH * cosd(side === :south ? φ₁ : φ₂) * deg2rad(360 / (SCALING * 360)) for _ in coord_m] :
             [R_EARTH * deg2rad(1 / SCALING) for _ in coord_m]
        model_area[side] = sum(max.(Hmodel, 0.0) .* wm)
        @printf(" %-6s  BSOSE wet %4d/%4d, model wet %4d/%4d, disagree %4d (%.1f%%)\n",
                side, count(bwet), length(bwet), count(mwet), length(mwet),
                count(bwet .!= mwet), 100 * count(bwet .!= mwet) / length(bwet))
        if any(both)
            d = Hmodel[both] .- Hb_on_m[both]
            @printf("         where both wet: model is %+.1f m deeper on average, range %+.1f to %+.1f m\n",
                    mean(d), minimum(d), maximum(d))
            @printf("         model deeper than BSOSE in %d of %d shared columns (%.1f%%)\n",
                    count(>(0), d), count(both), 100 * count(>(0), d) / count(both))
        end
    end
    @printf("\n model-mask face areas: west %.3e, east %.3e, south %.3e, north %.3e m2\n",
            model_area[:west], model_area[:east], model_area[:south], model_area[:north])
    @printf(" north / (west+east) = %.2fx on the model mask, %.2fx on BSOSE's.\n",
            model_area[:north] / max(model_area[:west] + model_area[:east], eps()),
            AN / max(AW + AE, eps()))
    @printf(" A %.2f m error in (etaB - etaExt) on the north face buys %+.1f Sv there.\n",
            0.24, 0.24 * sqrt(9.81 * safemean(HN[HN .> 0])) * model_area[:north] / 1e6 /
                  max(safemean(HN[HN .> 0]), eps()))
    println("""
 Columns the model calls wet but BSOSE calls dry get inpainted velocity, and columns where
 the model is deeper extend the integral below BSOSE's floor. Both inject transport that
 BSOSE never prescribed -- this is the ~+3 Sv the balance routine removes, and the reason
 the four faces do not sum to zero even though BSOSE's own do.""")
else
    @info "MODEL_MASK=0: skipping Q6."
end

# ------------------------------------------------------------------------------
# Output.
# ------------------------------------------------------------------------------
csv = joinpath(OUTDIR, "face_transports.csv")
open(csv, "w") do io
    println(io, "date,west_Sv,east_Sv,south_Sv,north_Sv,net4_Sv,netWE_Sv,band_T_C,band_S_PSU")
    for ti in 1:nt
        @printf(io, "%s,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                Dates.format(times[tsel[ti]], "yyyy-mm-dd"), Tr[:west][ti], Tr[:east][ti],
                Tr[:south][ti], Tr[:north][ti], net4[ti], netWE[ti], bandT[ti], bandS[ti])
    end
end
@info "wrote $csv"

if DO_FIGURES
    using CairoMakie

    dts = times[tsel]
    xs = 1:nt
    lbl = [Dates.format(d, "yyyy-mm-dd") for d in dts]
    step = max(1, nt ÷ 10)

    fig = Figure(size = (1100, 760))
    ax = Axis(fig[1, 1], title = "Prescribed face transport, BSOSE i$(dataset.iteration)",
              ylabel = "Sv", xticks = (xs[1:step:end], lbl[1:step:end]), xticklabelrotation = π/4)
    for (s, c) in ((:west, :dodgerblue), (:east, :firebrick), (:north, :seagreen), (:south, :orange))
        lines!(ax, xs, Tr[s], color = c, linewidth = 2, label = String(s))
    end
    axislegend(ax, position = :lt)

    ax2 = Axis(fig[2, 1], title = "Net imbalance: all four faces vs west+east only",
               ylabel = "Sv", xticks = (xs[1:step:end], lbl[1:step:end]), xticklabelrotation = π/4)
    lines!(ax2, xs, net4, color = :black, linewidth = 2, label = "4-face net")
    lines!(ax2, xs, netWE, color = :purple, linewidth = 2, linestyle = :dash, label = "west+east net")
    hlines!(ax2, [0.0], color = :gray, linestyle = :dot)
    axislegend(ax2, position = :lt)
    save(joinpath(OUTDIR, "face_transports.png"), fig)

    fig2 = Figure(size = (1200, 480))
    for (n, s) in enumerate((:west, :east))
        sec = u_sections[s]
        coord = φ_b
        local ax = Axis(fig2[1, n], title = "$(s) face: time-mean U (m/s)",
                  xlabel = "latitude", ylabel = "z (m)")
        hm = heatmap!(ax, coord, zc, sec, colormap = :balance,
                      colorrange = (-maximum(abs, sec), maximum(abs, sec)))
        Colorbar(fig2[1, n == 1 ? 1 : 2][1, 2], hm)
        ylims!(ax, -5000, 0)
    end
    save(joinpath(OUTDIR, "west_east_U_sections.png"), fig2)

    fig3 = Figure(size = (1000, 420))
    ax = Axis(fig3[1, 1], title = "Prescribed inflow in the $(Int(-BAND[2]))-$(Int(-BAND[1]))S band, upper 200 m",
              ylabel = "theta (C)", xticks = (xs[1:step:end], lbl[1:step:end]), xticklabelrotation = π/4)
    lines!(ax, xs, bandT, color = :firebrick, linewidth = 2)
    hlines!(ax, [-1.88], color = :blue, linestyle = :dash)
    text!(ax, 1, -1.88, text = " freezing", align = (:left, :bottom), color = :blue, fontsize = 11)
    ax2 = Axis(fig3[1, 2], title = "and its salinity", ylabel = "S (PSU)",
               xticks = (xs[1:step:end], lbl[1:step:end]), xticklabelrotation = π/4)
    lines!(ax2, xs, bandS, color = :seagreen, linewidth = 2)
    save(joinpath(OUTDIR, "band_inflow_TS.png"), fig3)

    @info "wrote figures to $OUTDIR"
end

for d in (uds, vds, tds, sds); close(d); end
println("\ndone.")
