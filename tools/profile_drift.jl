# ==============================================================================
# profile_drift.jl -- WHERE IN THE COLUMN does our salinity differ from B-SOSE's?
#
#     MODEL_RUN=2Y_wind_fw_sw_5day julia --project=. tools/profile_drift.jl
#
# WHY. `compare_bsose.jl` measures the SURFACE drift (-0.032 g/kg/yr, linear) and
# `mixing_energetics.jl` measures the COLUMN-INTEGRATED stratification excess (~40% at 70-65S
# and nowhere else). Neither says where between the two the error lives, and that is the thing
# that discriminates between the remaining candidates:
#
#     concentrated in the top ~50 m   surface freshwater piling up faster than it is exported.
#                                     A surface boundary-condition problem.
#     spread through 100-300 m        the halocline itself sits at the wrong depth. An
#                                     advection or initial-condition problem, not a surface one.
#     below the winter mixed layer    it arrived through the open boundaries.
#
# This is a measurement, not a hypothesis, and it needs no download: it uses the monthly Theta
# and Salt already on disk and the run's own output.
#
# BAND MEANS, NOT POINTWISE. The two grids differ horizontally (B-SOSE native vs our 1/6 deg),
# so each side is cos-lat weighted over its OWN wet cells inside the band and only the two
# resulting 1-D profiles are differenced -- the same convention `mixing_energetics.jl` uses for
# `req model` against `req BSOSE`. The approximation is that the two wet masks sample slightly
# different water; section 1 prints the cell counts so that is visible rather than assumed.
#
# LOGIN-NODE SAFE: NCDatasets only, one month resident at a time, no Oceananigans.
#
# ENV
#     BSOSE_DIR     directory holding the monthly files      (default: data)
#     MODEL_RUN     run name; reads <run>_eke.nc             (REQUIRED)
#     MODEL_DIR     where to look for it, then surface_tests (default: .)
#     MODEL_START   simulation time zero                     (default: 2014-06-01)
#     START_DATE    first month to compare                   (default: 2014-06-01)
#     END_DATE      last month to compare                    (default: 2016-06-01)
#     ZMAX          depth to compare over, m                 (default: 600)
#     OUTDIR        where the figures go                     (default: plots)
# ==============================================================================

using NCDatasets, Dates, Printf, Statistics, CairoMakie

const DIR    = get(ENV, "BSOSE_DIR", "data")
const RUN    = get(ENV, "MODEL_RUN", "")
const MDIRS  = [get(ENV, "MODEL_DIR", "."), "surface_tests"]
const MSTART = DateTime(get(ENV, "MODEL_START", "2014-06-01"))
const START  = DateTime(get(ENV, "START_DATE", "2014-06-01"))
const STOP   = DateTime(get(ENV, "END_DATE", "2016-06-01"))
const ZMAX   = parse(Float64, get(ENV, "ZMAX", "600"))
const OUTDIR = get(ENV, "OUTDIR", "plots")

isempty(RUN) && error("MODEL_RUN is required -- this tool compares OUR column against B-SOSE's")

const LON   = (90.0, 150.0)
const LAT   = (-70.0, -45.0)
const BANDS = [(-70.0, -65.0), (-65.0, -60.0), (-60.0, -55.0), (-55.0, -50.0), (-50.0, -45.0)]
const BINS  = [(0.0, 50.0), (50.0, 100.0), (100.0, 200.0), (200.0, 400.0), (400.0, 600.0)]

function window_center(stamp)
    stop = round(DateTime(stamp), Month)
    start = stop - Month(1)
    return start + Millisecond(Dates.value(Millisecond(stop - start)) ÷ 2)
end

bfile(token) = joinpath(DIR, "$(token)_bsoseI156_2013to2024_monthly.nc")

function coordvar(ds, cands, what)
    for c in cands
        haskey(ds, c) && return Float64.(coalesce.(ds[c][:], NaN))
    end
    error("no $what coordinate in $(path(ds)); tried $(join(cands, ", ")). " *
          "Keys: $(collect(keys(ds)))")
end

"""Linear interpolation of `v` given at positive depths `d` onto positive depths `t`.
Both are sorted ascending here; outside the source range the nearest value is held, which only
bites at the very top and bottom cells and is flagged by the caller's depth range."""
function interp_depth(d, v, t)
    ord = sortperm(d); d, v = d[ord], v[ord]
    keep = findall(i -> isfinite(d[i]) && isfinite(v[i]), eachindex(d))
    isempty(keep) && return fill(NaN, length(t))
    d, v = d[keep], v[keep]
    map(t) do z
        z <= d[1]   && return v[1]
        z >= d[end] && return v[end]
        k = searchsortedfirst(d, z)
        w = (z - d[k - 1]) / (d[k] - d[k - 1])
        return (1 - w) * v[k - 1] + w * v[k]
    end
end

# ── B-SOSE geometry and vertical grid ─────────────────────────────────────────────────────────
probe = Dataset(bfile("Salt"))
bx = coordvar(probe, ["XC"], "longitude")
by = coordvar(probe, ["YC"], "latitude")
bis = findall(x -> LON[1] <= x <= LON[2], bx)
bjs = findall(y -> LAT[1] <= y <= LAT[2], by)
bstamps = [DateTime(t) for t in probe["time"][:]]
close(probe)
const BLAT = by[bjs]

gfile = joinpath(DIR, "grid.nc")
isfile(gfile) || error("need $gfile for B-SOSE's vertical grid (RC)")
gds = Dataset(gfile); RC_all = Float64.(coalesce.(gds["RC"][:], NaN)); close(gds)
const NZ = something(findlast(z -> -z <= ZMAX, RC_all), length(RC_all))
const ZB = -RC_all[1:NZ]                      # positive depths, surface first

const SEL = [k for k in eachindex(bstamps) if START <= window_center(bstamps[k]) <= STOP]
isempty(SEL) && error("no B-SOSE record between $START and $STOP")
const LABELS = [Dates.format(window_center(bstamps[k]), "yyyy-mm") for k in SEL]

# ── our run ───────────────────────────────────────────────────────────────────────────────────
hit = findfirst(d -> isfile(joinpath(d, "$(RUN)_eke.nc")), MDIRS)
hit === nothing && error("no $(RUN)_eke.nc under $(join(MDIRS, " or "))")
const MPATH = joinpath(MDIRS[hit], "$(RUN)_eke.nc")
@info "model column from $MPATH"

mds = Dataset(MPATH)
const MZ_UP = coordvar(mds, ["z_aac", "zC", "z_c", "z"], "z")          # bottom-to-top, negative
const MLAT  = coordvar(mds, ["y_aca", "φ_aca", "φ_cca", "yC", "latitude", "lat", "y"], "latitude")
const MT    = Float64.(mds["time"][:])
const MWIN  = haskey(mds.attrib, "averaging_window_seconds") ?
              Float64(mds.attrib["averaging_window_seconds"]) : 30 * 86400.0
const MCEN  = [MSTART + Second(round(Int, x - MWIN / 2)) for x in MT]
close(mds)
const MZ = -reverse(MZ_UP)                                            # positive, surface first

@info "$(length(SEL)) months $(first(LABELS))..$(last(LABELS)); " *
      "B-SOSE $(NZ) levels to $(round(ZB[end], digits=1)) m, model $(length(MZ)) levels to " *
      "$(round(maximum(MZ), digits=1)) m"

"""Cos-lat weighted band-mean profile of a (ni, nj, nk) field, surface-first on return."""
function band_profile(A, lats, lo, hi; surface_first)
    nk = size(A, 3)
    num = zeros(nk); den = zeros(nk)
    for j in axes(A, 2)
        (lo <= lats[j] < hi) || continue
        w = cosd(lats[j])
        for i in axes(A, 1), k in 1:nk
            v = A[i, j, k]
            # Oceananigans writes immersed cells as EXACT zeros, and B-SOSE's land comes back
            # NaN; `isfinite && != 0` is the one wet test that is right for both.
            (isfinite(v) && v != 0) || continue
            num[k] += w * v; den[k] += w
        end
    end
    p = [den[k] == 0 ? NaN : num[k] / den[k] for k in 1:nk]
    return surface_first ? p : reverse(p)
end

"""Wet COLUMN count per band on each side, so the differing masks are visible, not assumed.
Counts a column if ANY level is wet, which is orientation-independent -- index 1 is the bottom
on our side and the surface on B-SOSE's, so sampling a single level would undercount one of
them wherever the bathymetry is shallow."""
function band_cells(A, lats, lo, hi)
    n = 0
    for j in axes(A, 2)
        (lo <= lats[j] < hi) || continue
        for i in axes(A, 1)
            any(k -> (v = A[i, j, k]; isfinite(v) && v != 0), axes(A, 3)) && (n += 1)
        end
    end
    return n
end

# ── pair each B-SOSE month with the nearest model window ──────────────────────────────────────
pairs = NamedTuple[]
for (n, k) in enumerate(SEL)
    c = window_center(bstamps[k])
    gap, m = findmin(abs(Dates.value(x - c)) for x in MCEN)
    gd = gap / 86400_000
    if gd > 20
        @warn "no model window within 20 days of $(LABELS[n]) (nearest $(round(gd, digits=1)) d) -- skipping"
        continue
    end
    push!(pairs, (label = LABELS[n], bk = k, mk = m, gap = gd))
end
isempty(pairs) && error("no month in range has a model window within 20 days")
@info "$(length(pairs)) month(s) paired, max offset $(round(maximum(p -> p.gap, pairs), digits=1)) days"

# ── the profiles ──────────────────────────────────────────────────────────────────────────────
dS = Dict(b => fill(NaN, NZ, length(pairs)) for b in BANDS)
dT = Dict(b => fill(NaN, NZ, length(pairs)) for b in BANDS)
cells = Dict(b => (0, 0) for b in BANDS)

bS = Dataset(bfile("Salt")); bT = Dataset(bfile("Theta")); md = Dataset(MPATH)
svar = haskey(bS, "SALT") ? "SALT" : "Salt"
tvar = haskey(bT, "THETA") ? "THETA" : "Theta"
for (n, p) in enumerate(pairs)
    BS = Float64.(coalesce.(bS[svar][bis, bjs, 1:NZ, p.bk], NaN))
    BT = Float64.(coalesce.(bT[tvar][bis, bjs, 1:NZ, p.bk], NaN))
    MS = Float64.(coalesce.(md["S"][:, :, :, p.mk], NaN))
    MM = Float64.(coalesce.(md["T"][:, :, :, p.mk], NaN))
    for b in BANDS
        bs = band_profile(BS, BLAT, b[1], b[2]; surface_first = true)
        bt = band_profile(BT, BLAT, b[1], b[2]; surface_first = true)
        ms = interp_depth(MZ, band_profile(MS, MLAT, b[1], b[2]; surface_first = false), ZB)
        mt = interp_depth(MZ, band_profile(MM, MLAT, b[1], b[2]; surface_first = false), ZB)
        dS[b][:, n] = ms .- bs
        dT[b][:, n] = mt .- bt
        n == 1 && (cells[b] = (band_cells(BS, BLAT, b[1], b[2]),
                               band_cells(MS, MLAT, b[1], b[2])))
    end
end
close(bS); close(bT); close(md)

# ── output ────────────────────────────────────────────────────────────────────────────────────
println("\n" * "="^100)
println("1. WET CELLS PER BAND -- the two sides are averaged over their OWN masks")
println("="^100)
@printf("\n    %-10s %12s %12s   %s\n", "band", "B-SOSE", "model", "note")
for b in BANDS
    nb, nm = cells[b]
    @printf("    %4.0fS-%2.0fS  %12d %12d   %s\n", -b[1], -b[2], nb, nm,
            nb == 0 || nm == 0 ? "EMPTY -- band means are meaningless here" : "")
end
println("\n  The counts differ because the grids differ; what matters is that neither is tiny.")

binof(z) = findfirst(i -> BINS[i][1] <= z < BINS[i][2], eachindex(BINS))

function bin_table(title, D, unit)
    println("\n" * "="^100)
    println("$title  (model - B-SOSE, $unit)")
    println("="^100)
    for b in BANDS
        @printf("\n  %.0fS to %.0fS\n", -b[1], -b[2])
        print("    " * rpad("depth", 11) * "|")
        for p in pairs; print(lpad(p.label[3:end], 9)); end
        println()
        for (lo, hi) in BINS
            ks = [k for k in 1:NZ if lo <= ZB[k] < hi]
            isempty(ks) && continue
            @printf("    %-11s|", @sprintf("%.0f-%.0f m", lo, hi))
            for n in eachindex(pairs)
                v = mean(filter(isfinite, D[b][ks, n]))
                print(lpad(isfinite(v) ? @sprintf("%.4f", v) : "--", 9))
            end
            println()
        end
    end
end

bin_table("2. SALINITY DIFFERENCE BY DEPTH BIN", dS, "g/kg")
bin_table("3. TEMPERATURE DIFFERENCE BY DEPTH BIN", dT, "degC")

println("\n" * "="^100)
println("4. WHERE THE SALINITY ERROR LIVES -- share of the column integral of |dS| per bin")
println("  This is the discriminator. Top-heavy means a surface boundary-condition problem;")
println("  weight at 100-300 m means the halocline is misplaced; weight below the winter")
println("  mixed layer means it came in through the open boundaries.")
println("="^100)
@printf("\n    %-10s", "band")
for (lo, hi) in BINS; @printf("%12s", @sprintf("%.0f-%.0f m", lo, hi)); end
@printf("%12s%12s\n", "peak |dS|", "at depth")
for b in BANDS
    prof = [mean(filter(isfinite, dS[b][k, :])) for k in 1:NZ]       # time mean, per level
    dz   = [k == 1 ? ZB[1] : ZB[k] - ZB[k - 1] for k in 1:NZ]
    tot  = sum(k -> isfinite(prof[k]) ? abs(prof[k]) * dz[k] : 0.0, 1:NZ)
    @printf("    %4.0fS-%2.0fS", -b[1], -b[2])
    # A band with no real drift still has float residue, and dividing that by itself returns
    # the LAYER THICKNESS distribution -- which looks like a result and is not one. 1e-6 g/kg.m
    # is six orders below anything physical here.
    if tot < 1.0e-6
        for _ in BINS; @printf("%12s", "--"); end
        @printf("%12s%12s\n", "--", "no drift")
        continue
    end
    for (lo, hi) in BINS
        ks = [k for k in 1:NZ if lo <= ZB[k] < hi]
        s = isempty(ks) ? 0.0 : sum(k -> isfinite(prof[k]) ? abs(prof[k]) * dz[k] : 0.0, ks)
        @printf("%11.1f%%", 100 * s / tot)
    end
    kk = argmax([isfinite(prof[k]) ? abs(prof[k]) : -Inf for k in 1:NZ])
    @printf("%12.4f%10.0f m\n", prof[kk], ZB[kk])
end

# ── figures ───────────────────────────────────────────────────────────────────────────────────
mkpath(OUTDIR)
for (name, D, unit, cmax) in (("salinity", dS, "g/kg", nothing), ("temperature", dT, "degC", nothing))
    lim = maximum(b -> maximum(x -> isfinite(x) ? abs(x) : 0.0, D[b]), BANDS)
    lim = lim == 0 ? 1.0 : lim
    fig = Figure(size = (330 * length(BANDS), 460))
    local hm
    for (n, b) in enumerate(BANDS)
        ax = Axis(fig[1, n], title = @sprintf("%.0fS to %.0fS", -b[1], -b[2]),
                  xlabel = "month", ylabel = n == 1 ? "depth (m)" : "",
                  yreversed = true,
                  xticks = (1:3:length(pairs), [pairs[i].label[3:end] for i in 1:3:length(pairs)]),
                  xticklabelsize = 9, xticklabelrotation = pi / 4)
        hm = heatmap!(ax, 1:length(pairs), ZB, permutedims(D[b]),
                      colormap = :balance, colorrange = (-lim, lim))
    end
    Colorbar(fig[1, length(BANDS) + 1], hm, label = "model - B-SOSE  ($unit)")
    Label(fig[0, :], "Where the $name error lives -- $RUN against B-SOSE i156",
          fontsize = 16, font = :bold)
    f = joinpath(OUTDIR, "profile_drift_$(name)_$(RUN).png"); save(f, fig); @info "wrote $f"
end
