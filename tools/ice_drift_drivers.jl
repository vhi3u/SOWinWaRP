# ==============================================================================================
# ice_drift_drivers.jl -- why does the run's ice drift north across 65S when B-SOSE's does not?
# Wind, ocean current, or the ice's response to them.
#
#     BSOSE_DIR=/storage/scratch1/1/vnguyen480/SOWinWaRP/data julia --project=. tools/ice_drift_drivers.jl 2Y_nested_bsose_ggl_hj_pr207_icein
#
# WHY. tools/ice_meridional_transport.jl on _icein (with B-SOSE's own SIvheff): June-October of both
# winters the run sent 1395 km³ of ice north across 65S, B-SOSE 139, at about the same h·ℵ: the run's
# ice moves north at 1-4 cm/s, B-SOSE's at about 0. The ice that crosses melts in 65-60S.
#
# WHAT, per line (DRIFT_LINES, default 65S and 58S) and calendar month:
#   VELOCITIES at the line, both components, cm/s (wind m/s):
#     ice     run: daily snapshots of u, v (<RUN>_sea_ice.nc), mean over iced faces (h·ℵ > 0.01 m);
#             B-SOSE: 5-day SIuice, SIvice, mean over iced faces (SIheff > 0.01 m)
#     ocean   top cell. run: 30-day means of u, v (<RUN>_eke.nc), each window to the month of its
#             centre; B-SOSE: monthly Uvel, Vvel top level (record centre = stamp - 15 days).
#             Mean over the line's wet columns.
#     wind    ERA5 10 m u, v, the run's own forcing (ERA5_DIR, default BSOSE_DIR/era5), hourly,
#             mean over the line inside the domain.
#   Southern Hemisphere ice drifts to the LEFT of the wind: an eastward wind pushes ice north, so a
#   north-south difference can come from the east-west wind as much as the north-south one.
#   STRESS on B-SOSE's ocean (5-day oceTAUX, oceTAUY) against a bulk ERA5 stress (Large & Yeager
#   2004 neutral drag, from hourly winds, averaged over the same 5 days and faces), mN/m², over:
#     open faces    (B-SOSE SIArea < 0.05): oceTAU is the wind stress itself, so oceTAU against
#                   ERA5 says whether B-SOSE's adjusted wind differs from the run's raw ERA5.
#     compact faces (SIArea > 0.8): oceTAU is what the ice passes on to the ocean; small against
#                   ERA5 = B-SOSE's ice holds the wind (internal stress), not the drag.
#   B-SOSE has no wind file (no EXF*, SItaux/SItauy in i156), so this is the only check on it.
#
# B-SOSE layout (i156): SIvice, SIvheff, oceTAUY, Vvel on (XC, YG) with 630 YG rows, rows 1:588 the
# cells' south faces; SIuice, oceTAUX, Uvel on (XG, YC); the line's face is the YG row nearest it,
# its two cells the YC rows either side.
#
# WRITES OUTDIR/<RUN>_ice_drift_drivers.png: per line, the monthly v of ice, ocean (both) and wind,
# and B-SOSE's oceTAUY against ERA5 over open and compact faces.
# ENV: DIR, OUTDIR, BSOSE_DIR, ERA5_DIR, DRIFT_LINES (-65,-58), WINTER (6,7,8,9,10)
# (not LINES: Julia reads LINES as the terminal height, and "-63,-60" there breaks its own printing)
# ==============================================================================================

using NCDatasets, CairoMakie, Dates, Printf, Statistics

ENV["HDF5_USE_FILE_LOCKING"] = "FALSE"

RUN = isempty(ARGS) ? get(ENV, "RUN", "") : ARGS[1]
isempty(RUN) && error("give the run name: julia --project=. tools/ice_drift_drivers.jl <RUN>")
DIR       = get(ENV, "DIR", ".")
OUTDIR    = get(ENV, "OUTDIR", "plots")
BSOSE_DIR = get(ENV, "BSOSE_DIR", "")
isempty(BSOSE_DIR) && error("set BSOSE_DIR")
ERA5_DIR  = get(ENV, "ERA5_DIR", joinpath(BSOSE_DIR, "era5"))
LINES     = sort(parse.(Float64, split(get(ENV, "DRIFT_LINES", "-65,-58"), ',')))
WINTER    = parse.(Int, split(get(ENV, "WINTER", "6,7,8,9,10"), ','))
const ρₐ = 1.225
const ICED = 0.01

g(A) = Float64.(coalesce.(A, NaN))
z(A) = replace(A, NaN => 0.0)
nl = length(LINES)
lname(l) = @sprintf("%.0fS", -LINES[l])
label(ym) = @sprintf("%04d-%02d", ym...)
"Large & Yeager (2004) neutral 10 m drag coefficient"
cd_ly(U) = 1e-3 * (2.7 / U + 0.142 + U / 13.09)
"Bulk wind stress (N/m²), both components, from 10 m winds"
bulk(u, v) = (U = max(hypot(u, v), 0.5); c = ρₐ * cd_ly(U) * U; (c * u, c * v))
nanmean(x) = (y = filter(isfinite, x); isempty(y) ? NaN : mean(y))
# sum and count accumulators, for means over masked faces and records
mutable struct Acc; s::Float64; n::Int; end
Acc() = Acc(0.0, 0)
add!(a::Acc, x) = isfinite(x) && (a.s += x; a.n += 1)
val(a::Acc) = a.n == 0 ? NaN : a.s / a.n

# ── the run: grid, months ──────────────────────────────────────────────────────────────────────
ice_path, eke_path = joinpath(DIR, "$(RUN)_sea_ice.nc"), joinpath(DIR, "$(RUN)_eke.nc")
isfile(ice_path) || error("$ice_path not found; set DIR to the run directory")
isfile(eke_path) || error("$eke_path not found; set DIR to the run directory")
ds = NCDataset(ice_path)
λ = g(ds["λ_caa"][:]); φ = g(ds["φ_aca"][:]); Nx, Ny = length(λ), length(φ)
start = DateTime(ds.attrib["start_date"])
ts = g(ds["time"][:]); Nt = length(ts)
wet = g(ds["bottom_height"][:, :]) .< 0
Δλ, Δφ = λ[2] - λ[1], φ[2] - φ[1]
φS = φ[1] - Δφ / 2
jm = [round(Int, (L - φS) / Δφ) + 1 for L in LINES]          # y-face j: south face of cell j
all(j -> 2 <= j <= Ny, jm) || error("a line in $LINES is outside the run's interior")
φm = [φS + (j - 1) * Δφ for j in jm]
wline = [wet[:, j-1] .& wet[:, j] for j in jm]
datetime(t) = start + Millisecond(round(Int, 1000t))
ym_of(d) = (year(d), month(d))
months = unique(ym_of.(datetime.(ts)))
filter!(ym -> count(t -> ym_of(datetime(t)) == ym, ts) >= daysinmonth(Date(ym...)) - 1, months)
nm = length(months); kof = Dict(ym => k for (k, ym) in enumerate(months))
mk() = [Acc() for _ in 1:nl, _ in 1:nm]

# ice: daily snapshots
rd(name, n, js) = (v = ds[name]; g(ndims(v) == 4 ? v[:, js, 1, n] : v[:, js, n]))
iu_r, iv_r = mk(), mk()
for n in 1:Nt
    k = get(kof, ym_of(datetime(ts[n])), 0); k == 0 && continue
    for l in 1:nl
        j = jm[l]
        V = z(rd("h", n, j-1:j)) .* z(rd("ℵ", n, j-1:j))
        iced = ((V[:, 1] .+ V[:, 2]) ./ 2 .> ICED) .& wline[l]
        v = rd("v", n, j)[1:Nx]
        u = rd("u", n, j-1:j)
        uc = (u[1:Nx, 1] .+ u[1:Nx, 2]) ./ 2                    # x-face u, the two rows either side of the line
        foreach(i -> (add!(iv_r[l, k], v[i]); add!(iu_r[l, k], uc[i])), findall(iced))
    end
end
close(ds)

# ocean: 30-day means, top cell, each window to the month of its centre
ou_r, ov_r = mk(), mk()
NCDataset(eke_path) do de
    te = g(de["time"][:]); win = haskey(de.attrib, "averaging_window_seconds") ? Float64(de.attrib["averaging_window_seconds"]) : 30 * 86400.0
    Nz = size(de["v"], 3)
    for (n, t) in enumerate(te)
        t < 0.99win && continue
        k = get(kof, ym_of(datetime(t - win / 2)), 0); k == 0 && continue
        for l in 1:nl
            j = jm[l]
            U = g(de["u"][:, j-1:j, Nz, n]); V = g(de["v"][:, j-1:j, Nz, n])
            for i in findall(wline[l])
                add!(ou_r[l, k], (U[i, 1] + U[i, 2]) / 2); add!(ov_r[l, k], (V[i, 1] + V[i, 2]) / 2)
            end
        end
    end
end

# ── ERA5: hourly 10 m wind at each line ─────────────────────────────────────────────────────────
"ERA5 yearly files holding `short` (u10/v10), by year"
function era5_files(short)
    # several files can hold the same year (a short smoke-test download beside the full year): take, per
    # year, the one with the most hours whose box covers the domain's longitudes and the lines
    cand = Dict{Int, Vector{Tuple{Int, String}}}()
    isdir(ERA5_DIR) || error("ERA5_DIR $ERA5_DIR not found")
    for f in readdir(ERA5_DIR; join = true)
        endswith(f, ".nc") || continue
        NCDataset(f) do d
            haskey(d, short) || return
            lo, la = g(d["longitude"][:]), g(d["latitude"][:])
            (minimum(lo) <= λ[1] && maximum(lo) >= λ[end] && minimum(la) <= minimum(LINES) && maximum(la) >= maximum(LINES)) || return
            tt = d["valid_time"][:]
            push!(get!(cand, year(first(tt)), Tuple{Int, String}[]), (length(tt), f))
        end
    end
    out = Dict{Int, String}()
    for (y, c) in cand
        sort!(c; rev = true)
        out[y] = c[1][2]
        length(c) > 1 && @warn "ERA5 $short $y: $(length(c)) files; using $(basename(c[1][2])) ($(c[1][1]) h), not " *
                               join(["$(basename(f)) ($n h)" for (n, f) in c[2:end]], ", ")
    end
    return out
end
fu, fv = era5_files("u10"), era5_files("v10")
eyears = sort(collect(intersect(keys(fu), keys(fv))))
all(ym -> ym[1] in eyears, months) || @warn "ERA5 u10/v10 missing for some of the run's years ($(sort(unique(first.(months))))), found $eyears"
elon, elat = NCDataset(fu[first(eyears)]) do d; g(d["longitude"][:]), g(d["latitude"][:]) end
ei = findall(x -> λ[1] - Δλ/2 <= x <= λ[end] + Δλ/2, elon)
ej = [argmin(abs.(elat .- p)) for p in φm]
wu, wv = mk(), mk()
# hourly bulk stress at each line, kept per hour for the 5-day B-SOSE windows below: (times, τx, τy) per line
etimes = DateTime[]; eτx = [Matrix{Float64}(undef, length(ei), 0) for _ in 1:nl]; eτy = deepcopy(eτx)
for y in eyears
    du, dv = NCDataset(fu[y]), NCDataset(fv[y])
    tt = DateTime.(du["valid_time"][:])
    tt == DateTime.(dv["valid_time"][:]) || error("ERA5 u10 and v10 $y hold different hours")
    append!(etimes, tt)
    for l in 1:nl
        U = g(du["u10"][ei, ej[l], :]); V = g(dv["v10"][ei, ej[l], :])
        for (h, t) in enumerate(tt)
            k = get(kof, ym_of(t), 0)
            k == 0 || (add!(wu[l, k], mean(U[:, h])); add!(wv[l, k], mean(V[:, h])))
        end
        τ = bulk.(U, V)
        eτx[l] = hcat(eτx[l], first.(τ)); eτy[l] = hcat(eτy[l], last.(τ))
    end
    close(du); close(dv)
end
for k in 1:nm, l in 1:nl
    need = 24 * daysinmonth(Date(months[k]...))
    if wu[l, k].n < 0.9need
        l == 1 && wu[l, k].n > 0 && @warn "ERA5 has $(wu[l, k].n) of $need hours in $(label(months[k])): its winds left out"
        wu[l, k] = Acc(); wv[l, k] = Acc()
    end
end

# ── B-SOSE ─────────────────────────────────────────────────────────────────────────────────────
function bfile(token, tag)
    want = lowercase("$(token)_bsoseI156_$(tag).nc")
    hits = filter(f -> lowercase(f) == want, readdir(BSOSE_DIR))
    isempty(hits) ? "" : joinpath(BSOSE_DIR, first(hits))
end
opens(p) = !isempty(p) && (try NCDataset(_ -> true, p) catch; false end)
pick(d, names) = d[first(k for k in names if haskey(d, k))]
fivetok = ("SIuice", "SIvice", "SIheff", "SIArea", "oceTAUX", "oceTAUY")
byears = filter(sort(unique(first.(months)))) do y
    bad = [t for t in fivetok if !opens(bfile(t, "$(y)_5day"))]
    isempty(bad) || @warn "B-SOSE $y skipped: $(join(bad, ", ")) 5-day file missing or unreadable"
    isempty(bad)
end
geom = joinpath(BSOSE_DIR, "surfTflx_bsoseI156_2013to2024_monthly.nc")
xc, yc, depth = NCDataset(geom) do d
    x = g(d["XC"][:]); y = g(d["YC"][:])
    (ndims(x) == 1 ? x : x[:, 1]), (ndims(y) == 1 ? y : y[1, :]), g(d["Depth"][:, :])
end
Ir = findall(x -> λ[1] - Δλ/2 <= x <= λ[end] + Δλ/2, xc)
yface = [(yc[j-1] + yc[j]) / 2 for j in 2:length(yc)]
jb = [1 + argmin(abs.(yface .- L)) for L in LINES]
φb = [(yc[j-1] + yc[j]) / 2 for j in jb]
bwet = [(depth[Ir, j-1] .> 0) .& (depth[Ir, j] .> 0) for j in jb]
eofB = [argmin(abs.(elon[ei] .- x)) for x in xc[Ir]]            # nearest ERA5 column to each B-SOSE face
function checkYG(d, what)
    haskey(d, "YG") || return
    yg = g(d["YG"][:])
    for l in 1:nl
        abs(yg[jb[l]] - φb[l]) < 0.01 || error(@sprintf("%s's YG[%d] = %.4f is not the face at %.4f", what, jb[l], yg[jb[l]], φb[l]))
    end
end

iu_b, iv_b, ou_b, ov_b = mk(), mk(), mk(), mk()
sτ = Dict((c, s) => mk() for c in (:open, :compact) for s in (:bx, :by, :ex, :ey))   # B-SOSE oceTAU, ERA5 bulk
nf = Dict(c => mk() for c in (:open, :compact))                                     # face-records counted
for y in byears
    F = Dict(t => NCDataset(bfile(t, "$(y)_5day")) for t in fivetok)
    Uv = pick(F["SIuice"], ("SIuice", "SIUICE")); Vv = pick(F["SIvice"], ("SIvice", "SIVICE"))
    Hv = pick(F["SIheff"], ("SIheff", "SIHeff", "SIHEFF")); Av = pick(F["SIArea"], ("SIarea", "SIArea", "SIAREA"))
    Tx = pick(F["oceTAUX"], ("oceTAUX", "OCETAUX")); Ty = pick(F["oceTAUY"], ("oceTAUY", "OCETAUY"))
    for (v, nm_) in ((Vv, "SIvice"), (Ty, "oceTAUY"))
        size(v, 1) == length(xc) && size(v, 2) >= length(yc) || error("$nm_ $(size(v)) against XC×YC ($(length(xc)), $(length(yc)))")
    end
    checkYG(F["SIvice"], "SIvice"); checkYG(F["oceTAUY"], "oceTAUY")
    stamps = DateTime.(F["SIvice"]["time"][:])
    for t in fivetok
        DateTime.(F[t]["time"][:]) == stamps || error("B-SOSE $t $y holds different records from SIvice")
    end
    for (n, s) in enumerate(stamps)
        c = s - Hour(60); k = get(kof, ym_of(c), 0); k == 0 && continue
        hrs = findall(t -> s - Day(5) < t <= s, etimes)
        for l in 1:nl
            j = jb[l]
            H = z(g(Hv[Ir, j-1:j, n])); A = z(g(Av[Ir, j-1:j, n]))
            Hf, Af = (H[:, 1] .+ H[:, 2]) ./ 2, (A[:, 1] .+ A[:, 2]) ./ 2
            v = g(Vv[Ir, j, n]); U = g(Uv[Ir, j-1:j, n]); uc = (U[:, 1] .+ U[:, 2]) ./ 2
            foreach(i -> (add!(iv_b[l, k], v[i]); add!(iu_b[l, k], uc[i])), findall((Hf .> ICED) .& bwet[l]))
            length(hrs) < 108 && continue                 # ERA5 must cover at least 90% of the 5 days
            ty = g(Ty[Ir, j, n]); X = g(Tx[Ir, j-1:j, n]); tx = (X[:, 1] .+ X[:, 2]) ./ 2
            ex = vec(mean(eτx[l][:, hrs], dims = 2)); ey = vec(mean(eτy[l][:, hrs], dims = 2))
            for (cls, sel) in ((:open, Af .< 0.05), (:compact, Af .> 0.8))
                for i in findall(sel .& bwet[l])
                    add!(sτ[(cls, :bx)][l, k], 1e3 * tx[i]); add!(sτ[(cls, :by)][l, k], 1e3 * ty[i])
                    add!(sτ[(cls, :ex)][l, k], 1e3 * ex[eofB[i]]); add!(sτ[(cls, :ey)][l, k], 1e3 * ey[eofB[i]])
                    nf[cls][l, k].n += 1
                end
            end
        end
    end
    foreach(close, values(F))
end

# monthly Uvel, Vvel: top level, record centre = stamp - 15 days
mU, mV = bfile("Uvel", "2013to2024_monthly"), bfile("Vvel", "2013to2024_monthly")
if opens(mU) && opens(mV)
    NCDataset(mU) do du; NCDataset(mV) do dv
        Uo = pick(du, ("UVEL", "Uvel")); Vo = pick(dv, ("VVEL", "Vvel"))
        checkYG(dv, "Vvel")
        for (n, s) in enumerate(DateTime.(dv["time"][:]))
            k = get(kof, ym_of(s - Day(15)), 0); k == 0 && continue
            for l in 1:nl
                j = jb[l]
                v = g(Vo[Ir, j, 1, n]); U = g(Uo[Ir, j-1:j, 1, n])
                for i in findall(bwet[l])
                    add!(ov_b[l, k], v[i]); add!(ou_b[l, k], (U[i, 1] + U[i, 2]) / 2)
                end
            end
        end
    end end
else
    @warn "B-SOSE monthly Uvel/Vvel not readable: no B-SOSE ocean velocity"
end

# ── tables ─────────────────────────────────────────────────────────────────────────────────────
c(a) = 100 * val(a)                       # m/s -> cm/s
bar = "="^150
println("\n", bar)
println("WHAT DRIVES THE ICE ACROSS EACH LINE, $RUN against B-SOSE")
println("  velocities cm/s (wind m/s): ice over iced faces (h·ℵ > $ICED m), ocean = top cell over the line, wind = ERA5 10 m (the run's forcing)")
println("  stress mN/m²: B-SOSE's oceTAU against ERA5 bulk (Large & Yeager neutral drag), same 5-day windows and faces;")
println("  open faces (B-SOSE SIArea < 0.05): wind stress, so B-SOSE's wind against ERA5; compact (> 0.8): what B-SOSE's ice passes to the ocean")
println(bar)
for l in 1:nl
    @printf("\n  %s   (run face %.3f, B-SOSE face %.3f, ERA5 row %.2f)\n", lname(l), φm[l], φb[l], elat[ej[l]])
    println("  month    |  ice u run B-SOSE |  ice v run B-SOSE | ocean u run B-SOSE | ocean v run B-SOSE | ERA5 u10  v10 || open: n  oceTAUX  ERA5  oceTAUY  ERA5 || compact: n  oceTAUX  ERA5  oceTAUY  ERA5")
    for k in 1:nm
        o, p = :open, :compact
        @printf("  %s  |  %6.2f  %6.2f |  %6.2f  %6.2f |   %6.2f  %6.2f  |   %6.2f  %6.2f  |  %6.2f %5.2f || %7d  %6.1f %6.1f  %6.1f %6.1f || %10d  %6.1f %6.1f  %6.1f %6.1f\n",
                label(months[k]), c(iu_r[l, k]), c(iu_b[l, k]), c(iv_r[l, k]), c(iv_b[l, k]),
                c(ou_r[l, k]), c(ou_b[l, k]), c(ov_r[l, k]), c(ov_b[l, k]), val(wu[l, k]), val(wv[l, k]),
                nf[o][l, k].n, val(sτ[(o, :bx)][l, k]), val(sτ[(o, :ex)][l, k]), val(sτ[(o, :by)][l, k]), val(sτ[(o, :ey)][l, k]),
                nf[p][l, k].n, val(sτ[(p, :bx)][l, k]), val(sτ[(p, :ex)][l, k]), val(sτ[(p, :by)][l, k]), val(sτ[(p, :ey)][l, k]))
    end
    wk = [k for k in 1:nm if months[k][2] in WINTER]
    wm(A, f = val) = nanmean([f(A[l, k]) for k in wk])
    @printf("  months %s mean: ice v run %.2f B-SOSE %.2f | ice u run %.2f B-SOSE %.2f | ocean v run %.2f B-SOSE %.2f | ERA5 u10 %.2f v10 %.2f\n",
            join(WINTER, ','), wm(iv_r, c), wm(iv_b, c), wm(iu_r, c), wm(iu_b, c), wm(ov_r, c), wm(ov_b, c), wm(wu), wm(wv))
    @printf("                  open oceTAUX/ERA5 %.2f, oceTAUY/ERA5 %.2f | compact oceTAUX/ERA5 %.2f, oceTAUY/ERA5 %.2f\n",
            wm(sτ[(:open, :bx)]) / wm(sτ[(:open, :ex)]), wm(sτ[(:open, :by)]) / wm(sτ[(:open, :ey)]),
            wm(sτ[(:compact, :bx)]) / wm(sτ[(:compact, :ex)]), wm(sτ[(:compact, :by)]) / wm(sτ[(:compact, :ey)]))
end
println()

# ── figure ─────────────────────────────────────────────────────────────────────────────────────
mkpath(OUTDIR)
x = 1:nm
ticks = (x[1:3:end], label.(months)[1:3:end])
fig = Figure(size = (1500, 320 * nl + 80), fontsize = 13)
Label(fig[0, 1:2], "$RUN: what drives the ice across each line"; font = :bold, fontsize = 15)
for l in 1:nl
    ax = Axis(fig[l, 1]; title = "$(lname(l)): northward velocity", ylabel = "cm/s (wind: m/s)", xticks = ticks, xticklabelrotation = π/6)
    hlines!(ax, [0]; color = :grey70)
    lines!(ax, x, [c(iv_r[l, k]) for k in x]; color = :royalblue, linewidth = 2.5, label = "ice, run")
    lines!(ax, x, [c(iv_b[l, k]) for k in x]; color = :black, linewidth = 2.5, label = "ice, B-SOSE")
    lines!(ax, x, [c(ov_r[l, k]) for k in x]; color = :royalblue, linestyle = :dash, label = "ocean, run")
    lines!(ax, x, [c(ov_b[l, k]) for k in x]; color = :black, linestyle = :dash, label = "ocean, B-SOSE")
    lines!(ax, x, [val(wv[l, k]) for k in x]; color = :darkorange, label = "ERA5 v10 (m/s)")
    lines!(ax, x, [val(wu[l, k]) for k in x]; color = :darkorange, linestyle = :dot, label = "ERA5 u10 (m/s)")
    l == 1 && axislegend(ax; position = :lt, backgroundcolor = (:white, 0.8), labelsize = 11)
    ax2 = Axis(fig[l, 2]; title = "$(lname(l)): northward stress, B-SOSE oceTAUY against ERA5 bulk", ylabel = "mN/m²",
               xticks = ticks, xticklabelrotation = π/6)
    hlines!(ax2, [0]; color = :grey70)
    for (cls, col) in ((:open, :seagreen), (:compact, :purple))
        lines!(ax2, x, [val(sτ[(cls, :by)][l, k]) for k in x]; color = col, linewidth = 2.5, label = "oceTAUY, $cls")
        lines!(ax2, x, [val(sτ[(cls, :ey)][l, k]) for k in x]; color = col, linestyle = :dash, label = "ERA5, $cls")
    end
    l == 1 && axislegend(ax2; position = :lt, backgroundcolor = (:white, 0.8), labelsize = 11)
end
out = joinpath(OUTDIR, "$(RUN)_ice_drift_drivers.png"); save(out, fig); @info "wrote $out"
