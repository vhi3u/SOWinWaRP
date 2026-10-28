# ==============================================================================
# surface_flux_budget.jl -- what B-SOSE's surface heat and salt fluxes DO, on B-SOSE's own grid.
#
#     julia --project=. tools/surface_flux_budget.jl
#
# THE POINT. A surface flux does not set T and S, it sets their TENDENCY. What actually happens
# is that tendency minus the divergence. B-SOSE is forced with exactly the fluxes `model.jl`
# applies and does not drift the way our model does, so the quantity that matters is
#
#     residual = (what the flux implies)  -  (what B-SOSE's ocean actually does)
#
# which is the heat and salt the circulation removes. That is the thing our model has to
# reproduce, and nothing in this project has ever measured it.
#
# Everything is reported in FLUX UNITS -- W/m^2 for heat, g/m^2/s for salt -- so the forcing,
# the ocean's response and the residual can be read against each other directly. The content
# tendencies are
#
#     heat:  rho0 * cp * d/dt integral(THETA dz)      [W/m^2]
#     salt:  rho0      * d/dt integral(SALT  dz)      [g/m^2/s]
#
# over a fixed layer 0..ZTOP, using partial cell heights.
#
# PAIRING THE RECORDS. B-SOSE monthly files are MEANS stamped at the END of the month they
# average, so the change between consecutive records is the change between consecutive month
# CENTRES, and the flux to compare it against is the mean flux over that same interval. Both
# are built that way below -- a snapshot-minus-snapshot reading of monthly means is wrong.
#
# WHY THE NATIVE GRID. Regridding onto the model grid puts interpolation error into a budget
# that is supposed to close. Nothing here is compared cell-for-cell against model output.
#
# LOGIN-NODE SAFE: NCDatasets only, subset reads, no Oceananigans and no NumericalEarth.
#
# ENV OVERRIDES
#     BSOSE_DIR    directory holding the monthly files
#     START_DATE   default 2014-06-01        END_DATE   default 2015-02-01
#     ZTOP         budget layer depth, m, default 100
#     OUTDIR       figures, default tools/bsose_comparison
# ==============================================================================

using NCDatasets, Dates, Printf, Statistics, CairoMakie

const LON   = (90.0, 150.0)
const LAT   = (-70.0, -45.0)
const ρ₀    = 1026.0        # same value check_freshwater_flux.jl uses
const cₚ    = 3994.0        # J/(kg K), MITgcm's HeatCapacity_Cp
const START_DATE = DateTime(get(ENV, "START_DATE", "2014-06-01"))
const END_DATE   = DateTime(get(ENV, "END_DATE",   "2015-02-01"))
const ZTOP   = parse(Float64, get(ENV, "ZTOP", "100"))
const OUTDIR = get(ENV, "OUTDIR", joinpath(@__DIR__, "bsose_comparison"))
mkpath(OUTDIR)

# Latitude bands, the same ones compare_bsose.jl reports.
const BANDS = [(-70.0, -65.0), (-65.0, -60.0), (-60.0, -55.0), (-55.0, -50.0), (-50.0, -45.0)]

# ── file access, copied from check_freshwater_flux.jl so this stays import-free ───────────────
function bsose_directory()
    haskey(ENV, "BSOSE_DIR") && isdir(ENV["BSOSE_DIR"]) && return abspath(ENV["BSOSE_DIR"])
    for d in ("/storage/scratch1/1/vnguyen480/SOWinWaRP/data",
              normpath(joinpath(@__DIR__, "..", "data")), "data")
        isdir(d) && return abspath(d)
    end
    error("No BSOSE data directory found. Set BSOSE_DIR.")
end

const DIR = bsose_directory()
path(token) = joinpath(DIR, "$(token)_bsoseI156_2013to2024_monthly.nc")

function require(token)
    p = path(token)
    isfile(p) || error("Missing $(basename(p)) in $DIR.")
    filesize(p) > 0 || error("$(basename(p)) is zero bytes -- the download has not finished.")
    return p
end
have(token) = (p = path(token); isfile(p) && filesize(p) > 0)

bsose_averaged_month(stamp) = round(DateTime(stamp), Month) - Month(1)
bsose_window_center(stamp)  = bsose_averaged_month(stamp) + Day(15)

"""First of `names` that exists in `ds`, or an error naming all of them."""
function pick(ds, names...)
    for n in names
        haskey(ds, n) && return n
    end
    error("None of $(join(names, ", ")) found. Have: $(join(sort(collect(keys(ds))), ", "))")
end

num(A) = Float64.(coalesce.(A, NaN))

println("B-SOSE directory: ", DIR)

# ── open the flux files ───────────────────────────────────────────────────────────────────────
tf_ds = NCDataset(require("surfTflx"))
sw_ds = NCDataset(require("oceQsw"))
sf_ds = NCDataset(require("surfSflx"))
fw_ds = NCDataset(require("oceFWflx"))
has_TS = have("Theta") && have("Salt")
th_ds = has_TS ? NCDataset(require("Theta")) : nothing
sa_ds = has_TS ? NCDataset(require("Salt"))  : nothing
has_TS || @warn "No Theta/Salt here -- sections 3 and 4 (the budget) are skipped."

const V_TFLUX = pick(tf_ds, "TFLUX")
const V_QSW   = pick(sw_ds, "oceQsw")
const V_SFLUX = pick(sf_ds, "SFLUX")
const V_FW    = pick(fw_ds, "oceFWflx")

# ── geometry ──────────────────────────────────────────────────────────────────────────────────
xc = num(fw_ds["XC"][:]); yc = num(fw_ds["YC"][:])
iu = findall(x -> LON[1] <= x <= LON[2], xc)
ju = findall(y -> LAT[1] <= y <= LAT[2], yc)
(isempty(iu) || isempty(ju)) && error("No BSOSE cells in $LON x $LAT.")
lat   = yc[ju]
area  = num(fw_ds["rA"][iu, ju])
depth = num(fw_ds["Depth"][iu, ju])
wet   = isfinite.(depth) .& (depth .> 0) .& isfinite.(area)
ni, nj = length(iu), length(ju)

@printf("domain %d x %d cells, %.2f-%.2fE %.2f-%.2fS, %d wet (%.0f%%)\n",
        ni, nj, xc[first(iu)], xc[last(iu)], -lat[1], -lat[end], count(wet),
        100count(wet) / length(wet))

# Vertical grid and partial cell heights, for the layer integral.
if has_TS
    drF   = num(th_ds[pick(th_ds, "drF")][:])
    hFacC = num(th_ds[pick(th_ds, "hFacC")][iu, ju, :])
    nk    = length(drF)

    # Thickness of each cell lying inside 0..ZTOP, partial cells included.
    wz = zeros(ni, nj, nk)
    for i in 1:ni, j in 1:nj
        zt = 0.0
        for k in 1:nk
            h = drF[k] * hFacC[i, j, k]
            h <= 0 && break
            if zt + h <= ZTOP
                wz[i, j, k] = h
            elseif zt < ZTOP
                wz[i, j, k] = ZTOP - zt
                break
            else
                break
            end
            zt += h
        end
    end
    thick = dropdims(sum(wz, dims = 3), dims = 3)
    # Only columns that hold the WHOLE layer enter the budget: a shallower one would charge
    # bathymetry to the circulation.
    deep = wet .& (thick .> 0.99ZTOP)
    @printf("budget layer 0-%.0f m: %d of %d wet columns are deep enough (%.0f%%)\n",
            ZTOP, count(deep), count(wet), 100count(deep) / count(wet))
end

# ── records shared by every file, inside the window ───────────────────────────────────────────
months_of(ds) = Dict(bsose_averaged_month(s) => n for (n, s) in enumerate(ds["time"][:]))
idx = Dict(:tf => months_of(tf_ds), :sw => months_of(sw_ds),
           :sf => months_of(sf_ds), :fw => months_of(fw_ds))
if has_TS
    idx[:th] = months_of(th_ds)
    idx[:sa] = months_of(sa_ds)
end

# intersect over the KEYS: intersect(::Dict, ::Dict) would compare key=>value pairs instead.
months = sort(collect(intersect((keys(d) for d in values(idx))...)))
months = filter(m -> START_DATE <= m + Day(15) <= END_DATE, months)
isempty(months) && error("No records shared by all files inside $START_DATE..$END_DATE.")
nm = length(months)
labels = Dates.format.(months, "yyyy-mm")
@printf("%d shared monthly records, %s to %s\n\n", nm, first(labels), last(labels))

# ══════════════════════════════════════════════════════════════════════════════════════════════
# 0. AUDIT -- units, signs and ranges, before anything is believed
# ══════════════════════════════════════════════════════════════════════════════════════════════
println("="^100)
println("0. AUDIT: what these files claim to be, and whether the signs come out right")
println("="^100)
for (lbl, ds, v) in (("surfTflx", tf_ds, V_TFLUX), ("oceQsw", sw_ds, V_QSW),
                     ("surfSflx", sf_ds, V_SFLUX), ("oceFWflx", fw_ds, V_FW))
    a = ds[v].attrib
    @printf("  %-9s %-9s units=%-12s %s\n", lbl, v,
            get(a, "units", "?"), get(a, "long_name", "?"))
end
println()
@printf("  conversions in use: rho0=%.0f kg/m^3, cp=%.0f J/(kg K) -> rho0*cp = %.3e J/(m^3 K)\n",
        ρ₀, cₚ, ρ₀ * cₚ)
@printf("  so 10 W/m^2 for 30 days over 100 m = %.3f degC\n", 10 * 2.592e6 / (ρ₀ * cₚ * 100))
@printf("  and 1e-5 kg/m^2/s of freshwater, same window, at S=34 = %.4f g/kg\n\n",
        -34 * 1e-5 * 2.592e6 / (ρ₀ * 100))

# ══════════════════════════════════════════════════════════════════════════════════════════════
# load the monthly series
# ══════════════════════════════════════════════════════════════════════════════════════════════
wmean(A, mask) = (sel = mask .& isfinite.(A); count(sel) == 0 ? NaN :
                  sum(A[sel] .* area[sel]) / sum(area[sel]))

# zonal (area-weighted) mean over wet cells at each latitude row
function zonal(A, mask)
    out = fill(NaN, nj)
    for j in 1:nj
        num_ = den = 0.0
        for i in 1:ni
            (mask[i, j] && isfinite(A[i, j])) || continue
            num_ += A[i, j] * area[i, j]; den += area[i, j]
        end
        den > 0 && (out[j] = num_ / den)
    end
    return out
end

TFLUX = zeros(ni, nj, nm); QSW = zeros(ni, nj, nm)
SFLUX = zeros(ni, nj, nm); FW  = zeros(ni, nj, nm); SURF_S = zeros(ni, nj, nm)
HC = has_TS ? zeros(ni, nj, nm) : nothing      # heat content of the layer, J/m^2
SC = has_TS ? zeros(ni, nj, nm) : nothing      # salt content of the layer, g/m^2

for (n, m) in enumerate(months)
    TFLUX[:, :, n] = num(tf_ds[V_TFLUX][iu, ju, idx[:tf][m]])
    QSW[:, :, n]   = num(sw_ds[V_QSW][iu, ju, idx[:sw][m]])
    SFLUX[:, :, n] = num(sf_ds[V_SFLUX][iu, ju, idx[:sf][m]])
    FW[:, :, n]    = num(fw_ds[V_FW][iu, ju, idx[:fw][m]])

    if has_TS
        T = num(th_ds[pick(th_ds, "THETA")][iu, ju, :, idx[:th][m]])
        S = num(sa_ds[pick(sa_ds, "SALT")][iu, ju, :, idx[:sa][m]])
        SURF_S[:, :, n] = map(s -> isfinite(s) && s > 0 ? s : 34.0, S[:, :, 1])
        for i in 1:ni, j in 1:nj
            deep[i, j] || continue
            ht = st = 0.0
            for k in axes(wz, 3)
                w = wz[i, j, k]; w <= 0 && continue
                isfinite(T[i, j, k]) && (ht += T[i, j, k] * w)
                isfinite(S[i, j, k]) && (st += S[i, j, k] * w)
            end
            HC[i, j, n] = ρ₀ * cₚ * ht
            SC[i, j, n] = ρ₀ * st
        end
    else
        SURF_S[:, :, n] .= 34.0
    end
end

NONSOLAR = TFLUX .- QSW
VIRT     = .-SURF_S .* FW          # virtual salt flux, g/m^2/s
SALTTOT  = SFLUX .+ VIRT

# sign sanity: summer in the north should warm
jn = findall(y -> y >= -50, lat)
dec = findfirst(m -> month(m) == 12, months)
if dec !== nothing && !isempty(jn)
    v = wmean(TFLUX[:, :, dec], wet .& reshape([y >= -50 for y in lat], 1, :))
    @printf("  SIGN CHECK  Dec TFLUX north of 50S = %+.1f W/m^2 -- %s\n\n", v,
            v > 0 ? "positive, warming: convention is +into ocean, as assumed" :
                    "NEGATIVE. The sign convention is not what model.jl assumes. STOP HERE.")
end

# ══════════════════════════════════════════════════════════════════════════════════════════════
# 1. HOVMOLLER -- every forcing term, latitude x time
# ══════════════════════════════════════════════════════════════════════════════════════════════
println("="^100)
println("1. THE FORCING ITSELF: zonal-mean of each term, latitude x time")
println("="^100)

panels = [("TFLUX (total)", TFLUX, "W/m²"), ("oceQsw (solar)", QSW, "W/m²"),
          ("TFLUX − oceQsw (non-solar)", NONSOLAR, "W/m²"),
          ("SFLUX (brine)", SFLUX, "g/m²/s"), ("−S·oceFWflx (dilution)", VIRT, "g/m²/s"),
          ("salt total", SALTTOT, "g/m²/s")]

fig1 = Figure(size = (1500, 900))
Label(fig1[0, 1:6], "B-SOSE surface forcing, $(first(labels)) to $(last(labels)), " *
      "zonal mean over $(Int(LON[1]))–$(Int(LON[2]))°E", fontsize = 18, font = :bold)
for (p, (ttl, A, un)) in enumerate(panels)
    Z = hcat([zonal(A[:, :, n], wet) for n in 1:nm]...)'       # (nm, nj)
    fin = filter(isfinite, Z)
    lim = isempty(fin) ? 1.0 : maximum(abs, fin)
    r, c = divrem(p - 1, 3) .+ (1, 1)
    ax = Axis(fig1[r, 2c - 1], title = ttl, xlabel = "month", ylabel = "latitude (°N)",
              xticks = (1:max(1, nm ÷ 6):nm, labels[1:max(1, nm ÷ 6):nm]),
              xticklabelrotation = π / 4)
    hm = heatmap!(ax, 1:nm, lat, Z, colormap = :balance, colorrange = (-lim, lim),
                  nan_color = :gray85)
    Colorbar(fig1[r, 2c], hm, label = un)
end
f1 = joinpath(OUTDIR, "bsose_surface_forcing_hovmoller.png")
save(f1, fig1); @info "wrote $f1"

# ══════════════════════════════════════════════════════════════════════════════════════════════
# 2. THE SAME FORCING IN UNITS YOU CAN COMPARE TO A BIAS
# ══════════════════════════════════════════════════════════════════════════════════════════════
println()
println("="^100)
@printf("2. IMPLIED TENDENCY if the flux went into a fixed layer (per 30 days)\n")
println("   degC/month and (g/kg)/month. This is what makes +130 W/m2 commensurable with +0.5 C.")
println("="^100)
secs = 30 * 86400.0
for h in (25.0, 50.0, 100.0)
    @printf("\n  layer 0-%.0f m\n", h)
    @printf("  %-9s |", "month")
    for (lo, hi) in BANDS; @printf("%9s", @sprintf("%.0f-%.0fS", -lo, -hi)); end
    @printf("   |")
    for (lo, hi) in BANDS; @printf("%9s", @sprintf("%.0f-%.0fS", -lo, -hi)); end
    println()
    @printf("  %-9s |%s dT (degC/mo) %s|%s dS (g/kg/mo)%s\n", "", " "^8, " "^8, " "^9, " "^9)
    println("  " * "-"^(12 + 2 * 9 * length(BANDS) + 4))
    for n in 1:nm
        @printf("  %-9s |", labels[n])
        for (lo, hi) in BANDS
            msk = wet .& reshape([lo <= y < hi for y in lat], 1, :)
            @printf("%9.3f", wmean(TFLUX[:, :, n], msk) * secs / (ρ₀ * cₚ * h))
        end
        @printf("   |")
        for (lo, hi) in BANDS
            msk = wet .& reshape([lo <= y < hi for y in lat], 1, :)
            @printf("%9.4f", wmean(SALTTOT[:, :, n], msk) * secs / (ρ₀ * h))
        end
        println()
    end
end

# ══════════════════════════════════════════════════════════════════════════════════════════════
# 3 & 4. THE BUDGET -- implied minus actual = what the circulation removes
# ══════════════════════════════════════════════════════════════════════════════════════════════
if has_TS
    println()
    println("="^100)
    @printf("3. BUDGET over 0-%.0f m: flux implied vs B-SOSE's own change\n", ZTOP)
    if ZTOP < 250
        @printf("   !! ZTOP = %.0f m is INSIDE the winter mixed layer (B-SOSE runs 177-215 m here\n", ZTOP)
        println("      in Jun-Oct), so the residual is mostly VERTICAL exchange with the water just")
        println("      below. Rerun with ZTOP=300 to enclose it; the difference is the vertical part.")
    else
        @printf("   ZTOP = %.0f m encloses B-SOSE's winter mixed layer (177-215 m Jun-Oct), so the\n", ZTOP)
        println("   residual here is lateral convergence plus exchange ACROSS this depth, not the")
        println("   mixed layer trading with the water below it. Compare against a ZTOP=100 run.")
    end
    println("   Both in FLUX UNITS, so they read against each other directly.")
    println("     implied  = mean surface flux over the interval between month centres")
    println("     actual   = d/dt of the layer's heat/salt content over the same interval")
    println("     residual = implied - actual = removed by advection and mixing")
    println("   A residual near the implied value means the ocean exports essentially all of it.")
    println("="^100)

    nt = nm - 1
    Qimp = zeros(ni, nj, nt); Qact = zeros(ni, nj, nt)
    Simp = zeros(ni, nj, nt); Sact = zeros(ni, nj, nt)
    tlab = String[]
    for n in 1:nt
        Δt = Dates.value(Second((months[n + 1] + Day(15)) - (months[n] + Day(15))))
        push!(tlab, labels[n] * "→" * labels[n + 1][end - 1:end])
        for i in 1:ni, j in 1:nj
            deep[i, j] || continue
            Qimp[i, j, n] = 0.5 * (TFLUX[i, j, n] + TFLUX[i, j, n + 1])
            Simp[i, j, n] = 0.5 * (SALTTOT[i, j, n] + SALTTOT[i, j, n + 1])
            Qact[i, j, n] = (HC[i, j, n + 1] - HC[i, j, n]) / Δt
            Sact[i, j, n] = (SC[i, j, n + 1] - SC[i, j, n]) / Δt
        end
    end
    Qres = Qimp .- Qact
    Sres = Simp .- Sact

    for (name, imp, act, res, un, fmt) in (("HEAT", Qimp, Qact, Qres, "W/m²", "%10.2f"),
                                           ("SALT", Simp, Sact, Sres, "g/m²/s", "%10.3e"))
        @printf("\n  %s (%s)\n", name, un)
        @printf("  %-14s |", "interval")
        for (lo, hi) in BANDS; @printf("%11s", @sprintf("%.0f-%.0fS", -lo, -hi)); end
        @printf("%11s\n", "all")
        println("  " * "-"^(17 + 11 * (length(BANDS) + 1)))
        wide = (fmt == "%10.2f")
        cell(v) = isnan(v) ? lpad("---", 11) :
                  lpad(wide ? @sprintf("%.2f", v) : @sprintf("%.3e", v), 11)
        for n in 1:nt
            for (tag, A) in (("implied", imp), ("actual", act), ("residual", res))
                @printf("  %-14s |", tag == "implied" ? tlab[n] : "  " * tag)
                for (lo, hi) in BANDS
                    msk = deep .& reshape([lo <= y < hi for y in lat], 1, :)
                    print(cell(wmean(A[:, :, n], msk)))
                end
                print(cell(wmean(A[:, :, n], deep)))
                println()
            end
            println()
        end
        # the headline: what fraction of the forcing is exported
        # Signed run means first. Do NOT ratio them: the surface flux is a seasonal cycle whose
        # signed mean is near zero, so implied/residual ratios blow up and mean nothing.
        si = [wmean(imp[:, :, n], deep) for n in 1:nt]
        sa = [wmean(act[:, :, n], deep) for n in 1:nt]
        sr = [wmean(res[:, :, n], deep) for n in 1:nt]
        @printf("  run mean  : implied %s   actual %s   residual %s\n",
                strip(cell(mean(si))), strip(cell(mean(sa))), strip(cell(mean(sr))))
        @printf("  typical   : mean|implied| %s   mean|residual| %s -> the circulation moves\n",
                strip(cell(mean(abs, si))), strip(cell(mean(abs, sr))))
        @printf("              %.0f%% as much as the surface does, %s\n",
                100 * mean(abs, sr) / mean(abs, si),
                mean(sr) * mean(si) < 0 ? "OPPOSING the forcing on average" :
                                          "reinforcing the forcing on average")
    end

    # ── figure: implied / actual / residual, latitude x time ──────────────────────────────────
    fig3 = Figure(size = (1500, 850))
    Label(fig3[0, 1:6], "Upper $(Int(ZTOP)) m budget: surface flux vs B-SOSE's own change " *
          "(residual = exported)", fontsize = 18, font = :bold)
    for (r, (nm_, trio, un)) in enumerate((("heat", (Qimp, Qact, Qres), "W/m²"),
                                           ("salt", (Simp, Sact, Sres), "g/m²/s")))
        Zs = [hcat([zonal(A[:, :, n], deep) for n in 1:nt]...)' for A in trio]
        fin = filter(isfinite, vcat(vec.(Zs)...))
        lim = isempty(fin) ? 1.0 : quantile(abs.(fin), 0.99)
        for (c, (sub, Z)) in enumerate(zip(("implied", "actual", "residual"), Zs))
            ax = Axis(fig3[r, 2c - 1], title = "$nm_ $sub ($un)",
                      xlabel = "interval", ylabel = c == 1 ? "latitude (°N)" : "",
                      xticks = (1:max(1, nt ÷ 4):nt, tlab[1:max(1, nt ÷ 4):nt]),
                      xticklabelrotation = π / 4)
            hm = heatmap!(ax, 1:nt, lat, Z, colormap = :balance,
                          colorrange = (-lim, lim), nan_color = :gray85)
            c == 3 && Colorbar(fig3[r, 6], hm, label = un)
        end
    end
    f3 = joinpath(OUTDIR, "bsose_surface_flux_budget.png")
    save(f3, fig3); @info "wrote $f3"

    # ══════════════════════════════════════════════════════════════════════════════════════════
    # 5. THE S_ref = 34.0 APPROXIMATION
    # ══════════════════════════════════════════════════════════════════════════════════════════
    println()
    println("="^100)
    println("4. THE S_ref = 34.0 APPROXIMATION: -34*oceFWflx against -S_BSOSE*oceFWflx")
    println("   model.jl uses the constant. This is the error that introduces.")
    println("="^100)
    VIRT34 = .-34.0 .* FW
    ERR = VIRT34 .- VIRT
    println("\n  (domain mean over wet cells)")
    @printf("  %-9s |%11s%11s%13s%14s\n", "month", "S mean", "S min", "err g/m²/s",
            "as % of term")
    println("  " * "-"^62)
    for n in 1:nm
        sm = wmean(SURF_S[:, :, n], wet); se = wmean(ERR[:, :, n], wet)
        vt = wmean(VIRT[:, :, n], wet)
        smin = minimum(filter(isfinite, SURF_S[:, :, n][wet]); init = Inf)
        @printf("  %-9s |%11.4f%11.4f%13.3e%13.2f%%\n", labels[n], sm, smin, se,
                abs(vt) > 0 ? 100se / vt : NaN)
    end

    fig4 = Figure(size = (1250, 430))
    Label(fig4[0, 1:4], "Virtual salt flux: constant S_ref = 34.0 minus B-SOSE's own surface S",
          fontsize = 17, font = :bold)
    Em = dropdims(mean(ERR, dims = 3), dims = 3); Em[.!wet] .= NaN
    Sm = dropdims(mean(SURF_S, dims = 3), dims = 3); Sm[.!wet] .= NaN
    for (c, (ttl, A, cmap, un)) in enumerate((("mean surface S", Sm, :haline, "g/kg"),
                                              ("error in the virtual flux", Em, :balance,
                                               "g/m²/s")))
        fin = filter(isfinite, A)
        cr = c == 1 ? extrema(fin) : (-maximum(abs, fin), maximum(abs, fin))
        ax = Axis(fig4[1, 2c - 1], title = ttl, xlabel = "longitude (°E)",
                  ylabel = c == 1 ? "latitude (°N)" : "")
        hm = heatmap!(ax, xc[iu], lat, A, colormap = cmap, colorrange = cr, nan_color = :gray85)
        Colorbar(fig4[1, 2c], hm, label = un)
    end
    f4 = joinpath(OUTDIR, "bsose_sref_error.png")
    save(f4, fig4); @info "wrote $f4"
end

# ══════════════════════════════════════════════════════════════════════════════════════════════
# 6. THE COASTAL CANCELLATION, CELL BY CELL
# ══════════════════════════════════════════════════════════════════════════════════════════════
println()
println("="^100)
println("5. COASTAL CANCELLATION: model.jl records SFLUX -0.140 against -S*FW +0.127 at 70-65S,")
println("   a BAND MEAN that nearly cancels. Where does the cancellation hold, and where not?")
println("="^100)
Sf = dropdims(mean(SFLUX, dims = 3), dims = 3)
Vr = dropdims(mean(VIRT, dims = 3), dims = 3)
Tt = Sf .+ Vr
for A in (Sf, Vr, Tt); A[.!wet] .= NaN; end

fig5 = Figure(size = (1700, 430))
Label(fig5[0, 1:6], "Run-mean salt forcing: brine, dilution, and their sum (g/m²/s)",
      fontsize = 17, font = :bold)
fin = filter(isfinite, vcat(vec(Sf), vec(Vr), vec(Tt)))
lim = isempty(fin) ? 1.0 : quantile(abs.(fin), 0.99)
for (c, (ttl, A)) in enumerate((("SFLUX (brine)", Sf), ("−S·oceFWflx (dilution)", Vr),
                                ("sum", Tt)))
    ax = Axis(fig5[1, 2c - 1], title = ttl, xlabel = "longitude (°E)",
              ylabel = c == 1 ? "latitude (°N)" : "")
    hm = heatmap!(ax, xc[iu], lat, A, colormap = :balance, colorrange = (-lim, lim),
                  nan_color = :gray85)
    c == 3 && Colorbar(fig5[1, 6], hm, label = "g/m²/s")
end
f5 = joinpath(OUTDIR, "bsose_salt_forcing_terms.png")
save(f5, fig5); @info "wrote $f5"

@printf("\n  %-9s |%12s%12s%12s%10s\n", "band", "SFLUX", "-S*FW", "sum", "cancel %")
println("  " * "-"^58)
for (lo, hi) in BANDS
    msk = wet .& reshape([lo <= y < hi for y in lat], 1, :)
    a, b = wmean(Sf, msk), wmean(Vr, msk)
    @printf("  %-9s |%12.3e%12.3e%12.3e%9.0f%%\n", @sprintf("%.0f-%.0fS", -lo, -hi),
            a, b, a + b, abs(a) > 0 ? 100 * (1 - abs(a + b) / max(abs(a), abs(b))) : NaN)
end

# ══════════════════════════════════════════════════════════════════════════════════════════════
# 6. IS THERE A RELAXATION TERM HIDING IN WHAT WE APPLY?
#
# model.jl forces with TFLUX and SFLUX. MITgcm describes those as matching heat- and
# salt-CONTENT variations, which means they contain every term that changed the content --
# including `TRELAX`/`SRELAX`, the restoring of the model's own surface toward observations.
#
# That distinction matters to us. A restoring term depends on B-SOSE's state, not ours. Applying
# it to a model whose surface has drifted is the same frozen-feedback problem as the ice-ocean
# heat flux: we would be handing our model a correction computed for a different ocean.
#
# TRELAX and SRELAX are not published on the data server. But MITgcm's decomposition is
#
#     TFLUX = oceQnet + oceFreez + TRELAX  (+ the heat carried by the freshwater flux)
#     SFLUX = oceSflux          + SRELAX  (+ the salt carried by the freshwater flux)
#
# so the residual RECOVERS them, up to the freshwater-advection terms. Near zero means there is
# no state correction in what we apply and TFLUX/SFLUX are safe to force with. Large means they
# are not, and `oceQnet + oceFreez` / `oceSflux` are the physical-only alternatives.
#
# File token and variable name are NOT always the same here -- surfTflx holds TFLUX -- so both
# are stated explicitly below rather than inferred.
# ══════════════════════════════════════════════════════════════════════════════════════════════
println()
println("="^100)
println("6. IMPLIED RELAXATION: TFLUX - (oceQnet + oceFreez), and SFLUX - oceSflux")
println("   TRELAX/SRELAX are not published; this recovers them by difference.")
println("="^100)

function load_field(token, varname)
    have(token) || (@warn "no $(basename(path(token))) -- skipping it"; return nothing)
    ds = NCDataset(path(token))
    haskey(ds, varname) || (close(ds);
        @warn "$(basename(path(token))) has no \"$varname\""; return nothing)
    mi = months_of(ds)
    all(m -> haskey(mi, m), months) || (close(ds);
        @warn "$token does not cover $(first(labels))..$(last(labels))"; return nothing)
    A = zeros(ni, nj, nm)
    for (n, m) in enumerate(months); A[:, :, n] = num(ds[varname][iu, ju, mi[m]]); end
    close(ds)
    @info "  loaded $token/$varname"
    return A
end

QNET  = load_field("oceQnet",  "oceQnet")
FREEZ = load_field("oceFreez", "oceFreez")
SFLX  = load_field("oceSflux", "oceSflux")

bandmask(lo, hi) = wet .& reshape([lo <= y < hi for y in lat], 1, :)

function report_residual(label, base, parts, partnames, unit, fmt)
    R = copy(base)
    for P in parts; R .-= P; end
    bm = [wmean(base[:, :, n], wet) for n in 1:nm]
    rm_ = [wmean(R[:, :, n], wet) for n in 1:nm]
    @printf("\n  %s - (%s)\n", label, join(partnames, " + "))
    @printf("  %-9s |", "month")
    for (lo, hi) in BANDS; @printf("%12s", @sprintf("%.0f-%.0fS", -lo, -hi)); end
    @printf("%12s\n", "all")
    println("  " * "-"^(12 + 12 * (length(BANDS) + 1)))
    for n in 1:nm
        @printf("  %-9s |", labels[n])
        for (lo, hi) in BANDS
            @printf("%12s", fmt(wmean(R[:, :, n], bandmask(lo, hi))))
        end
        @printf("%12s\n", fmt(rm_[n]))
    end
    frac = 100 * mean(abs, rm_) / mean(abs, bm)
    @printf("\n  mean|%s| = %s,  mean|residual| = %s  ->  %.1f%% of the applied field\n",
            label, fmt(mean(abs, bm)), fmt(mean(abs, rm_)), frac)
    println(frac < 5 ? "  => NEGLIGIBLE. No state correction worth worrying about in $label." :
            "  => NOT negligible. $label carries a term that is not a physical surface flux.")
    return R, frac
end

wfmt(v) = isnan(v) ? "---" : @sprintf("%.2f", v)
sfmt(v) = isnan(v) ? "---" : @sprintf("%.2e", v)

Tres = Sres_ = nothing
if QNET !== nothing
    parts, names = Any[QNET], ["oceQnet"]
    FREEZ === nothing || (push!(parts, FREEZ); push!(names, "oceFreez"))
    Tres, _ = report_residual("TFLUX", TFLUX, parts, names, "W/m²", wfmt)
end
if SFLX !== nothing
    Sres_, _ = report_residual("SFLUX", SFLUX, Any[SFLX], ["oceSflux"], "g/m²/s", sfmt)
end

# oceFreez on its own is the ice-ocean heat flux we have been trying to parameterise.
if FREEZ !== nothing
    fm = [wmean(FREEZ[:, :, n], wet) for n in 1:nm]
    @printf("\n  oceFreez (heating from freezing) domain mean by month, W/m²:\n   ")
    for n in 1:nm; @printf(" %s=%.2f", labels[n][end-1:end], fm[n]); end
    println()
    println("    This is the term a Losch-style freezing relaxation would have to reproduce.")
end

panels6 = Tuple{String,Array{Float64,3},String}[("TFLUX", TFLUX, "W/m²")]
QNET  === nothing || push!(panels6, ("oceQnet", QNET, "W/m²"))
FREEZ === nothing || push!(panels6, ("oceFreez", FREEZ, "W/m²"))
Tres  === nothing || push!(panels6, ("implied TRELAX", Tres, "W/m²"))
panels6s = Tuple{String,Array{Float64,3},String}[("SFLUX", SFLUX, "g/m²/s")]
SFLX  === nothing || push!(panels6s, ("oceSflux", SFLX, "g/m²/s"))
Sres_ === nothing || push!(panels6s, ("implied SRELAX", Sres_, "g/m²/s"))

if length(panels6) > 1 || length(panels6s) > 1
    ncol6 = max(length(panels6), length(panels6s))
    fig6 = Figure(size = (300 * ncol6 + 160, 760))
    Label(fig6[0, 1:(ncol6 + 1)],
          "TFLUX and SFLUX decomposed, zonal mean -- the last panel of each row is what we apply " *
          "that is NOT a physical surface flux", fontsize = 17, font = :bold)
    for (row, group) in enumerate((panels6, panels6s))
        Zs = [hcat([zonal(A[:, :, n], wet) for n in 1:nm]...)' for (_, A, _) in group]
        local fin = filter(isfinite, vcat(vec.(Zs)...))
        local lim = isempty(fin) ? 1.0 : quantile(abs.(fin), 0.99)
        for (c, ((ttl, _, un), Z)) in enumerate(zip(group, Zs))
            ax = Axis(fig6[row, c], title = ttl, xlabel = "month",
                      ylabel = c == 1 ? "latitude (°N)" : "",
                      xticks = (1:max(1, nm ÷ 4):nm, labels[1:max(1, nm ÷ 4):nm]),
                      xticklabelrotation = π / 4)
            hm = heatmap!(ax, 1:nm, lat, Z, colormap = :balance, colorrange = (-lim, lim),
                          nan_color = :gray85)
            c == length(group) && Colorbar(fig6[row, ncol6 + 1], hm, label = un)
        end
    end
    f6 = joinpath(OUTDIR, "bsose_flux_decomposition.png")
    save(f6, fig6); @info "wrote $f6"
end

println("\nReading this:")
println("  * Section 0 must pass before anything else is believed. A wrong sign here is silent.")
println("  * Section 2 turns W/m2 into degC/month, which is the unit our biases are in.")
println("  * Section 3 is the one that matters: if the residual is most of the implied flux,")
println("    the ocean exports nearly everything it receives, and our model must do the same or")
println("    it will drift. If the residual is small, the forcing accumulates locally instead.")
println("  * Section 4 bounds the S_ref = 34.0 approximation in model.jl.")
println("  * Section 5 locates the coastal salting that the Aug-Oct surface maps confirmed.")
println("  * Section 6 is the test of whether TFLUX/SFLUX are safe to force with at all. A large")
println("    residual means we are applying B-SOSE's correction to its OWN state as if it were")
println("    a physical flux -- in which case oceQnet+oceFreez and oceSflux are the alternatives.")
