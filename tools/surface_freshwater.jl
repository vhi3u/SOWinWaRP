# ==============================================================================================
# surface_freshwater.jl -- fresh water into the ocean by latitude band and month, run against
# B-SOSE, split into what the atmosphere delivers and what the ice exchanges.
#
#     BSOSE_DIR=/storage/scratch1/1/vnguyen480/SOWinWaRP/data julia --project=. tools/surface_freshwater.jl 2Y_nested_bsose_ggl_hj_pr207_icein
#
# WHY. under_ice_heat.jl (_icein): at 60-55S the run's surface is 0.1-0.26 g/kg saltier than B-SOSE's.
# ice_meridional_transport.jl: ice arriving across 60S explains a third of that at most. This asks
# where the rest of B-SOSE's fresh water comes from: its atmosphere (precipitation is adjusted in the
# state estimate) or its ice.
#
# TERMS, all FRESH WATER, kg m⁻² s⁻¹ shown as cm per month (1 cm = 10 kg/m²), + = INTO the ocean or cell:
#   into ocean   B-SOSE oceFWflx; run freshwater_flux (<RUN>_surface_fluxes.nc, m/s) × ρ₀ (RHO0, 1020,
#                NumericalEarth's default). Both include the ice exchange.
#   atmosphere   what the atmosphere (and land) delivers to the CELL, ice and ocean together:
#                B-SOSE SIatmFW (MITgcm seaice_growth.F: all precipitation, open-water evaporation,
#                runoff, sublimation). Run: into ocean − ice-to-ocean + snowfall on the ice.
#   ice          into ocean − atmosphere: melt minus freezing, less the snow piling up on the ice.
#                Run: −(ice_mass_tendency + snow_mass_tendency) − snowfall (<RUN>_sea_ice_fluxes.nc;
#                NumericalEarth's ice-to-ocean fresh water is exactly −(Eᵢ + Eₛ)/ρ₀, and ClimaSeaIce's
#                snow tendency leaves out the snowfall the ice absorbs, which `snowfall` carries).
# The run's `snowfall` is taken as the snowfall the ice absorbs; it is ClimaSeaIce's top snowfall
# flux, so where the ice takes less than it is offered the run's atmosphere term is a little high
# and its ice term a little low, by the same amount (their sum, into ocean, is exact).
# Band means are area-weighted over the run's wet cells; B-SOSE at the nearest cell. B-SOSE 5-day
# records go to the month holding their centre (stamp − 60 h); the run's daily means to the month
# of their middle.
#
# WRITES OUTDIR/<RUN>_surface_freshwater.png: per band, monthly atmosphere and ice terms of both.
# ENV: DIR, OUTDIR, BSOSE_DIR (required), BANDS (-70,-65,-60,-55,-50,-45), RHO0 (1020), WINTER (6,7,8,9,10)
# ==============================================================================================

using NCDatasets, CairoMakie, Dates, Printf, Statistics

ENV["HDF5_USE_FILE_LOCKING"] = "FALSE"

RUN = isempty(ARGS) ? get(ENV, "RUN", "") : ARGS[1]
isempty(RUN) && error("give the run name: julia --project=. tools/surface_freshwater.jl <RUN>")
DIR       = get(ENV, "DIR", ".")
OUTDIR    = get(ENV, "OUTDIR", "plots")
BSOSE_DIR = get(ENV, "BSOSE_DIR", "")
isempty(BSOSE_DIR) && error("set BSOSE_DIR")
EDGES     = parse.(Float64, split(get(ENV, "BANDS", "-70,-65,-60,-55,-50,-45"), ','))
RHO0      = parse(Float64, get(ENV, "RHO0", "1020"))
WINTER    = parse.(Int, split(get(ENV, "WINTER", "6,7,8,9,10"), ','))

g(A) = Float64.(coalesce.(A, NaN))
z(A) = replace(A, NaN => 0.0)
nb = length(EDGES) - 1
band_name(b) = @sprintf("%.0f-%.0fS", -EDGES[b], -EDGES[b+1])
label(ym) = @sprintf("%04d-%02d", ym...)
ym_of(d) = (year(d), month(d))
cm_month(x, ym) = x * daysinmonth(Date(ym...)) * 86400 / 10      # kg m⁻² s⁻¹ -> cm of fresh water per month

# ── the run ──────────────────────────────────────────────────────────────────────────────────
sf_path, si_path, ice_path = (joinpath(DIR, "$(RUN)_$(s).nc") for s in ("surface_fluxes", "sea_ice_fluxes", "sea_ice"))
for p in (sf_path, si_path, ice_path)
    isfile(p) || error("$p not found; set DIR to the run directory")
end
λ, φ, wet, start = NCDataset(ice_path) do d
    g(d["λ_caa"][:]), g(d["φ_aca"][:]), g(d["bottom_height"][:, :]) .< 0, DateTime(d.attrib["start_date"])
end
Nx, Ny = length(λ), length(φ)
band_of = [findfirst(b -> EDGES[b] <= φ[j] < EDGES[b+1], 1:nb) for j in 1:Ny]
W = [wet[i, j] && band_of[j] !== nothing ? cosd(φ[j]) : 0.0 for i in 1:Nx, j in 1:Ny]
bandsum(F) = [sum(W[:, band_of .== b] .* z(F[:, band_of .== b])) / sum(W[:, band_of .== b]) for b in 1:nb]

rd(ds, name, n) = (v = ds[name]; g(ndims(v) == 4 ? v[:, :, 1, n] : v[:, :, n]))
# per band and month: sums and day counts of (into ocean, ice to ocean, snowfall)
acc = Dict{Tuple{Int, Int}, Vector{Float64}}(); days = Dict{Tuple{Int, Int}, Int}()
sf, si = NCDataset(sf_path), NCDataset(si_path)
tf, ti = g(sf["time"][:]), g(si["time"][:])
for (n, t) in enumerate(tf)
    t <= 0 && continue
    m = findfirst(s -> abs(s - t) < 60, ti); m === nothing && continue
    ym = ym_of(start + Millisecond(round(Int, 1000t)) - Hour(12))
    into = bandsum(rd(sf, "freshwater_flux", n) .* RHO0)
    ice = bandsum(-(rd(si, "ice_mass_tendency", m) .+ z(rd(si, "snow_mass_tendency", m))))
    snow = haskey(si, "snowfall") ? bandsum(rd(si, "snowfall", m)) : zeros(nb)
    for b in 1:nb
        a = get!(acc, (b, ym[1] * 100 + ym[2]), zeros(3)); a .+= (into[b], ice[b], snow[b])
    end
    days[(0, ym[1] * 100 + ym[2])] = get(days, (0, ym[1] * 100 + ym[2]), 0) + 1
end
close(sf); close(si)
keys_ym = sort(unique(k[2] for k in keys(days)))
months = [(div(k, 100), k % 100) for k in keys_ym if days[(0, k)] >= daysinmonth(Date(div(k, 100), k % 100)) - 1]
nm = length(months)
R = Dict(q => fill(NaN, nb, nm) for q in (:into, :atm, :ice))
for (k, ym) in enumerate(months), b in 1:nb
    key = ym[1] * 100 + ym[2]; a = acc[(b, key)] ./ days[(0, key)]
    into, ice_oc, snow = a
    R[:into][b, k] = into
    R[:ice][b, k] = ice_oc - snow                 # cell convention: snow kept on the ice is taken from the atmosphere's share
    R[:atm][b, k] = into - R[:ice][b, k]
end

# ── B-SOSE ───────────────────────────────────────────────────────────────────────────────────
function bfile(token, y)
    want = lowercase("$(token)_bsoseI156_$(y)_5day.nc")
    hits = filter(f -> lowercase(f) == want, readdir(BSOSE_DIR))
    isempty(hits) ? "" : joinpath(BSOSE_DIR, first(hits))
end
opens(p) = !isempty(p) && (try NCDataset(_ -> true, p) catch; false end)
pick(d, names) = d[first(k for k in names if haskey(d, k))]
byears = filter(sort(unique(first.(months)))) do y
    bad = [t for t in ("oceFWflx", "SIatmFW") if !opens(bfile(t, y))]
    isempty(bad) || @warn "B-SOSE $y skipped: $(join(bad, ", ")) 5-day file missing or unreadable"
    isempty(bad)
end
B = Dict(q => fill(NaN, nb, nm) for q in (:into, :atm, :ice))
if !isempty(byears)
    xc, yc = NCDataset(bfile("oceFWflx", first(byears))) do d
        x = g(d["XC"][:]); y = g(d["YC"][:]); (ndims(x) == 1 ? x : x[:, 1]), (ndims(y) == 1 ? y : y[1, :])
    end
    ix = [argmin(abs.(xc .- x)) for x in λ]; jy = [argmin(abs.(yc .- y)) for y in φ]
    ir, jr = minimum(ix):maximum(ix), minimum(jy):maximum(jy)
    il, jl = ix .- first(ir) .+ 1, jy .- first(jr) .+ 1
    kof = Dict(ym => k for (k, ym) in enumerate(months))
    sums = Dict(q => zeros(nb, nm) for q in (:into, :atm)); cnt = zeros(Int, nm)
    for y in byears
        df, da = NCDataset(bfile("oceFWflx", y)), NCDataset(bfile("SIatmFW", y))
        F, A = pick(df, ("oceFWflx", "OCEFWFLX")), pick(da, ("SIatmFW", "SIATMFW"))
        stamps = DateTime.(df["time"][:])
        stamps == DateTime.(da["time"][:]) || error("B-SOSE oceFWflx and SIatmFW $y hold different records")
        for (n, s) in enumerate(stamps)
            k = get(kof, ym_of(s - Hour(60)), 0); k == 0 && continue
            sums[:into][:, k] .+= bandsum(g(F[ir, jr, n])[il, jl]); sums[:atm][:, k] .+= bandsum(g(A[ir, jr, n])[il, jl])
            cnt[k] += 1
        end
        close(df); close(da)
    end
    for k in 1:nm
        cnt[k] >= 4 || continue
        B[:into][:, k] = sums[:into][:, k] ./ cnt[k]; B[:atm][:, k] = sums[:atm][:, k] ./ cnt[k]
        B[:ice][:, k] = B[:into][:, k] .- B[:atm][:, k]
    end
end

# ── tables ───────────────────────────────────────────────────────────────────────────────────
bar = "="^120
println("\n", bar)
println("FRESH WATER INTO THE OCEAN, $RUN against B-SOSE -- cm per month, + = into the ocean (or the cell)")
println("  into ocean = atmosphere (to ice + ocean) + ice (melt − freezing − snow kept on the ice)")
println("  B-SOSE: oceFWflx, SIatmFW, their difference; run: freshwater_flux × $(RHO0), the ice and snow mass tendencies, snowfall")
println(bar)
tot(x) = (y = filter(isfinite, x); isempty(y) ? NaN : sum(y))
for b in 1:nb
    @printf("\n  %s\n", band_name(b))
    println("  month    |  into ocean run  B-SOSE |  atmosphere run  B-SOSE |   ice run  B-SOSE | Δ atm   Δ ice  (run − B-SOSE)")
    for (k, ym) in enumerate(months)
        c(q, X) = cm_month(X[q][b, k], ym)
        @printf("  %s  |       %7.2f  %6.2f |       %7.2f  %6.2f | %7.2f  %6.2f | %+6.2f  %+6.2f\n", label(ym),
                c(:into, R), c(:into, B), c(:atm, R), c(:atm, B), c(:ice, R), c(:ice, B),
                c(:atm, R) - c(:atm, B), c(:ice, R) - c(:ice, B))
    end
    wk = [k for k in 1:nm if months[k][2] in WINTER]
    s(q, X) = tot([cm_month(X[q][b, k], months[k]) for k in wk])
    @printf("  months %s, all years: into ocean run %.1f B-SOSE %.1f | atmosphere run %.1f B-SOSE %.1f | ice run %.1f B-SOSE %.1f cm\n",
            join(WINTER, ','), s(:into, R), s(:into, B), s(:atm, R), s(:atm, B), s(:ice, R), s(:ice, B))
end
println()

# ── figure ───────────────────────────────────────────────────────────────────────────────────
mkpath(OUTDIR)
x = 1:nm
ticks = (x[1:3:end], label.(months)[1:3:end])
fig = Figure(size = (1300, 260 * nb + 80), fontsize = 13)
Label(fig[0, 1:2], "$RUN: fresh water into the ocean, cm/month (+ = into)"; font = :bold, fontsize = 15)
for b in 1:nb
    for (col, q, ttl) in ((1, :atm, "from the atmosphere"), (2, :ice, "from the ice (melt − freezing)"))
        ax = Axis(fig[b, col]; title = "$(band_name(b)): $ttl", ylabel = "cm/month", xticks = ticks, xticklabelrotation = π/6)
        hlines!(ax, [0]; color = :grey70)
        lines!(ax, x, [cm_month(R[q][b, k], months[k]) for k in x]; color = :royalblue, linewidth = 2.5, label = "run")
        lines!(ax, x, [cm_month(B[q][b, k], months[k]) for k in x]; color = :black, linestyle = :dash, linewidth = 2, label = "B-SOSE")
        b == 1 && col == 1 && axislegend(ax; position = :lt, backgroundcolor = (:white, 0.8))
    end
end
out = joinpath(OUTDIR, "$(RUN)_surface_freshwater.png"); save(out, fig); @info "wrote $out"
