# ==============================================================================================
# ice_meridional_transport.jl -- sea ice volume crossing interior latitude lines (default 65S and
# 60S), model against B-SOSE: does the run send less ice north, to melt and freshen the surface?
#
#     BSOSE_DIR=/storage/scratch1/1/vnguyen480/SOWinWaRP/data julia --project=. tools/ice_meridional_transport.jl 2Y_nested_bsose_ggl_hj_pr207_icein
#
# WHY. tools/under_ice_heat.jl on _icein: at 60-55S the run's surface is 0.1-0.26 g/kg saltier than
# B-SOSE's (its deep layer matches), its top-to-100-290 m density step a third of B-SOSE's, and it
# has almost no ice where B-SOSE has 0.3-0.45 over water ~1 C above freezing. B-SOSE's fresh cap
# looks like melt of ice arriving from the south. This measures that arrival.
#
# HOW. Flux through each y-face on the line = v × upwind h·ℵ × face length, km³ per month,
# POSITIVE NORTHWARD, summed along the line and by longitude sector.
#   model    <RUN>_sea_ice.nc daily SNAPSHOTS of v, h, ℵ; upwind cell from the sign of v.
#   B-SOSE   SIvice (on the SOUTH face of cell j) and SIheff, 5-day means; the face nearest the
#            line; upwind SIheff. A record counts toward the month holding its centre (stamp - 60 h).
#            A product of means: an estimate, low where ice and velocity covary within 5 days.
# The line's face is the nearest one in each grid; both latitudes are printed.
# Also per month: the line-mean h·ℵ and v (to tell "less ice reaches the line" from "it moves
# slower"), and the northward transport as fresh water spread over the band from the line to the
# next line north (or NORTH_EDGE), in cm, with ice salinity S_ICE.
#
# WRITES (OUTDIR, default plots/):
#   <RUN>_ice_meridional_transport.png          monthly net per line, model and B-SOSE
#   <RUN>_ice_meridional_transport_profile.png  where along each line ice crosses, WINTER mean, m²/s
# ENV: DIR, OUTDIR, BSOSE_DIR, LINES (-65,-60), SECTORS (longitude edges, 90,110,130,150),
#      NORTH_EDGE (-55), S_ICE (4, g/kg), WINTER (6,7,8,9,10)
# ==============================================================================================

using NCDatasets, CairoMakie, Dates, Printf, Statistics

ENV["HDF5_USE_FILE_LOCKING"] = "FALSE"

RUN = isempty(ARGS) ? get(ENV, "RUN", "") : ARGS[1]
isempty(RUN) && error("give the run name: julia --project=. tools/ice_meridional_transport.jl <RUN>")
DIR        = get(ENV, "DIR", ".")
OUTDIR     = get(ENV, "OUTDIR", "plots")
BSOSE_DIR  = get(ENV, "BSOSE_DIR", "")
LINES      = sort(parse.(Float64, split(get(ENV, "LINES", "-65,-60"), ',')))
SECTORS    = parse.(Float64, split(get(ENV, "SECTORS", "90,110,130,150"), ','))
NORTH_EDGE = parse(Float64, get(ENV, "NORTH_EDGE", "-55"))
S_ICE      = parse(Float64, get(ENV, "S_ICE", "4"))
WINTER     = parse.(Int, split(get(ENV, "WINTER", "6,7,8,9,10"), ','))
const R, KM3 = 6.371e6, 1e9
const ρᵢ, ρ_fw, S_REF = 900.0, 1000.0, 34.0

g(A) = Float64.(coalesce.(A, NaN))
z(A) = replace(A, NaN => 0.0)
nl, ns = length(LINES), length(SECTORS) - 1
sector_of(x) = findfirst(s -> SECTORS[s] <= x < SECTORS[s+1], 1:ns)
north_of(l) = l < nl ? LINES[l+1] : NORTH_EDGE
lname(l) = @sprintf("%.0fS", -LINES[l])
sname(s) = @sprintf("%.0f-%.0fE", SECTORS[s], SECTORS[s+1])
label(ym) = @sprintf("%04d-%02d", ym...)
band_area(lo, hi, λ0, λ1) = R^2 * deg2rad(λ1 - λ0) * (sind(hi) - sind(lo))
# fresh water (m³) from melting a volume of ice, net of the salt the ice carries
fresh(V) = V * ρᵢ / ρ_fw * (1 - S_ICE / S_REF)

# ── model ────────────────────────────────────────────────────────────────────────────────────
ice_path = joinpath(DIR, "$(RUN)_sea_ice.nc")
isfile(ice_path) || error("$ice_path not found; set DIR to the run directory")
ds = NCDataset(ice_path)
λ = g(ds["λ_caa"][:]); φ = g(ds["φ_aca"][:])
Nx, Ny = length(λ), length(φ)
start = DateTime(ds.attrib["start_date"])
ts = g(ds["time"][:]); Nt = length(ts)
Δλ, Δφ = λ[2] - λ[1], φ[2] - φ[1]
φS = φ[1] - Δφ / 2
# y-face j is the south face of cell j, at φS + (j-1)Δφ
jm = [round(Int, (L - φS) / Δφ) + 1 for L in LINES]
all(j -> 2 <= j <= Ny, jm) || error("a line in $LINES is outside the run's interior ($(φS) to $(φS + Ny*Δφ))")
φm = [φS + (j - 1) * Δφ for j in jm]
dxm = [R * cosd(p) * deg2rad(Δλ) for p in φm]
rd(name, n, js) = (v = ds[name]; z(g(ndims(v) == 4 ? v[:, js, 1, n] : v[:, js, n])))
mflux = zeros(nl, Nx, Nt); mhA = zeros(nl, Nx, Nt); mv = zeros(nl, Nx, Nt)
for n in 1:Nt, l in 1:nl
    j = jm[l]
    v = rd("v", n, j)[1:Nx]
    V = rd("h", n, j-1:j) .* rd("ℵ", n, j-1:j)
    up = ifelse.(v .> 0, V[:, 1], V[:, 2])
    mflux[l, :, n] = v .* up .* dxm[l]
    mhA[l, :, n] = (V[:, 1] .+ V[:, 2]) ./ 2
    mv[l, :, n] = v
end
close(ds)
datetime(t) = start + Millisecond(round(Int, 1000t))
snap_month(t) = (d = datetime(t); (year(d), month(d)))
months = unique(snap_month.(ts))
filter!(ym -> count(n -> snap_month(ts[n]) == ym, 1:Nt) >= daysinmonth(Date(ym...)) - 1, months)
nm = length(months)
mrecs = [[n for n in 1:Nt if snap_month(ts[n]) == ym] for ym in months]
msec = [sector_of(x) for x in λ]

# per line, month (and sector): km³ = mean flux × month length; line means of h·ℵ and v over cells with ice
km3(F, cols, recs, ym) = isempty(recs) ? NaN : mean(sum(F[cols, recs], dims = 1)) * daysinmonth(Date(ym...)) * 86400 / KM3
iced_mean(X, H, cols, recs) = (m = H[cols, recs] .> 0.01; any(m) ? mean(X[cols, recs][m]) : NaN)
M = (; net = [km3(mflux[l, :, :], 1:Nx, mrecs[k], months[k]) for l in 1:nl, k in 1:nm],
       sec = [km3(mflux[l, :, :], findall(==(s), msec), mrecs[k], months[k]) for l in 1:nl, s in 1:ns, k in 1:nm],
       hA  = [isempty(mrecs[k]) ? NaN : mean(mhA[l, :, mrecs[k]]) for l in 1:nl, k in 1:nm],
       v   = [iced_mean(mv[l, :, :], mhA[l, :, :], 1:Nx, mrecs[k]) for l in 1:nl, k in 1:nm],
       prof = [mean(mflux[l, i, [n for n in 1:Nt if month(datetime(ts[n])) in WINTER]]) / dxm[l] for l in 1:nl, i in 1:Nx])

# ── B-SOSE ───────────────────────────────────────────────────────────────────────────────────
B = nothing
if isempty(BSOSE_DIR)
    @warn "BSOSE_DIR not set: model only"
else
    function bfile(token, y)
        want = lowercase("$(token)_bsoseI156_$(y)_5day.nc")
        hits = filter(f -> lowercase(f) == want, readdir(BSOSE_DIR))
        isempty(hits) ? "" : joinpath(BSOSE_DIR, first(hits))
    end
    opens(p) = !isempty(p) && (try NCDataset(_ -> true, p) catch; false end)
    years = filter(first(months)[1]:last(months)[1]) do y
        bad = [t for t in ("SIvice", "SIheff") if !opens(bfile(t, y))]
        isempty(bad) || @warn "B-SOSE $y skipped: $(join(bad, ", ")) 5-day file missing or unreadable"
        isempty(bad)
    end
    if isempty(years)
        @warn "no B-SOSE year readable: model only"
    else
        # cell-centre XC, YC from the SIheff file (SIvice's file carries the face coordinate, not YC; and
        # grid.nc's XG/YG are corrupt in i156), else from the monthly surfTflx file ice_side_transport.jl uses
        has_xy(p) = NCDataset(d -> haskey(d, "XC") && haskey(d, "YC"), p)
        geom = has_xy(bfile("SIheff", first(years))) ? bfile("SIheff", first(years)) :
               joinpath(BSOSE_DIR, "surfTflx_bsoseI156_2013to2024_monthly.nc")
        xc, yc = NCDataset(geom) do d
            g(d["XC"][:]), g(d["YC"][:])
        end
        xc = ndims(xc) == 1 ? xc : xc[:, 1]; yc = ndims(yc) == 1 ? yc : yc[1, :]
        Ir = findall(x -> λ[1] - Δλ/2 <= x <= λ[end] + Δλ/2, xc)
        dλB = xc[2] - xc[1]
        yface = [(yc[j-1] + yc[j]) / 2 for j in 2:length(yc)]
        jb = [1 + argmin(abs.(yface .- L)) for L in LINES]          # SIvice[:, j] is on the south face of cell j
        φb = [(yc[j-1] + yc[j]) / 2 for j in jb]
        dxb = [R * cosd(p) * deg2rad(dλB) for p in φb]
        bsec = [sector_of(x) for x in xc[Ir]]
        cent = DateTime[]; bflux = [Vector{Vector{Float64}}() for _ in 1:nl]
        bhA = [Vector{Vector{Float64}}() for _ in 1:nl]; bv = [Vector{Vector{Float64}}() for _ in 1:nl]
        for y in years
            dv, dh = NCDataset(bfile("SIvice", y)), NCDataset(bfile("SIheff", y))
            pick(d, names) = d[first(k for k in names if haskey(d, k))]
            Vv, Hv = pick(dv, ("SIvice", "SIVICE")), pick(dh, ("SIheff", "SIHeff", "SIHEFF"))
            size(Vv)[1:2] == size(Hv)[1:2] == (length(xc), length(yc)) ||
                error("SIvice $(size(Vv)), SIheff $(size(Hv)), XC×YC ($(length(xc)), $(length(yc))): not the C-grid layout assumed")
            for n in eachindex(dv["time"][:])
                push!(cent, DateTime(dv["time"][n]) - Hour(60))
                for l in 1:nl
                    v = z(g(Vv[Ir, jb[l], n])); H = z(g(Hv[Ir, jb[l]-1:jb[l], n]))
                    push!(bflux[l], v .* ifelse.(v .> 0, H[:, 1], H[:, 2]) .* dxb[l])
                    push!(bhA[l], (H[:, 1] .+ H[:, 2]) ./ 2); push!(bv[l], v)
                end
            end
            close(dv); close(dh)
        end
        brecs = [[n for n in eachindex(cent) if (year(cent[n]), month(cent[n])) == ym] for ym in months]
        wb = [n for n in eachindex(cent) if month(cent[n]) in WINTER]
        F = [reduce(hcat, bflux[l]) for l in 1:nl]; HA = [reduce(hcat, bhA[l]) for l in 1:nl]; VV = [reduce(hcat, bv[l]) for l in 1:nl]
        nIr = length(Ir)
        B = (; net = [km3(F[l], 1:nIr, brecs[k], months[k]) for l in 1:nl, k in 1:nm],
               sec = [km3(F[l], findall(==(s), bsec), brecs[k], months[k]) for l in 1:nl, s in 1:ns, k in 1:nm],
               hA  = [isempty(brecs[k]) ? NaN : mean(HA[l][:, brecs[k]]) for l in 1:nl, k in 1:nm],
               v   = [iced_mean(VV[l], HA[l], 1:nIr, brecs[k]) for l in 1:nl, k in 1:nm],
               prof = [isempty(wb) ? NaN : mean(F[l][i, wb]) / dxb[l] for l in 1:nl, i in 1:nIr],
               x = xc[Ir], φ = φb)
    end
end

# ── tables ───────────────────────────────────────────────────────────────────────────────────
bar = "="^140
println("\n", bar)
println("SEA ICE VOLUME CROSSING LATITUDE LINES, $RUN -- km³ per month, + = NORTHWARD")
println("  model: daily snapshots, v × upwind h·ℵ; B-SOSE: 5-day SIvice × upwind SIheff. h·ℵ: line mean (m); v: mean over iced faces (cm/s)")
println("  fresh water: the month's net northward ice, melted (ice salinity $(S_ICE) g/kg), spread over the band north of the line, cm")
println(bar)
tot(x) = (y = filter(!isnan, x); isempty(y) ? NaN : sum(y))
bget(f, idx...) = B === nothing ? NaN : getfield(B, f)[idx...]
for l in 1:nl
    lo, hi = LINES[l], north_of(l)
    A = band_area(lo, hi, λ[1] - Δλ/2, λ[end] + Δλ/2)
    @printf("\n  %s   (model face %.3f, B-SOSE face %s; fresh water over %.0f-%.0fS, %.2e km²)\n", lname(l), φm[l],
            B === nothing ? "-" : @sprintf("%.3f", B.φ[l]), -lo, -hi, A / 1e6)
    print("  month    |  net model  B-SOSE |  h·ℵ model B-SOSE |   v model B-SOSE | fresh cm model B-SOSE |")
    foreach(s -> @printf(" %s model B-SOSE |", sname(s)), 1:ns); println()
    for k in 1:nm
        @printf("  %s  | %9.1f %7.1f |  %8.3f  %6.3f | %7.2f  %6.2f |   %8.1f  %7.1f |", label(months[k]),
                M.net[l, k], bget(:net, l, k), M.hA[l, k], bget(:hA, l, k), 100 * M.v[l, k], 100 * bget(:v, l, k),
                100 * fresh(M.net[l, k] * KM3) / A, 100 * fresh(bget(:net, l, k) * KM3) / A)
        foreach(s -> @printf(" %15.1f %7.1f |", M.sec[l, s, k], bget(:sec, l, s, k)), 1:ns); println()
    end
    wk = [k for k in 1:nm if months[k][2] in WINTER]
    @printf("  months %s, all years: model %.0f km³, B-SOSE %.0f km³ northward (fresh water %.1f vs %.1f cm)\n",
            join(WINTER, ','), tot(M.net[l, wk]), B === nothing ? NaN : tot(B.net[l, wk]),
            100 * fresh(tot(M.net[l, wk]) * KM3) / A, B === nothing ? NaN : 100 * fresh(tot(B.net[l, wk]) * KM3) / A)
end
println()

# ── figures ──────────────────────────────────────────────────────────────────────────────────
mkpath(OUTDIR)
x = 1:nm
ticks = (x[1:3:end], label.(months)[1:3:end])
fig = Figure(size = (1100, 280 * nl + 80), fontsize = 13)
Label(fig[0, 1:2], "$RUN: sea ice volume crossing latitude lines (km³/month, + = northward)"; font = :bold, fontsize = 15)
for l in 1:nl
    ax = Axis(fig[l, 1]; title = lname(l), ylabel = "km³/month", xticks = ticks, xticklabelrotation = π/6)
    hlines!(ax, [0]; color = :grey70)
    lines!(ax, x, M.net[l, :]; color = :royalblue, linewidth = 2.5, label = "model")
    B === nothing || lines!(ax, x, B.net[l, :]; color = :black, linestyle = :dash, linewidth = 2, label = "B-SOSE")
    Legend(fig[l, 2], ax; framevisible = false)
end
out1 = joinpath(OUTDIR, "$(RUN)_ice_meridional_transport.png"); save(out1, fig); @info "wrote $out1"

fig2 = Figure(size = (1100, 280 * nl + 80), fontsize = 13)
Label(fig2[0, 1:2], "$RUN: where ice crosses each line, months $(join(WINTER, ',')) mean (m²/s per metre of line, + = northward)"; font = :bold, fontsize = 15)
for l in 1:nl
    ax = Axis(fig2[l, 1]; title = lname(l), xlabel = "longitude (°E)", ylabel = "m²/s")
    hlines!(ax, [0]; color = :grey70)
    lines!(ax, λ, M.prof[l, :]; color = :royalblue, linewidth = 2, label = "model")
    B === nothing || lines!(ax, B.x, B.prof[l, :]; color = :black, linestyle = :dash, linewidth = 2, label = "B-SOSE")
    Legend(fig2[l, 2], ax; framevisible = false)
end
out2 = joinpath(OUTDIR, "$(RUN)_ice_meridional_transport_profile.png"); save(out2, fig2); @info "wrote $out2"
