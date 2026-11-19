# ==============================================================================================
# ice_side_transport.jl -- sea ice volume crossing each open side of the nest, model against
# B-SOSE: is the domain losing more ice through its boundaries than B-SOSE does?
#
#     BSOSE_DIR=/storage/scratch1/1/vnguyen480/SOWinWaRP/data julia --project=. tools/ice_side_transport.jl 2Y_nested_bsose_ggl_hj_pr207
#
# WHY. tools/ice_growth_budget.jl found the domain's ice "drift" (volume change minus freezing and
# melting, which over the whole domain can only be ice crossing the sides) at -350 to -550 km³ per
# June-September, about half the peak ice volume, and the run's fastest ice at the west edge.
#
# HOW. Flux through a face = normal ice velocity × upwind h·ℵ × face length, km³ per day,
# POSITIVE INTO THE DOMAIN, summed along each side (west 90E, east 150E, north 45S, south where
# wet) and accumulated by calendar month.
#   model    <RUN>_sea_ice.nc, DAILY SNAPSHOTS of u, v, h, ℵ. Upwind h·ℵ is the first interior
#            cell (the ghost value is not written), so inflow is approximate. Snapshots, not
#            means: a month's total is good to the extent one instant stands for each day.
#            With <RUN>_sea_ice_fluxes.nc present, the domain's drift from the budget is printed
#            beside the sum of the sides, as a check.
#   B-SOSE   SIuice, SIvice, SIheff 5-day means (BSOSE_DIR). MITgcm C grid: SIuice[i] on the WEST
#            face of cell i, SIvice[j] on the SOUTH face of cell j, same array sizes as XC, YC.
#            Faces placed from XC, YC (grid.nc's XG/YG are corrupt in i156). Upwind SIheff from
#            the cell on the upstream side, which B-SOSE has on both sides of our boundary.
#            A record counts toward the month holding its window's centre (stamp - 60 h).
#   With SIuheff/SIvheff 5-day files for every year (B-SOSE's own ice volume transport per metre,
#   m²/s, on the same faces, accumulated every time step) those are used and B-SOSE's flux is exact.
#   Without them, a product of 5-day means is used, with a warning: not the 5-day mean of the
#   product, so an estimate, low where ice and velocity vary together within 5 days.
#
# WRITES (OUTDIR, default plots/):
#   <RUN>_ice_side_transport.png          monthly net flux per side, model and B-SOSE
#   <RUN>_ice_side_transport_profile.png  where along each side ice leaves, Jun-Oct mean, m²/s
# ENV: DIR, OUTDIR, BSOSE_DIR, WINTER (months for the profile, default 6,7,8,9,10)
# ==============================================================================================

using NCDatasets, CairoMakie, Dates, Printf, Statistics

ENV["HDF5_USE_FILE_LOCKING"] = "FALSE"

RUN = isempty(ARGS) ? get(ENV, "RUN", "") : ARGS[1]
isempty(RUN) && error("give the run name: julia --project=. tools/ice_side_transport.jl <RUN>")
DIR       = get(ENV, "DIR", ".")
OUTDIR    = get(ENV, "OUTDIR", "plots")
BSOSE_DIR = get(ENV, "BSOSE_DIR", "")
WINTER    = parse.(Int, split(get(ENV, "WINTER", "6,7,8,9,10"), ','))
const R = 6.371e6
const KM3 = 1e9

g(A) = Float64.(coalesce.(A, NaN))
z(A) = replace(A, NaN => 0.0)

# ── model ────────────────────────────────────────────────────────────────────────────────────
ice_path = joinpath(DIR, "$(RUN)_sea_ice.nc")
isfile(ice_path) || error("$ice_path not found; set DIR to the run directory")
ds = NCDataset(ice_path)
λ = g(ds["λ_caa"][:]); φ = g(ds["φ_aca"][:])
Nx, Ny = length(λ), length(φ)
wet = g(ds["bottom_height"][:, :]) .< 0
start = DateTime(ds.attrib["start_date"])
ts = g(ds["time"][:])
Nt = length(ts)
Δλ, Δφ = λ[2] - λ[1], φ[2] - φ[1]
λW, λE, φS, φN = λ[1] - Δλ/2, λ[end] + Δλ/2, φ[1] - Δφ/2, φ[end] + Δφ/2
dy = R * deg2rad(Δφ)                                   # length of an x-face
dxN, dxS = R * cosd(φN) * deg2rad(Δλ), R * cosd(φS) * deg2rad(Δλ)

rd(name, n) = (v = ds[name]; z(g(ndims(v) == 4 ? v[:, :, 1, n] : v[:, :, n])))
sides = (:west, :east, :north, :south)
along = Dict(:west => φ, :east => φ, :north => λ, :south => λ)
model_flux = Dict(s => zeros(length(along[s]), Nt) for s in sides)    # m³/s per boundary cell, + in

for n in 1:Nt
    u, v = rd("u", n), rd("v", n)
    V = rd("h", n) .* rd("ℵ", n)
    uW = u[1, :]
    uE = size(u, 1) == Nx + 1 ? u[Nx+1, :] : u[Nx, :]
    vS = v[:, 1]
    vN = size(v, 2) == Ny + 1 ? v[:, Ny+1] : v[:, Ny]
    model_flux[:west][:, n]  = ifelse.(wet[1, :],  uW .* V[1, :] .* dy, 0.0)
    model_flux[:east][:, n]  = ifelse.(wet[Nx, :], -uE .* V[Nx, :] .* dy, 0.0)
    model_flux[:north][:, n] = ifelse.(wet[:, Ny], -vN .* V[:, Ny] .* dxN, 0.0)
    model_flux[:south][:, n] = ifelse.(wet[:, 1],  vS .* V[:, 1] .* dxS, 0.0)
end
close(ds)

datetime(t) = start + Millisecond(round(Int, 1000t))
snap_month(t) = (d = datetime(t); (year(d), month(d)))
months = unique(snap_month.(ts))
filter!(ym -> count(n -> snap_month(ts[n]) == ym, 1:Nt) >= daysinmonth(Date(ym...)) - 1, months)
label(ym) = @sprintf("%04d-%02d", ym...)

# km³ per month: the month's mean flux × its length
monthly(F, recs, ndays) = isempty(recs) ? NaN : mean(sum(F[:, recs], dims = 1)) * ndays * 86400 / KM3
model_in  = Dict(s => [monthly(max.(model_flux[s], 0), [n for n in 1:Nt if snap_month(ts[n]) == ym], daysinmonth(Date(ym...))) for ym in months] for s in sides)
model_out = Dict(s => [monthly(min.(model_flux[s], 0), [n for n in 1:Nt if snap_month(ts[n]) == ym], daysinmonth(Date(ym...))) for ym in months] for s in sides)
model_net = Dict(s => model_in[s] .+ model_out[s] for s in sides)

# the budget's domain drift, as a check
drift = fill(NaN, length(months))
flux_path = joinpath(DIR, "$(RUN)_sea_ice_fluxes.nc")
if isfile(flux_path)
    fs = NCDataset(flux_path); di = NCDataset(ice_path)
    ρᵢ = Float64(get(fs.attrib, "ice_density", 900.0))
    A = g(di["Δx_cca"][:, :]) .* g(di["Δy_cca"][:, :])
    tf = g(fs["time"][:])
    Vsum(n) = (v = di["h"]; a = di["ℵ"]; sum(z(g(v[:, :, n]) .* g(a[:, :, n]))[wet] .* A[wet]))
    Msum(n) = (m = fs["ice_mass_tendency"]; M = z(g(ndims(m) == 4 ? m[:, :, 1, n] : m[:, :, n])); sum(M[wet] .* A[wet]) / ρᵢ)
    wmonth(t) = snap_month(t - 43200)
    for (k, ym) in enumerate(months)
        recs = [n for n in eachindex(tf) if tf[n] > 0 && wmonth(tf[n]) == ym]
        isempty(recs) && continue
        a = findfirst(s -> abs(s - (tf[first(recs)] - 86400)) < 60, ts); e = findfirst(s -> abs(s - tf[last(recs)]) < 60, ts)
        (a === nothing || e === nothing) && continue
        thermo = sum(Msum(n) * 86400 for n in recs)
        drift[k] = (Vsum(e) - Vsum(a) - thermo) / KM3
    end
    close(fs); close(di)
end

# ── B-SOSE ───────────────────────────────────────────────────────────────────────────────────
B = nothing
if !isempty(BSOSE_DIR)
    file(token, y) = joinpath(BSOSE_DIR, "$(token)_bsoseI156_$(y)_5day.nc")
    years = (first(months)[1]):(last(months)[1])
    need = [file(t, y) for y in years for t in ("SIuice", "SIvice", "SIHeff")]
    geom = joinpath(BSOSE_DIR, "surfTflx_bsoseI156_2013to2024_monthly.nc")
    missing_files = filter(!isfile, [need; geom])
    if !isempty(missing_files)
        @warn "B-SOSE not drawn: missing $(join(basename.(missing_files), ", "))"
    else
        gds = NCDataset(geom); xc, yc = g(gds["XC"][:]), g(gds["YC"][:]); depth = g(gds["Depth"][:, :]); close(gds)
        bwet = depth .> 0
        dλB = xc[2] - xc[1]
        # faces: x-face i sits at xc[i] - dλB/2; y-face j sits between yc[j-1] and yc[j]
        iW = argmin(abs.(xc .- dλB/2 .- λW)); iE = argmin(abs.(xc .- dλB/2 .- λE))
        yface = [(yc[j-1] + yc[j]) / 2 for j in 2:length(yc)]
        jN = 1 + argmin(abs.(yface .- φN)); jS = 1 + argmin(abs.(yface .- φS))
        Jr = findall(y -> φS <= y <= φN, yc); Ir = findall(x -> λW <= x <= λE, xc)
        dyB = [R * deg2rad((yc[min(j+1, end)] - yc[max(j-1, 1)]) / (min(j+1, length(yc)) - max(j-1, 1))) for j in eachindex(yc)]
        dxB(j) = R * cosd((yc[j-1] + yc[j]) / 2) * deg2rad(dλB)
        @info @sprintf("B-SOSE faces: west %.3fE, east %.3fE, north %.3fS, south %.3fS",
                       xc[iW] - dλB/2, xc[iE] - dλB/2, -(yc[jN-1] + yc[jN]) / 2, -(yc[jS-1] + yc[jS]) / 2)
        up(u, Vup, Vdn) = u > 0 ? u * Vup : u * Vdn          # upwind along the + direction
        exact = all(y -> isfile(file("SIuheff", y)) && isfile(file("SIvheff", y)), years)
        exact || @warn "no SIuheff/SIvheff 5-day files for every year: B-SOSE flux is velocity × upwind SIheff (an estimate)"
        @info exact ? "B-SOSE flux: SIuheff/SIvheff (exact)" : "B-SOSE flux: SIuice/SIvice × upwind SIheff (estimate)"
        centres = DateTime[]
        bf = Dict(s => Vector{Vector{Float64}}() for s in sides)
        for y in years
            du, dv, dh = NCDataset(file("SIuice", y)), NCDataset(file("SIvice", y)), NCDataset(file("SIHeff", y))
            pick(d, names) = first(k for k in names if haskey(d, k))
            U = d -> d[pick(d, ("SIuice", "SIUICE"))]; Vv = d -> d[pick(d, ("SIvice", "SIVICE"))]
            Hv = dh[pick(dh, ("SIheff", "SIHeff", "SIHEFF"))]
            size(U(du))[1:2] == size(depth) || error("SIuice is $(size(U(du))), XC×YC is $(size(depth)): not the layout assumed")
            stamps = [DateTime(s) for s in du["time"][:]]
            # i156's SIvice (and SIvheff) carry 630 YG rows: rows 1:588 are the cells' south faces, the rest
            # run past the domain; SIuice/SIuheff are XG × YC
            size(Vv(dv), 1) == size(depth, 1) && size(Vv(dv), 2) >= size(depth, 2) ||
                error("SIvice is $(size(Vv(dv))), XC×YC is $(size(depth)): not the layout assumed")
            dqu = exact ? NCDataset(file("SIuheff", y)) : nothing; dqv = exact ? NCDataset(file("SIvheff", y)) : nothing
            QU = exact ? dqu[pick(dqu, ("SIuheff", "SIUHEFF"))] : nothing
            QV = exact ? dqv[pick(dqv, ("SIvheff", "SIVHEFF"))] : nothing
            if exact
                size(QU)[1:2] == size(depth) || error("SIuheff is $(size(QU)), XC×YC is $(size(depth))")
                size(QV, 1) == size(depth, 1) && size(QV, 2) >= size(depth, 2) || error("SIvheff is $(size(QV)), XC×YC is $(size(depth))")
                DateTime.(dqu["time"][:]) == stamps == DateTime.(dqv["time"][:]) || error("SIuheff/SIvheff $y records differ from SIuice's")
            end
            for n in eachindex(stamps)
                push!(centres, stamps[n] - Hour(60))
                uW = z(g(U(du)[iW, Jr, n])); uE = z(g(U(du)[iE, Jr, n]))
                vN = z(g(Vv(dv)[Ir, jN, n])); vS = z(g(Vv(dv)[Ir, jS, n]))
                HW = z(g(Hv[iW-1:iW, Jr, n])); HE = z(g(Hv[iE-1:iE, Jr, n]))
                HN = z(g(Hv[Ir, jN-1:jN, n])); HS = z(g(Hv[Ir, jS-1:jS, n]))
                # per-metre transport on each face: B-SOSE's own (exact) or velocity × upwind SIheff
                qW = exact ? z(g(QU[iW, Jr, n])) : [up(uW[k], HW[1, k], HW[2, k]) for k in eachindex(Jr)]
                qE = exact ? z(g(QU[iE, Jr, n])) : [up(uE[k], HE[1, k], HE[2, k]) for k in eachindex(Jr)]
                qN = exact ? z(g(QV[Ir, jN, n])) : [up(vN[k], HN[k, 1], HN[k, 2]) for k in eachindex(Ir)]
                qS = exact ? z(g(QV[Ir, jS, n])) : [up(vS[k], HS[k, 1], HS[k, 2]) for k in eachindex(Ir)]
                push!(bf[:west],  [bwet[iW, j] ? qW[k] * dyB[j] : 0.0 for (k, j) in enumerate(Jr)])
                push!(bf[:east],  [bwet[iE-1, j] ? -qE[k] * dyB[j] : 0.0 for (k, j) in enumerate(Jr)])
                push!(bf[:north], [bwet[i, jN-1] ? -qN[k] * dxB(jN) : 0.0 for (k, i) in enumerate(Ir)])
                push!(bf[:south], [bwet[i, jS] ? qS[k] * dxB(jS) : 0.0 for (k, i) in enumerate(Ir)])
            end
            close(du); close(dv); close(dh); exact && (close(dqu); close(dqv))
        end
        bmonth(c) = (year(c), month(c))
        Bnet = Dict(s => [monthly(reduce(hcat, bf[s]), [n for n in eachindex(centres) if bmonth(centres[n]) == ym], daysinmonth(Date(ym...))) for ym in months] for s in sides)
        Bin  = Dict(s => [monthly(max.(reduce(hcat, bf[s]), 0), [n for n in eachindex(centres) if bmonth(centres[n]) == ym], daysinmonth(Date(ym...))) for ym in months] for s in sides)
        Bout = Dict(s => [monthly(min.(reduce(hcat, bf[s]), 0), [n for n in eachindex(centres) if bmonth(centres[n]) == ym], daysinmonth(Date(ym...))) for ym in months] for s in sides)
        wrecs = [n for n in eachindex(centres) if month(centres[n]) in WINTER]
        Bprof = Dict(s => mean(reduce(hcat, bf[s])[:, wrecs], dims = 2)[:] for s in sides)
        Balong = Dict(:west => yc[Jr], :east => yc[Jr], :north => xc[Ir], :south => xc[Ir])
        Bseg = Dict(:west => dyB[Jr], :east => dyB[Jr], :north => fill(dxB(jN), length(Ir)), :south => fill(dxB(jS), length(Ir)))
        B = (; net = Bnet, in = Bin, out = Bout, prof = Bprof, along = Balong, seg = Bseg)
    end
end

# ── tables ───────────────────────────────────────────────────────────────────────────────────
nanz(x) = isnan(x) ? 0.0 : x
shown = [s for s in sides if any(!=(0), model_flux[s]) || (B !== nothing && any(!=(0), filter(!isnan, B.net[s])))]
println("\n", "="^110)
println("SEA ICE VOLUME THROUGH THE OPEN SIDES, $RUN -- km³ per month, + = INTO the domain")
println("  in/out/net per side, model (daily snapshots) and B-SOSE (5-day means)")
println("="^110)
for s in shown
    println("\n  ", uppercase(String(s)), " side")
    println("  month    |  model in    out     net |  B-SOSE in    out     net |  net, model - B-SOSE")
    for (k, ym) in enumerate(months)
        b = B === nothing ? (NaN, NaN, NaN) : (B.in[s][k], B.out[s][k], B.net[s][k])
        println(@sprintf("  %s  | %8.1f %7.1f %7.1f | %9.1f %7.1f %7.1f | %8.1f",
                         label(ym), model_in[s][k], model_out[s][k], model_net[s][k], b..., model_net[s][k] - b[3]))
    end
end

println("\n  ALL SIDES")
println("  month    |  model net   budget drift |  B-SOSE net |  model - B-SOSE")
for (k, ym) in enumerate(months)
    m = sum(model_net[s][k] for s in sides)
    b = B === nothing ? NaN : sum(nanz(B.net[s][k]) for s in sides)
    println(@sprintf("  %s  | %10.1f %14.1f | %11.1f | %10.1f", label(ym), m, drift[k], b, m - b))
end
tot(x) = (y = filter(!isnan, x); isempty(y) ? NaN : sum(y))
println(@sprintf("\n  whole record: model %.0f km³ net through the sides, B-SOSE %.0f, budget drift %.0f",
                 tot([sum(model_net[s][k] for s in sides) for k in eachindex(months)]),
                 B === nothing ? NaN : tot([sum(nanz(B.net[s][k]) for s in sides) for k in eachindex(months)]),
                 tot(drift)))
println()

# ── figures ──────────────────────────────────────────────────────────────────────────────────
mkpath(OUTDIR)
x = 1:length(months)
fig = Figure(size = (1100, 250 * (length(shown) + 1) + 80), fontsize = 13)
Label(fig[0, 1:2], "$RUN: sea ice volume through the open sides (km³/month, + = into the domain)"; font = :bold, fontsize = 15)
ticks = (x[1:3:end], label.(months)[1:3:end])
for (r, s) in enumerate([shown; :all])
    ax = Axis(fig[r, 1]; title = s === :all ? "all sides" : "$(s) side", ylabel = "km³/month", xticks = ticks, xticklabelrotation = π/6)
    hlines!(ax, [0]; color = :grey70)
    if s === :all
        lines!(ax, x, [sum(model_net[q][k] for q in sides) for k in x]; color = :royalblue, linewidth = 2.5, label = "model net")
        all(isnan, drift) || lines!(ax, x, drift; color = :darkorange, linewidth = 2, label = "model budget drift")
        B === nothing || lines!(ax, x, [sum(nanz(B.net[q][k]) for q in sides) for k in x]; color = :black, linestyle = :dash, linewidth = 2, label = "B-SOSE net")
    else
        lines!(ax, x, model_in[s]; color = :royalblue, linestyle = :dot, label = "model in")
        lines!(ax, x, model_out[s]; color = :royalblue, linestyle = :dashdot, label = "model out")
        lines!(ax, x, model_net[s]; color = :royalblue, linewidth = 2.5, label = "model net")
        B === nothing || lines!(ax, x, B.net[s]; color = :black, linestyle = :dash, linewidth = 2, label = "B-SOSE net")
    end
    Legend(fig[r, 2], ax; framevisible = false)
end
out1 = joinpath(OUTDIR, "$(RUN)_ice_side_transport.png"); save(out1, fig); @info "wrote $out1"

wn = [n for n in 1:Nt if month(datetime(ts[n])) in WINTER]
seglen = Dict(:west => fill(dy, Ny), :east => fill(dy, Ny), :north => fill(dxN, Nx), :south => fill(dxS, Nx))
fig2 = Figure(size = (1100, 260 * length(shown) + 80), fontsize = 13)
Label(fig2[0, 1:2], "$RUN: where ice crosses each side, months $(join(WINTER, ',')) mean (m²/s per metre of side, + = in)"; font = :bold, fontsize = 15)
for (r, s) in enumerate(shown)
    xlabel = s in (:west, :east) ? "latitude (°N)" : "longitude (°E)"
    ax = Axis(fig2[r, 1]; title = "$(s) side", xlabel, ylabel = "m²/s")
    hlines!(ax, [0]; color = :grey70)
    lines!(ax, along[s], mean(model_flux[s][:, wn], dims = 2)[:] ./ seglen[s]; color = :royalblue, linewidth = 2, label = "model")
    B === nothing || lines!(ax, B.along[s], B.prof[s] ./ B.seg[s]; color = :black, linestyle = :dash, linewidth = 2, label = "B-SOSE")
    Legend(fig2[r, 2], ax; framevisible = false)
end
out2 = joinpath(OUTDIR, "$(RUN)_ice_side_transport_profile.png"); save(out2, fig2); @info "wrote $out2"
