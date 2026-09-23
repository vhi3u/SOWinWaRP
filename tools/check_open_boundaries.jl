# ==============================================================================
# check_open_boundaries.jl
#
# Does fluid actually leave through the open boundaries, or does the domain behave
# like a bounded box? Three independent checks, all from the surface output of a
# finished run:
#
#   1. Is there flow through the faces at all? A wall holds the boundary-normal
#      velocity at exactly zero; an open boundary does not.
#   2. Does the boundary distort the interior? A reflecting or over-specified
#      boundary shows up as variance collapsing, and as a tracer front building,
#      in the outermost cells — distinct from the gradual damping of the sponge.
#   3. Do the boundary values track BSOSE? On inflow the model should follow the
#      prescribed data closely (nudging works); on outflow it should be free to
#      depart from it (the open boundary is not clamping the solution).
#
# This reads model_surface_fields.nc, so it sees the surface layer only. Net volume
# transport and free-surface pile-up need the boundary diagnostics that
# BOUNDARY_DIAGNOSTICS in src/model.jl writes for a run.
# ==============================================================================

using NCDatasets
using CairoMakie
using Printf
using Dates
using Statistics

surface_file = get(ENV, "SURFACE_FILE", "model_surface_fields.nc")
plot_dir = get(ENV, "OBC_PLOT_DIR", "plots")
start_date = DateTime(get(ENV, "START_DATE", "2014-01-01"))
sponge_width = parse(Float64, get(ENV, "SPONGE_WIDTH", "3.0"))   # degrees, to mark on the profiles
probe_cells = parse(Int, get(ENV, "PROBE_CELLS", "24"))          # how far in from each face to look
compare_bsose = get(ENV, "COMPARE_BSOSE", "1") != "0"

mkpath(plot_dir)

isfile(surface_file) || error("Surface file '$surface_file' not found! Set SURFACE_FILE.")

println("Loading surface fields from: ", surface_file)
ds = Dataset(surface_file)
λc = Float64.(ds["λ_caa"][:])
λf = Float64.(ds["λ_faa"][:])
φc = Float64.(ds["φ_aca"][:])
φf = Float64.(ds["φ_afa"][:])
times = Float64.(ds["time"][:])
Nx, Ny, Nt = length(λc), length(φc), length(times)
days = times ./ 86400

mask_c = ds["inactive_nodes_ccc"][:, :, 1] .!= 0   # true on land

np = max(2, min(probe_cells, Nx ÷ 3, Ny ÷ 3))
Δλ = λf[2] - λf[1]
Δφ = φf[2] - φf[1]

println(@sprintf("  - Grid: Nx = %d, Ny = %d, %d snapshots over %.1f days", Nx, Ny, Nt, days[end] - days[1]))
println(@sprintf("  - Probing %d cells in from each face (sponge is %.1f° ≈ %.0f cells in x, %.0f in y)",
    np, sponge_width, sponge_width / Δλ, sponge_width / Δφ))

# ── Per-face strips, in a canonical orientation ──────────────────────────────
# For every face: distance-from-boundary (1 = outermost) × along-boundary × time.
#   normal      : boundary-normal velocity, positive OUTWARD, on faces 1..np+1
#   tangential  : the along-boundary velocity component in the outermost np cells
#   T, S        : tracers in the outermost np cells
# The normal velocity is a Face quantity, so index 1 of `normal` sits exactly on the
# boundary itself — that is the value the open boundary condition sets.
face_names = (:west, :east, :south, :north)

normal = Dict{Symbol,Array{Float32,3}}()
tangential = Dict{Symbol,Array{Float32,3}}()
tracer_T = Dict{Symbol,Array{Float32,3}}()
tracer_S = Dict{Symbol,Array{Float32,3}}()
land = Dict{Symbol,BitMatrix}()   # distance × along, true on land

land[:west] = mask_c[1:np, :]
land[:east] = mask_c[Nx:-1:Nx-np+1, :]
land[:south] = permutedims(mask_c[:, 1:np], (2, 1))
land[:north] = permutedims(mask_c[:, Ny:-1:Ny-np+1], (2, 1))

for f in face_names
    along = f in (:west, :east) ? Ny : Nx
    normal[f] = Array{Float32}(undef, np + 1, along, Nt)
    tangential[f] = Array{Float32}(undef, np, along, Nt)
    tracer_T[f] = Array{Float32}(undef, np, along, Nt)
    tracer_S[f] = Array{Float32}(undef, np, along, Nt)
end

print("Reading boundary strips")
for t in 1:Nt
    t % max(1, Nt ÷ 20) == 0 && print(".")
    u = Float32.(ds["u"][:, :, 1, t])   # (Nx+1, Ny)
    v = Float32.(ds["v"][:, :, 1, t])   # (Nx, Ny+1)
    T = Float32.(ds["T"][:, :, 1, t])
    S = Float32.(ds["S"][:, :, 1, t])

    # v and u are centred onto the tracer grid where they are the tangential component
    vc = 0.5f0 .* (v[:, 1:end-1] .+ v[:, 2:end])   # (Nx, Ny)
    uc = 0.5f0 .* (u[1:end-1, :] .+ u[2:end, :])   # (Nx, Ny)

    # west: outward is -u; east: outward is +u
    normal[:west][:, :, t] = -u[1:np+1, :]
    normal[:east][:, :, t] = u[Nx+1:-1:Nx+1-np, :]
    tangential[:west][:, :, t] = vc[1:np, :]
    tangential[:east][:, :, t] = vc[Nx:-1:Nx-np+1, :]
    tracer_T[:west][:, :, t] = T[1:np, :]
    tracer_T[:east][:, :, t] = T[Nx:-1:Nx-np+1, :]
    tracer_S[:west][:, :, t] = S[1:np, :]
    tracer_S[:east][:, :, t] = S[Nx:-1:Nx-np+1, :]

    # south: outward is -v; north: outward is +v
    normal[:south][:, :, t] = permutedims(-v[:, 1:np+1], (2, 1))
    normal[:north][:, :, t] = permutedims(v[:, Ny+1:-1:Ny+1-np], (2, 1))
    tangential[:south][:, :, t] = permutedims(uc[:, 1:np], (2, 1))
    tangential[:north][:, :, t] = permutedims(uc[:, Ny:-1:Ny-np+1], (2, 1))
    tracer_T[:south][:, :, t] = permutedims(T[:, 1:np], (2, 1))
    tracer_T[:north][:, :, t] = permutedims(T[:, Ny:-1:Ny-np+1], (2, 1))
    tracer_S[:south][:, :, t] = permutedims(S[:, 1:np], (2, 1))
    tracer_S[:north][:, :, t] = permutedims(S[:, Ny:-1:Ny-np+1], (2, 1))
end
println(" done")
close(ds)

# Wet points along each boundary, used everywhere below. A face can be entirely land
# (the southern edge runs into Antarctica), in which case there is no boundary to check.
wet = Dict(f => .!land[f][1, :] for f in face_names)
active_faces = filter(f -> any(wet[f]), face_names)

for f in face_names
    f in active_faces || println(@sprintf("  - %s face is entirely land (%d cells): nothing to check there",
        f, length(wet[f])))
end
isempty(active_faces) && error("Every boundary face is land — nothing to check.")

nanmean(x) = (v = filter(!isnan, x); isempty(v) ? NaN : mean(v))
nanstd(x) = (v = filter(!isnan, x); length(v) < 2 ? NaN : std(v))

# ── 1. Is anything crossing the faces? ───────────────────────────────────────
println("\n" * "="^78)
println("1. FLOW THROUGH THE FACES  (surface layer, outward positive)")
println("="^78)
println(@sprintf("%-7s %10s %10s %10s %10s %12s", "face", "mean m/s", "max out", "max in", "|u|max", "% outflow"))

face_mean_out = Dict{Symbol,Vector{Float64}}()
outflow_fraction = Dict{Symbol,Vector{Float64}}()
exactly_zero = Dict{Symbol,Float64}()

for f in active_faces
    edge = normal[f][1, wet[f], :]                       # along × time, on the boundary itself
    face_mean_out[f] = [mean(edge[:, t]) for t in 1:Nt]
    outflow_fraction[f] = [count(>(0), edge[:, t]) / size(edge, 1) for t in 1:Nt]
    exactly_zero[f] = count(iszero, edge) / length(edge)
    println(@sprintf("%-7s %10.4f %10.4f %10.4f %10.4f %11.1f%%",
        f, mean(edge), maximum(edge), minimum(edge), maximum(abs, edge),
        100 * mean(outflow_fraction[f])))
end

println()
for f in active_faces
    z = exactly_zero[f]
    verdict = z > 0.99 ? "WALL: the normal velocity is pinned to zero" :
              z > 0.05 ? "suspicious: many exactly-zero points" :
              "open: flow crosses the boundary"
    println(@sprintf("  %-7s %5.1f%% of wet boundary points are exactly zero  ->  %s", f, 100z, verdict))
end

# ── 2. Does the boundary distort the interior? ───────────────────────────────
println("\n" * "="^78)
println("2. BOUNDARY IMPRINT ON THE INTERIOR")
println("="^78)

# Temporal variability as a function of distance from the boundary. A wall, or a
# tangential velocity clamped to monthly data, kills the variance in the outer cells.
tangential_std = Dict{Symbol,Vector{Float64}}()
normal_std = Dict{Symbol,Vector{Float64}}()
for f in active_faces
    tangential_std[f] = [nanmean([nanstd(tangential[f][d, a, :]) for a in findall(wet[f])]) for d in 1:np]
    normal_std[f] = [nanmean([nanstd(normal[f][d, a, :]) for a in findall(wet[f])]) for d in 1:np]
    inner = nanmean(tangential_std[f][max(1, np-4):np])
    ratio = tangential_std[f][1] / inner
    verdict = ratio < 0.25 ? "CLAMPED: edge variability is a fraction of the interior" :
              ratio < 0.6 ? "damped at the edge (sponge, or a stiff boundary)" :
              "free: edge variability matches the interior"
    println(@sprintf("  %-7s tangential std: edge %.4f m/s, %d cells in %.4f m/s, ratio %.2f  ->  %s",
        f, tangential_std[f][1], np, inner, ratio, verdict))
end

# Tracer build-up: the jump across the outermost cell face, against the typical
# interior jump at the same time. A boundary that cannot pass tracers grows a front.
println()
edge_gradient = Dict{Symbol,Matrix{Float64}}()   # 2 x Nt, rows = T, S; ratio to interior
for f in active_faces
    ratios = zeros(2, Nt)
    for (r, field) in enumerate((tracer_T[f], tracer_S[f]))
        for t in 1:Nt
            edge_jump = nanmean(abs.(field[1, wet[f], t] .- field[2, wet[f], t]))
            interior_jump = nanmean([nanmean(abs.(field[d, wet[f], t] .- field[d+1, wet[f], t]))
                                     for d in (np-4):(np-1)])
            ratios[r, t] = interior_jump == 0 ? NaN : edge_jump / interior_jump
        end
    end
    edge_gradient[f] = ratios
    first_month = nanmean(ratios[1, 1:min(30, Nt)])
    last_month = nanmean(ratios[1, max(1, Nt-29):Nt])
    trend = last_month / first_month
    verdict = last_month > 3 && trend > 1.5 ? "ACCUMULATING: the edge front grows through the run" :
              last_month > 3 ? "a standing edge front, not growing" :
              "no edge front in temperature"
    println(@sprintf("  %-7s T edge/interior gradient: first month %.2f, last month %.2f (x%.2f)  ->  %s",
        f, first_month, last_month, trend, verdict))
end

# ── 3. Model against the BSOSE data the boundaries prescribe ─────────────────
bsose_available = false
bsose_model = Dict{Symbol,Dict{Symbol,Matrix{Float64}}}()   # face => var => (along × time)
bsose_data = Dict{Symbol,Dict{Symbol,Matrix{Float64}}}()

if compare_bsose
    println("\n" * "="^78)
    println("3. MODEL vs BSOSE AT THE BOUNDARY")
    println("="^78)
    try
        include(joinpath(@__DIR__, "..", "src", "setup_bsose.jl"))

        # A single-level grid matching the output's horizontal grid: BSOSE is interpolated
        # onto exactly the cells the model wrote, through the same pipeline that built the
        # boundary conditions, so any mismatch is the model's and not the interpolation's.
        grid2d = LatitudeLongitudeGrid(CPU(); size=(Nx, Ny, 1),
            longitude=(λf[1], λf[end]), latitude=(φf[1], φf[end]),
            z=(-5, 0), halo=(7, 7, 7))

        dataset = BSOSEMonthly()
        available = all_dates(dataset, :temperature)
        year_dates = filter(d -> start_date <= d < start_date + Year(1), available)
        bc_dates = length(year_dates) >= 12 ? year_dates[1:12] : available[1:12]
        println(@sprintf("  Using %s, %d monthly records, %s to %s",
            summary(dataset), length(bc_dates), bc_dates[1], bc_dates[end]))

        region = bsose_region(grid2d)
        series = Dict{Symbol,Any}()
        for (key, name) in ((:T, :temperature), (:S, :salinity), (:u, :u_velocity), (:v, :v_velocity))
            metadata = Metadata(name; dataset, dates=bc_dates, region)
            fts = FieldTimeSeries(metadata, grid2d)
            fts.times .+= model_clock_offset(metadata, start_date)   # onto the model clock
            series[key] = fts
        end

        # Sample BSOSE at every model output time, at the outermost cell of each face
        for f in active_faces
            bsose_model[f] = Dict{Symbol,Matrix{Float64}}()
            bsose_data[f] = Dict{Symbol,Matrix{Float64}}()
            along = f in (:west, :east) ? Ny : Nx
            for var in (:T, :S, :normal)
                bsose_data[f][var] = fill(NaN, along, Nt)
            end
        end

        # Where each face sits in the horizontal, and the sign that makes its normal
        # velocity point out of the domain.
        edge_of = Dict(:west => (A -> A[1, :]), :east => (A -> A[Nx, :]),
            :south => (A -> A[:, 1]), :north => (A -> A[:, Ny]))
        normal_component = Dict(:west => :u, :east => :u, :south => :v, :north => :v)
        outward_sign = Dict(:west => -1, :east => 1, :south => -1, :north => 1)

        for t in 1:Nt
            snapshot = Dict(key => interior(fts[Oceananigans.Units.Time(times[t])], :, :, 1)
                            for (key, fts) in series)
            for f in active_faces
                bsose_data[f][:T][:, t] = edge_of[f](snapshot[:T])
                bsose_data[f][:S][:, t] = edge_of[f](snapshot[:S])
                bsose_data[f][:normal][:, t] = outward_sign[f] .* edge_of[f](snapshot[normal_component[f]])
            end
        end

        for f in active_faces
            bsose_model[f][:T] = Float64.(tracer_T[f][1, :, :])
            bsose_model[f][:S] = Float64.(tracer_S[f][1, :, :])
            bsose_model[f][:normal] = Float64.(normal[f][1, :, :])
        end
        global bsose_available = true

        # The discriminating statistic: agreement on inflow vs on outflow.
        println()
        println(@sprintf("%-7s %-8s %12s %12s %12s", "face", "var", "RMS inflow", "RMS outflow", "bias"))
        for f in active_faces
            w = findall(wet[f])
            outward = bsose_model[f][:normal][w, :]
            inflow = outward .< 0
            outflow = outward .> 0
            for var in (:normal, :T, :S)
                d = bsose_model[f][var][w, :] .- bsose_data[f][var][w, :]
                good = .!isnan.(d)
                rms(sel) = (x = d[sel .& good]; isempty(x) ? NaN : sqrt(mean(abs2, x)))
                println(@sprintf("%-7s %-8s %12.4f %12.4f %12.4f",
                    f, var, rms(inflow), rms(outflow), nanmean(d[good])))
            end
        end
        println("""
        Expected: RMS on inflow small (the boundary data is being followed), RMS on
        outflow larger (the interior solution is leaving freely rather than being
        clamped back to BSOSE). Similar values on both sides means the boundary is
        holding the solution to the data in each direction.""")
    catch e
        println("  Skipping BSOSE comparison: ", first(split(sprint(showerror, e), "\n")))
        println("  (set COMPARE_BSOSE=0 to silence, or BSOSE_DIR to point at the data)")
    end
end

# ── Figures ──────────────────────────────────────────────────────────────────
colors = Dict(:west => :steelblue, :east => :firebrick, :south => :seagreen, :north => :darkorange)

fig1 = Figure(size=(1500, 950), fontsize=14)
Label(fig1[0, 1:2], "Open boundary check — surface layer", fontsize=22, font=:bold)

ax = Axis(fig1[1, 1], title="Face-mean outward normal velocity", xlabel="Day", ylabel="m/s")
for f in active_faces
    lines!(ax, days, face_mean_out[f], color=colors[f], label=string(f))
end
hlines!(ax, [0.0], color=:black, linestyle=:dash)
axislegend(ax, position=:lb, framevisible=false)

ax = Axis(fig1[1, 2], title="Fraction of each face flowing outward", xlabel="Day", ylabel="fraction")
for f in active_faces
    lines!(ax, days, outflow_fraction[f], color=colors[f], label=string(f))
end
hlines!(ax, [0.5], color=:black, linestyle=:dash)
ylims!(ax, 0, 1)

ax = Axis(fig1[2, 1], title="Tangential velocity variability vs distance from boundary",
    xlabel="cells in from the boundary", ylabel="std (m/s)")
for f in active_faces
    lines!(ax, 1:np, tangential_std[f], color=colors[f], label=string(f))
    scatter!(ax, 1:np, tangential_std[f], color=colors[f], markersize=6)
end
vlines!(ax, [sponge_width / Δλ], color=:gray, linestyle=:dot)
text!(ax, sponge_width / Δλ, 0, text=" sponge edge (x)", align=(:left, :bottom), color=:gray, fontsize=11)
axislegend(ax, position=:rb, framevisible=false)

ax = Axis(fig1[2, 2], title="Temperature jump at the edge / typical interior jump",
    xlabel="Day", ylabel="ratio")
for f in active_faces
    lines!(ax, days, edge_gradient[f][1, :], color=colors[f], label=string(f))
end
hlines!(ax, [1.0], color=:black, linestyle=:dash)

out1 = joinpath(plot_dir, "obc_check.png")
save(out1, fig1)
println("\nSaved $out1")

# Hovmöller of the outward normal velocity on each face
fig2 = Figure(size=(1500, 950), fontsize=14)
Label(fig2[0, 1:4], "Outward normal velocity on each boundary (surface)", fontsize=22, font=:bold)
for (n, f) in enumerate(active_faces)
    row, col = divrem(n - 1, 2) .+ (1, 1)
    edge = replace(normal[f][1, :, :], 0.0f0 => NaN32)
    along = f in (:west, :east) ? φc : λc
    lim = maximum(filter(!isnan, abs.(edge)); init=0.1f0)
    axh = Axis(fig2[row, 2col-1], title=string(f),
        xlabel="Day", ylabel=f in (:west, :east) ? "Latitude (°N)" : "Longitude (°E)")
    hm = heatmap!(axh, days, along, permutedims(edge, (2, 1)),
        colormap=:balance, colorrange=(-lim, lim), nan_color=:gray30)
    Colorbar(fig2[row, 2col], hm, label="m/s (outward +)")
end
out2 = joinpath(plot_dir, "obc_hovmoller.png")
save(out2, fig2)
println("Saved $out2")

if bsose_available
    fig3 = Figure(size=(1500, 1250), fontsize=14)
    Label(fig3[0, 1:3], "Boundary values: model vs BSOSE (surface, face mean)", fontsize=22, font=:bold)
    for (row, f) in enumerate(active_faces)
        w = findall(wet[f])
        for (col, (var, unit)) in enumerate(((:normal, "m/s, outward +"), (:T, "°C"), (:S, "PSU")))
            axc = Axis(fig3[row, col], title="$(f) — $(var)", xlabel="Day", ylabel=unit)
            lines!(axc, days, [nanmean(bsose_data[f][var][w, t]) for t in 1:Nt],
                color=:gray40, linewidth=2.5, label="BSOSE")
            lines!(axc, days, [nanmean(bsose_model[f][var][w, t]) for t in 1:Nt],
                color=colors[f], linewidth=1.5, label="model")
            row == 1 && col == 1 && axislegend(axc, position=:lb, framevisible=false)
        end
    end
    out3 = joinpath(plot_dir, "obc_bsose_comparison.png")
    save(out3, fig3)
    println("Saved $out3")
end

println("\nDone.")
