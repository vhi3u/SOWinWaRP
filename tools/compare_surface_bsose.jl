# ==============================================================================
# compare_surface_bsose.jl -- monthly-mean SURFACE T and S, model minus B-SOSE.
#
#     julia --project=. tools/compare_surface_bsose.jl 2Y_nested_test
#     RUN=2Y_wind_fw_sw julia --project=. tools/compare_surface_bsose.jl
#
# Reads only <RUN>_surface.nc (daily, top level), so it works the same on a static-z run from
# src/model.jl and on a z-star run from examples/sowinwarp_nested.jl: the top level is the top
# level either way, and no 3-D grid has to be rebuilt.
#
# WHAT IS COMPARED. B-SOSE publishes MONTHLY MEANS. The model side is made the same thing: the
# mean of its daily surface snapshots over each calendar month, paired with the B-SOSE record
# whose averaging window is that month. The t = 0 snapshot is the initial condition (B-SOSE
# itself), not model output, and is left out. A month with fewer than MIN_DAYS daily snapshots
# (the partial months at either end of a run) is skipped rather than compared on a few days.
#
# B-SOSE is put onto the model's horizontal grid by NumericalEarth's own regridding, into a
# single layer centred on the depth of the model's TOP CELL (read off the file's z_aac or r_aac),
# so both sides describe the same depth: 2.1 m for a run on B-SOSE's own levels (exactly B-SOSE's
# top cell), 4 m for src/model.jl, 5.4 m for 2Y_nested_test. Both sides are masked to columns
# that are ocean in BOTH bathymetries (tools/bsose_masks.jl explains why `isfinite` masks nothing).
#
# CONVENTIONS. The model runs TEOS10 but is initialised and forced from B-SOSE's potential
# temperature and practical salinity without conversion, so its T and S carry B-SOSE's
# conventions, and the difference is like with like.
#
# OUTPUT (in OUTDIR):
#     <RUN>_surface_bias_T.png       one map per month, model - B-SOSE
#     <RUN>_surface_bias_S.png
#     <RUN>_surface_bias_series.png  domain-mean bias and RMS difference, month by month
# and a table of the same numbers, by latitude band, on stdout.
#
# ENV
#     RUN       run name prefix                    (or the first argument)
#     DIR       where <RUN>_surface.nc lives        (default ".")
#     OUTDIR    figures                            (default "plots")
#     START     model time zero                    (default: the file's start_date attribute)
#     MIN_DAYS  fewest daily snapshots for a month (default 20)
#     T_LIM     colour limit for ΔT maps, degC      (default: 99th percentile of |ΔT|)
#     S_LIM     colour limit for ΔS maps, g/kg      (default: 99th percentile of |ΔS|)
# ==============================================================================

using Oceananigans, Oceananigans.Units, Oceananigans.Grids
using NumericalEarth
using NCDatasets, CairoMakie, Printf, Statistics, Dates

include(joinpath(@__DIR__, "..", "src", "setup_bsose.jl"))
include(joinpath(@__DIR__, "bsose_masks.jl"))

const RUN      = length(ARGS) > 0 ? ARGS[1] : get(ENV, "RUN", "")
isempty(RUN) && error("usage: julia --project=. tools/compare_surface_bsose.jl RUN_NAME")
const DIR      = get(ENV, "DIR", ".")
const OUTDIR   = get(ENV, "OUTDIR", "plots")
const MIN_DAYS = parse(Int, get(ENV, "MIN_DAYS", "20"))
mkpath(OUTDIR)

surf_file = joinpath(DIR, "$(RUN)_surface.nc")
isfile(surf_file) || error("no $surf_file")

# ── the model's surface record ────────────────────────────────────────────────────────────────
ds = Dataset(surf_file)
λc = Float64.(ds["λ_caa"][:])
φc = Float64.(ds["φ_aca"][:])
Nx, Ny = length(λc), length(φc)
times = Float64.(ds["time"][:])

start = if haskey(ENV, "START")
    DateTime(ENV["START"])
elseif haskey(ds.attrib, "start_date")
    DateTime(string(ds.attrib["start_date"]))
else
    error("$surf_file has no start_date attribute; pass START=yyyy-mm-ddTHH:MM")
end
@info "$RUN: $(length(times)) surface snapshots from $start ($(round(maximum(times) / 86400, digits = 1)) days)"

# 4-D (x, y, z = 1, time) as Oceananigans writes a sliced field; 3-D tolerated
slab(var, n) = (v = ds[var]; A = ndims(v) == 4 ? v[:, :, 1, n] : v[:, :, n]; Float64.(coalesce.(A, NaN)))

# Oceananigans writes land as 0, not NaN, so the model's own mask is the only reliable one
haskey(ds, "inactive_nodes_ccc") || error("$surf_file has no inactive_nodes_ccc mask (written with include_grid_metrics = false?)")
model_land = let m = ds["inactive_nodes_ccc"]
    (ndims(m) == 3 ? m[:, :, 1] : m[:, :]) .!= 0
end

# ── calendar months with enough days ──────────────────────────────────────────────────────────
dates = [start + Millisecond(round(Int, 1000t)) for t in times]
months = Dict{Tuple{Int, Int}, Vector{Int}}()
for (n, d) in enumerate(dates)
    times[n] == 0 && continue                       # the initial condition, not model output
    push!(get!(months, (year(d), month(d)), Int[]), n)
end
keys_kept = sort([k for (k, v) in months if length(v) >= MIN_DAYS])
skipped   = sort([k for (k, v) in months if length(v) < MIN_DAYS])
isempty(skipped) || @info "skipping partial month(s) $(join(["$(y)-$(lpad(m, 2, '0')) ($(length(months[(y, m)])) d)" for (y, m) in skipped], ", "))"

# ── the matching B-SOSE records ───────────────────────────────────────────────────────────────
dataset   = BSOSEMonthly(iteration = 156)
available = all_dates(dataset, :temperature)
record_of = Dict((year(bsose_window_center(d)), month(bsose_window_center(d))) => d for d in available)
keys_kept = [k for k in keys_kept if haskey(record_of, k)]
isempty(keys_kept) && error("no month of $RUN has both $MIN_DAYS+ days of output and a B-SOSE record")
sel_dates = [record_of[k] for k in keys_kept]
labels    = [@sprintf("%d-%02d", y, m) for (y, m) in keys_kept]
nm        = length(keys_kept)
@info "comparing $nm months: $(first(labels)) to $(last(labels))"

# One layer centred on the model's top cell. z_aac on a static grid, r_aac (the reference depth)
# on a z-star one; the surface file holds just the one level.
ztop = let name = haskey(ds, "z_aac") ? "z_aac" : haskey(ds, "r_aac") ? "r_aac" : nothing
    isnothing(name) ? (@warn "no z_aac or r_aac in $surf_file; comparing at 5 m"; -5.0) :
                      Float64(only(ds[name][:]))
end
@info @sprintf("model top cell centred at %.1f m; B-SOSE sampled there", -ztop)
Δλ, Δφ = λc[2] - λc[1], φc[2] - φc[1]
grid = LatitudeLongitudeGrid(CPU(); size = (Nx, Ny, 1),
                             longitude = (λc[1] - Δλ / 2, λc[end] + Δλ / 2),
                             latitude = (φc[1] - Δφ / 2, φc[end] + Δφ / 2),
                             z = (2ztop, 0), halo = (3, 3, 3))
region = bsose_region(grid)
resident(v) = FieldTimeSeries(Metadata(v; dataset, dates = sel_dates, region), grid;
                              time_indices_in_memory = nm)
Tb, Sb = resident(:temperature), resident(:salinity)

wet = BitMatrix(.!model_land .& (bsose_depth(dataset, λc, φc) .> 0))
@info "$(count(wet)) of $(Nx * Ny) surface cells are ocean in both bathymetries"

# ── monthly means and differences ─────────────────────────────────────────────────────────────
ΔT = Vector{Matrix{Float64}}(undef, nm)
ΔS = Vector{Matrix{Float64}}(undef, nm)
for (k, key) in enumerate(keys_kept)
    idx = months[key]
    Tm = sum(slab("T", n) for n in idx) ./ length(idx)
    Sm = sum(slab("S", n) for n in idx) ./ length(idx)
    Tp = Array(interior(Tb[k]))[:, :, 1]
    Sp = Array(interior(Sb[k]))[:, :, 1]
    ΔT[k] = ifelse.(wet, Tm .- Tp, NaN)
    ΔS[k] = ifelse.(wet, Sm .- Sp, NaN)
end
close(ds)

"""cos(lat)-weighted mean of `f(x)` over the finite cells of a (Nx, Ny) field in a latitude band."""
function band_stat(A, lo, hi; f = identity)
    num = den = 0.0
    for i in 1:Nx, j in 1:Ny
        (lo <= φc[j] < hi && isfinite(A[i, j])) || continue
        w = cosd(φc[j]); num += w * f(A[i, j]); den += w
    end
    return den == 0 ? NaN : num / den
end
domain_bias(A) = band_stat(A, -90, 90)
domain_rms(A)  = sqrt(band_stat(A, -90, 90; f = abs2))

# ── table ─────────────────────────────────────────────────────────────────────────────────────
bands = [(-70.0, -65.0), (-65.0, -60.0), (-60.0, -55.0), (-55.0, -50.0), (-50.0, -45.0)]
cell(v, fmt) = isnan(v) ? lpad("--", 8) : Printf.format(Printf.Format(fmt), v)

println()
println("="^96)
println("SURFACE MONTHLY MEAN, model minus B-SOSE  ($RUN)")
println("  cos(lat)-weighted over cells that are ocean in both bathymetries. Bias = mean difference,")
println("  RMS = root-mean-square difference. Positive ΔT: model too WARM. Positive ΔS: too SALTY.")
println("="^96)
@printf("  %-8s | %8s %8s | %8s %8s\n", "month", "ΔT", "rms T", "ΔS", "rms S")
for k in 1:nm
    @printf("  %-8s | %s %s | %s %s\n", labels[k],
            cell(domain_bias(ΔT[k]), "%+8.3f"), cell(domain_rms(ΔT[k]), "%8.3f"),
            cell(domain_bias(ΔS[k]), "%+8.3f"), cell(domain_rms(ΔS[k]), "%8.3f"))
end

for (name, Δ, fmt) in (("ΔT (degC)", ΔT, "%+7.2f"), ("ΔS (g/kg)", ΔS, "%+7.3f"))
    println()
    println("  $name by latitude band")
    @printf("  %-8s |", "month")
    for (lo, hi) in bands; @printf(" %7s", "$(Int(-lo))-$(Int(-hi))S"); end
    println()
    for k in 1:nm
        @printf("  %-8s |", labels[k])
        for (lo, hi) in bands; print(" ", cell(band_stat(Δ[k], lo, hi), fmt)); end
        println()
    end
end
println()

# ── figures ───────────────────────────────────────────────────────────────────────────────────
function colour_limit(Δ, key)
    haskey(ENV, key) && return parse(Float64, ENV[key])
    v = filter(isfinite, abs.(reduce(vcat, vec.(Δ))))
    return isempty(v) ? 1.0 : max(quantile(v, 0.99), 1e-6)
end

ncol = min(nm, 6)
nrow = cld(nm, ncol)

for (var, Δ, unit, key) in (("T", ΔT, "°C", "T_LIM"), ("S", ΔS, "g/kg", "S_LIM"))
    lim = colour_limit(Δ, key)
    fig = Figure(size = (260 * ncol + 120, 190 * nrow + 120))
    Label(fig[0, 1:ncol], "$RUN — surface $var, monthly mean, model − B-SOSE", fontsize = 18, font = :bold)
    local hm
    for k in 1:nm
        r, c = fldmod1(k, ncol)
        ax = Axis(fig[r, c], title = labels[k], titlesize = 12,
                  xticklabelsvisible = r == nrow, yticklabelsvisible = c == 1,
                  xticklabelsize = 9, yticklabelsize = 9)
        hm = heatmap!(ax, λc, φc, Δ[k], colormap = :balance, colorrange = (-lim, lim), nan_color = :gray80)
    end
    Colorbar(fig[1:nrow, ncol + 1], hm, label = "Δ$var ($unit)")
    Label(fig[nrow + 1, 1:ncol],
          "red: model too $(var == "T" ? "warm" : "salty"); blue: too $(var == "T" ? "cold" : "fresh"). " *
          "Grey: land in either bathymetry. Colour limit ±$(round(lim, sigdigits = 2)) $unit" *
          (haskey(ENV, key) ? " (set by $key)." : " (99th percentile of |Δ$var|; set $key to fix it)."),
          fontsize = 11)
    f = joinpath(OUTDIR, "$(RUN)_surface_bias_$(var).png"); save(f, fig); @info "wrote $f"
end

fig = Figure(size = (900, 520))
for (row, (var, Δ, unit)) in enumerate((("T", ΔT, "°C"), ("S", ΔS, "g/kg")))
    ax = Axis(fig[row, 1], ylabel = "Δ$var ($unit)", xticks = (1:nm, labels), xticklabelrotation = π / 3,
              xticklabelsvisible = row == 2, title = row == 1 ? "$RUN — surface model − B-SOSE, domain" : "")
    hlines!(ax, [0], color = :gray60)
    scatterlines!(ax, 1:nm, domain_bias.(Δ), color = :firebrick, label = "mean")
    scatterlines!(ax, 1:nm, domain_rms.(Δ), color = :black, linestyle = :dash, label = "RMS")
    row == 1 && axislegend(ax; position = :lt, framevisible = false)
end
f = joinpath(OUTDIR, "$(RUN)_surface_bias_series.png"); save(f, fig); @info "wrote $f"
