# ==============================================================================
# hammond_jones_freshwater.jl -- the Hammond & Jones Southern Ocean freshwater field (ice-shelf melt,
# calving with iceberg tracks, runoff), on the nest's grid, for the coastal freshwater forcing.
#
#     git clone https://github.com/DaniJonesOcean/southern_fwflux <data>/southern_fwflux    # once
#     julia --project=. tools/hammond_jones_freshwater.jl
#
# THE SOURCE. Hammond & Jones (2016), Geosci. Data J. 3, 60-62, doi:10.1002/gdj3.43, made for SOSE;
# B-SOSE prescribes melt and calving from it. runoff_NVND.bin in the repository above (also at BODC,
# doi:10.5285/376e0b0f-d065-500a-e053-6c86abc0f5e9): one multiyear-mean field, m/yr of fresh water,
# big-endian Float32, 2160 x 320 on SOSE's regular 1/6 deg grid, cell centres (i - 1/2)/6 E and
# -78 + 1/24 + (j - 1/2)/6 N (-77.875 to -24.708; its grid.mat). Constant in time.
#
# THE FIELD FOR THE MODEL. Written as FIELD_FILE (coastal_freshwater_hj.nc) in the format of
# tools/coastal_freshwater.jl's coastal_freshwater_bsose.nc, so the nested scripts' coastal freshwater
# forcing reads either: lon, lat (the nest's cell centres) and freshwater_flux (kg m-2 s-1, + into the
# ocean). Regridded conservatively: the nest's longitudes coincide with SOSE's and its latitudes are
# 1/24 deg off, so each nest cell is the area-weighted mean of the SOSE cells it overlaps, and the
# total is kept. Nothing is moved off land here: that depends on the run's bathymetry, so it is the
# script's job; what lands on our land cells is reported below.
#
# THE COMPARISON. Gt/yr by sector and latitude band, and the largest cells, against the residual
# field the B-SOSE runs have used since fix5 (RESIDUAL_FILE, if present). A map of both is written to
# OUTDIR/coastal_freshwater_hj_vs_residual.png.
#
# LOGIN-NODE SAFE: NCDatasets and CairoMakie only.
#
# ENV
#     HJ_FILE        the binary            (default <data>/southern_fwflux/runoff_NVND.bin)
#     FIELD_FILE     the field to write    (default coastal_freshwater_hj.nc)
#     RESIDUAL_FILE  to compare against    (default coastal_freshwater_bsose.nc; skipped if absent)
#     OUTDIR         figures               (default plots)
# ==============================================================================

using NCDatasets, CairoMakie, Printf

const DATA_DIR = let d = "/storage/scratch1/1/vnguyen480/SOWinWaRP/data"
    isdir(d) ? d : normpath(joinpath(@__DIR__, "..", "data"))
end
const HJ_FILE       = get(ENV, "HJ_FILE", joinpath(DATA_DIR, "southern_fwflux", "runoff_NVND.bin"))
const FIELD_FILE    = get(ENV, "FIELD_FILE", "coastal_freshwater_hj.nc")
const RESIDUAL_FILE = get(ENV, "RESIDUAL_FILE", "coastal_freshwater_bsose.nc")
const OUTDIR        = get(ENV, "OUTDIR", "plots")

# The nest's grid (examples/sowinwarp_nested.jl, sowinwarp_glorys_baseline.jl), which the scripts check
const CHILD_LON, CHILD_LAT, CHILD_SCALING = (90.0, 150.0), (-70.0, -45.0), 6

const R_EARTH      = 6.371e6
const YEAR_SECONDS = 365.25 * 86400
const ρ_FRESH      = 1000.0

# ── read ─────────────────────────────────────────────────────────────────────────────────────
const NX_HJ, NY_HJ = 2160, 320
isfile(HJ_FILE) || error("$HJ_FILE not found: git clone https://github.com/DaniJonesOcean/southern_fwflux into $DATA_DIR")
filesize(HJ_FILE) == 4 * NX_HJ * NY_HJ ||
    error("$HJ_FILE is $(filesize(HJ_FILE)) bytes, not $(4 * NX_HJ * NY_HJ) (2160 x 320 Float32)")
hj = reshape(Float64.(ntoh.(reinterpret(Float32, read(HJ_FILE)))), NX_HJ, NY_HJ)     # m/yr
all(isfinite, hj) && minimum(hj) >= 0 && maximum(hj) < 1000 ||
    error("$HJ_FILE does not read as m/yr (range $(extrema(hj))): byte order or precision is not big-endian Float32")

λ_edges_hj = (0:NX_HJ) ./ 6
φ_edges_hj = -78 + 1/24 .+ (0:NY_HJ) ./ 6

# cell areas from the edges, m²
areas(λe, φe) = [R_EARTH^2 * deg2rad(λe[i+1] - λe[i]) * (sind(φe[j+1]) - sind(φe[j]))
                 for i in 1:length(λe)-1, j in 1:length(φe)-1]
A_hj = areas(λ_edges_hj, φ_edges_hj)
gt(F_myr, A) = sum(F_myr .* A) / 1e9          # m³ of fresh water a year = Gt/yr

# ── regrid conservatively onto the nest ──────────────────────────────────────────────────────
Nx = round(Int, CHILD_SCALING * (CHILD_LON[2] - CHILD_LON[1]))
Ny = round(Int, CHILD_SCALING * (CHILD_LAT[2] - CHILD_LAT[1]))
λ_edges = CHILD_LON[1] .+ (0:Nx) ./ CHILD_SCALING
φ_edges = CHILD_LAT[1] .+ (0:Ny) ./ CHILD_SCALING
λc = CHILD_LON[1] .+ ((1:Nx) .- 0.5) ./ CHILD_SCALING
φc = CHILD_LAT[1] .+ ((1:Ny) .- 0.5) ./ CHILD_SCALING

# W[d, s]: the fraction of destination cell d covered by source cell s, measured by `measure`
# (longitude itself; sin(latitude) for latitude, so the weights are areas)
function overlaps(dst, src, measure)
    W = zeros(length(dst) - 1, length(src) - 1)
    for d in 1:length(dst)-1, s in 1:length(src)-1
        lo, hi = max(dst[d], src[s]), min(dst[d+1], src[s+1])
        hi > lo && (W[d, s] = (measure(hi) - measure(lo)) / (measure(dst[d+1]) - measure(dst[d])))
    end
    all(sum(W; dims = 2) .≈ 1) || error("the nest is not inside the SOSE grid")
    return W
end
Wλ = overlaps(λ_edges, λ_edges_hj, identity)
Wφ = overlaps(φ_edges, φ_edges_hj, sind)
F_myr = Wλ * hj * Wφ'                         # m/yr on the nest
A = areas(λ_edges, φ_edges)

λ_hj = ((1:NX_HJ) .- 0.5) ./ 6
φ_hj = -78 + 1/24 .+ ((1:NY_HJ) .- 0.5) ./ 6
in_nest_hj = [CHILD_LON[1] <= λ_hj[i] <= CHILD_LON[2] && CHILD_LAT[1] <= φ_hj[j] <= CHILD_LAT[2] for i in 1:NX_HJ, j in 1:NY_HJ]
@printf("Hammond & Jones: %.0f Gt/yr in all (SOSE's grid, south of 24.7S); %.0f Gt/yr in the nest's box on SOSE's cells, %.0f on the nest's\n",
        gt(hj, A_hj), gt(hj[in_nest_hj], A_hj[in_nest_hj]), gt(F_myr, A))

F = F_myr .* ρ_FRESH ./ YEAR_SECONDS          # kg m⁻² s⁻¹

NCDataset(FIELD_FILE, "c") do out
    defDim(out, "lon", Nx); defDim(out, "lat", Ny)
    defVar(out, "lon", collect(λc), ("lon",); attrib = ["units" => "degrees_east"])
    defVar(out, "lat", collect(φc), ("lat",); attrib = ["units" => "degrees_north"])
    defVar(out, "freshwater_flux", F, ("lon", "lat");
           attrib = ["units" => "kg m-2 s-1", "positive" => "into the ocean",
                     "long_name" => "Antarctic ice-shelf melt, iceberg melt and runoff (Hammond & Jones 2016)"])
    out.attrib["method"] = "Hammond & Jones (2016) runoff_NVND.bin, conservatively regridded from SOSE's 1/6 deg grid"
    out.attrib["source"] = "https://github.com/DaniJonesOcean/southern_fwflux; doi:10.1002/gdj3.43"
    out.attrib["period"] = "multiyear mean (2010-2015), constant in time"
    out.attrib["total_Gt_per_year"] = gt(F_myr, A)
end
@info "wrote $FIELD_FILE"

# ── compare with the residual ────────────────────────────────────────────────────────────────
residual = nothing
if isfile(RESIDUAL_FILE)
    residual = NCDataset(RESIDUAL_FILE) do ds
        lon, lat = Float64.(ds["lon"][:]), Float64.(ds["lat"][:])
        (length(lon) == Nx && length(lat) == Ny && maximum(abs, lon .- λc) < 1e-6 && maximum(abs, lat .- φc) < 1e-6) ||
            error("$RESIDUAL_FILE is not on the nest's grid")
        Float64.(coalesce.(ds["freshwater_flux"][:, :], 0.0)) .* YEAR_SECONDS ./ ρ_FRESH   # m/yr
    end
else
    @warn "$RESIDUAL_FILE not found: no comparison with the residual field"
end

fields = residual === nothing ? (("H&J", F_myr),) : (("H&J", F_myr), ("residual", residual))
header = join([@sprintf("%10s", name) for (name, _) in fields])

println("\nGt/yr BY SECTOR (all latitudes of the nest)")
println("  sector     ", header)
for a in 90:10:140
    m = [a <= λ < a + 10 for λ in λc, φ in φc]
    println(@sprintf("  %3d-%3dE   ", a, a + 10), join([@sprintf("%10.1f", gt(f[m], A[m])) for (_, f) in fields]))
end
println(@sprintf("  all        "), join([@sprintf("%10.1f", gt(f, A)) for (_, f) in fields]))

println("\nGt/yr BY LATITUDE")
println("  band       ", header)
for (a, b) in ((-70, -66), (-66, -62), (-62, -58), (-58, -54), (-54, -50), (-50, -45))
    m = [a <= φ < b for λ in λc, φ in φc]
    println(@sprintf("  %3d-%3dS   ", -a, -b), join([@sprintf("%10.1f", gt(f[m], A[m])) for (_, f) in fields]))
end

println("\nHOW CONCENTRATED: Gt/yr in cells above each rate, and the largest cells (m/yr)")
println("  rate       ", header)
for r in (0.5, 1, 2, 5, 10, 20)
    println(@sprintf("  > %4.1f     ", r), join([@sprintf("%10.1f", gt(f[f .> r], A[f .> r])) for (_, f) in fields]))
end
for (name, f) in fields
    top = partialsortperm(vec(f), 1:5; rev = true)
    println("  $name largest: ", join([@sprintf("%.1f at %.2fE %.2fS", f[I], λc[I[1]], -φc[I[2]])
                                      for I in CartesianIndices(f)[top]], "; "))
end

# ── map ──────────────────────────────────────────────────────────────────────────────────────
mkpath(OUTDIR)
fig = Figure(size = (1000, 330 * length(fields) + 60))
for (n, (name, f)) in enumerate(fields)
    ax = Axis(fig[n, 1]; title = @sprintf("%s: %.0f Gt/yr, max %.1f m/yr", name, gt(f, A), maximum(f)),
              xlabel = "longitude", ylabel = "latitude", aspect = DataAspect())
    heatmap!(ax, λc, φc, replace(f, 0.0 => NaN); colormap = :viridis, colorrange = (0, 3), highclip = :red)
end
Colorbar(fig[1:length(fields), 2]; colormap = :viridis, colorrange = (0, 3), highclip = :red,
         label = "fresh water, m/yr (red: above 3)")
out = joinpath(OUTDIR, "coastal_freshwater_hj_vs_residual.png")
save(out, fig)
@info "wrote $out"
