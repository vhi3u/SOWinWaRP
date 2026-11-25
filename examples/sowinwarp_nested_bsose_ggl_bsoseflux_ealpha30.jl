# SOWinWaRP (90-150E, 70-45S, 1/6°) nested in B-SOSE, OCEAN ONLY, FORCED BY B-SOSE's OWN SURFACE FLUXES.
# ARM: '_ealpha30' -- CATKE's TKE mixing lengths x30 (E_ALPHA below), under the same B-SOSE fluxes. Run name bsose_ggl_hj_pr207_bsoseflux_ealpha30.
#
# The _ibobc run (examples/sowinwarp_nested_bsose_ggl_icein_mitbulk_mar_iceobc2_ibobc.jl) with the sea ice, the ERA5
# atmosphere and the coupler taken out, and B-SOSE i156's 5-day ocean-surface fluxes put in their place. Same grid,
# bathymetry, levels, closure, nest, boundaries (inverse barometer included) and dates. Run name
# bsose_ggl_hj_pr207_bsoseflux.
#
# WHY. The coastal salinity bias (+0.18 to +0.34 g/kg at the coast, 128E 66.9S at +1.3) did not move with the closure
# (_ealpha10, _ealpha30) and is the same in every configuration so far. With B-SOSE's own fluxes the ocean sees what
# B-SOSE's ocean saw, so:
#   coast salt gone       -> the cause is upstream of the ocean: the sea ice (growth, lead closing, brine), the bulk
#                            formulae, or ERA5 against B-SOSE's adjusted atmosphere
#   coast salt still there -> the ocean's own response: coastal circulation, bathymetry, mixing
# The same split for the too-warm, too-deep winter layer at 60-55S open water.
#
# THE FLUXES (5-day files <token>_bsoseI156_<year>_5day.nc, each record placed at its window's centre, stamp - 2.5 days,
# linear in time between records; regridded as the ice in _ibobc: B-SOSE land takes its nearest ocean value within 6
# cells, then bilinear). B-SOSE's conventions are "> 0 into the ocean"; Oceananigans' top fluxes are + upward.
#   heat       top:   -(TFLUX - oceQsw) / (ρ₀ cₚ)    the non-solar part
#              solar: oceQsw / (ρ₀ cₚ) through the column, NumericalEarth's TwoColorRadiation (as the coupler feeds it)
#   salt       top:   -(SFLUX - S_top · oceFWflx) / ρ₀, the model's own top-cell S: the virtual-salt form of B-SOSE's real
#              freshwater flux. oceFWflx carries P - E, runoff and the Hammond & Jones melt (tools/coastal_freshwater.jl
#              residual = oceFWflx - ice - P = runoff), and the freshwater part of ice growth and melt (the brine); SFLUX
#              is the salt the ice keeps (~4 g/kg). So the Hammond & Jones forcing of _ibobc is NOT added here: it is
#              already in oceFWflx.
#   momentum   top:   -oceTAUX / ρ₀, -oceTAUY / ρ₀: the stress the ocean receives, from the air in open water and
#              from the ice under it. Regridded at cell centres (MITgcm's C-grid faces averaged), then averaged
#              onto the nest's faces.
# These are B-SOSE's fluxes computed against B-SOSE's surface state, so they do not feed back: nothing stops the run's
# surface from drifting below freezing (no ice forms) or away from B-SOSE's SST. The progress line prints the coldest
# surface water; watch it at the coast in winter.
#
#     julia --project=. examples/sowinwarp_nested_bsose_ggl_bsoseflux.jl
#
# A fixed configuration: every setting is written below, nothing is read from the environment. Writes
# <RUN_NAME>_{surface,midlon,free_surface,surface_fluxes,eke,z105,z200}.nc (no _sea_ice.nc), so tools/compare_bsose.jl,
# compare_spira_mld.jl, compare_mld_5day.jl, zonal_bias.jl and ekman_profile.jl read it.

# HDF5 file locking off, before any library opens a file (see the _ibobc script).
ENV["HDF5_USE_FILE_LOCKING"] = "FALSE"

using CUDA                          # defines GPU()
using NumericalEarth
using NumericalEarth.DataWrangling: bracketing_dates
using Oceananigans
using Oceananigans.Units
using Oceananigans.Diagnostics: NaNChecker
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.TurbulenceClosures: HorizontalScalarDiffusivity, VerticalScalarDiffusivity
using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities: CATKEVerticalDiffusivity, CATKEMixingLength, CATKEEquation
using Oceananigans.TimeSteppers: VerticallyImplicitTimeDiscretization
using NCDatasets                    # loads Oceananigans' NetCDFWriter
using Dates, Printf

include(joinpath(@__DIR__, "..", "src", "setup_bsose.jl"))          # BSOSEMonthly, resolve_bsose_filename, load_coord

# ## Run settings

const SMOKE    = false      # true: 2 days, 6-hourly output
const RUN_NAME = SMOKE ? "bsose_bsoseflux_ealpha30_smoke" : "bsose_ggl_hj_pr207_bsoseflux_ealpha30"
const BDIR     = "/storage/scratch1/1/vnguyen480/SOWinWaRP/data"   # B-SOSE, its grid.nc, the 5-day files, era5/
const ARCH     = GPU()

# THE CLOSURE ARM. E_ALPHA scales CATKE's TKE mixing lengths (its TKE diffusivity; 1.0 is NumericalEarth's default, as
# _ibobc). BACKGROUND = true keeps _ibobc's constant νz, κz = 1e-3, 1e-5 m²/s under CATKE; false drops them (CATKE and
# the horizontal Laplacian only).
const E_ALPHA    = 30.0
const BACKGROUND = true

start_date = DateTime(2014, 3, 2, 6)       # a native B-SOSE stamp (see the _ibobc script)
end_date   = SMOKE ? DateTime(2014, 3, 4, 6) : DateTime(2016, 3, 1, 6)    # 730 days
const Δt_max = 30minutes
const CFL    = 0.8
output_interval = SMOKE ? 6hours : 1days

for name in ("RUN_NAME", "END_DATE", "SMOKE", "PICKUP", "ARCH", "BSOSE_DIR", "ERA5_DIR")
    haskey(ENV, name) && @warn "$name = $(ENV[name]) is set in the environment; this script ignores it"
end
ARCH isa GPU && !CUDA.functional() && error("CUDA is not functional on this node. Check the GPU allocation.")
@info "$RUN_NAME: $start_date to $end_date"
include(joinpath(@__DIR__, "..", "src", "run_name_guard.jl"))
guard_run_name(RUN_NAME)   # stop if another job is writing this run's files

# ## Grid (as _ibobc)

const λ₁, λ₂ = 90.0, 150.0
const φ₁, φ₂ = -70.0, -45.0
const SCALING = 6
const Nx, Ny = Int(SCALING * (λ₂ - λ₁)), Int(SCALING * (φ₂ - φ₁))   # 360 x 150

const BSOSE_DRF = [4.2, 5.0, 5.9, 6.9, 8.5, 9.5, 10.0, 10.0, 10.0, 10.0,
                   10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 13.0, 17.0, 20.0, 20.0,
                   20.0, 20.0, 20.0, 20.0, 22.0, 30.0, 38.0, 45.0, 50.0, 50.0,
                   53.0, 72.0, 100.0, 100.0, 100.0, 100.0, 100.0, 150.0, 200.0, 200.0,
                   200.0, 220.0, 300.0, 380.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0,
                   400.0, 400.0]
z = reverse(-vcat(0.0, cumsum(BSOSE_DRF))) .+ 0.0
const Nz = length(z) - 1

underlying_grid = LatitudeLongitudeGrid(ARCH; size = (Nx, Ny, Nz), halo = (7, 7, 7), z,
                                        longitude = (λ₁, λ₂), latitude = (φ₁, φ₂))

# ## Bathymetry: B-SOSE's own (as _ibobc)

function bsose_bottom_height(path)
    ds = NCDataset(path)
    X = Float64.(ds["XC"][:, 1])
    Y = Float64.(ds["YC"][1, :])
    D = Float64.(coalesce.(ds["Depth"][:, :], 0.0))
    close(ds)

    edges = vcat(1.5Y[1] - 0.5Y[2], (Y[1:end-1] .+ Y[2:end]) ./ 2, 1.5Y[end] - 0.5Y[end-1])
    A = diff(edges) .* cosd.(Y)
    columns = findall(x -> λ₁ < x < λ₂, X)
    length(columns) == Nx || error("$path has $(length(columns)) longitudes in $(λ₁)-$(λ₂)E, the nest $Nx")

    area, wet, volume = zeros(Nx, Ny), zeros(Nx, Ny), zeros(Nx, Ny)
    for jb in eachindex(Y), ib in columns
        i = floor(Int, (X[ib] - λ₁) * SCALING) + 1
        j = floor(Int, (Y[jb] - φ₁) * SCALING) + 1
        1 <= j <= Ny || continue
        area[i, j] += A[jb]
        D[ib, jb] > 0 && (wet[i, j] += A[jb]; volume[i, j] += A[jb] * D[ib, jb])
    end
    zb = [wet[i, j] >= 0.5 * area[i, j] ? -max(volume[i, j] / wet[i, j], 10.0) : 0.0 for i in 1:Nx, j in 1:Ny]

    reached = falses(Nx, Ny)
    queue = [(i, j) for i in 1:Nx, j in 1:Ny if (i == 1 || i == Nx || j == Ny) && zb[i, j] < 0]
    foreach(c -> reached[c...] = true, queue)
    while !isempty(queue)
        i, j = pop!(queue)
        for (a, b) in ((i-1, j), (i+1, j), (i, j-1), (i, j+1))
            (1 <= a <= Nx && 1 <= b <= Ny && !reached[a, b] && zb[a, b] < 0) || continue
            reached[a, b] = true
            push!(queue, (a, b))
        end
    end
    zb[.!reached] .= 0.0
    return zb
end

function fill_dead_end_pockets!(zb)
    land(i, j) = 1 <= i <= Nx && 1 <= j <= Ny && zb[i, j] >= 0
    filled = 0
    while true
        pockets = [(i, j) for i in 1:Nx, j in 1:Ny
                   if zb[i, j] < 0 && count((land(i-1, j), land(i+1, j), land(i, j-1), land(i, j+1))) >= 3]
        isempty(pockets) && return filled
        foreach(c -> zb[c...] = 0, pockets)
        filled += length(pockets)
    end
end

zb = bsose_bottom_height(joinpath(BDIR, "grid.nc"))
filled = fill_dead_end_pockets!(zb)
bottom_height = Field{Center, Center, Nothing}(underlying_grid)
set!(bottom_height, reshape(zb, Nx, Ny, 1))
fill_halo_regions!(bottom_height)
@info "bathymetry: B-SOSE i156 Depth, $(count(zb .< 0)) wet columns, $filled dead-end pocket cells filled"

grid = ImmersedBoundaryGrid(underlying_grid, PartialCellBottom(bottom_height; minimum_fractional_cell_height = 0.2);
                            active_cells_map = true)

# ## Closure (as _ibobc, scaled by E_ALPHA, backgrounds per BACKGROUND)

dataset = BSOSEMonthly(dir = BDIR, iteration = 156)
parent_records = length(bracketing_dates(dataset, :temperature, (start_date, end_date)))

const ML_DEFAULT = CATKEMixingLength()
catke = CATKEVerticalDiffusivity(VerticallyImplicitTimeDiscretization(), Float64;
                                 mixing_length = CATKEMixingLength(Cᵇ = 0.01,
                                                                   Cʰⁱe = E_ALPHA * ML_DEFAULT.Cʰⁱe,
                                                                   Cˡᵒe = E_ALPHA * ML_DEFAULT.Cˡᵒe,
                                                                   Cᵘⁿe = E_ALPHA * ML_DEFAULT.Cᵘⁿe,
                                                                   Cᶜe  = E_ALPHA * ML_DEFAULT.Cᶜe,
                                                                   Cᵉe  = E_ALPHA * ML_DEFAULT.Cᵉe),
                                 turbulent_kinetic_energy_equation = CATKEEquation(Cᵂϵ = 1.0))
closure = BACKGROUND ?
    (catke, HorizontalScalarDiffusivity(ν = 10.0, κ = 10.0),
     VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(); ν = 1e-3, κ = 1e-5)) :
    (catke, HorizontalScalarDiffusivity(ν = 10.0, κ = 10.0))
let ml = catke.mixing_length
    @info @sprintf("closure: CATKE, TKE mixing lengths x%.1f (Cʰⁱe %.3f, Cˡᵒe %.3f, Cᵘⁿe %.3f, Cᶜe %.3f), Laplacian 10 m²/s horizontal, %s",
                   E_ALPHA, ml.Cʰⁱe, ml.Cˡᵒe, ml.Cᵘⁿe, ml.Cᶜe,
                   BACKGROUND ? "νz, κz = 1e-3, 1e-5 m²/s" : "NO constant vertical background")
end

# ## Parent η without the inverse barometer (as _ibobc; see that script's header)

import NumericalEarth.NestedModels: sample_parent!
using Oceananigans.Architectures: architecture, on_architecture

const IB_ρ₀g  = 1027.0 * 9.81
const IB_HALO = 7

function inverse_barometer_months(first_date, last_date, era5_dir, wet)
    files = filter(f -> startswith(f, "mean_sea_level_pressure_") && endswith(f, ".nc"), readdir(era5_dir))
    λc = [λ₁ + (i - 0.5) / SCALING for i in 1-IB_HALO:Nx+IB_HALO]
    φc = [φ₁ + (j - 0.5) / SCALING for j in 1-IB_HALO:Ny+IB_HALO]
    times, Xs = Float64[], Matrix{Float64}[]
    m = DateTime(year(first_date), month(first_date)) - Month(1)
    while m <= DateTime(year(last_date), month(last_date)) + Month(1)
        fy = filter(f -> startswith(f, "mean_sea_level_pressure_$(year(m))_"), files)
        isempty(fy) && error("no ERA5 mean_sea_level_pressure_$(year(m))_*.nc in $era5_dir for the inverse barometer")
        P = NCDataset(joinpath(era5_dir, first(fy))) do e
            lon, lat = mod.(Float64.(e["longitude"][:]), 360), Float64.(e["latitude"][:])
            t = e["valid_time"][:]
            sel = findall(d -> year(d) == year(m) && month(d) == month(m), t)
            isempty(sel) && error("$(first(fy)) has no hours in $(Dates.format(m, "yyyy-mm"))")
            Pm = dropdims(sum(Float64.(e["msl"].var[:, :, first(sel):last(sel)]); dims = 3); dims = 3) ./ length(sel)
            Pm[[argmin(abs.(lon .- mod(x, 360))) for x in λc], [argmin(abs.(lat .- y)) for y in φc]]
        end
        interior = P[IB_HALO+1:IB_HALO+Nx, IB_HALO+1:IB_HALO+Ny]
        p̄ = sum(interior[wet]) / count(wet)
        push!(Xs, (P .- p̄) ./ IB_ρ₀g)
        push!(times, Dates.value(m + Millisecond(round(Int, Dates.value(Day(daysinmonth(m))) * 43_200_000)) - first_date) / 1000)
        m += Month(1)
    end
    return times, Xs
end

const IB_TIMES, IB_X = inverse_barometer_months(start_date, end_date, joinpath(BDIR, "era5"), zb .< 0)

function add_inverse_barometer!(field, time)
    n = clamp(searchsortedlast(IB_TIMES, time), 1, length(IB_TIMES) - 1)
    w = clamp((time - IB_TIMES[n]) / (IB_TIMES[n+1] - IB_TIMES[n]), 0.0, 1.0)
    is, js = axes(field.data, 1) .+ IB_HALO, axes(field.data, 2) .+ IB_HALO
    X = (1 - w) .* IB_X[n][is, js] .+ w .* IB_X[n+1][is, js]
    d = parent(field.data)
    d .+= on_architecture(architecture(field), reshape(X, size(d)))
    return nothing
end

function sample_parent!(field, loc::Tuple{Center, Center, Nothing}, source, time)
    invoke(sample_parent!, Tuple{Any, Any, Any, Any}, field, loc, source, time)
    add_inverse_barometer!(field, time)
    return field
end

@assert which(sample_parent!, Tuple{Any, Tuple{Center, Center, Nothing}, Any, Any}).module == @__MODULE__
@assert which(sample_parent!, Tuple{Any, Tuple{Face, Center, Center}, Any, Any}).module != @__MODULE__

# ## B-SOSE's 5-day surface fluxes, on the nest's cell centres
#
# The regridding of _ibobc's bsose_ice: B-SOSE land takes its nearest B-SOSE ocean value within 6 cells, then
# bilinear onto the nest's centres; nest land is 0.

const ρ₀ = 1026.0               # as src/setup_bsose.jl's surface builders (B-SOSE's rhoConst differs by < 1%)
const cₚ = 3991.86795711963

fiveday_file(token, y) = joinpath(BDIR, "$(token)_bsoseI156_$(y)_5day.nc")

function bsose_regridder()
    ds = NCDataset(joinpath(BDIR, resolve_bsose_filename(BDIR, :surface_heat_flux, 156)))
    xc, yc = load_coord(ds, "XC", 1), load_coord(ds, "YC", 2)
    iB = let r = findall(x -> λ₁ - 1 <= x <= λ₂ + 1, xc); minimum(r):maximum(r) end
    jB = let r = findall(y -> φ₁ - 1 <= y <= φ₂ + 1, yc); minimum(r):maximum(r) end
    wetB = Float64.(coalesce.(ds["Depth"][iB, jB], 0.0)) .> 0
    close(ds)
    xs, ys = xc[iB], yc[jB]

    lin = LinearIndices(wetB)
    source = map(CartesianIndices(wetB)) do I
        best, d² = 0, Inf
        for di in -6:6, dj in -6:6
            J = I + CartesianIndex(di, dj)
            (checkbounds(Bool, wetB, J) && wetB[J] && di^2 + dj^2 < d²) || continue
            best, d² = lin[J], di^2 + dj^2
        end
        best
    end

    λc = λ₁ .+ ((1:Nx) .- 0.5) ./ SCALING
    φc = φ₁ .+ ((1:Ny) .- 0.5) ./ SCALING
    bracket(s, x) = (a = clamp(searchsortedlast(s, x), 1, length(s) - 1); (a, clamp((x - s[a]) / (s[a+1] - s[a]), 0, 1)))
    function stencil(A, a, wx, b, wy)
        num = den = 0.0
        for (di, w₁) in ((0, 1 - wx), (1, wx)), (dj, w₂) in ((0, 1 - wy), (1, wy))
            s = source[a+di, b+dj]
            s > 0 && (num += w₁ * w₂ * A[s]; den += w₁ * w₂)
        end
        return den > 0 ? num / den : 0.0
    end
    wet = zb .< 0
    regrid(A) = [wet[i, j] ? stencil(A, bracket(xs, λc[i])..., bracket(ys, φc[j])...) : 0.0 for i in 1:Nx, j in 1:Ny]
    return (; iB, jB, regrid)
end

# token => variable names inside its files (as tools/single_column_bsose.jl)
const FLUX_TOKENS = (TFLUX = ("surfTflx", ("TFLUX",)), SFLUX = ("surfSflx", ("SFLUX",)),
                     Qsw = ("oceQsw", ("oceQsw",)), FW = ("oceFWflx", ("oceFWflx",)),
                     τx = ("oceTAUX", ("oceTAUX",)), τy = ("oceTAUY", ("oceTAUY",)))

function bsose_fiveday_fluxes(first_date, last_date)
    R = bsose_regridder()
    # the run's years, and a neighbour year if every token has it, so a run near New Year is still bracketed
    years = [y for y in year(first_date)-1:year(last_date)+1
             if year(first_date) <= y <= year(last_date) || all(isfile(fiveday_file(t, y)) for (t, _) in values(FLUX_TOKENS))]
    for y in years, (token, _) in values(FLUX_TOKENS)
        isfile(fiveday_file(token, y)) || error("missing $(fiveday_file(token, y)) (download_bsose_5day.sh)")
    end
    out = Dict{Symbol, Vector{Matrix{Float64}}}(k => Matrix{Float64}[] for k in keys(FLUX_TOKENS))
    dates = DateTime[]
    for y in years, (key, (token, names)) in pairs(FLUX_TOKENS)
        ds = NCDataset(fiveday_file(token, y))
        name = first(filter(n -> haskey(ds, n), names))
        # MITgcm C grid: oceTAUX[i] on the west face of cell i, oceTAUY[j] on the south face of cell j; a cell's centre
        # value is the mean of its two faces (one more face read on the east / north side)
        A = if key === :τx
            Uf = Float64.(coalesce.(ds[name][first(R.iB):last(R.iB)+1, R.jB, :], 0.0))
            0.5 .* (Uf[1:end-1, :, :] .+ Uf[2:end, :, :])
        elseif key === :τy
            Vf = Float64.(coalesce.(ds[name][R.iB, first(R.jB):last(R.jB)+1, :], 0.0))
            0.5 .* (Vf[:, 1:end-1, :] .+ Vf[:, 2:end, :])
        else
            Float64.(coalesce.(ds[name][R.iB, R.jB, :], 0.0))
        end
        stamps = DateTime.(ds["time"][:])
        close(ds)
        key === :TFLUX && append!(dates, stamps .- Hour(60))      # window centre: stamps are the END of 5 days
        append!(out[key], [R.regrid(A[:, :, n]) for n in axes(A, 3)])
    end
    all(length(out[k]) == length(dates) for k in keys(out)) ||
        error("B-SOSE 5-day fluxes: record counts differ: " * join(["$k $(length(out[k]))" for k in keys(out)], ", "))
    issorted(dates) && all(diff(dates) .== Day(5)) || error("B-SOSE 5-day flux records are not contiguous 5-day steps")
    (first(dates) <= first_date && last(dates) >= last_date) ||
        error("B-SOSE 5-day fluxes cover $(first(dates)) to $(last(dates)), not $first_date to $last_date")
    keep = max(1, searchsortedlast(dates, first_date)):min(length(dates), searchsortedfirst(dates, last_date))
    times = [Dates.value(d - first_date) / 1000 for d in dates[keep]]
    return (; times, (k => out[k][keep] for k in keys(out))...)
end

const BF = bsose_fiveday_fluxes(start_date, end_date)
let wet = zb .< 0, A = [cosd(φ₁ + (j - 0.5) / SCALING) for i in 1:Nx, j in 1:Ny] .* wet
    m(X) = sum(sum(x .* A) for x in X) / (length(X) * sum(A))
    @info @sprintf("B-SOSE 5-day fluxes: %d records (%s to %s centres); means over the run: TFLUX %+.1f W/m², oceQsw %+.1f W/m², SFLUX %+.2e g/m²/s, oceFWflx %+.2f m/yr, oceTAUX %+.3f N/m²",
                   length(BF.times), start_date + Millisecond(round(Int, 1000 * first(BF.times))),
                   start_date + Millisecond(round(Int, 1000 * last(BF.times))),
                   m(BF.TFLUX), m(BF.Qsw), m(BF.SFLUX), m(BF.FW) * 365.25 * 86400 / 1000, m(BF.τx))
    length(BF.times) > 100 || error("only $(length(BF.times)) 5-day records for a two-year run: the 5-day files were not read")
end

# The kinematic flux fields the top boundary conditions read, filled every step by update_surface_fluxes! below
Jᵘ = Field{Face, Center, Nothing}(grid)
Jᵛ = Field{Center, Face, Nothing}(grid)
Jᵀ = Field{Center, Center, Nothing}(grid)
Jᴮ = Field{Center, Center, Nothing}(grid)        # -SFLUX / ρ₀, the salt the ice keeps (+ up)
Jᵂ = Field{Center, Center, Nothing}(grid)        # oceFWflx / ρ₀, m/s, + into the ocean

@inline prescribed_flux(i, j, grid, clock, fields, J) = @inbounds J[i, j, 1]
@inline salt_flux(i, j, grid, clock, fields, p) = @inbounds p.brine[i, j, 1] + fields.S[i, j, grid.Nz] * p.fw[i, j, 1]

surface_bcs = (u = FieldBoundaryConditions(top = FluxBoundaryCondition(prescribed_flux; discrete_form = true, parameters = Jᵘ)),
               v = FieldBoundaryConditions(top = FluxBoundaryCondition(prescribed_flux; discrete_form = true, parameters = Jᵛ)),
               T = FieldBoundaryConditions(top = FluxBoundaryCondition(prescribed_flux; discrete_form = true, parameters = Jᵀ)),
               S = FieldBoundaryConditions(top = FluxBoundaryCondition(salt_flux; discrete_form = true,
                                                                       parameters = (; brine = Jᴮ, fw = Jᵂ))))

# ## Ocean, nested in B-SOSE (as _ibobc, without the Hammond & Jones forcing: it is in oceFWflx)

ocean = nested_ocean_simulation(grid, dataset;
                                dates = (start_date, end_date),
                                dir = BDIR,
                                parent_time_indices_in_memory = parent_records,
                                boundary_conditions = surface_bcs,
                                closure,
                                Δt = 1minutes,
                                stop_time = Dates.value(end_date - start_date) / 1000,
                                free_surface = SplitExplicitFreeSurface(grid; cfl = 0.7, fixed_Δt = Δt_max))

child = ocean.model.child

# Stop here if the merge kept NumericalEarth's coupler-fed top conditions instead of these
for (name, f) in pairs((; u = child.velocities.u, v = child.velocities.v, T = child.tracers.T, S = child.tracers.S))
    top = f.boundary_conditions.top
    (top.condition isa Oceananigans.BoundaryConditions.DiscreteBoundaryFunction &&
     top.condition.func in (prescribed_flux, salt_flux)) ||
        error("the child's top boundary condition on $name is not B-SOSE's flux: $(summary(top))")
end
@info "top boundary conditions: B-SOSE's fluxes on u, v, T, S (checked)"

radiation = NumericalEarth.Oceans.get_radiative_forcing(child)
radiation isa NumericalEarth.Oceans.TwoColorRadiation ||
    error("the child has no TwoColorRadiation: the shortwave would have nowhere to go")

# Fill every flux field at the model's time: linear between the bracketing 5-day centres, clamped at the ends
const NFX = size(interior(Jᵘ), 1)
const NFY = size(interior(Jᵛ), 2)
# face i lies between centres i-1 and i; a boundary face (Bounded: Nx + 1 faces) takes its one centre
to_x_faces(C) = NFX == Nx + 1 ? vcat(C[1:1, :], 0.5 .* (C[1:end-1, :] .+ C[2:end, :]), C[end:end, :]) :
                                0.5 .* (circshift(C, (1, 0)) .+ C)
to_y_faces(C) = NFY == Ny + 1 ? hcat(C[:, 1:1], 0.5 .* (C[:, 1:end-1] .+ C[:, 2:end]), C[:, end:end]) :
                                0.5 .* (circshift(C, (0, 1)) .+ C)

function update_surface_fluxes!(t)
    n = clamp(searchsortedlast(BF.times, t), 1, length(BF.times) - 1)
    w = clamp((t - BF.times[n]) / (BF.times[n+1] - BF.times[n]), 0.0, 1.0)
    at(X) = (1 - w) .* X[n] .+ w .* X[n+1]
    TF, Qs = at(BF.TFLUX), at(BF.Qsw)
    set!(Jᵀ, reshape(.-(TF .- Qs) ./ (ρ₀ * cₚ), Nx, Ny, 1))
    set!(radiation.surface_flux, reshape(Qs ./ (ρ₀ * cₚ), Nx, Ny, 1))
    set!(Jᴮ, reshape(.-at(BF.SFLUX) ./ ρ₀, Nx, Ny, 1))
    set!(Jᵂ, reshape(at(BF.FW) ./ ρ₀, Nx, Ny, 1))
    set!(Jᵘ, reshape(to_x_faces(.-at(BF.τx) ./ ρ₀), NFX, Ny, 1))
    set!(Jᵛ, reshape(to_y_faces(.-at(BF.τy) ./ ρ₀), Nx, NFY, 1))
    return nothing
end
update_surface_fluxes!(time(ocean))
ocean.callbacks[:surface_fluxes] = Callback(sim -> update_surface_fluxes!(time(sim)), IterationInterval(1))
@info @sprintf("surface fluxes filled at t = %.0f s: max |Jᵘ| %.2e m²/s², Jᵀ %.2e to %.2e °C m/s",
               time(ocean), maximum(abs, Jᵘ), minimum(Jᵀ), maximum(Jᵀ))

# ## Simulation: the wizard on the ocean's horizontal Courant number only (no ice)

ocean_timescale() = Oceananigans.Advection.cell_advection_timescale(child.grid,
                        (child.velocities.u, child.velocities.v, Oceananigans.Fields.ZeroField()))
wizard = TimeStepWizard(; cfl = CFL, max_change = 1.1, max_Δt = Δt_max, min_Δt = 1,
                        cell_advection_timescale = _ -> ocean_timescale())
ocean.callbacks[:wizard] = Callback(wizard, IterationInterval(10))
ocean.callbacks[:nan_checker] = Callback(NaNChecker(fields = merge(child.velocities, child.tracers), erroring = true),
                                         IterationInterval(1))

wet_top = Array(child.grid.immersed_boundary.bottom_height[1:Nx, 1:Ny, 1]) .< 0
centred(A) = (A = size(A, 1) == Nx + 1 ? 0.5 .* (A[1:end-1, :] .+ A[2:end, :]) : A;
              size(A, 2) == Ny + 1 ? 0.5 .* (A[:, 1:end-1] .+ A[:, 2:end]) : A)
top_level(f) = Array(view(interior(f), :, :, size(interior(f), 3)))
place(I) = @sprintf("%.1fE %.1fS", λ₁ + (I[1] - 0.5) / SCALING, -(φ₁ + (I[2] - 0.5) / SCALING))

wall_time = Ref(time_ns())
function progress(sim)
    S, T = top_level(child.tracers.S), top_level(child.tracers.T)
    sₒ = ifelse.(wet_top, hypot.(centred(top_level(child.velocities.u)), centred(top_level(child.velocities.v))), 0.0)
    # how far the surface sits below its freezing point (TEOS-10 approx. -0.0575 S at the surface)
    Tf = ifelse.(wet_top, T .+ 0.0575 .* S, Inf)
    Iₒ, I_f = argmax(sₒ), argmin(Tf)
    @info @sprintf("day %.1f, iter %d, Δt %.1f min | surface S (%.2f, %.2f), T (%.2f, %.2f) | coldest T - T_f %+.2f at %s | ocean %.2f m/s at %s, max |w| %.1e, CFL %.2f | wall %s",
                   time(sim) / 86400, iteration(sim), sim.Δt / 60, extrema(S[wet_top])..., extrema(T[wet_top])...,
                   Tf[I_f], place(I_f), sₒ[Iₒ], place(Iₒ), maximum(abs, child.velocities.w), sim.Δt / ocean_timescale(),
                   prettytime(1e-9 * (time_ns() - wall_time[])))
    wall_time[] = time_ns()
    return nothing
end
ocean.callbacks[:progress] = Callback(progress, TimeInterval(output_interval))

# ## Output (as _ibobc, less the sea ice; _surface_fluxes holds what this run applies)

u, v, w = child.velocities
T, S = child.tracers.T, child.tracers.S

output_metadata = Dict("run_name" => RUN_NAME, "start_date" => string(start_date), "end_date" => string(end_date),
                "dataset" => "BSOSE",
                "setup" => "examples/sowinwarp_nested_bsose_ggl_bsoseflux.jl",
                "boundaries" => "one-way nest in B-SOSE monthly (NumericalEarth nested_ocean_simulation), parent eta without the inverse barometer",
                "surface_forcing" => "B-SOSE i156 5-day TFLUX - oceQsw (top), oceQsw (TwoColorRadiation), SFLUX - S_top*oceFWflx (virtual salt), oceTAUX/Y; no sea ice, no atmosphere",
                "coastal_freshwater" => "none added (Hammond & Jones is inside oceFWflx)",
                "closure" => @sprintf("CATKE (NumericalEarth default, TKE mixing lengths x%.1f) + Laplacian 10 m2/s horizontal%s",
                                      E_ALPHA, BACKGROUND ? " + nu_z 1e-3 / kappa_z 1e-5 vertical" : ", no vertical background"),
                "bathymetry" => "B-SOSE i156 Depth (grid.nc), area-aggregated",
                "vertical_coordinate" => "static z, B-SOSE i156's 52 levels")

ocean.output_writers[:surface] = NetCDFWriter(child, (; u, v, T, S);
                                              filename = "$(RUN_NAME)_surface.nc",
                                              schedule = TimeInterval(output_interval),
                                              indices = (:, :, Nz),
                                              global_attributes = output_metadata, overwrite_files = true)

ocean.output_writers[:mid_lon] = NetCDFWriter(child, (; u, v, T, S);
                                              filename = "$(RUN_NAME)_midlon.nc",
                                              schedule = TimeInterval(output_interval),
                                              indices = (div(Nx, 2), :, :),
                                              global_attributes = output_metadata, overwrite_files = true)

@inline surface_displacement(i, j, k, grid, η) = @inbounds η[i, j, grid.Nz + 1]
η_surface = Field(KernelFunctionOperation{Center, Center, Nothing}(surface_displacement, child.grid, child.free_surface.displacement.data))
ocean.output_writers[:free_surface] = NetCDFWriter(child, (; η = η_surface);
                                                   filename = "$(RUN_NAME)_free_surface.nc",
                                                   schedule = TimeInterval(output_interval),
                                                   global_attributes = output_metadata, overwrite_files = true)

ocean.output_writers[:surface_fluxes] =
    NetCDFWriter(child, (; heat_flux = Jᵀ, brine_salt_flux = Jᴮ, freshwater_flux = Jᵂ,
                           shortwave = radiation.surface_flux, zonal_momentum_flux = Jᵘ, meridional_momentum_flux = Jᵛ);
                 filename = "$(RUN_NAME)_surface_fluxes.nc",
                 schedule = AveragedTimeInterval(output_interval, window = output_interval),
                 global_attributes = merge(output_metadata, Dict(
                     "flux_convention" => "kinematic, + OUT of the ocean (heat_flux non-solar °C m/s; brine_salt_flux = -SFLUX/rho0; momentum m2/s2), except freshwater_flux = oceFWflx/rho0 (m/s, + into the ocean) and shortwave = oceQsw/(rho0 cp) (°C m/s, + down)",
                     "salt_flux_seen_by_ocean" => "brine_salt_flux + S_top * freshwater_flux",
                     "averaging_window_seconds" => Float64(output_interval))),
                 overwrite_files = true)

u_cc = @at (Center, Center, Center) u
v_cc = @at (Center, Center, Center) v
ocean.output_writers[:eke] = NetCDFWriter(child, (; u = u_cc, v = v_cc, T, S, uu = u_cc^2, vv = v_cc^2);
                                          filename = "$(RUN_NAME)_eke.nc",
                                          schedule = AveragedTimeInterval(30days, window = 30days, stride = 1),
                                          indices = (:, :, :),
                                          global_attributes = merge(output_metadata, Dict(
                                              "averaging_window_seconds" => Float64(30days),
                                              "averaging_stride" => 1,
                                              "moment_location" => "Center, Center, Center",
                                              "eke_recipe" => "EKE = 0.5*((uu - u^2) + (vv - v^2)); MKE = 0.5*(u^2 + v^2)")),
                                          overwrite_files = true)

z_centres = [(z[k] + z[k+1]) / 2 for k in 1:Nz]
for (name, depth) in ((:z105, 105.0), (:z200, 200.0))
    k = argmin(abs.(z_centres .+ depth))
    abs(z_centres[k] + depth) < 0.01 || error("no level centred at $(depth) m (nearest: $(-z_centres[k]) m)")
    ocean.output_writers[name] = NetCDFWriter(child, (; u, v, T, S);
                                              filename = "$(RUN_NAME)_$(name).nc",
                                              schedule = TimeInterval(output_interval),
                                              indices = (:, :, k),
                                              global_attributes = merge(output_metadata, Dict(
                                                  "level_index" => k,
                                                  "level_centre_depth_m" => depth,
                                                  "level_faces_m" => "$(-z[k+1])-$(-z[k])")),
                                              overwrite_files = true)
end

# ## Run (no checkpoints)

@info "output: $(RUN_NAME)_{surface,midlon,free_surface,surface_fluxes,eke,z105,z200}.nc in $(pwd())"
run!(ocean)
