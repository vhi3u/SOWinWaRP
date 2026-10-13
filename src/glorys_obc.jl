# ==============================================================================
# glorys_obc.jl
#
# GLORYS-driven open boundaries, following NumericalEarth's 02_mab_glorys_obc.jl as closely
# as this domain allows. The domain, bathymetry source, closures, surface forcing and output
# are ours; everything about how GLORYS is read and turned into boundary conditions is theirs.
#
# The one substantive change from 02 is the cadence: it uses `GLORYSDaily()` over a 12x8 degree
# box for ten days; this domain is 60x25 degrees for eight months, where daily would be ~250
# records and tens of GB. `GLORYSMonthly()` is 11 records for the same window. Everything else
# -- how the metadata is built, how the series are read, how Uᵉˣᵗ is formed, which scheme goes
# on which component -- is 02's. Pass `dataset = GLORYSDaily()` to get its cadence back.
#
# THE FULL TRIAD, as 02 assembles it:
#
#   u, v NORMAL      NormalFlowBoundaryCondition(Interpolated(GLORYS); scheme)
#   u, v TANGENTIAL  ValueBoundaryCondition(Interpolated(GLORYS); scheme)   <- see note
#   T, S             ValueBoundaryCondition(Interpolated(GLORYS); scheme)
#   U, V barotropic  Flather, fed the DEPTH-INTEGRATED GLORYS transport and zos
#   η                Chapman (radiates only; Flather is the data entry point)
#
# The barotropic pair is not optional. The split-explicit corrector replaces the depth mean of
# the boundary velocity with whatever the barotropic solver says, so driving `u` at the
# boundary while leaving `U` unconditioned gets the inflow wiped.
#
# ON THE TANGENTIAL COMPONENT. NumericalEarth's `parent_boundary_conditions` builds this whole
# structure from a parent FieldTimeSeries but applies `schemes` only on `NormalFlow` sides,
# and its docstring recommends a stiff Dirichlet for the tangential component. In 02's nested
# tests that was the WORST of three options (enstrophy 1.611 against a truth of 1.000, versus
# 0.330 for `Gradient(0)` and 0.923 radiated): imposing the tangential velocity puts a
# discontinuity straight into ∂v/∂x, and since ζ = ∂v/∂x − ∂u/∂y that appears as a vorticity
# band sitting on the boundary. So the conditions here are hand-built from `Interpolated` in
# order to give the tangential component a scheme too.
#
# ON Uᵉˣᵗ ("native" mode, 02's default). ū is the TRUE depth mean of `uo`/`vo` over GLORYS's
# OWN valid levels -- its real bottom, from the static `deptho` field -- computed on GLORYS's
# own 50-level vertical grid. Only H_wet, the model's own wet depth, is model-derived, because
# that is the H the model's barotropic mode needs. It is deliberately NOT
# `Integral(fts[n], dims=3)`: `src_grid` has no immersed bottom, so that integral runs over the
# whole column including the inpainted velocity filling cells below GLORYS's real sea floor,
# and on a shelf that transport is then divided by a column tens of metres deep. In the MAB
# that gave an implied depth-mean Uᵉˣᵗ/H of −3.4 m/s against a true +0.2 m/s. Matching the
# bathymetry cannot fix it; it only changes H, not the numerator. `report_uext` prints all
# three modes so the difference is visible here rather than assumed.
#
# ON THE SOURCE GRID. `Interpolated` evaluates the parent at the child's boundary FACE, which
# lies half a cell outside the outermost parent cell CENTRE, so the parent source must BRACKET
# the child grid rather than merely coincide with it. 02 achieves that by insetting its MODEL
# inside a fixed, already-downloaded GLORYS box. GLORYS is global and we download what we ask
# for, so we instead keep the model domain fixed and extend the SOURCE outward by `n_pad`
# cells. The index relation is identical -- source index s along a boundary is model index
# s − n_pad -- so `wet_transport` and `boundary_columns` carry over unchanged, and that is
# also the mapping used when the slabs are turned into per-face `FieldTimeSeries` below.
#
# RECONSTRUCTED: `match_boundary_bathymetry!`. 02 calls it from `glorys_bathymetry.jl`, which
# was not provided, so it is rebuilt here from that script's description of it: blend the model
# depth toward GLORYS `deptho` over `n_match` cells inward from each open edge. The reason it
# exists is that U = ∫u dz, so a depth mismatch at the boundary is a transport error; in the
# MAB the unmatched western shelf differed from GLORYS by up to ~60% and drove a spurious
# inflow jet (the SW-corner max|u| = 3.5 m/s artefact). This is the ONLY thing here that
# changes the bathymetry, and it touches an `n_match`-cell band at the open edges only.
# ==============================================================================

using CopernicusMarine                      # activates the GLORYS download backend
using NumericalEarth.NestedModels: Interpolated
using NumericalEarth.DataWrangling: Metadata, MetadataSet, BoundingBox, z_interfaces, all_dates
using Oceananigans.BoundaryConditions: PerturbationAdvection, NormalRadiation,
                                       GravityWaveRadiationBoundaryCondition,
                                       SurfaceWaveRadiationBoundaryCondition
using Oceananigans.Grids: znodes, halo_size
using Statistics: mean

"""
    glorys_directory()

Where GLORYS lives. `GLORYS_DIR` overrides; otherwise a `glorys` subdirectory of wherever the
BSOSE data already sits, so on the cluster it lands on scratch rather than in `\$HOME`.
"""
function glorys_directory()
    haskey(ENV, "GLORYS_DIR") && return ENV["GLORYS_DIR"]
    dir = joinpath(default_bsose_directory(), "glorys")
    mkpath(dir)
    return dir
end


"""
    host_bottom_height(x, Nx, Ny; halo = (0, 0))

The `(Nx, Ny)` interior of a bottom-height field, on the host.

`regrid_bathymetry` hands back a `Field`, but the one stored inside an `ImmersedBoundaryGrid`
is a bare `OffsetArray` -- and on the GPU that wraps a `CuArray`. `view`ing the OffsetArray and
then calling `Array` on the view does not reach the GPU copy machinery: it iterates element by
element and throws "Scalar indexing is disallowed". So the device-to-host copy is taken on
`parent`, which is contiguous, and the halo is stripped afterwards.
"""
function host_bottom_height(x, Nx, Ny; halo = (0, 0))
    x isa Field && return reshape(Array(interior(x, :, :, 1)), Nx, Ny)
    host = Array(parent(x))                       # one contiguous device-to-host copy
    Hx, Hy = halo
    return reshape(host[Hx .+ (1:Nx), Hy .+ (1:Ny), 1], Nx, Ny)
end


"""
    glorys_deptho_on_grid(target_grid; dir, region)

GLORYS's own sea floor (m, positive down) on `target_grid`'s cell centres, `NaN` on land.

This is the parent's bathymetry, not the model's. It decides which of GLORYS's native levels
carry real water when the true depth mean is formed, so `inpainting = nothing`: an inpainted
`deptho` would report ocean where GLORYS has none and defeat the whole point.

GLORYS's static file is on GLORYS's own grid, so it is resampled here. Nearest neighbour is
enough -- this selects levels, it does not interpolate values.
"""
function glorys_deptho_on_grid(target_grid; dir = glorys_directory(), region)
    m = first(Metadata(:depth; dataset = GLORYSStatic(), dir, region))
    Downloads.download(m)

    # Read the file directly rather than through `Field(::Metadatum)`: the static dataset is
    # downloaded under the CMEMS variable name `deptho`, but that reader asks the file for
    # `depth` and fails. The file is two coordinates and one 2-D array, so this is no loss.
    path = joinpath(m.dir, m.filename)
    ds = Dataset(path)
    A  = Float64.(coalesce.(ds["deptho"][:, :], NaN))
    λs = Float64.(ds["longitude"][:])
    φs = Float64.(ds["latitude"][:])
    close(ds)

    λt = collect(λnodes(target_grid, Center()))
    φt = collect(φnodes(target_grid, Center()))
    iidx = [argmin(abs.(λs .- λ)) for λ in λt]
    jidx = [argmin(abs.(φs .- φ)) for φ in φt]

    H = fill(NaN, length(λt), length(φt))
    for (jj, j) in enumerate(jidx), (ii, i) in enumerate(iidx)
        d = A[i, j]
        H[ii, jj] = (isfinite(d) && d > 0) ? d : NaN
    end
    return H
end


"""
    match_boundary_bathymetry!(bottom_height, grid; dir, region, n_match = 4, minimum_depth = 10)

Blend the model's depth toward GLORYS's `deptho` over `n_match` cells inward from each open
edge, in place.

Reconstructed from 02's description -- see the header. `U = ∫u dz`, so a depth mismatch at the
boundary is a transport error however carefully Uᵉˣᵗ is formed. The weight is 1 in the
outermost cell and falls linearly to 0 at `n_match` cells in, so the interior is untouched.
Only cells that are wet in both the model and GLORYS are blended: this adjusts depths, it does
not open or close columns.
"""
function match_boundary_bathymetry!(bottom_height, grid; dir = glorys_directory(), region,
                                    n_match = 4, minimum_depth = 10)
    Nx, Ny, = size(grid)
    Hg = glorys_deptho_on_grid(grid; dir, region)

    h = host_bottom_height(bottom_height, Nx, Ny; halo = halo_size(grid)[1:2])

    changed, Σδ = 0, 0.0
    for j in 1:Ny, i in 1:Nx
        d = min(i - 1, Nx - i, j - 1, Ny - j)        # cells to the nearest edge
        d >= n_match && continue
        h[i, j] < 0 || continue                       # model land stays land
        isfinite(Hg[i, j]) || continue                # GLORYS land: leave the model alone
        w = (n_match - d) / n_match
        hg = -max(Hg[i, j], minimum_depth)
        δ = w * (hg - h[i, j])
        h[i, j] += δ
        changed += 1; Σδ += abs(δ)
    end

    set!(bottom_height, reshape(h, Nx, Ny, 1))
    fill_halo_regions!(bottom_height)
    @info @sprintf(" -> boundary bathymetry matched to GLORYS over %d cells: %d columns changed, mean |Δdepth| %.1f m",
                   n_match, changed, changed == 0 ? 0.0 : Σδ / changed)
    return bottom_height
end


"""
    glorys_boundary_dates(start_date, end_date; dataset = GLORYSMonthly())

The dataset's own records covering the run, extended by one at each end.

02 can write `start_date : Day(1) : stop_date` because GLORYSDaily has a record on every one of
those days. A monthly dataset does not: its stamps are the first of each month, and asking for
a date it does not hold is what sends a series to `Cyclical()` wrapping. The extra record at
each end is what lets the series interpolate across the first and last half-month instead of
wrapping to the other end of the record -- the same reasoning as `bc_dates` in model.jl.
"""
function glorys_boundary_dates(start_date, end_date; dataset = GLORYSMonthly())
    available = all_dates(dataset, :temperature)

    (end_date < first(available) || start_date > last(available)) &&
        error("""$(summary(dataset)) holds no data for $start_date to $end_date. Its records run \
                 from $(first(available)) to $(last(available)).""")

    lo = something(findlast(d -> d <= start_date, available), firstindex(available))
    hi = something(findfirst(d -> d >= end_date, available), lastindex(available))
    lo = max(firstindex(available), lo - 1)
    hi = min(lastindex(available), hi + 1)
    return available[lo:hi]
end


struct GLORYSOpenBoundaries{B, S, R}
    boundary_conditions :: B
    series :: S
    region :: R
end


"""
    glorys_open_boundary_conditions(grid; dates, dir, n_pad = 2, tau_in = 1days,
                                    uext_mode = :native, tangential = :radiated,
                                    top_u = nothing, top_v = nothing,
                                    top_T = nothing, top_S = nothing)

02_mab_glorys_obc.jl's boundary conditions, for `grid`.

`uext_mode` is 02's `MAB_UEXT`: `:native` (its default and the one it validated), `:masked`, or
`:legacy`. `tangential` is its `MAB_TANGENTIAL`: `:radiated` (default) or `:prescribed`.
`top_*` carry our own surface forcing, which 02 has no equivalent of -- it gets ERA5 through
`OceanOnlyModel` -- and `FieldBoundaryConditions` is immutable, so they must be built in here.
"""
function glorys_open_boundary_conditions(grid;
                                         dates,
                                         dataset = GLORYSMonthly(),
                                         dir = glorys_directory(),
                                         n_pad = 2,
                                         tau_in = 1days,
                                         uext_mode = :native,
                                         tangential = :radiated,
                                         top_u = nothing, top_v = nothing,
                                         top_T = nothing, top_S = nothing)

    uext_mode in (:native, :masked, :legacy) ||
        error("uext_mode must be :native, :masked or :legacy; got $uext_mode.")
    tangential in (:radiated, :prescribed) ||
        error("tangential must be :radiated or :prescribed; got $tangential.")

    underlying = grid isa ImmersedBoundaryGrid ? grid.underlying_grid : grid
    Nx, Ny, Nz = size(underlying)
    arch = architecture(grid)

    λ = collect(λnodes(underlying, Face()))
    φ = collect(φnodes(underlying, Face()))
    Δλ = (λ[end] - λ[1]) / Nx
    Δφ = (φ[end] - φ[1]) / Ny

    λ_src = (λ[1] - n_pad * Δλ, λ[end] + n_pad * Δλ)
    φ_src = (φ[1] - n_pad * Δφ, φ[end] + n_pad * Δφ)
    Ns_x, Ns_y = Nx + 2n_pad, Ny + 2n_pad
    zi = collect(znodes(underlying, Face()))

    src_grid = LatitudeLongitudeGrid(arch; size = (Ns_x, Ns_y, Nz),
                                     longitude = λ_src, latitude = φ_src, z = zi, halo = (7, 7, 7))

    region = BoundingBox(longitude = (λ_src[1] - 0.5, λ_src[2] + 0.5),
                         latitude  = (φ_src[1] - 0.5, φ_src[2] + 0.5))

    meta(name) = Metadata(name; dataset, dates, dir, region)

    # Rough download size: GLORYS is 1/12 degree and 50 levels, and four of the five variables
    # are three-dimensional. Counted on GLORYS's own lattice, not the model's.
    gb = ((λ_src[2] - λ_src[1]) * 12) * ((φ_src[2] - φ_src[1]) * 12) * 50 * 4 * 4 * length(dates) / 1e9
    @info "GLORYS open boundaries (02_mab_glorys_obc.jl recipe), $(summary(dataset))"
    @info @sprintf(" -> model  λ %.4f to %.4f, φ %.4f to %.4f  (%d x %d x %d)",
                   λ[1], λ[end], φ[1], φ[end], Nx, Ny, Nz)
    @info @sprintf(" -> source λ %.4f to %.4f, φ %.4f to %.4f  (%d x %d), n_pad = %d",
                   λ_src[1], λ_src[2], φ_src[1], φ_src[2], Ns_x, Ns_y, n_pad)
    @info @sprintf(" -> %d GLORYS records, %s to %s; about %.1f GB to download if not cached",
                   length(dates), first(dates), last(dates), gb)
    @info " -> GLORYS directory: $dir"

    fts_u = FieldTimeSeries(meta(:u_velocity),   src_grid)
    fts_v = FieldTimeSeries(meta(:v_velocity),   src_grid)
    fts_T = FieldTimeSeries(meta(:temperature),  src_grid)
    fts_S = FieldTimeSeries(meta(:salinity),     src_grid)
    fts_η = FieldTimeSeries(meta(:free_surface), src_grid)
    @info @sprintf(" -> built: %d times", length(fts_u.times))

    # ---------------- the barotropic exterior, from GLORYS ----------------
    zc_src = 0.5 .* (zi[1:end-1] .+ zi[2:end])
    Δz_src = diff(zi)

    bh = grid isa ImmersedBoundaryGrid ?
          host_bottom_height(grid.immersed_boundary.bottom_height, Nx, Ny;
                             halo = halo_size(underlying)[1:2]) :
          zeros(Nx, Ny)
    floors = ((bh[1, :], bh[end, :]),       # (west, east) rows      -- u boundaries
              (bh[:, 1], bh[:, end]))       # (south, north) columns -- v boundaries

    function wet_transport(col, floor_line)
        Ns = size(col, 1)
        T, Hwet = zeros(Ns), zeros(Ns)
        for s in 1:Ns
            b = floor_line[clamp(s - n_pad, 1, length(floor_line))]
            for k in 1:size(col, 2)
                if zc_src[k] > b
                    T[s]    += Δz_src[k] * col[s, k]
                    Hwet[s] += Δz_src[k]
                end
            end
        end
        return T, Hwet
    end

    function boundary_columns(A, dim; Nz_expected = Nz)
        (size(A, 3) == Nz_expected && size(A, 1) in (Ns_x, Ns_x + 1) && size(A, 2) in (Ns_y, Ns_y + 1)) ||
            error("unexpected GLORYS frame size $(size(A)); expected ($Ns_x, $Ns_y, $Nz_expected)")
        p = n_pad + 1
        return dim == 1 ? (A[p, 1:Ns_y, :], A[end-n_pad, 1:Ns_y, :]) :
                          (A[1:Ns_x, p, :], A[1:Ns_x, end-n_pad, :])
    end

    function boundary_transport_slabs(fts, dim)
        times = collect(fts.times)
        slabs = map(eachindex(times)) do n
            if uext_mode === :legacy
                F = Field(Integral(fts[n], dims = 3)); compute!(F)
                I = Array(interior(F)); p = n_pad + 1
                dim == 1 ? (I[p, 1:Ns_y, 1], I[end-n_pad, 1:Ns_y, 1]) :
                           (I[1:Ns_x, p, 1], I[1:Ns_x, end-n_pad, 1])
            else
                lo, hi = boundary_columns(Array(interior(fts[n])), dim)
                (wet_transport(lo, floors[dim][1])[1], wet_transport(hi, floors[dim][2])[1])
            end
        end
        return times, first.(slabs), last.(slabs)
    end

    # ---------------- "native" mode: the true depth mean on GLORYS's OWN vertical grid -------
    local fts_u_native, fts_v_native, true_floors, zc_nat, Δz_nat
    local tU, U_lo, U_hi, tV, V_lo, V_hi
    if uext_mode === :native
        z_native = z_interfaces(meta(:u_velocity))
        Nz_native = length(z_native) - 1
        src_grid_native_z = LatitudeLongitudeGrid(arch; size = (Ns_x, Ns_y, Nz_native),
                                                  longitude = λ_src, latitude = φ_src,
                                                  z = z_native, halo = (7, 7, 7))
        @info @sprintf(" -> native-resolution GLORYS u/v for the true depth mean (%d levels vs the model's %d)",
                       Nz_native, Nz)
        fts_u_native = FieldTimeSeries(meta(:u_velocity), src_grid_native_z)
        fts_v_native = FieldTimeSeries(meta(:v_velocity), src_grid_native_z)

        Hg = glorys_deptho_on_grid(src_grid; dir, region)
        p = n_pad + 1
        true_floors = ((-Hg[p, :], -Hg[end-n_pad, :]),
                       (-Hg[:, p], -Hg[:, end-n_pad]))
        zc_nat = collect(znodes(src_grid_native_z, Center()))
        Δz_nat = diff(collect(znodes(src_grid_native_z, Face())))
    end

    # `b` is NaN where GLORYS itself has no ocean; `zc[k] > NaN` is false for every k, so those
    # columns fall out naturally (H stays 0 => ubar 0) without a separate land check.
    function true_depth_mean(col, zc, Δz, floor_line)
        Ns = size(col, 1)
        ubar = zeros(Ns)
        for s in 1:Ns
            b = floor_line[clamp(s, 1, length(floor_line))]
            T = H = 0.0
            for k in 1:size(col, 2)
                v = col[s, k]
                if zc[k] > b && isfinite(v)
                    T += Δz[k] * v
                    H += Δz[k]
                end
            end
            ubar[s] = H > 0 ? T / H : 0.0
        end
        return ubar
    end

    function boundary_transport_slabs_native(fts, floor_pair, dim)
        times = collect(fts.times)
        slabs = map(eachindex(times)) do n
            lo, hi = boundary_columns(Array(interior(fts[n])), dim; Nz_expected = length(zc_nat))
            ū_lo = true_depth_mean(lo, zc_nat, Δz_nat, floor_pair[1])
            ū_hi = true_depth_mean(hi, zc_nat, Δz_nat, floor_pair[2])
            Hwet_lo = last(wet_transport(zeros(size(lo, 1), Nz), floors[dim][1]))
            Hwet_hi = last(wet_transport(zeros(size(hi, 1), Nz), floors[dim][2]))
            (ū_lo .* Hwet_lo, ū_hi .* Hwet_hi)
        end
        return times, first.(slabs), last.(slabs)
    end

    # 02's own diagnostic: implied depth-mean boundary velocity from all three modes, frame 1.
    # legacy tracks the spike; masked and native should both track GLORYS, native more precisely.
    function report_uext(fts, fts_native, dim, names)
        A = Array(interior(fts[1]))
        F = Field(Integral(fts[1], dims = 3)); compute!(F)
        I = Array(interior(F)); p = n_pad + 1
        cols = boundary_columns(A, dim)
        legs = dim == 1 ? (I[p, 1:Ns_y, 1], I[end-n_pad, 1:Ns_y, 1]) :
                          (I[1:Ns_x, p, 1], I[1:Ns_x, end-n_pad, 1])
        cols_n = boundary_columns(Array(interior(fts_native[1])), dim; Nz_expected = length(zc_nat))

        for side in 1:2
            T, Hwet = wet_transport(cols[side], floors[dim][side])
            wet = findall(Hwet .> 0)
            isempty(wet) && continue
            ū_leg = legs[side][wet] ./ Hwet[wet]
            ū_msk = T[wet]          ./ Hwet[wet]
            ū_nat = true_depth_mean(cols_n[side], zc_nat, Δz_nat, true_floors[dim][side])[wet]
            iL, iM, iN = argmax(abs.(ū_leg)), argmax(abs.(ū_msk)), argmax(abs.(ū_nat))
            @info @sprintf("  Uᵉˣᵗ/H_wet, %-5s frame 1: legacy max|ū| = %6.3f m/s at %3d   masked max|ū| = %6.3f m/s at %3d   native max|ū| = %6.3f m/s at %3d",
                           names[side], abs(ū_leg[iL]), wet[iL] - n_pad, abs(ū_msk[iM]), wet[iM] - n_pad,
                           abs(ū_nat[iN]), wet[iN] - n_pad)
        end
    end
    if uext_mode === :native
        report_uext(fts_u, fts_u_native, 1, ("west", "east"))
        report_uext(fts_v, fts_v_native, 2, ("south", "north"))
    end

    if uext_mode === :native
        tU, U_lo, U_hi = boundary_transport_slabs_native(fts_u_native, true_floors[1], 1)   # west, east
        tV, V_lo, V_hi = boundary_transport_slabs_native(fts_v_native, true_floors[2], 2)   # south, north
    else
        tU, U_lo, U_hi = boundary_transport_slabs(fts_u, 1)
        tV, V_lo, V_hi = boundary_transport_slabs(fts_v, 2)
    end

    η_slabs = map(eachindex(fts_η.times)) do n
        A = Array(interior(fts_η[n]))[:, :, 1]
        p = n_pad + 1
        (A[p, :], A[end-n_pad, :], A[:, p], A[:, end-n_pad])   # west, east, south, north
    end
    η_w = [sl[1] for sl in η_slabs]
    η_e = [sl[2] for sl in η_slabs]
    η_s = [sl[3] for sl in η_slabs]
    η_n = [sl[4] for sl in η_slabs]

    # ---------------------------------------------------------------------------
    # From host slabs to boundary conditions.
    #
    # 02 feeds Flather `discrete_form` closures that interpolate the slabs on the fly. That is
    # CPU-only: a boundary condition is passed INTO a GPU kernel, so it must be `isbits`, and a
    # closure over `Vector{Vector{Float64}}` is not -- nor are the `Core.Box`es Julia creates
    # for variables assigned inside a branch and captured afterwards. The kernel compiler
    # rejects both by name.
    #
    # So the slabs become `FieldTimeSeries` instead, the same construction
    # `bsose_barotropic_transport` uses and the same one the 3-D `Interpolated` conditions
    # already rely on: Oceananigans adapts them to device arrays before launch, and they carry
    # their own time interpolation, so nothing is lost.
    #
    # The times are taken from the parent series rather than recomputed. Whatever epoch
    # NumericalEarth put `fts_u.times` on, the 3-D conditions are already on it, so reusing it
    # is what keeps the barotropic and baroclinic boundaries on one clock.
    cpu_grid = arch isa GPU ? on_architecture(CPU(), grid) : grid

    """Host slab (source-indexed) -> per-face `FieldTimeSeries` on the model grid."""
    function face_series(slabs, times, side)
        along_x = side in (:south, :north)
        N = along_x ? Nx : Ny
        fts = along_x ?
              FieldTimeSeries{Center,Nothing,Nothing}(cpu_grid, times; indices = (:, 1, 1)) :
              FieldTimeSeries{Nothing,Center,Nothing}(cpu_grid, times; indices = (1, :, 1))
        for t in eachindex(times)
            slab = slabs[t]
            # model index n  <->  source index n + n_pad, as in 02's `lerp_slab`.
            v = [slab[clamp(n + n_pad, 1, length(slab))] for n in 1:N]
            along_x ? (interior(fts[t], :, 1, 1) .= v) : (interior(fts[t], 1, :, 1) .= v)
            fill_halo_regions!(fts[t])
        end
        return arch isa GPU ? on_architecture(arch, fts) : fts
    end

    U_west_ext,  η_west_ext  = face_series(U_lo, tU, :west),  face_series(η_w, tU, :west)
    U_east_ext,  η_east_ext  = face_series(U_hi, tU, :east),  face_series(η_e, tU, :east)
    V_south_ext, η_south_ext = face_series(V_lo, tV, :south), face_series(η_s, tV, :south)
    V_north_ext, η_north_ext = face_series(V_hi, tV, :north), face_series(η_n, tV, :north)

    report_face_transport(" -> west ", U_lo, grid, :west,  n_pad)
    report_face_transport(" -> east ", U_hi, grid, :east,  n_pad)
    report_face_transport(" -> south", V_lo, grid, :south, n_pad)
    report_face_transport(" -> north", V_hi, grid, :north, n_pad)

    # ---------------- the boundary conditions ----------------
    normal_scheme  = PerturbationAdvection(inflow_timescale = tau_in, outflow_timescale = Inf)
    tangential_sch = NormalRadiation(inflow_timescale = tau_in, outflow_timescale = Inf)
    tracer_scheme  = NormalRadiation(inflow_timescale = tau_in, outflow_timescale = Inf)

    tangential_bc(fts) = tangential === :prescribed ?
        ValueBoundaryCondition(Interpolated(fts)) :
        ValueBoundaryCondition(Interpolated(fts); scheme = tangential_sch)

    maybe_top(bc) = bc === nothing ? NamedTuple() : (; top = bc)

    u_bcs = FieldBoundaryConditions(;
        west  = NormalFlowBoundaryCondition(Interpolated(fts_u); scheme = normal_scheme),
        east  = NormalFlowBoundaryCondition(Interpolated(fts_u); scheme = normal_scheme),
        south = tangential_bc(fts_u),
        north = tangential_bc(fts_u),
        maybe_top(top_u)...)

    v_bcs = FieldBoundaryConditions(;
        south = NormalFlowBoundaryCondition(Interpolated(fts_v); scheme = normal_scheme),
        north = NormalFlowBoundaryCondition(Interpolated(fts_v); scheme = normal_scheme),
        west  = tangential_bc(fts_v),
        east  = tangential_bc(fts_v),
        maybe_top(top_v)...)

    tracer_bcs(fts, top) = FieldBoundaryConditions(;
        west  = ValueBoundaryCondition(Interpolated(fts); scheme = tracer_scheme),
        east  = ValueBoundaryCondition(Interpolated(fts); scheme = tracer_scheme),
        south = ValueBoundaryCondition(Interpolated(fts); scheme = tracer_scheme),
        north = ValueBoundaryCondition(Interpolated(fts); scheme = tracer_scheme),
        maybe_top(top)...)

    U_bcs = FieldBoundaryConditions(grid, (Face(), Center(), nothing);
        west = GravityWaveRadiationBoundaryCondition((U_west_ext,  η_west_ext)),
        east = GravityWaveRadiationBoundaryCondition((U_east_ext,  η_east_ext)))
    V_bcs = FieldBoundaryConditions(grid, (Center(), Face(), nothing);
        south = GravityWaveRadiationBoundaryCondition((V_south_ext, η_south_ext)),
        north = GravityWaveRadiationBoundaryCondition((V_north_ext, η_north_ext)))
    η_bcs = FieldBoundaryConditions(grid, (Center(), Center(), Face());
        west  = SurfaceWaveRadiationBoundaryCondition(), east  = SurfaceWaveRadiationBoundaryCondition(),
        south = SurfaceWaveRadiationBoundaryCondition(), north = SurfaceWaveRadiationBoundaryCondition())

    @info " -> normal: PerturbationAdvection(τ_in = $(tau_in/days) d, τ_out = Inf)"
    @info " -> tangential: $(tangential === :prescribed ? "prescribed (Dirichlet)" : "NormalRadiation(τ_in = $(tau_in/days) d, τ_out = Inf)")"
    @info " -> tracers: NormalRadiation(τ_in = $(tau_in/days) d, τ_out = Inf)"
    @info " -> barotropic: Flather on all four faces; η: Chapman on all four faces"

    bcs = (u = u_bcs, v = v_bcs, T = tracer_bcs(fts_T, top_T), S = tracer_bcs(fts_S, top_S),
           U = U_bcs, V = V_bcs, η = η_bcs)
    series = (u = fts_u, v = fts_v, T = fts_T, S = fts_S, η = fts_η, src_grid = src_grid)
    return GLORYSOpenBoundaries(bcs, series, region)
end


"""
    report_face_transport(label, slabs, grid, side, n_pad)

Net transport each face is handed, in Sv, and the implied depth-mean speed, for the
iteration-0 log.
"""
function report_face_transport(label, slabs, grid, side, n_pad)
    widths, depths = face_geometry(side, grid)
    N = length(widths)
    at(slab, n) = slab[clamp(n + n_pad, 1, length(slab))]
    sv = [sum(at(slab, n) * widths[n] for n in 1:N) / 1e6 for slab in slabs]
    wet = count(>(0), depths)
    peak = wet == 0 ? 0.0 :
           maximum(abs(at(slabs[1], n)) / max(depths[n], eps()) for n in 1:N if depths[n] > 0)
    @info @sprintf("%s Uᵉˣᵗ: %+7.1f Sv (range %+.1f to %+.1f), max |U|/H = %.3f m/s over %d wet columns",
                   label, sum(sv)/length(sv), minimum(sv), maximum(sv), peak, wet)
    return nothing
end


"""
    glorys_initial_conditions!(model; date, dataset, dir, region, remove_eta_mean = true)

Initialise the model from GLORYS, exactly as 02 does: `u`, `v`, `T`, `S` from the frame for
`date`, then the free surface from `zos`.

`remove_eta_mean` is 02's own step. It is kept because the instruction is to follow that
script, but it is worth watching: 02 removes the domain mean from the model's η while feeding
RAW `zos` to Flather, and those two cannot both be the reference. For the MAB the mean is
small enough not to matter. The removed value is logged here so it can be checked -- at this
domain's Flather sensitivity, every centimetre of mismatch is a few Sverdrups, and if it comes
out large the fix is 05's: keep both raw ("removing the domain mean here would make the
boundaries import or export that mean in the first few hours").
"""
function glorys_initial_conditions!(model; date, dataset = GLORYSMonthly(),
                                    dir = glorys_directory(), region,
                                    remove_eta_mean = true)
    set!(model, MetadataSet(:temperature, :salinity, :u_velocity, :v_velocity;
                            dataset, date, dir, region))
    set!((; free_surface = model.free_surface.displacement),
         MetadataSet(:free_surface; dataset, date, dir, region))

    η = model.free_surface.displacement
    η₀ = Array(interior(η))
    wet = filter(isfinite, η₀)
    μ = isempty(wet) ? 0.0 : mean(wet)
    if remove_eta_mean
        parent(η) .-= μ
        fill_halo_regions!(η)
        @info @sprintf(" -> free surface from GLORYS zos; domain mean %+.4f m removed (02's step), residual range %+.3f to %+.3f m",
                       μ, minimum(wet) - μ, maximum(wet) - μ)
        abs(μ) > 0.05 && @warn @sprintf("The removed mean is %.3f m. Flather is still being fed RAW zos, so the interior and the boundary now disagree by that much -- several hundred Sv at this domain's sensitivity. Consider remove_eta_mean = false (05's choice).", μ)
    else
        @info @sprintf(" -> free surface from GLORYS zos, datum intact: mean %+.4f m, range %+.3f to %+.3f m",
                       μ, minimum(wet), maximum(wet))
    end
    return model
end
