# ==============================================================================================
# salt_budget.jl -- where the salt (and so the fresh water) of the bands south of 60S goes, run against B-SOSE.
#
#     BSOSE_DIR=/storage/scratch1/1/vnguyen480/SOWinWaRP/data julia --project=. tools/salt_budget.jl <RUN> [<RUN> ...]
#
# WHY. On B-SOSE's own surface fluxes (_bsoseflux) the 3-D ocean builds a fresh cap south of 60S that B-SOSE does not:
# the same surface input, so the difference is in how each ocean moves the salt around. This splits each box's salt
# change into the surface input and the transport through each side, for the run and for B-SOSE, over the same span.
#
# BOXES (BANDS, default -65,-60): the band from the coast (70S) to the first edge, and from there to the next; each
# 90-150E, full depth and 0-LAYER m (default 210, a level face on B-SOSE i156's grid, which the nest shares).
#
# TERMS, each as the change it makes to the BOX-MEAN salinity (g/kg), summed over a period:
#   storage    change of the box's salt content (∫ S dV) from first to last record centre
#   surface    salt put in at the top, in virtual-salt form: -(salt flux out) for the run (its _surface_fluxes.nc:
#              ice_salt_flux or brine_salt_flux, + S_top * freshwater_flux, + the Hammond & Jones flux the run added
#              as a forcing, if its metadata names it), SFLUX/ρ₀ - S_top oceFWflx/ρ₀ for B-SOSE (its monthly files)
#   north, south, west, east   transport INTO the box through that side, of (S - S_REF): the freshwater-carrying part.
#              Taking S_REF out makes the run (no net volume through the box: virtual salt flux) and B-SOSE (real
#              fresh water, which leaves sideways as volume) comparable; S_REF (default 34.5) is close to the box S.
#      run     EXACT if the run wrote <RUN>_faces.nc (FACE_FLUXES): the model's own advective salt flux and volume
#              transport through each face, eddies in, and (0-LAYER) the floor term Fz + Dz - S_REF Qz.
#              Otherwise MEAN FLOW ONLY: the 30-day means of u, v, S in <RUN>_eke.nc, at cell centres, averaged onto
#              each face; the eddy transport and the rebuild error land in the residual.
#      B-SOSE  total: ADVx/ADVy + DFxE/DFyE _SLT (monthly, read remotely from BUDGET_URL), minus S_REF x the volume
#              through the face (monthly UVEL/VVEL). Also its mean-flow part from the monthly UVEL, VVEL, SALT, done
#              the run's way, so B-SOSE's eddy part = total - mean flow.
#   bottom     (0-LAYER only, B-SOSE only) ADVr + DFrI _SLT - S_REF WVEL rA through the layer's floor, + upward = into the layer
#   residual   storage - surface - sides (- bottom): for the run the eddy transport + (layer) vertical exchange +
#              errors; for B-SOSE full depth, the closure error of this bookkeeping (S_top vs S_REF, η)
#
# PERIODS: year 1 and year 2 split at SPLIT (default 2015-03-01), and the whole span. Each side's records sit at their
# own times (run: 30-day window centres; B-SOSE: monthly means), so a period is integrated on each side's own clock.
#
# WRITES OUTDIR/salt_budget_<RUN>_<box>.png (cumulative terms against time, run beside B-SOSE) and the tables on stdout.
# B-SOSE's numbers are cached in OUTDIR (.salt_budget_bsose_<key>.jls): the remote reads take minutes.
#
# ENV: DIR (.), OUTDIR (plots), BSOSE_DIR (required), BUDGET_URL, BANDS (-65,-60), LAYER (210), S_REF (34.5),
#      SPLIT (2015-03-01), COASTAL (coastal_freshwater_hj.nc; used only for a run whose metadata names Hammond & Jones)
# ==============================================================================================

ENV["HDF5_USE_FILE_LOCKING"] = "FALSE"
using NCDatasets, CairoMakie, Dates, Printf, Statistics, Serialization
include(joinpath(@__DIR__, "model_grid.jl"))   # file_levels

const RUNS = copy(ARGS)
isempty(RUNS) && error("give one or more run names: julia --project=. tools/salt_budget.jl <RUN> [<RUN> ...]")
const DIR        = get(ENV, "DIR", ".")
const OUTDIR     = get(ENV, "OUTDIR", "plots")
const BSOSE_DIR  = get(ENV, "BSOSE_DIR", "")
isempty(BSOSE_DIR) && error("set BSOSE_DIR to the directory with B-SOSE's monthly Salt, Uvel, Vvel, surfSflx, oceFWflx")
const BUDGET_URL = get(ENV, "BUDGET_URL", "https://sose.ucsd.edu/SO6/ITER156/MONTHLY/")
const EDGES      = parse.(Float64, split(get(ENV, "BANDS", "-65,-60"), ','))
const LAYER      = parse(Float64, get(ENV, "LAYER", "210"))
const S_REF      = parse(Float64, get(ENV, "S_REF", "34.5"))
const SPLIT      = DateTime(get(ENV, "SPLIT", "2015-03-01"))
const COASTAL    = get(ENV, "COASTAL", "coastal_freshwater_hj.nc")
const ρ₀         = 1026.0
const R_EARTH    = 6.371e6
const SOUTH      = -70.0                      # the nest's southern wall; B-SOSE is cut there too
const λW, λE     = 90.0, 150.0
const REACH      = 6                          # as src/coastal_freshwater.jl
const DAY        = 86400.0

f64(A) = Float64.(coalesce.(A, NaN))
fill0(A) = (B = f64(A); B[.!isfinite.(B)] .= 0; B)
coord(ds, name, dim) = (v = f64(ds[name][:]); ndims(v) == 1 ? v : dim == 1 ? v[:, 1] : v[1, :])

# Boxes: (name, southern edge, northern edge); the first starts at the coast
const BOXES = [(i == 1 ? @sprintf("coast-%.0fS", -EDGES[1]) : @sprintf("%.0f-%.0fS", -EDGES[i-1], -EDGES[i]),
                i == 1 ? SOUTH : EDGES[i-1], EDGES[i]) for i in eachindex(EDGES)]
const LAYERS = [("full depth", Inf), (@sprintf("0-%.0f m", LAYER), LAYER)]
const SIDES = (:north, :south, :west, :east)

# ── the run ──────────────────────────────────────────────────────────────────────────────────────
"""
Per 30-day window of <RUN>_eke.nc and per (box, layer): salt content, side transports of (S - S_REF) and volume, and
the surface input averaged over the window. Every flux in psu m³/s, contents in psu m³.
"""
function run_budget(run)
    eke_path = joinpath(DIR, "$(run)_eke.nc")
    isfile(eke_path) || error("$eke_path not found (set DIR to the run directory)")
    eke = NCDataset(eke_path)
    λ, φ = f64(eke["λ_caa"][:]), f64(eke["φ_aca"][:])
    lv = file_levels(eke); lv === nothing && error("$eke_path carries no vertical levels")
    zc, zf, _ = lv                                       # bottom to top (Oceananigans)
    Nx, Ny, Nz = length(λ), length(φ), length(zc)
    size(eke["S"], 3) == Nz || error("$eke_path holds $(size(eke["S"], 3)) levels of S, not the full $Nz")
    hb = haskey(eke, "bottom_height") ? fill0(eke["bottom_height"][:, :]) :
         NCDataset(d -> fill0(d["bottom_height"][:, :]), joinpath(DIR, "$(run)_surface.nc"))
    start = DateTime(string(eke.attrib["start_date"]))
    window = haskey(eke.attrib, "averaging_window_seconds") ? Float64(eke.attrib["averaging_window_seconds"]) : 30DAY
    # the run ADDED Hammond & Jones as a forcing (not in its flux file): _ibobc's "constant, from coastal_freshwater_hj.nc
    # (Hammond & Jones 2016)". _bsoseflux's "none added (Hammond & Jones is inside oceFWflx)" must not count.
    cf = lowercase(string(get(eke.attrib, "coastal_freshwater", "")))
    hj = occursin("hammond", cf) && !startswith(cf, "none")
    @info "$run: coastal_freshwater metadata \"$cf\" -> Hammond & Jones $(hj ? "ADDED to" : "not added to") the surface term"
    tend = f64(eke["time"][:])
    recs = findall(>(0), tend)
    times = [start + Millisecond(round(Int, 1000 * (t - window / 2))) for t in tend[recs]]

    Δλ, Δφ = deg2rad(λ[2] - λ[1]), deg2rad(φ[2] - φ[1])
    area = [R_EARTH^2 * Δλ * Δφ * cosd(φ[j]) for i in 1:Nx, j in 1:Ny]
    dy = R_EARTH * Δφ
    dxf(φf) = R_EARTH * Δλ * cosd(φf)
    dz = [hb[i, j] < 0 ? max(0.0, zf[k+1] - max(zf[k], hb[i, j])) : 0.0 for i in 1:Nx, j in 1:Ny, k in 1:Nz]
    inlayer(L) = [zf[k] >= -L - 1e-6 for k in 1:Nz]

    # rows of each box; a box edge is a face of the run's grid
    rows(b) = findall(y -> BOXES[b][2] < y < BOXES[b][3], φ)
    for b in eachindex(BOXES)
        jr = rows(b)
        isempty(jr) && error("box $(BOXES[b][1]) has no rows in the run")
        @info @sprintf("%s box %s: rows %d-%d, northern face %.2fS", run, BOXES[b][1], first(jr), last(jr),
                       -(φ[last(jr)] + (φ[2] - φ[1]) / 2))
    end

    # surface input per day (psu m³/s per box), then onto the windows
    fl_path, sf_path = joinpath(DIR, "$(run)_surface_fluxes.nc"), joinpath(DIR, "$(run)_surface.nc")
    fl, sf = NCDataset(fl_path), NCDataset(sf_path)
    saltname = haskey(fl, "brine_salt_flux") ? "brine_salt_flux" : haskey(fl, "ice_salt_flux") ? "ice_salt_flux" :
               error("$fl_path has neither brine_salt_flux nor ice_salt_flux")
    tf, ts = f64(fl["time"][:]), f64(sf["time"][:])
    fwin = haskey(fl.attrib, "averaging_window_seconds") ? Float64(fl.attrib["averaging_window_seconds"]) : DAY
    readday(d, name, n) = (v = d[name]; fill0(ndims(v) == 4 ? v[:, :, 1, n] : v[:, :, n]))
    wet = hb .< 0
    G = hj ? coastal_field(λ, φ, wet) : zeros(Nx, Ny)
    hj && @info @sprintf("%s: Hammond & Jones from %s, %.0f Gt/yr on the run's wet cells, added to the surface term",
                         run, COASTAL, sum(G .* area) * 365.25DAY / 1e12)
    frecs = findall(>(0), tf)
    fin = zeros(length(frecs), length(BOXES))
    for (q, n) in enumerate(frecs)
        s = argmin(abs.(ts .- (tf[n] - fwin / 2)))
        Stop = readday(sf, "S", s)
        out = readday(fl, saltname, n) .+ Stop .* readday(fl, "freshwater_flux", n) .+ Stop .* G ./ ρ₀
        for b in eachindex(BOXES)
            jr = rows(b)
            fin[q, b] = -sum(out[:, jr] .* area[:, jr] .* wet[:, jr])
        end
    end
    close(fl); close(sf)

    nb, nl, nt = length(BOXES), length(LAYERS), length(recs)
    C = zeros(nt, nb, nl)
    side = Dict(s => zeros(nt, nb, nl) for s in SIDES)
    vol = Dict(s => zeros(nt, nb, nl) for s in SIDES)
    surf = zeros(nt, nb)
    for (q, n) in enumerate(recs)
        S = fill0(eke["S"][:, :, :, n]); u = fill0(eke["u"][:, :, :, n]); v = fill0(eke["v"][:, :, :, n])
        Sa = S .- S_REF
        for b in eachindex(BOXES), (l, (_, L)) in enumerate(LAYERS)
            jr, ks = rows(b), inlayer(L)
            for k in 1:Nz
                ks[k] || continue
                C[q, b, l] += sum(S[:, jr, k] .* area[:, jr] .* dz[:, jr, k])
                # north face, between rows jN and jN+1 (into the box = southward)
                for (sd, jin, jout, sgn) in ((:north, last(jr), last(jr) + 1, -1.0), (:south, first(jr), first(jr) - 1, 1.0))
                    1 <= jout <= Ny || continue
                    φf = (φ[jin] + φ[jout]) / 2
                    for i in 1:Nx
                        h = min(dz[i, jin, k], dz[i, jout, k]); h > 0 || continue
                        vf = (v[i, jin, k] + v[i, jout, k]) / 2
                        side[sd][q, b, l] += sgn * vf * (Sa[i, jin, k] + Sa[i, jout, k]) / 2 * dxf(φf) * h
                        vol[sd][q, b, l] += sgn * vf * dxf(φf) * h
                    end
                end
                # west and east: the domain's open sides; the first and last column's centre values
                for (sd, i, sgn) in ((:west, 1, 1.0), (:east, Nx, -1.0)), j in jr
                    h = dz[i, j, k]; h > 0 || continue
                    side[sd][q, b, l] += sgn * u[i, j, k] * Sa[i, j, k] * dy * h
                    vol[sd][q, b, l] += sgn * u[i, j, k] * dy * h
                end
            end
        end
        lo, hi = tend[n] - window, tend[n]
        sel = [q2 for (q2, m) in enumerate(frecs) if lo < tf[m] <= hi]
        isempty(sel) || (surf[q, :] .= vec(mean(fin[sel, :], dims = 1)))
        q == 1 && @info @sprintf("%s: %d windows; first %s", run, nt, times[1])
    end
    V = [sum(area[:, rows(b)] .* dropdims(sum(dz[:, rows(b), :] .* reshape(inlayer(L), 1, 1, Nz), dims = 3), dims = 3))
         for b in eachindex(BOXES), (_, L) in LAYERS]

    # EXACT sides and floor from <RUN>_faces.nc when the run wrote it (FACE_FLUXES): the model's own advective salt flux
    # (Fx, Fy, Fz, psu m³/s, eddies in) and volume transport (Qx, Qy, Qz) through every face, and its vertical diffusive
    # flux Dz, 30-day means on the same windows as _eke.nc. They replace the mean-flow rebuild above.
    faces_path = joinpath(DIR, "$(run)_faces.nc")
    exact = isfile(faces_path)
    bottom = floorvol = fwvol = nothing
    if exact
        fc = NCDataset(faces_path)
        tfc = f64(fc["time"][:])
        size(fc["Fx"], 1) == Nx + 1 && size(fc["Fy"], 2) == Ny + 1 && size(fc["Fz"], 3) == Nz + 1 ||
            error("$faces_path: Fx/Fy/Fz are $(size(fc["Fx"]))/$(size(fc["Fy"]))/$(size(fc["Fz"])), not on the faces of $(Nx)x$(Ny)x$(Nz)")
        bottom, floorvol, fwvol = zeros(nt, nb), zeros(nt, nb), zeros(nt, nb)   # static z: no fresh water as volume
        for (q, n) in enumerate(recs)
            m = argmin(abs.(tfc .- tend[n]))
            abs(tfc[m] - tend[n]) < 3600 || error("$faces_path has no window ending at $(tend[n]) s (nearest $(tfc[m]))")
            Fx, Qx = fill0(fc["Fx"][:, :, :, m]), fill0(fc["Qx"][:, :, :, m])
            Fy, Qy = fill0(fc["Fy"][:, :, :, m]), fill0(fc["Qy"][:, :, :, m])
            for b in eachindex(BOXES), (l, (_, L)) in enumerate(LAYERS)
                jr, ks = rows(b), findall(inlayer(L))
                jS, jN = first(jr), last(jr)
                Ssum(F, Q) = sum(F) - S_REF * sum(Q)
                side[:north][q, b, l] = -Ssum(Fy[:, jN+1, ks], Qy[:, jN+1, ks]); vol[:north][q, b, l] = -sum(Qy[:, jN+1, ks])
                side[:south][q, b, l] =  Ssum(Fy[:, jS, ks], Qy[:, jS, ks]);     vol[:south][q, b, l] =  sum(Qy[:, jS, ks])
                side[:west][q, b, l]  =  Ssum(Fx[1, jr, ks], Qx[1, jr, ks]);     vol[:west][q, b, l]  =  sum(Qx[1, jr, ks])
                side[:east][q, b, l]  = -Ssum(Fx[Nx+1, jr, ks], Qx[Nx+1, jr, ks]); vol[:east][q, b, l] = -sum(Qx[Nx+1, jr, ks])
            end
            # the 0-LAYER floor: the face under the layer's lowest level, + up = into the layer; Dz only where wet on
            # both sides (it is ∂S/∂z across the floor elsewhere)
            k0 = findfirst(inlayer(LAYER))
            Fz, Qz, Dz = fill0(fc["Fz"][:, :, k0, m]), fill0(fc["Qz"][:, :, k0, m]), fill0(fc["Dz"][:, :, k0, m])
            k0 > 1 || error("the layer reaches the bottom of the grid")
            wetboth = (dz[:, :, k0] .> 0) .& (dz[:, :, k0-1] .> 0)
            for b in eachindex(BOXES)
                jr = rows(b)
                bottom[q, b] = sum(Fz[:, jr] .+ Dz[:, jr] .* wetboth[:, jr])
                floorvol[q, b] = sum(Qz[:, jr])
            end
        end
        close(fc)
        @info @sprintf("%s: EXACT side and floor fluxes from %s (layer floor at %.1f m)", run, faces_path, -zf[findfirst(inlayer(LAYER))])
    else
        @info "$run: no $(run)_faces.nc; sides are the mean flow rebuilt from _eke.nc (eddies and errors in the residual)"
    end
    close(eke)
    out = (; name = run, times, C, side, vol, surf, V, mean_flow_only = !exact, bottom)
    return exact ? merge(out, (; floorvol, fwvol)) : out
end

"""The run's coastal fresh water on its grid (kg m⁻² s⁻¹, + in), land cells spread over wet ones within REACH, as
src/coastal_freshwater.jl does."""
function coastal_field(lon, lat, wet)
    isfile(COASTAL) || error("$COASTAL not found: run from the repo root, or set COASTAL to the file the run used")
    F, flon, flat, move = NCDataset(COASTAL) do d
        fill0(d["freshwater_flux"][:, :]), f64(d["lon"][:]), f64(d["lat"][:]), startswith(lowercase(get(d.attrib, "on_land", "")), "move")
    end
    Nx, Ny = size(wet)
    (size(F) == (Nx, Ny) && maximum(abs, flon .- lon) < 1e-6 && maximum(abs, flat .- lat) < 1e-6) ||
        error("$COASTAL is not on the run's grid")
    area = [cosd(lat[j]) for i in 1:Nx, j in 1:Ny]
    G = ifelse.(wet, F, 0.0)
    move || return G
    for I in CartesianIndices(F)
        (wet[I] || F[I] <= 0) && continue
        i, j = Tuple(I)
        ts = [(ii, jj) for di in -REACH:REACH, dj in -REACH:REACH for (ii, jj) in ((i + di, j + dj),)
              if 1 <= ii <= Nx && 1 <= jj <= Ny && wet[ii, jj] && (di * cosd(lat[j]))^2 + dj^2 <= REACH^2]
        isempty(ts) && continue
        ΔF = F[I] * area[I] / sum(area[t...] for t in ts)
        for t in ts; G[t...] += ΔF; end
    end
    return G
end

# ── B-SOSE ───────────────────────────────────────────────────────────────────────────────────────
const MONTHLY_SPAN = "2013to2024_monthly"

"A monthly file: BSOSE_DIR's copy if there is one, else BUDGET_URL read by byte ranges."
function monthly_path(token)
    want = lowercase("$(token)_bsoseI156_$(MONTHLY_SPAN).nc")
    hits = isdir(BSOSE_DIR) ? filter(f -> lowercase(f) == want, readdir(BSOSE_DIR)) : String[]
    isempty(hits) || return joinpath(BSOSE_DIR, first(hits))
    return BUDGET_URL * "$(token)_bsoseI156_$(MONTHLY_SPAN).nc#mode=bytes"
end

function open_monthly(f, path)
    for attempt in 1:3
        try
            return NCDataset(f, path)
        catch e
            (attempt == 3 || !endswith(path, "#mode=bytes")) && rethrow()
            @warn "reading $path failed (attempt $attempt of 3); retrying in 30 s" exception = e
            sleep(30)
        end
    end
end

data_var(ds, names) = (for n in names; haskey(ds, n) && return n; end;
                       c = [k for k in keys(ds) if ndims(ds[k]) >= 3]; length(c) == 1 && return only(c);
                       m = [k for k in c if any(n -> occursin(lowercase(n), lowercase(k)), names)];
                       length(m) == 1 ? only(m) : error("none of $names; $(c)"))

"(year, month) a monthly record averages: its stamp is the END of the month and drifts by hours."
monthkey(s) = (d = DateTime(s) - Day(14); (year(d), month(d)))

function month_index(ds, months, path)
    have = monthkey.(ds["time"][:])
    idx = [findfirst(==(m), have) for m in months]
    any(isnothing, idx) && error("$path has no record for $(months[findfirst(isnothing, idx)])")
    idx == collect(first(idx):last(idx)) || error("$path: the monthly records are not consecutive")
    return first(idx):last(idx)
end

"Read `token`'s variable over (is, js, levels, months); a 2-D field (x, y, time) ignores `levels`."
function read_monthly(token, names, is, js, months; levels = Colon())
    path = monthly_path(token)
    open_monthly(path) do ds
        m = month_index(ds, months, path)
        v = ds[data_var(ds, names)]
        fill0(ndims(v) == 3 ? v[is, js, m] : v[is, js, levels, m])
    end
end

function bsose_budget(months)
    key = @sprintf("%s_L%.0f_ref%.2f_%04d%02d-%04d%02d", join(string.(Int.(EDGES)), "_"), LAYER, S_REF, months[1]..., months[end]...)
    cache = joinpath(OUTDIR, ".salt_budget_bsose_$key.jls")
    isfile(cache) && (@info "B-SOSE budget from cache $cache"; return deserialize(cache))
    t0 = time()
    # geometry, from the budget files that carry it, read on the region only (rows js0:jtop+1, the columns I and
    # one either side): rA, hFacC, drF (TOTTTEND, as tools/single_column_bsose.jl reads them); hFacW, dyG on the west and east faces (ADVx); hFacS, dxG (ADVy)
    xc, yc, drF, RF = open_monthly(monthly_path("TOTTTEND")) do ds
        dr = f64(ds["drF"][:])
        coord(ds, "XC", 1), coord(ds, "YC", 2), dr, haskey(ds, "RF") ? f64(ds["RF"][:]) : vcat(0.0, -cumsum(dr))
    end
    Nz = length(drF)
    I = findall(x -> λW < x < λE, xc); i0, i1 = first(I), last(I)
    length(I) == 360 || @warn "B-SOSE has $(length(I)) columns in $(λW)-$(λE)E (the nest 360)"
    js0, jtop = findfirst(>(SOUTH), yc), findlast(<(EDGES[end]), yc)
    JR = js0:jtop+1                                                            # region rows, one past the last face
    sh(j) = j - js0 + 1
    need(ds, n, path) = haskey(ds, n) || error("$path carries no $n (needed for the box geometry)")
    rA, hC = open_monthly(monthly_path("TOTTTEND")) do ds
        need(ds, "rA", "TOTTTEND"); need(ds, "hFacC", "TOTTTEND")
        fill0(ds["rA"][I, JR]), fill0(ds["hFacC"][I, JR, :])
    end
    hW, dyG = open_monthly(monthly_path("ADVx_SLT")) do ds                    # the west (1) and east (2) faces
        need(ds, "hFacW", "ADVx_SLT"); need(ds, "dyG", "ADVx_SLT")
        cat(fill0(ds["hFacW"][i0:i0, JR, :]), fill0(ds["hFacW"][i1+1:i1+1, JR, :]), dims = 1),
        vcat(fill0(ds["dyG"][i0:i0, JR]), fill0(ds["dyG"][i1+1:i1+1, JR]))
    end
    hS, dxG = open_monthly(monthly_path("ADVy_SLT")) do ds
        need(ds, "hFacS", "ADVy_SLT"); need(ds, "dxG", "ADVy_SLT")
        fill0(ds["hFacS"][I, JR, :]), fill0(ds["dxG"][I, JR])
    end
    # the local monthly files must be on the same (global) grid as the remote budget files. Compared by the data
    # variable's horizontal size, not by coordinate names: UVEL sits on XG, VVEL on YG, the rest on XC, YC.
    for (tok, names) in (("Salt", ("SALT", "Salt")), ("Uvel", ("UVEL", "Uvel")), ("Vvel", ("VVEL", "Vvel")),
                         ("surfSflx", ("SFLUX",)), ("oceFWflx", ("oceFWflx",)))
        open_monthly(monthly_path(tok)) do ds
            nx, ny = size(ds[data_var(ds, names)])[1:2]
            (nx == length(xc) && ny == length(yc)) ||
                error("$(monthly_path(tok)) is $(nx)x$(ny), the budget files $(length(xc))x$(length(yc)): not the same grid")
        end
    end
    ygf = vcat(1.5yc[1] - 0.5yc[2], (yc[1:end-1] .+ yc[2:end]) ./ 2)          # south faces (YG in grid.nc is corrupt)
    kL = findfirst(k -> -RF[k+1] >= LAYER - 1e-3, 1:Nz)                        # last level of the layer
    abs(-RF[kL+1] - LAYER) < 1 || @warn @sprintf("B-SOSE: the layer's floor is at %.1f m, not %.0f", -RF[kL+1], LAYER)

    nb, nl, nt = length(BOXES), length(LAYERS), length(months)
    C = zeros(nt, nb, nl); surf = zeros(nt, nb); bottom = zeros(nt, nb)
    side = Dict(s => zeros(nt, nb, nl) for s in SIDES)               # total (ADV + DF) - S_REF vol
    meanf = Dict(s => zeros(nt, nb, nl) for s in SIDES)              # mean flow, the run's way
    vol = Dict(s => zeros(nt, nb, nl) for s in SIDES)
    V = zeros(nb, nl)
    edges_used = Float64[]
    ksel(l) = l == 1 ? (1:Nz) : (1:kL)

    # SALT over the region once (local monthly file), top-down levels, one column either side of I
    S = read_monthly("Salt", ("SALT", "Salt"), i0-1:i1+1, JR, months)
    Sx(i, j, k, n) = S[i - (i0 - 1) + 1, sh(j), k, n]
    for (b, (_, lo, hi)) in enumerate(BOXES)
        jr = findall(y -> lo < y < hi, yc) ∩ (js0:jtop)
        jS, jN = first(jr), last(jr)
        push!(edges_used, ygf[jN + 1])
        cellV = rA[:, sh.(jr)] .* hC[:, sh.(jr), :] .* reshape(drF, 1, 1, Nz)
        for l in 1:nl
            V[b, l] = sum(cellV[:, :, ksel(l)])
            for n in 1:nt
                C[n, b, l] = sum(S[2:end-1, sh(jS):sh(jN), ksel(l), n] .* cellV[:, :, ksel(l)])
            end
        end
        # side transports: ADV + DF of salt (psu m³/s), and the volume (UVEL/VVEL)
        faces = ((:north, :y, jN + 1, -1.0), (:south, :y, jS, 1.0), (:west, :x, i0, 1.0), (:east, :x, i1 + 1, -1.0))
        for (sd, dir, f, sgn) in faces
            (sd == :south && b == 1) && continue                                # the coast
            if dir == :y
                A = read_monthly("ADVy_SLT", ("ADVy_SLT",), I, f:f, months) .+ read_monthly("DFyE_SLT", ("DFyE_SLT",), I, f:f, months)
                Vv = read_monthly("Vvel", ("VVEL", "Vvel"), I, f:f, months)
                area_f = dxG[:, sh(f)] .* hS[:, sh(f), :] .* reshape(drF, 1, Nz)        # (i, k)
                Sf = [(Sx(i, f - 1, k, n) + Sx(i, f, k, n)) / 2 for i in I, k in 1:Nz, n in 1:nt]
            else
                A = read_monthly("ADVx_SLT", ("ADVx_SLT",), f:f, jr, months) .+ read_monthly("DFxE_SLT", ("DFxE_SLT",), f:f, jr, months)
                Vv = read_monthly("Uvel", ("UVEL", "Uvel"), f:f, jr, months)
                c = f == i0 ? 1 : 2
                area_f = dyG[c, sh.(jr)] .* hW[c, sh.(jr), :] .* reshape(drF, 1, Nz)
                Sf = [(Sx(f - 1, j, k, n) + Sx(f, j, k, n)) / 2 for j in jr, k in 1:Nz, n in 1:nt]
            end
            A, Vv = dropdims(A, dims = dir == :y ? 2 : 1), dropdims(Vv, dims = dir == :y ? 2 : 1)   # (along, k, n)
            for l in 1:nl, n in 1:nt
                ks = ksel(l)
                Q = Vv[:, ks, n] .* area_f[:, ks]
                vol[sd][n, b, l] = sgn * sum(Q)
                side[sd][n, b, l] = sgn * (sum(A[:, ks, n]) - S_REF * sum(Q))
                meanf[sd][n, b, l] = sgn * sum(Q .* (Sf[:, ks, n] .- S_REF))
            end
        end
        # surface: SFLUX/ρ₀ - S_top oceFWflx/ρ₀ over the box's wet top cells
        wetTop = hC[:, sh.(jr), 1] .> 0
        SF = read_monthly("surfSflx", ("SFLUX",), I, jr, months; levels = 1:1)
        FW = read_monthly("oceFWflx", ("oceFWflx",), I, jr, months; levels = 1:1)
        SF = ndims(SF) == 4 ? SF[:, :, 1, :] : SF; FW = ndims(FW) == 4 ? FW[:, :, 1, :] : FW
        for n in 1:nt
            Stop = S[2:end-1, sh(jS):sh(jN), 1, n]
            surf[n, b] = sum((SF[:, :, n] .- Stop .* FW[:, :, n]) ./ ρ₀ .* rA[:, sh.(jr)] .* wetTop)
        end
        # the layer's floor: ADVr + DFrI on the top face of level kL+1 (+ up = into the layer)
        R = read_monthly("ADVr_SLT", ("ADVr_SLT",), I, jr, months; levels = kL+1:kL+1) .+
            read_monthly("DFrI_SLT", ("DFrI_SLT",), I, jr, months; levels = kL+1:kL+1)
        bottom[:, b] = [sum(R[:, :, 1, n]) for n in 1:nt]
        @info @sprintf("B-SOSE box %s: rows %d-%d (%.2f-%.2fS), read in %.0f s so far", BOXES[b][1], jS, jN,
                       -ygf[jS], -ygf[jN + 1], time() - t0)
    end
    times = [DateTime(m[1], m[2], 1) + Hour(12 * daysinmonth(DateTime(m[1], m[2], 1))) for m in months]
    out = (; name = "B-SOSE", times, C, side, meanf, vol, surf, V, bottom, edges_used, layer_floor = -RF[kL+1])
    mkpath(OUTDIR); serialize(cache, out)
    return out
end

"""
B-SOSE's VOLUME fluxes the salt cache lacks, per month and box (m³/s): through the layer's floor (WVEL on the top face
of level kL+1 times rA, + up = into the layer) and through the surface (oceFWflx/ρ₀ times rA, + into the ocean).
The bottom salt term needs the first: sides carry (S - S_REF), so the floor must too, ADVr + DFrI - S_REF W rA.
Without it the 0-LAYER bottom was S_REF x the layer's vertical volume flux (±22 g/kg). Cached on its own, so the
salt cache (hours of remote reads) stays valid.
"""
function bsose_volumes(months)
    key = @sprintf("%s_L%.0f_%04d%02d-%04d%02d", join(string.(Int.(EDGES)), "_"), LAYER, months[1]..., months[end]...)
    cache = joinpath(OUTDIR, ".salt_budget_bsose_vol_$key.jls")
    isfile(cache) && (@info "B-SOSE volumes from cache $cache"; return deserialize(cache))
    t0 = time()
    xc, yc, RF = open_monthly(monthly_path("TOTTTEND")) do ds
        dr = f64(ds["drF"][:])
        coord(ds, "XC", 1), coord(ds, "YC", 2), haskey(ds, "RF") ? f64(ds["RF"][:]) : vcat(0.0, -cumsum(dr))
    end
    Nz = length(RF) - 1
    I = findall(x -> λW < x < λE, xc)
    js0, jtop = findfirst(>(SOUTH), yc), findlast(<(EDGES[end]), yc)
    kL = findfirst(k -> -RF[k+1] >= LAYER - 1e-3, 1:Nz)
    rA = open_monthly(ds -> fill0(ds["rA"][I, js0:jtop]), monthly_path("TOTTTEND"))
    W = read_monthly("Wvel", ("WVEL", "Wvel"), I, js0:jtop, months; levels = kL+1:kL+1)
    F = read_monthly("oceFWflx", ("oceFWflx",), I, js0:jtop, months; levels = 1:1)
    W = ndims(W) == 4 ? W[:, :, 1, :] : W; F = ndims(F) == 4 ? F[:, :, 1, :] : F
    nb, nt = length(BOXES), length(months)
    floorvol, fwvol = zeros(nt, nb), zeros(nt, nb)
    for (b, (_, lo, hi)) in enumerate(BOXES)
        r = (findall(y -> lo < y < hi, yc) ∩ (js0:jtop)) .- js0 .+ 1
        for n in 1:nt
            floorvol[n, b] = sum(W[:, r, n] .* rA[:, r])
            fwvol[n, b] = sum(F[:, r, n] ./ ρ₀ .* rA[:, r])
        end
    end
    @info @sprintf("B-SOSE WVEL at %.1f m and oceFWflx read in %.0f s", -RF[kL+1], time() - t0)
    out = (; floorvol, fwvol)
    mkpath(OUTDIR); serialize(cache, out)
    return out
end

"""
B-SOSE's volume through the box sides and the layer's floor from its MASS-WEIGHTED velocities (UVELMASS, VVELMASS,
WVELMASS: hFac and the r* thickness factor already in, so the face area is dyG drF, dxG drF, rA), per month, box and
layer (m³/s), with the mean-flow transport of (S - S_REF) done with them. UVEL x hFac misses r*'s (1 + η/H): 0.01 Sv
on these boxes, which S_REF made 0.05-0.2 g/kg of residual. Cached on its own; the cached ADV + DF stay valid.
"""
function bsose_mass(months)
    key = @sprintf("%s_L%.0f_ref%.2f_%04d%02d-%04d%02d", join(string.(Int.(EDGES)), "_"), LAYER, S_REF, months[1]..., months[end]...)
    cache = joinpath(OUTDIR, ".salt_budget_bsose_mass_$key.jls")
    isfile(cache) && (@info "B-SOSE mass transports from cache $cache"; return deserialize(cache))
    t0 = time()
    xc, yc, drF, RF = open_monthly(monthly_path("TOTTTEND")) do ds
        dr = f64(ds["drF"][:])
        coord(ds, "XC", 1), coord(ds, "YC", 2), dr, haskey(ds, "RF") ? f64(ds["RF"][:]) : vcat(0.0, -cumsum(dr))
    end
    Nz = length(drF)
    I = findall(x -> λW < x < λE, xc); i0, i1 = first(I), last(I)
    js0, jtop = findfirst(>(SOUTH), yc), findlast(<(EDGES[end]), yc)
    JR = js0:jtop+1
    sh(j) = j - js0 + 1
    rA = open_monthly(ds -> fill0(ds["rA"][I, JR]), monthly_path("TOTTTEND"))
    dyG = open_monthly(ds -> vcat(fill0(ds["dyG"][i0:i0, JR]), fill0(ds["dyG"][i1+1:i1+1, JR])), monthly_path("ADVx_SLT"))
    dxG = open_monthly(ds -> fill0(ds["dxG"][I, JR]), monthly_path("ADVy_SLT"))
    kL = findfirst(k -> -RF[k+1] >= LAYER - 1e-3, 1:Nz)
    nb, nl, nt = length(BOXES), length(LAYERS), length(months)
    vol = Dict(s => zeros(nt, nb, nl) for s in SIDES)
    meanf = Dict(s => zeros(nt, nb, nl) for s in SIDES)
    floorvol = zeros(nt, nb)
    ksel(l) = l == 1 ? (1:Nz) : (1:kL)
    S = read_monthly("Salt", ("SALT", "Salt"), i0-1:i1+1, JR, months)
    Sx(i, j, k, n) = S[i - (i0 - 1) + 1, sh(j), k, n]
    for (b, (_, lo, hi)) in enumerate(BOXES)
        jr = findall(y -> lo < y < hi, yc) ∩ (js0:jtop)
        jS, jN = first(jr), last(jr)
        faces = ((:north, :y, jN + 1, -1.0), (:south, :y, jS, 1.0), (:west, :x, i0, 1.0), (:east, :x, i1 + 1, -1.0))
        for (sd, dir, f, sgn) in faces
            (sd == :south && b == 1) && continue
            if dir == :y
                Vv = read_monthly("VvelMass", ("VVELMASS", "VvelMass"), I, f:f, months)
                area_f = dxG[:, sh(f)] .* reshape(drF, 1, Nz)
                Sf = [(Sx(i, f - 1, k, n) + Sx(i, f, k, n)) / 2 for i in I, k in 1:Nz, n in 1:nt]
            else
                Vv = read_monthly("UvelMass", ("UVELMASS", "UvelMass"), f:f, jr, months)
                area_f = dyG[f == i0 ? 1 : 2, sh.(jr)] .* reshape(drF, 1, Nz)
                Sf = [(Sx(f - 1, j, k, n) + Sx(f, j, k, n)) / 2 for j in jr, k in 1:Nz, n in 1:nt]
            end
            Vv = dropdims(Vv, dims = dir == :y ? 2 : 1)
            for l in 1:nl, n in 1:nt
                ks = ksel(l)
                Q = Vv[:, ks, n] .* area_f[:, ks]
                vol[sd][n, b, l] = sgn * sum(Q)
                meanf[sd][n, b, l] = sgn * sum(Q .* (Sf[:, ks, n] .- S_REF))
            end
        end
        W = read_monthly("WvelMass", ("WVELMASS", "WvelMass"), I, jr, months; levels = kL+1:kL+1)
        W = ndims(W) == 4 ? W[:, :, 1, :] : W
        for n in 1:nt
            floorvol[n, b] = sum(W[:, :, n] .* rA[:, sh.(jr)])
        end
        @info @sprintf("B-SOSE mass transports, box %s: read in %.0f s so far", BOXES[b][1], time() - t0)
    end
    out = (; vol, meanf, floorvol)
    mkpath(OUTDIR); serialize(cache, out)
    return out
end

# ── integrals over a period on each side's own clock ─────────────────────────────────────────────
secs(d) = Dates.value(d - DateTime(2000)) / 1000     # seconds since 2000, a plain Float64

"x at time s (seconds), linear between records, held at the ends."
function at_s(t, x, s)
    s <= t[1] && return x[1]; s >= t[end] && return x[end]
    n = searchsortedlast(t, s); w = (s - t[n]) / (t[n+1] - t[n])
    return (1 - w) * x[n] + w * x[n+1]
end
at(times, x, d) = at_s(secs.(times), x, secs(d))

"∫ from a to b (dates) of a flux known at the record times, trapezoidal on the records' own clock (seconds)."
function integral(times, f, a, b)
    t = secs.(times); sa, sb = secs(a), secs(b)
    g = vcat(sa, filter(s -> sa < s < sb, t), sb)
    return sum((g[k+1] - g[k]) * (at_s(t, f, g[k]) + at_s(t, f, g[k+1])) / 2 for k in 1:length(g)-1; init = 0.0)
end

function terms(B, b, l, a, z)
    V = B.V[b, l]
    tr(x) = integral(B.times, x, a, z) / V
    st = (at(B.times, B.C[:, b, l], z) - at(B.times, B.C[:, b, l], a)) / V
    sf = tr(B.surf[:, b])                         # the top cell is in both layers
    sides = Dict(s => tr(B.side[s][:, b, l]) for s in SIDES)
    bot = (l == 2 && B.bottom !== nothing) ? tr(B.bottom[:, b] .- S_REF .* B.floorvol[:, b]) : NaN
    resid = st - sf - sum(values(sides)) - (isnan(bot) ? 0.0 : bot)
    mf = hasproperty(B, :meanf) ? Dict(s => tr(B.meanf[s][:, b, l]) for s in SIDES) : nothing
    return (; st, sf, sides, bot, resid, mf)
end

# ── main ─────────────────────────────────────────────────────────────────────────────────────────
runs = [run_budget(r) for r in RUNS]
t_first = minimum(first(r.times) for r in runs); t_last = maximum(last(r.times) for r in runs)
months = let m = Tuple{Int,Int}[], d = DateTime(year(t_first), month(t_first))
    while d <= t_last; push!(m, (year(d), month(d))); d += Month(1); end; m
end
bs = let B = bsose_budget(months), bm = bsose_mass(months), bv = bsose_volumes(months)
    # the cached side = sgn (ADV + DF - S_REF Q_uvel), vol = sgn Q_uvel: swap Q_uvel for the mass-weighted Q
    side = Dict(s => B.side[s] .+ S_REF .* (B.vol[s] .- bm.vol[s]) for s in SIDES)
    merge(B, (; side, vol = bm.vol, meanf = bm.meanf, floorvol = bm.floorvol, fwvol = bv.fwvol))
end
@info "B-SOSE box edges used (its faces nearest yours): " * join([@sprintf("%.2fS", -e) for e in bs.edges_used], ", ") *
      @sprintf("; layer floor %.1f m", bs.layer_floor)

periods = let a = max(first(bs.times), minimum(first(r.times) for r in runs)), z = min(last(bs.times), minimum(last(r.times) for r in runs))
    [(@sprintf("%s to %s", Dates.format(a, "yyyy-mm-dd"), Dates.format(SPLIT, "yyyy-mm-dd")), a, SPLIT),
     (@sprintf("%s to %s", Dates.format(SPLIT, "yyyy-mm-dd"), Dates.format(z, "yyyy-mm-dd")), SPLIT, z),
     ("whole span", a, z)]
end

println("\n", "="^118)
println("SALT BUDGET of the bands south of $(-EDGES[end])S, 90-150E: each term as the change it makes to the BOX-MEAN salinity (g/kg)")
@printf("  sides: transport INTO the box of (S - %.2f)\n", S_REF)
for r in runs
    println("  ", r.name, r.mean_flow_only ? ": MEAN FLOW of its 30-day means only (eddies and rebuild errors in its residual)" :
                                             ": EXACT, the model's own face fluxes (_faces.nc); residual = closure error")
end
println("  B-SOSE sides = ADV + DF (total); 'B-SOSE mean' = its monthly-mean flow, done the run's way; eddy = total - mean")
println("  residual = storage - surface - sides (- bottom): run = eddies + vertical (layer) + errors; B-SOSE = vertical (layer)/closure")
println("="^118)
for (b, (bname, _, _)) in enumerate(BOXES), (l, (lname, _)) in enumerate(LAYERS)
    @printf("\n  %s, %s   (box volume: run %.3g m³, B-SOSE %.3g m³)\n", uppercase(bname), lname, runs[1].V[b, l], bs.V[b, l])
    @printf("  %-26s %-14s | %8s %8s | %8s %8s %8s %8s | %8s %8s\n", "period", "source", "storage", "surface",
            "north", "south", "west", "east", "bottom", "residual")
    for (pname, a, z) in periods
        for B in (runs..., bs)
            T = terms(B, b, l, a, z)
            @printf("  %-26s %-14s | %+8.4f %+8.4f | %+8.4f %+8.4f %+8.4f %+8.4f | %8s %+8.4f\n", pname,
                    first(B.name, 14), T.st, T.sf, T.sides[:north], T.sides[:south], T.sides[:west], T.sides[:east],
                    isnan(T.bot) ? "-" : @sprintf("%+8.4f", T.bot), T.resid)
            if T.mf !== nothing
                @printf("  %-26s %-14s | %8s %8s | %+8.4f %+8.4f %+8.4f %+8.4f |\n", "", "  B-SOSE mean", "", "",
                        T.mf[:north], T.mf[:south], T.mf[:west], T.mf[:east])
            end
        end
    end
end

# volume through the sides (Sv), whole span: the run's mean flow against B-SOSE's
println("\n  VOLUME INTO each box through each side, mean over the whole span (Sv; + into the box)")
println("  net = sides (+ B-SOSE's fresh water through the surface; + its floor, layer): ~0 if the bookkeeping closes")
println("  B-SOSE volumes from UVELMASS/VVELMASS/WVELMASS (r* included)")
println("  (the run's full-depth net must be 0: static z, virtual salt flux; S_REF x net x span / box volume is its salt error)")
let a = periods[end][2], z = periods[end][3]
    for (b, (bname, _, _)) in enumerate(BOXES), (l, (lname, _)) in enumerate(LAYERS)
        @printf("  %-12s %-12s", bname, lname)
        for B in (runs..., bs)
            mv(x) = integral(B.times, x, a, z) / (secs(z) - secs(a)) / 1e6
            v = [mv(B.vol[s][:, b, l]) for s in SIDES]
            @printf(" | %s N %+6.2f S %+6.2f W %+6.2f E %+6.2f", first(B.name, 10), v...)
            if hasproperty(B, :fwvol)
                fw = mv(B.fwvol[:, b])                    # the top cell is in both layers
                fl = l == 2 ? mv(B.floorvol[:, b]) : 0.0
                @printf(" FW %+6.3f floor %+6.3f net %+6.3f", fw, fl, sum(v) + fw + fl)
            else
                @printf(" net %+6.3f%s", sum(v), l == 1 ? "" : " (+floor)")
            end
        end
        println()
    end
end

# figures: cumulative terms from the start of the whole span, per box and layer
mkpath(OUTDIR)
for r in runs
    fig = Figure(size = (1500, 420 * length(BOXES)))
    (_, a, z) = periods[end]
    for (b, (bname, _, _)) in enumerate(BOXES), (l, (lname, _)) in enumerate(LAYERS)
        ax = Axis(fig[b, l]; title = "$bname, $lname", ylabel = "Δ box-mean S (g/kg)",
                  xticklabelrotation = π / 5)
        for (B, style) in ((r, :solid), (bs, :dash))
            ts = filter(d -> a <= d <= z, B.times)
            T = [terms(B, b, l, a, d) for d in ts]
            x = [secs(d) / DAY for d in ts]
            lines!(ax, x, [t.st for t in T]; color = :black, linestyle = style, label = "storage")
            lines!(ax, x, [t.sf for t in T]; color = :darkorange, linestyle = style, label = "surface")
            lines!(ax, x, [sum(values(t.sides)) for t in T]; color = :royalblue, linestyle = style, label = "sides")
            lines!(ax, x, [t.sides[:north] for t in T]; color = :teal, linestyle = style, label = "north side")
            lines!(ax, x, [t.resid for t in T]; color = :gray50, linestyle = style, label = "residual")
        end
        tk = [DateTime(y, m, 1) for y in year(a):year(z) for m in (3, 9) if a <= DateTime(y, m, 1) <= z]
        ax.xticks = ([secs(d) / DAY for d in tk], Dates.format.(tk, "yyyy-mm"))
        hlines!(ax, 0; color = :gray70)
        (b, l) == (1, 1) && axislegend(ax; position = :lb, merge = true, framevisible = false)
    end
    Label(fig[0, :], "$(r.name): salt budget, run (solid) against B-SOSE (dashed); sides = transport of (S - $(S_REF)) into the box";
          fontsize = 18, font = :bold)
    out = joinpath(OUTDIR, "salt_budget_$(r.name).png")
    save(out, fig); @info "wrote $out"
end
