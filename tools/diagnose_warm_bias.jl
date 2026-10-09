# ==============================================================================
# diagnose_warm_bias.jl
#
# Why is the model 10-15 C too warm between 65S and 55S in summer, when the
# warmest water anywhere in that column in BSOSE is +3.9 C?
#
# The anomaly grows through each austral summer and collapses each winter, so it
# is not the initial condition and not a steady injection at a boundary. Two
# explanations survive that seasonality:
#
#   A. REDISTRIBUTION. The column takes up roughly the right amount of heat, but
#      deposits it in far too thin a layer -- no penetrating shortwave, or a
#      summer mixed layer that CATKE leaves too shallow. BSOSE's +130 W/m2 summer
#      flux spread over 50 m is +3.8 C; the same flux trapped in 10 m is +19 C.
#   B. IMPORT. The column actually gains heat it should not have, advected across
#      the Polar Front or injected through the western boundary.
#
# These two make opposite predictions about column heat content, which is what
# this script measures. Run it as
#
#     julia --project=. -e 'include("tools/diagnose_warm_bias.jl")'
#
# and read the three outputs in this order:
#
#   1. plots/RUN_heat_budget.png   0-HEAT_DEPTH m heat content, SST and mixed
#      layer depth in the LAT_BAND, model against BSOSE, over the whole run.
#      THIS IS THE DECISIVE ONE. If heat content tracks BSOSE while SST runs
#      away, it is (A) and the fix is in the mixing/radiation, not the forcing.
#      If heat content diverges too, it is (B) and the fix is at the boundary.
#   2. plots/RUN_section_DATE.png  theta and S at the mid-longitude transect,
#      model / BSOSE / difference, over the upper FOCUS_DEPTH metres, with each
#      solution's mixed layer depth drawn on it. Shows how deep the bias reaches.
#   3. plots/RUN_westface_DATE.png the western boundary, model against BSOSE at
#      the same longitude. Rules the boundary in or out directly.
#
# A summary table of the same numbers is printed to stdout.
#
# Configuration, all through the environment:
#
#     RUN_NAME         prefix of the model's output files ("2YS6_surface_flux")
#     RUN_DIR          directory holding them (".")
#     BSOSE_DIR        directory holding the BSOSE NetCDF files
#     BSOSE_ITERATION  156 (2013-2024, 1/6 deg) or 105 (2008-2012, 1/3 deg)
#     SECTION_DATES    comma-separated dates to draw sections for
#     LAT_BAND         "lo,hi" for the time series ("-65,-60")
#     FOCUS_DEPTH      depth of the section plots in metres (300)
#     HEAT_DEPTH       depth the heat content is integrated over (200)
#     MLD_THRESHOLD    density threshold for the mixed layer, kg/m3 (0.03)
#     MLD_REF_DEPTH    reference depth for it, metres below the surface (0 --
#                      the top wet cell; set 10 for the conventional number)
#     OUTPUT_DIR       where the PNGs go ("plots")
# ==============================================================================

using NCDatasets
using CairoMakie
using Printf
using Dates
using Statistics
using SeawaterPolynomials
using SeawaterPolynomials.TEOS10

const EOS = TEOS10EquationOfState()
const RHO0 = 1026.0
const CP = 3991.86795711963

# ── Configuration ─────────────────────────────────────────────────────────────

function env_pair(key, fallback)
    haskey(ENV, key) || return fallback
    raw = strip(ENV[key])
    isempty(raw) && return fallback
    parts = [parse(Float64, strip(p)) for p in split(raw, ',') if !isempty(strip(p))]
    length(parts) == 2 || error("$key must be a lo,hi pair; got \"$raw\"")
    parts[1] < parts[2] || error("$key must be increasing; got \"$raw\"")
    return (parts[1], parts[2])
end

const RUN_NAME = get(ENV, "RUN_NAME", "2YS6_surface_flux")
const RUN_DIR = get(ENV, "RUN_DIR", ".")
const BSOSE_ITERATION = parse(Int, get(ENV, "BSOSE_ITERATION", "156"))
const LAT_BAND = env_pair("LAT_BAND", (-65.0, -60.0))
const FOCUS_DEPTH = parse(Float64, get(ENV, "FOCUS_DEPTH", "300"))
const HEAT_DEPTH = parse(Float64, get(ENV, "HEAT_DEPTH", "200"))
const MLD_THRESHOLD = parse(Float64, get(ENV, "MLD_THRESHOLD", "0.03"))
const MLD_REF_DEPTH = parse(Float64, get(ENV, "MLD_REF_DEPTH", "0.0"))
const OUTPUT_DIR = get(ENV, "OUTPUT_DIR", "plots")

const SECTION_DATES = [DateTime(strip(s)) for s in
                       split(get(ENV, "SECTION_DATES",
                                 "2014-06-01,2015-01-12,2015-05-08,2016-01-04,2016-04-23"), ',')
                       if !isempty(strip(s))]

# ── Where the BSOSE data is (mirrors tools/animate_bsose_surface.jl) ──────────

function bsose_directory()
    haskey(ENV, "BSOSE_DIR") && isdir(ENV["BSOSE_DIR"]) && return abspath(ENV["BSOSE_DIR"])
    isdir("/storage/scratch1/1/vnguyen480/SOWinWaRP/data") && return "/storage/scratch1/1/vnguyen480/SOWinWaRP/data"
    repo_data = normpath(joinpath(@__DIR__, "..", "data"))
    isdir(repo_data) && return repo_data
    return abspath("data")
end

function bsose_file(dir, shortname, iteration)
    cands = iteration == 156 ?
            ["$(shortname)_bsoseI156_2013to2024_monthly.nc",
             "bsose_i156_2013to2024_monthly_$(shortname).nc"] :
            ["bsose_i105_2008to2012_monthly_$(shortname).nc",
             "$(shortname)_bsoseI105_2008to2012_monthly.nc"]

    for cand in cands
        isfile(joinpath(dir, cand)) && return joinpath(dir, cand)
    end

    if isdir(dir)
        matches = filter(f -> endswith(f, ".nc") && occursin(lowercase(shortname), lowercase(f)), readdir(dir))
        for token in ("i$(iteration)", "$(iteration)")
            for f in matches
                occursin(token, lowercase(f)) && return joinpath(dir, f)
            end
        end
    end

    error("""No BSOSE iteration-$(iteration) $(shortname) file found in $dir.
             Expected one of: $(join(cands, ", ")).
             Set BSOSE_DIR to the directory holding the monthly BSOSE NetCDF files.""")
end

# A BSOSE monthly record is stamped at the END of the month it averages, to within
# about a day, so rounding the stamp recovers the month. See src/setup_bsose.jl.
bsose_averaged_month(stamp) = round(DateTime(stamp), Month) - Month(1)
month_index_map(ds) = Dict(bsose_averaged_month(s) => i for (i, s) in enumerate(ds["time"][:]))

function record_index(ds, path, month)
    idx = get(month_index_map(ds), month, nothing)
    idx === nothing && error("$(basename(path)) holds no record for $(Dates.format(month, "yyyy-mm")).")
    return idx
end

# BSOSE marks land with a NaN _FillValue, which NCDatasets hands back as `missing`.
nanread(x) = Float64.(coalesce.(x, NaN))

"The first coordinate variable of `ds` named in `names`, as a Float64 vector."
function coordinate(ds, names...)
    for n in names
        haskey(ds, n) && return Float64.(ds[n][:])
    end
    error("None of $(names) is a variable of this file; it holds $(sort(collect(keys(ds)))).")
end

# ── Vertical grid, density and the two column integrals ───────────────────────

"""
    faces_from_centers(zc)

Cell faces (length `Nz+1`, ascending) reconstructed from ascending cell centres
with the top face at `z = 0`. Exact for any stretching, which matters here: the
model's spacing runs from 5 m at the surface to 200 m below.
"""
function faces_from_centers(zc)
    Nz = length(zc)
    f = zeros(Float64, Nz + 1)
    f[Nz+1] = 0.0
    for k in Nz:-1:1
        f[k] = 2zc[k] - f[k+1]
    end
    return f
end

"Potential density referenced to the surface, NaN-safe."
sigma0(t, s) = (isnan(t) || isnan(s)) ? NaN : SeawaterPolynomials.ρ(t, s, 0.0, EOS)

"""
    mixed_layer_depth(theta, salt, z)

Density-threshold mixed layer depth in positive metres, from a column ordered
surface-to-deep. A temperature threshold would be misleading south of the Polar
Front, where the stratification is held up by salinity rather than heat -- which
is exactly the structure the model may be losing.

The reference level is the top wet cell, not the conventional 10 m. That matters
here: the model's top cell is 5 m thick and the hypothesis under test is that
summer heat is being trapped in the first cell or two. A 10 m reference sits
BELOW such a layer, takes its density from the cold water underneath, and reports
a deep mixed layer for a column that is in fact capped by a 5 m warm skin --
hiding the one thing this diagnostic exists to find. Set `MLD_REF_DEPTH = 10` for
the conventional number once the question is settled.
"""
function mixed_layer_depth(theta, salt, z; dsigma=MLD_THRESHOLD, z_ref=-MLD_REF_DEPTH)
    n = length(z)
    kref = 0
    for k in 1:n
        (isnan(theta[k]) || isnan(salt[k])) && continue
        kref = k
        z[k] <= z_ref && break
    end
    kref == 0 && return NaN

    sref = sigma0(theta[kref], salt[kref])
    isnan(sref) && return NaN
    starget = sref + dsigma

    zprev, sprev = z[kref], sref
    for k in (kref+1):n
        (isnan(theta[k]) || isnan(salt[k])) && break
        sk = sigma0(theta[k], salt[k])
        isnan(sk) && break
        if sk >= starget
            w = sk > sprev ? (starget - sprev) / (sk - sprev) : 0.0
            return -(zprev + w * (z[k] - zprev))
        end
        zprev, sprev = z[k], sk
    end
    return -zprev   # no threshold crossing: mixed to the bottom of the valid column
end

"""
    column_heat_content(theta, z_top, z_bot, H)

`(rho0 cp integral(theta dz), depth covered)` over the upper `H` metres, from a
column ordered surface-to-deep. `z_top`/`z_bot` are each cell's faces, so a BSOSE
partial cell enters with `z_bot = z_top - drF*hFacC` and contributes only its wet
thickness. The covered depth comes back separately so a column that bottoms out
above `H` is not silently read as a cold one.
"""
function column_heat_content(theta, z_top, z_bot, H)
    acc = 0.0
    covered = 0.0
    for k in eachindex(theta)
        zt = min(z_top[k], 0.0)
        zb = max(z_bot[k], -H)
        thick = zt - zb
        thick <= 0 && continue
        isnan(theta[k]) && continue
        acc += theta[k] * thick
        covered += thick
    end
    return RHO0 * CP * acc, covered
end

"`v` sampled at `zs` linearly interpolated onto `zd`; both surface-to-deep."
function interp_z(zs, v, zd)
    out = fill(NaN, length(zd))
    n = length(zs)
    for (i, z) in enumerate(zd)
        if z >= zs[1]
            out[i] = v[1]
            continue
        end
        for k in 1:(n-1)
            if zs[k] >= z >= zs[k+1]
                v1, v2 = v[k], v[k+1]
                if !(isnan(v1) || isnan(v2))
                    w = (zs[k] - z) / (zs[k] - zs[k+1])
                    out[i] = v1 + w * (v2 - v1)
                end
                break
            end
        end
    end
    return out
end

"Index of the element of `xs` nearest `x`."
nearest_index(xs, x) = argmin(abs.(xs .- x))

# ── Reading the model ─────────────────────────────────────────────────────────

"The run's start date: the file's global attribute, else the one in src/model.jl."
function resolve_start_date(ds)
    if haskey(ds.attrib, "start_date")
        try
            return DateTime(string(ds.attrib["start_date"]))
        catch
            @warn "Ignoring unparseable start_date attribute."
        end
    end
    path = normpath(joinpath(@__DIR__, "..", "src", "model.jl"))
    if isfile(path)
        m = match(r"^[ \t]*start_date[ \t]*=[ \t]*DateTime\(\"([^\"]+)\"\)"m, read(path, String))
        m !== nothing && return DateTime(m.captures[1])
    end
    error("Cannot resolve the run's start date; the file has no start_date attribute.")
end

"""
    model_section(path)

A model transect file (`RUN_midlon.nc` or `RUN_face_west.nc`) as a NamedTuple.
Fields keep the file's own ascending-z order for plotting; `ztop`/`zbot`/`zs2d`
are the surface-to-deep companions the column integrals take.
"""
function model_section(path)
    ds = Dataset(path)
    lat = coordinate(ds, "φ_aca")
    z = coordinate(ds, "z_aac")
    times = Float64.(ds["time"][:])
    lon = haskey(ds, "λ_caa") ? Float64(ds["λ_caa"][1]) : NaN
    lon_face = haskey(ds, "λ_faa") ? Float64(ds["λ_faa"][1]) : NaN
    start_date = resolve_start_date(ds)
    Ny, Nz, Nt = length(lat), length(z), length(times)

    mask = haskey(ds, "inactive_nodes_ccc") ? (ds["inactive_nodes_ccc"][1, :, :] .!= 0) : falses(Ny, Nz)

    T = Array{Float32}(undef, Ny, Nz, Nt)
    S = Array{Float32}(undef, Ny, Nz, Nt)
    u = Array{Float32}(undef, Ny, Nz, Nt)
    for t in 1:Nt
        Tt = Float32.(ds["T"][1, :, :, t]); Tt[mask] .= NaN32; T[:, :, t] = Tt
        St = Float32.(ds["S"][1, :, :, t]); St[mask] .= NaN32; S[:, :, t] = St
        if haskey(ds, "u")
            ut = Float32.(ds["u"][1, :, :, t]); ut[mask] .= NaN32; u[:, :, t] = ut
        end
    end
    close(ds)

    f = faces_from_centers(z)
    return (; lat, z, zs2d=reverse(z),
        ztop=reverse(f[2:end]), zbot=reverse(f[1:end-1]),
        times, dates=[start_date + Second(round(Int, t)) for t in times],
        start_date, lon, lon_face, T, S, u, Ny, Nz, Nt)
end

# ── Reading BSOSE on the same transect ────────────────────────────────────────

"""
    bsose_section(lon, lat_bounds, months; want_u, face_longitude)

BSOSE theta, salt and (optionally) uvel on a single meridional transect. With
`face_longitude = true` the column picked is the one whose WESTERN face sits at
`lon`, which is where MITgcm stores `UVEL` and where the model's open boundary
is; otherwise the tracer cell nearest `lon` is used.
"""
function bsose_section(lon, lat_bounds, months; want_u=false, face_longitude=false)
    dir = bsose_directory()
    theta_path = bsose_file(dir, "Theta", BSOSE_ITERATION)
    salt_path = bsose_file(dir, "Salt", BSOSE_ITERATION)

    dsT = Dataset(theta_path)
    XC = Float64.(dsT["XC"][:]); YC = Float64.(dsT["YC"][:])
    Z = Float64.(dsT["Z"][:])             # descending: already surface-to-deep
    drF = Float64.(dsT["drF"][:])
    dx = length(XC) > 1 ? XC[2] - XC[1] : 1 / 6
    ix = face_longitude ? nearest_index(XC .- dx / 2, lon) : nearest_index(XC, lon)
    iy = findall(y -> lat_bounds[1] <= y <= lat_bounds[2], YC)
    isempty(iy) && error("BSOSE has no rows between $(lat_bounds[1]) and $(lat_bounds[2]).")

    lat = YC[iy]
    hFac = nanread(dsT["hFacC"][ix, iy, :])     # (Ny, Nz), 0 on land, <1 in a partial cell
    hFac[isnan.(hFac)] .= 0.0

    # Cell faces from the thicknesses; the bottom face of each column is pulled up
    # by its partial-cell fraction so the integrals see only wet water.
    ztop = vcat(0.0, -cumsum(drF)[1:end-1])
    zbot = [ztop[k] - drF[k] * hFac[j, k] for j in eachindex(iy), k in eachindex(drF)]

    Nm = length(months)
    T = Array{Float32}(undef, length(iy), length(Z), Nm)
    S = Array{Float32}(undef, length(iy), length(Z), Nm)
    U = Array{Float32}(undef, length(iy), length(Z), Nm)

    dsS = Dataset(salt_path)
    dsU = nothing
    if want_u
        try
            dsU = Dataset(bsose_file(dir, "Uvel", BSOSE_ITERATION))
        catch err
            @warn "No BSOSE Uvel file found; the velocity panels will be skipped. ($err)"
        end
    end

    for (m, month) in enumerate(months)
        T[:, :, m] = Float32.(nanread(dsT["THETA"][ix, iy, :, record_index(dsT, theta_path, month)]))
        S[:, :, m] = Float32.(nanread(dsS["SALT"][ix, iy, :, record_index(dsS, salt_path, month)]))
        dsU !== nothing &&
            (U[:, :, m] = Float32.(nanread(dsU["UVEL"][ix, iy, :, record_index(dsU, "Uvel", month)])))
    end

    # Land is stored as 0.0, not NaN, in some BSOSE products; hFacC is the truth.
    for j in axes(T, 1), k in axes(T, 2)
        if hFac[j, k] == 0
            T[j, k, :] .= NaN32
            S[j, k, :] .= NaN32
            U[j, k, :] .= NaN32
        end
    end

    close(dsT); close(dsS); dsU !== nothing && close(dsU)

    return (; lat, zs2d=Z, z=reverse(Z), ztop, zbot, hFac,
        lon=XC[ix], lon_face=XC[ix] - dx / 2, months, T, S,
        u=(dsU === nothing ? nothing : U))
end

# ── Column statistics in a latitude band ──────────────────────────────────────

"""
    band_statistics(lat, zs2d, ztop, zbot, theta, salt)

`(sst, mld, heat_content, column_mean_theta)` averaged over the columns of one
snapshot, where `ztop`/`zbot` are either vectors (the model's uniform column) or
`(Ny, Nz)` matrices (BSOSE's partial cells). Columns that do not reach
`HEAT_DEPTH` are dropped rather than averaged in, so the shelf does not drag the
band mean toward a shallow, warm value.
"""
function band_statistics(lat, zs2d, ztop, zbot, theta, salt)
    ssts, mlds, hcs, means = Float64[], Float64[], Float64[], Float64[]
    for j in eachindex(lat)
        th = Float64.(theta[j, :]); sa = Float64.(salt[j, :])
        all(isnan, th) && continue
        zt = ztop isa AbstractMatrix ? ztop[j, :] : ztop
        zb = zbot isa AbstractMatrix ? zbot[j, :] : zbot
        hc, covered = column_heat_content(th, zt, zb, HEAT_DEPTH)
        covered < 0.9 * HEAT_DEPTH && continue
        push!(ssts, th[findfirst(!isnan, th)])
        push!(mlds, mixed_layer_depth(th, sa, zs2d))
        push!(hcs, hc)
        push!(means, hc / (RHO0 * CP * covered))
    end
    nanmean(v) = (w = filter(!isnan, v); isempty(w) ? NaN : mean(w))
    return (sst=nanmean(ssts), mld=nanmean(mlds), hc=nanmean(hcs), tbar=nanmean(means))
end

# ── Diagnostic 1: the heat budget (the decisive one) ──────────────────────────

"Every month from the month containing `d1` to the month containing `d2`."
function month_range(d1, d2)
    m = DateTime(year(d1), month(d1), 1)
    out = DateTime[]
    while m <= d2
        push!(out, m)
        m += Month(1)
    end
    return out
end

"""
    date_ticks(start_date, dates)

Quarter-start tick positions and labels, in days since `start_date`. The marks
are generated from the span of `dates`, not selected from them: the face files
are written every 5 days, so a filter over the sampled dates would land on a
first-of-the-quarter only by luck and leave the axis with one tick on it.
"""
function date_ticks(start_date, dates)
    d1, d2 = first(dates), last(dates)
    marks = DateTime[]
    m = DateTime(year(d1), 3 * ((month(d1) - 1) ÷ 3) + 1, 1)
    while m <= d2
        m >= d1 && push!(marks, m)
        m += Month(3)
    end
    isempty(marks) && (marks = [d1, d2])
    pos = [Dates.value(Second(mk - start_date)) / 86400 for mk in marks]
    return (pos, [Dates.format(mk, "u yyyy") for mk in marks])
end

function heat_budget(model, bs, outpath)
    jm = findall(y -> LAT_BAND[1] <= y <= LAT_BAND[2], model.lat)
    jb = findall(y -> LAT_BAND[1] <= y <= LAT_BAND[2], bs.lat)
    isempty(jm) && error("The model transect has no rows in $(LAT_BAND).")

    println("\nDiagnostic 1: heat budget over $(LAT_BAND[1])..$(LAT_BAND[2])S at ",
        @sprintf("%.2f", model.lon), "E")
    println("  model columns: $(length(jm)), BSOSE columns: $(length(jb))")

    m_days = model.times ./ 86400
    m_sst = fill(NaN, model.Nt); m_mld = similar(m_sst); m_tbar = similar(m_sst)
    for t in 1:model.Nt
        st = band_statistics(model.lat[jm], model.zs2d, model.ztop, model.zbot,
            reverse(model.T[jm, :, t], dims=2), reverse(model.S[jm, :, t], dims=2))
        m_sst[t], m_mld[t], m_tbar[t] = st.sst, st.mld, st.tbar
    end

    Nm = length(bs.months)
    b_days = [Dates.value(Second(bs.months[m] + Day(14) - model.start_date)) / 86400 for m in 1:Nm]
    b_sst = fill(NaN, Nm); b_mld = similar(b_sst); b_tbar = similar(b_sst)
    for m in 1:Nm
        st = band_statistics(bs.lat[jb], bs.zs2d, bs.ztop, bs.zbot[jb, :],
            bs.T[jb, :, m], bs.S[jb, :, m])
        b_sst[m], b_mld[m], b_tbar[m] = st.sst, st.mld, st.tbar
    end

    # The table. This is the output to read first.
    println()
    println("  month      | SST model / BSOSE / diff | mean T 0-$(Int(HEAT_DEPTH))m model / BSOSE / diff | MLD model / BSOSE")
    println("  " * "-"^106)
    for m in 1:Nm
        it = argmin(abs.(m_days .- b_days[m]))
        abs(m_days[it] - b_days[m]) > 20 && continue
        @printf("  %-10s | %+6.2f %+6.2f %+6.2f C      | %+6.2f %+6.2f %+6.2f C                    | %6.1f %6.1f m\n",
            Dates.format(bs.months[m], "yyyy-mm"),
            m_sst[it], b_sst[m], m_sst[it] - b_sst[m],
            m_tbar[it], b_tbar[m], m_tbar[it] - b_tbar[m],
            m_mld[it], b_mld[m])
    end
    println("  " * "-"^106)
    println("""
      READ IT LIKE THIS. The right-hand pair is the whole story:

        mean T over 0-$(Int(HEAT_DEPTH)) m agrees but SST does not
            -> the column holds the right amount of heat and is burying it in too
               thin a layer. Suspect missing penetrating shortwave (the whole net
               flux is landing in the top 5 m cell) and a summer mixed layer CATKE
               is leaving too shallow. The MLD column should confirm it.
        mean T over 0-$(Int(HEAT_DEPTH)) m also runs warm
            -> the column is genuinely gaining heat it should not have. Go to
               diagnostic 3 and the sponge/inflow settings.""")

    tickpos, ticklab = date_ticks(model.start_date, model.dates)
    fig = Figure(size=(1250, 1000), fontsize=14)
    Label(fig[0, 1], @sprintf("%s vs BSOSE i%d  --  %.0f to %.0fS at %.1f E",
            RUN_NAME, BSOSE_ITERATION, LAT_BAND[1], LAT_BAND[2], model.lon),
        fontsize=20, font=:bold)

    panels = ((1, "Mean θ over upper $(Int(HEAT_DEPTH)) m  (∝ column heat content)", "°C", m_tbar, b_tbar, false),
        (2, "Sea surface temperature", "°C", m_sst, b_sst, false),
        (3, "Mixed layer depth  (Δσ = $(MLD_THRESHOLD) kg m⁻³ from $(MLD_REF_DEPTH) m)", "m", m_mld, b_mld, true))

    for (row, title, unit, mv, bv, flip) in panels
        ax = Axis(fig[row, 1]; title, ylabel=unit,
            xlabel=(row == 3 ? "" : ""), xticks=(tickpos, ticklab),
            yreversed=flip)
        lines!(ax, m_days, mv, color=:firebrick, linewidth=1.8, label="model")
        scatterlines!(ax, b_days, bv, color=:navy, linewidth=2.0, markersize=7, label="BSOSE")
        row == 1 && axislegend(ax, position=:lt, framevisible=false)
    end

    save(outpath, fig)
    println("\n  wrote ", outpath)
    return nothing
end

# ── Diagnostics 2 and 3: sections, model against BSOSE ────────────────────────

"BSOSE field `A[:, :, m]` put on the model's (lat, z) grid, ascending z."
function regrid(bs, A, m, model)
    out = fill(NaN, length(model.lat), length(model.z))
    for (j, y) in enumerate(model.lat)
        jb = nearest_index(bs.lat, y)
        col = interp_z(bs.zs2d, Float64.(A[jb, :, m]), model.zs2d)
        out[j, :] = reverse(col)
    end
    return out
end

"MLD along a transect, one value per latitude."
mld_profile(lat, zs2d, T, S) =
    [mixed_layer_depth(Float64.(T[j, :]), Float64.(S[j, :]), zs2d) for j in eachindex(lat)]

function limits_of(arrays...; pad=0.0)
    v = Float64[]
    for a in arrays
        append!(v, filter(!isnan, vec(a)))
    end
    isempty(v) && return (-1.0, 1.0)
    lo, hi = minimum(v), maximum(v)
    lo == hi && return (lo - 1, hi + 1)
    return (lo - pad * (hi - lo), hi + pad * (hi - lo))
end

"""
    symmetric_limits(a, span)

Limits centred on zero for a difference panel, never narrower than 1% of `span`,
the range of the fields differenced. Without that floor a panel where the model
matches BSOSE rescales onto its own float32 round-off and draws a few parts in
10^6 as vivid noise, which reads as a difference when it is the absence of one.
"""
function symmetric_limits(a, span)
    v = filter(!isnan, vec(a))
    m = isempty(v) ? 0.0 : maximum(abs, v)
    m = max(m, 0.01 * abs(span))
    return m == 0 ? (-1.0, 1.0) : (-m, m)
end

"""
    comparison_figure(model, bs, it, im, rows, depth, title, outpath)

Three columns -- model, BSOSE, model minus BSOSE -- for each field in `rows`,
over the upper `depth` metres. `rows` is a tuple of
`(label, unit, model_field, bsose_field, colormap)`.
"""
function comparison_figure(model, bs, it, im, rows, depth, title, outpath; mld=nothing)
    fig = Figure(size=(1750, 420 * length(rows) + 90), fontsize=14)
    Label(fig[0, 1:5], title, fontsize=20, font=:bold)
    bounds = (minimum(model.lat), maximum(model.lat), -depth, 0.0)

    for (row, (label, unit, mfield, bfield, cmap)) in enumerate(rows)
        msl = Float64.(mfield[:, :, it])
        bsl = regrid(bs, bfield, im, model)
        dsl = msl .- bsl

        kept = model.z .>= -depth
        lims = limits_of(msl[:, kept], bsl[:, kept])
        dlims = symmetric_limits(dsl[:, kept], lims[2] - lims[1])

        for (col, (sub, name, cm, cl)) in enumerate(((msl, "model", cmap, lims),
            (bsl, "BSOSE i$(BSOSE_ITERATION)", cmap, lims),
            (dsl, "model − BSOSE", :balance, dlims)))
            axcol = col == 3 ? 4 : col
            ax = Axis(fig[row, axcol]; title="$label — $name",
                xlabel="Latitude (°N)", ylabel=(axcol == 1 ? "Depth (m)" : ""),
                limits=bounds)
            hm = heatmap!(ax, model.lat, model.z, sub, colormap=cm, colorrange=cl,
                nan_color=:gray30)
            if mld !== nothing && col <= 2
                lines!(ax, model.lat, -mld[col], color=:white, linewidth=2.5)
                lines!(ax, model.lat, -mld[col], color=:black, linewidth=1.2, linestyle=:dash)
            end
            col == 2 && Colorbar(fig[row, 3], hm, label=unit)
            col == 3 && Colorbar(fig[row, 5], hm, label=unit)
        end
    end

    save(outpath, fig)
    println("  wrote ", outpath)
    return nothing
end

# ── Driver ────────────────────────────────────────────────────────────────────

"Index of the snapshot of `dates` nearest `d`."
nearest_date(dates, d) = argmin([abs(Dates.value(Second(dd - d))) for dd in dates])

"BSOSE's per-latitude MLD put on the model's latitudes."
function bsose_mld_on(model, bs, im)
    prof = mld_profile(bs.lat, bs.zs2d, bs.T[:, :, im], bs.S[:, :, im])
    return [prof[nearest_index(bs.lat, y)] for y in model.lat]
end

function main()
    mkpath(OUTPUT_DIR)

    midlon_path = joinpath(RUN_DIR, "$(RUN_NAME)_midlon.nc")
    face_path = joinpath(RUN_DIR, "$(RUN_NAME)_face_west.nc")
    isfile(midlon_path) || error("""No $(midlon_path).
                                    Set RUN_NAME and RUN_DIR to point at the run's output.""")

    println("Reading ", midlon_path)
    model = model_section(midlon_path)
    @printf("  transect at %.2f E, Ny = %d, Nz = %d, %d snapshots\n",
        model.lon, model.Ny, model.Nz, model.Nt)
    @printf("  %s to %s, top cell %.1f m thick\n",
        Dates.format(first(model.dates), "yyyy-mm-dd"),
        Dates.format(last(model.dates), "yyyy-mm-dd"),
        model.ztop[1] - model.zbot[1])

    months = month_range(first(model.dates), last(model.dates))
    lat_pad = (minimum(model.lat) - 0.5, maximum(model.lat) + 0.5)

    println("Reading BSOSE i$(BSOSE_ITERATION) from ", bsose_directory())
    bs = bsose_section(model.lon, lat_pad, months)
    @printf("  transect at %.2f E, Ny = %d, Nz = %d, %d months\n",
        bs.lon, length(bs.lat), length(bs.zs2d), length(months))

    # 1. The heat budget.
    heat_budget(model, bs, joinpath(OUTPUT_DIR, "$(RUN_NAME)_heat_budget.png"))

    # 2. Upper-ocean sections on the requested dates.
    println("\nDiagnostic 2: upper-$(Int(FOCUS_DEPTH)) m sections at $(round(model.lon, digits=1))E")
    for d in SECTION_DATES
        (d < first(model.dates) || d > last(model.dates)) &&
            (println("  skipping $(Dates.format(d, "yyyy-mm-dd")): outside the run"); continue)
        it = nearest_date(model.dates, d)
        im = findfirst(==(DateTime(year(d), month(d), 1)), bs.months)
        im === nothing && (println("  skipping $(Dates.format(d, "yyyy-mm-dd")): no BSOSE month"); continue)

        mlds = (mld_profile(model.lat, model.zs2d,
                reverse(model.T[:, :, it], dims=2), reverse(model.S[:, :, it], dims=2)),
            bsose_mld_on(model, bs, im))

        comparison_figure(model, bs, it, im,
            (("θ", "°C", model.T, bs.T, :thermal),
                ("S", "PSU", model.S, bs.S, :haline)),
            FOCUS_DEPTH,
            @sprintf("%s  —  %s   (BSOSE %s)   λ = %.1f°E",
                RUN_NAME, Dates.format(model.dates[it], "d u yyyy"),
                Dates.format(bs.months[im], "u yyyy"), model.lon),
            joinpath(OUTPUT_DIR, "$(RUN_NAME)_section_$(Dates.format(model.dates[it], "yyyy-mm-dd")).png");
            mld=mlds)
    end

    # 3. The western boundary.
    println("\nDiagnostic 3: western open boundary")
    if !isfile(face_path)
        println("  no $(face_path) -- set BOUNDARY_DIAGNOSTICS = true in src/model.jl to write it.")
        return nothing
    end

    face = model_section(face_path)
    bf_lon = isnan(face.lon_face) ? face.lon : face.lon_face
    @printf("  face at %.2f E, %d snapshots\n", bf_lon, face.Nt)
    bsf = bsose_section(bf_lon, (minimum(face.lat) - 0.5, maximum(face.lat) + 0.5), months;
        want_u=true, face_longitude=true)

    rows_for(bsu) = bsu === nothing ?
                    (("θ", "°C", face.T, bsf.T, :thermal), ("S", "PSU", face.S, bsf.S, :haline)) :
                    (("θ", "°C", face.T, bsf.T, :thermal), ("u", "m/s", face.u, bsu, :balance))

    for d in SECTION_DATES
        (d < first(face.dates) || d > last(face.dates)) && continue
        it = nearest_date(face.dates, d)
        im = findfirst(==(DateTime(year(d), month(d), 1)), bsf.months)
        im === nothing && continue

        comparison_figure(face, bsf, it, im, rows_for(bsf.u),
            abs(minimum(face.z)),
            @sprintf("%s  —  WEST FACE  —  %s   (BSOSE %s)   λ = %.1f°E",
                RUN_NAME, Dates.format(face.dates[it], "d u yyyy"),
                Dates.format(bsf.months[im], "u yyyy"), bf_lon),
            joinpath(OUTPUT_DIR, "$(RUN_NAME)_westface_$(Dates.format(face.dates[it], "yyyy-mm-dd")).png"))
    end

    println("\nDone. PNGs are in $(OUTPUT_DIR)/.")
    return nothing
end

main()
