# ==============================================================================
# bsose_extremes.jl -- B-SOSE's extremes over our box (90-150E, 70-45S) through time, and where they are: the largest
# h·ℵ in any cell, the lowest and highest surface temperature and salinity, and the ice extent, area and volume.
# From the MONTHLY and the 5-DAY distributions, so a run's progress line has a yardstick at either cadence.
#
#     julia --project=. tools/bsose_extremes.jl                              # 2014-06-01 to 2016-06-01, both cadences
#     CADENCE=fiveday START_DATE=2014-06-01 END_DATE=2015-06-01 julia --project=. tools/bsose_extremes.jl
#
# WHAT, per record (a monthly mean or a 5-day mean):
#   max h·ℵ   the largest SIheff in one cell (m): SIheff is ice volume per unit cell area, i.e. h·ℵ, what the runs'
#             progress line calls "max h·ℵ" (and their coastal piles reach 8-33 m)
#   T, S      the lowest and highest top-level (0-4.2 m, the same cell as ours) temperature (°C) and salinity (g/kg):
#             the runs' "surface S (min, max), T (min, max)"
#   extent    area of cells with ℵ > 0.15 (10⁶ km²); area Σ ℵ·A (10⁶ km², the runs' startup "area"); volume Σ SIheff·A (km³)
# each extreme with its place. B-SOSE's land is zero and is left out. Means, not snapshots: the runs' daily
# snapshots reach further, more so against the monthly means than the 5-day ones.
#
# WRITES, per cadence:
#   stdout                         each year's extremes with where and when, and every month's
#   OUTDIR/bsose_extremes_<cadence>.csv   every record: date, each quantity and its lon/lat
#   OUTDIR/bsose_extremes_<cadence>.png   the series through time (top) and maps of where each extreme sat,
#                                  coloured by date (bottom): max h·ℵ; lowest and highest S; lowest and highest T
#
# READS (BSOSE_DIR, default the PACE data directory). Ice and surface tokens are matched whatever their case
# (B-SOSE writes both SIArea and SIarea):
#   monthly  SIarea, SIheff, Theta, Salt  as <token>_bsoseI156_2013to2024_monthly.nc
#   5-day    SIArea, SIHeff, Theta, Salt  as <token>_bsoseI156_<year>_5day.nc
# Only the top level of Theta and Salt is read, one record at a time. LOGIN-NODE SAFE.
#
# DATES: a monthly record stamped at the end of its month is the month before; it is placed mid-month. A 5-day
# record stamped at the end of its window is placed at the window centre (stamp - 2.5 days).
# ==============================================================================

using NCDatasets, CairoMakie, Dates, Printf

const BDIR       = get(ENV, "BSOSE_DIR", "/storage/scratch1/1/vnguyen480/SOWinWaRP/data")
const OUTDIR     = get(ENV, "OUTDIR", "plots")
const START_DATE = DateTime(get(ENV, "START_DATE", "2014-06-01"))
const END_DATE   = DateTime(get(ENV, "END_DATE", "2016-06-01"))
const CADENCES   = Symbol.(split(get(ENV, "CADENCE", "monthly,fiveday"), ","))
const LON, LAT   = (90.0, 150.0), (-70.0, -45.0)
const R          = 6.371e6
mkpath(OUTDIR)

f64(A) = Float64.(coalesce.(A, NaN))
coord(ds, name, dim) = (v = f64(ds[name][:]); ndims(v) == 1 ? v : dim == 1 ? v[:, 1] : v[1, :])
first_var(ds, names, path) = (found = filter(n -> haskey(ds, n), names);
                              isempty(found) ? error("none of $(names) in $path; it holds $(collect(keys(ds)))") : first(found))

"The file `<token>_bsoseI156_<span>.nc` in BDIR, matching the token whatever its case."
function bsose_file(token, span)
    want = lowercase("$(token)_bsoseI156_$(span).nc")
    hits = filter(f -> lowercase(f) == want, readdir(BDIR))
    isempty(hits) && error("no $(token)_bsoseI156_$(span).nc (any case) in $BDIR")
    return joinpath(BDIR, first(hits))
end

# One source per quantity and cadence: the files, and each record's date and index in its file
struct Source
    paths :: Vector{String}
    names :: Vector{String}
    dates :: Vector{DateTime}
    where :: Vector{Tuple{Int, Int}}     # (file, record)
end

const NAMES = Dict(:ℵ => ("SIarea", "SIArea", "SIAREA"), :V => ("SIheff", "SIHeff", "SIHEFF"),
                   :T => ("THETA", "Theta"), :S => ("SALT", "Salt"))
const TOKENS = Dict(:ℵ => "SIarea", :V => "SIheff", :T => "Theta", :S => "Salt")

record_date(::Val{:monthly}, stamp) = round(DateTime(stamp), Month) - Month(1) + Day(14)   # mid-month of the month averaged
record_date(::Val{:fiveday}, stamp) = DateTime(stamp) - Hour(60)                             # centre of the 5-day window

function source(q, cadence)
    spans = cadence === :monthly ? ["2013to2024_monthly"] : ["$(y)_5day" for y in year(START_DATE):year(END_DATE)]
    paths, names, dates, where = String[], String[], DateTime[], Tuple{Int, Int}[]
    for span in spans
        path = try bsose_file(TOKENS[q], span) catch err
            cadence === :fiveday && !occursin(string(year(END_DATE)), span) && rethrow()
            @warn sprint(showerror, err); continue        # the end year's 5-day file may not be needed
        end
        NCDataset(path) do ds
            push!(paths, path); push!(names, first_var(ds, NAMES[q], path))
            for (n, stamp) in enumerate(ds["time"][:])
                d = record_date(Val(cadence), stamp)
                START_DATE <= d < END_DATE || continue
                push!(dates, d); push!(where, (length(paths), n))
            end
        end
    end
    isempty(dates) && error("no $(TOKENS[q]) $(cadence) record between $START_DATE and $END_DATE in $BDIR")
    return Source(paths, names, dates, where)
end

"One 2-D record of a source over the box (the top level if the variable is 3-D), NaN where B-SOSE has land (zero)."
function read_record(src, n, i, j)
    f, k = src.where[n]
    NCDataset(src.paths[f]) do ds
        v = ds[src.names[f]]
        A = f64(ndims(v) == 4 ? v[i, j, 1, k] : v[i, j, k])
        A[A .== 0] .= NaN
        A
    end
end

nanargmax(A) = argmax(map(a -> isnan(a) ? -Inf : a, A))
nanargmin(A) = argmin(map(a -> isnan(a) ? Inf : a, A))

function extremes(cadence)
    srcs = Dict(q => source(q, cadence) for q in (:ℵ, :V, :T, :S))
    dates = srcs[:V].dates
    for q in (:ℵ, :T, :S)
        srcs[q].dates == dates || error("$(TOKENS[q]) and SIheff hold different $(cadence) records")
    end
    # the box, and cell areas from the cell-centre coordinates (latitude spacing is Mercator-like)
    x, yc = NCDataset(srcs[:V].paths[1]) do ds; coord(ds, "XC", 1), coord(ds, "YC", 2) end
    i = findall(λ -> LON[1] < λ < LON[2], x); j = findall(φ -> LAT[1] < φ < LAT[2], yc)
    i, j = first(i):last(i), first(j):last(j)
    λ, φ = x[i], yc[j]
    edges = vcat(1.5yc[1] - 0.5yc[2], (yc[1:end-1] .+ yc[2:end]) ./ 2, 1.5yc[end] - 0.5yc[end-1])
    cell = [R^2 * deg2rad(x[2] - x[1]) * abs(sind(edges[jj+1]) - sind(edges[jj])) for _ in i, jj in j]

    rows = NamedTuple[]
    land = nothing
    @info "B-SOSE $(cadence): $(length(dates)) records from $(first(dates)) to $(last(dates))"
    for n in eachindex(dates)
        ℵ, V, T, S = (read_record(srcs[q], n, i, j) for q in (:ℵ, :V, :T, :S))
        land === nothing && (land = isnan.(S))
        ℵ0, V0 = replace(ℵ, NaN => 0.0), replace(V, NaN => 0.0)
        kv, ks, kS, kt, kT = nanargmax(V), nanargmin(S), nanargmax(S), nanargmin(T), nanargmax(T)
        at(k) = (λ[k[1]], φ[k[2]])
        push!(rows, (; date = dates[n],
                       hv = V[kv], hv_at = at(kv),
                       Smin = S[ks], Smin_at = at(ks), Smax = S[kS], Smax_at = at(kS),
                       Tmin = T[kt], Tmin_at = at(kt), Tmax = T[kT], Tmax_at = at(kT),
                       extent = sum(cell[ℵ0 .> 0.15]) / 1e12, area = sum(ℵ0 .* cell) / 1e12, volume = sum(V0 .* cell) / 1e9))
    end
    return rows, λ, φ, land
end

place((x, y)) = @sprintf("%6.2fE %5.2fS", x, -y)

function report(cadence, rows)
    label = cadence === :monthly ? "MONTHLY MEANS" : "5-DAY MEANS"
    println("\n", "="^118)
    println("B-SOSE $label over 90-150E, 70-45S, $(Date(START_DATE)) to $(Date(END_DATE)) ($(length(rows)) records)")
    println("="^118)
    for y in unique(year.(getfield.(rows, :date)))
        r = filter(t -> year(t.date) == y, rows)
        best(f, get) = r[f(get.(r))]
        a = best(argmax, t -> t.hv); b = best(argmin, t -> t.Smin); c = best(argmax, t -> t.Smax)
        d = best(argmin, t -> t.Tmin); e = best(argmax, t -> t.Tmax); g = best(argmax, t -> t.extent)
        println("\n  $y")
        @printf("    max h·ℵ        %7.2f m      at %s on %s\n", a.hv, place(a.hv_at), Date(a.date))
        @printf("    lowest S       %7.2f g/kg   at %s on %s\n", b.Smin, place(b.Smin_at), Date(b.date))
        @printf("    highest S      %7.2f g/kg   at %s on %s\n", c.Smax, place(c.Smax_at), Date(c.date))
        @printf("    lowest T       %7.2f °C     at %s on %s\n", d.Tmin, place(d.Tmin_at), Date(d.date))
        @printf("    highest T      %7.2f °C     at %s on %s\n", e.Tmax, place(e.Tmax_at), Date(e.date))
        @printf("    max extent     %7.2f 10⁶ km² on %s (area %.2f, volume %.0f km³)\n", g.extent, Date(g.date), g.area, g.volume)
    end
    println("\n  BY MONTH (the extreme over the month's records, and where the largest h·ℵ sat)")
    @printf("  %-8s | %7s %-16s | %7s %7s | %7s %7s | %7s %6s %7s\n",
            "month", "max h·ℵ", "  where", "S low", "S high", "T low", "T high", "extent", "area", "volume")
    for m in unique(Date.(year.(getfield.(rows, :date)), month.(getfield.(rows, :date))))
        r = filter(t -> Date(year(t.date), month(t.date)) == m, rows)
        k = argmax(getfield.(r, :hv))
        mean(f) = sum(getfield.(r, f)) / length(r)
        @printf("  %-8s | %7.2f %-16s | %7.2f %7.2f | %7.2f %7.2f | %7.2f %6.2f %7.0f\n", Dates.format(m, "yyyy-mm"),
                r[k].hv, place(r[k].hv_at), minimum(getfield.(r, :Smin)), maximum(getfield.(r, :Smax)),
                minimum(getfield.(r, :Tmin)), maximum(getfield.(r, :Tmax)), mean(:extent), mean(:area), mean(:volume))
    end
end

function write_csv(cadence, rows)
    path = joinpath(OUTDIR, "bsose_extremes_$(cadence).csv")
    open(path, "w") do io
        println(io, "date,max_hv_m,hv_lon,hv_lat,S_min,Smin_lon,Smin_lat,S_max,Smax_lon,Smax_lat,T_min,Tmin_lon,Tmin_lat,T_max,Tmax_lon,Tmax_lat,extent_1e6km2,area_1e6km2,volume_km3")
        for t in rows
            println(io, join((Date(t.date), t.hv, t.hv_at..., t.Smin, t.Smin_at..., t.Smax, t.Smax_at...,
                              t.Tmin, t.Tmin_at..., t.Tmax, t.Tmax_at..., t.extent, t.area, t.volume), ","))
        end
    end
    @info "wrote $path"
end

function plot_extremes(cadence, rows, λ, φ, land)
    dates = getfield.(rows, :date)
    days = [Dates.value(d - START_DATE) / 86_400_000 for d in dates]
    ticks = [DateTime(y, m, 1) for y in year(START_DATE):year(END_DATE) for m in (1, 4, 7, 10)
             if START_DATE <= DateTime(y, m, 1) <= END_DATE]
    tickpos = [Dates.value(d - START_DATE) / 86_400_000 for d in ticks]
    ticklab = Dates.format.(ticks, "yyyy-mm")
    label = cadence === :monthly ? "monthly means" : "5-day means"

    fig = Figure(size = (1700, 1150))
    panels = ((1, "largest h·ℵ in a cell (m)", [(:hv, :black, "max h·ℵ")]),
              (2, "surface salinity (g/kg)", [(:Smin, :steelblue, "lowest"), (:Smax, :firebrick, "highest")]),
              (3, "surface temperature (°C)", [(:Tmin, :steelblue, "lowest"), (:Tmax, :firebrick, "highest")]))
    for (col, title, lines) in panels
        ax = Axis(fig[1, col]; title = "B-SOSE $title, $label", xticks = (tickpos, ticklab), xticklabelrotation = π / 4)
        for (f, c, l) in lines
            scatterlines!(ax, days, getfield.(rows, f); color = c, label = l, markersize = cadence === :monthly ? 8 : 4)
        end
        length(lines) > 1 && axislegend(ax; position = :lt)
    end

    # where: the land as a grey background, each record's extreme coloured by date (▲ highest, ▼ lowest)
    maps = ((1, "where the largest h·ℵ sat", [(:hv_at, :utriangle)]),
            (2, "where the lowest (▼) and highest (▲) S sat", [(:Smin_at, :dtriangle), (:Smax_at, :utriangle)]),
            (3, "where the lowest (▼) and highest (▲) T sat", [(:Tmin_at, :dtriangle), (:Tmax_at, :utriangle)]))
    for (col, title, marks) in maps
        ax = Axis(fig[2, col]; title, aspect = DataAspect(), xlabel = "longitude", ylabel = "latitude",
                  limits = (LON..., LAT...))
        heatmap!(ax, λ, φ, Float64.(land); colormap = [:white, :gray70], colorrange = (0, 1))
        for (f, marker) in marks
            pts = getfield.(rows, f)
            scatter!(ax, first.(pts), last.(pts); color = days, colormap = :viridis, colorrange = extrema(days),
                     marker, markersize = 12, strokewidth = 0.5, strokecolor = :black)
        end
    end
    Colorbar(fig[2, 4]; colormap = :viridis, colorrange = extrema(days), ticks = (tickpos, ticklab), label = "date")
    path = joinpath(OUTDIR, "bsose_extremes_$(cadence).png")
    save(path, fig)
    @info "wrote $path"
end

for cadence in CADENCES
    cadence in (:monthly, :fiveday) || error("CADENCE must be monthly and/or fiveday, got $cadence")
    rows, λ, φ, land = extremes(cadence)
    report(cadence, rows)
    write_csv(cadence, rows)
    plot_extremes(cadence, rows, λ, φ, land)
end
