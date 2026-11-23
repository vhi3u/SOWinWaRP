# ==============================================================================================
# era5_stress.jl -- Large & Yeager (2004) open-water wind stress from the ERA5 files the runs read
# (tools/download_era5_ncar.jl: <ERA5_DIR>/<CDS variable>_<YYYY>_<W>_<E>_<S>_<N>.nc, (longitude, latitude,
# valid_time), hourly). Used by compare_wind_stress.jl and single_column_bsose.jl (STRESS=era5_*).
#
# τ = ρa Cd(U10) U10 (u10, v10): NumericalEarth's PolynomialNeutralDragCoefficient (768ed1d defaults), NEUTRAL (no
# stability correction), absolute wind (no ocean current). ρa from ERA5's msl, t2m and d2m (ideal gas, virtual
# temperature) as NumericalEarth's coupler computes it, or a constant (EXF's atmrho, 1.2).
# ==============================================================================================

using NCDatasets, Dates, Statistics

neutral_cd(U) = (U = max(U, 0.5); U < 33 ? (2.7 / U + 0.142 + U / 13.09 - 3.14807e-10 * U^6) / 1000 : 2.34e-3)

"Moist air density (kg/m³) from pressure (Pa), temperature and dewpoint (K)."
function air_density(p, T, Td)
    Tdc = Td - 273.15
    e = 611.2 * exp(17.67 * Tdc / (Tdc + 243.5))      # vapour pressure, Pa (Bolton 1980)
    q = 0.622 * e / (p - 0.378 * e)
    return p / (287.04 * T * (1 + 0.608 * q))
end

"The ERA5 file holding `short` (u10, v10, t2m, d2m, msl) for year y in dir, found by opening the candidates."
function era5_file(dir, short, y)
    for f in sort(readdir(dir; join = true))
        (endswith(f, ".nc") && occursin("_$(y)_", basename(f))) || continue
        NCDataset(f) do ds; haskey(ds, short) end && return f
    end
    error("no ERA5 file with $short for $y in $dir (tools/download_era5_ncar.jl writes them)")
end

era5_timevar(ds) = haskey(ds, "valid_time") ? "valid_time" : "time"   # download_era5_ncar.jl writes CDS's valid_time

"""
Hourly ERA5 stress over a lon-lat box, the mean of each ERA5 cell's stress (not the stress of the mean wind) over
the cells whose centres fall in the box (the nearest cell if none does), for the hours in [t0, t1].
Returns (times, τx, τy, ρa, u10, v10), all box means; rho = nothing uses ERA5's air density, a number that constant.
every = 6 keeps only the hours 00, 06, 12, 18 of the wind, temperature, dewpoint and pressure and interpolates them
linearly to every hour BEFORE the stress is computed, as MITgcm's EXF does with 6-hourly fields.
area = d (degrees) instead takes the cells in a d x d square on the box's centre and averages the wind, temperature,
dewpoint and pressure over it BEFORE the stress: one smoothed wind, a stand-in for a coarser product (ERA-Interim).
"""
function era5_box_stress(dir, lon, lat, t0, t1; rho = nothing, every = 1, area = nothing)
    times, τx, τy, ρs, us, vs = DateTime[], Float64[], Float64[], Float64[], Float64[], Float64[]
    for y in year(t0):year(t1)
        files = Dict(s => era5_file(dir, s, y) for s in ("u10", "v10", "t2m", "d2m", "msl"))
        NCDataset(files["u10"]) do ds
            lo = mod.(Float64.(ds[haskey(ds, "longitude") ? "longitude" : "lon"][:]), 360)
            la = Float64.(ds[haskey(ds, "latitude") ? "latitude" : "lat"][:])
            lonr, latr = area === nothing ? (lon, lat) :
                         (((lon[1] + lon[2]) / 2 - area / 2, (lon[1] + lon[2]) / 2 + area / 2),
                          ((lat[1] + lat[2]) / 2 - area / 2, (lat[1] + lat[2]) / 2 + area / 2))
            ib = findall(x -> mod(lonr[1], 360) - 1e-6 <= x <= mod(lonr[2], 360) + 1e-6, lo)
            jb = findall(x -> latr[1] - 1e-6 <= x <= latr[2] + 1e-6, la)
            isempty(ib) && (ib = [argmin(abs.(lo .- mod((lon[1] + lon[2]) / 2, 360)))])
            isempty(jb) && (jb = [argmin(abs.(la .- (lat[1] + lat[2]) / 2))])
            I, J = minimum(ib):maximum(ib), minimum(jb):maximum(jb)
            t = DateTime.(ds[era5_timevar(ds)][:])
            idx = findall(s -> t0 <= s <= t1, t); isempty(idx) && return
            K = first(idx):last(idx)
            rd(s) = NCDataset(f -> Float64.(coalesce.(f[s][I, J, K], NaN)), files[s])
            avg(A) = area === nothing ? A : mean(A; dims = (1, 2))           # one smoothed cell
            sub = every == 1 ? avg : A -> subsample_hours(avg(A), t[K], every)
            u, v = sub(rd("u10")), sub(rd("v10"))
            ρ = rho === nothing ? air_density.(sub(rd("msl")), sub(rd("t2m")), sub(rd("d2m"))) : fill(Float64(rho), size(u))
            U = sqrt.(u .^ 2 .+ v .^ 2); k = ρ .* neutral_cd.(U) .* U
            append!(times, t[K])
            append!(τx, vec(mean(k .* u; dims = (1, 2)))); append!(τy, vec(mean(k .* v; dims = (1, 2))))
            append!(ρs, vec(mean(ρ; dims = (1, 2))))
            append!(us, vec(mean(u; dims = (1, 2)))); append!(vs, vec(mean(v; dims = (1, 2))))
        end
    end
    isempty(times) && error("no ERA5 hours between $t0 and $t1 in $dir")
    return times, τx, τy, ρs, us, vs
end

"A (x, y, hour) sampled at the hours divisible by `every` and linearly interpolated back to every hour (held at the ends)."
function subsample_hours(A, t, every)
    keep = findall(s -> hour(s) % every == 0, t)
    length(keep) < 2 && error("fewer than two $(every)-hourly samples")
    B = similar(A)
    for n in eachindex(t)
        m = clamp(searchsortedlast(keep, n), 1, length(keep) - 1)
        a, b = keep[m], keep[m+1]
        w = clamp((n - a) / (b - a), 0.0, 1.0)
        B[:, :, n] .= (1 - w) .* view(A, :, :, a) .+ w .* view(A, :, :, b)
    end
    return B
end
