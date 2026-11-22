# ==============================================================================================
# bsose_windows.jl -- B-SOSE i156 averaged over one of the run's 30-day windows, on a box of cells,
# from its MONTHLY files (default) or its 5-DAY files. Used by compare_mld_5day.jl and under_ice_heat.jl.
#
# SOURCE=monthly (default): <token>_bsoseI156_2013to2024_monthly.nc. Each record is a calendar-month mean
#   (stamped at the month's end; the month is the stamp minus 14 days, as in single_column_bsose.jl). A
#   window is the mean of the months it overlaps, each weighted by its days in the window: exact for a
#   time mean of T and S, since the run's windows are 30 days and drift off the calendar (2014-06-01 +
#   n x 30 days). The months must cover at least 99% of the window.
# SOURCE=5day: <token>_bsoseI156_<year>_5day.nc, the old rule kept exactly: every record whose centre
#   (stamp - 2.5 days) falls in the window, equal weights, at least min_records of them.
#
# READS. The B-SOSE files are NetCDF-4, uncompressed and contiguous, 2160 x 588 (x 52) per record. A box
# 360 cells wide is ~5,900 short strided pieces per 3-D record; full-width rows (v[:, jr, k, n]) are one
# contiguous block per level. So each record is read at full width over the box's rows and cut to its
# columns in memory: ~50 MB per 3-D record to 1400 m.
# ==============================================================================================

using NCDatasets, Dates

const BSOSE_SOURCE = get(ENV, "SOURCE", "monthly")
BSOSE_SOURCE in ("monthly", "5day") || error("SOURCE must be monthly or 5day, got $BSOSE_SOURCE")

"One B-SOSE record: file, variable, record index, and the period it averages, [t0, t1)."
struct BRecord
    path::String
    var::String
    n::Int
    t0::DateTime
    t1::DateTime
end

"`<token>_bsoseI156_2013to2024_monthly.nc` (year = nothing) or `<token>_bsoseI156_<year>_5day.nc` in dir, any case; \"\" if absent."
function bsose_path(dir, token, year = nothing)
    want = lowercase(year === nothing ? "$(token)_bsoseI156_2013to2024_monthly.nc" : "$(token)_bsoseI156_$(year)_5day.nc")
    hits = filter(f -> lowercase(f) == want, readdir(dir))
    return isempty(hits) ? "" : joinpath(dir, first(hits))
end

bsose_opens(p) = !isempty(p) && (try NCDataset(_ -> true, p) catch; false end)

"The data variable of a B-SOSE file: the first of `names` present, else its only variable over (x, y, time) or (x, y, z, time)."
function bsose_data_var(ds, names, path)
    for n in names
        haskey(ds, n) && return n
    end
    cands = [k for k in keys(ds) if ndims(ds[k]) >= 3]
    length(cands) == 1 && return only(cands)
    error("none of $names in $path, and no single data variable ($(cands))")
end

"""
The years (5-day) for which every token's file opens, warning about the rest; for monthly, checks each token's
one file and errors naming any that is missing. Returns the years to use (monthly: all of `years`).
"""
function bsose_usable_years(dir, tokens, years)
    if BSOSE_SOURCE == "monthly"
        bad = [t for t in tokens if !bsose_opens(bsose_path(dir, t))]
        isempty(bad) || error("B-SOSE monthly file(s) missing or unreadable in $dir: " *
                              join(("$(t)_bsoseI156_2013to2024_monthly.nc" for t in bad), ", ") *
                              " (or run with SOURCE=5day)")
        return collect(years)
    end
    ok = filter(years) do y
        bad = [t for t in tokens if !bsose_opens(bsose_path(dir, t, y))]
        isempty(bad) || @warn "B-SOSE $y skipped: $(join(bad, ", ")) 5-day file missing or unreadable (truncated download?)"
        isempty(bad)
    end
    isempty(ok) && error("no year with B-SOSE's 5-day $(join(tokens, ", ")) all readable in $dir")
    return ok
end

"Every record of a token: monthly from the one 2013to2024 file, 5-day from each year's file."
function bsose_records(dir, token, names, years)
    recs = BRecord[]
    paths = BSOSE_SOURCE == "monthly" ? [bsose_path(dir, token)] : [bsose_path(dir, token, y) for y in years]
    for path in paths
        NCDataset(path) do ds
            var = bsose_data_var(ds, names, path)
            for (n, stamp) in enumerate(ds["time"][:])
                s = DateTime(stamp)
                if BSOSE_SOURCE == "monthly"
                    d = s - Day(14)
                    t0 = DateTime(Dates.year(d), Dates.month(d), 1)
                    push!(recs, BRecord(path, var, n, t0, t0 + Month(1)))
                else
                    push!(recs, BRecord(path, var, n, s - Day(5), s))
                end
            end
        end
    end
    return recs
end

"(record, weight) pairs making up the window (w0, w1], or nothing if B-SOSE does not cover it."
function bsose_weights(recs, w0, w1; min_records = 4)
    if BSOSE_SOURCE == "5day"
        sel = filter(r -> w0 < r.t0 + Hour(60) <= w1, recs)
        length(sel) < min_records && return nothing
        return [(r, 1 / length(sel)) for r in sel]
    end
    ov(r) = max(0, Dates.value(min(r.t1, w1) - max(r.t0, w0)))
    sel = filter(r -> ov(r) > 0, recs)
    tot = sum(ov, sel; init = 0)
    tot < 0.99 * Dates.value(w1 - w0) && return nothing
    return [(r, ov(r) / tot) for r in sel]
end

"""
B-SOSE's window mean over the box ir × jr: levels 1:levels (3-D) or the surface field (levels = nothing).
Each record is read at full width (contiguous on disk) and cut to ir in memory. Missing values become NaN.
"""
function bsose_window_mean(recs, w0, w1, ir, jr, levels; min_records = 4)
    wts = bsose_weights(recs, w0, w1; min_records)
    wts === nothing && return nothing
    acc = nothing
    for (r, wt) in wts
        A = NCDataset(r.path) do ds
            v = ds[r.var]
            B = levels === nothing ? (ndims(v) == 4 ? v[:, jr, 1, r.n] : v[:, jr, r.n]) : v[:, jr, 1:levels, r.n]
            Float64.(coalesce.(levels === nothing ? B[ir, :] : B[ir, :, :], NaN))
        end
        acc = acc === nothing ? wt .* A : acc .+ wt .* A
    end
    return acc
end

"How a window was built, for the header line."
bsose_source_note() = BSOSE_SOURCE == "monthly" ?
    "B-SOSE monthly means, each window the overlap-weighted blend of its calendar months" :
    "B-SOSE 5-day records with centres in each window, equal weights"
