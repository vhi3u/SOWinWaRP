# Emits the before/after evidence for the find_bsose_time_index fix, as markdown.
#
#     julia --project=. tools/show_date_offset.jl
#
# Reads only the `time` axis of a B-SOSE monthly file, so it runs anywhere the data lives and
# needs no grid, no GPU and no Oceananigans. `old_index` is the resolver as it stood before the
# fix; `new_index` is the current one, reimplemented here so the two can be printed side by side
# without reverting anything.

using NCDatasets, Dates

const DATA = get(ENV, "DATA", joinpath(@__DIR__, "..", "data"))
const F156 = joinpath(DATA, get(ENV, "F156", "surfSflx_bsoseI156_2013to2024_monthly.nc"))
const F105 = joinpath(DATA, get(ENV, "F105", "bsose_i105_2008to2012_monthly_Theta.nc"))

function bsose_averaging_window(stamp)
    w = round(DateTime(stamp), Month)
    return (w - Month(1), w)
end

function bsose_window_center(stamp)
    a, b = bsose_averaging_window(stamp)
    return a + Millisecond(Dates.value(Millisecond(b - a)) ÷ 2)
end
label(stamp) = Dates.format(bsose_window_center(stamp), "yyyy-mm")

"""The resolver before the fix: scan for the window containing the target."""
function old_index(stamps, target)
    t = DateTime(target)
    for (i, s) in enumerate(stamps)
        ws, wp = bsose_averaging_window(s)
        ws <= t < wp && return i
    end
    return argmin([abs((DateTime(s) - t).value) for s in stamps])
end

"""The resolver after the fix: a record stamp means that record."""
function new_index(stamps, target)
    t = DateTime(target)
    e = findfirst(s -> DateTime(s) == t, stamps)
    isnothing(e) || return e
    return old_index(stamps, target)
end

function read_stamps(f)
    ds = Dataset(f)
    try
        return [DateTime(t) for t in ds["time"][:]]
    finally
        close(ds)
    end
end

st156 = read_stamps(F156)
st105 = read_stamps(F105)

# ── 1. the whole record axis ──────────────────────────────────────────────────────────────────
println("### Every record stamp, resolved against its own file\n")
println("| dataset | records | correct before | correct after |")
println("|---|---|---|---|")
for (nm, st) in (("B-SOSE i156 monthly", st156), ("B-SOSE i105 monthly", st105))
    ob = count(k -> old_index(st, st[k]) == k, eachindex(st))
    nb = count(k -> new_index(st, st[k]) == k, eachindex(st))
    println("| $nm | $(length(st)) | **$ob** | $nb |")
end

# ── 2. the 2-year run ─────────────────────────────────────────────────────────────────────────
start_date, end_date = DateTime("2014-06-01"), DateTime("2016-06-01")
inside = findall(d -> start_date <= bsose_window_center(d) <= end_date, st156)
lo = max(firstindex(st156), first(inside) - 1)
hi = min(lastindex(st156), last(inside) + 1)

println("\n### `bc_dates` for `2Y_wind_fw_sw` (2014-06-01 → 2016-06-01)\n")
println("`model.jl:535` passes these stamps to the winds, surface fluxes, shortwave, open")
println("boundary conditions and sponge forcing.\n")
println("| model clock | stamp | month loaded BEFORE | month loaded AFTER |")
println("|---|---|---|---|")
wrong = 0
for k in lo:hi
    o, n = old_index(st156, st156[k]), new_index(st156, st156[k])
    ol, nl = label(st156[o]), label(st156[n])
    bad = ol != label(st156[k]); global wrong += bad
    println("| $(label(st156[k])) | $(st156[k]) | $ol$(bad ? " ❌" : "") | $nl |")
end
n = hi - lo + 1
println("\n**$wrong of $n records loaded the wrong month.** ",
        "Months never loaded at all: ",
        join([label(st156[k]) for k in lo:hi
              if !(k in [old_index(st156, st156[j]) for j in lo:hi])], ", "), ".")

# ── 3. the symptom in the MLD table ───────────────────────────────────────────────────────────
sel = [d for d in st156 if month(bsose_window_center(d)) in [6,7,8,9,10,11,12,1] &&
                           DateTime("2014-06-01") <= bsose_window_center(d) <= DateTime("2015-02-01")]
println("\n### The visible symptom: `compare_mld.jl`, 8 months\n")
println("```")
println("before: ", [old_index(st156, d) for d in sel], "  -> ",
        length(unique([old_index(st156, d) for d in sel])), " distinct records for 8 months")
println("after : ", [new_index(st156, d) for d in sel], "  -> ",
        length(unique([new_index(st156, d) for d in sel])), " distinct records for 8 months")
println("```")
println("\nThe repeated indices are why the B-SOSE row printed `188.7` twice, `215.0` twice and")
println("`28.1` twice, in all five latitude bands and in the section table.")

# ── 4. why ────────────────────────────────────────────────────────────────────────────────────
println("\n### Why\n")
println("A B-SOSE stamp sits at the **end** of the window it averages, and the stamps drift")
println("about ten hours per record, so most land just past their own month boundary and fall")
println("inside the *next* record's window:\n")
println("```")
for k in 1:4
    ws, wp = bsose_averaging_window(st156[k])
    println("stamp $(st156[k])  averages $(label(st156[k]))  window [$ws, $wp)  ",
            st156[k] < wp ? "stamp is inside  -> correct" : "stamp is PAST wp -> next record")
end
println("```")
