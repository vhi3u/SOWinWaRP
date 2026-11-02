# ==============================================================================================
# nested_probe.jl -- does BSOSEMonthly satisfy the nested-ocean dataset interface?
#
#     julia --project=. examples/nested_probe.jl
#
# WHY THIS AND NOT A MODEL SCRIPT. A model script dies on the first missing method, you fix it,
# it dies on the second. This runs every check independently and prints the WHOLE list, so one
# login-node run (no GPU, no job) tells us exactly what to write.
#
# Every check is wrapped: a failure is recorded and the probe carries on. The summary at the end
# is the work list.
#
# WHAT THE NESTED CODE ACTUALLY CALLS on a parent dataset -- read off
# src/DataWrangling/prescribed_ocean.jl and src/NestedModels/nested_ocean_model.jl on the
# ss/nested-ocean-model branch:
#
#     Metadata(name; dataset, dates, region, dir)   for :temperature, :salinity,
#                                                   :u_velocity, :v_velocity, :free_surface
#     FieldTimeSeries(metadata, arch; time_indices_in_memory, time_indexing, start_time)
#     all_dates(dataset, :temperature)
#     expand_dates(dataset, name, dates)            generic fallback returns `dates`
#     window_center(Metadatum(name; dataset, date))
#     default_download_directory(dataset)
#     default_horizontal_padding(dataset)           NO fallback -- `function ... end`
#     matching_bathymetry(dataset; region, dir)     generic fallback returns `nothing`
# ==============================================================================================

using Dates, Printf

const REPO = normpath(joinpath(@__DIR__, ".."))

const RESULTS = Tuple{String,Bool,String}[]

"""Run one check, record it, never throw. `f` returns a short string on success."""
function check(name, f)
    print(rpad(name, 52))
    try
        note = f()
        push!(RESULTS, (name, true, note))
        println("PASS  ", note)
    catch e
        msg = first(split(sprint(showerror, e), '\n'))
        length(msg) > 88 && (msg = msg[1:88] * "…")
        push!(RESULTS, (name, false, msg))
        println("FAIL  ", msg)
    end
    return nothing
end

println("\n", "="^104)
println("NESTED-OCEAN INTERFACE PROBE -- BSOSEMonthly against NumericalEarth ss/nested-ocean-model")
println("="^104, "\n")

# ── 0. the packages, and whether our own setup file still loads ───────────────────────────────
check("NumericalEarth loads") do
    @eval using NumericalEarth
    @eval using Oceananigans
    @eval using Oceananigans.Units
    string(pkgversion(NumericalEarth))
end

# setup_bsose.jl was written against NumericalEarth 0.8.2. If the nested branch moved an API it
# uses, THIS is where we find out, and it is the single most important line in the probe: every
# check below needs BSOSEMonthly, which lives in that file.
check("src/setup_bsose.jl loads") do
    @eval module BS
        include(joinpath($REPO, "src", "setup_bsose.jl"))
    end
    "module BS"
end

ok(name) = any(r -> r[1] == name && r[2], RESULTS)
if !ok("src/setup_bsose.jl loads")
    println("\n  setup_bsose.jl does not load against this NumericalEarth. Nothing below can run.")
    println("  Fix that first -- the error above names the API that moved.\n")
    exit(1)
end

# ── 1. the dataset object ─────────────────────────────────────────────────────────────────────
const DIR = get(ENV, "BSOSE_DIR", BS.default_bsose_directory())
dataset = nothing
check("BSOSEMonthly constructs") do
    global dataset = BS.BSOSEMonthly(dir = DIR, iteration = 156)
    BS.summary(dataset)
end
dataset === nothing && (println("\n  no dataset; stopping.\n"); exit(1))

# ── 2. the five names the nested code asks for ────────────────────────────────────────────────
const NEEDED = (:temperature, :salinity, :u_velocity, :v_velocity, :free_surface)
for name in NEEDED
    check("variable :$name resolves to a file") do
        tok, _ = BS.BSOSE_FILES[dataset.iteration][name]
        f = BS.resolve_bsose_filename(dataset.dir, name, dataset.iteration)
        isfile(joinpath(dataset.dir, f)) ? "$tok -> $f" : "$tok -> $f  (NOT ON DISK)"
    end
end

# ── 3. the generics, one method at a time ─────────────────────────────────────────────────────
const DATES = (DateTime(2014, 6, 1), DateTime(2014, 6, 16))

check("all_dates(dataset, :temperature)") do
    d = NumericalEarth.DataWrangling.all_dates(dataset, :temperature)
    "$(length(collect(d))) dates, $(first(d)) .. $(last(d))"
end

check("expand_dates(dataset, :temperature, dates)") do
    d = NumericalEarth.DataWrangling.expand_dates(dataset, :temperature, DATES)
    # The generic fallback returns `dates` unchanged. For a (start, end) TUPLE that is wrong --
    # the nested code expects the tuple expanded to the dataset's native cadence -- so a result
    # that is still a 2-tuple means we must write a method, even though this check "passes".
    d isa Tuple ? "returned the tuple unchanged -- NEEDS A METHOD" : "$(length(collect(d))) native dates"
end

check("window_center(Metadatum(:temperature))") do
    md = NumericalEarth.Metadatum(:temperature; dataset, date = DateTime(2014, 6, 1))
    string(NumericalEarth.DataWrangling.window_center(md))
end

check("default_download_directory(dataset)") do
    string(NumericalEarth.DataWrangling.default_download_directory(dataset))
end

check("default_horizontal_padding(dataset)") do
    # Declared as `function default_horizontal_padding end` with NO fallback, so this is
    # expected to fail until we add a method. It is the one mandatory addition.
    string(NumericalEarth.DataWrangling.default_horizontal_padding(dataset))
end

check("matching_bathymetry(dataset)") do
    b = NumericalEarth.DataWrangling.matching_bathymetry(dataset; dir = DIR)
    b === nothing ? "nothing (generic fallback -- fine, the seafloor comes from missing T)" : string(typeof(b))
end

# ── 4. a Metadata and a FieldTimeSeries, the two things PrescribedOcean actually builds ───────
check("Metadata(:temperature; dataset, dates, region, dir)") do
    reg = NumericalEarth.BoundingBox(longitude = (90.0, 150.0), latitude = (-70.0, -45.0))
    md = NumericalEarth.Metadata(:temperature; dataset, dates = [DateTime(2014, 6, 1)],
                                 region = reg, dir = DIR)
    string(typeof(md).name.name)
end

check("FieldTimeSeries from that Metadata") do
    reg = NumericalEarth.BoundingBox(longitude = (90.0, 150.0), latitude = (-70.0, -45.0))
    md = NumericalEarth.Metadata(:temperature; dataset, dates = [DateTime(2014, 6, 1)],
                                 region = reg, dir = DIR)
    fts = Oceananigans.OutputReaders.FieldTimeSeries(md, CPU())
    "size $(size(fts))"
end

# ── 5. the whole adapter ──────────────────────────────────────────────────────────────────────
check("PrescribedOcean(region, dates, dataset)") do
    reg = NumericalEarth.BoundingBox(longitude = (90.0, 150.0), latitude = (-70.0, -45.0))
    po = NumericalEarth.Oceans.PrescribedOcean(reg, DATES, dataset; architecture = CPU(), dir = DIR)
    "built; η is $(po.η === nothing ? "NOTHING -- nested needs it" : "present")"
end

# ── summary ───────────────────────────────────────────────────────────────────────────────────
println("\n", "="^104)
nfail = count(r -> !r[2], RESULTS)
@printf("%d of %d checks passed\n", length(RESULTS) - nfail, length(RESULTS))
println("="^104)
if nfail > 0
    println("\nWORK LIST -- each line is a method to write, or a file to download:\n")
    for (name, passed, msg) in RESULTS
        passed || @printf("  %-50s %s\n", name, msg)
    end
end
println("\nAlso read any check that PASSED with a caveat in its note -- `expand_dates` returning")
println("the tuple unchanged is a pass that still needs a method.\n")
