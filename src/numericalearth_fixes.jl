# numericalearth_fixes.jl -- workarounds for NumericalEarth behaviour that hurts our runs, in one place.
# Included by every run script (examples/sowinwarp_nested.jl through src/setup_bsose.jl,
# examples/sowinwarp_glorys_baseline.jl, examples/nested_glorys_mini.jl). Each is guarded, so it is a
# no-op on a NumericalEarth that lacks the hook, and changes how data are read, never what.
#
# ERA5's yearly files: no whole-file pre-read. NumericalEarth (#773, from d59fa0a) `read`s every file of a
# series' window WHOLE before loading it, to warm the page cache, and again in the background for the next
# window: meant for one-record-per-file datasets. ERA5YearlySingleLevel keeps a year of hourly records of a
# variable in ONE file, so each daily reload of the 24 records in memory pulled all 9 variables' yearly files
# in whole, twice: the suspected cause of nested_glorys_mini's 10-20 min of wall time per model day
# (2026-11-11), a cost per model day, not per step. The series reads only its records anyway. (B-SOSE has the same override in src/setup_bsose.jl.)

using NumericalEarth

if isdefined(NumericalEarth.DataWrangling, :window_paths) && isdefined(NumericalEarth.DataWrangling, :DatasetBackend)
    NumericalEarth.DataWrangling.window_paths(::NumericalEarth.DataWrangling.DatasetBackend{<:Any, <:Any, <:Any,
                                                  <:NumericalEarth.DataWrangling.Metadata{<:NumericalEarth.ERA5YearlySingleLevel}},
                                              indices) = String[]
end
