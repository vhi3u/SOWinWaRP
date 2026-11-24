# ==============================================================================
# download_nsidc.jl -- fetch the NOAA/NSIDC Sea Ice Concentration CDR (G02202 version 6), southern hemisphere,
# 25 km polar stereographic: the monthly files for YEARS and the ancillary grid file (latitude, longitude, land).
# Public HTTPS, no login (noaadata.apps.nsidc.org). ~0.5 MB a month, ~6 MB a year, plus ~5 MB of ancillary.
# Files already there are skipped, so it can be rerun.
#
#     julia tools/download_nsidc.jl                       # 2014-2016 into $BSOSE_DIR/nsidc (or ./data/nsidc)
#     YEARS=2013-2017 DEST=/some/dir julia tools/download_nsidc.jl
#
# Then tools/compare_nsidc.jl reads them from the same place (NSIDC_DIR, default $BSOSE_DIR/nsidc).
# Uses only Julia's standard library: run it anywhere with internet (the PACE login node, or locally and rsync DEST).
# ==============================================================================

import Downloads

const BASE = "https://noaadata.apps.nsidc.org/NOAA/G02202_V6"
const ANCILLARY = "G02202-ancillary-pss25-v06r00.nc"

years = let s = get(ENV, "YEARS", "2014-2016")
    occursin('-', s) ? (parse(Int, split(s, '-')[1]):parse(Int, split(s, '-')[2])) : [parse(Int, y) for y in split(s, ',')]
end
dest = get(ENV, "DEST", haskey(ENV, "BSOSE_DIR") ? joinpath(ENV["BSOSE_DIR"], "nsidc") : joinpath("data", "nsidc"))
mkpath(dest)

function fetch(url, file)
    path = joinpath(dest, file)
    isfile(path) && filesize(path) > 0 && return :kept
    tmp = path * ".part"
    Downloads.download(url, tmp)
    mv(tmp, path; force = true)
    return :new
end

listing = String(take!(Downloads.download("$BASE/south/monthly/", IOBuffer())))
names = unique(m.captures[1] for m in eachmatch(r"href=\"(sic_pss25_(\d{6})_[^\"]*\.nc)\"", listing))
wanted = filter(f -> parse(Int, f[11:14]) in years, names)
isempty(wanted) && error("no monthly files for $years in $BASE/south/monthly/ (the listing format may have changed)")

counts = Dict(:new => 0, :kept => 0)
counts[fetch("$BASE/ancillary/$ANCILLARY", ANCILLARY)] += 1
for f in sort(wanted)
    counts[fetch("$BASE/south/monthly/$f", f)] += 1
end
found = sort(unique(f[11:16] for f in wanted))
println("NSIDC CDR v6 south, years $(first(years))-$(last(years)): $(length(wanted)) monthly files ($(found[1])..$(found[end])) ",
        "+ ancillary in $dest; $(counts[:new]) downloaded, $(counts[:kept]) already there")
length(found) == 12 * length(years) ||
    @warn "expected $(12 * length(years)) months, found $(length(found)): some months are missing upstream"
