# ==============================================================================
# compare_obc_tests.jl
#
# Ranks the open boundary configurations submitted by ./run_obc_tests.
#
# The question each test is asking is whether the boundary still holds a fast jet
# along its own edge, so the headline number is the ratio of the peak boundary
# velocity to the peak interior velocity. A configuration that has fixed the
# problem has a ratio near 1; the clamped reference should be well above it.
#
#     julia --project=. tools/compare_obc_tests.jl
#     julia --project=. tools/compare_obc_tests.jl some/other/dir
# ==============================================================================

using NCDatasets
using Printf
using Statistics

const EDGE_CELLS = 2   # cells counted as "the boundary"
const SKIN = 5         # cells excluded around the edge when measuring the whole interior
const REF_OFFSET = 24  # distance in from the west face of the latitude-matched reference strip

"""
    summarise(surface_file)

Boundary-versus-interior velocity statistics for one run's surface output.
Returns `nothing` if the file cannot be read.

The headline comparison is against a reference strip the same shape as the edge strip and
at the same latitudes, `REF_OFFSET` cells further in — just outside the sponge. Comparing
the edge against the whole interior instead would confound the boundary with geography,
since the west face spans the energetic ACC band and the quiet south together while a
domain-wide average is dominated by the quiet part.
"""
function summarise(surface_file)
    ds = Dataset(surface_file)
    try
        Nt = length(ds["time"][:])
        times = Float64.(ds["time"][:])
        mask = ds["inactive_nodes_ccc"][:, :, 1] .!= 0
        Nx, Ny = size(mask)

        edge_v = 0.0; edge_u = 0.0; inner_v = 0.0; inner_u = 0.0
        ref_v = 0.0
        edge_v_sum = 0.0; ref_v_sum = 0.0; inner_v_sum = 0.0; n = 0

        for t in 1:Nt
            u_raw = Float64.(ds["u"][:, :, 1, t])
            v_raw = Float64.(ds["v"][:, :, 1, t])
            uc = 0.5 .* (u_raw[1:end-1, :] .+ u_raw[2:end, :])
            vc = 0.5 .* (v_raw[:, 1:end-1] .+ v_raw[:, 2:end])
            uc[mask] .= NaN
            vc[mask] .= NaN

            nanmax(a) = (b = filter(isfinite, a); isempty(b) ? 0.0 : maximum(abs, b))
            nanmean(a) = (b = filter(isfinite, a); isempty(b) ? 0.0 : mean(abs, b))

            # the western edge is where the jet sits
            edge = vc[1:EDGE_CELLS, :]
            ref = vc[REF_OFFSET+1:REF_OFFSET+EDGE_CELLS, :]   # same latitudes, further in
            edge_v = max(edge_v, nanmax(edge))
            ref_v = max(ref_v, nanmax(ref))
            inner_v = max(inner_v, nanmax(vc[SKIN+1:Nx-SKIN, SKIN+1:Ny-SKIN]))
            edge_u = max(edge_u, nanmax(uc[:, Ny-EDGE_CELLS+1:Ny]))
            inner_u = max(inner_u, nanmax(uc[SKIN+1:Nx-SKIN, SKIN+1:Ny-SKIN]))
            edge_v_sum += nanmean(edge)
            ref_v_sum += nanmean(ref)
            inner_v_sum += nanmean(vc[SKIN+1:Nx-SKIN, SKIN+1:Ny-SKIN])
            n += 1
        end

        return (; days=times[end] / 86400, snapshots=Nt,
            edge_v, ref_v, inner_v, edge_u, inner_u,
            ratio_max=ref_v == 0 ? NaN : edge_v / ref_v,
            ratio_mean=ref_v_sum == 0 ? NaN : (edge_v_sum / n) / (ref_v_sum / n),
            ratio_domain=inner_v == 0 ? NaN : edge_v / inner_v)
    catch e
        @warn "could not read $surface_file" exception = e
        return nothing
    finally
        close(ds)
    end
end

function compare_obc_tests(dir="obc_tests")
    if !isdir(dir)
        println("No $dir/ directory here — run this from the repository root, or pass the ",
            "directory holding the *_surface.nc files.")
        return nothing
    end

    files = sort(filter(readdir(dir; join=true)) do f
        endswith(f, "_surface.nc") && !endswith(f, "_free_surface.nc")
    end)

    if isempty(files)
        println("No *_surface.nc files in $dir/ yet.")
        println("Submit the tests with ./run_obc_tests, then rerun this once they finish.")
        return nothing
    end

    println("Open boundary configurations in $dir/")
    println("  edge   = west face, outermost $EDGE_CELLS columns")
    println("  ref    = same latitudes, $REF_OFFSET cells further in (just outside the sponge)")
    println("  ratios compare edge to ref, so geography cancels. Near 1 = nothing pinned.\n")
    @printf("%-22s %6s %9s %9s %9s %9s %9s\n",
        "config", "days", "edge|v|", "ref|v|", "max ratio", "mean rat", "vs domain")
    println("-"^86)

    rows = []
    for f in files
        name = replace(basename(f), "_surface.nc" => "")
        s = summarise(f)
        s === nothing && continue
        push!(rows, (name, s))
    end

    isempty(rows) && return nothing

    # A diverged run stops early, so anything well short of the longest run is incomplete.
    full = maximum(r -> r[2].days, rows)
    complete(r) = r[2].snapshots > 1 && r[2].days >= 0.9 * full

    for (name, s) in rows
        @printf("%-22s %6.1f %9.3f %9.3f %9.2f %9.2f %9.2f %s\n",
            name, s.days, s.edge_v, s.ref_v, s.ratio_max, s.ratio_mean, s.ratio_domain,
            complete((name, s)) ? "" : "  DIVERGED — numbers are the initial state, ignore")
    end

    println()
    ok = filter(r -> complete(r) && isfinite(r[2].ratio_max), rows)
    if isempty(ok)
        println("No configuration ran to completion — nothing to rank.")
    else
        best = argmin(r -> r[2].ratio_max, ok)
        @printf("Least boundary-pinned of the %d that completed: %s (peak west |v| is %.2fx the interior peak)\n",
            length(ok), best[1], best[2].ratio_max)
    end

    return rows
end

# Run on include as well as from the command line, matching the other analysis scripts in
# tools/ — `include("tools/compare_obc_tests.jl")` is how these are usually invoked here.
# Call `compare_obc_tests(dir)` directly to point it somewhere else.
compare_obc_tests(isempty(ARGS) ? "obc_tests" : ARGS[1])
