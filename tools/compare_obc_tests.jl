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
const SKIN = 5         # cells excluded around the edge when measuring the interior

"""
    summarise(surface_file)

Boundary-versus-interior velocity statistics for one run's surface output.
Returns `nothing` if the file cannot be read.
"""
function summarise(surface_file)
    ds = Dataset(surface_file)
    try
        Nt = length(ds["time"][:])
        times = Float64.(ds["time"][:])
        mask = ds["inactive_nodes_ccc"][:, :, 1] .!= 0
        Nx, Ny = size(mask)

        edge_v = 0.0; edge_u = 0.0; inner_v = 0.0; inner_u = 0.0
        edge_v_sum = 0.0; inner_v_sum = 0.0; n = 0

        for t in 1:Nt
            u_raw = Float64.(ds["u"][:, :, 1, t])
            v_raw = Float64.(ds["v"][:, :, 1, t])
            uc = 0.5 .* (u_raw[1:end-1, :] .+ u_raw[2:end, :])
            vc = 0.5 .* (v_raw[:, 1:end-1] .+ v_raw[:, 2:end])
            uc[mask] .= NaN
            vc[mask] .= NaN

            nanmax(a) = (b = filter(isfinite, a); isempty(b) ? 0.0 : maximum(abs, b))
            nanmean(a) = (b = filter(isfinite, a); isempty(b) ? 0.0 : mean(abs, b))

            # the western edge is where the jet sits; keep it separate from the interior
            ev = nanmax(vc[1:EDGE_CELLS, :])
            iv = nanmax(vc[SKIN+1:Nx-SKIN, SKIN+1:Ny-SKIN])
            edge_v = max(edge_v, ev)
            inner_v = max(inner_v, iv)
            edge_u = max(edge_u, nanmax(uc[:, Ny-EDGE_CELLS+1:Ny]))
            inner_u = max(inner_u, nanmax(uc[SKIN+1:Nx-SKIN, SKIN+1:Ny-SKIN]))
            edge_v_sum += nanmean(vc[1:EDGE_CELLS, :])
            inner_v_sum += nanmean(vc[SKIN+1:Nx-SKIN, SKIN+1:Ny-SKIN])
            n += 1
        end

        return (; days=times[end] / 86400, snapshots=Nt,
            edge_v, inner_v, edge_u, inner_u,
            ratio_max=inner_v == 0 ? NaN : edge_v / inner_v,
            ratio_mean=inner_v_sum == 0 ? NaN : (edge_v_sum / n) / (inner_v_sum / n))
    catch e
        @warn "could not read $surface_file" exception = e
        return nothing
    finally
        close(ds)
    end
end

function compare_obc_tests(dir="obc_tests")
    files = sort(filter(f -> endswith(f, "_surface.nc"), readdir(dir; join=true)))

    if isempty(files)
        println("No *_surface.nc files in $dir/ yet.")
        println("Submit the tests with ./run_obc_tests, then rerun this once they finish.")
        return nothing
    end

    println("Open boundary configurations in $dir/")
    println("  west edge = outermost $EDGE_CELLS cells; interior excludes $SKIN cells all round.")
    println("  A configuration that is not pinning the flow has ratios near 1.\n")
    @printf("%-22s %6s %9s %9s %9s %9s %9s\n",
        "config", "days", "edge|v|", "int|v|", "max ratio", "mean rat", "edge|u| N")
    println("-"^80)

    rows = []
    for f in files
        name = replace(basename(f), "_surface.nc" => "")
        s = summarise(f)
        s === nothing && continue
        push!(rows, (name, s))
        @printf("%-22s %6.1f %9.3f %9.3f %9.2f %9.2f %9.3f\n",
            name, s.days, s.edge_v, s.inner_v, s.ratio_max, s.ratio_mean, s.edge_u)
    end

    isempty(rows) && return nothing

    println()
    ok = filter(r -> isfinite(r[2].ratio_max), rows)
    if !isempty(ok)
        best = argmin(r -> r[2].ratio_max, ok)
        @printf("Least boundary-pinned: %s (peak west |v| is %.2fx the interior peak)\n",
            best[1], best[2].ratio_max)
    end

    expected = ["gradient", "clamped", "radiation", "radiation_bv",
        "pa_gravity", "pa_plain", "radiation_both", "gradient_radnormal"]
    missing = filter(n -> !any(r -> r[1] == n, rows), expected)
    isempty(missing) ||
        println("No output from: ", join(missing, ", "),
            " — check logs/obc_*.out; a configuration that diverges writes nothing.")

    return rows
end

# Run it when the file is executed directly, so `julia tools/compare_obc_tests.jl` works.
if abspath(PROGRAM_FILE) == @__FILE__
    compare_obc_tests(isempty(ARGS) ? "obc_tests" : ARGS[1])
end
