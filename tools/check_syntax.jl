# ==============================================================================
# check_syntax.jl -- does this file actually parse?
#
#     julia tools/check_syntax.jl src/model.jl src/setup_bsose.jl
#     julia tools/check_syntax.jl                 # defaults to every .jl in src/
#
# Why this exists rather than a one-line `Meta.parseall(read(f, String))`:
#
# `Meta.parseall` does NOT reliably throw on bad syntax. It returns an expression tree with
# the failure buried in it as an `Expr(:error, ...)` node. So
#
#     Meta.parseall(read(f, String)); println("parses")
#
# prints "parses" for a file containing a stray `end`, and you find out instead when the run
# dies forty minutes in, after the grid, the boundary conditions and the initial conditions
# have all been built:
#
#     ERROR: LoadError: ParseError:
#     # Error @ src/model.jl:626:1
#         end
#     end
#     └─┘ ── invalid identifier
#
# `include` evaluates top-level expressions one at a time, so everything above the bad line
# runs first. A dangling `end` left by deleting an `if`/`elseif` head is the usual cause, and
# it is invisible to the eye because the body below it is still correctly indented.
#
# This script checks for BOTH failure modes -- a thrown ParseError and an embedded error node
# -- and self-tests against a known-bad string first, so a silent regression in the checker
# itself shows up immediately.
#
# It does NOT catch lowering or runtime errors (undefined names, `local` misuse, bad method
# calls). It only answers "is this valid Julia syntax".
# ==============================================================================

"""Return a list of syntax problems in `src`, empty if it is clean."""
function syntax_errors(src::AbstractString, name::AbstractString)
    errs = String[]

    ex = try
        Meta.parseall(src; filename = name)
    catch e
        push!(errs, "THROWN: " * first(sprint(showerror, e), 400))
        return errs
    end

    walk(e) = if e isa Expr
        if e.head === :error || e.head === :incomplete
            push!(errs, string(e.head, ": ", first(string(e.args), 400)))
        end
        foreach(walk, e.args)
    end
    walk(ex)

    return errs
end

# ---- self-test: the checker must flag a file with a stray `end` ----
const PROBE = "x = 1\nif x > 0\n    y = 2\nend\nend\n"

if isempty(syntax_errors(PROBE, "selftest.jl"))
    println("*** check_syntax.jl is BROKEN: it did not flag a stray `end`. Fix it before trusting a pass. ***")
    exit(2)
end

# ---- check the requested files ----
files = isempty(ARGS) ?
        sort(filter(f -> endswith(f, ".jl"),
                    readdir(joinpath(@__DIR__, "..", "src"); join = true))) :
        ARGS

failed = 0
for f in files
    global failed
    errs = syntax_errors(read(f, String), f)
    if isempty(errs)
        println("OK     ", f)
    else
        failed += 1
        println("FAIL   ", f)
        for e in errs
            println("         ", replace(e, "\\n" => "\n         "))
        end
    end
end

println()
println(failed == 0 ? "All $(length(files)) file(s) parse." : "$failed of $(length(files)) file(s) FAILED to parse.")
exit(failed == 0 ? 0 : 1)
