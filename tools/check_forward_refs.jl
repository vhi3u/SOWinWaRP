# ==============================================================================
# check_forward_refs.jl -- find globals USED before they are DEFINED in a top-to-bottom script.
#
#     julia tools/check_forward_refs.jl                  # src/model.jl
#     julia tools/check_forward_refs.jl src/model.jl tools/compare_bsose.jl
#
# WHY. `src/model.jl` is a 1200-line script that runs top to bottom, and a statement reading a
# global defined further down parses fine, lints fine, and then dies with `UndefVarError` the
# moment the job starts -- after the queue wait, the module load and the bathymetry build. Twice
# now, each costing a submitted job:
#
#   * `SPONGE_DEG = SPONGE_CELLS / SCALING` ten lines above `const SCALING`
#     (tools/compare_bsose.jl, 2026-10-27)
#   * a callback scheduled on `output_interval` sixteen lines above it
#     (src/model.jl, 2026-10-28)
#
# HOW. This walks the AST from `Meta.parseall` rather than matching text. A regex version of
# this file was written first and reported fourteen findings, every one of them false: `let`
# bindings, tuple destructuring, keyword-argument names, and words inside triple-quoted
# strings. None of those exist in the AST, so none of them can be mistaken for code.
#
# WHAT COUNTS. Only top-level executable statements. A function BODY may reference anything,
# because the name is resolved when the function is called, not when it is defined -- so bodies
# are skipped. Local bindings introduced by `let`, `for` and comprehensions are skipped too.
#
# Exit status is 1 if anything is found, so it can gate a submit script.
# ==============================================================================

const TARGETS = isempty(ARGS) ? ["src/model.jl"] : ARGS

isfuncdef(ex) = ex isa Expr && (ex.head === :function ||
                (ex.head === :(=) && ex.args[1] isa Expr && ex.args[1].head === :call))

"""Names this expression BINDS at top level: assignments, destructuring, function names."""
function collect_defs!(ex, out)
    ex isa Expr || return
    if ex.head === :const || ex.head === :global
        for a in ex.args; collect_defs!(a, out); end
    elseif ex.head === :function
        sig = ex.args[1]
        sig isa Expr && sig.head === :call && sig.args[1] isa Symbol && push!(out, sig.args[1])
    elseif ex.head === :(=)
        lhs = ex.args[1]
        if lhs isa Symbol
            push!(out, lhs)
        elseif lhs isa Expr && lhs.head === :tuple          # a, b = f()
            for a in lhs.args; a isa Symbol && push!(out, a); end
        elseif lhs isa Expr && lhs.head === :call           # f(x) = ...
            lhs.args[1] isa Symbol && push!(out, lhs.args[1])
        elseif lhs isa Expr && lhs.head === :(::)           # x::T = ...
            lhs.args[1] isa Symbol && push!(out, lhs.args[1])
        end
    elseif ex.head === :struct
        nm = ex.args[2]
        nm isa Symbol && push!(out, nm)
        nm isa Expr && nm.head === :(<:) && nm.args[1] isa Symbol && push!(out, nm.args[1])
    elseif ex.head in (:abstract, :primitive)
        nm = ex.args[1]
        nm isa Symbol && push!(out, nm)
        nm isa Expr && nm.head === :(<:) && nm.args[1] isa Symbol && push!(out, nm.args[1])
    elseif ex.head in (:block, :if, :elseif, :for, :while, :let, :try, :toplevel, :macrocall)
        # :macrocall covers docstrings -- `"""..."""` attaches to the definition that follows,
        # so the definition lives inside the macrocall and would otherwise go unrecorded.
        for a in ex.args; collect_defs!(a, out); end
    end
end

"""Every name BOUND anywhere inside this statement: `let` bindings, loop variables, lambda
arguments, nested destructuring, and assignments in nested blocks. All of these are in scope by
the time they are read, so none of them is a forward reference."""
function push_bound!(t, out)
    if t isa Symbol
        push!(out, t)
    elseif t isa Expr
        if t.head === :tuple || t.head === :parameters
            for a in t.args; push_bound!(a, out); end
        elseif t.head === :(::) || t.head === :kw || t.head === :(...)
            push_bound!(t.args[1], out)
        end
    end
end

function collect_locals!(ex, out)
    ex isa Expr || return
    if ex.head === :->                                   # k -> f(k)
        push_bound!(ex.args[1], out)
    elseif ex.head === :function
        sig = ex.args[1]
        if sig isa Expr && sig.head === :call
            for a in sig.args[2:end]; push_bound!(a, out); end
        end
    elseif ex.head === :let
        b = ex.args[1]
        for bind in (b isa Expr && b.head === :block ? b.args : [b])
            bind isa Expr && bind.head === :(=) ? push_bound!(bind.args[1], out) :
                                                  push_bound!(bind, out)
        end
    elseif ex.head in (:for, :generator, :comprehension, :typed_comprehension, :filter)
        for a in ex.args
            a isa Expr && a.head === :(=) && push_bound!(a.args[1], out)
        end
    elseif ex.head === :(=)
        push_bound!(ex.args[1], out)                     # including inside nested blocks
    elseif ex.head === :call && !isempty(ex.args) && ex.args[1] isa Symbol
        nothing
    end
    for a in ex.args; collect_locals!(a, out); end
end

"""Names this expression READS. Skips function bodies, keyword names and field accesses."""
function collect_uses!(ex, out)
    if ex isa Symbol
        push!(out, ex)
    elseif ex isa Expr
        isfuncdef(ex) && return                      # body resolved at call time
        # A struct's own name is a definition, not a read; its field types are declarations.
        ex.head in (:struct, :abstract, :primitive) && return
        if ex.head === :kw                            # f(name = value): only `value` is read
            collect_uses!(ex.args[2], out)
        elseif ex.head === :.                         # a.b: only `a` is read
            collect_uses!(ex.args[1], out)
        elseif ex.head === :quote || ex.head === :line
            return
        elseif ex.head === :(=)
            lhs = ex.args[1]
            lhs isa Symbol || collect_uses!(lhs, out) # `A[i] = x` reads A and i
            collect_uses!(ex.args[2], out)
        else
            for a in ex.args; collect_uses!(a, out); end
        end
    end
end

total = 0
for path in TARGETS
    isfile(path) || (println("skip (no such file): $path"); continue)
    top = Meta.parseall(read(path, String))

    # (line, expr) for every top-level statement, in order
    stmts = Tuple{Int,Any}[]
    line = 0
    for a in top.args
        if a isa LineNumberNode
            line = a.line
        else
            push!(stmts, (line, a))
        end
    end

    defs = Dict{Symbol,Int}()
    for (ln, ex) in stmts
        names = Symbol[]; collect_defs!(ex, names)
        for n in names; haskey(defs, n) || (defs[n] = ln); end
    end

    bad = Tuple{Int,Symbol,Int}[]
    for (ln, ex) in stmts
        uses = Symbol[];    collect_uses!(ex, uses)
        bound = Symbol[];   collect_locals!(ex, bound)
        skip = Set(bound)
        for u in unique(uses)
            u in skip && continue
            d = get(defs, u, 0)
            d > ln && push!(bad, (ln, u, d))
        end
    end

    if isempty(bad)
        println("ok   $path")
    else
        global total += length(bad)
        println("FAIL $path")
        src = readlines(path)
        for (ln, name, d) in bad
            println("  line $ln uses `$name`, not defined until line $d")
            1 <= ln <= length(src) && println("      $(strip(src[ln]))")
        end
    end
end

total > 0 && (println("\n$total forward reference(s) -- each is an UndefVarError at job start."); exit(1))
println("\nclean")
