# update_numericalearth.jl -- move this project to the newest commit of NumericalEarth's nested-ocean
# branch, pinned to that exact commit, with a way back.
#
#     julia --project=. tools/update_numericalearth.jl            # update, if the branch has moved
#     julia --project=. tools/update_numericalearth.jl check      # only say whether it has moved
#     julia --project=. tools/update_numericalearth.jl rollback   # back to the Manifest before the last update
#
# RUN IT ON A GPU NODE (salloc ... --gres=gpu:V100:1). It precompiles, and precompiling on a CPU node
# breaks CUDA for every later GPU job in this checkout; it refuses to update without a GPU.
#
# What an update does:
#   1. asks GitHub for the branch's head commit (git ls-remote; nothing is cloned);
#   2. stops if the project already has it;
#   3. copies Manifest.toml to Manifest.toml.<old commit>.bak (gitignored) -- `rollback` restores the newest;
#   4. Pkg.add(url = ..., rev = <the commit>): the exact commit, not the branch name, so the Manifest says
#      what every later run used. Pkg moves Oceananigans, Breeze, ... only as far as the branch requires;
#   5. precompiles, then loads CUDA, Oceananigans and NumericalEarth in a fresh Julia and checks the GPU works.
# It changes Manifest.toml (and Project.toml the first time). Commit those yourself once a SMOKE run is good.
#
#     NE_URL     the repository   (default https://github.com/NumericalEarth/NumericalEarth.jl)
#     NE_BRANCH  the branch       (default ss/nested-ocean-model)

using Pkg, Dates, UUIDs

const NE_UUID   = UUID("904d977b-046a-4731-8b86-9235c0d1ef02")
const NE_URL    = get(ENV, "NE_URL", "https://github.com/NumericalEarth/NumericalEarth.jl")
const NE_BRANCH = get(ENV, "NE_BRANCH", "ss/nested-ocean-model")
const PROJECT   = dirname(Base.active_project())
const MANIFEST  = joinpath(PROJECT, "Manifest.toml")
const MODE      = isempty(ARGS) ? "update" : ARGS[1]
MODE in ("update", "check", "rollback") || error("usage: julia --project=. tools/update_numericalearth.jl [check|rollback]")
isfile(joinpath(PROJECT, "Project.toml")) && occursin("SOWinWaRP", PROJECT) ||
    @warn "the active project is $PROJECT; run this with --project=. from the SOWinWaRP checkout"

has_gpu() = try success(pipeline(`nvidia-smi -L`; stdout = devnull, stderr = devnull)) catch; false end
require_gpu(what) = has_gpu() ||
    error("no GPU on this node: $what precompiles, which on a CPU node breaks CUDA for later GPU jobs. " *
          "Get a GPU node first (salloc ... --gres=gpu:V100:1).")

"""What the project has now: the commit it is pinned to (or nothing), and a short description."""
function current()
    info = get(Pkg.dependencies(), NE_UUID, nothing)
    info === nothing && return (rev = nothing, label = "not installed")
    rev = info.git_revision                       # the rev it was added with: a commit, a branch, or nothing
    label = "v$(info.version)" * (info.git_source === nothing ? " (registered release)" :
                                  " from $(info.git_source) at $(something(rev, "?"))")
    return (; rev, label)
end

"""The branch's head commit on GitHub."""
function remote_head()
    out = read(`git ls-remote $NE_URL refs/heads/$NE_BRANCH`, String)
    isempty(strip(out)) && error("$NE_URL has no branch $NE_BRANCH")
    return String(first(split(out)))
end

same_commit(a, b) = a !== nothing && length(a) >= 7 && (startswith(b, a) || startswith(a, b))

"""Load the packages the runs use in a fresh Julia, on the GPU."""
function load_test()
    code = """
        using CUDA, Oceananigans, NumericalEarth
        CUDA.functional() || error("CUDA is not functional")
        println("  loads: NumericalEarth ", pkgversion(NumericalEarth), ", Oceananigans ", pkgversion(Oceananigans),
                ", CUDA ", pkgversion(CUDA), " on ", CUDA.name(CUDA.device()))
        """
    run(`$(Base.julia_cmd()) --project=$PROJECT -e $code`)
end

backups() = sort(filter(f -> startswith(f, "Manifest.toml.") && endswith(f, ".bak"), readdir(PROJECT));
                 by = f -> mtime(joinpath(PROJECT, f)))

# ── rollback ──────────────────────────────────────────────────────────────────────────────────
if MODE == "rollback"
    b = backups()
    isempty(b) && error("no Manifest.toml.*.bak in $PROJECT to roll back to")
    require_gpu("rollback")
    cp(joinpath(PROJECT, last(b)), MANIFEST; force = true)
    @info "restored $(last(b)) as Manifest.toml"
    Pkg.instantiate()
    Pkg.precompile()
    load_test()
    @info "rolled back: NumericalEarth $(current().label)"
    exit(0)
end

# ── check / update ────────────────────────────────────────────────────────────────────────────
now_ = current()
head = remote_head()
@info "NumericalEarth: the project has $(now_.label); $NE_BRANCH is at $(head[1:9])"

if same_commit(now_.rev, head)
    @info "already at the head of $NE_BRANCH: nothing to do"
    exit(0)
end
if MODE == "check"
    base = now_.rev === nothing ? "" : "$(NE_URL)/compare/$(now_.rev)...$(head[1:9])"
    @info "an update is available" * (isempty(base) ? "" : "; what changed: $base")
    exit(0)
end

require_gpu("an update")
old = something(now_.rev, "v" * string(get(Pkg.dependencies(), NE_UUID, (; version = "none")).version))
backup = joinpath(PROJECT, "Manifest.toml.$(first(old, 9)).bak")
isfile(MANIFEST) && cp(MANIFEST, backup; force = true)
@info "backed up the Manifest as $(basename(backup)); `rollback` restores it"

t₀ = time()
try
    Pkg.add(PackageSpec(url = NE_URL, rev = head))
    Pkg.precompile()
    load_test()
catch err
    @error "the update failed; restoring the Manifest from $(basename(backup))" exception = err
    isfile(backup) && cp(backup, MANIFEST; force = true)
    Pkg.instantiate()
    rethrow()
end

Pkg.status(["NumericalEarth", "Oceananigans", "ClimaSeaIce", "Breeze", "CUDA"])
@info "updated NumericalEarth $(first(old, 9)) -> $(head[1:9]) in $(round((time() - t₀) / 60, digits = 1)) min. " *
      "Next: SMOKE=1 runs of the run scripts, then commit Manifest.toml and Project.toml."
