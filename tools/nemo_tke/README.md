# NEMO TKE environment

A separate Julia environment for `tools/single_column_bsose.jl` with NEMO's TKE closure (`CLOSURES=nemo,nemo_glorys,...`).

## What it is

- **The closure:** `NEMOTKEVerticalDiffusivity` lives on Oceananigans' `fjp/nemo-tke` branch (0.118.2 plus NEMO TKE). It's not a PR yet.
- **The pin:** `Project.toml` fixes the branch to commit `5fde76f`, so a later push to the branch changes nothing here. `Manifest.toml` is resolved on PACE at the first build and is gitignored.
- **Why separate:** the runs stay on Oceananigans 0.117.2 under NumericalEarth 768ed1d. This environment never touches them.
- **No CUDA:** the CPU single column loads no CUDA (it is only a weak dependency of Oceananigans), so building this environment on a CPU node can't spoil the GPU cache.

## Setup (once, on a compute node)

The login node can't build CairoMakie (4 GiB limit). The `ekman` mode of `run_single_column` instantiates the environment at the start of its job. By hand:

```bash
module purge; module load julia/1.12.5; unset LD_LIBRARY_PATH
julia --project=tools/nemo_tke -e 'using Pkg; Pkg.instantiate(); using Oceananigans, NCDatasets, CairoMakie'
```

## Use

```bash
CLOSURES=catke,nemo,nemo_glorys BSOSE_DIR=/storage/scratch1/1/vnguyen480/SOWinWaRP/data \
    julia --project=tools/nemo_tke tools/single_column_bsose.jl
```

The CATKE schemes run here too, on Oceananigans 0.118.2. Run `catke` in both environments to see what the version change alone does.

## The NEMO settings

| Name | Settings |
|---|---|
| `nemo` | NEMO 4 reference namelist: surface TKE 67.83 u★² (wave breaking), νᵇ 1.2e-4, κᵇ 1.2e-5 |
| `nemo_glorys` | GLORYS12's NEMO 3.1 settings: surface TKE 3.75 u★² (no waves), surface length 0.4 m, Riᶜ 0.2, νᵇ 1e-4, κᵇ 1e-5 |
| `nemo[_glorys]_ebb<x>_nub<x>_kb<x>` | either one, with the surface coefficient, background viscosity or diffusivity changed |

NEMO TKE works only as a single closure, so it carries its own backgrounds rather than the runs' νz 1e-3 and κz 1e-5. Use `nub1e-3_kb1e-5` to match the runs.
