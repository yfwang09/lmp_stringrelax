# Minimal test commands

Run these in a WSL Bash terminal. The example uses case 25, FCC → BCC.
The short runs below check execution; they do not establish convergence.

## 1. Load the environment

```bash
source /home/yfwang09/miniforge3/etc/profile.d/conda.sh
conda activate md_env
cd /home/yfwang09/Codes/lmp_stringrelax
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
```

Loads the installed Python, LAMMPS, and MPICH environment. The thread settings
use one CPU thread per process.

## 2. Choose state A and state B

```bash
state_a="$PWD/states/minimized.25_FCC.lmp"
state_b="$PWD/states/minimized_25_BCC.lmp"
```

A is the initial state; B is the final state. Replace these paths to test
another pair. The files must have matching atom IDs, types, and cells.

## 3. Run native LAMMPS NEB (`in.neb`)

First prepare the endpoint inputs in a new output directory:

```bash
neb_out="$PWD/outputs/minimal_neb_$(date +%Y%m%d_%H%M%S_%N)"
python stringrelax_lammps.py "$state_a" "$state_b" \
    --prepare-only --output "$neb_out"
```

This validates the endpoints and writes `state_A.lammps` and `neb.final` with
consistent periodic coordinates. It does not run string relaxation.

```bash
mpiexec -n 8 lmp -partition 8x1 -in in.neb \
    -var first "$neb_out/state_A.lammps" \
    -var last "$neb_out/neb.final" \
    -var potential "$PWD/Fe-O.tersoff.zbl" \
    -var out "$neb_out" \
    -var ordinary_steps 100 -var climbing_steps 0 -var report_every 10 \
    -log "$neb_out/log.lammps" -screen "$neb_out/screen"
```

Runs eight NEB images, one MPICH process per image, for up to 100 ordinary
iterations with no climbing iterations. Progress is recorded every 10
iterations. These `-var` arguments override the corresponding input defaults.
Increase `ordinary_steps` for a longer run; keep it divisible by `report_every`.

Results are in `$neb_out`: the combined `log.lammps`, per-image logs and dumps,
and final `replica.0.lammps` through `replica.7.lammps` configurations.

## 4. Run the serial stringrelax example

```bash
string_out="$PWD/outputs/stringrelax_$(date +%Y%m%d_%H%M%S_%N)"
python stringrelax_lammps.py "$state_a" "$state_b" \
    --nchain 8 --nstep 10 --relax-step 5 --output "$string_out"
```

Runs eight images serially for 10 relax/redistribute cycles, with up to five
LAMMPS CG iterations per interior-image relaxation. Both endpoints stay fixed.
Increase `--nstep` for a longer run.

The command prints its new `$string_out` directory. It
contains `energy_profile.png`, `energy_profile.csv`, final image data files,
and settings/convergence diagnostics.

## 5. Plot the final paths and convergence

```bash
python plot_results.py "$neb_out" "$string_out"
```

Reads the existing results without rerunning either calculation. It prints a
new visualization directory containing:

- `results.png` / `results.pdf`: final energy overlay, NEB and string barriers
  versus iteration, and the maximum NEB force versus iteration.
- `neb_path.dump` / `string_path.dump`: the final atomic paths. In OVITO
  Desktop, use **File → Load File** and move the timeline slider through the
  images. Frames are MEP image indices, not simulation time.
- CSV files with the energy profiles and barrier histories.

For runs from an earlier terminal, replace `$neb_out` and `$string_out` with
their actual output-directory paths. No LAMMPS or MPI launch is needed to plot.
