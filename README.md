# Serial LAMMPS stringrelax

This is the original string's **relax, then redistribute** algorithm with a
LAMMPS Python driver. It linearly initializes images, keeps both endpoints
fixed, briefly relaxes each interior image, and redistributes along cumulative
Euclidean arc length. Images run serially through one LAMMPS instance.

The original scripts in `old_scripts/` are unchanged. State files are supplied
by the user and are never rewritten by either workflow. The triangular mesh
workflow is not part of this implementation.

## Run the example

Read [ENVIRONMENT.md](ENVIRONMENT.md) for the existing installation. No additional
packages or changes to the environment are required.

```bash
source /home/yfwang09/miniforge3/etc/profile.d/conda.sh
conda activate md_env
cd /home/yfwang09/Codes/lmp_stringrelax
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python stringrelax_lammps.py \
    states/minimized.25_FCC.lmp states/minimized_25_BCC.lmp
```

State A and B are required positional arguments. Optional arguments are
`--output NEW_DIRECTORY`, `--nchain`, `--nstep`, `--relax-step`, and `--path-tol`.
The output directory must not already exist. Remaining numerical settings are
at the top of `stringrelax_lammps.py`. Defaults are eight
images, 50 outer cycles, and up to five CG iterations per interior image per
cycle. CG uses `etol=0`, `ftol=1e-8` eV/Angstrom, `maxeval=1000`, and
`dmax=0.01` Angstrom. A relaxation can stop earlier according to LAMMPS's
minimizer stopping criteria; the full minimization summaries are in the run log.

`path_tol=None` runs every outer cycle. Setting a nonnegative tolerance stops
when the largest image displacement between two complete cycles is at most that
tolerance. The norm includes all 3N coordinates. This reports **path stability**,
not vanishing physical forces or proof of an MEP. A calculation reaching its
cycle limit is explicitly reported as `max_cycles`.

This driver rejects more than one MPI process. Use it directly with Python;
there is no image parallelization or MPI decomposition path in this version.
For deterministic checks, use one OpenMP and BLAS thread as above.

## Inputs and scientific meaning of this test

The current three pairs are:

| Case | State A | State B |
| --- | --- | --- |
| 15 | `states/minimized.15_FCC.lmp` | `states/minimized_15_BCC.lmp` |
| 25 | `states/minimized.25_FCC.lmp` | `states/minimized_25_BCC.lmp` |
| 32.5 | `states/minimized.32.5_FCC.lmp` | `states/minimized_32.5_BCC.lmp` |

Every pair contains matching IDs/types for 864 Fe atoms and one O atom, and has
a matching fixed triclinic cell. Cells can differ between cases. These updated
FCC/BCC pairs have substantial internal differences, with nearest-image RMS
displacements of approximately 0.374, 0.368, and 0.361 Angstrom after removing
translation. The previous translation-only inputs and their outputs were
replaced/deleted at the user's request.

Both methods use `boundary p p p`, `atom_style atomic`, and all atoms mobile in
interior images. Neither method minimizes either endpoint or relaxes the cell.
The driver still detects translation-only pairs if such inputs are supplied.

The supplied [NIST potential](https://www.ctcms.nist.gov/potentials/Download/2019--Byggmastar-J-Nagel-M-Albe-K-et-al--Fe-O/1/Fe-O.tersoff.zbl)
is saved as `Fe-O.tersoff.zbl`, with its source header and eight Fe/O triplets.
It was retrieved through the web text reader on 2026-10-27 after the direct
download returned HTTP 403. The local text file has SHA-256
`a77f475c706fbe599a5112aaec2e076d20ba4aaaf6b3f1c46e876981abf9d00d`.
Its commands are:

```lammps
units metal
pair_style tersoff/zbl
pair_coeff * * Fe-O.tersoff.zbl Fe O
```

Type 1 maps to Fe and type 2 to O. Energy is total potential energy in eV,
coordinates are in Angstrom, and forces are in eV/Angstrom. `thermo_modify norm
no` explicitly disables per-atom thermo normalization. This potential has no
charge degrees of freedom or charge-equilibration fix.

## Small interfaces and deliberate changes

`stringrelax_util.py` defines only two functions:

```python
reparamChain(Rchain) -> Rchain
stringrelax(rA, rB, E_func=None, F_func=None, ..., relax_func=None,
            path_tol=None) -> Rp, Eclist, info
```

`Rchain` has shape `(nchain, *configuration_shape)`. Atomistic configurations
have shape `(natoms, 3)`. `stringrelax` copies its input endpoints. The returned
`info` dictionary contains `completed_cycles`, `final_path_change`, `stop_reason`,
and `path_changes` for individual cycles. Without `E_func`, `Eclist` is `None`.
The new utility consistently returns three values, including for force-only
analytic use.

The driver defines just two callbacks, which use its single initialized engine:

```python
evaluate(R) -> total_energy, physical_forces
relax(R, relax_step) -> relaxed_coordinates

Rp, Eclist, info = stringrelax(
    rA, rB, E_func=lambda R: evaluate(R)[0], relax_func=relax,
    nchain=nchain, nstep=nstep, relax_step=relax_step, path_tol=path_tol)
```

Importing the driver does not start an engine or calculation. Its example setup
and output code are in a single `if __name__ == "__main__"` block. The callbacks
are usable with a configured engine and the module's `atom_ids`, `atom_types`,
`cell`, and `origin`; there is no backend class or plugin interface.

Important numerical choices:

- LAMMPS relaxation uses a new limited CG minimization for each image/cycle.
  This deliberately replaces fixed-step gradient descent. `dmax` limits motion
  within a line search; it is not the prototype's `step_size` multiplier. No NEB
  springs or force projection are applied to the string.
- The analytic branch retains `F_func = +grad(E)` and subtracts it. LAMMPS
  `evaluate` returns the physical force `F = -grad(E)`; a direct force descent
  step would add `step_size * F`. LAMMPS minimization handles this sign itself.
- With only `E_func`, finite differences use that function, correcting the
  prototype's hardcoded `Vnew`. Matrix configurations are flattened only for
  numerical differentiation, then reshaped.
- `energySpike=None` is retained for compatibility. Non-None values raise an
  error because adaptive spike handling is not implemented.
- Redistribution removes repeated arc-length samples. A wholly zero-length
  chain is returned unchanged. Samples are uniformly spaced on the **input
  polyline's arc**; straight-line distances across corners can differ.
- Invalid inputs and non-finite or failed evaluations raise errors. String
  evaluation failures identify the cycle and image; they are not convergence.

## Atom IDs and periodic coordinates

The Python path is copied, ID-ordered, continuous **unwrapped Cartesian** data.
Both endpoints must have matching positive unique IDs, atom types, and fixed
cells. The driver chooses the nearest periodic image of every B atom relative
to A using the triclinic cell. It enumerates the lattice candidates within a
bound obtained from the smallest cell singular value; component-wise rounding
alone is not assumed to give the Cartesian nearest image. No overall
translation is removed or atom permutation performed. The chosen image of B,
including any lattice offset, is then fixed throughout the string calculation.

`evaluate` wraps coordinates into the cell for engine transfer, resets image
flags, and executes `run 0` to refresh neighbors, ghost coordinates, energy, and
forces. It neither integrates nor minimizes. Owned atoms are mapped by sorted
IDs before updating positions and again after the engine call. Extracted arrays
that must survive subsequent calls are copied.

`relax` retrieves `xu, yu, zu` after minimization and restores the input's whole
cell offsets, preserving continuous motion across periodic faces. The cell is
never changed. Fixed subsets of atoms and variable-cell paths are not exposed
in this first example; supporting those would require an explicit update and
distance convention.

## Results

Every run creates a new `outputs/stringrelax_<timestamp>/` directory exclusively,
or uses a new directory supplied with `--output`.
Source inputs and existing results are not overwritten. It contains:

| File | Contents |
| --- | --- |
| `path.npy` | Final unwrapped coordinates, shape `(nchain, natoms, 3)`, sorted by ID |
| `path.npz` | Coordinates, atom IDs/types, row-vector cell, origin, original endpoints, and periodic endpoint representatives used |
| `image_000.lammps`, etc. | Each final image as LAMMPS atomic data, wrapped in the fixed cell with reset image flags; potential coefficients are supplied separately |
| `energy_profile.csv` | Fresh final image index, cumulative 3N arc length, total energy, energy minus final fixed-A energy, and largest per-atom force norm |
| `energy_profile.png` | Headless final energy profile versus arc length |
| `history.csv` | Energy rows recorded **before** each outer cycle |
| `settings.json` | Inputs and hashes, units, numerical settings, cell, coordinate conventions, engine version, and endpoint diagnosis |
| `convergence.json` | Cycle count, stopping reason, path changes, final energy reference, and force diagnostics |
| `lammps.log` | Engine output, including each short CG minimization's stopping reason |
| `endpoints.npz`, `state_A.lammps`, `neb.final` | Validated endpoint metadata and matching inputs usable by native NEB |

`history.csv` preserves the prototype's columns: zero-based cycle, one relative
energy per image, current A energy, and initial A energy. Its last row is **not**
the final profile. Always use `energy_profile.csv` for the returned path.
The log's `No fixes with time integration` warning during `run 0` is expected:
evaluating an image deliberately does not advance its coordinates.

## Independent native NEB reference

`in.neb` remains separate from the Python string loop. Its runner accepts the
same required endpoint arguments:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python run_neb.py \
    states/minimized.25_FCC.lmp states/minimized_25_BCC.lmp
```

The runner invokes `stringrelax_lammps.py --prepare-only` for endpoint
validation and periodic representatives, then launches native LAMMPS NEB. No
string relaxation is run during NEB preparation. Each run prepares its own
`state_A.lammps` and `neb.final`; there is no global hardcoded endpoint file.

The input uses the same potential, cell, and endpoints, eight replicas,
`fix neb` with spring constant 1 eV/Angstrom², FIRE, a 0.001 ps timestep,
`dmax=0.01` Angstrom, the default FIRE maximum timestep multiplier of 10, and 1000 ordinary
plus 1000 climbing iterations. Both
endpoint replicas have zero velocities and an explicit `fix setforce`, so their
positions stay fixed. Other atoms are mobile. The iteration counts and report
interval are adjustable with `--ordinary-steps`, `--climbing-steps`, and
`--report-every`; each iteration count must be a multiple of the interval.
`--nchain`, `--ftol`, `--timestep`, `--tmax`, and `--output` are also available.
These stepping controls are explicit so numerical stability can be investigated
without changing nudging forces or the potential.

[Native LAMMPS NEB](https://docs.lammps.org/neb.html) requires multiple MPI
partitions and a damped-dynamics minimizer such as FIRE; CG cannot be used for
native NEB. This requirement applies only to the separate reference input.

The runner uses `mpiexec` and `lmp` from the active Python environment, with one
MPICH process per replica, and supplies all required input variables. For
manual launches, prepare inputs first and then use:

```bash
python stringrelax_lammps.py states/minimized.25_FCC.lmp \
    states/minimized_25_BCC.lmp --prepare-only --output outputs/neb_manual
neb_out="$PWD/outputs/neb_manual"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 mpiexec -n 8 lmp \
    -partition 8x1 -in in.neb -var out "$neb_out" \
    -var first "$neb_out/state_A.lammps" -var last "$neb_out/neb.final" \
    -var potential "$PWD/Fe-O.tersoff.zbl" \
    -log "$neb_out/log.lammps" -screen "$neb_out/screen"
```

This generates separate logs, trajectory dumps, and final data files for each
replica. If changing the replica count manually, also pass `-var replicas N`
and update the partition/process counts. `run_neb.py` handles those together.
After NEB it removes artificial forces, executes `run 0`, and exports a fresh
physical energy/force profile. NEB output includes `path.npy`, `path.npz`,
`energy_profile.csv`, `energy_profile.png`, and `convergence.json`, in the same
conventions as the string output. Convergence diagnostics record the final
NEB force norm separately from the exported physical atomic forces.

## Visualize separately launched NEB and string runs

```bash
python plot_results.py NEB_OUTPUT_DIRECTORY STRING_OUTPUT_DIRECTORY
```

This accepts the separate directories printed by the minimal test commands,
including a direct `in.neb` run with zero climbing iterations. It reads existing
final snapshots, energies, and histories; neither simulation is rerun. A new
`outputs/visualization_<timestamp>/` directory contains `results.png`/`.pdf`,
the energy/barrier CSV files, and `neb_path.dump`/`string_path.dump`.

The figure overlays final energy versus normalized arc length, plots the NEB
and string barriers against their respective iteration counts, and shows the
maximum NEB replica force. A string outer cycle differs from a native NEB
iteration. The barrier includes both endpoints. Native history is read from
every image's reported energy, with the final value checked against the fresh
exported energies. Actual NEB iteration limits and force tolerance are read
from the expanded command in `log.lammps.0`, rather than preparation settings.
The zero-iteration climbing setup row is retained in the CSV and omitted from
the iteration plots when no climbing iterations were requested.

For atomic visualization, load either path dump in [OVITO Desktop](https://www.ovito.org/manual/usage/import.html)
and use the timeline slider. Every frame is one final image along the path,
not a relaxation timestep. Atom IDs/types and the fixed triclinic cell are
preserved. The installed OVITO Python module is sufficient to inspect these
files headlessly; an interactive desktop viewer is a separate application.

See [MINIMAL_TEST.md](MINIMAL_TEST.md) for an end-to-end command list, including
the output-directory variables used by this plotting command.

## Run and overlay all three cases

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python compare_cases.py
```

This runs FCC → BCC for cases 15, 25, and 32.5, sequentially by case. Each case
has `string/` and `neb/` subdirectories and `comparison.png`/`comparison.pdf`.
The batch root also has a three-panel `comparison.png`/`comparison.pdf` and
`summary.json`. Optional arguments include `--states DIRECTORY`,
`--output NEW_DIRECTORY`, `--nchain`, `--nstep`, `--relax-step`,
`--ordinary-steps`, `--climbing-steps`, and `--report-every`.

Overlay plots use **normalized configuration-space arc length** for each path.
The energy reference for both methods in a panel is the same fixed state A.
Endpoint energies are checked to agree within 1e-8 eV before plotting. Separate
CSV files retain absolute arc lengths and energies. Unmet NEB force tolerances
are labeled on the plots; finite-iteration profiles are not automatically
converged MEPs. Existing results can be redrawn without calculations:

```bash
python compare_cases.py --plot-only --output outputs/fcc_bcc_comparison
```

An auxiliary `comparison_before_climbing.png`/`.pdf` overlays the final string
with the ordinary NEB stage immediately before climbing. Its NEB points come
from the rounded universe-log energies/reaction coordinates and are exported
as `neb/ordinary_profile_from_log.csv`. The primary `comparison.png` always
uses the freshly evaluated final profiles, including the climbing stage.

### Earlier three-case calculations (2026-10-27)

These records describe the earlier batch, whose outputs were subsequently
removed. Newly launched standalone runs can be visualized with `plot_results.py`
without recreating that three-case directory structure.

All three cases were run with eight images, 50 string cycles (five CG iterations
per interior relaxation), and 1000 ordinary plus 1000 climbing NEB steps. Results
are in `outputs/fcc_bcc_comparison/`. Every exported image was independently
reloaded: all 48 energies, forces, IDs/types, endpoint positions modulo lattice
translations, and arc-length profiles passed verification. The maximum energy
discrepancy was 7.74e-12 eV; `verification.json` contains per-run checks.

**Convergence was not established.** The strings reached the requested cycle
limit, and all NEB runs failed to meet the 1e-8 eV/Angstrom force tolerance.
Cases 15 and 25 developed large energy spikes during climbing; their final
profiles must not be reported as physical barriers. The pre-climbing plot is
included to make the behavior visible. A separate case-15 check with `--tmax 1`
reduced but did not remove the spike, so the published comparison retains the
original default stepping. No alternative potential, spring model, or atom
correspondence was introduced to make the curves agree.

## Plot barrier convergence from saved runs

```bash
python plot_convergence.py outputs/fcc_bcc_comparison
```

This reads existing logs and histories without rerunning a simulation. It writes
`convergence.png`/`.pdf` (NEB barrier, NEB force, and string barrier histories),
`convergence_ordinary_neb.png`/`.pdf` (ordinary-stage detail),
`convergence_summary.json`, and `barrier_history.csv` under each method folder.
The plotted forward barrier is `max(E_i) - E_A` over **all** images, including
both endpoints. Native per-image energies are printed with approximately 1e-4
eV resolution for these runs. String histories include pre-cycle rows and the
freshly evaluated final path at cycle 50. A string outer cycle is distinct from
a single native NEB minimizer iteration.

Both iteration-1000 rows are retained: ordinary NEB ends there, then climbing
starts with the same coordinates and a different force on the climbing image.
In the installed LAMMPS version, `EBF` during climbing follows the selected
climber, which can cease to be the highest image. Thus these plots compute the
barrier from all reported image energies and also save the original `EBF` field.

The current curves show ordinary NEB approaching barriers of approximately
8.315, 1.630, and 0.034 eV in cases 15, 25, and 32.5. Its maximum replica forces
at iteration 1000 are 0.0327, 0.0112, and 0.0895 eV/Angstrom, still above the
requested 1e-8 tolerance. Climbing then begins automatically at that iteration
limit. The large energy and force growth in cases 15 and 25 occurs afterward.

FIRE was selected as one of the native NEB damped-dynamics options; Quickmin is
also supported. Native NEB excludes CG/SD/HFTN because their inner line searches
cannot be synchronized across replicas. See the [22Jul2025 NEB documentation](https://github.com/lammps/lammps/blob/stable_22Jul2025/doc/src/neb.rst)
and [status-output implementation](https://github.com/lammps/lammps/blob/stable_22Jul2025/src/REPLICA/neb.cpp).

## Validation

```bash
# Analytic tests need NumPy, SciPy, Matplotlib, and regex used by the old reference.
python validate_stringrelax.py --analytic-only

# Backend checks, plus independent verification of a completed run:
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python validate_stringrelax.py \
    --results outputs/stringrelax_<timestamp>
```

The validator compares both analytic branches with the original 40-image 2D
example (`atol=1e-8`), differentiates a different quadratic energy with matrix
coordinates, and checks endpoints, history timing, redistribution, repeated and
zero-length chains, stopping, and invalid/non-finite inputs. Backend checks
exercise repeat evaluations, fresh coordinates, copied forces, ID mapping after
an observed storage reorder, finite-difference force signs (`h=1e-5` Angstrom,
`rtol=1e-5`, `atol=1e-4` eV/Angstrom), downhill motion, periodic lattice shifts,
and an observed triclinic face crossing during minimization. The checks
also exercise three full relax/redistribute cycles with a perturbed
endpoint, so the CG callback demonstrably moves interior images. Saved-image
energies are checked by independently reopening every final data file in a
fresh engine (`atol=1e-8` eV). Reports go to a fresh
`outputs/validation_<timestamp>/validation.json` directory.

For backend checks without an existing run, supply endpoint arguments directly:
`python validate_stringrelax.py STATE_A STATE_B`. With `--results`, the validator
reads the endpoint paths from that run's settings. Generated `outputs/`
directories are kept on disk and ignored by Git.

In a restricted runner where MPICH/UCX cannot create its normal local transport,
prefix **serial** commands with `UCX_TLS=self`; checks here used that temporary
process setting and `MPLCONFIGDIR=/tmp/stringrelax-matplotlib`. Do not use
`UCX_TLS=self` for a manual multi-process NEB launch; `run_neb.py` removes that
serial-only setting from its MPI subprocess. No installed environment settings
were changed. The Python string remains serial. Native NEB uses MPI partitions.
