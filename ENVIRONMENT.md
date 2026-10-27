# md_env: software and package information

Installation and verification date: **2026-10-27 (Asia/Tokyo)**.

This document describes the installed WSL environment used to develop the LAMMPS Python implementation of stringrelax. It is a snapshot of a working installation; the version numbers below are the installed versions, not instructions to upgrade automatically.

## Platform and locations

| Item | Installed value |
| --- | --- |
| Platform | Ubuntu 26.04.1 LTS, WSL2, Linux x86_64 |
| Project directory | `/home/yfwang09/Codes/lmp_stringrelax` |
| Previous project directory | `/home/yfwang09/Codes/lmp_string` |
| Miniforge distribution | Miniforge3 26.9.4-1; official installer SHA-256 verified |
| Miniforge directory | `/home/yfwang09/miniforge3` |
| Conda | 26.9.4 |
| Environment name | `md_env` |
| Environment directory | `/home/yfwang09/miniforge3/envs/md_env` |
| Python | 3.12.16 |
| Python executable | `/home/yfwang09/miniforge3/envs/md_env/bin/python` |
| MPI implementation | MPICH 4.3.2 |
| Jupyter kernel | `md_env` (display name: `Python-md_env`) |

Conda uses **conda-forge** with **strict channel priority**. Bash initialization is installed, and automatic activation of the `base` environment is disabled. OVITO and LAMMPS were installed from precompiled PyPI wheels after installing the scientific stack and MPI through Conda. No source compilation was needed. The OVITO desktop application was not installed.

## Activate the environment

In a WSL Bash terminal:

```bash
source /home/yfwang09/miniforge3/etc/profile.d/conda.sh
conda activate md_env
cd /home/yfwang09/Codes/lmp_stringrelax
```

In a newly opened terminal, Conda is already initialized, so `conda activate md_env` is sufficient.

The environment sets:

```text
LD_LIBRARY_PATH=/home/yfwang09/miniforge3/envs/md_env/lib
```

This setting lets the LAMMPS wheel locate `libmpi.so.12` from the same MPICH installation used by `mpi4py` and `mpiexec`. It is scoped to Conda activation of `md_env`; use the activated environment for both Python and command-line runs. Keep MPICH as the MPI runtime for this wheel rather than mixing it with OpenMPI.

## Installed package versions

| Package | Version | Installed through |
| --- | --- | --- |
| NumPy | 2.5.4 | Conda |
| SciPy | 1.18.1 | Conda |
| Matplotlib (`matplotlib-base`) | 3.11.2 | Conda |
| pandas | 3.1.0 | Conda |
| SymPy | 1.14.0 | Conda |
| h5py | 3.16.0 | Conda |
| IPython | 9.17.1 | Conda |
| JupyterLab | 4.6.4 | Conda |
| ipykernel | 7.4.0 | Conda |
| regex | 2026.9.29 | Conda |
| mpi4py | 4.1.2 | Conda, matching MPICH |
| OVITO | 3.17.1 | PyPI wheel |
| LAMMPS Python distribution | 2025.7.22.0.2 | PyPI wheel |
| PySide6-Essentials | 6.12.0 | PyPI, OVITO dependency |
| shiboken6 | 6.12.0 | PyPI, Qt dependency |
| traits | 7.2.0 | PyPI, OVITO dependency |

OVITO 3.17.1 requires NumPy 2 or newer. The old OVITO 3.7.12, NumPy 1.26, and Qt 6.4 compatibility pins are not used in this environment. Qt runtime libraries are required even when OVITO is used only for headless Python analysis.

## LAMMPS version and capabilities

The wheel is the community-maintained, precompiled distribution `lammps==2025.7.22.0.2`. Its upstream source tag is **`stable_22Jul2025`**, and `lammps.version()` returns **`20250722`**. The `.0.2` wheel suffix identifies packaging revisions; it does not select a later upstream LAMMPS stable update.

Use the standard Python interface for the stringrelax implementation:

```python
from lammps import lammps

with lammps(cmdargs=["-log", "none", "-screen", "none"]) as engine:
    print(engine.version())  # 20250722
```

The constructor argument for this release is `cmdargs`. With a configured simulation, energy can be read using `engine.get_thermo("pe")`, and coordinate and force arrays through `engine.numpy.extract_atom("x")` and `engine.numpy.extract_atom("f")`. Recompute after changing coordinates and respect local atom ownership under MPI; see [AGENTS.md](AGENTS.md).

This is a CPU/MPI installation with the `OPENMP` package available. GPU acceleration is not configured. The requested `MANYBODY` (including Tersoff/EAM), `MEAM`, `KSPACE`, `REAXFF`, `QEQ`, `MC`, and `REPLICA` (including NEB) packages are present.

Complete installed LAMMPS package list:

```text
AMOEBA ASPHERE BOCS BODY BPM BROWNIAN CG-DNA CG-SPICA CLASS2 COLLOID
COLVARS COMPRESS CORESHELL DIELECTRIC DIFFRACTION DIPOLE DPD-BASIC
DPD-MESO DPD-REACT DPD-SMOOTH DRUDE EFF ELECTRODE EXTRA-COMMAND
EXTRA-COMPUTE EXTRA-DUMP EXTRA-FIX EXTRA-MOLECULE EXTRA-PAIR FEP
GRANULAR INTERLAYER KSPACE LEPTON MACHDYN MANYBODY MC MEAM MESONT MISC
ML-IAP ML-POD ML-SNAP ML-UF3 MOFFF MOLECULE OPENMP OPT ORIENT PERI
PHONON PLUGIN POEMS QEQ REACTION REAXFF REPLICA RHEO RIGID SHOCK SPH
SPIN SRD TALLY UEF VORONOI YAFF
```

The installed Python package directory contains bundled potential files under:

```text
/home/yfwang09/miniforge3/envs/md_env/lib/python3.12/site-packages/lammps/share/lammps/potentials
```

The existence of a potential file is not validation of its suitability for FeO reduction or the intended transition.

### Python interface limitations

The standard `lammps` class can drive simulations and obtain energies and forces. The compiled **`PYTHON` package is absent**: calling embedded Python from LAMMPS input commands is unavailable, and the obsolete `PyLammps` wrapper cannot automatically capture run thermo output. Basic `PyLammps` calls were tested, but new work should use the standard `lammps` class and explicit data access.

OVITO emits a generic advisory when its PyPI wheel is used inside a Conda environment. Imports and headless analysis passed in this installation. Keep the advisory distinct from an actual import or runtime failure; do not assume every future Qt error is harmless.

## Quick checks and launch commands

Run these after activation:

```bash
python --version
python -m pip check
lmp -h
```

Check the installed versions and MPI vendor:

```bash
python - <<'PY'
import importlib.metadata as metadata
from mpi4py import MPI
from lammps import lammps

for package in ("numpy", "scipy", "ovito", "lammps", "mpi4py"):
    print(package, metadata.version(package))
print("MPI:", MPI.get_vendor())
with lammps(cmdargs=["-log", "none", "-screen", "none"]) as engine:
    assert engine.version() == 20250722
    print("LAMMPS:", engine.version())
PY
```

Check a two-process MPI launch:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 mpiexec -n 2 python -c 'from mpi4py import MPI; c = MPI.COMM_WORLD; print("rank", c.rank, "of", c.size, flush=True)'
```

Check OVITO without a display server:

```bash
env -u DISPLAY -u WAYLAND_DISPLAY QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=1 python -c 'import ovito; print(ovito.version_string)'
```

For headless Matplotlib scripts, set `MPLBACKEND=Agg`. Import `ovito` before importing PySide6 on minimal Linux systems. Headless data analysis was verified; image rendering is a separate capability and may require a Vulkan driver or a software renderer.

Start notebooks with `jupyter lab` after activation, and select the `Python-md_env` kernel.

## Existing verification and reproducibility records

The installation verification passed for:

- Headless OVITO common-neighbor analysis of 256 FCC atoms.
- Standard LAMMPS and basic PyLammps calls; NumPy coordinate/force access and energy changes after a coordinate update.
- Short Tersoff and ReaxFF calculations, including charge equilibration.
- Monte Carlo atom swaps (20 attempted and 20 accepted in the smoke test).
- Two-process MPI runs through both Python and the LAMMPS executable.
- Four-replica NEB execution and reference MinEngMesh stringrelax dependency imports.
- `pip check` and fresh Bash activation.

These are installation smoke tests, not validation of the future stringrelax implementation or a FeO interatomic potential.

The existing records are in:

```text
/mnt/c/Users/yfwang09/OneDrive - OIST/Documents/research-notes/outputs/md_env
```

| File | Purpose |
| --- | --- |
| `environment.yml` | Resolved Conda packages, pip packages, and environment variables |
| `conda-requested.yml` | Originally requested Conda package specifications |
| `conda-linux-64-explicit.txt` | Exact Linux Conda package builds and download locations |
| `pip-requirements.txt` | Pins for the five packages installed through pip |
| `pip-freeze.txt` | Full pip-visible package snapshot, including Conda-installed distributions |
| `setup-metadata.json` | Installer checksum, paths, configuration, and limitations |
| `verification-results.json` | Detailed smoke-test results |
| `verify_md_env.py` | Installation verification script; its reference-prototype check targets the research-notes copy |
| `mpi-*.log`, `neb-*.log` | MPI and NEB verification logs |

Use the Conda explicit list together with `pip-requirements.txt` when reconstructing the exact environment. The full pip freeze includes Conda-managed distributions and should not be used to replace the Conda installation wholesale. The exported prefix and `LD_LIBRARY_PATH` are specific to this WSL user and must be adjusted for another installation location.

The reference verification script can be rerun as follows; it updates its reports in the records directory:

```bash
MD_ENV_RECORDS='/mnt/c/Users/yfwang09/OneDrive - OIST/Documents/research-notes/outputs/md_env'
env -u DISPLAY -u WAYLAND_DISPLAY QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python "$MD_ENV_RECORDS/verify_md_env.py"
```

## Upstream references

- [LAMMPS wheel packaging release](https://github.com/njzjz/lammps-wheel/releases/tag/v2025.7.22.0.2)
- [LAMMPS original 22Jul2025 release](https://github.com/lammps/lammps/releases/tag/stable_22Jul2025)
- [OVITO 3.17.1 Python distribution](https://pypi.org/project/ovito/3.17.1/)
- [OVITO installation documentation](https://www.ovito.org/manual/python/introduction/getting_started/installation.html)

The local package snapshot above is the authority for this installation; current upstream documentation may describe newer releases.
