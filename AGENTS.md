# Agent instructions: LAMMPS stringrelax implementation

These instructions apply to `/home/yfwang09/Codes/lmp_stringrelax` and its subdirectories.

## Objective

Implement a Python driver for the existing **stringrelax algorithm**, using LAMMPS to evaluate the potential energy and atomic forces of each image. The broader research context is the FeO/Fe phase-transition project discussed with Alex. A specific FeO potential, physical transition, and endpoint structures have not yet been selected in this repository.

Use `from lammps import lammps` to control the installed engine. LAMMPS NEB may be used as a comparison calculation; it is not a replacement for implementing the prototype's relaxation-and-redistribution algorithm.

This file defines the development task. Its creation does not itself implement the LAMMPS driver.

## Environment

Read [ENVIRONMENT.md](ENVIRONMENT.md) before development or execution. Use the existing WSL `md_env` environment:

```bash
source /home/yfwang09/miniforge3/etc/profile.d/conda.sh
conda activate md_env
cd /home/yfwang09/Codes/lmp_stringrelax
```

The installed engine is LAMMPS 22Jul2025, distributed as the precompiled `2025.7.22.0.2` wheel. Use its `cmdargs` constructor keyword. The environment supplies MPICH and the required library path. Python control and energy/force access work even though the embedded LAMMPS `PYTHON` package is absent. Do not depend on PyLammps automatic thermo capture.

Use headless plotting/analysis for automated checks. Keep CPU/MPI tooling consistent with the installed MPICH runtime. Assess evidence of a missing capability before changing the installed environment, and update the environment documentation if changes are needed.

## Reference files and baseline behavior

- `stringrelax_util.py`: `stringrelax`, `reparamChain`, and the analytic potential functions.
- `stringrelax_2d_PEL.py`: the analytic two-dimensional demonstration, with both energy-based and gradient-based usage.

Preserve these original files as the comparison baseline. Add the LAMMPS implementation separately and document any deliberate correction or numerical change. The files also include mesh-related utilities; implementing the triangular mesh algorithm is outside the stringrelax task unless separately requested.

The baseline algorithm is:

1. Linearly interpolate `nchain` images between the endpoint configurations.
2. Keep both endpoints fixed.
3. Relax each interior image for `relax_step` gradient-descent steps.
4. Redistribute the images uniformly by cumulative Euclidean arc length in the full configuration space.
5. Repeat the relaxation and redistribution cycle for the requested iterations, recording energies and reporting convergence.

The current prototype uses full-gradient relaxation followed by redistribution. Do not silently add NEB springs, a perpendicular-force projection, or a different optimization algorithm and claim equivalence to the baseline.

## Important prototype pitfalls

- **Force sign:** despite its name, the prototype's `F_func` supplies the positive gradient `grad(E)`, which is subtracted during relaxation. LAMMPS supplies physical forces `F = -grad(E)`. A descent step using those forces is `R_new = R + step_size * F`. Make the backend convention explicit and verify it numerically.
- **Hardcoded gradient fallback:** when only `E_func` is supplied, the prototype calls `approx_fprime(x, Vnew, f_epsilon)` rather than differentiating the supplied `E_func`. A generic port must not inherit that hardcoding. Validate the correction with an energy function other than `Vnew`.
- **History timing:** `Eclist` is recorded before each outer relaxation/redistribution cycle. Its final recorded row is not necessarily the energy of the returned final path. Re-evaluate the final images and label the output timing clearly.
- **Inactive parameter:** `energySpike` is accepted and documented in the prototype but has no implemented behavior. Do not claim adaptive spike handling without implementing and validating it explicitly.
- **Degenerate interpolation:** repeated images can produce duplicate arc-length values, and a zero-length chain cannot be interpolated normally. Handle those cases explicitly with a documented result or a clear error.

## LAMMPS backend requirements

Keep the string algorithm separate from the LAMMPS-specific energy/force evaluator so analytic potentials can still exercise the algorithm independently.

- Represent atomistic images using a documented coordinate convention, normally arrays of shape `(natoms, 3)`. Validate shapes, finite values, atom identities/types, and compatibility of endpoints before relaxation.
- Preserve atom correspondence across images. LAMMPS storage order may change after sorting, migration, or neighbor updates; coordinate and force mappings must use stable atom IDs rather than assuming array indices remain atom identities.
- After updating coordinates, recompute energy and forces for the new configuration, ordinarily with `run 0`. Ensure neighbors, ghost coordinates, and any potential-specific charge equilibration are refreshed correctly. Do not inadvertently minimize or integrate while evaluating an image.
- Be explicit about total potential energy, unit conventions, and LAMMPS thermo normalization. Use consistent energy values across images and record the reference used for relative energies.
- Coordinate/force arrays returned through the NumPy interface may be views into engine memory. Copy data that must survive subsequent engine calls or atom reordering.
- Under MPI, local arrays contain owned/ghost subsets, not an automatically complete, consistently ordered configuration. Gather/scatter by atom ID as needed, and distinguish MPI decomposition of a single image from parallel execution of different images.
- Specify wrapped versus unwrapped coordinates and the periodic displacement convention for interpolation and arc length. Avoid artificial long paths across periodic boundaries. Account explicitly for any fixed atoms or constraints in the update and distance calculations.
- Release engine instances and isolate per-image/per-run logs and output files. Make small validation runs deterministic.

Do not assume an FeO potential, a reaction mechanism, changes in atom number/species, or cell interpolation. Obtain the required scientific inputs when a real atomistic case is introduced. Variable-cell pathways need a separate, explicit choice of degrees of freedom and distance metric.

## Deliverables and validation

Provide a reusable stringrelax implementation, a LAMMPS energy/force adapter, and a small runnable example with documented commands. Expose the image count, relaxation iterations/step size, stopping criteria, and necessary simulation inputs. Write final image coordinates, a fresh energy profile versus arc length, and settings/convergence diagnostics in documented formats. Report non-finite values or failed evaluations clearly rather than silently treating them as convergence.

Establish correctness with small calculations before running a production FeO/Fe system:

- Reproduce the analytic two-dimensional reference behavior with documented tolerances; test the gradient fallback on a different potential.
- Check force signs against finite-difference energy gradients and verify that a sufficiently small descent step lowers energy away from a stationary point.
- Verify fixed endpoints, equal-arc-length redistribution, repeated-image/zero-length handling, and invalid input errors.
- Check that repeated energy/force evaluations of the same atomistic configuration agree, and that coordinate changes trigger fresh evaluations.
- Exercise atom-ID mapping and periodic-boundary cases. Verify serial/MPI consistency if the implementation provides an MPI execution path.
- Demonstrate the complete relax/redistribute loop on a small reproducible LAMMPS system, saving the path and final energy profile with convergence information.

The installation smoke tests described in `ENVIRONMENT.md` establish available software capabilities; they do not establish correctness of the new stringrelax implementation. A short NEB comparison can provide additional evidence once matched endpoints and a suitable example are available.

## Working practices

Keep the original prototype, user changes, and Git history intact. Add implementation code and generated results with clear names, and do not overwrite baseline inputs or expensive results. Keep documentation accurate about what has been implemented and which validations were actually run. The separate MD++-dependent `ta_disl` workflow from MinEngMesh is not an installed dependency or an existing LAMMPS implementation in this repository.
