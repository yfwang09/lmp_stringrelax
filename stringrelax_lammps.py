"""Run serial stringrelax using a single LAMMPS Python engine.

Usage
-----
python stringrelax_lammps.py STATE_A STATE_B [--output NEW_DIRECTORY]
python stringrelax_lammps.py STATE_A STATE_B --prepare-only

Inputs
------
LAMMPS atomic endpoint data files with matching IDs, types, and fully periodic
fixed cells. The Fe/O potential and editable defaults are set below.

Outputs
-------
A new run directory contains settings/logs and prepared NEB endpoints.
A full string run also saves continuous path arrays, per-image data files,
fresh energy CSV/plot, pre-cycle history CSV, and convergence JSON.

Notes
-----
Images use unwrapped (natoms, 3) Cartesian arrays sorted by atom ID.
evaluate(R) -> (total_energy_eV, physical_forces); relax(R, relax_step)
-> relaxed_coordinates. Their shared globals are initialized in the main block.
Images are processed sequentially; multiple MPI processes are rejected.
Importing defines callbacks without starting an engine or calculation.
"""

import argparse
from datetime import datetime
from itertools import product
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from lammps import lammps

from stringrelax_util import stringrelax

# Settings: all lengths in Angstrom, energies in eV, forces in eV/Angstrom.
root = Path(__file__).resolve().parent
potential = root / "Fe-O.tersoff.zbl"
nchain, nstep, relax_step = 8, 50, 5
min_etol, min_ftol, min_maxeval, dmax = 0.0, 1e-8, 1000, 0.001
path_tol = None                    # Optional max image displacement between cycles.
output_parent = root / "outputs"


def evaluate(R):
    """Set one image's coordinates and evaluate its energy and forces.

    Parameters
    ----------
    R : array_like, shape (natoms, 3)
        Finite, continuous unwrapped Cartesian coordinates in Angstrom.
        Rows must follow the module's sorted atom_ids array, with species
        matching atom_types. The supplied array is not modified.

    Returns
    -------
    energy : float
        Fresh TOTAL potential energy in eV from get_thermo("pe").
        Setup must use units metal and thermo_modify norm no.
    forces : numpy.ndarray, shape (natoms, 3)
        Physical forces in eV/Angstrom, in sorted atom-ID order, with sign
        F = -grad(E). This is an independent copy that remains valid after
        later LAMMPS calls or atom reordering.

    Raises
    ------
    ValueError
        If coordinates have an invalid shape/non-finite values, or current
        engine atom IDs or types differ from the stored endpoint identities.
    RuntimeError
        If the engine uses multiple MPI processes, does not own every atom,
        or returns non-finite energy or forces. LAMMPS errors also propagate.

    Notes
    -----
    Requires initialized module globals engine, atom_ids, atom_types, cell,
    and origin. cell contains three lattice vectors as ROWS in Angstrom;
    origin is the Cartesian box origin. The same single, fully periodic
    engine is reused for every image.

    This call mutates engine coordinates and image flags, and refreshes
    neighbors, ghost coordinates, energy, and forces using run 0. It does not
    integrate or minimize. Coordinates are wrapped for transfer and image
    flags reset to zero; the input's whole-cell offsets are restored by relax
    when continuous relaxed positions are needed. Engine storage can be
    reordered by run 0, so atom-ID order is resolved again afterwards.
    Engine logging follows the active run-specific LAMMPS log settings.

    Examples
    --------
    After the driver has initialized its engine and identity/cell globals::

        energy_eV, physical_forces = evaluate(R)
        positive_gradient = -physical_forces  # Analytic F_func convention.
    """
    R = np.asarray(R, dtype=float)
    natoms = len(atom_ids)
    if R.shape != (natoms, 3) or not np.isfinite(R).all():
        raise ValueError(f"Coordinates must be finite with shape ({natoms}, 3)")
    if engine.extract_setting("world_size") != 1 or engine.extract_setting("nlocal") != natoms:
        raise RuntimeError("This driver requires one MPI process owning every atom")
    # NumPy extraction exposes engine memory in its current storage order.
    # Match owned atoms to stable IDs instead of assuming storage is unchanged.
    order = np.argsort(engine.numpy.extract_atom("id")[:natoms])
    if (not np.array_equal(engine.numpy.extract_atom("id")[order], atom_ids)
            or not np.array_equal(engine.numpy.extract_atom("type")[order], atom_types)):
        raise ValueError("Engine atom IDs or types no longer match the endpoints")

    # Wrap fractional coordinates into the triclinic box and reset packed
    # image flags so this image cannot inherit another image's lattice offsets.
    fractional = (R - origin) @ np.linalg.inv(cell)
    wrapped = origin + (fractional - np.floor(fractional)) @ cell
    engine.numpy.extract_atom("x")[order] = wrapped
    engine.numpy.extract_atom("image")[:natoms] = engine.encode_image_flags(0, 0, 0)
    # run 0 refreshes neighbors, ghosts, forces, and energy without dynamics.
    # It can reorder atoms; resolve IDs again before copying physical forces.
    engine.command("run 0")
    order = np.argsort(engine.numpy.extract_atom("id")[:natoms])
    energy = float(engine.get_thermo("pe"))
    forces = engine.numpy.extract_atom("f")[order].copy()
    if not np.isfinite(energy) or not np.isfinite(forces).all():
        raise RuntimeError("LAMMPS returned non-finite energy or forces")
    return energy, forces


def relax(R, relax_step):
    """Minimize one interior image with a limited LAMMPS CG calculation.

    Parameters
    ----------
    R : array_like, shape (natoms, 3)
        Finite, continuous unwrapped Cartesian coordinates in Angstrom,
        ordered by the initialized module's sorted atom IDs. Not modified.
    relax_step : int
        Nonnegative maximum number of CG iterations for this call. Zero still
        refreshes the engine through evaluate and returns a copy of R.

    Returns
    -------
    relaxed_coordinates : numpy.ndarray, shape (natoms, 3)
        Independent, finite Cartesian coordinate array in Angstrom and sorted
        atom-ID order. Whole-cell offsets from R and any boundary crossings
        during minimization are retained, preserving a continuous image.
        Coordinates equal a copy of R when relax_step is zero.

    Raises
    ------
    ValueError
        If the iteration count is not a nonnegative integer, or evaluate
        rejects the coordinates or current atom identities.
    RuntimeError
        If evaluation fails or minimization produces non-finite positions.
        LAMMPS minimization errors also propagate.

    Notes
    -----
    Requires the same globals as evaluate, plus min_etol, min_ftol, and
    min_maxeval. The engine must already have min_style cg,
    min_modify dmax ..., and compute unwrapped all property/atom xu yu zu
    configured.

    minimize etol ftol maxiter maxeval can stop early on energy/force tolerance
    or a minimizer stopping condition; the iteration budget does not guarantee
    convergence. min_etol is a relative energy tolerance, min_ftol is a
    force-norm tolerance in eV/Angstrom, and min_maxeval caps force/energy
    evaluations. CG starts afresh for each image/cycle. dmax limits CG
    line-search movement in Angstrom; it is not the prototype's gradient-
    descent multiplier step_size.

    Mutates the shared engine, allowing all atoms to move in the fixed cell.
    This is a deliberate replacement of fixed-step analytic descent; it adds
    neither NEB springs nor a perpendicular-force projection. Only interior
    images are passed here by stringrelax; that utility fixes endpoints.

    Examples
    --------
    After initializing the driver's shared engine and globals::

        relaxed_R = relax(R, relax_step=5)
        energy_eV, forces = evaluate(relaxed_R)  # Fresh physical evaluation.
    """
    if isinstance(relax_step, (bool, np.bool_)) or not isinstance(relax_step, (int, np.integer)) or relax_step < 0:
        raise ValueError("relax_step must be a nonnegative integer")
    # Load this image into the shared engine and refresh its physical state.
    evaluate(R)
    if relax_step == 0:
        return np.array(R, dtype=float, copy=True)
    # Keep the input lattice offsets separately because evaluate() wraps R.
    # minimize arguments: relative energy tolerance, force-norm tolerance,
    # maximum CG iterations, and maximum force/energy evaluations.
    offsets = np.floor((R - origin) @ np.linalg.inv(cell)) @ cell
    engine.command(f"minimize {min_etol} {min_ftol} {relax_step} {min_maxeval}")
    order = np.argsort(engine.numpy.extract_atom("id")[:len(atom_ids)])
    # extract_compute("unwrapped", 1, 2): 1 means per-atom data; 2 means
    # a 2D array. xu/yu/zu include crossings during CG. Copy in ID order,
    # then add the input lattice offsets to recover continuous coordinates.
    result = engine.numpy.extract_compute("unwrapped", 1, 2)[order].copy() + offsets
    if not np.isfinite(result).all():
        raise RuntimeError("LAMMPS minimization returned non-finite coordinates")
    return result


if __name__ == "__main__":
    # Read endpoint paths and optional overrides; other controls are above.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state_A", type=Path, help="Initial LAMMPS atomic data file")
    parser.add_argument("state_B", type=Path, help="Final LAMMPS atomic data file")
    parser.add_argument("--output", type=Path, help="New output directory; must not exist")
    parser.add_argument("--nchain", type=int, default=nchain)
    parser.add_argument("--nstep", type=int, default=nstep)
    parser.add_argument("--relax-step", type=int, default=relax_step)
    parser.add_argument("--path-tol", type=float, default=path_tol)
    parser.add_argument("--prepare-only", action="store_true", help="Prepare matched NEB inputs without running the string")
    args = parser.parse_args()
    state_A, state_B = args.state_A.resolve(), args.state_B.resolve()
    nchain, nstep, relax_step, path_tol = args.nchain, args.nstep, args.relax_step, args.path_tol
    if nchain < 3 or nstep < 0 or relax_step < 0:
        parser.error("Require nchain >= 3, nstep >= 0, and relax-step >= 0")
    for path in (state_A, state_B, potential):
        if not path.is_file():
            raise FileNotFoundError(path)
    if (not np.isfinite([min_etol, min_ftol, dmax]).all()
            or min_etol < 0 or min_ftol < 0 or dmax <= 0
            or not isinstance(min_maxeval, int) or min_maxeval < 1):
        raise ValueError("Invalid CG tolerances, dmax, or evaluation limit")

    # Start without a log, reject MPI, then create a new directory exclusively.
    # A failed calculation leaves its log and inputs available for inspection.
    with lammps(cmdargs=["-log", "none", "-screen", "none"]) as engine:
        if engine.extract_setting("world_size") != 1:
            raise RuntimeError("Run python stringrelax_lammps.py without mpiexec")
        output = args.output.resolve() if args.output else output_parent / datetime.now().astimezone().strftime("stringrelax_%Y%m%d_%H%M%S_%f")
        output.mkdir(parents=True, exist_ok=False)
        log_name = "preparation.log" if args.prepare_only else "lammps.log"
        engine.command(f'log "{output / log_name}"')

        # Read both endpoints using the same engine. The clear command resets
        # the model without starting another engine or changing source files.
        endpoints = []
        for path in (state_A, state_B):
            engine.commands_string(f'''# Reset the previous model while retaining this LAMMPS instance.
clear

# Use eV/Angstrom/ps units, atomic particles, ID lookup, and periodic faces.
units metal
atom_style atomic
atom_modify map array
boundary p p p
read_data "{path}"

# Load the Fe/O interaction; type 1 maps to Fe and type 2 maps to O.
pair_style tersoff/zbl
pair_coeff * * "{potential}" Fe O
# Use a 2-Angstrom neighbor skin and check for rebuilds every step.
neighbor 2.0 bin
neigh_modify every 1 delay 0 check yes

# Request total PE and force diagnostics without per-atom normalization.
# The named compute supplies continuous unwrapped Cartesian positions.
thermo_style custom step pe fmax fnorm
thermo_modify norm no
compute unwrapped all property/atom xu yu zu

# Select short CG relaxation and its movement limit; run 0 initializes
# coordinates, neighbors, and diagnostics without performing relaxation.
min_style cg
min_modify dmax {dmax}
run 0
''')
            # Copy owned atom/compute arrays before the next engine command.
            # extract_box returns lo, hi, xy, yz, xz, periodic flags, change flag.
            # Build ROW vectors: Cartesian = origin + fractional @ cell.
            count = engine.get_natoms()
            ids = engine.numpy.extract_atom("id")[:count].copy()
            order = np.argsort(ids)
            types = engine.numpy.extract_atom("type")[order].copy()
            coords = engine.numpy.extract_compute("unwrapped", 1, 2)[order].copy()
            lo, hi, xy, yz, xz, periodic, _ = engine.extract_box()
            box = np.array([[hi[0] - lo[0], 0, 0],
                            [xy, hi[1] - lo[1], 0],
                            [xz, yz, hi[2] - lo[2]]])
            if count == 0 or len(np.unique(ids)) != count or np.any(ids <= 0):
                raise ValueError("Endpoint IDs must be positive and unique, with at least one atom")
            if not np.isfinite(coords).all() or not np.isfinite(box).all() or np.any(np.diag(box) <= 0):
                raise ValueError("Invalid endpoint coordinates or simulation cell")
            if periodic != [1, 1, 1] or not np.isin(types, [1, 2]).all():
                raise ValueError("This example requires periodic boundaries and types 1=Fe, 2=O")
            if endpoints:
                if (not np.array_equal(ids[order], atom_ids) or not np.array_equal(types, atom_types)
                        or not np.allclose(box, cell, rtol=0, atol=1e-10)
                        or not np.allclose(lo, origin, rtol=0, atol=1e-10)):
                    raise ValueError("Endpoints must have identical atom IDs, types, and fixed cells")
            else:
                atom_ids, atom_types = ids[order].copy(), types
                origin, cell = np.array(lo), box
            endpoints.append(coords)

        # Choose the true nearest lattice image of each B atom relative to A.
        # Enumerate all possible improvements within the best candidate's
        # Cartesian length, using the cell's smallest singular value as a bound.
        # This handles skew cells without assuming component-wise rounding is
        # always the Cartesian minimum image. It is done only once at startup.
        rA, raw_rB = endpoints
        fractional_delta = (raw_rB - rA) @ np.linalg.inv(cell)
        reduced = fractional_delta - np.rint(fractional_delta)
        displacement = reduced @ cell
        best_squared = np.sum(displacement**2, axis=1)
        radius = int(np.ceil(np.linalg.norm(reduced, axis=1).max()
                             + np.sqrt(best_squared.max()) / np.linalg.svd(cell, compute_uv=False)[-1]))
        for offset in product(range(-radius, radius + 1), repeat=3):
            candidate = (reduced - offset) @ cell
            squared = np.sum(candidate**2, axis=1)
            better = squared < best_squared
            displacement[better], best_squared[better] = candidate[better], squared[better]
        rB = rA + displacement
        internal_rms = float(np.sqrt(np.mean(np.sum((displacement - displacement.mean(axis=0))**2, axis=1))))
        print(f"Endpoint internal RMS displacement after removing translation: {internal_rms:.6g} Angstrom")
        if internal_rms < 1e-8:
            print("Workflow test: endpoints have the same internal structure; no transition barrier can be inferred.")

        # Save input provenance and numerical conventions alongside results.
        settings = {
            "state_A": str(state_A), "state_B": str(state_B), "potential": str(potential),
            "potential_sha256": hashlib.sha256(potential.read_bytes()).hexdigest(),
            "endpoint_sha256": [hashlib.sha256(path.read_bytes()).hexdigest() for path in (state_A, state_B)],
            "lammps_version": engine.version(), "units": "metal", "atom_types": {"1": "Fe", "2": "O"},
            "nchain": nchain, "nstep": nstep, "relax_step": relax_step, "min_style": "cg",
            "min_etol": min_etol, "min_ftol": min_ftol, "min_maxeval": min_maxeval, "dmax": dmax,
            "path_tol": path_tol, "boundary": "p p p", "fixed_cell": True, "mobile_atoms": "all",
            "cell_rows_Angstrom": cell.tolist(), "origin_Angstrom": origin.tolist(),
            "coordinates": "continuous unwrapped Cartesian, sorted by atom ID; nearest periodic endpoint images",
            "arc_length": "Euclidean norm over all 3N coordinates in Angstrom",
            "history_timing": "before each relax/redistribute cycle; final profile evaluated separately",
            "endpoint_internal_rms_Angstrom": internal_rms,
            "workflow_test_only": internal_rms < 1e-8,
            "method": "initialization" if args.prepare_only else "stringrelax",
        }
        (output / "settings.json").write_text(json.dumps(settings, indent=2, allow_nan=False) + "\n")

        # The standalone NEB runner uses the same validated geometry and nearest
        # periodic endpoint representatives, without depending on a string run.
        evaluate(rA)
        # write_data saves wrapped A geometry/cell. nocoeff omits pair
        # coefficients because the native NEB input reloads the potential.
        engine.command(f'write_data "{output / "state_A.lammps"}" nocoeff')
        # Native "neb ... final FILE" reads an atom count then ID/x/y/z rows.
        # Use the same nearest-periodic B representative as the string.
        with (output / "neb.final").open("x") as final_file:
            final_file.write(f"# Prepared from {state_B}\n{len(atom_ids)}\n")
            np.savetxt(final_file, np.column_stack((atom_ids, rB)), fmt=["%d", "%.17g", "%.17g", "%.17g"])
        np.savez(output / "endpoints.npz", atom_ids=atom_ids, atom_types=atom_types,
                 endpoints_input=np.array(endpoints), endpoints_used=np.array([rA, rB]), cell=cell, origin=origin)
        if args.prepare_only:
            print(f"Prepared endpoints: {output}")
            raise SystemExit(0)

        # The same loop as the 2D example, with a LAMMPS relaxation callback.
        Rp, Eclist, info = stringrelax(
            rA, rB, E_func=lambda R: evaluate(R)[0], relax_func=relax,
            nchain=nchain, nstep=nstep, relax_step=relax_step,
            path_tol=path_tol, verbose=True)

        # Re-evaluate the actual final path, including both fixed endpoints.
        energies, max_forces = np.empty(nchain), np.empty(nchain)
        for i, R in enumerate(Rp):
            try:
                energies[i], forces = evaluate(R)
                max_forces[i] = np.linalg.norm(forces, axis=1).max()
                # Save fresh geometry for independent reloads; path.npz
                # below retains the continuous unwrapped coordinates.
                engine.command(f'write_data "{output / f"image_{i:03d}.lammps"}" nocoeff')
            except Exception as exc:
                raise RuntimeError(f"Final evaluation/output failed at image {i}: {exc}") from exc
        arc = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(Rp, axis=0).reshape(nchain - 1, -1), axis=1))))
        relative = energies - energies[0]
        np.save(output / "path.npy", Rp)
        np.savez(output / "path.npz", coordinates=Rp, atom_ids=atom_ids, atom_types=atom_types,
                 cell=cell, origin=origin, endpoints_input=np.array(endpoints), endpoints_used=np.array([rA, rB]))
        np.savetxt(output / "energy_profile.csv", np.column_stack((np.arange(nchain), arc, energies, relative, max_forces)),
                   delimiter=",", header="image,arc_Angstrom,energy_eV,energy_minus_EA_eV,max_atom_force_eV_per_Angstrom", comments="")
        history_header = "cycle," + ",".join(f"image_{i}_minus_EA_eV" for i in range(nchain)) + ",EA_eV,initial_EA_eV"
        np.savetxt(output / "history.csv", Eclist, delimiter=",", header=history_header, comments="")
        info.update({"energy_reference": "fixed state A, total potential energy", "EA_eV": float(energies[0]),
                     "profile_timing": "after the final redistribution", "final_arc_length_Angstrom": float(arc[-1]),
                     "forward_profile_max_minus_EA_eV": float(relative.max()),
                     "final_max_interior_atom_force_eV_per_Angstrom": float(max_forces[1:-1].max()) if nchain > 2 else None,
                     "force_converged": None, "workflow_test_only": settings["workflow_test_only"]})
        (output / "convergence.json").write_text(json.dumps(info, indent=2, allow_nan=False) + "\n")

    # Render a headless profile after the engine context has closed.
    fig, ax = plt.subplots()
    ax.plot(arc, relative, "o-")
    ax.set_xlabel("Configuration-space arc length (Angstrom)")
    ax.set_ylabel(r"Total potential energy $E-E_A$ (eV)")
    if settings["workflow_test_only"]:
        ax.set_title("Workflow test: endpoints differ only by translation")
    fig.tight_layout()
    fig.savefig(output / "energy_profile.png", dpi=160)
    plt.close(fig)
    print(f"Finished {info['completed_cycles']} cycles: {info['stop_reason']}")
    print(f"Results: {output}")
