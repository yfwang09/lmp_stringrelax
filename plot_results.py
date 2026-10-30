"""Visualize standalone NEB/string results without rerunning LAMMPS.

Usage
-----
python plot_results.py NEB_DIRECTORY STRING_DIRECTORY [--output NEW_DIRECTORY]

Inputs
------
NEB output from in.neb or run_neb.py and a stringrelax_lammps.py output for
matching endpoints, IDs/types, cell, and potential. Requires native final
dumps/energies/logs, endpoint metadata, and string path/profile/history files.

Outputs
-------
A new directory receives results.png/pdf, profile/history CSV files, summary
JSON, and both paths as NumPy arrays and multi-frame LAMMPS dumps for OVITO.
Each dump frame is a final path image, not a relaxation timestep.
Source results are read without modification.

Notes
-----
NEB iterations and string outer cycles represent different work units.
Zero-iteration climbing setup rows stay in CSV but are omitted from curves.
This script executes at top level and provides no importable function API.
"""

import argparse
from datetime import datetime
from itertools import product
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("neb", type=Path, help="Output directory from in.neb or run_neb.py")
parser.add_argument("string", type=Path, help="Output directory from stringrelax_lammps.py")
parser.add_argument("--output", type=Path, help="New visualization directory; must not exist")
args = parser.parse_args()
neb, string = args.neb.resolve(), args.string.resolve()
output = args.output.resolve() if args.output else root / "outputs" / datetime.now().astimezone().strftime("visualization_%Y%m%d_%H%M%S_%f")
output.mkdir(parents=True, exist_ok=False)

# Load copied coordinates/identities, and require matching physical inputs.
endpoints = np.load(neb / "endpoints.npz")
string_saved = np.load(string / "path.npz")
ids, types, cell, origin = [endpoints[key] for key in ("atom_ids", "atom_types", "cell", "origin")]
for key in ("atom_ids", "atom_types"):
    np.testing.assert_array_equal(endpoints[key], string_saved[key])
for key in ("cell", "origin", "endpoints_used"):
    np.testing.assert_allclose(endpoints[key], string_saved[key], rtol=0, atol=1e-9)
neb_settings = json.loads((neb / "settings.json").read_text())
string_settings = json.loads((string / "settings.json").read_text())
if neb_settings["potential_sha256"] != string_settings["potential_sha256"]:
    raise ValueError("The two calculations used different potential files")
string_path = string_saved["coordinates"]
string_profile = np.loadtxt(string / "energy_profile.csv", delimiter=",", skiprows=1)
string_info = json.loads((string / "convergence.json").read_text())

# Direct in.neb runs already export fresh physical energies and final dumps;
# they do not need to have been postprocessed by run_neb.py.
image_ids = sorted(int(path.name.split(".")[1]) for path in neb.glob("final.*.txt"))
if image_ids != list(range(len(image_ids))) or len(image_ids) < 3:
    raise ValueError("Expected consecutive final.0.txt, final.1.txt, ... for at least three NEB images")
neb_path, neb_energies, neb_forces = [], [], []
for i in image_ids:
    data = np.loadtxt(neb / f"final.{i}.dump", skiprows=9)
    if data.shape != (len(ids), 8) or not np.isfinite(data).all():
        raise ValueError(f"Invalid final NEB image {i}")
    np.testing.assert_array_equal(data[:, 0], ids)
    np.testing.assert_array_equal(data[:, 1], types)
    neb_path.append(data[:, 2:5])
    neb_energies.append(float(np.loadtxt(neb / f"final.{i}.txt")[0]))
    neb_forces.append(float(np.linalg.norm(data[:, 5:8], axis=1).max()))
neb_path, neb_energies = np.array(neb_path), np.array(neb_energies)

# Make adjacent-image coordinates continuous using the same nearest-lattice
# search as the driver. This also gives an arc suitable for the energy plot.
for i in range(1, len(neb_path)):
    fractional = (neb_path[i] - neb_path[i - 1]) @ np.linalg.inv(cell)
    reduced = fractional - np.rint(fractional)
    displacement = reduced @ cell
    best_squared = np.sum(displacement**2, axis=1)
    radius = int(np.ceil(np.linalg.norm(reduced, axis=1).max()
                         + np.sqrt(best_squared.max()) / np.linalg.svd(cell, compute_uv=False)[-1]))
    for offset in product(range(-radius, radius + 1), repeat=3):
        candidate = (reduced - offset) @ cell
        squared = np.sum(candidate**2, axis=1)
        better = squared < best_squared
        displacement[better], best_squared[better] = candidate[better], squared[better]
    neb_path[i] = neb_path[i - 1] + displacement
for path in (neb_path, string_path):
    if not np.isfinite(path).all():
        raise ValueError("Non-finite final path coordinates")
    for image, endpoint in ((0, 0), (-1, 1)):
        fractional = (path[image] - endpoints["endpoints_used"][endpoint]) @ np.linalg.inv(cell)
        np.testing.assert_allclose(fractional, np.rint(fractional), rtol=0, atol=1e-9)
np.testing.assert_allclose(neb_energies[[0, -1]], string_profile[[0, -1], 2], rtol=0, atol=1e-8)
EA = float(string_profile[0, 2])
neb_arc = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(neb_path, axis=0).reshape(len(neb_path) - 1, -1), axis=1))))
np.savetxt(output / "neb_energy_profile.csv", np.column_stack((image_ids, neb_arc, neb_energies, neb_energies - EA, neb_forces)),
           delimiter=",", header="image,arc_Angstrom,energy_eV,energy_minus_EA_eV,max_atom_force_eV_per_Angstrom", comments="")

# Read native convergence, retaining its stage labels and actual image energies.
# During climbing the EBF column follows the chosen climber, so use every PE
# column to obtain the maximum energy along the whole path, including endpoints.
rows, stages, stage = [], [], 0
for line in (neb / "log.lammps").read_text().splitlines():
    if line.startswith("Climbing replica ="):
        stage = 1
    fields = line.split()
    if len(fields) == 9 + 2 * len(neb_path):
        try:
            rows.append([float(value) for value in fields])
            stages.append(stage)
        except ValueError:
            pass
data, stages = np.array(rows), np.array(stages)
if data.ndim != 2 or not np.isfinite(data).all():
    raise ValueError("Missing or non-finite native NEB history")
neb_barrier = np.max(data[:, 10::2] - data[:, 10, None], axis=1)
np.testing.assert_allclose(neb_barrier[-1], np.max(neb_energies - EA), rtol=1e-6, atol=1e-4)

# Extract the expanded command from the per-replica log. Preparation settings
# can differ from the actual -var overrides used by a manual in.neb launch.
controls, native_style = None, None
for line in (neb / "log.lammps.0").read_text().splitlines():
    fields = line.split()
    if len(fields) == 2 and fields[0] == "min_style" and "$" not in fields[1]:
        native_style = fields[1]
    if fields and fields[0] == "neb" and len(fields) >= 7:
        try:
            controls = {"etol": float(fields[1]), "ftol": float(fields[2]),
                        "ordinary_steps": int(fields[3]), "climbing_steps": int(fields[4]),
                        "report_every": int(fields[5])}
        except ValueError:
            pass
if controls is None:
    raise ValueError("Cannot find the actual NEB command in log.lammps.0")
controls["min_style"] = native_style
np.savetxt(output / "neb_barrier_history.csv", np.column_stack((data[:, 0], stages, neb_barrier, data[:, 1], data[:, 6])),
           delimiter=",", header="iteration,stage_0_ordinary_1_climbing,barrier_eV,max_neb_replica_force_eV_per_Angstrom,reported_EBF_eV", comments="")
history = np.loadtxt(string / "history.csv", delimiter=",", skiprows=1, ndmin=2)
if history.size:
    np.testing.assert_allclose(history[:, -2], EA, rtol=0, atol=1e-8)
    string_cycles = np.concatenate((history[:, 0], [string_info["completed_cycles"]]))
    string_barrier = np.concatenate((history[:, 1:1 + len(string_path)].max(axis=1), [string_profile[:, 3].max()]))
else:
    string_cycles, string_barrier = np.array([0]), np.array([string_profile[:, 3].max()])
np.savetxt(output / "string_barrier_history.csv", np.column_stack((string_cycles, string_barrier)),
           delimiter=",", header="outer_cycle,barrier_eV", comments="")

figure, axes = plt.subplots(2, 2, figsize=(11, 8))
for arc, energy, name, style in ((neb_arc, neb_energies - EA, f"NEB ({native_style})", "s--"),
                                (string_profile[:, 1], string_profile[:, 2] - EA, f"String ({string_settings['min_style']})", "o-")):
    if arc[-1] <= 0 or not np.isfinite(energy).all():
        raise ValueError(f"Invalid {name} final energy profile")
    axes[0, 0].plot(arc / arc[-1], energy, style, label=name)
axes[0, 0].set(xlabel="Normalized configuration-space arc length", ylabel=r"$E-E_A$ (eV)", title="Final path energy profiles")
axes[0, 0].legend()
for phase, name, color in ((0, "Ordinary NEB", "tab:blue"), (1, "Climbing NEB", "tab:orange")):
    selected = stages == phase
    # LAMMPS still prints one climbing SETUP row when zero climbing iterations
    # were requested. Keep it in the CSV, but omit it from the iteration curves.
    if not np.any(selected) or (phase == 1 and controls["climbing_steps"] == 0):
        continue
    axes[0, 1].plot(data[selected, 0], neb_barrier[selected], "o-", markersize=3, label=name, color=color)
    axes[1, 1].semilogy(data[selected, 0], data[selected, 1], "o-", markersize=3, label=name, color=color)
axes[0, 1].set(xlabel="NEB iteration", ylabel=r"Barrier $E_b$ (eV)", title="NEB barrier history")
axes[0, 1].legend()
axes[1, 0].plot(string_cycles, string_barrier, "o-", markersize=3, color="tab:green")
axes[1, 0].plot(string_cycles[-1], string_barrier[-1], "ko", label="Fresh final path")
axes[1, 0].set(xlabel="String outer cycle (relax + redistribute)", ylabel=r"Barrier $E_b$ (eV)", title="String barrier history")
axes[1, 0].legend()
axes[1, 1].axhline(controls["ftol"], color="black", linestyle=":", label=f"Tolerance {controls['ftol']:.0e}")
axes[1, 1].set(xlabel="NEB iteration", ylabel="Max NEB replica force (eV/Å)", title="NEB force history")
axes[1, 1].legend()
for axis in axes.flat:
    axis.grid(alpha=0.25)
figure.suptitle("Final paths and convergence · iteration counts have different meanings", fontsize=13)
figure.tight_layout()
figure.savefig(output / "results.png", dpi=180)
figure.savefig(output / "results.pdf")
plt.close(figure)

# Export each final path as an animation: frame index is IMAGE INDEX, not MD
# time. Both calculations have the same cell, so preserve LAMMPS's validated
# triclinic dump-box header rather than reconstructing bounding-box conventions.
box_header = "\n".join((neb / "final.0.dump").read_text().splitlines()[4:8]) + "\n"
for name, path in (("neb", neb_path), ("string", string_path)):
    with (output / f"{name}_path.dump").open("x") as stream:
        for image, coordinates in enumerate(path):
            stream.write(f"ITEM: TIMESTEP\n{image}\nITEM: NUMBER OF ATOMS\n{len(ids)}\n")
            stream.write(box_header)
            stream.write("ITEM: ATOMS id type xu yu zu\n")
            np.savetxt(stream, np.column_stack((ids, types, coordinates)), fmt=["%d", "%d", "%.17g", "%.17g", "%.17g"])
    np.save(output / f"{name}_path.npy", path)
report = {"neb_directory": str(neb), "string_directory": str(string), "simulation_rerun": False,
          "neb_controls_from_actual_log": controls, "neb_images": len(neb_path), "string_images": len(string_path),
          "final_neb_barrier_eV": float(np.max(neb_energies - EA)),
          "final_string_barrier_eV": float(string_profile[:, 3].max()),
          "string_completed_cycles": string_info["completed_cycles"],
          "barrier_definition": "max over all images, including endpoints, of E_i minus E_A",
          "neb_history_note": "Zero-iteration climbing setup row remains in CSV but is omitted from curves",
          "path_dump_frames": "ordered final images along the MEP; not simulation time"}
(output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
print(f"Final NEB barrier: {report['final_neb_barrier_eV']:.8g} eV")
print(f"Final string barrier: {report['final_string_barrier_eV']:.8g} eV")
print(f"Plots and atomic paths: {output}")
