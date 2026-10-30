"""Relax/redistribute utilities independent of the energy/force backend.

reparamChain(Rchain) returns a redistributed chain; stringrelax(...) returns
(Rp, Eclist, info). Callers supply consistent units and periodic coordinates.
Analytic F_func means positive energy gradient; relax_func can instead provide
engine-driven relaxation. See each function's full call/return documentation.
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import approx_fprime


def reparamChain(Rchain):
    """Redistribute interior images uniformly along the input polyline.

    Parameters
    ----------
    Rchain : array_like, shape (nchain, *configuration_shape)
        Finite coordinates of at least two nonempty configurations. Distances
        use the Euclidean norm over every configuration component. A writable
        float64 NumPy array is modified in place; other inputs are converted
        with np.asarray(..., dtype=float) and may produce a new array.

    Returns
    -------
    Rchain : numpy.ndarray, shape (nchain, *configuration_shape)
        Redistributed float64 chain, with the original endpoints unchanged.
        Consecutive duplicate arc samples are omitted during interpolation.
        A completely zero-length chain is returned unchanged after conversion.
        The result can share storage with the input; it is not an independent
        snapshot.

    Raises
    ------
    ValueError
        If there are fewer than two images, configurations are empty, or
        coordinates or cumulative arc lengths are non-finite.

    Notes
    -----
    Equal intervals are measured along the original polyline. Straight-line
    distances between the returned images can differ when an interval crosses
    a corner. No periodic wrapping or atom matching is performed here; supply
    continuous coordinates with matching components before calling.

    Examples
    --------
    A collinear chain is redistributed without moving its endpoints::

        chain = np.array([[0.0, 0.0], [1.0, 0.0], [4.0, 0.0]])
        redistributed = reparamChain(chain)  # Middle image becomes [2, 0].
    """
    Rchain = np.asarray(Rchain, dtype=float)
    if Rchain.ndim < 2 or len(Rchain) < 2 or Rchain[0].size == 0:
        raise ValueError("Rchain must contain at least two nonempty configurations")
    if not np.isfinite(Rchain).all():
        raise ValueError("Rchain contains non-finite coordinates")

    nchain = len(Rchain)
    lengths = np.linalg.norm(np.diff(Rchain, axis=0).reshape(nchain - 1, -1), axis=1)
    arc = np.concatenate(([0.0], np.cumsum(lengths)))
    if not np.isfinite(arc).all():
        raise ValueError("Non-finite arc length")
    if arc[-1] == 0.0:
        return Rchain
    keep = np.concatenate(([True], np.diff(arc) > 0.0))
    interpolate = interp1d(arc[keep], Rchain[keep], axis=0)
    targets = np.arange(1, nchain - 1) / (nchain - 1) * arc[-1]
    Rchain[1:-1] = interpolate(targets)
    return Rchain


def stringrelax(rA, rB, E_func=None, F_func=None, f_epsilon=1e-5,
                nchain=6, nstep=10, relax_step=10, step_size=1e-2,
                energySpike=None, verbose=False, relax_func=None, path_tol=None):
    """Initialize a path, relax its interior images, and redistribute it.

    Parameters
    ----------
    rA, rB : array_like, shape configuration_shape
        Finite endpoint configurations with the same nonempty shape, such as
        (2,) for an analytic example or (natoms, 3) for atoms. Both are copied
        and remain fixed throughout the calculation. Supply continuous
        coordinates in a common cell and atom order for periodic systems.
    E_func : callable, optional
        E_func(R) -> energy returns one finite scalar for a configuration of
        shape configuration_shape. Used for pre-cycle energy history and, when
        neither relaxation nor gradient callbacks are supplied, for forward
        finite-difference gradients. Treat its input as read-only.
    F_func : callable, optional
        F_func(R) -> gradient returns a finite array of the same shape as R.
        Despite the prototype name, this is the POSITIVE gradient grad(E);
        analytic descent subtracts it. LAMMPS physical forces have the opposite
        sign. Treat its input as read-only. Ignored when relax_func is supplied.
    f_epsilon : float, default 1e-5
        Positive, finite coordinate increment for numerical differentiation of
        the supplied E_func. No hardcoded analytic potential is used.
    nchain : int, default 6
        Number of images including the two endpoints; must be at least two.
    nstep : int, default 10
        Maximum number of complete relaxation/redistribution cycles; may be
        zero to return just the initial linear path.
    relax_step : int, default 10
        Nonnegative descent-step count per interior image, or the iteration
        budget passed to relax_func.
    step_size : float, default 1e-2
        Positive, finite multiplier in R -= step_size * gradient for the
        analytic branch. Unused by a supplied relax_func.
    energySpike : None, optional
        Reserved prototype argument. Any non-None value is rejected because
        adaptive energy-spike handling is not implemented.
    verbose : bool, default False
        Print the maximum image displacement after each completed cycle.
    relax_func : callable, optional
        relax_func(R, relax_step) -> relaxed_coordinates replaces the analytic
        descent loop. Receives a COPY of one interior configuration and must
        return finite coordinates with exactly the same shape. It may use
        fewer iterations than requested. The LAMMPS driver uses short CG
        minimizations here, a deliberate numerical change from the prototype's
        fixed-step gradient descent.
    path_tol : float or None, optional
        Nonnegative, finite stopping tolerance for the maximum Euclidean norm
        of an image's coordinate change between complete cycles. The norm
        covers the entire configuration, e.g. all 3N atomic coordinates.
        None disables early stopping. This tests path stability, not force
        convergence.

    Returns
    -------
    Rp : numpy.ndarray, shape (nchain, *configuration_shape)
        Final float64 path after the last redistribution, including the fixed
        endpoints. Does not share coordinate storage with the endpoint inputs.
    Eclist : numpy.ndarray, shape (completed_cycles, nchain + 3), or None
        History recorded BEFORE each cycle. Columns are the zero-based cycle,
        E(image_i) - EA for each image, the current evaluated EA, and the
        initially evaluated EA. With zero cycles the array has zero rows;
        without E_func it is None. Re-evaluate Rp for a fresh final energy
        profile; the final history row precedes the final cycle.
    info : dict
        completed_cycles is the number of completed cycles. final_path_change
        is the maximum image displacement in the last cycle, or 0.0 for zero
        cycles. path_changes lists these values for every cycle. stop_reason
        is "path_tol" when the optional tolerance is met, otherwise
        "max_cycles".

    Raises
    ------
    ValueError
        If endpoint shapes/values, iteration counts, or scalar controls are
        invalid, or energySpike is not None. Positive step_size and f_epsilon
        are required even when their branches are unused.
    TypeError
        If a supplied callback is not callable, or none of the three callbacks
        is supplied.
    RuntimeError
        If energy evaluation or relaxation fails, returns an invalid value,
        or produces a non-finite path change. Callback failures include cycle
        and image context and retain the original exception as their cause.

    Notes
    -----
    At least one of E_func, F_func, or relax_func is required. Relaxation
    precedence is relax_func, then F_func, then numerical differentiation of
    E_func. Coordinates, energies, gradient units, and descent multipliers
    must be consistent; this utility imposes no unit system. Redistribution
    uses unweighted Euclidean configuration-space arc length and does not add
    NEB springs or project out tangential forces.

    Examples
    --------
    Energy-only analytic calculation::

        energy = lambda R: float(np.sum(R**2))
        Rp, Eclist, info = stringrelax([-1.0, 1.0], [1.0, 1.0],
                                      E_func=energy, nchain=8, nstep=10)

    After initializing the LAMMPS driver's shared engine and callbacks::

        Rp, Eclist, info = stringrelax(
            rA, rB, E_func=lambda R: evaluate(R)[0], relax_func=relax,
            nchain=8, nstep=50, relax_step=5)
    """
    # Copy endpoints and validate controls before allocating or changing the path.
    rA, rB = np.array(rA, dtype=float, copy=True), np.array(rB, dtype=float, copy=True)
    if rA.ndim == 0 or rA.size == 0 or rA.shape != rB.shape:
        raise ValueError("Endpoints must have the same nonempty configuration shape")
    if not np.isfinite(rA).all() or not np.isfinite(rB).all():
        raise ValueError("Endpoints contain non-finite coordinates")
    for name, value, minimum in (("nchain", nchain, 2), ("nstep", nstep, 0),
                                  ("relax_step", relax_step, 0)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    for name, value in (("step_size", step_size), ("f_epsilon", f_epsilon)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if path_tol is not None and (not np.isfinite(path_tol) or path_tol < 0):
        raise ValueError("path_tol must be None or finite and nonnegative")
    if energySpike is not None:
        raise ValueError("energySpike has no implemented behavior")
    for name, func in (("E_func", E_func), ("F_func", F_func), ("relax_func", relax_func)):
        if func is not None and not callable(func):
            raise TypeError(f"{name} must be callable")
    if relax_func is None and E_func is None and F_func is None:
        raise TypeError("Supply E_func, F_func, or relax_func")

    # Initialize a linear path; every history row describes the path BEFORE a cycle.
    Rp = np.linspace(rA, rB, nchain)
    Eclist = [] if E_func is not None else None
    E0 = None
    info = {"completed_cycles": 0, "final_path_change": 0.0,
            "stop_reason": "max_cycles", "path_changes": []}

    for step in range(nstep):
        previous = Rp.copy()
        if E_func is not None:
            energies = np.empty(nchain)
            for i in range(nchain):
                try:
                    value = np.asarray(E_func(Rp[i]), dtype=float)
                    if value.ndim != 0 or not np.isfinite(value):
                        raise ValueError("Energy must be one finite scalar")
                    energies[i] = value
                except Exception as exc:
                    raise RuntimeError(f"Energy evaluation failed at cycle {step}, image {i}: {exc}") from exc
            if E0 is None:
                E0 = energies[0]
            Eclist.append(np.concatenate(([step], energies - energies[0], [energies[0], E0])))

        # Only interior images are relaxed. A callback replaces the analytic
        # descent loop; otherwise use a supplied gradient or differentiate E_func.
        for i in range(1, nchain - 1):
            try:
                if relax_func is not None:
                    relaxed = np.asarray(relax_func(Rp[i].copy(), relax_step), dtype=float)
                    if relaxed.shape != rA.shape or not np.isfinite(relaxed).all():
                        raise ValueError("Relaxation returned an invalid shape or non-finite coordinates")
                    Rp[i] = relaxed
                else:
                    for _ in range(relax_step):
                        if F_func is not None:
                            grad = np.asarray(F_func(Rp[i]), dtype=float)
                        else:
                            # Flatten only for scipy, then restore the original configuration shape.
                            grad = approx_fprime(
                                Rp[i].ravel(), lambda x: E_func(x.reshape(rA.shape)),
                                f_epsilon).reshape(rA.shape)
                        if grad.shape != rA.shape or not np.isfinite(grad).all():
                            raise ValueError("Gradient has an invalid shape or non-finite values")
                        Rp[i] -= step_size * grad
                        if not np.isfinite(Rp[i]).all():
                            raise ValueError("Descent produced non-finite coordinates")
            except Exception as exc:
                raise RuntimeError(f"Relaxation failed at cycle {step}, image {i}: {exc}") from exc

        # Redistribute the relaxed path, restore fixed endpoints, then measure
        # the maximum full-image displacement over this complete cycle.
        Rp = reparamChain(Rp)
        Rp[0], Rp[-1] = rA, rB
        change = float(np.linalg.norm((Rp - previous).reshape(nchain, -1), axis=1).max())
        if not np.isfinite(change):
            raise RuntimeError(f"Non-finite path change at cycle {step}")
        info["completed_cycles"] = step + 1
        info["final_path_change"] = change
        info["path_changes"].append(change)
        if verbose:
            print(f"Cycle {step + 1}/{nstep}: max image displacement = {change:.8g}", flush=True)
        if path_tol is not None and change <= path_tol:
            info["stop_reason"] = "path_tol"
            break

    if Eclist is not None:
        Eclist = np.asarray(Eclist).reshape(-1, nchain + 3)
    return Rp, Eclist, info
