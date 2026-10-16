

"""Search for liquid-phase instability at one five-solvent composition.

Run beside mixture_thermodynamics.py and project_config.py.
Default: calculated surfaces, five equimolar solvents, 298.15 K.
"""

import argparse
from itertools import combinations
import json

import numpy as np
import pandas as pd
from scipy.constants import R
from scipy.optimize import minimize

from mixture_thermodynamics import build_model, mixture_properties
from project_config import MOLECULES, PROJECT, surface_paths


TEMPERATURE = 298.15
FLOOR = 1e-10
TPD_TOLERANCE = 1e-6  # Dimensionless; much larger than solver roundoff.


def starting_compositions(z, count, seed):
    """Include the original liquid, component-rich, pair-rich and random starts."""
    n = len(z)
    starts = [z.copy()]
    starts.extend(0.95 * np.eye(n) + 0.05 / n)
    for i, j in combinations(range(n), 2):
        w = np.full(n, 0.05 / n)
        w[[i, j]] += 0.475
        starts.append(w)
    rng = np.random.default_rng(seed)
    while len(starts) < count:
        concentration = (0.25, 1.0, 3.0)[len(starts) % 3]
        w = rng.dirichlet(np.full(n, concentration))
        starts.append(FLOOR + (1 - n * FLOOR) * w)
    return starts[:count]


def search_tpd(log_activity, z, count=4000, seed=88):
    """Minimize dimensionless TPD from several starting compositions."""
    original_log_a = log_activity(z)
    best_value, best_w = 0.0, z.copy()  # TPD is exactly zero at w = z.
    rows = []

    # z is the overall composition. w is a candidate new liquid composition.
    # TPD/RT = sum(w * [ln(a(w)) - ln(a(z))]). A negative value shows that
    # forming a small amount of this liquid can lower total Gibbs energy.
    # Trial liquids are free to explore compositions far from equimolar.
    for number, start in enumerate(starting_compositions(z, count, seed), 1):
        lowest_value, lowest_w = np.inf, start.copy()

        def objective(u):
            nonlocal lowest_value, lowest_w
            total = u.sum()
            w = u / total
            difference = log_activity(w) - original_log_a
            value = float(w @ difference)
            # Gibbs-Duhem gives this gradient for the normalized composition.
            gradient = (difference - value) / total
            if value < lowest_value:
                lowest_value, lowest_w = value, w.copy()
            return value, gradient

        result = minimize(
            objective, start, jac=True, method="SLSQP",
            bounds=[(FLOOR, 1.0)] * len(z),
            constraints={"type": "eq", "fun": lambda u: u.sum() - 1.0,
                         "jac": lambda u: np.ones_like(u)},
            options={"ftol": 1e-10, "maxiter": 300},
        )
        # Retain negative points even if an optimizer later fails to converge.
        if lowest_value < best_value:
            best_value, best_w = lowest_value, lowest_w.copy()
        row = {
            "start": number, "converged": bool(result.success),
            "iterations": result.nit, "evaluations": result.nfev,
            "lowest_tpd_over_RT": lowest_value,
            "final_tpd_over_RT": float(result.fun), "message": str(result.message),
        }
        row.update({f"initial_x_{i}": value for i, value in enumerate(start)})
        row.update({f"best_x_{i}": value for i, value in enumerate(lowest_w)})
        rows.append(row)
        print(f"Start {number:3d}/{count}: lowest TPD/RT = {lowest_value:+.8f}; "
              f"converged = {result.success}", flush=True)

    return best_value, best_w, pd.DataFrame(rows)


def check_small_split(log_activity, z, w):
    """Verify a negative TPD with an explicit, mass-balanced two-liquid trial."""
    RT = R * TEMPERATURE / 1000
    g_original = RT * (z @ log_activity(z))
    g_new = RT * (w @ log_activity(w))
    fraction = min(0.01, 0.5 * np.min(z / w))
    for _ in range(10):
        remainder = (z - fraction * w) / (1 - fraction)
        g_remainder = RT * (remainder @ log_activity(remainder))
        change = fraction * g_new + (1 - fraction) * g_remainder - g_original
        if change < -1e-10:
            return fraction, remainder, float(change)
        fraction /= 10
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x", nargs=5, type=float, default=[0.2] * 5,
                        help="Overall mole fractions in project order")
    parser.add_argument("--starts", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=88)
    args = parser.parse_args()
    z = np.asarray(args.x)
    if not np.all(np.isfinite(z)) or np.any(z < FLOOR) or not np.isclose(
        z.sum(), 1.0, rtol=0, atol=1e-12
    ):
        parser.error("Provide five positive mole fractions summing to one.")
    if args.starts < 16:
        parser.error(
            "Use at least 16 starts to include the fixed starting points.")
    z = z / z.sum()

    model = build_model("calculated")
    # Tighten the numerical COSMOspace solver for a sensitive stability test.
    # The molecular surfaces and physical model parameters are unchanged.
    model.par.cosmospace_conv_thresh = 1e-10
    model.par.cosmospace_max_iter = 3000

    def log_activity(x):
        table, _ = mixture_properties(model, x, TEMPERATURE)
        return np.log(x) + table["ln_gamma"].to_numpy()

    print("Molecule order:", ", ".join(MOLECULES))
    print("Overall mole fractions:", z)
    best_value, best_w, trials = search_tpd(
        log_activity, z, args.starts, args.seed)
    failed = int((~trials["converged"]).sum())
    if best_value < -TPD_TOLERANCE:
        conclusion = "Instability found within the model."
    elif failed:
        conclusion = "No instability found; some searches did not converge."
    else:
        conclusion = "No instability found in this multistart search."
    print("\n" + conclusion)
    print(f"Lowest TPD/RT: {best_value:+.9f}")
    print(f"Lowest TPD: {best_value * R * TEMPERATURE / 1000:+.9f} "
          "kJ/mol of trial liquid")
    print("A finite search does not certify the global minimum.")

    tag = "x_" + "_".join(f"{value:.8g}" for value in z)
    output = PROJECT / "results/calculated/phase_stability" / tag
    output.mkdir(parents=True, exist_ok=True)
    trials.to_csv(output / "searches.csv", index=False)
    summary = {
        "temperature_K": TEMPERATURE, "molecule_order": list(MOLECULES),
        "overall_x": z.tolist(), "starts": args.starts, "seed": args.seed,
        "lowest_tpd_over_RT": best_value, "best_trial_x": best_w.tolist(),
        "nonconverged_searches": failed, "conclusion": conclusion,
        "tpd_tolerance": TPD_TOLERANCE, "mole_fraction_floor": FLOOR,
        "cosmospace_convergence": model.par.cosmospace_conv_thresh,
        "surface_files": {name: str(path.relative_to(PROJECT))
                          for name, path in surface_paths("calculated").items()},
    }
    if best_value < -TPD_TOLERANCE:
        split = check_small_split(log_activity, z, best_w)
        if split is not None:
            fraction, remainder, change = split
            print(f"Explicit check: put {100 * fraction:.6g}% of total moles "
                  f"in the trial liquid; change in G = {change:+.9f} kJ/mol.")
            print("These are trial compositions, not final equilibrium layers.")
            pd.DataFrame({"molecule": MOLECULES, "overall_x": z,
                          "trial_liquid_x": best_w, "remaining_liquid_x": remainder,
                          "trial_mole_fraction_of_total": fraction}).to_csv(
                output / "trial_split.csv", index=False
            )
            summary.update({"trial_mole_fraction_of_total": fraction,
                            "explicit_split_change_kJ_mol": change})
        else:
            print("Negative TPD found, but finite-split verification needs inspection.")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("Saved:", output)


if __name__ == "__main__":
    main()
