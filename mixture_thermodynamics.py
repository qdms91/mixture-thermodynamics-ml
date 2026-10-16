"""Activities and mixing thermodynamics for two to five equimolar liquids."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.constants import R
from opencosmorspy.cosmors import COSMORS
from opencosmorspy.parameterization import openCOSMORS24a

from project_config import MOLECULES, PROJECT, surface_paths


def build_model(source="calculated"):
    """Register all five surfaces in the fixed project order."""
    crs = COSMORS(par=openCOSMORS24a())
    for path in surface_paths(source).values():
        crs.add_molecule([str(path)])
    return crs


def mixture_properties(crs, x, temperature=298.15):
    """Calculate activities and molar free energies of one homogeneous liquid."""
    x = np.asarray(x, dtype=float)
    if (
        x.shape != (len(MOLECULES),)
        or not np.all(np.isfinite(x))
        or np.any(x < 0)
        or not np.isclose(x.sum(), 1.0, rtol=0, atol=1e-12)
    ):
        raise ValueError("Provide five finite, nonnegative mole fractions summing to one.")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive, in kelvin.")
    x = x / x.sum()
    present = x > 0

    # add_job registers the composition and temperature using the pure liquids
    # as reference states. calculate() returns ln(gamma), so np.exp converts
    # these values to activity coefficients. The dimensionless activity is x*gamma.
    # clear_jobs removes earlier requests; it keeps the registered molecules.
    crs.clear_jobs()
    crs.add_job(x, temperature, refst="pure_component")
    ln_gamma = np.asarray(crs.calculate()["tot"]["lng"][0])
    if not np.all(np.isfinite(ln_gamma)):
        raise RuntimeError("The activity calculation returned nonfinite values.")
    gamma = np.exp(ln_gamma)

    # These are molar mixing free energies, relative to the separated pure
    # liquids at the same temperature. The ideal term is RT*sum(x*ln(x)), and
    # the excess term is RT*sum(x*ln(gamma)). We omit zero fractions from ln(x).
    # R is in J/(mol K), so dividing by 1000 gives energies in kJ/mol of mixture.
    RT = R * temperature / 1000
    g_ideal = RT * np.sum(x[present] * np.log(x[present]))
    g_excess = RT * np.sum(x[present] * ln_gamma[present])
    table = pd.DataFrame({
        "molecule": MOLECULES,
        "temperature_K": temperature,
        "mole_fraction": x,
        "ln_gamma": ln_gamma,
        "gamma": gamma,
        "activity": x * gamma,
    })
    properties = {
        "g_ideal_kJ_mol": g_ideal,
        "g_excess_kJ_mol": g_excess,
        "g_mix_kJ_mol": g_ideal + g_excess,
    }
    return table.loc[present].reset_index(drop=True), properties


def mixture_thermodynamics(crs, x, temperature=298.15, dT=1.0):
    """Add model-derived molar mixing entropy and enthalpy."""
    if not np.isfinite(dT) or not 0 < dT < temperature:
        raise ValueError("Require a finite temperature step with 0 < dT < temperature.")
    table, properties = mixture_properties(crs, x, temperature)
    _, lower = mixture_properties(crs, x, temperature - dT)
    _, upper = mixture_properties(crs, x, temperature + dT)

    # Excess entropy is minus the temperature derivative of excess free energy
    # at fixed composition and pressure. We approximate it using T-dT and T+dT.
    # The same molecular surfaces are used at all three temperatures: this probes
    # the COSMO-RS model's temperature dependence, rather than reoptimizing the
    # molecules or calculating gas-phase vibrational thermochemistry with ORCA.
    s_excess = -1000 * (
        upper["g_excess_kJ_mol"] - lower["g_excess_kJ_mol"]
    ) / (2 * dT)
    s_ideal = -1000 * properties["g_ideal_kJ_mol"] / temperature
    h_mix = properties["g_excess_kJ_mol"] + temperature * s_excess / 1000
    properties.update({
        "s_ideal_J_mol_K": s_ideal,
        "s_excess_J_mol_K": s_excess,
        "s_mix_J_mol_K": s_ideal + s_excess,
        "h_mix_kJ_mol": h_mix,
        "minus_T_s_mix_kJ_mol": -temperature * (s_ideal + s_excess) / 1000,
    })
    return table, properties


def plot_results(summary, activities, directory, source):
    """Plot the free-energy contributions and the component properties."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    groups = [
        [("g_ideal_kJ_mol", "Ideal"), ("g_excess_kJ_mol", "Excess"),
         ("g_mix_kJ_mol", "Total mixing")],
        [("h_mix_kJ_mol", "Enthalpy"), ("minus_T_s_mix_kJ_mol", "-T × entropy"),
         ("g_mix_kJ_mol", "Mixing free energy")],
    ]
    for ax, columns in zip(axes, groups):
        for column, label in columns:
            ax.plot(summary["components"], summary[column], "o-", label=label)
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_xticks([2, 3, 4, 5])
        ax.set_xlabel("Number of components in the selected equimolar mixtures")
        ax.set_ylabel("kJ/mol of mixture")
        ax.legend()
        ax.grid(alpha=0.2)
    fig.suptitle(f"Mixing thermodynamics at 298.15 K: {source} surfaces")
    fig.tight_layout()
    fig.savefig(directory / "mixing_thermodynamics.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for name in MOLECULES:
        rows = activities.loc[activities["molecule"] == name]
        axes[0].plot(rows["components"], rows["gamma"], "o-", label=name)
        axes[1].plot(rows["components"], rows["activity"], "o-", label=name)
    axes[0].axhline(1, color="gray", linestyle="--", label="Ideal gamma")
    axes[1].plot([2, 3, 4, 5], 1 / np.arange(2, 6), "k--", label="Ideal activity (1/N)")
    for ax, ylabel in zip(axes, ["Activity coefficient", "Activity"]):
        ax.set_xticks([2, 3, 4, 5])
        ax.set_xlabel("Number of components in the selected equimolar mixtures")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.2)
    fig.suptitle(f"Component properties at 298.15 K: {source} surfaces")
    fig.tight_layout()
    fig.savefig(directory / "activities.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["calculated", "reference"], default="calculated")
    args = parser.parse_args()
    crs = build_model(args.source)
    results_dir = PROJECT / "results" / args.source
    figures_dir = PROJECT / "figures" / args.source
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    rows, tables = [], []

    # This sequence is water/methanol, then acetone, then acetonitrile, then
    # benzene. Each calculation represents a separate equimolar composition.
    # It is not a time-dependent simulation of sequential pouring, and changing
    # the number of components also changes their identities and mole fractions.
    for n in range(2, 6):
        x = np.zeros(len(MOLECULES))
        x[:n] = 1 / n
        table, properties = mixture_thermodynamics(crs, x)
        _, wider_step = mixture_thermodynamics(crs, x, dT=2.0)
        properties["h_step_change_kJ_mol"] = abs(
            wider_step["h_mix_kJ_mol"] - properties["h_mix_kJ_mol"]
        )
        rows.append({"source": args.source, "components": n, "temperature_K": 298.15, **properties})
        tables.append(table.assign(source=args.source, components=n))
        print(f"\n{n} components at 298.15 K ({args.source} surfaces)")
        print(table.round(6).to_string(index=False))

    summary = pd.DataFrame(rows)
    activities = pd.concat(tables, ignore_index=True)
    summary.to_csv(results_dir / "equimolar_thermodynamics.csv", index=False)
    activities.to_csv(results_dir / "equimolar_activities.csv", index=False)
    pd.DataFrame([
        {"molecule": name, "surface_file": str(path.relative_to(PROJECT))}
        for name, path in surface_paths(args.source).items()
    ]).to_csv(results_dir / "mixture_inputs.csv", index=False)
    columns = ["components", "g_ideal_kJ_mol", "g_excess_kJ_mol", "g_mix_kJ_mol",
               "h_mix_kJ_mol", "s_mix_J_mol_K", "h_step_change_kJ_mol"]
    print("\n" + summary[columns].round(6).to_string(index=False))
    plot_results(summary, activities, figures_dir, args.source)
    print(f"\nSaved tables: {results_dir}")
    print(f"Saved figures: {figures_dir}")
    print("These are homogeneous-liquid predictions; phase stability has not been tested.")


if __name__ == "__main__":
    main()
