"""Save and compare the five molecular sigma profiles."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from opencosmorspy.input_parsers import SigmaProfileParser
from opencosmorspy.parameterization import openCOSMORS24a

from project_config import PROJECT, surface_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["calculated", "reference"], default="calculated")
    args = parser.parse_args()
    molecules = surface_paths(args.source)
    results_dir = PROJECT / "results" / args.source
    figures_dir = PROJECT / "figures" / args.source
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    parameterization = openCOSMORS24a()
    summaries = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Here we parse each COSMO surface and generate its sigma profile. Averaging
    # uses the same radius as the 24a parameterization and the interactive viewer.
    # The profile stores surface area in each sigma bin. Dividing by total area
    # gives an area fraction, which helps compare differently sized molecules.
    # The normalized bins sum to one; they are fractions per bin, not densities.
    for name, path in molecules.items():
        surface = SigmaProfileParser(str(path))
        surface.calculate_averaged_sigmas(averaging_radius=parameterization.r_av)
        sigma, area = surface.cluster_and_create_sigma_profile()
        np.testing.assert_allclose(area.sum(), surface["seg_area"].sum(), rtol=1e-8)
        fraction = area / area.sum()
        np.savetxt(
            results_dir / f"{name}_sigma.csv",
            np.column_stack((sigma, area, fraction)),
            delimiter=",", comments="",
            header="sigma_e_per_A2,area_A2,area_fraction",
        )
        axes[0].plot(sigma, area, label=name)
        axes[1].plot(sigma, fraction, label=name)
        summaries.append({
            "molecule": name,
            "source_file": str(path.relative_to(PROJECT)),
            "segments": len(surface["seg_area"]),
            "surface_area_A2": surface["area"],
            "profile_area_A2": area.sum(),
        })
        print(f"{name}: {len(surface['seg_area'])} segments; profile area = {area.sum():.4f} A^2")

    # Both panels describe the individual molecules, using either absolute area
    # or area fraction. Negative screening charge is associated with positive
    # molecular regions such as hydroxyl hydrogens; positive screening charge
    # is associated with electron-rich regions such as oxygen acceptor sites.
    for ax in axes:
        ax.axvline(0, color="gray", linewidth=0.8)
        ax.set_xlabel(r"$\sigma$ ($e/\mathrm{\AA}^2$)")
        ax.set_xlim(-0.025, 0.025)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel(r"Area per bin ($\mathrm{\AA}^2$)")
    axes[1].set_ylabel("Surface area fraction per bin")
    axes[0].set_title("Absolute surface areas")
    axes[1].set_title("Normalized profiles")
    axes[1].legend()
    fig.suptitle(f"Five molecular sigma profiles: {args.source} surfaces")
    fig.tight_layout()
    figure_file = figures_dir / "sigma_profiles.png"
    fig.savefig(figure_file, dpi=200)
    plt.close(fig)
    pd.DataFrame(summaries).to_csv(results_dir / "surface_summary.csv", index=False)
    print(f"Saved profiles: {results_dir}")
    print(f"Saved figure: {figure_file}")


if __name__ == "__main__":
    main()
