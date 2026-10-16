"""Scan water–benzene mixing and test a 50:50 liquid for phase splitting."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mixture_thermodynamics import build_model, mixture_properties
from project_config import PROJECT


model = build_model("calculated")
benzene_fraction = np.linspace(0.0, 1.0, 101)
g_mix = np.zeros(len(benzene_fraction))

# Pure liquids have zero mixing free energy. Calculate the 99 mixtures.
for k, z in enumerate(benzene_fraction[1:-1], start=1):
    x = [1 - z, 0, 0, 0, z]  # water, methanol, acetone, acetonitrile, benzene
    _, properties = mixture_properties(model, x, temperature=298.15)
    g_mix[k] = properties["g_mix_kJ_mol"]

# At overall x_benzene = 0.5, compare one liquid with pairs of liquids.
# The lever rule gives the fraction of liquid at composition a.
feed_index = 50
single_g = g_mix[feed_index]
best_g = single_g
best_pair = None

for i in range(feed_index):
    for j in range(feed_index + 1, len(benzene_fraction)):
        a = benzene_fraction[i]
        b = benzene_fraction[j]
        fraction_at_a = (b - 0.5) / (b - a)
        split_g = fraction_at_a * g_mix[i] + (1 - fraction_at_a) * g_mix[j]

        if split_g < best_g:
            best_g = split_g
            best_pair = (i, j)

results = PROJECT / "results/calculated"
figures = PROJECT / "figures/calculated"
results.mkdir(parents=True, exist_ok=True)
figures.mkdir(parents=True, exist_ok=True)

pd.DataFrame({
    "x_benzene": benzene_fraction,
    "g_mix_kJ_mol": g_mix,
}).to_csv(results / "water_benzene_scan.csv", index=False)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(benzene_fraction, g_mix, label="One liquid")
ax.plot(0.5, single_g, "ko", label="50:50 overall composition")
ax.axhline(0, color="gray", linewidth=0.8)

print(f"One liquid at 50:50: {single_g:.4f} kJ/mol")

if best_pair is not None and single_g - best_g > 0.001:
    i, j = best_pair
    ax.plot(
        benzene_fraction[[i, j]], g_mix[[i, j]], "r--",
        label="Lower-energy two-liquid candidate",
    )
    print(
        f"Two-liquid candidate: x_benzene = "
        f"{benzene_fraction[i]:.2f} and {benzene_fraction[j]:.2f}"
    )
    print(f"Lower by: {single_g - best_g:.4f} kJ/mol")
else:
    print("No convincingly lower two-liquid state found on this grid.")

ax.set(
    xlabel="Benzene mole fraction",
    ylabel="Mixing Gibbs energy (kJ/mol)",
    title="Water–benzene at 298.15 K",
)
ax.legend()
ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(figures / "water_benzene_stability.png", dpi=200)
plt.close(fig)

print("Saved scan and figure.")
