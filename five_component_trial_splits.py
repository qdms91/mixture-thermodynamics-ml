"""Test whether selected splits lower the five-component mixing free energy."""

import numpy as np

from mixture_thermodynamics import build_model, mixture_properties


model = build_model("calculated")
overall = np.full(5, 0.20)  # water, methanol, acetone, acetonitrile, benzene


def free_energy(composition):
    _, properties = mixture_properties(model, composition, temperature=298.15)
    return properties["g_mix_kJ_mol"]


one_liquid = free_energy(overall)
print(f"One liquid: {one_liquid:.6f} kJ/mol")

# Make two equally sized liquids. Move water into A and benzene into B.
# Their average composition remains 20% of every component.
for shift in [0.05, 0.10, 0.15, 0.19]:
    phase_a = overall.copy()
    phase_b = overall.copy()

    phase_a[0] += shift
    phase_a[4] -= shift
    phase_b[0] -= shift
    phase_b[4] += shift

    two_liquids = (
        free_energy(phase_a) + free_energy(phase_b)
    ) / 2

    print(
        f"shift={shift:.2f}: two liquids={two_liquids:.6f}, "
        f"change={two_liquids - one_liquid:+.6f} kJ/mol"
    )
