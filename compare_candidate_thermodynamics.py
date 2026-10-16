"""Compare the equimolar liquid with two selected five-solvent recipes."""

import pandas as pd

from mixture_thermodynamics import build_model, mixture_thermodynamics
from project_config import MOLECULES, PROJECT


TEMPERATURE = 298.15

# Mole fractions follow the same order as our other calculations:
# water, methanol, acetone, acetonitrile, benzene.
RECIPES = {
    "equimolar": [0.20, 0.20, 0.20, 0.20, 0.20],
    "acetone_32": [0.15, 0.22, 0.32, 0.16, 0.15],
    "acetone_33": [0.15, 0.22, 0.33, 0.15, 0.15],
}


def main():
    crs = build_model("calculated")
    crs.par.cosmospace_conv_thresh = 1e-10
    crs.par.cosmospace_max_iter = 3000
    rows, tables = [], []

    for name, x in RECIPES.items():
        # The existing function uses T-1, T and T+1 K to obtain the entropy.
        # Each temperature calculation keeps this recipe's composition fixed.
        table, properties = mixture_thermodynamics(crs, x, TEMPERATURE)

        # Repeat with a wider interval to check the numerical derivative.
        _, wider = mixture_thermodynamics(crs, x, TEMPERATURE, dT=2.0)
        properties["h_step_change_kJ_mol"] = abs(
            wider["h_mix_kJ_mol"] - properties["h_mix_kJ_mol"]
        )
        rows.append({
            "recipe": name,
            "temperature_K": TEMPERATURE,
            **{f"x_{molecule}": fraction for molecule, fraction in zip(MOLECULES, x)},
            **properties,
        })
        tables.append(table.assign(recipe=name))

    summary = pd.DataFrame(rows)
    baseline = summary.iloc[0]

    # Subtract the equimolar result to explain each candidate's advantage.
    # A negative change in H or in -T*S favors a more negative mixing G.
    for quantity in ("g_mix_kJ_mol", "h_mix_kJ_mol", "minus_T_s_mix_kJ_mol"):
        summary[f"change_{quantity}"] = summary[quantity] - baseline[quantity]

    directory = PROJECT / "results/calculated/candidate_comparison"
    directory.mkdir(parents=True, exist_ok=True)
    summary.to_csv(directory / "candidate_thermodynamics_top.csv", index=False)
    pd.concat(tables, ignore_index=True).to_csv(
        directory / "candidate_activities_top.csv", index=False
    )

    columns = ["recipe", "g_mix_kJ_mol", "h_mix_kJ_mol", "s_ideal_J_mol_K",
               "s_excess_J_mol_K", "s_mix_J_mol_K", "minus_T_s_mix_kJ_mol",
               "h_step_change_kJ_mol"]
    print(summary[columns].round(6).to_string(index=False))
    changes = ["recipe", "change_g_mix_kJ_mol", "change_h_mix_kJ_mol",
               "change_minus_T_s_mix_kJ_mol"]
    print("\nChanges relative to the equimolar recipe (kJ/mol):")
    print(summary[changes].round(6).to_string(index=False))
    print("\nSaved:", directory)
    print("These are homogeneous-liquid properties; stability is assessed separately.")


if __name__ == "__main__":
    main()
