# Starting geometries

These five XYZ files were generated for this project using **RDKit 2025.09.6**,
ETKDGv3 embedding with random seed 42, followed by MMFF94 optimization. All five
MMFF optimizations converged. Coordinates are in angstrom, hydrogens are explicit,
and every molecule has charge 0 and multiplicity 1.

These are molecular-mechanics starting structures, not DFT geometries or
experimental measurements. `run_cosmo_calculations.py` subsequently optimizes
them with ORCA in the conductor environment. The existing reference surfaces
are not required to run these new calculations.

| Molecule | SMILES | Atoms including H |
| --- | --- | --- |
| water | O | 3 |
| methanol | CO | 6 |
| acetone | CC(=O)C | 10 |
| acetonitrile | CC#N | 6 |
| benzene | c1ccccc1 | 12 |

RDKit is not a dependency of the main workflow because the XYZ files are already
included. For reproducibility, the generation procedure was:

```python
from pathlib import Path
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem

structures = Path("data/structures")
structures.mkdir(parents=True, exist_ok=True)
smiles = {
    "water": "O", "methanol": "CO", "acetone": "CC(=O)C",
    "acetonitrile": "CC#N", "benzene": "c1ccccc1",
}

for name, text in smiles.items():
    molecule = Chem.AddHs(Chem.MolFromSmiles(text))
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = 42
    assert AllChem.EmbedMolecule(molecule, parameters) == 0
    assert AllChem.MMFFHasAllMoleculeParams(molecule)
    assert AllChem.MMFFOptimizeMolecule(
        molecule, maxIters=1000, mmffVariant="MMFF94"
    ) == 0
    lines = Chem.MolToXYZBlock(molecule).splitlines()
    lines[1] = (
        f"{name}; SMILES={text}; RDKit {rdBase.rdkitVersion}; "
        "ETKDGv3 seed=42 + MMFF94; angstrom; charge=0; multiplicity=1"
    )
    (structures / f"{name}.xyz").write_text("\n".join(lines) + "\n")
```

Documentation: [RDKit molecular embedding and force-field optimization](https://www.rdkit.org/docs/GettingStartedInPython.html#working-with-3d-molecules).
