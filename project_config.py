"""Molecule order and file locations shared by the project scripts."""

from pathlib import Path


PROJECT = Path(__file__).resolve().parent
REFERENCE = (
    PROJECT / "data/reference/openCOSMO-RS_24a_SI_data_qm_files/qm_files"
)

# This order also defines the order of mole fractions in the composition x.
# Keeping the names in one place lets the calculation, plotting and mixture
# scripts refer to exactly the same five compounds and surface files.
REFERENCE_FILES = {
    "water": "H2O_c000.orcacosmo",
    "methanol": "methanol_c000.orcacosmo",
    "acetone": "acetone_c000.orcacosmo",
    "acetonitrile": "acetonitrile_CnfS1_c000.orcacosmo",
    "benzene": "benzene_CnfS1_c000.orcacosmo",
}
MOLECULES = tuple(REFERENCE_FILES)


def surface_paths(source="calculated", names=None):
    """Return the requested surfaces, checking that every file exists."""
    if source not in ("calculated", "reference"):
        raise ValueError("source must be 'calculated' or 'reference'.")
    names = MOLECULES if names is None else names
    paths = {}
    for name in names:
        if name not in MOLECULES:
            raise ValueError(f"Unknown molecule: {name}")
        if source == "reference":
            path = REFERENCE / REFERENCE_FILES[name]
        else:
            path = PROJECT / "calculations" / name / f"{name}.solute.orcacosmo"
        if not path.is_file():
            raise FileNotFoundError(f"Missing {source} surface: {path}")
        paths[name] = path
    return paths
