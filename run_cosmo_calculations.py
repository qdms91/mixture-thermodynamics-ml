"""Optimize the five molecules with ORCA/OPI, then generate COSMO surfaces."""

import argparse
from datetime import datetime
import os
from pathlib import Path
import shutil

from opi.core import Calculator
from opi.input.structures import Structure
from opencosmorspy.input_parsers import SigmaProfileParser

from project_config import MOLECULES, PROJECT


ORCA = Path(os.environ.get("OPI_ORCA", str(Path.home() / "orca_6_1_1" / "orca")))

# Here we define a conductor-phase geometry optimization at BP86/def2-TZVP.
# The H, C, N and O cavity radii come from the authors' conformer pipeline.
# cut_area is in bohr squared: 0.035710648 corresponds to 0.01 angstrom squared.
# These settings keep the geometry preparation close to the 24a protocol.
# Source: https://github.com/TUHH-TVT/openCOSMO-RS_conformer_pipeline
OPTIMIZATION = """! BP86 def2-TZVP TightSCF TightOpt CPCM
%cpcm
  radius[1] 1.30
  radius[6] 2.00
  radius[7] 1.83
  radius[8] 1.72
  cut_area 0.035710648
end
"""

# COSMORS(WATER) uses ORCA's built-in BP86/def2-TZVPD surface workflow.
# It also performs gas-phase and solvent single points for its own solvation
# calculation. We retain the solute surface for our Python mixture calculations.
# WATER here selects the auxiliary solvent job; it does not fix the composition
# of the mixtures we subsequently calculate with openCOSMO-RS Python.
SURFACE = "! COSMORS(WATER)"


def make_calculator(xyz, directory, basename, keywords, cores):
    """Use OPI to combine the structure and calculation settings."""
    calc = Calculator(basename=basename, working_dir=directory)
    calc.structure = Structure.from_xyz(xyz, charge=0, multiplicity=1)
    calc.input.add_arbitrary_string(keywords)
    calc.input.ncores = cores
    # MB per process: approximately 6 GB with four cores.
    calc.input.memory = 1500
    return calc


def run_job(calc, product, optimize=False):
    """Run one stage, or reuse its completed output when the input is unchanged."""
    input_file = calc.working_dir / f"{calc.basename}.inp"
    output_file = calc.working_dir / f"{calc.basename}.out"
    previous = input_file.read_text() if input_file.exists() else None
    calc.write_input()

    # We can rerun the script after completing some molecules. A finished stage
    # is reused only if its input is identical and its expected output exists.
    # A changed or unsuccessful calculation is kept for inspection; --restart
    # moves the entire molecule folder to an archive before starting again.
    if output_file.exists():
        same_input = previous == input_file.read_text()
        if not same_input:
            if previous is not None:
                input_file.write_text(previous)
            raise RuntimeError(
                f"Input changed for {input_file}. Use --restart.")
        output = calc.get_output()
        converged = not optimize or output.geometry_optimization_converged()
        if output.terminated_normally() and converged and product.is_file():
            print(f"Reusing completed stage: {calc.basename}", flush=True)
            return
        raise RuntimeError(
            f"Inspect {output_file}; use --restart to repeat it.")

    print(f"Running {calc.basename}; output: {output_file}", flush=True)
    if not calc.run():
        raise RuntimeError(
            f"ORCA did not terminate normally. Inspect {output_file}")
    if optimize and not calc.get_output().geometry_optimization_converged():
        raise RuntimeError(
            f"Geometry optimization did not converge: {output_file}")
    if not product.is_file():
        raise FileNotFoundError(
            f"ORCA did not create the expected file: {product}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("molecules", nargs="*",
                        help="Names to run; default: all five")
    parser.add_argument("--cores", type=int, choices=range(1, 5), default=1)
    parser.add_argument("--orca", type=Path, default=ORCA)
    parser.add_argument("--restart", action="store_true",
                        help="Archive and repeat selected molecules")
    args = parser.parse_args()
    names = args.molecules or MOLECULES
    if len(set(names)) != len(names) or any(name not in MOLECULES for name in names):
        parser.error(f"Choose distinct names from: {', '.join(MOLECULES)}")
    orca = args.orca.expanduser().resolve()
    if not orca.is_file() or not os.access(orca, os.X_OK):
        parser.error(f"ORCA executable not found or not executable: {orca}")
    if args.cores > 1 and shutil.which("mpirun") is None:
        parser.error(
            "Parallel ORCA needs a compatible Open MPI installation; otherwise use --cores 1.")
    os.environ["OPI_ORCA"] = str(orca)

    # Each molecule is processed sequentially in its own calculation folder.
    # The first job optimizes the starting XYZ. The second job reads optimize.xyz,
    # so its COSMO surface is calculated from our newly optimized geometry.
    # The initial structures are ordinary molecular-mechanics starting geometries;
    # their preparation is documented in data/structures/README.md.
    for name in names:
        xyz = PROJECT / "data/structures" / f"{name}.xyz"
        if not xyz.is_file():
            raise FileNotFoundError(f"Missing starting geometry: {xyz}")
        directory = PROJECT / "calculations" / name
        if args.restart and directory.exists():
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            archived = PROJECT / "calculations/archive" / f"{name}_{stamp}"
            archived.parent.mkdir(parents=True, exist_ok=True)
            directory.rename(archived)
            print(f"Archived: {archived}")
        directory.mkdir(parents=True, exist_ok=True)
        optimized = directory / "optimize.xyz"
        print(f"\n{name} ({args.cores} core(s))", flush=True)
        calc = make_calculator(xyz, directory, "optimize",
                               OPTIMIZATION, args.cores)
        run_job(calc, optimized, optimize=True)

        surface_file = directory / f"{name}.solute.orcacosmo"
        calc = make_calculator(optimized, directory, name, SURFACE, args.cores)
        run_job(calc, surface_file)
        surface = SigmaProfileParser(str(surface_file))
        print(f"Surface: {surface_file}")
        print(
            f"Segments: {len(surface['seg_area'])}; area: {surface['area']:.4f} A^2")


if __name__ == "__main__":
    main()
