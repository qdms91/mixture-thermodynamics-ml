# Reproducing the calculations and learning experiments

The workflow follows the scientific sequence described in the accompanying manuscript: molecular preparation, mixture thermodynamics, reference instability searches, P1 regression, P2 classification, P3 active learning, and P4 adaptation and replay. Commands are run from the repository root. The solvent order is always **water, methanol, acetone, acetonitrile, benzene**.

## Choose a starting point

| Interest | Starting materials | Follow |
| --- | --- | --- |
| Recalculate the chemistry and then train models | Starting XYZ files, ORCA, and the Python environment | Sections 1–9 |
| Train models using the recorded reference calculations | Cached datasets and their metadata, frozen splits, and generated surfaces | Sections 4–9; skip the reference calculations where indicated |
| Explore existing results or make predictions | Numerical tables and selected pretrained models | Sections 10–11 |

## 1. Software and project directory

Clone the repository and run the following commands from its root directory to create the environment:

```bash
conda env create -f environment.yml
conda activate cosmo-rs-1
python -m pip check
```

The workflows were developed on Ubuntu/Linux. Several campaign scripts use POSIX file locks. `environment.yml` records the project's installed direct dependencies, including pip packages. The scientific runs used **scikit-learn 1.8.0**, **orca-pi 2.0.0**, and openCOSMO-RS_py commit **`c2bc186762e4a2a3e6bb547fc9cbc917ed35d9b0`** with `openCOSMORS24a`. ORCA itself is installed separately.

For molecular regeneration, install **ORCA 6.1.1** following the [ORCA distribution and installation instructions](https://www.faccts.de/orca/). Set the ORCA executable path, for example:

```bash
export OPI_ORCA="$HOME/orca_6_1_1/orca"
python run_cosmo_calculations.py --help
```

Adjust that executable location to your installation, or pass `--orca /absolute/path/to/orca` to the calculation command. For `--cores` greater than one, install the MPI version compatible with your ORCA distribution. A single-core ORCA run avoids that additional requirement.

The examples below use three workers for reference-data collection and four for most learning/search jobs. The original ML benchmarks used eight workers. Worker count can be reduced for a smaller machine; preserve the scientific seeds, budgets, features, and optimization settings. Recorded durations will depend on the machine and concurrency.

## 2. Molecular geometries and COSMO surfaces

The five starting geometries are:

```text
data/structures/water.xyz
data/structures/methanol.xyz
data/structures/acetone.xyz
data/structures/acetonitrile.xyz
data/structures/benzene.xyz
```

Their preparation is described in `data/structures/README.md`. Run the complete molecular sequence with:

```bash
python run_cosmo_calculations.py --cores 1 --restart
```

`--restart` archives existing molecule calculation directories under `calculations/archive/` before regeneration. On a directory without earlier calculation files, it simply starts the jobs. Each molecule is processed sequentially: conductor-phase BP86/def2-TZVP geometry optimization with the specified cavity settings, followed by ORCA's `COSMORS(WATER)` surface workflow. The Python mixture calculations use the resulting solute surfaces:

```text
calculations/water/water.solute.orcacosmo
calculations/methanol/methanol.solute.orcacosmo
calculations/acetone/acetone.solute.orcacosmo
calculations/acetonitrile/acetonitrile.solute.orcacosmo
calculations/benzene/benzene.solute.orcacosmo
```

To skip the ORCA stage, use these five supplied generated surfaces at those exact paths. The downstream calculations use `project_config.py` to find them. The `calculated` surface set underlies the reported P1–P4 experiments; molecular regeneration can produce slightly different file bytes or numerical results on another software setup.

An optional earlier comparison used the authors' reference surfaces from the [openCOSMO-RS 24a paper's Supporting Information](https://www.sciencedirect.com/science/article/pii/S0378381224002255). Obtain the QM-file archive from its source and place these files inside `data/reference/openCOSMO-RS_24a_SI_data_qm_files/qm_files/`:

| Solvent | Reference filename |
| --- | --- |
| Water | `H2O_c000.orcacosmo` |
| Methanol | `methanol_c000.orcacosmo` |
| Acetone | `acetone_c000.orcacosmo` |
| Acetonitrile | `acetonitrile_CnfS1_c000.orcacosmo` |
| Benzene | `benzene_CnfS1_c000.orcacosmo` |

The calculated surfaces used in the study are provided as project inputs. The third-party reference collection is available from the cited source.

## 3. Equimolar thermodynamics and reference datasets

Calculate the equimolar two-, three-, four-, and five-component sequence:

```bash
python mixture_thermodynamics.py --source calculated
```

Results are written under `results/calculated/`, including `equimolar_thermodynamics.csv` and `equimolar_activities.csv`. The calculation evaluates activity coefficients and Gibbs mixing contributions at 298.15 K and estimates enthalpy and entropy with neighboring-temperature calculations. To repeat the optional reference-surface comparison after installing those inputs:

```bash
python mixture_thermodynamics.py --source reference
```

The ML reference campaigns use the calculated surfaces. If using the supplied datasets, continue to Section 4. To regenerate them, first preserve any existing campaign directories. The collector has fixed output names and resumes from its local database; a saved CSV alone is not a resumable calculation checkpoint.

Run this preservation step **only when choosing to regenerate the datasets**:

```bash
saved_runs=$(mktemp -d ../cosmo_rs_1_saved_reference_runs_XXXXXX)
for campaign_dir in \
    results/ml/thermo_seed88 \
    results/ml/stability_seed88_starts40 \
    results/ml/project2_stability/total2000_sample2026_search88_starts40
do
    if [ -d "$campaign_dir" ]; then
        mv -- "$campaign_dir" "$saved_runs/"
    fi
done
```

Then collect the thermodynamic grid and the initial 600 instability references, and extend the latter to 2,000 recipes:

```bash
python collect_ml_data.py --campaign thermo --seed 88 --workers 3
python collect_ml_data.py --campaign stability --starts 40 --seed 88 --workers 3
python extend_stability_dataset.py --total 2000 --sample-seed 2026 --workers 4
```

| Dataset | Output directory |
| --- | --- |
| 23,751 bounded-grid thermodynamic recipes | `results/ml/thermo_seed88/` |
| Initial 600 instability-search recipes | `results/ml/stability_seed88_starts40/` |
| Extended 2,000-recipe instability dataset | `results/ml/project2_stability/total2000_sample2026_search88_starts40/` |

The datasets include compositions, computed quantities, recipe identifiers, and recorded durations. The instability label follows a 40-start tangent-plane search and verification of an energy-lowering split. Here, 40 is the number of optimization starts. The initial source region contains all five components at 15–40 mol% each. Unresolved searches retain their review status in the collection stage.

Collection can resume by repeating the same command with unchanged inputs and environment. `--limit` denotes an absolute prefix of the recipe manifest, including recipes already completed. Partial collections are useful for checking an installation but do not reproduce the manuscript experiments.

## 4. P1 — Regression and extrapolation

Use a fresh output directory for a new fit:

```bash
P1_RUN=results/ml/project1_regression/reproduction_v1
python train_free_energy_models.py \
    --data results/ml/thermo_seed88/dataset.csv \
    --output "$P1_RUN" \
    --sizes 100 300 1000 3000 10000 \
    --seeds 42 88 123 --workers 4
```

The full training pool is added automatically. Models learn excess Gibbs energy from four independent mole fractions. The analytical ideal term is added to obtain total mixing Gibbs energy. The script saves fitted models, selected settings, learning curves, predictions, and validation/test metrics.

The original run is `results/ml/project1_regression/run_20261004_134348_552708/`. For downstream experiments using its saved models, assign that directory to `P1_RUN` instead.

To evaluate the frozen regression models on fresh off-grid, broader-composition, named, and water–benzene binary challenges:

```bash
python stress_test_free_energy.py \
    --model-run "$P1_RUN" \
    --data results/ml/thermo_seed88/dataset.csv \
    --samples 200 --seed 2026 --workers 4 \
    --output results/ml/project1_extrapolation/reproduction_v1
```

This command calculates new openCOSMO-RS references. The reference challenge and its results are retained in `results/ml/project1_extrapolation/run_20261004_153338_888728/` for inspection without repeating those calculations.

## 5. P2 — Instability classification

To retain the original train/validation/test splits, use the archived prepared dataset:

```bash
P2_PREPARED=results/ml/project2_classification/prepared_seed2026
P2_RUN=results/ml/project2_classification/reproduction_v1
```

The prepared directory includes the feature table, feature definitions, descriptor constants, frozen split, and `preparation.json`. Its train/validation/test sizes are **1,191/314/495**. All seven prepared files are supplied and required to load the original dataset. Published metadata paths are relative to the repository root, so run these commands from that directory. No local path editing is needed.

Alternatively, prepare a new split from the collected 2,000-recipe source using the same preparation implementation:

```bash
P2_PREPARED=results/ml/project2_classification/prepared_reproduction
python prepare_stability_classification.py \
    --source results/ml/project2_stability/total2000_sample2026_search88_starts40 \
    --output "$P2_PREPARED"
```

Train and evaluate:

```bash
python train_stability_classifiers.py \
    --prepared "$P2_PREPARED" --output "$P2_RUN" --workers 4
```

This runs the recorded feature groups and model families, grouped training cross-validation, validation threshold selection, and final test evaluation. `--describe` can be added to print the plan before fitting. To use the saved models for subsequent stages, set `P2_RUN=results/ml/project2_classification/benchmark_v1` instead.

## 6. P3 — Active learning

```bash
P3_RUN=results/ml/project3_active_learning/reproduction_v1
python run_active_learning.py \
    --prepared "$P2_PREPARED" --run-dir "$P3_RUN" \
    --seeds 11 22 33 44 55 --initial 100 --budgets 200 400 800 \
    --batch 25 --committee 3 --layers 128 64 --alpha 0.01 \
    --max-epochs 1200 --workers 4
python evaluate_active_learning.py \
    --run-dir "$P3_RUN" --prepared "$P2_PREPARED"
```

The experiment compares random, space-filling, and uncertainty/diversity acquisition, with a full-pool reference. It queries the already computed source labels retrospectively. Query order, revealed labels, per-round fitting information, checkpoints, and reporting thresholds are saved. Evaluation follows completion of the acquisition paths. To resume an interrupted learning run, repeat the learning command with `--resume` and identical settings.

The original run is `results/ml/project3_active_learning/benchmark_v1/`. The active-200, active-800, and full-pool reference models can be loaded from this directory when the corresponding model files are available. Set `P3_RUN` to this directory to reuse the saved models in the next stage.

## 7. P4 — Frozen out-of-domain challenge

The root file `project4_input_recipes.csv` supplies the P1 challenge recipe manifest used in the original experiment. The preparation script samples its off-grid, mild-OOD, and broad-OOD groups, retains its named examples, adds all ten binary systems and pure endpoints, and freezes predictions before reference labeling.

```bash
P4_FROZEN=results/ml/project4_frozen_ood/reproduction_v1
python stress_test_stability.py prepare \
    --recipes project4_input_recipes.csv --run-dir "$P4_FROZEN" \
    --prepared "$P2_PREPARED" \
    --project2-run "$P2_RUN" --project3-run "$P3_RUN" \
    --p3-budgets 200 800 --per-group 40 --binary-grid pilot --seed 20261006
python stress_test_stability.py search \
    --run-dir "$P4_FROZEN" --workers 4 --starts 40 --search-seed 88
python stress_test_stability.py report --run-dir "$P4_FROZEN"
```

The study’s frozen challenge contains **337 recipes**. Its aggregate labels, predictions, and domain metrics are under `results/ml/project4_frozen_ood/pilot_v1/`; detailed search and reference records can be obtained from the accompanying reference-search archive.

If your immediate interest is target adaptation using cached labels, the `prepare` command above is sufficient to create a fresh local model registry and the challenge composition set needed by adaptation. The challenge `search` and `report` commands can be skipped in that route. This also avoids relying on model paths copied from another machine.

## 8. P4 — Target adaptation

Prepare the 1,000-recipe target campaign, with 600/200/200 train/validation/test recipes:

```bash
P4_TARGET=results/ml/project4_adaptation/target_reproduction
python project4_adaptation.py prepare \
    --run-dir "$P4_TARGET" --frozen-run "$P4_FROZEN" \
    --prepared "$P2_PREPARED" --source p3 \
    --seed 20261008 --seeds 11 22 33 44 55
```

Choose **one** reference-label route. To calculate target labels:

```bash
python project4_adaptation.py search \
    --run-dir "$P4_TARGET" --workers 4 --starts 40 --search-seed 88
```

To reuse the completed reference target campaign, obtain its metadata and the individual `references/target_*.json` files from the supporting reference-data archive:

```bash
python project4_adaptation.py search \
    --run-dir "$P4_TARGET" --workers 4 --starts 40 --search-seed 88 \
    --reuse-labels results/ml/project4_adaptation/target_v1
```

The reuse command checks the complete reference protocol and recipe/split equality before importing labels. It needs the individual reference JSON files in addition to `dataset.csv`. Use the supplied calculated surfaces for this route so their recorded hashes match.

Then train and report:

```bash
python project4_adaptation.py train \
    --run-dir "$P4_TARGET" --experiment benchmark_v1 \
    --budgets 60 150 300 --batch 30 --epochs 1200 --workers 4
python project4_adaptation.py report \
    --run-dir "$P4_TARGET" --experiment benchmark_v1
```

The training stage evaluates scratch-random, scratch-active, transfer-random, transfer-active, scratch training on the transfer-active selections, and a full-pool scratch reference. The reference results are under `results/ml/project4_adaptation/target_v1/benchmark_v1/`.

## 9. P4 — Retention, energy replay, and the downstream bridge

The retention experiment uses the completed target-adaptation run and its exact transfer-active acquisition paths. It adds the P2 composition and molecular sources and the P3 active-200 sources. Replay uses 20 source examples and a 20% domain-weighted contribution.

```bash
RETENTION="$P4_TARGET/benchmark_v2_retention"
python project4_retention.py prepare \
    --run-dir "$RETENTION" --target-run "$P4_TARGET" \
    --previous-experiment benchmark_v1 --prepared "$P2_PREPARED" \
    --p2-run "$P2_RUN" --p3-run "$P3_RUN"
python project4_retention.py train --run-dir "$RETENTION" --stage 1 --workers 4
python project4_retention.py train --run-dir "$RETENTION" --stage 2 --workers 4
python project4_retention.py report --run-dir "$RETENTION" --stage all
```

Stages 1–2 establish the no-replay comparisons and duration/random replay comparisons, including matched scratch controls. Energy replay follows only after those fits and their report are complete:

```bash
ENERGY="$RETENTION/stage3_energy"
python project4_energy_replay.py prepare \
    --run-dir "$ENERGY" --retention-run "$RETENTION" --ridge-run "$P1_RUN"
python project4_energy_replay.py train --run-dir "$ENERGY" --workers 4
python project4_energy_replay.py report --run-dir "$ENERGY"
```

The Ridge model predicts excess Gibbs energy; the exact ideal term is added before ordering the eligible source examples. The strict bridge selects ten positives immediately preceding the first negative in descending energy order, followed by the first ten negatives from that anchor downward. The original P3 source pools supported this strict rule only for seed 55. Infeasible seeds are recorded rather than replaced with a different rule.

The later P3-only extension uses the first ten eligible positives in descending energy order, allowing intervening negatives, and then the first ten eligible negatives **after the tenth selected positive**. Its balanced random control uses the same eligible pool:

```bash
BRIDGE="$RETENTION/stage3_bridge_extension"
python project4_bridge_extension.py prepare --run-dir "$BRIDGE" --stage3-run "$ENERGY"
python project4_bridge_extension.py train --run-dir "$BRIDGE" --workers 4
python project4_bridge_extension.py report --run-dir "$BRIDGE"
```

The strict and downstream rules are separate evaluated policies. The extension is exploratory and was conducted after inspection of Stage 3. Its original workload consisted of 54 new committee fits and six reused control fits. Feasibility, reuse, and outcomes can change when upstream models are retrained; the archived manifests identify the exact original memories and target examples.

The original retention directory is `results/ml/project4_adaptation/target_v1/benchmark_v2_retention/`. Selected source models and the Ridge snapshot remain at the paths expected by the stage scripts. Additional fitted replay checkpoints and their `record.json` files are distributed separately where available. They are needed when inspecting individual fitted replay models or rerunning complete saved-run reports.

## 10. Explore saved models and numerical results

The stored results can be read directly without rerunning a campaign. These are the main numerical entry points:

| Experiment | Result file relative to the root |
| --- | --- |
| Equimolar mixtures | `results/calculated/equimolar_thermodynamics.csv` |
| P1 held-out errors | `results/ml/project1_regression/run_20261004_134348_552708/test_metrics.csv` |
| P1 extrapolation | `results/ml/project1_extrapolation/run_20261004_153338_888728/metrics_by_group.csv` |
| P2 classifier comparison | `results/ml/project2_classification/benchmark_v1/test_metrics.csv` |
| P3 learning comparison | `results/ml/project3_active_learning/benchmark_v1/report/summary.csv` |
| P4 frozen challenge | `results/ml/project4_frozen_ood/pilot_v1/report/metrics_by_domain.csv` |
| P4 target adaptation | `results/ml/project4_adaptation/target_v1/benchmark_v1/report/summary.csv` |
| Duration replay | `results/ml/project4_adaptation/target_v1/benchmark_v2_retention/report_stage12/summary.csv` |
| Energy replay | `results/ml/project4_adaptation/target_v1/benchmark_v2_retention/stage3_energy/report/summary.csv` |
| Downstream bridge | `results/ml/project4_adaptation/target_v1/benchmark_v2_retention/stage3_bridge_extension/report/summary.csv` |

For example, predict the equimolar mixing energy with the saved P1 Ridge model:

```python
from pathlib import Path
import joblib
import numpy as np
from scipy.constants import R
from scipy.special import xlogy
run = Path("results/ml/project1_regression/run_20261004_134348_552708")
model = joblib.load(run / "ridge_model.joblib")
x = np.array([[0.2, 0.2, 0.2, 0.2, 0.2]])
g_excess = model.predict(x[:, :4])
g_mix = g_excess + R * 298.15 / 1000 * xlogy(x, x).sum(axis=1)
print("g_excess, g_mix [kJ/mol]:", g_excess, g_mix)
```

The saved classifiers include fitted preprocessing. For a P2 molecular committee:

```python
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import project2_common as common
run = Path("results/ml/project2_classification/benchmark_v1")
selected = json.loads((run / "selections.json").read_text())["models"]
entry = next(r for r in selected if r["group"] == "b_molecular" and r["family"] == "mlp")
package = joblib.load(run / entry["model_file"])
recipes = pd.DataFrame([[0.2] * 5], columns=common.X_COLUMNS)
features = common.add_descriptors(recipes)[entry["columns"]].to_numpy(float)
score = np.mean([m.predict_proba(features)[:, 1] for m in package["models"]], axis=0)
print("score:", score, "predicted instability:", score >= entry["threshold"])
```

Use the pinned environment when loading these scikit-learn models. Source labels mean detected instability under the recorded reference-search protocol. Average precision summarizes ranking; recall and F1 use the stored validation-selected cutoff. When comparing replay policies, match source family, seed, target budget, scope, initialization, and threshold rule.

## 11. Costs, plots, and checks

Numerical manuscript tables are retained under `Mixture_Thermodynamics_Manuscript_Bundle/Mixture_Thermodynamics_Manuscript/tables/`. The bridge-extension cost correction is retained alongside them under `cost_correction/`. It attributes only new fits to the extension and counts reused fits once. To regenerate that derived accounting into a fresh directory:

```bash
python Mixture_Thermodynamics_Manuscript_Bundle/Mixture_Thermodynamics_Manuscript/cost_correction/correct_incremental_costs.py \
    --costs results/ml/project4_adaptation/target_v1/benchmark_v2_retention/stage3_bridge_extension/report/costs.csv \
    --output results/ml/project4_adaptation/target_v1/bridge_costs_reproduction
```

Search-duration sums and recorded per-fit durations describe resource accounting; elapsed campaign wall time also depends on concurrency and machine setup.

The training and reporting scripts generate their own diagnostic plots. Molecular sigma-profile plots can be regenerated with:

```bash
python plot_sigma_profiles.py --source calculated
```

Manuscript figure assembly is separate from the scientific workflows documented here. The underlying numerical tables are provided.

To run the existing checks after installation:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg \
python -m unittest -v \
    test_project2_workflow test_project3_active_learning \
    test_project4_adaptation test_project4_retention \
    test_project4_energy_replay test_project4_bridge_extension \
    test_stress_test_stability
```

Output directories are created by the scripts. For fresh experiments, choose unused `reproduction_v1` directories or another run name. For archived experiments, preserve their frozen datasets, configurations, and integrity checks. Extract accompanying data or model archives into the documented root-relative paths so the scripts can locate the required files.

The metadata portability update restores 18 original records using repository-relative paths and updates the five linked completion/report manifests. `metadata_portability.json` records the original and public file hashes; `RUN_SETTINGS.json` retains its original provenance hashes. Scientific settings, scripts, datasets, model weights, and numerical results are unchanged. Metadata files with hash references must be kept together at the same revision. The v1.0.0 supplementary archives remain compatible with these metadata updates.

The cloned repository contains the files needed to load the P2 prepared data, validate the P3/P4 protocols, and start fresh ML runs from the published datasets. Importing cached target labels with `--reuse-labels` also requires the target reference JSON files from `reference_searches.zip`. Re-evaluating every historical P3 checkpoint or replay fit requires the additional models archive. Incomplete-run resumption retains the scripts' original strict checks on paths, code, and software; use fresh run directories for new training on another machine.
