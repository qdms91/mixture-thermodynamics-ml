"""Project 4A: freeze predictions, calculate OOD labels, then compare them.

Three separate commands: prepare, search, report. No model is fitted here.
Run beside project2_common.py, mixture_thermodynamics.py, phase_stability.py
and project_config.py. Existing Project 2/3 files are read only.
"""

import os
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import redirect_stdout
from datetime import datetime, timezone
import fcntl
import hashlib
from importlib.metadata import PackageNotFoundError, version
from itertools import combinations, zip_longest
import json
import multiprocessing
from pathlib import Path
import signal
from time import perf_counter
import warnings

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import project2_common as common

ROOT = Path(__file__).resolve().parent
TEMPERATURE = 298.15
MODEL = None
THERMO = None
STABILITY = None


def require(condition, message):
    if not condition:
        raise ValueError(message)


def composition_key(x):
    return tuple(np.round(np.asarray(x, float), 12))


def make_recipes(source, per_group, binary_grid, seed):
    """Sample old continuous challenge compositions and add all ten binaries.

    We read only compositions and group names from the Project 1 manifest.
    Neither its energies nor any classifier output influences the sampling.
    """
    source = source.copy()
    require({"recipe_id", "test_group", *common.X_COLUMNS}.issubset(source.columns),
            "The supplied file must be the Project 1 recipes.csv manifest.")
    rng = np.random.default_rng(seed)
    groups = []
    for name in ["off_grid_in_domain", "mild_ood", "broad_ood"]:
        part = source[source.test_group == name].sort_values("recipe_id")
        require(len(part) >= per_group, f"Only {len(part)} recipes available in {name}.")
        chosen = rng.choice(len(part), per_group, replace=False)
        groups.append([{"test_group": name, "source_recipe_id": str(row.recipe_id),
                        **{c: float(row[c]) for c in common.X_COLUMNS}}
                       for _, row in part.iloc[chosen].iterrows()])
    named = source[source.test_group == "named_ood"].sort_values("recipe_id")
    groups.append([{"test_group": "named_ood", "source_recipe_id": str(row.recipe_id),
                    **{c: float(row[c]) for c in common.X_COLUMNS}}
                   for _, row in named.iterrows()])
    require(len(named) > 0, "Missing named OOD recipes.")

    if binary_grid == "pilot":
        fractions = [1e-6, 1e-5, 1e-4, .001, .005, .01, .05, .1, .2, .35,
                     .5, .65, .8, .9, .95, .99, .995, .999, .9999, .99999, .999999]
    else:
        trace = [1e-6, 1e-5, 1e-4, .001, .005]
        fractions = trace + [i / 100 for i in range(1, 100)] + [1-v for v in trace]
    fractions = sorted(fractions, key=lambda v: (abs(v-.5), v))
    for i, j in combinations(range(5), 2):
        rows = []
        name = f"binary_{common.MOLECULES[i]}_{common.MOLECULES[j]}"
        for value in fractions:
            x = np.zeros(5); x[i], x[j] = value, 1-value
            rows.append({"test_group": name, "source_recipe_id": "generated_binary",
                         **dict(zip(common.X_COLUMNS, x))})
        groups.append(rows)
    groups.append([{"test_group": "pure_reference_check", "source_recipe_id": "generated_pure",
                    **dict(zip(common.X_COLUMNS, x))} for x in np.eye(5)])

    # Interleave domains, so a short --limit run exercises several different
    # cases. --limit counts the first N total recipes, not N additional recipes.
    rows = [r for layer in zip_longest(*groups) for r in layer if r is not None]
    result = pd.DataFrame(rows)
    x = result[common.X_COLUMNS].to_numpy(float)
    require(np.isfinite(x).all() and (x >= 0).all()
            and np.allclose(x.sum(axis=1), 1, rtol=0, atol=1e-12), "Invalid composition.")
    require(len({composition_key(v) for v in x}) == len(x), "Duplicate challenge compositions.")
    result.insert(0, "recipe_id", [f"ood_{i:04d}" for i in range(1, len(result)+1)])
    result["recipe_number"] = np.arange(1, len(result)+1)
    result["present_components"] = (x > 0).sum(axis=1)
    result["max_bound_violation_mol_percent"] = 100*np.maximum(np.maximum(.15-x, x-.4), 0).max(axis=1)
    result["role"] = "exploratory_frozen_challenge"
    return result


def model_registry(project2, project3, budgets):
    """Keep existing model choices and thresholds; never select using OOD labels."""
    selections_path = project2 / "selections.json"
    thresholds_path = project3 / "report/frozen_thresholds.json"
    selections = json.loads(selections_path.read_text())
    thresholds = json.loads(thresholds_path.read_text())
    models = []
    for row in selections["models"]:
        if row["group"] == "a_composition" or (row["group"] == "b_molecular" and row["family"] == "mlp"):
            models.append({"model_id": f"p2_{row['group']}__{row['family']}",
                           "source_project": 2, "family": row["family"],
                           "source_training_labels": 1191, "source_seed": None,
                           "columns": row["columns"], "threshold": row["threshold"],
                           "model_file": str((project2 / row["model_file"]).resolve()),
                           "model_sha256": row["model_sha256"]})
    require(len(models) == 5, "Expected Project 2's four composition models and molecular MLP.")
    for row in thresholds:
        if row["strategy"] != "uncertainty_diversity" or row["training_labels"] not in budgets:
            continue
        folder = project3 / f"seed_{row['seed']}" / row["strategy"]
        state = json.loads((folder / "state.json").read_text())
        checkpoint = next(c for c in state["checkpoints"] if c["training_labels"] == row["training_labels"])
        models.append({"model_id": f"p3_active_n{row['training_labels']}_seed{row['seed']}",
                       "source_project": 3, "family": "mlp", "columns": common.COMPOSITION,
                       "source_training_labels": row["training_labels"], "source_seed": row["seed"],
                       "threshold": row["threshold"],
                       "model_file": str((folder / checkpoint["model_file"]).resolve()),
                       "model_sha256": checkpoint["model_sha256"]})
    for budget in budgets:
        require(any(m["source_project"] == 3 and m["source_training_labels"] == budget for m in models),
                f"No active-learning checkpoints found for budget {budget}.")
    require(len({m['model_id'] for m in models}) == len(models), "Duplicate model IDs.")
    inputs = {str(p.resolve()): common.sha256(p) for p in [selections_path, thresholds_path]}
    return models, inputs


def frozen_predictions(recipes, models):
    """Use saved scalers, weights and cutoffs. No .fit or calibration is called."""
    features = common.add_descriptors(recipes)
    rows = []
    for item in models:
        path = Path(item["model_file"])
        require(common.sha256(path) == item["model_sha256"], f"Source model changed: {path}")
        package = joblib.load(path)  # Only the user's own saved model packages.
        require(package["columns"] == item["columns"], "Saved feature order differs.")
        x = features[item["columns"]].to_numpy(float)
        with threadpool_limits(limits=1):
            if package.get("constant") is not None:
                score = np.full(len(x), float(package["constant"]))
            elif item["family"] == "svm":
                score = package["models"][0].decision_function(x)
            else:
                score = np.mean([m.predict_proba(x)[:, 1] for m in package["models"]], axis=0)
        require(np.isfinite(score).all(), "A source model returned nonfinite scores.")
        rows.append(pd.DataFrame({"recipe_id": recipes.recipe_id, "model_id": item["model_id"],
                                 "score": score, "threshold": item["threshold"],
                                 "predicted": (score >= item["threshold"]).astype(int)}))
    return pd.concat(rows, ignore_index=True)


def prepare(args):
    require(args.recipes is not None, "prepare requires --recipes PATH.")
    require(not args.run_dir.exists(), "Run directory exists. Use search/report, or choose a new directory.")
    recipes = make_recipes(pd.read_csv(args.recipes), args.per_group, args.binary_grid, args.seed)
    original, _, _ = common.load_prepared(args.prepared)
    old = {composition_key(v) for v in original[common.X_COLUMNS].to_numpy(float)}
    require(not any(composition_key(v) in old for v in recipes[common.X_COLUMNS].to_numpy(float)),
            "A challenge recipe duplicates an existing source recipe.")
    models, inputs = model_registry(args.project2_run, args.project3_run, args.p3_budgets)
    predictions = frozen_predictions(recipes, models)
    args.run_dir.mkdir(parents=True)
    common.write_csv(args.run_dir / "recipes.csv", recipes)
    common.write_csv(args.run_dir / "frozen_predictions.csv", predictions)
    common.write_json(args.run_dir / "models.json", models)
    inputs[str(args.recipes.resolve())] = common.sha256(args.recipes)
    inputs[str((args.prepared / 'preparation.json').resolve())] = common.sha256(args.prepared / 'preparation.json')
    common.write_json(args.run_dir / "protocol.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(), "temperature_K": TEMPERATURE,
        "sample_seed": args.seed, "per_group": args.per_group, "binary_grid": args.binary_grid,
        "role": "Exploratory frozen challenge; do not recycle these labels into a claimed independent final test.",
        "source_selection": "Project 2 composition families plus molecular MLP; Project 3 active committees at declared budgets, all available seeds. Budget 200 was highlighted after source-test inspection.",
        "inputs_sha256": inputs, "script_sha256": common.sha256(Path(__file__)),
        "common_sha256": common.sha256(Path(common.__file__)),
        "versions": {p: version(p) for p in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'joblib']},
        "frozen_sha256": {n: common.sha256(args.run_dir / n) for n in
                          ['recipes.csv', 'frozen_predictions.csv', 'models.json']}})
    print(recipes.groupby('test_group').size().to_string())
    print(f"\nSaved {len(recipes)} compositions and predictions from {len(models)} frozen models.")
    print("No reference calculations have been performed. Saved:", args.run_dir)


def validate_run(directory):
    protocol = json.loads((directory / "protocol.json").read_text())
    require(protocol["script_sha256"] == common.sha256(Path(__file__)), "Script changed; keep this run's version.")
    require(protocol["common_sha256"] == common.sha256(Path(common.__file__)), "Shared helper changed.")
    for filename, expected in protocol["frozen_sha256"].items():
        require(common.sha256(directory / filename) == expected, f"Frozen file changed: {filename}.")
    return protocol


def make_log_activity(model, present, thermo):
    """Only present solvents can appear in trial phases; absent entries stay zero."""
    names = np.asarray(common.MOLECULES)[present].tolist()

    def log_activity(w):
        full_x = np.zeros(5); full_x[present] = w
        table, _ = thermo.mixture_properties(model, full_x, TEMPERATURE)
        lng = table.set_index('molecule').loc[names, 'ln_gamma'].to_numpy(float)
        answer = np.log(w) + lng
        if not np.isfinite(answer).all():
            raise RuntimeError("Nonfinite log activity in reference calculation.")
        return answer
    return log_activity


def classify_search(best, trials, split, tolerance):
    numerical = np.isfinite(best) and np.isfinite(
        trials[['lowest_tpd_over_RT', 'final_tpd_over_RT']].to_numpy(float)).all()
    failed = int((~trials.converged).sum())
    if numerical and best < -tolerance and split is not None:
        return 1, 'instability_found', failed
    if numerical and best >= -tolerance and failed == 0:
        return 0, 'no_instability_found', failed
    return None, 'needs_review', failed


def initialize_worker():
    global MODEL, THERMO, STABILITY
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    import mixture_thermodynamics as thermo
    import phase_stability as stability
    THERMO, STABILITY = thermo, stability
    MODEL = thermo.build_model('calculated')
    MODEL.par.cosmospace_conv_thresh = 1e-10
    MODEL.par.cosmospace_max_iter = 3000


def calculate_reference(recipe, starts, seed, directory):
    """Reference worker receives a recipe, never classifier predictions or weights."""
    started = perf_counter()
    row = {**recipe, 'temperature_K': TEMPERATURE}
    x = np.asarray([recipe[c] for c in common.X_COLUMNS], float); x /= x.sum()
    present = x > 0
    z = x[present]
    if len(z) == 1:
        # At this T and pressure our composition model has no alternative
        # liquid composition for a pure component. Keep this as an endpoint
        # sanity check, separate from the nontrivial stability search metrics.
        return {**row, 'instability_label': 0, 'label_status': 'pure_component_check',
                'starts': 0, 'lowest_tpd_over_RT': 0., 'stability_seconds': perf_counter()-started}
    require((z >= STABILITY.FLOOR).all(), "Positive fraction below the reference solver floor.")
    log_activity = make_log_activity(MODEL, present, THERMO)
    digest = hashlib.sha256(np.asarray(x, dtype='<f8').tobytes()).digest()
    search_seed = int(np.random.SeedSequence([seed, int.from_bytes(digest[:4], 'little')]).generate_state(1)[0])
    with open(os.devnull, 'w') as quiet, redirect_stdout(quiet):
        best, best_w, trials = STABILITY.search_tpd(log_activity, z, starts, search_seed)
    split = STABILITY.check_small_split(log_activity, z, best_w) if best < -STABILITY.TPD_TOLERANCE else None
    if split is not None:
        fraction, remainder, change = split
        require((remainder >= 0).all() and np.allclose(fraction*best_w+(1-fraction)*remainder,
                z, rtol=0, atol=1e-12), "Trial split failed component mass balance.")
        require(change < 0, "Trial split does not lower free energy.")
    label, status, failed = classify_search(best, trials, split, STABILITY.TPD_TOLERANCE)
    row.update({'instability_label': label, 'label_status': status, 'starts': starts,
                'search_seed': search_seed, 'lowest_tpd_over_RT': float(best),
                'nonconverged_searches': failed})
    full_w = np.zeros(5); full_w[present] = best_w
    row.update({f'best_trial_x_{name}': float(v) for name, v in zip(common.MOLECULES, full_w)})
    if split is not None:
        full_r = np.zeros(5); full_r[present] = remainder
        row.update({'trial_fraction': float(fraction), 'split_change_kJ_mol': float(change),
                    **{f'remaining_x_{name}': float(v) for name, v in zip(common.MOLECULES, full_r)}})
    common.write_csv(Path(directory) / 'searches' / f"{recipe['recipe_id']}.csv", trials)
    row['stability_seconds'] = perf_counter()-started
    return row


def reference_protocol(starts, seed):
    import mixture_thermodynamics as thermo
    import phase_stability as stability
    import project_config as config
    from opencosmorspy import cosmors, parameterization
    require(list(config.MOLECULES) == common.MOLECULES and stability.TEMPERATURE == TEMPERATURE,
            "The established molecule order and 298.15 K are required.")
    paths = {f'surface:{n}': p for n, p in config.surface_paths('calculated').items()}
    paths.update({Path(m.__file__).name: Path(m.__file__) for m in [thermo, stability, config]})
    paths.update({f'opencosmorspy/{Path(m.__file__).name}': Path(m.__file__)
                  for m in [cosmors, parameterization]})
    versions = {}
    for name in ['numpy', 'pandas', 'scipy', 'opencosmorspy']:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = 'metadata_unavailable; source hashes recorded'
    return {'starts': starts, 'search_seed': seed, 'temperature_K': TEMPERATURE,
            'tpd_tolerance': stability.TPD_TOLERANCE, 'floor': stability.FLOOR,
            'cosmospace_conv_thresh': 1e-10, 'cosmospace_max_iter': 3000,
            'sha256': {n: common.sha256(p) for n, p in paths.items()},
            'versions': versions}


def export_references(directory):
    rows = [json.loads(p.read_text()) for p in sorted((directory/'references').glob('ood_*.json'))]
    if rows:
        data = pd.DataFrame(rows).sort_values('recipe_number')
        common.write_csv(directory / 'reference_labels.csv', data)
        return data
    return pd.DataFrame()


def search(args):
    validate_run(args.run_dir)
    reference = reference_protocol(args.starts, args.search_seed)
    path = args.run_dir / 'reference_protocol.json'
    if path.exists():
        require(json.loads(path.read_text()) == reference, 'Reference settings, code or surfaces changed.')
    else:
        common.write_json(path, reference)
    for name in ['references', 'searches']:
        (args.run_dir / name).mkdir(exist_ok=True)
    recipes = pd.read_csv(args.run_dir/'recipes.csv')
    selected = recipes.iloc[:args.limit] if args.limit is not None else recipes
    done = {p.stem for p in (args.run_dir/'references').glob('ood_*.json')}
    pending = selected[~selected.recipe_id.isin(done)].to_dict('records')
    print(f"Selected total: {len(selected)}; previously completed in selection: {len(selected)-len(pending)}; pending: {len(pending)}.", flush=True)
    print('Limit is a total prefix, including completed recipes.', flush=True)
    if pending:
        pool = ProcessPoolExecutor(max_workers=args.workers, initializer=initialize_worker,
                                   mp_context=multiprocessing.get_context('spawn'))
        todo, futures = iter(pending), {}
        started, completed = perf_counter(), 0

        def submit_next():
            recipe = next(todo, None)
            if recipe is not None:
                f = pool.submit(calculate_reference, recipe, args.starts, args.search_seed, str(args.run_dir))
                futures[f] = recipe

        def save_future(f):
            row = f.result()
            common.write_json(args.run_dir/'references'/f"{row['recipe_id']}.json", row)
            return row

        try:
            for _ in range(min(args.workers, len(pending))):
                submit_next()
            while futures:
                ready, _ = wait(futures, return_when=FIRST_COMPLETED)
                for f in ready:
                    futures.pop(f)
                    row = save_future(f)
                    completed += 1
                    eta = (perf_counter()-started)/completed*(len(pending)-completed)/3600
                    print(f"New references: {completed}/{len(pending)}; {row['recipe_id']}; {row['label_status']}; estimated remainder {eta:.2f} h", flush=True)
                    if completed % 10 == 0:
                        export_references(args.run_dir)
                    submit_next()
        except KeyboardInterrupt:
            print('\nStopping submissions; saving running jobs when they finish. Repeat this command to resume.', flush=True)
        finally:
            for f in futures:
                f.cancel()
            pool.shutdown(wait=True, cancel_futures=True)
            for f in futures:
                if not f.cancelled() and f.done() and f.exception() is None:
                    save_future(f)
            export_references(args.run_dir)
    data = export_references(args.run_dir)
    if len(data):
        print(data.label_status.value_counts().to_string())
    if len(data) == len(recipes):
        files = sorted((args.run_dir/'references').glob('ood_*.json'))
        common.write_json(args.run_dir/'references_complete.json', {
            'reference_protocol_sha256': common.sha256(path),
            'reference_labels_sha256': common.sha256(args.run_dir/'reference_labels.csv'),
            'rows_sha256': {p.name: common.sha256(p) for p in files}})
        print('Reference campaign complete. Run the report stage.')


def safe_metrics(y, score, threshold):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='No positive class found in y_true.*', category=UserWarning)
        values = common.metrics(y, score, threshold)
    if len(np.unique(y)) < 2:
        # Ranking quality needs both classes for this benchmark. In particular,
        # AP=1 in an all-positive group would give a misleading impression.
        for key in ['average_precision', 'pr_auc_trapezoid', 'roc_auc', 'balanced_accuracy']:
            values[key] = None
        values['ranking_note'] = 'One class only; ranking metrics omitted.'
    else:
        values['ranking_note'] = ''
    if np.sum(y) == 0:
        values['recall_instability'] = None
        values['f1_instability'] = None
    values['accuracy'] = (values['tp']+values['tn'])/len(y)
    negatives = values['tn']+values['fp']
    values['false_positive_rate'] = values['fp']/negatives if negatives else None
    return values


def report(args):
    validate_run(args.run_dir)
    require((args.run_dir/'references_complete.json').exists(), 'Complete the reference campaign before reporting.')
    completion = json.loads((args.run_dir/'references_complete.json').read_text())
    require(common.sha256(args.run_dir/'reference_labels.csv') == completion['reference_labels_sha256'],
            'Reference dataset changed after completion.')
    predictions = pd.read_csv(args.run_dir/'frozen_predictions.csv')
    references = pd.read_csv(args.run_dir/'reference_labels.csv')
    require(references.recipe_id.is_unique, 'Duplicate reference IDs.')
    joined = predictions.merge(references, on='recipe_id', how='left', validate='many_to_one')
    require(joined.label_status.notna().all(), 'A prediction has no reference result.')
    destination = args.run_dir/'report'; destination.mkdir(exist_ok=True)
    common.write_csv(destination/'predictions_with_references.csv', joined)
    coverage = references.groupby('test_group').agg(total=('recipe_id', 'count'),
                labelled=('instability_label', 'count'), positives=('instability_label', 'sum'))
    coverage['needs_review'] = coverage.total - coverage.labelled
    common.write_csv(destination/'label_coverage.csv', coverage.reset_index())
    rows = []
    for (model_id, group), part in joined.groupby(['model_id', 'test_group']):
        usable = part[part.instability_label.notna()]
        record = {'model_id': model_id, 'test_group': group, 'unresolved': len(part)-len(usable)}
        if len(usable):
            threshold = float(usable.threshold.iloc[0])
            record.update(safe_metrics(usable.instability_label.to_numpy(int), usable.score, threshold))
            record['threshold'] = threshold
        rows.append(record)
    metrics = pd.DataFrame(rows)
    common.write_csv(destination/'metrics_by_domain.csv', metrics)
    registry = pd.DataFrame(json.loads((args.run_dir/'models.json').read_text()))
    active = metrics.merge(registry[['model_id', 'source_project', 'source_training_labels']], on='model_id')
    active = active[active.source_project == 3]
    columns = ['average_precision', 'recall_instability', 'f1_instability', 'false_positive_rate', 'fp', 'fn']
    summary = active.groupby(['source_training_labels', 'test_group'])[columns].agg(['mean', 'std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    common.write_csv(destination/'project3_summary_across_seeds.csv', summary.reset_index())
    print(metrics[['model_id', 'test_group', 'recipes', 'average_precision', 'recall_instability', 'f1_instability', 'fn', 'fp', 'unresolved']].round(4).to_string(index=False))
    print('\nSaved:', destination)
    print('Fixed source thresholds; no fitting. Review unresolved labels before strong performance claims.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'search', 'report'])
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--recipes', type=Path)
    parser.add_argument('--prepared', type=Path, default=ROOT/'results/ml/project2_classification/prepared_seed2026')
    parser.add_argument('--project2-run', type=Path, default=ROOT/'results/ml/project2_classification/benchmark_v1')
    parser.add_argument('--project3-run', type=Path, default=ROOT/'results/ml/project3_active_learning/benchmark_v1')
    parser.add_argument('--p3-budgets', type=int, nargs='+', default=[200, 800])
    parser.add_argument('--per-group', type=int, default=40)
    parser.add_argument('--binary-grid', choices=['pilot', 'dense'], default='pilot')
    parser.add_argument('--seed', type=int, default=20261006)
    parser.add_argument('--starts', type=int, default=40)
    parser.add_argument('--search-seed', type=int, default=88)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    require(args.workers >= 1 and args.starts >= 16 and args.per_group >= 1
            and args.seed >= 0 and args.search_seed >= 0, 'Invalid count or seed.')
    require(args.limit is None or args.limit > 0, '--limit must be positive.')
    require(len(set(args.p3_budgets)) == len(args.p3_budgets), 'Duplicate source budgets.')
    args.run_dir = args.run_dir.resolve()
    if args.stage == 'prepare':
        prepare(args)
        return
    require(args.run_dir.is_dir(), 'Run prepare first.')
    with (args.run_dir/'run.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('This run is already in use.') from error
        if args.stage == 'search':
            search(args)
        else:
            report(args)


if __name__ == '__main__':
    main()
