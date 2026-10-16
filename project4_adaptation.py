"""Project 4B: prepare, label, learn, and evaluate a frozen target benchmark."""

import os
for variable in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS']:
    os.environ[variable] = '1'
os.environ.setdefault('MPLBACKEND', 'Agg')

import argparse
import fcntl
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from importlib.metadata import version
from itertools import combinations, zip_longest
import json
import multiprocessing
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.spatial.distance import cdist

import project2_common as common
import project3_common as active
import project4_learning as learning
import stress_test_stability as reference

ROOT = Path(__file__).resolve().parent
DEFAULT = ROOT / 'results/ml/project4_adaptation/target_v1'
ARMS = ['scratch_random', 'transfer_random', 'scratch_active', 'transfer_active']
require = learning.require


def read_json(path):
    return json.loads(Path(path).read_text())


def code_hashes():
    modules = [common, active, learning, reference]
    paths = [Path(__file__)] + [Path(m.__file__) for m in modules]
    return {p.name: common.sha256(p) for p in paths}


def check_run(directory):
    protocol = read_json(directory / 'protocol.json')
    require(protocol['code_sha256'] == code_hashes(), 'Code changed. Keep the frozen code with this run.')
    for name, digest in protocol['files_sha256'].items():
        require(common.sha256(directory / name) == digest, f'Frozen file changed: {name}')
    return protocol


def split_blocks(count, rng, binary=False):
    # Entire local composition blocks belong to one split. The assignment uses
    # geometry only; no stability label influences it. Binary test/validation
    # each receive an interior block and a more dilute block.
    roles = np.full(count, 'train', dtype=object)
    if binary:
        inner = rng.permutation([3, 4, 5, 6])
        outer = rng.permutation([0, 1, 2, 7, 8, 9])
        roles[[inner[0], outer[0]]] = 'test'
        roles[[inner[1], outer[1]]] = 'validation'
    else:
        order = rng.permutation(count)
        roles[order[:count // 5]] = 'test'
        roles[order[count // 5:2 * count // 5]] = 'validation'
    return roles


def make_target(seed):
    """1000 recipes: 600 train, 200 validation, 200 test, with disjoint blocks."""
    rng = np.random.default_rng(seed)
    rows = []
    for n in [2, 3, 4, 5]:
        for subset in combinations(range(5), n):
            names = [common.MOLECULES[i] for i in subset]
            domain = '_'.join(names)
            blocks, per_block = (50, 6) if n == 5 else (10, 4 if n == 2 else 2)
            roles = split_blocks(blocks, rng, binary=n == 2)
            anchors = None
            if n > 2:
                anchors = np.vstack([rng.dirichlet(np.full(n, [.25, .5, 1., 2.][j % 4]))
                                     for j in range(blocks)])
            for block in range(blocks):
                if n == 2:
                    # Equal logit intervals resolve both trace limits without
                    # pretending a uniform mole-fraction grid would do so.
                    edges = np.linspace(-np.log(999999.), np.log(999999.), 11)
                    first = expit(rng.uniform(edges[block], edges[block + 1], per_block))
                    fractions = np.column_stack([first, 1 - first])
                else:
                    fractions = []
                    while len(fractions) < per_block:
                        trial = rng.dirichlet(anchors[block] * 150 + .05)
                        trial = 1e-6 + (1 - n * 1e-6) * trial
                        # Assign to the nearest anchor, so local Voronoi blocks
                        # cannot cross roles. No thermodynamic values are used.
                        if np.argmin(((anchors - trial) ** 2).sum(axis=1)) == block:
                            fractions.append(trial)
                    fractions = np.asarray(fractions)
                for fraction in fractions:
                    x = np.zeros(5); x[list(subset)] = fraction
                    number = len(rows)
                    rows.append({'recipe_id': f'target_{number:04d}', 'recipe_number': number,
                                 'test_group': domain, 'components': n,
                                 'block_id': f'{domain}__{block:02d}', 'split': roles[block],
                                 **dict(zip(common.X_COLUMNS, x))})
    # Interleave component subsets so a short pilot includes binary, ternary,
    # quaternary and five-component work rather than only the first binary.
    domains = list(dict.fromkeys(r['test_group'] for r in rows))
    groups = [[r for r in rows if r['test_group'] == domain] for domain in domains]
    rows = [r for layer in zip_longest(*groups) for r in layer if r is not None]
    for number, row in enumerate(rows):
        row['recipe_id'], row['recipe_number'] = f'target_{number:04d}', number
    return pd.DataFrame(rows)


def prepare(args):
    require(version('scikit-learn') == '1.8.0', 'Use scikit-learn 1.8.0 with the existing source models.')
    require(not args.run_dir.exists(), 'Output already exists. Resume another stage; do not prepare again.')
    old = args.frozen_run
    reference.validate_run(old)
    registry = read_json(old / 'models.json')
    ids = [f'p3_active_n200_seed{s}' for s in args.seeds] if args.source == 'p3' else ['p2_a_composition__mlp']
    sources = [r for r in registry if r['model_id'] in ids]
    require(len(sources) == len(ids), 'Requested source models are missing from models.json.')
    for row in sources:
        require(common.sha256(row['model_file']) == row['model_sha256'], 'Source weights changed.')
        package = joblib.load(row['model_file'])
        learning.source_networks(package)  # Check architecture before any campaign.
    recipes = make_target(args.seed)
    source_data, _, _ = common.load_prepared(args.prepared)
    source_keys = {tuple(v) for v in source_data[common.X_COLUMNS].round(12).to_numpy()}
    require(not any(tuple(v) in source_keys for v in recipes[common.X_COLUMNS].round(12).to_numpy()),
            'A target recipe duplicates a source recipe.')
    retention = source_data.loc[source_data.split == 'test',
                                ['recipe_id', *common.X_COLUMNS, 'instability_label']].copy()
    previous = pd.read_csv(old / 'recipes.csv')
    old_keys = {tuple(v) for v in previous[common.X_COLUMNS].round(12).to_numpy()}
    require(not any(tuple(v) in old_keys for v in recipes[common.X_COLUMNS].round(12).to_numpy()),
            'A target recipe duplicates a previously inspected challenge recipe.')
    require(recipes.groupby('block_id').split.nunique().max() == 1, 'A block crosses splits.')
    require(recipes.split.value_counts().to_dict() == {'train': 600, 'validation': 200, 'test': 200},
            'Unexpected split sizes.')
    require(recipes[common.X_COLUMNS].round(12).drop_duplicates().shape[0] == len(recipes),
            'Duplicate target compositions.')
    predictions = reference.frozen_predictions(recipes, sources)
    args.run_dir.mkdir(parents=True)
    common.write_csv(args.run_dir / 'recipes.csv', recipes)
    common.write_csv(args.run_dir / 'frozen_predictions.csv', predictions)
    common.write_csv(args.run_dir / 'source_retention.csv', retention)
    common.write_csv(args.run_dir / 'source_retention_frozen.csv', reference.frozen_predictions(retention, sources))
    distances = recipes[['recipe_id', 'split', 'components', 'test_group']].copy()
    x = recipes[common.X_COLUMNS].to_numpy(float)
    distances['nearest_train_distance'] = cdist(x, x[recipes.split == 'train']).min(axis=1)
    common.write_csv(args.run_dir / 'composition_distances.csv', distances)
    common.write_json(args.run_dir / 'sources.json', sources)
    common.write_json(args.run_dir / 'protocol.json', {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'sample_seed': args.seed,
        'source': args.source, 'seeds': args.seeds, 'temperature_K': 298.15,
        'role': 'New target block split; design informed by the inspected 337-recipe challenge.',
        'prior_challenge': str(old.resolve()),
        'source_validation_labels': int((source_data.split == 'validation').sum()),
        'prior_recipes_sha256': common.sha256(old / 'recipes.csv'),
        'code_sha256': code_hashes(), 'sklearn_version': version('scikit-learn'),
        'files_sha256': {name: common.sha256(args.run_dir / name) for name in
                         ['recipes.csv', 'frozen_predictions.csv', 'sources.json',
                          'source_retention.csv', 'source_retention_frozen.csv', 'composition_distances.csv']}})
    print(recipes.groupby(['components', 'split']).size().unstack().to_string())
    print('Saved 1000 fresh recipes and source predictions. No labels were calculated.')
    print('Output:', args.run_dir)


def export_labels(directory):
    rows = [read_json(p) for p in sorted((directory / 'references').glob('target_*.json'))]
    if not rows:
        return pd.DataFrame()
    data = pd.DataFrame(rows).sort_values('recipe_number')
    common.write_csv(directory / 'dataset.csv', data)
    return data


def search(args):
    check_run(args.run_dir)
    protocol = reference.reference_protocol(args.starts, args.search_seed)
    path = args.run_dir / 'reference_protocol.json'
    if path.exists():
        require(read_json(path) == protocol, 'Reference settings/code/surfaces changed.')
    else:
        common.write_json(path, protocol)
    for name in ['references', 'searches']:
        (args.run_dir / name).mkdir(exist_ok=True)
    recipes = pd.read_csv(args.run_dir / 'recipes.csv')
    if args.reuse_labels is not None:
        old_data, _ = load_complete(args.reuse_labels)
        require(read_json(args.reuse_labels / 'reference_protocol.json') == protocol,
                'Cached reference settings differ.')
        previous = pd.read_csv(args.reuse_labels / 'recipes.csv')
        require(recipes.equals(previous), 'Cached recipes/splits differ.')
        for row in old_data.to_dict('records'):
            source = args.reuse_labels / 'references' / f"{row['recipe_id']}.json"
            destination = args.run_dir / 'references' / source.name
            if destination.exists():
                require(common.sha256(destination) == common.sha256(source), 'Conflicting cached reference.')
            else:
                common.write_json(destination, read_json(source))
        print('Reused the identical completed reference campaign; no new calculations needed.')
    chosen = recipes.iloc[:args.limit] if args.limit is not None else recipes
    done = {p.stem for p in (args.run_dir / 'references').glob('target_*.json')}
    pending = chosen[~chosen.recipe_id.isin(done)].to_dict('records')
    print(f'Selected total: {len(chosen)}; already complete: {len(chosen)-len(pending)}; pending: {len(pending)}.', flush=True)
    # Submit at most one recipe per worker. An interrupt finishes those jobs,
    # saves them, and stops. --limit means total rows, including existing rows.
    pool = ProcessPoolExecutor(max_workers=args.workers, initializer=reference.initialize_worker,
                               mp_context=multiprocessing.get_context('spawn'))
    iterator, futures = iter(pending), {}
    started, completed = perf_counter(), 0

    def submit():
        recipe = next(iterator, None)
        if recipe is not None:
            f = pool.submit(reference.calculate_reference, recipe, args.starts,
                            args.search_seed, str(args.run_dir))
            futures[f] = recipe['recipe_id']

    def save(f):
        row = f.result()
        common.write_json(args.run_dir / 'references' / f"{row['recipe_id']}.json", row)

    try:
        for _ in range(min(len(pending), args.workers)):
            submit()
        while futures:
            f = next(as_completed(futures))
            futures.pop(f); save(f); completed += 1
            hours = (perf_counter() - started) / completed * (len(pending) - completed) / 3600
            print(f'New labels: {completed}/{len(pending)}; estimated remainder {hours:.2f} h', flush=True)
            if completed % 10 == 0:
                export_labels(args.run_dir)
            submit()
    except KeyboardInterrupt:
        print('Stopping after running jobs finish; repeat the same command to resume.', flush=True)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        for f in futures:
            if f.done() and not f.cancelled() and f.exception() is None:
                save(f)
        data = export_labels(args.run_dir)
    if len(data):
        print(data.label_status.value_counts().to_string())
    if len(data) == len(recipes):
        common.write_json(args.run_dir / 'labels_complete.json', {
            'dataset_sha256': common.sha256(args.run_dir / 'dataset.csv'),
            'reference_protocol_sha256': common.sha256(path)})
        print('All references saved. Ready for train.')


def load_complete(directory):
    protocol = check_run(directory)
    complete = read_json(directory / 'labels_complete.json')
    require(complete['dataset_sha256'] == common.sha256(directory / 'dataset.csv'), 'Labels changed.')
    require(complete['reference_protocol_sha256'] == common.sha256(directory / 'reference_protocol.json'),
            'Reference protocol changed.')
    data = pd.read_csv(directory / 'dataset.csv')
    require(len(data) == 1000 and data.recipe_id.is_unique, 'Reference campaign is incomplete.')
    unresolved = data[~data.instability_label.isin([0, 1])]
    # Never silently turn numerical failures into negative labels, or silently
    # discard difficult test points. Inspection is required before benchmarking.
    require(unresolved.empty,
            'Unresolved references: ' + ', '.join(unresolved.recipe_id) +
            '. Inspect their searches before training; do not edit labels by hand.')
    return data, protocol


def atomic_model(path, package):
    temporary = path.with_suffix('.tmp')
    joblib.dump(package, temporary)
    temporary.replace(path)


def path_run(directory, arm, seed, pool, sources, config, matched=None):
    folder = directory / f'seed_{seed}' / arm
    folder.mkdir(parents=True, exist_ok=True)
    x = pool[common.X_COLUMNS].to_numpy(float)
    oracle = active.LabelOracle(pool.instability_label.to_numpy(int))
    selected = active.initial_positions(len(pool), config['budgets'][0], seed)
    records = []
    budget = config['budgets'][0]
    while True:
        if matched is not None:
            selected = read_json(matched / f'round_{budget:04d}.json')['positions']
        stamp = folder / f'round_{budget:04d}.json'
        model_file = folder / f'model_{budget:04d}.joblib'
        if stamp.exists():
            record = read_json(stamp)
            require(record['positions'] == selected, 'Saved acquisition path disagrees with replay.')
            require(common.sha256(model_file) == record['model_sha256'], 'Saved model changed.')
            package = joblib.load(model_file)
        else:
            labels = oracle.reveal(selected)
            package = learning.fit_committee(x[selected], labels, sources,
                                             arm.startswith('transfer'), seed, config)
            atomic_model(model_file, package)
            record = {'arm': arm, 'seed': seed, 'budget': budget, 'positions': selected.copy(),
                      'recipe_ids': pool.iloc[selected].recipe_id.tolist(),
                      'positive_labels': int(labels.sum()),
                      'label_seconds': float(pool.iloc[selected].stability_seconds.sum()),
                      'fit_seconds': package['fit_seconds'], 'acquisition_seconds': 0.,
                      'diagnostics': package['diagnostics'],
                      'model_file': str(model_file.relative_to(directory)),
                      'model_sha256': common.sha256(model_file)}
            common.write_json(stamp, record)
        records.append(record)
        print(f'Seed {seed}: {arm}, {budget} target labels', flush=True)
        if budget == config['budgets'][-1]:
            break
        next_budget = min(budget + config['batch'],
                          next(b for b in config['budgets'] if b > budget))
        if matched is None:
            if 'next_positions' in record:
                selected = record['next_positions']
            else:
                strategy = 'random' if arm.endswith('random') else 'active'
                started = perf_counter()
                selected = selected + learning.select_next(strategy, x, selected,
                            next_budget - budget, seed, package)
                record['acquisition_seconds'] = perf_counter() - started
                record['next_positions'] = selected
                common.write_json(stamp, record)
        budget = next_budget
    return records


def train_seed(directory, pool, source_row, seed, config):
    # Different seeds run in different processes. Each process fits its own
    # networks with one numerical thread; no target labels cross path selection.
    directory = Path(directory)
    require(common.sha256(source_row['model_file']) == source_row['model_sha256'], 'Source changed.')
    source_package = joblib.load(source_row['model_file'])
    sources = learning.source_networks(source_package)
    records = []
    for arm in ARMS:
        records.extend(path_run(directory, arm, seed, pool, sources, config))
    records.extend(path_run(directory, 'scratch_matched_transfer_active', seed, pool, sources,
                            config, directory / f'seed_{seed}/transfer_active'))
    full = directory / f'seed_{seed}/full_pool_scratch'
    full.mkdir(parents=True, exist_ok=True)
    model_file, stamp = full / 'model.joblib', full / 'record.json'
    if not stamp.exists():
        package = learning.fit_committee(pool[common.X_COLUMNS].to_numpy(float),
                        pool.instability_label.to_numpy(int), sources, False, seed, config)
        atomic_model(model_file, package)
        common.write_json(stamp, {'arm': 'full_pool_scratch', 'seed': seed, 'budget': len(pool),
            'positions': list(range(len(pool))), 'recipe_ids': pool.recipe_id.tolist(),
            'positive_labels': int(pool.instability_label.sum()),
            'label_seconds': float(pool.stability_seconds.sum()),
            'fit_seconds': package['fit_seconds'], 'diagnostics': package['diagnostics'],
            'acquisition_seconds': 0.,
            'model_file': str(model_file.relative_to(directory)), 'model_sha256': common.sha256(model_file)})
    full_record = read_json(stamp)
    require(common.sha256(model_file) == full_record['model_sha256'], 'Full-pool model changed.')
    records.append(full_record)
    common.write_json(directory / f'seed_{seed}/complete.json', records)
    return seed


def train(args):
    data, protocol = load_complete(args.run_dir)
    require(version('scikit-learn') == protocol['sklearn_version'], 'Use the frozen sklearn version.')
    require(version('scikit-learn') == '1.8.0', 'This transfer adapter was verified with scikit-learn 1.8.0.')
    sources = read_json(args.run_dir / 'sources.json')
    seeds = protocol['seeds']
    pool = data[data.split == 'train'].reset_index(drop=True)
    require(pool.instability_label.nunique() == 2, 'Target pool lacks one class.')
    config = {'budgets': args.budgets, 'batch': args.batch, 'epochs': args.epochs,
              'min_epochs': min(100, args.epochs), 'patience': 60, 'tol': 1e-5,
              'alpha': .01, 'learning_rate': .001, 'seeds': seeds,
              'dataset_sha256': common.sha256(args.run_dir / 'dataset.csv'),
              'source': protocol['source'], 'code_sha256': code_hashes()}
    require(sorted(set(args.budgets)) == args.budgets and 2 <= args.budgets[0]
            and args.budgets[-1] < len(pool), 'Use increasing target budgets below pool size.')
    require(args.batch > 0 and args.epochs > 0, 'Batch and epochs must be positive.')
    directory = args.run_dir / args.experiment
    directory.mkdir(exist_ok=True)
    path = directory / 'training_protocol.json'
    if path.exists():
        require(read_json(path) == config, 'Training settings changed. Choose a new --experiment name.')
    else:
        common.write_json(path, config)
    # Only training rows enter workers. Validation and test labels are never
    # supplied to model fitting or acquisition; evaluation is a separate stage.
    with ProcessPoolExecutor(max_workers=min(args.workers, len(seeds)),
                             mp_context=multiprocessing.get_context('spawn')) as executor:
        jobs = []
        for seed in seeds:
            row = next((r for r in sources if r['source_seed'] == seed), sources[0])
            jobs.append(executor.submit(train_seed, str(directory), pool, row, seed, config))
        for f in as_completed(jobs):
            print('Completed seed:', f.result(), flush=True)
    common.write_json(directory / 'training_complete.json', {
        'training_protocol_sha256': common.sha256(path),
        'seed_sha256': {str(s): common.sha256(directory / f'seed_{s}/complete.json') for s in seeds}})
    print('Paths are complete. Run report once to freeze validation cutoffs and evaluate test.')


def report(args):
    data, protocol = load_complete(args.run_dir)
    directory = args.run_dir / args.experiment
    config = read_json(directory / 'training_protocol.json')
    require(config['code_sha256'] == code_hashes(), 'Training code changed.')
    require(config['dataset_sha256'] == common.sha256(args.run_dir / 'dataset.csv'), 'Dataset changed.')
    complete = read_json(directory / 'training_complete.json')
    require(complete['training_protocol_sha256'] == common.sha256(directory / 'training_protocol.json'),
            'Training protocol changed.')
    final_out = directory / 'report'
    require(not final_out.exists(), 'Report already exists. Preserve it; do not retune on its test results.')
    out = directory / 'report_in_progress'
    validation = data[data.split == 'validation'].copy()
    require(validation.instability_label.nunique() == 2, 'Validation has only one class; thresholds need review.')
    xval = validation[common.X_COLUMNS].to_numpy(float)
    fixed, costs = [], []
    for seed in config['seeds']:
        path = directory / f'seed_{seed}/complete.json'
        require(common.sha256(path) == complete['seed_sha256'][str(seed)], 'Seed manifest changed.')
        history = read_json(path)
        for record in history:
            if record['budget'] not in config['budgets'] and record['arm'] != 'full_pool_scratch':
                continue
            model_file = directory / record['model_file']
            require(common.sha256(model_file) == record['model_sha256'], 'A fitted model changed.')
            score = learning.scores(joblib.load(model_file), xval).mean(axis=0)
            threshold = common.choose_threshold(validation.instability_label, score)
            fixed.append({**record, 'threshold': threshold})
            preceding = [r for r in history if r['arm'] == record['arm'] and r['budget'] <= record['budget']]
            costs.append({'seed': seed, 'arm': record['arm'], 'training_labels': record['budget'],
                          'validation_labels': len(validation), 'test_labels': int((data.split == 'test').sum()),
                          'source_training_labels': 200 if protocol['source'] == 'p3' else 1191,
                          'source_validation_labels': protocol['source_validation_labels'],
                          'target_label_seconds': record['label_seconds'],
                          'target_validation_seconds': float(validation.stability_seconds.sum()),
                          'cumulative_fit_seconds': sum(r['fit_seconds'] for r in preceding),
                          'cumulative_acquisition_seconds': sum(r['acquisition_seconds'] for r in preceding
                                                               if r['budget'] < record['budget']),
                          'epoch_cap_members': sum(d['epoch_cap_reached'] for d in record['diagnostics'])})
    # A zero-shot baseline keeps the original cutoff. Threshold-only adaptation
    # uses target validation labels but performs no neural-network updates.
    frozen = pd.read_csv(args.run_dir / 'frozen_predictions.csv')
    sources = read_json(args.run_dir / 'sources.json')
    for source in sources:
        v = validation[['recipe_id', 'instability_label']].merge(
                frozen[frozen.model_id == source['model_id']], on='recipe_id', validate='one_to_one')
        for mode in ['zero_shot', 'threshold_only']:
            threshold = source['threshold'] if mode == 'zero_shot' else common.choose_threshold(v.instability_label, v.score)
            fixed.append({'arm': mode, 'seed': source['source_seed'] or 0, 'budget': 0,
                          'threshold': threshold, 'source_model_id': source['model_id']})
    out.mkdir(exist_ok=True)
    if (out / 'frozen_thresholds.json').exists():
        require(read_json(out / 'frozen_thresholds.json') == fixed, 'Frozen evaluation cutoffs changed.')
    else:
        common.write_json(out / 'frozen_thresholds.json', fixed)
    print('All paths and validation cutoffs fixed. Reading target test labels now.', flush=True)
    test = data[data.split == 'test'].copy()
    retention = pd.read_csv(args.run_dir / 'source_retention.csv')
    retention_frozen = pd.read_csv(args.run_dir / 'source_retention_frozen.csv')
    pure = pd.DataFrame(np.eye(5), columns=common.X_COLUMNS)
    pure['recipe_id'] = ['pure_' + name for name in common.MOLECULES]
    pure_frozen = reference.frozen_predictions(pure, sources)
    prediction_rows, metric_rows, retention_rows, endpoint_rows = [], [], [], []
    for item in fixed:
        if 'source_model_id' in item:
            scored = test.merge(frozen[frozen.model_id == item['source_model_id']][['recipe_id', 'score']],
                                on='recipe_id', validate='one_to_one')
            retained = retention.merge(retention_frozen[retention_frozen.model_id == item['source_model_id']],
                                       on='recipe_id', validate='one_to_one')
            retained_score = retained.score.to_numpy(float)
            retained_y = retained.instability_label.to_numpy(int)
            pure_score = pure.merge(pure_frozen[pure_frozen.model_id == item['source_model_id']],
                                    on='recipe_id', validate='one_to_one').score.to_numpy(float)
        else:
            scored = test.copy()
            scored['score'] = learning.scores(joblib.load(directory / item['model_file']),
                                              scored[common.X_COLUMNS].to_numpy(float)).mean(axis=0)
            retained_score = learning.scores(joblib.load(directory / item['model_file']),
                              retention[common.X_COLUMNS].to_numpy(float)).mean(axis=0)
            retained_y = retention.instability_label.to_numpy(int)
            pure_score = learning.scores(joblib.load(directory / item['model_file']),
                                         pure[common.X_COLUMNS].to_numpy(float)).mean(axis=0)
        for name, score in zip(common.MOLECULES, pure_score):
            endpoint_rows.append({'arm': item['arm'], 'seed': item['seed'], 'training_labels': item['budget'],
                                  'molecule': name, 'score': float(score),
                                  'predicted': int(score >= item['threshold']),
                                  'reference_label': 0, 'threshold': item['threshold']})
        retention_rows.append({'arm': item['arm'], 'seed': item['seed'], 'training_labels': item['budget'],
            **reference.safe_metrics(retained_y, retained_score, item['threshold'])})
        scored['predicted'] = (scored.score >= item['threshold']).astype(int)
        scored['arm'], scored['seed'], scored['training_labels'] = item['arm'], item['seed'], item['budget']
        scored['threshold'] = item['threshold']
        prediction_rows.append(scored)
        groups = [('overall', scored)]
        groups += [(f'{n}_components', r) for n, r in scored.groupby('components')]
        groups += [(name, r) for name, r in scored.groupby('test_group') if r.components.iloc[0] == 2]
        for name, rows in groups:
            metric_rows.append({'arm': item['arm'], 'seed': item['seed'], 'training_labels': item['budget'],
                'domain': name, **reference.safe_metrics(rows.instability_label.to_numpy(int),
                                                       rows.score.to_numpy(float), item['threshold'])})
    predictions, metrics = pd.concat(prediction_rows, ignore_index=True), pd.DataFrame(metric_rows)
    common.write_csv(out / 'test_predictions.csv', predictions)
    common.write_csv(out / 'metrics.csv', metrics)
    common.write_csv(out / 'costs.csv', pd.DataFrame(costs))
    common.write_csv(out / 'source_retention_metrics.csv', pd.DataFrame(retention_rows))
    common.write_csv(out / 'pure_endpoint_predictions.csv', pd.DataFrame(endpoint_rows))
    fields = ['average_precision', 'pr_auc_trapezoid', 'recall_instability', 'precision_instability',
              'f1_instability', 'false_positive_rate', 'fn', 'fp']
    summary = metrics.groupby(['arm', 'training_labels', 'domain'])[fields].agg(['mean', 'std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    common.write_csv(out / 'summary.csv', summary.reset_index())
    common.write_csv(out / 'coverage.csv', data.groupby(['split', 'components']).instability_label.agg(['count', 'sum']).reset_index())
    # Paired differences use the same seed, target test recipes and label budget.
    paired = []
    comparisons = [('transfer_random', 'scratch_random'),
                   ('scratch_active', 'scratch_random'),
                   ('transfer_active', 'transfer_random'),
                   ('transfer_active', 'scratch_active'),
                   ('transfer_active', 'scratch_matched_transfer_active')]
    for left, right in comparisons:
        joined = metrics[metrics.arm == left].merge(metrics[metrics.arm == right],
                      on=['seed', 'training_labels', 'domain'], suffixes=('_left', '_right'))
        for _, row in joined.iterrows():
            paired.append({'comparison': f'{left}_minus_{right}', 'seed': row.seed,
                           'training_labels': row.training_labels, 'domain': row.domain,
                           **{f'delta_{f}': row[f'{f}_left'] - row[f'{f}_right'] for f in fields}})
    common.write_csv(out / 'paired_differences.csv', pd.DataFrame(paired))
    # A declared descriptive criterion, not a statistical equivalence test:
    # AP within .02, recall within .05 and FPR no worse by more than .05 of
    # full-pool scratch. Report each domain too; aggregate success can hide failure.
    efficiency = []
    full = metrics[metrics.arm == 'full_pool_scratch']
    for (seed, domain), baseline in full.groupby(['seed', 'domain']):
        base = baseline.iloc[0]
        for arm in ARMS + ['scratch_matched_transfer_active']:
            candidates = metrics[(metrics.arm == arm) & (metrics.seed == seed) & (metrics.domain == domain)]
            good = candidates[(candidates.average_precision >= base.average_precision - .02)
                 & (candidates.recall_instability >= base.recall_instability - .05)
                 & (candidates.false_positive_rate <= base.false_positive_rate + .05)]
            smallest = int(good.training_labels.min()) if len(good) else None
            efficiency.append({'seed': seed, 'domain': domain, 'arm': arm,
                               'full_pool_labels': int(base.training_labels),
                               'smallest_passing_budget': smallest,
                               'fraction_of_full_pool': smallest / base.training_labels if smallest else None})
    common.write_csv(out / 'label_efficiency.csv', pd.DataFrame(efficiency))
    print(summary.reset_index().query("domain == 'overall'")[[
        'arm', 'training_labels', 'average_precision_mean', 'recall_instability_mean', 'f1_instability_mean']].to_string(index=False))
    out.rename(final_out)
    print('Saved:', final_out)
    print('Means and seed spread reuse the same test set; they are not independent-test confidence intervals.')
    print('Labels are finite COSMO-RS search outcomes. Inspect domain metrics and training epoch caps.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ['prepare', 'search', 'train', 'report']:
        p = sub.add_parser(command)
        p.add_argument('--run-dir', type=Path, default=DEFAULT)
        if command == 'prepare':
            p.add_argument('--frozen-run', type=Path, default=ROOT / 'results/ml/project4_frozen_ood/pilot_v1')
            p.add_argument('--source', choices=['p3', 'p2'], default='p3')
            p.add_argument('--seed', type=int, default=20261008)
            p.add_argument('--seeds', nargs='+', type=int, default=[11, 22, 33, 44, 55])
            p.add_argument('--prepared', type=Path, default=common.DEFAULT_PREPARED)
        if command == 'search':
            p.add_argument('--starts', type=int, default=40)
            p.add_argument('--search-seed', type=int, default=88)
            p.add_argument('--limit', type=int)
            p.add_argument('--reuse-labels', type=Path)
        if command in ['search', 'train']:
            p.add_argument('--workers', type=int, default=8)
        if command in ['train', 'report']:
            p.add_argument('--experiment', default='benchmark_v1')
        if command == 'train':
            p.add_argument('--budgets', type=int, nargs='+', default=[60, 150, 300])
            p.add_argument('--batch', type=int, default=30)
            p.add_argument('--epochs', type=int, default=1200)
    args = parser.parse_args()
    if hasattr(args, 'workers'):
        require(1 <= args.workers <= 8, 'Use 1 to 8 workers.')
    if args.command == 'search':
        require(args.starts >= 16 and (args.limit is None or args.limit > 0), 'Invalid starts or limit.')
    args.run_dir.parent.mkdir(parents=True, exist_ok=True)
    with (args.run_dir.parent / (args.run_dir.name + '.lock')).open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another command is using this run directory.') from error
        globals()[args.command](args)


if __name__ == '__main__':
    main()
