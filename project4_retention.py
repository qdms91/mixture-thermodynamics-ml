"""Project 4 retention: prepare, train Stages 1/2, and report.

Uses completed reference labels and fixed target acquisition paths. It never
runs ORCA/COSMO-RS, acquires new labels, or selects replay by energy.
"""

import os
for variable in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS']:
    os.environ[variable] = '1'
os.environ.setdefault('MPLBACKEND', 'Agg')

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
from importlib.metadata import version
import json
import multiprocessing
from pathlib import Path
import shutil
import tempfile
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import project2_common as common
import project3_common as active
import project4_learning as legacy
import project4_retention_learning as learning


ROOT = Path(__file__).resolve().parent
TARGET = ROOT / 'results/ml/project4_adaptation/target_v1'
DEFAULT = TARGET / 'benchmark_v2_retention'
FAMILIES = ['p3_active200', 'p2_composition', 'p2_molecular']
REPLAY_FAMILIES = ['p3_active200', 'p2_molecular']
SEEDS = [11, 22, 33, 44, 55]
BUDGETS = [60, 150, 300]
BASE_COLUMNS = ['recipe_id', 'block_id', 'split', *common.X_COLUMNS,
                'instability_label', 'stability_seconds']
require = learning.require


def read_json(path):
    return json.loads(Path(path).read_text())


def hashes():
    modules = [common, active, legacy, learning]
    paths = [Path(__file__)] + [Path(m.__file__) for m in modules]
    return {p.name: common.sha256(p) for p in paths}


def checked(path, expected):
    path = Path(path)
    require(path.is_file(), f'Missing file: {path}')
    require(common.sha256(path) == expected, f'File changed: {path}')
    return path


def atomic_model(path, model):
    temporary = path.with_suffix('.tmp')
    joblib.dump(model, temporary)
    temporary.replace(path)


def composition_keys(data):
    return set(map(tuple, data[common.X_COLUMNS].round(12).to_numpy()))


def check_split(data, counts, name):
    require(data.recipe_id.is_unique, f'{name}: duplicate recipe IDs.')
    require(data.split.value_counts().to_dict() == counts, f'{name}: unexpected split counts.')
    require(data.groupby('block_id').split.nunique().max() == 1, f'{name}: a block crosses splits.')
    require(data.instability_label.isin([0, 1]).all(), f'{name}: unresolved labels.')
    learning.features(data, common.COMPOSITION)
    require(len(composition_keys(data)) == len(data), f'{name}: duplicate compositions.')
    for split, rows in data.groupby('split'):
        require(rows.instability_label.nunique() == 2, f'{name}/{split}: missing label class.')


def load_target(directory, previous):
    """Check the completed chemistry campaign and original training manifests."""
    protocol = read_json(directory / 'protocol.json')
    require(protocol['sklearn_version'] == '1.8.0', 'Expected the original sklearn 1.8.0 run.')
    require(protocol['source'] == 'p3', 'Baseline reuse expects the completed Project 3 transfer run.')
    require(protocol['seeds'] == SEEDS, 'Expected the five original adaptation seeds.')
    for filename, digest in protocol['files_sha256'].items():
        checked(directory / filename, digest)
    complete = read_json(directory / 'labels_complete.json')
    checked(directory / 'dataset.csv', complete['dataset_sha256'])
    checked(directory / 'reference_protocol.json', complete['reference_protocol_sha256'])
    data = pd.read_csv(directory / 'dataset.csv')
    check_split(data, {'train': 600, 'validation': 200, 'test': 200}, 'Target')
    require(np.allclose(data.temperature_K, 298.15), 'Target temperature changed.')
    require(data.starts.eq(40).all(), 'This retention protocol expects 40-start labels.')
    recipes = pd.read_csv(directory / 'recipes.csv').set_index('recipe_id')
    observed = data.set_index('recipe_id')
    require(set(recipes.index) == set(observed.index), 'Target recipe manifest disagrees with labels.')
    require(np.allclose(recipes.loc[observed.index, common.X_COLUMNS], observed[common.X_COLUMNS],
                        atol=1e-12, rtol=0), 'Target compositions changed.')
    require(recipes.loc[observed.index, 'split'].equals(observed['split']), 'Target split changed.')
    config = read_json(previous / 'training_protocol.json')
    checked(directory / 'dataset.csv', config['dataset_sha256'])
    require(config['budgets'] == BUDGETS and config['seeds'] == SEEDS, 'Unexpected original budgets/seeds.')
    require(config['code_sha256']['project4_learning.py'] == common.sha256(legacy.__file__),
            'Original learning helper changed; baseline compatibility needs inspection.')
    completion = read_json(previous / 'training_complete.json')
    checked(previous / 'training_protocol.json', completion['training_protocol_sha256'])
    paths, reused = {}, {}
    pool = data.loc[data.split == 'train'].reset_index(drop=True)
    for seed in SEEDS:
        manifest = checked(previous / f'seed_{seed}/complete.json', completion['seed_sha256'][str(seed)])
        records = read_json(manifest)
        for budget in BUDGETS:
            pair = {}
            for initialization, arm in [('transfer', 'transfer_active'),
                                        ('scratch', 'scratch_matched_transfer_active')]:
                choices = [r for r in records if r['arm'] == arm and r['budget'] == budget]
                require(len(choices) == 1, f'Missing original {arm}, seed {seed}, budget {budget}.')
                record = choices[0]
                require(len(record['recipe_ids']) == budget and len(set(record['recipe_ids'])) == budget,
                        'Invalid recorded target path.')
                require(pool.iloc[record['positions']].recipe_id.tolist() == record['recipe_ids'],
                        'Original target IDs no longer match their pool positions.')
                checked(previous / record['model_file'], record['model_sha256'])
                pair[initialization] = record
                reused[f'{seed}:{budget}:{initialization}'] = record
            require(pair['transfer']['recipe_ids'] == pair['scratch']['recipe_ids'],
                    'The matched scratch control used different target recipes.')
            paths[f'{seed}:{budget}'] = pair['transfer']['recipe_ids']
        require(set(paths[f'{seed}:60']) <= set(paths[f'{seed}:150']) <= set(paths[f'{seed}:300']),
                'Target curriculum is not nested.')
    return data, protocol, config, paths, reused


def source_entry(family, seed, args, prepared, p2_selections, p3_registry):
    """Resolve original weights; never substitute a different fitted model."""
    if family == 'p3_active200':
        matches = [r for r in p3_registry if r['model_id'] == f'p3_active_n200_seed{seed}']
        require(len(matches) == 1, f'Missing source registry entry for seed {seed}.')
        entry = matches[0]
        path = Path(entry['model_file'])
        if not path.is_file():
            path = args.p3_run / f'seed_{seed}/uncertainty_diversity/model_0200.joblib'
        checked(path, entry['model_sha256'])
        package = joblib.load(path)
        ids = package.get('training_recipe_ids', [])
        require(len(ids) == 200, f'{path}: expected 200 checkpoint training IDs.')
        columns, cutoff = common.COMPOSITION, entry['threshold']
    else:
        group = 'a_composition' if family == 'p2_composition' else 'b_molecular'
        matches = [r for r in p2_selections['models'] if r['group'] == group and r['family'] == 'mlp']
        require(len(matches) == 1, f'Missing selected Project 2 model: {group}.')
        entry = matches[0]
        path = checked(args.p2_run / entry['model_file'], entry['model_sha256'])
        package = joblib.load(path)
        ids = prepared.loc[prepared.split == 'train', 'recipe_id'].tolist()
        columns, cutoff = entry['columns'], entry['threshold']
    model = learning.adapt_source(package, columns)
    training = prepared.loc[prepared.split == 'train']
    require(set(ids) <= set(training.recipe_id), 'A source checkpoint includes held-out recipes.')
    probe = pd.concat([training.loc[training.recipe_id.isin(ids), common.X_COLUMNS],
                       pd.DataFrame(np.eye(5), columns=common.X_COLUMNS)], ignore_index=True)
    error = learning.check_source_predictions(package, model, probe)
    return model, ids, {'family': family, 'seed': seed, 'original_file': str(path.resolve()),
                       'original_sha256': common.sha256(path), 'original_cutoff': float(cutoff),
                       'source_training_labels': len(ids), 'columns': columns,
                       'source_conversion_max_abs_error': error}


def job_id(job):
    return f"{job['family']}__s{job['seed']}__{job['policy']}__{job['initialization']}__n{job['budget']}"


def job_folder(directory, job):
    return directory / f"stage{job['stage']}" / 'fits' / job['job_id']


def fit_record(job, package, model_file, extra=None):
    record = {**job, 'model_sha256': common.sha256(model_file),
              'fit_seconds': float(package['fit_seconds']),
              'epoch_cap_members': sum(int(d['epoch_cap_reached']) for d in package['diagnostics']),
              'diagnostics': package['diagnostics'], 'reused': False,
              'replay_size': int(package['replay_size']),
              'replay_weight_share': float(package['replay_weight_share'])}
    if extra:
        record.update(extra)
    return record


def prepare(args):
    require(not args.run_dir.exists(), f'Already exists: {args.run_dir}. Use train/status; do not prepare again.')
    require(args.run_dir.resolve() != args.target_run.resolve(), 'Output cannot replace the target dataset.')
    prepared, groups, prep_meta = common.load_prepared(args.prepared)
    check_split(prepared, {'train': 1191, 'validation': 314, 'test': 495}, 'Source')
    descriptors = common.add_descriptors(prepared[common.X_COLUMNS])
    require(np.allclose(descriptors[groups['b_molecular']], prepared[groups['b_molecular']],
                        rtol=1e-10, atol=1e-10), 'Descriptor formulas/constants changed.')
    calculation_meta = prep_meta['source_metadata']['calculation_metadata']
    require(calculation_meta['starts'] == 40 and calculation_meta['temperature_K'] == 298.15,
            'Source reference protocol differs from the planned timing comparison.')
    p2_meta = read_json(args.p2_run / 'run_metadata.json')
    checked(args.prepared / 'preparation.json', p2_meta['preparation_sha256'])
    selections = read_json(args.p2_run / 'selections.json')
    previous = args.target_run / args.previous_experiment
    target, target_protocol, old_config, paths, reused = load_target(args.target_run, previous)
    require(not composition_keys(prepared).intersection(composition_keys(target)),
            'A target recipe duplicates an original source recipe.')
    retained = pd.read_csv(args.target_run / 'source_retention.csv').set_index('recipe_id')
    source_test = prepared.loc[prepared.split == 'test'].set_index('recipe_id')
    require(set(retained.index) == set(source_test.index), 'Source-retention test IDs changed.')
    require(np.array_equal(retained.loc[source_test.index, 'instability_label'], source_test.instability_label),
            'Source-retention labels changed.')
    registry = read_json(args.target_run / 'sources.json')
    config = {k: old_config[k] for k in ['epochs', 'min_epochs', 'patience', 'tol', 'alpha', 'learning_rate']}
    config['replay_share'] = .2
    source_train = prepared.loc[prepared.split == 'train', BASE_COLUMNS].copy()
    if 'data_origin' in prepared:
        source_train['data_origin'] = prepared.loc[source_train.index, 'data_origin']

    # Build under a temporary sibling directory. A failed preparation removes
    # only its own incomplete staging files; original calculations remain intact.
    args.run_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.retention_prepare_', dir=args.run_dir.parent))
    try:
        for folder in ['inputs', 'source_models', 'replay_manifests', 'code_snapshot']:
            (staging / folder).mkdir()
        for role in ['train', 'validation', 'test']:
            cols = BASE_COLUMNS + [c for c in ['components', 'test_group'] if c in target]
            common.write_csv(staging / f'inputs/target_{role}.csv', target.loc[target.split == role, cols])
            common.write_csv(staging / f'inputs/source_{role}.csv', prepared.loc[prepared.split == role, BASE_COLUMNS])
        source_specs, source_objects, memory_index, audit_rows = {}, {}, {}, []
        for family in FAMILIES:
            for seed in SEEDS:
                tag = f'{family}__s{seed}'
                source, ids, spec = source_entry(family, seed, args, prepared, selections, registry)
                source_objects[tag] = source
                spec['snapshot'] = f'source_models/{tag}.joblib'
                atomic_model(staging / spec['snapshot'], source)
                candidates = learning.eligible_memory(source_train, ids, source, spec['original_cutoff'])
                common.write_csv(staging / f'replay_manifests/{tag}__candidates.csv', candidates)
                if family in REPLAY_FAMILIES:
                    memories = learning.choose_memories(candidates, seed)
                    for policy, memory in memories.items():
                        filename = f'replay_manifests/{tag}__{policy}.csv'
                        common.write_csv(staging / filename, memory)
                        memory_index[f'{tag}:{policy}'] = filename
                        audit_rows.append({'family': family, 'seed': seed, 'policy': policy,
                                           'eligible_training_examples': int(candidates.eligible.sum()),
                                           'memory_size': len(memory), 'positive_labels': int(memory.instability_label.sum()),
                                           'recorded_search_seconds': float(memory.stability_seconds.sum()),
                                           'distinct_blocks': int(memory.block_id.nunique())})
                source_specs[tag] = spec
        jobs = []
        for stage in [1, 2]:
            for family in FAMILIES if stage == 1 else REPLAY_FAMILIES:
                for seed in SEEDS:
                    tag = f'{family}__s{seed}'
                    for policy in ['none'] if stage == 1 else learning.POLICIES:
                        for budget in BUDGETS:
                            for initialization in ['transfer', 'scratch']:
                                job = {'stage': stage, 'family': family, 'seed': seed, 'policy': policy,
                                       'initialization': initialization, 'budget': budget,
                                       'source_tag': tag, 'source_file': source_specs[tag]['snapshot'],
                                       'target_recipe_ids': paths[f'{seed}:{budget}'],
                                       'memory_file': None if policy == 'none' else memory_index[f'{tag}:{policy}']}
                                job['job_id'] = job_id(job)
                                jobs.append(job)
                                if family == 'p3_active200' and policy == 'none':
                                    # Import the exact completed target-only models, not fitted substitutes.
                                    record = reused[f'{seed}:{budget}:{initialization}']
                                    fitted = joblib.load(previous / record['model_file'])
                                    package = {**source_objects[tag], **fitted,
                                               'replay_size': 0, 'replay_weight_share': 0.}
                                    folder = job_folder(staging, job); folder.mkdir(parents=True)
                                    atomic_model(folder / 'model.joblib', package)
                                    probes = target.loc[target.recipe_id.isin(job['target_recipe_ids'])]
                                    before = legacy.scores(fitted, probes[common.X_COLUMNS].to_numpy()).mean(axis=0)
                                    after = learning.scores(package, probes).mean(axis=0)
                                    require(np.allclose(before, after, atol=1e-12, rtol=1e-10), 'Reused model changed predictions.')
                                    common.write_json(folder / 'record.json', fit_record(job, package, folder / 'model.joblib', {
                                        'reused': True, 'fit_seconds': 0.,
                                        'historical_fit_seconds': float(fitted['fit_seconds']),
                                        'reused_from': str((previous / record['model_file']).resolve()),
                                        'reused_original_sha256': record['model_sha256']}))
        common.write_json(staging / 'inputs/jobs.json', jobs)
        common.write_json(staging / 'inputs/sources.json', source_specs)
        common.write_json(staging / 'inputs/descriptor_constants.json', read_json(args.prepared / 'descriptor_constants.json'))
        common.write_csv(staging / 'replay_manifests/memory_summary.csv', pd.DataFrame(audit_rows))
        common.write_json(staging / 'input_audit.json', {
            'source_split_counts': {str(k): int(v) for k, v in prepared.split.value_counts().items()},
            'target_split_counts': {str(k): int(v) for k, v in target.split.value_counts().items()},
            'source_target_composition_overlap': 0, 'source_test_ids_match_previous_run': True,
            'source_timing_protocol': calculation_meta,
            'source_timing_caveat': 'Historical elapsed search durations; CPU load and campaign cohort can affect ranking.',
            'reused_p3_no_replay_models': 30, 'stage1_new_fits': 60, 'stage2_new_fits': 180,
            'source_models': source_specs})
        for name in hashes():
            shutil.copy2(ROOT / name, staging / 'code_snapshot' / name)
        frozen_files = [p for folder in ['inputs', 'source_models', 'replay_manifests', 'code_snapshot']
                        for p in (staging / folder).rglob('*') if p.is_file()]
        frozen_files.append(staging / 'input_audit.json')
        common.write_json(staging / 'protocol.json', {
            'created_utc': datetime.now(timezone.utc).isoformat(), 'schema': 'retention_stages12_v1',
            'code_sha256': hashes(), 'sklearn_version': version('scikit-learn'),
            'versions': {n: version(n) for n in ['numpy', 'pandas', 'scipy', 'joblib', 'threadpoolctl']},
            'files_sha256': {str(p.relative_to(staging)): common.sha256(p) for p in frozen_files},
            'training': config, 'seeds': SEEDS, 'budgets': BUDGETS, 'memory_size': 20,
            'families': FAMILIES, 'replay_families': REPLAY_FAMILIES, 'policies': learning.POLICIES,
            'target_acquisition': 'Fixed original transfer_active paths; no new acquisition or energy selection.',
            'previous_experiment': str(previous.resolve()),
            'provenance': {'prepared': str(args.prepared.resolve()), 'p2_run': str(args.p2_run.resolve()),
                           'target_dataset_sha256': common.sha256(args.target_run / 'dataset.csv'),
                           'prepared_metadata_sha256': common.sha256(args.prepared / 'preparation.json'),
                           'previous_training_protocol_sha256': common.sha256(previous / 'training_protocol.json')},
            'evaluation': 'Fixed target validation F1 cutoff; source-validation cutoff is a separate diagnostic.',
            'stage3': 'Deferred. Energy-ranked and energy-transition memories are not implemented here.'})
        staging.rename(args.run_dir)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    print(pd.DataFrame(audit_rows).to_string(index=False))
    print('\nPrepared:', args.run_dir)
    print('Imported 30 P3 no-replay fits. Stage 1: 60 new committee fits; Stage 2: 180.')
    print('No new reference calculations were performed. No energy-based selector was used.')


def load_run(directory):
    protocol = read_json(directory / 'protocol.json')
    require(protocol['schema'] == 'retention_stages12_v1', 'Unsupported protocol schema.')
    require(protocol['code_sha256'] == hashes(), 'Code changed after preparation; preserve this frozen run.')
    require(version('scikit-learn') == protocol['sklearn_version'] == '1.8.0', 'Use scikit-learn 1.8.0.')
    for filename, digest in protocol['files_sha256'].items():
        checked(directory / filename, digest)
    return protocol, read_json(directory / 'inputs/jobs.json'), read_json(directory / 'inputs/sources.json')


def completed(directory, job):
    folder = job_folder(directory, job)
    path = folder / 'record.json'
    if not path.exists():
        return None
    record = read_json(path)
    require(all(record.get(key) == value for key, value in job.items()), f'Job manifest changed: {job["job_id"]}')
    checked(folder / 'model.joblib', record['model_sha256'])
    return record


def execute_job(directory, job, config):
    """The worker reads only frozen source models and training rows."""
    directory = Path(directory)
    existing = completed(directory, job)
    if existing is not None:
        return job['job_id']
    target = pd.read_csv(directory / 'inputs/target_train.csv').set_index('recipe_id', drop=False)
    target = target.loc[job['target_recipe_ids']].reset_index(drop=True)
    memory = pd.read_csv(directory / job['memory_file']) if job['memory_file'] else target.iloc[:0].copy()
    require(target.split.eq('train').all() and (memory.empty or memory.split.eq('train').all()),
            'A held-out row reached a worker.')
    require(not composition_keys(target).intersection(composition_keys(memory)), 'Target/replay duplicate.')
    source = joblib.load(directory / job['source_file'])
    package = learning.fit_committee(source, target, memory, job['initialization'] == 'transfer',
                                     job['seed'], config)
    folder = job_folder(directory, job); folder.mkdir(parents=True, exist_ok=True)
    atomic_model(folder / 'model.joblib', package)
    record = fit_record(job, package, folder / 'model.joblib', {
        'target_positive_labels': int(target.instability_label.sum()),
        'replay_positive_labels': int(memory.instability_label.sum()),
        'target_reference_seconds': float(target.stability_seconds.sum()),
        'replay_historical_reference_seconds': float(memory.stability_seconds.sum()),
        'replay_recipe_ids': memory.recipe_id.tolist()})
    common.write_json(folder / 'record.json', record)
    return job['job_id']


def train(args):
    protocol, jobs, _ = load_run(args.run_dir)
    stages = [1, 2] if args.stage == 'all' else [int(args.stage)]
    selected = [j for j in jobs if j['stage'] in stages]
    if args.limit is not None:
        selected = selected[:args.limit]  # TOTAL selected jobs, including completed/imported jobs.
    pending = [j for j in selected if completed(args.run_dir, j) is None]
    print(f'Selected jobs: {len(selected)}; completed/imported: {len(selected)-len(pending)}; pending: {len(pending)}.', flush=True)
    executor = ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'))
    iterator, futures = iter(pending), {}

    def submit():
        job = next(iterator, None)
        if job is not None:
            future = executor.submit(execute_job, str(args.run_dir), job, protocol['training'])
            futures[future] = job['job_id']

    stopped = False
    try:
        for _ in range(min(args.workers, len(pending))):
            submit()
        count = 0
        while futures:
            future = next(as_completed(futures))
            futures.pop(future)
            result = future.result(); count += 1
            print(f'Completed new fit {count}/{len(pending)}: {result}', flush=True)
            submit()
    except KeyboardInterrupt:
        stopped = True
        print('\nStopping after currently running fits save. Repeat the same command to resume.', flush=True)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        # Workers commit their own outputs atomically; interrupted fits are
        # retried only when they have no valid completion record.
    if not stopped:
        for future in futures:
            if not future.cancelled():
                future.result()
    status(args)


def status(args):
    _, jobs, _ = load_run(args.run_dir)
    rows = []
    for stage in [1, 2]:
        selected = [j for j in jobs if j['stage'] == stage]
        records = [completed(args.run_dir, j) for j in selected]
        rows.append({'stage': stage, 'planned': len(selected), 'complete': sum(r is not None for r in records),
                     'reused': sum(bool(r and r['reused']) for r in records)})
    print(pd.DataFrame(rows).to_string(index=False))


def safe_metrics(y, score, cutoff):
    y = np.asarray(y, int)
    require(np.isfinite(score).all(), 'Nonfinite model score.')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        result = common.metrics(y, score, cutoff)
    if len(np.unique(y)) < 2:
        for key in ['average_precision', 'pr_auc_trapezoid', 'roc_auc']:
            result[key] = None
    result['false_positive_rate'] = result['fp'] / (result['fp'] + result['tn']) if result['fp'] + result['tn'] else None
    return result


def evaluate_rows(rows, score, cutoff, identifiers, scope, threshold_rule):
    prediction = rows.copy()
    prediction['score'], prediction['predicted'] = score, (score >= cutoff).astype(int)
    prediction['threshold'] = cutoff
    for key, value in {**identifiers, 'scope': scope, 'threshold_rule': threshold_rule}.items():
        prediction[key] = value
    groups = [('overall', prediction)]
    if scope == 'target':
        groups += [(f'{n}_components', group) for n, group in prediction.groupby('components')]
        groups += [(name, group) for name, group in prediction.groupby('test_group') if group.components.iloc[0] == 2]
    metrics = [{**identifiers, 'scope': scope, 'domain': name, 'threshold_rule': threshold_rule,
                **safe_metrics(group.instability_label, group.score.to_numpy(), cutoff)} for name, group in groups]
    return prediction, metrics


def plot_report(metrics, directory):
    data = metrics.loc[(metrics.domain == 'overall') & (metrics.threshold_rule == 'target_validation') &
                       (metrics.initialization == 'transfer')]
    colors = {'none': '#666666', 'random20': '#397EAF', 'slowest20': '#D57927', 'time_class_matched20': '#008775'}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for row, family in enumerate(REPLAY_FAMILIES):
        for col, scope in enumerate(['source', 'target']):
            ax = axes[row, col]
            subset = data.loc[(data.family == family) & (data.scope == scope)]
            for policy, group in subset.groupby('policy', sort=False):
                stats = group.groupby('budget').average_precision.agg(['mean', 'std'])
                ax.plot(stats.index, stats['mean'], 'o-', color=colors[policy], label=policy)
                ax.fill_between(stats.index, np.clip(stats['mean']-stats['std'], 0, 1),
                                np.clip(stats['mean']+stats['std'], 0, 1), color=colors[policy], alpha=.1)
            ax.set(title=f'{family}: {scope} test', ylabel='Average precision', ylim=(0, 1.02), xticks=BUDGETS)
            ax.grid(alpha=.2); ax.legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel('Target training labels (replay memories contain 20 source examples)')
    fig.suptitle('Source retention and target adaptation: transfer models')
    fig.text(.5, .01, 'Means ±1 sample SD across adaptation seeds; common held-out recipes, not independent-test confidence intervals.', ha='center', fontsize=8)
    fig.tight_layout(rect=[0, .04, 1, .96])
    fig.savefig(directory / 'retention_target_AP.png', dpi=180)
    plt.close(fig)


def report(args):
    protocol, all_jobs, sources = load_run(args.run_dir)
    stages = [1] if args.stage == '1' else [1, 2]
    jobs = [j for j in all_jobs if j['stage'] in stages]
    records = [completed(args.run_dir, job) for job in jobs]
    missing = [j['job_id'] for j, r in zip(jobs, records) if r is None]
    require(not missing, f'Complete the requested fits before report. Missing {len(missing)}; first: {missing[:3]}')
    output = args.run_dir / ('report_stage1' if stages == [1] else 'report_stage12')
    require(not output.exists(), f'Report already exists: {output}. It is frozen; do not overwrite it.')
    building = args.run_dir / (output.name + '_in_progress'); building.mkdir(exist_ok=True)
    target_val = pd.read_csv(args.run_dir / 'inputs/target_validation.csv')
    source_val = pd.read_csv(args.run_dir / 'inputs/source_validation.csv')
    fixed = []
    for job, record in zip(jobs, records):
        package = joblib.load(job_folder(args.run_dir, job) / 'model.joblib')
        target_score = learning.scores(package, target_val).mean(axis=0)
        source_score = learning.scores(package, source_val).mean(axis=0)
        fixed.append({'job_id': job['job_id'], 'model_sha256': record['model_sha256'],
                      'target_cutoff': common.choose_threshold(target_val.instability_label, target_score),
                      'source_cutoff': common.choose_threshold(source_val.instability_label, source_score)})
    frozen = building / 'frozen_thresholds.json'
    if frozen.exists():
        require(read_json(frozen) == fixed, 'Validation cutoffs changed during report resumption.')
    else:
        common.write_json(frozen, fixed)
    print('All fits and validation cutoffs fixed. Reading held-out source and target labels now.', flush=True)
    target_test = pd.read_csv(args.run_dir / 'inputs/target_test.csv')
    source_test = pd.read_csv(args.run_dir / 'inputs/source_test.csv')
    pure = pd.DataFrame(np.eye(5), columns=common.X_COLUMNS)
    predictions, metric_rows, baseline_rows, endpoint_rows, cost_rows = [], [], [], [], []
    baseline_by_tag = {}
    for tag, spec in sources.items():
        package = joblib.load(args.run_dir / spec['snapshot'])
        for scope, rows in [('source', source_test), ('target', target_test)]:
            score = learning.scores(package, rows).mean(axis=0)
            ids = {'family': spec['family'], 'seed': spec['seed'], 'policy': 'frozen_source',
                   'initialization': 'pretrained', 'budget': 0}
            prediction, metrics = evaluate_rows(rows, score, spec['original_cutoff'], ids, scope, 'original_source_validation')
            predictions.append(prediction); baseline_rows.extend(metrics)
            if scope == 'source':
                baseline_by_tag[tag] = metrics[0]['average_precision']
    for job, record, threshold in zip(jobs, records, fixed):
        package = joblib.load(job_folder(args.run_dir, job) / 'model.joblib')
        ids = {k: job[k] for k in ['family', 'seed', 'policy', 'initialization', 'budget']}
        for scope, rows in [('target', target_test), ('source', source_test)]:
            score = learning.scores(package, rows).mean(axis=0)
            prediction, metrics = evaluate_rows(rows, score, threshold['target_cutoff'], ids, scope, 'target_validation')
            predictions.append(prediction); metric_rows.extend(metrics)
            if scope == 'source':
                _, diagnostics = evaluate_rows(rows, score, threshold['source_cutoff'], ids, scope, 'source_validation_diagnostic')
                metric_rows.extend(diagnostics)
        for name, score in zip(common.MOLECULES, learning.scores(package, pure).mean(axis=0)):
            endpoint_rows.append({**ids, 'molecule': name, 'score': float(score),
                                  'predicted': int(score >= threshold['target_cutoff']), 'reference_label': 0})
        memory = pd.read_csv(args.run_dir / job['memory_file']) if job['memory_file'] else None
        train = pd.read_csv(args.run_dir / 'inputs/target_train.csv').set_index('recipe_id').loc[job['target_recipe_ids']]
        cost_rows.append({**ids, 'reused_fit': record['reused'], 'fit_seconds_this_benchmark': record['fit_seconds'],
                          'historical_fit_seconds': record.get('historical_fit_seconds', 0.),
                          'epoch_cap_members': record['epoch_cap_members'], 'source_memory_labels': 0 if memory is None else len(memory),
                          'replay_weight_share': record['replay_weight_share'], 'target_validation_labels': len(target_val),
                          'source_validation_labels_for_diagnostic': len(source_val),
                          'source_pretraining_labels_in_weights': sources[job['source_tag']]['source_training_labels'] if job['initialization']=='transfer' else 0,
                          'memory_selection_uses_source_teacher': memory is not None,
                          'target_curriculum_uses_previous_transfer_teacher': True,
                          'target_reference_seconds_sum': float(train.stability_seconds.sum()),
                          'replay_historical_reference_seconds_sum': 0. if memory is None else float(memory.stability_seconds.sum()),
                          'new_reference_calculations': 0})
    metrics = pd.DataFrame(metric_rows)
    keys = ['family', 'policy', 'initialization', 'budget', 'scope', 'domain', 'threshold_rule']
    fields = ['average_precision', 'pr_auc_trapezoid', 'f1_instability', 'precision_instability',
              'recall_instability', 'recall_no_instability', 'fp', 'fn']
    summary = metrics.groupby(keys)[fields].agg(['mean', 'std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    paired = []
    no_replay = metrics.loc[metrics.policy == 'none']
    join_keys = ['family', 'seed', 'initialization', 'budget', 'scope', 'domain', 'threshold_rule']
    joined = metrics.loc[metrics.policy != 'none'].merge(no_replay, on=join_keys, suffixes=('', '_baseline'), validate='many_to_one')
    for _, row in joined.iterrows():
        paired.append({**{k: row[k] for k in join_keys}, 'policy': row.policy,
                       **{f'delta_{f}': row[f] - row[f'{f}_baseline'] for f in fields}})
    retention = metrics.loc[(metrics.scope == 'source') & (metrics.domain == 'overall') &
                            (metrics.threshold_rule == 'target_validation') & (metrics.initialization == 'transfer')].copy()
    retention['original_source_AP'] = [baseline_by_tag[f'{r.family}__s{r.seed}'] for r in retention.itertuples()]
    retention['source_AP_change_from_pretraining'] = retention.average_precision - retention.original_source_AP
    common.write_csv(building / 'test_predictions.csv', pd.concat(predictions, ignore_index=True))
    common.write_csv(building / 'metrics.csv', metrics)
    common.write_csv(building / 'summary.csv', summary.reset_index())
    common.write_csv(building / 'frozen_source_metrics.csv', pd.DataFrame(baseline_rows))
    common.write_csv(building / 'source_retention_changes.csv', retention)
    common.write_csv(building / 'paired_replay_minus_none.csv', pd.DataFrame(paired))
    # These paired contrasts separate replay selection, class balance and
    # initial weights. Every row compares the same seed and target budget.
    contrasts = []
    for comparator in ['random20', 'time_class_matched20']:
        pair = metrics.loc[metrics.policy == 'slowest20'].merge(
            metrics.loc[metrics.policy == comparator], on=join_keys,
            suffixes=('', '_control'), validate='one_to_one')
        for _, row in pair.iterrows():
            contrasts.append({**{k: row[k] for k in join_keys},
                              'contrast': f'slowest20_minus_{comparator}',
                              **{f'delta_{f}': row[f] - row[f'{f}_control'] for f in fields}})
    weight_keys = [k for k in join_keys if k != 'initialization'] + ['policy']
    pair = metrics.loc[metrics.initialization == 'transfer'].merge(
        metrics.loc[metrics.initialization == 'scratch'], on=weight_keys,
        suffixes=('', '_control'), validate='one_to_one')
    for _, row in pair.iterrows():
        contrasts.append({**{k: row[k] for k in weight_keys},
                          'contrast': 'transfer_minus_matched_scratch',
                          **{f'delta_{f}': row[f] - row[f'{f}_control'] for f in fields}})
    common.write_csv(building / 'paired_controls.csv', pd.DataFrame(contrasts))
    common.write_csv(building / 'costs.csv', pd.DataFrame(cost_rows))
    common.write_csv(building / 'pure_endpoint_predictions.csv', pd.DataFrame(endpoint_rows))
    shutil.copy2(args.run_dir / 'replay_manifests/memory_summary.csv', building / 'memory_summary.csv')
    plot_report(metrics, building)
    common.write_json(building / 'report_manifest.json', {
        'protocol_sha256': common.sha256(args.run_dir / 'protocol.json'), 'stages': stages,
        'evaluated_fits': len(jobs), 'files_sha256': {p.name: common.sha256(p) for p in building.iterdir()
                                                   if p.is_file() and p.name != 'report_manifest.json'}})
    building.rename(output)
    selected = summary.reset_index().query("domain == 'overall' and threshold_rule == 'target_validation'")
    print(selected[['family', 'policy', 'initialization', 'budget', 'scope', 'average_precision_mean',
                    'recall_instability_mean', 'f1_instability_mean']].round(4).to_string(index=False))
    print('\nSaved:', output)
    print('These are retained/adapted predictions of finite stability screens, not experimental miscibility.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for command in ['prepare', 'train', 'status', 'report']:
        p = commands.add_parser(command)
        p.add_argument('--run-dir', type=Path, default=DEFAULT)
        if command == 'prepare':
            p.add_argument('--target-run', type=Path, default=TARGET)
            p.add_argument('--previous-experiment', default='benchmark_v1')
            p.add_argument('--prepared', type=Path, default=common.DEFAULT_PREPARED)
            p.add_argument('--p2-run', type=Path, default=ROOT/'results/ml/project2_classification/benchmark_v1')
            p.add_argument('--p3-run', type=Path, default=ROOT/'results/ml/project3_active_learning/benchmark_v1')
        if command == 'train':
            p.add_argument('--stage', choices=['1', '2', 'all'], default='all')
            p.add_argument('--workers', type=int, default=8)
            p.add_argument('--limit', type=int, help='Total selected jobs, INCLUDING completed/imported jobs.')
        if command == 'report':
            p.add_argument('--stage', choices=['1', 'all'], default='all')
    args = parser.parse_args()
    require(version('scikit-learn') == '1.8.0', 'Activate cosmo-rs-1 with scikit-learn 1.8.0; do not upgrade the saved-model environment.')
    if args.command == 'train':
        require(1 <= args.workers <= 8, 'Choose 1 to 8 workers.')
        require(args.limit is None or args.limit > 0, '--limit must be positive.')
    args.run_dir = args.run_dir.resolve()
    args.run_dir.parent.mkdir(parents=True, exist_ok=True)
    with (args.run_dir.parent / (args.run_dir.name + '.lock')).open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another command is using this retention run. Use one terminal per run directory.') from error
        globals()[args.command](args)


if __name__ == '__main__':
    main()
