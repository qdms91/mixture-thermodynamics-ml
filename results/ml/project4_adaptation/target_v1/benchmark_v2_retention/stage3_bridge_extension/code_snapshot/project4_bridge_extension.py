"""Post-Stage-3 exploratory P3 downstream bridge extension.

Run prepare, train, report. Existing Stage 3 is read only. All five seeds,
budgets, source checkpoints, target paths and fitting settings are inherited.
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
import multiprocessing
from pathlib import Path
import shutil
import tempfile

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import project2_common as common
import project4_retention as base
import project4_retention_learning as learning
import project4_energy_replay as stage3
import project4_bridge_memory as selector

ROOT = Path(__file__).resolve().parent
DEFAULT = base.DEFAULT / 'stage3_bridge_extension'
FAMILY = 'p3_active200'
SCHEMA = 'p3_downstream_bridge_extension_v1'
NEW = 'bridge_extension_new'
REUSED = 'stage3_control_reused'
FIELDS, KEYS = stage3.FIELDS, stage3.KEYS
read_json, require = base.read_json, base.require


def code_hashes():
    return {**stage3.code_hashes(), Path(__file__).name: common.sha256(__file__),
            Path(selector.__file__).name: common.sha256(selector.__file__)}


def prepare(args):
    parent = args.stage3_run.resolve()
    destination = args.run_dir.resolve()
    require(not destination.exists(), f'Already prepared: {destination}. Use train/status.')
    require(destination != parent and parent not in destination.parents and destination not in parent.parents,
            'Use a separate sibling directory, outside the frozen Stage 3 run.')
    protocol, all_jobs, all_sources = stage3.load_run(parent)
    require(protocol['seeds'] == base.SEEDS and protocol['budgets'] == base.BUDGETS,
            'Expected the original five seeds and budgets 60/150/300.')
    require(np.isclose(protocol['training']['replay_share'], .2, rtol=0, atol=1e-12),
            'Expected the original 20% replay weighting.')
    old_jobs = [j for j in all_jobs if j['family'] == FAMILY]
    records = {j['job_id']: base.completed(parent, j) for j in old_jobs}
    require(old_jobs and all(r is not None for r in records.values()), 'Finish P3 Stage 3 fits first.')
    sources = {k: v for k, v in all_sources.items() if v['family'] == FAMILY}
    require(sorted(s['seed'] for s in sources.values()) == base.SEEDS, 'Missing P3 source seed.')
    require(all(s['source_training_labels'] == 200 and s['columns'] == common.COMPOSITION
                for s in sources.values()), 'Expected active-200 composition checkpoints.')
    report = parent / 'report'
    manifest = read_json(report / 'report_manifest.json')
    require(manifest['protocol_sha256'] == common.sha256(parent / 'protocol.json'),
            'Stage 3 report/protocol mismatch.')
    required_report = ['metrics.csv', 'costs.csv', 'frozen_source_metrics.csv', 'frozen_thresholds.json']
    for name in required_report:
        base.checked(report / name, manifest['files_sha256'][name])
    old_cutoffs = {r['job_id']: r for r in read_json(report / 'frozen_thresholds.json')}
    for j in old_jobs:
        require(old_cutoffs[j['job_id']]['model_sha256'] == records[j['job_id']]['model_sha256'],
                'Stage 3 cutoff/checkpoint mismatch.')

    # Check every path rather than silently taking the last policy's recipe IDs.
    paths = {}
    for job in old_jobs:
        key = (job['seed'], job['budget'])
        ids = job['target_recipe_ids']
        require(len(ids) == len(set(ids)) == job['budget'], 'Invalid target path.')
        require(key not in paths or paths[key] == ids, 'Stage 3 policies used different target paths.')
        paths[key] = ids
    require(set(paths) == {(s, b) for s in base.SEEDS for b in base.BUDGETS}, 'Missing target path.')
    target_train = pd.read_csv(parent / 'inputs/target_train.csv')
    require(target_train.split.eq('train').all(), 'Target training file contains held-out rows.')
    for ids in paths.values():
        require(set(ids) <= set(target_train.recipe_id), 'Target path includes a nontraining recipe.')
    source = pd.read_csv(parent / 'inputs/source_train.csv')
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.bridge_prepare_', dir=destination.parent))
    try:
        for name in ['inputs', 'source_models', 'replay_manifests', 'code_snapshot', 'baseline']:
            (staging / name).mkdir()
        for path in (parent / 'inputs').glob('*.csv'):
            if path.name in [f'{scope}_{split}.csv' for scope in ['source', 'target']
                             for split in ['train', 'validation', 'test']]:
                shutil.copy2(path, staging / 'inputs' / path.name)
        shutil.copy2(parent / 'inputs/descriptor_constants.json', staging / 'inputs/descriptor_constants.json')
        jobs, imported_cutoffs, memory_summary, availability, overlap = [], [], [], [], []
        for tag, spec in sources.items():
            shutil.copy2(parent / spec['snapshot'], staging / spec['snapshot'])
            candidates = pd.read_csv(parent / f'replay_manifests/{tag}__energy_ranked_candidates.csv')
            require(candidates.eligible.eq(True).all() and candidates.split.eq('train').all(),
                    'Frozen ranked pool contains ineligible or held-out rows.')
            original = source.set_index('recipe_id').loc[candidates.recipe_id]
            require(np.array_equal(original.instability_label, candidates.instability_label) and
                    np.allclose(original[common.X_COLUMNS], candidates[common.X_COLUMNS], rtol=0, atol=1e-12),
                    'Frozen candidate labels/compositions differ from source training data.')
            model = joblib.load(parent / spec['snapshot'])
            rechecked = learning.eligible_memory(source, candidates.recipe_id.tolist(), model, spec['original_cutoff'])
            require(rechecked.eligible.all(), 'Source correctness eligibility changed.')
            memories, info, ranked = selector.choose_memories(candidates, spec['seed'])
            availability.append({'family': FAMILY, 'seed': spec['seed'], **info})
            common.write_csv(staging / f'replay_manifests/{tag}__energy_ranked_candidates.csv', ranked)
            for policy, memory in memories.items():
                name = f'replay_manifests/{tag}__{policy}.csv'
                common.write_csv(staging / name, memory)
                memory_summary.append({'family': FAMILY, 'seed': spec['seed'], 'policy': policy,
                    'memory_size': len(memory), 'positive_labels': int(memory.instability_label.sum()),
                    'g_mix_min_kJ_mol': float(memory[stage3.energy.ENERGY].min()),
                    'g_mix_max_kJ_mol': float(memory[stage3.energy.ENERGY].max()),
                    'distinct_blocks': int(memory.block_id.nunique())})
                for old_policy in stage3.POLICIES:
                    old_file = parent / f'replay_manifests/{tag}__{old_policy}.csv'
                    if old_file.exists():
                        old_memory = pd.read_csv(old_file)
                        overlap.append({'family': FAMILY, 'seed': spec['seed'], 'policy': policy,
                                        'compared_with': old_policy,
                                        'shared_examples': len(set(memory.recipe_id) & set(old_memory.recipe_id))})
                reusable = parent / f'replay_manifests/{tag}__bridge_class_matched20.csv'
                reuse = False
                if policy == selector.CONTROL and reusable.exists():
                    previous = pd.read_csv(reusable)
                    # Match fitting order, labels and inputs, not merely set membership.
                    reuse = (previous.recipe_id.tolist() == memory.recipe_id.tolist() and
                             np.array_equal(previous.instability_label, memory.instability_label) and
                             np.allclose(previous[common.X_COLUMNS], memory[common.X_COLUMNS], rtol=0, atol=1e-12))
                for budget in base.BUDGETS:
                    for initialization in ['transfer', 'scratch']:
                        job = {'stage': 3, 'family': FAMILY, 'seed': spec['seed'], 'policy': policy,
                            'initialization': initialization, 'budget': budget, 'source_tag': tag,
                            'source_file': spec['snapshot'], 'target_recipe_ids': paths[(spec['seed'], budget)],
                            'memory_file': name, 'origin': REUSED if reuse else NEW}
                        job['job_id'] = base.job_id(job)
                        if reuse:
                            matches = [j for j in old_jobs if j['seed'] == spec['seed'] and j['budget'] == budget
                                       and j['initialization'] == initialization and j['policy'] == 'bridge_class_matched20']
                            require(len(matches) == 1, 'Missing reusable control fit.')
                            old = matches[0]; record = records[old['job_id']]
                            require(old['target_recipe_ids'] == job['target_recipe_ids'] and
                                    record['replay_size'] == 20 and np.isclose(record['replay_weight_share'], .2),
                                    'Reusable control training inputs differ.')
                            folder = base.job_folder(staging, job); folder.mkdir(parents=True)
                            shutil.copy2(base.job_folder(parent, old) / 'model.joblib', folder / 'model.joblib')
                            common.write_json(folder / 'record.json', {**record, **job, 'reused': True,
                                'fit_seconds': 0., 'historical_fit_seconds': record.get('historical_fit_seconds', 0.) + record['fit_seconds'],
                                'reused_from_job_id': old['job_id']})
                            imported_cutoffs.append({**old_cutoffs[old['job_id']], 'job_id': job['job_id']})
                        jobs.append(job)
            overlap.append({'family': FAMILY, 'seed': spec['seed'], 'policy': selector.POLICY,
                            'compared_with': selector.CONTROL,
                            'shared_examples': len(set(memories[selector.POLICY].recipe_id) & set(memories[selector.CONTROL].recipe_id))})

        require(len(jobs) == 60 and len({j['job_id'] for j in jobs}) == 60, 'Expected exactly 60 extension entries.')
        # Historical metrics are copied unchanged; they do not enter selection or fitting.
        for name in required_report[:3]:
            table = pd.read_csv(report / name)
            common.write_csv(staging / 'baseline' / name, table.loc[table.family == FAMILY])
        baseline_metrics = pd.read_csv(staging / 'baseline/metrics.csv')
        identities = ['family', 'seed', 'policy', 'initialization', 'budget']
        require(len(baseline_metrics[identities].drop_duplicates()) == len(old_jobs),
                'Stage 3 report does not cover all completed P3 entries.')
        common.write_json(staging / 'inputs/jobs.json', jobs)
        common.write_json(staging / 'inputs/sources.json', sources)
        common.write_json(staging / 'inputs/imported_cutoffs.json', imported_cutoffs)
        for name, rows in [('memory_summary', memory_summary), ('availability', availability), ('memory_overlap', overlap)]:
            common.write_csv(staging / f'replay_manifests/{name}.csv', pd.DataFrame(rows))
        for name in code_hashes():
            shutil.copy2(ROOT / name, staging / 'code_snapshot' / name)
        frozen = [p for name in ['inputs', 'source_models', 'replay_manifests', 'code_snapshot', 'baseline']
                  for p in (staging / name).rglob('*') if p.is_file()]
        common.write_json(staging / 'protocol.json', {'schema': SCHEMA,
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'exploratory_after_stage3_test_inspection': True,
            'selection_rule': 'First 10 eligible positives descending in predicted g_mix; then first 10 eligible negatives AFTER the tenth selected positive. Earlier negatives excluded. Ties: recipe_id ascending.',
            'control_rule': 'Random 10 positives and 10 negatives from entire eligible pool; Stage 3 seed convention.',
            'code_sha256': code_hashes(),
            'files_sha256': {str(p.relative_to(staging)): common.sha256(p) for p in frozen},
            'sklearn_version': version('scikit-learn'), 'training': protocol['training'],
            'seeds': base.SEEDS, 'budgets': base.BUDGETS, 'family': FAMILY,
            'parent_run': str(parent), 'parent_protocol_sha256': common.sha256(parent / 'protocol.json'),
            'parent_report_sha256': common.sha256(report / 'report_manifest.json'),
            'baseline_entries': len(old_jobs), 'planned_entries': len(jobs),
            'reused_controls': len(imported_cutoffs), 'new_fits': len(jobs) - len(imported_cutoffs),
            'ridge_training_recipe_count': protocol['ridge_training_recipe_count'],
            'energy_values_reused_from_stage3': True, 'new_reference_calculations': 0})
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(pd.DataFrame(availability).to_string(index=False))
    print(pd.DataFrame(memory_summary).to_string(index=False))
    print(f'Prepared 60 extension entries: {len(imported_cutoffs)} reused controls; {60-len(imported_cutoffs)} new committee fits.')
    print('Prepared:', destination)


def load_run(directory):
    protocol = read_json(directory / 'protocol.json')
    require(protocol['schema'] == SCHEMA, 'Unexpected extension schema.')
    require(protocol['code_sha256'] == code_hashes(), 'A frozen script changed after preparation.')
    require(version('scikit-learn') == protocol['sklearn_version'] == '1.8.0', 'Use the existing sklearn 1.8.0 environment.')
    for filename, digest in protocol['files_sha256'].items():
        base.checked(directory / filename, digest)
    return protocol, read_json(directory / 'inputs/jobs.json'), read_json(directory / 'inputs/sources.json')


def train(args):
    protocol, jobs, _ = load_run(args.run_dir)
    selected = [j for j in jobs if j['origin'] == NEW]
    if args.limit is not None:
        selected = selected[:args.limit]
    pending = [j for j in selected if base.completed(args.run_dir, j) is None]
    print(f'Selected new entries: {len(selected)}; pending: {len(pending)}.', flush=True)
    if pending:
        executor = ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'))
        futures = []
        try:
            futures = [executor.submit(base.execute_job, str(args.run_dir), j, protocol['training']) for j in pending]
            for i, future in enumerate(as_completed(futures), 1):
                print(f'Completed {i}/{len(pending)}: {future.result()}', flush=True)
        finally:
            for future in futures:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
    status(args)


def status(args):
    protocol, jobs, _ = load_run(args.run_dir)
    rows = [{k: j[k] for k in ['policy', 'origin']} | {'complete': base.completed(args.run_dir, j) is not None} for j in jobs]
    print(pd.DataFrame(rows).groupby(['policy', 'origin']).complete.agg(planned='size', complete='sum').to_string())
    print('Frozen historical P3 entries:', protocol['baseline_entries'])


def paired_contrasts(metrics):
    controls = ['none', 'random20', 'slowest20', 'highest_g20', 'lowest_g20',
                'energy_bridge20', 'bridge_class_matched20', *selector.POLICIES]
    comparisons = [(p, q) for p in selector.POLICIES for q in controls if p != q]
    rows = []
    for a, b in comparisons:
        pair = metrics.loc[metrics.policy == a].merge(metrics.loc[metrics.policy == b],
                 on=KEYS, suffixes=('', '_control'), validate='one_to_one')
        for _, r in pair.iterrows():
            rows.append({**{k: r[k] for k in KEYS}, 'contrast': f'{a}_minus_{b}', 'policy': '',
                         **{f'delta_{f}': r[f] - r[f'{f}_control'] for f in FIELDS}})
    keys = [k for k in KEYS if k != 'initialization'] + ['policy']
    new = metrics.loc[metrics.policy.isin(selector.POLICIES)]
    pair = new.loc[new.initialization == 'transfer'].merge(new.loc[new.initialization == 'scratch'],
        on=keys, suffixes=('', '_control'), validate='one_to_one')
    for _, r in pair.iterrows():
        rows.append({**{k: r[k] for k in keys}, 'initialization': 'transfer_vs_scratch',
                     'contrast': 'transfer_minus_matched_scratch',
                     **{f'delta_{f}': r[f] - r[f'{f}_control'] for f in FIELDS}})
    return pd.DataFrame(rows)


def plot_report(metrics, output):
    selected = metrics.query("domain=='overall' and threshold_rule=='target_validation' and initialization=='transfer'")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    policies = {'none': '#666666', 'random20': '#387cab', 'slowest20': '#b37024',
                selector.POLICY: '#875ca6', selector.CONTROL: '#26846c'}
    for ax, scope in zip(axes, ['source', 'target']):
        for policy, color in policies.items():
            group = selected.loc[(selected.scope == scope) & (selected.policy == policy)].groupby('budget').average_precision.agg(['mean', 'std', 'count'])
            if group.empty:
                continue
            ax.plot(group.index, group['mean'], 'o-', color=color, label=f'{policy} (n={int(group["count"].min())})')
            ax.fill_between(group.index, (group['mean']-group['std']).clip(0, 1),
                            (group['mean']+group['std']).clip(0, 1), color=color, alpha=.09)
        ax.set(title=f'P3 active-200: {scope}', xlabel='Target training labels', ylabel='Average precision',
               ylim=(0, 1.02), xticks=base.BUDGETS)
        ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.suptitle('Exploratory downstream bridge extension')
    fig.text(.5, .015, 'Means and sample SD across adaptation seeds. Strict Stage 3 bridge comparisons use only shared seeds in the tables.', ha='center', fontsize=8)
    fig.tight_layout(rect=[0, .05, 1, .94]); fig.savefig(output / 'bridge_extension_AP.png', dpi=180); plt.close(fig)


def report(args):
    protocol, jobs, _ = load_run(args.run_dir)
    records = [base.completed(args.run_dir, j) for j in jobs]
    require(all(r is not None for r in records), 'Finish all planned fits before reporting.')
    output = args.run_dir / 'report'
    require(not output.exists(), f'Report is already frozen: {output}')
    building = args.run_dir / 'report_in_progress'; building.mkdir(exist_ok=True)
    imported = {r['job_id']: r for r in read_json(args.run_dir / 'inputs/imported_cutoffs.json')}
    source_val = pd.read_csv(args.run_dir / 'inputs/source_validation.csv')
    target_val = pd.read_csv(args.run_dir / 'inputs/target_validation.csv')
    cutoffs = []
    for job, record in zip(jobs, records):
        if job['origin'] == REUSED:
            cutoff = imported[job['job_id']]
            require(cutoff['model_sha256'] == record['model_sha256'], 'Reused cutoff/model mismatch.')
        else:
            package = joblib.load(base.job_folder(args.run_dir, job) / 'model.joblib')
            cutoff = {'job_id': job['job_id'], 'model_sha256': record['model_sha256'],
                'target_cutoff': common.choose_threshold(target_val.instability_label, learning.scores(package, target_val).mean(axis=0)),
                'source_cutoff': common.choose_threshold(source_val.instability_label, learning.scores(package, source_val).mean(axis=0))}
        cutoffs.append(cutoff)
    frozen = building / 'frozen_thresholds.json'
    if frozen.exists():
        require(read_json(frozen) == cutoffs, 'Frozen cutoffs changed.')
    else:
        common.write_json(frozen, cutoffs)
    print('All extension fits and validation cutoffs fixed. Reading test labels now.', flush=True)
    source = pd.read_csv(args.run_dir / 'inputs/source_test.csv')
    target = pd.read_csv(args.run_dir / 'inputs/target_test.csv')
    predictions, values, costs = [], [], []
    for job, record, cutoff in zip(jobs, records, cutoffs):
        package = joblib.load(base.job_folder(args.run_dir, job) / 'model.joblib')
        ids = {k: job[k] for k in ['family', 'seed', 'policy', 'initialization', 'budget']}
        for scope, rows in [('source', source), ('target', target)]:
            scores = learning.scores(package, rows).mean(axis=0)
            pred, metrics = base.evaluate_rows(rows, scores, cutoff['target_cutoff'], ids, scope, 'target_validation')
            predictions.append(pred); values.extend(metrics)
            if scope == 'source':
                _, diagnostic = base.evaluate_rows(rows, scores, cutoff['source_cutoff'], ids, scope, 'source_validation_diagnostic')
                values.extend(diagnostic)
        costs.append({**ids, 'origin': job['origin'], 'fit_seconds_this_stage': record['fit_seconds'],
            'historical_fit_seconds': record.get('historical_fit_seconds', 0.),
            'epoch_cap_members': record['epoch_cap_members'], 'source_memory_labels': 20,
            'replay_weight_share': record['replay_weight_share'],
            'source_pretraining_labels_in_weights': 200 if job['initialization'] == 'transfer' else 0,
            'target_validation_labels': len(target_val), 'source_validation_labels_for_diagnostic': len(source_val),
            'ridge_historical_training_recipes': protocol['ridge_training_recipe_count'] if job['policy'] == selector.POLICY else 0,
            'ridge_used_for_memory_or_control_counts': job['policy'] == selector.POLICY,
            'target_reference_seconds_sum': record['target_reference_seconds'],
            'replay_historical_reference_seconds_sum': record['replay_historical_reference_seconds'],
            'target_curriculum_uses_previous_transfer_teacher': True, 'new_reference_calculations': 0})
    new_metrics = pd.DataFrame(values)
    metrics = pd.concat([pd.read_csv(args.run_dir / 'baseline/metrics.csv'), new_metrics], ignore_index=True)
    require(not metrics.duplicated(KEYS + ['policy']).any(), 'Duplicate report metric identity.')
    keys = [k for k in KEYS if k != 'seed'] + ['policy']
    summary = metrics.groupby(keys)[FIELDS].agg(['mean', 'std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary = summary.join(metrics.groupby(keys).seed.nunique().rename('seeds_evaluated')).reset_index()
    paired = paired_contrasts(metrics)
    pair_keys = [k for k in keys if k != 'policy'] + ['contrast', 'policy']
    paired_summary = paired.groupby(pair_keys)[[f'delta_{f}' for f in FIELDS]].agg(['mean', 'std'])
    paired_summary.columns = ['_'.join(c) for c in paired_summary.columns]
    paired_summary = paired_summary.join(paired.groupby(pair_keys).seed.nunique().rename('paired_seeds'))
    paired_summary = paired_summary.join(paired.assign(ap_win=paired.delta_average_precision > 1e-12).groupby(pair_keys).ap_win.sum().rename('AP_positive_seed_count'))
    frozen_source = pd.read_csv(args.run_dir / 'baseline/frozen_source_metrics.csv')
    original = frozen_source.query("scope=='source' and domain=='overall'").set_index('seed').average_precision
    retention = metrics.query("scope=='source' and domain=='overall' and initialization=='transfer' and threshold_rule=='target_validation'").copy()
    retention['original_source_AP'] = retention.seed.map(original)
    retention['source_AP_change_from_pretraining'] = retention.average_precision - retention.original_source_AP
    old_costs = pd.read_csv(args.run_dir / 'baseline/costs.csv')
    # Preserve Stage 3's recorded cost values exactly. The extension report
    # appends new entries; it must not rewrite the historical accounting of the
    # frozen parent run.
    old_costs['origin'] = 'stage3_metrics_reused'
    for name, table in [('metrics', metrics), ('summary', summary), ('paired_comparisons', paired),
                        ('paired_summary', paired_summary.reset_index()), ('source_retention_changes', retention),
                        ('costs', pd.concat([old_costs, pd.DataFrame(costs)], ignore_index=True)),
                        ('frozen_source_metrics', frozen_source)]:
        common.write_csv(building / f'{name}.csv', table)
    pd.concat(predictions, ignore_index=True).to_csv(building / 'extension_test_predictions.csv.gz', index=False,
                                                   compression={'method': 'gzip', 'mtime': 0})
    for name in ['memory_summary.csv', 'availability.csv', 'memory_overlap.csv']:
        shutil.copy2(args.run_dir / 'replay_manifests' / name, building / name)
    plot_report(metrics, building)
    common.write_json(building / 'report_manifest.json', {'protocol_sha256': common.sha256(args.run_dir / 'protocol.json'),
        'exploratory_after_stage3_test_inspection': True, 'extension_entries': len(jobs),
        'baseline_entries_reused': protocol['baseline_entries'],
        'predictions_scope': 'Extension policies only; historical predictions remain in original Stage 3 report.',
        'files_sha256': {p.name: common.sha256(p) for p in building.iterdir() if p.is_file() and p.name != 'report_manifest.json'}})
    building.rename(output)
    show = summary.query("domain=='overall' and initialization=='transfer' and threshold_rule=='target_validation'")
    print(show[['policy', 'budget', 'scope', 'seeds_evaluated', 'average_precision_mean', 'recall_instability_mean', 'f1_instability_mean']].round(4).to_string(index=False))
    print('\nSaved:', output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for command in ['prepare', 'train', 'status', 'report']:
        sub = commands.add_parser(command)
        sub.add_argument('--run-dir', type=Path, default=DEFAULT)
        if command == 'prepare':
            sub.add_argument('--stage3-run', type=Path, default=stage3.DEFAULT)
        if command == 'train':
            sub.add_argument('--workers', type=int, default=8)
            sub.add_argument('--limit', type=int, help='First N new entries, including any already completed.')
    args = parser.parse_args()
    require(version('scikit-learn') == '1.8.0', 'Use the existing cosmo-rs-1 environment with sklearn 1.8.0.')
    if args.command == 'train':
        require(1 <= args.workers <= 8 and (args.limit is None or args.limit > 0), 'Invalid workers or limit.')
    args.run_dir = args.run_dir.resolve(); args.run_dir.parent.mkdir(parents=True, exist_ok=True)
    with (args.run_dir.parent / (args.run_dir.name + '.lock')).open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another command is using this extension run.') from error
        globals()[args.command](args)


if __name__ == '__main__':
    main()
