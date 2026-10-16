"""Stage 3: frozen-energy replay on the completed retention benchmark.

Root entry point: prepare -> train -> report. No ORCA/COSMO-RS calls.
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
import project4_energy_memory as energy

ROOT = Path(__file__).resolve().parent
PARENT = base.DEFAULT
DEFAULT = PARENT / 'stage3_energy'
RIDGE_RUN = ROOT / 'results/ml/project1_regression/run_20261004_134348_552708'
FAMILIES = base.REPLAY_FAMILIES
POLICIES = ['none', 'random20', 'slowest20', 'time_class_matched20',
            'highest_g20', 'highest_g_class_matched20',
            'lowest_g20', 'lowest_g_class_matched20',
            'energy_bridge20', 'bridge_class_matched20']
FIELDS = ['average_precision', 'pr_auc_trapezoid', 'f1_instability',
          'precision_instability', 'recall_instability', 'recall_no_instability', 'fp', 'fn']
KEYS = ['family', 'seed', 'initialization', 'budget', 'scope', 'domain', 'threshold_rule']
read_json, require = base.read_json, base.require


def code_hashes():
    return {**base.hashes(), Path(__file__).name: common.sha256(__file__),
            Path(energy.__file__).name: common.sha256(energy.__file__)}


def canonical_inputs(parent, source, candidates, spec):
    """Recheck the frozen candidate pool against source training data and weights."""
    require(len(candidates) == spec['source_training_labels'], 'Checkpoint candidate count changed.')
    require(set(candidates.recipe_id) <= set(source.recipe_id), 'Candidate is outside source training.')
    original = source.set_index('recipe_id').loc[candidates.recipe_id]
    require(np.array_equal(original.instability_label, candidates.instability_label), 'Source labels differ.')
    require(np.allclose(original[common.X_COLUMNS], candidates[common.X_COLUMNS], rtol=0, atol=1e-12),
            'Source compositions differ.')
    model = joblib.load(parent / spec['snapshot'])
    reconstructed = learning.eligible_memory(source, candidates.recipe_id.tolist(), model, spec['original_cutoff'])
    require(np.array_equal(reconstructed.eligible, candidates.eligible), 'Frozen correctness eligibility changed.')
    return model


def prepare(args):
    require(not args.run_dir.exists(), f'Already prepared: {args.run_dir}. Use train/status.')
    parent = args.retention_run.resolve()
    require(args.run_dir.resolve() != parent, 'Stage 3 must have its own output directory.')
    old_protocol, old_jobs, sources = base.load_run(parent)
    require(all(base.completed(parent, j) is not None for j in old_jobs), 'Complete Stages 1/2 first.')
    report_dir = parent / 'report_stage12'
    report_manifest = read_json(report_dir / 'report_manifest.json')
    require(report_manifest['protocol_sha256'] == common.sha256(parent / 'protocol.json'),
            'Parent report belongs to a different protocol.')
    base.checked(report_dir / 'frozen_thresholds.json', report_manifest['files_sha256']['frozen_thresholds.json'])
    old_cutoffs = {r['job_id']: r for r in read_json(report_dir / 'frozen_thresholds.json')}
    audit = read_json(parent / 'input_audit.json')
    ridge, ridge_metadata, ridge_selection = energy.load_ridge(args.ridge_run, audit['source_timing_protocol'])
    source = pd.read_csv(parent / 'inputs/source_train.csv')
    predicted, prediction_seconds = energy.predicted_energies(ridge, source)
    energy_columns = ['recipe_id', 'predicted_g_excess_kJ_mol', 'analytical_g_ideal_kJ_mol', energy.ENERGY]
    sources = {k: v for k, v in sources.items() if v['family'] in FAMILIES}
    # Stage into a new directory; parent files are read-only throughout.
    args.run_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.energy_prepare_', dir=args.run_dir.parent))
    try:
        for name in ['inputs', 'source_models', 'replay_manifests', 'ridge', 'code_snapshot']:
            (staging / name).mkdir()
        for scope in ['source', 'target']:
            for split in ['train', 'validation', 'test']:
                name = f'inputs/{scope}_{split}.csv'
                shutil.copy2(parent / name, staging / name)
        shutil.copy2(parent / 'inputs/descriptor_constants.json', staging / 'inputs/descriptor_constants.json')
        common.write_csv(staging / 'inputs/source_training_predicted_energies.csv', predicted)
        for name in ['ridge_model.joblib', 'run_metadata.json', 'selected_settings.json']:
            shutil.copy2(args.ridge_run / name, staging / 'ridge' / name)
        memories, memory_rows, availability, candidates_by_tag = {}, [], [], {}
        for tag, spec in sources.items():
            shutil.copy2(parent / spec['snapshot'], staging / spec['snapshot'])
            candidates = pd.read_csv(parent / f'replay_manifests/{tag}__candidates.csv')
            canonical_inputs(parent, source, candidates, spec)
            candidates = candidates.merge(predicted[energy_columns], on='recipe_id', validate='one_to_one')
            new_memories, statuses, ranked = energy.choose_energy_memories(candidates, spec['seed'])
            common.write_csv(staging / f'replay_manifests/{tag}__energy_ranked_candidates.csv', ranked)
            candidates_by_tag[tag] = ranked
            availability += [{**r, 'family': spec['family'], 'seed': spec['seed']} for r in statuses]
            all_memories = {policy: pd.read_csv(parent / f'replay_manifests/{tag}__{policy}.csv').merge(
                predicted[energy_columns], on='recipe_id', validate='one_to_one') for policy in learning.POLICIES}
            all_memories.update(new_memories)
            for policy, memory in all_memories.items():
                require(len(memory) == 20 and memory.recipe_id.nunique() == 20, 'Memory must have 20 distinct examples.')
                require(set(memory.recipe_id) <= set(ranked.recipe_id), 'Memory includes ineligible examples.')
                name = f'replay_manifests/{tag}__{policy}.csv'
                common.write_csv(staging / name, memory)
                memories[f'{tag}:{policy}'] = name
                memory_rows.append({'family': spec['family'], 'seed': spec['seed'], 'policy': policy,
                                    'eligible_examples': len(ranked), 'memory_size': 20,
                                    'positive_labels': int(memory.instability_label.sum()),
                                    'g_mix_min_kJ_mol': float(memory[energy.ENERGY].min()),
                                    'g_mix_max_kJ_mol': float(memory[energy.ENERGY].max()),
                                    'g_mix_mean_kJ_mol': float(memory[energy.ENERGY].mean()),
                                    'recorded_search_seconds': float(memory.stability_seconds.sum()),
                                    'distinct_blocks': int(memory.block_id.nunique())})
        # Import all four existing policies as exact checkpoints. No baseline
        # is retrained and its original validation cutoffs are retained.
        jobs, imported_cutoffs = [], []
        old_selected = [j for j in old_jobs if j['family'] in FAMILIES]
        for old in old_selected:
            record = base.completed(parent, old)
            job = {**old, 'stage': 3, 'origin': 'stage12_reused'}
            job['memory_file'] = None if old['policy'] == 'none' else memories[f"{old['source_tag']}:{old['policy']}"]
            folder = base.job_folder(staging, job); folder.mkdir(parents=True)
            shutil.copy2(base.job_folder(parent, old) / 'model.joblib', folder / 'model.joblib')
            require(common.sha256(folder / 'model.joblib') == record['model_sha256'], 'Baseline copy changed.')
            cutoff = old_cutoffs[old['job_id']]
            require(cutoff['model_sha256'] == record['model_sha256'], 'Parent cutoff/model mismatch.')
            imported_cutoffs.append(cutoff)
            common.write_json(folder / 'record.json', {**record, **job, 'reused': True, 'fit_seconds': 0.,
                'historical_fit_seconds': record.get('historical_fit_seconds', 0.) + record['fit_seconds'],
                'reused_from': str((base.job_folder(parent, old) / 'model.joblib').resolve())})
            jobs.append(job)
        paths = {(j['seed'], j['budget']): j['target_recipe_ids'] for j in old_selected}
        for tag, spec in sources.items():
            for policy in list(energy.PAIRS) + list(energy.PAIRS.values()):
                if f'{tag}:{policy}' not in memories:
                    continue  # Infeasible strict bridge; explicitly listed in availability.csv.
                for budget in old_protocol['budgets']:
                    for initialization in ['transfer', 'scratch']:
                        job = {'stage': 3, 'family': spec['family'], 'seed': spec['seed'],
                               'policy': policy, 'initialization': initialization, 'budget': budget,
                               'source_tag': tag, 'source_file': spec['snapshot'],
                               'target_recipe_ids': paths[(spec['seed'], budget)],
                               'memory_file': memories[f'{tag}:{policy}'], 'origin': 'energy_stage3'}
                        job['job_id'] = base.job_id(job); jobs.append(job)
        require(len({j['job_id'] for j in jobs}) == len(jobs), 'Duplicate planned job.')
        common.write_json(staging / 'inputs/jobs.json', jobs)
        common.write_json(staging / 'inputs/sources.json', sources)
        common.write_json(staging / 'inputs/imported_cutoffs.json', imported_cutoffs)
        common.write_csv(staging / 'replay_manifests/memory_summary.csv', pd.DataFrame(memory_rows))
        common.write_csv(staging / 'replay_manifests/availability.csv', pd.DataFrame(availability))
        overlap = []
        for tag, spec in sources.items():
            subsets = {key.split(':', 1)[1]: set(pd.read_csv(staging / filename).recipe_id)
                       for key, filename in memories.items() if key.startswith(tag+':')}
            for a, ids in subsets.items():
                for b, other in subsets.items():
                    if a < b:
                        overlap.append({'family': spec['family'], 'seed': spec['seed'],
                                        'policy_a': a, 'policy_b': b, 'shared_examples': len(ids & other)})
        common.write_csv(staging / 'replay_manifests/memory_overlap.csv', pd.DataFrame(overlap))
        for name in code_hashes():
            shutil.copy2(ROOT / name, staging / 'code_snapshot' / name)
        frozen = [p for name in ['inputs', 'source_models', 'replay_manifests', 'ridge', 'code_snapshot']
                  for p in (staging / name).rglob('*') if p.is_file()]
        common.write_json(staging / 'protocol.json', {
            'schema': 'energy_replay_stage3_v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
            'code_sha256': code_hashes(), 'files_sha256': {str(p.relative_to(staging)): common.sha256(p) for p in frozen},
            'sklearn_version': version('scikit-learn'), 'training': old_protocol['training'],
            'seeds': old_protocol['seeds'], 'budgets': old_protocol['budgets'], 'families': FAMILIES,
            'parent_run': str(parent), 'parent_protocol_sha256': common.sha256(parent / 'protocol.json'),
            'ridge_run': str(args.ridge_run.resolve()), 'ridge_training_recipe_count': ridge_metadata['training_sizes'][-1],
            'ridge_split_counts': ridge_metadata['split_counts'], 'ridge_selection': ridge_selection,
            'ridge_predictions_seconds': prediction_seconds, 'temperature_K': energy.TEMPERATURE,
            'new_reference_calculations': 0, 'imported_fits': len(old_selected),
            'new_fits': len(jobs)-len(old_selected), 'bridge_infeasibility': 'Skip that family/seed and its paired control; never substitute a different anchor.',
            'target_curriculum': 'Same previous transfer_active paths for all arms; no new active acquisition.',
            'energy_role': 'Frozen predicted homogeneous molar mixing free energy for memory selection only.',
            'evaluation': 'Existing source/target held-out tests; follow-up comparison, not new untouched confirmation.'})
        staging.rename(args.run_dir)
    except BaseException:
        if staging.exists(): shutil.rmtree(staging)
        raise
    show = pd.DataFrame(memory_rows)
    print(show[['family','seed','policy','positive_labels','g_mix_mean_kJ_mol']].to_string(index=False))
    print('\nStrict bridge feasibility:')
    print(pd.DataFrame(availability).query("policy=='energy_bridge20'")[
        ['family','seed','status','reason']].to_string(index=False))
    print(f'\nImported {len(old_selected)} fits; planned {len(jobs)-len(old_selected)} new committee fits.')
    print('Prepared:', args.run_dir)


def load_run(directory):
    protocol = read_json(directory / 'protocol.json')
    require(protocol['schema'] == 'energy_replay_stage3_v1', 'Unexpected Stage 3 schema.')
    require(protocol['code_sha256'] == code_hashes(), 'A frozen script changed after preparation.')
    require(version('scikit-learn') == protocol['sklearn_version'] == '1.8.0', 'Use sklearn 1.8.0.')
    for filename, digest in protocol['files_sha256'].items(): base.checked(directory / filename, digest)
    return protocol, read_json(directory / 'inputs/jobs.json'), read_json(directory / 'inputs/sources.json')


def train(args):
    protocol, jobs, _ = load_run(args.run_dir)
    selected = [j for j in jobs if j['origin'] == 'energy_stage3']
    if args.limit is not None: selected = selected[:args.limit]
    pending = [j for j in selected if base.completed(args.run_dir, j) is None]
    print(f'New-policy entries selected: {len(selected)}; already completed: {len(selected)-len(pending)}; pending: {len(pending)}.', flush=True)
    executor = ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'))
    iterator, futures = iter(pending), {}
    def submit():
        job = next(iterator, None)
        if job is not None:
            future = executor.submit(base.execute_job, str(args.run_dir), job, protocol['training'])
            futures[future] = job['job_id']
    try:
        for _ in range(min(args.workers, len(pending))): submit()
        count = 0
        while futures:
            future = next(as_completed(futures)); futures.pop(future)
            result = future.result(); count += 1
            print(f'Completed {count}/{len(pending)}: {result}', flush=True); submit()
    except KeyboardInterrupt:
        print('\nStopping submissions; allow running fits to save, then repeat this command to resume.', flush=True)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    status(args)


def status(args):
    protocol, jobs, _ = load_run(args.run_dir)
    rows = [{**{k: j[k] for k in ['family','policy','origin']},
             'complete': base.completed(args.run_dir, j) is not None} for j in jobs]
    print(pd.DataFrame(rows).groupby(['family','policy','origin']).complete.agg(planned='size', complete='sum').to_string())
    print('New reference calculations:', protocol['new_reference_calculations'])


def paired_contrasts(metrics):
    """Compare exactly aligned seeds; never subtract unpaired policy averages."""
    comparisons = []
    for policy in [p for p in POLICIES if p != 'none']:
        comparisons.append((policy, 'none'))
    for policy, matched in energy.PAIRS.items():
        comparisons.extend([(policy, 'random20'), (policy, matched)])
    comparisons += [('highest_g20', 'lowest_g20'), ('highest_g20','slowest20'),
                    ('lowest_g20','slowest20'), ('energy_bridge20','slowest20'),
                    ('energy_bridge20','highest_g20'), ('energy_bridge20','lowest_g20')]
    rows = []
    for a, b in comparisons:
        pair = metrics.loc[metrics.policy == a].merge(metrics.loc[metrics.policy == b], on=KEYS,
                                                    suffixes=('', '_control'), validate='one_to_one')
        for _, row in pair.iterrows():
            rows.append({**{k: row[k] for k in KEYS}, 'contrast': f'{a}_minus_{b}',
                         **{f'delta_{f}': row[f]-row[f'{f}_control'] for f in FIELDS}})
    weight_keys = [k for k in KEYS if k != 'initialization'] + ['policy']
    pair = metrics.loc[metrics.initialization == 'transfer'].merge(
        metrics.loc[metrics.initialization == 'scratch'], on=weight_keys,
        suffixes=('', '_control'), validate='one_to_one')
    for _, row in pair.iterrows():
        rows.append({**{k: row[k] for k in weight_keys}, 'initialization': 'transfer_vs_scratch',
                     'contrast': 'transfer_minus_matched_scratch',
                     **{f'delta_{f}': row[f]-row[f'{f}_control'] for f in FIELDS}})
    return pd.DataFrame(rows)


def plot_report(metrics, directory):
    main = metrics.query("domain=='overall' and threshold_rule=='target_validation' and initialization=='transfer'")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    colors = {'none':'#666666', 'random20':'#387cab', 'slowest20':'#b37024',
              'highest_g20':'#bb4053', 'lowest_g20':'#26846c', 'energy_bridge20':'#875ca6'}
    for i, family in enumerate(FAMILIES):
        for j, scope in enumerate(['source','target']):
            ax = axes[i,j]
            for policy, color in colors.items():
                selected = main.query('family==@family and scope==@scope and policy==@policy')
                if selected.empty: continue
                grouped = selected.groupby('budget').average_precision.agg(['mean','std','count'])
                n = int(grouped['count'].min())
                ax.plot(grouped.index, grouped['mean'], 'o-', color=color, label=f'{policy} (n={n})')
                ax.fill_between(grouped.index, np.clip(grouped['mean']-grouped['std'].fillna(0),0,1),
                                np.clip(grouped['mean']+grouped['std'].fillna(0),0,1), color=color, alpha=.08)
            ax.set(title=f'{family}: {scope} test', ylabel='Average precision', ylim=(0,1.02), xticks=base.BUDGETS)
            ax.grid(alpha=.2); ax.legend(fontsize=8)
    for ax in axes[-1]: ax.set_xlabel('Target training labels; replay policies add 20 source examples')
    fig.suptitle('Energy-selected replay: source retention and target adaptation')
    fig.text(.5,.01,'Means and sample SD across available seeds; not independent-test confidence intervals. Matched controls are in the tables.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.035,1,.96]); fig.savefig(directory/'energy_retention_AP.png',dpi=180);plt.close(fig)


def report(args):
    protocol, jobs, sources = load_run(args.run_dir)
    records = [base.completed(args.run_dir,j) for j in jobs]
    require(all(r is not None for r in records), 'Finish planned fits before reporting.')
    output = args.run_dir / 'report'
    require(not output.exists(), f'Report already exists and is frozen: {output}')
    building = args.run_dir / 'report_in_progress'; building.mkdir(exist_ok=True)
    imported = {r['job_id']:r for r in read_json(args.run_dir/'inputs/imported_cutoffs.json')}
    target_val = pd.read_csv(args.run_dir/'inputs/target_validation.csv')
    source_val = pd.read_csv(args.run_dir/'inputs/source_validation.csv')
    cutoffs=[]
    for job, record in zip(jobs,records):
        if job['origin']=='stage12_reused':
            entry=imported[job['job_id']]
            require(entry['model_sha256']==record['model_sha256'],'Imported cutoff no longer matches model.')
        else:
            package=joblib.load(base.job_folder(args.run_dir,job)/'model.joblib')
            entry={'job_id':job['job_id'],'model_sha256':record['model_sha256'],
                   'target_cutoff':common.choose_threshold(target_val.instability_label,learning.scores(package,target_val).mean(axis=0)),
                   'source_cutoff':common.choose_threshold(source_val.instability_label,learning.scores(package,source_val).mean(axis=0))}
        cutoffs.append(entry)
    frozen=building/'frozen_thresholds.json'
    if frozen.exists(): require(read_json(frozen)==cutoffs,'Frozen validation cutoffs changed.')
    else: common.write_json(frozen,cutoffs)
    print('All fits and validation cutoffs fixed. Evaluating source and target tests now.',flush=True)
    target=pd.read_csv(args.run_dir/'inputs/target_test.csv');source=pd.read_csv(args.run_dir/'inputs/source_test.csv')
    predictions,metric_rows,original_rows,endpoints,costs=[],[],[],[],[]
    original_ap={}
    for tag,spec in sources.items():
        package=joblib.load(args.run_dir/spec['snapshot'])
        ids={'family':spec['family'],'seed':spec['seed'],'policy':'frozen_source','initialization':'pretrained','budget':0}
        for scope,rows in [('source',source),('target',target)]:
            pred,values=base.evaluate_rows(rows,learning.scores(package,rows).mean(axis=0),spec['original_cutoff'],ids,scope,'original_source_validation')
            predictions.append(pred);original_rows.extend(values)
            if scope=='source':original_ap[tag]=values[0]['average_precision']
    pure=pd.DataFrame(np.eye(5),columns=common.X_COLUMNS)
    for job,record,threshold in zip(jobs,records,cutoffs):
        package=joblib.load(base.job_folder(args.run_dir,job)/'model.joblib')
        ids={k:job[k] for k in ['family','seed','policy','initialization','budget']}
        for scope,rows in [('source',source),('target',target)]:
            score=learning.scores(package,rows).mean(axis=0)
            pred,values=base.evaluate_rows(rows,score,threshold['target_cutoff'],ids,scope,'target_validation')
            predictions.append(pred);metric_rows.extend(values)
            if scope=='source':
                _,diagnostic=base.evaluate_rows(rows,score,threshold['source_cutoff'],ids,scope,'source_validation_diagnostic')
                metric_rows.extend(diagnostic)
        for name,score in zip(common.MOLECULES,learning.scores(package,pure).mean(axis=0)):
            endpoints.append({**ids,'molecule':name,'score':float(score),'reference_label':0,
                              'predicted':int(score>=threshold['target_cutoff'])})
        memory=pd.read_csv(args.run_dir/job['memory_file']) if job['memory_file'] else None
        train=pd.read_csv(args.run_dir/'inputs/target_train.csv').set_index('recipe_id').loc[job['target_recipe_ids']]
        costs.append({**ids,'origin':job['origin'],'fit_seconds_this_stage':record['fit_seconds'],
                      'historical_fit_seconds':record.get('historical_fit_seconds',0.),
                      'epoch_cap_members':record['epoch_cap_members'],
                      'source_memory_labels':0 if memory is None else len(memory),
                      'replay_weight_share':record['replay_weight_share'],
                      'target_validation_labels':len(target_val),'source_validation_labels_for_diagnostic':len(source_val),
                      'source_pretraining_labels_in_weights':sources[job['source_tag']]['source_training_labels'] if job['initialization']=='transfer' else 0,
                      'ridge_historical_training_recipes':protocol['ridge_training_recipe_count'] if job['origin']=='energy_stage3' else 0,
                      'ridge_used_for_memory_or_control_counts':job['origin']=='energy_stage3',
                      'target_reference_seconds_sum':float(train.stability_seconds.sum()),
                      'replay_historical_reference_seconds_sum':0. if memory is None else float(memory.stability_seconds.sum()),
                      'target_curriculum_uses_previous_transfer_teacher':True,'new_reference_calculations':0})
    metrics=pd.DataFrame(metric_rows)
    keys=[k for k in KEYS if k!='seed']+['policy']
    summary=metrics.groupby(keys)[FIELDS].agg(['mean','std'])
    summary.columns=['_'.join(c) for c in summary.columns]
    summary=summary.join(metrics.groupby(keys).seed.nunique().rename('seeds_evaluated')).reset_index()
    paired=paired_contrasts(metrics)
    pair_keys=['family','initialization','budget','scope','domain','threshold_rule','contrast']
    # The policy column is relevant only for transfer-versus-scratch contrasts.
    paired['policy']=paired.get('policy',pd.Series('',index=paired.index)).fillna('')
    pair_keys+=['policy']
    delta_fields=[f'delta_{f}' for f in FIELDS]
    pair_summary=paired.groupby(pair_keys)[delta_fields].agg(['mean','std'])
    pair_summary.columns=['_'.join(c) for c in pair_summary.columns]
    pair_summary=pair_summary.join(paired.groupby(pair_keys).seed.nunique().rename('paired_seeds'))
    pair_summary=pair_summary.join(paired.assign(ap_win=paired.delta_average_precision>1e-12).groupby(pair_keys).ap_win.sum().rename('AP_positive_seed_count'))
    retention=metrics.query("scope=='source' and domain=='overall' and threshold_rule=='target_validation' and initialization=='transfer'").copy()
    retention['original_source_AP']=[original_ap[f'{r.family}__s{r.seed}'] for r in retention.itertuples()]
    retention['source_AP_change_from_pretraining']=retention.average_precision-retention.original_source_AP
    # Compressed predictions keep the per-recipe audit available without the
    # large uncompressed CSV that accumulated in earlier reports.
    pd.concat(predictions,ignore_index=True).to_csv(building/'test_predictions.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    for name,table in [('metrics',metrics),('summary',summary),('paired_comparisons',paired),
                       ('paired_summary',pair_summary.reset_index()),('source_retention_changes',retention),
                       ('frozen_source_metrics',pd.DataFrame(original_rows)),('costs',pd.DataFrame(costs)),
                       ('pure_endpoint_predictions',pd.DataFrame(endpoints))]:
        common.write_csv(building/f'{name}.csv',table)
    for name in ['memory_summary.csv','availability.csv','memory_overlap.csv']:
        shutil.copy2(args.run_dir/'replay_manifests'/name,building/name)
    plot_report(metrics,building)
    common.write_json(building/'report_manifest.json',{'protocol_sha256':common.sha256(args.run_dir/'protocol.json'),
        'evaluated_fits':len(jobs),'files_sha256':{p.name:common.sha256(p) for p in building.iterdir() if p.is_file() and p.name!='report_manifest.json'}})
    building.rename(output)
    show=summary.query("domain=='overall' and threshold_rule=='target_validation' and initialization=='transfer'")
    print(show[['family','policy','budget','scope','seeds_evaluated','average_precision_mean','recall_instability_mean','f1_instability_mean']].round(4).to_string(index=False))
    print('\nSaved:',output)
    print('Energy rank is a replay heuristic; these labels remain finite model-based stability screens.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    for command in ['prepare','train','status','report']:
        p=commands.add_parser(command);p.add_argument('--run-dir',type=Path,default=DEFAULT)
        if command=='prepare':
            p.add_argument('--retention-run',type=Path,default=PARENT)
            p.add_argument('--ridge-run',type=Path,default=RIDGE_RUN)
        if command=='train':
            p.add_argument('--workers',type=int,default=8)
            p.add_argument('--limit',type=int,help='Total NEW-policy entries selected, including already completed new-policy fits.')
    args=parser.parse_args()
    require(version('scikit-learn')=='1.8.0','Use the existing cosmo-rs-1 environment with sklearn 1.8.0.')
    if args.command=='train':
        require(1<=args.workers<=8 and (args.limit is None or args.limit>0),'Invalid workers or limit.')
    args.run_dir=args.run_dir.resolve();args.run_dir.parent.mkdir(parents=True,exist_ok=True)
    with (args.run_dir.parent/(args.run_dir.name+'.lock')).open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:raise RuntimeError('Another command is using this Stage 3 run.') from error
        globals()[args.command](args)


if __name__=='__main__':
    main()
