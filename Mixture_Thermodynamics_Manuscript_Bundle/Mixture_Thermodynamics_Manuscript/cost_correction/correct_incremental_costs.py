#!/usr/bin/env python3
"""Create a derived cost presentation; never edit the frozen scientific report.

Uses only Python's standard library. No fitting or reference calculations.
Usage: python correct_incremental_costs.py --costs PATH/report/costs.csv --output NEW_DIRECTORY
"""
import argparse
import csv
import hashlib
import json
from decimal import Decimal
from pathlib import Path

NEW = 'bridge_extension_new'
OLD = 'stage3_metrics_reused'
REUSED = 'stage3_control_reused'
CONTROL = 'downstream_bridge_class_matched20'

def number(value):
    return Decimal(value or '0')

def fit_id(row, policy=None):
    return f"{row['family']}__s{row['seed']}__{policy or row['policy']}__{row['initialization']}__n{row['budget']}"

def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--costs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = args.costs.resolve()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    with source.open(newline='') as handle:
        original = list(csv.DictReader(handle))
    manifest_path = source.parent / 'report_manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        expected = manifest.get('files_sha256', {}).get(source.name)
        if expected is not None and digest != expected:
            raise ValueError('Input costs do not match their frozen report manifest.')
    if not original or set(row['origin'] for row in original) != {NEW, OLD, REUSED}:
        raise ValueError('Expected the completed P3 bridge-extension cost report.')
    ids = [fit_id(row) for row in original]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate report identity.')
    parent = {fit_id(row): row for row in original if row['origin'] == OLD}
    corrected, unique = [], []
    for row in original:
        origin = row['origin']
        reported_fit = number(row['fit_seconds_this_stage'])
        reported_history = number(row['historical_fit_seconds'])
        if min(reported_fit, reported_history) < 0:
            raise ValueError('Negative duration.')
        canonical = fit_id(row)
        incremental = reported_fit if origin == NEW else Decimal(0)
        history = reported_history if origin != OLD else reported_fit + reported_history
        alias = ''
        if origin == NEW and reported_history != 0:
            raise ValueError('Unexpected inherited fitting time on a new extension fit.')
        if origin == REUSED:
            if row['policy'] != CONTROL or reported_fit != 0:
                raise ValueError('Unexpected reused-control record.')
            canonical = fit_id(row, 'bridge_class_matched20')
            alias = canonical
            previous = parent[canonical]
            previous_time = number(previous['fit_seconds_this_stage']) + number(previous['historical_fit_seconds'])
            if abs(previous_time - history) > Decimal('0.000000001'):
                raise ValueError('Reused control does not reconcile with its parent fitting cost.')
        unique_history = Decimal(0) if alias else history
        out = {k: v for k, v in row.items() if k not in ['fit_seconds_this_stage', 'historical_fit_seconds']}
        out.update({
            'report_fit_id': fit_id(row),
            'canonical_fit_id': canonical,
            'fit_cost_alias_of': alias,
            'include_in_unique_fit_cost_total': not bool(alias),
            'reported_fit_seconds_this_stage': row['fit_seconds_this_stage'],
            'reported_historical_fit_seconds': row['historical_fit_seconds'],
            'incremental_fit_seconds': str(incremental),
            'historical_fit_seconds_before_extension': str(history),
            'historical_fit_seconds_unique_contribution': str(unique_history),
            'total_recorded_fit_seconds_unique_contribution': str(incremental + unique_history),
        })
        corrected.append(out)
        if not alias:
            unique.append({k: out[k] for k in ['canonical_fit_id','family','seed','policy','initialization','budget','origin','incremental_fit_seconds','historical_fit_seconds_unique_contribution','total_recorded_fit_seconds_unique_contribution']})
    if len({row['canonical_fit_id'] for row in unique}) != len(unique):
        raise ValueError('Duplicate fit in the unique ledger.')
    summary = []
    for origin, policy in sorted({(row['origin'],row['policy']) for row in corrected}):
        rows = [row for row in corrected if (row['origin'],row['policy']) == (origin,policy)]
        summary.append({
            'origin': origin, 'policy': policy, 'report_entries': len(rows),
            'new_committee_fits': len(rows) if origin == NEW else 0,
            'reused_alias_entries': sum(bool(row['fit_cost_alias_of']) for row in rows),
            'incremental_fit_seconds': str(sum((number(row['incremental_fit_seconds']) for row in rows), Decimal(0))),
            'historical_fit_seconds_unique_contribution': str(sum((number(row['historical_fit_seconds_unique_contribution']) for row in rows), Decimal(0))),
        })
    incremental = sum((number(row['incremental_fit_seconds']) for row in corrected), Decimal(0))
    historical = sum((number(row['historical_fit_seconds_unique_contribution']) for row in corrected), Decimal(0))
    removed = sum((number(row['reported_fit_seconds_this_stage']) for row in corrected if row['origin']==OLD), Decimal(0))
    totals = {
        'input_sha256': digest,
        'report_entries': len(corrected), 'unique_recorded_comparison_fits': len(unique),
        'new_committee_fits': sum(row['origin']==NEW for row in original),
        'reused_control_aliases': sum(row['origin']==REUSED for row in original),
        'historical_comparison_entries': len(parent),
        'incremental_fit_seconds': str(incremental),
        'incremental_fit_minutes': str(incremental / 60),
        'historical_fit_seconds_unique_contribution': str(historical),
        'total_recorded_comparison_fit_seconds': str(incremental + historical),
        'parent_fit_seconds_removed_from_incremental_total': str(removed),
        'new_reference_calculations': sum(int(row['new_reference_calculations']) for row in original),
        'duration_interpretation': 'Sum of recorded per-job elapsed fitting times; not campaign wall time or CPU time.',
        'cost_scope': 'Recorded comparison-model fits only. Not a complete end-to-end cost including source pretraining, Ridge training, acquisition, data generation, validation, feature generation or reporting.',
    }
    # This correction targets the audited completed experiment, not partial runs.
    assert (len(corrected),len(unique),totals['new_committee_fits'],totals['reused_control_aliases'],len(parent)) == (312,306,54,6,252)
    output = args.output.resolve()
    if output == source.parent or output.exists():
        raise ValueError('Choose a new derived-output directory; frozen files are never overwritten.')
    output.mkdir(parents=True)
    write_csv(output/'costs_corrected.csv', corrected)
    write_csv(output/'unique_fit_cost_ledger.csv', unique)
    write_csv(output/'cost_summary.csv', summary)
    (output/'cost_totals.json').write_text(json.dumps(totals, indent=2)+'\n')
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    print(json.dumps(totals,indent=2))

if __name__ == '__main__':
    main()
