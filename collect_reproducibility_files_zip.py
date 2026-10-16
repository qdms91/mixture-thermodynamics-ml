#!/usr/bin/env python3
"""Collect frozen Methods evidence without changing any source file.

Run from cosmo_rs_1 (or supply --root):
    python collect_reproducibility_files_zip.py --dry-run
    python collect_reproducibility_files_zip.py

Optional: --include-chemistry adds five molecular surfaces and a small, explicit
ORCA provenance/citation set. It does not collect entire calculation folders.

This is a Methods evidence package, not a complete reproducibility release.
Environment/machine records, the full source-code release, raw reference-search
files, the reference QM collection and intermediate fit trees are excluded.
Saved code snapshots for the retention campaigns ARE included.

All required paths were checked against directory_structure_1210261052.txt.
Preflight checks the current files again. A directory listing cannot establish
that file contents or scientific results are correct.

Nothing is deleted, moved, or overwritten. No model is loaded or executed.
On interruption, an incomplete ZIP may remain: it is never reported as success
and must be renamed manually before rerunning. Run after campaigns have stopped
writing; the script checks for changes to each file while it is copied, but
cannot obtain an atomic snapshot of several independently changing files.
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

SEEDS = (11, 22, 33, 44, 55)
ARCHIVE_ROOT = 'files_reproducibility_methods_repository'
ZIP_NAME = ARCHIVE_ROOT + '.zip'


class Collection:
    def __init__(self, root):
        self.root = root
        self.files = set()
        self.errors = []

    def safe(self, path):
        # Refuse symlinks, including directory symlinks, instead of accidentally
        # collecting data outside the selected project.
        relative = path.relative_to(self.root)
        current = self.root
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                self.errors.append('Symlink not collected: ' + str(relative))
                return False
        return True

    def file(self, relative):
        path = self.root / relative
        if not self.safe(path):
            return
        if not path.is_file():
            self.errors.append('Missing file: ' + str(relative))
        else:
            self.files.add(path)

    def directory(self, relative):
        self.pattern(relative, '*', recursive=True)

    def pattern(self, relative, pattern, recursive=False):
        directory = self.root / relative
        if not self.safe(directory):
            return
        if not directory.is_dir():
            self.errors.append('Missing directory: ' + str(relative))
            return
        matches = directory.rglob(pattern) if recursive else directory.glob(pattern)
        count = 0
        for path in matches:
            if not self.safe(path):
                continue
            if path.is_file():
                self.files.add(path)
                count += 1
        if count == 0:
            self.errors.append('No files matched: ' + str(relative / pattern))


def collect(root, include_chemistry=False):
    c = Collection(root)
    p1 = Path('results/ml/project1_regression/run_20261004_134348_552708')
    for name in ('ridge_model.joblib', 'run_metadata.json', 'selected_settings.json',
                 'validation_search.csv', 'test_metrics.csv'):
        c.file(p1 / name)

    p2 = Path('results/ml/project2_classification')
    c.directory(p2 / 'prepared_seed2026')
    # Enforce key members, not just the existence of a nonempty directory.
    for name in ('classification_data.csv', 'descriptor_constants.json',
                 'feature_groups.json', 'frozen_split.csv', 'preparation.json',
                 'split_counts.csv', 'excluded_needs_review.csv'):
        c.file(p2 / 'prepared_seed2026' / name)
    # These compact records document all learners, not only retained MLPs.
    for name in ('run_metadata.json', 'selections.json', 'cv_results.csv',
                 'fit_quality.json', 'invocations.jsonl', 'test_metrics.csv'):
        c.file(p2 / 'benchmark_v1' / name)
    p2_labels = Path('results/ml/project2_stability/total2000_sample2026_search88_starts40')
    c.file(p2_labels / 'metadata.json')
    c.file(p2_labels / 'dataset.csv')

    p3 = Path('results/ml/project3_active_learning/benchmark_v1')
    for name in ('protocol.json', 'invocations.jsonl', 'frozen_roles.csv', 'learning_complete.json'):
        c.file(p3 / name)
    c.directory(p3 / 'report')
    for seed in SEEDS:
        active = p3 / f'seed_{seed}' / 'uncertainty_diversity'
        for name in ('state.json', 'queries.csv', 'model_0200.joblib', 'model_0800.joblib'):
            c.file(active / name)

    ood = Path('results/ml/project4_frozen_ood/pilot_v1')
    for name in ('models.json', 'protocol.json', 'recipes.csv', 'reference_labels.csv',
                 'reference_protocol.json', 'frozen_predictions.csv'):
        c.file(ood / name)

    p4 = Path('results/ml/project4_adaptation/target_v1')
    c.pattern(p4, '*.json')
    c.pattern(p4, '*.csv')
    for name in ('dataset.csv', 'recipes.csv', 'protocol.json', 'reference_protocol.json',
                 'labels_complete.json', 'composition_distances.csv', 'sources.json'):
        c.file(p4 / name)
    bench = p4 / 'benchmark_v1'
    for name in ('training_protocol.json', 'training_complete.json'):
        c.file(bench / name)
    c.directory(bench / 'report')
    for seed in SEEDS:
        seed_dir = bench / f'seed_{seed}'
        c.pattern(seed_dir, '*.json', recursive=True)
        for budget in (60, 150, 300):
            c.file(seed_dir / 'transfer_active' / f'model_{budget:04d}.joblib')
            c.file(seed_dir / 'transfer_active' / f'round_{budget:04d}.json')

    ret = p4 / 'benchmark_v2_retention'
    for name in ('protocol.json', 'input_audit.json'):
        c.file(ret / name)
    for name in ('inputs', 'replay_manifests', 'report_stage12', 'source_models', 'code_snapshot'):
        c.directory(ret / name)
    energy = ret / 'stage3_energy'
    c.file(energy / 'protocol.json')
    for name in ('inputs', 'replay_manifests', 'report', 'ridge', 'code_snapshot'):
        c.directory(energy / name)
    bridge = ret / 'stage3_bridge_extension'
    c.file(bridge / 'protocol.json')
    for name in ('inputs', 'replay_manifests', 'report', 'baseline', 'code_snapshot'):
        c.directory(bridge / name)

    if include_chemistry:
        for molecule in ('water', 'methanol', 'acetone', 'acetonitrile', 'benzene'):
            folder = Path('calculations') / molecule
            for name in (f'{molecule}.solute.orcacosmo', 'optimize.xyz', 'optimize.inp',
                         'optimize.out', 'optimize.bibtex', f'{molecule}.inp',
                         f'{molecule}.out', f'{molecule}.bibtex',
                         f'{molecule}.solute_cpcm.inp', f'{molecule}.solute_cpcm.bibtex'):
                c.file(folder / name)
    return c


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, default=Path.cwd(), help='Project root (default: current directory)')
    parser.add_argument('--dry-run', action='store_true', help='Validate and list files; write nothing')
    parser.add_argument('--include-chemistry', action='store_true', help='Add selected ORCA provenance and five surfaces')
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()
    c = collect(root, args.include_chemistry)
    if c.errors:
        print('Preflight failed; no ZIP was created.', file=sys.stderr)
        for error in sorted(set(c.errors)):
            print('  - ' + error, file=sys.stderr)
        return 1
    ordered = sorted(c.files, key=lambda p: p.relative_to(root).as_posix())
    output = root / ZIP_NAME
    total = sum(p.stat().st_size for p in ordered)
    print(f'Project root: {root}\nOutput: {output}\nSource files: {len(ordered)}\nSource size: {total / 1024**2:.1f} MiB')
    if args.dry_run:
        for path in ordered:
            print(path.relative_to(root).as_posix())
        print('Dry run complete. Nothing was written.')
        return 0
    if output.exists() or output.is_symlink():
        print('Output already exists; it will not be overwritten.', file=sys.stderr)
        return 1
    records = []
    try:
        with ZipFile(output, mode='x', compression=ZIP_DEFLATED, compresslevel=6) as archive:
            for number, path in enumerate(ordered, 1):
                relative = path.relative_to(root).as_posix()
                if not c.safe(path):
                    raise RuntimeError('Source changed to a symlink: ' + relative)
                before = path.stat()
                digest = hashlib.sha256()
                copied = 0
                with path.open('rb') as source, archive.open(ARCHIVE_ROOT + '/' + relative, 'w', force_zip64=True) as dest:
                    while True:
                        block = source.read(1024 * 1024)
                        if not block:
                            break
                        dest.write(block)
                        digest.update(block)
                        copied += len(block)
                after = path.stat()
                if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino) or copied != before.st_size:
                    raise RuntimeError('Source changed while copying: ' + relative)
                records.append({'path': relative, 'bytes': copied, 'sha256': digest.hexdigest()})
                if number % 50 == 0 or number == len(ordered):
                    print(f'  Added {number}/{len(ordered)} files')
            manifest = {'created_utc': datetime.now(timezone.utc).isoformat(),
                        'purpose': 'Compact Methods evidence; not a complete reproducibility release',
                        'include_chemistry': args.include_chemistry, 'source_files': records}
            archive.writestr(ARCHIVE_ROOT + '/collection_manifest.json', json.dumps(manifest, indent=2) + '\n')
        with ZipFile(output) as archive:
            bad = archive.testzip()
            if bad is not None:
                raise RuntimeError('ZIP integrity check failed: ' + bad)
    except (OSError, RuntimeError, KeyboardInterrupt) as exc:
        print(f'Collection did not finish successfully: {exc}', file=sys.stderr)
        print(f'If an incomplete ZIP exists at {output}, rename it manually before retrying.', file=sys.stderr)
        return 1
    print(f'Success: {output}\nZIP size: {output.stat().st_size / 1024**2:.1f} MiB')
    print(f'Included {len(ordered)} source files plus a SHA-256 manifest. No source files were changed, moved, or deleted.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
