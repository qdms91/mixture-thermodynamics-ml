# Project 4 bridge extension: corrected incremental-cost presentation

This is a derived publication companion to the frozen extension report. The original report, code, checkpoints, cutoffs, predictions, and performance results are unchanged. Do not replace a file inside the frozen report with these files: its recorded checksum would no longer match.

## Publication-ready cost statement

The exploratory P3 bridge extension added 54 committee fits and reused six existing matched-control fits. The 54 new fits comprised 30 downstream-energy-bridge fits and 24 random class-matched-control fits, requiring a summed recorded fitting duration of **775.15 seconds (12.92 minutes)**. These are summed per-job elapsed times, not campaign wall time or measured CPU time. The extension performed **zero new reference calculations**, using previously generated source and target labels and frozen energy predictions. Historical data-generation, source-model and energy-surrogate costs remain inherited costs; the incremental fitting time is not an end-to-end cost estimate.

| Extension policy | New committee fits | Reused committee fits | Incremental recorded fit duration (s) |
|---|---:|---:|---:|
| Downstream energy bridge | 30 | 0 | 416.32 |
| Random class-matched 10/10 control | 24 | 6 | 358.83 |
| Total | 54 | 6 | 775.15 |

Each new committee contains three networks, so these 54 new committee fits contain 162 newly fitted networks.

## What was corrected

The combined source table has 312 report entries: 252 historical comparison entries and 60 extension entries. Historical entries retain 1,975.79 seconds of parent-stage fitting in the original `fit_seconds_this_stage` column. Summing that mixed-origin column would incorrectly attribute 2,750.94 seconds to the extension.

The corrected table provides an unambiguous `incremental_fit_seconds` column: it is zero for all reused or historical entries. The two original duration fields are retained under `reported_...` names for traceability.

The six reused controls are aliases of historical `bridge_class_matched20` fits at seed 55. Their 103.86 seconds of recorded historical fitting are already represented among the 252 historical entries. They must not be added again. `canonical_fit_id` links each alias to the historical fit, and `include_in_unique_fit_cost_total` is false for these six aliases.

## Files and columns

- `costs_corrected.csv`: all 312 report entries, with all non-duration source columns retained and explicit incremental/historical accounting.
- `unique_fit_cost_ledger.csv`: 306 unique recorded comparison fits (252 historical plus 54 new), with aliases removed.
- `cost_summary.csv`: policy/origin summaries.
- `cost_totals.json`: exact reconciliation and input hash.
- `correct_incremental_costs.py`: standalone Python script to reproduce the derived tables; uses only the standard library and refuses to overwrite an existing output directory.

Safe duration totals:

| Column | Meaning | Safe to sum across the corrected table? |
|---|---|---|
| `incremental_fit_seconds` | New fitting performed for this extension | Yes |
| `historical_fit_seconds_before_extension` | Historical duration attached to each report entry | No: reused aliases repeat historical fits |
| `historical_fit_seconds_unique_contribution` | Historical duration counted once per canonical comparison fit | Yes |
| `total_recorded_fit_seconds_unique_contribution` | Incremental plus deduplicated historical comparison-fit duration | Yes |
| `reported_fit_seconds_this_stage` | Original mixed-origin duration column, preserved verbatim | No, not as extension cost |
| `reported_historical_fit_seconds` | Original historical duration column, preserved verbatim | No, not as a deduplicated historical total |

The unique historical comparison fits account for 3,716.61 recorded seconds; including the new fits gives 4,491.76 recorded seconds. These totals describe only the comparison fits present in this report. They are not a complete project cost: source pretraining, Ridge training, acquisition overhead, reference generation, feature generation, validation/evaluation and reporting are not fully accounted for here.

Repeated recipe-count and reference-duration columns retain their original per-entry meanings. They must not be summed as counts of unique labeled recipes or unique reference jobs; the same recipes and checkpoints are reused across policies and budgets.

## Reproduction

From the project root, using the script in this bundle:

```bash
python correct_incremental_costs.py \
  --costs results/ml/project4_adaptation/target_v1/benchmark_v2_retention/stage3_bridge_extension/report/costs.csv \
  --output results/ml/project4_adaptation/target_v1/benchmark_v2_retention/stage3_bridge_extension_costs_corrected
```

The script checks the input hash against the source report manifest when available, validates the reused-control cost aliases, and verifies that the source CSV is unchanged. It does not read labels, fit models, load checkpoints, or run COSMO-RS.

Audited input SHA-256: `1ad12e12f3f9e9871b72390a8a08cc5511dcd5d71c1613d62b57f0d423c7c536`.
