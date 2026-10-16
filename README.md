# Learning mixture thermodynamics with efficient screening, adaptation, and thermodynamically guided replay

How much of a multicomponent mixture's behavior can we learn from its composition, and how can we make that learning less expensive?

We investigate water, methanol, acetone, acetonitrile, and benzene at 298.15 K using ORCA and openCOSMO-RS. Starting with equimolar mixtures, we move from predicting mixing free energies to identifying instabilities, selecting informative calculations, adapting models to new composition regimes, and retaining what those models learned in their original domain.

The project connects four experiments:

| Project | Question and principal result |
| --- | --- |
| **P1 — Thermodynamic regression** | Can composition alone predict excess Gibbs energy? A fourth-degree polynomial Ridge model reached a held-out MAE of **0.00990 J/mol** against the openCOSMO-RS reference. Adding the analytical ideal contribution gives total mixing Gibbs energy. Modest extrapolation remained useful; binary-limit extrapolation exposed clear failures. |
| **P2 — Instability classification** | Can a classifier predict the outcome of an expensive stability search? Composition and simple molecular features performed well; activity-based MLP committees reached **AP = 1.0000** on the held-out source test. We compared the additional feature cost with the cost of the full search. |
| **P3 — Active learning** | Can we choose which compositions deserve reference labels? Across five seeds, active learning reached mean **AP = 0.9930 with 200 training labels**, compared with 0.9250 for random acquisition and 0.9928 for a matched 1,191-label reference. |
| **P4 — Adaptation and replay** | Can a source model adapt to broader compositions while retaining source performance? Adaptation improved target predictions but caused source forgetting. Small memories selected using historical search duration or predicted mixing energy supported retention. The strict energy bridge was particularly effective with the larger P2 molecular source pool. |

The **energy bridge** is a central idea in the replay experiments. We order correctly classified source-training examples by predicted mixing free energy and retain examples with opposing reference labels around the first transition from the high-energy end. With a 20-example memory, the P2 strict bridge reached source AP 0.9518 at 150 target labels, compared with 0.7906 for its class-matched random control. Target performance and source retention trade off, and the smaller active-200 source pools gave a more mixed picture. A strict P3 bridge and its later downstream extension are reported separately.

The accompanying **manuscript, Version 10**, develops the scientific questions, methods, results, and limitations.

## Reproduce or explore

See **[REPRODUCE.md](REPRODUCE.md)** for the calculation sequence, verified command-line arguments, and input/output locations. You can start from molecular calculations, reuse cached thermodynamic datasets and instability labels, or explore the saved models and numerical results.

- The Python scripts remain in the repository root.
- `data/structures/` contains starting molecular geometries; `calculations/` contains the generated surfaces and selected calculation records.
- `results/ml/` retains the original P1–P4 organization, including datasets, splits, acquisition records, replay memories, selected models, and result tables.
- [RUN_SETTINGS.json](RUN_SETTINGS.json) summarizes settings extracted from the original run records.
- Larger checkpoint collections and detailed searches are assigned to GitHub Release archives; their download links will be added with the first release.

The calculations use [openCOSMO-RS_py](https://github.com/TUHH-TVT/openCOSMO-RS_py), its [24a parameterization](https://www.sciencedirect.com/science/article/pii/S0378381224002255), and [ORCA](https://www.faccts.de/orca/). ORCA is needed when regenerating molecular surfaces. The Python calculations can start directly from the supplied surfaces.

**Author:** Daniel Morales Salazar.
