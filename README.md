# Normalization as Implicit Reweighting — code and data

Companion repository for

> *Normalization as Implicit Reweighting: A Pre-Ranking Audit for
> Multi-Criteria Decision Support*, submitted to the International Journal of
> Information Technology & Decision Making.

It contains the audit (Section 4), the ranking methods and agreement measures
(Supplement S1), the reactor-component case study with its Bayesian network
and Monte Carlo engine (Section 5, Supplement S3–S4), the corpus tooling for
the 28 published decision matrices (Section 6, Supplement S5), and the
simulation (Section 7, Supplement S6), together with scripts that regenerate
every table and figure.

## Layout

```
normaudit/                    the package
  audit.py                    audit(): kappa, effective weights, nu, delta_w, flagged pairs
  methods.py                  min-max / vector normalisation; additive, WSM-V, TOPSIS,
                              TOPSIS-min-max, VIKOR; Kendall tau_b, inversion counting
  io.py                       CSV format for decision matrices
case_study/
  config.json                 all case-study inputs, each tagged with its source
  bn.py                       two-layer Bayesian network, exact inference
rao2007_decision_matrices/    corpus: metadata.json, expected_audit.json, CSV format (see its README)
scripts/
  worked_example.py           Table 1
  run_case_study.py           Tables 2-5, S3.4 posteriors, S4 results
  run_audit.py                Table 6, Table S2.6, the S5 tables, Section 6 pooled numbers
  validate_corpus.py          checks transcribed matrices against the published audit output
  simulation.py               Fig. 3 and the S6 tables
  calibrate_multipliers.py    see "Reconstructed inputs" below
  run_all.sh                  runs everything in order
tests/                        pytest suite (paper numbers, algebraic identities, pipeline)
outputs/                      generated tables and figures
```

## Requirements

Python 3.10+, NumPy, SciPy, Matplotlib (`pip install -r requirements.txt`).
Nothing else. Tested with NumPy 2.4, SciPy 1.17, Matplotlib 3.10.

## Reproducing the paper

```
bash scripts/run_all.sh
```

or step by step:

| command | produces | runtime |
|---|---|---|
| `python tests/test_audit.py && python tests/test_pipeline.py` | 12 checks incl. Table 1 numbers | seconds |
| `python scripts/worked_example.py` | `outputs/table1_worked_example.csv` (Table 1) | seconds |
| `python scripts/run_case_study.py` | `outputs/case_*.csv` (Tables 2–5, S3.4, S4) | ~3 min |
| `python scripts/run_audit.py` | `outputs/audit_*.csv`, `audit_summary.txt` (Table 6, S2.6, S5) | seconds |
| `python scripts/simulation.py` | `outputs/sim_*.csv`, `outputs/fig_sim.pdf` (Fig. 3, S6) | ~20 s |

All random draws use `numpy.random.default_rng` with fixed seeds
(simulation: 20260918, as stated in S6; case study: 20260918, set in
`case_study/config.json`).

### What reproduces exactly

* **Table 1** (worked example): κ = (0.921, 0.144), w̃ = (0.865, 0.135),
  ν = 6.41, δ_w = 0.365, σ₁ = −0.105, σ₂ = +0.292 — to every printed digit.
* **Supplement S6** (simulation): bin counts 5 / 50 / 306 / 806 / 1132 / 1701,
  every mean τ_b, every non-identical rate and every OLS coefficient in
  Tables S6.1–S6.2 — to the printed precision. Note that the main text of
  Section 7 still quotes older numbers (slope −0.177, t = −18, τ_b from 0.968
  to 0.656); the supplement's values are the ones this code produces.
* **Case study, deterministic quantities**: κ for severity (0.501) and
  detectability cost (0.626), the posterior-mean matrix audit
  κ = (0.655, 0.501, 0.626) with δ_w = 0.047, mean AHP weights
  (0.635, 0.260, 0.106) and Pr(CR < 0.1) = 0.988.

### What reproduces to Monte Carlo precision

The stochastic case-study results (Table 4 κ intervals, Table 5 agreement,
flag precision/recall, per-pair rates) match the paper to within sampling
error given the reconstructed network inputs described next.

## Reconstructed inputs — read before release

Two kinds of input were not present in the manuscript and are therefore
**not** the authors' original values. They are labelled as such inside the
files and must be replaced before the repository is made public.

1. **Bayesian-network parent sets and stress multipliers**
   (`case_study/config.json`, keys `parents` and `multipliers`, tagged
   `RECONSTRUCTED`). Supplement S3 marks these `[[fill from repository]]`.
   The values shipped here were obtained by `scripts/calibrate_multipliers.py`,
   which chooses per-parent factors so that the exact posteriors under the
   three scenarios match the Monte Carlo means reported in S3.4 (they do, to
   four decimals). Replace them with the elicited multiplier table and delete
   the calibration script.

2. **The 28 decision matrices** (`rao2007_decision_matrices/*.csv`). The
   supplement gives per-criterion κ and weights but not the matrix entries,
   which cannot be recovered from them. `metadata.json` (sources, orientations,
   stated weights, caveats) and `expected_audit.json` (every published audit
   number) are complete; transcribe each table from Rao (2007) in the format
   described in that folder's README and run `scripts/validate_corpus.py`,
   which verifies each file against the published κ_j, w̃, ν, δ_w, flagged
   pairs and TOPSIS inversions. `run_audit.py` works on whatever subset is
   present and reports which tables are still missing.

Everything else — expert scores, base rates, stressor priors, Beta
concentration, AHP comparison distributions, scenarios, iteration count,
simulation generator and seed — is taken verbatim from the manuscript and is
tagged `source: manuscript` in `config.json`.

## Using the audit on your own matrix

```python
import numpy as np
from normaudit import audit

X = np.array([[0.020, 0.70], [0.005, 0.90], [0.001, 0.80]])   # n x m, non-negative
o = np.array([1, 1])          # +1 benefit, -1 cost
w = np.array([0.5, 0.5])      # stated weights
res = audit(X, o, w)          # cost_convention="raw" (default) or "max_minus_x"
print("\n".join(res.summary_lines(["C1", "C2"], ["A", "B", "C"])))
```

or from a CSV in the corpus format:

```python
from normaudit import load_matrix_csv, audit
p = load_matrix_csv("rao2007_decision_matrices/rao_table_29_1.csv")
res = audit(p.X, p.o, p.w)
```

`res.kappa`, `res.w_eff`, `res.nu`, `res.delta_w`, `res.flagged_pairs`
(0-based index pairs), `res.dropped` (zero-range criteria) and `res.sigma`
(σ₁, σ₂ per pair) are the outputs of steps 1–4 in Section 4.

## Conventions

* κ_j is computed on the column **as published**; cost criteria are not
  transformed before vector normalisation (the original TOPSIS convention).
  `cost_convention="max_minus_x"` reproduces Table S2.6's second pair of
  columns.
* Tolerances: ε_R = ε_m = 10⁻¹².
* VIKOR uses v = 0.5 and compares the complete ordering by Q only.
* Kendall τ_b is computed on average ranks; two rankings are "identical" when
  their rank vectors are equal; "same top" when the arg-max coincides.

## Citation and licence

Code: MIT licence (see `LICENSE`). The decision matrices are reprinted data
from the sources listed in `rao2007_decision_matrices/metadata.json`; cite
Rao (2007) and the original papers when using them. See `CITATION.cff`.
