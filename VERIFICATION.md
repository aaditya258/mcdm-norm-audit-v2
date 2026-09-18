# Verification against the manuscript (run of 2026-09-18)

Produced by `bash scripts/run_all.sh` on a clean checkout. "Paper" values are
from the main text or supplement as submitted.

## Exact reproductions

| quantity | paper | this code |
|---|---|---|
| Table 1: κ₁, κ₂ | 0.921, 0.144 | 0.921, 0.144 |
| Table 1: w̃ | (0.865, 0.135) | (0.865, 0.135) |
| Table 1: ν, δ_w | 6.41, 0.365 | 6.411, 0.365 |
| Table 1: σ₁, σ₂ for (A,B) | −0.105, +0.292 | −0.105, +0.292 |
| S6.1 bin counts | 5 / 50 / 306 / 806 / 1132 / 1701 | identical |
| S6.1 mean τ_b, all 18 cells | e.g. 0.654 / 0.899 / 0.615 (ν > 6) | identical to 3 d.p. |
| S6.2 OLS slopes, SEs, t, R², ρ_S | e.g. −0.197, 0.0092, −21.3, 0.151, −0.402 | identical |
| S4.2 posterior-mean matrix κ, δ_w | (0.655, 0.501, 0.626), 0.047 | (0.654, 0.501, 0.626), 0.047 |
| AHP mean weights | (0.635, 0.260, 0.106) | (0.635, 0.259, 0.105) |
| Pr(CR < 0.1) | 0.988 | 0.988 |
| S3.4 exact posteriors, 3 scenarios × 4 modes | as printed | matched to 4 d.p. by calibration |

## Monte Carlo reproductions (N = 5000 per scenario, reconstructed BN inputs)

| quantity | paper | this code |
|---|---|---|
| Table 4 nominal κ₁ (95 %) | 0.893 (0.708–0.999) | 0.899 (0.711–0.999) |
| Table 4 nominal w̃ | (0.741, 0.172, 0.087) | (0.743, 0.171, 0.087) |
| Table 4 nominal shift | +10.6 / −8.8 / −1.8 pp | +10.7 / −8.9 / −1.9 pp |
| Table 4 nominal ν, δ_w | 1.78, 0.106 | 1.80, 0.108 |
| flag vs TOPSIS precision / recall | 0.995 / 0.465 | 0.994 / 0.468 |
| sign-test accuracy vs WSM-V | 1.000 | 1.000 |
| τ_b TOPSIS vs additive (nominal) | 0.976 | 0.981 |
| τ_b TOPSIS-min-max vs additive | 0.987 | 0.989 |
| different complete ranking: norm-only / aggr-only / both | 3.5 / 3.8 / 7.0 % | 2.7 / 3.2 / 5.3 % |
| same top-ranked, every comparison | ≥ 99 % | ≥ 98.8 % |
| Turbine rank-1, AHP / entropy / uniform | 0.491 / 0.510 / 0.636 | 0.503 / 0.516 / 0.643 |
| uniform regime reverses leader in High T+M | yes | yes |
| δ_w over Beta s = 5 … 50 | 0.113 → 0.090 | 0.113 → 0.092 |
| Pump posterior 95 % interval, nominal | 0.0005–0.0281 | 0.0004–0.0273 |
| Pump posterior 95 % interval, High T+M | 0.0017–0.1992 | 0.0016–0.2042 |

## Known differences attributable to reconstructed inputs

* Sensor–Valve is non-dominated in 43 % of nominal draws here against 79 % in
  the paper, and is flagged in 1.3 % (paper 2.3 %) and inverted by TOPSIS in
  2.0 % (paper 4.0 %). The paper's figures require the authors' parent sets
  and multiplier table, which govern how much the Sensor and Valve likelihood
  posteriors overlap. Insert them in `case_study/config.json` and re-run.
* Corpus results (Table 6, S2.6, S5) cannot be produced until the 28 CSVs are
  transcribed; `validate_corpus.py` will confirm each one against S5.

## Manuscript items noticed while building

* Section 7 main text quotes the pre-update simulation numbers (slope −0.177,
  t = −18, τ_b 0.968 → 0.656, R² 0.36 vs 0.12); Supplement S6 and this code
  give −0.197, t = −21.3, τ_b 0.982 → 0.654 (1.000 in the 5-problem bin),
  R² 0.362 vs 0.151. The `[[re-run ... and update]]` note in Section 7 refers
  to this.
* S3.4 lists the Valve nominal posterior as 0.0009 and Vibration as 0.0008;
  the exact value is 0.000845 in both (Valve is not a child of V), so the
  difference is Monte Carlo rounding.
