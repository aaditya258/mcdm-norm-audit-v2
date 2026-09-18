"""Reactor-component prioritisation case (Section 5, Supplement S3-S4).

At each Monte Carlo iteration the engine
  1. perturbs every calibrated CPT entry mu by Beta(mu*s, (1-mu)*s),
  2. computes exact BN posteriors of the four failure modes under the scenario,
  3. draws AHP weights from triangular pairwise comparisons,
  4. runs the audit and the five ranking methods on the resulting matrix,
and both the matrix and the weight vector go unchanged to every method.

Outputs (outputs/case_*.csv, see README):
  case_posteriors.csv        MC mean and 95% interval of failure posteriors (S3.4)
  case_audit.csv             Table 4: kappa, stated / effective weights, shift, nu, delta_w
  case_flag_vs_topsis.csv    precision / recall of the pairwise flag against TOPSIS, pooled
  case_pairs.csv             per-pair rates: non-dominated, flagged, inverted by TOPSIS
  case_agreement.csv         Table 5: tau_b, identical-ranking and same-top rates vs additive
  case_rank1.csv             rank-1 acceptability by method and scenario
  case_weight_regimes.csv    S4.2: additive rank-1 acceptability under AHP / entropy / uniform weights
  case_beta_sweep.csv        S4.2: delta_w and agreement as the Beta concentration varies
  case_deterministic.csv     audit on the posterior-mean matrix
  case_cpt.txt               the calibrated conditional probability tables
"""
from __future__ import annotations

import argparse
import csv
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from case_study.bn import FailureNetwork, load_config, scenarios          # noqa: E402
from normaudit import (audit, inverted_pairs, kendall_tau_b, same_ranking,  # noqa: E402
                       same_top, saw_minmax, topsis, topsis_minmax, vikor, wsm_vector)

METHODS = {"additive": saw_minmax, "wsm_v": wsm_vector, "topsis_mm": topsis_minmax,
           "topsis": topsis, "vikor": vikor}
CELL_LABEL = {"wsm_v": "normalisation only", "topsis_mm": "aggregation only",
              "topsis": "both (TOPSIS)", "vikor": "VIKOR"}


# ------------------------------------------------------------------ weights
def ahp_weights(rng, cfg):
    a = cfg["weights_ahp"]
    a12 = rng.triangular(*a["a12"])
    a13 = rng.triangular(*a["a13"])
    a23 = rng.triangular(*a["a23"])
    A = np.array([[1, a12, a13], [1 / a12, 1, a23], [1 / a13, 1 / a23, 1]])
    vals, vecs = np.linalg.eig(A)
    k = np.argmax(vals.real)
    w = np.abs(vecs[:, k].real)
    w /= w.sum()
    lam = vals[k].real
    cr = (lam - 3) / 2 / a["random_index_n3"]
    return w, cr


def entropy_weights(X):
    P = X / X.sum(axis=0)
    n = X.shape[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        e = -(np.where(P > 0, P * np.log(P), 0)).sum(axis=0) / np.log(n)
    d = 1 - e
    return d / d.sum() if d.sum() > 0 else np.full(X.shape[1], 1 / X.shape[1])


# ------------------------------------------------------------------ one iteration
def sample_tables(rng, net, s):
    out = {}
    for a in net.alternatives:
        mu = np.array([r["p_fail"] for r in net.tables[a]["rows"]])
        out[a] = rng.beta(mu * s, (1 - mu) * s)
    return out


def iterate(rng, net, cfg, evidence, s, fixed_cols):
    tables = sample_tables(rng, net, s)
    like = net.posteriors(evidence, tables)
    X = np.column_stack([like, fixed_cols])
    w, cr = ahp_weights(rng, cfg)
    return X, w, cr


# ------------------------------------------------------------------ main loop
def run_scenario(rng, net, cfg, evidence, N, s, fixed_cols, o):
    n = len(net.alternatives)
    pairs = list(combinations(range(n), 2))
    rec = {"post": [], "kappa": [], "w": [], "w_eff": [], "nu": [], "delta_w": [], "cr": [],
           "flagged": np.zeros(len(pairs)), "nondom": np.zeros(len(pairs)),
           "topsis_inv": np.zeros(len(pairs)), "flag_and_inv": np.zeros(len(pairs)),
           "wsmv_inv": np.zeros(len(pairs)), "flag_and_wsmv": np.zeros(len(pairs)),
           "tau": {m: [] for m in METHODS}, "ident": {m: [] for m in METHODS},
           "top": {m: [] for m in METHODS}, "rank1": {m: np.zeros(n) for m in METHODS},
           "rank1_regime": {"ahp": np.zeros(n), "entropy": np.zeros(n), "uniform": np.zeros(n)},
           "sv_share_topsis": [0, 0], "sv_share_vikor": [0, 0]}
    sv = pairs.index((2, 3))  # Sensor--Valve
    for _ in range(N):
        X, w, cr = iterate(rng, net, cfg, evidence, s, fixed_cols)
        res = audit(X, o, w)
        rec["post"].append(X[:, 0]); rec["kappa"].append(res.kappa); rec["w"].append(w)
        rec["w_eff"].append(res.w_eff); rec["nu"].append(res.nu); rec["delta_w"].append(res.delta_w)
        rec["cr"].append(cr)
        scores = {m: f(X, o, w) for m, f in METHODS.items()}
        ref = scores["additive"]
        inv_t = set(inverted_pairs(ref, scores["topsis"]))
        inv_v = set(inverted_pairs(ref, scores["wsm_v"]))
        fl = set(res.flagged_pairs)
        for k, (i, j) in enumerate(pairs):
            diff = o * (X[i] - X[j])
            rec["nondom"][k] += not (np.all(diff >= 0) or np.all(diff <= 0))
            rec["flagged"][k] += (i, j) in fl
            rec["topsis_inv"][k] += (i, j) in inv_t
            rec["flag_and_inv"][k] += (i, j) in fl and (i, j) in inv_t
            rec["wsmv_inv"][k] += (i, j) in inv_v
            rec["flag_and_wsmv"][k] += ((i, j) in fl) == ((i, j) in inv_v)  # accuracy
        for m, sc in scores.items():
            rec["tau"][m].append(kendall_tau_b(ref, sc))
            rec["ident"][m].append(same_ranking(ref, sc))
            rec["top"][m].append(same_top(ref, sc))
            rec["rank1"][m][int(np.argmax(sc))] += 1
        # share of divergent draws explained by the single Sensor--Valve inversion
        for m, key in (("topsis", "sv_share_topsis"), ("vikor", "sv_share_vikor")):
            inv = inverted_pairs(ref, scores[m])
            if inv:
                rec[key][1] += 1
                rec[key][0] += inv == [(2, 3)]
        # weight regimes for the additive kernel (SMAA-2 style)
        rec["rank1_regime"]["ahp"][int(np.argmax(ref))] += 1
        rec["rank1_regime"]["entropy"][int(np.argmax(saw_minmax(X, o, entropy_weights(X))))] += 1
        rec["rank1_regime"]["uniform"][int(np.argmax(saw_minmax(X, o, rng.dirichlet(np.ones(3)))))] += 1
    for k in ("post", "kappa", "w", "w_eff"):
        rec[k] = np.array(rec[k])
    rec["pairs"] = pairs
    return rec


def q(x, lo=2.5, hi=97.5):
    return np.percentile(x, [lo, hi])


def write(path, rows):
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=str(ROOT / "outputs"))
    ap.add_argument("--no-sweep", action="store_true", help="skip the Beta concentration sweep")
    args = ap.parse_args()
    cfg = load_config()
    N = args.iterations or cfg["monte_carlo"]["iterations"]
    seed = args.seed if args.seed is not None else cfg["monte_carlo"]["seed"]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    net = FailureNetwork(cfg)
    alts, crit = cfg["alternatives"], cfg["criteria"]
    o = np.array(cfg["orientation"])
    sev = np.array(cfg["expert_scores"]["severity"])
    det_cost = 1 - np.array(cfg["expert_scores"]["detectability_D"])
    fixed = np.column_stack([sev, det_cost])
    s0 = cfg["epistemic"]["beta_concentration"]
    (out / "case_cpt.txt").write_text(net.describe() + "\n")

    rng = np.random.default_rng(seed)
    results = {name: run_scenario(rng, net, cfg, ev, N, s0, fixed, o)
               for name, ev in scenarios(cfg).items()}

    # ---- posteriors (S3.4)
    rows = []
    for name, r in results.items():
        exact = net.posteriors(scenarios(cfg)[name])
        for i, a in enumerate(alts):
            lo, hi = q(r["post"][:, i])
            rows.append({"scenario": name, "alternative": a, "mc_mean": r["post"][:, i].mean(),
                         "q2.5": lo, "q97.5": hi, "exact_posterior": exact[i]})
    write(out / "case_posteriors.csv", rows)

    # ---- Table 4 audit
    rows = []
    for name, r in results.items():
        for j, c in enumerate(crit):
            lo, hi = q(r["kappa"][:, j])
            rows.append({"scenario": name, "criterion": c, "kappa_mean": r["kappa"][:, j].mean(),
                         "kappa_q2.5": lo, "kappa_q97.5": hi, "w_stated_mean": r["w"][:, j].mean(),
                         "w_eff_mean": r["w_eff"][:, j].mean(),
                         "shift_pp": 100 * (r["w_eff"][:, j].mean() - r["w"][:, j].mean()),
                         "nu_mean": np.mean(r["nu"]), "delta_w_mean": np.mean(r["delta_w"]),
                         "pr_cr_lt_0.1": np.mean(np.array(r["cr"]) < 0.1)})
    write(out / "case_audit.csv", rows)

    # ---- flag vs TOPSIS, pooled
    F = sum(r["flagged"].sum() for r in results.values())
    T = sum(r["topsis_inv"].sum() for r in results.values())
    B = sum(r["flag_and_inv"].sum() for r in results.values())
    n_pairs = len(next(iter(results.values()))["pairs"])
    acc = sum(r["flag_and_wsmv"].sum() for r in results.values()) / (len(results) * N * n_pairs)
    write(out / "case_flag_vs_topsis.csv", [{
        "pair_evaluations": len(results) * N * n_pairs, "flagged": int(F), "topsis_inversions": int(T),
        "flagged_and_inverted": int(B), "precision": B / F if F else np.nan, "recall": B / T if T else np.nan,
        "sign_test_accuracy_vs_wsm_v": acc}])

    # ---- per-pair rates
    rows = []
    for name, r in results.items():
        for k, (i, j) in enumerate(r["pairs"]):
            rows.append({"scenario": name, "pair": f"{alts[i]}--{alts[j]}",
                         "non_dominated_pct": 100 * r["nondom"][k] / N,
                         "flagged_pct": 100 * r["flagged"][k] / N,
                         "topsis_inverted_pct": 100 * r["topsis_inv"][k] / N})
    write(out / "case_pairs.csv", rows)

    # ---- Table 5 agreement vs additive
    rows = []
    for name, r in results.items():
        for m in ("wsm_v", "topsis_mm", "topsis", "vikor"):
            rows.append({"scenario": name, "method": m, "cell": CELL_LABEL[m],
                         "mean_tau_b": np.mean(r["tau"][m]),
                         "different_complete_ranking_pct": 100 * (1 - np.mean(r["ident"][m])),
                         "same_top_pct": 100 * np.mean(r["top"][m])})
        rows.append({"scenario": name, "method": "sensor_valve_share", "cell": "share of divergent draws that are exactly the Sensor--Valve inversion",
                     "mean_tau_b": np.nan,
                     "different_complete_ranking_pct": 100 * r["sv_share_topsis"][0] / max(r["sv_share_topsis"][1], 1),
                     "same_top_pct": 100 * r["sv_share_vikor"][0] / max(r["sv_share_vikor"][1], 1)})
    write(out / "case_agreement.csv", rows)

    # ---- rank-1 acceptability
    rows = []
    for name, r in results.items():
        for m in METHODS:
            row = {"scenario": name, "method": m}
            row.update({a: r["rank1"][m][i] / N for i, a in enumerate(alts)})
            rows.append(row)
    write(out / "case_rank1.csv", rows)
    rows = []
    for name, r in results.items():
        for reg in ("ahp", "entropy", "uniform"):
            row = {"scenario": name, "weight_regime": reg}
            row.update({a: r["rank1_regime"][reg][i] / N for i, a in enumerate(alts)})
            rows.append(row)
    write(out / "case_weight_regimes.csv", rows)

    # ---- deterministic posterior-mean matrix
    Xd = np.column_stack([net.posteriors({}), fixed])
    wbar = results["Nominal"]["w"].mean(axis=0)
    rd = audit(Xd, o, wbar)
    write(out / "case_deterministic.csv", [{"criterion": c, "kappa": rd.kappa[j], "w_stated": wbar[j],
                                            "w_eff": rd.w_eff[j], "nu": rd.nu, "delta_w": rd.delta_w}
                                           for j, c in enumerate(crit)])

    # ---- Beta concentration sweep (nominal)
    if not args.no_sweep:
        rows = []
        for s in cfg["epistemic"]["beta_concentration_sweep"]:
            rs = results["Nominal"] if s == s0 else run_scenario(np.random.default_rng(seed + s), net, cfg, {}, N, s, fixed, o)
            rows.append({"beta_s": s, "delta_w_mean": np.mean(rs["delta_w"]),
                         "kappa_likelihood_mean": rs["kappa"][:, 0].mean(),
                         "mean_tau_topsis_vs_additive": np.mean(rs["tau"]["topsis"]),
                         "mean_tau_vikor_vs_additive": np.mean(rs["tau"]["vikor"])})
        write(out / "case_beta_sweep.csv", rows)

    # ---- console summary
    print(f"N = {N} iterations per scenario, seed {seed}, Beta s = {s0}")
    for name, r in results.items():
        print(f"\n[{name}]  nu = {np.mean(r['nu']):.2f}  delta_w = {np.mean(r['delta_w']):.3f}  "
              f"Pr(CR<0.1) = {np.mean(np.array(r['cr'])<0.1):.3f}")
        for j, c in enumerate(crit):
            lo, hi = q(r["kappa"][:, j])
            print(f"  {c:<20} kappa {r['kappa'][:, j].mean():.3f} ({lo:.3f}-{hi:.3f})  "
                  f"w {r['w'][:, j].mean():.3f} -> {r['w_eff'][:, j].mean():.3f}  "
                  f"shift {100*(r['w_eff'][:, j].mean()-r['w'][:, j].mean()):+.1f} pp")
        for m in ("wsm_v", "topsis_mm", "topsis", "vikor"):
            print(f"  vs additive: {m:<10} tau_b {np.mean(r['tau'][m]):.3f}  "
                  f"diff. ranking {100*(1-np.mean(r['ident'][m])):.1f}%  same top {100*np.mean(r['top'][m]):.1f}%")
    print(f"\nflag vs TOPSIS pooled: flagged {int(F)}, TOPSIS inversions {int(T)}, both {int(B)}, "
          f"precision {B/F:.3f}, recall {B/T:.3f}; sign-test accuracy vs WSM-V {acc:.3f}")
    print(f"deterministic matrix: kappa {np.round(rd.kappa,3)}  delta_w {rd.delta_w:.3f}")


if __name__ == "__main__":
    main()
