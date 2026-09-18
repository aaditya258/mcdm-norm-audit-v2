"""Run the audit on every decision matrix in rao2007_decision_matrices/ and
produce the corpus outputs of Section 6 and Supplement S2.6 / S5.

For each CSV present it computes: kappa, effective weights, nu, delta_w under
both cost conventions, the flagged pairs, the pairs inverted by classical
TOPSIS relative to the min-max additive model, the aggregation-only
inversions (TOPSIS under min-max vs additive), the Kendall tau_b between
methods, and whether TOPSIS and the additive model agree on the top-ranked
alternative.

Outputs:
  outputs/audit_results_full.csv      one row per matrix (Table 6 + Table S2.6 columns)
  outputs/audit_per_criterion.csv     one row per matrix x criterion (the S5 tables)
  outputs/audit_flagged_pairs.csv     one row per flagged pair, with TOPSIS status
  outputs/audit_summary.txt           the pooled numbers quoted in Section 6

Matrices that have not yet been transcribed are simply absent; the summary
reports how many of the 28 were found.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from normaudit import (audit, inverted_pairs, kendall_tau_b, load_matrix_csv,  # noqa: E402
                       same_top, saw_minmax, topsis, topsis_minmax)

CORPUS = ROOT / "rao2007_decision_matrices"


def analyse(problem, meta):
    X, o, w = problem.X, problem.o, problem.w
    raw = audit(X, o, w, cost_convention="raw")
    tr = audit(X, o, w, cost_convention="max_minus_x")
    add = saw_minmax(X, o, w)
    tp = topsis(X, o, w)
    tpm = topsis_minmax(X, o, w)
    inv_topsis = set(inverted_pairs(add, tp))
    inv_aggr = set(inverted_pairs(add, tpm))
    flagged = set(raw.flagged_pairs)
    names = problem.alternatives
    ret = raw.retained
    shift = raw.shift_pp
    jmax = max(ret, key=lambda j: abs(shift[j]))
    row = {
        "rao_table": meta["rao_table"], "problem": meta["problem"], "source": meta["source"],
        "type_code": meta.get("type_code"), "n": X.shape[0], "m": X.shape[1],
        "m_retained": len(ret), "dropped_criteria": ";".join(problem.criteria[j] for j in raw.dropped),
        "nu_raw": raw.nu, "delta_w_raw": raw.delta_w,
        "nu_max_minus_x": tr.nu, "delta_w_max_minus_x": tr.delta_w,
        "largest_shift_criterion": problem.criteria[jmax], "largest_shift_pp": shift[jmax],
        "largest_loss_is_cost": int(o[min(ret, key=lambda j: shift[j])] < 0),
        "n_pairs": X.shape[0] * (X.shape[0] - 1) // 2,
        "n_flagged": len(flagged), "n_topsis_inversions": len(inv_topsis),
        "n_topsis_inversions_flagged": len(flagged & inv_topsis),
        "n_aggregation_only_inversions": len(inv_aggr),
        "tau_topsis_vs_additive": kendall_tau_b(add, tp),
        "tau_topsis_mm_vs_additive": kendall_tau_b(add, tpm),
        "same_top_topsis_additive": int(same_top(add, tp)),
        "top_additive": names[int(np.argmax(add))], "top_topsis": names[int(np.argmax(tp))],
        "top_swap_pair_flagged": "",
        "weights_note": ";".join(problem.notes),
    }
    if not row["same_top_topsis_additive"]:
        i, k = sorted((int(np.argmax(add)), int(np.argmax(tp))))
        row["top_swap_pair_flagged"] = int((i, k) in flagged)
    per_crit = [{"rao_table": meta["rao_table"], "criterion": problem.criteria[j],
                 "orientation": int(o[j]), "w_stated": raw.w_stated[j], "kappa": raw.kappa[j],
                 "w_eff": raw.w_eff[j], "shift_pp": shift[j],
                 "kappa_max_minus_x": tr.kappa[j], "w_eff_max_minus_x": tr.w_eff[j]}
                for j in range(X.shape[1])]
    pairs = [{"rao_table": meta["rao_table"], "alt_1": names[i], "alt_2": names[k],
              "flagged": 1, "inverted_by_topsis": int((i, k) in inv_topsis)} for i, k in sorted(flagged)]
    pairs += [{"rao_table": meta["rao_table"], "alt_1": names[i], "alt_2": names[k],
               "flagged": 0, "inverted_by_topsis": 1} for i, k in sorted(inv_topsis - flagged)]
    return row, per_crit, pairs


def write(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(CORPUS))
    ap.add_argument("--out", default=str(ROOT / "outputs"))
    args = ap.parse_args()
    corpus, out = Path(args.corpus), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.load(open(corpus / "metadata.json"))["matrices"]

    rows, per_crit, pairs, missing = [], [], [], []
    for tab in sorted(meta, key=lambda t: tuple(int(x) for x in t.split("."))):
        f = corpus / meta[tab]["csv_file"]
        if not f.exists():
            missing.append(tab)
            continue
        r, pc, pr = analyse(load_matrix_csv(f), meta[tab])
        rows.append(r); per_crit += pc; pairs += pr

    lines = [f"{len(rows)} of {len(meta)} matrices found in {corpus}"]
    if missing:
        lines.append("not yet transcribed: " + ", ".join(missing))
    if rows:
        rows.sort(key=lambda r: r["delta_w_raw"])
        write(out / "audit_results_full.csv", rows)
        write(out / "audit_per_criterion.csv", per_crit)
        write(out / "audit_flagged_pairs.csv", pairs)
        dw = np.array([r["delta_w_raw"] for r in rows])
        F = sum(r["n_flagged"] for r in rows)
        T = sum(r["n_topsis_inversions"] for r in rows)
        B = sum(r["n_topsis_inversions_flagged"] for r in rows)
        A = sum(r["n_aggregation_only_inversions"] for r in rows)
        top_dis = [r for r in rows if not r["same_top_topsis_additive"]]
        lines += [
            f"delta_w: median {np.median(dw):.3f}, IQR {np.percentile(dw,25):.3f}-{np.percentile(dw,75):.3f}",
            f"matrices with delta_w >= 0.10: {(dw>=0.10).sum()}; >= 0.20: {(dw>=0.20).sum()}; < 0.02: {(dw<0.02).sum()}",
            f"matrices with at least one flagged pair: {sum(1 for r in rows if r['n_flagged'])}",
            f"pooled: flagged {F}, TOPSIS inversions {T}, both {B} -> precision {B/F if F else float('nan'):.2f}, recall {B/T if T else float('nan'):.2f}",
            f"aggregation-only inversions (TOPSIS under min-max vs additive): {A}",
            f"top-rank disagreements TOPSIS vs additive: {len(top_dis)}; swap pair flagged in {sum(r['top_swap_pair_flagged']==1 for r in top_dis)}",
            f"criterion losing most weight is a cost in {sum(r['largest_loss_is_cost'] for r in rows)} of {len(rows)}",
        ]
        for code, label in (("R", "raw"), ("M", "mixed")):
            sub = [r["delta_w_raw"] for r in rows if r["type_code"] == code]
            if sub:
                lines.append(f"median delta_w, {label} ({len(sub)}): {np.median(sub):.3f}")
        if len(rows) >= 5:
            t1 = np.array([r["tau_topsis_vs_additive"] for r in rows])
            t2 = np.array([r["tau_topsis_mm_vs_additive"] for r in rows])
            r1, r2 = spearmanr(dw, t1), spearmanr(dw, t2)
            lines.append(f"Spearman(delta_w, tau TOPSIS-additive) = {r1.statistic:+.2f} (p={r1.pvalue:.3f}); "
                         f"Spearman(delta_w, tau aggregation-only) = {r2.statistic:+.2f} (p={r2.pvalue:.3f})")
    text = "\n".join(lines)
    (out / "audit_summary.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
