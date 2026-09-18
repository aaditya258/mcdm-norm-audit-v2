"""Validate transcribed Rao (2007) matrices against the published audit output.

For every CSV in rao2007_decision_matrices/ that has an entry in
expected_audit.json, this script checks:
  * shape, criterion names and orientations against metadata.json
  * kappa_j, effective weights and the shift for each criterion (S5, 3 d.p.)
  * nu and delta_w under both cost conventions (S5 and Table S2.6)
  * the set of flagged pairs and which of them TOPSIS inverts (S5)
  * the TOPSIS inversions that are not flagged (S5)
A matrix passes when every number agrees to the printed precision. This is
the transcription check to run after entering each table from the book.

Usage:  python scripts/validate_corpus.py [--tol 0.0015] [--update-metadata]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from normaudit import (audit, inverted_pairs, load_matrix_csv, saw_minmax,  # noqa: E402
                       topsis)

CORPUS = ROOT / "rao2007_decision_matrices"


def check(problem, meta, exp, tol):
    msgs = []
    n, m = problem.shape
    if (n, m) != (meta["n_alternatives"], meta["n_criteria"]):
        msgs.append(f"shape {n}x{m} != {meta['n_alternatives']}x{meta['n_criteria']}")
    if list(problem.criteria) != meta["criteria"]:
        msgs.append(f"criteria {problem.criteria} != {meta['criteria']}")
    if list(map(int, problem.o)) != meta["orientation"]:
        msgs.append(f"orientation {list(map(int, problem.o))} != {meta['orientation']}")
    if msgs:
        return msgs
    raw = audit(problem.X, problem.o, problem.w, cost_convention="raw")
    tr = audit(problem.X, problem.o, problem.w, cost_convention="max_minus_x")
    for j, row in enumerate(exp["per_criterion"]):
        for key, val in (("w_stated", raw.w_stated[j]), ("kappa", raw.kappa[j]), ("w_eff", raw.w_eff[j])):
            if abs(val - row[key]) > tol:
                msgs.append(f"{row['criterion']}: {key} {val:.3f} != {row[key]:.3f}")
        if abs(raw.shift_pp[j] - row["shift_pp"]) > 0.15:
            msgs.append(f"{row['criterion']}: shift {raw.shift_pp[j]:+.1f} != {row['shift_pp']:+.1f}")
    if abs(raw.nu - exp["nu"]) > 0.015:
        msgs.append(f"nu {raw.nu:.2f} != {exp['nu']:.2f}")
    if abs(raw.delta_w - exp["delta_w"]) > tol:
        msgs.append(f"delta_w {raw.delta_w:.3f} != {exp['delta_w']:.3f}")
    conv = exp.get("convention_comparison")
    if conv:
        if abs(tr.nu - conv["nu_max_minus_x"]) > 0.015:
            msgs.append(f"nu (max-x) {tr.nu:.2f} != {conv['nu_max_minus_x']:.2f}")
        if abs(tr.delta_w - conv["delta_w_max_minus_x"]) > tol:
            msgs.append(f"delta_w (max-x) {tr.delta_w:.3f} != {conv['delta_w_max_minus_x']:.3f}")
    names = problem.alternatives
    got_flag = {frozenset((names[i], names[k])) for i, k in raw.flagged_pairs}
    exp_flag = {frozenset(p["pair"]) for p in exp["flagged_pairs"]}
    if got_flag != exp_flag:
        msgs.append(f"flagged pairs differ: extra {sorted(map(sorted, got_flag - exp_flag))}, "
                    f"missing {sorted(map(sorted, exp_flag - got_flag))}")
    inv = {frozenset((names[i], names[k])) for i, k in
           inverted_pairs(saw_minmax(problem.X, problem.o, problem.w), topsis(problem.X, problem.o, problem.w))}
    exp_inv = {frozenset(p["pair"]) for p in exp["flagged_pairs"] if p["inverted_by_topsis"]}
    exp_inv |= {frozenset(p) for p in exp["topsis_inversions_not_flagged"]}
    if inv != exp_inv:
        msgs.append(f"TOPSIS inversions differ: extra {sorted(map(sorted, inv - exp_inv))}, "
                    f"missing {sorted(map(sorted, exp_inv - inv))}")
    return msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=0.0015, help="tolerance on 3-d.p. quantities")
    ap.add_argument("--update-metadata", action="store_true",
                    help="set 'transcribed': true in metadata.json for matrices that pass")
    args = ap.parse_args()
    meta_doc = json.load(open(CORPUS / "metadata.json"))
    meta = meta_doc["matrices"]
    exp = json.load(open(CORPUS / "expected_audit.json"))["matrices"]
    passed, failed, missing = [], [], []
    for tab in sorted(meta, key=lambda t: tuple(int(x) for x in t.split("."))):
        f = CORPUS / meta[tab]["csv_file"]
        if not f.exists():
            missing.append(tab)
            continue
        try:
            msgs = check(load_matrix_csv(f), meta[tab], exp[tab], args.tol)
        except Exception as e:  # noqa: BLE001
            msgs = [f"could not load: {e}"]
        if msgs:
            failed.append(tab)
            print(f"FAIL  Table {tab} ({meta[tab]['csv_file']})")
            for msg in msgs:
                print(f"      - {msg}")
        else:
            passed.append(tab)
            print(f"pass  Table {tab}")
    print(f"\n{len(passed)} pass, {len(failed)} fail, {len(missing)} not yet transcribed")
    if missing:
        print("missing: " + ", ".join(missing))
    if args.update_metadata:
        for tab in meta:
            meta[tab]["transcribed"] = tab in passed
        json.dump(meta_doc, open(CORPUS / "metadata.json", "w"), indent=1)
        print("metadata.json updated")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
