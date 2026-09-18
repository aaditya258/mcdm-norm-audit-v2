"""Table 1 of the main paper (Section 3.5): the three-alternative worked example."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from normaudit import audit, minmax, ranking, saw_minmax, vector, wsm_vector  # noqa: E402

X = np.array([[0.020, 0.70], [0.005, 0.90], [0.001, 0.80]])
o = np.array([1, 1])
w = np.array([0.5, 0.5])
alts = ["A", "B", "C"]


def main():
    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    res = audit(X, o, w)
    u, r = minmax(X, o), vector(X)
    U_mm = saw_minmax(X, o, w)
    U_vec = wsm_vector(X, o, w)
    U_eff = saw_minmax(X, o, res.w_eff)
    rows = []
    for i, a in enumerate(alts):
        rows.append({"alt": a, "x1": X[i, 0], "x2": X[i, 1], "u1": u[i, 0], "u2": u[i, 1],
                     "r1": r[i, 0], "r2": r[i, 1], "U_minmax_w": U_mm[i], "U_vec_w": U_vec[i],
                     "U_minmax_weff": U_eff[i], "rank_minmax": int(ranking(U_mm)[i]),
                     "rank_vec": int(ranking(U_vec)[i])})
    with open(out / "table1_worked_example.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)
    print("\n".join(res.summary_lines(["C1", "C2"], alts)))
    print(f"R = {X.max(0)-X.min(0)}, norms = {np.sqrt((X**2).sum(0))}")
    s1, s2 = res.sigma[(0, 1)]
    print(f"pair (A,B): sum a_j = {s1:+.3f}, sum kappa_j a_j = {s2:+.3f}")
    print(f"{'alt':<4}{'u1':>7}{'u2':>7}{'r1':>7}{'r2':>7}{'U_mm(w)':>9}{'U_vec(w)':>10}{'U_mm(w~)':>10}")
    for rw in rows:
        print(f"{rw['alt']:<4}{rw['u1']:7.3f}{rw['u2']:7.3f}{rw['r1']:7.3f}{rw['r2']:7.3f}"
              f"{rw['U_minmax_w']:9.3f}{rw['U_vec_w']:10.3f}{rw['U_minmax_weff']:10.3f}")


if __name__ == "__main__":
    main()
