"""Simulation of Section 7 / Supplement S6.

Generator (S6): n ~ U{4,5,6,8,10}, m ~ U{3,4,5}, l ~ U(0.2,4), offset o ~ U(0,5)
once per problem; per criterion d_j ~ U(0,l); x_ij = 10^{U(-d_j,0)} + o*U(0,1)*xbar_j
with one U(0,1) draw per column and xbar_j the column mean before the offset;
w ~ Dirichlet(1). All criteria are benefits. 4000 problems, seed 20260918.

Outputs (in outputs/):
    sim_problems.csv        one row per problem: n, m, nu, delta_w, tau_b and
                            identical-ranking indicator for the three cells
    sim_table_bins.csv      Table S6.1 (mean tau_b by nu bin, bootstrap CI, non-identical %)
    sim_table_ols.csv       Table S6.2 (OLS with n, m controls, HC0 SEs, Spearman rho)
    fig_sim.pdf / .png      Fig. 3 of the main paper
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from normaudit import (audit, kendall_tau_b, same_ranking, saw_minmax,   # noqa: E402
                       topsis, topsis_minmax, wsm_vector)

BINS = [(0, 1.05, "<=1.05"), (1.05, 1.25, "1.05-1.25"), (1.25, 1.75, "1.25-1.75"),
        (1.75, 3.0, "1.75-3.0"), (3.0, 6.0, "3.0-6.0"), (6.0, np.inf, ">6.0")]
CELLS = [("norm_only", "Normalisation only (WSM-V)"),
         ("aggr_only", "Aggregation only (TOPSIS, min-max)"),
         ("both", "Both (TOPSIS)")]


def generate_problem(rng: np.random.Generator):
    n = int(rng.choice([4, 5, 6, 8, 10]))
    m = int(rng.choice([3, 4, 5]))
    ell = rng.uniform(0.2, 4.0)
    off = rng.uniform(0.0, 5.0)
    X = np.empty((n, m))
    for j in range(m):
        d = rng.uniform(0.0, ell)
        col = 10 ** rng.uniform(-d, 0.0, n)
        X[:, j] = col + off * rng.uniform() * col.mean()
    w = rng.dirichlet(np.ones(m))
    return X, np.ones(m), w


def run(n_problems: int, seed: int):
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_problems):
        X, o, w = generate_problem(rng)
        res = audit(X, o, w)
        ref = saw_minmax(X, o, w)
        cells = {"norm_only": wsm_vector(X, o, w),
                 "aggr_only": topsis_minmax(X, o, w),
                 "both": topsis(X, o, w)}
        row = {"problem": p, "n": X.shape[0], "m": X.shape[1],
               "nu": res.nu, "delta_w": res.delta_w, "n_flagged": len(res.flagged_pairs)}
        for key, s in cells.items():
            row[f"tau_{key}"] = kendall_tau_b(ref, s)
            row[f"ident_{key}"] = int(same_ranking(ref, s))
        rows.append(row)
    return rows


def bootstrap_ci(x, rng, B=2000):
    x = np.asarray(x, float)
    if len(x) == 0:
        return (np.nan, np.nan)
    means = np.array([rng.choice(x, len(x)).mean() for _ in range(B)])
    return tuple(np.percentile(means, [2.5, 97.5]))


def table_bins(rows, rng):
    nu = np.array([r["nu"] for r in rows])
    out = []
    for lo, hi, label in BINS:
        mask = (nu > lo) & (nu <= hi) if lo > 0 else (nu <= hi)
        rec = {"nu_bin": label, "count": int(mask.sum())}
        for key, _ in CELLS:
            t = np.array([r[f"tau_{key}"] for r in rows])[mask]
            ci = bootstrap_ci(t, rng)
            rec[f"tau_{key}"] = t.mean() if mask.any() else np.nan
            rec[f"tau_{key}_lo"], rec[f"tau_{key}_hi"] = ci
            rec[f"nonident_{key}_pct"] = 100 * (1 - np.array([r[f"ident_{key}"] for r in rows])[mask].mean()) if mask.any() else np.nan
        out.append(rec)
    return out


def ols_hc0(y, Xd):
    """OLS with HC0 robust SE; returns (beta, se, r2)."""
    Xd = np.column_stack([np.ones(len(y)), Xd])
    XtX_inv = np.linalg.inv(Xd.T @ Xd)
    beta = XtX_inv @ Xd.T @ y
    resid = y - Xd @ beta
    meat = (Xd * resid[:, None] ** 2).T @ Xd
    cov = XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.diag(cov))
    r2 = 1 - resid.var() / y.var()
    return beta, se, r2


def table_ols(rows):
    n = np.array([r["n"] for r in rows], float)
    m = np.array([r["m"] for r in rows], float)
    preds = {"log10_nu": np.log10([r["nu"] for r in rows]),
             "delta_w": np.array([r["delta_w"] for r in rows])}
    out = []
    for pname, x in preds.items():
        for key, label in CELLS:
            y = np.array([r[f"tau_{key}"] for r in rows])
            beta, se, r2 = ols_hc0(y, np.column_stack([x, n, m]))
            out.append({"predictor": pname, "component": label, "slope": beta[1],
                        "robust_se": se[1], "t": beta[1] / se[1], "r2": r2,
                        "spearman_rho": spearmanr(x, y).statistic})
    return out


def write_csv(path, rows):
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)


def make_figure(bins, path_stem):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = [b["nu_bin"] for b in bins]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    styles = {"norm_only": ("o-", "Normalisation only"),
              "aggr_only": ("s--", "Aggregation only"),
              "both": ("^-.", "Both (TOPSIS)")}
    for key, (st, lab) in styles.items():
        t = np.array([b[f"tau_{key}"] for b in bins])
        lo = np.array([b[f"tau_{key}_lo"] for b in bins])
        hi = np.array([b[f"tau_{key}_hi"] for b in bins])
        axes[0].errorbar(x, t, yerr=[t - lo, hi - t], fmt=st, capsize=3, label=lab)
        axes[1].plot(x, [b[f"nonident_{key}_pct"] for b in bins], st, label=lab)
    axes[0].set_ylabel(r"mean Kendall $\tau_b$ vs. min-max additive")
    axes[1].set_ylabel("non-identical complete rankings (%)")
    for ax, t in zip(axes, ("(a)", "(b)")):
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30)
        ax.set_xlabel(r"$\nu$ bin")
        ax.set_title(t, loc="left")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{path_stem}.pdf")
    fig.savefig(f"{path_stem}.png", dpi=160)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-problems", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=20260918)
    ap.add_argument("--out", default=str(ROOT / "outputs"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = run(args.n_problems, args.seed)
    write_csv(out / "sim_problems.csv", rows)
    bins = table_bins(rows, np.random.default_rng(args.seed + 1))
    write_csv(out / "sim_table_bins.csv", bins)
    ols = table_ols(rows)
    write_csv(out / "sim_table_ols.csv", ols)
    make_figure(bins, str(out / "fig_sim"))

    print(f"{len(rows)} problems, seed {args.seed}")
    print(f"{'nu bin':<11}{'count':>6}  {'norm':>6} {'aggr':>6} {'both':>6}   non-identical % (norm/aggr/both)")
    for b in bins:
        print(f"{b['nu_bin']:<11}{b['count']:>6}  {b['tau_norm_only']:6.3f} {b['tau_aggr_only']:6.3f} "
              f"{b['tau_both']:6.3f}   {b['nonident_norm_only_pct']:5.1f} / {b['nonident_aggr_only_pct']:5.1f} / {b['nonident_both_pct']:5.1f}")
    print()
    for r in ols:
        print(f"{r['predictor']:<9} {r['component']:<36} slope {r['slope']:+.3f}  se {r['robust_se']:.4f}  "
              f"t {r['t']:+.1f}  R2 {r['r2']:.3f}  rho_S {r['spearman_rho']:+.3f}")


if __name__ == "__main__":
    main()
