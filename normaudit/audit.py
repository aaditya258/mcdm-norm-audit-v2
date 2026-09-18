"""Pre-ranking normalization audit (Section 4 of the main paper).

Given a decision matrix X (n alternatives x m criteria, non-negative), an
orientation vector o (+1 benefit, -1 cost) and stated weights w, the audit
reports the reweighting factors kappa_j = R_j / ||x_j||_2, the effective
weights w~ that vector normalization implicitly applies, the two summary
numbers nu and delta_w (Eq. 4), and the set P of alternative pairs whose
order under a weighted sum depends only on the normalization rule (Eq. 5).

All quantities use the *untransformed* cost-column convention of the paper:
kappa_j is computed on the column as published, whether benefit or cost.
Pass ``cost_convention="max_minus_x"`` to reproduce the alternative
convention of Supplement S2.6 (cost columns replaced by max - x before
vector normalization).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import List, Sequence, Tuple

import numpy as np


@dataclass
class AuditResult:
    kappa: np.ndarray            # length m; NaN for dropped criteria
    w_stated: np.ndarray         # stated weights renormalised over retained criteria (NaN if dropped)
    w_eff: np.ndarray            # effective weights w~ (NaN if dropped)
    nu: float
    delta_w: float
    flagged_pairs: List[Tuple[int, int]]
    retained: List[int] = field(default_factory=list)   # J
    dropped: List[int] = field(default_factory=list)    # Z
    sigma: dict = field(default_factory=dict)           # (i,i') -> (sigma1, sigma2)

    @property
    def shift_pp(self) -> np.ndarray:
        """w~_j - w_j in percentage points."""
        return 100.0 * (self.w_eff - self.w_stated)

    def summary_lines(self, criteria: Sequence[str] | None = None,
                      alternatives: Sequence[str] | None = None) -> List[str]:
        m = len(self.kappa)
        criteria = list(criteria) if criteria is not None else [f"C{j+1}" for j in range(m)]
        lines = [f"nu = {self.nu:.3f}   delta_w = {self.delta_w:.3f}",
                 f"{'criterion':<20}{'w':>8}{'kappa':>8}{'w_eff':>8}{'shift(pp)':>11}"]
        for j in range(m):
            if j in self.dropped:
                lines.append(f"{criteria[j]:<20}{'dropped (zero range)':>35}")
            else:
                lines.append(f"{criteria[j]:<20}{self.w_stated[j]:8.3f}{self.kappa[j]:8.3f}"
                             f"{self.w_eff[j]:8.3f}{self.shift_pp[j]:+11.1f}")
        if alternatives is None:
            names = None
        else:
            names = list(alternatives)
        if self.flagged_pairs:
            pairs = [f"{names[i]}--{names[k]}" if names else f"A{i+1}--A{k+1}"
                     for i, k in self.flagged_pairs]
            lines.append(f"flagged pairs ({len(self.flagged_pairs)}): " + "; ".join(pairs))
        else:
            lines.append("no flagged pairs")
        return lines


def kappa_factors(X: np.ndarray, o: np.ndarray, cost_convention: str = "raw",
                  eps_R: float = 1e-12) -> Tuple[np.ndarray, np.ndarray, List[int], List[int]]:
    """Step 1 (Measure). Returns (kappa, R, retained, dropped)."""
    X = np.asarray(X, dtype=float)
    o = np.asarray(o, dtype=float)
    Xc = X.copy()
    if cost_convention == "max_minus_x":
        for j in range(X.shape[1]):
            if o[j] < 0:
                Xc[:, j] = X[:, j].max() - X[:, j]
    elif cost_convention != "raw":
        raise ValueError("cost_convention must be 'raw' or 'max_minus_x'")
    R = Xc.max(axis=0) - Xc.min(axis=0)
    norm = np.sqrt((Xc ** 2).sum(axis=0))
    retained = [j for j in range(X.shape[1]) if R[j] > eps_R]
    dropped = [j for j in range(X.shape[1]) if R[j] <= eps_R]
    kappa = np.full(X.shape[1], np.nan)
    for j in retained:
        kappa[j] = R[j] / norm[j]
    return kappa, R, retained, dropped


def effective_weights(w: np.ndarray, kappa: np.ndarray, retained: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Step 2 (Reweight). Returns (w renormalised over J, w~)."""
    w = np.asarray(w, dtype=float)
    w_ret = np.full(len(w), np.nan)
    w_eff = np.full(len(w), np.nan)
    idx = list(retained)
    s = w[idx].sum()
    w_ret[idx] = w[idx] / s
    denom = float(np.sum(w_ret[idx] * kappa[idx]))
    w_eff[idx] = w_ret[idx] * kappa[idx] / denom
    return w_ret, w_eff


def audit(X, o, w, eps_R: float = 1e-12, eps_m: float = 1e-12,
          cost_convention: str = "raw") -> AuditResult:
    """Run the four-step audit of Section 4 and return an AuditResult."""
    X = np.asarray(X, dtype=float)
    o = np.asarray(o, dtype=float)
    w = np.asarray(w, dtype=float)
    n, m = X.shape
    if (X < 0).any():
        raise ValueError("decision matrix must be non-negative")
    if not np.all(np.isin(o, (1, -1))):
        raise ValueError("orientation entries must be +1 or -1")
    if (w <= 0).any():
        raise ValueError("weights must be strictly positive")

    # Step 1
    kappa, R, retained, dropped = kappa_factors(X, o, cost_convention, eps_R)
    if not retained:
        raise ValueError("every criterion has zero range")
    # Step 2
    w_ret, w_eff = effective_weights(w, kappa, retained)
    # Step 3
    k = kappa[retained]
    nu = float(k.max() / k.min())
    delta_w = float(0.5 * np.abs(w_eff[retained] - w_ret[retained]).sum())
    # Step 4: pairwise screen.  Under the max-minus-x convention the oriented
    # column is exactly the transformed column, so a_j is computed on it.
    Xo = X.copy()
    if cost_convention == "max_minus_x":
        for j in retained:
            if o[j] < 0:
                Xo[:, j] = X[:, j].max() - X[:, j]
        oo = np.ones(m)
    else:
        oo = o
    flagged, sigma = [], {}
    for i, i2 in combinations(range(n), 2):
        a = np.array([oo[j] * w_ret[j] * (Xo[i, j] - Xo[i2, j]) / R[j] for j in retained])
        s1 = float(a.sum())
        s2 = float((k * a).sum())
        sigma[(i, i2)] = (s1, s2)
        if abs(s1) > eps_m and abs(s2) > eps_m and np.sign(s1) != np.sign(s2):
            flagged.append((i, i2))
    return AuditResult(kappa=kappa, w_stated=w_ret, w_eff=w_eff, nu=nu, delta_w=delta_w,
                       flagged_pairs=flagged, retained=retained, dropped=dropped, sigma=sigma)
