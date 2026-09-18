"""Ranking methods and agreement measures (Supplement S1).

Every function takes X (n x m, non-negative), o (+1/-1) and w (sums to 1)
and returns a score vector in which *larger is better* (VIKOR's Q is
returned negated for that reason), so that the same helpers can compare
any pair of methods.
"""
from __future__ import annotations

from itertools import combinations
from typing import List, Tuple

import numpy as np
from scipy.stats import kendalltau, rankdata

EPS_M = 1e-12


# ---------------------------------------------------------------- normalisation
def minmax(X, o) -> np.ndarray:
    """Oriented min-max normalisation u_ij in [0,1], 1 = best."""
    X = np.asarray(X, float)
    o = np.asarray(o, float)
    lo, hi = X.min(axis=0), X.max(axis=0)
    R = hi - lo
    R = np.where(R > 0, R, 1.0)
    u = np.where(o > 0, (X - lo) / R, (hi - X) / R)
    return u


def vector(X) -> np.ndarray:
    """Vector normalisation r_ij = x_ij / ||x_j||_2 on the column as published."""
    X = np.asarray(X, float)
    norm = np.sqrt((X ** 2).sum(axis=0))
    norm = np.where(norm > 0, norm, 1.0)
    return X / norm


# ---------------------------------------------------------------- methods
def saw_minmax(X, o, w) -> np.ndarray:
    """Additive model under range normalisation (reference cell)."""
    return minmax(X, o) @ np.asarray(w, float)


def wsm_vector(X, o, w) -> np.ndarray:
    """Weighted sum of signed vector-normalised values (normalisation-only cell)."""
    return vector(X) @ (np.asarray(o, float) * np.asarray(w, float))


def topsis(X, o, w) -> np.ndarray:
    """Classical TOPSIS (Hwang & Yoon 1981): vector normalisation on the column
    as published; ideal = max for a benefit, min for a cost."""
    o = np.asarray(o, float)
    v = vector(X) * np.asarray(w, float)
    vmax, vmin = v.max(axis=0), v.min(axis=0)
    ideal = np.where(o > 0, vmax, vmin)
    anti = np.where(o > 0, vmin, vmax)
    dp = np.sqrt(((v - ideal) ** 2).sum(axis=1))
    dm = np.sqrt(((v - anti) ** 2).sum(axis=1))
    den = dp + dm
    return np.where(den > 0, dm / den, 0.5)


def topsis_minmax(X, o, w) -> np.ndarray:
    """TOPSIS with min-max in place of vector normalisation (aggregation-only cell)."""
    v = minmax(X, o) * np.asarray(w, float)
    ideal, anti = v.max(axis=0), v.min(axis=0)
    dp = np.sqrt(((v - ideal) ** 2).sum(axis=1))
    dm = np.sqrt(((v - anti) ** 2).sum(axis=1))
    den = dp + dm
    return np.where(den > 0, dm / den, 0.5)


def vikor(X, o, w, v: float = 0.5) -> np.ndarray:
    """VIKOR on min-max distances; returns -Q so that larger is better."""
    D = minmax(X, o)
    w = np.asarray(w, float)
    S = ((1 - D) * w).sum(axis=1)
    Rr = ((1 - D) * w).max(axis=1)
    Sstar, Sminus = S.min(), S.max()
    Rstar, Rminus = Rr.min(), Rr.max()
    termS = (S - Sstar) / (Sminus - Sstar) if Sminus > Sstar else np.zeros_like(S)
    termR = (Rr - Rstar) / (Rminus - Rstar) if Rminus > Rstar else np.zeros_like(Rr)
    Q = v * termS + (1 - v) * termR
    return -Q


METHODS = {
    "additive": saw_minmax,        # min-max, weighted sum        (reference)
    "wsm_v": wsm_vector,           # vector,  weighted sum        (normalisation only)
    "topsis_mm": topsis_minmax,    # min-max, TOPSIS aggregation  (aggregation only)
    "topsis": topsis,              # vector,  TOPSIS              (both)
    "vikor": vikor,
}


# ---------------------------------------------------------------- agreement
def ranking(scores) -> np.ndarray:
    """Rank 1 = best (largest score); ties get average rank."""
    return rankdata(-np.asarray(scores, float), method="average")


def kendall_tau_b(s1, s2) -> float:
    r1, r2 = ranking(s1), ranking(s2)
    if np.all(r1 == r1[0]) or np.all(r2 == r2[0]):
        return 1.0
    return float(kendalltau(r1, r2).statistic)


def same_ranking(s1, s2) -> bool:
    return bool(np.array_equal(ranking(s1), ranking(s2)))


def same_top(s1, s2) -> bool:
    return int(np.argmax(s1)) == int(np.argmax(s2))


def inverted_pairs(s1, s2, eps: float = EPS_M) -> List[Tuple[int, int]]:
    """Pairs (i,i') ordered oppositely by scores s1 and s2 (Supplement S1)."""
    s1 = np.asarray(s1, float)
    s2 = np.asarray(s2, float)
    out = []
    for i, k in combinations(range(len(s1)), 2):
        d1, d2 = s1[i] - s1[k], s2[i] - s2[k]
        if abs(d1) > eps and abs(d2) > eps and np.sign(d1) != np.sign(d2):
            out.append((i, k))
    return out
