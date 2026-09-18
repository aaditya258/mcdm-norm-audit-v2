"""CSV format for decision matrices (Supplement S7).

    ,C1,C2,C3            <- header: first cell blank (or 'alternative'), then criteria
    __orientation__,1,-1,1
    __weight__,0.5,0.3,0.2
    A1,0.02,0.70,3.1
    A2,...

Orientation may be written 1/-1, +/-, or benefit/cost (case-insensitive).
Weights need not sum to one; they are renormalised on load and a warning
is recorded in ``DecisionProblem.notes`` when the printed sum is not 1.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np

_BENEFIT = {"1", "+1", "+", "benefit", "b", "max"}
_COST = {"-1", "-", "cost", "c", "min"}


@dataclass
class DecisionProblem:
    name: str
    criteria: List[str]
    alternatives: List[str]
    X: np.ndarray
    o: np.ndarray
    w: np.ndarray
    notes: List[str] = field(default_factory=list)

    @property
    def shape(self):
        return self.X.shape


def _parse_orientation(tok: str) -> int:
    t = tok.strip().lower()
    if t in _BENEFIT:
        return 1
    if t in _COST:
        return -1
    raise ValueError(f"unrecognised orientation token {tok!r}")


def load_matrix_csv(path) -> DecisionProblem:
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as fh:
        rows = [r for r in csv.reader(fh) if any(c.strip() for c in r)]
    header = rows[0]
    criteria = [c.strip() for c in header[1:]]
    o = w = None
    alts, data = [], []
    for r in rows[1:]:
        key = r[0].strip()
        vals = r[1:1 + len(criteria)]
        if key == "__orientation__":
            o = np.array([_parse_orientation(v) for v in vals])
        elif key == "__weight__":
            w = np.array([float(v) for v in vals])
        elif key.startswith("__"):
            continue  # reserved for future metadata rows
        else:
            alts.append(key)
            data.append([float(v) for v in vals])
    if o is None or w is None:
        raise ValueError(f"{path.name}: missing __orientation__ or __weight__ row")
    X = np.array(data, float)
    notes = []
    s = w.sum()
    if abs(s - 1.0) > 1e-3:
        notes.append(f"printed weights sum to {s:.3f}; renormalised")
    w = w / s
    return DecisionProblem(path.stem, criteria, alts, X, o, w, notes)


def save_matrix_csv(path, problem: DecisionProblem) -> None:
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["alternative"] + list(problem.criteria))
        wr.writerow(["__orientation__"] + [int(v) for v in problem.o])
        wr.writerow(["__weight__"] + [f"{v:.6g}" for v in problem.w])
        for name, row in zip(problem.alternatives, problem.X):
            wr.writerow([name] + [f"{v:.10g}" for v in row])
