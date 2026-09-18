"""normaudit: pre-ranking normalization audit for multi-criteria decision support.

Companion code for "Normalization as Implicit Reweighting: A Pre-Ranking
Audit for Multi-Criteria Decision Support".
"""
from .audit import AuditResult, audit, effective_weights, kappa_factors
from .io import DecisionProblem, load_matrix_csv, save_matrix_csv
from .methods import (METHODS, inverted_pairs, kendall_tau_b, minmax, ranking,
                      same_ranking, same_top, saw_minmax, topsis, topsis_minmax,
                      vector, vikor, wsm_vector)

__version__ = "1.0.0"
__all__ = [
    "AuditResult", "audit", "effective_weights", "kappa_factors",
    "DecisionProblem", "load_matrix_csv", "save_matrix_csv",
    "METHODS", "inverted_pairs", "kendall_tau_b", "minmax", "ranking",
    "same_ranking", "same_top", "saw_minmax", "topsis", "topsis_minmax",
    "vector", "vikor", "wsm_vector",
]
