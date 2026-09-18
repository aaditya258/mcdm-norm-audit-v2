"""Checks against numbers stated in the main paper and supplement.

Run with:  python -m pytest tests/    (or simply  python tests/test_audit.py)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from normaudit import (audit, inverted_pairs, ranking, saw_minmax, topsis,   # noqa: E402
                       wsm_vector)

# Worked example, Section 3.5 / Table 1
X_WE = np.array([[0.020, 0.70],
                 [0.005, 0.90],
                 [0.001, 0.80]])
O_WE = np.array([1, 1])
W_WE = np.array([0.5, 0.5])


def test_worked_example_numbers():
    res = audit(X_WE, O_WE, W_WE)
    assert np.allclose(res.kappa, [0.921, 0.144], atol=5e-4), res.kappa
    assert np.allclose(res.w_eff, [0.865, 0.135], atol=5e-4), res.w_eff
    assert abs(res.nu - 6.41) < 0.01, res.nu
    assert abs(res.delta_w - 0.365) < 5e-4, res.delta_w
    assert res.flagged_pairs == [(0, 1)]
    s1, s2 = res.sigma[(0, 1)]
    assert abs(s1 + 0.105) < 1e-3 and abs(s2 - 0.292) < 1e-3


def test_worked_example_orders():
    # min-max: B > A > C ; vector: A > B > C
    assert list(ranking(saw_minmax(X_WE, O_WE, W_WE))) == [2, 1, 3]
    assert list(ranking(wsm_vector(X_WE, O_WE, W_WE))) == [1, 2, 3]
    # Eq. (2): min-max with effective weights reproduces the vector order
    res = audit(X_WE, O_WE, W_WE)
    assert list(ranking(saw_minmax(X_WE, O_WE, res.w_eff))) == [1, 2, 3]


def test_identity_random_matrices():
    """WSM-V under w ranks exactly as the additive model under w~ (Eq. 2),
    for benefit and cost criteria alike."""
    rng = np.random.default_rng(1)
    for _ in range(300):
        n, m = rng.integers(3, 9), rng.integers(2, 7)
        X = rng.uniform(0, 1, (n, m)) * 10 ** rng.uniform(-3, 3, m) + rng.uniform(0, 5, m)
        o = rng.choice([1, -1], m)
        w = rng.dirichlet(np.ones(m))
        res = audit(X, o, w)
        r1 = ranking(wsm_vector(X, o, w))
        r2 = ranking(saw_minmax(X, o, res.w_eff))
        assert np.array_equal(r1, r2)
        # the sign test reproduces every inversion between the two weighted sums
        inv = set(inverted_pairs(saw_minmax(X, o, w), wsm_vector(X, o, w)))
        assert inv == set(res.flagged_pairs)


def test_kappa_bounds_and_scale_invariance():
    rng = np.random.default_rng(2)
    X = rng.uniform(0, 1, (6, 4))
    o = np.ones(4)
    w = np.full(4, 0.25)
    k = audit(X, o, w).kappa
    assert np.all(k > 0) and np.all(k <= 1)
    # scaling leaves kappa alone; a positive offset lowers it (S2.2)
    assert np.allclose(audit(X * 7.3, o, w).kappa, k)
    assert np.all(audit(X + 2.0, o, w).kappa < k)


def test_zero_range_criterion_dropped():
    X = np.array([[1, 5, 2.0], [3, 5, 1.0], [2, 5, 4.0]])
    res = audit(X, [1, 1, -1], [0.4, 0.3, 0.3])
    assert res.dropped == [1] and res.retained == [0, 2]
    assert np.isnan(res.kappa[1]) and abs(np.nansum(res.w_eff) - 1) < 1e-12


def test_cost_convention_changes_kappa():
    X = np.array([[100, 1.0], [110, 2.0], [130, 4.0]])
    o = np.array([-1, 1])
    w = np.array([0.5, 0.5])
    raw = audit(X, o, w, cost_convention="raw")
    tr = audit(X, o, w, cost_convention="max_minus_x")
    assert not np.allclose(raw.kappa, tr.kappa)
    # with no cost criteria the two conventions agree
    o2 = np.array([1, 1])
    assert np.allclose(audit(X, o2, w, cost_convention="raw").kappa,
                       audit(X, o2, w, cost_convention="max_minus_x").kappa)


def test_topsis_sanity():
    # a dominating alternative must be ranked first by TOPSIS
    X = np.array([[9, 1.0], [5, 5.0], [1, 9.0]])
    assert np.argmax(topsis(X, [1, -1], [0.5, 0.5])) == 0


if __name__ == "__main__":
    import inspect
    fns = [f for name, f in globals().items() if name.startswith("test_") and inspect.isfunction(f)]
    for f in fns:
        f()
        print(f"ok  {f.__name__}")
    print(f"{len(fns)} tests passed")
