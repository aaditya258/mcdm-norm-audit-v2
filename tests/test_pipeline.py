"""End-to-end checks of the CSV loader, corpus analysis and validator, using the
worked example of Table 1 as a stand-in matrix (no Rao data required)."""
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from normaudit import DecisionProblem, load_matrix_csv, save_matrix_csv  # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _worked_problem():
    return DecisionProblem("we", ["C1", "C2"], ["A", "B", "C"],
                           np.array([[0.020, 0.70], [0.005, 0.90], [0.001, 0.80]]),
                           np.array([1, 1]), np.array([0.5, 0.5]))


def test_csv_roundtrip():
    p = _worked_problem()
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "we.csv"
        save_matrix_csv(f, p)
        q = load_matrix_csv(f)
    assert q.criteria == p.criteria and q.alternatives == p.alternatives
    assert np.allclose(q.X, p.X) and np.array_equal(q.o, p.o) and np.allclose(q.w, p.w)


def test_template_loads():
    q = load_matrix_csv(ROOT / "rao2007_decision_matrices" / "TEMPLATE.csv")
    assert q.shape == (3, 3) and list(q.o) == [1, -1, 1]


def test_run_audit_analyse():
    ra = _load("run_audit")
    meta = {"rao_table": "0.0", "problem": "worked example", "source": "paper", "type_code": "R"}
    row, per_crit, pairs = ra.analyse(_worked_problem(), meta)
    assert abs(row["nu_raw"] - 6.41) < 0.01 and abs(row["delta_w_raw"] - 0.365) < 1e-3
    assert row["n_flagged"] == 1 and len(per_crit) == 2
    assert pairs[0]["alt_1"] == "A" and pairs[0]["alt_2"] == "B"
    # with only benefit criteria the two cost conventions coincide
    assert abs(row["nu_raw"] - row["nu_max_minus_x"]) < 1e-12


def test_validator_passes_and_fails():
    vc = _load("validate_corpus")
    p = _worked_problem()
    meta = {"n_alternatives": 3, "n_criteria": 2, "criteria": ["C1", "C2"], "orientation": [1, 1]}
    exp = {"nu": 6.41, "delta_w": 0.365,
           "per_criterion": [{"criterion": "C1", "w_stated": 0.5, "kappa": 0.921, "w_eff": 0.865, "shift_pp": 36.5},
                             {"criterion": "C2", "w_stated": 0.5, "kappa": 0.144, "w_eff": 0.135, "shift_pp": -36.5}],
           "flagged_pairs": [{"pair": ["A", "B"], "inverted_by_topsis": True}],
           "topsis_inversions_not_flagged": [], "convention_comparison": None}
    msgs = vc.check(p, meta, exp, tol=0.0015)
    assert msgs == [], msgs
    # a transcription error must be caught
    bad = _worked_problem()
    bad.X[1, 1] = 0.95
    assert vc.check(bad, meta, exp, tol=0.0015)


def test_expected_json_consistent_with_metadata():
    corpus = ROOT / "rao2007_decision_matrices"
    meta = json.load(open(corpus / "metadata.json"))["matrices"]
    exp = json.load(open(corpus / "expected_audit.json"))["matrices"]
    assert set(meta) == set(exp) and len(meta) == 28
    for tab, m in meta.items():
        e = exp[tab]
        assert len(m["criteria"]) == m["n_criteria"] == len(e["per_criterion"])
        assert abs(sum(m["stated_weights_renormalised"]) - 1) < 0.01
        assert abs(sum(r["w_eff"] for r in e["per_criterion"]) - 1) < 0.01
        # delta_w is half the L1 distance between the printed weight vectors (within rounding)
        dw = 0.5 * sum(abs(r["w_eff"] - r["w_stated"]) for r in e["per_criterion"])
        assert abs(dw - e["delta_w"]) < 0.004, (tab, dw, e["delta_w"])
        k = [r["kappa"] for r in e["per_criterion"]]
        assert abs(max(k) / min(k) - e["nu"]) / e["nu"] < 0.05, (tab, max(k) / min(k), e["nu"])
    assert sum(len(e["flagged_pairs"]) for e in exp.values()) == 79
    assert sum(e["topsis_inversions_total"] for e in exp.values()) == 77


if __name__ == "__main__":
    import inspect
    fns = [f for name, f in list(globals().items()) if name.startswith("test_") and inspect.isfunction(f)]
    for f in fns:
        f()
        print(f"ok  {f.__name__}")
    print(f"{len(fns)} tests passed")
