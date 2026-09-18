"""Two-layer Bayesian network for the HPCI failure-mode case (Supplement S3).

Structure: binary stressor nodes (nominal/abnormal) with independent priors;
binary failure-mode nodes each with a subset of the stressors as parents.
CPT entries are P(F_i = fail | pa_i) = c_i * lambda_i * m_{i,pa_i}, with c_i
chosen so that the prior marginal equals the published base rate lambda_i
exactly. Evidence is applied to stressor nodes only and posteriors are
obtained by exact marginalisation over parent configurations.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


class FailureNetwork:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.stressors: List[str] = cfg["stressors"]["names"]
        self.prior = dict(zip(self.stressors, cfg["stressors"]["prior_abnormal"]))
        self.alternatives: List[str] = cfg["alternatives"]
        self.base = dict(zip(self.alternatives, cfg["base_rates"]["per_demand"]))
        self.parents: Dict[str, List[str]] = {a: cfg["parents"][a] for a in self.alternatives}
        self.factors = cfg["multipliers"]["per_parent_factor"]
        self.tables = {a: self._calibrated_table(a) for a in self.alternatives}

    # ---- CPTs -------------------------------------------------------------
    def multiplier(self, alt: str, config: Dict[str, int]) -> float:
        m = 1.0
        for p in self.parents[alt]:
            if config[p] == 1:
                m *= self.factors[alt][p]
        return m

    def parent_configs(self, alt: str):
        pa = self.parents[alt]
        for states in itertools.product((0, 1), repeat=len(pa)):
            cfg = dict(zip(pa, states))
            prob = 1.0
            for p, s in cfg.items():
                prob *= self.prior[p] if s == 1 else 1 - self.prior[p]
            yield cfg, prob

    def _calibrated_table(self, alt: str) -> dict:
        lam = self.base[alt]
        expected_m = sum(self.multiplier(alt, c) * pr for c, pr in self.parent_configs(alt))
        c_i = 1.0 / expected_m
        rows = []
        for cfg, pr in self.parent_configs(alt):
            m = self.multiplier(alt, cfg)
            rows.append({"config": cfg, "prior_prob": pr, "m": m, "p_fail": c_i * lam * m})
        return {"c": c_i, "rows": rows}

    # ---- inference --------------------------------------------------------
    def posterior(self, alt: str, evidence: Dict[str, int] | None = None,
                  table_override: np.ndarray | None = None) -> float:
        """Exact P(F_alt = fail | evidence on stressors).

        table_override lets the Monte Carlo loop pass a sampled vector of CPT
        entries (same order as self.tables[alt]['rows'])."""
        evidence = evidence or {}
        rows = self.tables[alt]["rows"]
        num = den = 0.0
        for k, row in enumerate(rows):
            cfg = row["config"]
            if any(p in evidence and cfg[p] != evidence[p] for p in cfg):
                continue
            # probability of this parent configuration given evidence
            pr = 1.0
            for p, s in cfg.items():
                if p in evidence:
                    continue
                pr *= self.prior[p] if s == 1 else 1 - self.prior[p]
            p_fail = row["p_fail"] if table_override is None else table_override[k]
            num += pr * p_fail
            den += pr
        return num / den

    def posteriors(self, evidence=None, tables=None) -> np.ndarray:
        return np.array([self.posterior(a, evidence, None if tables is None else tables[a])
                         for a in self.alternatives])

    def describe(self) -> str:
        out = []
        for a in self.alternatives:
            t = self.tables[a]
            out.append(f"{a}: parents {self.parents[a]}, lambda={self.base[a]:.3e}, c={t['c']:.4f}")
            for r in t["rows"]:
                cfgs = ",".join(f"{k}={'abn' if v else 'nom'}" for k, v in r["config"].items())
                out.append(f"    [{cfgs}]  m={r['m']:.3f}  P(fail)={r['p_fail']:.5f}")
        return "\n".join(out)


def load_config(path=None) -> dict:
    path = Path(path) if path else Path(__file__).with_name("config.json")
    with open(path) as fh:
        return json.load(fh)


def scenarios(cfg: dict) -> Dict[str, Dict[str, int]]:
    """Scenario name -> evidence dict, skipping the 'source' annotation."""
    return {k: v for k, v in cfg["scenarios"].items() if k != "source"}
