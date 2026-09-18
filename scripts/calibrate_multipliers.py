"""Calibrate the RECONSTRUCTED per-parent stress factors in case_study/config.json
so that the exact BN posteriors under the three scenarios match the Monte
Carlo means reported in Supplement S3.4.

This script exists only because the elicited multipliers were marked
[[fill from repository]] in the supplement. Once the authors insert their
own parent sets and multipliers, this script is no longer needed.

Usage:  python scripts/calibrate_multipliers.py [--write]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from case_study.bn import FailureNetwork, load_config  # noqa: E402


def objective(theta, cfg, alt, keys, targets):
    cfg = json.loads(json.dumps(cfg))
    for k, v in zip(keys, np.exp(theta)):
        cfg["multipliers"]["per_parent_factor"][alt][k] = float(v)
    net = FailureNetwork(cfg)
    idx = cfg["alternatives"].index(alt)
    err = 0.0
    for scen, ev in cfg["scenarios"].items():
        if scen in ("Nominal", "source"):
            continue  # nominal is matched exactly by calibration of c_i
        post = net.posterior(alt, ev)
        tgt = targets[scen][idx]
        err += (np.log(post) - np.log(tgt)) ** 2
    # weak ridge toward equal T and M factors where both are parents (identifiability)
    if "T" in keys and "M" in keys:
        err += 1e-3 * (theta[keys.index("T")] - theta[keys.index("M")]) ** 2
    return err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="write calibrated factors back to config.json")
    args = ap.parse_args()
    cfg = load_config()
    targets = {k: v for k, v in cfg["reported_posterior_means"].items() if not k.startswith("source")}
    for alt in cfg["alternatives"]:
        keys = [p for p in cfg["parents"][alt] if p != "P"]  # P is never in evidence; unidentifiable, keep as set
        theta0 = np.log([cfg["multipliers"]["per_parent_factor"][alt][k] for k in keys])
        res = minimize(objective, theta0, args=(cfg, alt, keys, targets), method="Nelder-Mead",
                       options={"xatol": 1e-6, "fatol": 1e-10, "maxiter": 5000})
        for k, v in zip(keys, np.exp(res.x)):
            cfg["multipliers"]["per_parent_factor"][alt][k] = round(float(v), 3)
    net = FailureNetwork(cfg)
    print("scenario      " + "  ".join(f"{a[:8]:>9}" for a in cfg["alternatives"]))
    for scen, ev in cfg["scenarios"].items():
        if scen == "source":
            continue
        post = net.posteriors(ev)
        print(f"{scen:<12} " + "  ".join(f"{p:9.4f}" for p in post) + "   (exact)")
        print(f"{'':<12} " + "  ".join(f"{p:9.4f}" for p in targets[scen]) + "   (reported S3.4)")
    print("\ncalibrated per-parent factors:")
    print(json.dumps(cfg["multipliers"]["per_parent_factor"], indent=2))
    if args.write:
        with open(ROOT / "case_study" / "config.json", "w") as fh:
            json.dump(cfg, fh, indent=2)
        print("written to case_study/config.json")


if __name__ == "__main__":
    main()
