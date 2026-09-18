#!/usr/bin/env bash
# Regenerate every table and figure. Run from the repository root.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "== tests";            python tests/test_audit.py && python tests/test_pipeline.py
echo "== Table 1";          python scripts/worked_example.py
echo "== simulation";       python scripts/simulation.py
echo "== corpus";           python scripts/run_audit.py
echo "== corpus check";     python scripts/validate_corpus.py || true
echo "== case study";       python scripts/run_case_study.py
echo "done: see outputs/"
