#!/usr/bin/env bash
# Regulated Generative Analytics (RGA)
# Authors: Vivek Kumar Singh, Fnu Rahul, Vivek Bhatt, Sandeep Anand
# Licence: MIT (see LICENSE)
# Tests the whole pipeline on SYNTHETIC data with a MOCK model in a temporary copy. Produces no real results.
set -eo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"; T="$(mktemp -d)"; cp -r "$HERE" "$T/rga_v2"; cd "$T/rga_v2"
rm -f artifacts/*.csv artifacts/*.json results/* runs/*
python scripts/make_synthetic_data.py "$T/syn.csv" 30000
export RGA_DATA="$T/syn.csv" RGA_MODELS="$T/models_mock.json"
echo '[{"name":"mockA","provider":"mock","model_id":"mock","noise":1.0},{"name":"mockB","provider":"mock","model_id":"mock","noise":0.6}]' > "$T/models_mock.json"
python scripts/s00_profile.py && python scripts/s06_credibility_eval.py >/dev/null && python scripts/s01_build_benchmark.py | tail -5
python scripts/s03_collect.py --models mockA mockB --mock | tail -3 && python scripts/s03_collect.py --models mockA mockB --mock --b1b | tail -2 && python scripts/s04_score.py | tail -3 && python scripts/s05_analyze.py | tail -3 && python scripts/s07_validate.py | grep -E "^(PASS|FAIL|SUMMARY)|Traceback|Error" && python scripts/s08_unsure_rule.py | head -8
test -s results/scored.csv && test -s results/primary_contrasts.csv && test -s runs/raw_mockA.jsonl || { echo "SMOKE TEST FAILED: expected output files are missing"; exit 1; }
echo "SMOKE TEST PASSED (temp dir $T). Delete it when done."
