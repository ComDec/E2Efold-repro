#!/usr/bin/env bash
#
# Evaluate all 7 E2Efold benchmarks + pseudoknot-aware metrics on ArchiveII/iPKnot.
#
# Assumes:
#   - Data already preprocessed (scripts/preprocess_all.sh).
#   - Trained checkpoints in ./models_ckpt/ (from scripts/train_all.sh OR from
#     the Google Drive release package: UniRNA/baselines/e2efold/models_ckpt/).
#
# Usage:
#   bash scripts/eval_all.sh [GPU_ID]
#
# Wall time: ~30–60 min on a single H100 (dominated by ArchiveII / iPKnot CPU-bound
# postprocess + ~10 min of Python-loop pseudoknot scoring).
#
set -euo pipefail

GPU_ID="${1:-0}"
cd "$(dirname "$0")/.."
mkdir -p logs

set_gpu() {
    local cfg="$1"
    python3 -c "
import json
p = '$cfg'
with open(p) as f: d = json.load(f)
d['gpu'] = '$GPU_ID'
with open(p, 'w') as f: json.dump(d, f, indent=2)
"
}

echo "============================================================"
echo "Evaluating all benchmarks on GPU $GPU_ID"
echo "============================================================"

# ---- Stage 1 + Stage 3 experiments (use --test True on the e2e script) ----

echo; echo "[1/7] RNAStralign — Stage 3 e2e"
set_gpu experiment_rnastralign_repro/config.json
(cd experiment_rnastralign_repro && \
    python3 ../run_exp.py evR e2e_learning_stage3.py -c config.json --test True \
    > ../logs/rnastralign_eval.log 2>&1)

echo; echo "[2/7] Rivals — Stage 3 e2e on TestSetA (val) + TestSetB (test)"
set_gpu experiment_rivals/config.json
(cd experiment_rivals && \
    python3 ../run_exp.py evV evaluate_rivals.py -c config.json \
    > ../logs/rivals_eval.log 2>&1)

echo; echo "[3/7] bpRNA-1m (TR0→VL0→TS0) — Stage 3 e2e"
set_gpu experiment_bprna/config.json
(cd experiment_bprna && \
    python3 ../run_exp.py evB evaluate.py -c config.json \
    > ../logs/bprna_eval.log 2>&1)

# ---- Stage 1 only experiments (non-learned postprocess at eval time) -----

echo; echo "[4/7] UniRNA-SS — Stage 1 + non-learned PP"
set_gpu experiment_unirna_ss/config.json
(cd experiment_unirna_ss && \
    python3 ../run_exp.py evU evaluate.py -c config.json \
    > ../logs/unirna_ss_eval.log 2>&1)

echo; echo "[5/7] iPKnot — Stage 1 + non-learned PP"
set_gpu experiment_ipknot/config.json
(cd experiment_ipknot && \
    python3 ../run_exp.py evK evaluate.py -c config.json \
    > ../logs/ipknot_eval.log 2>&1)

echo; echo "[6/7] ArchiveII ≤600bp — Stage 1 + non-learned PP"
set_gpu experiment_archiveii_full/config.json
(cd experiment_archiveii_full && \
    python3 ../run_exp.py evA evaluate.py -c config.json \
    > ../logs/archiveii_eval.log 2>&1)

echo; echo "[7/7] bpRNA-1m-new — shared Stage 1 checkpoint with bpRNA-1m"
set_gpu experiment_bprna_s1/config.json
(cd experiment_bprna_s1 && \
    python3 ../run_exp.py evN evaluate.py -c config.json \
    > ../logs/bprna_new_eval.log 2>&1)

# ---- Pseudoknot-aware evaluation (CPU only) ----

echo; echo "[PK-1] UniRNA-SS pseudoknot metrics"
python3 evaluate_pseudoknot.py \
    --predictions experiment_unirna_ss/test_predictions.pkl \
    --dataset_name "UniRNA-SS (n=1041)" \
    | tee logs/unirna_ss_pseudoknot.log

echo; echo "[PK-2] ArchiveII pseudoknot metrics"
python3 evaluate_pseudoknot.py \
    --predictions experiment_archiveii_full/test_predictions.pkl \
    --dataset_name "ArchiveII (≤600bp, n=3911)" \
    | tee logs/archiveii_pseudoknot.log

echo; echo "[PK-3] iPKnot pseudoknot metrics"
python3 evaluate_pseudoknot.py \
    --predictions experiment_ipknot/test_predictions.pkl \
    --dataset_name "iPKnot (bpRNA-PK-TS0-1K)" \
    | tee logs/ipknot_pseudoknot.log

echo
echo "============================================================"
echo "Evaluation complete. See logs/*.log for full output."
echo "Summary tables are in README.md §3 and §4."
echo "============================================================"
