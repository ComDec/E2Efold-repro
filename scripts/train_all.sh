#!/usr/bin/env bash
#
# Train all 7 E2Efold benchmarks sequentially on a single GPU.
#
# Benchmarks:
#   1. RNAStralign (anchor)   Stage 1 + Stage 3
#   2. Rivals                 Stage 1 + Stage 3
#   3. bpRNA-1m (TR0/VL0/TS0) Stage 1 + Stage 3
#   4. UniRNA-SS              Stage 1 only
#   5. iPKnot                 Stage 1 only
#   6. ArchiveII (≤600bp)     Stage 1 only
#   7. bpRNA-1m-new           No extra training (reuses #3 Stage 1 checkpoint)
#
# Wall time on one H100 NVL: ~15 h total (Stage 1 dominated by ArchiveII + RNAStralign).
#
# Usage:
#   bash scripts/train_all.sh [GPU_ID]
#
# Assumes data has already been preprocessed (run scripts/preprocess_all.sh first).
#
set -euo pipefail

GPU_ID="${1:-0}"
cd "$(dirname "$0")/.."
mkdir -p logs

set_gpu() {
    # Patch the "gpu" field of a config.json in place. Reverts to 0 if GPU_ID is 0.
    local cfg="$1"
    python3 -c "
import json, sys
p = '$cfg'
with open(p) as f: d = json.load(f)
d['gpu'] = '$GPU_ID'
with open(p, 'w') as f: json.dump(d, f, indent=2)
"
}

echo "============================================================"
echo "Training all benchmarks on GPU $GPU_ID"
echo "============================================================"

# -------------------------- Stage 1 + Stage 3 -------------------------------

echo; echo "[1/7] RNAStralign (Stage 1 + Stage 3)"
set_gpu experiment_rnastralign_repro/config.json
(cd experiment_rnastralign_repro && \
    python3 ../run_exp.py expR1 e2e_learning_stage1.py -c config.json > ../logs/rnastralign_stage1.log 2>&1 && \
    python3 ../run_exp.py expR3 e2e_learning_stage3.py -c config.json > ../logs/rnastralign_stage3.log 2>&1)

echo; echo "[2/7] Rivals (Stage 1 + Stage 3, Lag_PP_mixed)"
set_gpu experiment_rivals/config.json
(cd experiment_rivals && \
    python3 ../run_exp.py expV1 e2e_learning_stage1.py -c config.json > ../logs/rivals_stage1.log 2>&1 && \
    python3 ../run_exp.py expV3 e2e_learning_stage3.py -c config.json > ../logs/rivals_stage3.log 2>&1)

echo; echo "[3/7] bpRNA-1m (Stage 1 + Stage 3, Lag_PP_mixed)"
set_gpu experiment_bprna/config.json
(cd experiment_bprna && \
    python3 ../run_exp.py expB1 e2e_learning_stage1.py -c config.json > ../logs/bprna_stage1.log 2>&1 && \
    python3 ../run_exp.py expB3 e2e_learning_stage3.py -c config.json > ../logs/bprna_stage3.log 2>&1)

# -------------------------- Stage 1 only ------------------------------------

echo; echo "[4/7] UniRNA-SS (Stage 1 only)"
set_gpu experiment_unirna_ss/config.json
(cd experiment_unirna_ss && \
    python3 ../run_exp.py expU1 e2e_learning_stage1.py -c config.json > ../logs/unirna_ss_stage1.log 2>&1)

echo; echo "[5/7] iPKnot (Stage 1 only)"
set_gpu experiment_ipknot/config.json
(cd experiment_ipknot && \
    python3 ../run_exp.py expK1 e2e_learning_stage1.py -c config.json > ../logs/ipknot_stage1.log 2>&1)

echo; echo "[6/7] ArchiveII ≤600bp (Stage 1 only)"
set_gpu experiment_archiveii_full/config.json
(cd experiment_archiveii_full && \
    python3 ../run_exp.py expA1 e2e_learning_stage1.py -c config.json > ../logs/archiveii_stage1.log 2>&1)

echo; echo "[7/7] bpRNA-1m-new — no additional training (reuses bpRNA-1m Stage 1 checkpoint)"

echo
echo "============================================================"
echo "All training complete. Checkpoints in ./models_ckpt/"
echo "Run scripts/eval_all.sh to evaluate all benchmarks."
echo "============================================================"
