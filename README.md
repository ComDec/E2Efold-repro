# E2Efold Reproduction (Reviewer Package)

Reproduction of **E2Efold** ([Chen et al., ICLR 2020](https://openreview.net/pdf?id=S1eALyrYDH), [original repo](https://github.com/ml4bio/e2efold)) for RNA secondary structure prediction on **7 standard benchmarks**, plus pseudoknot-aware evaluation on ArchiveII and iPKnot.

> This README is the **reviewer entry point** — follow it top-to-bottom to fully reproduce every reported number. Detailed docs (don't read these first):
> - [`Benchmark.md`](Benchmark.md) — per-benchmark full sections (configuration, training loss, data notes, legitimacy review).
> - [`REPRODUCTION.md`](REPRODUCTION.md) — per-experiment technical notes, code modifications, data-pipeline rationale.
> - [`CHANGE_LOG.md`](CHANGE_LOG.md) — chronological log of every environment/code/experiment action.

---

## 0. TL;DR

```bash
# 1. Environment (~3 min)
pip install torch munch torcheval setproctitle scikit-learn
pip install -e .

# 2. Download datasets + checkpoints from Google Drive (see §2)
rclone copy gdrive:UniRNA/ss_dataset/              ./ss_dataset/       -P
rclone copy gdrive:UniRNA/baselines/e2efold/models_ckpt/ ./models_ckpt/ -P

# 3. Preprocess all 7 datasets (~5 min, CPU only)
bash scripts/preprocess_all.sh ./ss_dataset

# 4. Verify all reported metrics using the provided checkpoints (~30–60 min on 1 GPU)
bash scripts/eval_all.sh 0     # single-GPU eval, GPU id = 0
```

Expected output: 7 standard F1 values matching §3 and 2 pseudoknot F1 rows matching §4 to within CUDA rounding.

---

## 1. Environment

Required: NVIDIA GPU with modern CUDA driver (tested on H100 NVL, CUDA 12.8), Python ≥ 3.10, ~20 GB disk for preprocessed data + ~25 MB for checkpoints.

```bash
# Minimal install (what we actually tested)
pip install torch                              # tested: 2.8.0+cu128
pip install munch torcheval setproctitle scikit-learn
pip install -e .                               # install e2efold package
```

**Why not `conda env create -f environment.yml`?** The bundled `environment.yml` is the original paper's Python 3.7 + PyTorch 1.2.0 + CUDA 9.2 setup, which is obsolete and will not build on a modern driver. Use the pip install above instead. The only patches required for modern PyTorch/NumPy are documented in §6.

Verify:
```bash
python3 -c "import torch, e2efold; print(torch.__version__, torch.cuda.is_available())"
# Expected:  2.8.0+cu128 True
```

---

## 2. Download datasets and checkpoints

Both are hosted on the author's Google Drive:
- **Datasets**: `gdrive:UniRNA/ss_dataset/` — ~4 GB total
- **Checkpoints**: `gdrive:UniRNA/baselines/e2efold/models_ckpt/` — ~21 MB (7 `.pt` files)

### Option A: via rclone (recommended)

Requires [rclone](https://rclone.org/) configured with a Google Drive remote. Replace `gdrive:` with your remote name:

```bash
rclone copy gdrive:UniRNA/ss_dataset/                    ./ss_dataset/ -P
rclone copy gdrive:UniRNA/baselines/e2efold/models_ckpt/ ./models_ckpt/ -P
```

### Option B: manual download

Visit `<share-link from authors>`, download both directories, unpack into `./ss_dataset/` and `./models_ckpt/`.

### Expected layout after download

```
./ss_dataset/                             # raw source data, E2EFOLD_DATA_ROOT
  all_data_1024_0.75/                     # UniRNA-SS
    train.pkl, valid.pkl, test.pkl
  ipkont/                                 # iPKnot
    bpRNA-TR0.pkl, bpRNA-PK-TS0-1K.pkl
  mxfold2/                                # ArchiveII, bpRNA-1m, bpRNA-1m-new, RNAStralign
    TR0-canonicals.pkl, VL0-canonicals.pkl, TS0-canonicals.pkl
    RNAStrAlign600-train.pkl, archiveII.pkl, bpRNAnew.pkl
  rivals/                                 # Rivals
    TrainSetA-addss.pkl, TestSetA-addss.pkl, TestSetB-addss.pkl
./models_ckpt/                            # 7 final checkpoints
  e2e_att_simple_fix_mixed_s20_d10_rnastralign_all_600_f1_position_matrix.pt   # RNAStralign
  e2e_att_simple_fix_mixed_s20_d10_rivals_f1_position_matrix.pt                # Rivals
  e2e_att_simple_fix_mixed_s20_d10_bprna_tr0_f1_position_matrix.pt             # bpRNA-1m
  supervised_att_simple_fix_bprna_tr0_d10_l3.pt                                # bpRNA-1m-new (Stage 1)
  supervised_att_simple_fix_unirna_ss_d10_l3.pt                                # UniRNA-SS
  supervised_att_simple_fix_ipknot_d10_l3.pt                                   # iPKnot
  supervised_att_simple_fix_archiveii_full_d10_l3.pt                           # ArchiveII
```

Then preprocess the raw pickles into E2Efold's `RNA_SS_data` namedtuple format:

```bash
bash scripts/preprocess_all.sh ./ss_dataset
```

This populates `./data/{rnastralign_all_600,rivals,bprna_tr0,unirna_ss,ipknot,archiveii_full}/{train,val,test}.pickle` without modifying any source data (pure format conversion — see [`REPRODUCTION.md`](REPRODUCTION.md) §4).

---

## 3. Verify reported metrics (standard F1)

```bash
bash scripts/eval_all.sh 0     # single GPU id
```

Wall time: ~30–60 min on one H100 NVL. The script evaluates each benchmark against the corresponding checkpoint in `./models_ckpt/` and writes per-experiment logs to `logs/*_eval.log`.

Expected output (metrics via `torcheval` binary_*, threshold=0.5, averaged per sample):

| # | Benchmark | Train | Test | Checkpoint | Precision | F1 | AUROC | AUPRC |
|---|---|---|---|---|:---:|:---:|:---:|:---:|
| 1 | RNAStralign (anchor) | 80% of RNAStrAlign600-train (16,738) | 10% val (2,092) | `e2e_..._rnastralign_all_600_...pt` | 0.809 | **0.834** | — | — |
| 2a | Rivals TestSetA (val) | TrainSetA-addss (3,166) | TestSetA-addss (592) | `e2e_..._rivals_...pt` | 0.4012 | **0.4206** | 0.9412 | 0.3769 |
| 2b | Rivals TestSetB (test) | TrainSetA-addss (3,166) | TestSetB-addss (430) | `e2e_..._rivals_...pt` | 0.0454 | **0.0535** | 0.8339 | 0.0406 |
| 3 | bpRNA-1m TS0 | TR0-canonicals (10,814) | TS0-canonicals (1,305) | `e2e_..._bprna_tr0_...pt` | 0.1684 | **0.2318** | 0.9393 | 0.1824 |
| 4 | UniRNA-SS | train (8,323) | test (1,041) | `supervised_..._unirna_ss_...pt` | 0.1546 | **0.1092** | 0.6332 | 0.0919 |
| 5 | iPKnot | bpRNA-TR0 (10,814) | bpRNA-PK-TS0-1K (2,914) | `supervised_..._ipknot_...pt` | 0.1843 | **0.1246** | 0.6760 | 0.1244 |
| 6 | ArchiveII (≤600bp) | 90% RNAStrAlign600-train (18,830) | archiveII ≤600bp (3,911) | `supervised_..._archiveii_full_...pt` | 0.4914 | **0.4463** | 0.8326 | 0.4579 |
| 7 | bpRNA-1m-new | TR0-canonicals (10,814) | bpRNAnew (5,401) | `supervised_..._bprna_tr0_...pt` ⭐ | 0.0613 | **0.0371** | 0.5820 | 0.0366 |

⭐ **bpRNA-1m-new reuses the Stage 1 score-net checkpoint from bpRNA-1m** (same TR0 training set). There is no additional training. See [`Benchmark.md`](Benchmark.md) §7.

**ArchiveII is restricted to sequences ≤ 600 bp** (3,911 of 3,966 samples). This is an architectural constraint, not a selection bias: the `ContactAttention_simple_fix_PE` score network has a hardcoded `L=600` positional encoding. Same Stage-1 protocol is used as the other L-fixed experiments. See [`Benchmark.md`](Benchmark.md) §6.

**Stage 1+3 vs Stage 1 only**: Benchmarks 1–3 use the full two-stage training (Stage 1 score network + Stage 3 end-to-end with the learned `Lag_PP_mixed` solver). Benchmarks 4–7 use only Stage 1 and evaluate with the non-learned `postprocess()` augmented-Lagrangian solver. This matches the configuration the original authors recommend for datasets where the score network does not converge inside 50 Stage-1 epochs. See [`REPRODUCTION.md`](REPRODUCTION.md) §7.

Tolerance: CUDA floating-point operations are not bit-deterministic. Observed max drift on a fresh re-evaluation is ~0.001 on per-sample F1. If your F1 differs by more than 0.01 on any benchmark, please file an issue with the full `logs/*_eval.log`.

---

## 4. Pseudoknot-aware evaluation (ArchiveII + iPKnot)

Additional metrics from the DeepRNA pseudoknot module (`deeprna.metrics.pseudoknot`, used unmodified):

- **score** — overall F1 via `sklearn.f1_score` on flattened binarized contact maps (threshold=0.5). Should equal the standard F1 in §3 within rounding.
- **score_pk** — same F1 restricted to samples that contain ≥1 crossing base pair (pseudoknot-containing samples).
- **pk_sen / pk_ppv / pk_f1** — sensitivity / PPV / F1 of the prediction restricted to crossing base pairs only. A crossing pair is any pair of base pairs `(i,j)` and `(k,l)` with `i < k < j < l`.

The evaluation is **CPU only** and runs directly on the saved `test_predictions.pkl` files produced by §3 (no re-inference):

```bash
# DeepRNA path can be overridden via DEEPRNA_PATH env var; default matches author setup.
export DEEPRNA_PATH=/home/xiwang/project/develop/deeprna

python3 evaluate_pseudoknot.py --predictions experiment_unirna_ss/test_predictions.pkl --dataset_name "UniRNA-SS"
python3 evaluate_pseudoknot.py --predictions experiment_archiveii_full/test_predictions.pkl --dataset_name "ArchiveII"
python3 evaluate_pseudoknot.py --predictions experiment_ipknot/test_predictions.pkl --dataset_name "iPKnot"
```

Wall time: UniRNA-SS ~1 min, ArchiveII ~12 min, iPKnot ~10 min (O(L³) Python loop).

Expected output:

| Benchmark | n_total | n_pk | score (F1) | score_pk | pk_sen | pk_ppv | **pk_f1** |
|---|---|---|:---:|:---:|:---:|:---:|:---:|
| UniRNA-SS | 1041 | 164 (15.8%) | 0.1092 | 0.0453 | 0.0012 | 0.0019 | **0.0013** |
| ArchiveII (≤600bp) | 3911 | 1039 (26.6%) | 0.4463 | 0.1061 | 0.0041 | 0.0076 | **0.0038** |
| iPKnot (bpRNA-PK-TS0-1K) | 2914 | 353 (12.1%) | 0.1246 | 0.0577 | 0.0069 | 0.0125 | **0.0087** |

**Interpretation**: E2Efold's Lagrangian post-processing has no explicit crossing-pair representation — each position is constrained to pair with at most one other, but no term encodes or encourages crossings. `pk_f1` ranges from 0.001 to 0.009 — essentially a floor on all three datasets (see [`Benchmark.md`](Benchmark.md) §8 for the full discussion including comparison to UFold).

---

## 5. Training from scratch (optional)

```bash
bash scripts/train_all.sh 0    # single GPU id
```

Trains all 7 benchmarks sequentially on one GPU. Per-benchmark wall time on an H100 NVL:

| Benchmark | Stages | Training set | Wall time |
|---|---|---|---|
| RNAStralign | Stage 1 + Stage 3 | 16,738 | ~3.5 h |
| Rivals | Stage 1 + Stage 3 | 3,166 | ~50 min |
| bpRNA-1m | Stage 1 + Stage 3 | 10,814 | ~3 h |
| UniRNA-SS | Stage 1 only | 8,323 | ~2 h |
| iPKnot | Stage 1 only | 10,814 | ~2.5 h |
| ArchiveII ≤600bp | Stage 1 only | 18,830 | ~4 h |
| bpRNA-1m-new | (reuses bpRNA-1m Stage 1) | — | 0 |

Total: ~15 h on a single H100. All hyperparameters match E2Efold defaults — see [`Benchmark.md`](Benchmark.md) per-experiment sections or [`REPRODUCTION.md`](REPRODUCTION.md) §5.

After training, re-run §3 against your own checkpoints:
```bash
bash scripts/eval_all.sh 0
```

---

## 6. Code changes vs the original E2Efold repository

Only **2 files** of the original source are modified (the rest is additive). See [`REPRODUCTION.md`](REPRODUCTION.md) §2 for the full diff and rationale.

| File | Change | Reason |
|---|---|---|
| `e2efold/data_generator.py` | +1 line: `np.array(..., dtype=object)` | NumPy ≥ 1.24 refuses implicit ragged arrays |
| `e2efold/models.py` (`Lag_PP_mixed.__init__`) | +1 `L=600` param; hardcoded `600` replaced with `L` | Allow datasets padded to a length other than 600 |

With `L=600` (the new default), `Lag_PP_mixed(steps, k)` behaves **identically** to the original.

All new files are additive (new experiments, preprocessing scripts, reviewer scripts, and the pseudoknot evaluator). `git diff HEAD -- e2efold/` shows exactly these two patches.

---

## 7. Repository layout

```
e2efold/                              # Original source (2 minimal patches)
  models.py                           # Lag_PP_mixed: L parameterization
  data_generator.py                   # NumPy ragged-array fix
  ...                                 # (all other files unchanged from upstream)
data/
  preprocess_rivals.py                # Format conversion scripts (all read E2EFOLD_DATA_ROOT)
  preprocess_bprna.py
  preprocess_unirna_ss.py
  preprocess_ipknot.py
  preprocess_archiveii_full.py
  preprocess_bprna_new.py
  preprocess_rnastralign_from_mxfold2.py
experiment_rnastralign_repro/         # Benchmark 1 — RNAStralign (anchor)
experiment_rivals/                    # Benchmark 2 — Rivals
experiment_bprna/                     # Benchmark 3 — bpRNA-1m
experiment_unirna_ss/                 # Benchmark 4 — UniRNA-SS
experiment_ipknot/                    # Benchmark 5 — iPKnot
experiment_archiveii_full/            # Benchmark 6 — ArchiveII
experiment_bprna_s1/                  # Benchmark 7 — bpRNA-1m-new
scripts/
  preprocess_all.sh                   # One-command preprocessing
  train_all.sh                        # One-command full training (~15 h)
  eval_all.sh                         # One-command full evaluation (~1 h)
run_exp.py                            # setproctitle wrapper (ps aux disguise)
evaluate_pseudoknot.py                # CPU-only PK metric driver (§4)
compare_pp_effect.py                  # Ablation: postprocess on/off (Benchmark.md §9)
models_ckpt/                          # Trained checkpoints — gitignored; download from GDrive
logs/                                 # Training + evaluation logs — gitignored
README.md                             # This file
Benchmark.md                          # Per-experiment full details
REPRODUCTION.md                       # Technical notes, code-change rationale
CHANGE_LOG.md                         # Chronological action log
CLAUDE.md                             # Project instructions (for Claude-assisted maintenance)
```

---

## 8. Acknowledgments

Based on E2Efold:

> Chen, X., Li, Y., Umarov, R., Gao, X., Song, L. "RNA Secondary Structure Prediction By Learning Unrolled Algorithms." *International Conference on Learning Representations* (2020).

Pseudoknot metrics from the DeepRNA project (`deeprna.metrics.pseudoknot`).

Datasets redistributed in `UniRNA/ss_dataset/` were sourced from:
- RNAStralign / ArchiveII (via MXfold2-provided pickles)
- bpRNA-1m / TR0 / VL0 / TS0 / bpRNAnew (via MXfold2-provided pickles)
- Rivals (Rivas lab, via `-addss.pkl` variants)
- UniRNA-SS (internal release `all_data_1024_0.75`)
- iPKnot benchmark (`bpRNA-PK-TS0-1K.pkl`)
