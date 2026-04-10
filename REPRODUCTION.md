# REPRODUCTION.md — E2Efold Reproduction Technical Notes

> **Per-experiment results, training curves, and reproduction commands live in [`Benchmark.md`](Benchmark.md).** This document contains the cross-cutting technical content: environment, code-change rationale, data-pipeline design, evaluation protocol, and known limitations. It is intentionally short — everything experiment-specific has been moved to `Benchmark.md` to avoid duplication.

---

## 1. Environment

| Component | Paper | This reproduction |
|---|:-:|:-:|
| Python | 3.7.3 | 3.13.12 |
| PyTorch | 1.2.0+cu92 | 2.8.0+cu128 |
| CUDA | 9.2 | 12.8 |
| GPU | not specified | NVIDIA H100 NVL (95 GB) |
| OS | not specified | Linux 6.8.0 |

**Deviation from the original `environment.yml`.** The paper's conda environment requires Python 3.7, which is obsolete and will not build on a modern CUDA driver. We use the system Python 3.13 and install the minimal dependencies with pip:

```bash
pip install torch                                  # 2.8.0+cu128
pip install munch torcheval setproctitle scikit-learn
pip install -e .                                   # install e2efold package
```

Two source patches are required to make the original code run on modern PyTorch/NumPy — §2.

---

## 2. Original-repo code modifications

Exactly **two files** of the upstream E2Efold source are modified. Verified via `git diff HEAD -- e2efold/`:

### 2.1 `e2efold/data_generator.py` — 1 line

```diff
-        self.pairs = np.array([instance[-1] for instance in self.data])
+        self.pairs = np.array([instance[-1] for instance in self.data], dtype=object)
```

NumPy ≥ 1.24 refuses to auto-coerce ragged arrays into `dtype=float`. Each sample has a different number of base pairs, so this list is ragged. `dtype=object` restores the original behavior exactly.

### 2.2 `e2efold/models.py` — `Lag_PP_mixed.__init__`, 6 lines

```diff
-    def __init__(self, steps, k, rho_mode='fix'):
+    def __init__(self, steps, k, rho_mode='fix', L=600):
         ...
-        self.rho_m = nn.Parameter(torch.randn(600, 600))
+        self.rho_m = nn.Parameter(torch.randn(L, L))
         ...
-        pos_j, pos_i = np.meshgrid(np.arange(1,600+1)/600.0,
-            np.arange(1,600+1)/600.0)
+        pos_j, pos_i = np.meshgrid(np.arange(1,L+1)/float(L),
+            np.arange(1,L+1)/float(L))
```

`Lag_PP_mixed` hardcodes a 600×600 learnable sparsity matrix `rho_m`. When the data is padded to a different length (768 for Rivals, 499 for bpRNA), this causes a shape-mismatch `RuntimeError` inside `update_rule`. The patch adds an `L` parameter defaulting to `600` — so calling `Lag_PP_mixed(steps, k)` behaves *identically* to the original. Only code that explicitly passes `L=<non-600>` sees new behavior.

**Caveat**: since `rho_m` is L×L learnable parameters, changing L changes the model's parameter count. This is by design — the original `Lag_PP_mixed` is L-specific; the paper's L-agnostic alternative is `Lag_PP_final` (scalar rho, no `L` dependence).

---

## 3. `experiment_rnastralign_repro/` — why a new directory

The original `experiment_rnastralign/` scripts cannot be used directly because:

1. **`upsampling=True`** calls `upsampling_data()`, which does `name.split('/')[2]` — this crashes on mxfold2-provided pickle IDs that have no `/` separator.
2. **`torch.load(path)`** fails on PyTorch 2.x with namedtuple pickles because the default changed to `weights_only=True`. We pass `weights_only=False` explicitly.
3. **Hardcoded test path** `../data/rnastralign_all/test_no_redundant_600.pickle` — that file is not in the upstream repository (requires SharePoint download) and we use an i.i.d. val split instead.

`experiment_rnastralign_repro/` is a copy of `experiment_rnastralign/` with only these three patches applied. Every other training detail (loss functions, optimizer, gradient accumulation, architecture instantiation) is byte-identical.

---

## 4. Data preprocessing pipeline

All conversion scripts under `data/preprocess_*.py` follow the same pure-format-conversion pattern:

```python
# Input:  list of dicts {id: str, seq: str, label: ndarray(N, N)}
# Output: list of RNA_SS_data namedtuples, pickled

RNA_SS_data = namedtuple('RNA_SS_data', 'seq ss_label length name pairs')

for item in source_data:
    seq_padded = one_hot_encode(item['seq'], maxlen)    # [maxlen, 4]
    ss_padded  = contact_to_ss_label(item['label'])     # [maxlen, 3] (unused at train time)
    pairs      = np.where(item['label'] > 0.5)          # both (i,j) and (j,i) directions
    result.append(RNA_SS_data(seq_padded, ss_padded, len(item['seq']), item['id'], pairs))
```

**No modification of the underlying data.** `pairs2map(contact_to_pairs(label))` exactly reproduces the original `label` — verified for every dataset.

Every preprocessing script reads its data directory from the `E2EFOLD_DATA_ROOT` environment variable (with a default matching the author's local setup). Reviewers set `E2EFOLD_DATA_ROOT=/path/to/ss_dataset` and run `scripts/preprocess_all.sh`.

### Expected layout under `E2EFOLD_DATA_ROOT`

```
$E2EFOLD_DATA_ROOT/
  rivals/                       # Rivals
    TrainSetA-addss.pkl, TestSetA-addss.pkl, TestSetB-addss.pkl
  mxfold2/                      # bpRNA, bpRNA-new, RNAStralign, ArchiveII
    TR0-canonicals.pkl, VL0-canonicals.pkl, TS0-canonicals.pkl
    RNAStrAlign600-train.pkl, archiveII.pkl, bpRNAnew.pkl
  all_data_1024_0.75/           # UniRNA-SS
    train.pkl, valid.pkl, test.pkl
  ipkont/                       # iPKnot
    bpRNA-TR0.pkl, bpRNA-PK-TS0-1K.pkl
```

### Output layout under `./data/`

```
./data/
  rnastralign_all_600/{train,val,test}.pickle
  rivals/{train,val,test}.pickle                 # train=TrainSetA, val=TestSetA, test=TestSetB
  bprna_tr0/{train,val,test,test_new}.pickle     # test=TS0, test_new=bpRNAnew
  unirna_ss/{train,val,test}.pickle              # predefined splits
  ipknot/{train,val,test}.pickle                 # val == test (same source file)
  archiveii_full/{train,val,test}.pickle         # train=90% RNAStrAlign600-train, test=archiveII≤600bp
```

---

## 5. Training protocol

### Architecture

- **Score network**: `ContactAttention_simple_fix_PE` (`d=10`, Transformer encoder + Conv1d + fixed positional encoding). Input: one-hot sequence + `L×L` PE. Output: raw score matrix `L×L`.
- **Post-processing (Stage 1 only evaluations)**: `e2efold.postprocess.postprocess(logits, seq, lr_min=0.01, lr_max=0.1, num_itr=50, rho=1.0, with_l1=True)` — the non-learned augmented Lagrangian solver.
- **Post-processing (Stage 1+3 evaluations)**: `Lag_PP_mixed(pp_steps=20, k=1, rho_per_position='matrix', L=<seq_len>)` — the learned augmented Lagrangian solver with per-position `rho_m`.
- **End-to-end wrapper**: `RNA_SS_e2e(contact_net, lag_pp_net)`.

### Two-stage training (identical to the paper)

1. **Stage 1** (50 epochs): pre-train the score network with `BCEWithLogitsLoss(pos_weight=300)` on the masked `pred_contacts * contact_masks` logits. Adam optimizer at default `lr=0.001`, `batch_size=20`. Evaluation every 5 epochs on the val set; the final-epoch checkpoint is saved to `../models_ckpt/supervised_<model_type>_<data_type>_d<d>_l3.pt`.

2. **Stage 3** (10 epochs): end-to-end joint training with `loss = loss_u (BCE) + loss_a (F1 on contact map)`, gradient accumulation over 30 forward passes, `batch_size=8`. The `F1` loss is differentiable (`e2efold.common.utils.f1_loss`). Evaluation every epoch; final-epoch checkpoint saved to `../models_ckpt/e2e_<model_type>_<pp_type>_d<d>_<data_type>_<pp_loss>_position_<rho_per_position>.pt`.

### What `upsampling=False` means

The original `RNASSDataGenerator.upsampling_data()` rebalances the 8 RNAStralign RNA families using hardcoded indices that require `/`-separated path IDs. This function is specific to RNAStralign and crashes on every other dataset. Setting `upsampling=False` means the training data distribution is used as-is — the only legitimate option for non-RNAStralign benchmarks.

---

## 6. Evaluation protocol

### DeepRNA metrics (primary — matches `secondary_structure_metircs`)

```python
from torcheval.metrics.functional import (
    binary_precision, binary_f1_score, binary_auroc, binary_auprc,
)

for each sample:
    pred = model_output[:seq_len, :seq_len].flatten()    # continuous [0, ~1]
    target = ground_truth[:seq_len, :seq_len].flatten()  # binary {0, 1}
    precision = binary_precision(pred, target, threshold=0.5)
    f1        = binary_f1_score(pred, target, threshold=0.5)
    auroc     = binary_auroc(pred, target)
    auprc     = binary_auprc(pred, target)

final = macro mean over all samples
```

### E2Efold native metrics (secondary — for continuity with the paper)

```python
from e2efold.common.utils import evaluate_exact, evaluate_shifted

binary_pred = (model_output > 0.5).float()
precision,  recall,  f1   = evaluate_exact(binary_pred, ground_truth)      # exact position match
precision_s, recall_s, f1_s = evaluate_shifted(binary_pred, ground_truth)  # ±1-position tolerance
```

### Pseudoknot metrics (ArchiveII + iPKnot only)

Uses `deeprna.metrics.pseudoknot.evaluate_structure_metrics`, called unmodified on the saved `test_predictions.pkl` (no re-inference). See [`Benchmark.md`](Benchmark.md) §8 for the detailed setup and [`evaluate_pseudoknot.py`](evaluate_pseudoknot.py) for the script.

### Inference pipeline (verified identical to the paper)

```
input → one_hot_encode → contact_net(PE, seq, zeros)
     → (Stage 3) lag_pp_net(logits, seq)
       (Stage 1) postprocess(logits, seq, 0.01, 0.1, 50, 1.0, True)
     → threshold(0.5)
```

Symmetry is enforced internally by `contact_a()`. Masking is implicit via the AU/CG/GU constraint matrix built inside the score network.

---

## 7. Known issues and limitations

1. **Diverse datasets are beyond the model capacity.** E2Efold was designed for the 8 RNAStralign families and uses a Transformer encoder with a narrow `d=10` hidden dimension and an L×L learnable sparsity matrix. On bpRNA, UniRNA-SS, and iPKnot (which cover many RNA families), the Stage-1 score network saturates at val F1 ≈ 0.1–0.15 and Stage-3 fine-tuning diverges. These are the Stage-1-only experiments in the reproduction. See [`Benchmark.md`](Benchmark.md) §3–5 for per-dataset analysis.

2. **`Lag_PP_mixed` is L-dependent.** The learnable `rho_m` matrix is L×L. Changing L changes model capacity. For data padded to length ≠ 600, we have to either (a) change L and accept a different model size (what we do for Rivals / bpRNA), or (b) switch to `Lag_PP_final` (scalar rho, L-agnostic). We report (a) as the primary result and (b) as an ablation.

3. **Training instability.** With longer Stage-3 training (> 30 epochs) on bpRNA, the joint loss can diverge to NaN. The optimal Stage-3 length on bpRNA is ~10–25 epochs; going longer does not help and eventually breaks. We use the paper's 10-epoch default everywhere.

4. **E2Efold cannot predict pseudoknots.** The augmented Lagrangian enforces at-most-one-partner-per-position but has no crossing-pair term. On iPKnot and the PK subset of ArchiveII, `pk_f1 ≈ 0.004–0.009` is essentially a floor. See [`Benchmark.md`](Benchmark.md) §8.

5. **Checkpoint path collision across PP variants.** When running `Lag_PP_mixed` and `Lag_PP_final` in parallel on the same dataset, both Stage-1 checkpoints go to the same path (`supervised_<data>_d10_l3.pt`). This is functionally correct — both variants use the *same* Stage-1 score network — but operationally you cannot run the two variants' Stage-1 training concurrently (the second will overwrite the first). Stage-3 paths include `pp_type` and do not collide.

6. **ArchiveII is restricted to ≤600 bp.** The `ContactAttention_simple_fix_PE` positional encoding is created at `L=600` in the paper's recipe and cannot process longer sequences without retraining. 55 of 3,966 ArchiveII samples are excluded. This is an architectural constraint of E2Efold, not a selection bias of this reproduction. See [`Benchmark.md`](Benchmark.md) §6.
