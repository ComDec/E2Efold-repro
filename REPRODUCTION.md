# REPRODUCTION.md — E2Efold Reproduction Technical Details

## 1. Environment

| Component | Original (Paper) | This Reproduction |
|-----------|:-:|:-:|
| Python | 3.7.3 | 3.13.12 |
| PyTorch | 1.2.0+cu92 | 2.8.0+cu128 |
| CUDA | 9.2 | 12.8 |
| GPU | Not specified | NVIDIA H100 NVL (95 GB) |
| NumPy | 1.16.4 | (system default) |
| OS | Not specified | Linux 6.8.0-101-generic |

**Deviation**: Did not use `conda env create -f environment.yml` (requires obsolete Python 3.7). Used existing Python 3.13 with `pip install munch torcheval` for missing dependencies.

## 2. Original Repo Code Modifications

**Only 2 files modified** (verified via `git diff HEAD --stat`):

### 2.1 `e2efold/data_generator.py` (1 line)

```diff
-        self.pairs = np.array([instance[-1] for instance in self.data])
+        self.pairs = np.array([instance[-1] for instance in self.data], dtype=object)
```

**Why**: NumPy ≥1.24 raises `ValueError` on implicit ragged arrays. Each sample has a different number of base pairs, making this array ragged. `dtype=object` restores original behavior.

### 2.2 `e2efold/models.py` (6 lines in `Lag_PP_mixed.__init__`)

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

**Why**: `rho_m` is hardcoded to 600×600. When data is padded to a different length (e.g., 768 for Rivals, 499 for bpRNA), a shape mismatch RuntimeError occurs. Default `L=600` preserves backward compatibility — calling `Lag_PP_mixed(steps, k)` behaves identically to the original.

## 3. `experiment_rnastralign_repro/` — Why New Scripts

The original `experiment_rnastralign/` scripts cannot be used directly because:

1. **`upsampling=True`** calls `upsampling_data()` which does `name.split('/')[2]` — crashes on mxfold2 data IDs (no `/` separators)
2. **`torch.load(path)`** fails on PyTorch 2.x (default changed to `weights_only=True`)
3. **Hardcoded test path** `../data/rnastralign_all/test_no_redundant_600` — data unavailable (requires SharePoint download)

The repro scripts are minimal copies with only these 3 adaptations. All training logic (loss functions, optimizer, gradient accumulation, architecture selection) is identical.

## 4. Data Preprocessing Pipeline

All conversion scripts follow the same pattern — pure format conversion, no data modification:

```python
# Input: list of dicts {id: str, seq: str, label: ndarray(N,N)}
# Output: list of RNA_SS_data namedtuples, pickled

RNA_SS_data = namedtuple('RNA_SS_data', 'seq ss_label length name pairs')

for item in source_data:
    seq_padded  = one_hot_encode(item['seq'], maxlen)    # [maxlen, 4]
    ss_padded   = contact_to_ss_label(item['label'])     # [maxlen, 3] (unused in training)
    pairs       = np.where(item['label'] > 0.5)          # both (i,j) and (j,i)
    → RNA_SS_data(seq_padded, ss_padded, len(item['seq']), item['id'], pairs)
```

Key: `pairs2map(contact_to_pairs(label))` exactly reproduces the original `label` — verified for all datasets.

## 5. Training Details

### Architecture
- Score network: `ContactAttention_simple_fix_PE` (d=10, Transformer encoder + Conv1d + fixed positional encoding)
- Post-processing: `Lag_PP_mixed` (20-step unrolled augmented Lagrangian with learned per-position rho) or `Lag_PP_final` (scalar rho)
- End-to-end wrapper: `RNA_SS_e2e`

### Training Procedure (identical to paper)
1. **Stage 1** (50 epochs): Pre-train score network with `BCEWithLogitsLoss(pos_weight=300)`, Adam optimizer
2. **Stage 3** (10 epochs): End-to-end joint training with `loss = loss_u(BCE) + loss_a(F1)`, gradient accumulation every 30 steps

### What `upsampling=False` Means
The original `upsampling_data()` in `data_generator.py` rebalances 8 RNAStralign RNA families with hardcoded indices. This function crashes on non-RNAStralign data because it assumes `/`-separated file paths. Disabling upsampling means the training data distribution is used as-is. This is the only option for non-RNAStralign datasets.

## 6. Evaluation Details

### DeepRNA Metrics (primary, matches `secondary_structure_metircs`)

```python
from torcheval.metrics.functional import binary_precision, binary_f1_score, binary_auroc, binary_auprc

for each sample:
    pred = model_output[:seq_len, :seq_len].flatten()   # continuous [0,1]
    target = ground_truth[:seq_len, :seq_len].flatten()  # binary {0,1}
    precision = binary_precision(pred, target, threshold=0.5)
    f1        = binary_f1_score(pred, target, threshold=0.5)
    auroc     = binary_auroc(pred, target)
    auprc     = binary_auprc(pred, target)

final = mean over all samples
```

### E2Efold Native Metrics (secondary, for comparison with paper)

```python
from e2efold.common.utils import evaluate_exact, evaluate_shifted

binary_pred = (model_output > 0.5).float()
precision, recall, f1 = evaluate_exact(binary_pred, ground_truth)     # exact match
precision_s, recall_s, f1_s = evaluate_shifted(binary_pred, ground_truth)  # 1-position shift allowed
```

### Inference Pipeline (verified identical to paper)
```
input_sequence → one_hot_encode → contact_net(PE, seq, zeros) → lag_pp_net(logits, seq) → threshold(0.5)
```
No additional post-processing. Symmetry enforced internally by `contact_a()`. Masking implicit via constraint matrix.

## 7. Known Issues and Limitations

1. **E2Efold struggles on diverse datasets**: Designed for 8 RNAStralign families. bpRNA (diverse) and Rivals (different distribution) yield much lower F1.
2. **`Lag_PP_mixed` L dependency**: The per-position `rho_m` matrix is L×L learnable parameters. Changing L changes model capacity. For non-600bp data, `Lag_PP_final` (L-agnostic) is the paper's recommended alternative.
3. **Training instability**: With longer training (>30 epochs Stage 3), loss can diverge to NaN on bpRNA. The optimal Stage 3 training length is ~10-25 epochs.
4. **Shared Stage 1 checkpoint paths**: When running Lag_PP_mixed and Lag_PP_final in parallel on the same dataset, they share the Stage 1 model path (same score network). This is functionally correct but operationally fragile.
