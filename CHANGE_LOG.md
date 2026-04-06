# CHANGE_LOG.md

All modifications made to the original E2Efold repository for reproduction experiments.

## Environment Setup

- **Date**: 2026-04-05
- **Base Python**: 3.13.12 (system miniforge3), NOT the paper's conda env (Python 3.7 + PyTorch 1.2.0)
- **PyTorch**: 2.8.0+cu128 (vs original 1.2.0+cu92)
- **Additional packages installed**: `pip install munch torcheval`
- **e2efold installed**: `pip install -e .` from repo root
- **Hardware**: NVIDIA H100 NVL (95 GB), single GPU per experiment
- **Deviation from README**: Did NOT use `conda env create -f environment.yml` (original env requires Python 3.7 which is obsolete). Used existing Python 3.13 environment instead.

## Code Modifications to Original Repo

### 1. `e2efold/data_generator.py` (line 35)

**Change**: `np.array([...])` → `np.array([...], dtype=object)`

**Reason**: NumPy >= 1.24 raises ValueError on ragged arrays (each sample has different number of pairs). Adding `dtype=object` restores the original behavior.

**Impact**: None on model behavior. Pure compatibility fix.

```diff
-        self.pairs = np.array([instance[-1] for instance in self.data])
+        self.pairs = np.array([instance[-1] for instance in self.data], dtype=object)
```

### 2. `e2efold/models.py` — `Lag_PP_mixed.__init__` (lines 732, 743, 761-764)

**Change**: Added `L=600` default parameter to `__init__`, replaced hardcoded `600` with `L` in `rho_m` and `rho_pos_fea`.

**Reason**: Original code hardcodes `rho_m = nn.Parameter(torch.randn(600, 600))`. When data is padded to a length ≠ 600, this causes a shape mismatch RuntimeError in `update_rule`. Default `L=600` preserves backward compatibility.

**Impact**: When called with `L=600` (default), behavior is identical to original. When called with different L, creates appropriately-sized parameters. This is necessary for datasets with max length ≠ 600.

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

## New Files Added

### Data Preprocessing Scripts

| File | Purpose | Source Data |
|------|---------|-------------|
| `data/preprocess_rivals.py` | Convert Rivals pkl → e2efold format | `/home/xiwang/project/develop/data/rivals/` |
| `data/preprocess_bprna.py` | Convert bpRNA TR0/VL0/TS0 pkl → e2efold format | `/data/xiwang_home/project/develop/data/mxfold2/` |
| `data/preprocess_rnastralign_from_mxfold2.py` | Convert mxfold2 RNAStralign pkl → e2efold format | `/data/xiwang_home/project/develop/data/mxfold2/` |

### Experiment Directories

| Directory | Dataset | Configs |
|-----------|---------|---------|
| `experiment_rivals/` | Rivals TrainSetA/TestSetA/TestSetB | `config.json` (mixed), `config_final.json` (final) |
| `experiment_bprna/` | bpRNA TR0/VL0/TS0 | `config.json`, `config_final.json`, `config_v2.json` |
| `experiment_rnastralign_repro/` | RNAStralign (from mxfold2) | `config.json` |

### Evaluation Scripts

- `experiment_rivals/evaluate_rivals.py` — DeepRNA + E2Efold native metrics
- `experiment_bprna/evaluate.py` — Same, for bpRNA

## Data Preprocessing Details

All preprocessing scripts follow the same pattern:
1. Load source pkl (list of dicts with `{id, seq, label}`)
2. Encode `seq` to 4-dim one-hot (matching `e2efold/common/utils.py:seq_dict`)
3. Extract pairs from `label` matrix via `np.where(label > 0.5)` (both directions)
4. Pad all to global max length across train/val/test
5. Save as list of `RNA_SS_data` namedtuples in pickle format

**No data modification**: The preprocessing only converts format. No filtering, augmentation, or alteration of the original data.

## Experiment-Specific Training Script Changes vs Original

| Change | Reason |
|--------|--------|
| `upsampling=False` for Rivals/bpRNA | Original upsampling function hardcodes RNAStralign family indices and `name.split('/')` which crashes on non-RNAStralign data |
| `torch.load(..., weights_only=False)` | PyTorch 2.x default changed to `weights_only=True`; namedtuples need `False` |
| `Lag_PP_mixed(pp_steps, k, rho_per_position, L=seq_len)` | Pass actual seq_len to avoid shape mismatch |
| `config_v2.json`: `grad_accum=10`, `epoches_first=100`, `epoches_third=50` | Reviewer-recommended optimization for small datasets |
