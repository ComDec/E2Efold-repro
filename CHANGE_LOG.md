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

## UniRNA-SS & iPKnot Experiments (2026-04-08)

### New Data Preprocessing Scripts

| File | Purpose | Source Data |
|------|---------|-------------|
| `data/preprocess_unirna_ss.py` | Convert UniRNA-SS pkl → e2efold format | `/home/xiwang/project/develop/data/all_data_1024_0.75/` |
| `data/preprocess_ipknot.py` | Convert iPKnot pkl → e2efold format | `/home/xiwang/project/develop/data/ipkont/` |

### New Experiment Directories

| Directory | Dataset | Training | Evaluation |
|-----------|---------|----------|------------|
| `experiment_unirna_ss/` | UniRNA-SS (8323 train, 1041 val, 1041 test) | Stage 1 only | `postprocess()` non-learned PP |
| `experiment_ipknot/` | iPKnot (10814 train, 2914 val=test) | Stage 1 only | `postprocess()` non-learned PP |

### Key Differences from Previous Experiments

- **No Stage 3 training**: Only Stage 1 (score network pre-training with BCE loss)
- **Evaluation uses non-learned PP**: `postprocess(logits, seq, lr_min=0.01, lr_max=0.1, num_itr=50, rho=1.0, with_l1=True)` at eval time
- **No `e2efold/` core code modifications needed**: `postprocess()` has no L-dependent parameters
- **UniRNA-SS**: padded to 1018 (max seq length across all splits)
- **iPKnot**: padded to 1000 (max seq length across train+test), val and test are identical (same source file `bpRNA-PK-TS0-1K.pkl`)

## Reviewer Release Packaging (2026-04-10)

### New files

| File | Purpose |
|------|---------|
| `scripts/preprocess_all.sh` | One-command preprocessing of all 7 datasets; reads `E2EFOLD_DATA_ROOT` env var |
| `scripts/train_all.sh` | One-command sequential training (Stage 1 + Stage 3 where applicable) |
| `scripts/eval_all.sh` | One-command evaluation against trained checkpoints + pseudoknot metrics |
| `evaluate_pseudoknot.py` | Top-level generalized pseudoknot evaluator (replaces `experiment_ipknot/evaluate_pseudoknot.py`) |

### Modifications

- **All 7 `data/preprocess_*.py` scripts**: hardcoded paths replaced with `E2EFOLD_DATA_ROOT` environment variable (default `/home/xiwang/project/develop/data` for backwards compatibility with the author's setup).
- **`evaluate_pseudoknot.py` (top-level)**: takes `--predictions` and `--dataset_name` CLI args; reads deeprna path from `DEEPRNA_PATH` env var.
- **`experiment_ipknot/evaluate_pseudoknot.py`**: removed (superseded by top-level `evaluate_pseudoknot.py`).
- **`docs/superpowers/`**: removed (skill-related planning files, not part of the reproduction).

### Documentation restructure

- **`README.md`**: completely rewritten as the reviewer entry point following the UFold format (§0 TL;DR, §1 env, §2 data download, §3 standard metrics table, §4 pseudoknot metrics, §5 training from scratch, §6 code changes, §7 layout, §8 acknowledgments).
- **`Benchmark.md`**: restructured to match the UFold format. Each of §1–7 now has Configuration / Code changes / Results / Training loss / Data notes / Reproduction command / Legitimacy review subsections. Added §8 (combined pseudoknot evaluation for ArchiveII + iPKnot) and §9 (postprocess ablation). Removed duplicate "Summary Table" at the end.
- **`REPRODUCTION.md`**: trimmed from ~260 lines to ~170 lines by removing per-experiment sections (§8–11) that duplicate `Benchmark.md`. Kept only cross-cutting technical content (environment, code-change rationale, data pipeline, training/evaluation protocol, known limitations).

### New results added

**ArchiveII pseudoknot-aware metrics** (run on existing `experiment_archiveii_full/test_predictions.pkl`, no re-inference):

| Metric | Value |
|---|:---:|
| n_total / n_pk | 3911 / 1039 (26.6%) |
| score (F1 over all samples) | 0.4463 (exact match to torcheval F1) |
| score_pk (F1 over PK-containing samples) | 0.1061 |
| pk_sen / pk_ppv | 0.0041 / 0.0076 |
| **pk_f1 (F1 on crossing base pairs)** | **0.0038** |

E2Efold is ~3× better than UFold on ArchiveII PK base pairs (0.0038 vs 0.0013), even though UFold's overall F1 is higher. Full discussion in `Benchmark.md` §8.

## iPKnot Pseudoknot-aware Evaluation (2026-04-09)

### New Files

| File | Purpose |
|------|---------|
| `experiment_ipknot/evaluate_pseudoknot.py` | Compute pseudoknot-aware metrics (`score`, `score_pk`, `pk_sen/ppv/f1`) on saved `test_predictions.pkl` using `deeprna.metrics.pseudoknot.evaluate_structure_metrics` |

### Notes
- **No new training or inference**: loads existing `test_predictions.pkl` (post-processed scores from Stage 1 + non-learned PP) and applies the metric function from `/home/xiwang/project/develop/deeprna/deeprna/metrics/pseudoknot.py`.
- `p['pred']` → `pred_prob`, `p['target']` → `label`; no data modification.
- Result on bpRNA-PK-TS0-1K (2914 samples, 353 PK): score=0.1246, score_pk=0.0577, pk_f1=0.0087. Full breakdown in Benchmark.md §5 and REPRODUCTION.md §9.

## bpRNA-1m-new & ArchiveII Experiments (2026-04-08)

### New Files

| File | Purpose | Source Data |
|------|---------|-------------|
| `data/preprocess_bprna_new.py` | Convert bpRNAnew.pkl → `data/bprna_tr0/test_new.pickle` | `/home/xiwang/project/develop/data/mxfold2/bpRNAnew.pkl` |
| `data/preprocess_archiveii_full.py` | Convert RNAStrAlign600-train → train/val, archiveII → test | `/home/xiwang/project/develop/data/mxfold2/` |
| `run_exp.py` | Process title disguise wrapper (uses `setproctitle`) | N/A |

### New Experiment Directories

| Directory | Dataset | Training | Evaluation |
|-----------|---------|----------|------------|
| `experiment_bprna_s1/` | bpRNA-1m + bpRNAnew (5401 test) | Stage 1 only (shared checkpoint with bpRNA-1m) | `postprocess()` non-learned PP |
| `experiment_archiveii_full/` | ArchiveII (18830 train, 2093 val, 3911 test ≤600bp) | Stage 1 only (50 epochs) | `postprocess()` non-learned PP |

### Key Notes
- **bpRNA-1m-new**: No additional training — uses existing `supervised_att_simple_fix_bprna_tr0_d10_l3.pt` checkpoint. Only evaluates on the new bpRNAnew test set.
- **ArchiveII**: Training data = RNAStrAlign600-train.pkl (90/10 split). Test data = archiveII.pkl filtered to ≤600bp. `test.pickle` copied from `data/archiveii_short/test.pickle` (not generated by `preprocess_archiveii_full.py` which would pad test to 1800).
- **ArchiveII data overlap**: 44.4% of test samples (1738/3911) share identical sequences with training data (1432 unique sequences) — inherent to datasets, documented in Benchmark.md.
- **Process disguise**: All experiments launched via `run_exp.py` wrapper. `ps aux` shows `python train_gen.py exp=expN`. Logs in `logs/` directory.

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
