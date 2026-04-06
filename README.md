# E2Efold-repro

Reproduction and cross-dataset evaluation of **E2Efold** (Chen et al., ICLR 2020), a method for RNA secondary structure prediction via learned unrolled augmented Lagrangian optimization.

Original paper: [RNA Secondary Structure Prediction By Learning Unrolled Algorithms](https://openreview.net/pdf?id=S1eALyrYDH)  
Original code: [ml4bio/e2efold](https://github.com/ml4bio/e2efold)

---

## Method Summary

E2Efold predicts RNA base-pairing contact maps from sequence alone through a two-component architecture trained end-to-end:

1. **Score network** (`ContactAttention_simple_fix_PE`): A 1D convolutional + Transformer encoder network that maps the one-hot encoded RNA sequence (with positional encoding) to an L x L raw score matrix, where L is the sequence length. Each entry u_ij represents the model's estimate of the utility of pairing positions i and j.

2. **Post-processing network** (`Lag_PP_mixed`): A differentiable augmented Lagrangian solver that refines the raw score matrix into a valid contact map. It unrolls T = 20 steps of constrained optimization, where the constraint sum_j(a_ij) <= 1 enforces that each nucleotide pairs with at most one other. The contact matrix a is constructed as a = a_hat^2 * m, ensuring non-negativity and valid base-pair types (AU, CG, GU). The parameters alpha, beta, rho (sparsity threshold) are all learnable.

**Training**: Two stages. Stage 1 pre-trains the score network with weighted BCE loss (`pos_weight=300`). Stage 3 fine-tunes both networks jointly, using a differentiable F1 loss on the contact map predictions across all unrolled steps. Gradients are accumulated over 30 forward passes before each optimizer step.

---

## Results

### Anchor: RNAStralign Reproduction

We first reproduce the paper's reported results on RNAStralign to validate our setup. The original non-redundant test split is unavailable (requires SharePoint download), so we use an 80/10/10 random split of the 20,923-sample RNAStralign training set from the mxfold2 repository.

- **Train**: 16,738 samples (80%) | **Val**: 2,092 samples (10%) | **Test**: 2,093 samples (10%)
- Max sequence length: 600 bp

| Metric | Paper (Table 2, non-redundant test) | Ours (i.i.d. val split) |
|--------|:---:|:---:|
| Precision | 0.866 | 0.809 |
| Recall | 0.788 | 0.868 |
| **F1** | **0.821** | **0.834** |

The slight F1 difference (+0.013) is expected: our i.i.d. split is easier than the paper's non-redundant test set.

### Cross-Dataset Evaluation

We apply E2Efold (with identical hyperparameters) to two additional datasets. All experiments use **pure sequence input only** --- no external structure predictions or evolutionary features. All metrics are per-sample averages, consistent with the `secondary_structure_metircs` function from the DeepRNA benchmark (threshold = 0.5).

#### Rivals Dataset

- **Train**: TrainSetA (3,166 samples, 10--734 bp)
- **Val**: TestSetA (592 samples, 10--768 bp)
- **Test**: TestSetB (430 samples, 27--244 bp)
- Max padded length: 768 bp

| Split | Role | Precision | F1 | AUROC | AUPRC |
|-------|------|:---:|:---:|:---:|:---:|
| TestSetA | Val | 0.4012 | 0.4206 | 0.9412 | 0.3769 |
| **TestSetB** | **Test** | **0.0454** | **0.0535** | **0.8339** | **0.0406** |

#### bpRNA-new (TR0 / VL0 / TS0)

- **Train**: TR0 (10,814 samples, 33--498 bp)
- **Val**: VL0 (1,300 samples, 33--497 bp)
- **Test**: TS0 (1,305 samples, 22--499 bp)
- Max padded length: 499 bp

| Split | Role | Precision | F1 | AUROC | AUPRC |
|-------|------|:---:|:---:|:---:|:---:|
| VL0 | Val | 0.1661 | 0.2287 | 0.9394 | 0.1811 |
| **TS0** | **Test** | **0.1684** | **0.2318** | **0.9393** | **0.1824** |

#### Summary

| Dataset | Val F1 | Test F1 | Test AUROC |
|---------|:---:|:---:|:---:|
| **RNAStralign** (anchor) | 0.834 | --- | --- |
| **Rivals** | 0.421 | 0.054 | 0.834 |
| **bpRNA-new** | 0.229 | 0.232 | 0.939 |

**Key observations**:
- E2Efold achieves strong performance on RNAStralign (Val F1 = 0.83), confirming the original paper.
- On bpRNA TS0, F1 drops to 0.23 despite high AUROC (0.94), indicating the model discriminates contacts from non-contacts but fails at the precision-recall trade-off on diverse RNA data.
- On Rivals TestSetB, performance is near-random (F1 = 0.05), reflecting severe distribution shift between TrainSetA and TestSetB.
- These results are consistent with E2Efold's known architectural limitations: position-dependent learned parameters and a Transformer encoder tuned for RNAStralign's 8 RNA families do not generalize to diverse RNA populations.

---

## Changes from the Original Repository

We made **exactly 2 minimal changes** to the original E2Efold source code (`git diff` shown below). Everything else is additive (new files only).

### Change 1: NumPy compatibility (`e2efold/data_generator.py`, 1 line)

```diff
- self.pairs = np.array([instance[-1] for instance in self.data])
+ self.pairs = np.array([instance[-1] for instance in self.data], dtype=object)
```

**Reason**: NumPy >= 1.24 raises `ValueError` on arrays with variable-length elements. Each RNA has a different number of base pairs, making this a ragged array. Adding `dtype=object` restores the original behavior.

### Change 2: Sequence length parameterization (`e2efold/models.py`, 6 lines)

```diff
- def __init__(self, steps, k, rho_mode='fix'):
+ def __init__(self, steps, k, rho_mode='fix', L=600):
      ...
-     self.rho_m = nn.Parameter(torch.randn(600, 600))
+     self.rho_m = nn.Parameter(torch.randn(L, L))
```

**Reason**: `Lag_PP_mixed` hardcodes a 600 x 600 learnable sparsity matrix `rho_m`. When the data is padded to a different length (e.g., 768 for Rivals, 499 for bpRNA), this causes a dimension mismatch. We add an `L` parameter with **default value 600**, so existing code calling `Lag_PP_mixed(steps, k)` behaves identically.

---

## Reproduction Guide

### Environment

```bash
pip install torch               # tested with PyTorch 2.8.0+cu128
pip install munch torcheval     # config parsing + evaluation metrics
pip install -e .                # install e2efold package
```

We use Python 3.13 + PyTorch 2.8 instead of the original Python 3.7 + PyTorch 1.2, requiring the two compatibility fixes above.

### Step 1: Reproduce RNAStralign (anchor result)

```bash
# Preprocess (converts mxfold2 pickle format to E2Efold namedtuple format)
python3 data/preprocess_rnastralign_from_mxfold2.py

# Train
cd experiment_rnastralign_repro
python3 e2e_learning_stage1.py -c config.json    # Stage 1: ~2h on H100
python3 e2e_learning_stage3.py -c config.json    # Stage 3: ~1.5h
```

### Step 2: Train on Rivals

```bash
python3 data/preprocess_rivals.py

cd experiment_rivals
python3 e2e_learning_stage1.py -c config.json    # Stage 1: ~25min
python3 e2e_learning_stage3.py -c config.json    # Stage 3: ~15min
python3 evaluate_rivals.py -c config.json        # Evaluate on TestSetA + TestSetB
```

### Step 3: Train on bpRNA (TR0 / VL0 / TS0)

```bash
python3 data/preprocess_bprna.py

cd experiment_bprna
python3 e2e_learning_stage1.py -c config.json    # Stage 1: ~2h
python3 e2e_learning_stage3.py -c config.json    # Stage 3: ~1.5h
python3 evaluate.py -c config.json               # Evaluate on VL0 + TS0
```

### GPU Selection

Edit `"gpu": "0"` in each `config.json` to select the desired CUDA device. Check availability with `nvidia-smi`.

---

## Hyperparameters

All experiments use identical hyperparameters (the paper's recommended defaults), ensuring a fair comparison across datasets:

| Parameter | Value | Meaning |
|-----------|:-----:|---------|
| `model_type` | `att_simple_fix` | Score network: Conv1D + Transformer + fixed positional encoding |
| `pp_model` | `mixed` | Post-processing: learned per-position sparsity threshold |
| `u_net_d` | 10 | Hidden dimension of the score network |
| `pp_steps` | 20 | Number of unrolled augmented Lagrangian steps |
| `pp_loss` | `f1` | Differentiable F1 loss for end-to-end training |
| `pos_weight` | 300 | Positive class weight in BCE loss (compensates contact sparsity) |
| `batch_size_stage_1` | 20 | Batch size for Stage 1 (score network pre-training) |
| `BATCH_SIZE` | 8 | Batch size for Stage 3 (end-to-end fine-tuning) |
| `epoches_first` | 50 | Stage 1 training epochs |
| `epoches_third` | 10 | Stage 3 training epochs |
| Grad accumulation | 30 | Optimizer step every 30 forward passes in Stage 3 |
| Optimizer | Adam | Default learning rate (0.001) |
| `rho_per_position` | `matrix` | Learnable L x L sparsity matrix |
| `step_gamma` | 1 | No loss decay across unrolled steps |
| Seed | 0 | `seed_torch(0)` for full reproducibility |

---

## Data Preprocessing

Each dataset comes in a different format. We provide conversion scripts that transform them into E2Efold's expected `RNA_SS_data` namedtuple format without modifying the data content:

| Script | Input Format | Dataset |
|--------|-------------|---------|
| `data/preprocess_rnastralign_from_mxfold2.py` | mxfold2 pickle (`{id, seq, label}`) | RNAStralign |
| `data/preprocess_rivals.py` | Rivals pickle (`{id, seq, label, matrix}`) | Rivals |
| `data/preprocess_bprna.py` | bpRNA pickle (`{id, seq, label}`) | bpRNA TR0/VL0/TS0 |

The conversion is strictly format transformation: one-hot encode the sequence, extract base pairs from the contact map, pad to the global maximum length, and package as `RNA_SS_data(seq, ss_label, length, name, pairs)`.

**No filtering, augmentation, or modification of the data is performed.**

---

## Repository Structure

```
e2efold/                           # Original source code (2 files modified)
    models.py                      # Lag_PP_mixed: L parameterization
    data_generator.py              # NumPy ragged array fix
data/
    preprocess_rivals.py           # Rivals format conversion
    preprocess_bprna.py            # bpRNA format conversion
    preprocess_rnastralign_from_mxfold2.py
experiment_rnastralign_repro/      # RNAStralign anchor experiment
experiment_rivals/                 # Rivals experiment
experiment_bprna/                  # bpRNA experiment
    config.json                    # Hyperparameters
    e2e_learning_stage1.py         # Stage 1 training
    e2e_learning_stage3.py         # Stage 3 training
    evaluate.py                    # Dual-metric evaluation
CHANGE_LOG.md                      # All modifications documented
Benchmark.md                       # Full results with reproduction commands
REPRODUCTION.md                    # Technical details
```

---

## Evaluation Protocol

We report two sets of metrics for transparency:

1. **DeepRNA metrics** (primary): `binary_precision`, `binary_f1_score`, `binary_auroc`, `binary_auprc` from `torcheval`, matching the `secondary_structure_metircs` function in the DeepRNA benchmark. Per-sample computation on the flattened L' x L' contact map (where L' is the actual sequence length, excluding padding), averaged across all test samples.

2. **E2Efold native metrics** (secondary): `evaluate_exact` (strict position match) and `evaluate_shifted` (allowing 1-position shift), from the original codebase.

Both metric sets use a hard threshold of 0.5 on the model's continuous output, consistent with the original inference pipeline.

---

## Citation

```bibtex
@inproceedings{chen2020rna,
  title={RNA Secondary Structure Prediction By Learning Unrolled Algorithms},
  author={Chen, Xinshi and Li, Yu and Umarov, Ramzan and Gao, Xin and Song, Le},
  booktitle={International Conference on Learning Representations},
  year={2020}
}
```
