# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

E2Efold is a PyTorch implementation of "RNA Secondary Structure Prediction By Learning Unrolled Algorithms" (ICLR 2020). It predicts RNA secondary structure (base-pair contact maps) by combining a deep learning score network with a differentiable augmented Lagrangian post-processing step, trained end-to-end.

## Setup

```bash
conda env create -f environment.yml
source activate rna_ss
pip install -e .
```

Key dependencies: Python 3.7, PyTorch 1.2.0, CUDA 9.2, NumPy, SciPy, scikit-learn, pandas, munch.

## Training Commands

All training scripts must be run from within their experiment directory (paths are relative with `../`).

### Short sequences (<600 bp) - RNAStralign

```bash
cd experiment_rnastralign
python e2e_learning_stage1.py -c config.json          # Stage 1: pre-train score network
python e2e_learning_stage3.py -c config.json           # Stage 3: end-to-end fine-tuning
python e2e_learning_stage3.py -c config.json --test True  # Test only
```

### Long sequences (600-1800 bp) - RNAStralign

```bash
cd experiment_rnastralign
python e2e_learning_stage1_rnastralign_all_long.py -c config_long.json
python e2e_learning_stage3_rnastralign_all_long.py -c config_long.json
```

### ArchiveII evaluation (uses models trained on RNAStralign)

```bash
cd experiment_archiveii
python e2e_learning_stage3.py -c config.json                              # short
python e2e_learning_stage3_rnastralign_all_long.py -c config_long.json    # long
```

### Inference on new sequences

```bash
cd e2efold_productive
sh main_short.sh    # sequences <600 bp
sh main_long.sh     # sequences 600-1800 bp
```

Input: CT format files in `short_seqs/` or `long_seqs/`. Output: CT files in `short_cts/` or `long_cts/`. Paths are configured in `config.json` / `config_long.json`.

## Architecture

### Two-stage training pipeline

1. **Stage 1** (`e2e_learning_stage1*.py`): Supervised pre-training of the contact/score network. Uses BCE loss with positive weighting to predict contact maps from sequence embeddings. Saves model to `models_ckpt/supervised_*.pt`.

2. **Stage 3** (`e2e_learning_stage3*.py`): End-to-end training of score network + post-processing network jointly. The `RNA_SS_e2e` wrapper module combines both networks. Supports both L2 and F1 loss (`pp_loss` config). Gradients are accumulated over 30 steps before optimizer step. Saves to `models_ckpt/e2e_*.pt`.

There is no Stage 2 in the codebase.

### Core modules (`e2efold/`)

- **`models.py`** - All neural network architectures:
  - **Score networks** (predict raw contact maps): `ContactAttention_simple_fix_PE` (default, with fixed positional encoding), `ContactAttention`, `ContactAttention_simple`, `ContactNetwork`, `ContactNetwork_test`, `ContactNetwork_fc`
  - **Post-processing networks** (enforce constraints via learned augmented Lagrangian): `Lag_PP_mixed` (default for short), `Lag_PP_final` (default for long), `Lag_PP_zero`, `Lag_PP_perturb`, `Lag_PP_NN`
  - **End-to-end wrapper**: `RNA_SS_e2e` - combines contact net + lag_pp net
  - All Lag_PP variants unroll constrained optimization for `pp_steps` iterations, enforcing that each nucleotide pairs with at most one other

- **`postprocess.py`** - Non-learned augmented Lagrangian post-processing (`postprocess()` function). Used as a baseline comparison; the learned Lag_PP networks replace this.

- **`data_generator.py`** - `RNASSDataGenerator` loads preprocessed pickle data (train/val/test splits). `Dataset` wraps it for PyTorch DataLoader. `Dataset_1800` handles long sequences via chunk-based processing (300 bp chunks with pairwise combinations).

- **`common/utils.py`** - Sequence encoding (`seq_dict`: IUPAC codes to 4-dim one-hot), evaluation metrics (`evaluate_exact`, `evaluate_shifted`, `F1_low_tri`), constraint matrix construction (`constraint_matrix_batch`: encodes valid AU/CG/GU base pairs), positional encoding (`get_pe`), CT format I/O.

- **`common/long_seq_pre_post_process.py`** - Chunk-based processing for long sequences: splits into 300 bp chunks, creates pairwise combinations, recombines with averaging on diagonals.

- **`common/config.py`** - JSON config loading via `munch` (attribute-style access).

- **`evaluation.py`** - Full test set evaluation: `model_eval_all_test` (compares learned vs non-learned PP), `all_test_only_e2e` (e2e evaluation only).

### Config system

JSON config files control all hyperparameters. Key fields:
- `model_type`: score network architecture (default: `att_simple_fix`)
- `pp_model`: post-processing variant (`mixed` for short, `final` for long)
- `pp_steps`: unrolled optimization iterations (default: 20)
- `pp_loss`: loss type (`f1` or `l2`)
- `data_type`: dataset folder name under `data/`
- `u_net_d`: hidden dimension (default: 10)
- `rho_per_position`: sparsity mode (`matrix` or `single`)
- `gpu`: CUDA device index as string

### Data format

Preprocessed data lives in `data/` as pickle files (`train.pickle`, `val.pickle`, `test_no_redundant_600.pickle`). Each sample is a named tuple `RNA_SS_data(seq, ss_label, length, name, pairs)` where `seq` is 4-dim one-hot encoded and `ss_label` is 3-dim (dot/open/close).

Model checkpoints go in `models_ckpt/`. Checkpoint filenames encode the full model configuration.


### 复现的核心注意事项和学术正直
- 维护CHANGE_LOG.md文件用于记录你的环境创建和代码修改，实验运行行为
- 尽可能按照原始仓库的ReadME来构建环境，自定义的适应操作是允许的，但是必须记录在CHANGE_LOG.md中
- 如果用户要求使用额外数据集进行训练和评测，最佳做法是写一个转化函数，将用户数据集转化成原始仓库的支持格式，并完整记录转化函数
- 最佳评测工作流是直接使用用户指明的验证和测试集，并尽可能不做修改的使用用户的评测函数，对于最终结果需要推理的到最终结果，便于用户后续直接评测
- 核心的训练集，验证集（如果有）和测试集必须由用户指明，你永远不允许修改训练中使用的数据集，或者对数据集做任何改动，每一次实验都需要调用subagents来判断实验的合法性
- 每次完成环境配置，原始论文结果复现，新数据集训练和测试后，都需要更新到REPRODUCTION.md文档中，所有的subagents都需要遵循这个原则
- 只使用Opus High Effort模型作为你的Subagents
- 每次新增的REPRODUCTION.md文档的内容，都需要额外调用Subagents做代码Review，着重评估代码Bug，评测策略和数据是否符合要求，是否存在任何数据集泄露，测试集训练，静默替换测试集的行为
- 维护Benchmark.md文档，记录官方数据集复现结果（和论文数值对比），以及所有额外数据集的结果，每一个数据集的结果都需要记录详细的日志，额外的代码改动，超参数
- 复现尽可能少添加和修改代码，所有的代码改动必须直观展示在git中，便于用户review，在未经允许的情况下，不要做git add/commit/push
- 复现过程中可能会出现中间数据文件，你可以添加到gitignore中，但必须明确告知用户
- Benchmark.md文档必须要写清楚新的Benchmark要怎么做，以及注意事项和禁止开展的行为