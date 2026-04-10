"""
Generalized pseudoknot-aware evaluation script.

Loads a saved `test_predictions.pkl` (produced by any experiment's evaluate.py)
and computes `score`, `score_pk`, `pk_sen/ppv/f1` using the unmodified
`deeprna.metrics.pseudoknot.evaluate_structure_metrics` function.

Expects predictions file to be a dict with keys:
  - 'pred'   : list of (L, L) float arrays
  - 'target' : list of (L, L) binary ground-truth contact maps

Example:
  python3 evaluate_pseudoknot.py \
      --predictions experiment_archiveii_full/test_predictions.pkl \
      --dataset_name ArchiveII
"""
import argparse
import os
import sys
import _pickle as pickle

# Path to the DeepRNA repo (for the pseudoknot metric module).
# Override via DEEPRNA_PATH env var if needed.
_deeprna_path = os.environ.get('DEEPRNA_PATH', '/home/xiwang/project/develop/deeprna')
if _deeprna_path not in sys.path:
    sys.path.insert(0, _deeprna_path)
from deeprna.metrics.pseudoknot import evaluate_structure_metrics  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', required=True,
                        help='Path to test_predictions.pkl')
    parser.add_argument('--dataset_name', default='dataset',
                        help='Name used in the output header')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Binarization threshold (default: 0.5)')
    args = parser.parse_args()

    print(f'Loading {args.predictions}...')
    with open(args.predictions, 'rb') as f:
        p = pickle.load(f)

    preds_list = [
        {'pred_prob': pr, 'label': tg}
        for pr, tg in zip(p['pred'], p['target'])
    ]

    print(f'Computing pseudoknot metrics on {len(preds_list)} samples '
          f'(threshold={args.threshold})...')
    result = evaluate_structure_metrics(preds_list, threshold=args.threshold)

    print()
    print('=' * 60)
    print(f'{args.dataset_name} — pseudoknot.py metrics')
    print('=' * 60)
    print(f"  n_total      : {result['n_total']}")
    print(f"  n_pk         : {result['n_pk']}  "
          f"({100 * result['n_pk'] / max(1, result['n_total']):.1f}%)")
    print(f"  score        : {result['score']:.4f}  (F1 over ALL samples)")
    print(f"  score_pk     : {result['score_pk']:.4f}  (F1 over PK-containing samples)")
    print(f"  pk_sen       : {result['pk_sen']:.4f}  (PK base-pair sensitivity)")
    print(f"  pk_ppv       : {result['pk_ppv']:.4f}  (PK base-pair PPV)")
    print(f"  pk_f1        : {result['pk_f1']:.4f}  (PK base-pair F1)")


if __name__ == '__main__':
    main()
