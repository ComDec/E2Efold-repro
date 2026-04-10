"""
Evaluate Stage 1 score network on UniRNA-SS with non-learned post-processing.

Pipeline: score_net(seq) -> postprocess(logits, seq, lr_min=0.01, lr_max=0.1, 50 itr) -> threshold 0.5
Saves test predictions to test_predictions.pkl.

Reports:
  1. DeepRNA metrics: binary_precision, binary_f1_score, binary_auroc, binary_auprc
  2. E2Efold native: evaluate_exact, evaluate_shifted
"""
import torch
from torch.utils import data
import numpy as np
import collections
import os
import _pickle as cPickle

from e2efold.models import ContactAttention_simple_fix_PE
from e2efold.common.utils import get_pe, seed_torch, get_args, evaluate_exact, evaluate_shifted
from e2efold.common.config import process_config
from e2efold.postprocess import postprocess
from e2efold.data_generator import RNASSDataGenerator, Dataset

from torcheval.metrics.functional import (
    binary_auprc,
    binary_auroc,
    binary_f1_score,
    binary_precision,
)

RNA_SS_data = collections.namedtuple('RNA_SS_data',
    'seq ss_label length name pairs')

args = get_args()
config_file = args.config
config = process_config(config_file)

os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu

d = config.u_net_d
data_type = config.data_type
model_type = config.model_type

model_path = '../models_ckpt/supervised_{}_{}_d{}_l3.pt'.format(model_type, data_type, d)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed_torch(0)

# Load train data only to determine seq_len
train_data = RNASSDataGenerator('../data/{}/'.format(data_type), 'train', False)
seq_len = train_data.data_y.shape[-2]
print('Max seq length:', seq_len)
del train_data

# Build score network only (no Lag_PP)
contact_net = ContactAttention_simple_fix_PE(d=d, L=seq_len, device=device).to(device)

print('Loading score network from:', model_path)
contact_net.load_state_dict(torch.load(model_path, weights_only=False))
contact_net.eval()


def evaluate_dataset(split_name, data_path, split, save_predictions=False):
    """Evaluate using non-learned postprocess()."""
    test_data = RNASSDataGenerator(data_path, split)
    test_set = Dataset(test_data)
    test_loader = data.DataLoader(test_set, batch_size=1, shuffle=False,
                                   num_workers=4, drop_last=False)

    deeprna_metrics = {'precision': [], 'f1': [], 'auroc': [], 'auprc': []}
    e2e_exact = []
    e2e_shifted = []

    # For saving predictions
    all_seq_lens = []
    all_preds = []
    all_preds_binary = []
    all_targets = []

    batch_n = 0
    for contacts, seq_embeddings, matrix_reps, seq_lens_batch in test_loader:
        if batch_n % 100 == 0:
            print(f'  Processing batch {batch_n}/{len(test_loader)}...')
        batch_n += 1

        contacts_batch = torch.Tensor(contacts.float()).to(device)
        seq_embedding_batch = torch.Tensor(seq_embeddings.float()).to(device)
        state_pad = torch.zeros(contacts.shape).to(device)
        PE_batch = get_pe(seq_lens_batch, contacts.shape[-1]).float().to(device)

        with torch.no_grad():
            pred_contacts = contact_net(PE_batch, seq_embedding_batch, state_pad)

        # Non-learned post-processing (same params as original Stage 1 eval)
        u_pp = postprocess(pred_contacts, seq_embedding_batch, 0.01, 0.1, 50, 1.0, True)
        pred_scores = u_pp.cpu()
        gt = contacts_batch.cpu()
        final_pred = (pred_scores > 0.5).float()

        for i in range(pred_scores.shape[0]):
            sl = int(seq_lens_batch[i].item())

            # DeepRNA metrics
            p = pred_scores[i, :sl, :sl].flatten()
            t = gt[i, :sl, :sl].flatten()

            deeprna_metrics['precision'].append(
                binary_precision(p, t, threshold=0.5).item())
            deeprna_metrics['f1'].append(
                binary_f1_score(p, t, threshold=0.5).item())
            deeprna_metrics['auroc'].append(binary_auroc(p, t).item())
            deeprna_metrics['auprc'].append(binary_auprc(p, t).item())

            # E2Efold native metrics (trimmed to actual seq length)
            e2e_exact.append(evaluate_exact(final_pred[i, :sl, :sl], gt[i, :sl, :sl]))
            e2e_shifted.append(evaluate_shifted(final_pred[i, :sl, :sl], gt[i, :sl, :sl]))

            if save_predictions:
                all_seq_lens.append(sl)
                all_preds.append(pred_scores[i, :sl, :sl].numpy())
                all_preds_binary.append(final_pred[i, :sl, :sl].numpy())
                all_targets.append(gt[i, :sl, :sl].numpy())

    # Print DeepRNA metrics
    print(f'\n=== {split_name}: DeepRNA Metrics ({len(deeprna_metrics["f1"])} samples) ===')
    for name in ['precision', 'f1', 'auroc', 'auprc']:
        vals = deeprna_metrics[name]
        print(f'  {name:>12s}: {np.mean(vals):.4f} (std={np.std(vals):.4f})')

    # Print e2efold native metrics
    exact_p, exact_r, exact_f1 = zip(*e2e_exact)
    shift_p, shift_r, shift_f1 = zip(*e2e_shifted)
    print(f'\n=== {split_name}: E2Efold Native Metrics ===')
    print(f'  Exact  - Precision: {np.average(exact_p):.4f}, Recall: {np.average(exact_r):.4f}, F1: {np.average(exact_f1):.4f}')
    print(f'  Shift  - Precision: {np.average(shift_p):.4f}, Recall: {np.average(shift_r):.4f}, F1: {np.average(shift_f1):.4f}')

    predictions = None
    if save_predictions:
        predictions = {
            'seq_len': all_seq_lens,
            'pred': all_preds,
            'pred_binary': all_preds_binary,
            'target': all_targets,
        }

    return deeprna_metrics, {'exact': e2e_exact, 'shifted': e2e_shifted}, predictions


print('\n' + '=' * 60)
print('Evaluating on val...')
mV_deep, mV_e2e, _ = evaluate_dataset('val', '../data/{}/'.format(data_type), 'val')

print('\n' + '=' * 60)
print('Evaluating on test (saving predictions)...')
mT_deep, mT_e2e, test_preds = evaluate_dataset('test', '../data/{}/'.format(data_type), 'test',
                                                  save_predictions=True)

# Save test predictions
pred_path = 'test_predictions.pkl'
with open(pred_path, 'wb') as f:
    cPickle.dump(test_preds, f)
print(f'\nTest predictions saved to {pred_path}')
