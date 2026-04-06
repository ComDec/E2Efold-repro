"""
Evaluate trained e2efold model on val (TestSetA) and test (TestSetB).

Reports TWO metric sets:
  1. DeepRNA-consistent (secondary_structure_metircs from deepprotein/tasks/utils.py):
     binary_precision, binary_f1_score, binary_auroc, binary_auprc (threshold=0.5, flatten full matrix)
  2. E2Efold native: evaluate_exact (precision, recall, F1 on symmetric contact map)
"""
import torch
from torch.utils import data
import numpy as np
import collections
import os

from e2efold.models import ContactAttention_simple_fix_PE
from e2efold.models import Lag_PP_mixed, Lag_PP_final, RNA_SS_e2e
from e2efold.common.utils import get_pe, seed_torch, get_args, evaluate_exact, evaluate_shifted
from e2efold.common.config import process_config
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
pp_steps = config.pp_steps
pp_loss = config.pp_loss
data_type = config.data_type
model_type = config.model_type
pp_type = '{}_s{}'.format(config.pp_model, pp_steps)
rho_per_position = config.rho_per_position
k = config.k

e2e_model_path = '../models_ckpt/e2e_{}_{}_d{}_{}_{}_position_{}.pt'.format(
    model_type, pp_type, d, data_type, pp_loss, rho_per_position)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed_torch(0)

# Load train data only to determine seq_len
train_data = RNASSDataGenerator('../data/{}/'.format(data_type), 'train', False)
seq_len = train_data.data_y.shape[-2]
print('Max seq length:', seq_len)
del train_data

# Build model
contact_net = ContactAttention_simple_fix_PE(d=d, L=seq_len, device=device).to(device)

if 'mixed' in pp_type:
    lag_pp_net = Lag_PP_mixed(pp_steps, k, rho_per_position, L=seq_len).to(device)
elif 'final' in pp_type:
    lag_pp_net = Lag_PP_final(pp_steps, k, rho_per_position).to(device)

rna_ss_e2e = RNA_SS_e2e(contact_net, lag_pp_net)

print('Loading e2e model from:', e2e_model_path)
rna_ss_e2e.load_state_dict(torch.load(e2e_model_path, weights_only=False))
rna_ss_e2e.eval()


def evaluate_dataset(split_name, data_path, split):
    """Evaluate on a dataset using both DeepRNA and e2efold native metrics."""
    test_data = RNASSDataGenerator(data_path, split)
    test_set = Dataset(test_data)
    test_loader = data.DataLoader(test_set, batch_size=1, shuffle=False,
                                   num_workers=4, drop_last=False)

    # DeepRNA metrics (secondary_structure_metircs style)
    deeprna_metrics = {'precision': [], 'f1': [], 'auroc': [], 'auprc': []}

    # E2Efold native metrics
    e2e_exact = []     # (precision, recall, f1)
    e2e_shifted = []   # (precision, recall, f1) with 1-position shift allowed

    batch_n = 0
    for contacts, seq_embeddings, matrix_reps, seq_lens in test_loader:
        if batch_n % 100 == 0:
            print(f'  Processing batch {batch_n}/{len(test_loader)}...')
        batch_n += 1

        contacts_batch = torch.Tensor(contacts.float()).to(device)
        seq_embedding_batch = torch.Tensor(seq_embeddings.float()).to(device)
        state_pad = torch.zeros(contacts.shape).to(device)
        PE_batch = get_pe(seq_lens, contacts.shape[-1]).float().to(device)

        with torch.no_grad():
            pred_contacts = contact_net(PE_batch, seq_embedding_batch, state_pad)
            a_pred_list = lag_pp_net(pred_contacts, seq_embedding_batch)

        pred_scores = a_pred_list[-1].cpu()  # continuous values
        gt = contacts_batch.cpu()
        final_pred = (pred_scores > 0.5).float()

        for i in range(pred_scores.shape[0]):
            sl = int(seq_lens[i].item())

            # --- DeepRNA metrics: flatten full valid region ---
            # Matches secondary_structure_metircs exactly:
            #   p, t = p.flatten(), t.flatten()
            #   No .int() cast, no skip on zero contacts, no try/except
            p = pred_scores[i, :sl, :sl].flatten()
            t = gt[i, :sl, :sl].flatten()

            deeprna_metrics['precision'].append(
                binary_precision(p, t, threshold=0.5).item())
            deeprna_metrics['f1'].append(
                binary_f1_score(p, t, threshold=0.5).item())
            deeprna_metrics['auroc'].append(binary_auroc(p, t).item())
            deeprna_metrics['auprc'].append(binary_auprc(p, t).item())

            # --- E2Efold native metrics: on full symmetric contact map ---
            e2e_exact.append(evaluate_exact(final_pred[i], gt[i]))
            e2e_shifted.append(evaluate_shifted(final_pred[i], gt[i]))

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

    return deeprna_metrics, {'exact': e2e_exact, 'shifted': e2e_shifted}


print('\n' + '=' * 60)
print('Evaluating on val (VL0)...')
mV_deep, mV_e2e = evaluate_dataset('VL0 (val)', '../data/{}/'.format(data_type), 'val')

print('\n' + '=' * 60)
print('Evaluating on test (TS0)...')
mT_deep, mT_e2e = evaluate_dataset('TS0 (test)', '../data/{}/'.format(data_type), 'test')
