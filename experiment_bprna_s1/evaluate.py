"""
Evaluate Stage 1 score network on bpRNA-1m (TR0/VL0/TS0) with non-learned post-processing.

Pipeline: score_net(seq) -> postprocess(logits, seq, lr_min=0.01, lr_max=0.1, 50 itr) -> threshold 0.5
Also evaluates raw sigmoid scores (without PP) for comparison.
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
import sys
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
    """Evaluate using non-learned postprocess() AND raw sigmoid (no PP)."""
    test_data = RNASSDataGenerator(data_path, split)
    test_set = Dataset(test_data)
    test_loader = data.DataLoader(test_set, batch_size=1, shuffle=False,
                                   num_workers=4, drop_last=False)

    # Metrics with PP
    deeprna_pp = {'precision': [], 'f1': [], 'auroc': [], 'auprc': []}
    e2e_exact_pp = []
    e2e_shifted_pp = []

    # Metrics without PP (raw sigmoid)
    deeprna_raw = {'precision': [], 'f1': [], 'auroc': [], 'auprc': []}
    e2e_exact_raw = []
    e2e_shifted_raw = []

    # For saving predictions
    all_seq_lens = []
    all_preds_pp = []
    all_preds_pp_binary = []
    all_preds_raw = []
    all_preds_raw_binary = []
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

        # Raw sigmoid scores (without PP)
        raw_scores = torch.sigmoid(pred_contacts).cpu()
        raw_binary = (raw_scores > 0.5).float()

        # Non-learned post-processing
        u_pp = postprocess(pred_contacts, seq_embedding_batch, 0.01, 0.1, 50, 1.0, True)
        pp_scores = u_pp.cpu()
        pp_binary = (pp_scores > 0.5).float()

        gt = contacts_batch.cpu()

        for i in range(pp_scores.shape[0]):
            sl = int(seq_lens_batch[i].item())

            # --- With PP metrics ---
            p_pp = pp_scores[i, :sl, :sl].flatten()
            t = gt[i, :sl, :sl].flatten()

            deeprna_pp['precision'].append(
                binary_precision(p_pp, t, threshold=0.5).item())
            deeprna_pp['f1'].append(
                binary_f1_score(p_pp, t, threshold=0.5).item())
            deeprna_pp['auroc'].append(binary_auroc(p_pp, t).item())
            deeprna_pp['auprc'].append(binary_auprc(p_pp, t).item())

            e2e_exact_pp.append(evaluate_exact(pp_binary[i, :sl, :sl], gt[i, :sl, :sl]))
            e2e_shifted_pp.append(evaluate_shifted(pp_binary[i, :sl, :sl], gt[i, :sl, :sl]))

            # --- Without PP metrics (raw sigmoid) ---
            p_raw = raw_scores[i, :sl, :sl].flatten()

            deeprna_raw['precision'].append(
                binary_precision(p_raw, t, threshold=0.5).item())
            deeprna_raw['f1'].append(
                binary_f1_score(p_raw, t, threshold=0.5).item())
            deeprna_raw['auroc'].append(binary_auroc(p_raw, t).item())
            deeprna_raw['auprc'].append(binary_auprc(p_raw, t).item())

            e2e_exact_raw.append(evaluate_exact(raw_binary[i, :sl, :sl], gt[i, :sl, :sl]))
            e2e_shifted_raw.append(evaluate_shifted(raw_binary[i, :sl, :sl], gt[i, :sl, :sl]))

            if save_predictions:
                all_seq_lens.append(sl)
                all_preds_pp.append(pp_scores[i, :sl, :sl].numpy())
                all_preds_pp_binary.append(pp_binary[i, :sl, :sl].numpy())
                all_preds_raw.append(raw_scores[i, :sl, :sl].numpy())
                all_preds_raw_binary.append(raw_binary[i, :sl, :sl].numpy())
                all_targets.append(gt[i, :sl, :sl].numpy())

    # Print WITH PP metrics
    print(f'\n=== {split_name}: WITH Post-Processing (DeepRNA Metrics, {len(deeprna_pp["f1"])} samples) ===')
    for name in ['precision', 'f1', 'auroc', 'auprc']:
        vals = deeprna_pp[name]
        print(f'  {name:>12s}: {np.mean(vals):.4f} (std={np.std(vals):.4f})')

    exact_p, exact_r, exact_f1 = zip(*e2e_exact_pp)
    shift_p, shift_r, shift_f1 = zip(*e2e_shifted_pp)
    print(f'\n=== {split_name}: WITH Post-Processing (E2Efold Native Metrics) ===')
    print(f'  Exact  - Precision: {np.average(exact_p):.4f}, Recall: {np.average(exact_r):.4f}, F1: {np.average(exact_f1):.4f}')
    print(f'  Shift  - Precision: {np.average(shift_p):.4f}, Recall: {np.average(shift_r):.4f}, F1: {np.average(shift_f1):.4f}')

    # Print WITHOUT PP metrics
    print(f'\n=== {split_name}: WITHOUT Post-Processing / Raw Sigmoid (DeepRNA Metrics, {len(deeprna_raw["f1"])} samples) ===')
    for name in ['precision', 'f1', 'auroc', 'auprc']:
        vals = deeprna_raw[name]
        print(f'  {name:>12s}: {np.mean(vals):.4f} (std={np.std(vals):.4f})')

    exact_p_r, exact_r_r, exact_f1_r = zip(*e2e_exact_raw)
    shift_p_r, shift_r_r, shift_f1_r = zip(*e2e_shifted_raw)
    print(f'\n=== {split_name}: WITHOUT Post-Processing (E2Efold Native Metrics) ===')
    print(f'  Exact  - Precision: {np.average(exact_p_r):.4f}, Recall: {np.average(exact_r_r):.4f}, F1: {np.average(exact_f1_r):.4f}')
    print(f'  Shift  - Precision: {np.average(shift_p_r):.4f}, Recall: {np.average(shift_r_r):.4f}, F1: {np.average(shift_f1_r):.4f}')

    predictions = None
    if save_predictions:
        predictions = {
            'seq_len': all_seq_lens,
            'pred_pp': all_preds_pp,
            'pred_pp_binary': all_preds_pp_binary,
            'pred_raw': all_preds_raw,
            'pred_raw_binary': all_preds_raw_binary,
            'target': all_targets,
        }

    return {
        'deeprna_pp': deeprna_pp,
        'e2e_exact_pp': e2e_exact_pp,
        'e2e_shifted_pp': e2e_shifted_pp,
        'deeprna_raw': deeprna_raw,
        'e2e_exact_raw': e2e_exact_raw,
        'e2e_shifted_raw': e2e_shifted_raw,
    }, predictions


# --- Evaluate val (VL0) ---
print('\n' + '=' * 60)
print('Evaluating on val (VL0)...')
val_results, _ = evaluate_dataset('bpRNA-1m VL0 (val)', '../data/{}/'.format(data_type), 'val')

# --- Evaluate test (TS0) ---
print('\n' + '=' * 60)
print('Evaluating on test (TS0), saving predictions...')
test_results, test_preds = evaluate_dataset('bpRNA-1m TS0 (test)', '../data/{}/'.format(data_type), 'test',
                                              save_predictions=True)

# Save test predictions
pred_path = 'test_predictions.pkl'
with open(pred_path, 'wb') as f:
    cPickle.dump(test_preds, f)
print(f'\nTest predictions saved to {pred_path}')

# --- Evaluate test_new (bpRNAnew) if available ---
test_new_path = '../data/{}/test_new.pickle'.format(data_type)
if os.path.exists(test_new_path):
    print('\n' + '=' * 60)
    print('Evaluating on test_new (bpRNA-1m-new / bpRNAnew), saving predictions...')
    new_results, new_preds = evaluate_dataset('bpRNA-1m-new (test_new)', '../data/{}/'.format(data_type), 'test_new',
                                                save_predictions=True)
    new_pred_path = 'test_new_predictions.pkl'
    with open(new_pred_path, 'wb') as f:
        cPickle.dump(new_preds, f)
    print(f'\ntest_new predictions saved to {new_pred_path}')
else:
    print(f'\nSkipping test_new: {test_new_path} not found.')
    new_results = None

# --- Final summary ---
print('\n' + '=' * 70)
print('FINAL RESULTS SUMMARY')
print('=' * 70)
print(f'Model: {model_path}')
print(f'Data:  {data_type}')
print()

def print_summary_row(label, metrics):
    exact_p, exact_r, exact_f1 = zip(*metrics['e2e_exact_pp'])
    raw_exact_p, raw_exact_r, raw_exact_f1 = zip(*metrics['e2e_exact_raw'])
    print(f'  {label}')
    print(f'    With PP:    E2E-Exact F1={np.average(exact_f1):.4f}, P={np.average(exact_p):.4f}, R={np.average(exact_r):.4f} | DeepRNA F1={np.mean(metrics["deeprna_pp"]["f1"]):.4f}, AUROC={np.mean(metrics["deeprna_pp"]["auroc"]):.4f}, AUPRC={np.mean(metrics["deeprna_pp"]["auprc"]):.4f}')
    print(f'    Without PP: E2E-Exact F1={np.average(raw_exact_f1):.4f}, P={np.average(raw_exact_p):.4f}, R={np.average(raw_exact_r):.4f} | DeepRNA F1={np.mean(metrics["deeprna_raw"]["f1"]):.4f}, AUROC={np.mean(metrics["deeprna_raw"]["auroc"]):.4f}, AUPRC={np.mean(metrics["deeprna_raw"]["auprc"]):.4f}')

print_summary_row('bpRNA-1m VL0 (val):', val_results)
print_summary_row('bpRNA-1m TS0 (test):', test_results)
if new_results is not None:
    print_summary_row('bpRNA-1m-new (test_new):', new_results)

print('\n' + '=' * 70)
