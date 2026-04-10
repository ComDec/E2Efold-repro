"""
Compare metrics WITH and WITHOUT postprocess() for Stage 1 models.
Runs on RNAStralign, UniRNA-SS, iPKnot.
"""
import torch
from torch.utils import data
import numpy as np
import collections
import os
import sys

from e2efold.models import ContactAttention_simple_fix_PE
from e2efold.common.utils import get_pe, seed_torch, evaluate_exact, evaluate_shifted
from e2efold.postprocess import postprocess
from e2efold.data_generator import RNASSDataGenerator, Dataset

from torcheval.metrics.functional import (
    binary_auprc, binary_auroc, binary_f1_score, binary_precision,
)

RNA_SS_data = collections.namedtuple('RNA_SS_data', 'seq ss_label length name pairs')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed_torch(0)


def evaluate_with_and_without_pp(data_dir, model_path, split, label, gpu="6"):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    train_data = RNASSDataGenerator(data_dir, 'train', False)
    seq_len = train_data.data_y.shape[-2]
    del train_data

    contact_net = ContactAttention_simple_fix_PE(d=10, L=seq_len, device=device).to(device)
    contact_net.load_state_dict(torch.load(model_path, weights_only=False))
    contact_net.eval()

    test_data = RNASSDataGenerator(data_dir, split)
    test_set = Dataset(test_data)
    test_loader = data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4, drop_last=False)

    # Metrics accumulators
    raw_metrics = {'precision': [], 'f1': [], 'auroc': [], 'auprc': []}
    pp_metrics  = {'precision': [], 'f1': [], 'auroc': [], 'auprc': []}
    raw_exact, pp_exact = [], []

    n = 0
    for contacts, seq_embeddings, matrix_reps, seq_lens_batch in test_loader:
        if n % 200 == 0:
            print(f'  {label}: {n}/{len(test_loader)}...')
        n += 1

        contacts_batch = contacts.float().to(device)
        seq_embedding_batch = seq_embeddings.float().to(device)
        state_pad = torch.zeros(contacts.shape).to(device)
        PE_batch = get_pe(seq_lens_batch, contacts.shape[-1]).float().to(device)

        with torch.no_grad():
            pred_contacts = contact_net(PE_batch, seq_embedding_batch, state_pad)

        # Raw sigmoid (no postprocess)
        raw_scores = torch.sigmoid(pred_contacts).cpu()
        # With postprocess
        pp_scores = postprocess(pred_contacts, seq_embedding_batch, 0.01, 0.1, 50, 1.0, True).cpu()

        gt = contacts_batch.cpu()

        for i in range(raw_scores.shape[0]):
            sl = int(seq_lens_batch[i].item())
            t = gt[i, :sl, :sl].flatten()

            # Skip samples with no positive contacts (avoid AUROC errors)
            if t.sum() == 0:
                continue

            for scores, metrics in [(raw_scores, raw_metrics), (pp_scores, pp_metrics)]:
                p = scores[i, :sl, :sl].flatten()
                metrics['precision'].append(binary_precision(p, t, threshold=0.5).item())
                metrics['f1'].append(binary_f1_score(p, t, threshold=0.5).item())
                try:
                    metrics['auroc'].append(binary_auroc(p, t).item())
                    metrics['auprc'].append(binary_auprc(p, t).item())
                except:
                    pass

            # E2Efold exact F1
            raw_bin = (raw_scores[i, :sl, :sl] > 0.5).float()
            pp_bin  = (pp_scores[i, :sl, :sl] > 0.5).float()
            gt_trim = gt[i, :sl, :sl]
            raw_exact.append(evaluate_exact(raw_bin, gt_trim))
            pp_exact.append(evaluate_exact(pp_bin, gt_trim))

    print(f'\n{"="*60}')
    print(f'{label} ({len(raw_metrics["f1"])} samples with contacts)')
    print(f'{"="*60}')
    print(f'{"Metric":<12} {"Raw (no PP)":>12} {"With PP":>12} {"Delta":>10}')
    print(f'{"-"*46}')
    for name in ['precision', 'f1', 'auroc', 'auprc']:
        r = np.mean(raw_metrics[name])
        p = np.mean(pp_metrics[name])
        d = p - r
        print(f'{name:<12} {r:>12.4f} {p:>12.4f} {d:>+10.4f}')

    raw_p, raw_r, raw_f1 = zip(*raw_exact)
    pp_p, pp_r, pp_f1 = zip(*pp_exact)
    print(f'\nE2Efold Exact:')
    print(f'{"Metric":<12} {"Raw (no PP)":>12} {"With PP":>12} {"Delta":>10}')
    print(f'{"-"*46}')
    print(f'{"Precision":<12} {np.nanmean(raw_p):>12.4f} {np.nanmean(pp_p):>12.4f} {np.nanmean(pp_p)-np.nanmean(raw_p):>+10.4f}')
    print(f'{"Recall":<12} {np.nanmean(raw_r):>12.4f} {np.nanmean(pp_r):>12.4f} {np.nanmean(pp_r)-np.nanmean(raw_r):>+10.4f}')
    print(f'{"F1":<12} {np.nanmean(raw_f1):>12.4f} {np.nanmean(pp_f1):>12.4f} {np.nanmean(pp_f1)-np.nanmean(raw_f1):>+10.4f}')

    return raw_metrics, pp_metrics


# === RNAStralign (Stage 1 model, val split) ===
print('\n' + '#'*60)
print('# RNAStralign - Stage 1 Only')
print('#'*60)
evaluate_with_and_without_pp(
    '../data/rnastralign_all_600/',
    '../models_ckpt/supervised_att_simple_fix_rnastralign_all_600_d10_l3.pt',
    'val', 'RNAStralign Val', gpu='6')

# === UniRNA-SS (Stage 1 model, test split) ===
print('\n' + '#'*60)
print('# UniRNA-SS - Stage 1 Only')
print('#'*60)
evaluate_with_and_without_pp(
    '../data/unirna_ss/',
    '../models_ckpt/supervised_att_simple_fix_unirna_ss_d10_l3.pt',
    'test', 'UniRNA-SS Test', gpu='6')

# === iPKnot (Stage 1 model, test split) ===
print('\n' + '#'*60)
print('# iPKnot - Stage 1 Only')
print('#'*60)
evaluate_with_and_without_pp(
    '../data/ipknot/',
    '../models_ckpt/supervised_att_simple_fix_ipknot_d10_l3.pt',
    'test', 'iPKnot Test', gpu='5')
