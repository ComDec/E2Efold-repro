"""
Preprocess rivals dataset (pkl format) into e2efold's expected pickle format.
Uses only 'seq' and 'label' fields (no matrix features).
"""
import os
import sys
import collections
import _pickle as cPickle
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

# Sequence encoding: 4-dim one-hot (must match e2efold/common/utils.py seq_dict)
seq_dict = {
    'A': np.array([1, 0, 0, 0]),
    'U': np.array([0, 1, 0, 0]),
    'C': np.array([0, 0, 1, 0]),
    'G': np.array([0, 0, 0, 1]),
    'N': np.array([0, 0, 0, 0]),
    'T': np.array([0, 1, 0, 0]),  # treat T as U
}

# Structure label encoding
label_dict = {
    '.': np.array([1, 0, 0]),
    '(': np.array([0, 1, 0]),
    ')': np.array([0, 0, 1]),
}

RNA_SS_data = collections.namedtuple('RNA_SS_data',
    'seq ss_label length name pairs')


def seq_encoding(seq_str):
    """Encode RNA sequence string to one-hot array [L, 4]."""
    return np.stack([seq_dict.get(c, np.array([0, 0, 0, 0])) for c in seq_str], axis=0)


def padding(arr, maxlen):
    """Pad array to maxlen along first axis."""
    if arr.ndim == 1:
        return np.pad(arr, (0, maxlen - len(arr)), 'constant')
    a, b = arr.shape
    return np.pad(arr, ((0, maxlen - a), (0, 0)), 'constant')


def contact_to_pairs(label_matrix):
    """
    Extract base pairs from symmetric contact map.
    Returns list of [i, j] for ALL nonzero entries (both directions),
    matching e2efold's get_pairings which returns both (i,j) and (j,i).
    """
    indices = np.where(label_matrix > 0.5)
    pairs = list(zip(indices[0].tolist(), indices[1].tolist()))
    return pairs


def contact_to_ss_label(label_matrix, seq_len):
    """
    Convert contact map to secondary structure label [L, 3].
    Simple heuristic: if position has any pair, mark as ( or ) based on pair direction.
    If unpaired, mark as dot.
    """
    ss = np.zeros((seq_len, 3))
    pairs = contact_to_pairs(label_matrix)
    paired_positions = set()
    for i, j in pairs:
        paired_positions.add(i)
        paired_positions.add(j)

    for pos in range(seq_len):
        if pos not in paired_positions:
            ss[pos] = label_dict['.']
        else:
            # Check if this position is the "left" or "right" in its pair
            is_left = any(p[0] == pos for p in pairs)
            if is_left:
                ss[pos] = label_dict['(']
            else:
                ss[pos] = label_dict[')']
    return ss


def process_dataset(pkl_path, maxlen):
    """Load a rivals pkl file and convert to list of RNA_SS_data."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    rna_ss_list = []
    for item in data:
        seq_str = item['seq']
        label = item['label']  # N x N contact map (NOT matrix)
        seq_len = len(seq_str)
        name = item['id']

        # One-hot encode sequence and pad
        seq_enc = seq_encoding(seq_str)
        seq_padded = padding(seq_enc, maxlen)

        # Generate ss_label and pad
        ss_label = contact_to_ss_label(label, seq_len)
        ss_padded = padding(ss_label, maxlen)

        # Extract pairs
        pairs = contact_to_pairs(label)

        rna_ss_list.append(RNA_SS_data(
            seq=seq_padded,
            ss_label=ss_padded,
            length=seq_len,
            name=name,
            pairs=pairs
        ))

    return rna_ss_list


def main():
    rivals_dir = '/home/xiwang/project/develop/data/rivals'
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rivals')

    # Use addss variant (base reference, no extra features)
    train_pkl = os.path.join(rivals_dir, 'TrainSetA-addss.pkl')
    testA_pkl = os.path.join(rivals_dir, 'TestSetA-addss.pkl')
    testB_pkl = os.path.join(rivals_dir, 'TestSetB-addss.pkl')

    # Determine global max length across all sets
    print("Scanning all datasets for max length...")
    all_lens = []
    for pkl_path in [train_pkl, testA_pkl, testB_pkl]:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        lens = [len(d['seq']) for d in data]
        all_lens.extend(lens)
        print(f"  {os.path.basename(pkl_path)}: {len(lens)} samples, "
              f"min={min(lens)}, max={max(lens)}, mean={np.mean(lens):.1f}")

    maxlen = max(all_lens)
    print(f"Global max length: {maxlen}")

    # Process all datasets
    print("\nProcessing TrainSetA...")
    train_all = process_dataset(train_pkl, maxlen)
    print(f"  {len(train_all)} samples")

    print("Processing TestSetA...")
    testA = process_dataset(testA_pkl, maxlen)
    print(f"  {len(testA)} samples")

    print("Processing TestSetB...")
    testB = process_dataset(testB_pkl, maxlen)
    print(f"  {len(testB)} samples")

    # Full TrainSetA as train, TestSetA as val, TestSetB as test
    print(f"\nSplit: {len(train_all)} train (full TrainSetA), "
          f"{len(testA)} val (TestSetA), {len(testB)} test (TestSetB)")

    # Save
    os.makedirs(output_dir, exist_ok=True)

    for split_name, split_data in [
        ('train', train_all),
        ('val', testA),
        ('test', testB),
    ]:
        out_path = os.path.join(output_dir, f'{split_name}.pickle')
        with open(out_path, 'wb') as f:
            cPickle.dump(split_data, f)
        print(f"Saved {split_name}: {len(split_data)} samples -> {out_path}")

    print(f"\nDone! Data saved to {output_dir}")
    print(f"Max sequence length (padded): {maxlen}")


if __name__ == '__main__':
    main()
