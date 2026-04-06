"""
Preprocess RNAStrAlign dataset from mxfold2 pkl format into e2efold's expected pickle format.
Also processes ArchiveII (short sequences only) as a separate test set.
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
            is_left = any(p[0] == pos and p[1] > pos for p in pairs)
            if is_left:
                ss[pos] = label_dict['(']
            else:
                ss[pos] = label_dict[')']
    return ss


def process_items(items, maxlen):
    """Convert a list of dicts {id, seq, label} to list of RNA_SS_data."""
    rna_ss_list = []
    for item in items:
        seq_str = item['seq']
        label = item['label']  # N x N contact map
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
    data_dir = os.path.dirname(os.path.abspath(__file__))

    # Paths
    rnastralign_pkl = '/data/xiwang_home/project/develop/data/mxfold2/RNAStrAlign600-train.pkl'
    archiveii_pkl = '/data/xiwang_home/project/develop/data/mxfold2/archiveII.pkl'

    maxlen = 600  # Fixed max length for short sequences

    # =====================
    # Process RNAStrAlign
    # =====================
    print("Loading RNAStrAlign data...")
    with open(rnastralign_pkl, 'rb') as f:
        rnastralign_data = pickle.load(f)
    print(f"  Total samples: {len(rnastralign_data)}")

    lens = [len(d['seq']) for d in rnastralign_data]
    print(f"  Seq lengths: min={min(lens)}, max={max(lens)}, mean={np.mean(lens):.1f}")

    # Split 80/10/10 with random_state=42
    train_items, temp_items = train_test_split(
        rnastralign_data, test_size=0.2, random_state=42)
    val_items, test_items = train_test_split(
        temp_items, test_size=0.5, random_state=42)

    print(f"  Split: {len(train_items)} train, {len(val_items)} val, {len(test_items)} test")

    # Convert to RNA_SS_data format
    print("Processing train set...")
    train_ss = process_items(train_items, maxlen)
    print("Processing val set...")
    val_ss = process_items(val_items, maxlen)
    print("Processing test set...")
    test_ss = process_items(test_items, maxlen)

    # Save RNAStrAlign
    output_dir = os.path.join(data_dir, 'rnastralign_all_600')
    os.makedirs(output_dir, exist_ok=True)

    for split_name, split_data in [('train', train_ss), ('val', val_ss), ('test', test_ss)]:
        out_path = os.path.join(output_dir, f'{split_name}.pickle')
        with open(out_path, 'wb') as f:
            cPickle.dump(split_data, f)
        print(f"  Saved {split_name}: {len(split_data)} samples -> {out_path}")

    # =====================
    # Process ArchiveII (short only)
    # =====================
    print("\nLoading ArchiveII data...")
    with open(archiveii_pkl, 'rb') as f:
        archiveii_data = pickle.load(f)
    print(f"  Total samples: {len(archiveii_data)}")

    # Filter to sequences <= 600 bp
    archiveii_short = [d for d in archiveii_data if len(d['seq']) <= maxlen]
    print(f"  Samples <= {maxlen} bp: {len(archiveii_short)}")

    print("Processing ArchiveII short set...")
    archiveii_ss = process_items(archiveii_short, maxlen)

    # Save ArchiveII
    archiveii_dir = os.path.join(data_dir, 'archiveii_short')
    os.makedirs(archiveii_dir, exist_ok=True)
    out_path = os.path.join(archiveii_dir, 'test.pickle')
    with open(out_path, 'wb') as f:
        cPickle.dump(archiveii_ss, f)
    print(f"  Saved test: {len(archiveii_ss)} samples -> {out_path}")

    print("\nDone!")
    print(f"RNAStrAlign data saved to {output_dir}")
    print(f"ArchiveII data saved to {archiveii_dir}")
    print(f"Max sequence length (padded): {maxlen}")


if __name__ == '__main__':
    main()
