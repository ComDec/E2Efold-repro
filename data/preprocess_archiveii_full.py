"""
Preprocess ArchiveII dataset (full) into e2efold pickle format.

Training data: RNAStrAlign600-train.pkl (20923 samples, max_len=600)
  -> 90/10 split (random_state=42): 18830 train, 2093 val
  -> Padded to 600

Test data: archiveII.pkl (3966 samples, max_len=1800)
  -> Padded to 1800

Uses only 'seq' and 'label' fields (pure sequence, no matrix features).
"""
import os
import collections
import _pickle as cPickle
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

seq_dict = {
    'A': np.array([1, 0, 0, 0]),
    'U': np.array([0, 1, 0, 0]),
    'C': np.array([0, 0, 1, 0]),
    'G': np.array([0, 0, 0, 1]),
    'N': np.array([0, 0, 0, 0]),
    'T': np.array([0, 1, 0, 0]),
}

label_dict = {
    '.': np.array([1, 0, 0]),
    '(': np.array([0, 1, 0]),
    ')': np.array([0, 0, 1]),
}

RNA_SS_data = collections.namedtuple('RNA_SS_data',
    'seq ss_label length name pairs')


def seq_encoding(seq_str):
    return np.stack([seq_dict.get(c, np.array([0, 0, 0, 0])) for c in seq_str], axis=0)


def padding(arr, maxlen):
    if arr.ndim == 1:
        return np.pad(arr, (0, maxlen - len(arr)), 'constant')
    a, b = arr.shape
    return np.pad(arr, ((0, maxlen - a), (0, 0)), 'constant')


def contact_to_pairs(label_matrix):
    indices = np.where(label_matrix > 0.5)
    return list(zip(indices[0].tolist(), indices[1].tolist()))


def contact_to_ss_label(label_matrix, seq_len):
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
            is_left = any(p[0] == pos and p[1] > pos for p in pairs)
            if is_left:
                ss[pos] = label_dict['(']
            else:
                ss[pos] = label_dict[')']
    return ss


def process_items(items, maxlen):
    """Convert a list of dicts to RNA_SS_data namedtuples."""
    rna_ss_list = []
    for item in items:
        seq_str = item['seq']
        label = item['label']
        seq_len = len(seq_str)
        name = item['id']
        seq_enc = seq_encoding(seq_str)
        seq_padded = padding(seq_enc, maxlen)
        ss_label = contact_to_ss_label(label, seq_len)
        ss_padded = padding(ss_label, maxlen)
        pairs = contact_to_pairs(label)
        rna_ss_list.append(RNA_SS_data(
            seq=seq_padded, ss_label=ss_padded,
            length=seq_len, name=name, pairs=pairs))
    return rna_ss_list


def main():
    # Data root can be overridden via E2EFOLD_DATA_ROOT env var.
    # Expected layout: $E2EFOLD_DATA_ROOT/mxfold2/{RNAStrAlign600-train,archiveII}.pkl
    data_root = os.environ.get('E2EFOLD_DATA_ROOT', '/home/xiwang/project/develop/data')
    train_src = os.path.join(data_root, 'mxfold2', 'RNAStrAlign600-train.pkl')
    test_src = os.path.join(data_root, 'mxfold2', 'archiveII.pkl')
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'archiveii_full')

    os.makedirs(output_dir, exist_ok=True)

    # --- Training data: load and split 90/10 ---
    print("Loading training data from:", train_src)
    with open(train_src, 'rb') as f:
        train_all = pickle.load(f)
    print(f"  Total samples: {len(train_all)}")

    train_lens = [len(d['seq']) for d in train_all]
    train_maxlen = max(train_lens)
    print(f"  Min len: {min(train_lens)}, Max len: {train_maxlen}, Mean: {np.mean(train_lens):.1f}")

    # Split 90/10 with random_state=42
    train_items, val_items = train_test_split(train_all, test_size=0.1, random_state=42)
    print(f"  Split: {len(train_items)} train, {len(val_items)} val")

    # Pad training data to 600 (max len of training data)
    print(f"\nProcessing train split (pad to {train_maxlen})...")
    train_ss = process_items(train_items, train_maxlen)
    train_path = os.path.join(output_dir, 'train.pickle')
    with open(train_path, 'wb') as f:
        cPickle.dump(train_ss, f)
    print(f"  Saved {len(train_ss)} samples -> {train_path}")

    print(f"\nProcessing val split (pad to {train_maxlen})...")
    val_ss = process_items(val_items, train_maxlen)
    val_path = os.path.join(output_dir, 'val.pickle')
    with open(val_path, 'wb') as f:
        cPickle.dump(val_ss, f)
    print(f"  Saved {len(val_ss)} samples -> {val_path}")

    # --- Test data: archiveII.pkl ---
    print("\nLoading test data from:", test_src)
    with open(test_src, 'rb') as f:
        test_all = pickle.load(f)
    print(f"  Total samples: {len(test_all)}")

    test_lens = [len(d['seq']) for d in test_all]
    test_maxlen = max(test_lens)
    print(f"  Min len: {min(test_lens)}, Max len: {test_maxlen}, Mean: {np.mean(test_lens):.1f}")

    print(f"\nProcessing test split (pad to {test_maxlen})...")
    test_ss = process_items(test_all, test_maxlen)
    test_path = os.path.join(output_dir, 'test.pickle')
    with open(test_path, 'wb') as f:
        cPickle.dump(test_ss, f)
    print(f"  Saved {len(test_ss)} samples -> {test_path}")

    print(f"\nDone! Data saved to {output_dir}")
    print(f"  Train: {len(train_ss)} samples, padded to {train_maxlen}")
    print(f"  Val:   {len(val_ss)} samples, padded to {train_maxlen}")
    print(f"  Test:  {len(test_ss)} samples, padded to {test_maxlen}")


if __name__ == '__main__':
    main()
