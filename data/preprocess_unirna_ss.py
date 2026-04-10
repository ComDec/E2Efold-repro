"""
Preprocess UniRNA-SS (all_data_1024_0.75) into e2efold pickle format.
Uses only 'seq' and 'label' fields (pure sequence, no matrix features).

Source: /home/xiwang/project/develop/data/all_data_1024_0.75/{train,valid,test}.pkl
Output: data/unirna_ss/{train,val,test}.pickle
"""
import os
import collections
import _pickle as cPickle
import pickle
import numpy as np

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


def process_dataset(pkl_path, maxlen):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    rna_ss_list = []
    for item in data:
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
    # Expected layout: $E2EFOLD_DATA_ROOT/all_data_1024_0.75/{train,valid,test}.pkl
    data_root = os.environ.get('E2EFOLD_DATA_ROOT', '/home/xiwang/project/develop/data')
    src = os.path.join(data_root, 'all_data_1024_0.75')
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'unirna_ss')

    files = {
        'train': os.path.join(src, 'train.pkl'),
        'val':   os.path.join(src, 'valid.pkl'),
        'test':  os.path.join(src, 'test.pkl'),
    }

    # Determine global max length across all splits
    print("Scanning all datasets for max length...")
    all_lens = []
    for split, path in files.items():
        with open(path, 'rb') as f:
            data = pickle.load(f)
        lens = [len(d['seq']) for d in data]
        all_lens.extend(lens)
        print(f"  {split}: {len(lens)} samples, min={min(lens)}, max={max(lens)}, mean={np.mean(lens):.1f}")

    maxlen = max(all_lens)
    if maxlen % 2 != 0:
        maxlen += 1
    print(f"Global max length: {maxlen} (padded)")

    os.makedirs(output_dir, exist_ok=True)

    for split, path in files.items():
        print(f"\nProcessing {split}...")
        ss_list = process_dataset(path, maxlen)
        out_path = os.path.join(output_dir, f'{split}.pickle')
        with open(out_path, 'wb') as f:
            cPickle.dump(ss_list, f)
        print(f"  Saved {len(ss_list)} samples -> {out_path}")

    print(f"\nDone! Data saved to {output_dir}, padded to {maxlen}")


if __name__ == '__main__':
    main()
