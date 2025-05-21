import torch
from torch.utils.data import Dataset, DataLoader, Subset

import torch
from torch.utils.data import Dataset


class HandwritingPredictionDataset(Dataset):
    def __init__(self, sequences, seq_len, stride):
        """
        Args:
            sequences: list of [T_i, F] tensors (one per handwriting sample)
            seq_len: length of input sequence window
            stride: step size between windows
        """
        self.sequences = sequences
        self.stride = stride
        self.seq_len = seq_len

        # For each sequence, compute valid start indices for sliding windows
        self.starts = []
        for i, seq in enumerate(sequences):
            n_valid = len(seq) - seq_len - 1
            if n_valid<0:
                continue
            for start in range(0, n_valid+1, stride):
                self.starts.append((i, start))  # (sequence index, start index)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        seq_idx, start = self.starts[idx]
        seq = self.sequences[seq_idx]

        x = seq[start : start + self.seq_len]       # Input window
        y = seq[start+1 : start + self.seq_len+1]          # Next point as target

        return x, y


def get_prediction_loaders(data, seq_len, stride, batch_size, train_pct, val_pct):
    total_len = len(data)
    train_end = int(train_pct * total_len)
    val_end = train_end + int(val_pct * total_len)

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    train_set = HandwritingPredictionDataset(train_data, seq_len, stride)
    val_set = HandwritingPredictionDataset(val_data, seq_len, stride)
    test_set = HandwritingPredictionDataset(test_data, seq_len, stride)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    return train_loader, val_loader, test_loader


class HandwritingPredictionDataset2(Dataset):
    def __init__(self, sequences):
        """
        Args:
            sequences: list of [T_i, F] tensors (one per handwriting sample)
        """
        self.sequences = sequences


    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]

        x = seq[ : -1]       # Input window
        y = seq[1 : ]        # Next point as target

        return x, y

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence, PackedSequence

def collate_fn(batch):
    x_seqs, y_seqs = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in x_seqs])

    lengths, perm_idx = lengths.sort(0, descending=True)
    x_seqs = [x_seqs[i] for i in perm_idx]
    y_seqs = [y_seqs[i] for i in perm_idx]

    x_padded = pad_sequence(x_seqs, batch_first=True)
    y_padded = pad_sequence(y_seqs, batch_first=True)

    return x_padded, y_padded, lengths

def get_prediction_loaders2(data, batch_size, train_pct, val_pct):
    total_len = len(data)
    train_end = int(train_pct * total_len)
    val_end = train_end + int(val_pct * total_len)

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    train_set = HandwritingPredictionDataset2(train_data)
    val_set = HandwritingPredictionDataset2(val_data)
    test_set = HandwritingPredictionDataset2(test_data)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size,collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=batch_size,collate_fn=collate_fn)

    return train_loader, val_loader, test_loader


if __name__=="__main__":
    # Let's say each sequence is a [T, 3] tensor (delta_x, delta_y, lift_point)
    seqs = [
        torch.tensor([[1,2,3,4,5]]).T,  # seq 1
        torch.tensor([[6,7,8]]).T,  # seq 1
    ]

    dataset = HandwritingPredictionDataset(sequences=seqs, seq_len=3, stride=1)
    loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    print("Sequences: ")
    print(seqs)

    for x, y in loader:
        print("x: ")
        print(x)
        print("y: ")
        print(y)
        break
