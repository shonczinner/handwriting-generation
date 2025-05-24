import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence, PackedSequence
import random

class HandwritingPredictionDataset(Dataset):
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

def collate_fn_prediction(batch):
    x_seqs, y_seqs = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in x_seqs])

    lengths, perm_idx = lengths.sort(0, descending=True)
    x_seqs = [x_seqs[i] for i in perm_idx]
    y_seqs = [y_seqs[i] for i in perm_idx]

    x_padded = pad_sequence(x_seqs, batch_first=True)
    y_padded = pad_sequence(y_seqs, batch_first=True)

    return x_padded, y_padded, lengths

def get_prediction_loaders(data, batch_size, train_pct, val_pct):
    total_len = len(data)
    train_end = int(train_pct * total_len)
    val_end = train_end + int(val_pct * total_len)

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    train_set = HandwritingPredictionDataset(train_data)
    val_set = HandwritingPredictionDataset(val_data)
    test_set = HandwritingPredictionDataset(test_data)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,collate_fn=collate_fn_prediction)
    val_loader = DataLoader(val_set, batch_size=batch_size,collate_fn=collate_fn_prediction)
    test_loader = DataLoader(test_set, batch_size=batch_size,collate_fn=collate_fn_prediction)

    return train_loader, val_loader, test_loader

class HandwritingSynthesisDataset(Dataset):
    def __init__(self, strokes, ascii):
        """
        Args:
            strokes: list of [T_i, F] tensors (one per handwriting sample)
            ascii: list of [U_i, V] tensors (one per handwriting sample)
        """
        assert len(strokes) == len(ascii), "strokes and ascii must be the same length"
        self.strokes = strokes
        self.ascii = ascii

    def __len__(self):
        return len(self.strokes)

    def __getitem__(self, idx):
        stroke_seq = self.strokes[idx]  # [T, F]
        ascii_seq = self.ascii[idx]     # [U, V]

        x = stroke_seq[:-1]  # Input: all but last point
        y = stroke_seq[1:]   # Target: all but first point (i.e., next point prediction)

        return x, y, ascii_seq



def collate_fn_synthesis(batch):
    x_seqs, y_seqs, ascii_seqs = zip(*batch)

    # Sort by stroke length (required for packing)
    stroke_lengths = torch.tensor([len(seq) for seq in x_seqs],device=x_seqs[0].device)
    stroke_lengths, perm_idx = stroke_lengths.sort(0, descending=True)

    # Reorder batch
    x_seqs = [x_seqs[i] for i in perm_idx]
    y_seqs = [y_seqs[i] for i in perm_idx]
    ascii_seqs = [ascii_seqs[i] for i in perm_idx]

    # Pad strokes
    x_padded = pad_sequence(x_seqs, batch_first=True)  # [B, T, 3]
    y_padded = pad_sequence(y_seqs, batch_first=True)  # [B, T, 3]

    # Pad ASCII sequences (not packed)
    ascii_lengths = torch.tensor([seq.size(0) for seq in ascii_seqs],device=x_seqs[0].device)
    ascii_padded = pad_sequence(ascii_seqs, batch_first=True, padding_value=0)  # [B, U, V]

    return x_padded, y_padded, stroke_lengths, ascii_padded, ascii_lengths


def get_synthesis_loaders(strokes, ascii, batch_size, train_pct, val_pct):
    assert len(strokes) == len(ascii), "strokes and ascii must be the same length"

    total_len = len(strokes)
    train_end = int(train_pct * total_len)
    val_end = train_end + int(val_pct * total_len)

    train_set = HandwritingSynthesisDataset(strokes[:train_end], ascii[:train_end])
    val_set = HandwritingSynthesisDataset(strokes[train_end:val_end], ascii[train_end:val_end])
    test_set = HandwritingSynthesisDataset(strokes[val_end:], ascii[val_end:])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_synthesis)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_synthesis)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_synthesis)

    return train_loader, val_loader, test_loader

if __name__=="__main__":
    # Generate some dummy data: list of variable-length sequences
    # Each sequence is a tensor of shape [T_i, F] (e.g., F = 3)
    dummy_sequences = [torch.randn(random.randint(5, 15), 3) for _ in range(100)]

    # Create data loaders
    train_loader, val_loader, test_loader = get_prediction_loaders(
        dummy_sequences,
        batch_size=8,
        train_pct=0.7,
        val_pct=0.2
    )

    # Fetch one batch to test
    for x_batch, y_batch, lengths in train_loader:
        print("Input batch shape:", x_batch.shape)  # (batch_size, max_seq_len, feature_dim)
        print("Target batch shape:", y_batch.shape)
        print("Lengths:", lengths)
        break  # Only one batch is enough for test

