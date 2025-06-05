import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import random

class HandwritingPredictionDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = []
        for seq in sequences:
            zero_feature = torch.zeros(1, seq.shape[1], device=seq.device if isinstance(seq, torch.Tensor) else None)
            seq_with_zero = torch.cat([zero_feature, seq], dim=0)
            self.sequences.append(seq_with_zero)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        x = seq[:-1]  # Input
        y = seq[1:]   # Target
        return x, y

def collate_prediction(batch):
    """
    Collate function that pads variable-length sequences and returns lengths.

    Args:
        batch: list of tuples [(x1, y1), (x2, y2), ...]

    Returns:
        padded_x: [B, T, 3]
        padded_y: [B, T, 3]
        lengths: [B]
    """
    xs, ys = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in xs], dtype=torch.long,device=xs[0].device)

    padded_x = pad_sequence(xs, batch_first=True)  # [B, T, 3]
    padded_y = pad_sequence(ys, batch_first=True)  # [B, T, 3]

    return padded_x, padded_y, lengths


def get_prediction_loaders(data, batch_size, train_pct, val_pct):
    total_len = len(data)
    train_end = int(train_pct * total_len)
    val_end = train_end + int(val_pct * total_len)

    train_set = HandwritingPredictionDataset(data[:train_end])
    val_set = HandwritingPredictionDataset(data[train_end:val_end])
    test_set = HandwritingPredictionDataset(data[val_end:])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_prediction)
    val_loader = DataLoader(val_set, batch_size=batch_size, collate_fn=collate_prediction)
    test_loader = DataLoader(test_set, batch_size=batch_size, collate_fn=collate_prediction)

    return train_loader, val_loader, test_loader


class HandwritingSynthesisDataset(Dataset):
    def __init__(self, strokes, ascii_seqs):
        assert len(strokes) == len(ascii_seqs)
        self.strokes = []
        self.ascii_seqs = ascii_seqs
        for seq in strokes:
            zero_feature = torch.zeros(1, seq.shape[1], device=seq.device if isinstance(seq, torch.Tensor) else None)
            seq_with_zero = torch.cat([zero_feature, seq], dim=0)
            self.strokes.append(seq_with_zero)

    def __len__(self):
        return len(self.strokes)

    def __getitem__(self, idx):
        stroke_seq = self.strokes[idx]
        ascii_seq = self.ascii_seqs[idx]
        x = stroke_seq[:-1]
        y = stroke_seq[1:]
        return x, y, ascii_seq

    
def collate_synthesis(batch):
    """
    Collate function that pads variable-length sequences and returns lengths.

    Args:
        batch: list of tuples [(x1, y1), (x2, y2), ...]

    Returns:
        padded_x: [B, T, 3]
        padded_y: [B, T, 3]
        lengths: [B]
    """
    xs, ys, ascii = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in xs], dtype=torch.long,device=xs[0].device)
    ascii_lengths = torch.tensor([len(seq) for seq in ascii], dtype=torch.long,device=ascii[0].device)

    padded_x = pad_sequence(xs, batch_first=True)  # [B, T, 3]
    padded_y = pad_sequence(ys, batch_first=True)  # [B, T, 3]
    pad_ascii = pad_sequence(ascii, batch_first=True)

    return padded_x, padded_y, lengths, pad_ascii, ascii_lengths

def get_synthesis_loaders(strokes, ascii_seqs, batch_size, train_pct, val_pct):
    total_len = len(strokes)
    train_end = int(train_pct * total_len)
    val_end = train_end + int(val_pct * total_len)

    train_set = HandwritingSynthesisDataset(strokes[:train_end], ascii_seqs[:train_end])
    val_set = HandwritingSynthesisDataset(strokes[train_end:val_end], ascii_seqs[train_end:val_end])
    test_set = HandwritingSynthesisDataset(strokes[val_end:], ascii_seqs[val_end:])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,collate_fn=collate_synthesis)
    val_loader = DataLoader(val_set, batch_size=batch_size,collate_fn=collate_synthesis)
    test_loader = DataLoader(test_set, batch_size=batch_size,collate_fn=collate_synthesis)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    dummy_sequences = [torch.randn(random.randint(5, 15), 3) for _ in range(100)]

    # Test prediction loader
    train_loader, val_loader, test_loader = get_prediction_loaders(
        dummy_sequences,
        train_pct=0.7,
        val_pct=0.2
    )

    for x_batch, y_batch, lengths in train_loader:
        print("Input shape:", x_batch.shape)   # [1, T, 3]
        print("Target shape:", y_batch.shape)  # [1, T, 3]
        print("lengths:", lengths)
        break
