import os
import torch
import pandas as pd

from models.synthesis_model import SynthesisModel
from utils.config import Config
from utils.dataset import get_synthesis_loaders
from utils.plot_metrics import plot_metrics
from constants import SAVE_PATH


class SynthesisTrainer:
    def __init__(self, strokes,ascii, config: Config, device):
        self.config = config
        self.device = device

        self.train_loader, self.val_loader, self.test_loader = get_synthesis_loaders(
            strokes,
            ascii,
            config.batch_size,
            config.train_pct,
            config.val_pct
        )

        self.save_path =  os.path.join(SAVE_PATH, "synthesis")
        os.makedirs(self.save_path, exist_ok=True)
        self.model_path = os.path.join(self.save_path, "model.pth")

        self.model = SynthesisModel(config).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.loss_fn = torch.nn.GaussianNLLLoss()

        self.train_losses, self.val_losses = [], []
        self.load_model()


    def load_model(self):
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, weights_only=True))
            self.model = self.model.to(self.device)
            
            metrics = pd.read_csv(os.path.join(self.save_path, "metrics.csv"),header=0)
            self.train_losses = metrics["train_losses"].tolist()
            self.val_losses = metrics["val_losses"].tolist()

            print("Model loaded from",self.model_path)

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)
        metrics = pd.DataFrame({
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            })
        metrics.to_csv(os.path.join(self.save_path, "metrics.csv"), index=False)

        self.config.save(self.save_path)

        print("Model saved at",self.model_path)

    def run_epoch(self, loader, train):
        self.model.train() if train else self.model.eval()
        total_loss = 0

        with torch.set_grad_enabled(train):
            for x, y, lengths, ascii, ascii_length in loader:
                # x, y, lengths = x.to(self.device), y.to(self.device), lengths.to(self.device)
                # ascii, ascii_length = ascii.to(self.device), ascii_length.to(self.device)

                if train:
                    self.optimizer.zero_grad()

                # Assume model.loss accepts (x, y, lengths)
                loss, _ = self.model.loss(x, ascii, y, lengths)

                if train:
                    loss.backward()
                    torch.nn.utils.clip_grad_value_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                total_loss += loss.detach().item()

        return total_loss / len(loader)

    def train(self):
        for epoch in range(self.config.epochs):
            train_loss = self.run_epoch(self.train_loader,True)
            val_loss = self.run_epoch(self.val_loader,False)
            print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f}")
            print(f" Val Loss: {val_loss:.4f}")
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

        self.save_model()
        plot_metrics(None, self.train_losses, self.val_losses, self.save_path, "Loss")
     

    def evaluate(self):
        test_loss = self.run_epoch(self.test_loader,False)
        print(f"Test Loss: {test_loss:.4f}")

if __name__ == "__main__":
    from constants import PROCESSED_STROKES_PATH, RAW_ASCII_PATH
    from utils.tokenizer import CharTokenizer

    config = Config()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #device = 'cpu'
    print(f"Device: {device}")

    df_strokes = pd.read_csv(PROCESSED_STROKES_PATH)
    print("Strokes: ")
    print(df_strokes.head())

    df_ascii = pd.read_csv(RAW_ASCII_PATH)

    tokenizer = CharTokenizer("".join(df_ascii["text"].astype(str).to_list())) # only do if not already done
    tokenizer.save(os.path.join(SAVE_PATH,"tokenizer.json"))

    config.vocab_size = tokenizer.vocab_size

    strokes = []
    ascii = []
    # First, group both DataFrames by ('line', 'code')
    grouped_strokes = dict(tuple(df_strokes.groupby(['line', 'code'])))
    grouped_ascii = dict(tuple(df_ascii.groupby(['line', 'code'])))

    # Iterate only over the keys that are common to both
    common_keys = sorted(set(grouped_strokes.keys()) & set(grouped_ascii.keys()))

    print("# Stroke keys only:", len(set(grouped_strokes.keys()) - set(grouped_ascii.keys())))
    print("# ASCII keys only:", len(set(grouped_ascii.keys()) - set(grouped_strokes.keys())))
    print("# in common:",len(common_keys))

    for key in common_keys:
        stroke_group = grouped_strokes[key]
        ascii_group = grouped_ascii[key]

        features = torch.tensor(stroke_group[['delta_x', 'delta_y', 'lift_point']].values, dtype=torch.float32).to(device)
        strokes.append(features)

        tokens = torch.tensor(tokenizer.encode(str(ascii_group["text"].item())), dtype=torch.long).to(device)
        ascii.append(tokens)

    trainer = SynthesisTrainer(strokes,ascii,config,device)
    trainer.train()
    trainer.evaluate()





