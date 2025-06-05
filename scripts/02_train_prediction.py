import os
import torch
import pandas as pd

from models.prediction_model import PredictionModel
from utils.config import Config
from utils.dataset import get_prediction_loaders
from utils.plot_metrics import plot_metrics
from constants import SAVE_PATH


class PredictionTrainer:
    def __init__(self, data, config: Config, device):
        self.config = config
        self.device = device

        self.train_loader, self.val_loader, self.test_loader = get_prediction_loaders(
            data,
            batch_size = config.batch_size,
            train_pct=config.train_pct,
            val_pct=config.val_pct
        )

        self.save_path = os.path.join(SAVE_PATH, "prediction")
        os.makedirs(self.save_path, exist_ok=True)
        self.model_path = os.path.join(self.save_path, "model.pth")

        self.model = PredictionModel(config).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)

        self.train_losses, self.val_losses = [], []
        self.load_model()

    def load_model(self):
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path))
            self.model = self.model.to(self.device)
            
            metrics_path = os.path.join(self.save_path, "metrics.csv")
            if os.path.exists(metrics_path):
                metrics = pd.read_csv(metrics_path)
                self.train_losses = metrics["train_losses"].tolist()
                self.val_losses = metrics["val_losses"].tolist()
                print("Model loaded from", self.model_path)

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)
        metrics = pd.DataFrame({
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        })
        metrics.to_csv(os.path.join(self.save_path, "metrics.csv"), index=False)
        self.config.save(self.save_path)
        print("Model saved at", self.model_path)

    def run_epoch(self, loader, train):
        self.model.train() if train else self.model.eval()
        total_loss = 0

        with torch.set_grad_enabled(train):
            for x, y,lengths in loader:
                if train:
                    self.optimizer.zero_grad()

                loss, _ = self.model.loss(x, y,lengths=lengths)

                if train:
                    loss.backward()
                    torch.nn.utils.clip_grad_value_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                total_loss += loss.detach().item()

        return total_loss / len(loader)

    def train(self):
        for epoch in range(self.config.epochs):
            train_loss = self.run_epoch(self.train_loader, True)
            val_loss = self.run_epoch(self.val_loader, False)

            print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f}")
            print(f" Val Loss: {val_loss:.4f}")
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

        self.save_model()
        plot_metrics(None, self.train_losses, self.val_losses, self.save_path, "Loss")

    def evaluate(self):
        test_loss = self.run_epoch(self.test_loader, False)
        print(f"Test Loss: {test_loss:.4f}")


if __name__ == "__main__":
    from constants import PROCESSED_STROKES_PATH

    config = Config()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    df_strokes = pd.read_csv(PROCESSED_STROKES_PATH)
    print("Dataset: ")
    print(df_strokes.head())

    sequences = []
    for _, group in df_strokes.groupby(['line', 'code']):
        features = torch.tensor(group[['delta_x', 'delta_y', 'lift_point']].values, dtype=torch.float32).to(device)
        sequences.append(features)

    
    trainer = PredictionTrainer(sequences,config,device)
    trainer.train()
    trainer.evaluate()





