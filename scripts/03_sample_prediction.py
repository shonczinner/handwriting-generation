import torch
import pandas as pd




if __name__ == "__main__":
    import os
    from pathlib import Path
    from models.prediction_model import PredictionModel
    from utils.config import Config
    from constants import SAVE_PATH, PROCESSED_STROKES_STATS_PATH
    from utils.display_strokes import plot_tensor

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load config and model
    model_folder = os.path.join(SAVE_PATH, "prediction")
    config = Config.load(model_folder)
    model = PredictionModel(config)
    model_path = os.path.join(model_folder, "model.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Model loaded from:", model_path)

    model = model.to(device)

    # Generate sample
    output = model.full_sample(device, max_length=1000, temperature=1)

    # Plot strokes with denormalized values
    save_path = save_path=os.path.join(model_folder,"sample.svg")
    plot_tensor(output,denormalize_stats=PROCESSED_STROKES_STATS_PATH,save_path=save_path)