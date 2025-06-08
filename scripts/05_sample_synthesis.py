import torch
import pandas as pd




if __name__ == "__main__":
    import os
    from pathlib import Path
    from models.synthesis_model import SynthesisModel
    from utils.config import Config
    from constants import SAVE_PATH, PROCESSED_STROKES_STATS_PATH
    from utils.display_strokes import plot_tensor
    from utils.tokenizer import CharTokenizer
    from utils.plot_attention import plot_attention

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load config and model
    model_folder = os.path.join(SAVE_PATH, "synthesis")
    config = Config.load(model_folder)
    model = SynthesisModel(config)
    model_path = os.path.join(model_folder, "model.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Model loaded from:", model_path)

    model = model.to(device)

    tokenizer = CharTokenizer.load(os.path.join(model_folder,"tokenizer.json"))

    # Generate sample
    text = "Hello World"
    ascii = torch.tensor(tokenizer.encode(text), dtype=torch.long).to(device).unsqueeze(0)
    # Generate sample
    output,phis = model.full_sample(ascii, device, max_length=500, temperature=0.5)

    # Plot strokes with denormalized values
    save_path = os.path.join(model_folder,"sample.svg")
    plot_tensor(output,denormalize_stats=PROCESSED_STROKES_STATS_PATH,ascii=text,save_path=save_path)
    plot_attention(text,torch.concat(phis,dim=1).squeeze(),save_path = os.path.join(model_folder,"sample_attention.png"))
    
