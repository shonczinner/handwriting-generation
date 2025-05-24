import torch
import pandas as pd

@torch.no_grad()
def sample(model, ascii, device, max_length=1000):
    model.eval()

    start = torch.zeros((1, 1, 3), dtype=torch.float32).to(device)  # B, T, F

    generated = start
    hidden = None
    for _ in range(max_length):
        sample, hidden = model.sample(generated[:, -1:],ascii, hidden)
        generated = torch.cat((generated, sample), dim=1)

    return generated

def tensor_to_df(output, denormalize_stats=None):
    df = pd.DataFrame(output.squeeze().detach().cpu().numpy(), columns=['delta_x', 'delta_y', 'lift_point'])
    df['line'] = 0

    if denormalize_stats is not None:
        stats = pd.read_csv(denormalize_stats).set_index('stat')['value']
        mu_dx = stats['mu_dx']
        sd_dx = stats['sd_dx']
        mu_dy = stats['mu_dy']
        sd_dy = stats['sd_dy']

        df['delta_x'] = df['delta_x'] * sd_dx + mu_dx
        df['delta_y'] = df['delta_y'] * sd_dy + mu_dy

    return df

if __name__ == "__main__":
    import os
    from pathlib import Path
    from models.synthesis_model import SynthesisModel
    from utils.config import Config
    from constants import SAVE_PATH, PROCESSED_STROKES_STATS_PATH
    from utils.display_strokes import plot_strokes
    from utils.tokenizer import CharTokenizer

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load config and model
    model_folder = os.path.join(SAVE_PATH, "synthesis")
    config = Config.load(model_folder)
    model = SynthesisModel(config)
    model_path = os.path.join(model_folder, "model.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Model loaded from:", model_path)

    model = model.to(device)

    tokenizer = CharTokenizer.load(os.path.join(SAVE_PATH,"tokenizer.json"))

    # Generate sample
    text = "hello world"
    ascii = torch.tensor(tokenizer.encode(text), dtype=torch.long).to(device).unsqueeze(0)
    output = sample(model, ascii, device, max_length=500)

    # Plot strokes with denormalized values
    plot_strokes(tensor_to_df(output, denormalize_stats=PROCESSED_STROKES_STATS_PATH),save_path=os.path.join(model_folder,"sample.png"))
