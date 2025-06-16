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
    from utils.get_primed_sequence import HandwritingRecorder, df_to_tensor
    import tkinter as tk
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

    prime_text = "The dog sat on"

    root = tk.Tk()
    app = HandwritingRecorder(root)
    root.mainloop()

    df = app.get_stroke_df()

    #prime_tensor = df_to_tensor(df,PROCESSED_STROKES_STATS_PATH).to(device)
    prime_tensor = df_to_tensor(df).to(device)
    print(prime_tensor.shape)

    text = "the mat"
    prime_ascii = torch.tensor(tokenizer.encode(prime_text), dtype=torch.long).to(device).unsqueeze(0)
    ascii = torch.tensor(tokenizer.encode(text), dtype=torch.long).to(device).unsqueeze(0)

    cat_ascii = torch.concat((prime_ascii,ascii),dim=-1)
    prime_hidden = model.get_primed_hidden(prime_tensor,cat_ascii)
    hidden = list(model.get_initial_hidden_state(1))
    hidden[0]=prime_hidden[0]
    hidden[1] = hidden[1].to(device)
    hidden[2] = hidden[2].to(device)

    output,phis = model.full_sample(ascii, device, hidden = hidden, max_length=1000, temperature=0.7)

    # Plot strokes with denormalized values
    save_path = os.path.join(model_folder,"primed_sample.svg")

    plot_tensor(prime_tensor,denormalize_stats=PROCESSED_STROKES_STATS_PATH,ascii=prime_text,save_path= os.path.join(model_folder,"primer_sample.svg"))
    plot_tensor(output,denormalize_stats=PROCESSED_STROKES_STATS_PATH,ascii=text,save_path=save_path)
    plot_attention(text,torch.concat(phis,dim=1)[:,:,-(len(text)+1):].squeeze(), os.path.join(model_folder,"primed_sample_attention.png"))