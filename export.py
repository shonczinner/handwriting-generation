import os
from pathlib import Path
from models.synthesis_model import SynthesisModel
import torch
from torch import nn
from utils.config import Config
from constants import SAVE_PATH, PROCESSED_STROKES_STATS_PATH
from utils.display_strokes import plot_tensor
from utils.tokenizer import CharTokenizer


# Load config and model
model_folder = os.path.join(SAVE_PATH, "synthesis")
config = Config.load(model_folder)
model = SynthesisModel(config)
model_path = os.path.join(model_folder, "model.pth")
model.load_state_dict(torch.load(model_path))
print("Model loaded from:", model_path)

dummy_input = None
dummy_hidden = None
dummy_ascii = None

class ExportableSampleModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x,c,hidden=None):  # simulate the sample function
        return self.model.sample(x,c,hidden)
    
class ExportablePrimeModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x,c,hidden=None):  # simulate the sample function
        return self.model.generate_prime_hidden(x,c,hidden)

    
exportable_model = ExportableSampleModel(model)

torch.onnx.export(
     exportable_model,
    (dummy_input, dummy_hidden,ascii),  # Pass both input and hidden state
    "model.onnx",
    input_names=["input", "ascii","hidden"],  # Input names for both input and hidden state
    output_names=["output", "hidden"],  # Output names for both (x,y,lift) and hidden state
    dynamic_axes={
        "input": {1: "seq_len"},  # Make seq_len dynamic
        "ascii": {1: "ascii_len"}
    },
    opset_version=11
)

import json
tokenizer = CharTokenizer.load(os.path.join(model_folder,"tokenizer.json"))

chars = tokenizer.chars
vocab = {ch: i for i, ch in enumerate(chars)}

# Save vocab.json
with open("vocab.json", "w") as f:
    json.dump(vocab, f, indent=2)