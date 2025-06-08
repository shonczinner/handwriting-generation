import os
from pathlib import Path
from models.synthesis_model import SynthesisModel
import torch
from torch import nn
from utils.config import Config
from constants import SAVE_PATH, PROCESSED_STROKES_STATS_PATH
from utils.display_strokes import plot_tensor
from utils.tokenizer import CharTokenizer
import json



# Load config and model
model_folder = os.path.join(SAVE_PATH, "synthesis")
config = Config.load(model_folder)
model = SynthesisModel(config)
model_path = os.path.join(model_folder, "model.pth")
model.load_state_dict(torch.load(model_path))
print("Model loaded from:", model_path)

#load tokenizer

tokenizer = CharTokenizer.load(os.path.join(model_folder,"tokenizer.json"))

chars = tokenizer.chars
vocab = {ch: i for i, ch in enumerate(chars)}
vocab_size = tokenizer.vocab_size

# Save vocab.json
with open("vocab.json", "w") as f:
    json.dump(vocab, f, indent=2)

dummy_input = torch.zeros(1,1,3)

dummy_hidden_core = [torch.zeros(1, 1, config.hidden_dim) for _ in range(config.num_layers)]
dummy_kappa = torch.zeros((1, config.n_heads))    # or whatever shape is expected
dummy_w0 = torch.zeros((1, 1, vocab_size))  # or whatever shape is expected

dummy_hidden = (dummy_hidden_core, dummy_kappa,dummy_w0)


dummy_ascii = torch.zeros(1, 1).long()

class ExportableSampleModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x,c,hidden=None):  # simulate the sample function
        sample, hidden = self.model.sample(x,c,hidden)
        return sample, hidden
    
class ExportablePrimeModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x,c,hidden=None):  # simulate the sample function
        hidden = self.model.get_primed_hidden(x,c,hidden)
        return hidden

    
exportable_sample_model = ExportableSampleModel(model)
exportable_prime_model = ExportablePrimeModel(model)

torch.onnx.export(
     exportable_sample_model,
    (dummy_input,dummy_ascii, dummy_hidden),  # Pass both input and hidden state
    "sample_model.onnx",
    input_names=["input", "ascii","hidden"],  # Input names for both input and hidden state
    output_names=["output", "hidden"],  # Output names for both stroke and hidden state
    dynamic_axes={
        "input": {1: "seq_len"},  # Make seq_len dynamic
        "ascii": {1: "ascii_len"}
    },
    opset_version=11
)

torch.onnx.export(
     exportable_prime_model,
    (dummy_input,dummy_ascii),  # Pass both input and hidden state
    "prime_model.onnx",
    input_names=["input", "ascii"],  # Input names for both input and hidden state
    output_names=["hidden"],  # Output names for both stroke and hidden state
    dynamic_axes={
        "input": {1: "seq_len"},  # Make seq_len dynamic
        "ascii": {1: "ascii_len"}
    },
    opset_version=11
)



