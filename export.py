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
import pandas as pd


# Load config and model
model_folder = os.path.join(SAVE_PATH, "synthesis")
config = Config.load(model_folder)
model = SynthesisModel(config)
model_path = os.path.join(model_folder, "model.pth")
model.load_state_dict(torch.load(model_path))
print("Model loaded from:", model_path)

# Load tokenizer
tokenizer = CharTokenizer.load(os.path.join(model_folder, "tokenizer.json"))

chars = tokenizer.chars
vocab = {ch: i+1 for i, ch in enumerate(chars)}
vocab_size = tokenizer.vocab_size

# Save vocab.json
with open("vocab.json", "w") as f:
    json.dump(vocab, f, indent=2)

# Load CSV
df = pd.read_csv(PROCESSED_STROKES_STATS_PATH)

# Convert to dictionary: {stat: value}
stats_dict = dict(zip(df['stat'], df['value']))

# Dump to JSON file
with open("normalize_stats.json", "w") as f:
    json.dump(stats_dict, f, indent=2)

print("JSON saved:", stats_dict)

# Dummy inputs
dummy_input = torch.zeros(1, 1, 3)
dummy_ascii = torch.zeros(1, 1).long()

dummy_hidden_core = [torch.zeros(1, 1, config.hidden_dim) for _ in range(config.num_layers)]
dummy_kappa = torch.zeros((1, config.n_heads))
dummy_w0 = torch.zeros((1, 1, vocab_size))

dummy_hidden_parts = dummy_hidden_core + [dummy_kappa, dummy_w0]



# === Exportable sample model wrapper ===
class ExportableSampleModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, c, *hidden_parts):
        num_layers = self.model.num_layers
        hidden_core = list(hidden_parts[:num_layers])
        kappa = hidden_parts[num_layers]
        w0 = hidden_parts[num_layers + 1]
        hidden = (hidden_core, kappa, w0)

        sample, hidden_out,phi_termination = self.model.sample(x, c, hidden)
        hidden_core_out, kappa_out, w0_out = hidden_out
        return sample, *hidden_core_out, kappa_out, w0_out, phi_termination

exportable_sample_model = ExportableSampleModel(model)


# === Exportable initial hidden state model wrapper ===
class ExportableInitialHiddenModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self):
        # Returns initial hidden state tuple for given batch size
        hidden = self.model.get_initial_hidden_state(1)
        hidden_core, kappa, w0 = hidden
        # Flatten outputs for ONNX export (no tuple support)
        return (*hidden_core, kappa, w0)

exportable_initial_hidden_model = ExportableInitialHiddenModel(model)


# === Export ONNX for sample model ===
torch.onnx.export(
    exportable_sample_model,
    (dummy_input, dummy_ascii, *dummy_hidden_parts),
    "sample_model.onnx",
    input_names=["input", "ascii"] + [f"hidden_core_{i}" for i in range(config.num_layers)] + ["kappa", "w0"],
    output_names=["output"] + [f"hidden_core_{i}" for i in range(config.num_layers)] + ["kappa", "w0","phi"],
    dynamic_axes={
        "ascii": {1: "ascii_len"},
        "phi": {2: "ascii_len_p1"},
    },
    opset_version=11,
)

# === Export ONNX for initial hidden state model ===
torch.onnx.export(
    exportable_initial_hidden_model,
    (),  # dummy batch_size=1, device CPU
    "initial_state_model.onnx",
    input_names=[],
    output_names=[f"hidden_core_{i}" for i in range(config.num_layers)] + ["kappa", "w0"],
    opset_version=11
)

if __name__ == "__main__":
    import onnxruntime as ort
    import numpy as np

    # --- Utility ---
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else tensor

    # --- Test initial_state_model.onnx first to get initial hidden states ---
    initial_sess = ort.InferenceSession("initial_state_model.onnx")

    initial_outputs = initial_sess.run(
        None,
        {},
    )
    print("\ninitial_state_model.onnx outputs:")
    for i, out in enumerate(initial_outputs):
        print(f"Output {i} shape: {out.shape}")

    # These outputs correspond to hidden_core_0, hidden_core_1, ..., kappa, w0
    # Assume config.num_layers is known:
    num_layers = config.num_layers

    hidden_core_np = initial_outputs[:num_layers]  # list of arrays for hidden cores
    kappa_np = initial_outputs[num_layers]
    w0_np = initial_outputs[num_layers + 1]

    # --- Prepare inputs for sample_model.onnx ---
    dummy_input_np = to_numpy(dummy_input)

    dummy_ascii = torch.zeros(1, 6).long() # making sure ascii is variable,
    dummy_ascii_np = to_numpy(dummy_ascii)

    sample_sess = ort.InferenceSession("sample_model.onnx")

    sample_inputs = {
        "input": dummy_input_np,
        "ascii": dummy_ascii_np,
    }
    # Feed initial hidden states obtained from initial_state_model.onnx
    for i in range(num_layers):
        sample_inputs[f"hidden_core_{i}.1"] = hidden_core_np[i]
    sample_inputs["kappa.1"] = kappa_np
    sample_inputs["w0.1"] = w0_np

    # Run sample model with the initial hidden states
    sample_outputs = sample_sess.run(None, sample_inputs)
    print("\nsample_model.onnx outputs (using initial hidden state):")
    for i, out in enumerate(sample_outputs):
        print(f"Output {i} shape: {out.shape}")


    # now make sure it can run again using the new hidden outputs
    sample_out = sample_outputs[0]
    new_hidden_core_np = sample_outputs[1:1+num_layers]
    new_kappa_np = sample_outputs[1+num_layers]
    new_w0_np = sample_outputs[2+num_layers]
    new_phi_np = sample_outputs[3+num_layers]  # this is phi

    print("\nRunning second step with updated hidden state...")

    # Prepare new input for second step
    sample_inputs_2 = {
        "input": to_numpy(sample_out[None,None,:]),  # could use different input for actual next stroke
        "ascii": dummy_ascii_np,
    }
    for i in range(num_layers):
        sample_inputs_2[f"hidden_core_{i}.1"] = new_hidden_core_np[i]
    sample_inputs_2["kappa.1"] = new_kappa_np
    sample_inputs_2["w0.1"] = new_w0_np

    # Run model again
    sample_outputs_2 = sample_sess.run(None, sample_inputs_2)

    print("\nSecond step sample_model.onnx outputs:")
    for i, out in enumerate(sample_outputs_2):
        print(f"Output {i} shape: {out.shape}")
