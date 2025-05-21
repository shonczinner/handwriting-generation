from dataclasses import dataclass, asdict
import json
import os

@dataclass
class Config:
    # Data and Training
    batch_size: int = 128
    max_seq_len: int = 500
    learning_rate: float = 1e-4
    epochs: int = 10
    train_pct: float = 0.8
    val_pct: float = 0.1
    test_pct: float = 0.1

    # Apply to all Models
    input_dim: int = 3
    hidden_dim: int = 400
    num_layers: int = 3
    n_components: int = 20

    # for synthesis 
    vocab_size: int = 0 # taken from tokenizer
    embed_dim: int = 256

    def save(self, path: str):
        with open(os.path.join(path,"config.json"), 'w') as f:
            json.dump(asdict(self), f, indent=4)

    @staticmethod
    def load(path: str) -> "Config":
        with open(os.path.join(path,"config.json"), 'r') as f:
            data = json.load(f)
        return Config(**data)
        
