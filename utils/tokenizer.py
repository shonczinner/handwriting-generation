import json
import os

class CharTokenizer:
    def __init__(self, text=None):
        if text:
            self.chars = sorted(list(set(text)))
            self.stoi = {ch: i+1 for i, ch in enumerate(self.chars)} #0 saved for padding
            self.itos = {i: ch for ch, i in self.stoi.items()}
            self.vocab_size = len(self.chars)+1
        else:
            self.chars = []
            self.stoi = {}
            self.itos = {}
            self.vocab_size = 0

    def encode(self, text):
        return [self.stoi[ch] for ch in text]

    def decode(self, tokens):
        return ''.join([self.itos[i] for i in tokens])

    def save(self, path):
        tokenizer_data = {
            "chars": self.chars
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(tokenizer_data, f, ensure_ascii=False, indent=4)

        print("Tokenizer saved to", path)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        print("Tokenizer loaded from", path)
        tokenizer = cls()
        tokenizer.chars = data["chars"]
        tokenizer.stoi = {ch: i+1 for i, ch in enumerate(tokenizer.chars)}
        tokenizer.itos = {i: ch for ch, i in tokenizer.stoi.items()}
        tokenizer.vocab_size = len(tokenizer.chars)+1
        return tokenizer
