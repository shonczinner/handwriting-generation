import torch
import torch.nn as nn
from models.bivariate_bernoulli_mixture_head import BivariateBernoulliMixtureHead


class PredictionModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Multilayer RNN with skip connections

        self.rnns = []
        for n in range(config.num_layers):
            if n==0:
                input_dim = config.input_dim
            else:
                input_dim = config.input_dim+config.hidden_dim
            rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=config.hidden_dim,
                num_layers=1,
                batch_first=True
            )
            self.rnns.append(rnn)

        self.rnns = nn.ModuleList(self.rnns)

        self.head = BivariateBernoulliMixtureHead(config.hidden_dim*config.num_layers, config.n_components)
    

    def network(self, x, hidden=None):
        """
        Forward pass through stacked RNNs

        Args:
            x: [B, T, input_dim]
            hidden: list of hidden states (or None)

        Returns:
            out: [B, T, hidden_dim]
            new_hidden: list of final hidden states for each layer
        """
        if hidden is None:
            hidden = [None] * len(self.rnns)

        new_hidden = []

        inputs = []

        out, h = self.rnns[0](x, hidden[0])              # First RNN layer
        new_hidden.append(h)

        inputs.append(out)

        for i, rnn in enumerate(self.rnns[1:], start=1):
            out, h = rnn(torch.concat((x,out),dim=-1), hidden[i])               # Next RNN layer
            new_hidden.append(h)
            inputs.append(out)

        return torch.concat(inputs,dim=-1), new_hidden

    def forward(self, x, hidden=None):
        """
        Forward model prediction.

        Args:
            x: [B, T, input_dim]

        Returns:
            Tuple from mixture head + hidden states
        """
        out, hidden = self.network(x, hidden)
        return (*self.head(out), hidden)

    def loss(self, x, y, hidden=None, lengths=None):
        """
        Computes masked loss for a batch of sequences.

        Args:
            x: [B, T, input_dim]
            y: [B, T, 3]
            lengths: [B] or None

        Returns:
            Scalar loss, updated hidden states
        """
        out, hidden = self.network(x, hidden)
        loss = self.head.loss(out, y, lengths=lengths)
        return loss, hidden

    @torch.no_grad()
    def sample(self, x, hidden=None, temperature=1.0):
        """
        Samples next output given last input.

        Args:
            x: [1, 1, 3]
            temperature: float

        Returns:
            sample: [3], hidden
        """
        out, hidden = self.network(x, hidden)
        sample = self.head.sample(out, temperature)  # [1, 1, H] -> [1, H]
        return sample, hidden
    
    @torch.no_grad()
    def full_sample(self,device, hidden=None,temperature=1.0,max_length=500):
        self.eval()

        start = torch.zeros((1,1,3), dtype=torch.float32, device=device)  # T,F

        generated = start
        hidden = None
        for _ in range(max_length):
            sample, hidden = self.sample(generated[:,-1:], hidden,temperature=temperature)
            generated = torch.cat((generated, sample.unsqueeze(0).unsqueeze(0)), dim=1)

        return generated


# === Demo usage ===
if __name__ == "__main__":
    from types import SimpleNamespace

    config = SimpleNamespace(
        input_dim=3,
        hidden_dim=32,
        num_layers=2,
        n_components=5
    )

    model = PredictionModel(config)

    B, T = 4, 20
    x = torch.randn(B, T, config.input_dim)
    y = torch.cat([
        torch.randn(B, T, 2),
        torch.randint(0, 2, (B, T, 1)).float()
    ], dim=-1)

    # --- Forward pass ---
    means, stdevs, log_weights, correlations, last_logit, hidden = model(x)
    print("Forward pass shapes:")
    print("Means:", means.shape)           # [B, T, C, 2]
    print("Stdevs:", stdevs.shape)         # [B, T, C, 2]
    print("Log Weights:", log_weights.shape)  # [B, T, C]
    print("Correlations:", correlations.shape)  # [B, T, C]
    print("Last Logit:", last_logit.shape)     # [B, T]

    # --- Loss ---
    lengths = torch.tensor([20, 18, 15, 12])  # example padding lengths
    loss, _ = model.loss(x, y, lengths=lengths)
    print("\nLoss:", loss.item())

    # --- Sampling ---
    x_sample = torch.zeros(1, 1, 3)
    sample, _ = model.sample(x_sample)
    print("\nSample shape:", sample.shape)
    print("Sample:", sample)
