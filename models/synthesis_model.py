import torch
import torch.nn as nn
from models.bivariate_bernoulli_mixture_head import BivariateBernoulliMixtureHead
from models.soft_window import SoftWindow

class SynthesisModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size, config.embed_dim)

        self.rnn1 = nn.RNN(config.input_dim, config.hidden_dim, config.num_layers, batch_first=True)
        self.softwindow = SoftWindow(config.hidden_dim, config.n_components)
        self.rnn2 = nn.RNN(config.hidden_dim+config.embed_dim, config.hidden_dim, config.num_layers, batch_first=True)
        self.binomialbernoullimixturehead = BivariateBernoulliMixtureHead(config.hidden_dim, config.n_components)

    def network(self, x, c, hidden=None):
        # hidden: (hidden1, kappa, hidden2) or None
        if hidden is not None:
            hidden1, kappa, hidden2 = hidden
        else:
            hidden1 = hidden2 = kappa = None

        c_emb = self.embed(c)  # [B, U, embed_dim]
        out, hidden1 = self.rnn1(x, hidden1)
        window_out, kappa = self.softwindow(out, c_emb, kappa) # B, T, embed_dim
        out, hidden2 = self.rnn2(torch.cat((out,window_out),dim=-1), hidden2)

        return out, (hidden1, kappa, hidden2)

    def forward(self, x, c, hidden=None):
        out, hidden = self.network(x, c, hidden)
        means, stdevs, log_weights, correlations, last_logit = self.binomialbernoullimixturehead(out)
        return means, stdevs, log_weights, correlations, last_logit, hidden

    def loss(self, x, c, y, lengths, hidden=None):
        out, hidden = self.network(x, c, hidden)
        loss = self.binomialbernoullimixturehead.loss(out, y, lengths)
        return loss, hidden

    @torch.no_grad()
    def sample(self, x, c, hidden=None):
        out, hidden = self.network(x, c, hidden)
        sample = self.binomialbernoullimixturehead.sample(out)
        return sample, hidden

    @torch.no_grad()
    def plot_heatmap(self, y, x, c, hidden=None, save_path=None):
        out, hidden = self.network(x, c, hidden)
        self.binomialbernoullimixturehead.plot_heatmap(y, out, save_path=save_path)

# ---------------------------------------------------------------
# Test block
# ---------------------------------------------------------------
if __name__ == "__main__":
    from types import SimpleNamespace

    # Dummy config
    config = SimpleNamespace(
        input_dim=10,
        hidden_dim=32,
        num_layers=1,
        n_components=5,
        vocab_size=40,
        embed_dim=16
    )

    # Instantiate the model
    model = SynthesisModel(config)

    # Dummy inputs
    batch_size = 4
    seq_len = 20
    cond_len = 12

    x = torch.randn(batch_size, seq_len, config.input_dim)
    c = torch.randint(0, config.vocab_size, (batch_size, cond_len))  # character sequence (long)
    y = torch.randn(batch_size, seq_len, 3)  # target

    # --- Forward pass ---
    means, stdevs, log_weights, correlations, last_logit, hidden = model(x, c)
    print("Forward pass:")
    print("Means shape:", means.shape)
    print("Stdevs shape:", stdevs.shape)
    print("Log Weights shape:", log_weights.shape)
    print("Correlations shape:", correlations.shape)
    print("Last Logit shape:", last_logit.shape)

    # --- Loss computation ---
    loss, hidden = model.loss(x, c, y, torch.tensor([seq_len]*batch_size))
    print("\nLoss computation:")
    print("Loss:", loss.item())

    # --- Sampling ---
    sample_input = torch.randn(1, 1, config.input_dim)
    c_sample = torch.randint(0, config.vocab_size, (1, cond_len))
    sample, _ = model.sample(sample_input, c_sample)
    print("\nSampling:")
    print("Sample shape:", sample.shape)
