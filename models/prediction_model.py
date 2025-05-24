import torch
import torch.nn as nn
from models.bivariate_bernoulli_mixture_head import BivariateBernoulliMixtureHead
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class PredictionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.rnn = nn.RNN(config.input_dim, config.hidden_dim, config.num_layers, batch_first=True)

        self.binomialbernoullimixturehead = BivariateBernoulliMixtureHead(config.hidden_dim, config.n_components)

    def forward(self, x, hidden=None):
        # x: [batch, seq_len, input_dim]
        # h: None or [num_layers, batch, hidden_size]
        out, hidden = self.rnn(x, hidden) # [batch, seq_len, hidden],[num_layers, batch, hidden_size]
        means, stdevs, log_weights, correlations, last_logit = self.binomialbernoullimixturehead(out) # 
        return means, stdevs, log_weights, correlations, last_logit, hidden # ..., [num_layers, batch, hidden_size]
    
    def loss(self,x,y,lengths,hidden=None):
        # x: [batch, seq_len, input_dim]
        # h: None or [num_layers, batch, hidden_size]
        out, hidden = self.rnn(x, hidden) # [batch, seq_len, hidden],[num_layers, batch, hidden_size]
        loss = self.binomialbernoullimixturehead.loss(out,y,lengths) # 
        return loss, hidden # ..., [num_layers, batch, hidden_size]
    
    @torch.no_grad()
    def sample(self, x, hidden=None):
        # x: [1, 1, input_dim]
        # h: None or [num_layers, batch, hidden_size]
        out, hidden = self.rnn(x, hidden) # [1, 1, hidden],[num_layers, batch, hidden_size]
        sample = self.binomialbernoullimixturehead.sample(out) # 
        return sample, hidden # [1, 1, 3] [num_layers, batch, hidden_size]
    

    @torch.no_grad()
    def plot_heatmap(self, y, x, hidden=None,save_path=None):
        # x: [1, T, input_dim]
        # h: None or [num_layers, batch, hidden_size]
        out, hidden = self.rnn(x, hidden) # [1, 1, hidden],[num_layers, batch, hidden_size]
        self.binomialbernoullimixturehead.plot_heatmap(y,out,save_path=save_path) 





if __name__ == "__main__":
    from types import SimpleNamespace

    # Dummy config for testing
    config = SimpleNamespace(
        input_dim=10,
        hidden_dim=32,
        num_layers=1,
        n_components=5  # number of mixture components
    )

    # Instantiate the model
    model = PredictionModel(config)

    # Dummy input (batch_size=4, sequence_length=20, input_dim=config.input_dim)
    batch_size = 4
    seq_len = 20
    x = torch.randn(batch_size, seq_len, config.input_dim)

    # Simulate sequence lengths (e.g., as if sequences were padded)
    lengths = torch.randint(low=5, high=seq_len + 1, size=(batch_size,))

    # Dummy target: 2D points + bernoulli indicator
    y = torch.cat([
        torch.randn(batch_size, seq_len, 2),
        torch.randint(0, 2, (batch_size, seq_len, 1)).float()
    ], dim=-1)

    # Zero out padded targets (optional but good for testing)
    for i in range(batch_size):
        y[i, lengths[i]:] = 0.0

    # --- Forward pass ---
    means, stdevs, log_weights, correlations, last_logit, hidden = model(x)
    print("Forward pass:")
    print("Means shape:", means.shape)
    print("Stdevs shape:", stdevs.shape)
    print("Log Weights shape:", log_weights.shape)
    print("Correlations shape:", correlations.shape)
    print("Last Logit shape:", last_logit.shape)

    # --- Loss computation ---
    loss, hidden = model.loss(x, y, lengths)
    print("\nLoss computation:")
    print("Loss:", loss.item())

    # --- Sampling ---
    sample_input = x[0:1, -1:, :]  # last timestep of first sample
    sample, _ = model.sample(sample_input)
    print("\nSampling:")
    print("Sample shape:", sample.shape)
    print("Sample:", sample)
