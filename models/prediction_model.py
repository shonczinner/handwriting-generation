import torch
import torch.nn as nn
from models.bivariate_bernoulli_mixture_head import BivariateBernoulliMixtureHead

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
    
    def loss(self,x,y,hidden=None):
        # x: [batch, seq_len, input_dim]
        # h: None or [num_layers, batch, hidden_size]
        out, hidden = self.rnn(x, hidden) # [batch, seq_len, hidden],[num_layers, batch, hidden_size]
        loss = self.binomialbernoullimixturehead.loss(out,y) # 
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

    # Dummy target assuming output is 3-dimensional: (x, y, bernoulli_mask)
    y = torch.randint(0, 2, (batch_size, seq_len, 3)).float()

    # --- Forward pass ---
    means, stdevs, log_weights, correlations, last_logit, hidden = model(x)
    print("Forward pass:")
    print("Means shape:", means.shape)
    print("Stdevs shape:", stdevs.shape)
    print("Log Weights shape:", log_weights.shape)
    print("Correlations shape:", correlations.shape)
    print("Last Logit shape:", last_logit.shape)

    # --- Loss computation ---
    loss, hidden = model.loss(x, y)
    print("\nLoss computation:")
    print("Loss:", loss.item())

    # --- Sampling ---
    sample, hidden = model.sample(x)
    print("\nSampling:")
    print("Sample shape:", sample.shape)


