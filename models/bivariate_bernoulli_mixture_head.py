import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Bernoulli, Independent
import matplotlib.pyplot as plt
import numpy as np

class BivariateBernoulliMixtureHead(nn.Module):
    def __init__(self, input_dim, n_components):
        super().__init__()
        self.n_components = n_components
        self.output_dim = 6 * n_components + 1
        self.net = nn.Linear(input_dim, self.output_dim)

    def forward(self, x):
        # x: [B, T, H]
        B,_,_ = x.shape
        out = self.net(x)  # [B, T, 6*n + 1]

        # Split into mixture outputs and final Bernoulli
        mixture, last_prob = out[:, :, :-1], out[:, :, -1]

        # Chunk mixture into parameter groups
        means, std_raw, weight_raw, corr_raw = torch.split(
            mixture,
            [2 * self.n_components, 2 * self.n_components, self.n_components, self.n_components],
            dim=2
        )

        means = means.reshape(B,-1,self.n_components,2)
        stdevs = torch.exp(std_raw).reshape(B,-1,self.n_components,2)
        log_weights = F.log_softmax(weight_raw, dim=2)
        correlations = torch.tanh(corr_raw)
        last_logit = last_prob

        return means, stdevs, log_weights, correlations, last_logit
    
    def loss(self, x, y):
        # x: [B, T, H]
        # y: [B, T, 3]
        means, stdevs, log_weights, correlations, last_logit = self.forward(x)        

        z0 = (y[...,:2].unsqueeze(-2)-means)/stdevs # [B, T, n_components, 2]
        Z = z0[...,0].pow(2)+z0[...,1].pow(2)-2*correlations*z0[...,0]*z0[...,1]

        log_2pi = torch.log(torch.tensor(2 * torch.pi, device=stdevs.device, dtype=stdevs.dtype))
        log_probs = -0.5 * Z / (1 - correlations.pow(2)) \
            - log_2pi \
            - torch.log(stdevs[..., 0]) \
            - torch.log(stdevs[..., 1]) \
            - 0.5 * torch.log(1 - correlations.pow(2))
        
        gaussian_loss = -torch.logsumexp(log_weights+log_probs,dim=-1)
        bernoulli_loss = F.binary_cross_entropy_with_logits(last_logit,y[...,2])
        return (gaussian_loss+bernoulli_loss).mean()

    @torch.no_grad() 
    def sample(self, x):
        # x: [1, 1, H]
        means, stdevs, log_weights, correlations, last_logit = self.forward(x)
        # Output shapes: [1, 1, n, 2], [1, 1, n, 2], [1, 1, n], [1, 1, n], [1, 1]

        means = means[0, 0]             # [n, 2]
        stdevs = stdevs[0, 0]           # [n, 2]
        log_weights = log_weights[0, 0] # [n]
        correlations = correlations[0, 0] # [n]
        last_logit = last_logit[0, 0]     # scalar

        weights = torch.exp(log_weights)  # [n]
        idx = torch.distributions.Categorical(weights).sample()  # scalar

        selected_mean = means[idx]        # [2]
        selected_stdev = stdevs[idx]      # [2]
        selected_corr = correlations[idx] # scalar

        cov = self._build_covariance_matrix(selected_stdev, selected_corr)  # [2, 2]

        mvn = MultivariateNormal(loc=selected_mean, covariance_matrix=cov)
        gaussian_sample = mvn.sample()  # [2]

        bernoulli_sample = Bernoulli(logits=last_logit).sample()  # scalar

        full_sample = torch.cat([gaussian_sample, bernoulli_sample.unsqueeze(0)], dim=0)  # [3]
        return full_sample.unsqueeze(0).unsqueeze(0)  # [1, 1, 3]
    


    @torch.no_grad()
    def plot_heatmap(self, y, x, save_path=None):
        # y: [1, T, 3]
        # x: [1, T, H]
        B, T, H = x.shape
        assert B == 1, "plot_heatmap expects a single batch (B=1)"

        # Forward pass
        means, stdevs, log_weights, correlations, last_logit = self.forward(x)
        weights = torch.exp(log_weights)[0]             # [T, n]
        means = means[0]                                # [T, n, 2]
        stdevs = stdevs[0]                              # [T, n, 2]
        correlations = correlations[0]                  # [T, n]
        last_prob = torch.sigmoid(last_logit[0])        # [T]

        # Compute absolute positions and predicted means
        actual_pos = torch.cumsum(y[0, :, :2], dim=0)   # [T, 2]
        predicted_pos = actual_pos.unsqueeze(1) + means # [T, n, 2]

        covs = self._build_covariance_matrix(stdevs, correlations)  # [T, n, 2, 2]

        device = predicted_pos.device

        headmap_2d = None

        # Plotting setup
        fig, axs = plt.subplots(3, 1, figsize=(12, 10),
                                gridspec_kw={'height_ratios': [5, 1, 1]})

        # Heatmap + Trajectory
        im0 = axs[0].imshow(heatmap_2d, extent=(x_grid[0].item(), x_grid[-1].item(),
                                                y_grid[0].item(), y_grid[-1].item()),
                            origin='lower', cmap='hot', alpha=0.7)
        fig.colorbar(im0, ax=axs[0], orientation='vertical', label='Density')

        axs[0].scatter(actual_pos[:, 0].cpu().numpy(),
                    actual_pos[:, 1].cpu().numpy(),
                    color='cyan', label='Actual Trajectory', s=2)
        axs[0].set_title("Predicted Heatmap + Actual Trajectory")
        axs[0].set_ylabel("Y")
        axs[0].axis('equal')
        axs[0].legend()

        # Mixture Weights
        im1 = axs[1].imshow(weights.T.cpu().numpy(), aspect='auto', cmap='viridis',
                            extent=[0, T, 0, weights.shape[1]])
        axs[1].set_title("Mixture Weights")
        axs[1].set_ylabel("Component")
        fig.colorbar(im1, ax=axs[1], orientation='vertical', label='Weight')

        # Bernoulli Probabilities
        im2 = axs[2].imshow(last_prob.unsqueeze(0).cpu().numpy(), aspect='auto', cmap='coolwarm',
                            extent=[0, T, 0, 1])
        axs[2].set_title("Bernoulli Probability")
        axs[2].set_yticks([])
        axs[2].set_xlabel("Timestep")
        fig.colorbar(im2, ax=axs[2], orientation='vertical', label='Probability')

        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()


    @staticmethod
    def _build_covariance_matrix(stdevs, correlations):
        """
        Build 2x2 covariance matrices from stddevs and correlations.

        Parameters:
            stdevs: Tensor of shape [..., 2]
            correlations: Tensor of shape [...]

        Returns:
            cov: Tensor of shape [..., 2, 2]
        """
        sigma_x, sigma_y = stdevs[..., 0], stdevs[..., 1]
        rho = correlations

        shape = sigma_x.shape
        cov = torch.zeros(*shape, 2, 2, device=stdevs.device)

        cov[..., 0, 0] = sigma_x ** 2
        cov[..., 1, 1] = sigma_y ** 2
        cov[..., 0, 1] = cov[..., 1, 0] = rho * sigma_x * sigma_y

        return cov




    
        
if __name__ == "__main__":
    from pathlib import Path
    from constants import TEST_RESULTS_PATH 
    torch.manual_seed(42)

    B, T, H = 4, 10, 32  # Batch size, time steps, input dimension
    n_components = 5

    # Create random input and target tensors
    x = torch.randn(B, T, H)
    # y: 2D target values + Bernoulli (0 or 1)
    y = torch.cat([torch.randn(B, T, 2), torch.randint(0, 2, (B, T, 1)).float()], dim=-1)

    # Initialize model
    model = BivariateBernoulliMixtureHead(input_dim=H, n_components=n_components)

    # Forward pass
    means, stdevs, log_weights, correlations, last_logit = model(x)
    print("Means shape:", means.shape)
    print("Stdevs shape:", stdevs.shape)
    print("Log Weights shape:", log_weights.shape)
    print("Correlations shape:", correlations.shape)
    print("Last logit shape:", last_logit.shape)

    # Compute loss
    loss = model.loss(x, y)
    print("Loss shape:", loss.shape)  # Should be a scalar
    print("Loss sample:", loss.item())

    # Sample from the model
    sample = model.sample(x[:,-1:,:])
    print("Sample shape:", sample.shape)  # Should be [B, 1, 3]
    print("Sample example:", sample[0, 0])

    from utils.parse_strokes import parse_strokes
    code = 'a01-000u'
    df = parse_strokes(code)
    y = torch.tensor(df.loc[df['line']==0][['delta_x','delta_y','lift_point']].to_numpy()).unsqueeze(0)
    model.plot_heatmap(y,torch.randn(1, y.size(1), H),save_path=Path(TEST_RESULTS_PATH)/"heatmap.png")

        
