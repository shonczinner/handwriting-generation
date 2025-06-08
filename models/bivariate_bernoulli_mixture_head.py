import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Bernoulli

LOG_TWO_PI = math.log(2 * math.pi)

class BivariateBernoulliMixtureHead(nn.Module):
    def __init__(self, input_dim, n_components):
        """
        Args:
            input_dim (int): Input dimension from the network.
            n_components (int): Number of mixture components.
        """
        super().__init__()
        self.n_components = n_components
        self.output_dim = 6 * n_components + 1  # 2 means + 2 stds + 1 weight + 1 corr per component, plus 1 Bernoulli
        self.net = nn.Linear(input_dim, self.output_dim)

    def forward(self, x):
        """
        Args:
            x (Tensor): [B, T, input_dim] — batched RNN output.

        Returns:
            Tuple of tensors:
                means:         [B, T, n_components, 2]
                stdevs:        [B, T, n_components, 2]
                log_weights:   [B, T, n_components]
                correlations:  [B, T, n_components]
                last_prob:     [B, T]
        """
        out = self.net(x)  # [B, T, 6*n + 1]
        return self._parse_raw_outputs(out)

    def _parse_raw_outputs(self, out):
        # out: [B, T, 6*n + 1]
        mixture, last_prob = out[..., :-1], out[..., -1]  # [B, T, 6*n], [B, T]

        means, std_raw, weight_raw, corr_raw = torch.split(
            mixture,
            [2 * self.n_components, 2 * self.n_components, self.n_components, self.n_components],
            dim=-1
        )

        means = means.view(*out.shape[:-1], self.n_components, 2)        # [B, T, C, 2]
        stdevs = torch.exp(std_raw).view(*out.shape[:-1], self.n_components, 2)  # [B, T, C, 2]
        log_weights = F.log_softmax(weight_raw, dim=-1)                 # [B, T, C]
        correlations = torch.tanh(corr_raw).clamp(min=-0.999,max=0.999)                           # [B, T, C]

        return means, stdevs, log_weights, correlations, last_prob

    def loss(self, x, y, lengths=None):
        """
        Computes loss over a sequence, with optional masking for padded positions.

        Args:
            x (Tensor): [B, T, input_dim] — input features.
            y (Tensor): [B, T, 3] — ground truth (dx, dy, lift).
            lengths (Tensor or None): [B] — actual sequence lengths (before padding).
                                    If None, no masking is applied.

        Returns:
            Scalar tensor — average loss.
        """
        B, T, _ = x.shape
        device = x.device

        means, stdevs, log_weights, correlations, last_logit = self.forward(x)

        z = (y[:, :, :2].unsqueeze(2) - means) / stdevs  # [B, T, C, 2]
        log_probs = self._bivariate_gaussian_log_prob(z, stdevs, correlations)  # [B, T, C]

        gaussian_loss = -torch.logsumexp(log_weights + log_probs, dim=-1)  # [B, T]
        bernoulli_loss = F.binary_cross_entropy_with_logits(
            last_logit, y[:, :, 2], reduction='none'
        )  # [B, T]

        total_loss = gaussian_loss + bernoulli_loss  # [B, T]
        assert not torch.isnan(total_loss).any(), "NaN detected in loss computation!"

        if lengths is not None:
            mask = torch.arange(T, device=device).unsqueeze(0) < lengths.unsqueeze(1)  # [B, T]
            total_loss = total_loss * mask  # [B, T]
            return total_loss.sum() / mask.sum()
        else:
            return total_loss.mean()



    def _bivariate_gaussian_log_prob(self, z, stdevs, correlations):
        """
        Computes log-likelihood of z under bivariate Gaussian mixture.

        Args:
            z:            [B, T, C, 2]
            stdevs:       [B, T, C, 2]
            correlations: [B, T, C]

        Returns:
            log_probs:    [B, T, C]
        """
        z1, z2 = z[..., 0], z[..., 1]
        corr_sq = correlations ** 2
        one_minus_corr_sq = 1 - corr_sq

        Z = (z1**2 + z2**2 - 2 * correlations * z1 * z2) / one_minus_corr_sq
        log_det = (
            torch.log(stdevs[..., 0]) +
            torch.log(stdevs[..., 1]) +
            0.5 * torch.log1p(-corr_sq)
        )

        return -0.5 * Z - log_det - LOG_TWO_PI

    @torch.no_grad()
    def sample(self, x, temperature=1.0):
        """
        Samples a single output from the mixture model.
        Args:
            x (Tensor): [1, 1, H] — hidden state from RNN.
            temperature (float): Controls sampling randomness.
        Returns:
            Tensor: [3] — (delta_x, delta_y, lift_point)
        """
        device = x.device
        x = x.squeeze(0).squeeze(0)  # [H]

        out = self.net(x)  # [6*n + 1]
        means, stdevs, log_weights, correlations, last_logit = self._parse_raw_outputs(out.unsqueeze(0).unsqueeze(0))

        means = means.squeeze(0).squeeze(0)             # [C, 2]
        stdevs = stdevs.squeeze(0).squeeze(0) * temperature  # [C, 2]
        log_weights = log_weights.squeeze(0).squeeze(0) / temperature  # [C]
        correlations = correlations.squeeze(0).squeeze(0)  # [C]

        weights = torch.softmax(log_weights, dim=-1)
        idx = torch.distributions.Categorical(weights).sample()

        mean = means[idx]
        stdev = stdevs[idx]
        corr = correlations[idx]

        gaussian_sample = self._sample_bivariate_gaussian(mean, stdev, corr)
        bernoulli_sample = Bernoulli(logits=last_logit.squeeze().to(device) / temperature).sample()

        return torch.cat([gaussian_sample, bernoulli_sample.unsqueeze(0)], dim=0)


    @staticmethod
    def _build_covariance_matrix(stdevs, correlation):
        """
        Builds 2x2 covariance matrix from stdevs and correlation.

        Args:
            stdevs (Tensor): [2] — standard deviations.
            correlation (Tensor): scalar — correlation coefficient.

        Returns:
            Tensor: [2, 2] covariance matrix.
        """
        sx, sy = stdevs[0], stdevs[1]
        rho = correlation
        cov = torch.stack([
            torch.stack([sx**2, rho * sx * sy], dim=-1),
            torch.stack([rho * sx * sy, sy**2], dim=-1)
        ], dim=-2)
        return cov
    
    def _sample_bivariate_gaussian(self, mean, stdev, rho):
        """
        ONNX-traceable sampling from a bivariate Gaussian using reparameterization.

        Args:
            mean (Tensor): [2] — mean vector.
            stdev (Tensor): [2] — standard deviations.
            rho (Tensor): scalar — correlation coefficient.

        Returns:
            Tensor: [2] — sample from the bivariate normal.
        """
        eps = torch.randn(2, device=mean.device)
        eps1, eps2 = eps[0], eps[1]

        sx, sy = stdev[0], stdev[1]
        x = sx * eps1
        y = sy * (rho * eps1 + torch.sqrt(1 - rho**2 + 1e-6) * eps2)

        return mean + torch.stack([x, y])
