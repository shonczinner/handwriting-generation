import torch
import torch.nn as nn

import torch
import torch.nn as nn

import torch
import torch.nn as nn


class SoftWindow(nn.Module):
    def __init__(self, input_dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.linear = nn.Linear(input_dim, 3 * n_heads)

    def forward(self, x, c, hidden=None, ascii_lengths=None):
        """
        Args:
            x: [B, 1, H] - input at current time step
            c: [B, U, V] - character sequence (context)
            hidden: [B, n_heads] - previous attention centers (optional), phi_termination
            ascii_lengths: [B] - actual lengths of character sequences (optional)
        Returns:
            w: [B, 1, V] - weighted sum of context (window)
            hidden:, new_kappa: [B, n_heads] - updated attention centers and phi_termination
        """

        B, T, H = x.shape
        assert T == 1, "SoftWindow only supports a single time step (T=1)"
        Bc, U, V = c.shape
        assert B == Bc, "Batch size mismatch between input x and context c"

        # Initialize kappa if not provided
        if hidden is None:
            kappa = torch.zeros(B, self.n_heads, device=x.device, dtype=x.dtype)
        else:
            kappa = hidden[0]

        # Compute attention parameters: alpha, beta, kappa
        linear_out = self.linear(x).squeeze(1)  # [B, 3 * n_heads]
        alpha_hat, beta_hat, kappa_hat = linear_out.chunk(3, dim=-1)  # Each: [B, n_heads]

        alpha = torch.exp(alpha_hat)             # [B, n_heads]
        beta = torch.exp(beta_hat)               # [B, n_heads]
        new_kappa = kappa + torch.exp(kappa_hat-3.9) # [B, n_heads] #-3.9 makes window steps start smaller

        # Prepare character positions u: [1, U, 1]
        u = torch.arange(U+1, device=x.device, dtype=x.dtype).view(1, U+1, 1)

        # Broadcast parameters to shape [B, U, n_heads]
        kappa_exp = new_kappa.unsqueeze(1)  # [B, 1, n_heads]
        beta_exp = beta.unsqueeze(1)        # [B, 1, n_heads]
        alpha_exp = alpha.unsqueeze(1)      # [B, 1, n_heads]

        # Compute attention weights φ: [B, U+1, n_heads]
        phi_components = alpha_exp * torch.exp(-beta_exp * (kappa_exp - u) ** 2)

        # Sum over components: [B, U] → [B, 1, U+1]
        phi_termination = phi_components.sum(dim=-1).unsqueeze(1)
        phi = phi_termination[:,:,:-1]

        # Optional masking for padded characters
        if ascii_lengths is not None:
            mask = torch.arange(U, device=x.device).unsqueeze(0).expand(B, U)
            mask = mask < ascii_lengths.unsqueeze(1)  # [B, U]
            mask = mask.unsqueeze(1)  # [B, 1, U]
            phi = phi.masked_fill(~mask, 0.0)

        # Compute window: [B, 1, U] @ [B, U, V] → [B, 1, V]
        w = torch.bmm(phi, c)

        return w, (new_kappa,phi_termination)



if __name__ == "__main__":
    def test_soft_window():
        B, T, H, U, V, N = 2, 1, 16, 10, 20, 3  # batch, time, input_dim, seq_len, embedding_dim, n_heads
        x = torch.randn(B, T, H)
        c = torch.randn(B, U, V)
        model = SoftWindow(input_dim=H, n_heads=N)

        w, new_kappa = model(x, c)

        assert w.shape == (B, T, V), f"Expected w shape {(B,T,V)}, got {w.shape}"
        assert new_kappa[0].shape == (B, N), f"Expected kappa shape {(B,N)}, got {new_kappa[0].shape}"
        print("SoftWindow test passed.")

    test_soft_window()
