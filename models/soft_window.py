import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftWindow(nn.Module):
    def __init__(self, input_dim, n_components):
        super().__init__()
        self.n_components = n_components
        self.output_dim = 3 * n_components
        self.net = nn.Linear(input_dim, self.output_dim)

    def forward(self, x, c, kappa=None):
        # x: [B, T, H]
        # c: [B, U, embedding_dim]
        # kappa: [B, n_components]

        B, T, _ = x.shape
        Bc, U, embedding_dim = c.shape
        assert B == Bc, "Batch size mismatch between x and c"

        if kappa is None:
            kappa = torch.zeros(B, self.n_components, device=x.device, dtype=x.dtype)

        out = self.net(x)  # [B, T, 3 * n_components]
        alpha_hat, beta_hat, kappa_hat = torch.chunk(out, 3, dim=2)  # Each: [B, T, n_components]

        alpha = torch.exp(alpha_hat).unsqueeze(2)  # [B, T, 1, n_components]
        beta = torch.exp(beta_hat).unsqueeze(2)    # [B, T, 1, n_components]

        kappa = kappa.unsqueeze(1) + torch.cumsum(torch.exp(kappa_hat), dim=1)
        kappa_expanded = kappa.unsqueeze(2)  # [B, T, 1, n_components]

        u = torch.arange(U, device=x.device, dtype=x.dtype).view(1, 1, U, 1)  # [1, 1, U, 1]

        phi = alpha * torch.exp(-beta * ((kappa_expanded - u) ** 2))  # [B, T, U, n_components]
        phi = phi.sum(dim=3)  # [B, T, U]

        w = torch.bmm(phi, c)  # [B, T, embedding_dim]

        return w, kappa[:, -1, :]  # [B, T, embedding_dim], [B, n_components]



if __name__ == "__main__":
    def test_soft_window():
        B, T, H, U, V, N = 2, 5, 16, 10, 20, 3  # batch, time, input_dim, seq_len, embedding_dim, n_components
        x = torch.randn(B, T, H)
        c = torch.randn(B, U, V)
        model = SoftWindow(input_dim=H, n_components=N)

        w, new_kappa = model(x, c)

        assert w.shape == (B, T, V), f"Expected w shape {(B,T,V)}, got {w.shape}"
        assert new_kappa.shape == (B, N), f"Expected kappa shape {(B,N)}, got {new_kappa.shape}"
        print("SoftWindow test passed.")

    test_soft_window()
