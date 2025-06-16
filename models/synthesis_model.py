import torch
import torch.nn as nn
import torch.nn.functional as F
from models.bivariate_bernoulli_mixture_head import BivariateBernoulliMixtureHead
from models.soft_window import SoftWindow

class SynthesisModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.num_layers = config.num_layers
        self.hidden_dim = config.hidden_dim
        self.n_heads = config.n_heads
        self.softwindow = SoftWindow(config.hidden_dim, config.n_heads)

        self.rnns = []

        for n in range(config.num_layers):
            if n==0:
                input_dim = config.input_dim+config.vocab_size
            else:
                input_dim = config.input_dim+config.hidden_dim+config.vocab_size


            rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=config.hidden_dim,
                num_layers=1,
                batch_first=True
            )
            self.rnns.append(rnn)
        
        self.rnns = nn.ModuleList(self.rnns)

        self.head = BivariateBernoulliMixtureHead(config.hidden_dim*config.num_layers, config.n_components)

    def get_initial_hidden_state(self, batch_size):
        # Instead of None, create zero tensors with correct shape:
        hidden_core = [torch.zeros(1, batch_size, self.hidden_dim) for _ in range(self.num_layers)]
        kappa = torch.zeros(batch_size, self.n_heads)
        w0 = torch.zeros(batch_size, 1, self.vocab_size)
        return hidden_core, kappa, w0


    def network(self, x, c, hidden=None,ascii_lengths=None):
        # hidden: (hidden1, kappa, w0) or None
        # x: [B, T, H] - input 
        # c: [B, U] - character sequence (context)
        B, T,_ = x.shape
        if hidden is None:
            hidden, kappa, w0 = self.get_initial_hidden_state(x.size(0))
            hidden = [h.to(x.device) for h in hidden]
            kappa = kappa.to(x.device)
            w0 = w0.to(x.device) 
        else:
            hidden, kappa, w0 = hidden

        new_hidden = []

        c_emb = F.one_hot(c,num_classes=self.vocab_size).type(x.dtype)


        out = []
        w = []
        for t in range(T):
            out_t, h_t = self.rnns[0](torch.cat((x[:,t:t+1],w0),dim=2), hidden[0])
            w0, kappa,phi_termination = self.softwindow(out_t,c_emb,kappa,ascii_lengths)         
            hidden[0]=h_t
            out.append(out_t)
            w.append(w0)

        out = torch.cat(out,dim=1)
        w = torch.cat(w, dim=1)

        inputs = [out]
        new_hidden.append(hidden[0])

        for i, rnn in enumerate(self.rnns[1:], start=1):
            out, h = rnn(torch.cat((x,out,w),dim=2), hidden[i])   
            new_hidden.append(h)
            inputs.append(out)


        new_hidden = (new_hidden, kappa, w0)
        return torch.concat(inputs,dim=-1), new_hidden,phi_termination

    def forward(self, x, c, hidden=None):
        out, hidden,phi_termination = self.network(x, c, hidden)
        means, stdevs, log_weights, correlations, last_logit = self.head(out)
        return means, stdevs, log_weights, correlations, last_logit, hidden,phi_termination

    def loss(self, x, c, y, lengths = None, ascii_lengths=None,hidden=None):
        out, hidden,_ = self.network(x, c, hidden, ascii_lengths)
        loss = self.head.loss(out, y, lengths)
        return loss, hidden

    @torch.no_grad()
    def get_primed_hidden(self, prime_x, c, hidden=None):
        _, hidden,_ = self.network(prime_x, c, hidden)
        return hidden
    
    @torch.no_grad()
    def sample(self, x, c, hidden=None, temperature = 1):
        out, hidden,phi_termination = self.network(x, c, hidden)
        sample = self.head.sample(out ,temperature=temperature)
        return sample, hidden,phi_termination

    @torch.no_grad()
    def full_sample(self, ascii, device,  hidden = None, temperature = 1.0,max_length=1000):
        self.eval()
        
        start = torch.zeros((1, 1, 3), dtype=torch.float32).to(device)  # B, T, F

        U = ascii.shape[1]

        phis = []
        generated = start
        for _ in range(max_length):
            sample, hidden,phi_termination = self.sample(generated[:, -1:],ascii, hidden, temperature=temperature)
            generated = torch.cat((generated, sample.unsqueeze(0).unsqueeze(0)), dim=1)
            # termination condition
            if phi_termination[:,:,U]==phi_termination.max():
                break
            phis.append(phi_termination)

        return generated,phis


# ---------------------------------------------------------------
# Test block
# ---------------------------------------------------------------
if __name__ == "__main__":
    from types import SimpleNamespace

    # Dummy config
    config = SimpleNamespace(
        input_dim=10,
        hidden_dim=32,
        num_layers=3,
        n_components=10,
        n_heads=5,
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
    means, stdevs, log_weights, correlations, last_logit, hidden,phi_termination = model(x, c)
    print("Forward pass:")
    print("Means shape:", means.shape)
    print("Stdevs shape:", stdevs.shape)
    print("Log Weights shape:", log_weights.shape)
    print("Correlations shape:", correlations.shape)
    print("Last Logit shape:", last_logit.shape)
    print("phi shape", phi_termination.shape)

    # --- Loss computation ---
    loss, hidden = model.loss(x, c, y, torch.tensor([seq_len]*batch_size))
    print("\nLoss computation:")
    print("Loss:", loss.item())

    # --- Sampling ---
    sample_input = torch.randn(1, 1, config.input_dim)
    c_sample = torch.randint(0, config.vocab_size, (1, cond_len))
    sample, _,_ = model.sample(sample_input, c_sample)
    print("\nSampling:")
    print("Sample shape:", sample.shape)


    def test_stepwise_vs_full_forward(model, x, c, tolerance=1e-5):
        model.eval()
        batch_size, seq_len, input_dim = x.shape

        # --- Full-sequence forward pass ---
        full_out, full_hidden,phi_termination = model.network(x, c)

        # --- Step-by-step forward pass ---
        stepwise_outputs = []
        hidden = None
        for t in range(seq_len):
            out_t, hidden, phi_termination = model.network(x[:, t:t+1], c, hidden)
            stepwise_outputs.append(out_t)

        stepwise_out = torch.cat(stepwise_outputs, dim=1)

        # --- Compare outputs ---
        if not torch.allclose(full_out, stepwise_out, atol=tolerance):
            diff = (full_out - stepwise_out).abs().max()
            print(f"[❌] Stepwise and full outputs differ. Max diff: {diff.item():.6f}")
        else:
            print("[✅] Stepwise and full outputs match.")

        # Optional: return tensors for inspection
        return full_out, stepwise_out

    test_stepwise_vs_full_forward(model, x, c)