import torch
import torch.nn as nn


class Ens(nn.Module):
    """Ensemble of K MLP dynamics models predicting delta-state."""

    def __init__(self, K=5, din=4, dout=2, hid=64, depth=3):
        super().__init__()
        self.K = K
        dims = [din] + [hid] * depth + [dout]
        self.W = nn.ParameterList()
        self.b = nn.ParameterList()
        for i in range(len(dims) - 1):
            w = torch.randn(K, dims[i], dims[i + 1]) * (2.0 / dims[i]) ** 0.5
            self.W.append(nn.Parameter(w))
            self.b.append(nn.Parameter(torch.zeros(K, 1, dims[i + 1])))

    def forward(self, x, a):
        """x,a: (K,B,*) or (B,*) (broadcast to K). -> next state (K,B,2)."""
        if x.dim() == 2:
            x = x.unsqueeze(0).expand(self.K, -1, -1)
            a = a.unsqueeze(0).expand(self.K, -1, -1)
        h = torch.cat([x, a], -1)
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            h = torch.baddbmm(b, h, w)
            if i < len(self.W) - 1:
                h = torch.relu(h)
        return x + h


def fit(model, X, A, Xn, epochs=300, bs=256, lr=2e-3, seed=0):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = X.shape[0]
    K = model.K
    for ep in range(epochs):
        # bootstrap resampling per member
        idx = torch.randint(0, n, (K, bs), generator=g)
        x, a, y = X[idx], A[idx], Xn[idx]
        loss = ((model(x, a) - y) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return model
