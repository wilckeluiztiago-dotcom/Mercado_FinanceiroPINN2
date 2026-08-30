"""
PINN completa e funcional – Hull-White
Inspirado no livro de Luiz Tiago Wilcke (Cap. 18)
Rede com Fourier features + residual blocks, perda composta, autograd de 2ª ordem.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, List

class FourierEmbedding(nn.Module):
    def __init__(self, in_dim, embed_dim=64, scale=10.0):
        super().__init__()
        self.register_buffer("B", torch.randn(in_dim, embed_dim) * scale)
        self.out_dim = embed_dim * 2
    def forward(self, x):
        proj = 2 * np.pi * x @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.Tanh()
    def forward(self, x):
        return self.act(self.norm(x + self.fc2(self.act(self.fc1(x)))))

class PINN(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=128, n_blocks=5, use_fourier=True):
        super().__init__()
        self.use_fourier = use_fourier
        if use_fourier:
            self.embed = FourierEmbedding(in_dim, 64)
            d0 = self.embed.out_dim
        else:
            d0 = in_dim
        self.proj = nn.Linear(d0, hidden_dim)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(n_blocks)])
        self.out = nn.Linear(hidden_dim, 1)
        self.act = nn.Tanh()
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.7)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, *args):
        x = torch.cat(args, dim=-1)
        if self.use_fourier:
            x = self.embed(x)
        h = self.act(self.proj(x))
        for b in self.blocks:
            h = b(h)
        return self.out(h)

class PhysicsLoss:
    def __init__(self, model, params: Dict):
        self.model = model
        self.p = params

    def residual(self, *inputs):
        r, t = inputs[0], inputs[1]
        P = self.model(r, t)
        Pr = torch.autograd.grad(P, r, torch.ones_like(P), create_graph=True)[0]
        Pt = torch.autograd.grad(P, t, torch.ones_like(P), create_graph=True)[0]
        Prr = torch.autograd.grad(Pr, r, torch.ones_like(Pr), create_graph=True)[0]
        a = self.p.get("a", 0.1); sig = self.p.get("sigma", 0.01)
        # theta(t) simple constant for demo; real version uses network
        theta = self.p.get("theta", 0.03)
        res = Pt + (theta - a*r)*Pr + 0.5*sig**2 * Prr - r*P

        return res

    def terminal_loss(self, *inputs):
        V = self.model(*inputs)
        S = inputs[0]
        K = self.p.get("K", 100.0)
        return ((V - torch.relu(S - K))**2).mean()

    def total_loss(self, col: Dict) -> Tuple[torch.Tensor, Dict]:
        Xi = col["interior"]
        cols = [Xi[:, i:i+1].requires_grad_(True) for i in range(Xi.shape[1])]
        res = self.residual(*cols)
        loss_pde = (res**2).mean()
        loss_ic = torch.tensor(0.0, device=Xi.device)
        if "terminal" in col:
            Xt = col["terminal"]
            tcols = [Xt[:, i:i+1] for i in range(Xt.shape[1])]
            loss_ic = self.terminal_loss(*tcols)
        total = self.p.get("lambda_pde", 1.0) * loss_pde + self.p.get("lambda_ic", 12.0) * loss_ic
        return total, {"pde": loss_pde.item(), "ic": loss_ic.item(), "total": total.item()}

def make_collocation(bounds, n_int, n_term, T, device):
    from utils.amostragem import LatinHypercubeSampler
    Xi = LatinHypercubeSampler(bounds).sample_torch(n_int, device=device)
    Xx = LatinHypercubeSampler(bounds[:-1]).sample_torch(n_term, device=device)
    Xt = torch.cat([Xx, torch.full((n_term,1), T, device=device)], 1).requires_grad_(True)
    return {"interior": Xi, "terminal": Xt}
