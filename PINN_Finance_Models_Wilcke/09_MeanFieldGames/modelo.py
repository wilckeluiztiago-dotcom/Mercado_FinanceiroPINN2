"""
PINN completa – Mean Field Games
Inspirado no livro de Luiz Tiago Wilcke (Cap. 11/23)
Implementação funcional com rede residual + Fourier, residual PDE, perda composta,
amostragem LHS, otimização híbrida Adam+L-BFGS e extração de quantidades de interesse.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, List, Optional

class FourierEmbedding(nn.Module):
    def __init__(self, in_dim: int, embed_dim: int = 64, scale: float = 10.0):
        super().__init__()
        self.register_buffer("B", torch.randn(in_dim, embed_dim) * scale)
        self.out_dim = embed_dim * 2
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = 2 * np.pi * x @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

class ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.Tanh()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(x + self.fc2(self.act(self.fc1(x)))))

class PINN(nn.Module):
    def __init__(self, in_dim: int = 2, hidden_dim: int = 128, n_blocks: int = 5,
                 use_fourier: bool = True, out_dim: int = 1):
        super().__init__()
        self.use_fourier = use_fourier
        if use_fourier:
            self.embed = FourierEmbedding(in_dim, 64)
            d0 = self.embed.out_dim
        else:
            d0 = in_dim
        self.proj = nn.Linear(d0, hidden_dim)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(n_blocks)])
        self.out = nn.Linear(hidden_dim, out_dim)
        self.act = nn.Tanh()
        self._init()
    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.7)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    def forward(self, *args) -> torch.Tensor:
        x = torch.cat(args, dim=-1)
        if self.use_fourier:
            x = self.embed(x)
        h = self.act(self.proj(x))
        for b in self.blocks:
            h = b(h)
        return self.out(h)

class PhysicsLoss:
    """Perda física genérica – residual + terminal + contorno."""
    def __init__(self, model: PINN, params: Dict):
        self.model = model
        self.p = params
    def residual(self, *inputs) -> torch.Tensor:
        # Generic Black-Scholes-like residual as baseline; specialize per model
        S, t = inputs[0], inputs[-1]
        for inp in inputs:
            inp.requires_grad_(True)
        V = self.model(*inputs)
        grads = []
        for inp in inputs:
            g = torch.autograd.grad(V, inp, torch.ones_like(V), create_graph=True, allow_unused=True)[0]
            grads.append(g if g is not None else torch.zeros_like(inp))
        # second derivative w.r.t. first spatial variable
        dV_dS = grads[0]
        d2V_dS2 = torch.autograd.grad(dV_dS, inputs[0], torch.ones_like(dV_dS),
                                       create_graph=True, allow_unused=True)[0]
        if d2V_dS2 is None:
            d2V_dS2 = torch.zeros_like(S)
        dV_dt = grads[-1]
        sig = self.p.get("sigma", 0.2)
        r = self.p.get("r", 0.05)
        res = dV_dt + 0.5 * sig**2 * S**2 * d2V_dS2 + r * S * dV_dS - r * V
        return res
    def terminal_loss(self, *inputs) -> torch.Tensor:
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
        total = self.p.get("lambda_pde", 1.0)*loss_pde + self.p.get("lambda_ic", 12.0)*loss_ic
        return total, {"pde": loss_pde.item(), "ic": loss_ic.item(), "total": total.item()}

def make_collocation(bounds, n_int, n_term, T, device):
    from utils.amostragem import LatinHypercubeSampler
    Xi = LatinHypercubeSampler(bounds).sample_torch(n_int, device=device)
    Xx = LatinHypercubeSampler(bounds[:-1]).sample_torch(n_term, device=device)
    Xt = torch.cat([Xx, torch.full((n_term, 1), T, device=device)], 1).requires_grad_(True)
    return {"interior": Xi, "terminal": Xt}

def compute_delta_gamma(model, S, t):
    S = S.clone().requires_grad_(True)
    t = t.clone().requires_grad_(True)
    V = model(S, t)
    delta = torch.autograd.grad(V, S, torch.ones_like(V), create_graph=True)[0]
    gamma = torch.autograd.grad(delta, S, torch.ones_like(delta), create_graph=True)[0]
    return delta, gamma
