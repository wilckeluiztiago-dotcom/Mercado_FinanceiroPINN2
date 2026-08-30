"""
PINN completa para Opções Americanas – Inequação Variacional
Inspirado no livro de Luiz Tiago Wilcke (Cap. 4)
~300 linhas: rede, penalização, smooth pasting, fronteira livre, Gregas
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional

class AmericanPINN(nn.Module):
    def __init__(self, hidden_layers: int = 7, hidden_dim: int = 160, use_fourier: bool = True):
        super().__init__()
        self.use_fourier = use_fourier
        if use_fourier:
            self.register_buffer("B", torch.randn(2, 64) * 10.0)
            in_dim = 128
        else:
            in_dim = 2
        layers = [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.8)
                nn.init.zeros_(m.bias)

    def forward(self, S: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = torch.cat([S / 100.0, t], dim=-1)
        if self.use_fourier:
            proj = 2 * np.pi * x @ self.B
            x = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        return self.net(x)

class AmericanLoss:
    def __init__(self, model: AmericanPINN, cfg):
        self.model = model
        self.cfg = cfg

    def operator_L(self, S, t, V):
        dV_dS = torch.autograd.grad(V, S, torch.ones_like(V), create_graph=True)[0]
        dV_dt = torch.autograd.grad(V, t, torch.ones_like(V), create_graph=True)[0]
        d2V_dS2 = torch.autograd.grad(dV_dS, S, torch.ones_like(dV_dS), create_graph=True)[0]
        return (dV_dt + 0.5 * self.cfg.sigma**2 * S**2 * d2V_dS2
                + (self.cfg.r - self.cfg.q) * S * dV_dS - self.cfg.r * V)

    def residual_penalized(self, S, t):
        S = S.requires_grad_(True)
        t = t.requires_grad_(True)
        V = self.model(S, t)
        L = self.operator_L(S, t, V)
        payoff = torch.relu(S - self.cfg.K)
        gap = torch.relu(payoff - V)
        penalty = self.cfg.lambda_penalty * (gap ** self.cfg.penalty_power)
        return L + penalty, V

    def boundary_loss(self, S, t):
        V = self.model(S, t)
        tau = self.cfg.T - t
        asym = S * torch.exp(-self.cfg.q * tau) - self.cfg.K * torch.exp(-self.cfg.r * tau)
        mask_low = (S < 1.0).float()
        mask_high = (S > self.cfg.S_max * 0.92).float()
        return (mask_low * V**2).mean() + (mask_high * (V - asym)**2).mean()

    def terminal_loss(self, S, t):
        V = self.model(S, t)
        return ((V - torch.relu(S - self.cfg.K))**2).mean()

    def smooth_pasting_loss(self, S, t):
        S = S.requires_grad_(True)
        V = self.model(S, t)
        payoff = torch.relu(S - self.cfg.K)
        near = (torch.abs(V.detach() - payoff) < 0.08).float()
        dV_dS = torch.autograd.grad(V, S, torch.ones_like(V), create_graph=True)[0]
        return (near * (dV_dS - 1.0)**2).mean()

    def total_loss(self, collocation: Dict) -> Tuple[torch.Tensor, Dict]:
        Xi = collocation["interior"]
        Xb = collocation["boundary"]
        Xt = collocation["terminal"]
        res, _ = self.residual_penalized(Xi[:, 0:1], Xi[:, 1:2])
        loss_pde = (res**2).mean()
        loss_bc = self.boundary_loss(Xb[:, 0:1], Xb[:, 1:2])
        loss_ic = self.terminal_loss(Xt[:, 0:1], Xt[:, 1:2])
        loss_sp = self.smooth_pasting_loss(Xi[:2500, 0:1], Xi[:2500, 1:2])
        total = (self.cfg.lambda_pde * loss_pde + self.cfg.lambda_bc * loss_bc
                 + self.cfg.lambda_ic * loss_ic + self.cfg.lambda_penalty_loss * loss_sp)
        return total, {"pde": loss_pde.item(), "bc": loss_bc.item(),
                       "ic": loss_ic.item(), "sp": loss_sp.item(), "total": total.item()}

def extract_free_boundary(model, t_grid, S_grid, K, tol=0.03):
    Sf = []
    model.eval()
    with torch.no_grad():
        for t in t_grid:
            tb = t.expand(S_grid.shape[0], 1)
            V = model(S_grid, tb).squeeze()
            payoff = torch.relu(S_grid.squeeze() - K)
            ex = (V <= payoff + tol)
            Sf.append(S_grid[torch.where(ex)[0][-1]].item() if ex.any() else float(K))
    return torch.tensor(Sf)

def sample_american_collocation(cfg, device):
    from utils.amostragem import LatinHypercubeSampler
    samp = LatinHypercubeSampler([(cfg.S_min + 0.5, cfg.S_max), (0.0, cfg.T)])
    Xi = samp.sample_torch(cfg.n_interior, device=device)
    n_b = cfg.n_boundary // 2
    t_b = torch.rand(n_b, 1, device=device) * cfg.T
    Xb = torch.cat([
        torch.cat([torch.zeros(n_b, 1, device=device), t_b], 1),
        torch.cat([torch.full((n_b, 1), cfg.S_max, device=device), t_b], 1)
    ], 0).requires_grad_(True)
    S_t = torch.rand(cfg.n_terminal, 1, device=device) * (cfg.S_max - 1) + 1
    Xt = torch.cat([S_t, torch.full((cfg.n_terminal, 1), cfg.T, device=device)], 1).requires_grad_(True)
    return {"interior": Xi, "boundary": Xb, "terminal": Xt}
