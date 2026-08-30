"""
PINN bidimensional completa para o Modelo de Heston
Formulação EDP + Gregas + condições de contorno: Luiz Tiago Wilcke – Capítulo 5
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple

class HestonPINN(nn.Module):
    def __init__(self, hidden_layers=8, hidden_dim=180):
        super().__init__()
        layers = [nn.Linear(3, hidden_dim), nn.Tanh()]
        for _ in range(hidden_layers-1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.7)
                nn.init.zeros_(m.bias)

    def forward(self, S, v, t):
        # Normalização
        x = torch.cat([S/100.0, v/0.1, t], dim=-1)
        return self.net(x)

class HestonLoss:
    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg

    def residual(self, S, v, t):
        S = S.requires_grad_(True)
        v = v.requires_grad_(True)
        t = t.requires_grad_(True)
        V = self.model(S, v, t)

        dV_dS = torch.autograd.grad(V, S, torch.ones_like(V), create_graph=True)[0]
        dV_dv = torch.autograd.grad(V, v, torch.ones_like(V), create_graph=True)[0]
        dV_dt = torch.autograd.grad(V, t, torch.ones_like(V), create_graph=True)[0]
        d2V_dS2 = torch.autograd.grad(dV_dS, S, torch.ones_like(dV_dS), create_graph=True)[0]
        d2V_dv2 = torch.autograd.grad(dV_dv, v, torch.ones_like(dV_dv), create_graph=True)[0]
        d2V_dSdv = torch.autograd.grad(dV_dS, v, torch.ones_like(dV_dS), create_graph=True)[0]

        c = self.cfg
        res = (dV_dt
               + 0.5 * v * S**2 * d2V_dS2
               + c.rho * c.xi * v * S * d2V_dSdv
               + 0.5 * c.xi**2 * v * d2V_dv2
               + (c.r - c.q) * S * dV_dS
               + c.kappa * (c.theta - v) * dV_dv
               - c.r * V)
        return res

    def terminal_loss(self, S, v, t):
        V = self.model(S, v, t)
        payoff = torch.relu(S - self.cfg.K)
        return ((V - payoff)**2).mean()

    def boundary_loss(self, S, v, t):
        V = self.model(S, v, t)
        # v=0: redução a BS com vol=0 ; S=0: V=0 ; S grande: asymptotic
        loss = 0.0
        mask_S0 = (S < 1.0).float()
        loss = loss + (mask_S0 * V**2).mean()
        return loss

    def total_loss(self, collocation):
        X = collocation["interior"]
        S, v, t = X[:,0:1], X[:,1:2], X[:,2:3]
        res = self.residual(S, v, t)
        loss_pde = (res**2).mean()
        # terminal
        Xt = collocation["terminal"]
        loss_ic = self.terminal_loss(Xt[:,0:1], Xt[:,1:2], Xt[:,2:3])
        total = self.cfg.lambda_pde * loss_pde + self.cfg.lambda_ic * loss_ic
        return total, {"pde": loss_pde.item(), "ic": loss_ic.item(), "total": total.item()}

def sample_heston_collocation(cfg, device):
    from utils.sampling import LatinHypercubeSampler
    sampler = LatinHypercubeSampler([(0.5, cfg.S_max), (1e-4, cfg.v_max), (0.0, cfg.T)])
    X_int = sampler.sample_torch(cfg.n_interior, device=device)
    # terminal
    sampler_t = LatinHypercubeSampler([(0.5, cfg.S_max), (1e-4, cfg.v_max)])
    Xy = sampler_t.sample_torch(cfg.n_terminal, device=device)
    t_term = torch.full((cfg.n_terminal,1), cfg.T, device=device)
    X_term = torch.cat([Xy, t_term], dim=1).requires_grad_(True)
    return {"interior": X_int, "terminal": X_term}
