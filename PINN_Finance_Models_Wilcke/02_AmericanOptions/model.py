"""
PINN completa para Opções Americanas – Inequação Variacional + Penalização Contínua
Formulação: Luiz Tiago Wilcke – Capítulo 4
"""
import torch
import torch.nn as nn
from typing import Dict, Tuple

class AmericanPINN(nn.Module):
    def __init__(self, hidden_layers=7, hidden_dim=160):
        super().__init__()
        layers = [nn.Linear(2, hidden_dim), nn.Tanh()]
        for _ in range(hidden_layers-1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.8)
                nn.init.zeros_(m.bias)

    def forward(self, S, t):
        x = torch.cat([S/100.0, t], dim=-1)
        return self.net(x)

class AmericanLoss:
    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg

    def residual_with_penalty(self, S, t):
        S = S.requires_grad_(True)
        t = t.requires_grad_(True)
        V = self.model(S, t)
        dV_dS = torch.autograd.grad(V, S, torch.ones_like(V), create_graph=True)[0]
        dV_dt = torch.autograd.grad(V, t, torch.ones_like(V), create_graph=True)[0]
        d2V_dS2 = torch.autograd.grad(dV_dS, S, torch.ones_like(dV_dS), create_graph=True)[0]
        L = dV_dt + 0.5*self.cfg.sigma**2 * S**2 * d2V_dS2 + (self.cfg.r - self.cfg.q)*S*dV_dS - self.cfg.r * V
        payoff = torch.relu(S - self.cfg.K)
        pen = self.cfg.lambda_penalty * torch.relu(payoff - V)**2
        return L + pen

    def total_loss(self, collocation):
        X = collocation["interior"]
        S, t = X[:,0:1], X[:,1:2]
        res = self.residual_with_penalty(S, t)
        loss_pde = (res**2).mean()
        # terminal
        Xt = collocation["terminal"]
        Vt = self.model(Xt[:,0:1], Xt[:,1:2])
        payoff = torch.relu(Xt[:,0:1] - self.cfg.K)
        loss_ic = ((Vt - payoff)**2).mean()
        total = loss_pde + 15.0 * loss_ic
        return total, {"pde": loss_pde.item(), "ic": loss_ic.item(), "total": total.item()}
