"""
Modelo PINN para a EDP de Black-Scholes (Call Europeia).
Formulação e arquitetura inspiradas nos Capítulos 1 e 3 de Luiz Tiago Wilcke.
"""

import torch
import torch.nn as nn
import math
from typing import Tuple


class FourierFeatureEmbedding(nn.Module):
    """Embedding de Fourier para melhorar a representação de altas frequências (opcional)."""
    def __init__(self, in_dim: int, embed_dim: int = 64, scale: float = 10.0):
        super().__init__()
        self.B = nn.Parameter(torch.randn(in_dim, embed_dim) * scale, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = 2 * math.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class BlackScholesPINN(nn.Module):
    """
    Rede neural totalmente conectada com ativação tanh (C^∞).
    Entrada: (S, t)  →  Saída: V(S,t)
    """

    def __init__(
        self,
        hidden_layers: int = 6,
        hidden_dim: int = 128,
        use_fourier: bool = False
    ):
        super().__init__()
        self.use_fourier = use_fourier
        if use_fourier:
            self.embed = FourierFeatureEmbedding(2, 32)
            in_dim = 64
        else:
            in_dim = 2

        layers = [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

        # Inicialização Xavier (boa prática para PINNs)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, S: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = torch.cat([S, t], dim=-1)
        if self.use_fourier:
            x = self.embed(x)
        return self.net(x)


class BlackScholesLoss:
    """
    Função de perda composta (PDE + contorno + terminal)
    conforme formulação do Capítulo 3 de Wilcke.
    """

    def __init__(self, model: BlackScholesPINN, config):
        self.model = model
        self.cfg = config
        self.device = config.device

    def pde_residual(self, S: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Resíduo da EDP de Black-Scholes via autograd."""
        S = S.requires_grad_(True)
        t = t.requires_grad_(True)
        V = self.model(S, t)

        # Derivadas de primeira ordem
        dV_dS = torch.autograd.grad(V, S, grad_outputs=torch.ones_like(V), create_graph=True)[0]
        dV_dt = torch.autograd.grad(V, t, grad_outputs=torch.ones_like(V), create_graph=True)[0]

        # Segunda derivada espacial
        d2V_dS2 = torch.autograd.grad(dV_dS, S, grad_outputs=torch.ones_like(dV_dS), create_graph=True)[0]

        # Operador de Black-Scholes
        residual = (
            dV_dt
            + 0.5 * self.cfg.sigma**2 * S**2 * d2V_dS2
            + self.cfg.r * S * dV_dS
            - self.cfg.r * V
        )
        return residual

    def boundary_loss(self, S: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Condições de contorno em S=0 e S→∞."""
        V = self.model(S, t)
        # Em S=0 → V=0; em S grande → V ≈ S - K e^{-r(T-t)}
        mask_low = (S < 1e-3).float()
        mask_high = (S > self.cfg.S_max * 0.95).float()
        target_high = S - self.cfg.K * torch.exp(-self.cfg.r * (self.cfg.T - t))
        loss = (mask_low * V**2).mean() + (mask_high * (V - target_high)**2).mean()
        return loss

    def terminal_loss(self, S: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Payoff no vencimento."""
        V = self.model(S, t)
        payoff = torch.relu(S - self.cfg.K)
        return ((V - payoff)**2).mean()

    def total_loss(self, collocation: dict) -> Tuple[torch.Tensor, dict]:
        X_int = collocation["interior"]
        X_bc = collocation["boundary"]
        X_term = collocation["terminal"]

        S_int, t_int = X_int[:, 0:1], X_int[:, 1:2]
        S_bc, t_bc = X_bc[:, 0:1], X_bc[:, 1:2]
        S_term, t_term = X_term[:, 0:1], X_term[:, 1:2]

        res = self.pde_residual(S_int, t_int)
        loss_pde = (res**2).mean()
        loss_bc = self.boundary_loss(S_bc, t_bc)
        loss_ic = self.terminal_loss(S_term, t_term)

        total = (
            self.cfg.lambda_pde * loss_pde
            + self.cfg.lambda_bc * loss_bc
            + self.cfg.lambda_ic * loss_ic
        )
        components = {
            "pde": loss_pde.item(),
            "bc": loss_bc.item(),
            "ic": loss_ic.item(),
            "total": total.item()
        }
        return total, components


def analytical_call(S, K, T, t, r, sigma):
    """Solução analítica de Black-Scholes para validação."""
    from scipy.stats import norm
    import numpy as np
    tau = T - t
    if tau <= 0:
        return np.maximum(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    return S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)
