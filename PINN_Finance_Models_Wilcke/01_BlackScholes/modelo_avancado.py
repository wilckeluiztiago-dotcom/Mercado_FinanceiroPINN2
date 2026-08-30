"""
PINN Avançada para Black-Scholes
Arquitetura: Fourier Features + Residual Blocks + Adaptive Loss Weights + Hard Constraint opcional
Baseado nas recomendações dos Capítulos 2, 3 e 25 de Luiz Tiago Wilcke
"""

import sys
sys.path.append("..")
import torch
import torch.nn as nn
from typing import Dict, Tuple
from utils.redes_avancadas import (
    AdvancedPINN, AdaptiveLossWeights, HardConstraintCall,
    DGMNetwork, ResidualAdaptiveSampler, causal_weight
)
from configuracao import BSConfig


class AdvancedBlackScholesLoss:
    """
    Perda composta com:
    - Pesos auto-adaptativos (aprendíveis)
    - Máscara causal temporal
    - Residual Adaptive Refinement ready
    """

    def __init__(self, model: nn.Module, cfg: BSConfig, use_adaptive_weights: bool = True):
        self.model = model
        self.cfg = cfg
        self.use_adaptive = use_adaptive_weights
        if use_adaptive_weights:
            self.adaptive = AdaptiveLossWeights(n_terms=3, init_weights=[1.0, 2.0, 15.0])
        else:
            self.adaptive = None

    def pde_residual(self, S: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        S = S.requires_grad_(True)
        t = t.requires_grad_(True)
        V = self.model(S, t)

        dV_dS = torch.autograd.grad(V, S, grad_outputs=torch.ones_like(V), create_graph=True)[0]
        dV_dt = torch.autograd.grad(V, t, grad_outputs=torch.ones_like(V), create_graph=True)[0]
        d2V_dS2 = torch.autograd.grad(dV_dS, S, grad_outputs=torch.ones_like(dV_dS), create_graph=True)[0]

        residual = (
            dV_dt
            + 0.5 * self.cfg.sigma**2 * S**2 * d2V_dS2
            + self.cfg.r * S * dV_dS
            - self.cfg.r * V
        )
        return residual

    def boundary_loss(self, S: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        V = self.model(S, t)
        mask_low = (S < 1e-3).float()
        mask_high = (S > self.cfg.S_max * 0.95).float()
        target_high = S - self.cfg.K * torch.exp(-self.cfg.r * (self.cfg.T - t))
        return (mask_low * V**2).mean() + (mask_high * (V - target_high)**2).mean()

    def terminal_loss(self, S: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        V = self.model(S, t)
        payoff = torch.relu(S - self.cfg.K)
        return ((V - payoff)**2).mean()

    def total_loss(self, collocation: Dict) -> Tuple[torch.Tensor, Dict]:
        X_int = collocation["interior"]
        X_bc  = collocation["boundary"]
        X_term = collocation["terminal"]

        S_int, t_int = X_int[:, 0:1], X_int[:, 1:2]
        S_bc, t_bc   = X_bc[:, 0:1], X_bc[:, 1:2]
        S_term, t_term = X_term[:, 0:1], X_term[:, 1:2]

        res = self.pde_residual(S_int, t_int)

        # Causal weighting (opcional)
        causal_w = causal_weight(t_int, self.cfg.T)
        loss_pde = (causal_w * res**2).mean()

        loss_bc  = self.boundary_loss(S_bc, t_bc)
        loss_ic  = self.terminal_loss(S_term, t_term)

        if self.use_adaptive and self.adaptive is not None:
            total = self.adaptive.weighted_sum([loss_pde, loss_bc, loss_ic])
            weights = self.adaptive().detach().cpu().numpy()
        else:
            total = (self.cfg.lambda_pde * loss_pde
                     + self.cfg.lambda_bc * loss_bc
                     + self.cfg.lambda_ic * loss_ic)
            weights = [self.cfg.lambda_pde, self.cfg.lambda_bc, self.cfg.lambda_ic]

        components = {
            "pde": loss_pde.item(),
            "bc": loss_bc.item(),
            "ic": loss_ic.item(),
            "total": total.item(),
            "weights": weights
        }
        return total, components


def build_advanced_bs_model(cfg: BSConfig, architecture: str = "residual_fourier") -> nn.Module:
    """
    Factory de arquiteturas avançadas.
    architecture ∈ {"residual_fourier", "dgm", "hard_constraint", "vanilla"}
    """
    if architecture == "residual_fourier":
        net = AdvancedPINN(
            in_dim=2,
            hidden_dim=cfg.hidden_dim,
            n_res_blocks=cfg.hidden_layers,
            fourier_dim=128,
            fourier_scale=12.0,
            use_fourier=True,
            use_residual=True
        )
    elif architecture == "dgm":
        net = DGMNetwork(in_dim=2, hidden_dim=cfg.hidden_dim, n_layers=cfg.hidden_layers)
    elif architecture == "hard_constraint":
        base = AdvancedPINN(in_dim=2, hidden_dim=cfg.hidden_dim, n_res_blocks=4, use_fourier=True)
        net = HardConstraintCall(base, K=cfg.K, T=cfg.T)
    else:
        # fallback vanilla
        layers = [nn.Linear(2, cfg.hidden_dim), nn.Tanh()]
        for _ in range(cfg.hidden_layers - 1):
            layers += [nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(cfg.hidden_dim, 1))
        net = nn.Sequential(*layers)

    return net
