"""
Rede PINN Complexa de Produção
==============================
Arquitetura avançada para EDPs financeiras e de física.

Componentes:
  - Fourier Feature Embedding (multi-scale)
  - Residual / Highway blocks com LayerNorm
  - Self-Attention espacial (opcional)
  - Multi-head output (sistemas acoplados)
  - Adaptive activation (adaptive tanh / swish aprendível)
  - Soft / Hard constraint layers
  - Gradient pathology mitigation (gradient balancing)
  - Causal temporal weighting
  - Residual Adaptive Refinement (RAR) sampler
  - Deep Galerkin (DGM) cells
  - Physics loss com pesos auto-adaptativos

Inspirado em técnicas modernas de PINNs e no livro de Luiz Tiago Wilcke.
"""

from __future__ import annotations
import math
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. Embeddings e ativações
# =============================================================================

class MultiScaleFourierEmbedding(nn.Module):
    """
    Fourier features em várias escalas (baixa + média + alta frequência).
    Ajuda a capturar camadas limite, singularidades e rugosidade.
    """
    def __init__(
        self,
        in_dim: int,
        n_features: int = 128,
        scales: Tuple[float, ...] = (1.0, 10.0, 50.0),
        learnable: bool = False,
    ):
        super().__init__()
        self.scales = scales
        B_list = []
        for s in scales:
            B = torch.randn(in_dim, n_features // (2 * len(scales))) * s
            B_list.append(B)
        B_cat = torch.cat(B_list, dim=1)
        if learnable:
            self.B = nn.Parameter(B_cat)
        else:
            self.register_buffer("B", B_cat)
        self.out_dim = B_cat.shape[1] * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = 2 * math.pi * x @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class AdaptiveTanh(nn.Module):
    """tanh com coeficiente de inclinação aprendível (Jagtap et al.)."""
    def __init__(self, init_a: float = 1.0):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(init_a))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.a * x)


class AdaptiveSwish(nn.Module):
    def __init__(self, init_beta: float = 1.0):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(init_beta))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.beta * x)


# =============================================================================
# 2. Blocos de rede
# =============================================================================

class ResidualBlock(nn.Module):
    def __init__(self, dim: int, activation: str = "adaptive_tanh", dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        if activation == "adaptive_tanh":
            self.act = AdaptiveTanh()
        elif activation == "swish":
            self.act = AdaptiveSwish()
        else:
            self.act = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.fc1(x)))
        h = self.drop(h)
        h = self.fc2(h)
        return self.act(self.norm2(x + h))


class HighwayBlock(nn.Module):
    """Highway network gate (Srivastava et al.)."""
    def __init__(self, dim: int):
        super().__init__()
        self.H = nn.Linear(dim, dim)
        self.T = nn.Linear(dim, dim)
        self.act = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.H(x))
        t = torch.sigmoid(self.T(x))
        return h * t + x * (1.0 - t)


class DGMCell(nn.Module):
    """
    Célula Deep Galerkin Method (Sirignano & Spiliopoulos, 2018).
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.Z = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.G = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.R = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.H = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.act = nn.Tanh()

    def forward(self, S: torch.Tensor, H_prev: torch.Tensor) -> torch.Tensor:
        X = torch.cat([S, H_prev], dim=-1)
        Z = torch.sigmoid(self.Z(X))
        G = torch.sigmoid(self.G(X))
        R = torch.sigmoid(self.R(X))
        H_hat = self.act(self.H(torch.cat([S, R * H_prev], dim=-1)))
        return (1.0 - G) * H_hat + Z * H_prev


class SpatialAttention(nn.Module):
    """Self-attention leve sobre features (útil em multi-ativo / alta dimensão)."""
    def __init__(self, dim: int, n_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D) -> (B, 1, D)
        x_seq = x.unsqueeze(1)
        out, _ = self.attn(x_seq, x_seq, x_seq)
        return self.norm(x + out.squeeze(1))


# =============================================================================
# 3. Rede principal complexa
# =============================================================================

class ComplexPINN(nn.Module):
    """
    PINN de produção com múltiplos caminhos e técnicas avançadas.

    Parâmetros
    ----------
    in_dim : dimensão de entrada (ex.: 2 para (S,t), 3 para Heston)
    out_dim : dimensão de saída (1 padrão; >1 para multi-head / regimes)
    hidden_dim : largura das camadas
    n_blocks : número de blocos residuais
    use_fourier : ativa Fourier multi-scale
    use_attention : ativa self-attention
    use_highway : usa Highway em vez de Residual puro
    use_dgm : usa células DGM empilhadas
    activation : 'adaptive_tanh' | 'swish' | 'tanh'
    """
    def __init__(
        self,
        in_dim: int = 2,
        out_dim: int = 1,
        hidden_dim: int = 128,
        n_blocks: int = 6,
        use_fourier: bool = True,
        fourier_scales: Tuple[float, ...] = (1.0, 10.0, 40.0),
        use_attention: bool = False,
        use_highway: bool = False,
        use_dgm: bool = False,
        activation: str = "adaptive_tanh",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_fourier = use_fourier
        self.use_dgm = use_dgm

        if use_fourier:
            self.embed = MultiScaleFourierEmbedding(in_dim, n_features=128, scales=fourier_scales)
            d0 = self.embed.out_dim
        else:
            d0 = in_dim

        self.input_proj = nn.Linear(d0, hidden_dim)

        if use_dgm:
            self.dgm_cells = nn.ModuleList(
                [DGMCell(in_dim if not use_fourier else d0, hidden_dim) for _ in range(n_blocks)]
            )
            self.blocks = None
        else:
            Block = HighwayBlock if use_highway else ResidualBlock
            if use_highway:
                self.blocks = nn.ModuleList([HighwayBlock(hidden_dim) for _ in range(n_blocks)])
            else:
                self.blocks = nn.ModuleList(
                    [ResidualBlock(hidden_dim, activation=activation, dropout=dropout)
                     for _ in range(n_blocks)]
                )
            self.dgm_cells = None

        self.attention = SpatialAttention(hidden_dim) if use_attention else None
        self.output = nn.Linear(hidden_dim, out_dim)
        self.act = AdaptiveTanh() if activation == "adaptive_tanh" else nn.Tanh()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.7)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, *args) -> torch.Tensor:
        raw = torch.cat(args, dim=-1)
        x = self.embed(raw) if self.use_fourier else raw
        h = self.act(self.input_proj(x))

        if self.use_dgm and self.dgm_cells is not None:
            for cell in self.dgm_cells:
                h = cell(x, h)
        else:
            for block in self.blocks:
                h = block(h)

        if self.attention is not None:
            h = self.attention(h)

        return self.output(h)


# =============================================================================
# 4. Hard / Soft constraints
# =============================================================================

class HardConstraintOutput(nn.Module):
    """
    Força condições de contorno/terminal por construção.
    Exemplo Call: V = payoff + tau * softplus(network)
    """
    def __init__(self, base_net: nn.Module, constraint_fn: Callable):
        super().__init__()
        self.net = base_net
        self.constraint_fn = constraint_fn

    def forward(self, *args) -> torch.Tensor:
        u = self.net(*args)
        return self.constraint_fn(*args, u)


def european_call_constraint(S, t, u, K: float = 100.0, T: float = 1.0):
    """Hard constraint para call europeia."""
    tau = (T - t).clamp(min=0.0)
    payoff = torch.relu(S - K)
    return payoff + tau * F.softplus(u)


# =============================================================================
# 5. Pesos adaptativos e perda física
# =============================================================================

class AdaptiveLossWeights(nn.Module):
    """
    Pesos de perda aprendíveis com regularização log.
    w_i = exp(lambda_i), lambda treinável.
    """
    def __init__(self, n_terms: int = 3, init: Optional[List[float]] = None):
        super().__init__()
        if init is None:
            init = [1.0] * n_terms
        self.log_w = nn.Parameter(torch.log(torch.tensor(init, dtype=torch.float32)))

    def forward(self) -> torch.Tensor:
        return torch.exp(self.log_w)

    def weighted_sum(self, losses: List[torch.Tensor]) -> torch.Tensor:
        w = self.forward()
        return sum(w[i] * losses[i] for i in range(len(losses)))


def causal_weights(t: torch.Tensor, t_max: float, strength: float = 1.0) -> torch.Tensor:
    """Curriculum temporal: mais peso no início do tempo."""
    return torch.exp(-strength * t / (t_max + 1e-8))


class ComplexPhysicsLoss:
    """
    Perda física completa com:
      - residual PDE
      - contorno
      - terminal
      - pesos adaptativos
      - máscara causal
    """
    def __init__(
        self,
        model: nn.Module,
        residual_fn: Callable,
        params: Dict,
        use_adaptive_weights: bool = True,
        use_causal: bool = True,
    ):
        self.model = model
        self.residual_fn = residual_fn
        self.p = params
        self.use_causal = use_causal
        self.adaptive = AdaptiveLossWeights(3, [1.0, 2.0, 15.0]) if use_adaptive_weights else None

    def total_loss(self, collocation: Dict) -> Tuple[torch.Tensor, Dict]:
        Xi = collocation["interior"]
        cols = [Xi[:, i:i+1].requires_grad_(True) for i in range(Xi.shape[1])]
        res = self.residual_fn(self.model, *cols, self.p)

        if self.use_causal and cols[-1] is not None:
            w_c = causal_weights(cols[-1], self.p.get("T", 1.0))
            loss_pde = (w_c * res**2).mean()
        else:
            loss_pde = (res**2).mean()

        loss_bc = torch.tensor(0.0, device=Xi.device)
        loss_ic = torch.tensor(0.0, device=Xi.device)

        if "boundary" in collocation:
            Xb = collocation["boundary"]
            Vb = self.model(*[Xb[:, i:i+1] for i in range(Xb.shape[1])])
            # exemplo: V≈0 em S≈0
            mask = (Xb[:, 0:1] < 1.0).float()
            loss_bc = (mask * Vb**2).mean()

        if "terminal" in collocation:
            Xt = collocation["terminal"]
            Vt = self.model(*[Xt[:, i:i+1] for i in range(Xt.shape[1])])
            K = self.p.get("K", 100.0)
            payoff = torch.relu(Xt[:, 0:1] - K)
            loss_ic = ((Vt - payoff)**2).mean()

        if self.adaptive is not None:
            total = self.adaptive.weighted_sum([loss_pde, loss_bc, loss_ic])
            weights = self.adaptive().detach().cpu().tolist()
        else:
            total = (
                self.p.get("lambda_pde", 1.0) * loss_pde
                + self.p.get("lambda_bc", 2.0) * loss_bc
                + self.p.get("lambda_ic", 15.0) * loss_ic
            )
            weights = [
                self.p.get("lambda_pde", 1.0),
                self.p.get("lambda_bc", 2.0),
                self.p.get("lambda_ic", 15.0),
            ]

        return total, {
            "pde": loss_pde.item(),
            "bc": loss_bc.item(),
            "ic": loss_ic.item(),
            "total": total.item(),
            "weights": weights,
        }


# =============================================================================
# 6. Residual Adaptive Refinement (RAR)
# =============================================================================

class ResidualAdaptiveSampler:
    """
    Reamostra pontos de colocação onde o residual é maior.
    """
    def __init__(self, residual_fn: Callable, base_sampler: Callable, device: str = "cpu"):
        self.residual_fn = residual_fn
        self.base_sampler = base_sampler
        self.device = device

    def sample(self, n_points: int, n_candidates_factor: int = 8) -> torch.Tensor:
        candidates = self.base_sampler(n_points * n_candidates_factor)
        with torch.enable_grad():
            res = torch.abs(self.residual_fn(candidates)).detach().flatten()
        idx = torch.argsort(res, descending=True)[:n_points]
        return candidates[idx].clone().requires_grad_(True)


# =============================================================================
# 7. Factory e exemplo de residual Black-Scholes
# =============================================================================

def black_scholes_residual(model, S, t, params):
    S = S.requires_grad_(True)
    t = t.requires_grad_(True)
    V = model(S, t)
    dV_dS = torch.autograd.grad(V, S, torch.ones_like(V), create_graph=True)[0]
    dV_dt = torch.autograd.grad(V, t, torch.ones_like(V), create_graph=True)[0]
    d2V_dS2 = torch.autograd.grad(dV_dS, S, torch.ones_like(dV_dS), create_graph=True)[0]
    r = params.get("r", 0.05)
    sigma = params.get("sigma", 0.2)
    return dV_dt + 0.5 * sigma**2 * S**2 * d2V_dS2 + r * S * dV_dS - r * V


def build_complex_pinn(
    in_dim: int = 2,
    out_dim: int = 1,
    architecture: str = "residual_fourier",
    hidden_dim: int = 128,
    n_blocks: int = 6,
) -> nn.Module:
    """
    Factory de arquiteturas complexas.

    architecture ∈ {
        'residual_fourier',  # recomendado
        'dgm',
        'highway_fourier',
        'attention_fourier',
        'full'               # fourier + residual + attention
    }
    """
    if architecture == "residual_fourier":
        return ComplexPINN(
            in_dim=in_dim, out_dim=out_dim, hidden_dim=hidden_dim, n_blocks=n_blocks,
            use_fourier=True, use_attention=False, use_dgm=False,
        )
    if architecture == "dgm":
        return ComplexPINN(
            in_dim=in_dim, out_dim=out_dim, hidden_dim=hidden_dim, n_blocks=n_blocks,
            use_fourier=True, use_dgm=True,
        )
    if architecture == "highway_fourier":
        return ComplexPINN(
            in_dim=in_dim, out_dim=out_dim, hidden_dim=hidden_dim, n_blocks=n_blocks,
            use_fourier=True, use_highway=True,
        )
    if architecture == "attention_fourier":
        return ComplexPINN(
            in_dim=in_dim, out_dim=out_dim, hidden_dim=hidden_dim, n_blocks=n_blocks,
            use_fourier=True, use_attention=True,
        )
    if architecture == "full":
        return ComplexPINN(
            in_dim=in_dim, out_dim=out_dim, hidden_dim=hidden_dim, n_blocks=n_blocks,
            use_fourier=True, use_attention=True, use_highway=False, use_dgm=False,
            activation="adaptive_tanh",
        )
    raise ValueError(f"Arquitetura desconhecida: {architecture}")


# =============================================================================
# 8. Demo rápida
# =============================================================================

if __name__ == "__main__":
    print("=== Complex PINN Demo ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for arch in ["residual_fourier", "dgm", "highway_fourier", "attention_fourier", "full"]:
        net = build_complex_pinn(in_dim=2, architecture=arch, hidden_dim=64, n_blocks=3).to(device)
        S = torch.rand(32, 1, device=device) * 100 + 50
        t = torch.rand(32, 1, device=device)
        V = net(S, t)
        nparams = sum(p.numel() for p in net.parameters())
        print(f"{arch:20s} | out={tuple(V.shape)} | params={nparams:,}")

    # Teste residual + loss
    net = build_complex_pinn(architecture="full", hidden_dim=64, n_blocks=3).to(device)
    params = {"r": 0.05, "sigma": 0.2, "K": 100.0, "T": 1.0}
    loss_fn = ComplexPhysicsLoss(net, black_scholes_residual, params)
    Xi = torch.rand(500, 2, device=device)
    Xi[:, 0] = Xi[:, 0] * 200 + 10
    col = {"interior": Xi.requires_grad_(True)}
    total, comps = loss_fn.total_loss(col)
    print(f"\nLoss demo: total={comps['total']:.4e} | weights={comps['weights']}")
    print(f"\nLoss demo: total={comps['total']:.4e} | weights={comps['weights']}")
    print("OK - ComplexPINN pronta para uso.")
