"""
Módulo de amostragem avançada para PINNs financeiras.
Baseado nas técnicas de Latin Hypercube Sampling e reamostragem adaptativa
descritas nos Capítulos 3 e 5 da obra de Luiz Tiago Wilcke.
"""

import numpy as np
from pyDOE import lhs
import torch
from typing import Tuple, Optional, List


class LatinHypercubeSampler:
    """Amostrador Latin Hypercube para domínios financeiros."""

    def __init__(self, bounds: List[Tuple[float, float]], seed: Optional[int] = None):
        self.bounds = np.array(bounds)
        self.dim = len(bounds)
        if seed is not None:
            np.random.seed(seed)

    def sample(self, n: int) -> np.ndarray:
        """Gera n pontos no hipercubo unitário e escala para os bounds."""
        unit = lhs(self.dim, samples=n)
        scaled = self.bounds[:, 0] + (self.bounds[:, 1] - self.bounds[:, 0]) * unit
        return scaled

    def sample_torch(self, n: int, device: str = "cpu", requires_grad: bool = True) -> torch.Tensor:
        pts = self.sample(n)
        t = torch.tensor(pts, dtype=torch.float32, device=device)
        if requires_grad:
            t.requires_grad_(True)
        return t


class AdaptiveResidualSampler:
    """
    Reamostragem adaptativa baseada em residual da PDE.
    Técnica avançada do Capítulo 5 (Heston) e Capítulo 3.
    """

    def __init__(self, base_sampler: LatinHypercubeSampler, residual_fn, device: str = "cpu"):
        self.base = base_sampler
        self.residual_fn = residual_fn
        self.device = device

    def sample(self, n: int, n_candidates: int = 5) -> torch.Tensor:
        """Seleciona os n pontos de maior residual entre n_candidates * n candidatos."""
        candidates = self.base.sample_torch(n * n_candidates, device=self.device, requires_grad=True)
        with torch.no_grad():
            res = torch.abs(self.residual_fn(candidates)).detach().cpu().numpy().flatten()
        idx = np.argsort(res)[-n:]
        selected = candidates[idx].detach().clone().requires_grad_(True)
        return selected


def collocation_points_bs(
    S_min: float, S_max: float, T: float,
    n_interior: int = 10000,
    n_boundary: int = 2000,
    n_terminal: int = 2000,
    device: str = "cpu"
) -> dict:
    """Gera pontos de colocação para Black-Scholes 1D."""
    sampler_int = LatinHypercubeSampler([(S_min, S_max), (0.0, T)])
    X_int = sampler_int.sample_torch(n_interior, device=device)

    # Contorno S=0 e S=S_max
    t_b = torch.rand(n_boundary, 1, device=device) * T
    S_low = torch.zeros(n_boundary // 2, 1, device=device)
    S_high = torch.full((n_boundary // 2, 1), S_max, device=device)
    t_low = t_b[: n_boundary // 2]
    t_high = t_b[n_boundary // 2 :]
    X_bound = torch.cat([
        torch.cat([S_low, t_low], dim=1),
        torch.cat([S_high, t_high], dim=1)
    ], dim=0).requires_grad_(True)

    # Terminal t=T
    S_term = torch.rand(n_terminal, 1, device=device) * (S_max - S_min) + S_min
    t_term = torch.full((n_terminal, 1), T, device=device)
    X_term = torch.cat([S_term, t_term], dim=1).requires_grad_(True)

    return {"interior": X_int, "boundary": X_bound, "terminal": X_term}
