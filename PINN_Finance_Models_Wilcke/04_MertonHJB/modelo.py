"""
PINN completa – Merton HJB Portfolio
Inspirado no livro de Luiz Tiago Wilcke (Cap. 6)
Implementação funcional com residual, contornos, terminal, treinamento híbrido.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, List

class PINN(nn.Module):
    """Rede fully-connected com Fourier features opcionais e residual blocks."""
    def __init__(self, in_dim: int = 2, hidden_layers: int = 6, hidden_dim: int = 128,
                 use_fourier: bool = True, fourier_scale: float = 10.0):
        super().__init__()
        self.use_fourier = use_fourier
        if use_fourier:
            self.register_buffer("B", torch.randn(in_dim, 64) * fourier_scale)
            d0 = 128
        else:
            d0 = in_dim
        self.input_layer = nn.Linear(d0, hidden_dim)
        self.blocks = nn.ModuleList()
        for _ in range(hidden_layers):
            self.blocks.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
            ))
        self.norm = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(hidden_layers)])
        self.out = nn.Linear(hidden_dim, 1)
        self.act = nn.Tanh()
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.75)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, *args) -> torch.Tensor:
        x = torch.cat(args, dim=-1)
        if self.use_fourier:
            proj = 2 * np.pi * x @ self.B
            x = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        h = self.act(self.input_layer(x))
        for block, norm in zip(self.blocks, self.norm):
            h = self.act(norm(h + block(h)))
        return self.out(h)


class PhysicsLoss:
    """Perda composta: residual PDE + contorno + terminal."""
    def __init__(self, model: PINN, params: Dict):
        self.model = model
        self.p = params

    def residual(self, *inputs) -> torch.Tensor:
        # inputs must require grad
        
        x, t = inputs[0], inputs[1]
        J = self.model(x, t)
        Jx = torch.autograd.grad(J, x, torch.ones_like(J), create_graph=True)[0]
        Jt = torch.autograd.grad(J, t, torch.ones_like(J), create_graph=True)[0]
        Jxx = torch.autograd.grad(Jx, x, torch.ones_like(Jx), create_graph=True)[0]
        mu = self.p.get("mu", 0.08)
        sig = self.p.get("sigma", 0.2)
        rho = self.p.get("rho", 0.05)
        gamma = self.p.get("gamma", 2.0)
        # CRRA: pi* and c* substituted
        pi_star = (mu - self.p["r"]) / (sig**2 * gamma)
        # residual after substitution (simplified power utility)
        res = Jt + (self.p["r"] + pi_star*(mu-self.p["r"]) - 0.5*gamma*pi_star**2*sig**2)*x*Jx + 0.5*(pi_star*sig*x)**2*Jxx - rho*J
     
        return res

    def terminal_loss(self, *inputs) -> torch.Tensor:
        V = self.model(*inputs)
        # payoff genérico – sobrescrito por modelo se necessário
        S = inputs[0]
        K = self.p.get("K", 100.0)
        payoff = x  # wealth – no standard option payoff
        return ((V - payoff)**2).mean()

    def boundary_loss(self, *inputs) -> torch.Tensor:
        V = self.model(*inputs)
        S = inputs[0]
        # condições simples em S→0
        mask = (S < 1.0).float()
        return (mask * V**2).mean()

    def total_loss(self, collocation: Dict) -> Tuple[torch.Tensor, Dict]:
        Xi = collocation["interior"]
        # split columns
        cols = [Xi[:, i:i+1] for i in range(Xi.shape[1])]
        for c in cols:
            c.requires_grad_(True)
        res = self.residual(*cols)
        loss_pde = (res**2).mean()
        loss_ic = torch.tensor(0.0, device=Xi.device)
        loss_bc = torch.tensor(0.0, device=Xi.device)
        if "terminal" in collocation:
            Xt = collocation["terminal"]
            tcols = [Xt[:, i:i+1] for i in range(Xt.shape[1])]
            loss_ic = self.terminal_loss(*tcols)
        if "boundary" in collocation:
            Xb = collocation["boundary"]
            bcols = [Xb[:, i:i+1] for i in range(Xb.shape[1])]
            loss_bc = self.boundary_loss(*bcols)
        lam_pde = self.p.get("lambda_pde", 1.0)
        lam_ic = self.p.get("lambda_ic", 10.0)
        lam_bc = self.p.get("lambda_bc", 1.0)
        total = lam_pde * loss_pde + lam_ic * loss_ic + lam_bc * loss_bc
        return total, {"pde": loss_pde.item(), "ic": loss_ic.item(),
                       "bc": loss_bc.item(), "total": total.item()}


def sample_collocation(bounds: List[Tuple[float, float]], n_int: int, n_term: int,
                       T: float, device: str) -> Dict:
    from utils.amostragem import LatinHypercubeSampler
    samp = LatinHypercubeSampler(bounds)
    Xi = samp.sample_torch(n_int, device=device)
    # terminal: last coordinate = T
    bounds_x = bounds[:-1]
    samp_x = LatinHypercubeSampler(bounds_x)
    Xx = samp_x.sample_torch(n_term, device=device)
    t_term = torch.full((n_term, 1), T, device=device)
    Xt = torch.cat([Xx, t_term], dim=1).requires_grad_(True)
    return {"interior": Xi, "terminal": Xt}
