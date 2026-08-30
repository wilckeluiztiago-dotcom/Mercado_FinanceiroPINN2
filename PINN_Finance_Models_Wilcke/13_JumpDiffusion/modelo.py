"""
PINN completa – Merton Jump Diffusion PIDE
Inspirado no livro de Luiz Tiago Wilcke (Cap. 17)
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
        
        S, t = inputs[0], inputs[1]
        V = self.model(S, t)
        VS = torch.autograd.grad(V, S, torch.ones_like(V), create_graph=True)[0]
        Vt = torch.autograd.grad(V, t, torch.ones_like(V), create_graph=True)[0]
        VSS = torch.autograd.grad(VS, S, torch.ones_like(VS), create_graph=True)[0]
        sig = self.p.get("sigma", 0.2)
        lam = self.p.get("lam", 0.5)
        muJ = self.p.get("muJ", -0.1)
        sigJ = self.p.get("sigJ", 0.15)
        kappa = np.exp(muJ + 0.5*sigJ**2) - 1.0
        L = Vt + 0.5*sig**2 * S**2 * VSS + (self.p["r"] - lam*kappa)*S*VS - self.p["r"]*V
        # Monte-Carlo jump integral inside graph
        M = 32
        Y = muJ + sigJ * torch.randn(S.shape[0], M, device=S.device)
        S_jump = S * torch.exp(Y)
        V_jump = self.model(S_jump.reshape(-1,1), t.expand(S.shape[0]*M, 1)).reshape(S.shape[0], M)
        jump_term = lam * (V_jump.mean(dim=1, keepdim=True) - V)
        res = L + jump_term
     
        return res

    def terminal_loss(self, *inputs) -> torch.Tensor:
        V = self.model(*inputs)
        # payoff genérico – sobrescrito por modelo se necessário
        S = inputs[0]
        K = self.p.get("K", 100.0)
        payoff = torch.relu(S - self.p['K'])
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
