"""
Extração nativa de Gregas via diferenciação automática.
Baseado nas técnicas dos Capítulos 3, 5 e 9 de Luiz Tiago Wilcke.
"""

import torch
from typing import Dict, Callable


def compute_greeks_1d(model: Callable, S: torch.Tensor, t: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Gregas clássicas para modelos 1D (Black-Scholes, American, etc.)."""
    S = S.clone().requires_grad_(True)
    t = t.clone().requires_grad_(True)
    V = model(S, t)

    # Delta
    delta = torch.autograd.grad(V, S, grad_outputs=torch.ones_like(V), create_graph=True)[0]

    # Gamma
    gamma = torch.autograd.grad(delta, S, grad_outputs=torch.ones_like(delta), create_graph=True)[0]

    # Theta
    theta = torch.autograd.grad(V, t, grad_outputs=torch.ones_like(V), create_graph=True)[0]

    # Vega (aproximado por diferenciação finita se sigma não for entrada)
    return {"delta": delta, "gamma": gamma, "theta": theta}


def compute_greeks_heston(model: Callable, S: torch.Tensor, v: torch.Tensor, t: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Gregas avançadas para Heston (Δ, Γ, Vega, Vanna, Volga)."""
    S = S.clone().requires_grad_(True)
    v = v.clone().requires_grad_(True)
    t = t.clone().requires_grad_(True)
    V = model(S, v, t)

    delta = torch.autograd.grad(V, S, torch.ones_like(V), create_graph=True)[0]
    gamma = torch.autograd.grad(delta, S, torch.ones_like(delta), create_graph=True)[0]
    vega = torch.autograd.grad(V, v, torch.ones_like(V), create_graph=True)[0]
    vanna = torch.autograd.grad(delta, v, torch.ones_like(delta), create_graph=True)[0]
    volga = torch.autograd.grad(vega, v, torch.ones_like(vega), create_graph=True)[0]
    theta = torch.autograd.grad(V, t, torch.ones_like(V), create_graph=True)[0]

    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "vanna": vanna,
        "volga": volga,
        "theta": theta
    }


def compute_greeks_multi(model: Callable, X: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Gregas para modelos multi-ativo (cesta)."""
    X = X.clone().requires_grad_(True)
    V = model(X)
    grads = torch.autograd.grad(V, X, torch.ones_like(V), create_graph=True)[0]
    # Hessiana para gammas cruzados
    n = X.shape[1]
    hessian = []
    for i in range(n):
        row = torch.autograd.grad(grads[:, i], X, torch.ones_like(grads[:, i]), create_graph=True, retain_graph=True)[0]
        hessian.append(row)
    return {"deltas": grads, "gammas": torch.stack(hessian, dim=1)}
