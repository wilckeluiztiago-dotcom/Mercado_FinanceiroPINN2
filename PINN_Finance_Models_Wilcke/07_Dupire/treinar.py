"""
Treinamento – Dupire Local Vol Inverse
Inspirado no livro de Luiz Tiago Wilcke (Cap. 9)
"""
import sys
sys.path.append("..")
import torch
import numpy as np
import matplotlib.pyplot as plt
from configuracao import Config
from modelo import PINN, PhysicsLoss, sample_collocation
from utils.otimizadores import HybridOptimizer

def main():
    cfg = Config()
    torch.manual_seed(getattr(cfg, "seed", 42))
    device = cfg.device
    print(f"[Dupire Local Vol Inverse] device={device}")

    model = PINN(in_dim=2, hidden_layers=cfg.hidden_layers,
                 hidden_dim=cfg.hidden_dim, use_fourier=True).to(device)
    print(f"Parâmetros: {sum(p.numel() for p in model.parameters()):,}")

    params = {
        "r": getattr(cfg, "r", 0.05),
        "sigma": getattr(cfg, "sigma", 0.2),
        "K": getattr(cfg, "K", 100.0),
        "T": getattr(cfg, "T", 1.0),
        "lambda_pde": 1.0,
        "lambda_ic": 12.0,
        "lambda_bc": 2.0,
        
    }
    loss_obj = PhysicsLoss(model, params)

    bounds = getattr(cfg, "bounds", [(0.5, 300.0), (0.0, 1.0)])
    collocation = sample_collocation(bounds, cfg.n_collocation, cfg.n_collocation // 4,
                                     params["T"], device)

    def loss_fn():
        total, _ = loss_obj.total_loss(collocation)
        return total

    hybrid = HybridOptimizer(model, loss_fn, lr_adam=cfg.lr_adam,
                             max_iter_lbfgs=100, device=device)
    history = []
    print("=== Adam ===")
    for ep in range(1, cfg.adam_epochs + 1):
        loss = hybrid.adam_step()
        history.append(loss)
        if ep % 2000 == 0:
            _, c = loss_obj.total_loss(collocation)
            print(f"Epoch {ep:5d} | total={c['total']:.4e} pde={c['pde']:.3e} ic={c['ic']:.3e}")
    print("=== L-BFGS ===")
    for i in range(cfg.lbfgs_epochs):
        loss = hybrid.lbfgs_step()
        history.append(loss)
        print(f"L-BFGS {i+1} | {loss:.6e}")

    plt.figure(figsize=(6, 4))
    plt.plot(history)
    plt.yscale("log")
    plt.title("Loss – Dupire Local Vol Inverse")
    plt.tight_layout()
    plt.savefig("dupire_local_vol_inverse_loss.png", dpi=120)
    torch.save(model.state_dict(), "dupire_local_vol_inverse_pinn.pth")
    print("Modelo e figura salvos.")

if __name__ == "__main__":
    main()
