"""
Treinamento completo – Opções Americanas PINN
Inspirado no livro de Luiz Tiago Wilcke (Cap. 4)
"""
import sys
sys.path.append("..")
import torch
import numpy as np
import matplotlib.pyplot as plt
from configuracao import AmericanConfig
from modelo import (AmericanPINN, AmericanLoss, extract_free_boundary,
                   sample_american_collocation)
from utils.otimizadores import HybridOptimizer

def main():
    cfg = AmericanConfig()
    torch.manual_seed(cfg.seed)
    device = cfg.device
    print(f"[American] device={device}")
    model = AmericanPINN(cfg.hidden_layers, cfg.hidden_dim, use_fourier=True).to(device)
    collocation = sample_american_collocation(cfg, device)
    loss_obj = AmericanLoss(model, cfg)

    def loss_fn():
        total, _ = loss_obj.total_loss(collocation)
        return total

    hybrid = HybridOptimizer(model, loss_fn, lr_adam=cfg.lr_adam, max_iter_lbfgs=120, device=device)
    history = []
    print("=== Adam ===")
    for ep in range(1, cfg.adam_epochs + 1):
        loss = hybrid.adam_step()
        history.append(loss)
        if ep % 1500 == 0:
            _, comps = loss_obj.total_loss(collocation)
            print(f"Epoch {ep:5d} | total={comps['total']:.4e} pde={comps['pde']:.3e} "
                  f"ic={comps['ic']:.3e} sp={comps['sp']:.3e}")
    print("=== L-BFGS ===")
    for i in range(cfg.lbfgs_epochs):
        loss = hybrid.lbfgs_step()
        history.append(loss)
        print(f"L-BFGS {i+1} | {loss:.6e}")

    t_grid = torch.linspace(0, cfg.T, 60, device=device)
    S_grid = torch.linspace(40, 160, 400, device=device).unsqueeze(1)
    Sf = extract_free_boundary(model, t_grid, S_grid, cfg.K)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(history); ax[0].set_yscale("log"); ax[0].set_title("Loss Americana")
    ax[1].plot(t_grid.cpu(), Sf.numpy()); ax[1].set_title("Fronteira Livre Sf(t)")
    ax[1].set_xlabel("t"); ax[1].set_ylabel("Sf")
    plt.tight_layout(); plt.savefig("american_results.png", dpi=140)
    torch.save(model.state_dict(), "american_pinn.pth")
    print("Salvo: american_pinn.pth / american_results.png")

if __name__ == "__main__":
    main()
