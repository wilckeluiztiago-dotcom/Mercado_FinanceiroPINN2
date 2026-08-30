"""
Pipeline completo de treinamento – Opções Americanas
Metodologia de penalização e fronteira livre: Luiz Tiago Wilcke – Capítulo 4
"""

import sys
sys.path.append("..")
import torch
import numpy as np
import matplotlib.pyplot as plt
from config import AmericanConfig
from model import AmericanPINN, AmericanLoss, extract_free_boundary
from utils.sampling import LatinHypercubeSampler, collocation_points_bs
from utils.optimizers import HybridOptimizer

def main():
    cfg = AmericanConfig()
    torch.manual_seed(cfg.seed)
    device = cfg.device
    print(f"[American] Dispositivo: {device}")

    model = AmericanPINN(cfg.hidden_layers, cfg.hidden_dim).to(device)
    collocation = collocation_points_bs(cfg.S_min, cfg.S_max, cfg.T,
                                        cfg.n_interior, cfg.n_boundary, cfg.n_terminal, device)

    loss_obj = AmericanLoss(model, cfg)
    def loss_fn():
        total, comps = loss_obj.total_loss(collocation)
        return total

    hybrid = HybridOptimizer(model, loss_fn, lr_adam=cfg.lr_adam, max_iter_lbfgs=150, device=device)
    history = hybrid.train(adam_epochs=cfg.adam_epochs, lbfgs_epochs=cfg.lbfgs_epochs, print_every=1500)

    # Extração da fronteira livre
    t_grid = torch.linspace(0, cfg.T, 50, device=device)
    S_grid = torch.linspace(50, 150, 300, device=device).unsqueeze(1)
    Sf = extract_free_boundary(model, t_grid, S_grid, cfg.K)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history)
    plt.yscale("log")
    plt.title("Loss Americana (Penalização)")
    plt.subplot(1, 2, 2)
    plt.plot(t_grid.cpu().numpy(), Sf.numpy())
    plt.title("Fronteira Livre Sf(t)")
    plt.xlabel("t")
    plt.ylabel("Sf")
    plt.tight_layout()
    plt.savefig("american_results.png", dpi=140)
    torch.save(model.state_dict(), "american_pinn.pth")
    print("Modelo e figura salvos.")

if __name__ == "__main__":
    main()
