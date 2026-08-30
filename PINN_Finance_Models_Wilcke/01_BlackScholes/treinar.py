"""
Pipeline de treinamento completo do PINN Black-Scholes.
Autor da metodologia: Luiz Tiago Wilcke (Capítulos 1 e 3)
"""

import sys
sys.path.append("..")
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from configuracao import BSConfig
from modelo import BlackScholesPINN, BlackScholesLoss, analytical_call
from utils.amostragem import collocation_points_bs
from utils.otimizadores import HybridOptimizer


def main():
    cfg = BSConfig()
    device = cfg.device
    print(f"Dispositivo: {device}")

    # Modelo
    model = BlackScholesPINN(
        hidden_layers=cfg.hidden_layers,
        hidden_dim=cfg.hidden_dim
    ).to(device)

    # Pontos de colocação
    collocation = collocation_points_bs(
        cfg.S_min, cfg.S_max, cfg.T,
        n_interior=cfg.n_interior,
        n_boundary=cfg.n_boundary,
        n_terminal=cfg.n_terminal,
        device=device
    )

    # Perda
    loss_obj = BlackScholesLoss(model, cfg)

    def loss_fn():
        total, _ = loss_obj.total_loss(collocation)
        return total

    # Otimizador híbrido
    hybrid = HybridOptimizer(
        model, loss_fn,
        lr_adam=cfg.lr_adam,
        max_iter_lbfgs=200,
        device=device
    )

    history = hybrid.train(
        adam_epochs=cfg.adam_epochs,
        lbfgs_epochs=cfg.lbfgs_epochs,
        print_every=1000
    )

    # Validação contra solução analítica
    S_test = np.linspace(50, 150, 100)
    t_test = np.full_like(S_test, 0.0)          # t=0
    V_true = analytical_call(S_test, cfg.K, cfg.T, t_test, cfg.r, cfg.sigma)

    S_t = torch.tensor(S_test, dtype=torch.float32, device=device).unsqueeze(1)
    t_t = torch.tensor(t_test, dtype=torch.float32, device=device).unsqueeze(1)
    with torch.no_grad():
        V_pred = model(S_t, t_t).cpu().numpy().flatten()

    rel_error = np.mean(np.abs(V_pred - V_true) / (np.abs(V_true) + 1e-8))
    print(f"\nErro relativo médio vs analítico: {rel_error:.4e}")

    # Plot
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history)
    plt.yscale("log")
    plt.title("Curva de Aprendizado (Loss)")
    plt.xlabel("Iteração")
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(S_test, V_true, "k-", label="Analítico")
    plt.plot(S_test, V_pred, "r--", label="PINN")
    plt.legend()
    plt.title("Preço da Call em t=0")
    plt.xlabel("S")
    plt.ylabel("V")
    plt.tight_layout()
    plt.savefig("bs_results.png", dpi=150)
    print("Figura salva em bs_results.png")

    # Salvar modelo
    torch.save(model.state_dict(), "bs_pinn.pth")
    print("Modelo salvo em bs_pinn.pth")


if __name__ == "__main__":
    main()
