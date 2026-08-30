"""
Treinamento Avançado – Black-Scholes PINN
Arquiteturas: Fourier + Residual + Adaptive Weights + Causal + RAR
Metodologia: Luiz Tiago Wilcke (Capítulos 2, 3, 25) + técnicas de última geração
"""

import sys
sys.path.append("..")
import torch
import numpy as np
import matplotlib.pyplot as plt
from configuracao import BSConfig
from modelo_avancado import AdvancedBlackScholesLoss, build_advanced_bs_model
from utils.amostragem import collocation_points_bs
from utils.otimizadores import HybridOptimizer
from utils.redes_avancadas import ResidualAdaptiveSampler, AdaptiveLossWeights


def main():
    cfg = BSConfig()
    device = cfg.device
    print(f"[Advanced BS] Device: {device}")

    # Escolha da arquitetura
    architecture = "residual_fourier"   # opções: residual_fourier | dgm | hard_constraint
    model = build_advanced_bs_model(cfg, architecture=architecture).to(device)
    print(f"Arquitetura: {architecture}")
    print(f"Parâmetros treináveis: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Colocação inicial
    collocation = collocation_points_bs(
        cfg.S_min, cfg.S_max, cfg.T,
        n_interior=cfg.n_interior,
        n_boundary=cfg.n_boundary,
        n_terminal=cfg.n_terminal,
        device=device
    )

    loss_obj = AdvancedBlackScholesLoss(model, cfg, use_adaptive_weights=True)

    # Incluir os pesos adaptativos no otimizador
    params = list(model.parameters())
    if loss_obj.adaptive is not None:
        params += list(loss_obj.adaptive.parameters())

    def loss_fn():
        total, comps = loss_obj.total_loss(collocation)
        return total

    # Otimizador híbrido
    hybrid = HybridOptimizer(
        model, loss_fn,
        lr_adam=cfg.lr_adam,
        max_iter_lbfgs=250,
        device=device
    )
    # Sobrescrever o otimizador Adam para incluir os pesos adaptativos
    hybrid.opt_adam = torch.optim.Adam(params, lr=cfg.lr_adam)

    print("=== Fase Adam (com pesos adaptativos e causal weighting) ===")
    history = []
    for epoch in range(1, cfg.adam_epochs + 1):
        hybrid.opt_adam.zero_grad()
        total, comps = loss_obj.total_loss(collocation)
        total.backward()
        hybrid.opt_adam.step()
        history.append(comps["total"])

        if epoch % 1000 == 0:
            w = comps.get("weights", [0,0,0])
            print(f"Epoch {epoch:5d} | Loss {comps['total']:.4e} | "
                  f"PDE {comps['pde']:.3e} | BC {comps['bc']:.3e} | IC {comps['ic']:.3e} | "
                  f"w={np.round(w,2)}")

        # Residual Adaptive Refinement a cada 2000 épocas
        if epoch % 2000 == 0 and epoch > 0:
            def res_fn(X):
                return loss_obj.pde_residual(X[:,0:1], X[:,1:2])
            from utils.amostragem import LatinHypercubeSampler
            base_sampler = LatinHypercubeSampler([(cfg.S_min, cfg.S_max), (0.0, cfg.T)])
            rar = ResidualAdaptiveSampler(res_fn, lambda n: base_sampler.sample_torch(n, device=device), device)
            new_points = rar.sample(3000, n_candidates=8)
            # Mistura com pontos antigos
            collocation["interior"] = torch.cat([collocation["interior"][:9000], new_points], dim=0)

    print("=== Fase L-BFGS ===")
    for i in range(cfg.lbfgs_epochs):
        loss = hybrid.lbfgs_step()
        history.append(loss)
        print(f"L-BFGS {i+1} | Loss = {loss:.6e}")

    # Validação
    from modelo import analytical_call
    S_test = np.linspace(50, 150, 120)
    t_test = np.zeros_like(S_test)
    V_true = analytical_call(S_test, cfg.K, cfg.T, t_test, cfg.r, cfg.sigma)
    S_t = torch.tensor(S_test, dtype=torch.float32, device=device).unsqueeze(1)
    t_t = torch.tensor(t_test, dtype=torch.float32, device=device).unsqueeze(1)
    with torch.no_grad():
        V_pred = model(S_t, t_t).cpu().numpy().flatten()
    rel_err = np.mean(np.abs(V_pred - V_true) / (np.abs(V_true) + 1e-8))
    print(f"\nErro relativo médio vs analítico: {rel_err:.4e}")

    # Plots
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(history)
    axes[0].set_yscale("log")
    axes[0].set_title("Loss (Advanced PINN)")
    axes[1].plot(S_test, V_true, "k-", lw=2, label="Analítico")
    axes[1].plot(S_test, V_pred, "r--", label="PINN Avançada")
    axes[1].legend()
    axes[1].set_title("Preço da Call t=0")
    # Delta
    S_t.requires_grad_(True)
    V = model(S_t, t_t)
    delta = torch.autograd.grad(V, S_t, torch.ones_like(V), create_graph=False)[0]
    axes[2].plot(S_test, delta.detach().cpu().numpy())
    axes[2].set_title("Delta (autograd)")
    plt.tight_layout()
    plt.savefig("bs_advanced_results.png", dpi=150)
    torch.save({
        "model": model.state_dict(),
        "adaptive_weights": loss_obj.adaptive.state_dict() if loss_obj.adaptive else None,
        "history": history,
        "architecture": architecture
    }, "bs_advanced_pinn.pth")
    print("Resultados salvos: bs_advanced_results.png / bs_advanced_pinn.pth")


if __name__ == "__main__":
    main()
