"""
Demo de treinamento com a rede PINN complexa (Black-Scholes).
Inspirado no livro de Luiz Tiago Wilcke.
"""
import sys
sys.path.append("..")
import torch
import numpy as np
import matplotlib.pyplot as plt
from pinn_complexa import (
    build_complex_pinn,
    ComplexPhysicsLoss,
    black_scholes_residual,
    ResidualAdaptiveSampler,
)
from amostragem import LatinHypercubeSampler
from otimizadores import HybridOptimizer


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Rede complexa
    model = build_complex_pinn(
        in_dim=2,
        architecture="full",      # fourier + residual + attention
        hidden_dim=128,
        n_blocks=6,
    ).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"Arquitetura: full | Parâmetros: {nparams:,}")

    params = {"r": 0.05, "sigma": 0.2, "K": 100.0, "T": 1.0, "S_max": 300.0}
    loss_obj = ComplexPhysicsLoss(
        model, black_scholes_residual, params,
        use_adaptive_weights=True, use_causal=True,
    )

    # Colocação inicial
    sampler = LatinHypercubeSampler([(1.0, params["S_max"]), (0.0, params["T"])])
    Xi = sampler.sample_torch(12000, device=device)
    S_term = torch.rand(3000, 1, device=device) * (params["S_max"] - 1) + 1
    Xt = torch.cat([S_term, torch.full((3000, 1), params["T"], device=device)], 1)
    collocation = {"interior": Xi, "terminal": Xt}

    # Incluir pesos adaptativos no otimizador
    all_params = list(model.parameters())
    if loss_obj.adaptive is not None:
        all_params += list(loss_obj.adaptive.parameters())

    opt = torch.optim.Adam(all_params, lr=1e-3)
    history = []

    print("=== Adam (com adaptive weights + causal + RAR periódico) ===")
    for ep in range(1, 6001):
        opt.zero_grad()
        total, comps = loss_obj.total_loss(collocation)
        total.backward()
        opt.step()
        history.append(comps["total"])

        if ep % 1000 == 0:
            w = np.round(comps["weights"], 2)
            print(f"Epoch {ep:5d} | loss={comps['total']:.4e} | "
                  f"pde={comps['pde']:.3e} ic={comps['ic']:.3e} | w={w}")

        # Residual Adaptive Refinement a cada 2000 épocas
        if ep % 2000 == 0:
            def res_fn(X):
                return black_scholes_residual(
                    model, X[:, 0:1], X[:, 1:2], params
                )
            rar = ResidualAdaptiveSampler(
                res_fn,
                lambda n: sampler.sample_torch(n, device=device),
                device,
            )
            new_pts = rar.sample(4000, n_candidates_factor=6)
            collocation["interior"] = torch.cat(
                [collocation["interior"][:8000], new_pts], dim=0
            )

    # L-BFGS refinamento
    print("=== L-BFGS ===")
    def closure():
        opt_lbfgs.zero_grad()
        total, _ = loss_obj.total_loss(collocation)
        total.backward()
        return total

    opt_lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=80, history_size=30,
                                  line_search_fn="strong_wolfe")
    for i in range(3):
        loss = opt_lbfgs.step(closure)
        history.append(loss.item() if torch.is_tensor(loss) else loss)
        print(f"L-BFGS {i+1} | {history[-1]:.6e}")

    # Validação rápida vs analítico
    from scipy.stats import norm
    S_np = np.linspace(60, 140, 80)
    tau = 1.0
    d1 = (np.log(S_np / 100) + (0.05 + 0.5 * 0.04) * tau) / (0.2 * np.sqrt(tau))
    d2 = d1 - 0.2 * np.sqrt(tau)
    V_true = S_np * norm.cdf(d1) - 100 * np.exp(-0.05 * tau) * norm.cdf(d2)

    S_t = torch.tensor(S_np, dtype=torch.float32, device=device).unsqueeze(1)
    t_t = torch.zeros_like(S_t)
    with torch.no_grad():
        V_pred = model(S_t, t_t).cpu().numpy().flatten()
    rel = np.mean(np.abs(V_pred - V_true) / (np.abs(V_true) + 1e-8))
    print(f"Erro relativo médio vs analítico: {rel:.4e}")

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history)
    plt.yscale("log")
    plt.title("Loss – Complex PINN")
    plt.subplot(1, 2, 2)
    plt.plot(S_np, V_true, "k-", lw=2, label="Analítico")
    plt.plot(S_np, V_pred, "r--", label="Complex PINN")
    plt.legend()
    plt.title("Call t=0")
    plt.tight_layout()
    plt.savefig("complex_pinn_demo.png", dpi=140)
    torch.save(model.state_dict(), "complex_pinn_demo.pth")
    print("Salvo: complex_pinn_demo.png / complex_pinn_demo.pth")


if __name__ == "__main__":
    main()
'''
