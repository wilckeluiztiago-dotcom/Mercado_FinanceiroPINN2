"""
Treinamento completo – Rough Volatility fPINN
Inspirado no livro de Luiz Tiago Wilcke (Cap. 22/27)
"""
import sys
sys.path.append("..")
import torch
import numpy as np
import matplotlib.pyplot as plt
from configuracao import Config
from modelo import PINN, PhysicsLoss, make_collocation, compute_delta_gamma
from utils.otimizadores import HybridOptimizer

def main():
    cfg = Config()
    torch.manual_seed(cfg.seed)
    device = cfg.device
    print(f"[Rough Volatility fPINN] device={device}")
    model = PINN(in_dim=2, hidden_dim=cfg.hidden_dim, n_blocks=cfg.hidden_layers).to(device)
    print(f"Parâmetros: {sum(p.numel() for p in model.parameters()):,}")
    params = {"r": cfg.r, "sigma": cfg.sigma, "K": cfg.K, "T": cfg.T,
              "lambda_pde": 1.0, "lambda_ic": 12.0}
    loss_obj = PhysicsLoss(model, params)
    col = make_collocation(cfg.bounds, cfg.n_collocation, cfg.n_collocation//4, cfg.T, device)
    def loss_fn():
        total, _ = loss_obj.total_loss(col)
        return total
    hybrid = HybridOptimizer(model, loss_fn, lr_adam=cfg.lr_adam, max_iter_lbfgs=80, device=device)
    history = []
    print("=== Adam ===")
    for ep in range(1, cfg.adam_epochs + 1):
        loss = hybrid.adam_step()
        history.append(loss)
        if ep % 2000 == 0:
            _, c = loss_obj.total_loss(col)
            print(f"Epoch {ep:5d} | total={c['total']:.4e} pde={c['pde']:.3e} ic={c['ic']:.3e}")
    print("=== L-BFGS ===")
    for i in range(cfg.lbfgs_epochs):
        loss = hybrid.lbfgs_step()
        history.append(loss)
        print(f"L-BFGS {i+1} | {loss:.6e}")
    # quick greek check
    S_test = torch.linspace(80, 120, 50, device=device).unsqueeze(1)
    t_test = torch.zeros_like(S_test)
    delta, gamma = compute_delta_gamma(model, S_test, t_test)
    print(f"Delta médio: {delta.mean().item():.4f} | Gamma médio: {gamma.mean().item():.6f}")
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1); plt.plot(history); plt.yscale("log"); plt.title("Loss")
    plt.subplot(1, 2, 2); plt.plot(S_test.cpu(), delta.detach().cpu()); plt.title("Delta")
    plt.tight_layout()
    plt.savefig("16_RoughVolatility_results.png", dpi=120)
    torch.save({"model": model.state_dict(), "history": history}, "16_RoughVolatility_pinn.pth")
    print("Salvo.")

if __name__ == "__main__":
    main()
