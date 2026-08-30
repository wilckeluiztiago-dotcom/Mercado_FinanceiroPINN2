"""
Pipeline de treinamento – Almgren-Chriss Liquidation
Inspirado no livro de Luiz Tiago Wilcke (Cap. 19)
"""
import sys
sys.path.append("..")
import torch
import matplotlib.pyplot as plt
from configuracao import Config
from modelo import PINN, PhysicsLoss, make_collocation
from utils.otimizadores import HybridOptimizer

def main():
    cfg = Config()
    torch.manual_seed(cfg.seed)
    device = cfg.device
    print(f"[Almgren-Chriss Liquidation] {device}")
    model = PINN(in_dim=3, hidden_dim=cfg.hidden_dim, n_blocks=cfg.hidden_layers).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"Parâmetros treináveis: {nparams:,}")
    params = {
        "r": cfg.r, "sigma": cfg.sigma, "K": cfg.K, "T": cfg.T,
        "lambda_pde": 1.0, "lambda_ic": 12.0,
        "eta": 1e-3, "gamma": 1e-4,
    }
    loss_obj = PhysicsLoss(model, params)
    col = make_collocation(cfg.bounds, cfg.n_collocation, cfg.n_collocation//4, cfg.T, device)
    def loss_fn():
        t, _ = loss_obj.total_loss(col)
        return t
    hybrid = HybridOptimizer(model, loss_fn, lr_adam=cfg.lr_adam, max_iter_lbfgs=80, device=device)
    history = []
    print("=== Adam ===")
    for ep in range(1, cfg.adam_epochs+1):
        loss = hybrid.adam_step()
        history.append(loss)
        if ep % 2000 == 0:
            _, c = loss_obj.total_loss(col)
            print(f"Epoch {ep:5d} | {c['total']:.4e} (pde={c['pde']:.3e} ic={c['ic']:.3e})")
    print("=== L-BFGS ===")
    for i in range(cfg.lbfgs_epochs):
        loss = hybrid.lbfgs_step()
        history.append(loss)
        print(f"L-BFGS {i+1} | {loss:.6e}")
    
    plt.figure(figsize=(6,4)); plt.plot(history); plt.yscale("log")
    plt.title("Loss – Almgren-Chriss Liquidation"); plt.tight_layout()
    fname = "15_AlmgrenChriss"
    plt.savefig(fname + "_loss.png", dpi=120)
    torch.save(model.state_dict(), fname + "_pinn.pth")
    print("Salvo.")

if __name__ == "__main__":
    main()
