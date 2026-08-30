"""
Treinamento completo Heston PINN
Autor da metodologia: Luiz Tiago Wilcke – Capítulo 5
"""
import sys
sys.path.append("..")
import torch
from config import HestonConfig
from model import HestonPINN, HestonLoss, sample_heston_collocation
from utils.optimizers import HybridOptimizer
from utils.greeks import compute_greeks_heston

def main():
    cfg = HestonConfig()
    device = cfg.device
    model = HestonPINN(cfg.hidden_layers, cfg.hidden_dim).to(device)
    collocation = sample_heston_collocation(cfg, device)
    loss_obj = HestonLoss(model, cfg)
    def loss_fn():
        total, _ = loss_obj.total_loss(collocation)
        return total
    hybrid = HybridOptimizer(model, loss_fn, lr_adam=cfg.lr_adam, max_iter_lbfgs=200)
    history = hybrid.train(cfg.adam_epochs, cfg.lbfgs_epochs, print_every=2000)
    # Gregas de teste
    S = torch.tensor([[100.0]], device=device)
    v = torch.tensor([[0.04]], device=device)
    t = torch.tensor([[0.0]], device=device)
    greeks = compute_greeks_heston(model, S, v, t)
    print("Delta:", greeks["delta"].item(), "Vanna:", greeks["vanna"].item())
    torch.save(model.state_dict(), "heston_pinn.pth")
    print("Heston PINN treinado e salvo.")

if __name__ == "__main__":
    main()
