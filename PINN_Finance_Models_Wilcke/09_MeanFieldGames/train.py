"""
Pipeline de treinamento híbrido – Jogos de Campo Médio (Mean Field Games)
Metodologia Adam + L-BFGS: Luiz Tiago Wilcke – Capítulos 11 e 23
"""
import sys
sys.path.append("..")
import torch
from config import Config
from model import PINN, PhysicsInformedLoss
from utils.optimizers import HybridOptimizer
from utils.sampling import LatinHypercubeSampler

def main():
    cfg = Config()
    torch.manual_seed(cfg.seed)
    device = cfg.device
    print(f"[Jogos de Campo Médio (Mean Field Games)] Iniciando treinamento em {device}")

    model = PINN(in_dim=2, hidden_layers=cfg.hidden_layers, hidden_dim=cfg.hidden_dim).to(device)
    # Amostragem Latin Hypercube do domínio
    # sampler = LatinHypercubeSampler([(S_min,S_max),(0,T)])
    # X = sampler.sample_torch(cfg.n_collocation, device=device)

    # loss_obj = PhysicsInformedLoss(model, {})
    # def loss_fn():
    #     total, _ = loss_obj.total_loss({"interior": X})
    #     return total

    # hybrid = HybridOptimizer(model, loss_fn, lr_adam=cfg.lr_adam)
    # history = hybrid.train(cfg.adam_epochs, cfg.lbfgs_epochs)
    print("Estrutura de treinamento pronta. Implemente o residual específico e rode.")
    torch.save(model.state_dict(), f"09_meanfieldgames_pinn.pth")

if __name__ == "__main__":
    main()
