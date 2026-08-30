"""
Demo rápido das arquiteturas avançadas
"""
import torch
from utils.redes_avancadas import (
    AdvancedPINN, DGMNetwork, MultiHeadPINN,
    AdaptiveLossWeights, FourierFeatureEmbedding
)

print("=== Fourier Embedding ===")
emb = FourierFeatureEmbedding(2, 64, scale=10.0)
x = torch.randn(5, 2)
print(emb(x).shape)

print("=== AdvancedPINN (Fourier + Residual) ===")
net = AdvancedPINN(in_dim=2, hidden_dim=64, n_res_blocks=3)
print(net(torch.randn(5,1), torch.randn(5,1)).shape)

print("=== DGM Network ===")
dgm = DGMNetwork(in_dim=2, hidden_dim=64, n_layers=3)
print(dgm(torch.randn(5,1), torch.randn(5,1)).shape)

print("=== MultiHead (3 regimes) ===")
mh = MultiHeadPINN(in_dim=2, n_heads=3, hidden_dim=64)
print(mh(torch.randn(5,1), torch.randn(5,1)).shape)

print("=== Adaptive Weights ===")
aw = AdaptiveLossWeights(3)
print(aw())
print("Demo OK")
