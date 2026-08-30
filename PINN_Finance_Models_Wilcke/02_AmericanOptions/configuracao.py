"""Config – Opções Americanas. Inspirado em Luiz Tiago Wilcke Cap. 4"""
from dataclasses import dataclass
@dataclass
class AmericanConfig:
    S0: float = 100.0
    K: float = 100.0
    T: float = 1.0
    r: float = 0.05
    sigma: float = 0.25
    q: float = 0.03
    S_min: float = 0.0
    S_max: float = 300.0
    lambda_penalty: float = 1500.0
    penalty_power: float = 2.0
    hidden_layers: int = 7
    hidden_dim: int = 160
    n_interior: int = 18000
    n_boundary: int = 4500
    n_terminal: int = 4500
    adam_epochs: int = 10000
    lbfgs_epochs: int = 8
    lr_adam: float = 7e-4
    lambda_pde: float = 1.0
    lambda_bc: float = 2.5
    lambda_ic: float = 18.0
    lambda_penalty_loss: float = 6.0
    device: str = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    seed: int = 42
