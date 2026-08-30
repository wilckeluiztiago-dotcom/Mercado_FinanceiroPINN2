"""
Configuração – Modelo de Heston completo
Autor: Luiz Tiago Wilcke – Capítulo 5
"""
from dataclasses import dataclass

@dataclass
class HestonConfig:
    # Dinâmica
    kappa: float = 2.0
    theta: float = 0.04
    xi: float = 0.3
    rho: float = -0.7
    r: float = 0.05
    q: float = 0.0
    S0: float = 100.0
    v0: float = 0.04
    K: float = 100.0
    T: float = 1.0
    S_max: float = 400.0
    v_max: float = 0.6

    # Rede
    hidden_layers: int = 8
    hidden_dim: int = 180

    # Amostragem
    n_interior: int = 25000
    n_boundary: int = 6000
    n_terminal: int = 6000

    # Treinamento
    adam_epochs: int = 15000
    lbfgs_epochs: int = 12
    lr_adam: float = 5e-4

    # Pesos
    lambda_pde: float = 1.0
    lambda_bc: float = 3.0
    lambda_ic: float = 20.0

    device: str = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    seed: int = 123
