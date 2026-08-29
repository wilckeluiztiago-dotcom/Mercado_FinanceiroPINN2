"""
Configuração do Modelo Black-Scholes PINN
Autor original da formulação: Luiz Tiago Wilcke (Capítulos 1 e 3)
"""

from dataclasses import dataclass

@dataclass
class BSConfig:
    # Parâmetros de mercado
    S0: float = 100.0
    K: float = 100.0
    T: float = 1.0
    r: float = 0.05
    sigma: float = 0.2
    S_min: float = 0.0
    S_max: float = 300.0

    # Arquitetura
    hidden_layers: int = 6
    hidden_dim: int = 128
    activation: str = "tanh"          # tanh é C^∞ e recomendado por Wilcke

    # Treinamento
    n_interior: int = 12000
    n_boundary: int = 3000
    n_terminal: int = 3000
    adam_epochs: int = 8000
    lbfgs_epochs: int = 5
    lr_adam: float = 1e-3
    device: str = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    # Pesos da perda composta (Cap. 3)
    lambda_pde: float = 1.0
    lambda_bc: float = 1.0
    lambda_ic: float = 10.0           # peso maior na condição terminal
