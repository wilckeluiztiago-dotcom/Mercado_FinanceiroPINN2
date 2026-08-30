"""
Configuração completa – Opções sobre Cesta Multiativo (Alta Dimensão)
Autor da formulação: Luiz Tiago Wilcke – Capítulo 15
"""
from dataclasses import dataclass

@dataclass
class Config:
    # Parâmetros de mercado e de rede – ver Capítulo Capítulo 15
    hidden_layers: int = 7
    hidden_dim: int = 140
    adam_epochs: int = 10000
    lbfgs_epochs: int = 8
    lr_adam: float = 1e-3
    n_collocation: int = 15000
    device: str = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    seed: int = 42
    # Adicione aqui os parâmetros específicos do modelo (r, sigma, kappa, etc.)
