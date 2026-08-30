"""Config – Regime Switching. Inspirado em Luiz Tiago Wilcke (Cap. 29)"""
from dataclasses import dataclass, field
from typing import List, Tuple
@dataclass
class Config:
    hidden_layers: int = 5
    hidden_dim: int = 128
    n_collocation: int = 12000
    adam_epochs: int = 8000
    lbfgs_epochs: int = 5
    lr_adam: float = 1e-3
    r: float = 0.05
    sigma: float = 0.2
    K: float = 100.0
    T: float = 1.0
    bounds: List[Tuple[float, float]] = field(default_factory=lambda: [(0.5, 300.0), (0.0, 1.0)])
    device: str = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    seed: int = 42
