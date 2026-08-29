"""
Otimizadores híbridos Adam + L-BFGS para PINNs financeiras.
Implementação fiel às recomendações dos Capítulos 3, 5 e 6 de Luiz Tiago Wilcke.
"""

import torch
from torch.optim import Adam, LBFGS
from typing import Callable, Optional, List


class HybridOptimizer:
    """
    Pipeline de treinamento em duas fases:
    1. Adam (exploração rápida)
    2. L-BFGS (refinamento de alta precisão)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: Callable,
        lr_adam: float = 1e-3,
        max_iter_lbfgs: int = 500,
        history_size: int = 50,
        device: str = "cpu"
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.device = device
        self.opt_adam = Adam(model.parameters(), lr=lr_adam)
        self.max_iter_lbfgs = max_iter_lbfgs
        self.history_size = history_size
        self.history: List[float] = []

    def adam_step(self, n_steps: int = 1) -> float:
        self.model.train()
        total = 0.0
        for _ in range(n_steps):
            self.opt_adam.zero_grad()
            loss = self.loss_fn()
            loss.backward()
            self.opt_adam.step()
            total += loss.item()
        return total / n_steps

    def lbfgs_step(self) -> float:
        """Uma chamada de L-BFGS com closure."""
        optimizer = LBFGS(
            self.model.parameters(),
            lr=1.0,
            max_iter=self.max_iter_lbfgs,
            history_size=self.history_size,
            line_search_fn="strong_wolfe"
        )

        def closure():
            optimizer.zero_grad()
            loss = self.loss_fn()
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        return loss.item() if torch.is_tensor(loss) else loss

    def train(
        self,
        adam_epochs: int = 5000,
        lbfgs_epochs: int = 3,
        print_every: int = 500,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ) -> List[float]:
        """Treinamento completo híbrido."""
        print("=== Fase Adam ===")
        for epoch in range(1, adam_epochs + 1):
            loss = self.adam_step()
            self.history.append(loss)
            if scheduler is not None:
                scheduler.step()
            if epoch % print_every == 0:
                print(f"Adam Epoch {epoch:5d} | Loss = {loss:.6e}")

        print("=== Fase L-BFGS ===")
        for i in range(lbfgs_epochs):
            loss = self.lbfgs_step()
            self.history.append(loss)
            print(f"L-BFGS Step {i+1} | Loss = {loss:.6e}")

        return self.history
