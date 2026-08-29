# Alocação de Portfólio de Merton – Equação HJB

**Autor da formulação original:** Luiz Tiago Wilcke  
**Capítulo de referência:** Capítulo 6

## Equação Principal

$$\frac{\partial J}{\partial t} + \sup_{\pi,c}\Bigl\{\mu\pi x J_x + \frac12\sigma^2\pi^2 x^2 J_{xx} - c J_x + U(c) - \rho J\Bigr\} = 0$$

## Descrição

Controle estocástico clássico resolvido por PINN com controles ótimos analíticos embutidos.

## Variáveis (português)

As variáveis seguem a notação padrão de finanças quantitativas apresentada na obra de Wilcke. Consulte o Capítulo indicado para a definição completa de cada símbolo.

## Arquitetura

- Rede fully-connected com ativação tanh (C∞)
- Função de perda composta (PDE + BC + IC/terminal)
- Otimização híbrida Adam + L-BFGS
- Amostragem Latin Hypercube / reamostragem adaptativa
- Extração de Gregas via autograd quando aplicável

## Execução

```bash
python train.py
```
