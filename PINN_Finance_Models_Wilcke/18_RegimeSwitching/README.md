# Mudança de Regime Macro-Financeira

**Autor da formulação original:** Luiz Tiago Wilcke  
**Capítulo de referência:** Capítulo 29

## Equação Principal

Sistema de EDPs acopladas com gerador infinitesimal de Markov multi-estado.

## Variáveis em Português

| $V_i$ | Preço no regime i |
| $q_{ij}$ | Taxa de transição i→j |
| Gerador | Matriz Q do processo de Markov |

## Arquitetura e Treinamento

- Rede fully-connected com ativação $\tanh$ (garantia de $C^\infty$)
- Função de perda composta (resíduo PDE + condições de contorno + terminal)
- Amostragem Latin Hypercube + reamostragem adaptativa por residual
- Otimização híbrida: Adam (exploração) → L-BFGS (refinamento de alta precisão)
- Extração de quantidades de interesse (Gregas, fronteira livre, controles ótimos) via autograd
- Suporte a GPU e salvamento de checkpoint

## Como executar

```bash
python train.py
```

Consulte o Capítulo Capítulo 29 da obra de Luiz Tiago Wilcke para a derivação completa e justificativa teórica de cada termo.
