# Deep Hedging – Cobertura Ótima

**Autor da formulação original:** Luiz Tiago Wilcke  
**Capítulo de referência:** Capítulo 16

## Equação Principal

Rede de política de cobertura δ_t = π_θ(S_t, t) treinada por minimização de CVaR ou variância do P&L.

## Variáveis em Português

| $\delta_t$ | Posição de cobertura |
| $PnL$ | Resultado da estratégia de hedge |
| CVaR | Conditional Value-at-Risk |

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

Consulte o Capítulo Capítulo 16 da obra de Luiz Tiago Wilcke para a derivação completa e justificativa teórica de cada termo.
