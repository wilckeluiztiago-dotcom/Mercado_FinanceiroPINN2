# Market Making Avellaneda-Stoikov + Fokker-Planck

**Autor da formulação original:** Luiz Tiago Wilcke  
**Capítulo de referência:** Capítulo 8

## Equação Principal

HJB de inventário acoplada à equação forward de Fokker-Planck da densidade de ordens no Limit Order Book.

## Variáveis em Português

| $q$ | Inventário do market maker |
| $\delta^{a,b}$ | Spreads ask/bid |
| $m$ | Densidade de probabilidade de estados |

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

Consulte o Capítulo Capítulo 8 da obra de Luiz Tiago Wilcke para a derivação completa e justificativa teórica de cada termo.
