# Risco de Crédito Estrutural (Merton + Black-Cox)

**Autor da formulação original:** Luiz Tiago Wilcke  
**Capítulo de referência:** Capítulo 26

## Equação Principal

Modelo de Merton como call sobre ativos da firma + barreira contínua de default (Black-Cox).

## Variáveis em Português

| $V$ | Valor dos ativos da firma |
| $D$ | Face value da dívida |
| $B(t)$ | Barreira de default |

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

Consulte o Capítulo Capítulo 26 da obra de Luiz Tiago Wilcke para a derivação completa e justificativa teórica de cada termo.
