# Modelo de Hull-White e Calibração Dinâmica

**Autor da formulação original:** Luiz Tiago Wilcke  
**Capítulo de referência:** Capítulo 18

## Equação Principal

$$dr_t = \bigl(\\theta(t) - a r_t\\bigr)dt + \\sigma dW_t$$

## Variáveis em Português

| $\theta(t)$ | Função de drift determinística (calibração) |
| $a$ | Velocidade de reversão |
| $P(r,t;T)$ | Preço do título |

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

Consulte o Capítulo Capítulo 18 da obra de Luiz Tiago Wilcke para a derivação completa e justificativa teórica de cada termo.
