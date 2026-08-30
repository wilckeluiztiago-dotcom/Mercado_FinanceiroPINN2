# Salto-Difusão de Merton – PIDE

**Autor da formulação original:** Luiz Tiago Wilcke  
**Capítulo de referência:** Capítulo 17

## Equação Principal

$$\partial_t V + \mathcal{L}V + \lambda\int_{-\infty}^\infty\bigl(V(Se^y,t)-V(S,t)\bigr)\nu(dy)=0$$

## Variáveis em Português

| $\lambda$ | Intensidade de saltos |
| $\nu(dy)$ | Medida de Lévy dos saltos |
| PIDE | Equação integro-diferencial parcial |

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

Consulte o Capítulo Capítulo 17 da obra de Luiz Tiago Wilcke para a derivação completa e justificativa teórica de cada termo.
