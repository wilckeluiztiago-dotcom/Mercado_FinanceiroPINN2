# Alocação de Portfólio de Merton – Equação HJB

**Autor da formulação original:** Luiz Tiago Wilcke  
**Capítulo de referência:** Capítulo 6

## Equação Principal

$$\frac{\partial J}{\partial t} + \sup_{\pi,c}\Bigl\{\mu\pi x J_x + \frac12\sigma^2\pi^2 x^2 J_{xx} - c J_x + U(c) - \rho J\Bigr\} = 0$$

## Variáveis em Português

| $J(x,t)$ | Função valor |
| $\pi$ | Fração do portfólio no ativo de risco |
| $c$ | Taxa de consumo |
| $U$ | Função de utilidade (CRRA/log) |

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

Consulte o Capítulo Capítulo 6 da obra de Luiz Tiago Wilcke para a derivação completa e justificativa teórica de cada termo.
