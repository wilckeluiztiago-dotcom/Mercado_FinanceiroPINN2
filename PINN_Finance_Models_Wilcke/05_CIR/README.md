# Modelo CIR de Taxa de Juros e Precificação de Títulos

**Autor da formulação original:** Luiz Tiago Wilcke  
**Capítulo de referência:** Capítulo 7

## Equação Principal

$$\frac{\partial P}{\partial t} + \kappa(\theta-r)\frac{\partial P}{\partial r} + \frac12\sigma^2 r\frac{\partial^2 P}{\partial r^2} - r P = 0$$

## Variáveis em Português

| $r$ | Taxa de juros de curto prazo |
| $\kappa$ | Velocidade de reversão |
| $\theta$ | Nível de longo prazo |
| $P(r,t)$ | Preço do título zero-cupom |

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

Consulte o Capítulo Capítulo 7 da obra de Luiz Tiago Wilcke para a derivação completa e justificativa teórica de cada termo.
