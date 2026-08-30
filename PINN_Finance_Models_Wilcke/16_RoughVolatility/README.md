# Volatilidade Rough e fPINN (Operador de Caputo)

**Autor da formulação original:** Luiz Tiago Wilcke  
**Capítulo de referência:** Capítulos 22 e 27

## Equação Principal

$${}^C D_t^\\alpha V = \\frac{1}{\\Gamma(1-\\alpha)}\\int_0^t (t-s)^{-\\alpha}\\partial_s V\\,ds$$

## Variáveis em Português

| $\alpha = H + 1/2$ | Ordem fracionária (H = expoente de Hurst) |
| fPINN | Physics-Informed Neural Network fracionária |

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

Consulte o Capítulo Capítulos 22 e 27 da obra de Luiz Tiago Wilcke para a derivação completa e justificativa teórica de cada termo.
