# Modelo SABR de Volatilidade

**Autor da formulação original:** Luiz Tiago Wilcke  
**Capítulo de referência:** Capítulo 13

## Equação Principal

$$dF_t = \alpha_t F_t^\beta dW^1_t,\quad d\alpha_t = \nu\alpha_t dW^2_t$$

## Variáveis em Português

| $F$ | Forward |
| $\alpha$ | Volatilidade estocástica |
| $\beta$ | Expoente CEV |
| $\nu$ | Vol-of-vol |

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

Consulte o Capítulo Capítulo 13 da obra de Luiz Tiago Wilcke para a derivação completa e justificativa teórica de cada termo.
