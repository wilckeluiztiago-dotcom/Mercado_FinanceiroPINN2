# Volatilidade Local de Dupire – Problema Inverso

**Autor da formulação original:** Luiz Tiago Wilcke  
**Capítulo de referência:** Capítulo 9

## Equação Principal

$$\frac{\partial C}{\partial T} = \frac12\sigma_{loc}^2(K,T)K^2\frac{\partial^2 C}{\partial K^2} - (r-q)K\frac{\partial C}{\partial K}$$

## Variáveis em Português

| $\sigma_{loc}(K,T)$ | Superfície de volatilidade local |
| $C(K,T)$ | Preço de call observado |

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

Consulte o Capítulo Capítulo 9 da obra de Luiz Tiago Wilcke para a derivação completa e justificativa teórica de cada termo.
