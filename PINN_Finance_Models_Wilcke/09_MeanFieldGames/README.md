# Jogos de Campo Médio (Mean Field Games)

**Autor da formulação original:** Luiz Tiago Wilcke  
**Capítulo de referência:** Capítulos 11 e 23

## Equação Principal

Sistema acoplado HJB + Fokker-Planck de Lasry-Lions para equilíbrio de Nash em população infinita.

## Variáveis em Português

| $u$ | Função valor individual |
| $m$ | Densidade de população |
| $H$ | Hamiltoniano |

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

Consulte o Capítulo Capítulos 11 e 23 da obra de Luiz Tiago Wilcke para a derivação completa e justificativa teórica de cada termo.
