# Opções sobre Cesta Multiativo (Alta Dimensão)

**Autor da formulação original:** Luiz Tiago Wilcke  
**Capítulo de referência:** Capítulo 15

## Equação Principal

EDP de dimensão d com matriz de correlação completa – solução meshless via PINN.

## Variáveis em Português

| $S_i$ | Preço do i-ésimo ativo |
| $\rho_{ij}$ | Correlação entre ativos |
| $d$ | Número de ativos (maldição da dimensionalidade) |

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

Consulte o Capítulo Capítulo 15 da obra de Luiz Tiago Wilcke para a derivação completa e justificativa teórica de cada termo.
