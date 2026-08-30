# Liquidação Ótima – Almgren-Chriss

**Autor da formulação original:** Luiz Tiago Wilcke  
**Capítulo de referência:** Capítulo 19

## Equação Principal

HJB de execução ótima de grandes lotes com impacto permanente e temporário de mercado.

## Variáveis em Português

| $x$ | Inventário restante |
| $v$ | Velocidade de liquidação |
| $\eta,\gamma$ | Coeficientes de impacto |

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

Consulte o Capítulo Capítulo 19 da obra de Luiz Tiago Wilcke para a derivação completa e justificativa teórica de cada termo.
