# DeepONet / Operador Neural para EDPs Financeiras

**Autor da formulação original:** Luiz Tiago Wilcke  
**Capítulo de referência:** Capítulo 10

## Equação Principal

Mapeamento funcional Branch-Trunk: parâmetros de mercado → superfície de preços de opções.

## Variáveis em Português

| Branch | Rede que processa a função de entrada (ex.: curva de volatilidade) |
| Trunk | Rede que processa o ponto de avaliação (S,t) |

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

Consulte o Capítulo Capítulo 10 da obra de Luiz Tiago Wilcke para a derivação completa e justificativa teórica de cada termo.
