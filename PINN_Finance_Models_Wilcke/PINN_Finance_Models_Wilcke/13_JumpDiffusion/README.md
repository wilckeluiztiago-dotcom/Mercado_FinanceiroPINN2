# Salto-Difusão de Merton – PIDE

**Autor da formulação original:** Luiz Tiago Wilcke  
**Capítulo de referência:** Capítulo 17

## Equação Principal

$$\partial_t V + \mathcal{L}V + \lambda\int (V(Se^y)-V)\nu(dy) = 0$$

## Descrição

Equação integro-diferencial com amostragem de saltos no grafo.

## Variáveis (português)

As variáveis seguem a notação padrão de finanças quantitativas apresentada na obra de Wilcke. Consulte o Capítulo indicado para a definição completa de cada símbolo.

## Arquitetura

- Rede fully-connected com ativação tanh (C∞)
- Função de perda composta (PDE + BC + IC/terminal)
- Otimização híbrida Adam + L-BFGS
- Amostragem Latin Hypercube / reamostragem adaptativa
- Extração de Gregas via autograd quando aplicável

## Execução

```bash
python train.py
```
