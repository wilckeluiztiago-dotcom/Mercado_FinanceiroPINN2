# Opções Americanas – Inequação Variacional e Fronteira Livre

**Autor da formulação original:** Luiz Tiago Wilcke  
**Capítulo de referência:** Capítulo 4

## Equação Principal

$$\min\left( -\frac{\partial V}{\partial t} - \mathcal{L}V,\ V - (S-K)^+ \right) = 0$$

## Descrição

Formulação de complementaridade com método de penalização contínua e extração da fronteira livre Sf(t).

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
