# Volatilidade Local de Dupire – Problema Inverso

**Autor da formulação original:** Luiz Tiago Wilcke  
**Capítulo de referência:** Capítulo 9

## Equação Principal

$$\frac{\partial C}{\partial T} = \frac12 \sigma_{loc}^2(K,T) K^2 \frac{\partial^2 C}{\partial K^2}$$

## Descrição

Calibração inversa de superfície de volatilidade local via PINN.

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
