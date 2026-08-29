# Modelo CIR de Taxa de Juros e Títulos Zero-Cupom

**Autor da formulação original:** Luiz Tiago Wilcke  
**Capítulo de referência:** Capítulo 7

## Equação Principal

$$\frac{\partial P}{\partial t} + \kappa(\theta-r)P_r + \frac12\sigma^2 r P_{rr} - r P = 0$$

## Descrição

Estabilização da singularidade em r=0 via PINN.

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
