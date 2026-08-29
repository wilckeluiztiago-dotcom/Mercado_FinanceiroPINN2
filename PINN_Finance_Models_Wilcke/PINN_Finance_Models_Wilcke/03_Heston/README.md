# Modelo de Volatilidade Estocástica de Heston

**Autor da formulação original:** Luiz Tiago Wilcke  
**Capítulo de referência:** Capítulo 5

## Equação Principal

$$\frac{\partial V}{\partial t} + \frac12 v S^2 V_{SS} + \rho\xi v S V_{Sv} + \frac12 \xi^2 v V_{vv} + r S V_S + \kappa(\theta-v)V_v - rV = 0$$

## Descrição

EDP bidimensional com correlação, vol-of-vol e extração de Gregas via autograd.

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
