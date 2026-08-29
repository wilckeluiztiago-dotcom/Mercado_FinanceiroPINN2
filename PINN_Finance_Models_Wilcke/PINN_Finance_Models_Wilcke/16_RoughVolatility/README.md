# Volatilidade Rough e fPINN (Caputo)

**Autor da formulação original:** Luiz Tiago Wilcke  
**Capítulo de referência:** Capítulos 22 e 27

## Equação Principal

$${}^C D_t^\alpha V = \frac1{\Gamma(1-\alpha)}\int_0^t (t-s)^{-\alpha} \partial_s V\, ds$$

## Descrição

Operador fracionário implementado com quadratura Gauss-Jacobi diferenciável.

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
