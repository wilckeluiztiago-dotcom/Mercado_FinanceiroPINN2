# 02 – Opções Americanas (Inequação Variacional)

**Autor da formulação:** Luiz Tiago Wilcke  
**Capítulo:** 4

## Formulação de Complementaridade

$$
\min\left( -\frac{\partial V}{\partial t} - \mathcal{L}V,\quad V - (S-K)^+ \right) = 0
$$

## Método de Penalização Contínua (Wilcke Cap. 4)

$$
\frac{\partial V}{\partial t} + \mathcal{L}V + \lambda \max\bigl((S-K)^+ - V, 0\bigr)^p = 0
$$

## Condições de Smooth Pasting

$$
V(S_f(t),t) = (S_f(t)-K)^+,\qquad \frac{\partial V}{\partial S}(S_f(t),t) = 1
$$

## Variáveis em Português

| Símbolo | Significado |
|---------|-------------|
| $V(S,t)$ | Preço da opção americana |
| $S_f(t)$ | Fronteira livre de exercício ótimo |
| $\lambda$ | Parâmetro de penalização |
| $q$ | Taxa de dividendos contínua |
| $\mathcal{L}$ | Operador de Black-Scholes |

## Arquitetura

- 7 camadas × 160 neurônios, ativação tanh
- Perda composta: PDE penalizada + contorno + terminal + probe de smooth pasting
- Extração automática da fronteira livre por varredura
- Otimização híbrida Adam → L-BFGS
