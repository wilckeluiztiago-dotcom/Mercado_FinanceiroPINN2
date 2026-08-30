# 01 – PINN Black-Scholes (Opções Europeias)

**Autor da formulação original:** Luiz Tiago Wilcke  
**Capítulos de referência:** 1 (Fundamentos + Lema de Itô) e 3 (Precificação + Convergência + Gregas)

## Equação Diferencial Parcial

$$
\frac{\partial V}{\partial t} + \frac12\sigma^2 S^2\frac{\partial^2 V}{\partial S^2} + r S\frac{\partial V}{\partial S} - r V = 0
$$

## Condições de Contorno e Terminal (Call)

$$
\begin{aligned}
V(S,T) &= \max(S-K,0)\\
V(0,t) &= 0\\
\lim_{S\to\infty}V(S,t) &= S - K e^{-r(T-t)}
\end{aligned}
$$

## Variáveis em Português

| Símbolo | Significado |
|---------|-------------|
| $S$ | Preço do ativo subjacente |
| $t$ | Tempo calendário |
| $\sigma$ | Volatilidade constante |
| $r$ | Taxa de juros livre de risco |
| $K$ | Strike (preço de exercício) |
| $V(S,t)$ | Valor da opção |

## Arquitetura

- Rede totalmente conectada com 6 camadas ocultas de 128 neurônios
- Ativação $\tanh$ (requisito de $C^\infty$ para autograd de 2ª ordem)
- Função de perda composta ponderada (PDE + contorno + terminal)
- Otimização híbrida Adam → L-BFGS
- Amostragem Latin Hypercube

## Como executar

```bash
python train.py
```
