# 03 – Modelo de Heston (Volatilidade Estocástica)

**Autor:** Luiz Tiago Wilcke – **Capítulo 5**

## Sistema de EDEs

$$
\begin{aligned}
dS_t &= (r-q)S_t dt + \sqrt{v_t} S_t dW^S_t \\
dv_t &= \kappa(\theta - v_t)dt + \xi\sqrt{v_t} dW^v_t
\end{aligned}
$$

## EDP Bidimensional

$$
\frac{\partial V}{\partial t} + \frac12 v S^2 V_{SS} + \rho\xi v S V_{Sv} + \frac12\xi^2 v V_{vv} + (r-q)S V_S + \kappa(\theta-v)V_v - r V = 0
$$

## Variáveis em Português

| Símbolo | Significado |
|---------|-------------|
| $v$ | Variância instantânea |
| $\kappa$ | Velocidade de reversão à média |
| $\theta$ | Nível de longo prazo da variância |
| $\xi$ | Volatilidade da volatilidade (vol-of-vol) |
| $\rho$ | Correlação entre os Brownianos |

## Recursos implementados

- Rede 8×180 com normalização de entradas
- Resíduo completo da EDP 2D via autograd
- Extração nativa de Δ, Γ, Vega, Vanna, Volga
- Amostragem Latin Hypercube 3D
- Otimização híbrida Adam + L-BFGS


## Extensão Avançada

Para usar Fourier Features + Residual Blocks + Adaptive Weights no Heston:

```python
from utils.advanced_networks import AdvancedPINN, AdaptiveLossWeights, MultiHeadPINN
model = AdvancedPINN(in_dim=3, hidden_dim=180, n_res_blocks=6, use_fourier=True)
```

A perda adaptativa e o Residual Adaptive Sampler (RAR) do módulo `utils.advanced_networks` podem ser plugados diretamente no `HestonLoss`.
