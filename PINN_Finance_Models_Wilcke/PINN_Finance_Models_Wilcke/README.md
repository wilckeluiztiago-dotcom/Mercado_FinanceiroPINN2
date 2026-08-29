# Redes Neurais Informadas pela Física (PINNs) aplicadas ao Mercado Financeiro

**Autor único:** Luiz Tiago Wilcke  
**Obra de referência:** *Redes Neurais Informadas pela Física – Volume II: Edição Especial – Precificação de Derivativos, Volatilidade Estocástica e Equações HJB de Larga Escala*  
**Licença de uso educacional:** Baseado integralmente nos fundamentos, derivações e arquiteturas apresentadas na obra de Luiz Tiago Wilcke.

Este repositório contém **20 modelos completos e de produção** de Physics-Informed Neural Networks (PINNs) para problemas de precificação, controle estocástico, volatilidade, risco e macro-finanças. Cada modelo é implementado de forma modular, com múltiplas classes, funções de perda compostas, amostragem Latin Hypercube, otimização híbrida Adam + L-BFGS, cálculo nativo de Gregas via autograd e suporte a GPU.

## Estrutura do Repositório

```
PINN_Finance_Models_Wilcke/
├── README.md                          (este arquivo)
├── requirements.txt
├── utils/
│   ├── sampling.py                    (Latin Hypercube + reamostragem adaptativa)
│   ├── optimizers.py                  (wrappers Adam + L-BFGS)
│   └── greeks.py                      (extração de Δ, Γ, Vega, Vanna, etc.)
├── 01_BlackScholes/
├── 02_AmericanOptions/
├── 03_Heston/
├── 04_MertonHJB/
├── 05_CIR/
├── 06_AvellanedaStoikov/
├── 07_Dupire/
├── 08_DeepONet/
├── 09_MeanFieldGames/
├── 10_SABR/
├── 11_MultiAsset/
├── 12_DeepHedging/
├── 13_JumpDiffusion/
├── 14_HullWhite/
├── 15_AlmgrenChriss/
├── 16_RoughVolatility/
├── 17_CreditRisk/
├── 18_RegimeSwitching/
├── 19_xVA/
└── 20_CarbonHJB/
```

Cada pasta contém:
- `README.md` específico com equações renderizadas
- `model.py` (arquitetura neural + perda física)
- `train.py` (pipeline de treinamento completo)
- `config.py` (hiperparâmetros)

---

## 1. Black-Scholes Europeu (Capítulos 1 e 3)

**Equação de Black-Scholes:**

$$
\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2\frac{\partial^2 V}{\partial S^2} + rS\frac{\partial V}{\partial S} - rV = 0
$$

**Condições de contorno (Call Europeia):**

$$
V(S,T) = \max(S-K,0),\qquad V(0,t)=0,\qquad V(S\to\infty,t)\sim S
$$

**Variáveis (português):**
- $S$ : preço do ativo subjacente
- $t$ : tempo
- $\sigma$ : volatilidade constante
- $r$ : taxa de juros livre de risco
- $K$ : preço de exercício
- $V(S,t)$ : valor da opção

**Origem:** Capítulos 1 (fundamentos + Lema de Itô) e 3 (PINN + convergência + Gregas via autograd).

---

## 2. Opções Americanas – Inequação Variacional (Capítulo 4)

**Formulação de complementaridade:**

$$
\min\left( -\frac{\partial V}{\partial t} - \mathcal{L}V,\quad V - (S-K)^+ \right) = 0
$$

onde $\mathcal{L}$ é o operador de Black-Scholes.

**Método de penalização contínua:**

$$
\frac{\partial V}{\partial t} + \mathcal{L}V + \lambda\max( (S-K)^+ - V, 0) = 0
$$

**Variáveis:**
- $\lambda$ : parâmetro de penalização
- $S_f(t)$ : fronteira livre de exercício ótimo

**Origem:** Capítulo 4 (smooth pasting + penalização em PINNs).

---

## 3. Modelo de Heston (Capítulo 5)

**Sistema de EDEs:**

$$
\begin{aligned}
dS_t &= rS_t\,dt + \sqrt{v_t}S_t\,dW_t^S\\
dv_t &= \kappa(\theta - v_t)\,dt + \xi\sqrt{v_t}\,dW_t^v
\end{aligned}
$$

**EDP bidimensional de Heston:**

$$
\frac{\partial V}{\partial t} + \frac12 vS^2\frac{\partial^2 V}{\partial S^2} + \rho\xi vS\frac{\partial^2 V}{\partial S\partial v} + \frac12\xi^2 v\frac{\partial^2 V}{\partial v^2} + rS\frac{\partial V}{\partial S} + \kappa(\theta-v)\frac{\partial V}{\partial v} - rV = 0
$$

**Variáveis:**
- $v$ : variância instantânea
- $\kappa$ : velocidade de reversão
- $\theta$ : nível de longo prazo
- $\xi$ : vol-of-vol
- $\rho$ : correlação entre Brownianos

**Origem:** Capítulo 5 (EDP 2D + função característica + Gregas Δ, Γ, Vanna).

---

## 4. Alocação de Portfólio de Merton – HJB (Capítulo 6)

**Equação de Hamilton-Jacobi-Bellman:**

$$
\frac{\partial J}{\partial t} + \sup_{\pi,c}\left\{ \mu\pi x\frac{\partial J}{\partial x} + \frac12\sigma^2\pi^2 x^2\frac{\partial^2 J}{\partial x^2} - c\frac{\partial J}{\partial x} + U(c) - \rho J \right\} = 0
$$

**Controles ótimos:**

$$
\pi^* = -\frac{\mu-r}{\sigma^2 x}\frac{J_x}{J_{xx}},\qquad c^* = (U')^{-1}(J_x)
$$

**Variáveis:**
- $J(x,t)$ : função valor
- $\pi$ : fração investida no ativo de risco
- $c$ : taxa de consumo
- $U$ : utilidade (CRRA ou log)

**Origem:** Capítulo 6.

---

## 5. Modelo CIR de Taxa de Juros (Capítulo 7)

**Dinâmica CIR:**

$$
dr_t = \kappa(\theta - r_t)\,dt + \sigma\sqrt{r_t}\,dW_t
$$

**EDP de título zero-cupom:**

$$
\frac{\partial P}{\partial t} + \kappa(\theta-r)\frac{\partial P}{\partial r} + \frac12\sigma^2 r\frac{\partial^2 P}{\partial r^2} - rP = 0
$$

**Origem:** Capítulo 7 (estabilização de singularidade em $r=0$).

---

## 6. Market Making – Avellaneda-Stoikov + Fokker-Planck (Capítulo 8)

**HJB de inventário + densidade de ordens via Fokker-Planck.**

**Origem:** Capítulo 8.

---

## 7. Volatilidade Local de Dupire – Problema Inverso (Capítulo 9)

**Equação de Dupire:**

$$
\frac{\partial C}{\partial T} = \frac12\sigma_{\text{loc}}^2(K,T)K^2\frac{\partial^2 C}{\partial K^2} - rK\frac{\partial C}{\partial K}
$$

**Calibração inversa via PINN:** $\sigma_{\text{loc}}(K,T)$ é parametrizado por uma rede neural.

**Origem:** Capítulo 9.

---

## 8. DeepONet para Operadores Financeiros (Capítulo 10)

Mapeamento funcional de parâmetros de mercado → superfície de preços.

**Origem:** Capítulo 10.

---

## 9. Jogos de Campo Médio (Mean Field Games) (Capítulos 11 e 23)

Sistema acoplado HJB + Fokker-Planck:

$$
\begin{aligned}
-\partial_t u - \frac12\sigma^2\partial_{xx}u + H(x,\partial_x u,m) &= 0\\
\partial_t m - \frac12\sigma^2\partial_{xx}m - \partial_x\bigl(m\partial_p H\bigr) &= 0
\end{aligned}
$$

**Origem:** Capítulos 11 e 23.

---

## 10. Modelo SABR (Capítulo 13)

**Dinâmica:**

$$
\begin{aligned}
dF_t &= \alpha_t F_t^\beta\,dW_t^1\\
d\alpha_t &= \nu\alpha_t\,dW_t^2
\end{aligned}
$$

**EDP SABR + estabilização de expoentes fracionários.**

**Origem:** Capítulo 13.

---

## 11. Opções sobre Cesta Multiativo (Capítulo 15)

EDP de dimensão $d$:

$$
\frac{\partial V}{\partial t} + \frac12\sum_{i,j}\rho_{ij}\sigma_i\sigma_j S_i S_j\frac{\partial^2 V}{\partial S_i\partial S_j} + r\sum_i S_i\frac{\partial V}{\partial S_i} - rV = 0
$$

Solução meshless via PINN (evita maldição da dimensionalidade).

**Origem:** Capítulo 15.

---

## 12. Deep Hedging (Capítulo 16)

Rede de política de cobertura $\delta_t = \pi_\theta(S_t,t)$ treinada por minimização de risco (CVaR ou variância).

**Origem:** Capítulo 16 (Buehler et al. 2019 adaptado a PINN).

---

## 13. Salto-Difusão de Merton – PIDE (Capítulo 17)

**Equação integro-diferencial:**

$$
\frac{\partial V}{\partial t} + \mathcal{L}V + \lambda\int_{-\infty}^{\infty}\bigl(V(Se^y,t)-V(S,t)\bigr)\nu(dy) = 0
$$

Integração por amostragem Monte-Carlo dentro do grafo computacional.

**Origem:** Capítulo 17.

---

## 14. Modelo de Hull-White (Capítulo 18)

$$
dr_t = \bigl(\theta(t) - ar_t\bigr)dt + \sigma\,dW_t
$$

Calibração dinâmica da função $\theta(t)$ via PINN sem diferenciação numérica ruidosa.

**Origem:** Capítulo 18.

---

## 15. Liquidação Ótima – Almgren-Chriss (Capítulo 19)

HJB de execução:

$$
\frac{\partial J}{\partial t} + \sup_v\bigl\{ -v\frac{\partial J}{\partial x} + \frac12\sigma^2\frac{\partial^2 J}{\partial S^2} - \eta v^2 - \gamma x v \bigr\} = 0
$$

**Origem:** Capítulo 19.

---

## 16. Volatilidade Rough / fPINN (Capítulos 22 e 27)

Operador de Caputo:

$$
{}^C D_t^\alpha V = \frac{1}{\Gamma(1-\alpha)}\int_0^t (t-s)^{-\alpha}\frac{\partial V}{\partial s}\,ds
$$

Implementado via quadratura de Gauss-Jacobi diferenciável.

**Origem:** Capítulos 22 e 27 (Rough Bergomi + fPINN).

---

## 17. Risco de Crédito Estrutural – Merton & Black-Cox (Capítulo 26)

Modelo de Merton como opção sobre ativos da firma + barreira contínua de Black-Cox.

**Origem:** Capítulo 26.

---

## 18. Mudança de Regime Macro-Financeira (Capítulo 29)

Sistema de EDPs acopladas com gerador infinitesimal de Markov:

$$
\frac{\partial V_i}{\partial t} + \mathcal{L}_i V_i + \sum_j q_{ij}V_j = 0
$$

Arquitetura multi-head.

**Origem:** Capítulo 29.

---

## 19. xVA Não-Linear (CVA/DVA/FVA) (Capítulo 39)

EDP com termo de fonte não-linear de funding e default bilateral.

**Origem:** Capítulo 39.

---

## 20. Abatimento Ótimo de Carbono – HJB (Capítulo 43)

Controle estocástico do preço do carbono com custo de abatimento.

**Origem:** Capítulo 43.

---

## Instalação

```bash
pip install -r requirements.txt
```

## Como usar

Cada pasta possui um `train.py` autônomo. Exemplo:

```bash
cd 01_BlackScholes
python train.py
```

Todos os códigos foram escritos de forma original, inspirados nas formulações, arquiteturas e estratégias de treinamento descritas na obra de **Luiz Tiago Wilcke**, mantendo fidelidade científica aos Capítulos referenciados.

---

**Citação sugerida**

> Wilcke, Luiz Tiago. *Redes Neurais Informadas pela Física – Volume II*. 2024/2025. Modelos implementados e adaptados para este repositório educacional.
