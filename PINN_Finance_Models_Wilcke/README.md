# Physics-Informed Neural Networks for Quantitative Finance

Modelos de PINNs para precificação de derivativos, volatilidade estocástica, controle ótimo e risco.  
Inspirado no livro *Redes Neurais Informadas pela Física – Volume II* de **Luiz Tiago Wilcke**.

---

## Instalação

```bash
pip install -r requirements.txt
```

---

## Estrutura

```
PINN_Finance_Models_Wilcke/
├── utils/                  # amostragem, otimizadores, Gregas, redes avançadas
├── 01_BlackScholes/        # Call europeia (+ versão avançada)
├── 02_AmericanOptions/     # Inequação variacional + fronteira livre
├── 03_Heston/              # Volatilidade estocástica 2D
├── 04_MertonHJB/           # Alocação ótima de portfólio
├── 05_CIR/                 # Taxa de juros + zero-cupom
├── 06_AvellanedaStoikov/   # Market making + Fokker-Planck
├── 07_Dupire/              # Volatilidade local (problema inverso)
├── 08_DeepONet/            # Operadores neurais
├── 09_MeanFieldGames/      # Jogos de campo médio
├── 10_SABR/                # Modelo SABR
├── 11_MultiAsset/          # Cesta multiativo
├── 12_DeepHedging/         # Cobertura ótima
├── 13_JumpDiffusion/       # Salto-difusão (PIDE)
├── 14_HullWhite/           # Hull-White
├── 15_AlmgrenChriss/       # Liquidação ótima
├── 16_RoughVolatility/     # Volatilidade rough / fPINN
├── 17_CreditRisk/          # Risco de crédito estrutural
├── 18_RegimeSwitching/     # Mudança de regime
├── 19_xVA/                 # CVA / DVA / FVA
└── 20_CarbonHJB/           # Abatimento de carbono
```

Cada pasta tem `config.py`, `model.py`, `train.py` e um `README.md` próprio.

---

## Arquiteturas avançadas (`utils/advanced_networks.py`)

- Fourier Feature Embedding  
- Residual Blocks + LayerNorm  
- DGM Cell (Deep Galerkin)  
- Self-Adaptive Loss Weights  
- Hard Constraint Layer  
- Multi-Head PINN  
- Causal Training Mask  
- Residual Adaptive Refinement (RAR)  

```bash
cd 01_BlackScholes
python train_advanced.py
```

---

## Modelos

### 01 – Black-Scholes

$$
\frac{\partial V}{\partial t} + \frac12\sigma^2 S^2\frac{\partial^2 V}{\partial S^2} + rS\frac{\partial V}{\partial S} - rV = 0
$$

Call europeia. Residual da EDP + contornos + payoff. Gregas via autograd.

### 02 – Opções Americanas

$$
\min\bigl(-\partial_t V - \mathcal{L}V,\ V-(S-K)^+\bigr)=0
$$

Penalização contínua + extração da fronteira livre $S_f(t)$ e smooth pasting.

### 03 – Heston

$$
\begin{aligned}
dS_t &= (r-q)S_t\,dt + \sqrt{v_t}S_t\,dW^S_t \\
dv_t &= \kappa(\theta-v_t)\,dt + \xi\sqrt{v_t}\,dW^v_t
\end{aligned}
$$

EDP 2D. Extração de $\Delta$, $\Gamma$, Vega, Vanna, Volga.

### 04 – Merton HJB

$$
\partial_t J + \sup_{\pi,c}\bigl\{\mu\pi x J_x + \tfrac12\sigma^2\pi^2 x^2 J_{xx} - c J_x + U(c) - \rho J\bigr\}=0
$$

Controles ótimos $\pi^*$ e $c^*$ por autograd.

### 05 – CIR

$$
\partial_t P + \kappa(\theta-r)\partial_r P + \tfrac12\sigma^2 r\partial_{rr}P - rP = 0
$$

Precificação de zero-cupom. Tratamento da singularidade em $r=0$.

### 06 – Avellaneda-Stoikov

HJB de inventário + Fokker-Planck da densidade de ordens no LOB.

### 07 – Dupire (inverso)

$$
\partial_T C = \tfrac12\sigma_{\mathrm{loc}}^2(K,T)K^2\partial_{KK}C
$$

$\sigma_{\mathrm{loc}}$ parametrizado por rede neural.

### 08 – DeepONet

Operador Branch-Trunk: parâmetros de mercado $\to$ superfície de preços. Versão físico-informada (PI-DeepONet).

### 09 – Mean Field Games

Sistema acoplado HJB + Fokker-Planck (Lasry-Lions).

### 10 – SABR

$$
dF_t = \alpha_t F_t^\beta\,dW^1_t,\quad d\alpha_t = \nu\alpha_t\,dW^2_t
$$

Smile de volatilidade. Regularização do expoente $\beta$.

### 11 – Multiativo

EDP de dimensão $d$ com matriz de correlação. Solução mesh-free (evita maldição da dimensionalidade).

### 12 – Deep Hedging

Rede de política $\delta_t=\pi_\theta(S_t,t)$ treinada por CVaR / variância do P&L.

### 13 – Salto-Difusão (Merton)

$$
\partial_t V + \mathcal{L}V + \lambda\int\bigl(V(Se^y)-V\bigr)\nu(dy)=0
$$

PIDE. Termo integral por amostragem Monte-Carlo no grafo.

### 14 – Hull-White

$$
dr_t = \bigl(\theta(t)-a r_t\bigr)dt + \sigma\,dW_t
$$

Calibração dinâmica de $\theta(t)$ via PINN.

### 15 – Almgren-Chriss

HJB de liquidação ótima com impacto permanente e temporário.

### 16 – Volatilidade Rough / fPINN

$$
{}^C D_t^\alpha V = \frac{1}{\Gamma(1-\alpha)}\int_0^t(t-s)^{-\alpha}\partial_s V\,ds
$$

Operador de Caputo por quadratura de Gauss-Jacobi diferenciável.

### 17 – Risco de Crédito (Merton + Black-Cox)

Equity como call sobre ativos da firma + barreira de default contínua.

### 18 – Mudança de Regime

$$
\partial_t V_i + \mathcal{L}_i V_i + \sum_j q_{ij}V_j = 0
$$

Sistema de EDPs acopladas. Arquitetura multi-head.

### 19 – xVA (CVA/DVA/FVA)

EDP com termo de fonte não-linear de funding e default bilateral.

### 20 – Abatimento de Carbono

HJB de controle ótimo da taxa de abatimento de emissões.

---

## Como rodar

```bash
cd 01_BlackScholes
python train.py              # clássico
python train_advanced.py     # Fourier + Residual + Adaptive + RAR
```

```bash
cd utils && python demo_advanced.py
```

---

## Notação

| Símbolo | Significado |
|---------|-------------|
| $S$ | Preço do ativo |
| $v$ | Variância |
| $r$ | Taxa livre de risco |
| $\sigma$ | Volatilidade |
| $K$ | Strike |
| $V$ | Preço do derivativo |
| $J$ | Função valor (HJB) |
| $\pi$ | Fração no ativo de risco |
| $\kappa,\theta,\xi$ | Reversão, nível de longo prazo, vol-of-vol |
| $\rho$ | Correlação |
| $S_f(t)$ | Fronteira livre |
| $\lambda$ | Intensidade de saltos / penalização |

---

Inspirado em: Wilcke, Luiz Tiago. *Redes Neurais Informadas pela Física – Volume II*.
