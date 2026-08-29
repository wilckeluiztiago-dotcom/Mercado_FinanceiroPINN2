# Physics-Informed Neural Networks for Quantitative Finance

**Autor da obra de referência:** Luiz Tiago Wilcke  
**Obra:** *Redes Neurais Informadas pela Física – Volume II*  
*(Edição Especial: Precificação de Derivativos, Volatilidade Estocástica e Equações HJB de Larga Escala)*

Este repositório contém **20 modelos completos** de Physics-Informed Neural Networks (PINNs) aplicados a problemas clássicos e avançados de finanças quantitativas, todos baseados nas formulações, derivações e arquiteturas apresentadas na obra de **Luiz Tiago Wilcke**.

---

## Índice

1. [Visão Geral](#1-visão-geral)
2. [Instalação](#2-instalação)
3. [Estrutura do Repositório](#3-estrutura-do-repositório)
4. [Arquiteturas PINN Avançadas](#4-arquiteturas-pinn-avançadas)
5. [Lista dos 20 Modelos](#5-lista-dos-20-modelos)
6. [Como Executar](#6-como-executar)
7. [Convenções e Notação](#7-convenções-e-notação)
8. [Citação](#8-citação)

---

## 1. Visão Geral

As Physics-Informed Neural Networks incorporam o residual de equações diferenciais parciais (EDPs) diretamente na função de perda. Isso permite:

- Resolver EDPs de precificação **sem malha** (mesh-free)
- Tratar problemas de **alta dimensionalidade** (cestas, Heston 2D, multi-ativo)
- Resolver **problemas inversos** (calibração de volatilidade local, parâmetros de Heston)
- Extrair **Gregas** de forma nativa via diferenciação automática
- Lidar com **fronteiras livres**, **saltos**, **rugosidade** e **jogos de campo médio**

Todos os códigos utilizam **PyTorch**, amostragem **Latin Hypercube**, otimização híbrida **Adam + L-BFGS** e ativação $\tanh$ (garantia de $C^\infty$).

---

## 2. Instalação

```bash
cd PINN_Finance_Models_Wilcke
pip install -r requirements.txt
```

**Requisitos principais**

| Pacote       | Versão mínima |
|--------------|---------------|
| PyTorch      | ≥ 2.0         |
| NumPy        | ≥ 1.24        |
| SciPy        | ≥ 1.10        |
| pyDOE        | ≥ 0.3.8       |
| Matplotlib   | ≥ 3.7         |
| tqdm         | ≥ 4.65        |

GPU (CUDA) é fortemente recomendada para os modelos 2D/3D e multi-ativo.

---

## 3. Estrutura do Repositório

```
PINN_Finance_Models_Wilcke/
│
├── README.md                          ← este arquivo
├── requirements.txt
│
├── utils/
│   ├── sampling.py                    # Latin Hypercube + reamostragem adaptativa
│   ├── optimizers.py                  # Hybrid Adam + L-BFGS
│   ├── greeks.py                      # Extração de Δ, Γ, Vega, Vanna, Volga
│   ├── advanced_networks.py           # Fourier, Residual, DGM, Adaptive Weights...
│   └── demo_advanced.py               # Demonstração rápida das arquiteturas
│
├── 01_BlackScholes/                   # Call europeia + versão avançada
├── 02_AmericanOptions/                # Inequação variacional + fronteira livre
├── 03_Heston/                         # Volatilidade estocástica 2D
├── 04_MertonHJB/                      # Alocação ótima de portfólio
├── 05_CIR/                            # Taxa de juros + títulos zero-cupom
├── 06_AvellanedaStoikov/              # Market making + Fokker-Planck
├── 07_Dupire/                         # Volatilidade local (problema inverso)
├── 08_DeepONet/                       # Operadores neurais
├── 09_MeanFieldGames/                 # Jogos de campo médio
├── 10_SABR/                           # Modelo SABR + smile
├── 11_MultiAsset/                     # Opções sobre cesta (alta dimensão)
├── 12_DeepHedging/                    # Cobertura ótima por rede de política
├── 13_JumpDiffusion/                  # Salto-difusão de Merton (PIDE)
├── 14_HullWhite/                      # Modelo de taxa de juros
├── 15_AlmgrenChriss/                  # Liquidação ótima
├── 16_RoughVolatility/                # Volatilidade rough + fPINN (Caputo)
├── 17_CreditRisk/                     # Risco de crédito estrutural
├── 18_RegimeSwitching/                # Mudança de regime macro-financeira
├── 19_xVA/                            # CVA / DVA / FVA não-linear
└── 20_CarbonHJB/                      # Abatimento ótimo de carbono
```

Cada pasta contém:

- `config.py` — hiperparâmetros e parâmetros de mercado  
- `model.py` — arquitetura da rede + função de perda física  
- `train.py` — pipeline de treinamento completo  
- `README.md` — equações, variáveis em português e instruções  

---

## 4. Arquiteturas PINN Avançadas

O módulo `utils/advanced_networks.py` implementa técnicas de última geração alinhadas aos Capítulos 2, 3, 5, 10, 25 e 27 da obra de Wilcke:

| Componente | Descrição | Uso recomendado |
|------------|-----------|-----------------|
| **Fourier Feature Embedding** | Mapeamento senoidal/cossenoidal de alta frequência | Fronteiras livres, smile, rugosidade |
| **Residual Blocks + LayerNorm** | Skip connections para redes profundas | Qualquer EDP de alta dimensão |
| **DGM Cell** | Célula Deep Galerkin (Sirignano & Spiliopoulos) | Capítulo 25 |
| **Self-Adaptive Loss Weights** | Pesos de perda aprendíveis | Balanceamento automático PDE × BC × IC |
| **Hard Constraint Layer** | Força payoff/contorno por construção | Opções europeias e americanas |
| **Multi-Head PINN** | Cabeças compartilhadas | Regime-switching, MFG |
| **Causal Training Mask** | Curriculum temporal | EDPs evolutivas |
| **Residual Adaptive Refinement (RAR)** | Reamostragem dos pontos de maior residual | Todos os modelos (Cap. 5) |

### Exemplo de uso (Black-Scholes avançado)

```bash
cd 01_BlackScholes
python train_advanced.py
```

Arquiteturas disponíveis no factory:

- `residual_fourier` (recomendado)
- `dgm`
- `hard_constraint`
- `vanilla`

---

## 5. Lista dos 20 Modelos

### 01 – Black-Scholes (Europeia)

**Capítulos:** 1 e 3

$$
\frac{\partial V}{\partial t} + \frac12\sigma^2 S^2\frac{\partial^2 V}{\partial S^2} + rS\frac{\partial V}{\partial S} - rV = 0
$$

**Variáveis:** $S$ (preço do ativo), $t$ (tempo), $\sigma$ (volatilidade), $r$ (taxa livre de risco), $K$ (strike), $V(S,t)$ (preço da opção).

---

### 02 – Opções Americanas (Inequação Variacional)

**Capítulo:** 4

$$
\min\left(-\frac{\partial V}{\partial t}-\mathcal{L}V,\quad V-(S-K)^+\right)=0
$$

Método de **penalização contínua** + extração da fronteira livre $S_f(t)$ e condições de *smooth pasting*.

---

### 03 – Modelo de Heston

**Capítulo:** 5

$$
\begin{aligned}
dS_t &= (r-q)S_t\,dt + \sqrt{v_t}S_t\,dW_t^S \\
dv_t &= \kappa(\theta-v_t)\,dt + \xi\sqrt{v_t}\,dW_t^v
\end{aligned}
$$

EDP bidimensional completa com extração nativa de $\Delta$, $\Gamma$, Vega, Vanna e Volga.

---

### 04 – Alocação de Portfólio de Merton (HJB)

**Capítulo:** 6

$$
\frac{\partial J}{\partial t} + \sup_{\pi,c}\Bigl\{\mu\pi x J_x + \tfrac12\sigma^2\pi^2 x^2 J_{xx} - c J_x + U(c) - \rho J\Bigr\}=0
$$

Controles ótimos $\pi^*$ e $c^*$ obtidos analiticamente e substituídos na equação.

---

### 05 – Modelo CIR (Taxa de Juros)

**Capítulo:** 7

$$
\frac{\partial P}{\partial t} + \kappa(\theta-r)\frac{\partial P}{\partial r} + \tfrac12\sigma^2 r\frac{\partial^2 P}{\partial r^2} - rP = 0
$$

Tratamento da singularidade em $r=0$.

---

### 06 – Market Making (Avellaneda-Stoikov)

**Capítulo:** 8

HJB de inventário acoplada à equação de Fokker-Planck da densidade de ordens no Limit Order Book.

---

### 07 – Volatilidade Local de Dupire (Problema Inverso)

**Capítulo:** 9

$$
\frac{\partial C}{\partial T} = \tfrac12\sigma_{\mathrm{loc}}^2(K,T)K^2\frac{\partial^2 C}{\partial K^2}
$$

$\sigma_{\mathrm{loc}}$ é parametrizado por uma rede neural.

---

### 08 – DeepONet (Operadores Neurais)

**Capítulo:** 10

Mapeamento funcional Branch-Trunk: parâmetros de mercado $\to$ superfície de preços.

---

### 09 – Jogos de Campo Médio (Mean Field Games)

**Capítulos:** 11 e 23

Sistema acoplado HJB + Fokker-Planck de Lasry-Lions.

---

### 10 – Modelo SABR

**Capítulo:** 13

$$
dF_t = \alpha_t F_t^\beta\,dW_t^1,\qquad d\alpha_t = \nu\alpha_t\,dW_t^2
$$

Estabilização de expoentes fracionários e geração de smile.

---

### 11 – Opções sobre Cesta Multiativo

**Capítulo:** 15

EDP de dimensão $d$ com matriz de correlação completa — solução *meshless* (evita a maldição da dimensionalidade).

---

### 12 – Deep Hedging

**Capítulo:** 16

Rede de política de cobertura $\delta_t=\pi_\theta(S_t,t)$ treinada por minimização de CVaR.

---

### 13 – Salto-Difusão de Merton (PIDE)

**Capítulo:** 17

$$
\partial_t V + \mathcal{L}V + \lambda\int\bigl(V(Se^y,t)-V(S,t)\bigr)\nu(dy)=0
$$

Termo integral amostrado dentro do grafo computacional.

---

### 14 – Modelo de Hull-White

**Capítulo:** 18

$$
dr_t = \bigl(\theta(t)-a r_t\bigr)dt + \sigma\,dW_t
$$

Calibração dinâmica de $\theta(t)$ sem diferenciação numérica ruidosa.

---

### 15 – Liquidação Ótima (Almgren-Chriss)

**Capítulo:** 19

HJB de execução ótima de grandes lotes com impacto permanente e temporário.

---

### 16 – Volatilidade Rough / fPINN

**Capítulos:** 22 e 27

$$
{}^C D_t^\alpha V = \frac{1}{\Gamma(1-\alpha)}\int_0^t(t-s)^{-\alpha}\partial_s V\,ds
$$

Operador de Caputo implementado via quadratura de Gauss-Jacobi diferenciável.

---

### 17 – Risco de Crédito Estrutural

**Capítulo:** 26

Modelo de Merton (firma como call) + barreira contínua de Black-Cox.

---

### 18 – Mudança de Regime Macro-Financeira

**Capítulo:** 29

Sistema de EDPs acopladas com gerador infinitesimal de Markov. Arquitetura multi-head.

---

### 19 – xVA Não-Linear (CVA/DVA/FVA)

**Capítulo:** 39

EDP com termo de fonte não-linear de funding e default bilateral.

---

### 20 – Abatimento Ótimo de Carbono (HJB)

**Capítulo:** 43

Controle estocástico do preço do carbono com custo de abatimento marginal.

---

## 6. Como Executar

### Modelo clássico (exemplo Black-Scholes)

```bash
cd 01_BlackScholes
python train.py
```

### Modelo avançado (Fourier + Residual + Adaptive Weights + RAR)

```bash
cd 01_BlackScholes
python train_advanced.py
```

### Demonstração das arquiteturas avançadas

```bash
cd utils
python demo_advanced.py
```

Cada `train.py` salva:

- checkpoint do modelo (`.pth`)
- figura de convergência e validação (`.png`)

---

## 7. Convenções e Notação

| Símbolo | Significado em português |
|---------|--------------------------|
| $S$ / $S_t$ | Preço do ativo subjacente |
| $v$ / $v_t$ | Variância instantânea |
| $r$ | Taxa de juros livre de risco |
| $\sigma$ | Volatilidade |
| $K$ | Preço de exercício (strike) |
| $T$ | Tempo até o vencimento |
| $V(S,t)$ | Valor do derivativo |
| $J(x,t)$ | Função valor (controle ótimo) |
| $\pi$ | Fração alocada no ativo de risco |
| $\kappa,\theta,\xi$ | Parâmetros de reversão à média / vol-of-vol |
| $\rho$ | Correlação entre Brownianos |
| $\lambda$ | Intensidade de saltos ou parâmetro de penalização |
| $S_f(t)$ | Fronteira livre de exercício ótimo |

Todas as variáveis e parâmetros seguem a notação da obra de **Luiz Tiago Wilcke**.

---

## 8. Citação

Se utilizar este repositório em trabalhos acadêmicos ou profissionais, cite a obra original:

> Wilcke, Luiz Tiago. *Redes Neurais Informadas pela Física – Volume II: Edição Especial – Precificação de Derivativos, Volatilidade Estocástica e Equações HJB de Larga Escala*.

Os códigos deste repositório foram escritos de forma original, inspirados nas formulações matemáticas, estratégias de amostragem, arquiteturas e pipelines de treinamento descritos na referida obra.

---

**Licença de uso:** educacional e de pesquisa.  
**Autor da formulação científica de referência:** Luiz Tiago Wilcke.
