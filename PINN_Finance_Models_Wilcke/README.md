# PINNs para Finanças Quantitativas

Modelos de Physics-Informed Neural Networks para precificação de derivativos, volatilidade estocástica, controle ótimo e risco.  
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
├── utils/
│   ├── amostragem.py           # Latin Hypercube + reamostragem
│   ├── otimizadores.py         # Adam + L-BFGS híbrido
│   ├── gregas.py               # Δ, Γ, Vega, Vanna via autograd
│   ├── redes_avancadas.py      # Fourier, Residual, DGM...
│   ├── pinn_complexa.py        # Rede PINN complexa de produção
│   └── treinar_pinn_complexa.py
├── 01_BlackScholes/
│   ├── configuracao.py
│   ├── modelo.py
│   ├── treinar.py
│   ├── modelo_avancado.py
│   └── treinar_avancado.py
├── 02_AmericanOptions/ ... 20_CarbonHJB/
│   ├── configuracao.py
│   ├── modelo.py
│   └── treinar.py
```

---

## Rede PINN complexa

```bash
cd utils
python treinar_pinn_complexa.py
```

Arquiteturas: `residual_fourier`, `dgm`, `highway_fourier`, `attention_fourier`, `full`.

---

## Modelos

| Pasta | Modelo | Equação principal |
|-------|--------|-------------------|
| 01_BlackScholes | Call europeia | $\partial_t V + \frac12\sigma^2 S^2 V_{SS} + rS V_S - rV = 0$ |
| 02_AmericanOptions | Americana | $\min(-\partial_t V-\mathcal{L}V,\ V-(S-K)^+)$ |
| 03_Heston | Vol. estocástica | EDP 2D em $(S,v,t)$ |
| 04_MertonHJB | Portfólio ótimo | HJB com $\pi^*$, $c^*$ |
| 05_CIR | Taxa de juros | $\partial_t P + \kappa(\theta-r)P_r + \frac12\sigma^2 r P_{rr} - rP = 0$ |
| 06_AvellanedaStoikov | Market making | HJB + Fokker-Planck |
| 07_Dupire | Vol. local (inverso) | $\partial_T C = \frac12\sigma_{loc}^2 K^2 \partial_{KK}C$ |
| 08_DeepONet | Operador neural | Branch-Trunk |
| 09_MeanFieldGames | Jogos de campo médio | HJB + FP acoplados |
| 10_SABR | SABR | $dF=\alpha F^\beta dW$ |
| 11_MultiAsset | Cesta multiativo | EDP dimensão $d$ |
| 12_DeepHedging | Cobertura ótima | Política $\delta_t=\pi_\theta$ |
| 13_JumpDiffusion | Saltos Merton | PIDE com integral |
| 14_HullWhite | Hull-White | $dr=(\theta(t)-ar)dt+\sigma dW$ |
| 15_AlmgrenChriss | Liquidação ótima | HJB de execução |
| 16_RoughVolatility | Vol. rough | Operador de Caputo |
| 17_CreditRisk | Risco de crédito | Merton + Black-Cox |
| 18_RegimeSwitching | Mudança de regime | EDPs acopladas |
| 19_xVA | CVA/DVA/FVA | Fonte não-linear |
| 20_CarbonHJB | Abatimento carbono | HJB de controle |

---

## Como executar

```bash
cd 01_BlackScholes
python treinar.py
python treinar_avancado.py
```

---

## Variáveis (português)

| Símbolo | Nome em português |
|---------|-------------------|
| $S$ | preço do ativo |
| $v$ | variância |
| $r$ | taxa de juros livre de risco |
| $\sigma$ | volatilidade |
| $K$ | preço de exercício (strike) |
| $T$ | vencimento |
| $V$ | preço do derivativo |
| $J$ | função valor |
| $\pi$ | fração no ativo de risco |
| $c$ | consumo / abatimento |
| $\kappa$ | velocidade de reversão |
| $\theta$ | nível de longo prazo |
| $\xi$ | vol-da-vol |
| $\rho$ | correlação |
| $S_f(t)$ | fronteira livre |
| $\lambda$ | intensidade de saltos / penalização |
| perda_pde | residual da equação |
| perda_contorno | erro nas bordas |
| perda_terminal | erro no payoff |
| dim_oculta | largura da rede |
| n_blocos | profundidade residual |

---

## Arquivos principais

| Arquivo | Função |
|---------|--------|
| `configuracao.py` | hiperparâmetros e dados de mercado |
| `modelo.py` | rede neural + função de perda física |
| `treinar.py` | pipeline Adam + L-BFGS |
| `amostragem.py` | Latin Hypercube e RAR |
| `otimizadores.py` | otimização híbrida |
| `gregas.py` | extração de sensibilidade |
| `pinn_complexa.py` | arquitetura avançada completa |

---

Inspirado em: Wilcke, Luiz Tiago. *Redes Neurais Informadas pela Física – Volume II*.
