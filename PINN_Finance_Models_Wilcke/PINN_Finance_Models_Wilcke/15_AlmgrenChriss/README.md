# Liquidação Ótima – Almgren-Chriss

**Autor da formulação original:** Luiz Tiago Wilcke  
**Capítulo de referência:** Capítulo 19

## Equação Principal

HJB de execução ótima de grandes lotes com impacto permanente e temporário.

## Descrição

Controle de velocidade de liquidação.

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
