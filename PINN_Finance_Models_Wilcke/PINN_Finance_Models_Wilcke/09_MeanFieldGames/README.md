# Jogos de Campo Médio (MFG)

**Autor da formulação original:** Luiz Tiago Wilcke  
**Capítulo de referência:** Capítulos 11 e 23

## Equação Principal

Sistema acoplado HJB + Fokker-Planck de Lasry-Lions.

## Descrição

Equilíbrio de Nash em população infinita de agentes.

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
