"""CAPA 2 — Predicción con Deep Learning (§2 del PDF).

Objetivo: a partir de precios históricos, entrenar un modelo DL que
prediga cuantiles del retorno futuro y usarlo para:
  (a) estimar probabilidades de régimen p_{i,k,t}  (§2.4)
  (b) generar N escenarios y reducirlos a 5 (§2.5)

Módulos:
- dataset:      §2.2  construcción de ventanas H, split cronológico
- model_dl:     §2.3  red cuantil (LSTM / Transformer)
- train:        §2.3  entrenamiento con pinball loss y early stopping
- regime_probs: §2.4  quintiles -> p_{i,bull,t}, p_{i,bear,t}
- scenarios:    §2.5  generación de N escenarios, reducción por quintiles
"""
