"""Modulo de inspeccion / diagnostico del pipeline SPC_Grid3.

Cada submodulo corresponde a un eslabon de la cadena
(features -> LSTM -> p_bull/escenarios -> mu/sigma_mix -> optimizador -> regret)
y produce sus salidas (PNG + CSV) en `inspeccion/<modulo>_out/`.
"""
