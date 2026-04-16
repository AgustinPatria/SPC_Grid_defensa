"""CAPA 1 — Optimizador (equivalente a ps.gms).

Implementa §1 del PDF:
- data_loader: §1.2–1.3  momentos por régimen y mezcla por periodo
- model:       §1.4       QP media-varianza con costos (Gurobi)
- simulation:  §1.5       capital ex-post (opt, BH, RB)

No depende de las capas de predicción ni calibración. Puede ejecutarse
de forma aislada con `python main.py`.
"""
from optimizer.data_loader import load_market_data
from optimizer.model       import solve_portfolio
from optimizer.simulation  import (
    simulate_capital_opt,
    simulate_naive_bh,
    simulate_naive_rb,
)

__all__ = [
    "load_market_data",
    "solve_portfolio",
    "simulate_capital_opt",
    "simulate_naive_bh",
    "simulate_naive_rb",
]
