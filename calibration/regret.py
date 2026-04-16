"""Paso 5b del Pipeline SPC-Grid: regret y reglas de selección (§3.4, ec. 21-24).

V_s^best = max_{g in G} V_{g,s}
R_{g,s}  = V_s^best - V_{g,s}   >= 0

Reglas:
  Regret promedio:  g* = argmin_g  (1/|S|) sum_s R_{g,s}
  Peor caso:        g* = argmin_g  max_s R_{g,s}
"""
import pandas as pd
import numpy as np


def compute_regret(V: pd.DataFrame):
    """Calcula la matriz de regret R_{g,s}.

    Args:
        V: DataFrame (|G| x |S|) con V_{g,s}.

    Returns:
        R: DataFrame (|G| x |S|) con R_{g,s}.
        V_best: Series con V_s^best por escenario.
    """
    V_best = V.max(axis=0)
    R = V_best - V
    return R, V_best


def select_best(R: pd.DataFrame, grid, rule: str = "avg"):
    """Selecciona g* según la regla de regret.

    Args:
        R: DataFrame (|G| x |S|) con R_{g,s}.
        grid: lista de GridPoint (mismo orden que filas de R).
        rule: "avg" (ec. 23) o "worst" (ec. 24).

    Returns:
        g_star: GridPoint seleccionado.
        summary: dict con métricas del punto seleccionado.
    """
    if rule == "avg":
        scores = R.mean(axis=1)
    elif rule == "worst":
        scores = R.max(axis=1)
    else:
        raise ValueError(f"Regla desconocida: {rule}. Usar 'avg' o 'worst'.")

    best_idx = scores.argmin()
    g_star = grid[best_idx]

    return g_star, {
        "rule":         rule,
        "g_star":       g_star,
        "score":        scores.iloc[best_idx],
        "regret_avg":   R.iloc[best_idx].mean(),
        "regret_worst": R.iloc[best_idx].max(),
    }
