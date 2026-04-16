"""Paso 5a del Pipeline SPC-Grid: evaluación V_{g,s} (§3.3, ec. 19-20).

Para cada g = (lambda_g, m_g):
  1. Optimizar UNA VEZ → política w^g_{i,t}
  2. Para cada escenario s:
       simular capital con los retornos r^s_{i,t} y costos base reales
       V_{g,s} = x^g_{T,s}  (capital terminal)
"""
import numpy as np
import pandas as pd

from calibration.grid    import GridPoint
from optimizer.model     import solve_portfolio


def _simulate_capital_on_scenario(w_sol, scenario, assets, c_base, T_vals, x0):
    """Simula capital de la política w^g sobre un escenario s.

    Args:
        w_sol: dict[(asset, t)] -> peso
        scenario: np.ndarray (T_future, n_assets) retornos del escenario
        assets: lista de nombres de activos
        c_base: dict[asset] -> costo base
        T_vals: lista de periodos
        x0: capital inicial

    Returns:
        capital terminal V_{g,s}
    """
    cap = x0
    for idx in range(1, len(T_vals)):
        t = T_vals[idx]
        t_prev = T_vals[idx - 1]
        r_port = sum(
            w_sol[assets[ai], t_prev] * scenario[idx - 1, ai]
            for ai in range(len(assets))
        )
        turn = sum(
            c_base[assets[ai]] * abs(
                w_sol[assets[ai], t] - w_sol[assets[ai], t_prev]
            )
            for ai in range(len(assets))
        )
        cap = cap * (1 + r_port) - cap * turn
    return cap


def evaluate_grid(grid, context, scenarios, theta=None):
    """Evalúa cada punto g sobre cada escenario s.

    Args:
        grid: lista de GridPoint
        context: dict del optimizer (necesita mu_mix, sigma_mix, etc.)
        scenarios: np.ndarray (n_scenarios, T_future, n_assets)
        theta: dict de sentimiento (default: neutral 1.0)

    Returns:
        V: pd.DataFrame con shape (|G|, |S|), valores V_{g,s}
        policies: dict[GridPoint] -> (z, w_sol, u_sol, v_sol)
    """
    assets = context["assets"]
    T_vals = context["T_vals"]
    c_base = context["c_base"]
    x0     = context["Capital_inicial"]
    n_scenarios = scenarios.shape[0]

    if theta is None:
        theta = {a: 1.0 for a in assets}

    V = np.zeros((len(grid), n_scenarios))
    policies = {}

    for gi, g in enumerate(grid):
        print(f"  [{gi+1}/{len(grid)}] {g.label()} ...", end=" ", flush=True)
        z, w_sol, u_sol, v_sol, status = solve_portfolio(
            theta, context, lambda_riesgo=g.lam, costo_mult=g.m
        )
        policies[g] = (z, w_sol, u_sol, v_sol)

        for si in range(n_scenarios):
            V[gi, si] = _simulate_capital_on_scenario(
                w_sol, scenarios[si], assets, c_base, T_vals, x0
            )
        print(f"z={z:.6f}  V_avg=${np.mean(V[gi]):,.2f}")

    g_labels = [g.label() for g in grid]
    s_labels = [f"S{si+1}" for si in range(n_scenarios)]
    V_df = pd.DataFrame(V, index=g_labels, columns=s_labels)

    return V_df, policies
