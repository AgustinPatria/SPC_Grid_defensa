"""Paso 3 del Pipeline SPC-Grid: generación y reducción de escenarios (§2.5).

Paso 1 — generar N escenarios candidatos:
  Desde la última ventana observada, para t = 1..T_future:
    - obtener cuantiles r_hat^{(q)}_{i,t}
    - elegir q uniforme en Q
    - fijar r^cand_{i,t} = r_hat^{(q)}_{i,t}
    - actualizar la ventana (rolling)

Paso 2 — reducir a 5 escenarios por quintiles:
  - ordenar los N candidatos por retorno acumulado del SPX
  - dividir en 5 grupos
  - elegir escenario mediano de cada grupo → |S| = 5
"""
import numpy as np
import torch


def generate_candidate_scenarios(model, last_window, N: int = 1000,
                                 T_future: int = 163, rng_seed: int = 42):
    """Genera N escenarios de retornos futuros por muestreo de cuantiles.

    Args:
        model: QuantileLSTM congelado.
        last_window: np.ndarray (H, n_assets), última ventana observada.
        N: número de escenarios candidatos.
        T_future: horizonte de predicción.
        rng_seed: semilla para reproducibilidad.

    Returns:
        np.ndarray (N, T_future, n_assets) con retornos simulados.
    """
    rng = np.random.default_rng(rng_seed)
    H = last_window.shape[0]
    n_assets = last_window.shape[1]
    n_q = len(model.quantiles)

    scenarios = np.zeros((N, T_future, n_assets))

    for s in range(N):
        window = last_window.copy()
        for t in range(T_future):
            x = torch.tensor(window[np.newaxis], dtype=torch.float32)
            with torch.no_grad():
                q_preds = model(x).numpy()[0]

            q_idx = rng.integers(0, n_q, size=n_assets)
            r_step = np.array([q_preds[ai, q_idx[ai]] for ai in range(n_assets)])
            scenarios[s, t] = r_step

            window = np.roll(window, -1, axis=0)
            window[-1] = r_step

    return scenarios


def reduce_to_quintiles(scenarios, summary_asset_idx: int = 0):
    """Reduce N escenarios a 5 representativos por quintiles.

    Args:
        scenarios: (N, T_future, n_assets)
        summary_asset_idx: índice del activo para calcular retorno acumulado (0=SPX).

    Returns:
        np.ndarray (5, T_future, n_assets), los 5 escenarios representativos.
    """
    N = scenarios.shape[0]
    cum_returns = np.prod(1 + scenarios[:, :, summary_asset_idx], axis=1) - 1

    order = np.argsort(cum_returns)
    quintile_size = N // 5
    representatives = []

    for q in range(5):
        start = q * quintile_size
        end = start + quintile_size if q < 4 else N
        group_indices = order[start:end]
        median_idx = group_indices[len(group_indices) // 2]
        representatives.append(scenarios[median_idx])

    return np.array(representatives)
