"""Probabilidades de régimen bull/bear desde los deciles predichos (PDF sec. 2.4).

Este modulo usa la regla **explicita** del PDF:
    bull ⇔ retorno >= BULL_THRESHOLD  (default 0.0)
    bear ⇔ retorno <  BULL_THRESHOLD
sobre cada decil predicho por la LSTM:
    p_bull(t+1) ≈ (1/|Q|) * Σ_{q ∈ Q} 1{ r_hat^(q)(t+1) >= BULL_THRESHOLD }
    p_bear(t+1) = 1 - p_bull(t+1)

Esto se hace por activo y por periodo, y cumple p_bull + p_bear = 1 como exige
ps.gms.

CAVEAT importante: los archivos `data/prob_*.csv` (p_hist) NO siguen esta misma
regla — fueron generados externamente con un criterio distinto (probablemente
HMM sobre volatilidad). Empiricamente accuracy(p_bull_hist > 0.5 ↔ r >= 0) es
≈ 52-56%. Por eso `mu_hat(bear)` puede ser MAYOR que `mu_hat(bull)` cuando se
calcula con `p_hist`. El pipeline DL post-unificacion (build_dl_context actual)
no toca `p_hist`, derivando todo de los candidatos del LSTM."""

from typing import Tuple

import numpy as np

from config import BULL_THRESHOLD
from .prediccion_deciles import LoadedModel, predict_deciles_batch


def regimen_from_deciles(decile_preds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    decile_preds: (..., n_deciles)  — retornos predichos por decil.
    return: (p_bull, p_bear) con la misma shape de entrada menos la última dim.
    """
    p_bull = (decile_preds >= BULL_THRESHOLD).mean(axis=-1).astype(np.float32)
    p_bear = (1.0 - p_bull).astype(np.float32)
    return p_bull, p_bear


def regimen_probabilities(
    model: LoadedModel,
    windows: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inferencia + conversión a régimen en un paso.

    windows: (N, H, n_assets) — ventanas de entrada del LSTM.
    return:  (p_bull, p_bear), ambos de shape (N, n_assets).
    """
    preds = predict_deciles_batch(model, windows)     # (N, A, Q)
    return regimen_from_deciles(preds)
