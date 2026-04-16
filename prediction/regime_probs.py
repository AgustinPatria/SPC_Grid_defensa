"""Paso 2c del Pipeline SPC-Grid: cuantiles → probabilidades de régimen (§2.4, ec. 15).

p_hat_{i, bull, t+1} = (1/|Q|) * sum_{q in Q} 1{ r_hat^{(q)}_{i,t+1} >= 0 }
p_hat_{i, bear, t+1} = 1 - p_hat_{i, bull, t+1}

Genera DataFrames con formato compatible con optimizer.data_loader.
"""
import numpy as np
import pandas as pd
import torch


def quantiles_to_regime_probs(model, X, t_indices, assets=("SPX", "CMC200")):
    """Convierte cuantiles predichos en probabilidades bear/bull por activo.

    Args:
        model: QuantileLSTM congelado.
        X: np.ndarray (n_samples, H, n_assets)
        t_indices: lista de t correspondientes.
        assets: nombres de activos en orden de columnas.

    Returns:
        dict[asset] -> DataFrame con columnas ["bear", "bull"], index = t.
    """
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(X, dtype=torch.float32)
        preds = model(x_tensor).numpy()

    n_q = preds.shape[2]
    probs = {}

    for ai, asset in enumerate(assets):
        p_bull = (preds[:, ai, :] >= 0).sum(axis=1) / n_q
        p_bear = 1.0 - p_bull
        probs[asset] = pd.DataFrame({
            "bear": p_bear,
            "bull": p_bull,
        }, index=t_indices)

    return probs
