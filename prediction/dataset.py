"""Paso 1 del Pipeline SPC-Grid: preparación de datos y partición temporal.

En cada fecha t se construye:
  x_t = (r_{i, t-H+1}, ..., r_{i,t})_{i in I}   features: ventana de H retornos
  y_t = (r_{i, t+1})_{i in I}                   target:  retorno siguiente

Split cronológico (sin fuga de información):
  train  ->  valid  ->  test
"""
import numpy as np
import pandas as pd
from pathlib import Path


def load_returns(data_dir: str) -> pd.DataFrame:
    """Carga retornos semanales de ambos activos en un DataFrame indexado por t."""
    base = Path(data_dir)
    spx = pd.read_csv(base / "ret_semanal_spx.csv")
    cmc = pd.read_csv(base / "ret_semanal_cmc200.csv")
    for df in (spx, cmc):
        df.columns = [c.strip() for c in df.columns]
        df["t"] = df["t"].astype(int)

    returns = pd.DataFrame({
        "t":      spx["t"],
        "SPX":    spx["ret_semanal_spx"].values,
        "CMC200": cmc["ret_semanal_cmc200"].values,
    }).set_index("t")
    return returns


def build_windows(returns: pd.DataFrame, H: int = 52):
    """Construye ventanas supervisadas (X, y) para la red cuantil.

    Args:
        returns: DataFrame con columnas ["SPX", "CMC200"], index = t (1..163).
        H: largo de la ventana de historia.

    Returns:
        X: np.ndarray de shape (n_samples, H, n_assets)
        y: np.ndarray de shape (n_samples, n_assets)
        t_indices: list de t correspondiente a cada muestra (t del target)
    """
    vals = returns.values
    t_vals = returns.index.values
    n_assets = vals.shape[1]

    X, y, t_indices = [], [], []
    for idx in range(H, len(vals)):
        X.append(vals[idx - H:idx])
        y.append(vals[idx])
        t_indices.append(t_vals[idx])

    return np.array(X), np.array(y), t_indices


def split_chronological(X, y, t_indices,
                        train_frac: float = 0.60,
                        valid_frac: float = 0.20):
    """Particiona en train/valid/test en orden temporal.

    Args:
        train_frac: fracción para entrenamiento.
        valid_frac: fracción para validación.
        El resto es test.

    Returns:
        dict con keys "train", "valid", "test", cada uno con (X, y, t_indices).
    """
    n = len(X)
    n_train = int(n * train_frac)
    n_valid = int(n * valid_frac)

    splits = {
        "train": (X[:n_train],
                  y[:n_train],
                  t_indices[:n_train]),
        "valid": (X[n_train:n_train + n_valid],
                  y[n_train:n_train + n_valid],
                  t_indices[n_train:n_train + n_valid]),
        "test":  (X[n_train + n_valid:],
                  y[n_train + n_valid:],
                  t_indices[n_train + n_valid:]),
    }
    return splits
