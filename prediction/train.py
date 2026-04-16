"""Paso 2b del Pipeline SPC-Grid: entrenamiento con pinball loss (§2.3, ec. 14).

L(psi) = (1/|train|) sum_t sum_i sum_q rho_q( r_{i,t+1} - r_hat^{(q)}_{i,t+1} )
rho_q(e) = max(q*e, (q-1)*e)

Early stopping sobre el error de validación.
Al finalizar, el modelo se CONGELA.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


def pinball_loss(y_true, y_pred, quantiles):
    """Pinball loss (regresión cuantil).

    Args:
        y_true:    (batch, n_assets)
        y_pred:    (batch, n_assets, n_quantiles)
        quantiles: tuple de niveles q
    """
    q = torch.tensor(quantiles, device=y_pred.device, dtype=y_pred.dtype)
    errors = y_true.unsqueeze(-1) - y_pred
    loss = torch.maximum(q * errors, (q - 1) * errors)
    return loss.mean()


def _make_loader(X, y, batch_size, shuffle):
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_model(model, splits, *,
                lr: float = 1e-3,
                batch_size: int = 32,
                max_epochs: int = 500,
                patience: int = 30):
    """Entrena con early stopping y devuelve el modelo congelado.

    Args:
        model: QuantileLSTM
        splits: dict con "train" y "valid", cada uno (X, y, t_indices)

    Returns:
        model congelado, dict con historial de loss.
    """
    X_tr, y_tr, _ = splits["train"]
    X_va, y_va, _ = splits["valid"]

    train_loader = _make_loader(X_tr, y_tr, batch_size, shuffle=True)
    valid_loader = _make_loader(X_va, y_va, batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    quantiles = model.quantiles

    best_val_loss = float("inf")
    best_state = None
    wait = 0
    history = {"train": [], "valid": []}

    for epoch in range(max_epochs):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            pred = model(xb)
            loss = pinball_loss(yb, pred, quantiles)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in valid_loader:
                pred = model(xb)
                val_losses.append(pinball_loss(yb, pred, quantiles).item())

        train_avg = np.mean(train_losses)
        val_avg = np.mean(val_losses)
        history["train"].append(train_avg)
        history["valid"].append(val_avg)

        if val_avg < best_val_loss:
            best_val_loss = val_avg
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stopping en epoch {epoch+1} (best val_loss={best_val_loss:.6f})")
                break

    model.load_state_dict(best_state)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    return model, history
