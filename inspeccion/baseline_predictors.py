"""Hay senal extractable en esta data?  Comparacion LSTM vs baselines simples.

Si modelos lineales/triviales obtienen metricas similares al LSTM, esta
demostrado que la data NO tiene senal extractable (no es culpa del LSTM
ni del pipeline). Si algun modelo simple bate al LSTM, hay senal pero el
LSTM no la captura bien.

Modelos comparados (misma TRAIN/VALID/TEST que el LSTM):
  1. Naive-zero        : predice r_{t+1} = 0
  2. Naive-mean        : predice r_{t+1} = mean(train)
  3. AR(1)             : r_{t+1} = a + b r_t
  4. AR(5)             : regresion lineal sobre ultimos 5 retornos
  5. Linear-H=60       : regresion lineal sobre toda la ventana (mismo input que LSTM)
  6. Logistic-sign     : regresion logistica sobre H para predecir sign(r_{t+1})
  7. Quantile-linear   : regresion cuantilica lineal (analogo lineal al LSTM)
  8. LSTM (existente)  : el modelo cuantilico entrenado

Tests estadisticos:
  - Autocorrelacion de retornos a lags 1..10
  - Ljung-Box test de "no autocorrelacion"

Metricas en TEST:
  - MSE (error cuadratico medio)
  - MAE (error absoluto medio)
  - Sign accuracy (porcentaje de signos correctos)
  - Pinball loss (promedio sobre 5 cuantiles, comparable con el LSTM)

Outputs (inspeccion/baseline_predictors_out/):
  1_autocorr_test.csv/png
  2_metrics_table.csv
  3_predictions_test.csv/png
  4_pinball_compare.csv/png
  5_resumen.csv
"""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from inspeccion._common import save_csv, save_fig

from config import ASSETS, CHECKPOINT_PATH, DECILES, SPLIT
from dl.prediccion_deciles import (
    build_windows,
    chrono_split,
    fit_standardizer,
    load_checkpoint,
    load_returns,
    predict_deciles_batch,
)


SUBDIR = "baseline_predictors"


# ================================================================
# Funciones auxiliares
# ================================================================

def pinball_loss_np(y_pred, y_true, quantiles):
    """y_pred: (N, Q), y_true: (N,), quantiles: list of Q quantiles.
    Devuelve loss promedio."""
    q = np.array(quantiles).reshape(1, -1)
    e = y_true.reshape(-1, 1) - y_pred                       # (N, Q)
    return np.maximum(q * e, (q - 1.0) * e).mean()


def sign_accuracy(y_pred, y_true):
    return float((np.sign(y_pred) == np.sign(y_true)).mean())


# ================================================================
# Test 1: Autocorrelacion de retornos
# ================================================================

def diag_autocorr(splits, asset_names):
    """Autocorrelacion en TRAIN/VALID/TEST por activo."""
    rows = []
    fig, axes = plt.subplots(len(asset_names), 1, figsize=(11, 4 * len(asset_names)),
                             sharex=True)
    if len(asset_names) == 1:
        axes = [axes]
    lags = list(range(1, 11))
    for ai, a in enumerate(asset_names):
        ax = axes[ai]
        for split_name, Y in [("TRAIN", splits.Y_train),
                              ("VALID", splits.Y_valid),
                              ("TEST",  splits.Y_test)]:
            y = Y[:, ai]
            acfs = []
            for lag in lags:
                if len(y) <= lag + 1:
                    acfs.append(np.nan)
                    continue
                y0 = y[:-lag]
                y1 = y[lag:]
                if y0.std() > 0 and y1.std() > 0:
                    acf = np.corrcoef(y0, y1)[0, 1]
                else:
                    acf = 0.0
                acfs.append(acf)
                rows.append({"asset": a, "split": split_name,
                             "lag": lag, "acf": float(acf)})
            ax.plot(lags, acfs, "o-", label=split_name, alpha=0.85)
        # banda de confianza ~ 2/sqrt(N) (TRAIN)
        n_train = len(splits.Y_train)
        band = 2.0 / np.sqrt(n_train)
        ax.axhline(+band, color="grey", ls="--", lw=0.6,
                   label=f"±2/√N TRAIN = ±{band:.3f}")
        ax.axhline(-band, color="grey", ls="--", lw=0.6)
        ax.axhline(0, color="black", lw=0.4)
        ax.set_title(f"{a}: autocorrelacion de retornos por lag y split")
        ax.set_xlabel("lag (semanas)")
        ax.set_ylabel("ACF")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=9)
    save_fig(fig, "1_autocorr_test", SUBDIR)
    save_csv(pd.DataFrame(rows), "1_autocorr_test", SUBDIR)


# ================================================================
# Predictores baseline (numpy / torch)
# ================================================================

def _design_matrix(X_windows, n_past):
    """X_windows: (N, H, A). Toma los ultimos n_past returns de cada ventana
    como features. Para AR(p), n_past = p. Para Linear-H, n_past = H."""
    return X_windows[:, -n_past:, :]  # (N, n_past, A)


def fit_linear_per_asset(X_tr, Y_tr, n_past):
    """Ajusta una regresion lineal independiente por activo.
    X_tr: (N, H, A), Y_tr: (N, A). Devuelve coefs (A, n_past+1) [con intercept]."""
    A = Y_tr.shape[1]
    feats = _design_matrix(X_tr, n_past).reshape(-1, n_past * A) if False else \
            _design_matrix(X_tr, n_past)
    coefs = []
    for ai in range(A):
        f = feats[:, :, ai]                    # (N, n_past)
        f1 = np.concatenate([np.ones((f.shape[0], 1)), f], axis=1)  # add intercept
        y = Y_tr[:, ai]
        b, *_ = np.linalg.lstsq(f1, y, rcond=None)
        coefs.append(b)
    return np.stack(coefs, axis=0)             # (A, n_past+1)


def predict_linear_per_asset(coefs, X, n_past):
    feats = _design_matrix(X, n_past)
    A = coefs.shape[0]
    preds = np.empty((X.shape[0], A))
    for ai in range(A):
        f = feats[:, :, ai]
        f1 = np.concatenate([np.ones((f.shape[0], 1)), f], axis=1)
        preds[:, ai] = f1 @ coefs[ai]
    return preds


class LogisticSignPerAsset(nn.Module):
    """Una logistica por activo, input = ventana flat."""
    def __init__(self, H, n_assets):
        super().__init__()
        self.heads = nn.ModuleList([nn.Linear(H, 1) for _ in range(n_assets)])
    def forward(self, x):
        # x: (B, H, A)
        outs = [self.heads[ai](x[:, :, ai]) for ai in range(x.shape[2])]
        return torch.cat(outs, dim=1)          # (B, A) logits


def fit_logistic_sign(X_tr, Y_tr, H, n_assets, epochs=500, lr=1e-2, device="cpu"):
    model = LogisticSignPerAsset(H, n_assets).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    X_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
    Y_t = torch.tensor((Y_tr >= 0).astype(np.float32), dtype=torch.float32, device=device)
    bce = nn.BCEWithLogitsLoss()
    for ep in range(epochs):
        opt.zero_grad()
        logits = model(X_t)
        loss = bce(logits, Y_t)
        loss.backward(); opt.step()
    return model


def predict_logistic_sign(model, X, device="cpu"):
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = model(X_t).numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))      # (N, A)
    return probs                                # probabilidad de r >= 0


class QuantileLinearPerAsset(nn.Module):
    """Para cada activo, una cabeza lineal que predice Q cuantiles."""
    def __init__(self, H, n_assets, n_quantiles):
        super().__init__()
        self.A = n_assets
        self.Q = n_quantiles
        self.heads = nn.ModuleList([
            nn.Linear(H, n_quantiles) for _ in range(n_assets)
        ])
    def forward(self, x):
        # x: (B, H, A) -> (B, A, Q)
        outs = [self.heads[ai](x[:, :, ai]).unsqueeze(1) for ai in range(self.A)]
        return torch.cat(outs, dim=1)


def pinball_torch(y_pred, y_true, quantiles):
    # y_pred: (B, A, Q), y_true: (B, A)
    q = torch.tensor(quantiles, dtype=y_pred.dtype, device=y_pred.device)
    q = q.view(1, 1, -1)
    e = y_true.unsqueeze(-1) - y_pred
    return torch.maximum(q * e, (q - 1.0) * e).mean()


def fit_quantile_linear(X_tr, Y_tr, H, n_assets, quantiles,
                        epochs=1000, lr=5e-3, device="cpu"):
    model = QuantileLinearPerAsset(H, n_assets, len(quantiles)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    X_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
    Y_t = torch.tensor(Y_tr, dtype=torch.float32, device=device)
    for ep in range(epochs):
        opt.zero_grad()
        preds = model(X_t)
        loss = pinball_torch(preds, Y_t, quantiles)
        loss.backward(); opt.step()
    return model


def predict_quantile_linear(model, X, device="cpu"):
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    with torch.no_grad():
        preds = model(X_t).numpy()             # (N, A, Q)
    # Sort para garantizar monotonicidad
    preds = np.sort(preds, axis=-1)
    return preds


# ================================================================
# Loop principal: ajustar, predecir, computar metricas
# ================================================================

def run_all_models(splits, H, assets):
    quantiles = list(DECILES)
    Q = len(quantiles)
    A = len(assets)

    results = {}                                # results[method][split][metric][asset]
    predictions = {}                            # predictions[method][split] = (N, A)

    # 1. Naive-zero
    pred_test_zero = np.zeros_like(splits.Y_test)
    pred_valid_zero = np.zeros_like(splits.Y_valid)
    predictions["Naive-zero"] = {"VALID": pred_valid_zero, "TEST": pred_test_zero}

    # 2. Naive-mean (mean of train per asset)
    mean_train = splits.Y_train.mean(axis=0)
    predictions["Naive-mean"] = {
        "VALID": np.tile(mean_train, (len(splits.Y_valid), 1)),
        "TEST":  np.tile(mean_train, (len(splits.Y_test),  1)),
    }

    # 3. AR(1)
    coefs_ar1 = fit_linear_per_asset(splits.X_train, splits.Y_train, n_past=1)
    predictions["AR(1)"] = {
        "VALID": predict_linear_per_asset(coefs_ar1, splits.X_valid, n_past=1),
        "TEST":  predict_linear_per_asset(coefs_ar1, splits.X_test,  n_past=1),
    }

    # 4. AR(5)
    coefs_ar5 = fit_linear_per_asset(splits.X_train, splits.Y_train, n_past=5)
    predictions["AR(5)"] = {
        "VALID": predict_linear_per_asset(coefs_ar5, splits.X_valid, n_past=5),
        "TEST":  predict_linear_per_asset(coefs_ar5, splits.X_test,  n_past=5),
    }

    # 5. Linear-H
    coefs_lH = fit_linear_per_asset(splits.X_train, splits.Y_train, n_past=H)
    predictions["Linear-H"] = {
        "VALID": predict_linear_per_asset(coefs_lH, splits.X_valid, n_past=H),
        "TEST":  predict_linear_per_asset(coefs_lH, splits.X_test,  n_past=H),
    }

    # 6. Logistic-sign (devuelve probabilidad de r>=0, no nivel)
    log_model = fit_logistic_sign(splits.X_train, splits.Y_train, H, A,
                                  epochs=500, lr=1e-2)
    predictions["Logistic-sign"] = {
        "VALID": predict_logistic_sign(log_model, splits.X_valid),
        "TEST":  predict_logistic_sign(log_model, splits.X_test),
    }

    # 7. Quantile-linear (devuelve quintiles)
    ql_model = fit_quantile_linear(splits.X_train, splits.Y_train, H, A,
                                   quantiles, epochs=1000, lr=5e-3)
    qpreds_valid = predict_quantile_linear(ql_model, splits.X_valid)
    qpreds_test  = predict_quantile_linear(ql_model, splits.X_test)
    # Para metricas de nivel: usar quintil mediano (q=0.5)
    predictions["Quantile-linear"] = {
        "VALID": qpreds_valid[:, :, Q // 2],
        "TEST":  qpreds_test [:, :, Q // 2],
    }
    quantile_preds = {"Quantile-linear":
                      {"VALID": qpreds_valid, "TEST": qpreds_test}}

    # 8. LSTM (cargado)
    model_lstm = load_checkpoint(CHECKPOINT_PATH)
    qpreds_valid_lstm = predict_deciles_batch(model_lstm, splits.X_valid)
    qpreds_test_lstm  = predict_deciles_batch(model_lstm, splits.X_test)
    predictions["LSTM"] = {
        "VALID": qpreds_valid_lstm[:, :, Q // 2],
        "TEST":  qpreds_test_lstm [:, :, Q // 2],
    }
    quantile_preds["LSTM"] = {"VALID": qpreds_valid_lstm, "TEST": qpreds_test_lstm}

    return predictions, quantile_preds, quantiles


def compute_metrics(predictions, quantile_preds, quantiles, splits, assets):
    rows = []
    for method, splits_preds in predictions.items():
        for split_name, Y_true in [("VALID", splits.Y_valid),
                                   ("TEST",  splits.Y_test)]:
            pred = splits_preds[split_name]
            for ai, a in enumerate(assets):
                mse = float(np.mean((pred[:, ai] - Y_true[:, ai]) ** 2))
                mae = float(np.mean(np.abs(pred[:, ai] - Y_true[:, ai])))
                # Para sign accuracy:
                if method == "Logistic-sign":
                    # pred = probabilidad de r>=0; signo predicho = (prob >= 0.5)
                    sign_pred = (pred[:, ai] >= 0.5).astype(float) * 2 - 1
                    sign_true = np.sign(Y_true[:, ai])
                    sign_true = np.where(sign_true == 0, 1, sign_true)
                    acc = float((sign_pred == sign_true).mean())
                    pinball = float("nan")
                else:
                    acc = sign_accuracy(pred[:, ai], Y_true[:, ai])
                    pinball = float("nan")
                # Pinball loss para los que tienen cuantiles
                if method in quantile_preds:
                    qp = quantile_preds[method][split_name][:, ai, :]  # (N, Q)
                    pinball = pinball_loss_np(qp, Y_true[:, ai], quantiles)
                rows.append({
                    "method": method, "split": split_name, "asset": a,
                    "MSE": mse, "MAE": mae,
                    "sign_acc": acc, "pinball": pinball,
                })
    return pd.DataFrame(rows)


# ================================================================
# Plots
# ================================================================

def plot_predictions(predictions, splits, assets, save_name):
    """Scatter de predicciones vs realidad en TEST por activo."""
    methods_to_plot = ["AR(1)", "Linear-H", "Quantile-linear", "LSTM"]
    fig, axes = plt.subplots(len(methods_to_plot), len(assets),
                             figsize=(5 * len(assets), 4 * len(methods_to_plot)),
                             sharex="col", sharey="col")
    if len(assets) == 1:
        axes = axes[:, None]
    if len(methods_to_plot) == 1:
        axes = axes[None, :]
    for mi, m in enumerate(methods_to_plot):
        if m not in predictions:
            continue
        for ai, a in enumerate(assets):
            ax = axes[mi, ai]
            y_pred = predictions[m]["TEST"][:, ai]
            y_true = splits.Y_test[:, ai]
            lim = max(np.abs(y_true).max(), np.abs(y_pred).max()) * 1.1
            ax.scatter(y_true * 100, y_pred * 100, alpha=0.6, s=15)
            ax.plot([-lim*100, lim*100], [-lim*100, lim*100],
                    color="grey", lw=0.6, ls="--", label="y=x ideal")
            ax.axhline(0, color="black", lw=0.4)
            ax.axvline(0, color="black", lw=0.4)
            acc = sign_accuracy(y_pred, y_true)
            ax.set_title(f"{m} | {a} TEST  (sign acc={acc:.2%})", fontsize=10)
            ax.set_xlabel("real [%]")
            ax.set_ylabel("predicho [%]")
            ax.grid(True, alpha=0.25)
    save_fig(fig, save_name, SUBDIR)


def plot_pinball_compare(quantile_preds, splits, assets, quantiles):
    """Pinball loss por metodo, split y activo."""
    methods = list(quantile_preds.keys())
    rows = []
    for m in methods:
        for split_name, Y_true in [("VALID", splits.Y_valid),
                                   ("TEST",  splits.Y_test)]:
            qp = quantile_preds[m][split_name]              # (N, A, Q)
            for ai, a in enumerate(assets):
                pb = pinball_loss_np(qp[:, ai, :], Y_true[:, ai], quantiles)
                rows.append({"method": m, "split": split_name,
                             "asset": a, "pinball": float(pb)})
    df = pd.DataFrame(rows)
    save_csv(df, "4_pinball_compare", SUBDIR)

    fig, axes = plt.subplots(1, len(assets), figsize=(6 * len(assets), 4.5),
                             sharey=True)
    if len(assets) == 1:
        axes = [axes]
    for ai, a in enumerate(assets):
        ax = axes[ai]
        for split_name, color in [("VALID", "#999"), ("TEST", "#1f77b4")]:
            d = df[(df["asset"] == a) & (df["split"] == split_name)]
            ax.bar([m + " " + split_name for m in d["method"]],
                   d["pinball"], color=color, alpha=0.85, label=split_name)
        ax.set_title(f"Pinball loss por método — {a}")
        ax.set_ylabel("pinball loss (menor es mejor)")
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.25, axis="y")
        ax.legend(fontsize=9)
    save_fig(fig, "4_pinball_compare", SUBDIR)


# ================================================================
# Main
# ================================================================

def main():
    print("=" * 70)
    print("BASELINE PREDICTORS: hay senal extractable en esta data?")
    print("=" * 70)
    print("\nCargando data y construyendo splits...")
    model_dummy = load_checkpoint(CHECKPOINT_PATH)
    H = model_dummy.config.H
    df = load_returns()
    X, Y, t_idx = build_windows(df, H)
    splits = chrono_split(X, Y, t_idx, SPLIT)
    print(f"  H = {H}")
    print(f"  TRAIN: {len(splits.X_train)} ventanas (t={splits.t_train[0]}..{splits.t_train[-1]})")
    print(f"  VALID: {len(splits.X_valid)} ventanas (t={splits.t_valid[0]}..{splits.t_valid[-1]})")
    print(f"  TEST : {len(splits.X_test)}  ventanas (t={splits.t_test[0]}..{splits.t_test[-1]})")

    assets = list(ASSETS)

    print("\n--- Test 1: autocorrelacion de retornos ---")
    diag_autocorr(splits, assets)

    print("\n--- Test 2: ajustando 7 modelos baseline + LSTM ---")
    predictions, quantile_preds, quantiles = run_all_models(splits, H, assets)

    print("\n--- Computando metricas ---")
    metrics = compute_metrics(predictions, quantile_preds, quantiles, splits, assets)
    save_csv(metrics, "2_metrics_table", SUBDIR)

    # Pivot para vision tabular
    print("\n=== METRICAS EN TEST (menor MSE/MAE/pinball, mayor sign_acc = mejor) ===")
    for a in assets:
        print(f"\n[{a}]")
        sub = metrics[(metrics["split"] == "TEST") & (metrics["asset"] == a)]
        for _, row in sub.iterrows():
            pb_str = f"{row['pinball']:.5f}" if not np.isnan(row['pinball']) else "  --   "
            print(f"  {row['method']:<18}  MSE={row['MSE']*10000:>7.2f}e-4  "
                  f"MAE={row['MAE']*100:>5.3f}%  sign_acc={row['sign_acc']*100:>5.1f}%  "
                  f"pinball={pb_str}")

    print("\n--- Plots de predicciones y pinball ---")
    plot_predictions(predictions, splits, assets, "3_predictions_test")
    plot_pinball_compare(quantile_preds, splits, assets, quantiles)

    # Resumen
    print("\n--- Resumen ---")
    summary_rows = []
    for a in assets:
        sub = metrics[(metrics["split"] == "TEST") & (metrics["asset"] == a)]
        # Naive-zero como baseline
        baseline_mse = sub[sub["method"] == "Naive-zero"]["MSE"].iloc[0]
        baseline_acc = 0.50
        for _, row in sub.iterrows():
            r2_vs_zero = 1.0 - row["MSE"] / baseline_mse if baseline_mse > 0 else 0.0
            acc_vs_chance = row["sign_acc"] - baseline_acc
            summary_rows.append({
                "asset": a, "method": row["method"],
                "R2_vs_naive_zero": float(r2_vs_zero),
                "sign_acc_pct": float(row["sign_acc"] * 100),
                "sign_acc_vs_chance_pp": float(acc_vs_chance * 100),
                "MSE_e-4": float(row["MSE"] * 10000),
                "pinball": float(row["pinball"]) if not np.isnan(row["pinball"]) else np.nan,
            })
    save_csv(pd.DataFrame(summary_rows), "5_resumen", SUBDIR)

    print("\n" + "=" * 70)
    print("Done. Outputs en inspeccion/baseline_predictors_out/")
    print("=" * 70)


if __name__ == "__main__":
    main()
