"""Por que p_bull < 0.5? Diagnostico de calibracion del LSTM.

Contexto: el LSTM en rollout produce p_bull ≈ 0.40 (SPX) y 0.40 (CMC),
sub-50%. Eso combinado con mu_hat (con regimen alineado p_sign) hace que
mu_mix sea NEGATIVO para ambos activos, lo que fuerza al FO a una solucion
de esquina (100% en el activo menos malo) sin posibilidad de diversificar.

Necesitamos entender de donde viene p_bull < 0.5. Tres hipotesis:

  H1. Los retornos reales tienen mediana negativa.
      Si median(r) < 0 en el periodo, entonces "r >= 0" naturalmente es
      menos del 50% de las semanas. p_bull(real) < 0.5 seria correcto.

  H2. El LSTM aprende a poner el quintil q=0.5 sesgado a negativo.
      Aunque la mediana real sea cercana a 0, el modelo podria predecir
      una mediana ligeramente debajo. Si q=0.5 < 0 entonces solo 2/5
      quintiles caen >= 0 => p_bull predicho = 0.4 exacto.

  H3. La discrepancia: p_bull(predicho) vs p_bull(real) en cada split.
      Si predicho << real, hay sesgo de calibracion del modelo.

Diagnosticos en este script:

  1_returns_stats_por_split    media, mediana, %dias bull, distribucion
                               de retornos reales en TRAIN/VALID/TEST.
  2_quintiles_predichos        trayectorias predichas q=0.1..0.9 sobre
                               cada split (walking).
  3_pbull_pred_vs_real         comparacion p_bull predicho vs sample real
                               por split (binario, ec 15).
  4_threshold_sensitivity      como cambia p_bull si usamos BULL_THRESHOLD
                               = mediana de train en vez de 0.

Outputs (inspeccion/lstm_calibration_out/):
  1_returns_stats.csv/png
  2_quintiles_pred.csv/png
  3_pbull_compare.csv/png
  4_threshold_sensitivity.csv/png
  5_diagnostico.csv  (resumen para defensa)

Corre con:
    python -m inspeccion.lstm_calibration
    (o)
    python inspeccion/lstm_calibration.py
"""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from inspeccion._common import save_csv, save_fig

from config import (
    ASSETS,
    BULL_THRESHOLD,
    CHECKPOINT_PATH,
    DECILES,
    SPLIT,
)
from dl.prediccion_deciles import (
    build_windows,
    chrono_split,
    load_checkpoint,
    load_returns,
    predict_deciles_batch,
)


SUBDIR = "lstm_calibration"


def load_state():
    print("Cargando modelo y datos...")
    model = load_checkpoint(CHECKPOINT_PATH)
    H = model.config.H
    df = load_returns()
    X, Y, t_idx = build_windows(df, H)
    split = chrono_split(X, Y, t_idx, SPLIT)
    return {
        "model": model, "H": H, "df": df,
        "split": split, "splits_names": ["TRAIN", "VALID", "TEST"],
    }


# ================================================================
# 1) Estadisticas de retornos reales por split
# ================================================================

def diag_returns_stats(state):
    sp = state["split"]
    splits = [("TRAIN", sp.Y_train), ("VALID", sp.Y_valid), ("TEST", sp.Y_test)]

    rows = []
    fig, axes = plt.subplots(len(ASSETS), len(splits),
                             figsize=(13, 3.5 * len(ASSETS)), sharey="row")
    if len(ASSETS) == 1:
        axes = axes[None, :]
    for ai, a in enumerate(ASSETS):
        for si, (split_name, Y) in enumerate(splits):
            y = Y[:, ai]
            n = len(y)
            n_bull = int((y >= BULL_THRESHOLD).sum())
            true_pbull = n_bull / n
            mu = float(y.mean())
            med = float(np.median(y))
            sd = float(y.std())
            skew = float(((y - mu) ** 3).mean() / (sd ** 3)) if sd > 0 else 0.0
            rows.append({
                "asset": a, "split": split_name, "n_weeks": n,
                "mean_pct": mu * 100, "median_pct": med * 100,
                "std_pct": sd * 100, "skew": skew,
                "n_bull": n_bull, "true_p_bull": true_pbull,
            })
            ax = axes[ai, si]
            ax.hist(y * 100, bins=20, color="#1f77b4", alpha=0.7, edgecolor="white")
            ax.axvline(BULL_THRESHOLD * 100, color="#E63946", lw=1.5,
                       label=f"BULL_THRESHOLD = {BULL_THRESHOLD * 100:.1f}%")
            ax.axvline(med * 100, color="#F2B705", lw=1.5, ls="--",
                       label=f"mediana = {med*100:+.2f}%")
            ax.set_title(f"{a} {split_name}  n={n}  "
                         f"true p_bull={true_pbull:.2%}")
            ax.set_xlabel("retorno semanal [%]")
            if si == 0:
                ax.set_ylabel("frecuencia")
            ax.legend(fontsize=8, loc="upper right")
            ax.grid(True, alpha=0.25)
    save_fig(fig, "1_returns_stats", SUBDIR)
    df = pd.DataFrame(rows)
    save_csv(df, "1_returns_stats", SUBDIR)
    print("\n--- Stats de retornos reales por split ---")
    print(df.to_string(index=False,
                       float_format=lambda x: f"{x:.3f}"))


# ================================================================
# 2) Trayectorias de quintiles predichos por split (walking)
# ================================================================

def diag_quintiles_pred(state):
    model = state["model"]
    sp = state["split"]
    splits = [
        ("TRAIN", sp.X_train, sp.Y_train, sp.t_train),
        ("VALID", sp.X_valid, sp.Y_valid, sp.t_valid),
        ("TEST",  sp.X_test,  sp.Y_test,  sp.t_test),
    ]

    rows = []
    fig, axes = plt.subplots(len(ASSETS), len(splits),
                             figsize=(15, 3.5 * len(ASSETS)), sharey="row")
    if len(ASSETS) == 1:
        axes = axes[None, :]

    qlabels = [f"q={q}" for q in DECILES]
    cmap = plt.get_cmap("viridis")
    Q = len(DECILES)

    for ai, a in enumerate(ASSETS):
        for si, (split_name, X, Y, t_idx) in enumerate(splits):
            preds = predict_deciles_batch(model, X)   # (N, A, Q)
            ax = axes[ai, si]
            for qi, q in enumerate(DECILES):
                qpred = preds[:, ai, qi]
                color = cmap(qi / (Q - 1))
                ax.plot(t_idx, qpred * 100, color=color, lw=1.0,
                        label=f"q={q}", alpha=0.85)
                for tk, vk in zip(t_idx, qpred):
                    rows.append({"asset": a, "split": split_name,
                                 "t": int(tk), "quintile": float(q),
                                 "pred_pct": float(vk * 100)})
            # Mediana realizada y BULL_THRESHOLD
            ax.plot(t_idx, Y[:, ai] * 100, color="#E63946", lw=0.6,
                    alpha=0.5, label="realizado")
            ax.axhline(BULL_THRESHOLD * 100, color="black", lw=0.6, ls="--",
                       label="threshold")
            ax.set_title(f"{a} {split_name} — quintiles predichos vs realizado")
            ax.set_xlabel("t")
            if si == 0:
                ax.set_ylabel("retorno [%]")
            ax.legend(fontsize=7, loc="best")
            ax.grid(True, alpha=0.25)
    save_fig(fig, "2_quintiles_pred", SUBDIR)
    save_csv(pd.DataFrame(rows), "2_quintiles_pred", SUBDIR)


# ================================================================
# 3) p_bull predicho vs sample p_bull por split
# ================================================================

def diag_pbull_compare(state):
    model = state["model"]
    sp = state["split"]
    splits = [
        ("TRAIN", sp.X_train, sp.Y_train),
        ("VALID", sp.X_valid, sp.Y_valid),
        ("TEST",  sp.X_test,  sp.Y_test),
    ]

    rows = []
    for split_name, X, Y in splits:
        preds = predict_deciles_batch(model, X)   # (N, A, Q)
        for ai, a in enumerate(ASSETS):
            # p_bull predicho POR VENTANA (ec. 15: fraccion de quintiles >= 0)
            pred_pbull_per_window = (preds[:, ai, :] >= BULL_THRESHOLD).mean(axis=-1)
            # p_bull real binario: r >= 0 cada semana
            real_pbull_per_window = (Y[:, ai] >= BULL_THRESHOLD).astype(float)
            rows.append({
                "asset": a, "split": split_name,
                "pred_pbull_mean": float(pred_pbull_per_window.mean()),
                "pred_pbull_std":  float(pred_pbull_per_window.std()),
                "real_pbull":      float(real_pbull_per_window.mean()),
                "median_quintile_pred_mean": float(preds[:, ai, len(DECILES)//2].mean()),
                "median_quintile_pred_median": float(np.median(preds[:, ai, len(DECILES)//2])),
                "frac_median_quintile_neg":  float((preds[:, ai, len(DECILES)//2] < 0).mean()),
            })

    df = pd.DataFrame(rows)
    save_csv(df, "3_pbull_compare", SUBDIR)
    print("\n--- p_bull predicho vs real por split ---")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Plot: bars side by side
    fig, axes = plt.subplots(1, len(ASSETS), figsize=(12, 4.5), sharey=True)
    if len(ASSETS) == 1:
        axes = [axes]
    splits_n = ["TRAIN", "VALID", "TEST"]
    x = np.arange(len(splits_n))
    width = 0.35
    for ai, a in enumerate(ASSETS):
        ax = axes[ai]
        pred = [df[(df["asset"] == a) & (df["split"] == s)]["pred_pbull_mean"].iloc[0]
                for s in splits_n]
        real = [df[(df["asset"] == a) & (df["split"] == s)]["real_pbull"].iloc[0]
                for s in splits_n]
        ax.bar(x - width/2, pred, width, label="predicho (ec.15)",
               color="#1f77b4", alpha=0.85)
        ax.bar(x + width/2, real, width, label="real (sample)",
               color="#E63946", alpha=0.85)
        ax.axhline(0.5, color="grey", lw=0.6, ls="--", label="0.5")
        ax.set_xticks(x); ax.set_xticklabels(splits_n)
        ax.set_title(f"{a}: p_bull predicho vs real")
        ax.set_ylabel("p_bull")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.25, axis="y")
        ax.legend(loc="best", fontsize=9)
        for xi, (p, r) in enumerate(zip(pred, real)):
            ax.text(xi - width/2, p + 0.01, f"{p:.2f}", ha="center", fontsize=8)
            ax.text(xi + width/2, r + 0.01, f"{r:.2f}", ha="center", fontsize=8)
    save_fig(fig, "3_pbull_compare", SUBDIR)


# ================================================================
# 4) Sensibilidad al threshold
# ================================================================

def diag_threshold_sensitivity(state):
    model = state["model"]
    sp = state["split"]
    df = state["df"]

    # Median of train returns por activo
    train_med = {a: float(np.median(sp.Y_train[:, ai]))
                 for ai, a in enumerate(ASSETS)}
    print(f"\n--- Mediana de retornos TRAIN por activo ---")
    for a in ASSETS:
        print(f"  {a}: {train_med[a]*100:+.3f}%/sem")

    # Predicciones sobre TEST
    preds_test = predict_deciles_batch(model, sp.X_test)   # (N, A, Q)

    rows = []
    thresholds = [-0.020, -0.010, -0.005, 0.0, +0.005, +0.010, +0.020]
    for thr in thresholds:
        for ai, a in enumerate(ASSETS):
            p_pred = float((preds_test[:, ai, :] >= thr).mean())
            p_real = float((sp.Y_test[:, ai] >= thr).mean())
            rows.append({
                "threshold_pct": thr * 100, "asset": a,
                "pred_p_bull": p_pred, "real_p_bull": p_real,
                "is_train_median": (abs(thr - train_med[a]) < 1e-6),
            })
    df_thr = pd.DataFrame(rows)
    save_csv(df_thr, "4_threshold_sensitivity", SUBDIR)

    fig, axes = plt.subplots(1, len(ASSETS), figsize=(13, 4.5), sharey=True)
    if len(ASSETS) == 1:
        axes = [axes]
    for ai, a in enumerate(ASSETS):
        ax = axes[ai]
        d = df_thr[df_thr["asset"] == a]
        ax.plot(d["threshold_pct"], d["pred_p_bull"], "-o",
                color="#1f77b4", label="predicho (TEST)", lw=1.5)
        ax.plot(d["threshold_pct"], d["real_p_bull"], "-s",
                color="#E63946", label="real (TEST)", lw=1.5)
        ax.axvline(0, color="black", lw=0.6, ls="--",
                   label="threshold actual (0)")
        ax.axvline(train_med[a] * 100, color="#F2B705", lw=1.5, ls=":",
                   label=f"mediana TRAIN = {train_med[a]*100:+.3f}%")
        ax.axhline(0.5, color="grey", lw=0.5, ls="--")
        ax.set_xlabel("BULL_THRESHOLD [%]")
        if ai == 0:
            ax.set_ylabel("p_bull")
        ax.set_title(f"{a}: p_bull(threshold) sobre TEST")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)
    save_fig(fig, "4_threshold_sensitivity", SUBDIR)


# ================================================================
# 5) Diagnostico resumen
# ================================================================

def diag_resumen(state):
    """Tabla resumen para defensa con las tres hipotesis."""
    model = state["model"]
    sp = state["split"]
    preds_train = predict_deciles_batch(model, sp.X_train)
    preds_test  = predict_deciles_batch(model, sp.X_test)
    Qmed = len(DECILES) // 2

    rows = []
    for ai, a in enumerate(ASSETS):
        # H1: mediana real de retornos
        med_train = float(np.median(sp.Y_train[:, ai]))
        med_test  = float(np.median(sp.Y_test[:, ai]))
        # H2: mediana del quintil 0.5 predicha
        med_q05_train = float(preds_train[:, ai, Qmed].mean())
        med_q05_test  = float(preds_test [:, ai, Qmed].mean())
        # H3: p_bull discrepancy
        pred_pbull_train = float((preds_train[:, ai, :] >= BULL_THRESHOLD).mean())
        pred_pbull_test  = float((preds_test [:, ai, :] >= BULL_THRESHOLD).mean())
        real_pbull_train = float((sp.Y_train[:, ai] >= BULL_THRESHOLD).mean())
        real_pbull_test  = float((sp.Y_test [:, ai] >= BULL_THRESHOLD).mean())

        rows.append({
            "asset": a,
            "H1_median_real_TRAIN": med_train * 100,
            "H1_median_real_TEST":  med_test * 100,
            "H2_q05_pred_TRAIN":    med_q05_train * 100,
            "H2_q05_pred_TEST":     med_q05_test * 100,
            "H3_real_pbull_TRAIN":  real_pbull_train,
            "H3_pred_pbull_TRAIN":  pred_pbull_train,
            "H3_real_pbull_TEST":   real_pbull_test,
            "H3_pred_pbull_TEST":   pred_pbull_test,
        })
    df = pd.DataFrame(rows)
    save_csv(df, "5_diagnostico", SUBDIR)
    print("\n--- Diagnostico resumen ---")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# ================================================================
# Main
# ================================================================

def main():
    print("=" * 70)
    print("CALIBRACION DEL LSTM: por que p_bull < 0.5?")
    print("=" * 70)
    state = load_state()
    diag_returns_stats(state)
    diag_quintiles_pred(state)
    diag_pbull_compare(state)
    diag_threshold_sensitivity(state)
    diag_resumen(state)
    print("\n" + "=" * 70)
    print("Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
