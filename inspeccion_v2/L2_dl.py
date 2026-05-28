"""L2 — Inspeccion del motor DL.

Genera, para cada activo:
  1. p_bull(t) — las 15 NNs superpuestas + ensemble (linea negra) sobre
     t=1..T_HORIZON, con marca vertical en t_test_start.
  2. Dispersion entre NNs por t: std y (max-min) de p_bull.
  3. Calibracion direccional: accuracy(p_bull(t) > 0.5  vs  sign(r_real(t)))
     para cada celda + ensemble. Tabla y barplot.
  4. mu_mix(t) por celda + mu_mix_ensemble (negro) + retorno real (gris fino).
  5. mu_hat por (activo, regimen) por celda (mu_hat es global por celda —
     no depende de t — pero las 15 celdas dan 15 valores).

Output: inspeccion_v2/L2_dl_out/{...}.png + .csv
"""
import argparse
import sys
from pathlib import Path

# Permite ejecutar `python inspeccion_v2/L2_dl.py` desde la raiz del proyecto.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from inspeccion_v2._common import (
    T_HORIZON,
    build_ensemble_model,
    hist_returns,
    load_contexts_is,
    load_nns,
    save_csv,
    save_fig,
)
from Regret_Grid import build_per_cell_context, compute_test_start_t
from config import SPLIT


SUBDIR = "L2_dl"


# ---------------------------------------------------------------- 1) p_bull
def plot_pbull_by_cell(contexts_is, ensemble_pbull_is, t_test_start, assets):
    """Una figura con 2 paneles (uno por activo); 15 lineas grises + ensemble negro."""
    fig, axes = plt.subplots(len(assets), 1, figsize=(11, 6), sharex=True)
    if len(assets) == 1:
        axes = [axes]
    T_vals = list(range(1, T_HORIZON + 1))
    for ai, asset in enumerate(assets):
        ax = axes[ai]
        for g, ctx in contexts_is.items():
            p = ctx["p_dl"][asset]["bull"].values
            ax.plot(T_vals, p, color="grey", alpha=0.35, linewidth=0.9)
        ax.plot(T_vals, ensemble_pbull_is[asset], color="black", linewidth=1.8,
                label="ensemble (avg logits)")
        ax.axvline(t_test_start, color="red", linestyle="--", alpha=0.7,
                   label=f"t_test_start={t_test_start}")
        ax.axhline(0.5, color="blue", linestyle=":", alpha=0.4)
        ax.set_ylabel(f"p_bull(t) — {asset}")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("t (semana)")
    fig.suptitle("L2 — p_bull(t) por celda (15 NNs) + ensemble", fontsize=12)
    save_fig(fig, "01_pbull_by_cell", SUBDIR)


def compute_pbull_matrix(contexts, assets, key="bull"):
    """Devuelve dict {asset: np.array (n_cells, T_vals_len)}."""
    cells = list(contexts.keys())
    out = {}
    for a in assets:
        rows = [contexts[g]["p_dl"][a][key].values for g in cells]
        out[a] = np.stack(rows, axis=0)         # (n_cells, T)
    return out, cells


# ---------------------------------------------------------------- 2) dispersion
def plot_dispersion(pbull_mat_is, t_test_start, assets):
    fig, axes = plt.subplots(len(assets), 1, figsize=(11, 6), sharex=True)
    if len(assets) == 1:
        axes = [axes]
    T_vals = list(range(1, T_HORIZON + 1))
    for ai, asset in enumerate(assets):
        ax = axes[ai]
        mat = pbull_mat_is[asset]               # (n_cells, T)
        std = mat.std(axis=0)
        spread = mat.max(axis=0) - mat.min(axis=0)
        ax.plot(T_vals, std, color="C0", label="std entre 15 NNs")
        ax.plot(T_vals, spread, color="C3", alpha=0.6, label="max - min")
        ax.axvline(t_test_start, color="red", linestyle="--", alpha=0.7)
        ax.set_ylabel(f"dispersion — {asset}")
        ax.set_ylim(-0.02, 0.65)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("t (semana)")
    fig.suptitle("L2 — dispersion de p_bull entre las 15 NNs", fontsize=12)
    save_fig(fig, "02_dispersion", SUBDIR)


# ---------------------------------------------------------------- 3) calibracion
def compute_direction_accuracy(pbull_mat, r_real_dict, cells, assets,
                               t_start=1, t_end=T_HORIZON):
    """
    accuracy(p_bull > 0.5  vs  r_real >= 0) por celda y por activo.
    pbull_mat: dict {asset: (n_cells, T)}, T = T_HORIZON, t-indexado desde 1.
    """
    rows = []
    idx_a, idx_b = t_start - 1, t_end  # python slicing
    for ai, a in enumerate(assets):
        r_real = r_real_dict[a][idx_a:idx_b]
        bull_real = (r_real >= 0).astype(np.float32)
        for ci, g in enumerate(cells):
            p = pbull_mat[a][ci, idx_a:idx_b]
            pred = (p > 0.5).astype(np.float32)
            acc = float((pred == bull_real).mean())
            base = float(bull_real.mean())
            rows.append({
                "asset": a, "lambda": g[0], "m": g[1],
                "accuracy_dir": acc, "base_rate_bull": base,
                "p_bull_mean": float(p.mean()),
                "p_bull_std":  float(p.std()),
                "t_start": t_start, "t_end": t_end,
            })
    return pd.DataFrame(rows)


def plot_calibration_bars(df_cal, assets, label: str):
    """Barplot: accuracy direccional por celda."""
    fig, axes = plt.subplots(1, len(assets), figsize=(12, 4), sharey=True)
    if len(assets) == 1:
        axes = [axes]
    for ai, a in enumerate(assets):
        ax = axes[ai]
        sub = df_cal[df_cal["asset"] == a].copy()
        labels = [f"l={l:.2f}\nm={m:.2f}" for l, m in zip(sub["lambda"], sub["m"])]
        ax.bar(range(len(sub)), sub["accuracy_dir"], color="C0")
        base = sub["base_rate_bull"].iloc[0]
        ax.axhline(base, color="red", linestyle="--", alpha=0.7,
                   label=f"base rate bull={base:.2f}")
        ax.axhline(0.5, color="grey", linestyle=":", alpha=0.5,
                   label="coin flip")
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels(labels, fontsize=7, rotation=90)
        ax.set_title(a)
        ax.set_ylim(0.3, 1.0)
        ax.grid(alpha=0.3, axis="y")
        ax.legend(loc="upper right", fontsize=8)
    axes[0].set_ylabel("accuracy direccional")
    fig.suptitle(f"L2 — calibracion direccional p_bull>0.5 vs r>=0 ({label})",
                 fontsize=12)
    save_fig(fig, f"03_calibration_{label}", SUBDIR)


# ---------------------------------------------------------------- 4) mu_mix
def plot_mu_mix(contexts_is, assets, r_real_dict, t_test_start):
    fig, axes = plt.subplots(len(assets), 1, figsize=(11, 6), sharex=True)
    if len(assets) == 1:
        axes = [axes]
    T_vals = list(range(1, T_HORIZON + 1))
    for ai, asset in enumerate(assets):
        ax = axes[ai]
        all_mu = []
        for g, ctx in contexts_is.items():
            mu = ctx["mu_mix"][asset].values
            ax.plot(T_vals, mu, color="grey", alpha=0.30, linewidth=0.8)
            all_mu.append(mu)
        mu_avg = np.mean(np.stack(all_mu, axis=0), axis=0)
        ax.plot(T_vals, mu_avg, color="black", linewidth=1.8,
                label="promedio entre celdas")
        # Retorno real (smoothed con media movil 4 sem para legibilidad)
        r = r_real_dict[asset]
        r_smooth = pd.Series(r).rolling(4, min_periods=1).mean().values
        ax.plot(T_vals, r_smooth, color="C1", alpha=0.6, linewidth=1.0,
                label="r_real(t) MA(4)")
        ax.axvline(t_test_start, color="red", linestyle="--", alpha=0.7)
        ax.axhline(0, color="black", linestyle=":", alpha=0.3)
        ax.set_ylabel(f"mu_mix(t) — {asset}")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("t (semana)")
    fig.suptitle("L2 — mu_mix(t) por celda + r_real(t) MA(4)", fontsize=12)
    save_fig(fig, "04_mu_mix_by_cell", SUBDIR)


# ---------------------------------------------------------------- 5) mu_hat
def mu_hat_table(contexts, assets, regimes=("bear", "bull")):
    """mu_hat por celda — un valor por (activo, regimen). 15 celdas x 4 columnas."""
    rows = []
    for g, ctx in contexts.items():
        mh = ctx["mu_hat"]
        row = {"lambda": g[0], "m": g[1]}
        for a in assets:
            for k in regimes:
                row[f"mu_hat_{a}_{k}"] = float(mh[(a, k)])
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- main
def main(force_retrain: bool = False):
    print("=" * 70)
    print("L2 — Inspeccion motor DL (15 NNs + ensemble)")
    print("=" * 70)

    nns = load_nns(force_retrain=force_retrain)
    ensemble = build_ensemble_model(nns)
    print(f"  ensemble: {len(ensemble.nets)} nets apiladas")

    H = ensemble.config.H
    t_test_start = compute_test_start_t(T=T_HORIZON, H=H, split=SPLIT)

    print("\n[1/5] Construyendo contextos IS (15 celdas, walking) ...")
    contexts_is = load_contexts_is(nns, mu_hat_source="p_hist")
    assets = contexts_is[list(contexts_is.keys())[0]]["assets"]
    print(f"      activos: {assets}")

    # Ensemble p_bull IS — usamos predict_pbull_walking sobre el ensemble.
    # Para no tocar Regret_Grid, reusamos el wrapper de build_per_cell_context
    # con el ensemble como NN.
    print("\n[2/5] Ensemble p_bull(t) IS ...")
    ctx_ens_is = build_per_cell_context(
        nn=ensemble, T=T_HORIZON, mu_hat_source="p_hist",
    )
    ensemble_pbull_is = {a: ctx_ens_is["p_dl"][a]["bull"].values for a in assets}

    # Matriz (n_cells, T) por activo
    pbull_mat_is, cells = compute_pbull_matrix(contexts_is, assets)

    # 1) curvas p_bull
    print("\n[3/5] Plots ...")
    plot_pbull_by_cell(contexts_is, ensemble_pbull_is, t_test_start, assets)

    # 2) dispersion
    plot_dispersion(pbull_mat_is, t_test_start, assets)

    # 3) calibracion direccional — IS post-warmup (t=H+1..T)
    # t<H+1 esta padded a un valor constante por predict_pbull_walking —
    # incluirlo distorsiona la accuracy. Usamos t=H+1..T.
    r_real = hist_returns(assets, T=T_HORIZON)
    df_cal_is = compute_direction_accuracy(
        pbull_mat_is, r_real, cells, assets, t_start=H + 1, t_end=T_HORIZON,
    )
    # ensemble como fila extra
    pbull_mat_ens = {a: ensemble_pbull_is[a][None, :] for a in assets}
    df_cal_ens_is = compute_direction_accuracy(
        pbull_mat_ens, r_real, [(-1, -1)], assets,
        t_start=H + 1, t_end=T_HORIZON,
    )
    df_cal_ens_is["lambda"] = "ensemble"
    df_cal_ens_is["m"] = "—"
    df_cal_is_full = pd.concat([df_cal_is, df_cal_ens_is], ignore_index=True)
    save_csv(df_cal_is_full, "03_calibration_IS", SUBDIR)
    plot_calibration_bars(df_cal_is, assets, label="IS")

    # Calibracion OOS (segmento test del split LSTM)
    df_cal_oos = compute_direction_accuracy(
        pbull_mat_is, r_real, cells, assets,
        t_start=t_test_start, t_end=T_HORIZON,
    )
    df_cal_ens_oos = compute_direction_accuracy(
        pbull_mat_ens, r_real, [(-1, -1)], assets,
        t_start=t_test_start, t_end=T_HORIZON,
    )
    df_cal_ens_oos["lambda"] = "ensemble"
    df_cal_ens_oos["m"] = "—"
    df_cal_oos_full = pd.concat([df_cal_oos, df_cal_ens_oos], ignore_index=True)
    save_csv(df_cal_oos_full, "03_calibration_OOS", SUBDIR)
    plot_calibration_bars(df_cal_oos, assets, label="OOS")

    # 4) mu_mix
    plot_mu_mix(contexts_is, assets, r_real, t_test_start)

    # 5) mu_hat por celda
    mu_hat_df = mu_hat_table(contexts_is, assets)
    save_csv(mu_hat_df, "05_mu_hat_per_cell_IS", SUBDIR)
    print("\n[4/5] mu_hat por celda IS (head):")
    print(mu_hat_df.to_string(index=False, float_format="{:+.6f}".format))

    # 6) Estadisticas resumen — bullets clave
    print("\n[5/5] Resumen IS:")
    for a in assets:
        mat = pbull_mat_is[a]
        std_avg = mat.std(axis=0).mean()
        spread_max = (mat.max(axis=0) - mat.min(axis=0)).max()
        print(f"  {a}: std(15NNs)_avg_over_t = {std_avg:.4f}  "
              f"max(max-min)_over_t = {spread_max:.4f}  "
              f"mean p_bull = {mat.mean():.3f}")
    print("\n  Accuracy direccional IS (resumen):")
    pivot = df_cal_is_full.pivot_table(
        index=["lambda", "m"], columns="asset", values="accuracy_dir")
    print(pivot.to_string(float_format="{:.3f}".format))

    print("\n  Accuracy direccional OOS (resumen):")
    pivot_oos = df_cal_oos_full.pivot_table(
        index=["lambda", "m"], columns="asset", values="accuracy_dir")
    print(pivot_oos.to_string(float_format="{:.3f}".format))

    # Persistencia full
    long_pbull = []
    for ci, g in enumerate(cells):
        for ai, a in enumerate(assets):
            for ti, t in enumerate(range(1, T_HORIZON + 1)):
                long_pbull.append({
                    "lambda": g[0], "m": g[1], "asset": a, "t": t,
                    "p_bull": float(pbull_mat_is[a][ci, ti]),
                })
    save_csv(pd.DataFrame(long_pbull), "00_pbull_long_IS", SUBDIR)

    print("\n  L2 listo.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--retrain", action="store_true",
                   help="Borra cache pickle + reentrena las 15 NNs.")
    args = p.parse_args()
    main(force_retrain=args.retrain)
