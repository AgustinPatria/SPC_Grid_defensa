"""L5 — Inspeccion del regret grid y seleccion g*.

  V[g, s] : capital terminal de la politica g aplicada al escenario s.
  R[g, s] = max_g' V[g', s] - V[g, s]  (ec. 22).
  g*_mean  = argmin_g  mean_s R[g, s]   (ec. 23)
  g*_worst = argmin_g  max_s  R[g, s]   (ec. 24)

Outputs:
  1. Heatmap V[g, s]  (15 x 5) IS y OOS.
  2. Heatmap R[g, s]  (15 x 5) IS y OOS.
  3. mean_regret en plano (lambda, m) (5 x 3) IS y OOS, con g*_mean marcado.
  4. worst_regret en plano (lambda, m) (5 x 3) IS y OOS, con g*_worst marcado.
  5. Curvas mean_regret vs lambda (una linea por m) y vs m (una linea por lambda).
  6. Per-cell: en cual escenario s* da peor V cada g (worst-scenario map).
  7. Comparativo IS vs OOS de la seleccion.

Output: inspeccion_v2/L5_regret_out/
"""
import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from inspeccion_v2._common import (
    LAMBDA_GRID,
    M_GRID,
    load_nns,
    run_or_load_grid,
    save_csv,
    save_fig,
)


SUBDIR = "L5_regret"


# ---------------------------------------------------------------- 1+2) heatmaps V/R
def plot_V_or_R(table: pd.DataFrame, label: str, kind: str,
                g_star: tuple | None = None):
    """Heatmap (lambda, m) x s con valores anotados."""
    arr = table.values
    cells = list(table.index)
    n_S = arr.shape[1]
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(cells) + 1.5))
    cmap = "RdYlGn" if kind == "V" else "Reds_r"
    im = ax.imshow(arr, aspect="auto", cmap=cmap)
    ax.set_xticks(range(n_S))
    ax.set_xticklabels([f"s={s}" for s in table.columns])
    ax.set_yticks(range(len(cells)))
    ax.set_yticklabels([f"l={lm[0]:.2f}, m={lm[1]:.2f}" for lm in cells],
                       fontsize=8)
    for ri in range(len(cells)):
        for sj in range(n_S):
            ax.text(sj, ri, f"${arr[ri, sj]:,.0f}", ha="center", va="center",
                    fontsize=7, color="black")
    if g_star is not None and g_star in cells:
        gi = cells.index(g_star)
        ax.add_patch(plt.Rectangle((-0.5, gi - 0.5), n_S, 1,
                                    fill=False, edgecolor="blue", linewidth=2))
    fig.colorbar(im, ax=ax, label=kind)
    ax.set_title(f"L5 — {kind}[g, s] ({label})")
    save_fig(fig, f"0{1 if kind == 'V' else 2}_{kind}_table_{label}", SUBDIR)


# ---------------------------------------------------------------- 3+4) plano lam-m
def plot_regret_plane(summary: pd.DataFrame, label: str, kind: str,
                      g_star: tuple):
    """Heatmap (lambda x m) de mean_regret o worst_regret."""
    col = "mean_regret" if kind == "mean" else "worst_regret"
    lam_vals = sorted(set(summary.index.get_level_values("lambda")))
    m_vals   = sorted(set(summary.index.get_level_values("m")))
    arr = np.zeros((len(lam_vals), len(m_vals)))
    for li, lam in enumerate(lam_vals):
        for mi, m_ in enumerate(m_vals):
            arr[li, mi] = summary.loc[(lam, m_), col]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(arr, cmap="Reds", aspect="auto", origin="lower")
    ax.set_xticks(range(len(m_vals)))
    ax.set_xticklabels([f"{m_:.2f}" for m_ in m_vals])
    ax.set_yticks(range(len(lam_vals)))
    ax.set_yticklabels([f"{l:.2f}" for l in lam_vals])
    ax.set_xlabel("m (costo_mult)")
    ax.set_ylabel("lambda")
    for li in range(len(lam_vals)):
        for mi in range(len(m_vals)):
            ax.text(mi, li, f"${arr[li, mi]:,.0f}", ha="center", va="center",
                    fontsize=8, color="black")
    if g_star in zip([l for l in lam_vals for _ in m_vals],
                      [m_ for _ in lam_vals for m_ in m_vals]):
        li = lam_vals.index(g_star[0])
        mi = m_vals.index(g_star[1])
        ax.add_patch(plt.Rectangle((mi - 0.5, li - 0.5), 1, 1,
                                    fill=False, edgecolor="blue", linewidth=3))
        ax.text(mi, li - 0.32, f"g*_{kind}", ha="center", va="bottom",
                color="blue", fontsize=9, fontweight="bold")
    fig.colorbar(im, ax=ax, label=col)
    ax.set_title(f"L5 — {col} en plano (lambda, m) ({label})")
    save_fig(fig, f"0{3 if kind == 'mean' else 4}_{kind}_plane_{label}",
              SUBDIR)


# ---------------------------------------------------------------- 5) curves
def plot_regret_curves(summary: pd.DataFrame, label: str):
    """mean_regret vs lambda (una linea por m) y vs m (una linea por lambda)."""
    lam_vals = sorted(set(summary.index.get_level_values("lambda")))
    m_vals   = sorted(set(summary.index.get_level_values("m")))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # vs lambda, una linea por m
    for m_ in m_vals:
        ys = [summary.loc[(lam, m_), "mean_regret"] for lam in lam_vals]
        axes[0].plot(lam_vals, ys, marker="o", label=f"m={m_:.2f}")
    axes[0].set_xlabel("lambda")
    axes[0].set_ylabel("mean_regret")
    axes[0].set_title("mean_regret vs lambda  (lineas: m)")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    # vs m, una linea por lambda
    for lam in lam_vals:
        ys = [summary.loc[(lam, m_), "mean_regret"] for m_ in m_vals]
        axes[1].plot(m_vals, ys, marker="o", label=f"l={lam:.2f}")
    axes[1].set_xlabel("m (costo_mult)")
    axes[1].set_ylabel("mean_regret")
    axes[1].set_title("mean_regret vs m  (lineas: lambda)")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    fig.suptitle(f"L5 — curvas de mean_regret ({label})", fontsize=12)
    save_fig(fig, f"05_curves_{label}", SUBDIR)


# ---------------------------------------------------------------- 6) worst-scen
def plot_worst_scenario_map(V_table: pd.DataFrame, label: str):
    """Para cada g, en cual escenario s da peor V (color = s_worst)."""
    cells = list(V_table.index)
    s_worst = V_table.values.argmin(axis=1)         # (n_cells,)
    lam_vals = sorted({g[0] for g in cells})
    m_vals   = sorted({g[1] for g in cells})
    arr = np.full((len(lam_vals), len(m_vals)), -1, dtype=int)
    for (lam, m_), s in zip(cells, s_worst):
        arr[lam_vals.index(lam), m_vals.index(m_)] = int(s)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(arr, cmap="viridis", aspect="auto", origin="lower",
                   vmin=0, vmax=V_table.shape[1] - 1)
    for li in range(len(lam_vals)):
        for mi in range(len(m_vals)):
            ax.text(mi, li, f"s={arr[li, mi]}", ha="center", va="center",
                    color="white", fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(m_vals)))
    ax.set_xticklabels([f"{m_:.2f}" for m_ in m_vals])
    ax.set_yticks(range(len(lam_vals)))
    ax.set_yticklabels([f"{l:.2f}" for l in lam_vals])
    ax.set_xlabel("m")
    ax.set_ylabel("lambda")
    fig.colorbar(im, ax=ax, label="s_worst (escenario peor)")
    ax.set_title(f"L5 — escenario peor por celda ({label})")
    save_fig(fig, f"06_worst_scenario_{label}", SUBDIR)


# ---------------------------------------------------------------- main
def main(force_retrain: bool = False):
    print("=" * 70)
    print("L5 — Inspeccion regret grid + seleccion g*")
    print("=" * 70)

    nns = load_nns(force_retrain=force_retrain)

    selections = {}
    for label in ["is", "oos"]:
        print(f"\n--- {label.upper()} ---")
        payload = run_or_load_grid(label, nns)
        res = payload["res"]
        V_table = res["V_table"]
        R_table = res["R_table"]
        summary = res["regret_summary"]
        g_mean  = res["g_mean"]
        g_worst = res["g_worst"]
        selections[label] = (g_mean, g_worst)

        print(f"  g*_mean  = {g_mean}   mean_regret = ${res['g_mean_metric']:,.2f}")
        print(f"  g*_worst = {g_worst}  worst_regret= ${res['g_worst_metric']:,.2f}")

        # 1) V table
        plot_V_or_R(V_table, label.upper(), kind="V", g_star=g_mean)
        # 2) R table
        plot_V_or_R(R_table, label.upper(), kind="R", g_star=g_mean)
        # 3) mean plane
        plot_regret_plane(summary, label.upper(), kind="mean", g_star=g_mean)
        # 4) worst plane
        plot_regret_plane(summary, label.upper(), kind="worst", g_star=g_worst)
        # 5) curves
        plot_regret_curves(summary, label.upper())
        # 6) worst scenario map
        plot_worst_scenario_map(V_table, label.upper())

        # CSV
        save_csv(V_table.reset_index(), f"01_V_table_{label.upper()}", SUBDIR)
        save_csv(R_table.reset_index(), f"02_R_table_{label.upper()}", SUBDIR)
        save_csv(summary.reset_index(), f"05_regret_summary_{label.upper()}",
                  SUBDIR)

        # console: pivot mean_regret vs (lambda, m)
        pivot_mean = summary["mean_regret"].unstack("m")
        print(f"\n  mean_regret pivot {label.upper()} (filas=lambda, cols=m):")
        print(pivot_mean.to_string(float_format="${:,.0f}".format))
        pivot_worst = summary["worst_regret"].unstack("m")
        print(f"\n  worst_regret pivot {label.upper()} (filas=lambda, cols=m):")
        print(pivot_worst.to_string(float_format="${:,.0f}".format))

    # Comparativo seleccion
    summ = pd.DataFrame([
        {"label": k.upper(), "g_mean": s[0], "g_worst": s[1]}
        for k, s in selections.items()
    ])
    save_csv(summ, "00_selection_summary", SUBDIR)
    print("\n  Resumen de selecciones:")
    print(summ.to_string(index=False))

    print("\n  L5 listo.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--retrain", action="store_true",
                   help="Borra cache pickle + reentrena las 15 NNs.")
    args = p.parse_args()
    main(force_retrain=args.retrain)
