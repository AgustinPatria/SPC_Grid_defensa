"""L4 — Inspeccion del motor del optimizador GAMS.

Para cada celda g = (lambda, m), descompone la solucion en sus piezas:

  FO:  z = sum_t [ sum_i w(i,t)*mu(i,t)
                 - lambda * ( sum_ij w_i(t)*w_j(t)*sigma_ij(t) - V_max )
                 - sum_i c_base(i)*costo_mult*(u(i,t)+v(i,t)) ]

Visualizaciones:
  1. Decomposicion de z en 3 terminos (retorno / riesgo / costo) por celda.
  2. Heatmap w(i, t): pesos por activo y tiempo, una pequena cuadricula
     por celda.
  3. Turnover acumulado sum_t (u+v) por celda y por activo.
  4. Peso medio de cada activo a lo largo del horizonte por celda
     -> ver como (lambda, m) modulan la mezcla.
  5. Mapa del plano (lambda, m): peso medio del activo "agresivo" (CMC200).

Output: inspeccion_v2/L4_optimizador_out/
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


SUBDIR = "L4_optimizador"


# ---------------------------------------------------------------- helpers
def policy_arrays(policy, assets, T_vals):
    """(w_sol, u_sol, v_sol, z) dicts -> arrays (T, A)."""
    w_sol, u_sol, v_sol, z = policy
    T = len(T_vals)
    A = len(assets)
    w = np.zeros((T, A))
    u = np.zeros((T, A))
    v = np.zeros((T, A))
    for ai, a in enumerate(assets):
        for ti, t in enumerate(T_vals):
            w[ti, ai] = w_sol[(a, t)]
            u[ti, ai] = u_sol[(a, t)]
            v[ti, ai] = v_sol[(a, t)]
    return w, u, v, z


def decompose_z(w, u, v, ctx, lam, m):
    """Devuelve (retorno, riesgo_term, costo, suma_check).

    riesgo_term  = lambda * (var_total - T*V_max)   (penalizacion neta)
    var_total    = sum_t sum_ij w_i(t)*w_j(t)*sigma_ij(t)
    """
    assets = ctx["assets"]
    T_vals = ctx["T_vals"]
    c_base = ctx["c_base"]
    V_max = ctx["V_max"]
    mu_mix = ctx["mu_mix"]
    sigma_mix = ctx["sigma_mix"]

    T = len(T_vals)
    A = len(assets)
    mu_arr = np.zeros((T, A))
    for ai, a in enumerate(assets):
        mu_arr[:, ai] = mu_mix[a].values
    sig_arr = np.zeros((T, A, A))
    for ai, a in enumerate(assets):
        for aj, b in enumerate(assets):
            sig_arr[:, ai, aj] = sigma_mix[a][b].values
    c_arr = np.array([c_base[a] for a in assets])

    retorno  = float((w * mu_arr).sum())
    var_t    = np.einsum("ti,tij,tj->t", w, sig_arr, w)             # (T,)
    var_total = float(var_t.sum())
    riesgo   = float(lam * (var_total - T * V_max))
    costo    = float((c_arr * m * (u + v)).sum())
    z_check  = retorno - riesgo - costo
    return {
        "retorno":   retorno,
        "var_total": var_total,
        "V_max":     V_max,
        "T":         T,
        "riesgo":    riesgo,
        "costo":     costo,
        "z_check":   z_check,
    }


# ---------------------------------------------------------------- 1) z decomp
def plot_z_decomposition(rows_df: pd.DataFrame, label: str):
    """Barras stacked por celda: retorno (verde), -riesgo (rojo), -costo (gris).

    Las barras suman z. Una etiqueta arriba muestra z.
    """
    cells = list(zip(rows_df["lambda"], rows_df["m"]))
    n = len(cells)
    x = np.arange(n)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x, rows_df["retorno"], color="C2", label="retorno")
    ax.bar(x, -rows_df["riesgo"], bottom=rows_df["retorno"],
           color="C3", label="-riesgo")
    ax.bar(x, -rows_df["costo"], bottom=rows_df["retorno"] - rows_df["riesgo"],
           color="grey", label="-costo")
    z = rows_df["retorno"] - rows_df["riesgo"] - rows_df["costo"]
    for xi, zi in zip(x, z):
        ax.text(xi, zi + 0.02 * max(abs(z)), f"{zi:.3f}",
                ha="center", fontsize=7, rotation=0)
    ax.set_xticks(x)
    ax.set_xticklabels([f"l={lm[0]:.2f}\nm={lm[1]:.2f}" for lm in cells],
                       fontsize=7, rotation=90)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("contribucion a z")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.suptitle(f"L4 — decomposicion de z por celda ({label})", fontsize=12)
    save_fig(fig, f"01_z_decomposition_{label}", SUBDIR)


# ---------------------------------------------------------------- 2) w heatmap
def plot_w_heatmaps(policies, contexts, assets, label: str):
    """Heatmap w(i, t) para cada celda en una cuadricula 5x3 (lambda x m)."""
    lam_vals = sorted({g[0] for g in policies.keys()})
    m_vals   = sorted({g[1] for g in policies.keys()})
    rows = len(lam_vals)
    cols = len(m_vals)
    A = len(assets)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.0, rows * 1.8),
                             sharex=True, sharey=True)
    if rows == 1: axes = axes[None, :]
    if cols == 1: axes = axes[:, None]
    for li, lam in enumerate(lam_vals):
        for mi, m_ in enumerate(m_vals):
            g = (lam, m_)
            ctx = contexts[g]
            w, _, _, _ = policy_arrays(policies[g], assets, ctx["T_vals"])
            ax = axes[li, mi]
            im = ax.imshow(w.T, aspect="auto", cmap="viridis",
                           vmin=0, vmax=1, origin="lower",
                           extent=[ctx["T_vals"][0], ctx["T_vals"][-1],
                                   -0.5, A - 0.5])
            ax.set_title(f"l={lam:.2f}, m={m_:.2f}", fontsize=9)
            ax.set_yticks(range(A))
            ax.set_yticklabels(assets, fontsize=8)
    fig.colorbar(im, ax=axes.ravel().tolist(), label="w(i,t)",
                 shrink=0.7, pad=0.02)
    fig.suptitle(f"L4 — pesos w(i, t) por celda ({label})", fontsize=12)
    save_fig(fig, f"02_w_heatmaps_{label}", SUBDIR)


# ---------------------------------------------------------------- 3) turnover
def turnover_table(policies, contexts, assets) -> pd.DataFrame:
    rows = []
    for g, pol in policies.items():
        ctx = contexts[g]
        w, u, v, z = policy_arrays(pol, assets, ctx["T_vals"])
        row = {"lambda": g[0], "m": g[1], "z": z}
        for ai, a in enumerate(assets):
            row[f"turnover_{a}"] = float((u[:, ai] + v[:, ai]).sum())
            row[f"w_mean_{a}"]   = float(w[:, ai].mean())
            row[f"w_init_{a}"]   = float(w[0, ai])
            row[f"w_final_{a}"]  = float(w[-1, ai])
        row["turnover_total"] = float((u + v).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def plot_turnover_bars(df: pd.DataFrame, label: str):
    cells = list(zip(df["lambda"], df["m"]))
    n = len(cells)
    x = np.arange(n)
    fig, ax = plt.subplots(figsize=(12, 5))
    cols_turn = [c for c in df.columns if c.startswith("turnover_") and c != "turnover_total"]
    bottoms = np.zeros(n)
    for i, col in enumerate(cols_turn):
        ax.bar(x, df[col], bottom=bottoms, label=col.replace("turnover_", ""))
        bottoms = bottoms + df[col].values
    ax.set_xticks(x)
    ax.set_xticklabels([f"l={lm[0]:.2f}\nm={lm[1]:.2f}" for lm in cells],
                       fontsize=7, rotation=90)
    ax.set_ylabel("turnover total = sum_t (u + v)")
    ax.legend(loc="upper right", fontsize=8, title="activo")
    ax.grid(alpha=0.3, axis="y")
    fig.suptitle(f"L4 — turnover total por celda ({label})", fontsize=12)
    save_fig(fig, f"03_turnover_{label}", SUBDIR)


# ---------------------------------------------------------------- 4) w mean
def plot_w_mean(df: pd.DataFrame, assets, label: str):
    """Para cada activo, scatter w_mean en plano (lambda, m)."""
    fig, axes = plt.subplots(1, len(assets), figsize=(5 * len(assets), 4.2))
    if len(assets) == 1: axes = [axes]
    for ai, a in enumerate(assets):
        ax = axes[ai]
        sc = ax.scatter(df["lambda"], df["m"], c=df[f"w_mean_{a}"],
                        s=200, cmap="viridis", vmin=0, vmax=1)
        for _, r in df.iterrows():
            ax.text(r["lambda"], r["m"], f"{r[f'w_mean_{a}']:.2f}",
                    ha="center", va="center", fontsize=7, color="white")
        ax.set_xlabel("lambda")
        ax.set_ylabel("m")
        ax.set_title(f"w_mean ({a})")
        ax.grid(alpha=0.3)
        fig.colorbar(sc, ax=ax, label=f"w_mean_{a}")
    fig.suptitle(f"L4 — peso medio por activo en plano (lambda, m) ({label})",
                 fontsize=12)
    save_fig(fig, f"04_w_mean_plane_{label}", SUBDIR)


# ---------------------------------------------------------------- 5) stacked w(t)
def plot_w_stacked(policies, contexts, assets, label: str):
    """Stacked area chart de w(i, t) por celda en grilla (lambda x m).

    La frontera entre las dos bandas ES la trayectoria de rebalanceo —
    plana = no rebalanceo, ondulante = rebalanceo activo.
    """
    lam_vals = sorted({g[0] for g in policies.keys()})
    m_vals   = sorted({g[1] for g in policies.keys()})
    rows = len(lam_vals)
    cols = len(m_vals)
    A = len(assets)
    # Colores por activo (consistentes entre celdas)
    palette = plt.cm.tab10(np.arange(A))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 2.1),
                             sharex=True, sharey=True)
    if rows == 1: axes = axes[None, :]
    if cols == 1: axes = axes[:, None]
    for li, lam in enumerate(lam_vals):
        for mi, m_ in enumerate(m_vals):
            g = (lam, m_)
            ctx = contexts[g]
            T_vals = ctx["T_vals"]
            w, _, _, _ = policy_arrays(policies[g], assets, T_vals)
            ax = axes[li, mi]
            ax.stackplot(T_vals, w.T, labels=assets,
                         colors=palette[:A], alpha=0.85,
                         edgecolor="white", linewidth=0.2)
            ax.set_ylim(0, 1)
            ax.set_xlim(T_vals[0], T_vals[-1])
            # turnover total como header
            turn = float(sum(abs(w[ti+1, ai] - w[ti, ai])
                             for ai in range(A) for ti in range(len(T_vals)-1)))
            ax.set_title(f"l={lam:.2f}, m={m_:.2f}   "
                         f"|Δw|_tot={turn:.2f}", fontsize=9)
            ax.grid(alpha=0.3, axis="x")
            ax.tick_params(labelsize=7)
            if li == rows - 1:
                ax.set_xlabel("t")
            if mi == 0:
                ax.set_ylabel("w(i, t)")
    # leyenda comun (usa el primer panel)
    handles, labels_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper center", ncol=A,
               bbox_to_anchor=(0.5, 1.02), fontsize=10)
    fig.suptitle(f"L4 — trayectoria de pesos w(i, t) por celda ({label})",
                 fontsize=12, y=1.05)
    save_fig(fig, f"05_w_stacked_{label}", SUBDIR)


# ---------------------------------------------------------------- main
def main(force_retrain: bool = False):
    print("=" * 70)
    print("L4 — Inspeccion motor optimizador")
    print("=" * 70)

    nns = load_nns(force_retrain=force_retrain)

    for label in ["is", "oos"]:
        print(f"\n--- {label.upper()} ---")
        payload = run_or_load_grid(label, nns)
        contexts = payload["contexts"]
        policies = payload["policies"]
        assets = contexts[list(contexts.keys())[0]]["assets"]

        # 1) z decomposition
        rows = []
        for g, pol in policies.items():
            ctx = contexts[g]
            w, u, v, z = policy_arrays(pol, assets, ctx["T_vals"])
            dec = decompose_z(w, u, v, ctx, lam=g[0], m=g[1])
            rows.append({
                "lambda": g[0], "m": g[1], "z_reported": z,
                "z_check": dec["z_check"], "retorno": dec["retorno"],
                "riesgo": dec["riesgo"], "costo": dec["costo"],
                "var_total": dec["var_total"], "T": dec["T"],
                "V_max": dec["V_max"],
            })
        dec_df = pd.DataFrame(rows).sort_values(["lambda", "m"]).reset_index(drop=True)
        save_csv(dec_df, f"01_z_decomposition_{label.upper()}", SUBDIR)

        # Sanity: z_check ~ z_reported (modulo precision IPOPT)
        diff = (dec_df["z_check"] - dec_df["z_reported"]).abs().max()
        print(f"  sanity max|z_check - z_reported| = {diff:.6f}  "
              f"(deberia ser << 1e-3)")

        plot_z_decomposition(dec_df, label.upper())

        # 2) w heatmaps
        plot_w_heatmaps(policies, contexts, assets, label.upper())

        # 3) turnover
        turn_df = turnover_table(policies, contexts, assets)
        save_csv(turn_df, f"03_turnover_{label.upper()}", SUBDIR)
        plot_turnover_bars(turn_df, label.upper())

        # 4) w_mean en plano (lambda, m)
        plot_w_mean(turn_df, assets, label.upper())

        # 5) trayectoria de pesos (stacked area)
        plot_w_stacked(policies, contexts, assets, label.upper())

        # 5) console summary
        print(f"\n  Decomposicion {label.upper()} (head):")
        print(dec_df[["lambda", "m", "retorno", "riesgo", "costo",
                      "z_reported"]].to_string(
            index=False, float_format="{:+.5f}".format))
        print(f"\n  Pesos medios {label.upper()}:")
        cols = ["lambda", "m"] + [f"w_mean_{a}" for a in assets] \
             + ["turnover_total"]
        print(turn_df[cols].to_string(
            index=False, float_format="{:.4f}".format))

    print("\n  L4 listo.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--retrain", action="store_true",
                   help="Borra cache pickle + reentrena las 15 NNs.")
    args = p.parse_args()
    main(force_retrain=args.retrain)
