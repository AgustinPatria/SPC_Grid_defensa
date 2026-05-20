"""Visualizacion del rebalanceo del portafolio en el tiempo.

Para cada uno de tres setups, resuelve el FO con sus parametros optimos y
grafica:

  1. w(SPX, t) y w(CMC200, t) — trayectoria de pesos
  2. turnover_t = sum_i |w(i,t) - w(i,t-1)| — rebalanceo por periodo
  3. costos acumulados de rebalanceo

Setups:
  - pdf_aligned : rollout + p_hist mu_hat (defaults actuales), g* del main.py
  - legacy      : walking + p_dl mu_hat (codigo viejo), mismo g*
  - opt_base    : sin DL (load_market_data), lambda=1.00, m=1.0 (baseline)

El plot deberia mostrar que pdf_aligned tiende a una politica casi
constante (rollout converge a punto fijo), legacy oscila siguiendo el
walking, y opt_base varia segun la dinamica de p_hist.

Outputs (inspeccion/rebalanceo_out/):
  1_pesos_compare.csv/png      w(i, t) por setup, panel por activo
  2_turnover_compare.csv/png   turnover_t por setup
  3_costos_acumulados.csv/png  costo acumulado de rebalanceo por setup
  4_metricas.csv               numeros agregados (rebalanceo total, costo final, etc.)

Corre con:
    python -m inspeccion.rebalanceo
    (o)
    python inspeccion/rebalanceo.py
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
    CHECKPOINT_PATH,
    DATA_DIR,
    N_CANDIDATES,
    N_SCENARIOS,
    SCENARIO_SEED,
    SUMMARY_ASSET,
    T_HORIZON,
)
from Regret_Grid import (
    build_dl_context,
    load_market_data,
    solve_portfolio,
)


SUBDIR = "rebalanceo"

# g* elegido por main.py en la ultima corrida PDF-aligned. Lambda en
# frontera del grid (limite inferior); m inerte porque mu_mix es constante.
G_MEAN = (0.30, 0.1)

SETUPS = [
    {
        "label":     "pdf_aligned",
        "color":     "#1f77b4",
        "p_method":  "rollout",
        "mu_hat":    "p_hist",
        "lambda":    G_MEAN[0],
        "m":         G_MEAN[1],
        "use_dl":    True,
    },
    {
        "label":     "legacy",
        "color":     "#E63946",
        "p_method":  "walking",
        "mu_hat":    "p_dl",
        "lambda":    G_MEAN[0],
        "m":         G_MEAN[1],
        "use_dl":    True,
    },
    {
        "label":     "opt_base",
        "color":     "#F2B705",
        "p_method":  None,        # no DL
        "mu_hat":    None,
        "lambda":    1.00,
        "m":         1.0,
        "use_dl":    False,
    },
]


def solve_all():
    print("Construyendo contextos y resolviendo FO en cada setup...")
    results = []
    for s in SETUPS:
        label = s["label"]
        print(f"\n  [{label}] lambda={s['lambda']:.2f}  m={s['m']:.1f}  "
              f"p_method={s['p_method']}  mu_hat={s['mu_hat']}")
        if s["use_dl"]:
            ctx = build_dl_context(
                data_dir=DATA_DIR, checkpoint_path=CHECKPOINT_PATH,
                T=T_HORIZON, N_candidates=N_CANDIDATES,
                n_scenarios=N_SCENARIOS, seed=SCENARIO_SEED,
                summary_asset=SUMMARY_ASSET,
                p_method=s["p_method"], mu_hat_source=s["mu_hat"],
            )
        else:
            ctx = load_market_data(str(DATA_DIR))
        z, w, u, v, status = solve_portfolio(
            ctx, lambda_riesgo=s["lambda"], costo_mult=s["m"],
        )
        print(f"    status={status}  z={z:.4f}")
        results.append({**s, "ctx": ctx, "w": w, "u": u, "v": v, "z": z})
    return results


# ================================================================
# 1) Pesos w(i, t) en el tiempo
# ================================================================

def diag_pesos(results):
    assets = results[0]["ctx"]["assets"]
    T_vals = results[0]["ctx"]["T_vals"]

    rows = []
    fig, axes = plt.subplots(len(assets), 1, figsize=(12, 4 * len(assets)),
                             sharex=True)
    if len(assets) == 1:
        axes = [axes]
    for ai, a in enumerate(assets):
        ax = axes[ai]
        ax.axhline(0.5, color="#666", ls="--", lw=0.8, label="w0 = 0.5")
        for r in results:
            w_series = [r["w"][a, t] for t in T_vals]
            ax.plot(T_vals, w_series, color=r["color"], lw=1.5,
                    label=f"{r['label']} (lambda={r['lambda']:.2f}, m={r['m']:.1f})")
            for t, w in zip(T_vals, w_series):
                rows.append({"asset": a, "setup": r["label"],
                             "t": int(t), "w": float(w)})
        ax.set_title(f"w({a}, t) por setup")
        ax.set_xlabel("t")
        ax.set_ylabel(f"w({a})")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=9)
    save_fig(fig, "1_pesos_compare", SUBDIR)
    save_csv(pd.DataFrame(rows), "1_pesos_compare", SUBDIR)


# ================================================================
# 2) Turnover por periodo
# ================================================================

def diag_turnover(results):
    assets = results[0]["ctx"]["assets"]
    T_vals = results[0]["ctx"]["T_vals"]

    rows = []
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for r in results:
        w0_dict = r["ctx"]["w0"]
        turn = []
        for idx, t in enumerate(T_vals):
            if idx == 0:
                # Rebalanceo inicial w0 -> w(t1).
                tt = sum(abs(r["w"][a, t] - w0_dict[a]) for a in assets)
            else:
                t_prev = T_vals[idx - 1]
                tt = sum(abs(r["w"][a, t] - r["w"][a, t_prev]) for a in assets)
            turn.append(tt)
            rows.append({"setup": r["label"], "t": int(t),
                         "turnover": float(tt)})
        ax.plot(T_vals, turn, color=r["color"], lw=1.3,
                label=f"{r['label']} (total = {sum(turn):.3f})")
    ax.set_title("Turnover por periodo: sum_i |w(i,t) - w(i,t-1)|  "
                 "(t=1 incluye rebalanceo inicial w0 -> w(t1))")
    ax.set_xlabel("t")
    ax.set_ylabel("turnover")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    save_fig(fig, "2_turnover_compare", SUBDIR)
    save_csv(pd.DataFrame(rows), "2_turnover_compare", SUBDIR)


# ================================================================
# 3) Costos acumulados de rebalanceo
# ================================================================

def diag_costos(results):
    assets = results[0]["ctx"]["assets"]
    T_vals = results[0]["ctx"]["T_vals"]

    rows = []
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for r in results:
        c_base = r["ctx"]["c_base"]
        acumulado = 0.0
        curve = []
        for t in T_vals:
            # Costo del rebalanceo HACIA w(t): c_base[i] * (u[i,t] + v[i,t]).
            cost_t = sum(c_base[a] * (r["u"][a, t] + r["v"][a, t]) for a in assets)
            acumulado += cost_t
            curve.append(acumulado)
            rows.append({"setup": r["label"], "t": int(t),
                         "costo_t": float(cost_t),
                         "costo_acum": float(acumulado)})
        ax.plot(T_vals, curve, color=r["color"], lw=1.3,
                label=f"{r['label']} (final = {curve[-1]:.4f})")
    ax.set_title("Costos acumulados de rebalanceo: "
                 "sum_{s<=t} sum_i c_base[i] * (u[i,s] + v[i,s])  "
                 "(fraccion del capital)")
    ax.set_xlabel("t")
    ax.set_ylabel("costo acumulado")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    save_fig(fig, "3_costos_acumulados", SUBDIR)
    save_csv(pd.DataFrame(rows), "3_costos_acumulados", SUBDIR)


# ================================================================
# 4) Metricas agregadas
# ================================================================

def diag_metricas(results):
    assets = results[0]["ctx"]["assets"]
    T_vals = results[0]["ctx"]["T_vals"]

    rows = []
    for r in results:
        w0_dict = r["ctx"]["w0"]
        c_base  = r["ctx"]["c_base"]
        total_turnover = 0.0
        for idx, t in enumerate(T_vals):
            if idx == 0:
                total_turnover += sum(abs(r["w"][a, t] - w0_dict[a]) for a in assets)
            else:
                t_prev = T_vals[idx - 1]
                total_turnover += sum(abs(r["w"][a, t] - r["w"][a, t_prev])
                                      for a in assets)
        total_cost = sum(c_base[a] * (r["u"][a, t] + r["v"][a, t])
                         for a in assets for t in T_vals)
        # Estadisticas por activo
        per_asset = {}
        for a in assets:
            ws = np.array([r["w"][a, t] for t in T_vals])
            per_asset[f"w_{a}_mean"] = float(ws.mean())
            per_asset[f"w_{a}_min"]  = float(ws.min())
            per_asset[f"w_{a}_max"]  = float(ws.max())
            per_asset[f"w_{a}_std"]  = float(ws.std())
        rows.append({
            "setup": r["label"],
            "lambda": float(r["lambda"]), "m": float(r["m"]),
            "z": float(r["z"]),
            "turnover_total": float(total_turnover),
            "costo_total":    float(total_cost),
            **per_asset,
        })
    df = pd.DataFrame(rows)
    save_csv(df, "4_metricas", SUBDIR)
    print("\n--- Metricas agregadas ---")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# ================================================================
# Main
# ================================================================

def main():
    print("=" * 70)
    print("REBALANCEO EN EL TIEMPO - 3 setups")
    print("=" * 70)
    results = solve_all()
    diag_pesos(results)
    diag_turnover(results)
    diag_costos(results)
    diag_metricas(results)
    print("\n" + "=" * 70)
    print("Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
