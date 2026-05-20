"""Rollout vs Scenarios: comparacion del metodo de construccion de p_dl(t).

Motivacion: el rollout deterministico (default actual) avanza la ventana con
el quintil mediano y colapsa rapido a un punto fijo => mu_mix(t) queda casi
constante => el FO se comporta como problema estatico. La opcion 'scenarios'
deriva p_dl(t, A) = (1/N) Σ_s 1{candidates[s, t, A] >= 0}, usando los mismos
N escenarios que despues se reducen a los 5 representativos. Esto inyecta
variabilidad temporal proveniente del muestreo estocastico de quintiles en
cada paso del rollout.

Setup (ambos con mu_hat_source='p_hist' para aislar el efecto de p_method):
  - rollout    : p_method='rollout'    (current default)
  - scenarios  : p_method='scenarios'  (option 2 del analisis)

Salidas (inspeccion/p_method_scenarios_out/):
  1_pbull_compare.csv/png     trayectorias p_bull(t) por activo
  2_mumix_compare.csv/png     trayectorias mu_mix(t) por activo
  3_grid_compare.csv/png      g*_mean y mean_regret por setup
  4_w_compare.csv/png         trayectoria w(t) bajo g*_mean por setup
  5_backtest_compare.csv/png  backtest historico de cada setup
  6_resumen.csv               numeros agregados

Corre con:
    python -m inspeccion.p_method_scenarios
    (o)
    python inspeccion/p_method_scenarios.py
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
    LAMBDA_GRID,
    M_GRID,
    N_CANDIDATES,
    N_SCENARIOS,
    SCENARIO_SEED,
    SUMMARY_ASSET,
    T_HORIZON,
)
from Regret_Grid import (
    build_dl_context,
    compute_regret_and_select,
    load_market_data,
    run_regret_grid,
    simulate_capital_opt,
    simulate_naive_bh,
    simulate_naive_rb,
    solve_portfolio,
)


SUBDIR = "p_method_scenarios"

SETUPS = [
    ("rollout",   "rollout",   "#1f77b4"),
    ("scenarios", "scenarios", "#E63946"),
]


def build_setups():
    print("Construyendo contextos con mu_hat_source='p_hist'...")
    ctxs = {}
    for label, p_method, _ in SETUPS:
        print(f"\n  [{label}] p_method={p_method}")
        ctx = build_dl_context(
            data_dir=DATA_DIR, checkpoint_path=CHECKPOINT_PATH,
            T=T_HORIZON, N_candidates=N_CANDIDATES, n_scenarios=N_SCENARIOS,
            seed=SCENARIO_SEED, summary_asset=SUMMARY_ASSET,
            p_method=p_method, mu_hat_source="p_hist",
        )
        ctxs[label] = ctx
        for a in ctx["assets"]:
            p = ctx["p_dl"][a]["bull"]
            mu = ctx["mu_mix"][a]
            print(f"    {a}: p_bull mean={p.mean():.3f} std={p.std():.3f}  "
                  f"min={p.min():.3f} max={p.max():.3f}  "
                  f"mu_mix std={mu.std():.6f}")
    opt_ctx = load_market_data(str(DATA_DIR))
    return {"ctxs": ctxs, "opt_ctx": opt_ctx}


# ================================================================
# 1) p_bull(t) por setup
# ================================================================

def diag_pbull(state):
    ctxs = state["ctxs"]
    assets = ctxs["rollout"]["assets"]
    T_vals = ctxs["rollout"]["T_vals"]

    rows = []
    fig, axes = plt.subplots(len(assets), 1, figsize=(12, 4 * len(assets)),
                             sharex=True)
    if len(assets) == 1:
        axes = [axes]
    for ai, a in enumerate(assets):
        ax = axes[ai]
        for label, _, color in SETUPS:
            p = ctxs[label]["p_dl"][a]["bull"]
            ax.plot(T_vals, p.values, color=color, lw=1.3,
                    label=f"{label} (mean={p.mean():.3f}, std={p.std():.3f})")
            for t, v in zip(T_vals, p.values):
                rows.append({"asset": a, "setup": label,
                             "t": int(t), "p_bull": float(v)})
        ax.axhline(0.5, color="grey", ls="--", lw=0.6)
        ax.set_title(f"p_bull({a}, t) — rollout vs scenarios")
        ax.set_xlabel("t")
        ax.set_ylabel(f"p_bull({a})")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=9)
    save_fig(fig, "1_pbull_compare", SUBDIR)
    save_csv(pd.DataFrame(rows), "1_pbull_compare", SUBDIR)


# ================================================================
# 2) mu_mix(t) por setup
# ================================================================

def diag_mumix(state):
    ctxs = state["ctxs"]
    assets = ctxs["rollout"]["assets"]
    T_vals = ctxs["rollout"]["T_vals"]

    rows = []
    fig, axes = plt.subplots(len(assets), 1, figsize=(12, 4 * len(assets)),
                             sharex=True)
    if len(assets) == 1:
        axes = [axes]
    for ai, a in enumerate(assets):
        ax = axes[ai]
        for label, _, color in SETUPS:
            mu = ctxs[label]["mu_mix"][a]
            ax.plot(T_vals, mu.values, color=color, lw=1.3,
                    label=f"{label} (mean={mu.mean():.4f}, std={mu.std():.6f})")
            for t, v in zip(T_vals, mu.values):
                rows.append({"asset": a, "setup": label,
                             "t": int(t), "mu_mix": float(v)})
        ax.axhline(0, color="grey", lw=0.6)
        ax.set_title(f"mu_mix({a}, t) — rollout vs scenarios")
        ax.set_xlabel("t")
        ax.set_ylabel(f"mu_mix({a})")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=9)
    save_fig(fig, "2_mumix_compare", SUBDIR)
    save_csv(pd.DataFrame(rows), "2_mumix_compare", SUBDIR)


# ================================================================
# 3) Regret grid por setup
# ================================================================

def run_grids(state):
    lambda_grid = list(LAMBDA_GRID)
    m_grid = list(M_GRID)
    grids = {}
    for label, _, _ in SETUPS:
        print(f"\n--- Grilla {len(lambda_grid)}x{len(m_grid)} solves en setup={label} ---")
        V_df, pol = run_regret_grid(state["ctxs"][label], lambda_grid, m_grid)
        res = compute_regret_and_select(V_df)
        grids[label] = {"V_df": V_df, "policies": pol, "res": res}
    return grids


def diag_grid(state, grids):
    rows = []
    for label, _, _ in SETUPS:
        res = grids[label]["res"]
        lam_m, m_m = res["g_mean"]
        lam_w, m_w = res["g_worst"]
        V_mean_row = res["V_table"].loc[(lam_m, m_m)]
        C0 = state["ctxs"][label]["Capital_inicial"]
        rows.append({
            "setup": label,
            "g_mean_lambda":  float(lam_m), "g_mean_m": float(m_m),
            "g_mean_regret":  float(res["g_mean_metric"]),
            "g_mean_V_avg":   float(V_mean_row.mean()),
            "g_mean_V_min":   float(V_mean_row.min()),
            "g_mean_V_max":   float(V_mean_row.max()),
            "g_mean_ret_avg": float(V_mean_row.mean()/C0 - 1),
            "g_worst_lambda": float(lam_w), "g_worst_m": float(m_w),
            "g_worst_regret": float(res["g_worst_metric"]),
        })
    save_csv(pd.DataFrame(rows), "3_grid_compare", SUBDIR)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (label, _, _) in zip(axes, SETUPS):
        res = grids[label]["res"]
        R_tab = res["R_table"]
        mean_R = R_tab.mean(axis=1).unstack("m").sort_index().sort_index(axis=1)
        ax.imshow(mean_R.values, cmap="RdYlGn_r", aspect="auto")
        ax.set_xticks(range(len(mean_R.columns)))
        ax.set_xticklabels([f"{m:.1f}" for m in mean_R.columns])
        ax.set_yticks(range(len(mean_R.index)))
        ax.set_yticklabels([f"{l:.2f}" for l in mean_R.index])
        ax.set_xlabel("m"); ax.set_ylabel("lambda")
        lam_m, m_m = res["g_mean"]
        ax.set_title(f"setup={label}  g*_mean=({lam_m:.2f},{m_m:.1f})\n"
                     f"regret=${res['g_mean_metric']:,.0f}")
        for i in range(mean_R.shape[0]):
            for j in range(mean_R.shape[1]):
                ax.text(j, i, f"${mean_R.values[i, j]:,.0f}",
                        ha="center", va="center", fontsize=7)
        j_opt = list(mean_R.columns).index(m_m)
        i_opt = list(mean_R.index).index(lam_m)
        ax.scatter([j_opt], [i_opt], s=200, marker="*", color="cyan",
                   edgecolor="k", linewidth=1.5, zorder=3)
    save_fig(fig, "3_grid_compare", SUBDIR)


# ================================================================
# 4) Politica w(t) bajo g*_mean por setup
# ================================================================

def diag_w(state, grids):
    ctxs = state["ctxs"]
    assets = ctxs["rollout"]["assets"]
    T_vals = ctxs["rollout"]["T_vals"]

    rows = []
    fig, axes = plt.subplots(len(assets), 1, figsize=(12, 4 * len(assets)),
                             sharex=True)
    if len(assets) == 1:
        axes = [axes]
    for ai, a in enumerate(assets):
        ax = axes[ai]
        ax.axhline(0.5, color="#666", ls="--", lw=0.8, label="w0 = 0.5")
        for label, _, color in SETUPS:
            res = grids[label]["res"]
            pol = grids[label]["policies"]
            lam_m, m_m = res["g_mean"]
            w_sol, _, _, _ = pol[(lam_m, m_m)]
            w_t = [w_sol[a, t] for t in T_vals]
            ax.plot(T_vals, w_t, color=color, lw=1.5,
                    label=f"{label} g*=({lam_m:.2f},{m_m:.1f}), "
                          f"mean={np.mean(w_t):.3f}, std={np.std(w_t):.3f}")
            for t, w in zip(T_vals, w_t):
                rows.append({"asset": a, "setup": label,
                             "t": int(t), "w": float(w)})
        ax.set_title(f"w({a}, t) bajo g*_mean — rollout vs scenarios")
        ax.set_xlabel("t")
        ax.set_ylabel(f"w({a})")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=9)
    save_fig(fig, "4_w_compare", SUBDIR)
    save_csv(pd.DataFrame(rows), "4_w_compare", SUBDIR)


# ================================================================
# 5) Backtest historico (diagnostico)
# ================================================================

def diag_backtest(state, grids):
    opt_ctx = state["opt_ctx"]
    rows = []
    print("\nResolviendo OPT base para backtest...")
    _, w_opt, u_opt, v_opt, _ = solve_portfolio(
        opt_ctx, lambda_riesgo=1.00, costo_mult=1.0,
    )

    fig, ax = plt.subplots(figsize=(11, 5))
    for label, _, color in SETUPS:
        ctx_dl = state["ctxs"][label]
        res = grids[label]["res"]
        pol = grids[label]["policies"]
        lam_m, m_m = res["g_mean"]
        w_rg, u_rg, v_rg, _ = pol[(lam_m, m_m)]
        cap_rg  = simulate_capital_opt(w_rg, u_rg, v_rg, ctx_dl)
        T_vals = ctx_dl["T_vals"]
        ax.plot(T_vals, [cap_rg[t] for t in T_vals], color=color, lw=1.5,
                label=f"RG g*_mean {label} (lam={lam_m:.2f}, m={m_m:.1f})  "
                      f"fin=${cap_rg[T_vals[-1]]:,.0f}")
        for t in T_vals:
            rows.append({"setup": label, "t": int(t),
                         "RG_cap": float(cap_rg[t])})

    # Comunes (mismo r_hist en todos los setups DL):
    ctx0 = state["ctxs"]["rollout"]
    T0 = ctx0["T_vals"]; C0 = ctx0["Capital_inicial"]
    cap_opt = simulate_capital_opt(w_opt, u_opt, v_opt, ctx0)
    cap_rb  = simulate_naive_rb(ctx0)
    cap_bh  = simulate_naive_bh(ctx0)
    ax.plot(T0, [cap_opt[t] for t in T0], color="#F2B705", lw=1.6,
            label=f"OPT base (lam=1.0, m=1.0)  fin=${cap_opt[T0[-1]]:,.0f}")
    ax.plot(T0, [cap_rb [t] for t in T0], color="#8B3A1F", lw=1.0,
            label=f"Naive 50/50 rebal  fin=${cap_rb[T0[-1]]:,.0f}")
    ax.plot(T0, [cap_bh [t] for t in T0], color="#999",    lw=1.0,
            label=f"Naive 50/50 B&H  fin=${cap_bh[T0[-1]]:,.0f}")
    ax.axhline(C0, color="#666", ls="--", lw=0.8, label=f"C0=${C0:,.0f}")
    ax.set_title("Backtest historico (diagnostico, fuera del PDF)")
    ax.set_xlabel("t"); ax.set_ylabel("Capital")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    save_fig(fig, "5_backtest_compare", SUBDIR)
    save_csv(pd.DataFrame(rows), "5_backtest_compare", SUBDIR)

    # Tabla final
    final_rows = []
    for label, _, _ in SETUPS:
        ctx_dl = state["ctxs"][label]
        res = grids[label]["res"]
        pol = grids[label]["policies"]
        lam_m, m_m = res["g_mean"]
        w_rg, u_rg, v_rg, _ = pol[(lam_m, m_m)]
        cap = simulate_capital_opt(w_rg, u_rg, v_rg, ctx_dl)
        T = ctx_dl["T_vals"]
        final_rows.append({
            "setup": label, "lambda": float(lam_m), "m": float(m_m),
            "RG_cap_final": float(cap[T[-1]]),
            "RG_ret_acum":  float(cap[T[-1]]/ctx_dl["Capital_inicial"] - 1),
        })
    final_rows.append({
        "setup": "OPT_base", "lambda": 1.00, "m": 1.0,
        "RG_cap_final": float(cap_opt[T0[-1]]),
        "RG_ret_acum":  float(cap_opt[T0[-1]]/C0 - 1),
    })
    final_rows.append({
        "setup": "Naive_rb", "lambda": np.nan, "m": np.nan,
        "RG_cap_final": float(cap_rb[T0[-1]]),
        "RG_ret_acum":  float(cap_rb[T0[-1]]/C0 - 1),
    })
    final_rows.append({
        "setup": "Naive_bh", "lambda": np.nan, "m": np.nan,
        "RG_cap_final": float(cap_bh[T0[-1]]),
        "RG_ret_acum":  float(cap_bh[T0[-1]]/C0 - 1),
    })
    save_csv(pd.DataFrame(final_rows), "6_resumen", SUBDIR)


# ================================================================
# Main
# ================================================================

def main():
    print("=" * 70)
    print("P_METHOD: ROLLOUT vs SCENARIOS")
    print("=" * 70)
    state = build_setups()
    diag_pbull(state)
    diag_mumix(state)
    grids = run_grids(state)
    diag_grid(state, grids)
    diag_w(state, grids)
    diag_backtest(state, grids)

    print("\n" + "=" * 70)
    print("RESUMEN")
    print("=" * 70)
    for label, _, _ in SETUPS:
        res = grids[label]["res"]
        lam_m, m_m = res["g_mean"]
        ctx = state["ctxs"][label]
        for a in ctx["assets"]:
            mu = ctx["mu_mix"][a]
            print(f"  [{label}] mu_mix({a}) mean={mu.mean():.4f} "
                  f"std={mu.std():.6f}  (max-min={mu.max()-mu.min():.4f})")
        print(f"  [{label}] g*_mean=(lam={lam_m:.2f}, m={m_m:.1f})  "
              f"mean_regret=${res['g_mean_metric']:,.0f}\n")
    print("=" * 70)


if __name__ == "__main__":
    main()
