"""mu_hat_source='p_sign': resuelve la inconsistencia de regimen del pipeline.

Diagnostico previo: la `p_hist` del CSV (HMM) tiene accuracy ~52-56% vs
(r >= 0), correlacion ~0. Como mu_hat se estima ponderando r_hist por p_hist
(PDF ec. 2), el resultado es que mu_hat(bull) ≈ mu_hat(bear) — gap de solo
~0.03% para SPX y signo INVERTIDO (bull < bear). Esto aplana mu_mix(t) sin
importar como se construya p_dl.

El LSTM en cambio define bull = (decile >= BULL_THRESHOLD) (ec. 15). Si
construyeramos mu_hat con la MISMA regla — el oraculo historico de signo —
el gap real entre regimenes es:
  SPX:  bull = +1.81%, bear = -1.86%  (gap = 3.67%, 120x mas que p_hist)
  CMC:  bull = +6.96%, bear = -7.06%  (gap = 14.02%, 24x mas)

Eso es lo que mu_hat_source='p_sign' implementa: usar el mismo regimen que
el LSTM tanto en el estimador como en la prediccion.

Experimento: tres setups (mismo grid, mismos escenarios DL, mismo T=163):

  rollout+phist   : p_method='rollout', mu_hat_source='p_hist' (default actual)
  rollout+psign   : p_method='rollout', mu_hat_source='p_sign' (solo cambia mu_hat)
  scenarios+psign : p_method='scenarios', mu_hat_source='p_sign' (ambos cambios)

Hipotesis: scenarios+psign deberia producir mu_mix(t) con variacion temporal
significativa para AMBOS activos, lo que permitiria al FO timear el mercado
en vez de degenerar a bet one-asset.

Salidas (inspeccion/mu_hat_signregime_out/):
  1_muhat_compare.csv          mu_hat por (asset, regimen) y setup
  2_mumix_compare.csv/png      trayectorias mu_mix(t)
  3_grid_compare.csv/png       g*_mean y mean_regret por setup
  4_w_compare.csv/png          politica w(t) bajo g*_mean
  5_backtest_compare.csv/png   backtest historico
  6_resumen.csv                tabla agregada
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
    REGIMES,
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


SUBDIR = "mu_hat_signregime"

SETUPS = [
    ("rollout+phist",   "rollout",   "p_hist", "#1f77b4"),
    ("rollout+psign",   "rollout",   "p_sign", "#F2B705"),
    ("scenarios+psign", "scenarios", "p_sign", "#E63946"),
]


def build_setups():
    print("Construyendo contextos...")
    ctxs = {}
    for label, p_method, mu_hat_source, _ in SETUPS:
        print(f"\n  [{label}] p_method={p_method}  mu_hat_source={mu_hat_source}")
        ctx = build_dl_context(
            data_dir=DATA_DIR, checkpoint_path=CHECKPOINT_PATH,
            T=T_HORIZON, N_candidates=N_CANDIDATES, n_scenarios=N_SCENARIOS,
            seed=SCENARIO_SEED, summary_asset=SUMMARY_ASSET,
            p_method=p_method, mu_hat_source=mu_hat_source,
        )
        ctxs[label] = ctx
        for a in ctx["assets"]:
            mu = ctx["mu_mix"][a]
            p  = ctx["p_dl"][a]["bull"]
            print(f"    {a}: mu_mix mean={mu.mean()*100:+.3f}%  "
                  f"std={mu.std()*100:.4f}%  "
                  f"(p_bull std={p.std():.3f})")
    opt_ctx = load_market_data(str(DATA_DIR))
    return {"ctxs": ctxs, "opt_ctx": opt_ctx}


# ================================================================
# 1) mu_hat por (asset, regimen) y setup
# ================================================================

def diag_muhat(state):
    rows = []
    for label, _, _, _ in SETUPS:
        ctx = state["ctxs"][label]
        for a in ctx["assets"]:
            for k in REGIMES:
                rows.append({
                    "setup": label, "asset": a, "regime": k,
                    "mu_hat": float(ctx["mu_hat"][(a, k)]),
                    "mu_hat_pct": float(ctx["mu_hat"][(a, k)] * 100),
                })
    df = pd.DataFrame(rows)
    save_csv(df, "1_muhat_compare", SUBDIR)
    print("\n--- mu_hat por (asset, regimen) ---")
    pivot = df.pivot_table(
        index=["asset", "regime"], columns="setup",
        values="mu_hat_pct", aggfunc="first",
    ).round(2)
    print(pivot.to_string())


# ================================================================
# 2) mu_mix(t) por setup
# ================================================================

def diag_mumix(state):
    ctxs = state["ctxs"]
    assets = ctxs[SETUPS[0][0]]["assets"]
    T_vals = ctxs[SETUPS[0][0]]["T_vals"]

    rows = []
    fig, axes = plt.subplots(len(assets), 1, figsize=(13, 4 * len(assets)),
                             sharex=True)
    if len(assets) == 1:
        axes = [axes]
    for ai, a in enumerate(assets):
        ax = axes[ai]
        for label, _, _, color in SETUPS:
            mu = ctxs[label]["mu_mix"][a]
            ax.plot(T_vals, mu.values * 100, color=color, lw=1.3,
                    label=f"{label} (mean={mu.mean()*100:+.2f}%, "
                          f"std={mu.std()*100:.3f}%)")
            for t, v in zip(T_vals, mu.values):
                rows.append({"asset": a, "setup": label,
                             "t": int(t), "mu_mix_pct": float(v * 100)})
        ax.axhline(0, color="grey", lw=0.6)
        ax.set_title(f"mu_mix({a}, t) por setup (en %/semana)")
        ax.set_xlabel("t")
        ax.set_ylabel(f"mu_mix({a})  [%/sem]")
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
    for label, _, _, _ in SETUPS:
        print(f"\n--- Grilla {len(lambda_grid)}x{len(m_grid)} solves en setup={label} ---")
        V_df, pol = run_regret_grid(state["ctxs"][label], lambda_grid, m_grid)
        res = compute_regret_and_select(V_df)
        grids[label] = {"V_df": V_df, "policies": pol, "res": res}
    return grids


def diag_grid(state, grids):
    rows = []
    for label, _, _, _ in SETUPS:
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

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, (label, _, _, _) in zip(axes, SETUPS):
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
        ax.set_title(f"setup={label}\n"
                     f"g*_mean=({lam_m:.2f},{m_m:.1f})  "
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
    assets = ctxs[SETUPS[0][0]]["assets"]
    T_vals = ctxs[SETUPS[0][0]]["T_vals"]

    rows = []
    fig, axes = plt.subplots(len(assets), 1, figsize=(13, 4 * len(assets)),
                             sharex=True)
    if len(assets) == 1:
        axes = [axes]
    for ai, a in enumerate(assets):
        ax = axes[ai]
        ax.axhline(0.5, color="#666", ls="--", lw=0.8, label="w0 = 0.5")
        for label, _, _, color in SETUPS:
            res = grids[label]["res"]
            pol = grids[label]["policies"]
            lam_m, m_m = res["g_mean"]
            w_sol, _, _, _ = pol[(lam_m, m_m)]
            w_t = [w_sol[a, t] for t in T_vals]
            ax.plot(T_vals, w_t, color=color, lw=1.5,
                    label=f"{label} g*=({lam_m:.2f},{m_m:.1f}), "
                          f"mean={np.mean(w_t):.3f}  std={np.std(w_t):.3f}")
            for t, w in zip(T_vals, w_t):
                rows.append({"asset": a, "setup": label,
                             "t": int(t), "w": float(w)})
        ax.set_title(f"w({a}, t) bajo g*_mean por setup")
        ax.set_xlabel("t")
        ax.set_ylabel(f"w({a})")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=9)
    save_fig(fig, "4_w_compare", SUBDIR)
    save_csv(pd.DataFrame(rows), "4_w_compare", SUBDIR)


# ================================================================
# 5) Backtest historico
# ================================================================

def diag_backtest(state, grids):
    opt_ctx = state["opt_ctx"]
    print("\nResolviendo OPT base para backtest...")
    _, w_opt, u_opt, v_opt, _ = solve_portfolio(
        opt_ctx, lambda_riesgo=1.00, costo_mult=1.0,
    )

    rows = []
    fig, ax = plt.subplots(figsize=(12, 5))
    for label, _, _, color in SETUPS:
        ctx_dl = state["ctxs"][label]
        res = grids[label]["res"]
        pol = grids[label]["policies"]
        lam_m, m_m = res["g_mean"]
        w_rg, u_rg, v_rg, _ = pol[(lam_m, m_m)]
        cap_rg = simulate_capital_opt(w_rg, u_rg, v_rg, ctx_dl)
        T_vals = ctx_dl["T_vals"]
        ax.plot(T_vals, [cap_rg[t] for t in T_vals], color=color, lw=1.5,
                label=f"RG {label} (lam={lam_m:.2f}, m={m_m:.1f})  "
                      f"fin=${cap_rg[T_vals[-1]]:,.0f}")
        for t in T_vals:
            rows.append({"setup": label, "t": int(t),
                         "RG_cap": float(cap_rg[t])})

    ctx0 = state["ctxs"][SETUPS[0][0]]
    T0 = ctx0["T_vals"]; C0 = ctx0["Capital_inicial"]
    cap_opt = simulate_capital_opt(w_opt, u_opt, v_opt, ctx0)
    cap_rb  = simulate_naive_rb(ctx0)
    cap_bh  = simulate_naive_bh(ctx0)
    ax.plot(T0, [cap_opt[t] for t in T0], color="#888", lw=1.5,
            label=f"OPT base  fin=${cap_opt[T0[-1]]:,.0f}")
    ax.plot(T0, [cap_rb [t] for t in T0], color="#8B3A1F", lw=1.0,
            label=f"Naive 50/50 rebal  fin=${cap_rb[T0[-1]]:,.0f}")
    ax.plot(T0, [cap_bh [t] for t in T0], color="#aaa", lw=1.0, ls=":",
            label=f"Naive 50/50 B&H  fin=${cap_bh[T0[-1]]:,.0f}")
    ax.axhline(C0, color="#666", ls="--", lw=0.8, label=f"C0=${C0:,.0f}")
    ax.set_title("Backtest historico (diagnostico, fuera del PDF)")
    ax.set_xlabel("t"); ax.set_ylabel("Capital")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    save_fig(fig, "5_backtest_compare", SUBDIR)
    save_csv(pd.DataFrame(rows), "5_backtest_compare", SUBDIR)

    final_rows = []
    for label, _, _, _ in SETUPS:
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
    final_rows.append({"setup": "OPT_base", "lambda": 1.00, "m": 1.0,
                       "RG_cap_final": float(cap_opt[T0[-1]]),
                       "RG_ret_acum":  float(cap_opt[T0[-1]]/C0 - 1)})
    final_rows.append({"setup": "Naive_rb", "lambda": np.nan, "m": np.nan,
                       "RG_cap_final": float(cap_rb[T0[-1]]),
                       "RG_ret_acum":  float(cap_rb[T0[-1]]/C0 - 1)})
    final_rows.append({"setup": "Naive_bh", "lambda": np.nan, "m": np.nan,
                       "RG_cap_final": float(cap_bh[T0[-1]]),
                       "RG_ret_acum":  float(cap_bh[T0[-1]]/C0 - 1)})
    save_csv(pd.DataFrame(final_rows), "6_resumen", SUBDIR)


# ================================================================
# Main
# ================================================================

def main():
    print("=" * 70)
    print("MU_HAT_SOURCE: p_hist vs p_sign  (regime consistency)")
    print("=" * 70)
    state = build_setups()
    diag_muhat(state)
    diag_mumix(state)
    grids = run_grids(state)
    diag_grid(state, grids)
    diag_w(state, grids)
    diag_backtest(state, grids)

    print("\n" + "=" * 70)
    print("RESUMEN")
    print("=" * 70)
    for label, _, _, _ in SETUPS:
        res = grids[label]["res"]
        lam_m, m_m = res["g_mean"]
        ctx = state["ctxs"][label]
        muSPX = ctx["mu_mix"]["SPX"]
        muCMC = ctx["mu_mix"]["CMC200"]
        print(f"  [{label}]")
        print(f"    mu_mix(SPX): mean={muSPX.mean()*100:+.2f}%/sem  "
              f"std={muSPX.std()*100:.3f}%")
        print(f"    mu_mix(CMC): mean={muCMC.mean()*100:+.2f}%/sem  "
              f"std={muCMC.std()*100:.3f}%")
        print(f"    g*_mean=(lam={lam_m:.2f}, m={m_m:.1f})  "
              f"mean_regret=${res['g_mean_metric']:,.0f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
