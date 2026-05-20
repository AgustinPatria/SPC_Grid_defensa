"""Test: usar `mu_hat_source='p_hist'` (en vez de 'p_sign') sobre data sintetica.

Hipotesis del usuario: usar p_hist para estimar mu_hat puede ser una mejor
opcion. La probamos sobre la data sintetica (donde p_hist fue construido
explicitamente como `Phi(mu(t)/sigma)`, alineado con la senal sinusoidal).

Notar: en data REAL el p_hist es HMM externo desalineado (accuracy ~52%
vs signo de r), pero en data SINTETICA el p_hist SI esta alineado con
el signo de r por construccion.

Setup: tres contextos sobre la MISMA data sintetica, mismo p_method=walking
(que ya validamos que funciona), variando solo el mu_hat_source:
  A. p_sign  (oraculo binario, default actual)
  B. p_hist  (probabilidades suaves del CSV sintetico)
  C. p_dl    (predicciones del LSTM, legacy)

Outputs (inspeccion/synthetic_phist_test_out/):
  P1_muhat.csv                  mu_hat por (asset, regime, source)
  P2_mumix_compare.csv/png      trayectorias mu_mix(t) por source
  P3_grid.csv                   resumen regret-grid por source
  P4_w_policy.csv/png           politicas w(t) por source
  P5_backtest.csv/png           capital backtest por source + baselines
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


SUBDIR = "synthetic_phist_test"
SYN_DATA_DIR = _THIS_DIR / "synthetic_experiment_out" / "data"
SYN_CKPT = _THIS_DIR / "synthetic_experiment_out" / "models" / "lstm_synthetic.pt"

SETUPS = [
    ("p_sign", "#1f77b4"),
    ("p_hist", "#E63946"),
    ("p_dl",   "#F2B705"),
]


def main():
    print("=" * 70)
    print("SINTETICO: comparacion mu_hat con p_sign vs p_hist vs p_dl")
    print("=" * 70)
    if not SYN_DATA_DIR.exists() or not SYN_CKPT.exists():
        raise SystemExit("Primero corre inspeccion/synthetic_experiment.py")

    # Construir los 3 contextos
    states = {}
    for src, _ in SETUPS:
        print(f"\nConstruyendo ctx mu_hat_source='{src}'...")
        ctx = build_dl_context(
            data_dir=SYN_DATA_DIR, checkpoint_path=SYN_CKPT,
            T=T_HORIZON, N_candidates=N_CANDIDATES, n_scenarios=N_SCENARIOS,
            seed=SCENARIO_SEED, summary_asset=SUMMARY_ASSET,
            p_method="walking", mu_hat_source=src,
        )
        states[src] = {"ctx": ctx}
        for a in ctx["assets"]:
            for k in REGIMES:
                print(f"  mu_hat[({a}, {k})] = {ctx['mu_hat'][(a, k)]*100:+.3f}%/sem")
            mu = ctx["mu_mix"][a]
            print(f"  mu_mix({a}): mean={mu.mean()*100:+.3f}%, std={mu.std()*100:.4f}%")

    opt_ctx = load_market_data(str(SYN_DATA_DIR))
    assets = states["p_sign"]["ctx"]["assets"]
    T_vals = states["p_sign"]["ctx"]["T_vals"]

    # --- P1: tabla mu_hat ---
    rows = []
    for src, _ in SETUPS:
        for a in assets:
            for k in REGIMES:
                rows.append({
                    "source": src, "asset": a, "regime": k,
                    "mu_hat_pct": float(states[src]["ctx"]["mu_hat"][(a, k)] * 100),
                })
    df = pd.DataFrame(rows)
    save_csv(df, "P1_muhat", SUBDIR)
    pivot = df.pivot_table(index=["asset", "regime"], columns="source",
                           values="mu_hat_pct", aggfunc="first").round(3)
    print("\n--- mu_hat por (asset, regime) ---")
    print(pivot.to_string())

    # --- P2: mu_mix trayectorias ---
    rows = []
    fig, axes = plt.subplots(len(assets), 1, figsize=(13, 4 * len(assets)),
                             sharex=True)
    if len(assets) == 1:
        axes = [axes]
    for ai, a in enumerate(assets):
        ax = axes[ai]
        for src, color in SETUPS:
            mu = states[src]["ctx"]["mu_mix"][a]
            ax.plot(T_vals, mu.values * 100, color=color, lw=1.4,
                    label=f"{src} (mean={mu.mean()*100:+.2f}%, std={mu.std()*100:.3f}%)")
            for t, v in zip(T_vals, mu.values):
                rows.append({"asset": a, "source": src,
                             "t": int(t), "mu_mix_pct": float(v * 100)})
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_title(f"mu_mix({a}, t) — sintetica, walking p_dl, distintos mu_hat_source")
        ax.set_xlabel("t"); ax.set_ylabel(f"mu_mix({a}) [%/sem]")
        ax.grid(True, alpha=0.25); ax.legend(fontsize=9, loc="best")
    save_fig(fig, "P2_mumix_compare", SUBDIR)
    save_csv(pd.DataFrame(rows), "P2_mumix_compare", SUBDIR)

    # --- P3: regret grid por source ---
    print("\n--- Grid por mu_hat source ---")
    rows = []
    for src, _ in SETUPS:
        print(f"\nGrilla {len(LAMBDA_GRID)}x{len(M_GRID)} en mu_hat_source='{src}'...")
        V_df, pol = run_regret_grid(states[src]["ctx"], list(LAMBDA_GRID), list(M_GRID))
        res = compute_regret_and_select(V_df)
        states[src]["V_df"] = V_df
        states[src]["policies"] = pol
        states[src]["res"] = res
        lam_m, m_m = res["g_mean"]
        V_row = res["V_table"].loc[(lam_m, m_m)]
        C0 = states[src]["ctx"]["Capital_inicial"]
        rows.append({
            "source": src, "g_mean_lambda": float(lam_m), "g_mean_m": float(m_m),
            "regret": float(res["g_mean_metric"]),
            "V_mean": float(V_row.mean()),
            "V_min":  float(V_row.min()),
            "V_max":  float(V_row.max()),
            "ret_avg": float(V_row.mean() / C0 - 1),
        })
        print(f"  g*=({lam_m:.2f}, {m_m:.1f})  regret=${res['g_mean_metric']:,.2f}  "
              f"V mean=${V_row.mean():,.0f}  range=[${V_row.min():,.0f}, ${V_row.max():,.0f}]")
    save_csv(pd.DataFrame(rows), "P3_grid", SUBDIR)

    # --- P4: politicas w(t) ---
    rows = []
    fig, axes = plt.subplots(len(assets), 1, figsize=(13, 4 * len(assets)),
                             sharex=True)
    if len(assets) == 1:
        axes = [axes]
    for ai, a in enumerate(assets):
        ax = axes[ai]
        ax.axhline(0.5, color="grey", ls="--", lw=0.6, label="w0=0.5")
        for src, color in SETUPS:
            res = states[src]["res"]
            pol = states[src]["policies"]
            lam_m, m_m = res["g_mean"]
            w_sol = pol[(lam_m, m_m)][0]
            w_t = [w_sol[(a, t)] for t in T_vals]
            ax.plot(T_vals, w_t, color=color, lw=1.5,
                    label=f"{src} g*=({lam_m:.2f},{m_m:.1f}) "
                          f"mean={np.mean(w_t):.3f} std={np.std(w_t):.3f}")
            for t, ww in zip(T_vals, w_t):
                rows.append({"asset": a, "source": src,
                             "t": int(t), "w": float(ww)})
        ax.set_title(f"Politica w({a}, t) por mu_hat source [sintetica + walking]")
        ax.set_xlabel("t"); ax.set_ylabel(f"w({a})")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.25); ax.legend(fontsize=9, loc="best")
    save_fig(fig, "P4_w_policy", SUBDIR)
    save_csv(pd.DataFrame(rows), "P4_w_policy", SUBDIR)

    # --- P5: backtest sobre data sintetica ---
    _, w_opt, u_opt, v_opt, _ = solve_portfolio(
        opt_ctx, lambda_riesgo=1.00, costo_mult=1.0,
    )
    ctx0 = states["p_sign"]["ctx"]
    T_vals = ctx0["T_vals"]; C0 = ctx0["Capital_inicial"]; t_f = T_vals[-1]
    cap_opt = simulate_capital_opt(w_opt, u_opt, v_opt, ctx0)
    cap_rb  = simulate_naive_rb(ctx0)
    cap_bh  = simulate_naive_bh(ctx0)

    rows = [{"setup": "OPT_base",   "cap_final": float(cap_opt[t_f]),
             "ret_acum": float(cap_opt[t_f]/C0 - 1)},
            {"setup": "Naive_RB",   "cap_final": float(cap_rb [t_f]),
             "ret_acum": float(cap_rb [t_f]/C0 - 1)},
            {"setup": "Naive_BH",   "cap_final": float(cap_bh [t_f]),
             "ret_acum": float(cap_bh [t_f]/C0 - 1)}]

    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.plot(T_vals, [cap_opt[t] for t in T_vals], color="#F2B705", lw=1.6, ls=":",
            label=f"OPT base (sin LSTM) fin=${cap_opt[t_f]:,.0f}  ret={cap_opt[t_f]/C0-1:+.1%}")
    ax.plot(T_vals, [cap_rb [t] for t in T_vals], color="#8B3A1F", lw=1.2,
            label=f"Naive RB  fin=${cap_rb[t_f]:,.0f}  ret={cap_rb[t_f]/C0-1:+.1%}")
    ax.plot(T_vals, [cap_bh [t] for t in T_vals], color="#aaa", lw=1.0, ls=":",
            label=f"Naive BH  fin=${cap_bh[t_f]:,.0f}  ret={cap_bh[t_f]/C0-1:+.1%}")
    for src, color in SETUPS:
        res = states[src]["res"]
        pol = states[src]["policies"]
        lam_m, m_m = res["g_mean"]
        w_rg, u_rg, v_rg, _ = pol[(lam_m, m_m)]
        cap_rg = simulate_capital_opt(w_rg, u_rg, v_rg, ctx0)
        ax.plot(T_vals, [cap_rg[t] for t in T_vals], color=color, lw=1.7,
                label=f"RG mu_hat={src} (lam={lam_m:.2f}, m={m_m:.1f})  "
                      f"fin=${cap_rg[t_f]:,.0f}  ret={cap_rg[t_f]/C0-1:+.1%}")
        rows.append({"setup": f"RG_{src}", "cap_final": float(cap_rg[t_f]),
                     "ret_acum": float(cap_rg[t_f]/C0 - 1)})
    ax.axhline(C0, color="#666", ls="--", lw=0.8, label=f"C0=${C0:,.0f}")
    ax.set_title("Backtest sintetico: RG con distintos mu_hat_source")
    ax.set_xlabel("t"); ax.set_ylabel("Capital")
    ax.grid(True, alpha=0.25); ax.legend(fontsize=8, loc="best")
    save_fig(fig, "P5_backtest", SUBDIR)
    save_csv(pd.DataFrame(rows), "P5_backtest", SUBDIR)

    print("\n" + "=" * 70)
    print("RESUMEN BACKTEST")
    print("=" * 70)
    for r in rows:
        print(f"  {r['setup']:<14} cap_final=${r['cap_final']:>10,.2f}  "
              f"ret={r['ret_acum']:+.2%}")


if __name__ == "__main__":
    main()
