"""Test: la data sintetica con p_method='walking' captura la senal?

Hipotesis: el rollout deterministico colapsa por feedback loop con su propia
mediana. Walking aplica la LSTM a ventanas REALES en cada paso => no hay
feedback de medianas auto-generadas => deberia capturar la senal sinusoidal
de la data sintetica.

Setup:
- Reusa la data sintetica generada en inspeccion/synthetic_experiment_out/data/
- Reusa el LSTM entrenado en inspeccion/synthetic_experiment_out/models/
- Construye 2 contextos: rollout (= experimento previo) vs walking (nuevo)
- Compara p_dl, mu_mix, w(t), grid, backtest

Outputs (inspeccion/synthetic_walking_test_out/):
  W1_pdl_compare.csv/png       p_dl(t) rollout vs walking vs p_hist_real
  W2_mumix_compare.csv/png     mu_mix(t)
  W3_w_policy.csv/png          politica w(t) bajo cada setup
  W4_grid_compare.csv          regret-grid resumen por setup
  W5_backtest.csv/png          backtest sobre data sintetica
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


SUBDIR = "synthetic_walking_test"
SYN_DATA_DIR = _THIS_DIR / "synthetic_experiment_out" / "data"
SYN_CKPT = _THIS_DIR / "synthetic_experiment_out" / "models" / "lstm_synthetic.pt"

SETUPS = [
    ("rollout",  "rollout",  "#1f77b4"),
    ("walking",  "walking",  "#E63946"),
]


def main():
    print("=" * 70)
    print("SINTETICO: rollout vs walking")
    print("=" * 70)
    if not SYN_DATA_DIR.exists():
        raise SystemExit(f"Falta data sintetica en {SYN_DATA_DIR}. "
                         f"Corre primero inspeccion/synthetic_experiment.py")
    if not SYN_CKPT.exists():
        raise SystemExit(f"Falta checkpoint en {SYN_CKPT}.")

    # Construyo contextos en ambos modos
    states = {}
    for label, p_method, _ in SETUPS:
        print(f"\nConstruyendo ctx '{label}' (p_method={p_method})...")
        ctx = build_dl_context(
            data_dir=SYN_DATA_DIR, checkpoint_path=SYN_CKPT,
            T=T_HORIZON, N_candidates=N_CANDIDATES, n_scenarios=N_SCENARIOS,
            seed=SCENARIO_SEED, summary_asset=SUMMARY_ASSET,
            p_method=p_method, mu_hat_source="p_sign",
        )
        for a in ctx["assets"]:
            p = ctx["p_dl"][a]["bull"]
            mu = ctx["mu_mix"][a]
            print(f"  {a}: p_bull mean={p.mean():.3f}, std={p.std():.3f}, "
                  f"range [{p.min():.3f}, {p.max():.3f}]")
            print(f"        mu_mix mean={mu.mean()*100:+.3f}%, "
                  f"std={mu.std()*100:.4f}%")
        states[label] = {"ctx": ctx}

    opt_ctx = load_market_data(str(SYN_DATA_DIR))

    # p_hist sintetico (la verdad)
    p_hist_real_spx = opt_ctx["p_hist"]["SPX"]["bull"]
    p_hist_real_cmc = opt_ctx["p_hist"]["CMC200"]["bull"]

    # --- W1: comparar p_dl trayectorias ---
    print("\n--- W1: p_dl rollout vs walking vs p_hist verdadero ---")
    assets = states["rollout"]["ctx"]["assets"]
    T_vals = states["rollout"]["ctx"]["T_vals"]
    rows = []
    fig, axes = plt.subplots(len(assets), 1, figsize=(14, 4 * len(assets)),
                             sharex=True)
    if len(assets) == 1:
        axes = [axes]
    for ai, a in enumerate(assets):
        ax = axes[ai]
        p_true = (p_hist_real_spx if a == "SPX" else p_hist_real_cmc).loc[T_vals]
        ax.plot(T_vals, p_true.values, color="black", lw=2.0, ls="--",
                label=f"p_hist VERDADERO (mean={p_true.mean():.3f}, "
                      f"std={p_true.std():.3f})")
        for label, _, color in SETUPS:
            p = states[label]["ctx"]["p_dl"][a]["bull"]
            ax.plot(T_vals, p.values, color=color, lw=1.4,
                    label=f"LSTM {label} (mean={p.mean():.3f}, "
                          f"std={p.std():.3f})")
            for t, v in zip(T_vals, p.values):
                rows.append({"asset": a, "method": label,
                             "t": int(t), "p_bull": float(v)})
        for t, v in zip(T_vals, p_true.values):
            rows.append({"asset": a, "method": "true",
                         "t": int(t), "p_bull": float(v)})
        ax.axhline(0.5, color="grey", lw=0.6, ls=":")
        ax.set_title(f"p_bull({a}, t) sobre data SINTETICA — comparacion 3 fuentes")
        ax.set_xlabel("t"); ax.set_ylabel(f"p_bull({a})")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.25); ax.legend(fontsize=9, loc="best")
    save_fig(fig, "W1_pdl_compare", SUBDIR)
    save_csv(pd.DataFrame(rows), "W1_pdl_compare", SUBDIR)

    # --- W2: mu_mix ---
    print("\n--- W2: mu_mix(t) ---")
    rows = []
    fig, axes = plt.subplots(len(assets), 1, figsize=(14, 4 * len(assets)),
                             sharex=True)
    if len(assets) == 1:
        axes = [axes]
    for ai, a in enumerate(assets):
        ax = axes[ai]
        for label, _, color in SETUPS:
            mu = states[label]["ctx"]["mu_mix"][a]
            ax.plot(T_vals, mu.values * 100, color=color, lw=1.4,
                    label=f"{label} (mean={mu.mean()*100:+.2f}%, "
                          f"std={mu.std()*100:.3f}%)")
            for t, v in zip(T_vals, mu.values):
                rows.append({"asset": a, "method": label,
                             "t": int(t), "mu_mix_pct": float(v * 100)})
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_title(f"mu_mix({a}, t) sobre data SINTETICA")
        ax.set_xlabel("t"); ax.set_ylabel(f"mu_mix({a}) [%/sem]")
        ax.grid(True, alpha=0.25); ax.legend(fontsize=9, loc="best")
    save_fig(fig, "W2_mumix_compare", SUBDIR)
    save_csv(pd.DataFrame(rows), "W2_mumix_compare", SUBDIR)

    # --- W3/W4: regret grid por setup ---
    print("\n--- W3/W4: regret grid + politicas ---")
    for label, _, _ in SETUPS:
        print(f"\nGrilla {len(LAMBDA_GRID)}x{len(M_GRID)} en setup={label}...")
        V_df, policies = run_regret_grid(
            states[label]["ctx"], list(LAMBDA_GRID), list(M_GRID),
        )
        res = compute_regret_and_select(V_df)
        states[label]["V_df"] = V_df
        states[label]["policies"] = policies
        states[label]["res"] = res
        lam_m, m_m = res["g_mean"]
        V_row = res["V_table"].loc[(lam_m, m_m)]
        print(f"  g*_mean=({lam_m:.2f}, {m_m:.1f})  regret=${res['g_mean_metric']:,.2f}  "
              f"V mean=${V_row.mean():,.0f}  range=[${V_row.min():,.0f}, ${V_row.max():,.0f}]")

    # Politicas w(t)
    rows = []
    fig, axes = plt.subplots(len(assets), 1, figsize=(14, 4 * len(assets)),
                             sharex=True)
    if len(assets) == 1:
        axes = [axes]
    for ai, a in enumerate(assets):
        ax = axes[ai]
        ax.axhline(0.5, color="grey", ls="--", lw=0.6, label="w0=0.5")
        for label, _, color in SETUPS:
            res = states[label]["res"]
            pol = states[label]["policies"]
            lam_m, m_m = res["g_mean"]
            w_sol = pol[(lam_m, m_m)][0]
            w_t = [w_sol[(a, t)] for t in T_vals]
            ax.plot(T_vals, w_t, color=color, lw=1.5,
                    label=f"{label} g*=({lam_m:.2f},{m_m:.1f}) "
                          f"mean={np.mean(w_t):.3f} std={np.std(w_t):.3f}")
            for t, ww in zip(T_vals, w_t):
                rows.append({"asset": a, "method": label,
                             "t": int(t), "w": float(ww)})
        ax.set_title(f"Politica w({a}, t) bajo g*_mean — rollout vs walking [sintetica]")
        ax.set_xlabel("t"); ax.set_ylabel(f"w({a})")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.25); ax.legend(fontsize=9, loc="best")
    save_fig(fig, "W3_w_policy", SUBDIR)
    save_csv(pd.DataFrame(rows), "W3_w_policy", SUBDIR)

    # Resumen grid
    rows = []
    for label, _, _ in SETUPS:
        res = states[label]["res"]
        lam_m, m_m = res["g_mean"]
        V_row = res["V_table"].loc[(lam_m, m_m)]
        rows.append({
            "setup": label,
            "g_mean_lambda": float(lam_m), "g_mean_m": float(m_m),
            "regret": float(res["g_mean_metric"]),
            "V_mean": float(V_row.mean()),
            "V_min":  float(V_row.min()),
            "V_max":  float(V_row.max()),
        })
    save_csv(pd.DataFrame(rows), "W4_grid_compare", SUBDIR)

    # --- W5: backtest ---
    print("\n--- W5: backtest sobre data sintetica realizada ---")
    _, w_opt, u_opt, v_opt, _ = solve_portfolio(
        opt_ctx, lambda_riesgo=1.00, costo_mult=1.0,
    )
    fig, ax = plt.subplots(figsize=(13, 5.5))
    rows = []
    ctx0 = states["rollout"]["ctx"]
    T_vals = ctx0["T_vals"]
    C0 = ctx0["Capital_inicial"]
    t_f = T_vals[-1]
    cap_opt = simulate_capital_opt(w_opt, u_opt, v_opt, ctx0)
    cap_rb  = simulate_naive_rb(ctx0)
    cap_bh  = simulate_naive_bh(ctx0)
    ax.plot(T_vals, [cap_opt[t] for t in T_vals], color="#F2B705", lw=1.8,
            label=f"OPT base (lam=1.0)  fin=${cap_opt[t_f]:,.0f}  "
                  f"ret={cap_opt[t_f]/C0-1:+.1%}")
    ax.plot(T_vals, [cap_rb [t] for t in T_vals], color="#8B3A1F", lw=1.2,
            label=f"Naive RB  fin=${cap_rb[t_f]:,.0f}  ret={cap_rb[t_f]/C0-1:+.1%}")
    ax.plot(T_vals, [cap_bh [t] for t in T_vals], color="#aaa", lw=1.0, ls=":",
            label=f"Naive BH  fin=${cap_bh[t_f]:,.0f}  ret={cap_bh[t_f]/C0-1:+.1%}")
    for label, _, color in SETUPS:
        res = states[label]["res"]
        pol = states[label]["policies"]
        lam_m, m_m = res["g_mean"]
        w_rg, u_rg, v_rg, _ = pol[(lam_m, m_m)]
        cap_rg = simulate_capital_opt(w_rg, u_rg, v_rg, ctx0)
        ax.plot(T_vals, [cap_rg[t] for t in T_vals], color=color, lw=1.8,
                label=f"RG {label} (lam={lam_m:.2f}, m={m_m:.1f})  "
                      f"fin=${cap_rg[t_f]:,.0f}  ret={cap_rg[t_f]/C0-1:+.1%}")
        rows.append({"setup": f"RG_{label}", "cap_final": float(cap_rg[t_f]),
                     "ret_acum": float(cap_rg[t_f]/C0 - 1)})
    rows.append({"setup": "OPT_base",  "cap_final": float(cap_opt[t_f]),
                 "ret_acum": float(cap_opt[t_f]/C0 - 1)})
    rows.append({"setup": "Naive_RB",  "cap_final": float(cap_rb [t_f]),
                 "ret_acum": float(cap_rb [t_f]/C0 - 1)})
    rows.append({"setup": "Naive_BH",  "cap_final": float(cap_bh [t_f]),
                 "ret_acum": float(cap_bh [t_f]/C0 - 1)})
    ax.axhline(C0, color="#666", ls="--", lw=0.8, label=f"C0=${C0:,.0f}")
    ax.set_title("Backtest sobre data sintetica realizada — rollout vs walking")
    ax.set_xlabel("t"); ax.set_ylabel("Capital")
    ax.grid(True, alpha=0.25); ax.legend(fontsize=8, loc="best")
    save_fig(fig, "W5_backtest", SUBDIR)
    save_csv(pd.DataFrame(rows), "W5_backtest", SUBDIR)

    print("\n" + "=" * 70)
    print("Done.")
    print("=" * 70)
    for r in rows:
        print(f"  {r['setup']:<15} cap_final=${r['cap_final']:>10,.2f}  "
              f"ret={r['ret_acum']:+.2%}")


if __name__ == "__main__":
    main()
