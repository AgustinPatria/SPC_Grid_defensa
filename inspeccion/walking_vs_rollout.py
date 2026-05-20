"""Walking vs rollout: comparacion del metodo de construccion de p_{i,k,t}.

CONTEXTO del PDF (Modelo_RegretGrid_DL_Portfolio.pdf):

  Sec. 2.5 paso 1 define que los escenarios se generan por rollout
  autoregresivo desde la ULTIMA ventana observada. Sec. 3.3 define que t=1..T
  son periodos FUTUROS y que la simulacion de capital usa los retornos del
  escenario (no historicos). Por consistencia, p_dl(t) que alimenta al FO
  deberia construirse de la misma forma (rollout) y NO por walking sobre el
  historico — el walking le entrega al LSTM los retornos reales como ventana
  de entrada, dandole info que en el futuro no tendria.

  Adicionalmente, sec. 1.3 ec. (2) define mu_hat con p_{i,k,t} = del CSV
  historico (HMM), no del DL. El codigo actual mezcla mu_hat con p_dl_walking,
  creando dependencia circular entre el estimador historico y la prediccion.

Experimento: tres setups con T=163, mismos escenarios, misma grilla.

  A "actual"     : p_method=walking,  mu_hat_source=p_dl     (codigo actual)
  B "rollout_pdl": p_method=rollout,  mu_hat_source=p_dl     (aisla cambio de p_method)
  C "pdf_aligned": p_method=rollout,  mu_hat_source=p_hist   (alineado al PDF)

Salidas (inspeccion/walking_vs_rollout_out/):
  1_pbull_compare.csv/png     trayectorias p_bull(t) por activo y setup
  2_mumix_compare.csv/png     trayectorias mu_mix(t) por activo y setup
  3_grid_compare.csv/png      g*_mean, regret por setup
  4_backtest_<setup>.csv/png  capital sobre r_hist (diagnostico)
  5_V_table_summary.csv       V[g, s] resumen (mean/min/max) por setup
  6_terminal_summary.csv/png  retorno acumulado en backtest historico

Notas:
- El backtest historico (Diag 4 y 6) NO esta en el algoritmo del PDF. Lo
  incluimos como diagnostico comun a los tres setups para visualizar la
  diferencia. La "evaluacion correcta" segun el PDF es V_{g,s} (Diag 5).
- Los escenarios son los mismos en los 3 setups (dependen solo del LSTM
  congelado y de initial_window, no de p_method ni mu_hat_source).

Corre con:
    python -m inspeccion.walking_vs_rollout
    (o)
    python inspeccion/walking_vs_rollout.py
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


SUBDIR = "walking_vs_rollout"

SETUPS = [
    ("actual",      "walking", "p_dl"),
    ("rollout_pdl", "rollout", "p_dl"),
    ("pdf_aligned", "rollout", "p_hist"),
]
SETUP_COLORS = {
    "actual":      "#E63946",
    "rollout_pdl": "#F2B705",
    "pdf_aligned": "#1f77b4",
}


# ================================================================
# 1) Construir los 3 setups
# ================================================================

def build_setups():
    print("Construyendo OPT base ctx (referencia)...")
    opt_ctx = load_market_data(str(DATA_DIR))

    ctxs = {}
    for label, p_method, mu_hat_source in SETUPS:
        print(f"\nConstruyendo DL ctx '{label}' "
              f"(p_method={p_method}, mu_hat_source={mu_hat_source})...")
        ctx = build_dl_context(
            data_dir=DATA_DIR, checkpoint_path=CHECKPOINT_PATH,
            T=T_HORIZON, N_candidates=N_CANDIDATES, n_scenarios=N_SCENARIOS,
            seed=SCENARIO_SEED, summary_asset=SUMMARY_ASSET,
            p_method=p_method, mu_hat_source=mu_hat_source,
        )
        ctxs[label] = ctx
        print(f"  p_bull mean: " + "  ".join(
            f"{a}={ctx['p_dl'][a]['bull'].mean():.3f}"
            for a in ctx["assets"]
        ))
        print(f"  p_bull std : " + "  ".join(
            f"{a}={ctx['p_dl'][a]['bull'].std():.3f}"
            for a in ctx["assets"]
        ))
    return {"opt_ctx": opt_ctx, "ctxs": ctxs}


# ================================================================
# 2) Diagnostico 1: p_bull(t) comparado por setup
# ================================================================

def diag_pbull_compare(ctx):
    ctxs = ctx["ctxs"]
    assets = ctxs["actual"]["assets"]
    T_vals = ctxs["actual"]["T_vals"]

    rows = []
    fig, axes = plt.subplots(len(assets), 1, figsize=(11, 3.5 * len(assets)))
    if len(assets) == 1:
        axes = [axes]
    for ai, a in enumerate(assets):
        ax = axes[ai]
        for label, _, _ in SETUPS:
            p = ctxs[label]["p_dl"][a]["bull"]
            ax.plot(T_vals, p.values, color=SETUP_COLORS[label], lw=1.2,
                    label=f"{label} (mean={p.mean():.3f}, std={p.std():.3f})")
            for t, v in zip(T_vals, p.values):
                rows.append({"asset": a, "setup": label,
                             "t": int(t), "p_bull": float(v)})
        ax.set_title(f"p_bull({a}) por setup")
        ax.set_xlabel("t")
        ax.set_ylabel(f"p_bull({a})")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="best")
    save_fig(fig, "1_pbull_compare", SUBDIR)
    save_csv(pd.DataFrame(rows), "1_pbull_compare", SUBDIR)


# ================================================================
# 3) Diagnostico 2: mu_mix(t) comparado por setup
# ================================================================

def diag_mumix_compare(ctx):
    ctxs = ctx["ctxs"]
    assets = ctxs["actual"]["assets"]
    T_vals = ctxs["actual"]["T_vals"]

    rows = []
    fig, axes = plt.subplots(len(assets), 1, figsize=(11, 3.5 * len(assets)))
    if len(assets) == 1:
        axes = [axes]
    for ai, a in enumerate(assets):
        ax = axes[ai]
        # Tambien graficamos mu_mix del OPT base (referencia historica).
        mu_opt = ctx["opt_ctx"]["mu_mix"][a]
        ax.plot(mu_opt.index, mu_opt.values, color="#333", lw=1.0, ls="--",
                label=f"OPT base (mean={mu_opt.mean():.4f})")
        for label, _, _ in SETUPS:
            mu = ctxs[label]["mu_mix"][a]
            ax.plot(T_vals, mu.values, color=SETUP_COLORS[label], lw=1.2,
                    label=f"{label} (mean={mu.mean():.4f}, std={mu.std():.4f})")
            for t, v in zip(T_vals, mu.values):
                rows.append({"asset": a, "setup": label,
                             "t": int(t), "mu_mix": float(v)})
        ax.axhline(0, color="grey", lw=0.6)
        ax.set_title(f"mu_mix({a}) por setup")
        ax.set_xlabel("t")
        ax.set_ylabel(f"mu_mix({a})  (retorno semanal esperado)")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="best")
    save_fig(fig, "2_mumix_compare", SUBDIR)
    save_csv(pd.DataFrame(rows), "2_mumix_compare", SUBDIR)


# ================================================================
# 4) Regret grid por setup
# ================================================================

def run_grids(ctx):
    lambda_grid = list(LAMBDA_GRID)
    m_grid = list(M_GRID)
    grids = {}
    for label, _, _ in SETUPS:
        print(f"\n--- Grilla {len(lambda_grid)}x{len(m_grid)}="
              f"{len(lambda_grid)*len(m_grid)} solves en setup={label} ---")
        V_df, pol = run_regret_grid(ctx["ctxs"][label], lambda_grid, m_grid)
        res = compute_regret_and_select(V_df)
        grids[label] = {"V_df": V_df, "policies": pol, "res": res}
    return grids


def diag_grid_compare(ctx, grids):
    rows = []
    for label, _, _ in SETUPS:
        res = grids[label]["res"]
        lam_m, m_m = res["g_mean"]
        lam_w, m_w = res["g_worst"]
        V_mean_row = res["V_table"].loc[(lam_m, m_m)]
        C0 = ctx["ctxs"][label]["Capital_inicial"]
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
# 5) Backtest historico (diagnostico fuera del algoritmo)
# ================================================================

def _backtest_setup(label, dl_ctx, opt_ctx, res, policies):
    lam_m, m_m = res["g_mean"]
    w_star, u_star, v_star, _z = policies[(lam_m, m_m)]
    _, w_opt, u_opt, v_opt, _ = solve_portfolio(
        opt_ctx, lambda_riesgo=1.00, costo_mult=1.0,
    )
    cap_opt = simulate_capital_opt(w_opt, u_opt, v_opt, dl_ctx)
    cap_rg  = simulate_capital_opt(w_star, u_star, v_star, dl_ctx)
    cap_rb  = simulate_naive_rb(dl_ctx)
    cap_bh  = simulate_naive_bh(dl_ctx)
    T_vals = dl_ctx["T_vals"]
    C0 = dl_ctx["Capital_inicial"]
    t_f = T_vals[-1]
    summary = {
        "setup": label,
        "n_periodos": len(T_vals),
        "OPT_cap_final":     float(cap_opt[t_f]),
        "RG_cap_final":      float(cap_rg [t_f]),
        "NaiveRB_cap_final": float(cap_rb [t_f]),
        "NaiveBH_cap_final": float(cap_bh [t_f]),
        "OPT_ret_acum":      float(cap_opt[t_f]/C0 - 1),
        "RG_ret_acum":       float(cap_rg [t_f]/C0 - 1),
        "NaiveRB_ret_acum":  float(cap_rb [t_f]/C0 - 1),
        "NaiveBH_ret_acum":  float(cap_bh [t_f]/C0 - 1),
        "g_mean_lambda": float(lam_m), "g_mean_m": float(m_m),
    }
    return {
        "summary": summary, "T_vals": T_vals,
        "cap_opt": cap_opt, "cap_rg": cap_rg,
        "cap_rb":  cap_rb,  "cap_bh": cap_bh,
        "lam_m": lam_m, "m_m": m_m,
    }


def diag_backtest(ctx, grids):
    bts = {}
    for label, _, _ in SETUPS:
        bt = _backtest_setup(
            label, ctx["ctxs"][label], ctx["opt_ctx"],
            grids[label]["res"], grids[label]["policies"],
        )
        bts[label] = bt
        T_vals = bt["T_vals"]
        rows = [{
            "t": int(t),
            "OPT":     float(bt["cap_opt"][t]),
            "RG":      float(bt["cap_rg" ][t]),
            "NaiveRB": float(bt["cap_rb" ][t]),
            "NaiveBH": float(bt["cap_bh" ][t]),
        } for t in T_vals]
        save_csv(pd.DataFrame(rows), f"4_backtest_{label}", SUBDIR)

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(T_vals, [bt["cap_opt"][t] for t in T_vals],
                color="#F2B705", lw=1.8, label="OPT base")
        ax.plot(T_vals, [bt["cap_rg" ][t] for t in T_vals],
                color="#1f77b4", lw=1.8,
                label=f"Regret-Grid g*_mean (lambda={bt['lam_m']:.2f}, m={bt['m_m']:.1f})")
        ax.plot(T_vals, [bt["cap_rb" ][t] for t in T_vals],
                color="#8B3A1F", lw=1.2, label="Naive 50/50 (rebal)")
        ax.plot(T_vals, [bt["cap_bh" ][t] for t in T_vals],
                color="#E63946", lw=1.2, label="Naive 50/50 (buy&hold)")
        ax.axhline(ctx["ctxs"][label]["Capital_inicial"],
                   color="#666", ls="--", lw=0.8, label="C0")
        ax.set_title(f"Backtest historico (diagnostico) - setup {label}  "
                     f"({len(T_vals)} periodos)")
        ax.set_xlabel("t"); ax.set_ylabel("Capital")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9, loc="best")
        save_fig(fig, f"4_backtest_{label}", SUBDIR)
    return bts


# ================================================================
# 6) Resumen del V_table (la "verdadera" evaluacion del PDF)
# ================================================================

def diag_V_table_summary(grids):
    rows = []
    for label, _, _ in SETUPS:
        res = grids[label]["res"]
        V_tab = res["V_table"]
        for (lam, m_), row in V_tab.iterrows():
            rows.append({
                "setup": label,
                "lambda": float(lam), "m": float(m_),
                "V_mean": float(row.mean()),
                "V_min":  float(row.min()),
                "V_max":  float(row.max()),
                "V_std":  float(row.std()),
            })
    save_csv(pd.DataFrame(rows), "5_V_table_summary", SUBDIR)


# ================================================================
# 7) Resumen terminal del backtest historico
# ================================================================

def diag_terminal_summary(bts):
    df = pd.DataFrame([bts[s[0]]["summary"] for s in SETUPS])
    save_csv(df, "6_terminal_summary", SUBDIR)

    politicas = ["OPT", "RG", "NaiveRB", "NaiveBH"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (label, _, _) in zip(axes, SETUPS):
        bt = bts[label]
        rets = [bt["summary"][f"{p}_ret_acum"] for p in politicas]
        colors = ["#F2B705", "#1f77b4", "#8B3A1F", "#E63946"]
        bars = ax.bar(politicas, [100*r for r in rets], color=colors, alpha=0.85)
        ax.axhline(0, color="grey", lw=0.6)
        ax.set_title(f"setup={label}  ({bt['summary']['n_periodos']} periodos)")
        ax.set_ylabel("Retorno acumulado [%]")
        for b, r in zip(bars, rets):
            ax.text(b.get_x() + b.get_width()/2, b.get_height(),
                    f"{r*100:+.1f}%", ha="center",
                    va="bottom" if r >= 0 else "top", fontsize=9)
        ax.grid(True, alpha=0.25, axis="y")
    fig.suptitle("Backtest historico (diagnostico fuera del PDF) por setup",
                 fontsize=11)
    save_fig(fig, "6_terminal_summary", SUBDIR)


# ================================================================
# Main
# ================================================================

def main():
    print("=" * 70)
    print("WALKING vs ROLLOUT - alineamiento al PDF sec. 2.5 + 1.3")
    print("=" * 70)

    ctx = build_setups()
    diag_pbull_compare(ctx)
    diag_mumix_compare(ctx)

    grids = run_grids(ctx)
    diag_grid_compare(ctx, grids)
    diag_V_table_summary(grids)

    bts = diag_backtest(ctx, grids)
    diag_terminal_summary(bts)

    print("\n" + "=" * 70)
    print("RESUMEN")
    print("=" * 70)
    for label, p_method, mu_hat_source in SETUPS:
        res = grids[label]["res"]
        lam_m, m_m = res["g_mean"]
        bt_s = bts[label]["summary"]
        print(f"  [{label:<13}] p={p_method:<8} mu_hat={mu_hat_source:<7}  "
              f"g*=(lam={lam_m:.2f}, m={m_m:.1f})  "
              f"regret=${res['g_mean_metric']:,.0f}")
        print(f"                  backtest hist: OPT={bt_s['OPT_ret_acum']:+.2%}  "
              f"RG={bt_s['RG_ret_acum']:+.2%}  "
              f"NaiveRB={bt_s['NaiveRB_ret_acum']:+.2%}  "
              f"NaiveBH={bt_s['NaiveBH_ret_acum']:+.2%}")
    print("=" * 70)


if __name__ == "__main__":
    main()
