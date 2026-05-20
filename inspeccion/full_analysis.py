"""Analisis comprehensivo del pipeline con defaults actuales (rollout + p_sign).

Genera un reporte completo de TODOS los objetos intermedios y outputs del
pipeline, no solo los finales. Pensado para que el usuario vea de un vistazo
el estado completo del modelo.

Secciones:

  A. ESCENARIOS
     - 5 representativos por quintil (PDF sec 2.5 paso 2)
     - Distribucion de los N=1000 candidatos (boxplots por activo, fan chart)
     - Retorno acumulado del SPX por quintil (PDF ec 17)

  B. PROBABILIDADES DE REGIMEN (p_bull, p_bear)
     - Trayectorias p_bull(t) por activo (rollout default)
     - Estadisticas mean, std, min, max
     - Comparacion con p_hist (CSV) y p_sign (oraculo)

  C. MOMENTOS POR REGIMEN Y MEZCLA
     - mu_hat[(i, k)] tabla
     - mu_mix(t) y sigma_mix(t) trayectorias
     - Descomposicion: contribucion de bull vs bear a mu_mix

  D. REGRET GRID
     - V[g, s] heatmap (lambda x m x scenario)
     - R[g, s] heatmap
     - Tabla de seleccion g* mean / worst

  E. POLITICAS w(t)
     - w(SPX, t) y w(CMC, t) para g*_mean y g*_worst
     - Turnover por periodo
     - Costos acumulados

  F. CAPITAL POR ESCENARIO
     - Trayectoria de capital bajo g*_mean en cada escenario
     - Bajo g*_worst en cada escenario
     - Distribucion final V_terminal

  G. BACKTEST HISTORICO (diagnostico)
     - OPT base vs RG g*_mean vs naive RB vs naive BH sobre r_hist

Outputs en inspeccion/full_analysis_out/:
  A1_escenarios_5_rep.csv/png
  A2_candidatos_distribution.csv/png
  A3_R_summary.csv
  B1_p_bull_compare.csv/png
  B2_p_summary.csv
  C1_mu_hat.csv
  C2_mu_mix_traj.csv/png
  C3_sigma_mix_traj.csv/png
  D1_V_table.csv/png
  D2_R_table.csv/png
  D3_g_seleccion.csv
  E1_w_g_mean.csv/png
  E2_w_g_worst.csv/png
  E3_turnover_costs.csv/png
  F1_capital_g_mean.csv/png
  F2_capital_g_worst.csv/png
  F3_V_terminal_dist.csv
  G1_backtest_historical.csv/png
  Z_resumen_final.csv  (todos los numeros importantes en una tabla)
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
    BULL_THRESHOLD,
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
from dl.generador_escenarios import generate_candidate_scenarios
from dl.prediccion_deciles import load_checkpoint
from Regret_Grid import (
    build_dl_context,
    compute_regret_and_select,
    load_market_data,
    predict_pbull_rollout,
    run_regret_grid,
    simulate_capital_on_scenario,
    simulate_capital_opt,
    simulate_naive_bh,
    simulate_naive_rb,
    solve_portfolio,
)


SUBDIR = "full_analysis"


def setup():
    print("=" * 70)
    print("ANALISIS COMPREHENSIVO DEL PIPELINE (defaults: rollout + p_sign)")
    print("=" * 70)
    print("\nConstruyendo contextos...")
    ctx_dl = build_dl_context(
        data_dir=DATA_DIR, checkpoint_path=CHECKPOINT_PATH,
        T=T_HORIZON, N_candidates=N_CANDIDATES, n_scenarios=N_SCENARIOS,
        seed=SCENARIO_SEED, summary_asset=SUMMARY_ASSET,
    )
    ctx_opt = load_market_data(str(DATA_DIR))
    model = load_checkpoint(CHECKPOINT_PATH)

    print(f"  Assets: {ctx_dl['assets']}")
    print(f"  T = {ctx_dl['nT']}, scenarios = {ctx_dl['scenarios'].shape}")
    return {"ctx_dl": ctx_dl, "ctx_opt": ctx_opt, "model": model}


# ================================================================
# A. ESCENARIOS
# ================================================================

def section_A_scenarios(state):
    print("\n--- A. ESCENARIOS ---")
    ctx = state["ctx_dl"]
    assets = ctx["assets"]
    T_vals = ctx["T_vals"]
    scenarios = ctx["scenarios"]                      # (5, T, A)
    summary_idx = assets.index(SUMMARY_ASSET)

    # A1: 5 escenarios representativos
    rows = []
    fig, axes = plt.subplots(len(assets), 1, figsize=(12, 4 * len(assets)),
                             sharex=True)
    if len(assets) == 1:
        axes = [axes]
    cmap = plt.get_cmap("RdYlGn")
    for ai, a in enumerate(assets):
        ax = axes[ai]
        for s in range(scenarios.shape[0]):
            r = scenarios[s, :, ai]
            cum = np.cumprod(1.0 + r) - 1.0
            color = cmap(s / max(scenarios.shape[0] - 1, 1))
            ax.plot(T_vals, cum * 100, color=color, lw=1.5,
                    label=f"s={s} (cum final = {cum[-1]*100:+.1f}%)")
            for t, rv, cv in zip(T_vals, r, cum):
                rows.append({"scenario": s, "asset": a, "t": int(t),
                             "r_pct": float(rv * 100),
                             "cum_pct": float(cv * 100)})
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_title(f"5 escenarios representativos - {a}")
        ax.set_xlabel("t")
        ax.set_ylabel(f"retorno acumulado {a} [%]")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=9)
    save_fig(fig, "A1_escenarios_5_rep", SUBDIR)
    save_csv(pd.DataFrame(rows), "A1_escenarios_5_rep", SUBDIR)

    # A2: distribucion de los N candidatos
    print("  Re-generando N=1000 candidatos para diagnostico...")
    initial_window = np.stack(
        [ctx["r"][i].sort_index().values[-state["model"].config.H:]
         for i in assets], axis=1,
    ).astype(np.float32)
    candidates = generate_candidate_scenarios(
        state["model"], initial_window, N=N_CANDIDATES, T=T_HORIZON,
        seed=SCENARIO_SEED,
    )                                                  # (N, T, A)

    # Cumulative return per scenario per asset
    cum_cand = np.cumprod(1.0 + candidates, axis=1) - 1.0  # (N, T, A)
    R_summary_a = cum_cand[:, -1, summary_idx]              # (N,)
    rows = []
    for s in range(len(R_summary_a)):
        rows.append({
            "scenario_idx": s,
            "R_cumul_SPX_final": float(R_summary_a[s]),
        })
    save_csv(pd.DataFrame(rows), "A3_R_summary", SUBDIR)
    print(f"  R_cum SPX (N={len(R_summary_a)}): mean={R_summary_a.mean()*100:+.2f}%  "
          f"median={np.median(R_summary_a)*100:+.2f}%  "
          f"std={R_summary_a.std()*100:.2f}%  "
          f"min={R_summary_a.min()*100:+.2f}%  max={R_summary_a.max()*100:+.2f}%")

    # Plot: fan chart de candidatos (percentiles)
    fig, axes = plt.subplots(len(assets), 1, figsize=(12, 4 * len(assets)),
                             sharex=True)
    if len(assets) == 1:
        axes = [axes]
    percs = [5, 25, 50, 75, 95]
    for ai, a in enumerate(assets):
        cum_a = np.cumprod(1.0 + candidates[:, :, ai], axis=1) - 1.0  # (N, T)
        pct = np.percentile(cum_a, percs, axis=0)                     # (5, T)
        ax = axes[ai]
        ax.fill_between(T_vals, pct[0] * 100, pct[-1] * 100,
                        color="#1f77b4", alpha=0.15, label="5-95%")
        ax.fill_between(T_vals, pct[1] * 100, pct[-2] * 100,
                        color="#1f77b4", alpha=0.3, label="25-75%")
        ax.plot(T_vals, pct[2] * 100, color="#1f77b4", lw=1.5, label="mediana")
        # Superponer los 5 representativos
        for s in range(scenarios.shape[0]):
            cum_s = np.cumprod(1.0 + scenarios[s, :, ai]) - 1.0
            ax.plot(T_vals, cum_s * 100, color="#E63946", lw=1.0,
                    alpha=0.7, label=f"rep s={s}" if s == 0 else None)
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_title(f"Distribucion N={N_CANDIDATES} candidatos + 5 reps - {a}")
        ax.set_xlabel("t")
        ax.set_ylabel(f"retorno acumulado {a} [%]")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)
    save_fig(fig, "A2_candidatos_distribution", SUBDIR)


# ================================================================
# B. PROBABILIDADES DE REGIMEN
# ================================================================

def section_B_pbull(state):
    print("\n--- B. PROBABILIDADES DE REGIMEN ---")
    ctx = state["ctx_dl"]
    assets = ctx["assets"]
    T_vals = ctx["T_vals"]
    model = state["model"]
    H = model.config.H

    # p_dl actual (rollout, ya en ctx)
    p_dl = ctx["p_dl"]

    # p_hist (CSV/HMM) truncada a T
    p_hist_T = {
        i: pd.DataFrame(
            state["ctx_opt"]["p_hist"][i].sort_index().values[:T_HORIZON, :],
            index=T_vals,
            columns=list(state["ctx_opt"]["p_hist"][i].columns),
        )
        for i in assets
    }
    # p_sign oracle sobre r_hist[:T]
    p_sign = {}
    for i in assets:
        r_i = pd.Series(
            ctx["r"][i].sort_index().values[:T_HORIZON], index=T_vals,
        )
        bull_i = (r_i >= BULL_THRESHOLD).astype(float)
        p_sign[i] = pd.DataFrame({"bear": 1.0 - bull_i.values,
                                  "bull": bull_i.values}, index=T_vals)

    rows = []
    fig, axes = plt.subplots(len(assets), 1, figsize=(13, 4 * len(assets)),
                             sharex=True)
    if len(assets) == 1:
        axes = [axes]
    for ai, a in enumerate(assets):
        ax = axes[ai]
        p1 = p_dl[a]["bull"]
        p2 = p_hist_T[a]["bull"]
        p3 = p_sign[a]["bull"]
        ax.plot(T_vals, p1.values, color="#1f77b4", lw=1.4,
                label=f"p_DL rollout (mean={p1.mean():.3f}, std={p1.std():.3f})")
        ax.plot(T_vals, p2.values, color="#F2B705", lw=1.2,
                label=f"p_HIST CSV (mean={p2.mean():.3f}, std={p2.std():.3f})")
        ax.plot(T_vals, p3.values, color="#E63946", lw=0.6, alpha=0.5,
                label=f"p_SIGN oracle (mean={p3.mean():.3f})")
        ax.axhline(0.5, color="grey", ls="--", lw=0.6)
        ax.set_title(f"p_bull({a}, t) — comparacion p_dl vs p_hist vs p_sign")
        ax.set_xlabel("t")
        ax.set_ylabel(f"p_bull({a})")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)
        for t, v in zip(T_vals, p1.values):
            rows.append({"asset": a, "source": "p_dl_rollout",
                         "t": int(t), "p_bull": float(v)})
        for t, v in zip(T_vals, p2.values):
            rows.append({"asset": a, "source": "p_hist_csv",
                         "t": int(t), "p_bull": float(v)})
        for t, v in zip(T_vals, p3.values):
            rows.append({"asset": a, "source": "p_sign_oracle",
                         "t": int(t), "p_bull": float(v)})
    save_fig(fig, "B1_p_bull_compare", SUBDIR)
    save_csv(pd.DataFrame(rows), "B1_p_bull_compare", SUBDIR)

    # Resumen estadistico
    summary_rows = []
    for source_name, source in [("p_dl_rollout", p_dl),
                                ("p_hist_csv", p_hist_T),
                                ("p_sign_oracle", p_sign)]:
        for a in assets:
            p = source[a]["bull"]
            summary_rows.append({
                "source": source_name, "asset": a,
                "p_bull_mean": float(p.mean()),
                "p_bull_std":  float(p.std()),
                "p_bull_min":  float(p.min()),
                "p_bull_max":  float(p.max()),
                "p_bull_<0.5_pct": float((p < 0.5).mean()),
            })
    df_sum = pd.DataFrame(summary_rows)
    save_csv(df_sum, "B2_p_summary", SUBDIR)
    print(df_sum.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# ================================================================
# C. MOMENTOS Y MEZCLA
# ================================================================

def section_C_moments(state):
    print("\n--- C. MOMENTOS Y MEZCLA ---")
    ctx = state["ctx_dl"]
    assets = ctx["assets"]
    T_vals = ctx["T_vals"]

    # C1: mu_hat y sigma_hat tabla
    rows = []
    for a in assets:
        for k in REGIMES:
            mu = ctx["mu_hat"][(a, k)]
            sd = np.sqrt(ctx["sigma_hat"][(a, a, k)])
            rows.append({
                "asset": a, "regime": k,
                "mu_hat_pct": float(mu * 100),
                "sigma_hat_pct": float(sd * 100),
            })
    df = pd.DataFrame(rows)
    save_csv(df, "C1_mu_hat", SUBDIR)
    print("\n  mu_hat y sigma_hat por (asset, regime):")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n  Cov SPX-CMC:")
    for k in REGIMES:
        cov = ctx["sigma_hat"][("SPX", "CMC200", k)]
        print(f"    {k}: {cov:.6f}")

    # C2: mu_mix trayectoria
    rows = []
    fig, axes = plt.subplots(len(assets), 1, figsize=(13, 4 * len(assets)),
                             sharex=True)
    if len(assets) == 1:
        axes = [axes]
    for ai, a in enumerate(assets):
        ax = axes[ai]
        mu = ctx["mu_mix"][a]
        ax.plot(T_vals, mu.values * 100, color="#1f77b4", lw=1.4,
                label=f"mu_mix({a}, t) mean={mu.mean()*100:+.3f}%")
        # Descomposicion: mu_bull * p_bull + mu_bear * p_bear
        p_bull = ctx["p_dl"][a]["bull"]
        contrib_bull = p_bull * ctx["mu_hat"][(a, "bull")]
        contrib_bear = (1 - p_bull) * ctx["mu_hat"][(a, "bear")]
        ax.plot(T_vals, contrib_bull * 100, color="#22c55e", lw=0.8,
                alpha=0.7, label="contrib bull (p_bull × mu_bull)")
        ax.plot(T_vals, contrib_bear * 100, color="#E63946", lw=0.8,
                alpha=0.7, label="contrib bear ((1-p_bull) × mu_bear)")
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_title(f"mu_mix({a}, t) y descomposicion por regimen")
        ax.set_xlabel("t")
        ax.set_ylabel(f"mu_mix({a}) [%/sem]")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)
        for t, v in zip(T_vals, mu.values):
            rows.append({"asset": a, "t": int(t),
                         "mu_mix_pct": float(v * 100),
                         "contrib_bull_pct": float(contrib_bull.iloc[T_vals.index(t)] * 100),
                         "contrib_bear_pct": float(contrib_bear.iloc[T_vals.index(t)] * 100)})
    save_fig(fig, "C2_mu_mix_traj", SUBDIR)
    save_csv(pd.DataFrame(rows), "C2_mu_mix_traj", SUBDIR)

    # C3: sigma_mix trayectoria (diagonal: var SPX, var CMC, y cov)
    rows = []
    fig, ax = plt.subplots(figsize=(13, 5))
    for ai, a in enumerate(assets):
        sig = ctx["sigma_mix"][a][a]
        ax.plot(T_vals, np.sqrt(sig.values) * 100, lw=1.3,
                label=f"sigma_mix({a}, {a}) sqrt  mean={(np.sqrt(sig.values)).mean()*100:.2f}%")
        for t, v in zip(T_vals, sig.values):
            rows.append({"pair": f"{a}-{a}", "t": int(t),
                         "var": float(v), "std_pct": float(np.sqrt(v) * 100)})
    cov_sc = ctx["sigma_mix"]["SPX"]["CMC200"]
    ax.plot(T_vals, cov_sc.values * 10000, color="#666", lw=1.0, ls=":",
            label="cov(SPX, CMC) × 10000 (escala 1e-4)")
    for t, v in zip(T_vals, cov_sc.values):
        rows.append({"pair": "SPX-CMC200", "t": int(t),
                     "var": float(v), "std_pct": float("nan")})
    ax.set_title("sigma_mix(t): desviacion estandar por activo + covarianza cruzada")
    ax.set_xlabel("t")
    ax.set_ylabel("sigma [%/sem] (lineas) / cov × 1e4 (puntos)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    save_fig(fig, "C3_sigma_mix_traj", SUBDIR)
    save_csv(pd.DataFrame(rows), "C3_sigma_mix_traj", SUBDIR)


# ================================================================
# D. REGRET GRID
# ================================================================

def section_D_regret(state):
    print("\n--- D. REGRET GRID ---")
    ctx = state["ctx_dl"]
    lambda_grid = list(LAMBDA_GRID)
    m_grid = list(M_GRID)
    V_df, policies = run_regret_grid(ctx, lambda_grid, m_grid)
    res = compute_regret_and_select(V_df)

    save_csv(V_df, "D1_V_table_long", SUBDIR)
    save_csv(res["V_table"].reset_index(), "D1_V_table_pivot", SUBDIR)
    save_csv(res["R_table"].reset_index(), "D2_R_table_pivot", SUBDIR)
    save_csv(res["regret_summary"].reset_index(), "D3_regret_summary", SUBDIR)

    # V[g,s] heatmap por escenario
    V_tab = res["V_table"]
    n_S = V_tab.shape[1]
    fig, axes = plt.subplots(1, n_S + 1, figsize=(3.5 * (n_S + 1), 5))
    for s in range(n_S):
        ax = axes[s]
        mat = V_tab[s].unstack("m").sort_index().sort_index(axis=1)
        im = ax.imshow(mat.values, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(mat.columns)))
        ax.set_xticklabels([f"{m:.1f}" for m in mat.columns])
        ax.set_yticks(range(len(mat.index)))
        ax.set_yticklabels([f"{l:.2f}" for l in mat.index])
        ax.set_xlabel("m"); ax.set_ylabel("lambda")
        ax.set_title(f"V[g, s={s}]")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"${mat.values[i, j]:,.0f}",
                        ha="center", va="center", fontsize=6,
                        color="white" if mat.values[i, j] < mat.values.mean() else "black")
    # Mean V
    mean_V = V_tab.mean(axis=1).unstack("m").sort_index().sort_index(axis=1)
    ax = axes[n_S]
    ax.imshow(mean_V.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(mean_V.columns)))
    ax.set_xticklabels([f"{m:.1f}" for m in mean_V.columns])
    ax.set_yticks(range(len(mean_V.index)))
    ax.set_yticklabels([f"{l:.2f}" for l in mean_V.index])
    ax.set_xlabel("m"); ax.set_ylabel("lambda")
    ax.set_title("mean V")
    for i in range(mean_V.shape[0]):
        for j in range(mean_V.shape[1]):
            ax.text(j, i, f"${mean_V.values[i, j]:,.0f}",
                    ha="center", va="center", fontsize=6,
                    color="white" if mean_V.values[i, j] < mean_V.values.mean() else "black")
    save_fig(fig, "D1_V_heatmap", SUBDIR)

    # R[g,s] heatmap
    R_tab = res["R_table"]
    mean_R = R_tab.mean(axis=1).unstack("m").sort_index().sort_index(axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, mat, title in [(axes[0], mean_R, "mean_R por g"),
                           (axes[1],
                            R_tab.max(axis=1).unstack("m").sort_index().sort_index(axis=1),
                            "max_R por g (peor caso)")]:
        ax.imshow(mat.values, cmap="RdYlGn_r", aspect="auto")
        ax.set_xticks(range(len(mat.columns)))
        ax.set_xticklabels([f"{m:.1f}" for m in mat.columns])
        ax.set_yticks(range(len(mat.index)))
        ax.set_yticklabels([f"{l:.2f}" for l in mat.index])
        ax.set_xlabel("m"); ax.set_ylabel("lambda")
        ax.set_title(title)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"${mat.values[i, j]:,.0f}",
                        ha="center", va="center", fontsize=7)
    save_fig(fig, "D2_R_heatmap", SUBDIR)

    # Tabla de seleccion
    sel = []
    lam_m, m_m = res["g_mean"]
    lam_w, m_w = res["g_worst"]
    sel.append({"tipo": "g_mean",  "lambda": float(lam_m), "m": float(m_m),
                "regret": float(res["g_mean_metric"])})
    sel.append({"tipo": "g_worst", "lambda": float(lam_w), "m": float(m_w),
                "regret": float(res["g_worst_metric"])})
    save_csv(pd.DataFrame(sel), "D3_g_seleccion", SUBDIR)
    print(f"  g*_mean  = ({lam_m:.2f}, {m_m:.1f})  regret = ${res['g_mean_metric']:,.0f}")
    print(f"  g*_worst = ({lam_w:.2f}, {m_w:.1f})  regret = ${res['g_worst_metric']:,.0f}")

    state["V_df"] = V_df
    state["policies"] = policies
    state["res"] = res


# ================================================================
# E. POLITICAS w(t)
# ================================================================

def section_E_policies(state):
    print("\n--- E. POLITICAS w(t) ---")
    ctx = state["ctx_dl"]
    assets = ctx["assets"]
    T_vals = ctx["T_vals"]
    c_base = ctx["c_base"]
    w0 = ctx["w0"]
    res = state["res"]
    policies = state["policies"]

    for label, g in [("g_mean", res["g_mean"]), ("g_worst", res["g_worst"])]:
        lam, m_ = g
        w, u, v, z = policies[(lam, m_)]
        rows = []
        fig, axes = plt.subplots(len(assets), 1, figsize=(13, 4 * len(assets)),
                                 sharex=True)
        if len(assets) == 1:
            axes = [axes]
        for ai, a in enumerate(assets):
            ax = axes[ai]
            w_t = [w[(a, t)] for t in T_vals]
            u_t = [u[(a, t)] for t in T_vals]
            v_t = [v[(a, t)] for t in T_vals]
            ax.plot(T_vals, w_t, color="#1f77b4", lw=1.5, label=f"w({a}, t)")
            ax.axhline(w0[a], color="grey", ls="--", lw=0.8,
                       label=f"w0 = {w0[a]}")
            ax.set_title(f"w({a}, t) bajo {label} (lambda={lam:.2f}, m={m_:.1f})")
            ax.set_xlabel("t")
            ax.set_ylabel(f"w({a})")
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best", fontsize=9)
            for t, ww, uu, vv in zip(T_vals, w_t, u_t, v_t):
                rows.append({"policy": label, "asset": a, "t": int(t),
                             "w": float(ww), "u": float(uu), "v": float(vv)})
        save_fig(fig, f"E1_w_{label}", SUBDIR)
        save_csv(pd.DataFrame(rows), f"E1_w_{label}", SUBDIR)

    # Turnover y costos acumulados
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for label, g in [("g_mean", res["g_mean"]), ("g_worst", res["g_worst"])]:
        lam, m_ = g
        w, u, v, _ = policies[(lam, m_)]
        turn = []
        cost_acum = []
        c_so_far = 0.0
        for idx, t in enumerate(T_vals):
            if idx == 0:
                tt = sum(abs(w[(a, t)] - w0[a]) for a in assets)
            else:
                t_prev = T_vals[idx - 1]
                tt = sum(abs(w[(a, t)] - w[(a, t_prev)]) for a in assets)
            cost_t = sum(c_base[a] * (u[(a, t)] + v[(a, t)]) for a in assets)
            c_so_far += cost_t
            turn.append(tt)
            cost_acum.append(c_so_far)
            rows.append({"policy": label, "t": int(t),
                         "turnover": float(tt),
                         "cost_t": float(cost_t),
                         "cost_acum": float(c_so_far)})
        axes[0].plot(T_vals, turn, lw=1.3,
                     label=f"{label} (total={sum(turn):.3f})")
        axes[1].plot(T_vals, cost_acum, lw=1.3,
                     label=f"{label} (final={c_so_far:.4f})")
    axes[0].set_title("Turnover por periodo")
    axes[0].set_xlabel("t"); axes[0].set_ylabel("sum_i |w(i,t) - w(i,t-1)|")
    axes[0].grid(True, alpha=0.25); axes[0].legend()
    axes[1].set_title("Costos acumulados de rebalanceo")
    axes[1].set_xlabel("t"); axes[1].set_ylabel("costo acumulado (fraccion)")
    axes[1].grid(True, alpha=0.25); axes[1].legend()
    save_fig(fig, "E3_turnover_costs", SUBDIR)
    save_csv(pd.DataFrame(rows), "E3_turnover_costs", SUBDIR)


# ================================================================
# F. CAPITAL POR ESCENARIO
# ================================================================

def section_F_capital(state):
    print("\n--- F. CAPITAL POR ESCENARIO ---")
    ctx = state["ctx_dl"]
    assets = ctx["assets"]
    T_vals = ctx["T_vals"]
    c_base = ctx["c_base"]
    C0 = ctx["Capital_inicial"]
    scenarios = ctx["scenarios"]
    res = state["res"]
    policies = state["policies"]

    rows = []
    for label, g in [("g_mean", res["g_mean"]), ("g_worst", res["g_worst"])]:
        lam, m_ = g
        w, u, v, _ = policies[(lam, m_)]
        fig, ax = plt.subplots(figsize=(11, 5))
        cmap = plt.get_cmap("RdYlGn")
        for s in range(scenarios.shape[0]):
            cap = simulate_capital_on_scenario(
                w, u, v, scenarios[s], assets, c_base, C0, T_vals,
            )
            color = cmap(s / max(scenarios.shape[0] - 1, 1))
            ax.plot(T_vals, [cap[t] for t in T_vals], color=color, lw=1.4,
                    label=f"s={s} (V={cap[T_vals[-1]]:,.0f}, "
                          f"ret={cap[T_vals[-1]]/C0 - 1:+.1%})")
            for t in T_vals:
                rows.append({"policy": label, "scenario": s, "t": int(t),
                             "capital": float(cap[t])})
        ax.axhline(C0, color="#666", ls="--", lw=0.8, label=f"C0=${C0:,.0f}")
        ax.set_title(f"Capital bajo {label} (lambda={lam:.2f}, m={m_:.1f}) "
                     f"en los 5 escenarios DL")
        ax.set_xlabel("t")
        ax.set_ylabel("Capital")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)
        save_fig(fig, f"F1_capital_{label}", SUBDIR)
    save_csv(pd.DataFrame(rows), "F1_capital_per_scenario", SUBDIR)

    # Distribucion V_terminal por (label, scenario)
    V_t_rows = []
    for label, g in [("g_mean", res["g_mean"]), ("g_worst", res["g_worst"])]:
        lam, m_ = g
        w, u, v, _ = policies[(lam, m_)]
        for s in range(scenarios.shape[0]):
            cap = simulate_capital_on_scenario(
                w, u, v, scenarios[s], assets, c_base, C0, T_vals,
            )
            V = cap[T_vals[-1]]
            V_t_rows.append({"policy": label, "scenario": s,
                             "V_terminal": float(V),
                             "ret_acum": float(V / C0 - 1)})
    save_csv(pd.DataFrame(V_t_rows), "F3_V_terminal_dist", SUBDIR)


# ================================================================
# G. BACKTEST HISTORICO (diagnostico)
# ================================================================

def section_G_backtest(state):
    print("\n--- G. BACKTEST HISTORICO (fuera del PDF) ---")
    ctx = state["ctx_dl"]
    ctx_opt = state["ctx_opt"]
    res = state["res"]
    policies = state["policies"]

    lam_m, m_m = res["g_mean"]
    w_rg, u_rg, v_rg, _ = policies[(lam_m, m_m)]
    _, w_opt, u_opt, v_opt, _ = solve_portfolio(
        ctx_opt, lambda_riesgo=1.00, costo_mult=1.0,
    )

    cap_opt = simulate_capital_opt(w_opt, u_opt, v_opt, ctx)
    cap_rg  = simulate_capital_opt(w_rg,  u_rg,  v_rg,  ctx)
    cap_rb  = simulate_naive_rb(ctx)
    cap_bh  = simulate_naive_bh(ctx)

    T_vals = ctx["T_vals"]
    C0 = ctx["Capital_inicial"]
    t_f = T_vals[-1]

    rows = []
    for t in T_vals:
        rows.append({"t": int(t),
                     "OPT":     float(cap_opt[t]),
                     "RG_g*":   float(cap_rg [t]),
                     "NaiveRB": float(cap_rb [t]),
                     "NaiveBH": float(cap_bh [t])})
    save_csv(pd.DataFrame(rows), "G1_backtest_historical", SUBDIR)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(T_vals, [cap_opt[t] for t in T_vals], color="#F2B705",
            lw=1.8, label=f"OPT base (lam=1.0)  fin=${cap_opt[t_f]:,.0f}  "
                         f"ret={cap_opt[t_f]/C0-1:+.2%}")
    ax.plot(T_vals, [cap_rg[t]  for t in T_vals], color="#1f77b4",
            lw=1.8, label=f"RG g*_mean (lam={lam_m:.2f}, m={m_m:.1f})  "
                         f"fin=${cap_rg[t_f]:,.0f}  ret={cap_rg[t_f]/C0-1:+.2%}")
    ax.plot(T_vals, [cap_rb[t]  for t in T_vals], color="#8B3A1F",
            lw=1.3, label=f"Naive 50/50 rebal  fin=${cap_rb[t_f]:,.0f}  "
                         f"ret={cap_rb[t_f]/C0-1:+.2%}")
    ax.plot(T_vals, [cap_bh[t]  for t in T_vals], color="#E63946",
            lw=1.3, label=f"Naive 50/50 B&H  fin=${cap_bh[t_f]:,.0f}  "
                         f"ret={cap_bh[t_f]/C0-1:+.2%}")
    ax.axhline(C0, color="#666", ls="--", lw=0.8, label=f"C0=${C0:,.0f}")
    ax.set_title("Backtest historico (diagnostico, fuera del PDF)")
    ax.set_xlabel("t"); ax.set_ylabel("Capital")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    save_fig(fig, "G1_backtest_historical", SUBDIR)

    print(f"  OPT base:   ${cap_opt[t_f]:,.2f}  ({cap_opt[t_f]/C0 - 1:+.2%})")
    print(f"  RG g*_mean: ${cap_rg [t_f]:,.2f}  ({cap_rg [t_f]/C0 - 1:+.2%})")
    print(f"  Naive RB:   ${cap_rb [t_f]:,.2f}  ({cap_rb [t_f]/C0 - 1:+.2%})")
    print(f"  Naive BH:   ${cap_bh [t_f]:,.2f}  ({cap_bh [t_f]/C0 - 1:+.2%})")

    # Resumen final consolidado
    final = {
        "Capital_inicial": float(C0),
        "OPT_cap_final":    float(cap_opt[t_f]),
        "OPT_ret":          float(cap_opt[t_f]/C0 - 1),
        "RG_cap_final":     float(cap_rg[t_f]),
        "RG_ret":           float(cap_rg[t_f]/C0 - 1),
        "NaiveRB_cap_final": float(cap_rb[t_f]),
        "NaiveRB_ret":      float(cap_rb[t_f]/C0 - 1),
        "NaiveBH_cap_final": float(cap_bh[t_f]),
        "NaiveBH_ret":      float(cap_bh[t_f]/C0 - 1),
        "g_mean_lambda":    float(lam_m),
        "g_mean_m":         float(m_m),
        "g_mean_regret":    float(res["g_mean_metric"]),
        "V_mean_avg_scenarios": float(res["V_table"].loc[(lam_m, m_m)].mean()),
        "V_min_g_mean":     float(res["V_table"].loc[(lam_m, m_m)].min()),
        "V_max_g_mean":     float(res["V_table"].loc[(lam_m, m_m)].max()),
    }
    save_csv(pd.DataFrame([final]), "Z_resumen_final", SUBDIR)


# ================================================================
# Main
# ================================================================

def main():
    state = setup()
    section_A_scenarios(state)
    section_B_pbull(state)
    section_C_moments(state)
    section_D_regret(state)
    section_E_policies(state)
    section_F_capital(state)
    section_G_backtest(state)
    print("\n" + "=" * 70)
    print(f"Done. Outputs en inspeccion/{SUBDIR}_out/")
    print("=" * 70)


if __name__ == "__main__":
    main()
