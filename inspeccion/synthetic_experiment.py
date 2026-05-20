"""Experimento con dataset sintetico: la prueba "el framework funciona con buena data".

PREGUNTA: si la LSTM tuviera una senal real y clara para aprender, y mu_mix
variara en el tiempo, el regret-grid produciria politicas distintas y un
rebalanceo real entre SPX y CMC?

Genera un dataset sintetico de T=163 semanas con:
- Senal periodica clara: mu(t) = drift + A*sin(2*pi*t/30)  (4 ciclos en 163w)
- SPX: amplitud +-1%/sem, vol 1.5%/sem  (regimes bull/bear bien definidos)
- CMC: amplitud +-3.5%/sem, vol 4%/sem (mas volatil pero con upside real)
- Correlacion SPX-CMC: 0.5 (realista)
- p_hist generada coherentemente con la senal (probabilidad de bull en cada t)

Pipeline completo:
1. Genera data sintetica en inspeccion/synthetic_experiment_out/data/
2. Entrena LSTM sobre esa data, guarda en inspeccion/synthetic_experiment_out/models/
3. Corre regret-grid con los defaults actuales (rollout + p_sign)
4. Compara con resultados sobre data real

Outputs en inspeccion/synthetic_experiment_out/:
  data/                    CSVs sinteticos (4 archivos, mismo formato que data/)
  models/                  LSTM entrenada sobre sintetica
  S1_synthetic_data.csv/png    Visualizacion de la data generada
  S2_pbull_compare.csv/png     p_bull(t) sintetica vs real
  S3_mumix_compare.csv/png     mu_mix(t) sintetica vs real
  S4_w_policy.csv/png          politica w(t) bajo g*_mean
  S5_grid_compare.csv/png      regret-grid sintetica vs real
  S6_capital_scenarios.csv/png capital por escenario
  Z_resumen.csv               resumen comparativo
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
    DATA_DIR,
    DLConfig,
    LAMBDA_GRID,
    M_GRID,
    N_CANDIDATES,
    N_SCENARIOS,
    SCENARIO_SEED,
    SUMMARY_ASSET,
    T_HORIZON,
)
from dl.prediccion_deciles import (
    load_checkpoint,
    save_checkpoint,
    train_deciles,
)
from Regret_Grid import (
    build_dl_context,
    compute_regret_and_select,
    load_market_data,
    run_regret_grid,
    simulate_capital_on_scenario,
    simulate_capital_opt,
    simulate_naive_bh,
    simulate_naive_rb,
    solve_portfolio,
)


SUBDIR = "synthetic_experiment"
OUT_DIR = _THIS_DIR / f"{SUBDIR}_out"
SYNTHETIC_DATA_DIR = OUT_DIR / "data"
SYNTHETIC_CHECKPOINT = OUT_DIR / "models" / "lstm_synthetic.pt"

# Parametros del proceso sintetico
T_SYN = T_HORIZON     # 163, mismo tamano que el dataset real
SYN_SEED = 7
PERIOD = 30           # semanas por ciclo regimen
# SPX: bull/bear simetricos chicos, vol moderada
DRIFT_SPX = 0.001
AMP_SPX = 0.010
SIGMA_SPX = 0.015
# CMC: bull/bear simetricos GRANDES, vol mayor (capta el "upside real" del cripto)
DRIFT_CMC = 0.002
AMP_CMC = 0.035
SIGMA_CMC = 0.040
RHO = 0.5


# ================================================================
# 1) Generar dataset sintetico
# ================================================================

def generate_synthetic_data():
    """Genera CSVs sinteticos en SYNTHETIC_DATA_DIR con formato compatible
    con el pipeline.
    """
    SYNTHETIC_DATA_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SYN_SEED)
    t = np.arange(1, T_SYN + 1)

    # Senal periodica
    phase = 0.0
    mu_spx_t = DRIFT_SPX + AMP_SPX * np.sin(2 * np.pi * t / PERIOD + phase)
    mu_cmc_t = DRIFT_CMC + AMP_CMC * np.sin(2 * np.pi * t / PERIOD + phase)

    # Noise correlacionado
    cov = np.array([
        [SIGMA_SPX ** 2,            RHO * SIGMA_SPX * SIGMA_CMC],
        [RHO * SIGMA_SPX * SIGMA_CMC, SIGMA_CMC ** 2],
    ])
    L = np.linalg.cholesky(cov)
    eps = rng.standard_normal(size=(T_SYN, 2)) @ L.T

    r_spx = (mu_spx_t + eps[:, 0]).astype(np.float32)
    r_cmc = (mu_cmc_t + eps[:, 1]).astype(np.float32)

    # p_hist coherente con la senal: P(r >= 0 | regime) = Phi(mu/sigma)
    from scipy.stats import norm
    p_bull_spx = norm.cdf(mu_spx_t / SIGMA_SPX)
    p_bull_cmc = norm.cdf(mu_cmc_t / SIGMA_CMC)

    # Save en formato esperado
    pd.DataFrame({"t": t, "ret_semanal_spx": r_spx}).to_csv(
        SYNTHETIC_DATA_DIR / "ret_semanal_spx.csv", index=False)
    pd.DataFrame({"t": t, "ret_semanal_cmc200": r_cmc}).to_csv(
        SYNTHETIC_DATA_DIR / "ret_semanal_cmc200.csv", index=False)
    pd.DataFrame({
        "t": t,
        "bear": 1.0 - p_bull_spx,
        "bull": p_bull_spx,
    }).to_csv(SYNTHETIC_DATA_DIR / "prob_spx.csv", index=False)
    pd.DataFrame({
        "t": t,
        "bear": 1.0 - p_bull_cmc,
        "bull": p_bull_cmc,
    }).to_csv(SYNTHETIC_DATA_DIR / "prob_cmc200.csv", index=False)

    print(f"  Generado dataset sintetico en {SYNTHETIC_DATA_DIR}")
    print(f"    SPX: mean ret = {r_spx.mean()*100:+.3f}%/sem,  "
          f"std = {r_spx.std()*100:.2f}%/sem,  "
          f"frac r>=0 = {(r_spx >= 0).mean()*100:.1f}%")
    print(f"    CMC: mean ret = {r_cmc.mean()*100:+.3f}%/sem,  "
          f"std = {r_cmc.std()*100:.2f}%/sem,  "
          f"frac r>=0 = {(r_cmc >= 0).mean()*100:.1f}%")
    print(f"    p_bull(SPX): mean = {p_bull_spx.mean():.3f},  "
          f"std = {p_bull_spx.std():.3f},  "
          f"rango [{p_bull_spx.min():.3f}, {p_bull_spx.max():.3f}]")
    print(f"    p_bull(CMC): mean = {p_bull_cmc.mean():.3f},  "
          f"std = {p_bull_cmc.std():.3f},  "
          f"rango [{p_bull_cmc.min():.3f}, {p_bull_cmc.max():.3f}]")

    return {"r_spx": r_spx, "r_cmc": r_cmc,
            "p_bull_spx": p_bull_spx, "p_bull_cmc": p_bull_cmc,
            "mu_spx_t": mu_spx_t, "mu_cmc_t": mu_cmc_t,
            "t": t}


def viz_synthetic_data(syn):
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)

    # Retornos
    ax = axes[0]
    ax.plot(syn["t"], syn["r_spx"] * 100, color="#1f77b4", lw=1.0,
            alpha=0.7, label=f"SPX (mean={syn['r_spx'].mean()*100:+.2f}%)")
    ax.plot(syn["t"], syn["mu_spx_t"] * 100, color="#1f77b4", lw=2.0, ls="--",
            label="mu_SPX(t) (regimen)")
    ax.plot(syn["t"], syn["r_cmc"] * 100, color="#E63946", lw=1.0,
            alpha=0.7, label=f"CMC (mean={syn['r_cmc'].mean()*100:+.2f}%)")
    ax.plot(syn["t"], syn["mu_cmc_t"] * 100, color="#E63946", lw=2.0, ls="--",
            label="mu_CMC(t) (regimen)")
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_title(f"Retornos sinteticos generados (T={T_SYN}, period={PERIOD}w, "
                 f"4 ciclos completos en horizonte)")
    ax.set_ylabel("retorno semanal [%]")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9, loc="best")

    # Capital cumulativo
    ax = axes[1]
    cum_spx = np.cumprod(1.0 + syn["r_spx"]) - 1.0
    cum_cmc = np.cumprod(1.0 + syn["r_cmc"]) - 1.0
    ax.plot(syn["t"], cum_spx * 100, color="#1f77b4", lw=1.5,
            label=f"SPX cum (final {cum_spx[-1]*100:+.1f}%)")
    ax.plot(syn["t"], cum_cmc * 100, color="#E63946", lw=1.5,
            label=f"CMC cum (final {cum_cmc[-1]*100:+.1f}%)")
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_title("Retorno acumulado sintetico")
    ax.set_ylabel("retorno acumulado [%]")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9, loc="best")

    # p_hist
    ax = axes[2]
    ax.plot(syn["t"], syn["p_bull_spx"], color="#1f77b4", lw=1.5,
            label=f"p_bull(SPX) (mean {syn['p_bull_spx'].mean():.2f})")
    ax.plot(syn["t"], syn["p_bull_cmc"], color="#E63946", lw=1.5,
            label=f"p_bull(CMC) (mean {syn['p_bull_cmc'].mean():.2f})")
    ax.axhline(0.5, color="grey", ls="--", lw=0.6, label="0.5")
    ax.set_title("p_hist sintetica (probabilidad de bull por regimen)")
    ax.set_xlabel("t [semanas]")
    ax.set_ylabel("p_bull")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9, loc="best")

    save_fig(fig, "S1_synthetic_data", SUBDIR)
    save_csv(pd.DataFrame({
        "t": syn["t"],
        "r_spx_pct":     syn["r_spx"] * 100,
        "r_cmc_pct":     syn["r_cmc"] * 100,
        "mu_spx_t_pct":  syn["mu_spx_t"] * 100,
        "mu_cmc_t_pct":  syn["mu_cmc_t"] * 100,
        "p_bull_spx":    syn["p_bull_spx"],
        "p_bull_cmc":    syn["p_bull_cmc"],
    }), "S1_synthetic_data", SUBDIR)


# ================================================================
# 2) Entrenar LSTM sobre sintetica
# ================================================================

def train_lstm_synthetic():
    SYNTHETIC_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nEntrenando LSTM sobre {SYNTHETIC_DATA_DIR}...")
    config = DLConfig()
    result = train_deciles(config, data_dir=SYNTHETIC_DATA_DIR)
    save_checkpoint(result, SYNTHETIC_CHECKPOINT)
    print(f"  best_valid = {result.best_valid:.6f}  (seed={result.best_seed})")
    print(f"  checkpoint: {SYNTHETIC_CHECKPOINT}")
    return result


# ================================================================
# 3) Pipeline completo sobre sintetica
# ================================================================

def run_pipeline_synthetic():
    print("\nConstruyendo contexto DL sintetico...")
    ctx_dl = build_dl_context(
        data_dir=SYNTHETIC_DATA_DIR, checkpoint_path=SYNTHETIC_CHECKPOINT,
        T=T_HORIZON, N_candidates=N_CANDIDATES, n_scenarios=N_SCENARIOS,
        seed=SCENARIO_SEED, summary_asset=SUMMARY_ASSET,
    )
    ctx_opt = load_market_data(str(SYNTHETIC_DATA_DIR))
    print(f"  p_dl(SPX): mean={ctx_dl['p_dl']['SPX']['bull'].mean():.3f}, "
          f"std={ctx_dl['p_dl']['SPX']['bull'].std():.3f}")
    print(f"  p_dl(CMC): mean={ctx_dl['p_dl']['CMC200']['bull'].mean():.3f}, "
          f"std={ctx_dl['p_dl']['CMC200']['bull'].std():.3f}")
    print(f"  mu_mix(SPX): mean={ctx_dl['mu_mix']['SPX'].mean()*100:+.3f}%, "
          f"std={ctx_dl['mu_mix']['SPX'].std()*100:.4f}%")
    print(f"  mu_mix(CMC): mean={ctx_dl['mu_mix']['CMC200'].mean()*100:+.3f}%, "
          f"std={ctx_dl['mu_mix']['CMC200'].std()*100:.4f}%")

    print(f"\nCorriendo regret-grid {len(LAMBDA_GRID)}x{len(M_GRID)} = "
          f"{len(LAMBDA_GRID)*len(M_GRID)} solves...")
    V_df, policies = run_regret_grid(ctx_dl, list(LAMBDA_GRID), list(M_GRID))
    res = compute_regret_and_select(V_df)
    return {"ctx_dl": ctx_dl, "ctx_opt": ctx_opt,
            "V_df": V_df, "policies": policies, "res": res}


# ================================================================
# 4) Comparacion sintetica vs real
# ================================================================

def diag_compare_pbull(state_syn):
    print("\n--- Comparando p_bull(t) sintetica vs real ---")
    ctx_real = build_dl_context(  # uses default DATA_DIR + checkpoint real
        data_dir=DATA_DIR,
        checkpoint_path=_PROJECT_ROOT / "models" / "decile_predictor.pt",
        T=T_HORIZON,
    )

    assets = state_syn["ctx_dl"]["assets"]
    T_vals = state_syn["ctx_dl"]["T_vals"]
    rows = []
    fig, axes = plt.subplots(len(assets), 1, figsize=(13, 4 * len(assets)),
                             sharex=True)
    if len(assets) == 1:
        axes = [axes]
    for ai, a in enumerate(assets):
        ax = axes[ai]
        p_syn  = state_syn["ctx_dl"]["p_dl"][a]["bull"]
        p_real = ctx_real["p_dl"][a]["bull"]
        ax.plot(T_vals, p_syn.values, color="#1f77b4", lw=1.4,
                label=f"sintetica (mean={p_syn.mean():.3f}, std={p_syn.std():.3f})")
        ax.plot(T_vals, p_real.values, color="#E63946", lw=1.2, alpha=0.7,
                label=f"real (mean={p_real.mean():.3f}, std={p_real.std():.3f})")
        ax.axhline(0.5, color="grey", ls="--", lw=0.6)
        ax.set_title(f"p_bull({a}, t) — LSTM sintetica vs LSTM real (rollout)")
        ax.set_xlabel("t")
        ax.set_ylabel(f"p_bull({a})")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9, loc="best")
        for t, vs, vr in zip(T_vals, p_syn.values, p_real.values):
            rows.append({"asset": a, "t": int(t),
                         "p_bull_synthetic": float(vs),
                         "p_bull_real": float(vr)})
    save_fig(fig, "S2_pbull_compare", SUBDIR)
    save_csv(pd.DataFrame(rows), "S2_pbull_compare", SUBDIR)
    return ctx_real


def diag_compare_mumix(state_syn, ctx_real):
    assets = state_syn["ctx_dl"]["assets"]
    T_vals = state_syn["ctx_dl"]["T_vals"]
    rows = []
    fig, axes = plt.subplots(len(assets), 1, figsize=(13, 4 * len(assets)),
                             sharex=True)
    if len(assets) == 1:
        axes = [axes]
    for ai, a in enumerate(assets):
        ax = axes[ai]
        mu_syn  = state_syn["ctx_dl"]["mu_mix"][a]
        mu_real = ctx_real["mu_mix"][a]
        ax.plot(T_vals, mu_syn.values * 100, color="#1f77b4", lw=1.4,
                label=f"sintetica (std={mu_syn.std()*100:.3f}%)")
        ax.plot(T_vals, mu_real.values * 100, color="#E63946", lw=1.2, alpha=0.7,
                label=f"real (std={mu_real.std()*100:.4f}%)")
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_title(f"mu_mix({a}, t) — sintetica vs real")
        ax.set_xlabel("t"); ax.set_ylabel(f"mu_mix({a}) [%/sem]")
        ax.grid(True, alpha=0.25); ax.legend(fontsize=9, loc="best")
        for t, vs, vr in zip(T_vals, mu_syn.values, mu_real.values):
            rows.append({"asset": a, "t": int(t),
                         "mu_mix_synthetic_pct": float(vs * 100),
                         "mu_mix_real_pct":      float(vr * 100)})
    save_fig(fig, "S3_mumix_compare", SUBDIR)
    save_csv(pd.DataFrame(rows), "S3_mumix_compare", SUBDIR)


def diag_policy(state_syn):
    res = state_syn["res"]
    lam_m, m_m = res["g_mean"]
    w, u, v, _ = state_syn["policies"][(lam_m, m_m)]
    ctx = state_syn["ctx_dl"]
    assets = ctx["assets"]
    T_vals = ctx["T_vals"]

    rows = []
    fig, axes = plt.subplots(len(assets), 1, figsize=(13, 4 * len(assets)),
                             sharex=True)
    if len(assets) == 1:
        axes = [axes]
    for ai, a in enumerate(assets):
        ax = axes[ai]
        w_t = [w[(a, t)] for t in T_vals]
        ax.plot(T_vals, w_t, color="#1f77b4", lw=1.5,
                label=f"w({a}, t)  mean={np.mean(w_t):.3f}, std={np.std(w_t):.3f}")
        ax.axhline(0.5, color="grey", ls="--", lw=0.6, label="w0=0.5")
        ax.set_title(f"Politica w({a}, t) bajo g*_mean=(lam={lam_m:.2f}, m={m_m:.1f}) "
                     f"[sintetica]")
        ax.set_xlabel("t"); ax.set_ylabel(f"w({a})")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.25); ax.legend(fontsize=9)
        for t, ww in zip(T_vals, w_t):
            rows.append({"asset": a, "t": int(t), "w": float(ww)})
    save_fig(fig, "S4_w_policy", SUBDIR)
    save_csv(pd.DataFrame(rows), "S4_w_policy", SUBDIR)


def diag_grid(state_syn):
    res = state_syn["res"]
    rows = []
    for (lam, m_), V_row in res["V_table"].iterrows():
        rows.append({
            "lambda": float(lam), "m": float(m_),
            "V_mean": float(V_row.mean()),
            "V_min":  float(V_row.min()),
            "V_max":  float(V_row.max()),
            "regret_mean": float(res["regret_summary"].loc[(lam, m_), "mean_regret"]),
        })
    save_csv(pd.DataFrame(rows), "S5_grid_compare", SUBDIR)

    R_tab = res["R_table"]
    mean_R = R_tab.mean(axis=1).unstack("m").sort_index().sort_index(axis=1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.imshow(mean_R.values, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(len(mean_R.columns)))
    ax.set_xticklabels([f"{m:.1f}" for m in mean_R.columns])
    ax.set_yticks(range(len(mean_R.index)))
    ax.set_yticklabels([f"{l:.2f}" for l in mean_R.index])
    ax.set_xlabel("m"); ax.set_ylabel("lambda")
    lam_m, m_m = res["g_mean"]
    ax.set_title(f"Regret-grid SINTETICA - mean(R) por g\n"
                 f"g*_mean=({lam_m:.2f},{m_m:.1f})  regret=${res['g_mean_metric']:,.0f}")
    for i in range(mean_R.shape[0]):
        for j in range(mean_R.shape[1]):
            ax.text(j, i, f"${mean_R.values[i, j]:,.0f}",
                    ha="center", va="center", fontsize=8)
    j_opt = list(mean_R.columns).index(m_m)
    i_opt = list(mean_R.index).index(lam_m)
    ax.scatter([j_opt], [i_opt], s=200, marker="*", color="cyan",
               edgecolor="k", linewidth=1.5, zorder=3)
    save_fig(fig, "S5_grid_compare", SUBDIR)


def diag_capital(state_syn):
    ctx = state_syn["ctx_dl"]
    res = state_syn["res"]
    assets = ctx["assets"]
    T_vals = ctx["T_vals"]
    c_base = ctx["c_base"]
    C0 = ctx["Capital_inicial"]
    scenarios = ctx["scenarios"]

    lam_m, m_m = res["g_mean"]
    w, u, v, _ = state_syn["policies"][(lam_m, m_m)]

    rows = []
    fig, ax = plt.subplots(figsize=(11, 5))
    cmap = plt.get_cmap("RdYlGn")
    for s in range(scenarios.shape[0]):
        cap = simulate_capital_on_scenario(
            w, u, v, scenarios[s], assets, c_base, C0, T_vals,
        )
        color = cmap(s / max(scenarios.shape[0] - 1, 1))
        cap_vals = [cap[t] for t in T_vals]
        ax.plot(T_vals, cap_vals, color=color, lw=1.5,
                label=f"s={s}  V=${cap[T_vals[-1]]:,.0f}  "
                      f"ret={cap[T_vals[-1]]/C0 - 1:+.1%}")
        for t in T_vals:
            rows.append({"scenario": s, "t": int(t),
                         "capital": float(cap[t])})
    ax.axhline(C0, color="#666", ls="--", lw=0.8, label=f"C0=${C0:,.0f}")
    ax.set_title(f"Capital bajo g*_mean en los 5 escenarios DL SINTETICOS")
    ax.set_xlabel("t"); ax.set_ylabel("Capital")
    ax.grid(True, alpha=0.25); ax.legend(fontsize=8, loc="best")
    save_fig(fig, "S6_capital_scenarios", SUBDIR)
    save_csv(pd.DataFrame(rows), "S6_capital_scenarios", SUBDIR)


def diag_backtest(state_syn):
    ctx = state_syn["ctx_dl"]
    ctx_opt = state_syn["ctx_opt"]
    res = state_syn["res"]
    lam_m, m_m = res["g_mean"]
    w_rg, u_rg, v_rg, _ = state_syn["policies"][(lam_m, m_m)]
    _, w_opt, u_opt, v_opt, _ = solve_portfolio(
        ctx_opt, lambda_riesgo=1.00, costo_mult=1.0,
    )
    cap_opt = simulate_capital_opt(w_opt, u_opt, v_opt, ctx)
    cap_rg  = simulate_capital_opt(w_rg, u_rg, v_rg, ctx)
    cap_rb  = simulate_naive_rb(ctx)
    cap_bh  = simulate_naive_bh(ctx)
    T_vals = ctx["T_vals"]
    C0 = ctx["Capital_inicial"]
    t_f = T_vals[-1]

    rows = [{"politica": "OPT base",     "cap_final": float(cap_opt[t_f]),
             "ret_acum": float(cap_opt[t_f]/C0 - 1)},
            {"politica": "RG g*_mean",   "cap_final": float(cap_rg [t_f]),
             "ret_acum": float(cap_rg [t_f]/C0 - 1)},
            {"politica": "Naive RB",     "cap_final": float(cap_rb [t_f]),
             "ret_acum": float(cap_rb [t_f]/C0 - 1)},
            {"politica": "Naive BH",     "cap_final": float(cap_bh [t_f]),
             "ret_acum": float(cap_bh [t_f]/C0 - 1)}]
    save_csv(pd.DataFrame(rows), "S7_backtest_synthetic", SUBDIR)
    print("\n--- Backtest sobre data sintetica (r realizado) ---")
    for r in rows:
        print(f"  {r['politica']:<15} cap_final=${r['cap_final']:>10,.2f}  "
              f"ret={r['ret_acum']:+.2%}")

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(T_vals, [cap_opt[t] for t in T_vals], color="#F2B705", lw=1.7,
            label=f"OPT base  fin=${cap_opt[t_f]:,.0f}")
    ax.plot(T_vals, [cap_rg [t] for t in T_vals], color="#1f77b4", lw=1.7,
            label=f"RG g*_mean (lam={lam_m:.2f}, m={m_m:.1f})  fin=${cap_rg[t_f]:,.0f}")
    ax.plot(T_vals, [cap_rb [t] for t in T_vals], color="#8B3A1F", lw=1.3,
            label=f"Naive RB  fin=${cap_rb[t_f]:,.0f}")
    ax.plot(T_vals, [cap_bh [t] for t in T_vals], color="#E63946", lw=1.3,
            label=f"Naive BH  fin=${cap_bh[t_f]:,.0f}")
    ax.axhline(C0, color="#666", ls="--", lw=0.8, label=f"C0=${C0:,.0f}")
    ax.set_title("Backtest sobre data SINTETICA realizada")
    ax.set_xlabel("t"); ax.set_ylabel("Capital")
    ax.grid(True, alpha=0.25); ax.legend(fontsize=9, loc="best")
    save_fig(fig, "S7_backtest_synthetic", SUBDIR)


def diag_resumen(state_syn):
    res = state_syn["res"]
    lam_m, m_m = res["g_mean"]
    ctx = state_syn["ctx_dl"]
    V_row = res["V_table"].loc[(lam_m, m_m)]
    C0 = ctx["Capital_inicial"]
    assets = ctx["assets"]

    final = {
        "Capital_inicial":   float(C0),
        "g_mean_lambda":     float(lam_m),
        "g_mean_m":          float(m_m),
        "mean_regret":       float(res["g_mean_metric"]),
        "V_mean_scenarios":  float(V_row.mean()),
        "V_min":             float(V_row.min()),
        "V_max":             float(V_row.max()),
        "ret_avg_scenarios": float(V_row.mean()/C0 - 1),
    }
    for a in assets:
        mu = ctx["mu_mix"][a]
        p  = ctx["p_dl"][a]["bull"]
        w  = state_syn["policies"][(lam_m, m_m)][0]
        w_t = [w[(a, t)] for t in ctx["T_vals"]]
        final[f"mu_mix_{a}_mean_pct"] = float(mu.mean() * 100)
        final[f"mu_mix_{a}_std_pct"]  = float(mu.std() * 100)
        final[f"p_bull_{a}_mean"]     = float(p.mean())
        final[f"p_bull_{a}_std"]      = float(p.std())
        final[f"w_{a}_mean"]          = float(np.mean(w_t))
        final[f"w_{a}_std"]           = float(np.std(w_t))
    save_csv(pd.DataFrame([final]), "Z_resumen", SUBDIR)


# ================================================================
# Main
# ================================================================

def main():
    print("=" * 70)
    print("EXPERIMENTO SINTETICO — pipeline con data de mejor senal")
    print("=" * 70)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    syn = generate_synthetic_data()
    viz_synthetic_data(syn)

    train_lstm_synthetic()

    state_syn = run_pipeline_synthetic()
    ctx_real = diag_compare_pbull(state_syn)
    diag_compare_mumix(state_syn, ctx_real)
    diag_policy(state_syn)
    diag_grid(state_syn)
    diag_capital(state_syn)
    diag_backtest(state_syn)
    diag_resumen(state_syn)

    print("\n" + "=" * 70)
    print(f"Done. Outputs en inspeccion/{SUBDIR}_out/")
    print("=" * 70)


if __name__ == "__main__":
    main()
