"""Cuadro final consolidado para la memoria.

Compara 5 politicas en backtest historico (r_hist) y en los 5 escenarios DL:

  1. OPT base          mu/sigma historico (sin DL); regret-grid sobre escenarios DL
  2. DL pre-unif.      mu_mix = p_dl x mu_hat_historico (formulacion original
                       del PDF, hereda momentos historicos via p_dl)
  3. DL post-unif.     mu_mix = mean(candidatos LSTM(t)); sigma_mix = cov(...)
                       (opcion a: optimizador y simulacion en el mismo mundo)
  4. Naive BH 50/50    buy & hold sin costos
  5. Naive RB 50/50    rebalanceo perfecto a 50/50 con costos

Para las 3 politicas optimizadas, se corre regret-grid sobre la grilla
estandar y se elige g*_mean. Las naive no tienen hiperparametros.

Corre con:
    python -m experimentos.cuadro_final

Salidas en `experimentos/cuadro_final_out/`:
    resultados.csv    tabla larga (politica x metrica)
    evolucion.png     curva de capital historico (las 5 politicas)
    escenarios.png    V terminal por escenario y politica (barras agrupadas)
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import (
    CHECKPOINT_PATH,
    DATA_DIR,
    LAMBDA_GRID,
    M_GRID,
    N_CANDIDATES,
    N_SCENARIOS,
    PROB_CSV,
    REGIMES,
    SCENARIO_SEED,
    SUMMARY_ASSET,
    T_HORIZON,
)
from Regret_Grid import (
    _compute_hist_moments,
    build_dl_context,
    compute_regret_and_select,
    load_market_data,
    run_regret_grid,
    simulate_capital_on_scenario,
    simulate_capital_opt,
    simulate_naive_bh,
    simulate_naive_rb,
)


OUT_DIR = PROJECT_ROOT / "experimentos" / "cuadro_final_out"


# ================================================================
# Helpers
# ================================================================
def build_dl_context_pre_unif(data_dir, checkpoint_path):
    """Reconstruye el ctx DL con la formulacion ANTERIOR a la unificacion:
    mu_mix(i, t) = sum_k p_dl(i, t, k) * mu_hat_historico(i, k).
    Los scenarios siguen siendo los mismos (mismo LSTM, mismo seed)."""
    post = build_dl_context(
        data_dir=data_dir, checkpoint_path=checkpoint_path,
        T=T_HORIZON, N_candidates=N_CANDIDATES,
        n_scenarios=N_SCENARIOS, seed=SCENARIO_SEED,
        summary_asset=SUMMARY_ASSET,
    )

    assets = post["assets"]
    regimes = list(REGIMES)
    r_hist = post["r"]
    T_vals = post["T_vals"]
    p_dl = post["p_dl"]

    # p_hist desde los CSVs (igual que el codigo viejo).
    p_hist = {}
    for a in assets:
        df = pd.read_csv(Path(data_dir) / PROB_CSV[a])
        df.columns = [c.strip() for c in df.columns]
        df["t"] = df["t"].astype(int)
        p_hist[a] = df.set_index("t")[regimes]

    mu_hat, sigma_hat = _compute_hist_moments(r_hist, p_hist, assets, regimes)

    mu_mix = {i: pd.Series(0.0, index=T_vals) for i in assets}
    sigma_mix = {i: {j: pd.Series(0.0, index=T_vals) for j in assets} for i in assets}
    for i in assets:
        for k in regimes:
            mu_mix[i] = mu_mix[i] + p_dl[i][k] * mu_hat[(i, k)]
    for i in assets:
        for j in assets:
            for k in regimes:
                sigma_mix[i][j] = (sigma_mix[i][j]
                                   + p_dl[i][k] * p_dl[j][k] * sigma_hat[(i, j, k)])
    for i in assets:
        for j in assets:
            sym = 0.5 * (sigma_mix[i][j] + sigma_mix[j][i])
            sigma_mix[i][j] = sym
            sigma_mix[j][i] = sym

    pre = dict(post)
    pre["mu_mix"]    = mu_mix
    pre["sigma_mix"] = sigma_mix
    return pre


def simulate_naive_bh_on_scenario(scenario, n_assets, C0, T_vals):
    """BH 50/50 sobre un escenario (sin costos)."""
    cap = {T_vals[0]: C0}
    w_const = 1.0 / n_assets
    for idx in range(1, len(T_vals)):
        t = T_vals[idx]; t_prev = T_vals[idx - 1]
        r_port = sum(w_const * scenario[idx - 1, ai] for ai in range(n_assets))
        cap[t] = cap[t_prev] * (1.0 + r_port)
    return cap


def simulate_naive_rb_on_scenario(scenario, assets, c_base, C0, T_vals):
    """Rebalanceo a 50/50 con costos sobre un escenario."""
    n_assets = len(assets)
    cap = {T_vals[0]: C0}
    w_target = 1.0 / n_assets
    for idx in range(1, len(T_vals)):
        t = T_vals[idx]; t_prev = T_vals[idx - 1]
        r_port = sum(w_target * scenario[idx - 1, ai] for ai in range(n_assets))
        w_drift = {a: w_target * (1.0 + scenario[idx - 1, ai]) / (1.0 + r_port)
                   for ai, a in enumerate(assets)}
        turn = sum(c_base[a] * abs(w_target - w_drift[a]) for a in assets)
        cap[t] = cap[t_prev] * (1.0 + r_port) - cap[t_prev] * turn
    return cap


def run_and_get_policy(ctx, name, lambda_grid, m_grid):
    """Corre regret-grid sobre ctx, selecciona g*_mean, devuelve policy + metricas."""
    print(f"\n[{name}] regret-grid {len(lambda_grid)}x{len(m_grid)} ...")
    V_df, pols = run_regret_grid(ctx, lambda_grid, m_grid)
    res = compute_regret_and_select(V_df)
    g = res["g_mean"]
    w, u, v, z = pols[g]
    return {
        "g_mean":      g,
        "mean_regret": float(res["g_mean_metric"]),
        "policy":      (w, u, v, z),
    }


def _V_on_scenarios(policy_or_kind, ctx, scenarios):
    """Simula V terminal por escenario. policy_or_kind:
       - (w, u, v) tupla -> politica optima
       - ("BH",) o ("RB",) -> naive."""
    assets = ctx["assets"]
    T_vals = ctx["T_vals"]
    c_base = ctx["c_base"]
    C0     = ctx["Capital_inicial"]
    n_S    = scenarios.shape[0]
    V_s = np.empty(n_S)
    for s in range(n_S):
        if policy_or_kind[0] == "BH":
            cap = simulate_naive_bh_on_scenario(
                scenarios[s], len(assets), C0, T_vals,
            )
        elif policy_or_kind[0] == "RB":
            cap = simulate_naive_rb_on_scenario(
                scenarios[s], assets, c_base, C0, T_vals,
            )
        else:
            w, u, v = policy_or_kind
            cap = simulate_capital_on_scenario(
                w, u, v, scenarios[s], assets, c_base, C0, T_vals,
            )
        V_s[s] = cap[T_vals[-1]]
    return V_s


# ================================================================
# Main
# ================================================================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("CUADRO FINAL DE TESIS - comparativa de pipelines")
    print("=" * 70)
    print(f"  output : {OUT_DIR}")
    print("-" * 70)

    print("Construyendo contextos...")
    opt_ctx  = load_market_data(str(DATA_DIR))
    dl_post  = build_dl_context(
        data_dir=DATA_DIR, checkpoint_path=CHECKPOINT_PATH,
        T=T_HORIZON, N_candidates=N_CANDIDATES,
        n_scenarios=N_SCENARIOS, seed=SCENARIO_SEED,
        summary_asset=SUMMARY_ASSET,
    )
    dl_pre   = build_dl_context_pre_unif(DATA_DIR, CHECKPOINT_PATH)
    hybrid_opt = dict(opt_ctx)
    hybrid_opt["scenarios"] = dl_post["scenarios"]
    print("  contextos OK (OPT base + DL pre-unif + DL post-unif)")

    lambda_grid = list(LAMBDA_GRID)
    m_grid      = list(M_GRID)

    out = {}
    out["OPT base"]     = run_and_get_policy(hybrid_opt, "OPT base", lambda_grid, m_grid)
    out["DL pre-unif"]  = run_and_get_policy(dl_pre,     "DL pre-unif",  lambda_grid, m_grid)
    out["DL post-unif"] = run_and_get_policy(dl_post,    "DL post-unif", lambda_grid, m_grid)

    assets    = dl_post["assets"]
    T_vals    = dl_post["T_vals"]
    C0        = dl_post["Capital_inicial"]
    c_base    = dl_post["c_base"]
    scenarios = dl_post["scenarios"]
    n_S       = scenarios.shape[0]

    # ============================================================
    # Construir tabla consolidada
    # ============================================================
    rows = []
    # Las 3 politicas optimizadas
    for name in ["OPT base", "DL pre-unif", "DL post-unif"]:
        w, u, v, _z = out[name]["policy"]
        cap_h = simulate_capital_opt(w, u, v, dl_post)
        V_h = cap_h[T_vals[-1]]
        V_s = _V_on_scenarios((w, u, v), dl_post, scenarios)
        rows.append({
            "policy":           name,
            "g_mean":           str(out[name]["g_mean"]),
            "mean_regret_$":    out[name]["mean_regret"],
            "V_hist":           float(V_h),
            "ret_hist_%":       float((V_h / C0 - 1) * 100),
            "V_scen_mean":      float(V_s.mean()),
            "V_scen_worst":     float(V_s.min()),
            "V_scen_best":      float(V_s.max()),
            "ret_scen_mean_%":  float((V_s.mean() / C0 - 1) * 100),
            "ret_scen_worst_%": float((V_s.min() / C0 - 1) * 100),
        })
        out[name]["V_h"] = V_h
        out[name]["V_s"] = V_s
        out[name]["cap_h"] = cap_h

    # Naives
    for label, kind, sim_hist in [
        ("Naive BH 50/50", ("BH",), simulate_naive_bh),
        ("Naive RB 50/50", ("RB",), simulate_naive_rb),
    ]:
        cap_h = sim_hist(dl_post)
        V_h = cap_h[T_vals[-1]]
        V_s = _V_on_scenarios(kind, dl_post, scenarios)
        rows.append({
            "policy":           label,
            "g_mean":           "-",
            "mean_regret_$":    float("nan"),
            "V_hist":           float(V_h),
            "ret_hist_%":       float((V_h / C0 - 1) * 100),
            "V_scen_mean":      float(V_s.mean()),
            "V_scen_worst":     float(V_s.min()),
            "V_scen_best":      float(V_s.max()),
            "ret_scen_mean_%":  float((V_s.mean() / C0 - 1) * 100),
            "ret_scen_worst_%": float((V_s.min() / C0 - 1) * 100),
        })
        out[label] = {"cap_h": cap_h, "V_h": V_h, "V_s": V_s}

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "resultados.csv", index=False)

    print("\n" + "=" * 70)
    print("CUADRO FINAL")
    print("=" * 70)
    pretty = df[["policy", "g_mean", "mean_regret_$",
                 "V_hist", "ret_hist_%",
                 "V_scen_mean", "V_scen_worst",
                 "ret_scen_mean_%", "ret_scen_worst_%"]].copy()
    print(pretty.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

    # ============================================================
    # Plot 1: evolucion del capital historico
    # ============================================================
    colors = {
        "OPT base":        "C2",
        "DL pre-unif":     "C3",
        "DL post-unif":    "C0",
        "Naive BH 50/50":  "C7",
        "Naive RB 50/50":  "C8",
    }
    fig, ax = plt.subplots(figsize=(13, 6))
    for name in ["OPT base", "DL pre-unif", "DL post-unif",
                 "Naive BH 50/50", "Naive RB 50/50"]:
        cap = out[name]["cap_h"]
        x = list(T_vals); y = [cap[t] for t in x]
        ls = "-" if name in {"OPT base", "DL pre-unif", "DL post-unif"} else "--"
        ax.plot(x, y, color=colors[name], lw=1.6, ls=ls,
                label=f"{name} (final ${y[-1]:,.0f})")
    ax.axhline(C0, color="grey", lw=0.7, ls=":")
    ax.set_title("Backtest historico: evolucion del capital por politica")
    ax.set_xlabel("t (semana)")
    ax.set_ylabel("Capital ($)")
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "evolucion.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ============================================================
    # Plot 2: V terminal por escenario y politica
    # ============================================================
    fig, ax = plt.subplots(figsize=(13, 6))
    pos = np.arange(n_S)
    width = 0.16
    policies_order = ["OPT base", "DL pre-unif", "DL post-unif",
                      "Naive BH 50/50", "Naive RB 50/50"]
    for pi, name in enumerate(policies_order):
        V_s = out[name]["V_s"]
        ax.bar(pos + width * (pi - 2), V_s, width,
               color=colors[name], label=name)
    ax.axhline(C0, color="grey", ls=":", lw=0.7)
    ax.set_xticks(pos)
    ax.set_xticklabels([f"s={s}" for s in range(n_S)])
    ax.set_xlabel("escenario")
    ax.set_ylabel("V terminal ($)")
    ax.set_title("Capital terminal por escenario y politica (5 escenarios DL)")
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "escenarios.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"\n--> resultados : {OUT_DIR / 'resultados.csv'}")
    print(f"--> evolucion  : {OUT_DIR / 'evolucion.png'}")
    print(f"--> escenarios : {OUT_DIR / 'escenarios.png'}")


if __name__ == "__main__":
    main()
