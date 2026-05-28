"""Analisis de sensibilidad de mu_hat (tarea #2 del meeting con Juan).

Pregunta: ¿a partir de que gap entre mu_hat(bull) y mu_hat(bear) el
Regret-Grid empieza a ganarle al naive 50/50 en backtest historico?

Diseno:
  - Mantiene fijo: las 15 NNs (cache), los 5 escenarios compartidos,
    sigma_hat (del baseline p_hist), y p_dl(t) por celda.
  - Varia: mu_hat_synth(bull, i) = mean_hist[i] + gap/2
           mu_hat_synth(bear, i) = mean_hist[i] - gap/2
    (preserva la media del activo y solo cambia "cuanto difieren los regimenes").

Output por cada gap:
  - g*_mean (lambda, m) seleccionado
  - V mean sobre los 5 escenarios DL
  - Backtest historico in-sample: cap_final del g*_mean aplicado a r_hist

Uso:    python inspeccion/sensibilidad_mu.py
Salida: inspeccion/sensibilidad_mu_out/  (tabla + plot)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import (                                                    # noqa: E402
    C_BASE, DATA_DIR, DLConfig, LAMBDA_GRID, M_GRID, MODELS_DIR,
    N_CANDIDATES, N_SCENARIOS, REGIMES, SCENARIO_SEED, T_HORIZON, W0,
)
from Regret_Grid import (                                               # noqa: E402
    build_ensemble_model,
    build_per_cell_context,
    build_shared_scenarios,
    compute_regret_and_select,
    load_market_data,
    run_per_cell_regret_grid,
    simulate_capital_opt,
    simulate_naive_bh,
    simulate_naive_rb,
    solve_portfolio,
    train_per_cell_nns,
)

OUT = Path(__file__).resolve().parent / "sensibilidad_mu_out"
OUT.mkdir(exist_ok=True)


# ====================================================================
# Construccion de ctx con mu_hat sintetico
# ====================================================================

def build_ctx_with_custom_mu(
    ctx_baseline: dict,
    mu_hat_synth: dict,
) -> dict:
    """Toma un ctx baseline (build_per_cell_context) y lo muta sustituyendo
    mu_mix por: mu_mix(i, t) = sum_k p_dl(t, k) * mu_hat_synth[(i, k)].

    sigma_mix se PRESERVA del baseline (depende de sigma_hat, que mantenemos).

    ctx_baseline:  output de build_per_cell_context (tiene p_dl, mu_hat, sigma_hat).
    mu_hat_synth:  dict {(asset, regime): float} con valores sinteticos.
    """
    assets  = ctx_baseline["assets"]
    regimes = list(REGIMES)
    T_vals  = ctx_baseline["T_vals"]
    p_dl    = ctx_baseline["p_dl"]

    mu_mix = {i: pd.Series(0.0, index=T_vals) for i in assets}
    for i in assets:
        for k in regimes:
            mu_mix[i] += p_dl[i][k] * mu_hat_synth[(i, k)]

    new_ctx = dict(ctx_baseline)
    new_ctx["mu_mix"]  = mu_mix
    new_ctx["mu_hat"]  = dict(mu_hat_synth)
    # sigma_mix queda como en baseline (no se recomputa).
    return new_ctx


def synthetic_mu_hat(
    r_mean_by_asset: dict,
    gap_pct: float,
    assets: list,
    regimes: list,
) -> dict:
    """Construye mu_hat sintetico con un gap dado entre bull y bear,
    manteniendo el mean del activo.

    r_mean_by_asset:  {asset: mean retorno historico}
    gap_pct:          gap en fraccion (ej. 0.02 = 2%/sem).
    """
    mu_hat = {}
    for i in assets:
        mean_i = r_mean_by_asset[i]
        for k in regimes:
            if k == "bull":
                mu_hat[(i, k)] = mean_i + gap_pct / 2.0
            else:  # bear
                mu_hat[(i, k)] = mean_i - gap_pct / 2.0
    return mu_hat


# ====================================================================
# Pipeline para un gap
# ====================================================================

def run_one_gap(
    gap_pct: float,
    contexts_baseline: dict,
    scenarios: np.ndarray,
    base_ctx: dict,
) -> dict:
    """Re-corre la grilla con mu_hat sintetico para un gap dado.

    Returns: dict con (lambda, m) g*_mean, V_mean sobre escenarios DL,
    backtest_hist del g*_mean.
    """
    assets  = list(base_ctx["assets"])
    regimes = list(REGIMES)
    r_hist  = base_ctx["r"]
    r_mean_by_asset = {i: float(r_hist[i].mean()) for i in assets}

    mu_hat_synth = synthetic_mu_hat(r_mean_by_asset, gap_pct, assets, regimes)

    # Construir contextos con mu_hat sintetico (mismo p_dl, mismo sigma_mix).
    contexts_synth = {
        g: build_ctx_with_custom_mu(ctx, mu_hat_synth)
        for g, ctx in contexts_baseline.items()
    }

    # Resolver grilla con esos contextos.
    V_df, policies = run_per_cell_regret_grid(
        contexts_synth, scenarios, list(LAMBDA_GRID), list(M_GRID),
    )
    res = compute_regret_and_select(V_df)

    lam_m, m_m = res["g_mean"]
    V_mean_row = res["V_table"].loc[(lam_m, m_m)]
    V_mean = float(V_mean_row.mean())

    # Backtest historico in-sample del g*_mean.
    w_star, u_star, v_star, _z = policies[(lam_m, m_m)]
    # Necesitamos un ctx historico (load_market_data) para el backtest.
    cap = simulate_capital_opt(w_star, u_star, v_star, base_ctx)
    cap_final = cap[base_ctx["T_vals"][-1]]
    C0 = base_ctx["Capital_inicial"]

    return {
        "gap_pct":           gap_pct,
        "mu_hat_bull_SPX":   mu_hat_synth[("SPX", "bull")],
        "mu_hat_bear_SPX":   mu_hat_synth[("SPX", "bear")],
        "mu_hat_bull_CMC":   mu_hat_synth[("CMC200", "bull")],
        "mu_hat_bear_CMC":   mu_hat_synth[("CMC200", "bear")],
        "g_mean_lambda":     lam_m,
        "g_mean_m":          m_m,
        "V_mean_dl":         V_mean,
        "backtest_cap":      cap_final,
        "backtest_ret":      cap_final / C0 - 1,
    }


# ====================================================================
# Main
# ====================================================================

def main():
    print("=" * 78)
    print("ANALISIS DE SENSIBILIDAD DE mu_hat")
    print("=" * 78)

    # 1. Cargar las 15 NNs (cache)
    print("\n[1/4] Cargando NNs por celda (cache)...")
    nns = train_per_cell_nns(
        lambda_grid=LAMBDA_GRID, m_grid=M_GRID,
        dl_config=DLConfig(), models_dir=MODELS_DIR, force_retrain=False,
    )

    # 2. Construir contextos baseline (con mu_hat de p_hist)
    print("\n[2/4] Construyendo contextos baseline (mu_hat con p_hist)...")
    contexts_baseline = {
        g: build_per_cell_context(nn=nn, data_dir=DATA_DIR, T=T_HORIZON,
                                    mu_hat_source="p_hist")
        for g, nn in nns.items()
    }

    # 3. Construir escenarios compartidos (1 sola vez, no dependen de mu_hat)
    print("\n[3/4] Construyendo escenarios compartidos (FO-aligned)...")
    ensemble = build_ensemble_model(nns)
    base_ctx = load_market_data(str(DATA_DIR))
    assets = list(base_ctx["assets"])
    r_hist = base_ctx["r"]
    H = ensemble.config.H
    returns_history = np.stack(
        [r_hist[i].sort_index().values[:T_HORIZON] for i in assets], axis=1,
    ).astype(np.float32)
    initial_window = returns_history[-H:, :]
    w_ref  = np.array([W0[i]    for i in assets], dtype=np.float32)
    c_base = np.array([C_BASE[i] for i in assets], dtype=np.float32)
    scenarios = build_shared_scenarios(
        ensemble_nn=ensemble, initial_window=initial_window,
        w_ref=w_ref, c_base=c_base,
        N=N_CANDIDATES, T=T_HORIZON, n_scenarios=N_SCENARIOS,
        seed=SCENARIO_SEED,
    )
    print(f"  scenarios.shape = {scenarios.shape}")

    # 4. Barrido de gaps
    # Gaps en %/semana. 0 = mu plano (peor caso); 8% = enorme (limite alto).
    gaps_pct = [
        0.0,    # baseline plano
        0.005,  # 0.5%
        0.01,   # 1%
        0.02,   # 2%
        0.04,   # 4%
        0.08,   # 8% (cerca del p_sign: SPX gap=3.7%, CMC gap=14%)
        0.12,   # 12%
        0.16,   # 16%
    ]
    print(f"\n[4/4] Barriendo {len(gaps_pct)} gaps de mu_hat...")
    print(f"      r_mean historico: SPX={r_hist['SPX'].mean():+.4%}/sem  "
          f"CMC={r_hist['CMC200'].mean():+.4%}/sem")

    rows = []
    for gap in gaps_pct:
        print(f"\n  --- gap = {gap:+.2%}/sem ---")
        result = run_one_gap(gap, contexts_baseline, scenarios, base_ctx)
        rows.append(result)
        print(f"    g*_mean=({result['g_mean_lambda']:.2f}, "
              f"{result['g_mean_m']:.2f})  "
              f"V_mean_dl=${result['V_mean_dl']:,.0f}  "
              f"backtest={result['backtest_ret']:+.2%}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "sensibilidad_mu_results.csv", index=False)

    # 5. Benchmarks para comparar
    naive_rb_cap = simulate_naive_rb(base_ctx)
    naive_bh_cap = simulate_naive_bh(base_ctx)
    C0 = base_ctx["Capital_inicial"]
    T_last = base_ctx["T_vals"][-1]
    naive_rb_ret = naive_rb_cap[T_last] / C0 - 1
    naive_bh_ret = naive_bh_cap[T_last] / C0 - 1

    # OPT base (lambda=1.0, m=1.0) sobre p_hist HMM
    z_opt, w_opt, u_opt, v_opt, _ = solve_portfolio(
        base_ctx, lambda_riesgo=1.0, costo_mult=1.0,
    )
    opt_cap = simulate_capital_opt(w_opt, u_opt, v_opt, base_ctx)
    opt_ret = opt_cap[T_last] / C0 - 1

    print("\n" + "=" * 78)
    print("RESULTADOS")
    print("=" * 78)
    print("\nTabla de sensibilidad:")
    print(df.to_string(index=False,
                        float_format=lambda x: f"{x:.4f}"))
    print(f"\nBenchmarks (constantes):")
    print(f"  Naive 50/50 rebal       : {naive_rb_ret:+.2%}")
    print(f"  Naive 50/50 buy & hold  : {naive_bh_ret:+.2%}")
    print(f"  OPT base (sin DL)       : {opt_ret:+.2%}")

    # 6. Plot principal: backtest vs gap
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(df["gap_pct"] * 100, df["backtest_ret"] * 100,
            "o-", color="#1f77b4", linewidth=2, markersize=9,
            label="RG g*_mean (backtest hist)")
    ax.axhline(naive_rb_ret * 100, color="#8B3A1F", linewidth=1.5,
               linestyle="--", label=f"Naive 50/50 rebal ({naive_rb_ret:+.1%})")
    ax.axhline(naive_bh_ret * 100, color="#E63946", linewidth=1.5,
               linestyle=":", label=f"Naive 50/50 B&H ({naive_bh_ret:+.1%})")
    ax.axhline(opt_ret * 100, color="#F2B705", linewidth=1.5,
               linestyle="-", alpha=0.7, label=f"OPT base ({opt_ret:+.1%})")
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.4)
    # Anotaciones de (lambda, m) en cada punto
    for _, r in df.iterrows():
        ax.annotate(f"({r['g_mean_lambda']:.1f},{r['g_mean_m']:.2f})",
                    (r["gap_pct"] * 100, r["backtest_ret"] * 100),
                    textcoords="offset points", xytext=(0, 10),
                    fontsize=8, ha="center", color="#1f77b4")
    ax.set_xlabel("gap entre mu_hat(bull) y mu_hat(bear) (%/semana)")
    ax.set_ylabel("Retorno backtest historico in-sample (%)")
    ax.set_title("Sensibilidad del Regret-Grid al gap de mu_hat\n"
                 "(in-sample 163 semanas, anotaciones = g*_mean=(λ, m))")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "sensibilidad_backtest_vs_gap.png", dpi=140)
    plt.close(fig)

    # 7. Plot secundario: V_mean_dl vs gap
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["gap_pct"] * 100, df["V_mean_dl"],
            "o-", color="#2ca02c", linewidth=2, markersize=8)
    ax.axhline(C0, color="black", linewidth=0.6, alpha=0.5,
               linestyle="--", label=f"Capital inicial (${C0:,.0f})")
    ax.set_xlabel("gap mu_hat (%/sem)")
    ax.set_ylabel("V mean sobre 5 escenarios DL ($)")
    ax.set_title("V mean del g*_mean sobre los 5 escenarios DL — vs gap")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "sensibilidad_Vmean_vs_gap.png", dpi=140)
    plt.close(fig)

    # 8. Identificar gap critico (donde RG cruza naive)
    naive_target = naive_rb_ret
    print("\n" + "-" * 78)
    print("GAP CRITICO (donde RG iguala/supera al naive 50/50 rebal):")
    print("-" * 78)
    crossing = df[df["backtest_ret"] >= naive_target]
    if len(crossing) == 0:
        print(f"  El RG NO supera al naive ({naive_target:+.2%}) en ningun gap probado.")
        print(f"  Maximo backtest del RG: {df['backtest_ret'].max():+.2%} "
              f"con gap={df.loc[df['backtest_ret'].idxmax(), 'gap_pct']:+.2%}")
    else:
        gap_critico = crossing.iloc[0]
        print(f"  Primer gap donde RG ≥ naive: gap={gap_critico['gap_pct']:+.2%}/sem  "
              f"(backtest={gap_critico['backtest_ret']:+.2%})")
        print(f"  Con g*=({gap_critico['g_mean_lambda']:.2f}, "
              f"{gap_critico['g_mean_m']:.2f})")

    print(f"\nOutputs: {OUT}")


if __name__ == "__main__":
    main()
