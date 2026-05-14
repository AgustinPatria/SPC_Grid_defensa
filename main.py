"""
Punto de entrada unico del pipeline SPC_Grid3.

Corre el flujo completo de punta a punta:
  1. Reentrena el LSTM cuantilico (con la `DLConfig` por defecto en config.py)
     y sobrescribe el checkpoint en `models/decile_predictor.pt`.
  2. Construye el contexto DL (mu_hat/sigma_hat historicos + p_dl(t) +
     5 escenarios representativos) sobre el horizonte forward.
  3. Resuelve la grilla `(lambda, m)` con GAMSPy + IPOPT, simula el capital
     terminal por escenario, y selecciona g*_mean (ec. 23) y g*_worst (ec. 24).
  4. Persiste tablas en `resultados/` y guarda la curva de capital de g*_mean.

Uso:
    python main.py
"""
from config import (
    CHECKPOINT_PATH,
    DATA_DIR,
    DLConfig,
    LAMBDA_GRID,
    M_GRID,
    N_CANDIDATES,
    N_SCENARIOS,
    RESULTS_DIR,
    SCENARIO_SEED,
    SUMMARY_ASSET,
    T_HORIZON,
)
from Regret_Grid import (
    build_dl_context,
    compute_regret_and_select,
    plot_capital_curves,
    run_historical_backtest,
    run_regret_grid,
)
from dl.prediccion_deciles import (
    DLConfig as _DLConfigAlias,  # noqa: F401  (asegura que el dataclass quede importable)
    save_checkpoint,
    train_deciles,
)


# ================================================================
# 1) Reentrenamiento del LSTM
# ================================================================

def train_and_save() -> None:
    config = DLConfig()
    print("=" * 70)
    print("ENTRENAMIENTO LSTM CUANTILICO")
    print("=" * 70)
    print(f"  H={config.H}  seeds={config.seeds}  deciles={config.n_quantiles}")
    print(f"  epochs={config.epochs}  patience={config.patience}  lr={config.lr}")
    print("-" * 70)

    result = train_deciles(config)
    save_checkpoint(result, CHECKPOINT_PATH)

    print("-" * 70)
    print(f"  best_valid = {result.best_valid:.6f}  (seed={result.best_seed})")
    print(f"  checkpoint guardado en: {CHECKPOINT_PATH}")


# ================================================================
# 2) Regret-grid completo
# ================================================================

def run_regret_pipeline() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("REGRET-GRID — DL -> optimizador -> seleccion (lambda, m)")
    print("=" * 70)
    print("Cargando datos y construyendo contexto DL ...")
    ctx = build_dl_context(
        data_dir=DATA_DIR,
        checkpoint_path=CHECKPOINT_PATH,
        T=T_HORIZON,
        N_candidates=N_CANDIDATES,
        n_scenarios=N_SCENARIOS,
        seed=SCENARIO_SEED,
        summary_asset=SUMMARY_ASSET,
    )
    print(f"  Assets     : {ctx['assets']}")
    print(f"  T          : {ctx['nT']} periodos forward (t1..t{ctx['nT']})")
    print(f"  Scenarios  : {ctx['scenarios'].shape} (S, T, A)")
    for i in ctx["assets"]:
        col = ctx["p_dl"][i]["bull"]
        print(f"  p_bull {i:<7}: min={col.min():.3f}  max={col.max():.3f}  "
              f"mean={col.mean():.3f}")

    lambda_grid = list(LAMBDA_GRID)
    m_grid      = list(M_GRID)

    print("\n" + "-" * 70)
    print(f"Corriendo {len(lambda_grid)}x{len(m_grid)}="
          f"{len(lambda_grid) * len(m_grid)} puntos x "
          f"{ctx['scenarios'].shape[0]} escenarios")
    print("-" * 70)
    V_df, policies = run_regret_grid(ctx, lambda_grid, m_grid)

    res = compute_regret_and_select(V_df)

    print("\n" + "=" * 70)
    print("RESULTADOS")
    print("=" * 70)

    print("\n--- V[g, s] — capital terminal por (lambda, m) y escenario ---")
    print(res["V_table"].to_string(float_format="${:,.2f}".format))

    print("\n--- R[g, s] = V_best_s - V[g, s] ---")
    print(res["R_table"].to_string(float_format="${:,.2f}".format))

    print("\n--- Resumen de regret por g ---")
    print(res["regret_summary"].to_string(float_format="${:,.2f}".format))

    lam_m, m_m = res["g_mean"]
    lam_w, m_w = res["g_worst"]
    C0 = ctx["Capital_inicial"]
    V_mean_row  = res["V_table"].loc[(lam_m, m_m)]
    V_worst_row = res["V_table"].loc[(lam_w, m_w)]
    print("\n--- Seleccion de g* ---")
    print(f"  g*_mean  (ec. 23): lambda={lam_m:.2f}  m={m_m:.1f}  "
          f"mean_regret=${res['g_mean_metric']:,.2f}")
    print(f"      V: mean=${V_mean_row.mean():>12,.2f}  "
          f"worst=${V_mean_row.min():>12,.2f}  "
          f"best=${V_mean_row.max():>12,.2f}  "
          f"(capital inicial=${C0:,.2f})")
    print(f"      retorno promedio sobre escenarios = {V_mean_row.mean()/C0 - 1:+.2%}")
    print(f"  g*_worst (ec. 24): lambda={lam_w:.2f}  m={m_w:.1f}  "
          f"worst_regret=${res['g_worst_metric']:,.2f}")
    print(f"      V: mean=${V_worst_row.mean():>12,.2f}  "
          f"worst=${V_worst_row.min():>12,.2f}  "
          f"best=${V_worst_row.max():>12,.2f}  "
          f"(capital inicial=${C0:,.2f})")
    print(f"      retorno en el peor escenario     = {V_worst_row.min()/C0 - 1:+.2%}")

    out_V = RESULTS_DIR / "regret_grid_results.csv"
    V_df.to_csv(out_V, index=False)
    print(f"\n  V_df (long)           : {out_V}")

    out_R = RESULTS_DIR / "regret_table.csv"
    res["R_table"].to_csv(out_R)
    print(f"  Tabla de regret       : {out_R}")

    out_summary = RESULTS_DIR / "regret_summary.csv"
    res["regret_summary"].to_csv(out_summary)
    print(f"  Resumen por g         : {out_summary}")

    w_star, u_star, v_star, _z = policies[(lam_m, m_m)]
    plot_capital_curves(
        w_star, u_star, v_star, ctx,
        title=f"Capital por escenario con g*_mean (lambda={lam_m:.2f}, m={m_m:.1f})",
        out_path=RESULTS_DIR / "regret_capital_curves.png",
    )

    # --- Backtest historico: OPT vs Naive vs Regret-Grid g*_mean ---
    run_historical_backtest(
        w_star, u_star, v_star, lam_m, m_m,
        V_mean_row=V_mean_row,
        n_scenarios=ctx["scenarios"].shape[0],
        data_dir=DATA_DIR,
        out_path=RESULTS_DIR / "evolucion_capital.png",
    )


# ================================================================
# 3) Bloque principal
# ================================================================

if __name__ == "__main__":
    train_and_save()
    run_regret_pipeline()
    print("\n" + "=" * 70)
    print("Pipeline completo.")
    print("=" * 70)
