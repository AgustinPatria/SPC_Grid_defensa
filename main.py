"""
Punto de entrada del pipeline SPC_Grid3 (paradigma per-cell NN).

Flujo nuevo:
  1. Entrena UNA NN cuantilica por celda g = (lambda, m) con seed
     deterministica = cell_seed(lam, m). Cachea en models/per_cell/.
  2. Para cada g: construye su contexto propio (mu_mix, sigma_mix) via
     walking + p_hist usando SU NN.
  3. ESCENARIOS COMPARTIDOS: ensemble de las 15 NNs (promedio de logits)
     genera N candidatos. Se reducen a 5 representativos rankeando por
     retorno de PORTAFOLIO bajo w_ref = w0 (50/50).
  4. Para cada g: resuelve solve_portfolio con su contexto, simula V[g, s]
     sobre los escenarios COMPARTIDOS.
  5. Calcula regret R[g, s] = V_best_s - V[g, s], elige g*_mean y g*_worst.
  6. Reporta diagnostico: dispersion de p_dl(t) entre las 15 NNs.
  7. BACKTEST IN-SAMPLE (control): aplica la politica seleccionada sobre
     todo r_hist[1..T]. Util como sanity check de que el modelo aprendio
     el train, pero CONTAMINADO con periodo de entrenamiento del LSTM.
  8. BACKTEST OUT-OF-SAMPLE: re-corre TODO el pipeline (ctx, escenarios,
     regret-grid, seleccion g*, backtest) restringido al segmento de
     test del split DL (t_test_start..T). Anchor w0=50/50 al inicio
     del test, mu_hat estimado solo sobre train+valid.

Uso:
    python main.py                # usa el cache .pt si existe
    python main.py --retrain      # fuerza re-entreno de las 15 NNs y
                                  # limpia inspeccion_v2/_cache (sino
                                  # L4/L5 leerian pickles stale)
"""
import argparse
import shutil

import numpy as np
import pandas as pd

from config import (
    DATA_DIR,
    DLConfig,
    LAMBDA_GRID,
    M_GRID,
    MODELS_DIR,
    N_CANDIDATES,
    N_SCENARIOS,
    PROJECT_ROOT,
    RESULTS_DIR,
    SCENARIO_SEED,
    SPLIT,
    T_HORIZON,
    W0,
)
from Regret_Grid import (
    build_ensemble_model,
    build_per_cell_context,
    build_shared_scenarios,
    compute_regret_and_select,
    compute_test_start_t,
    load_market_data,
    pdl_dispersion_diagnostic,
    plot_capital_curves,
    run_historical_backtest,
    run_historical_backtest_oos,
    run_per_cell_regret_grid,
    train_per_cell_nns,
)


# ================================================================
# Fase 1: 15 entrenamientos (uno por celda)
# ================================================================

def fase1_train_per_cell(force_retrain: bool = False):
    print("=" * 70)
    print("FASE 1 — ENTRENAMIENTO POR CELDA (15 NNs independientes)")
    print("=" * 70)
    dl_config = DLConfig()
    print(f"  DLConfig: H={dl_config.H}  hidden={dl_config.lstm_hidden}  "
          f"epochs={dl_config.epochs}  patience={dl_config.patience}")
    print(f"  Grilla: lambda x m = {len(LAMBDA_GRID)} x {len(M_GRID)} = "
          f"{len(LAMBDA_GRID) * len(M_GRID)} celdas")
    print(f"  Cache: {MODELS_DIR / 'per_cell'}")
    print(f"  force_retrain={force_retrain}")
    print("-" * 70)
    nns = train_per_cell_nns(
        lambda_grid=LAMBDA_GRID,
        m_grid=M_GRID,
        dl_config=dl_config,
        models_dir=MODELS_DIR,
        force_retrain=force_retrain,
    )
    print(f"-> {len(nns)} NNs disponibles en memoria.")
    return nns


# ================================================================
# Fase 2: contexto ex-ante por celda
# ================================================================

def fase2_per_cell_contexts(nns):
    print("\n" + "=" * 70)
    print("FASE 2 — CONTEXTO POR CELDA (mu_mix, sigma_mix con NN propia)")
    print("=" * 70)
    print(f"  mu_hat_source = 'p_hist' (lectura PDF literal sec 1.3)")
    print(f"  p_method      = 'walking' (validado en HALLAZGOS sec 4)")
    print("-" * 70)
    contexts = {}
    for g, nn in nns.items():
        lam, m = g
        ctx = build_per_cell_context(
            nn=nn, data_dir=DATA_DIR, T=T_HORIZON, mu_hat_source="p_hist",
        )
        contexts[g] = ctx
        p_bull_mean = np.mean([ctx["p_dl"][i]["bull"].mean()
                               for i in ctx["assets"]])
        print(f"  g=(lam={lam:.2f}, m={m:.2f})  "
              f"p_bull_mean(over assets)={p_bull_mean:.3f}")
    return contexts


# ================================================================
# Fase 3: escenarios compartidos (ensemble + ranking por portafolio)
# ================================================================

def fase3_shared_scenarios(nns, contexts):
    print("\n" + "=" * 70)
    print("FASE 3 — ESCENARIOS COMPARTIDOS (ensemble + ranking FO-aligned)")
    print("=" * 70)
    ensemble = build_ensemble_model(nns)
    print(f"  Ensemble construido: {len(ensemble.nets)} redes apiladas "
          f"(promedio de logits)")

    # initial_window: ultimas H semanas del historico, por activo.
    base_ctx = load_market_data(str(DATA_DIR))
    assets = list(base_ctx["assets"])
    r_hist = base_ctx["r"]
    H = ensemble.config.H
    returns_history = np.stack(
        [r_hist[i].sort_index().values[:T_HORIZON] for i in assets], axis=1,
    ).astype(np.float32)
    initial_window = returns_history[-H:, :]                  # (H, A)

    # w_ref = w0 (50/50): el agente de referencia mantiene su anchor inicial.
    # c_base: costos de transaccion por activo (config.C_BASE).
    # Con ambos, el ranking sigue la ec. 19 del PDF: cap_T = prod(1 + r_port - cost).
    w_ref  = np.array([W0[i]                  for i in assets], dtype=np.float32)
    c_base = np.array([base_ctx["c_base"][i]  for i in assets], dtype=np.float32)
    print(f"  w_ref (50/50): {dict(zip(assets, w_ref.tolist()))}")
    print(f"  c_base       : {dict(zip(assets, c_base.tolist()))}")
    print(f"  N candidates  = {N_CANDIDATES}")
    print(f"  n_scenarios   = {N_SCENARIOS} (reducidos por capital terminal FO)")

    scenarios = build_shared_scenarios(
        ensemble_nn=ensemble,
        initial_window=initial_window,
        w_ref=w_ref,
        c_base=c_base,
        N=N_CANDIDATES,
        T=T_HORIZON,
        n_scenarios=N_SCENARIOS,
        seed=SCENARIO_SEED,
    )
    print(f"  scenarios.shape = {scenarios.shape}  (n_S, T, A)")

    # Resumen por escenario: ret puro y capital terminal con costos (FO).
    r_port = np.einsum("sti,i->st", scenarios, w_ref)
    cum_port = np.prod(1.0 + r_port, axis=1) - 1.0
    denom = np.clip(1.0 + r_port, 1e-8, None)
    w_drift = (w_ref[None, None, :] * (1.0 + scenarios)) / denom[:, :, None]
    cost = np.einsum("sta,a->st", np.abs(w_ref[None, None, :] - w_drift), c_base)
    cap_T = np.prod(np.clip(1.0 + r_port - cost, 1e-8, None), axis=1) - 1.0
    print(f"  ret port (sin costos) por escenario:  {[f'{c:+.2%}' for c in cum_port]}")
    print(f"  cap FO (con costos)   por escenario:  {[f'{c:+.2%}' for c in cap_T]}")
    return scenarios, ensemble


# ================================================================
# Fase 4-5: regret grid (solve por celda, V[g,s] sobre escenarios compartidos)
# ================================================================

def fase4_regret_grid(contexts, scenarios):
    print("\n" + "=" * 70)
    print("FASE 4 — REGRET GRID (un solve por celda; V sobre escenarios comp.)")
    print("=" * 70)
    V_df, policies = run_per_cell_regret_grid(
        contexts, scenarios, list(LAMBDA_GRID), list(M_GRID),
    )
    res = compute_regret_and_select(V_df)
    return V_df, policies, res


# ================================================================
# Fase 6: reporte + persistencia
# ================================================================

def fase5_report(V_df, res, policies, contexts, scenarios, nns):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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
    ctx_m = contexts[(lam_m, m_m)]
    C0 = ctx_m["Capital_inicial"]
    V_mean_row  = res["V_table"].loc[(lam_m, m_m)]
    V_worst_row = res["V_table"].loc[(lam_w, m_w)]
    print("\n--- Seleccion de g* ---")
    print(f"  g*_mean  (ec. 23): lambda={lam_m:.2f}  m={m_m:.2f}  "
          f"mean_regret=${res['g_mean_metric']:,.2f}")
    print(f"      V: mean=${V_mean_row.mean():>12,.2f}  "
          f"worst=${V_mean_row.min():>12,.2f}  "
          f"best=${V_mean_row.max():>12,.2f}  "
          f"(capital inicial=${C0:,.2f})")
    print(f"      retorno promedio sobre escenarios = {V_mean_row.mean()/C0 - 1:+.2%}")
    print(f"  g*_worst (ec. 24): lambda={lam_w:.2f}  m={m_w:.2f}  "
          f"worst_regret=${res['g_worst_metric']:,.2f}")
    print(f"      V: mean=${V_worst_row.mean():>12,.2f}  "
          f"worst=${V_worst_row.min():>12,.2f}  "
          f"best=${V_worst_row.max():>12,.2f}  "
          f"(capital inicial=${C0:,.2f})")
    print(f"      retorno en el peor escenario     = {V_worst_row.min()/C0 - 1:+.2%}")

    # --- Persistencia ---
    out_V = RESULTS_DIR / "regret_grid_results.csv"
    V_df.to_csv(out_V, index=False)
    print(f"\n  V_df (long)           : {out_V}")

    out_R = RESULTS_DIR / "regret_table.csv"
    res["R_table"].to_csv(out_R)
    print(f"  Tabla de regret       : {out_R}")

    out_summary = RESULTS_DIR / "regret_summary.csv"
    res["regret_summary"].to_csv(out_summary)
    print(f"  Resumen por g         : {out_summary}")

    # --- Diagnostico: dispersion de p_dl entre las 15 NNs ---
    print("\n" + "-" * 70)
    print("DIAGNOSTICO — dispersion de p_dl(t) entre las 15 NNs por celda")
    print("-" * 70)
    disp = pdl_dispersion_diagnostic(nns, data_dir=DATA_DIR, T=T_HORIZON)
    out_disp = RESULTS_DIR / "pdl_dispersion.csv"
    disp.to_csv(out_disp, index=False)
    for asset in disp["asset"].unique():
        sub = disp[disp["asset"] == asset]
        print(f"  {asset:<7}  p_bull std promedio (sobre t)={sub['std'].mean():.4f}  "
              f"max std={sub['std'].max():.4f}  "
              f"max(max-min)={(sub['max']-sub['min']).max():.4f}")
    print(f"  -> {out_disp}")

    # --- Plot capital g*_mean ---
    w_star, u_star, v_star, _z = policies[(lam_m, m_m)]
    # plot_capital_curves espera 'scenarios' en el ctx; lo inyectamos en ctx_m.
    ctx_plot = dict(ctx_m)
    ctx_plot["scenarios"] = scenarios
    plot_capital_curves(
        w_star, u_star, v_star, ctx_plot,
        title=f"Capital por escenario con g*_mean (lambda={lam_m:.2f}, m={m_m:.2f})",
        out_path=RESULTS_DIR / "regret_capital_curves.png",
    )

    # --- Backtest historico IN-SAMPLE (control) ---
    print("\n" + "-" * 70)
    print("BACKTEST IN-SAMPLE (control) - r_hist[1..T] completo")
    print("-" * 70)
    print("  Nota: este backtest aplica la politica sobre toda la serie")
    print("        historica, INCLUYENDO el segmento que el LSTM uso para")
    print("        entrenar (train+valid del split). Es un sanity check de")
    print("        que el modelo aprendio el train; NO una metrica de")
    print("        generalizacion. Ver bloque OUT-OF-SAMPLE para esa.")
    run_historical_backtest(
        w_star, u_star, v_star, lam_m, m_m,
        V_mean_row=V_mean_row,
        n_scenarios=scenarios.shape[0],
        data_dir=DATA_DIR,
        out_path=RESULTS_DIR / "evolucion_capital.png",
    )


# ================================================================
# Fase 6: regret grid + backtest OUT-OF-SAMPLE (test segment del split LSTM)
# ================================================================

def fase6_out_of_sample(nns, ensemble):
    """Re-corre el pipeline restringido al segmento de test del split DL.

    Pasos:
      1. Calcular t_test_start desde SPLIT/H/T.
      2. Construir contextos OOS per-cell (mu_hat estimado solo en train+valid;
         T_vals = [t_test_start..T]).
      3. Construir escenarios OOS compartidos (ventana inicial = los H semanas
         antes de t_test_start; horizonte = T - t_test_start + 1).
      4. Resolver regret-grid OOS y seleccionar g*_mean_oos.
      5. Backtest OOS: OPT_oos / Naive / RG_oos sobre r_hist[t_test_start..T].
    """
    print("\n" + "=" * 70)
    print("FASE 6 — REGRET GRID + BACKTEST OUT-OF-SAMPLE (test del split LSTM)")
    print("=" * 70)
    H = ensemble.config.H
    t_test_start = compute_test_start_t(T=T_HORIZON, H=H, split=SPLIT)
    T_oos = T_HORIZON - t_test_start + 1
    print(f"  SPLIT={SPLIT}  H={H}  T={T_HORIZON}  =>  t_test_start={t_test_start}")
    print(f"  Horizonte OOS = t{t_test_start}..t{T_HORIZON} ({T_oos} periodos)")
    print(f"  mu_hat/sigma_hat estimados solo sobre t=1..{t_test_start-1} (train+valid)")
    print("-" * 70)

    # --- 6.1: contextos OOS per-cell ---
    print("  Construyendo contextos OOS per-cell ...")
    contexts_oos = {}
    for g, nn in nns.items():
        lam, m = g
        ctx_oos = build_per_cell_context(
            nn=nn, data_dir=DATA_DIR, T=T_HORIZON,
            mu_hat_source="p_hist",
            t_start=t_test_start,
            moments_window=(1, t_test_start - 1),
        )
        contexts_oos[g] = ctx_oos
        p_bull_mean = np.mean([ctx_oos["p_dl"][i]["bull"].mean()
                               for i in ctx_oos["assets"]])
        print(f"    g=(lam={lam:.2f}, m={m:.2f})  "
              f"p_bull_mean(OOS, over assets)={p_bull_mean:.3f}")

    # --- 6.2: escenarios OOS compartidos ---
    print("\n  Generando escenarios OOS (ventana inicial = retornos previos al test) ...")
    base_ctx = load_market_data(str(DATA_DIR))
    assets = list(base_ctx["assets"])
    r_hist = base_ctx["r"]
    # Ventana inicial: H semanas inmediatamente antes de t_test_start
    # (todas dentro de train+valid). r_hist es 1-indexed; usamos .loc.
    initial_window_oos = np.stack(
        [r_hist[i].loc[t_test_start - H : t_test_start - 1].values
         for i in assets],
        axis=1,
    ).astype(np.float32)                      # (H, A)
    w_ref = np.array([W0[i] for i in assets], dtype=np.float32)
    scenarios_oos = build_shared_scenarios(
        ensemble_nn=ensemble,
        initial_window=initial_window_oos,
        w_ref=w_ref,
        N=N_CANDIDATES,
        T=T_oos,
        n_scenarios=N_SCENARIOS,
        seed=SCENARIO_SEED,
    )
    print(f"    scenarios_oos.shape = {scenarios_oos.shape}  (n_S, T_oos, A)")
    r_port_oos = np.einsum("sti,i->st", scenarios_oos, w_ref)
    cum_port_oos = np.prod(1.0 + r_port_oos, axis=1) - 1.0
    print(f"    ret port (w_ref) por escenario OOS: "
          f"{[f'{c:+.2%}' for c in cum_port_oos]}")

    # --- 6.3: regret-grid OOS ---
    print("\n  Resolviendo regret-grid OOS (un solve por celda; T_oos periodos) ...")
    V_df_oos, policies_oos = run_per_cell_regret_grid(
        contexts_oos, scenarios_oos, list(LAMBDA_GRID), list(M_GRID),
    )
    res_oos = compute_regret_and_select(V_df_oos)

    # --- 6.4: reporte OOS ---
    print("\n--- V[g, s] OOS — capital terminal por (lambda, m) y escenario ---")
    print(res_oos["V_table"].to_string(float_format="${:,.2f}".format))
    print("\n--- R[g, s] OOS = V_best_s - V[g, s] ---")
    print(res_oos["R_table"].to_string(float_format="${:,.2f}".format))
    print("\n--- Resumen de regret OOS por g ---")
    print(res_oos["regret_summary"].to_string(float_format="${:,.2f}".format))

    lam_m_o, m_m_o = res_oos["g_mean"]
    lam_w_o, m_w_o = res_oos["g_worst"]
    ctx_oos_m = contexts_oos[(lam_m_o, m_m_o)]
    C0 = ctx_oos_m["Capital_inicial"]
    V_mean_row_oos  = res_oos["V_table"].loc[(lam_m_o, m_m_o)]
    V_worst_row_oos = res_oos["V_table"].loc[(lam_w_o, m_w_o)]

    print("\n--- Seleccion de g* (OOS) ---")
    print(f"  g*_mean_oos  (ec. 23): lambda={lam_m_o:.2f}  m={m_m_o:.2f}  "
          f"mean_regret=${res_oos['g_mean_metric']:,.2f}")
    print(f"      V_oos: mean=${V_mean_row_oos.mean():>12,.2f}  "
          f"worst=${V_mean_row_oos.min():>12,.2f}  "
          f"best=${V_mean_row_oos.max():>12,.2f}  "
          f"(capital inicial=${C0:,.2f})")
    print(f"      retorno promedio sobre escenarios OOS = "
          f"{V_mean_row_oos.mean()/C0 - 1:+.2%}")
    print(f"  g*_worst_oos (ec. 24): lambda={lam_w_o:.2f}  m={m_w_o:.2f}  "
          f"worst_regret=${res_oos['g_worst_metric']:,.2f}")
    print(f"      V_oos: mean=${V_worst_row_oos.mean():>12,.2f}  "
          f"worst=${V_worst_row_oos.min():>12,.2f}  "
          f"best=${V_worst_row_oos.max():>12,.2f}  "
          f"(capital inicial=${C0:,.2f})")
    print(f"      retorno en el peor escenario OOS     = "
          f"{V_worst_row_oos.min()/C0 - 1:+.2%}")

    # --- Persistencia OOS ---
    out_V_oos = RESULTS_DIR / "regret_grid_results_oos.csv"
    V_df_oos.to_csv(out_V_oos, index=False)
    print(f"\n  V_df OOS              : {out_V_oos}")
    out_R_oos = RESULTS_DIR / "regret_table_oos.csv"
    res_oos["R_table"].to_csv(out_R_oos)
    print(f"  Tabla de regret OOS   : {out_R_oos}")
    out_summary_oos = RESULTS_DIR / "regret_summary_oos.csv"
    res_oos["regret_summary"].to_csv(out_summary_oos)
    print(f"  Resumen por g OOS     : {out_summary_oos}")

    # --- Plot capital curves OOS bajo g*_mean_oos ---
    w_star_o, u_star_o, v_star_o, _z = policies_oos[(lam_m_o, m_m_o)]
    ctx_plot_oos = dict(ctx_oos_m)
    ctx_plot_oos["scenarios"] = scenarios_oos
    plot_capital_curves(
        w_star_o, u_star_o, v_star_o, ctx_plot_oos,
        title=(f"Capital por escenario OOS con g*_mean (lambda={lam_m_o:.2f}, "
               f"m={m_m_o:.2f})"),
        out_path=RESULTS_DIR / "regret_capital_curves_oos.png",
    )

    # --- Backtest historico OOS ---
    print("\n" + "-" * 70)
    print("BACKTEST OUT-OF-SAMPLE - r_hist[t_test_start..T]")
    print("-" * 70)
    run_historical_backtest_oos(
        w_star_o, u_star_o, v_star_o, lam_m_o, m_m_o,
        V_mean_row=V_mean_row_oos,
        n_scenarios=scenarios_oos.shape[0],
        data_dir=DATA_DIR,
        t_test_start=t_test_start,
        T=T_HORIZON,
        out_path=RESULTS_DIR / "evolucion_capital_oos.png",
    )

    return {
        "t_test_start": t_test_start,
        "contexts_oos": contexts_oos,
        "scenarios_oos": scenarios_oos,
        "V_df_oos": V_df_oos,
        "policies_oos": policies_oos,
        "res_oos": res_oos,
    }


# ================================================================
# Bloque principal
# ================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--retrain", action="store_true",
        help="Fuerza reentreno de las 15 NNs y limpia "
             "inspeccion_v2/_cache (pickles de L4/L5).",
    )
    args = parser.parse_args()

    if args.retrain:
        cache_dir = PROJECT_ROOT / "inspeccion_v2" / "_cache"
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            print(f"[retrain] cache pickle eliminado: {cache_dir}")

    nns                  = fase1_train_per_cell(force_retrain=args.retrain)
    contexts             = fase2_per_cell_contexts(nns)
    scenarios, ensemble  = fase3_shared_scenarios(nns, contexts)
    V_df, policies, res  = fase4_regret_grid(contexts, scenarios)
    fase5_report(V_df, res, policies, contexts, scenarios, nns)
    fase6_out_of_sample(nns, ensemble)

    print("\n" + "=" * 70)
    print("Pipeline per-cell completo (in-sample + out-of-sample).")
    print("=" * 70)
