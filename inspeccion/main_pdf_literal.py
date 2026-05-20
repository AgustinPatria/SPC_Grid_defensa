"""Run del pipeline con la interpretacion LITERAL del PDF.

Equivalente a `main.py` pero con `p_method='walking', mu_hat_source='p_hist'`
explicitos — es decir, la lectura literal del PDF sec 1.3 ec. (2): mu_hat se
estima con las probabilidades del CSV (HMM externo).

Ver HALLAZGOS sec 3 para discusion de las dos lecturas (PDF literal con
`p_hist` vs propuesta propia con `p_sign`).

NO re-entrena la LSTM — usa el checkpoint actual.
"""
from pathlib import Path
import sys

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

from config import (
    CHECKPOINT_PATH,
    DATA_DIR,
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


def main():
    out_dir = _THIS_DIR / "main_pdf_literal_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PIPELINE PDF-LITERAL  (mu_hat_source='p_hist', PDF sec 1.3 literal)")
    print("=" * 70)
    print("Construyendo contexto DL con interpretacion literal del PDF...")
    ctx = build_dl_context(
        data_dir=DATA_DIR, checkpoint_path=CHECKPOINT_PATH,
        T=T_HORIZON, N_candidates=N_CANDIDATES, n_scenarios=N_SCENARIOS,
        seed=SCENARIO_SEED, summary_asset=SUMMARY_ASSET,
        p_method="walking", mu_hat_source="p_hist",     # <-- LITERAL PDF
    )

    print(f"\n  Assets     : {ctx['assets']}")
    print(f"  T          : {ctx['nT']}")
    print(f"  Scenarios  : {ctx['scenarios'].shape}")
    for i in ctx["assets"]:
        col = ctx["p_dl"][i]["bull"]
        print(f"  p_bull {i:<7}: min={col.min():.3f}  max={col.max():.3f}  "
              f"mean={col.mean():.3f}")
        for k in ("bear", "bull"):
            print(f"  mu_hat[({i}, {k})] = {ctx['mu_hat'][(i, k)]*100:+.3f}%/sem")
        mu = ctx["mu_mix"][i]
        print(f"  mu_mix({i}) mean={mu.mean()*100:+.3f}%  std={mu.std()*100:.4f}%")

    lambda_grid = list(LAMBDA_GRID)
    m_grid = list(M_GRID)
    print(f"\nGrilla {len(lambda_grid)}x{len(m_grid)}={len(lambda_grid)*len(m_grid)} solves...")
    V_df, policies = run_regret_grid(ctx, lambda_grid, m_grid)
    res = compute_regret_and_select(V_df)

    print("\n" + "=" * 70)
    print("RESULTADOS — PDF LITERAL")
    print("=" * 70)
    print("\n--- Resumen de regret por g ---")
    print(res["regret_summary"].to_string(float_format="${:,.2f}".format))

    lam_m, m_m = res["g_mean"]
    lam_w, m_w = res["g_worst"]
    C0 = ctx["Capital_inicial"]
    V_mean_row = res["V_table"].loc[(lam_m, m_m)]

    print(f"\n--- Seleccion g* ---")
    print(f"  g*_mean  = (lambda={lam_m:.2f}, m={m_m:.1f})  mean_regret=${res['g_mean_metric']:,.2f}")
    print(f"      V: mean=${V_mean_row.mean():,.2f}  "
          f"worst=${V_mean_row.min():,.2f}  best=${V_mean_row.max():,.2f}")
    print(f"      retorno promedio sobre escenarios = {V_mean_row.mean()/C0 - 1:+.2%}")

    # Persistencia local (no toca resultados/ default)
    V_df.to_csv(out_dir / "regret_grid_results.csv", index=False)
    res["R_table"].to_csv(out_dir / "regret_table.csv")
    res["regret_summary"].to_csv(out_dir / "regret_summary.csv")

    # Capital curves bajo g*_mean
    w_star, u_star, v_star, _z = policies[(lam_m, m_m)]
    plot_capital_curves(
        w_star, u_star, v_star, ctx,
        title=f"Capital por escenario — PDF LITERAL g*_mean (lambda={lam_m:.2f}, m={m_m:.1f})",
        out_path=out_dir / "regret_capital_curves.png",
    )

    # Backtest historico
    run_historical_backtest(
        w_star, u_star, v_star, lam_m, m_m,
        V_mean_row=V_mean_row,
        n_scenarios=ctx["scenarios"].shape[0],
        data_dir=DATA_DIR,
        out_path=out_dir / "evolucion_capital.png",
    )

    print(f"\nOutputs en {out_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
