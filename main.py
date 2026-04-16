"""Orquestación del modelo base (CAPA 1 — optimizador).

Ejecuta solo la capa de optimización:
  - caso neutral
  - caso bullish SPX
  - grid de sensibilidad (lambda x c_mult)

El pipeline completo DL -> escenarios -> regret-grid vive en pipeline.py.
"""
from pathlib import Path

from optimizer.data_loader  import load_market_data
from optimizer.model        import solve_portfolio
from optimizer.simulation   import simulate_capital_opt, simulate_naive_bh, simulate_naive_rb
from optimizer.sensitivity  import run_sensitivity_grid


ROOT        = Path(__file__).parent
DATA_DIR    = ROOT / "data"
RESULTS_DIR = ROOT / "results"


def _summary_row(label, cap, T_end, C0):
    cf = cap[T_end]
    return f"  {label:<30}  ${cf:>12,.2f}  {cf/C0-1:>+8.2%}  {cf-C0:>+12,.2f}"


def main():
    print("Cargando datos...")
    context = load_market_data(str(DATA_DIR))
    assets  = context["assets"]
    T_vals  = context["T_vals"]
    C0      = context["Capital_inicial"]
    T_end   = T_vals[-1]
    print(f"Datos cargados: {len(T_vals)} periodos, activos: {assets}\n")

    print("=" * 65)
    print("CASO BASE - Neutral (theta=1.0, lambda=0.10, c_mult=1.0)")
    print("=" * 65)
    theta_neutral = {a: 1.0 for a in assets}
    z_neu, w_neu, u_neu, v_neu, status_neu = solve_portfolio(
        theta_neutral, context, lambda_riesgo=0.10
    )
    print(f"  Status : {status_neu}")
    print(f"  z      : {z_neu:.6f}")

    cap_opt = simulate_capital_opt(w_neu, u_neu, v_neu, context)
    cap_bh  = simulate_naive_bh(context)
    cap_rb  = simulate_naive_rb(context)

    print("\n--- cap_opt / cap_naive_rb / cap_naive_bh ---")
    print(f"  {'':30}  {'cap_final':>12}  {'ret_acum':>8}  {'inc_cap':>12}")
    print(f"  {'-'*65}")
    print(_summary_row("Optimo (cap_opt)",       cap_opt, T_end, C0))
    print(_summary_row("Naive 50/50 Rebalanceo", cap_rb,  T_end, C0))
    print(_summary_row("Naive Buy & Hold",       cap_bh,  T_end, C0))

    print("\n" + "=" * 65)
    print("CASO BULLISH SPX - theta_SPX=1.1")
    print("=" * 65)
    theta_bull = {a: 1.0 for a in assets}
    theta_bull["SPX"] = 1.10
    z_bull, w_bull, u_bull, v_bull, _ = solve_portfolio(
        theta_bull, context, lambda_riesgo=0.10
    )
    cap_bull = simulate_capital_opt(w_bull, u_bull, v_bull, context)
    print(f"  z      : {z_bull:.6f}")
    print(_summary_row("Optimo Bullish SPX", cap_bull, T_end, C0))

    print("\n" + "=" * 65)
    print("SENSIBILIDAD - Grid lambda (L1-L5) x c_mult (C1-C3)")
    print("=" * 65)
    df_grid = run_sensitivity_grid(context)

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "sensitivity_results.csv"
    df_grid.to_csv(out_path, index=False)
    print(f"\nResultados guardados en: {out_path}")


if __name__ == "__main__":
    main()
