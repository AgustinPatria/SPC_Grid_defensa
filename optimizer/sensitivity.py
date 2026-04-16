"""Grid de sensibilidad λ × multiplicador-de-costo (LOOP L,C del GAMS).

Este es el grid INTERNO del optimizador (§5 del .gms) que usa los
retornos observados como único "escenario". NO debe confundirse con
el regret-grid de la capa `calibration`, que evalúa la política sobre
múltiples escenarios generados por Deep Learning.
"""
import pandas as pd

from optimizer.model      import solve_portfolio
from optimizer.simulation import simulate_capital_opt


def run_sensitivity_grid(context,
                         lambda_grid=(0.05, 0.10, 0.20, 0.50, 1.00),
                         c_mult_grid=(0.5, 1.0, 2.0)):
    assets      = context["assets"]
    T_vals      = context["T_vals"]
    Capital_ini = context["Capital_inicial"]
    theta_neu   = {a: 1.0 for a in assets}

    labels_L = [f"L{i+1}" for i in range(len(lambda_grid))]
    labels_C = [f"C{i+1}" for i in range(len(c_mult_grid))]

    rows  = []
    total = len(lambda_grid) * len(c_mult_grid)
    run   = 0
    for li, lam in enumerate(lambda_grid):
        for ci, cm in enumerate(c_mult_grid):
            run += 1
            print(f"  [{run:>2}/{total}] {labels_L[li]}/{labels_C[ci]}  "
                  f"lambda={lam:.2f}  c_mult={cm:.1f} ...", end=" ", flush=True)
            z, w_sol, u_sol, v_sol, _ = solve_portfolio(
                theta_neu, context, lambda_riesgo=lam, costo_mult=cm
            )
            cap       = simulate_capital_opt(w_sol, u_sol, v_sol, context)
            cap_final = cap[T_vals[-1]]
            ret_acum  = cap_final / Capital_ini - 1
            print(f"z={z:.6f}  cap_final=${cap_final:,.2f}  ret={ret_acum:+.2%}")
            rows.append({
                "L": labels_L[li], "C": labels_C[ci],
                "lambda": lam,    "c_mult": cm,
                "z":         round(z, 6),
                "cap_final": round(cap_final, 2),
                "ret_acum":  round(ret_acum, 6),
            })

    return pd.DataFrame(rows)
