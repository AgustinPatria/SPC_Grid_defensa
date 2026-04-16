"""§1.4 del PDF: optimización media–varianza con costos (GAMSPy + IPOPT).

Variables de decisión:
  w[i,t]  pesos en [0,1]
  u[i,t]  compras (>=0)
  v[i,t]  ventas  (>=0)

Objetivo (ec. 9):
  max  sum_t [ sum_i w(i,t)*mu_mix(i,t)*theta(i)
               - lambda * sum_{i,j} w(i,t)*w(j,t)*sigma_mix(i,j,t)
               - sum_i c_eff(i) * (u(i,t) + v(i,t)) ]

Restricciones:
  (6)  sum_i w(i,t) = 1
  (7)  w(i,t) - w(i,t-1) = u(i,t) - v(i,t)      para t >= 2
  (8)  w(i,t1) - w0(i)   = u(i,t1) - v(i,t1)    (anclaje inicial)
"""
import sys
import pandas as pd
import gamspy as gp


def solve_portfolio(theta: dict, context: dict,
                    lambda_riesgo: float = 0.10,
                    costo_mult:    float = 1.0,
                    verbose:       bool  = False):
    mu_mix    = context["mu_mix"]
    sigma_mix = context["sigma_mix"]
    T_vals    = context["T_vals"]
    assets    = context["assets"]
    c_base    = context["c_base"]
    w0_dict   = context["w0"]

    T_labels = [f"t{n}" for n in T_vals]

    m = gp.Container()

    # --- Sets ---
    i_set = gp.Set(m, "i", records=assets,   description="activos")
    j_set = gp.Alias(m, "j", i_set)
    t_set = gp.Set(m, "t", records=T_labels, description="periodos")

    # --- Parameters ---
    mu_p = gp.Parameter(
        m, "mu_mix", domain=[i_set, t_set],
        records=pd.DataFrame(
            [[i, f"t{t}", mu_mix[i].loc[t] * theta[i]]
             for i in assets for t in T_vals],
            columns=["i", "t", "value"],
        ),
    )

    sig_p = gp.Parameter(
        m, "sigma_mix", domain=[i_set, j_set, t_set],
        records=pd.DataFrame(
            [[i, j, f"t{t}", sigma_mix[i][j].loc[t]]
             for i in assets for j in assets for t in T_vals],
            columns=["i", "j", "t", "value"],
        ),
    )

    c_eff_p = gp.Parameter(
        m, "c_eff", domain=[i_set],
        records=pd.DataFrame(
            [[i, c_base[i] * costo_mult] for i in assets],
            columns=["i", "value"],
        ),
    )

    w0_p = gp.Parameter(
        m, "w0", domain=[i_set],
        records=pd.DataFrame(
            [[i, w0_dict[i]] for i in assets],
            columns=["i", "value"],
        ),
    )

    lam_p = gp.Parameter(m, "lambda_riesgo", records=lambda_riesgo)

    # --- Variables ---
    z_var = gp.Variable(m, "z")
    w_var = gp.Variable(m, "w", domain=[i_set, t_set], type="positive")
    u_var = gp.Variable(m, "u", domain=[i_set, t_set], type="positive")
    v_var = gp.Variable(m, "v", domain=[i_set, t_set], type="positive")

    w_var.up[i_set, t_set] = 1.0

    # --- Equations ---
    fo = gp.Equation(m, "FO_media_var_costo")
    fo[...] = z_var == gp.Sum(
        t_set,
        gp.Sum(i_set, w_var[i_set, t_set] * mu_p[i_set, t_set])
        - lam_p * gp.Sum(
            (i_set, j_set),
            w_var[i_set, t_set] * w_var[j_set, t_set] * sig_p[i_set, j_set, t_set],
        )
        - gp.Sum(i_set, c_eff_p[i_set] * (u_var[i_set, t_set] + v_var[i_set, t_set]))
    )

    norm = gp.Equation(m, "normalizacion_pesos", domain=[t_set])
    norm[t_set] = gp.Sum(i_set, w_var[i_set, t_set]) == 1

    rebal = gp.Equation(m, "rebalanceo_lineal", domain=[i_set, t_set])
    rebal[i_set, t_set].where[gp.Ord(t_set) > 1] = (
        w_var[i_set, t_set] - w_var[i_set, t_set.lag(1)]
        == u_var[i_set, t_set] - v_var[i_set, t_set]
    )

    anclaje = gp.Equation(m, "anclaje_inicial", domain=[i_set])
    anclaje[i_set] = (
        w_var[i_set, "t1"] - w0_p[i_set]
        == u_var[i_set, "t1"] - v_var[i_set, "t1"]
    )

    # --- Solve ---
    portfolio = gp.Model(
        m,
        name="PortafolioEstadosCostos",
        equations=m.getEquations(),
        problem="NLP",
        sense=gp.Sense.MAX,
        objective=z_var,
    )

    portfolio.solve(solver="IPOPT", output=sys.stdout if verbose else None)

    if portfolio.status not in (
        gp.ModelStatus.OptimalLocal,
        gp.ModelStatus.OptimalGlobal,
    ):
        raise RuntimeError(
            f"GAMSPy/IPOPT no encontró solución óptima. Status: {portfolio.status}"
        )

    z_val = float(z_var.toValue())

    def _records_to_dict(var):
        sol = {}
        for _, row in var.records.iterrows():
            sol[row["i"], int(row["t"][1:])] = float(row["level"])
        return sol

    w_sol = _records_to_dict(w_var)
    u_sol = _records_to_dict(u_var)
    v_sol = _records_to_dict(v_var)

    status = ("optimal" if portfolio.status == gp.ModelStatus.OptimalGlobal
              else "optimal_local")
    return z_val, w_sol, u_sol, v_sol, status
