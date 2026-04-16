"""§1.5 del PDF: simulación ex-post de capital (ec. 10).

Tres políticas:
  - simulate_capital_opt: usa los pesos (u,v) del optimizador.
  - simulate_naive_bh:    50/50 buy & hold, sin costos.
  - simulate_naive_rb:    50/50 con rebalanceo semanal y costos.

Nota: simulate_capital_opt usa c_base (no c_eff); el multiplicador de costo
solo afecta la optimización, no la evaluación del capital real.
"""


def simulate_capital_opt(w_sol, u_sol, v_sol, context):
    T_vals  = context["T_vals"]
    assets  = context["assets"]
    r       = context["r"]
    c_base  = context["c_base"]
    Capital = context["Capital_inicial"]

    cap = {T_vals[0]: Capital}
    for idx in range(1, len(T_vals)):
        t, t_prev = T_vals[idx], T_vals[idx - 1]
        r_port = sum(w_sol[i, t_prev] * r[i].loc[t_prev] for i in assets)
        turn   = sum(c_base[i] * (u_sol[i, t] + v_sol[i, t]) for i in assets)
        cap[t] = cap[t_prev] * (1 + r_port) - cap[t_prev] * turn
    return cap


def simulate_naive_bh(context):
    T_vals  = context["T_vals"]
    assets  = context["assets"]
    r       = context["r"]
    Capital = context["Capital_inicial"]
    w_naive = {i: 0.5 for i in assets}

    cap = {T_vals[0]: Capital}
    for idx in range(1, len(T_vals)):
        t, t_prev = T_vals[idx], T_vals[idx - 1]
        r_port = sum(w_naive[i] * r[i].loc[t_prev] for i in assets)
        cap[t] = cap[t_prev] * (1 + r_port)
    return cap


def simulate_naive_rb(context):
    T_vals  = context["T_vals"]
    assets  = context["assets"]
    r       = context["r"]
    c_base  = context["c_base"]
    Capital = context["Capital_inicial"]
    w_target = 0.5

    cap = {T_vals[0]: Capital}
    for idx in range(1, len(T_vals)):
        t, t_prev = T_vals[idx], T_vals[idx - 1]
        r_port = sum(w_target * r[i].loc[t_prev] for i in assets)
        w_bh   = {i: w_target * (1 + r[i].loc[t_prev]) / (1 + r_port) for i in assets}
        turn   = sum(c_base[i] * abs(w_target - w_bh[i]) for i in assets)
        cap[t] = cap[t_prev] * (1 + r_port) - cap[t_prev] * turn
    return cap
