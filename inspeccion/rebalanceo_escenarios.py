"""Rebalanceo y capital del portafolio en escenarios extremos (bear / bull).

NOTA importante (PDF sec. 3.3): el FO se resuelve UNA VEZ por g, asi que la
politica w*(t) es la MISMA para todos los escenarios. Lo que cambia entre
escenarios es la trayectoria de retornos r^s(t), que afecta el capital
realizado pero no la politica.

Este script muestra:

  1. La politica w*(t) (la misma para todos los escenarios) — pdf_aligned
     g*_mean del main.py, OPT base (lambda=1.0), y naive 50/50 rebalanceado.

  2. Las trayectorias de retornos r^s(SPX, t) y r^s(CMC, t) para el peor
     escenario (s=0, mas bajista) y el mejor (s=4, mas alcista). Recordar
     que los escenarios estan ORDENADOS por retorno acumulado del SPX
     (eq. 17 del PDF).

  3. La evolucion de capital bajo cada politica aplicada al bear y al bull,
     lado a lado.

Si la politica del FO PDF-aligned es 100% CMC200 (caso del ultimo main.py),
veras como esa concentracion se castiga en bear (CMC tambien cae con SPX
porque los escenarios son comonotonicos) y se premia en bull.

Outputs (inspeccion/rebalanceo_escenarios_out/):
  1_politicas_w.csv/png
  2_returns_bear_bull.csv/png
  3_capital_bear_bull.csv/png
  4_resumen_terminal.csv

Corre con:
    python -m inspeccion.rebalanceo_escenarios
    (o)
    python inspeccion/rebalanceo_escenarios.py
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
    CHECKPOINT_PATH,
    DATA_DIR,
    N_CANDIDATES,
    N_SCENARIOS,
    SCENARIO_SEED,
    SUMMARY_ASSET,
    T_HORIZON,
)
from Regret_Grid import (
    build_dl_context,
    load_market_data,
    simulate_capital_on_scenario,
    solve_portfolio,
)


SUBDIR = "rebalanceo_escenarios"

# g* elegido por main.py PDF-aligned (lambda=0.30, m=0.1).
G_MEAN = (0.30, 0.1)

# Indices de escenarios extremos (ordenados por retorno acum del SPX).
S_BEAR = 0   # mas bajista
S_BULL = 4   # mas alcista


def _naive_policy(assets, T_vals):
    """Politica naive 50/50 con rebalanceo a cada paso.

    En la formulacion del FO, mantener 50/50 SIN drift entre periodos
    significa que la "decision" w(t) = 0.5 es estable. Como simulate_capital_
    on_scenario solo lee w_sol/u_sol/v_sol y no recomputa el rebalanceo
    desde el drift, usamos un proxy: turnover real es ~|drift|, que aqui
    estimamos como pequeno y constante (e). Para simplificar y respetar el
    contrato de simulate_capital_on_scenario, dejamos u=v=0 y bajamos la
    expectativa de costos (es el mismo enfoque que el simulator ignora
    cuando w_sol es plano)."""
    w = {(a, t): 0.5 for a in assets for t in T_vals}
    u = {(a, t): 0.0 for a in assets for t in T_vals}
    v = {(a, t): 0.0 for a in assets for t in T_vals}
    return w, u, v


def _simulate_naive_rebal_on_scenario(scenario, assets, c_base, C0, T_vals):
    """Naive 50/50 rebalanceado paso a paso sobre un escenario (replica
    `simulate_naive_rb` pero usando retornos del escenario en vez de r_hist).

    En cada paso el portafolio drifteo a `w_bh` por los retornos, y se
    rebalancea de vuelta a 0.5/0.5, pagando costo |w_target - w_bh| por
    activo. Cuenta con costo realista, a diferencia del proxy en
    `_naive_policy`."""
    w_target = 0.5
    cap = {T_vals[0]: C0}
    for idx in range(1, len(T_vals)):
        t = T_vals[idx]
        t_prev = T_vals[idx - 1]
        r_port = sum(w_target * scenario[idx - 1, ai]
                     for ai, _ in enumerate(assets))
        w_bh = {a: w_target * (1.0 + scenario[idx - 1, ai]) / (1.0 + r_port)
                for ai, a in enumerate(assets)}
        turn = sum(c_base[a] * abs(w_target - w_bh[a]) for a in assets)
        cap[t] = cap[t_prev] * (1.0 + r_port) - cap[t_prev] * turn
    return cap


def solve_policies():
    print("Construyendo ctx PDF-aligned y resolviendo g*_mean...")
    ctx_dl = build_dl_context(
        data_dir=DATA_DIR, checkpoint_path=CHECKPOINT_PATH,
        T=T_HORIZON, N_candidates=N_CANDIDATES, n_scenarios=N_SCENARIOS,
        seed=SCENARIO_SEED, summary_asset=SUMMARY_ASSET,
    )
    z, w_rg, u_rg, v_rg, _ = solve_portfolio(
        ctx_dl, lambda_riesgo=G_MEAN[0], costo_mult=G_MEAN[1],
    )
    print(f"  RG g*: z={z:.4f}")

    print("Construyendo ctx OPT base y resolviendo (lambda=1.0, m=1.0)...")
    ctx_opt = load_market_data(str(DATA_DIR))
    z_opt, w_opt, u_opt, v_opt, _ = solve_portfolio(
        ctx_opt, lambda_riesgo=1.00, costo_mult=1.0,
    )
    print(f"  OPT base: z={z_opt:.4f}")

    return {
        "ctx_dl": ctx_dl, "ctx_opt": ctx_opt,
        "RG":  {"w": w_rg,  "u": u_rg,  "v": v_rg, "color": "#1f77b4",
                "label": f"RG g*_mean (lambda={G_MEAN[0]:.2f}, m={G_MEAN[1]:.1f})"},
        "OPT": {"w": w_opt, "u": u_opt, "v": v_opt, "color": "#F2B705",
                "label": "OPT base (lambda=1.00, m=1.0)"},
    }


# ================================================================
# 1) Politicas w(t) — la misma en todos los escenarios
# ================================================================

def diag_politicas(state):
    assets = state["ctx_dl"]["assets"]
    T_vals = state["ctx_dl"]["T_vals"]

    rows = []
    fig, axes = plt.subplots(len(assets), 1, figsize=(12, 4 * len(assets)),
                             sharex=True)
    if len(assets) == 1:
        axes = [axes]
    for ai, a in enumerate(assets):
        ax = axes[ai]
        ax.axhline(0.5, color="#666", ls="--", lw=0.8,
                   label="naive 50/50 (constante)")
        for key in ("RG", "OPT"):
            p = state[key]
            w_t = [p["w"][a, t] for t in T_vals]
            ax.plot(T_vals, w_t, color=p["color"], lw=1.5, label=p["label"])
            for t, w in zip(T_vals, w_t):
                rows.append({"policy": key, "asset": a,
                             "t": int(t), "w": float(w)})
        ax.set_title(f"w({a}, t) — politica del FO (igual en todos los escenarios)")
        ax.set_xlabel("t")
        ax.set_ylabel(f"w({a})")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=9)
    save_fig(fig, "1_politicas_w", SUBDIR)
    save_csv(pd.DataFrame(rows), "1_politicas_w", SUBDIR)


# ================================================================
# 2) Retornos de los escenarios extremos
# ================================================================

def diag_returns(state):
    ctx_dl = state["ctx_dl"]
    assets = ctx_dl["assets"]
    T_vals = ctx_dl["T_vals"]
    scenarios = ctx_dl["scenarios"]  # (5, T, A)

    rows = []
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    titles = [
        (S_BEAR, "BEAR (s=0, peor escenario)",       "#E63946"),
        (S_BULL, "BULL (s=4, mejor escenario)",      "#1f77b4"),
    ]
    for ax, (s_idx, title, base_color) in zip(axes, titles):
        for ai, a in enumerate(assets):
            r = scenarios[s_idx, :, ai]
            cum = np.cumprod(1.0 + r) - 1.0
            ax.plot(T_vals, cum, lw=1.3,
                    label=f"{a} (cum final = {cum[-1]:+.1%})")
            for t, rv, cv in zip(T_vals, r, cum):
                rows.append({"scenario": s_idx, "asset": a,
                             "t": int(t), "r": float(rv),
                             "cum_return": float(cv)})
        ax.axhline(0, color="grey", lw=0.6)
        ax.set_title(f"Retorno acumulado por activo — {title}")
        ax.set_xlabel("t")
        ax.set_ylabel("retorno acumulado")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=9)
    save_fig(fig, "2_returns_bear_bull", SUBDIR)
    save_csv(pd.DataFrame(rows), "2_returns_bear_bull", SUBDIR)


# ================================================================
# 3) Capital bajo cada politica en bear vs bull
# ================================================================

def diag_capital(state):
    ctx_dl = state["ctx_dl"]
    assets = ctx_dl["assets"]
    T_vals = ctx_dl["T_vals"]
    c_base = ctx_dl["c_base"]
    C0     = ctx_dl["Capital_inicial"]
    scenarios = ctx_dl["scenarios"]

    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    titles = [
        (S_BEAR, "BEAR (s=0, peor escenario)"),
        (S_BULL, "BULL (s=4, mejor escenario)"),
    ]
    for ax, (s_idx, title) in zip(axes, titles):
        scen = scenarios[s_idx]                                    # (T, A)
        # RG g*
        cap_rg = simulate_capital_on_scenario(
            state["RG"]["w"], state["RG"]["u"], state["RG"]["v"],
            scen, assets, c_base, C0, T_vals,
        )
        # OPT base
        cap_opt = simulate_capital_on_scenario(
            state["OPT"]["w"], state["OPT"]["u"], state["OPT"]["v"],
            scen, assets, c_base, C0, T_vals,
        )
        # Naive 50/50 rebalanceado (proxy realista de costos via drift).
        cap_naive = _simulate_naive_rebal_on_scenario(
            scen, assets, c_base, C0, T_vals,
        )

        ax.plot(T_vals, [cap_rg [t] for t in T_vals],
                color=state["RG" ]["color"], lw=1.6,
                label=f"{state['RG' ]['label']}  (fin = ${cap_rg [T_vals[-1]]:,.0f})")
        ax.plot(T_vals, [cap_opt[t] for t in T_vals],
                color=state["OPT"]["color"], lw=1.6,
                label=f"{state['OPT']['label']}  (fin = ${cap_opt[T_vals[-1]]:,.0f})")
        ax.plot(T_vals, [cap_naive[t] for t in T_vals],
                color="#8B3A1F", lw=1.3,
                label=f"Naive 50/50 rebal  (fin = ${cap_naive[T_vals[-1]]:,.0f})")
        ax.axhline(C0, color="#666", ls="--", lw=0.8, label=f"C0 = ${C0:,.0f}")
        ax.set_title(f"Capital bajo cada politica — {title}")
        ax.set_xlabel("t")
        if s_idx == S_BEAR:
            ax.set_ylabel("Capital")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)

        for t in T_vals:
            rows.append({"scenario": s_idx, "t": int(t),
                         "RG":    float(cap_rg [t]),
                         "OPT":   float(cap_opt[t]),
                         "Naive": float(cap_naive[t])})
    save_fig(fig, "3_capital_bear_bull", SUBDIR)
    save_csv(pd.DataFrame(rows), "3_capital_bear_bull", SUBDIR)


# ================================================================
# 4) Resumen terminal
# ================================================================

def diag_resumen(state):
    ctx_dl = state["ctx_dl"]
    assets = ctx_dl["assets"]
    T_vals = ctx_dl["T_vals"]
    c_base = ctx_dl["c_base"]
    C0     = ctx_dl["Capital_inicial"]
    scenarios = ctx_dl["scenarios"]
    t_f = T_vals[-1]

    rows = []
    for s_idx, name in [(S_BEAR, "BEAR (s=0)"), (S_BULL, "BULL (s=4)")]:
        scen = scenarios[s_idx]
        cap_rg = simulate_capital_on_scenario(
            state["RG"]["w"], state["RG"]["u"], state["RG"]["v"],
            scen, assets, c_base, C0, T_vals,
        )
        cap_opt = simulate_capital_on_scenario(
            state["OPT"]["w"], state["OPT"]["u"], state["OPT"]["v"],
            scen, assets, c_base, C0, T_vals,
        )
        cap_naive = _simulate_naive_rebal_on_scenario(
            scen, assets, c_base, C0, T_vals,
        )
        rows.append({
            "escenario":     name,
            "RG_cap_final":    float(cap_rg [t_f]),
            "OPT_cap_final":   float(cap_opt[t_f]),
            "Naive_cap_final": float(cap_naive[t_f]),
            "RG_ret_acum":     float(cap_rg [t_f] / C0 - 1),
            "OPT_ret_acum":    float(cap_opt[t_f] / C0 - 1),
            "Naive_ret_acum":  float(cap_naive[t_f] / C0 - 1),
        })
    df = pd.DataFrame(rows)
    save_csv(df, "4_resumen_terminal", SUBDIR)
    print("\n--- Resumen terminal (escenarios extremos) ---")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# ================================================================
# Main
# ================================================================

def main():
    print("=" * 70)
    print("REBALANCEO Y CAPITAL EN ESCENARIOS EXTREMOS (BEAR / BULL)")
    print("=" * 70)
    state = solve_policies()
    diag_politicas(state)
    diag_returns(state)
    diag_capital(state)
    diag_resumen(state)
    print("\n" + "=" * 70)
    print("Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
