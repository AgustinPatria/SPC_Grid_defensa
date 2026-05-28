"""Investigacion: por que OPT base pierde contra Naive 50/50 en el OOS?

In-sample: OPT +45.22% vs Naive +20.92% (OPT gana fuerte).
OOS (t=148..163): OPT +22.26% vs Naive +29.75% (¡naive gana!).

Hipotesis a probar:
  A. In-sample OPT usa mu_hat estimado sobre TODA la serie incluyendo
     el periodo test → leakage informacional => benchmark injusto.
  B. OPT OOS sufre porque mu_hat estimado solo en train no refleja la
     dinamica del test → su politica esta "desactualizada".
  C. OPT paga mas costos de transaccion que naive en OOS.
  D. OPT overweight el activo equivocado para el periodo test.

Outputs: inspeccion/oos_investigacion_out/
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

from config import DATA_DIR, REGIMES, SPLIT, T_HORIZON                  # noqa: E402
from Regret_Grid import (                                               # noqa: E402
    build_hist_ctx_oos,
    compute_test_start_t,
    load_market_data,
    simulate_capital_opt,
    simulate_naive_bh,
    simulate_naive_rb,
    solve_portfolio,
    _compute_hist_moments,
)

OUT = Path(__file__).resolve().parent / "oos_investigacion_out"
OUT.mkdir(exist_ok=True)

_LOG = []


def say(*args):
    line = " ".join(str(a) for a in args)
    print(line)
    _LOG.append(line)


# ====================================================================
# Setup
# ====================================================================

def setup():
    t_test_start = compute_test_start_t(T=T_HORIZON, split=SPLIT)
    say("=" * 78)
    say("INVESTIGACION OOS — OPT vs Naive")
    say("=" * 78)
    say(f"\nSplit del LSTM: {SPLIT}  H={60}  T={T_HORIZON}")
    say(f"t_test_start = {t_test_start}  =>  test = t{t_test_start}..t{T_HORIZON}")
    say(f"Periodos de test: {T_HORIZON - t_test_start + 1}")

    ctx_full = load_market_data(str(DATA_DIR))                  # in-sample (cheating)
    ctx_oos  = build_hist_ctx_oos(DATA_DIR, t_test_start, T_HORIZON)  # honest OOS
    return ctx_full, ctx_oos, t_test_start


# ====================================================================
# 1. Comparar mu_hat in-sample vs train-only vs test-only
# ====================================================================

def investigacion1_muhat(ctx_full, ctx_oos, t_test_start):
    say("\n" + "=" * 78)
    say("INVESTIGACION 1 — mu_hat: in-sample vs train-only vs test-only")
    say("=" * 78)
    assets = list(ctx_full["assets"])
    regimes = list(REGIMES)
    r_full  = ctx_full["r"]
    p_full  = ctx_full["p_hist"]

    # mu_hat in-sample: usa toda la serie (lo que usa OPT base in-sample)
    mu_full, sigma_full = _compute_hist_moments(
        r_full, p_full, assets, regimes, moments_window=None,
    )
    # mu_hat train-only: t=1..t_test_start-1 (lo que usa OPT_oos)
    mu_train, sigma_train = _compute_hist_moments(
        r_full, p_full, assets, regimes,
        moments_window=(1, t_test_start - 1),
    )
    # mu_hat test-only: t=t_test_start..T (lo que "deberia haber sabido")
    mu_test, sigma_test = _compute_hist_moments(
        r_full, p_full, assets, regimes,
        moments_window=(t_test_start, T_HORIZON),
    )

    rows = []
    for i in assets:
        for k in regimes:
            rows.append({
                "asset":  i,
                "regime": k,
                "mu_full":  mu_full[(i, k)],
                "mu_train": mu_train[(i, k)],
                "mu_test":  mu_test[(i, k)],
                "train-full":  mu_train[(i, k)] - mu_full[(i, k)],
                "test-train":  mu_test[(i, k)]  - mu_train[(i, k)],
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "1_muhat_comparison.csv", index=False)
    say("\nmu_hat por activo y regimen, segun ventana de estimacion:")
    say(df.to_string(index=False, float_format=lambda x: f"{x:+.4%}"))

    say("\nLectura:")
    for i in assets:
        for k in regimes:
            ft = mu_full[(i, k)]
            tr = mu_train[(i, k)]
            te = mu_test[(i, k)]
            if abs(te - tr) > abs(tr) * 2 or (te * tr < 0 and abs(te) + abs(tr) > 0.005):
                say(f"  [!] {i} {k}: mu_train={tr:+.3%} pero mu_test={te:+.3%} "
                    f"-> OPT_oos usa estimador desactualizado vs realidad del test")

    # Realized r mean en el periodo test
    say("\nRetorno medio realizado en el periodo TEST (t148..t163):")
    for i in assets:
        r_t = r_full[i].loc[t_test_start:T_HORIZON]
        say(f"  {i}: mean={r_t.mean():+.4%}/sem  std={r_t.std():.4%}  "
            f"cum={(1+r_t).prod()-1:+.2%}")

    return mu_train, mu_test, df


# ====================================================================
# 2. Solve y comparar politicas: OPT in-sample, OPT OOS, Naive
# ====================================================================

def investigacion2_politicas(ctx_full, ctx_oos, t_test_start):
    say("\n" + "=" * 78)
    say("INVESTIGACION 2 — Politicas: OPT in-sample vs OPT_oos vs Naive")
    say("=" * 78)
    assets = list(ctx_oos["assets"])

    # OPT in-sample sobre toda la serie (uses future mu_hat — cheating)
    _, w_opt_full, u_opt_full, v_opt_full, _ = solve_portfolio(
        ctx_full, lambda_riesgo=1.00, costo_mult=1.0,
    )
    # OPT OOS: solo train data
    _, w_opt_oos, u_opt_oos, v_opt_oos, _ = solve_portfolio(
        ctx_oos, lambda_riesgo=1.00, costo_mult=1.0,
    )

    T_oos = ctx_oos["T_vals"]

    # Plot 1: w(SPX, t) y w(CMC, t) sobre el periodo OOS
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for ai, asset in enumerate(assets):
        w_full = [w_opt_full[asset, t] for t in T_oos]
        w_oos  = [w_opt_oos[asset, t]  for t in T_oos]
        axes[ai].plot(T_oos, w_full, "o-", color="#F2B705", linewidth=2,
                      markersize=6, label="OPT in-sample (con futuro)")
        axes[ai].plot(T_oos, w_oos,  "s-", color="#1f77b4", linewidth=2,
                      markersize=6, label="OPT_oos (solo train)")
        axes[ai].axhline(0.5, color="#8B3A1F", linewidth=1.5, linestyle="--",
                         label="Naive 50/50")
        axes[ai].set_ylabel(f"w({asset}, t)")
        axes[ai].set_ylim(-0.05, 1.05)
        axes[ai].grid(True, alpha=0.3)
        axes[ai].legend(loc="best", fontsize=9)
    axes[-1].set_xlabel("t")
    axes[0].set_title("Politicas en periodo OOS — OPT in-sample (cheating) vs OPT_oos vs Naive")
    fig.tight_layout()
    fig.savefig(OUT / "2_politicas_oos.png", dpi=140)
    plt.close(fig)

    # Tabla con los pesos
    rows = []
    for t in T_oos:
        row = {"t": t}
        for asset in assets:
            row[f"w_full_{asset}"] = w_opt_full[asset, t]
            row[f"w_oos_{asset}"]  = w_opt_oos[asset, t]
        rows.append(row)
    pol_df = pd.DataFrame(rows)
    pol_df.to_csv(OUT / "2_politicas_oos.csv", index=False)
    say("\nPoliticas en el OOS (primeras 8 semanas y ultimas 4):")
    say(pol_df.head(8).to_string(index=False,
                                  float_format=lambda x: f"{x:.4f}"))
    say("...")
    say(pol_df.tail(4).to_string(index=False,
                                  float_format=lambda x: f"{x:.4f}"))

    return (w_opt_full, u_opt_full, v_opt_full,
            w_opt_oos, u_opt_oos, v_opt_oos)


# ====================================================================
# 3. Descomponer capital por periodo: retorno bruto vs costos
# ====================================================================

def investigacion3_descomposicion(ctx_oos,
                                   w_opt_full, u_opt_full, v_opt_full,
                                   w_opt_oos, u_opt_oos, v_opt_oos):
    say("\n" + "=" * 78)
    say("INVESTIGACION 3 — Descomposicion por periodo: retorno vs costos")
    say("=" * 78)
    T_vals  = ctx_oos["T_vals"]
    assets  = list(ctx_oos["assets"])
    r       = ctx_oos["r"]
    c_base  = ctx_oos["c_base"]
    C0      = ctx_oos["Capital_inicial"]

    # Capital simulado para cada politica
    cap_opt_full = simulate_capital_opt(w_opt_full, u_opt_full, v_opt_full, ctx_oos)
    cap_opt_oos  = simulate_capital_opt(w_opt_oos,  u_opt_oos,  v_opt_oos,  ctx_oos)
    cap_rb       = simulate_naive_rb(ctx_oos)
    cap_bh       = simulate_naive_bh(ctx_oos)

    # Descomposicion: para cada politica, contribucion de retorno bruto y costo
    def decompose(w_sol, u_sol, v_sol, label):
        rows = []
        cum_ret = 1.0
        cum_cost = 1.0
        for idx in range(1, len(T_vals)):
            t      = T_vals[idx]
            t_prev = T_vals[idx - 1]
            r_port = sum(w_sol[i, t_prev] * r[i].loc[t_prev] for i in assets)
            turn   = sum(c_base[i] * (u_sol[i, t_prev] + v_sol[i, t_prev]) for i in assets)
            cum_ret  *= (1.0 + r_port)
            cum_cost *= (1.0 - turn)
            rows.append({
                "t":       t,
                "r_port":  r_port,
                "cost":    turn,
                "net":     r_port - turn,
                "cum_ret": cum_ret - 1,
                "cum_cost_drag": cum_cost - 1,
            })
        return pd.DataFrame(rows), cum_ret - 1, cum_cost - 1

    df_full, ret_full, cost_full = decompose(w_opt_full, u_opt_full, v_opt_full, "OPT in-sample")
    df_oos,  ret_oos,  cost_oos  = decompose(w_opt_oos,  u_opt_oos,  v_opt_oos,  "OPT_oos")

    # Naive RB: hay que re-derivar w(t), u(t), v(t) implicitos
    # (rebalance a 0.5 cada semana)
    say("\nDescomposicion del capital OOS (sin redondeo):")
    say(f"  OPT in-sample: ret bruto={ret_full:+.4f}  cost drag={cost_full:+.4f}  cap_final=${cap_opt_full[T_vals[-1]]:,.2f}")
    say(f"  OPT_oos:       ret bruto={ret_oos:+.4f}  cost drag={cost_oos:+.4f}  cap_final=${cap_opt_oos[T_vals[-1]]:,.2f}")
    say(f"  Naive RB:                                                cap_final=${cap_rb[T_vals[-1]]:,.2f}")
    say(f"  Naive BH:                                                cap_final=${cap_bh[T_vals[-1]]:,.2f}")

    df_full.to_csv(OUT / "3_decompose_opt_full.csv", index=False)
    df_oos.to_csv(OUT / "3_decompose_opt_oos.csv", index=False)

    # Plot: evolucion del capital
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(T_vals, [cap_opt_full[t] for t in T_vals], "o-",
            color="#F2B705", linewidth=2, markersize=5,
            label=f"OPT in-sample [CHEATING] (cap={cap_opt_full[T_vals[-1]]:,.0f})")
    ax.plot(T_vals, [cap_opt_oos[t]  for t in T_vals], "s-",
            color="#1f77b4", linewidth=2, markersize=5,
            label=f"OPT_oos (cap={cap_opt_oos[T_vals[-1]]:,.0f})")
    ax.plot(T_vals, [cap_rb[t] for t in T_vals], "v-",
            color="#8B3A1F", linewidth=2, markersize=5,
            label=f"Naive 50/50 RB (cap={cap_rb[T_vals[-1]]:,.0f})")
    ax.plot(T_vals, [cap_bh[t] for t in T_vals], "d-",
            color="#E63946", linewidth=1.5, markersize=5,
            label=f"Naive 50/50 BH (cap={cap_bh[T_vals[-1]]:,.0f})")
    ax.axhline(C0, color="black", linewidth=0.6, alpha=0.5,
               linestyle="--", label=f"Capital inicial (${C0:,.0f})")
    ax.set_xlabel("t (OOS, test del split LSTM)")
    ax.set_ylabel("Capital ($)")
    ax.set_title("Evolucion de capital en el periodo OOS — OPT in-sample (cheating) vs OPT_oos vs naive")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "3_evolucion_capital_oos_detallado.png", dpi=140)
    plt.close(fig)

    return cap_opt_full, cap_opt_oos, cap_rb, cap_bh


# ====================================================================
# 4. Veredicto y respuesta a la pregunta original
# ====================================================================

def investigacion4_veredicto(mu_train, mu_test, ctx_oos,
                              cap_opt_full, cap_opt_oos, cap_rb, cap_bh,
                              w_opt_oos):
    say("\n" + "=" * 78)
    say("INVESTIGACION 4 — Veredicto")
    say("=" * 78)
    assets = list(ctx_oos["assets"])
    T_vals = ctx_oos["T_vals"]
    r = ctx_oos["r"]
    C0 = ctx_oos["Capital_inicial"]
    t_first = T_vals[0]
    t_last  = T_vals[-1]

    # Retornos realizados (mean y cum)
    r_real = {}
    for i in assets:
        r_t = r[i]
        r_real[i] = {
            "mean": r_t.mean(),
            "cum":  (1 + r_t).prod() - 1,
            "std":  r_t.std(),
        }

    # mu_hat train: ¿coincide con la realidad del test?
    say("\nDiagnostico del 'lo que OPT_oos creyo vs lo que paso':")
    for i in assets:
        say(f"\n  {i}:")
        for k in ["bull", "bear"]:
            mt = mu_train[(i, k)]
            me = mu_test[(i, k)]
            say(f"    mu_hat({k}, train) = {mt:+.4%}/sem  "
                f"|  mu_hat({k}, test) = {me:+.4%}/sem  "
                f"|  diff = {me-mt:+.4%}")
        say(f"    Realidad test (mean retorno): {r_real[i]['mean']:+.4%}/sem  "
            f"(cum {r_real[i]['cum']:+.2%}, std {r_real[i]['std']:.4%})")

    # Peso medio de OPT_oos vs naive (50/50)
    say("\nPeso medio de la politica OPT_oos en el periodo OOS:")
    for asset in assets:
        ws = [w_opt_oos[asset, t] for t in T_vals]
        say(f"  {asset}: w_mean={np.mean(ws):.4f}  (naive seria 0.50)")

    # Resumen
    say("\n" + "-" * 78)
    say("RESUMEN")
    say("-" * 78)
    say(f"  OPT in-sample (cheating, mu_hat ve futuro): ${cap_opt_full[t_last]:,.2f}  "
        f"({cap_opt_full[t_last]/C0 - 1:+.2%})")
    say(f"  OPT_oos (honest, mu_hat solo train):        ${cap_opt_oos[t_last]:,.2f}  "
        f"({cap_opt_oos[t_last]/C0 - 1:+.2%})")
    say(f"  Naive 50/50 RB:                             ${cap_rb[t_last]:,.2f}  "
        f"({cap_rb[t_last]/C0 - 1:+.2%})")
    say(f"  Naive 50/50 BH:                             ${cap_bh[t_last]:,.2f}  "
        f"({cap_bh[t_last]/C0 - 1:+.2%})")
    diff_cheat = cap_opt_full[t_last] - cap_opt_oos[t_last]
    say(f"\n  Ventaja informacional del cheating: ${diff_cheat:,.2f}  "
        f"({(cap_opt_full[t_last] - cap_opt_oos[t_last]) / C0:+.2%} sobre capital inicial)")


# ====================================================================
# Main
# ====================================================================

def main():
    ctx_full, ctx_oos, t_test_start = setup()
    mu_train, mu_test, _ = investigacion1_muhat(ctx_full, ctx_oos, t_test_start)
    (w_opt_full, u_opt_full, v_opt_full,
     w_opt_oos,  u_opt_oos,  v_opt_oos) = investigacion2_politicas(
        ctx_full, ctx_oos, t_test_start,
    )
    cap_opt_full, cap_opt_oos, cap_rb, cap_bh = investigacion3_descomposicion(
        ctx_oos,
        w_opt_full, u_opt_full, v_opt_full,
        w_opt_oos,  u_opt_oos,  v_opt_oos,
    )
    investigacion4_veredicto(
        mu_train, mu_test, ctx_oos,
        cap_opt_full, cap_opt_oos, cap_rb, cap_bh, w_opt_oos,
    )

    (OUT / "log.txt").write_text("\n".join(_LOG), encoding="utf-8")
    say(f"\nLog guardado en: {OUT / 'log.txt'}")
    say(f"Plots y CSVs:    {OUT}")


if __name__ == "__main__":
    main()
