"""Ablacion del padding de p_bull en predict_pbull_walking.

Pregunta: el padding `p_bull[:H] = p_bull[H]` introduce un artefacto (los primeros
H=60 periodos tienen p_bull constante, por ende mu_mix/sigma_mix tambien). El
padding existe solo para preservar la alineacion T_vals = 1..163 del GAMS base.

Experimento: comparar el pipeline con padding (estado actual) vs el pipeline sin
padding, donde se trimea todo a t=H+1..T y se re-indexa a t=1..(T-H). Los dos
setups corren la misma grilla (lambda, m) y se comparan via:
  - g*_mean elegido por cada setup
  - V terminal en escenarios DL (mean/min/max)
  - backtest historico contra OPT base + Naive (rb) + Naive (B&H), cada uno
    evaluado sobre el calendario que le corresponde (full para el padded;
    semanas H+1..T del historico para el no-pad).

Calendarios:
  - Padded setup:  policia y baselines simulados sobre r_hist[t=1..T].
  - No-pad setup:  policia y baselines simulados sobre r_hist[t=H+1..T]
                   (re-indexado a t=1..T-H, capital inicial = C0). De este modo
                   ambos setups arrancan con el mismo C0 y la comparacion del
                   retorno % es coherente para cada uno.

NOTA: el horizonte es distinto (163 vs 103 semanas). El cambio de retorno
acumulado no es directamente comparable entre setups; lo informativo es:
  - dentro de cada setup, donde queda el regret-grid vs OPT vs naive.
  - g*_mean elegido: si cambia mucho, el padding estaba sesgando la seleccion.

Corre con:
    python -m inspeccion.padding_ablation
    (o)
    python inspeccion/padding_ablation.py

Outputs (en inspeccion/padding_ablation_out/):
  1_pbull_compare.png/csv      p_bull(t) padded vs no-pad por activo
  2_grid_compare.csv           g*_mean, g*_worst, regret por setup
  3_grid_compare.png           heatmaps mean_R por setup, lado-a-lado
  4_backtest_padded.csv/png    capital de OPT/Naive/RG en setup padded
  4_backtest_nopad.csv/png     capital de OPT/Naive/RG en setup no-pad
  5_terminal_summary.csv/png   V_final por (politica, setup), retorno acumulado
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
    LAMBDA_GRID,
    M_GRID,
    N_CANDIDATES,
    N_SCENARIOS,
    SCENARIO_SEED,
    SUMMARY_ASSET,
    T_HORIZON,
)
from dl.prediccion_deciles import load_checkpoint
from Regret_Grid import (
    build_dl_context,
    compute_regret_and_select,
    load_market_data,
    run_regret_grid,
    simulate_capital_opt,
    simulate_naive_bh,
    simulate_naive_rb,
    solve_portfolio,
    trim_post_warmup,
)


SUBDIR = "padding_ablation"


# ================================================================
# 1) Setup: construir los 4 contextos (DL y OPT base x padded y no-pad)
# ================================================================

def build_setups():
    print("Cargando checkpoint DL para leer H...")
    H = load_checkpoint(CHECKPOINT_PATH).config.H
    T = T_HORIZON
    T_eff = T - H
    print(f"  H = {H}  T = {T}  =>  no-pad: T_eff = {T_eff} periodos")

    print("\nConstruyendo DL ctx padded (lleva ~10s)...")
    # Pinned al setup legacy (walking + p_dl) porque este experimento
    # investiga el padding del walking. Con defaults nuevos (rollout) no
    # habria padding que ablacionar — no hay zona [1..H] sin prediccion.
    dl_padded = build_dl_context(
        data_dir=DATA_DIR, checkpoint_path=CHECKPOINT_PATH,
        T=T, N_candidates=N_CANDIDATES, n_scenarios=N_SCENARIOS,
        seed=SCENARIO_SEED, summary_asset=SUMMARY_ASSET,
        p_method="walking", mu_hat_source="p_dl",
    )
    print(f"  DL padded: T_vals = 1..{dl_padded['nT']}, "
          f"scenarios = {dl_padded['scenarios'].shape}")

    print("Construyendo DL ctx no-pad (trim_post_warmup)...")
    dl_nopad = trim_post_warmup(dl_padded, H=H, T_max=T)
    print(f"  DL no-pad: T_vals = 1..{dl_nopad['nT']}, "
          f"scenarios = {dl_nopad['scenarios'].shape}")

    print("Construyendo OPT base ctx padded (load_market_data)...")
    opt_padded = load_market_data(str(DATA_DIR))
    # Asegurar mismo T que DL para comparacion justa.
    opt_padded = trim_post_warmup(opt_padded, H=0, T_max=T)
    print(f"  OPT padded: T_vals = 1..{opt_padded['nT']}")

    print("Construyendo OPT base ctx no-pad (mismo trim que DL)...")
    opt_nopad = trim_post_warmup(load_market_data(str(DATA_DIR)), H=H, T_max=T)
    print(f"  OPT no-pad: T_vals = 1..{opt_nopad['nT']}")

    return {
        "H": H, "T": T, "T_eff": T_eff,
        "dl_padded":  dl_padded,
        "dl_nopad":   dl_nopad,
        "opt_padded": opt_padded,
        "opt_nopad":  opt_nopad,
    }


# ================================================================
# 2) Diagnostico 1: p_bull padded vs no-pad
# ================================================================

def diag_pbull_compare(ctx):
    dl_pad = ctx["dl_padded"]
    dl_nop = ctx["dl_nopad"]
    H = ctx["H"]
    assets = dl_pad["assets"]

    fig, axes = plt.subplots(len(assets), 1, figsize=(11, 3.5 * len(assets)))
    if len(assets) == 1:
        axes = [axes]
    rows = []
    for ai, a in enumerate(assets):
        ax = axes[ai]
        p_pad = dl_pad["p_dl"][a]["bull"]
        p_nop = dl_nop["p_dl"][a]["bull"]
        # Para visualizar el no-pad sobre el calendario original, lo desplazamos
        # H posiciones a la derecha (t_orig = H + t_nopad).
        t_pad = list(p_pad.index)
        t_nop_orig = [H + t for t in p_nop.index]

        ax.plot(t_pad, p_pad.values, color="#E63946", lw=1.2,
                label=f"padded ({len(p_pad)} pts)")
        ax.plot(t_nop_orig, p_nop.values, color="#1f77b4", lw=1.2,
                label=f"no-pad ({len(p_nop)} pts, t orig = H+k)")
        ax.axvspan(0, H, color="#ccc", alpha=0.4,
                   label=f"zona padded (t=1..H={H})")
        ax.set_title(f"p_bull({a}) - padded vs no-pad")
        ax.set_xlabel("t (calendario original)")
        ax.set_ylabel(f"p_bull({a})")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9, loc="best")

        for t, v in zip(t_pad, p_pad.values):
            rows.append({"asset": a, "setup": "padded",
                         "t_orig": int(t), "p_bull": float(v)})
        for t_orig, v in zip(t_nop_orig, p_nop.values):
            rows.append({"asset": a, "setup": "nopad",
                         "t_orig": int(t_orig), "p_bull": float(v)})

    save_fig(fig, "1_pbull_compare", SUBDIR)
    save_csv(pd.DataFrame(rows), "1_pbull_compare", SUBDIR)


# ================================================================
# 3) Diagnostico 2 y 3: regret grid en ambos setups
# ================================================================

def run_grids(ctx):
    lambda_grid = list(LAMBDA_GRID)
    m_grid = list(M_GRID)

    print(f"\n--- Corriendo grilla {len(lambda_grid)}x{len(m_grid)}="
          f"{len(lambda_grid)*len(m_grid)} solves en DL padded ---")
    V_df_pad, pol_pad = run_regret_grid(ctx["dl_padded"], lambda_grid, m_grid)
    res_pad = compute_regret_and_select(V_df_pad)

    print(f"\n--- Corriendo grilla {len(lambda_grid)}x{len(m_grid)}="
          f"{len(lambda_grid)*len(m_grid)} solves en DL no-pad ---")
    V_df_nop, pol_nop = run_regret_grid(ctx["dl_nopad"], lambda_grid, m_grid)
    res_nop = compute_regret_and_select(V_df_nop)

    return {
        "lambda_grid": lambda_grid, "m_grid": m_grid,
        "V_df_pad": V_df_pad, "pol_pad": pol_pad, "res_pad": res_pad,
        "V_df_nop": V_df_nop, "pol_nop": pol_nop, "res_nop": res_nop,
    }


def diag_grid_compare(ctx, grids):
    res_pad = grids["res_pad"]
    res_nop = grids["res_nop"]

    rows = []
    for label, res in [("padded", res_pad), ("nopad", res_nop)]:
        lam_m, m_m = res["g_mean"]
        lam_w, m_w = res["g_worst"]
        V_mean_row  = res["V_table"].loc[(lam_m, m_m)]
        V_worst_row = res["V_table"].loc[(lam_w, m_w)]
        C0 = ctx["dl_padded"]["Capital_inicial"]
        rows.append({
            "setup": label,
            "g_mean_lambda": float(lam_m), "g_mean_m": float(m_m),
            "g_mean_regret": float(res["g_mean_metric"]),
            "g_mean_V_avg":  float(V_mean_row.mean()),
            "g_mean_V_min":  float(V_mean_row.min()),
            "g_mean_V_max":  float(V_mean_row.max()),
            "g_mean_ret_avg":  float(V_mean_row.mean()/C0 - 1),
            "g_worst_lambda": float(lam_w), "g_worst_m": float(m_w),
            "g_worst_regret": float(res["g_worst_metric"]),
            "g_worst_V_avg":  float(V_worst_row.mean()),
            "g_worst_V_min":  float(V_worst_row.min()),
            "g_worst_V_max":  float(V_worst_row.max()),
            "g_worst_ret_min": float(V_worst_row.min()/C0 - 1),
        })
    save_csv(pd.DataFrame(rows), "2_grid_compare", SUBDIR)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, label, res in [
        (axes[0], "padded", res_pad),
        (axes[1], "nopad",  res_nop),
    ]:
        R_tab = res["R_table"]
        mean_R = R_tab.mean(axis=1).unstack("m").sort_index().sort_index(axis=1)
        im = ax.imshow(mean_R.values, cmap="RdYlGn_r", aspect="auto")
        ax.set_xticks(range(len(mean_R.columns)))
        ax.set_xticklabels([f"{m:.1f}" for m in mean_R.columns])
        ax.set_yticks(range(len(mean_R.index)))
        ax.set_yticklabels([f"{l:.2f}" for l in mean_R.index])
        ax.set_xlabel("m"); ax.set_ylabel("lambda")
        lam_m, m_m = res["g_mean"]
        ax.set_title(f"setup={label}   g*_mean=({lam_m:.2f}, {m_m:.1f})\n"
                     f"mean_regret = ${res['g_mean_metric']:,.0f}")
        for i in range(mean_R.shape[0]):
            for j in range(mean_R.shape[1]):
                ax.text(j, i, f"${mean_R.values[i, j]:,.0f}",
                        ha="center", va="center", fontsize=7)
        j_opt = list(mean_R.columns).index(m_m)
        i_opt = list(mean_R.index).index(lam_m)
        ax.scatter([j_opt], [i_opt], s=200, marker="*", color="cyan",
                   edgecolor="k", linewidth=1.5, zorder=3)
    save_fig(fig, "3_grid_compare", SUBDIR)


# ================================================================
# 4) Backtest historico en cada setup
# ================================================================

def _backtest_setup(label, dl_ctx, opt_ctx, res, policies):
    """Corre OPT base, naive (rb), naive (B&H) y g*_mean DL sobre el historico
    del setup. Devuelve dict con curvas y resumen."""
    lam_m, m_m = res["g_mean"]
    w_star, u_star, v_star, _z = policies[(lam_m, m_m)]

    # OPT base: re-solve sobre opt_ctx (mu_mix historico, lambda=1.00, m=1.0).
    _, w_opt, u_opt, v_opt, _ = solve_portfolio(
        opt_ctx, lambda_riesgo=1.00, costo_mult=1.0,
    )

    # Para simular sobre el historico del setup, usamos dl_ctx (que tiene r
    # ya alineado al calendario del setup). Los pesos w_opt y w_star comparten
    # las claves (i, t) con t = T_vals del setup.
    cap_opt = simulate_capital_opt(w_opt, u_opt, v_opt, dl_ctx)
    cap_rg  = simulate_capital_opt(w_star, u_star, v_star, dl_ctx)
    cap_rb  = simulate_naive_rb(dl_ctx)
    cap_bh  = simulate_naive_bh(dl_ctx)

    T_vals = dl_ctx["T_vals"]
    C0 = dl_ctx["Capital_inicial"]
    t_f = T_vals[-1]

    summary = {
        "setup": label,
        "n_periodos": len(T_vals),
        "Capital_inicial": float(C0),
        "OPT_cap_final":   float(cap_opt[t_f]),
        "RG_cap_final":    float(cap_rg [t_f]),
        "NaiveRB_cap_final": float(cap_rb [t_f]),
        "NaiveBH_cap_final": float(cap_bh [t_f]),
        "OPT_ret_acum":      float(cap_opt[t_f]/C0 - 1),
        "RG_ret_acum":       float(cap_rg [t_f]/C0 - 1),
        "NaiveRB_ret_acum":  float(cap_rb [t_f]/C0 - 1),
        "NaiveBH_ret_acum":  float(cap_bh [t_f]/C0 - 1),
        "g_mean_lambda": float(lam_m), "g_mean_m": float(m_m),
    }
    return {
        "summary": summary, "T_vals": T_vals,
        "cap_opt": cap_opt, "cap_rg": cap_rg,
        "cap_rb":  cap_rb,  "cap_bh": cap_bh,
        "lam_m": lam_m, "m_m": m_m,
    }


def diag_backtest(ctx, grids):
    bt_pad = _backtest_setup(
        "padded", ctx["dl_padded"], ctx["opt_padded"],
        grids["res_pad"], grids["pol_pad"],
    )
    bt_nop = _backtest_setup(
        "nopad",  ctx["dl_nopad"],  ctx["opt_nopad"],
        grids["res_nop"], grids["pol_nop"],
    )

    for tag, bt in [("padded", bt_pad), ("nopad", bt_nop)]:
        T_vals = bt["T_vals"]
        rows = []
        for t in T_vals:
            rows.append({
                "t": int(t),
                "OPT":    float(bt["cap_opt"][t]),
                "RG":     float(bt["cap_rg" ][t]),
                "NaiveRB":float(bt["cap_rb" ][t]),
                "NaiveBH":float(bt["cap_bh" ][t]),
            })
        save_csv(pd.DataFrame(rows), f"4_backtest_{tag}", SUBDIR)

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(T_vals, [bt["cap_opt"][t] for t in T_vals],
                color="#F2B705", lw=1.8, label="OPT base")
        ax.plot(T_vals, [bt["cap_rg" ][t] for t in T_vals],
                color="#1f77b4", lw=1.8,
                label=f"Regret-Grid g*_mean (lambda={bt['lam_m']:.2f}, m={bt['m_m']:.1f})")
        ax.plot(T_vals, [bt["cap_rb" ][t] for t in T_vals],
                color="#8B3A1F", lw=1.2, label="Naive 50/50 (rebal)")
        ax.plot(T_vals, [bt["cap_bh" ][t] for t in T_vals],
                color="#E63946", lw=1.2, label="Naive 50/50 (buy&hold)")
        ax.axhline(ctx["dl_padded"]["Capital_inicial"],
                   color="#666", ls="--", lw=0.8, label="C0")
        ax.set_title(f"Backtest historico - setup {tag} ({len(T_vals)} periodos)")
        ax.set_xlabel("t" + (" (re-indexado, t=1 -> semana H+1)" if tag == "nopad" else ""))
        ax.set_ylabel("Capital")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9, loc="best")
        save_fig(fig, f"4_backtest_{tag}", SUBDIR)

    return bt_pad, bt_nop


# ================================================================
# 5) Resumen terminal: V_final por (politica, setup)
# ================================================================

def diag_terminal_summary(ctx, bt_pad, bt_nop):
    rows = [bt_pad["summary"], bt_nop["summary"]]
    df = pd.DataFrame(rows)
    save_csv(df, "5_terminal_summary", SUBDIR)

    politicas = ["OPT", "RG", "NaiveRB", "NaiveBH"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, (tag, bt) in zip(axes, [("padded", bt_pad), ("nopad", bt_nop)]):
        rets = [bt["summary"][f"{p}_ret_acum"] for p in politicas]
        colors = ["#F2B705", "#1f77b4", "#8B3A1F", "#E63946"]
        bars = ax.bar(politicas, [100 * r for r in rets], color=colors, alpha=0.85)
        ax.axhline(0, color="grey", lw=0.6)
        ax.set_title(f"setup={tag}  ({bt['summary']['n_periodos']} periodos)")
        ax.set_ylabel("Retorno acumulado [%]")
        for b, r in zip(bars, rets):
            ax.text(b.get_x() + b.get_width()/2, b.get_height(),
                    f"{r*100:+.1f}%", ha="center",
                    va="bottom" if r >= 0 else "top", fontsize=9)
        ax.grid(True, alpha=0.25, axis="y")
    fig.suptitle("Retorno acumulado por politica y setup", fontsize=12)
    save_fig(fig, "5_terminal_summary", SUBDIR)


# ================================================================
# Main
# ================================================================


def main():
    print("=" * 70)
    print("PADDING ABLATION - p_bull[:H] padding vs trim post-warmup")
    print("=" * 70)

    ctx = build_setups()
    diag_pbull_compare(ctx)

    grids = run_grids(ctx)
    diag_grid_compare(ctx, grids)

    bt_pad, bt_nop = diag_backtest(ctx, grids)
    diag_terminal_summary(ctx, bt_pad, bt_nop)

    print("\n" + "=" * 70)
    print("RESUMEN")
    print("=" * 70)
    for label, res in [("padded", grids["res_pad"]), ("nopad", grids["res_nop"])]:
        lam_m, m_m = res["g_mean"]
        print(f"  [{label:<6}] g*_mean = (lambda={lam_m:.2f}, m={m_m:.1f}) "
              f"mean_regret = ${res['g_mean_metric']:,.0f}")
    for bt in [bt_pad, bt_nop]:
        s = bt["summary"]
        print(f"  [{s['setup']:<6}] backtest hist ({s['n_periodos']}p): "
              f"OPT={s['OPT_ret_acum']:+.2%}  "
              f"RG={s['RG_ret_acum']:+.2%}  "
              f"NaiveRB={s['NaiveRB_ret_acum']:+.2%}  "
              f"NaiveBH={s['NaiveBH_ret_acum']:+.2%}")
    print("=" * 70)


if __name__ == "__main__":
    main()
