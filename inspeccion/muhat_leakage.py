"""Leakage estructural en mu_hat: estimacion sobre la muestra completa
vs estimacion restringida a TRAIN del LSTM.

Setup del problema:
  En `_compute_hist_moments`, mu_hat[(i, k)] es UNA constante por (activo,
  regimen) estimada sumando sobre TODO el horizonte de r_hist y p (default).
  Entonces mu_mix(t) = sum_k p_dl(t) * mu_hat(k) hereda esa estimacion
  global incluso para t=1 — lo que implica que el FO al planificar en
  t=1 esta usando un mu_hat estimado con r_hist(t=163).

  Ese es un leakage estructural distinto al del LSTM (que verificamos en
  leakage_check.py): no depende del LSTM, ocurre tambien en OPT base.

Experimento: comparar dos setups, evaluados ambos sobre el mismo periodo
out-of-sample (semanas TEST del LSTM = t=148..163):

  full   mu_hat estimada con t=1..163 (lo que el codigo hace por default).
  train  mu_hat estimada solo con t=1..132 (TRAIN del LSTM, sin VALID/TEST).

Si full y train dan RG (y OPT) similares, mu_hat global no es el problema.
Si dan distintos, la diferencia mide el "boost" que el FO recibia del
leakage de info futura en la estimacion de momentos.

Outputs (inspeccion/muhat_leakage_out/):
  1_muhat_compare.csv       mu_hat(asset, regimen) y diff full vs train
  2_grid_compare.csv/png    g*_mean, regret por setup
  3_backtest_<setup>.png/csv  capital sobre TEST por politica
  4_terminal_summary.csv/png  retorno acumulado lado-a-lado

Corre con:
    python -m inspeccion.muhat_leakage
    (o)
    python inspeccion/muhat_leakage.py
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
    REGIMES,
    SCENARIO_SEED,
    SUMMARY_ASSET,
    T_HORIZON,
)
from dl.prediccion_deciles import load_checkpoint, load_returns
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


SUBDIR = "muhat_leakage"


# ================================================================
# 1) Computar la ventana TRAIN del LSTM
# ================================================================

def compute_train_end():
    """Devuelve la ultima semana que cae en el split TRAIN del LSTM.

    chrono_split parte sobre N = T_dataset - H ventanas. La n-esima ventana
    (n in [0, N-1]) predice la semana t = H + n + 1 (1-indexed). Train son
    las primeras int(N * r_tr) ventanas.
    """
    ckpt = load_checkpoint(CHECKPOINT_PATH)
    H = ckpt.config.H
    r_tr = ckpt.config.split[0]
    T_dataset = len(load_returns())
    N = T_dataset - H
    n_tr = int(N * r_tr)
    t_train_first = H + 1
    t_train_last  = H + n_tr
    return {
        "H": H, "T_dataset": T_dataset,
        "t_train_first": t_train_first, "t_train_last": t_train_last,
        "T_HORIZON": T_HORIZON,
    }


# ================================================================
# 2) Construir los dos setups (full vs train) trimeados a TEST
# ================================================================

def build_setups():
    info = compute_train_end()
    H = info["H"]
    t_train_last = info["t_train_last"]
    T = info["T_HORIZON"]
    print(f"  H = {H}  T_dataset = {info['T_dataset']}  T = {T}")
    print(f"  TRAIN window: t = {info['t_train_first']}..{t_train_last}")

    # Test window = ultimos 16 periodos (mismo recorte que leakage_check.py).
    test_window_size = 16
    H_eff = T - test_window_size
    print(f"  Test window: t = {H_eff + 1}..{T}  ({test_window_size} semanas)\n")

    print("Construyendo DL ctx FULL (mu_hat sobre 1..T, default)...")
    # Pinned al setup legacy (walking + p_dl) porque este experimento
    # contrasta dos ventanas de estimacion de mu_hat MANTENIENDO el resto
    # del pipeline igual al codigo original. Los defaults nuevos
    # (rollout + p_hist) cambiarian el experimento radicalmente.
    dl_full = build_dl_context(
        data_dir=DATA_DIR, checkpoint_path=CHECKPOINT_PATH,
        T=T, N_candidates=N_CANDIDATES, n_scenarios=N_SCENARIOS,
        seed=SCENARIO_SEED, summary_asset=SUMMARY_ASSET,
        moments_window=None,
        p_method="walking", mu_hat_source="p_dl",
    )
    print("Construyendo OPT ctx FULL (mu_hat sobre 1..T)...")
    opt_full = load_market_data(str(DATA_DIR), moments_window=None)

    print("Construyendo DL ctx TRAIN (mu_hat solo con t=1..t_train_last)...")
    dl_train = build_dl_context(
        data_dir=DATA_DIR, checkpoint_path=CHECKPOINT_PATH,
        T=T, N_candidates=N_CANDIDATES, n_scenarios=N_SCENARIOS,
        seed=SCENARIO_SEED, summary_asset=SUMMARY_ASSET,
        moments_window=(1, t_train_last),
        p_method="walking", mu_hat_source="p_dl",
    )
    print("Construyendo OPT ctx TRAIN (mu_hat solo con t=1..t_train_last)...")
    opt_train = load_market_data(str(DATA_DIR),
                                 moments_window=(1, t_train_last))

    print("\nRecortando ambos al TEST window via trim_post_warmup...")
    dl_full_t  = trim_post_warmup(dl_full,  H=H_eff, T_max=T)
    opt_full_t = trim_post_warmup(opt_full, H=H_eff, T_max=T)
    dl_train_t  = trim_post_warmup(dl_train,  H=H_eff, T_max=T)
    opt_train_t = trim_post_warmup(opt_train, H=H_eff, T_max=T)
    print(f"  Setup full : T_vals = 1..{dl_full_t['nT']}, "
          f"scenarios = {dl_full_t['scenarios'].shape}")
    print(f"  Setup train: T_vals = 1..{dl_train_t['nT']}, "
          f"scenarios = {dl_train_t['scenarios'].shape}")

    return {
        "info": info, "test_size": test_window_size,
        "full":  {"dl": dl_full_t,  "opt": opt_full_t,  "raw_dl_ctx": dl_full},
        "train": {"dl": dl_train_t, "opt": opt_train_t, "raw_dl_ctx": dl_train},
    }


# ================================================================
# 3) Diagnostico 1: mu_hat full vs train por (asset, regimen)
# ================================================================

def diag_muhat_compare(ctx):
    rows = []
    for asset_label in ctx["full"]["raw_dl_ctx"]["assets"]:
        for k in REGIMES:
            mu_f = ctx["full" ]["raw_dl_ctx"]["mu_hat"][(asset_label, k)]
            mu_t = ctx["train"]["raw_dl_ctx"]["mu_hat"][(asset_label, k)]
            sig_f = ctx["full" ]["raw_dl_ctx"]["sigma_hat"][(asset_label, asset_label, k)]
            sig_t = ctx["train"]["raw_dl_ctx"]["sigma_hat"][(asset_label, asset_label, k)]
            rows.append({
                "asset":      asset_label,
                "regime":     k,
                "mu_hat_full":  float(mu_f),
                "mu_hat_train": float(mu_t),
                "mu_diff_pct":  float((mu_t - mu_f) / abs(mu_f)) if mu_f != 0 else float("nan"),
                "sigma_hat_full":  float(sig_f),
                "sigma_hat_train": float(sig_t),
                "sigma_diff_pct":  float((sig_t - sig_f) / sig_f) if sig_f > 0 else float("nan"),
            })
    df = pd.DataFrame(rows)
    save_csv(df, "1_muhat_compare", SUBDIR)
    print("\n--- mu_hat full vs train ---")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


# ================================================================
# 4) Regret grid por setup
# ================================================================

def run_grids(ctx):
    lambda_grid = list(LAMBDA_GRID)
    m_grid = list(M_GRID)
    grids = {}
    for label in ("full", "train"):
        dl = ctx[label]["dl"]
        print(f"\n--- Grilla {len(lambda_grid)}x{len(m_grid)}="
              f"{len(lambda_grid)*len(m_grid)} solves en setup={label} ---")
        V_df, pol = run_regret_grid(dl, lambda_grid, m_grid)
        res = compute_regret_and_select(V_df)
        grids[label] = {"V_df": V_df, "policies": pol, "res": res}
    return grids


def diag_grid_compare(ctx, grids):
    rows = []
    for label, g in grids.items():
        res = g["res"]
        lam_m, m_m = res["g_mean"]
        lam_w, m_w = res["g_worst"]
        V_mean_row = res["V_table"].loc[(lam_m, m_m)]
        C0 = ctx[label]["dl"]["Capital_inicial"]
        rows.append({
            "setup": label,
            "g_mean_lambda":  float(lam_m), "g_mean_m": float(m_m),
            "g_mean_regret":  float(res["g_mean_metric"]),
            "g_mean_V_avg":   float(V_mean_row.mean()),
            "g_mean_V_min":   float(V_mean_row.min()),
            "g_mean_V_max":   float(V_mean_row.max()),
            "g_mean_ret_avg": float(V_mean_row.mean()/C0 - 1),
            "g_worst_lambda": float(lam_w), "g_worst_m": float(m_w),
            "g_worst_regret": float(res["g_worst_metric"]),
        })
    save_csv(pd.DataFrame(rows), "2_grid_compare", SUBDIR)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, label in zip(axes, ("full", "train")):
        res = grids[label]["res"]
        R_tab = res["R_table"]
        mean_R = R_tab.mean(axis=1).unstack("m").sort_index().sort_index(axis=1)
        ax.imshow(mean_R.values, cmap="RdYlGn_r", aspect="auto")
        ax.set_xticks(range(len(mean_R.columns)))
        ax.set_xticklabels([f"{m:.1f}" for m in mean_R.columns])
        ax.set_yticks(range(len(mean_R.index)))
        ax.set_yticklabels([f"{l:.2f}" for l in mean_R.index])
        ax.set_xlabel("m"); ax.set_ylabel("lambda")
        lam_m, m_m = res["g_mean"]
        ax.set_title(f"setup={label}  g*_mean=({lam_m:.2f},{m_m:.1f})\n"
                     f"regret = ${res['g_mean_metric']:,.0f}")
        for i in range(mean_R.shape[0]):
            for j in range(mean_R.shape[1]):
                ax.text(j, i, f"${mean_R.values[i, j]:,.0f}",
                        ha="center", va="center", fontsize=7)
        j_opt = list(mean_R.columns).index(m_m)
        i_opt = list(mean_R.index).index(lam_m)
        ax.scatter([j_opt], [i_opt], s=200, marker="*", color="cyan",
                   edgecolor="k", linewidth=1.5, zorder=3)
    save_fig(fig, "2_grid_compare", SUBDIR)


# ================================================================
# 5) Backtest historico sobre TEST
# ================================================================

def _backtest_setup(label, dl_ctx, opt_ctx, res, policies):
    lam_m, m_m = res["g_mean"]
    w_star, u_star, v_star, _z = policies[(lam_m, m_m)]
    _, w_opt, u_opt, v_opt, _ = solve_portfolio(
        opt_ctx, lambda_riesgo=1.00, costo_mult=1.0,
    )
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
        "OPT_cap_final":     float(cap_opt[t_f]),
        "RG_cap_final":      float(cap_rg [t_f]),
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
    bts = {}
    for label in ("full", "train"):
        bt = _backtest_setup(
            label, ctx[label]["dl"], ctx[label]["opt"],
            grids[label]["res"], grids[label]["policies"],
        )
        bts[label] = bt

        T_vals = bt["T_vals"]
        rows = [{
            "t": int(t),
            "OPT":    float(bt["cap_opt"][t]),
            "RG":     float(bt["cap_rg" ][t]),
            "NaiveRB":float(bt["cap_rb" ][t]),
            "NaiveBH":float(bt["cap_bh" ][t]),
        } for t in T_vals]
        save_csv(pd.DataFrame(rows), f"3_backtest_{label}", SUBDIR)

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
        ax.axhline(ctx[label]["dl"]["Capital_inicial"],
                   color="#666", ls="--", lw=0.8, label="C0")
        ax.set_title(f"Backtest TEST - setup mu_hat={label}  "
                     f"({len(T_vals)} periodos)")
        ax.set_xlabel("t")
        ax.set_ylabel("Capital")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9, loc="best")
        save_fig(fig, f"3_backtest_{label}", SUBDIR)
    return bts


# ================================================================
# 6) Resumen
# ================================================================

def diag_terminal_summary(bts):
    df = pd.DataFrame([bts[s]["summary"] for s in ("full", "train")])
    save_csv(df, "4_terminal_summary", SUBDIR)

    politicas = ["OPT", "RG", "NaiveRB", "NaiveBH"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, label in zip(axes, ("full", "train")):
        bt = bts[label]
        rets = [bt["summary"][f"{p}_ret_acum"] for p in politicas]
        colors = ["#F2B705", "#1f77b4", "#8B3A1F", "#E63946"]
        bars = ax.bar(politicas, [100*r for r in rets], color=colors, alpha=0.85)
        ax.axhline(0, color="grey", lw=0.6)
        ax.set_title(f"mu_hat = {label}  ({bt['summary']['n_periodos']} periodos)")
        ax.set_ylabel("Retorno acumulado [%]")
        for b, r in zip(bars, rets):
            ax.text(b.get_x() + b.get_width()/2, b.get_height(),
                    f"{r*100:+.1f}%", ha="center",
                    va="bottom" if r >= 0 else "top", fontsize=9)
        ax.grid(True, alpha=0.25, axis="y")
    fig.suptitle("Retorno acumulado en TEST por politica y origen de mu_hat",
                 fontsize=11)
    save_fig(fig, "4_terminal_summary", SUBDIR)


# ================================================================
# Main
# ================================================================

def main():
    print("=" * 70)
    print("MU_HAT LEAKAGE - estimacion full sample vs train-only")
    print("=" * 70)

    ctx = build_setups()
    diag_muhat_compare(ctx)

    grids = run_grids(ctx)
    diag_grid_compare(ctx, grids)

    bts = diag_backtest(ctx, grids)
    diag_terminal_summary(bts)

    print("\n" + "=" * 70)
    print("RESUMEN")
    print("=" * 70)
    for label in ("full", "train"):
        res = grids[label]["res"]
        lam_m, m_m = res["g_mean"]
        bt_s = bts[label]["summary"]
        print(f"  [mu_hat={label:<5}] g*_mean=(lam={lam_m:.2f}, m={m_m:.1f})  "
              f"regret=${res['g_mean_metric']:,.0f}")
        print(f"                       backtest: OPT={bt_s['OPT_ret_acum']:+.2%}  "
              f"RG={bt_s['RG_ret_acum']:+.2%}  "
              f"NaiveRB={bt_s['NaiveRB_ret_acum']:+.2%}  "
              f"NaiveBH={bt_s['NaiveBH_ret_acum']:+.2%}")
    print("=" * 70)


if __name__ == "__main__":
    main()
