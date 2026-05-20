"""Leakage check: el FO consume mu_mix construido con predicciones LSTM
walking que en gran parte caen sobre datos que la LSTM ya vio en training.

Pregunta: el algoritmo (PDF lineas 2-4) define un split cronologico train/
valid/test y congela el LSTM. La intencion implicita es usar las
predicciones SOLO sobre datos out-of-sample (test). En cambio, la
implementacion actual hace walking sobre todo `r_hist[:T]`, lo que mezcla
predicciones de train (memorizadas), valid (usadas para early stopping) y
test (honestas).

Setups comparados:

  full   T = 163 (con padding actual del pipeline).
         LSTM walking t = 61..163; 72w TRAIN + 15w VALID + 16w TEST;
         + 60w padded con la prediccion de t=61 (que vino de train).

  nopad  T = 103 (resultado del experimento padding_ablation).
         Sin padding pero misma proporcion train/valid/test en el walking.

  test   T = 16 (este experimento: solo TEST cronologico).
         LSTM walking t = 148..163; 100% predicciones honestas OOS.

Si "test" sale sustancialmente peor que "full"/"nopad", el pipeline estaba
viviendo del leakage. Si sale similar, el LSTM tiene poder predictivo real
y el problema esta en otro lado.

Calendarios para el backtest historico:
  full   r_hist[t = 1..163]
  nopad  r_hist[t = 61..163]   (reindexado a t = 1..103)
  test   r_hist[t = 148..163]  (reindexado a t = 1..16)

Cada setup compara g*_mean (RG) contra OPT base + Naive (rb) + Naive (B&H),
todos sobre el calendario de ese setup.

Outputs (inspeccion/leakage_check_out/):
  1_leakage_split.csv/png      composicion train/valid/test del input al FO
  2_grid_compare.csv/png       g*_mean, mean_regret por setup
  3_backtest_<setup>.csv/png   curvas de capital por setup
  4_terminal_summary.csv/png   retorno acumulado por (politica, setup)

Corre con:
    python -m inspeccion.leakage_check
    (o)
    python inspeccion/leakage_check.py
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
    DLConfig,
    LAMBDA_GRID,
    M_GRID,
    N_CANDIDATES,
    N_SCENARIOS,
    SCENARIO_SEED,
    SUMMARY_ASSET,
    T_HORIZON,
)
from dl.prediccion_deciles import load_checkpoint, load_returns, build_windows
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


SUBDIR = "leakage_check"


# ================================================================
# 1) Componer los 3 setups segun cuanta prediccion LSTM es OOS
# ================================================================

def compute_split_boundaries(H: int, T_dataset: int, ratios):
    """Replica la logica de chrono_split: devuelve la cantidad de ventanas
    de train/valid/test y los rangos temporales correspondientes.

    Una ventana indexada n predice la semana t = H+1+n (1-indexed sobre el
    dataset). N = T_dataset - H.
    """
    N = T_dataset - H
    r_tr, r_va, _ = ratios
    n_tr = int(N * r_tr)
    n_va = int(N * r_va)
    n_te = N - n_tr - n_va
    # Semanas predichas por cada split (1-indexed sobre el dataset original):
    t_train_first = H + 1
    t_train_last  = H + n_tr
    t_valid_first = H + n_tr + 1
    t_valid_last  = H + n_tr + n_va
    t_test_first  = H + n_tr + n_va + 1
    t_test_last   = H + N
    return {
        "N": N, "n_tr": n_tr, "n_va": n_va, "n_te": n_te,
        "t_train": (t_train_first, t_train_last),
        "t_valid": (t_valid_first, t_valid_last),
        "t_test":  (t_test_first,  t_test_last),
    }


def build_setups():
    print("Cargando checkpoint para leer H y split del LSTM...")
    ckpt = load_checkpoint(CHECKPOINT_PATH)
    H = ckpt.config.H
    ratios = ckpt.config.split
    T_dataset = len(load_returns())
    bounds = compute_split_boundaries(H, T_dataset, ratios)
    print(f"  H = {H}  T_dataset = {T_dataset}  split = {ratios}")
    print(f"  Ventanas: train={bounds['n_tr']}  valid={bounds['n_va']}  test={bounds['n_te']}")
    print(f"  TRAIN predice t = {bounds['t_train'][0]}..{bounds['t_train'][1]}")
    print(f"  VALID predice t = {bounds['t_valid'][0]}..{bounds['t_valid'][1]}")
    print(f"  TEST  predice t = {bounds['t_test' ][0]}..{bounds['t_test' ][1]}")

    print("\nConstruyendo DL ctx full (T=163, padded)...")
    # Pinned al setup legacy (walking + p_dl) porque este experimento mide
    # cuanto del horizonte del FO esta cubierto por predicciones in-sample
    # del LSTM, lo cual solo aplica al walking (rollout no usa retornos
    # reales como input despues de t=1).
    dl_full = build_dl_context(
        data_dir=DATA_DIR, checkpoint_path=CHECKPOINT_PATH,
        T=T_HORIZON, N_candidates=N_CANDIDATES, n_scenarios=N_SCENARIOS,
        seed=SCENARIO_SEED, summary_asset=SUMMARY_ASSET,
        p_method="walking", mu_hat_source="p_dl",
    )
    opt_full = trim_post_warmup(load_market_data(str(DATA_DIR)),
                                H=0, T_max=T_HORIZON)
    print(f"  dl_full:  T_vals = 1..{dl_full['nT']}")

    print("Construyendo DL ctx nopad (T=103, t=H+1..T del full)...")
    dl_nopad  = trim_post_warmup(dl_full, H=H, T_max=T_HORIZON)
    opt_nopad = trim_post_warmup(load_market_data(str(DATA_DIR)),
                                 H=H, T_max=T_HORIZON)
    print(f"  dl_nopad: T_vals = 1..{dl_nopad['nT']}")

    # "test": queremos los ULTIMOS bounds['n_te'] periodos del horizonte. Eso
    # corresponde a recortar las primeras (T_HORIZON - n_te) semanas. Como
    # trim_post_warmup trimea las primeras H semanas, usamos H_eff = T - n_te.
    n_te = bounds["n_te"]
    H_eff = T_HORIZON - n_te
    print(f"Construyendo DL ctx test (T={n_te}, solo TEST)...")
    dl_test  = trim_post_warmup(dl_full, H=H_eff, T_max=T_HORIZON)
    opt_test = trim_post_warmup(load_market_data(str(DATA_DIR)),
                                H=H_eff, T_max=T_HORIZON)
    print(f"  dl_test:  T_vals = 1..{dl_test['nT']}")

    return {
        "H": H, "T_HORIZON": T_HORIZON, "T_dataset": T_dataset,
        "bounds": bounds,
        "full":  {"dl": dl_full,  "opt": opt_full},
        "nopad": {"dl": dl_nopad, "opt": opt_nopad},
        "test":  {"dl": dl_test,  "opt": opt_test},
    }


# ================================================================
# 2) Diagnostico 1: composicion train/valid/test del input al FO
# ================================================================

def diag_leakage_split(ctx):
    bounds = ctx["bounds"]
    H = ctx["H"]
    rows = []
    for label, T_eff, t_first_orig in [
        ("full",  ctx["full"]["dl"]["nT"],  1),
        ("nopad", ctx["nopad"]["dl"]["nT"], H + 1),
        ("test",  ctx["test"]["dl"]["nT"],  ctx["T_HORIZON"] - ctx["test"]["dl"]["nT"] + 1),
    ]:
        t_last_orig = t_first_orig + T_eff - 1
        # Cuantas semanas del setup caen en cada split del LSTM:
        n_padded = max(0, min(H, t_last_orig) - t_first_orig + 1) if label == "full" else 0
        def _overlap(a1, a2, b1, b2):
            return max(0, min(a2, b2) - max(a1, b1) + 1)
        n_train_in = _overlap(t_first_orig, t_last_orig, *bounds["t_train"])
        n_valid_in = _overlap(t_first_orig, t_last_orig, *bounds["t_valid"])
        n_test_in  = _overlap(t_first_orig, t_last_orig, *bounds["t_test"])
        rows.append({
            "setup":        label,
            "T_eff":        T_eff,
            "t_first_orig": t_first_orig,
            "t_last_orig":  t_last_orig,
            "n_padded":     n_padded,
            "n_train_in_FO": n_train_in,
            "n_valid_in_FO": n_valid_in,
            "n_test_in_FO":  n_test_in,
            "pct_oos":      n_test_in / T_eff,
        })
    df = pd.DataFrame(rows)
    save_csv(df, "1_leakage_split", SUBDIR)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    labels = df["setup"].tolist()
    pads   = df["n_padded"].values
    trains = df["n_train_in_FO"].values
    valids = df["n_valid_in_FO"].values
    tests  = df["n_test_in_FO"].values

    y_pos = np.arange(len(labels))
    ax.barh(y_pos, pads,   color="#999",   label="padded (constante)")
    ax.barh(y_pos, trains, left=pads, color="#E63946",
            label="TRAIN (LSTM memorizo)")
    ax.barh(y_pos, valids, left=pads + trains, color="#F2B705",
            label="VALID (early stopping)")
    ax.barh(y_pos, tests,  left=pads + trains + valids, color="#1f77b4",
            label="TEST (OOS honesto)")
    for yi, (T_eff, pct) in enumerate(zip(df["T_eff"], df["pct_oos"])):
        ax.text(T_eff + 1, yi, f"T={T_eff}  OOS={pct:.0%}",
                va="center", fontsize=9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("semanas del horizonte del FO")
    ax.set_title("Composicion del input al FO por origen de la prediccion LSTM")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.25, axis="x")
    save_fig(fig, "1_leakage_split", SUBDIR)


# ================================================================
# 3) Regret grid por setup
# ================================================================

def run_grids(ctx):
    lambda_grid = list(LAMBDA_GRID)
    m_grid = list(M_GRID)
    grids = {}
    for label in ("full", "nopad", "test"):
        dl = ctx[label]["dl"]
        print(f"\n--- Corriendo grilla {len(lambda_grid)}x{len(m_grid)}="
              f"{len(lambda_grid)*len(m_grid)} solves en setup={label} "
              f"(T={dl['nT']}) ---")
        V_df, pol = run_regret_grid(dl, lambda_grid, m_grid)
        res = compute_regret_and_select(V_df)
        grids[label] = {"V_df": V_df, "policies": pol, "res": res}
    return {"lambda_grid": lambda_grid, "m_grid": m_grid, "grids": grids}


def diag_grid_compare(ctx, grids):
    rows = []
    for label, g in grids["grids"].items():
        res = g["res"]
        lam_m, m_m = res["g_mean"]
        lam_w, m_w = res["g_worst"]
        V_mean_row  = res["V_table"].loc[(lam_m, m_m)]
        V_worst_row = res["V_table"].loc[(lam_w, m_w)]
        C0 = ctx[label]["dl"]["Capital_inicial"]
        rows.append({
            "setup": label,
            "T_eff": ctx[label]["dl"]["nT"],
            "g_mean_lambda":  float(lam_m), "g_mean_m":  float(m_m),
            "g_mean_regret":  float(res["g_mean_metric"]),
            "g_mean_V_avg":   float(V_mean_row.mean()),
            "g_mean_V_min":   float(V_mean_row.min()),
            "g_mean_V_max":   float(V_mean_row.max()),
            "g_mean_ret_avg": float(V_mean_row.mean()/C0 - 1),
            "g_worst_lambda": float(lam_w), "g_worst_m": float(m_w),
            "g_worst_regret": float(res["g_worst_metric"]),
        })
    save_csv(pd.DataFrame(rows), "2_grid_compare", SUBDIR)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, label in zip(axes, ("full", "nopad", "test")):
        res = grids["grids"][label]["res"]
        R_tab = res["R_table"]
        mean_R = R_tab.mean(axis=1).unstack("m").sort_index().sort_index(axis=1)
        ax.imshow(mean_R.values, cmap="RdYlGn_r", aspect="auto")
        ax.set_xticks(range(len(mean_R.columns)))
        ax.set_xticklabels([f"{m:.1f}" for m in mean_R.columns])
        ax.set_yticks(range(len(mean_R.index)))
        ax.set_yticklabels([f"{l:.2f}" for l in mean_R.index])
        ax.set_xlabel("m"); ax.set_ylabel("lambda")
        lam_m, m_m = res["g_mean"]
        ax.set_title(f"setup={label} (T={ctx[label]['dl']['nT']})\n"
                     f"g*_mean=({lam_m:.2f},{m_m:.1f}) "
                     f"regret=${res['g_mean_metric']:,.0f}")
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
# 4) Backtest historico por setup
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
    for label in ("full", "nopad", "test"):
        bt = _backtest_setup(
            label, ctx[label]["dl"], ctx[label]["opt"],
            grids["grids"][label]["res"], grids["grids"][label]["policies"],
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
        ax.set_title(f"Backtest historico - setup {label} ({len(T_vals)} periodos)")
        ax.set_xlabel("t")
        ax.set_ylabel("Capital")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9, loc="best")
        save_fig(fig, f"3_backtest_{label}", SUBDIR)
    return bts


# ================================================================
# 5) Resumen terminal
# ================================================================

def diag_terminal_summary(bts):
    df = pd.DataFrame([bts[s]["summary"] for s in ("full", "nopad", "test")])
    save_csv(df, "4_terminal_summary", SUBDIR)

    politicas = ["OPT", "RG", "NaiveRB", "NaiveBH"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, label in zip(axes, ("full", "nopad", "test")):
        bt = bts[label]
        rets = [bt["summary"][f"{p}_ret_acum"] for p in politicas]
        colors = ["#F2B705", "#1f77b4", "#8B3A1F", "#E63946"]
        bars = ax.bar(politicas, [100*r for r in rets], color=colors, alpha=0.85)
        ax.axhline(0, color="grey", lw=0.6)
        ax.set_title(f"setup={label}  ({bt['summary']['n_periodos']} periodos)")
        ax.set_ylabel("Retorno acumulado [%]")
        for b, r in zip(bars, rets):
            ax.text(b.get_x() + b.get_width()/2, b.get_height(),
                    f"{r*100:+.1f}%", ha="center",
                    va="bottom" if r >= 0 else "top", fontsize=9)
        ax.grid(True, alpha=0.25, axis="y")
    fig.suptitle("Retorno acumulado por politica y setup (cuanto leakage = "
                 "izquierda; cero leakage = derecha)", fontsize=11)
    save_fig(fig, "4_terminal_summary", SUBDIR)


# ================================================================
# Main
# ================================================================

def main():
    print("=" * 70)
    print("LEAKAGE CHECK - cuanto del FO se alimenta de predicciones LSTM in-sample")
    print("=" * 70)

    ctx = build_setups()
    diag_leakage_split(ctx)

    grids = run_grids(ctx)
    diag_grid_compare(ctx, grids)

    bts = diag_backtest(ctx, grids)
    diag_terminal_summary(bts)

    print("\n" + "=" * 70)
    print("RESUMEN")
    print("=" * 70)
    for label in ("full", "nopad", "test"):
        res = grids["grids"][label]["res"]
        lam_m, m_m = res["g_mean"]
        bt_s = bts[label]["summary"]
        print(f"  [{label:<5}] T={bt_s['n_periodos']:>3}  "
              f"g*_mean=(lam={lam_m:.2f}, m={m_m:.1f})  "
              f"regret=${res['g_mean_metric']:,.0f}")
        print(f"          backtest: OPT={bt_s['OPT_ret_acum']:+.2%}  "
              f"RG={bt_s['RG_ret_acum']:+.2%}  "
              f"NaiveRB={bt_s['NaiveRB_ret_acum']:+.2%}  "
              f"NaiveBH={bt_s['NaiveBH_ret_acum']:+.2%}")
    print("=" * 70)


if __name__ == "__main__":
    main()
