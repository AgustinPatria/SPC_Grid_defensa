"""L3 — Inspeccion de los escenarios DL compartidos.

Genera, para IS (T=163) y OOS (T_oos=16):
  1. Las 5 trayectorias representativas: cum-return acumulado por activo y por
     portafolio w_ref (50/50).
  2. Fan chart de los N=1000 candidatos con los 5 reps superpuestos.
  3. Histograma del log-retorno acumulado de portafolio (criterio de ranking)
     con marcadores de los 5 percentiles representativos.
  4. Heatmap (escenario x activo) del cum-return.
  5. Correlacion SPX-CMC200 dentro de cada escenario representativo +
     correlacion historica como referencia.
  6. No-leakage check OOS: muestra grafico de initial_window OOS vs ventana
     IS para confirmar que la OOS son las H semanas inmediatamente previas
     al t_test_start.

Output: inspeccion_v2/L3_escenarios_out/
"""
import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from inspeccion_v2._common import (
    DATA_DIR,
    N_CANDIDATES,
    N_SCENARIOS,
    SCENARIO_SEED,
    T_HORIZON,
    build_ensemble_model,
    build_scenarios_is,
    build_scenarios_oos,
    hist_returns,
    load_initial_window_is,
    load_initial_window_oos,
    load_market_data,
    load_nns,
    save_csv,
    save_fig,
    w_ref_vector,
)
from dl.generador_escenarios import generate_candidate_scenarios


SUBDIR = "L3_escenarios"


# ---------------------------------------------------------------- helpers
def cum_return(returns_2d: np.ndarray) -> np.ndarray:
    """returns_2d (T, A) -> (T, A) cum return = prod(1+r) - 1."""
    return np.cumprod(1.0 + returns_2d, axis=0) - 1.0


def portfolio_returns(scenarios: np.ndarray, w_ref: np.ndarray) -> np.ndarray:
    """scenarios (N, T, A), w_ref (A,) -> (N, T) retorno semanal de portafolio."""
    return np.einsum("nti,i->nt", scenarios, w_ref)


def portfolio_cum_logret(scenarios: np.ndarray, w_ref: np.ndarray) -> np.ndarray:
    """log-ret cumulado por escenario — mismo criterio que reduce_by_portfolio_return."""
    r_port = portfolio_returns(scenarios, w_ref)
    return np.log(np.clip(1.0 + r_port, 1e-8, None)).sum(axis=1)        # (N,)


# ---------------------------------------------------------------- 1) reps
def plot_representative_paths(reps: np.ndarray, assets, w_ref, label: str):
    """Cum-return acumulado de los 5 escenarios — uno por activo + portafolio."""
    n_S, T, A = reps.shape
    n_panels = A + 1
    fig, axes = plt.subplots(n_panels, 1, figsize=(11, 2.5 * n_panels),
                             sharex=True)
    t_axis = np.arange(1, T + 1)
    cmap = plt.cm.viridis(np.linspace(0.0, 1.0, n_S))

    for ai, asset in enumerate(assets):
        ax = axes[ai]
        for s in range(n_S):
            cum = np.cumprod(1.0 + reps[s, :, ai]) - 1.0
            ax.plot(t_axis, cum, color=cmap[s], linewidth=1.5,
                    label=f"s={s}  ({['worst', 'low', 'mid', 'high', 'best'][s]})"
                    if n_S == 5 else f"s={s}")
        ax.axhline(0, color="black", linestyle=":", alpha=0.4)
        ax.set_ylabel(f"cum_ret — {asset}")
        ax.grid(alpha=0.3)
        if ai == 0:
            ax.legend(loc="upper left", fontsize=8, ncol=5)

    # portafolio
    ax = axes[-1]
    for s in range(n_S):
        r_p = reps[s] @ w_ref
        cum = np.cumprod(1.0 + r_p) - 1.0
        ax.plot(t_axis, cum, color=cmap[s], linewidth=1.5)
    ax.axhline(0, color="black", linestyle=":", alpha=0.4)
    ax.set_ylabel(f"cum_ret — portafolio w_ref")
    ax.set_xlabel("t")
    ax.grid(alpha=0.3)
    fig.suptitle(f"L3 — 5 escenarios representativos ({label})", fontsize=12)
    save_fig(fig, f"01_reps_paths_{label}", SUBDIR)


# ---------------------------------------------------------------- 2) fan
def plot_fan_with_reps(candidates: np.ndarray, reps: np.ndarray, w_ref,
                       label: str):
    """Fan chart del cum-return de portafolio de los N=1000 + reps superpuestos."""
    N, T, A = candidates.shape
    r_port_cand = portfolio_returns(candidates, w_ref)
    cum_cand = np.cumprod(1.0 + r_port_cand, axis=1) - 1.0           # (N, T)
    r_port_reps = portfolio_returns(reps, w_ref)
    cum_reps = np.cumprod(1.0 + r_port_reps, axis=1) - 1.0           # (n_S, T)
    t_axis = np.arange(1, T + 1)

    fig, ax = plt.subplots(figsize=(11, 5))
    qs = np.percentile(cum_cand, [5, 25, 50, 75, 95], axis=0)
    ax.fill_between(t_axis, qs[0], qs[4], color="C0", alpha=0.10,
                    label="5–95 pct candidatos")
    ax.fill_between(t_axis, qs[1], qs[3], color="C0", alpha=0.25,
                    label="25–75 pct candidatos")
    ax.plot(t_axis, qs[2], color="C0", linewidth=1.5, label="mediana candidatos")

    n_S = reps.shape[0]
    cmap = plt.cm.viridis(np.linspace(0.0, 1.0, n_S))
    for s in range(n_S):
        ax.plot(t_axis, cum_reps[s], color=cmap[s], linewidth=1.8,
                label=f"rep s={s}")
    ax.axhline(0, color="black", linestyle=":", alpha=0.4)
    ax.set_ylabel("cum_ret portafolio w_ref")
    ax.set_xlabel("t")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    fig.suptitle(f"L3 — fan chart {N} candidatos + 5 reps ({label})",
                 fontsize=12)
    save_fig(fig, f"02_fan_{label}", SUBDIR)


# ---------------------------------------------------------------- 3) hist
def plot_logret_histogram(candidates: np.ndarray, reps: np.ndarray, w_ref,
                          label: str):
    """Histograma del log-ret acumulado de portafolio (criterio del ranking)."""
    cum_cand_log = portfolio_cum_logret(candidates, w_ref)
    cum_reps_log = portfolio_cum_logret(reps, w_ref)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(cum_cand_log, bins=60, color="C0", alpha=0.45,
            label=f"N={len(cum_cand_log)} candidatos")
    n_S = len(cum_reps_log)
    cmap = plt.cm.viridis(np.linspace(0.0, 1.0, n_S))
    for s, val in enumerate(cum_reps_log):
        ax.axvline(val, color=cmap[s], linewidth=2,
                   label=f"rep s={s}: log_cum={val:+.3f}  "
                         f"(cum={np.exp(val) - 1:+.2%})")
    ax.set_xlabel("log-ret acumulado portafolio (criterio de ranking)")
    ax.set_ylabel("frecuencia")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.suptitle(f"L3 — distribucion de candidatos en el espacio de ranking ({label})",
                 fontsize=12)
    save_fig(fig, f"03_hist_logret_{label}", SUBDIR)


# ---------------------------------------------------------------- 4) heatmap
def plot_heatmap_cum(reps: np.ndarray, assets, label: str):
    """Heatmap (escenario x activo) del cum-return final."""
    n_S, T, A = reps.shape
    cum_final = np.prod(1.0 + reps, axis=1) - 1.0       # (n_S, A)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cum_final, cmap="RdYlGn", aspect="auto",
                   vmin=-np.abs(cum_final).max(),
                   vmax=+np.abs(cum_final).max())
    ax.set_xticks(range(A))
    ax.set_xticklabels(assets)
    ax.set_yticks(range(n_S))
    ax.set_yticklabels([f"s={s}" for s in range(n_S)])
    for s in range(n_S):
        for a in range(A):
            ax.text(a, s, f"{cum_final[s, a]:+.1%}", ha="center", va="center",
                    fontsize=10, color="black")
    fig.colorbar(im, ax=ax, label="cum_ret final")
    ax.set_title(f"L3 — cum_ret final por escenario x activo ({label})")
    save_fig(fig, f"04_heatmap_{label}", SUBDIR)


# ---------------------------------------------------------------- 5) correlation
def correlation_table(reps: np.ndarray, assets, r_real_dict, label: str):
    """corr(SPX, CMC200) por escenario + historico como referencia."""
    n_S, T, A = reps.shape
    rows = []
    for s in range(n_S):
        df = pd.DataFrame(reps[s], columns=assets)
        rows.append({"escenario": f"s={s} ({label})",
                     "corr_SPX_CMC200": float(df.corr().iloc[0, 1])})
    # historico (toda la serie)
    df_hist = pd.DataFrame({a: r_real_dict[a] for a in assets})
    rows.append({"escenario": "historico (toda la serie)",
                 "corr_SPX_CMC200": float(df_hist.corr().iloc[0, 1])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 6) no leakage
def plot_initial_windows(window_is: np.ndarray, window_oos: np.ndarray,
                         t_test_start: int, assets):
    """Compara las dos ventanas iniciales — confirma que OOS no toca el test."""
    H, A = window_is.shape
    t_is_axis = np.arange(T_HORIZON - H + 1, T_HORIZON + 1)
    t_oos_axis = np.arange(t_test_start - H, t_test_start)

    fig, axes = plt.subplots(A, 1, figsize=(11, 2.5 * A), sharex=True)
    if A == 1:
        axes = [axes]
    for ai, a in enumerate(assets):
        ax = axes[ai]
        ax.plot(t_is_axis, window_is[:, ai], color="C0",
                label=f"IS initial_window (t={t_is_axis[0]}..{t_is_axis[-1]})",
                marker="o", markersize=2)
        ax.plot(t_oos_axis, window_oos[:, ai], color="C3",
                label=f"OOS initial_window (t={t_oos_axis[0]}..{t_oos_axis[-1]})",
                marker="o", markersize=2)
        ax.axvline(t_test_start, color="red", linestyle="--", alpha=0.7,
                   label=f"t_test_start={t_test_start}")
        ax.axhline(0, color="black", linestyle=":", alpha=0.3)
        ax.set_ylabel(f"r_real — {a}")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("t")
    fig.suptitle("L3 — initial_window IS vs OOS (chequeo no-leakage)",
                 fontsize=12)
    save_fig(fig, "05_initial_windows", SUBDIR)


# ---------------------------------------------------------------- main
def main(force_retrain: bool = False):
    print("=" * 70)
    print("L3 — Inspeccion escenarios DL compartidos")
    print("=" * 70)

    nns = load_nns(force_retrain=force_retrain)
    ensemble = build_ensemble_model(nns)
    assets = list(ensemble.config.assets)
    w_ref = w_ref_vector(assets)
    print(f"  activos: {assets}    w_ref: {dict(zip(assets, w_ref.tolist()))}")

    # ---------------- IS ----------------
    print("\n[1/3] Generando candidatos + reps IS (N=1000, T=163) ...")
    init_is = load_initial_window_is(ensemble)
    cands_is = generate_candidate_scenarios(
        ensemble, init_is, N=N_CANDIDATES, T=T_HORIZON, seed=SCENARIO_SEED,
    )
    reps_is = build_scenarios_is(ensemble)
    print(f"  candidatos: {cands_is.shape}   reps: {reps_is.shape}")

    plot_representative_paths(reps_is, assets, w_ref, label="IS")
    plot_fan_with_reps(cands_is, reps_is, w_ref, label="IS")
    plot_logret_histogram(cands_is, reps_is, w_ref, label="IS")
    plot_heatmap_cum(reps_is, assets, label="IS")

    r_real = hist_returns(assets, T=T_HORIZON)
    corr_is = correlation_table(reps_is, assets, r_real, label="IS")
    save_csv(corr_is, "06_correlation_IS", SUBDIR)
    print("\n  Correlaciones IS:")
    print(corr_is.to_string(index=False, float_format="{:.3f}".format))

    # cum_ret final por escenario / activo (CSV)
    cum_is = pd.DataFrame(
        np.prod(1.0 + reps_is, axis=1) - 1.0,
        columns=assets,
    )
    cum_is.insert(0, "s", range(reps_is.shape[0]))
    cum_is["portafolio_w_ref"] = (np.prod(1.0 + reps_is @ w_ref, axis=1) - 1.0)
    save_csv(cum_is, "01_cum_final_IS", SUBDIR)
    print("\n  cum_ret final IS:")
    print(cum_is.to_string(index=False, float_format="{:+.2%}".format))

    # ---------------- OOS ----------------
    print("\n[2/3] Generando candidatos + reps OOS (T_oos = T - t_test_start + 1) ...")
    init_oos, t_test_start = load_initial_window_oos(ensemble)
    T_oos = T_HORIZON - t_test_start + 1
    print(f"  t_test_start={t_test_start}   T_oos={T_oos}")
    cands_oos = generate_candidate_scenarios(
        ensemble, init_oos, N=N_CANDIDATES, T=T_oos, seed=SCENARIO_SEED,
    )
    reps_oos, _ = build_scenarios_oos(ensemble)
    print(f"  candidatos: {cands_oos.shape}   reps: {reps_oos.shape}")

    plot_representative_paths(reps_oos, assets, w_ref, label="OOS")
    plot_fan_with_reps(cands_oos, reps_oos, w_ref, label="OOS")
    plot_logret_histogram(cands_oos, reps_oos, w_ref, label="OOS")
    plot_heatmap_cum(reps_oos, assets, label="OOS")

    corr_oos = correlation_table(reps_oos, assets, r_real, label="OOS")
    save_csv(corr_oos, "06_correlation_OOS", SUBDIR)
    print("\n  Correlaciones OOS:")
    print(corr_oos.to_string(index=False, float_format="{:.3f}".format))

    cum_oos = pd.DataFrame(
        np.prod(1.0 + reps_oos, axis=1) - 1.0,
        columns=assets,
    )
    cum_oos.insert(0, "s", range(reps_oos.shape[0]))
    cum_oos["portafolio_w_ref"] = (np.prod(1.0 + reps_oos @ w_ref, axis=1) - 1.0)
    save_csv(cum_oos, "01_cum_final_OOS", SUBDIR)
    print("\n  cum_ret final OOS:")
    print(cum_oos.to_string(index=False, float_format="{:+.2%}".format))

    # ---------------- no-leakage check ----------------
    print("\n[3/3] Plot initial_window IS vs OOS (no-leakage check) ...")
    plot_initial_windows(init_is, init_oos, t_test_start, assets)

    # comparativo IS vs OOS
    summary = pd.DataFrame({
        "metric": [
            "horizonte T",
            "candidates_min_cum_port",
            "candidates_median_cum_port",
            "candidates_max_cum_port",
            "reps_cum_port",
        ],
        "IS": [
            T_HORIZON,
            f"{np.exp(portfolio_cum_logret(cands_is, w_ref).min()) - 1:+.2%}",
            f"{np.exp(np.median(portfolio_cum_logret(cands_is, w_ref))) - 1:+.2%}",
            f"{np.exp(portfolio_cum_logret(cands_is, w_ref).max()) - 1:+.2%}",
            ", ".join(f"{v:+.2%}" for v in cum_is['portafolio_w_ref']),
        ],
        "OOS": [
            T_oos,
            f"{np.exp(portfolio_cum_logret(cands_oos, w_ref).min()) - 1:+.2%}",
            f"{np.exp(np.median(portfolio_cum_logret(cands_oos, w_ref))) - 1:+.2%}",
            f"{np.exp(portfolio_cum_logret(cands_oos, w_ref).max()) - 1:+.2%}",
            ", ".join(f"{v:+.2%}" for v in cum_oos['portafolio_w_ref']),
        ],
    })
    save_csv(summary, "00_summary_IS_vs_OOS", SUBDIR)
    print("\n  Resumen IS vs OOS:")
    print(summary.to_string(index=False))

    print("\n  L3 listo.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--retrain", action="store_true",
                   help="Borra cache pickle + reentrena las 15 NNs.")
    args = p.parse_args()
    main(force_retrain=args.retrain)
