"""Diagnostico de los 5 escenarios DL que alimentan el regret-grid.

Corre con:
    python -m inspeccion.escenarios
    (o)
    python inspeccion/escenarios.py

Produce 1 PNG + 1 CSV por diagnostico en `inspeccion/escenarios_out/`:

  1. sesgo            cumret_T de reps vs candidatos vs historico realizado
                      Hipotesis: los 5 reps son sistematicamente pesimistas y
                      empujan al regret-grid a la esquina conservadora.
  2. dispersion       fan chart de los N=1000 candidatos con los 5 reps encima
                      Hipotesis: el quintil bajo es tan extremo que minimax es trivial.
  3. correlacion      corr(SPX, CMC200) por escenario
                      Hipotesis: pese al q independiente, los activos quedan
                      comonotonos y el optimizador no diversifica (m inerte).
  4. path_dependence  histograma de retornos semanales por bloque temporal
                      Hipotesis: el rollout colapsa a media constante con t.
  5. sesgo_resumen    estratificacion por SPX cumret en el plano (SPX, CMC200)
                      Hipotesis: los "quintiles" representan a SPX pero no a CMC200.
  6. sanity           reproducibilidad (mismo seed) + dispersion vs otro seed.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from inspeccion._common import save_fig, save_csv, out_dir

from config import (
    CHECKPOINT_PATH,
    DATA_DIR,
    N_CANDIDATES,
    N_SCENARIOS,
    SCENARIO_POSITION,
    SCENARIO_SEED,
    SUMMARY_ASSET,
    T_HORIZON,
)
from dl.generador_escenarios import (
    generate_candidate_scenarios,
    reduce_to_representatives,
)
from dl.prediccion_deciles import load_checkpoint
from Regret_Grid import load_market_data


SUBDIR = "escenarios"


def build_context():
    """Genera N candidatos + 5 reps + historico realizado en el mismo horizonte."""
    base_ctx = load_market_data(str(DATA_DIR))
    assets = list(base_ctx["assets"])
    model = load_checkpoint(CHECKPOINT_PATH)
    H = model.config.H
    r_hist = base_ctx["r"]
    returns_history = np.stack(
        [r_hist[i].sort_index().values[:T_HORIZON] for i in assets], axis=1,
    ).astype(np.float32)
    initial_window = returns_history[-H:, :]

    candidates = generate_candidate_scenarios(
        model, initial_window, N=N_CANDIDATES, T=T_HORIZON, seed=SCENARIO_SEED,
    )                                                       # (N, T, A)
    summary_idx = assets.index(SUMMARY_ASSET)
    reps = reduce_to_representatives(
        candidates, summary_asset_idx=summary_idx,
        n_quintiles=N_SCENARIOS, position=SCENARIO_POSITION,
    )                                                       # (n_q, T, A)

    return {
        "assets":          assets,
        "candidates":      candidates,
        "reps":            reps,
        "r_hist":          returns_history,                 # (T, A)
        "summary_idx":     summary_idx,
        "initial_window":  initial_window,
        "model":           model,
    }


# ================================================================
# 1) Sesgo: distribucion del cumret terminal
# ================================================================
def diag_sesgo(ctx):
    assets = ctx["assets"]
    cand = ctx["candidates"]
    reps = ctx["reps"]
    rh = ctx["r_hist"]

    cum_cand = np.prod(1.0 + cand, axis=1) - 1.0     # (N, A)
    cum_reps = np.prod(1.0 + reps, axis=1) - 1.0     # (n_q, A)
    cum_hist = np.prod(1.0 + rh, axis=0) - 1.0       # (A,)

    fig, axes = plt.subplots(1, len(assets), figsize=(13, 4.5))
    if len(assets) == 1:
        axes = [axes]
    for ai, a in enumerate(assets):
        ax = axes[ai]
        ax.hist(cum_cand[:, ai], bins=40, alpha=0.5, color="C0",
                label=f"N={cand.shape[0]} candidatos")
        for s in range(reps.shape[0]):
            ax.axvline(cum_reps[s, ai], color="C1", lw=1.4,
                       label="5 reps" if s == 0 else None)
        ax.axvline(cum_hist[ai], color="k", lw=2.2,
                   label=f"historico ({cum_hist[ai]:+.2%})")
        ax.set_title(f"{a} - cumret terminal (T={T_HORIZON})")
        ax.set_xlabel("cumret")
        ax.legend(fontsize=8, loc="best")
    save_fig(fig, "1_sesgo_cumret_terminal", SUBDIR)

    rows = []
    for ai, a in enumerate(assets):
        rows.append({"asset": a, "grupo": "candidatos",
                     "n": cand.shape[0],
                     "mean": cum_cand[:, ai].mean(),
                     "median": float(np.median(cum_cand[:, ai])),
                     "std": cum_cand[:, ai].std(),
                     "min": cum_cand[:, ai].min(),
                     "max": cum_cand[:, ai].max()})
        rows.append({"asset": a, "grupo": "reps_5",
                     "n": reps.shape[0],
                     "mean": cum_reps[:, ai].mean(),
                     "median": float(np.median(cum_reps[:, ai])),
                     "std": cum_reps[:, ai].std(),
                     "min": cum_reps[:, ai].min(),
                     "max": cum_reps[:, ai].max()})
        rows.append({"asset": a, "grupo": "historico",
                     "n": 1,
                     "mean": float(cum_hist[ai]),
                     "median": float(cum_hist[ai]),
                     "std": 0.0,
                     "min": float(cum_hist[ai]),
                     "max": float(cum_hist[ai])})
    save_csv(pd.DataFrame(rows), "1_sesgo_cumret_terminal", SUBDIR)


# ================================================================
# 2) Dispersion: fan chart cumret(t)
# ================================================================
def diag_dispersion(ctx):
    assets = ctx["assets"]
    cand = ctx["candidates"]
    reps = ctx["reps"]
    rh = ctx["r_hist"]
    N, T, A = cand.shape

    cum_cand = np.cumprod(1.0 + cand, axis=1) - 1.0   # (N, T, A)
    cum_reps = np.cumprod(1.0 + reps, axis=1) - 1.0   # (n_q, T, A)
    cum_hist = np.cumprod(1.0 + rh, axis=0) - 1.0     # (T, A)

    quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
    qbands = np.quantile(cum_cand, quantiles, axis=0)  # (5, T, A)

    tt = np.arange(1, T + 1)
    fig, axes = plt.subplots(1, A, figsize=(13, 4.8))
    if A == 1:
        axes = [axes]
    for ai, a in enumerate(assets):
        ax = axes[ai]
        ax.fill_between(tt, qbands[0, :, ai], qbands[4, :, ai],
                        alpha=0.15, color="C0", label="P5-P95 cand")
        ax.fill_between(tt, qbands[1, :, ai], qbands[3, :, ai],
                        alpha=0.30, color="C0", label="P25-P75 cand")
        ax.plot(tt, qbands[2, :, ai], color="C0", lw=1.4, label="P50 cand")
        for s in range(reps.shape[0]):
            ax.plot(tt, cum_reps[s, :, ai], color="C1", lw=1.0, alpha=0.85,
                    label="5 reps" if s == 0 else None)
        ax.plot(tt, cum_hist[:, ai], color="k", lw=2.0, label="historico")
        ax.set_title(f"{a} - cumret(t)")
        ax.set_xlabel("semana")
        ax.set_ylabel("cumret")
        ax.legend(fontsize=8, loc="upper left")
    save_fig(fig, "2_dispersion_fan", SUBDIR)

    rows = []
    for ai, a in enumerate(assets):
        for qi, q in enumerate(quantiles):
            rows.append({
                "asset": a,
                "quantile": q,
                "cumret_T": float(qbands[qi, -1, ai]),
                "cumret_T_med_semana": float(qbands[qi, T // 2, ai]),
            })
    save_csv(pd.DataFrame(rows), "2_dispersion_fan", SUBDIR)


# ================================================================
# 3) Correlacion cross-asset por escenario
# ================================================================
def diag_correlacion(ctx):
    assets = ctx["assets"]
    cand = ctx["candidates"]
    reps = ctx["reps"]
    rh = ctx["r_hist"]
    N, T, A = cand.shape
    if A < 2:
        return
    i, j = 0, 1

    corr_cand = np.array([np.corrcoef(cand[n, :, i], cand[n, :, j])[0, 1]
                          for n in range(N)])
    corr_reps = np.array([np.corrcoef(reps[s, :, i], reps[s, :, j])[0, 1]
                          for s in range(reps.shape[0])])
    corr_hist = float(np.corrcoef(rh[:, i], rh[:, j])[0, 1])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(corr_cand, bins=40, alpha=0.55, color="C0",
            label=f"candidatos (mean={corr_cand.mean():+.3f})")
    for s, c in enumerate(corr_reps):
        ax.axvline(c, color="C1", lw=1.3,
                   label="5 reps" if s == 0 else None)
    ax.axvline(corr_hist, color="k", lw=2.2,
               label=f"historico ({corr_hist:+.3f})")
    ax.set_title(f"corr({assets[i]}, {assets[j]}) por escenario")
    ax.set_xlabel("Pearson r")
    ax.legend(fontsize=9)
    save_fig(fig, "3_correlacion_cross_asset", SUBDIR)

    rows = [{"grupo": "candidato", "id": n, "corr": float(corr_cand[n])}
            for n in range(N)]
    for s in range(reps.shape[0]):
        rows.append({"grupo": "rep", "id": s + 1, "corr": float(corr_reps[s])})
    rows.append({"grupo": "historico", "id": 0, "corr": corr_hist})
    save_csv(pd.DataFrame(rows), "3_correlacion_cross_asset", SUBDIR)


# ================================================================
# 4) Path dependence: retornos semanales por bloque temporal
# ================================================================
def diag_path_dependence(ctx):
    assets = ctx["assets"]
    cand = ctx["candidates"]
    rh = ctx["r_hist"]
    N, T, A = cand.shape

    blocks = [(0, 40), (40, 80), (80, 120), (120, T)]
    fig, axes = plt.subplots(A, len(blocks), figsize=(16, 4 * A))
    if A == 1:
        axes = axes[None, :]

    stats = []
    for ai, a in enumerate(assets):
        for bi, (lo, hi) in enumerate(blocks):
            block_cand = cand[:, lo:hi, ai].ravel()
            block_hist = rh[lo:hi, ai]
            ax = axes[ai, bi]
            ax.hist(block_cand, bins=50, alpha=0.55, density=True,
                    color="C0", label="cand")
            ax.hist(block_hist, bins=15, alpha=0.55, density=True,
                    color="k", label="hist")
            ax.set_title(f"{a}  t={lo+1}..{hi}\n"
                         f"cand μ={block_cand.mean():+.4f}  σ={block_cand.std():.4f}")
            ax.set_xlabel("r_semanal")
            ax.legend(fontsize=7)
            stats.append({
                "asset": a, "bloque": f"t={lo+1}..{hi}",
                "cand_mean": float(block_cand.mean()),
                "cand_std":  float(block_cand.std()),
                "hist_mean": float(block_hist.mean()),
                "hist_std":  float(block_hist.std()),
            })
    save_fig(fig, "4_path_dependence", SUBDIR)
    save_csv(pd.DataFrame(stats), "4_path_dependence", SUBDIR)


# ================================================================
# 5) Sesgo del activo de resumen
# ================================================================
def diag_sesgo_resumen(ctx):
    assets = ctx["assets"]
    cand = ctx["candidates"]
    reps = ctx["reps"]
    summary_idx = ctx["summary_idx"]
    if len(assets) < 2:
        return
    other_idx = 1 - summary_idx if len(assets) == 2 else (summary_idx + 1) % len(assets)

    cum_cand = np.prod(1.0 + cand, axis=1) - 1.0   # (N, A)
    cum_reps = np.prod(1.0 + reps, axis=1) - 1.0   # (n_q, A)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(cum_cand[:, summary_idx], cum_cand[:, other_idx],
               s=8, alpha=0.3, color="C0", label="candidatos")
    ax.scatter(cum_reps[:, summary_idx], cum_reps[:, other_idx],
               s=80, color="C1", edgecolor="k",
               label=f"5 reps (orden por {assets[summary_idx]})", zorder=3)
    for s in range(reps.shape[0]):
        ax.annotate(f"q{s+1}",
                    (cum_reps[s, summary_idx], cum_reps[s, other_idx]),
                    textcoords="offset points", xytext=(8, 6), fontsize=10)
    ax.axhline(0, color="grey", lw=0.5)
    ax.axvline(0, color="grey", lw=0.5)
    ax.set_xlabel(f"cumret {assets[summary_idx]} (activo de resumen)")
    ax.set_ylabel(f"cumret {assets[other_idx]}")
    ax.set_title("Estratificacion: posicion de los 5 reps en el plano cross-asset")
    ax.legend()
    save_fig(fig, "5_sesgo_resumen", SUBDIR)

    rows = []
    for s in range(reps.shape[0]):
        rows.append({
            "rep_quintil": s + 1,
            f"cumret_{assets[summary_idx]}": float(cum_reps[s, summary_idx]),
            f"cumret_{assets[other_idx]}":   float(cum_reps[s, other_idx]),
        })
    save_csv(pd.DataFrame(rows), "5_sesgo_resumen", SUBDIR)


# ================================================================
# 6) Sanity: reproducibilidad + sensibilidad al seed
# ================================================================
def diag_sanity(ctx):
    assets = ctx["assets"]
    model = ctx["model"]
    initial_window = ctx["initial_window"]
    cand = ctx["candidates"]

    cand_again = generate_candidate_scenarios(
        model, initial_window, N=N_CANDIDATES, T=T_HORIZON, seed=SCENARIO_SEED,
    )
    reproducible = bool(np.allclose(cand_again, cand, atol=1e-6))

    cand_other = generate_candidate_scenarios(
        model, initial_window, N=N_CANDIDATES, T=T_HORIZON,
        seed=SCENARIO_SEED + 1,
    )
    cum_a = np.prod(1.0 + cand, axis=1) - 1.0
    cum_b = np.prod(1.0 + cand_other, axis=1) - 1.0

    rows = [{"check": "reproducibilidad (mismo seed)",
             "valor": "OK" if reproducible else "FAIL"}]
    for ai, a in enumerate(assets):
        rows.append({"check": f"{a}  cumret_T  mean  seed={SCENARIO_SEED}",
                     "valor": f"{cum_a[:, ai].mean():+.4f}"})
        rows.append({"check": f"{a}  cumret_T  mean  seed={SCENARIO_SEED+1}",
                     "valor": f"{cum_b[:, ai].mean():+.4f}"})
        rows.append({"check": f"{a}  cumret_T  std   seed={SCENARIO_SEED}",
                     "valor": f"{cum_a[:, ai].std():+.4f}"})
        rows.append({"check": f"{a}  cumret_T  std   seed={SCENARIO_SEED+1}",
                     "valor": f"{cum_b[:, ai].std():+.4f}"})
    save_csv(pd.DataFrame(rows), "6_sanity", SUBDIR)

    fig, axes = plt.subplots(1, len(assets), figsize=(12, 4.2))
    if len(assets) == 1:
        axes = [axes]
    for ai, a in enumerate(assets):
        ax = axes[ai]
        ax.hist(cum_a[:, ai], bins=40, alpha=0.55,
                label=f"seed={SCENARIO_SEED}")
        ax.hist(cum_b[:, ai], bins=40, alpha=0.55,
                label=f"seed={SCENARIO_SEED+1}")
        ax.set_title(f"{a} - cumret_T por seed")
        ax.set_xlabel("cumret")
        ax.legend(fontsize=8)
    save_fig(fig, "6_sanity_seeds", SUBDIR)


# ================================================================
# Main
# ================================================================
def main():
    print("=" * 70)
    print("INSPECCION DE ESCENARIOS")
    print("=" * 70)
    print(f"  N_candidatos = {N_CANDIDATES}")
    print(f"  n_reps       = {N_SCENARIOS}")
    print(f"  T            = {T_HORIZON}")
    print(f"  seed         = {SCENARIO_SEED}")
    print(f"  position     = {SCENARIO_POSITION}")
    print(f"  summary      = {SUMMARY_ASSET}")
    print("-" * 70)
    print("Construyendo contexto...")
    ctx = build_context()
    print(f"  assets     : {ctx['assets']}")
    print(f"  candidates : {ctx['candidates'].shape}")
    print(f"  reps       : {ctx['reps'].shape}")
    print(f"  output dir : {out_dir(SUBDIR)}")
    print("-" * 70)

    print("[1/6] sesgo cumret terminal")
    diag_sesgo(ctx)
    print("[2/6] dispersion (fan chart)")
    diag_dispersion(ctx)
    print("[3/6] correlacion cross-asset")
    diag_correlacion(ctx)
    print("[4/6] path-dependence por bloque temporal")
    diag_path_dependence(ctx)
    print("[5/6] sesgo del activo de resumen")
    diag_sesgo_resumen(ctx)
    print("[6/6] sanity / reproducibilidad")
    diag_sanity(ctx)

    print("-" * 70)
    print(f"Listo. Resultados en {out_dir(SUBDIR)}")


if __name__ == "__main__":
    main()
