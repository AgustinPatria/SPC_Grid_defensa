"""Diagnostico del modulo de regimen / prediccion del LSTM cuantilico.

Corre con:
    python -m inspeccion.regimen
    (o)
    python inspeccion/regimen.py

Produce 1 PNG + 1 CSV por diagnostico en `inspeccion/regimen_out/`:

  1. pbull_walking_vs_rollout  walking-window (lo que ve mu_mix) vs rollout
                               autoregresivo (lo que ven los escenarios). La grieta
                               crucial: si difieren, el contexto del optimizador y
                               las trayectorias simuladas viven en mundos distintos.
  2. pbull_serie_dist          serie y distribucion de p_bull(t) walking. Detecta
                               colapso a 0 o a 1.
  3. rollout_step_by_step      los 5 deciles predichos en cada step del rollout para
                               una sola trayectoria. Si el rollout esta envenenado,
                               los deciles colapsan con t.
  4. sensibilidad_threshold    p_bull medio para distintos BULL_THRESHOLD.
                               Detecta si el umbral 0.0 esta corriendo al sesgo bear.
  5. calibracion_deciles       fraccion de retornos reales <= r_hat^q vs q nominal,
                               in-sample. Diagonal = perfecto. Curva por debajo
                               = el modelo sobreestima los retornos; por encima =
                               los subestima (sesgo bear).
  6. sesgo_deciles             histograma de deciles predichos vs realizados,
                               in-sample. Muestra el sesgo bruto del LSTM por
                               activo.

Diagnosticos 5 y 6 son in-sample (mismas semanas con las que se entreno) — sirven
para ver el sesgo aprendido, no para evaluar generalizacion.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from inspeccion._common import save_fig, save_csv, out_dir

from config import (
    BULL_THRESHOLD,
    CHECKPOINT_PATH,
    DATA_DIR,
    SCENARIO_SEED,
    T_HORIZON,
)
from dl.prediccion_deciles import load_checkpoint
from dl.regimen_predicted import regimen_from_deciles
from Regret_Grid import load_market_data, predict_pbull_walking


SUBDIR = "regimen"


def build_context():
    """Carga el LSTM y el historico recortado a T_HORIZON, y precomputa los
    deciles walking-window (idx in [H, T)). Esto evita recalcular en cada diag."""
    base_ctx = load_market_data(str(DATA_DIR))
    assets = list(base_ctx["assets"])
    model = load_checkpoint(CHECKPOINT_PATH)
    H = model.config.H
    Q = model.config.n_quantiles
    nominal_q = np.asarray(model.config.quantiles, dtype=np.float32)

    r_hist = base_ctx["r"]
    returns_history = np.stack(
        [r_hist[i].sort_index().values[:T_HORIZON] for i in assets], axis=1,
    ).astype(np.float32)
    initial_window = returns_history[-H:, :].astype(np.float32)

    # Deciles predichos para cada ventana walking idx in [H, T).
    A = len(assets)
    deciles_walk = np.full((T_HORIZON, A, Q), np.nan, dtype=np.float32)
    for idx in range(H, T_HORIZON):
        window = returns_history[idx - H: idx, :]
        x = ((window - model.mean) / model.std).astype(np.float32)[None, :, :]
        x_t = torch.from_numpy(x)
        with torch.no_grad():
            outs = [net(x_t).numpy()[0] for net in model.nets]
        preds = np.mean(np.stack(outs, axis=0), axis=0)
        preds = np.sort(preds, axis=-1)
        deciles_walk[idx] = preds

    return {
        "assets":          assets,
        "model":           model,
        "H":               H,
        "Q":               Q,
        "nominal_q":       nominal_q,
        "returns_history": returns_history,
        "initial_window":  initial_window,
        "deciles_walk":    deciles_walk,
    }


def _rollout_deciles(model, initial_window, T, seed):
    """Hace un rollout autoregresivo de largo T desde initial_window y devuelve
    los deciles predichos en cada paso (T, A, Q) y el retorno muestreado (T, A)."""
    A = initial_window.shape[1]
    Q = model.config.n_quantiles
    rng = np.random.default_rng(seed)
    window = initial_window.copy().astype(np.float32)
    deciles_t = np.empty((T, A, Q), dtype=np.float32)
    sampled_r = np.empty((T, A), dtype=np.float32)
    for t in range(T):
        x = ((window - model.mean) / model.std).astype(np.float32)[None, :, :]
        x_t = torch.from_numpy(x)
        with torch.no_grad():
            outs = [net(x_t).numpy()[0] for net in model.nets]
        preds = np.mean(np.stack(outs, axis=0), axis=0)
        preds = np.sort(preds, axis=-1)
        deciles_t[t] = preds
        q_idx = rng.integers(low=0, high=Q, size=A)
        r_t = preds[np.arange(A), q_idx]
        sampled_r[t] = r_t
        window = np.concatenate([window[1:], r_t[None, :]], axis=0)
    return deciles_t, sampled_r


# ================================================================
# 1) p_bull walking vs rollout autoregresivo
# ================================================================
def diag_pbull_walking_vs_rollout(ctx):
    assets = ctx["assets"]
    model = ctx["model"]
    H = ctx["H"]
    returns_history = ctx["returns_history"]
    initial_window = ctx["initial_window"]
    T = T_HORIZON

    p_walk = predict_pbull_walking(model, returns_history, T)   # (T, A)
    deciles_roll, _ = _rollout_deciles(model, initial_window, T, SCENARIO_SEED)
    p_roll, _ = regimen_from_deciles(deciles_roll)              # (T, A)

    fig, axes = plt.subplots(len(assets), 1, figsize=(13, 4.5 * len(assets)))
    if len(assets) == 1:
        axes = [axes]
    tt = np.arange(1, T + 1)
    for ai, a in enumerate(assets):
        ax = axes[ai]
        ax.plot(tt, p_walk[:, ai], color="C0", lw=1.4,
                label=f"walking (mean t>H = {p_walk[H:, ai].mean():.2f})")
        ax.plot(tt, p_roll[:, ai], color="C1", lw=1.0, alpha=0.85,
                label=f"rollout (mean = {p_roll[:, ai].mean():.2f})")
        ax.axhline(0.5, color="grey", lw=0.5)
        ax.set_title(f"{a} - p_bull(t): walking-window vs rollout autoregresivo")
        ax.set_xlabel("semana")
        ax.set_ylabel("p_bull")
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
    save_fig(fig, "1_pbull_walking_vs_rollout", SUBDIR)

    rows = []
    for ai, a in enumerate(assets):
        rows.append({"asset": a, "metodo": "walking",
                     "mean":   float(p_walk[H:, ai].mean()),
                     "median": float(np.median(p_walk[H:, ai])),
                     "std":    float(p_walk[H:, ai].std()),
                     "min":    float(p_walk[H:, ai].min()),
                     "max":    float(p_walk[H:, ai].max())})
        rows.append({"asset": a, "metodo": "rollout",
                     "mean":   float(p_roll[:, ai].mean()),
                     "median": float(np.median(p_roll[:, ai])),
                     "std":    float(p_roll[:, ai].std()),
                     "min":    float(p_roll[:, ai].min()),
                     "max":    float(p_roll[:, ai].max())})
    save_csv(pd.DataFrame(rows), "1_pbull_walking_vs_rollout", SUBDIR)


# ================================================================
# 2) Serie + distribucion de p_bull walking
# ================================================================
def diag_pbull_serie_dist(ctx):
    assets = ctx["assets"]
    model = ctx["model"]
    H = ctx["H"]
    returns_history = ctx["returns_history"]
    T = T_HORIZON

    p_walk = predict_pbull_walking(model, returns_history, T)

    fig, axes = plt.subplots(len(assets), 2, figsize=(13, 4 * len(assets)))
    if len(assets) == 1:
        axes = axes[None, :]
    tt = np.arange(1, T + 1)
    for ai, a in enumerate(assets):
        ax_s = axes[ai, 0]
        ax_h = axes[ai, 1]
        ax_s.plot(tt, p_walk[:, ai], color="C0", lw=1.2)
        ax_s.axhline(0.5, color="grey", lw=0.5)
        ax_s.set_title(f"{a} - p_bull(t) walking")
        ax_s.set_xlabel("semana")
        ax_s.set_ylabel("p_bull")
        ax_s.set_ylim(-0.05, 1.05)

        ax_h.hist(p_walk[H:, ai], bins=np.linspace(0, 1, 21),
                  color="C0", alpha=0.75)
        ax_h.set_title(f"{a} - distribucion (t > H)")
        ax_h.set_xlabel("p_bull")
        ax_h.set_xlim(-0.05, 1.05)
    save_fig(fig, "2_pbull_serie_dist", SUBDIR)

    rows = []
    for ai, a in enumerate(assets):
        col = p_walk[H:, ai]
        rows.append({"asset": a,
                     "mean":   float(col.mean()),
                     "median": float(np.median(col)),
                     "frac_eq_0": float((col == 0).mean()),
                     "frac_lt_0.2": float((col < 0.2).mean()),
                     "frac_gt_0.8": float((col > 0.8).mean()),
                     "frac_eq_1": float((col == 1).mean())})
    save_csv(pd.DataFrame(rows), "2_pbull_serie_dist", SUBDIR)


# ================================================================
# 3) Rollout step-by-step: 5 deciles vs t
# ================================================================
def diag_rollout_step_by_step(ctx):
    assets = ctx["assets"]
    model = ctx["model"]
    initial_window = ctx["initial_window"]
    Q = ctx["Q"]
    nominal_q = ctx["nominal_q"]
    T = T_HORIZON
    A = len(assets)

    deciles_t, sampled = _rollout_deciles(model, initial_window, T, SCENARIO_SEED)

    tt = np.arange(1, T + 1)
    fig, axes = plt.subplots(A, 1, figsize=(13, 4.5 * A))
    if A == 1:
        axes = [axes]
    colors = plt.cm.RdYlGn(np.linspace(0.15, 0.85, Q))
    for ai, a in enumerate(assets):
        ax = axes[ai]
        for q in range(Q):
            ax.plot(tt, deciles_t[:, ai, q], color=colors[q], lw=1.0,
                    label=f"q={nominal_q[q]:.1f}")
        ax.axhline(0.0, color="k", lw=0.6)
        ax.scatter(tt, sampled[:, ai], s=4, color="k", alpha=0.4,
                   label="sampled r_t")
        ax.set_title(f"{a} - deciles predichos durante el rollout (seed={SCENARIO_SEED})")
        ax.set_xlabel("semana del rollout")
        ax.set_ylabel("r_hat")
        ax.legend(fontsize=8, ncol=Q + 1)
    save_fig(fig, "3_rollout_step_by_step", SUBDIR)

    rows = []
    early = slice(0, 10)
    late = slice(T - 10, T)
    for ai, a in enumerate(assets):
        for q in range(Q):
            rows.append({
                "asset": a, "q_nominal": float(nominal_q[q]),
                "mean_t_1_10":   float(deciles_t[early, ai, q].mean()),
                "mean_t_last10": float(deciles_t[late, ai, q].mean()),
                "delta":         float(deciles_t[late, ai, q].mean()
                                       - deciles_t[early, ai, q].mean()),
            })
    save_csv(pd.DataFrame(rows), "3_rollout_step_by_step", SUBDIR)


# ================================================================
# 4) Sensibilidad a BULL_THRESHOLD
# ================================================================
def diag_sensibilidad_threshold(ctx):
    assets = ctx["assets"]
    H = ctx["H"]
    deciles_walk = ctx["deciles_walk"]      # (T, A, Q) con [:H] = NaN

    thresholds = [-0.03, -0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02, 0.03]
    rows = []
    for thr in thresholds:
        pbull = (deciles_walk[H:] >= thr).mean(axis=-1)   # (T-H, A)
        for ai, a in enumerate(assets):
            rows.append({"asset": a, "threshold": thr,
                         "p_bull_mean":     float(pbull[:, ai].mean()),
                         "frac_pbull_eq_0": float((pbull[:, ai] == 0).mean()),
                         "frac_pbull_eq_1": float((pbull[:, ai] == 1).mean())})
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, len(assets), figsize=(13, 4.7))
    if len(assets) == 1:
        axes = [axes]
    for ai, a in enumerate(assets):
        sub = df[df.asset == a]
        ax = axes[ai]
        ax.plot(sub.threshold, sub.p_bull_mean, "o-", color="C0",
                label="mean(p_bull)")
        ax.plot(sub.threshold, sub.frac_pbull_eq_0, "o--", color="C3",
                label="frac p_bull=0", alpha=0.7)
        ax.plot(sub.threshold, sub.frac_pbull_eq_1, "o--", color="C2",
                label="frac p_bull=1", alpha=0.7)
        ax.axvline(BULL_THRESHOLD, color="k", ls="--", lw=1,
                   label=f"actual ({BULL_THRESHOLD})")
        ax.set_title(f"{a} - p_bull vs threshold")
        ax.set_xlabel("threshold")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8)
    save_fig(fig, "4_sensibilidad_threshold", SUBDIR)
    save_csv(df, "4_sensibilidad_threshold", SUBDIR)


# ================================================================
# 5) Calibracion cuantilica in-sample
# ================================================================
def diag_calibracion_deciles(ctx):
    assets = ctx["assets"]
    H = ctx["H"]
    Q = ctx["Q"]
    nominal_q = ctx["nominal_q"]
    deciles_walk = ctx["deciles_walk"]
    returns_history = ctx["returns_history"]
    T = T_HORIZON
    A = len(assets)

    # realizado en cada idx in [H, T): returns_history[idx]
    realized = returns_history[H:T, :]              # (T-H, A)
    deciles = deciles_walk[H:T, :, :]               # (T-H, A, Q)

    fig, axes = plt.subplots(1, A, figsize=(13, 5))
    if A == 1:
        axes = [axes]
    rows = []
    for ai, a in enumerate(assets):
        emp_q = np.array([(realized[:, ai] <= deciles[:, ai, q]).mean()
                          for q in range(Q)])
        ax = axes[ai]
        ax.plot(nominal_q, emp_q, "o-", color="C0", lw=1.8, label="empirico")
        ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=1, label="perfecto")
        ax.set_title(f"{a} - calibracion cuantilica (in-sample, n={realized.shape[0]})")
        ax.set_xlabel("q nominal")
        ax.set_ylabel("frac(realizado <= r_hat^q)")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.legend(fontsize=9)
        for q in range(Q):
            rows.append({"asset": a,
                         "q_nominal":   float(nominal_q[q]),
                         "q_empirico":  float(emp_q[q]),
                         "desviacion":  float(emp_q[q] - nominal_q[q])})
    save_fig(fig, "5_calibracion_deciles", SUBDIR)
    save_csv(pd.DataFrame(rows), "5_calibracion_deciles", SUBDIR)


# ================================================================
# 6) Sesgo de deciles vs realizado in-sample
# ================================================================
def diag_sesgo_deciles(ctx):
    assets = ctx["assets"]
    H = ctx["H"]
    Q = ctx["Q"]
    nominal_q = ctx["nominal_q"]
    deciles_walk = ctx["deciles_walk"]
    returns_history = ctx["returns_history"]
    T = T_HORIZON
    A = len(assets)

    realized = returns_history[H:T, :]              # (T-H, A)
    deciles  = deciles_walk[H:T, :, :]              # (T-H, A, Q)

    fig, axes = plt.subplots(1, A, figsize=(13, 5))
    if A == 1:
        axes = [axes]
    rows = []
    for ai, a in enumerate(assets):
        ax = axes[ai]
        ax.hist(realized[:, ai], bins=30, alpha=0.55, density=True,
                color="k",
                label=f"realizado (μ={realized[:, ai].mean():+.4f})")
        ax.hist(deciles[:, ai, :].ravel(), bins=30, alpha=0.55, density=True,
                color="C1",
                label=f"deciles ({nominal_q.size}*n) (μ={deciles[:, ai, :].mean():+.4f})")
        ax.axvline(0, color="grey", lw=0.5)
        ax.set_title(f"{a} - distribucion semanal (in-sample)")
        ax.set_xlabel("r")
        ax.legend(fontsize=8)
        mr = float(realized[:, ai].mean())
        for q in range(Q):
            md = float(deciles[:, ai, q].mean())
            rows.append({
                "asset": a,
                "q_nominal":      float(nominal_q[q]),
                "mean_decil":     md,
                "mean_realizado": mr,
                "diff_decil_minus_real": md - mr,
            })
    save_fig(fig, "6_sesgo_deciles", SUBDIR)
    save_csv(pd.DataFrame(rows), "6_sesgo_deciles", SUBDIR)


# ================================================================
# Main
# ================================================================
def main():
    print("=" * 70)
    print("INSPECCION DE REGIMEN / LSTM")
    print("=" * 70)
    print(f"  BULL_THRESHOLD = {BULL_THRESHOLD}")
    print(f"  T              = {T_HORIZON}")
    print(f"  seed (rollout) = {SCENARIO_SEED}")
    print("-" * 70)
    print("Construyendo contexto (carga LSTM + precomputa deciles walking)...")
    ctx = build_context()
    print(f"  assets       : {ctx['assets']}")
    print(f"  H            : {ctx['H']}")
    print(f"  Q (deciles)  : {ctx['Q']}")
    print(f"  nominal_q    : {ctx['nominal_q'].tolist()}")
    print(f"  output dir   : {out_dir(SUBDIR)}")
    print("-" * 70)

    print("[1/6] p_bull walking vs rollout")
    diag_pbull_walking_vs_rollout(ctx)
    print("[2/6] p_bull(t) serie + distribucion")
    diag_pbull_serie_dist(ctx)
    print("[3/6] rollout step-by-step")
    diag_rollout_step_by_step(ctx)
    print("[4/6] sensibilidad a BULL_THRESHOLD")
    diag_sensibilidad_threshold(ctx)
    print("[5/6] calibracion cuantilica in-sample")
    diag_calibracion_deciles(ctx)
    print("[6/6] sesgo de deciles vs realizado")
    diag_sesgo_deciles(ctx)

    print("-" * 70)
    print(f"Listo. Resultados en {out_dir(SUBDIR)}")


if __name__ == "__main__":
    main()
