"""Diagnostico exhaustivo del pipeline per-cell (paradigma 2026-05-27).

Abre el motor: 5 capas del pipeline son auditadas independientemente. Cada
capa imprime tablas, guarda plots+CSVs en inspeccion/per_cell_diagnostico_out/,
y emite un veredicto rapido (OK / revisar / problema).

Capas:
  1. Las 15 NNs        — convergencia, fan chart por celda, distancia entre redes
  2. Contextos por celda — p_dl(t), mu_mix(t), sigma_mix(t), heterogeneidad
  3. Escenarios compartidos — distribucion N=1000, los 5 reps, correlacion
  4. Politicas optimizadas — w(i,t), turnover, V[g,s] heatmap, z por celda
  5. Regret y seleccion — mean/worst regret, ganador por escenario

Uso:    python inspeccion/per_cell_diagnostico.py
Salida: inspeccion/per_cell_diagnostico_out/
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import (                                                    # noqa: E402
    DATA_DIR, DLConfig, LAMBDA_GRID, M_GRID, MODELS_DIR, N_CANDIDATES,
    N_SCENARIOS, RETURN_CSV, RETURN_COL, SCENARIO_SEED, T_HORIZON, W0,
)
from Regret_Grid import (                                               # noqa: E402
    build_ensemble_model,
    build_per_cell_context,
    build_shared_scenarios,
    cell_seed,
    compute_regret_and_select,
    load_market_data,
    predict_pbull_walking,
    run_per_cell_regret_grid,
    solve_portfolio,
    train_per_cell_nns,
)
from dl.generador_escenarios import generate_candidate_scenarios        # noqa: E402
from dl.prediccion_deciles import (                                     # noqa: E402
    build_windows, chrono_split, load_returns, plot_fan_chart,
)

OUT = Path(__file__).resolve().parent / "per_cell_diagnostico_out"
OUT.mkdir(exist_ok=True)

_LOG = []
_VERDICTOS = []


def say(*args):
    line = " ".join(str(a) for a in args)
    print(line)
    _LOG.append(line)


def veredicto(capa: str, estado: str, motivo: str) -> None:
    """estado in {OK, REVISAR, PROBLEMA}."""
    _VERDICTOS.append((capa, estado, motivo))
    icon = {"OK": "[OK]", "REVISAR": "[?] ", "PROBLEMA": "[X] "}[estado]
    say(f"  {icon} {capa}: {motivo}")


# ====================================================================
# Setup: cargar/entrenar las 15 NNs y precomputar todo lo compartido
# ====================================================================

def setup_pipeline():
    say("=" * 78)
    say("SETUP — cargando NNs por celda (usa cache si existe)")
    say("=" * 78)
    dl_config = DLConfig()
    nns = train_per_cell_nns(
        lambda_grid=LAMBDA_GRID, m_grid=M_GRID,
        dl_config=dl_config, models_dir=MODELS_DIR, force_retrain=False,
    )
    say(f"  -> {len(nns)} NNs cargadas")
    return nns, dl_config


# ====================================================================
# CAPA 1: las 15 NNs
# ====================================================================

def capa1_nns(nns, dl_config):
    say("\n" + "=" * 78)
    say("CAPA 1 — Las 15 NNs (convergencia, fan charts, distancia entre redes)")
    say("=" * 78)

    # Tabla resumen: seed, best_valid (leyendo del checkpoint)
    rows = []
    per_cell_dir = MODELS_DIR / "per_cell"
    for (lam, m), nn in nns.items():
        ckpt = per_cell_dir / f"decile_predictor_l{lam:.2f}_m{m:.2f}.pt"
        payload = torch.load(ckpt, map_location="cpu", weights_only=False)
        rows.append({
            "lambda":     lam,
            "m":          m,
            "seed":       cell_seed(lam, m),
            "best_valid": float(payload.get("best_valid", float("nan"))),
            "epochs":     len(payload.get("history", {}).get("train", [])),
        })
    df = pd.DataFrame(rows).sort_values(["lambda", "m"]).reset_index(drop=True)
    df.to_csv(OUT / "capa1_nns_summary.csv", index=False)
    say("\nResumen de las 15 NNs:")
    say(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Boxplot de best_valid
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(df["best_valid"].values, vert=True)
    ax.scatter([1] * len(df), df["best_valid"].values, alpha=0.6, color="C0", zorder=3)
    ax.set_ylabel("pinball loss (valid)")
    ax.set_title("Distribucion de best_valid entre las 15 NNs")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "capa1_best_valid_boxplot.png", dpi=140)
    plt.close(fig)

    # Distancia L2 entre pares de NNs (vectorizar state_dict)
    def flatten_nn(nn):
        net = nn.nets[0]
        return np.concatenate([p.detach().cpu().numpy().flatten()
                               for p in net.parameters()])

    cell_keys = list(nns.keys())
    vecs = np.stack([flatten_nn(nns[g]) for g in cell_keys], axis=0)        # (15, P)
    G = vecs.shape[0]
    dist = np.zeros((G, G))
    for i in range(G):
        for j in range(G):
            dist[i, j] = np.linalg.norm(vecs[i] - vecs[j])

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(dist, cmap="viridis")
    ax.set_xticks(range(G))
    ax.set_yticks(range(G))
    labels = [f"({l:.1f},{m:.1f})" for (l, m) in cell_keys]
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title("Distancia L2 entre pesos de pares de NNs")
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(OUT / "capa1_l2_distance_matrix.png", dpi=140)
    plt.close(fig)

    # Fan charts de 3 NNs (esquinas + medio del grid) sobre TEST
    df_ret = load_returns()
    X, Y, t_idx = build_windows(df_ret, dl_config.H)
    split = chrono_split(X, Y, t_idx, dl_config.split)
    sample_cells = [
        (LAMBDA_GRID[0], M_GRID[0]),     # esquina (min, min)
        (LAMBDA_GRID[len(LAMBDA_GRID)//2], M_GRID[len(M_GRID)//2]),  # centro
        (LAMBDA_GRID[-1], M_GRID[-1]),   # esquina (max, max)
    ]
    for (lam, m) in sample_cells:
        plot_fan_chart(
            nns[(lam, m)], split.X_test, split.Y_test, split.t_test,
            out_path=OUT / f"capa1_fanchart_l{lam:.1f}_m{m:.1f}.png",
            show=False,
            title_suffix=f"NN celda (lambda={lam:.1f}, m={m:.1f})",
        )

    # Veredictos
    best_valid_std = df["best_valid"].std()
    iu = np.triu_indices(G, k=1)
    mean_pairwise = dist[iu].mean()
    say(f"\n  best_valid: mean={df['best_valid'].mean():.4f}  std={best_valid_std:.4f}")
    say(f"  L2 entre pares: mean={mean_pairwise:.4f}  min={dist[iu].min():.4f}  max={dist[iu].max():.4f}")
    if best_valid_std < 1e-4:
        veredicto("Capa 1", "PROBLEMA",
                  f"best_valid casi identico entre las 15 NNs (std={best_valid_std:.6f})")
    elif mean_pairwise < 0.1:
        veredicto("Capa 1", "REVISAR",
                  f"pesos de NNs muy parecidos (L2 promedio={mean_pairwise:.3f})")
    else:
        veredicto("Capa 1", "OK",
                  f"15 NNs convergieron a soluciones distintas (L2 prom={mean_pairwise:.2f})")


# ====================================================================
# CAPA 2: contextos por celda (p_dl, mu_mix, sigma_mix)
# ====================================================================

def capa2_contextos(nns):
    say("\n" + "=" * 78)
    say("CAPA 2 — Contextos por celda (p_dl(t), mu_mix(t), sigma_mix(t))")
    say("=" * 78)

    # Construir contextos para las 15 celdas (con mu_hat_source='p_hist')
    contexts = {}
    for g, nn in nns.items():
        contexts[g] = build_per_cell_context(nn=nn, data_dir=DATA_DIR,
                                              T=T_HORIZON, mu_hat_source="p_hist")
    base_ctx = load_market_data(str(DATA_DIR))
    assets = base_ctx["assets"]
    T_vals = contexts[next(iter(contexts))]["T_vals"]

    # mu_hat: debe ser identico entre celdas (usa p_hist + r_hist, no la NN)
    say("\nmu_hat (debe ser identico en las 15 celdas; depende de p_hist y r_hist):")
    first_key = next(iter(contexts))
    mu_hat = contexts[first_key]["mu_hat"]
    mu_hat_tbl = pd.DataFrame({
        "asset":   [k[0] for k in mu_hat],
        "regime":  [k[1] for k in mu_hat],
        "mu_hat":  [mu_hat[k] for k in mu_hat],
    })
    say(mu_hat_tbl.to_string(index=False, float_format=lambda x: f"{x:+.4%}"))
    # Sanity: el resto de celdas tiene los mismos numeros?
    mismatches = 0
    for g, ctx in contexts.items():
        for k in mu_hat:
            if abs(ctx["mu_hat"][k] - mu_hat[k]) > 1e-9:
                mismatches += 1
    say(f"  mismatches en mu_hat entre celdas: {mismatches} (esperado: 0)")

    # p_bull(t) por celda: las 15 trayectorias superpuestas
    fig, axes = plt.subplots(len(assets), 1, figsize=(10, 6), sharex=True)
    for ai, asset in enumerate(assets):
        ax = axes[ai]
        for g, ctx in contexts.items():
            ax.plot(T_vals, ctx["p_dl"][asset]["bull"].values,
                    alpha=0.4, linewidth=0.9)
        ax.set_ylabel(f"p_bull {asset}")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
    axes[0].set_title("p_bull(t) — 15 trayectorias (una por celda)")
    axes[-1].set_xlabel("t (forward)")
    fig.tight_layout()
    fig.savefig(OUT / "capa2_pbull_15_cells.png", dpi=140)
    plt.close(fig)

    # mu_mix(t): las 15 trayectorias
    fig, axes = plt.subplots(len(assets), 1, figsize=(10, 6), sharex=True)
    for ai, asset in enumerate(assets):
        ax = axes[ai]
        for g, ctx in contexts.items():
            ax.plot(T_vals, ctx["mu_mix"][asset].values,
                    alpha=0.4, linewidth=0.9)
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
        ax.set_ylabel(f"mu_mix {asset}")
        ax.grid(True, alpha=0.3)
    axes[0].set_title("mu_mix(t) — 15 trayectorias (una por celda)")
    axes[-1].set_xlabel("t (forward)")
    fig.tight_layout()
    fig.savefig(OUT / "capa2_mumix_15_cells.png", dpi=140)
    plt.close(fig)

    # Dispersion estadistica
    rows = []
    for ai, asset in enumerate(assets):
        # Apila p_bull y mu_mix de las 15 celdas: (15, T)
        pbull_stack = np.stack([ctx["p_dl"][asset]["bull"].values
                                for ctx in contexts.values()], axis=0)
        mumix_stack = np.stack([ctx["mu_mix"][asset].values
                                for ctx in contexts.values()], axis=0)
        # Para cada t, std entre celdas. Despues promediamos en t.
        rows.append({
            "asset":              asset,
            "p_bull std (cells)": float(pbull_stack.std(axis=0).mean()),
            "p_bull rango max":   float((pbull_stack.max(axis=0) -
                                          pbull_stack.min(axis=0)).max()),
            "mu_mix std (cells)": float(mumix_stack.std(axis=0).mean()),
            "mu_mix std (t)":     float(mumix_stack.mean(axis=0).std()),
        })
    disp = pd.DataFrame(rows)
    disp.to_csv(OUT / "capa2_dispersion.csv", index=False)
    say("\nDispersion entre celdas (promediada en t):")
    say(disp.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    avg_pbull_std = disp["p_bull std (cells)"].mean()
    avg_mumix_std_t = disp["mu_mix std (t)"].mean()
    if avg_pbull_std < 0.01:
        veredicto("Capa 2", "PROBLEMA",
                  f"p_bull casi identico entre celdas (std={avg_pbull_std:.4f})")
    elif avg_mumix_std_t < 1e-5:
        veredicto("Capa 2", "REVISAR",
                  f"mu_mix(t) apenas varia en t (std={avg_mumix_std_t:.6f}) — esperado con p_hist")
    else:
        veredicto("Capa 2", "OK",
                  f"p_bull std/celda={avg_pbull_std:.3f}, mu_mix std/t={avg_mumix_std_t:.5f}")

    return contexts, assets, T_vals


# ====================================================================
# CAPA 3: escenarios compartidos
# ====================================================================

def capa3_escenarios(nns, contexts, assets):
    say("\n" + "=" * 78)
    say("CAPA 3 — Escenarios compartidos (N=1000 candidatos -> 5 reps)")
    say("=" * 78)

    ensemble = build_ensemble_model(nns)
    H = ensemble.config.H
    base_ctx = load_market_data(str(DATA_DIR))
    r_hist = base_ctx["r"]
    returns_history = np.stack(
        [r_hist[i].sort_index().values[:T_HORIZON] for i in assets], axis=1,
    ).astype(np.float32)
    initial_window = returns_history[-H:, :]
    w_ref = np.array([W0[i] for i in assets], dtype=np.float32)

    # 1) Generar los N=1000 candidatos
    candidates = generate_candidate_scenarios(
        ensemble, initial_window, N=N_CANDIDATES, T=T_HORIZON, seed=SCENARIO_SEED,
    )                                                          # (N, T, A)
    # ret port acumulado por candidato
    r_port = np.einsum("nti,i->nt", candidates, w_ref)
    cum_port = np.prod(1.0 + r_port, axis=1) - 1.0             # (N,)

    # 2) Los 5 representativos (por la misma funcion que usa main.py)
    scenarios = build_shared_scenarios(
        ensemble_nn=ensemble, initial_window=initial_window, w_ref=w_ref,
        N=N_CANDIDATES, T=T_HORIZON, n_scenarios=N_SCENARIOS, seed=SCENARIO_SEED,
    )                                                          # (5, T, A)
    r_port_reps = np.einsum("sti,i->st", scenarios, w_ref)
    cum_port_reps = np.prod(1.0 + r_port_reps, axis=1) - 1.0

    # Histograma con los 5 reps marcados
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(cum_port * 100, bins=60, alpha=0.7, color="C0",
            label=f"N={N_CANDIDATES} candidatos")
    for s, cum in enumerate(cum_port_reps):
        ax.axvline(cum * 100, color="C3", linewidth=1.5,
                   label=(f"reps (5)" if s == 0 else None))
        ax.text(cum * 100, ax.get_ylim()[1] * 0.95, f"s{s}",
                ha="center", fontsize=8, color="C3")
    ax.set_xlabel("Retorno acumulado de portafolio (w_ref=50/50) %")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Distribucion ret port — N candidatos vs 5 reps elegidos")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "capa3_hist_candidatos.png", dpi=140)
    plt.close(fig)

    # 5 escenarios: trayectorias por activo y capital de portafolio bajo w_ref
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    cmap = plt.get_cmap("viridis")
    colors = [cmap(s / max(N_SCENARIOS - 1, 1)) for s in range(N_SCENARIOS)]
    T_idx = np.arange(1, T_HORIZON + 1)

    for s in range(N_SCENARIOS):
        for ai, asset in enumerate(assets):
            axes[ai].plot(T_idx, scenarios[s, :, ai] * 100,
                          color=colors[s], linewidth=1.0, alpha=0.85,
                          label=f"s{s} ({cum_port_reps[s]:+.1%})")
            axes[ai].set_ylabel(f"r {asset} (%)")
            axes[ai].grid(True, alpha=0.3)
            axes[ai].axhline(0, color="black", linewidth=0.4, alpha=0.5)
        # Capital de portafolio bajo w_ref=50/50
        cap = np.cumprod(1.0 + r_port_reps[s]) * 10_000
        axes[2].plot(T_idx, cap, color=colors[s], linewidth=1.2,
                     label=f"s{s}")
    axes[2].set_ylabel("Capital w_ref (50/50) $")
    axes[2].axhline(10_000, color="black", linewidth=0.5, alpha=0.5,
                    linestyle="--")
    axes[2].set_xlabel("t (forward)")
    axes[2].grid(True, alpha=0.3)
    axes[0].set_title("Los 5 escenarios elegidos")
    axes[0].legend(loc="upper left", fontsize=8, ncol=N_SCENARIOS)
    fig.tight_layout()
    fig.savefig(OUT / "capa3_5_scenarios.png", dpi=140)
    plt.close(fig)

    # Correlacion SPX-CMC en N candidatos vs historico
    # Promedio sobre tiempo de la correlacion por escenario.
    if len(assets) == 2:
        corr_per_scenario = np.array([
            np.corrcoef(candidates[n, :, 0], candidates[n, :, 1])[0, 1]
            for n in range(N_CANDIDATES)
        ])
        r_h = np.stack([r_hist[i].values for i in assets], axis=1)
        corr_hist = float(np.corrcoef(r_h[:, 0], r_h[:, 1])[0, 1])
        say(f"\nCorrelacion SPX-CMC:")
        say(f"  historico:                {corr_hist:+.3f}")
        say(f"  N candidatos (mean):      {corr_per_scenario.mean():+.3f}")
        say(f"  N candidatos (5/50/95%):  {np.percentile(corr_per_scenario, 5):+.3f}"
            f" / {np.percentile(corr_per_scenario, 50):+.3f}"
            f" / {np.percentile(corr_per_scenario, 95):+.3f}")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(corr_per_scenario, bins=40, alpha=0.7, color="C0")
        ax.axvline(corr_hist, color="C3", linewidth=2, label=f"hist={corr_hist:+.2f}")
        ax.set_xlabel("corr(SPX, CMC) por escenario")
        ax.set_ylabel("Frec")
        ax.set_title("Correlacion entre activos: candidatos vs historico")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / "capa3_correlacion_assets.png", dpi=140)
        plt.close(fig)

    # Tabla resumen de los 5 escenarios
    rows = []
    for s in range(N_SCENARIOS):
        d = {"escenario": f"s{s}"}
        for ai, asset in enumerate(assets):
            cum = float(np.prod(1.0 + scenarios[s, :, ai]) - 1.0)
            d[f"ret_{asset}"] = cum
        d["ret_port_wref"] = float(cum_port_reps[s])
        rows.append(d)
    reps_df = pd.DataFrame(rows)
    reps_df.to_csv(OUT / "capa3_5_scenarios_summary.csv", index=False)
    say("\nLos 5 escenarios elegidos:")
    say(reps_df.to_string(index=False, float_format=lambda x: f"{x:+.2%}"))

    # Veredicto
    spread_port = float(cum_port_reps.max() - cum_port_reps.min())
    if spread_port < 0.2:
        veredicto("Capa 3", "REVISAR",
                  f"5 reps muy concentrados (spread ret={spread_port:.1%}); ensemble podria estar plano")
    elif corr_per_scenario.mean() < 0.0 and corr_hist > 0.2:
        veredicto("Capa 3", "REVISAR",
                  f"correlacion candidatos ({corr_per_scenario.mean():+.2f}) muy distinta a hist ({corr_hist:+.2f})")
    else:
        veredicto("Capa 3", "OK",
                  f"5 reps abarcan ret port [{cum_port_reps.min():+.1%}..{cum_port_reps.max():+.1%}], corr ok")

    return scenarios, candidates


# ====================================================================
# CAPA 4: politicas optimizadas
# ====================================================================

def capa4_politicas(contexts, scenarios, assets, T_vals):
    say("\n" + "=" * 78)
    say("CAPA 4 — Politicas optimizadas (w(i,t), turnover, V[g,s], z)")
    say("=" * 78)

    V_df, policies = run_per_cell_regret_grid(
        contexts, scenarios, list(LAMBDA_GRID), list(M_GRID),
    )
    cell_keys = list(policies.keys())

    # w(SPX, t) y w(CMC, t) para las 15 celdas
    fig, axes = plt.subplots(len(assets), 1, figsize=(11, 7), sharex=True)
    cmap = plt.get_cmap("plasma")
    for ai, asset in enumerate(assets):
        for gi, g in enumerate(cell_keys):
            w_sol, _, _, _ = policies[g]
            w_series = [w_sol[asset, t] for t in T_vals]
            axes[ai].plot(T_vals, w_series, alpha=0.5, linewidth=0.9,
                          color=cmap(gi / max(len(cell_keys) - 1, 1)))
        axes[ai].axhline(0.5, color="black", linewidth=0.4, alpha=0.5,
                         linestyle="--")
        axes[ai].set_ylabel(f"w({asset}, t)")
        axes[ai].set_ylim(-0.02, 1.02)
        axes[ai].grid(True, alpha=0.3)
    axes[0].set_title("Politicas optimizadas — w(asset, t) para las 15 celdas")
    axes[-1].set_xlabel("t")
    fig.tight_layout()
    fig.savefig(OUT / "capa4_politicas_w.png", dpi=140)
    plt.close(fig)

    # Turnover total por celda y z objetivo
    rows = []
    for g in cell_keys:
        w_sol, u_sol, v_sol, z = policies[g]
        turnover = sum(u_sol[i, t] + v_sol[i, t] for i in assets for t in T_vals)
        w_spx_mean = np.mean([w_sol["SPX", t] for t in T_vals])
        rows.append({
            "lambda":     g[0],
            "m":          g[1],
            "z":          z,
            "turnover":   float(turnover),
            "w_SPX_mean": float(w_spx_mean),
            "w_SPX_max":  float(np.max([w_sol["SPX", t] for t in T_vals])),
            "w_SPX_min":  float(np.min([w_sol["SPX", t] for t in T_vals])),
        })
    pol_df = pd.DataFrame(rows).sort_values(["lambda", "m"]).reset_index(drop=True)
    pol_df.to_csv(OUT / "capa4_politicas_summary.csv", index=False)
    say("\nResumen de politicas por celda:")
    say(pol_df.to_string(index=False,
                          float_format=lambda x: f"{x:.4f}"))

    # Bar plot de turnover
    fig, ax = plt.subplots(figsize=(11, 4))
    labels = [f"({l:.1f},{m:.1f})" for l, m in cell_keys]
    ax.bar(labels, [pol_df.loc[(pol_df["lambda"] == l) & (pol_df["m"] == m),
                                "turnover"].iloc[0]
                     for l, m in cell_keys])
    ax.set_ylabel("turnover total")
    ax.set_title("Turnover total por celda (Sum_t (u+v))")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "capa4_turnover.png", dpi=140)
    plt.close(fig)

    # Heatmap V[g, s]
    V_pivot = V_df.pivot_table(index=["lambda", "m"], columns="s", values="V",
                                aggfunc="first")
    fig, ax = plt.subplots(figsize=(7, 8))
    im = ax.imshow(V_pivot.values, aspect="auto", cmap="RdYlGn")
    ax.set_xticks(range(V_pivot.shape[1]))
    ax.set_xticklabels([f"s{s}" for s in V_pivot.columns])
    ax.set_yticks(range(V_pivot.shape[0]))
    ax.set_yticklabels([f"({l:.1f},{m:.1f})" for l, m in V_pivot.index],
                       fontsize=8)
    for i in range(V_pivot.shape[0]):
        for j in range(V_pivot.shape[1]):
            ax.text(j, i, f"${V_pivot.values[i, j] / 1000:.1f}k",
                    ha="center", va="center", fontsize=7, color="black")
    plt.colorbar(im, ax=ax, label="V[g, s] ($)")
    ax.set_title("V[g, s] — capital terminal")
    fig.tight_layout()
    fig.savefig(OUT / "capa4_V_heatmap.png", dpi=140)
    plt.close(fig)

    # z vs lambda agrupado por m
    fig, ax = plt.subplots(figsize=(8, 4))
    for m_val in M_GRID:
        sub = pol_df[pol_df["m"] == m_val].sort_values("lambda")
        ax.plot(sub["lambda"], sub["z"], marker="o", label=f"m={m_val}")
    ax.set_xlabel("lambda")
    ax.set_ylabel("z (objetivo)")
    ax.set_title("Valor objetivo z vs lambda (por m)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "capa4_z_vs_lambda.png", dpi=140)
    plt.close(fig)

    # Veredicto
    z_monotone = pol_df.groupby("m").apply(
        lambda d: d.sort_values("lambda")["z"].is_monotonic_decreasing
    ).all()
    turnover_range = pol_df["turnover"].max() - pol_df["turnover"].min()
    w_extreme = (pol_df["w_SPX_max"] > 0.99).any() or (pol_df["w_SPX_min"] < 0.01).any()
    if not z_monotone:
        veredicto("Capa 4", "REVISAR",
                  "z NO es monotonicamente decreciente en lambda (esperado)")
    elif w_extreme:
        veredicto("Capa 4", "REVISAR",
                  "alguna celda va 100% a un activo — extremo, revisar V_max y mu_mix")
    elif turnover_range < 0.1:
        veredicto("Capa 4", "REVISAR",
                  f"turnover similar entre celdas (rango={turnover_range:.2f}) — m no responde")
    else:
        veredicto("Capa 4", "OK",
                  f"z monotono en lambda, turnover responde, sin extremos")

    return V_df, policies


# ====================================================================
# CAPA 5: regret y seleccion
# ====================================================================

def capa5_regret(V_df, policies, scenarios):
    say("\n" + "=" * 78)
    say("CAPA 5 — Regret y seleccion de g* (ec. 22-24)")
    say("=" * 78)
    res = compute_regret_and_select(V_df)

    summary = res["regret_summary"].sort_values("mean_regret").reset_index()
    summary.to_csv(OUT / "capa5_regret_summary_sorted.csv", index=False)
    say("\nRegret summary (ordenado por mean_regret):")
    say(summary.to_string(index=False,
                          float_format=lambda x: f"${x:,.2f}"))

    # Bar plot mean/worst regret
    labels = [f"({l:.1f},{m:.1f})" for l, m in summary[["lambda", "m"]].values]
    x = np.arange(len(labels))
    width = 0.4
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(x - width / 2, summary["mean_regret"], width, label="mean_regret")
    ax.bar(x + width / 2, summary["worst_regret"], width, label="worst_regret")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("regret ($)")
    ax.set_title("mean / worst regret por celda (ordenado por mean)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "capa5_regret_bars.png", dpi=140)
    plt.close(fig)

    # Ganador por escenario: cual celda maximiza V[g, s] para cada s
    V_table = res["V_table"]
    winners = V_table.idxmax(axis=0)                          # Series indexada por s
    say("\nGanador por escenario (celda que maximiza V):")
    for s in V_table.columns:
        g_win = winners[s]
        V_win = V_table.loc[g_win, s]
        say(f"  s{s}: lambda={g_win[0]:.2f}  m={g_win[1]:.2f}  V=${V_win:,.0f}")

    # Heatmap "winner heat": cuantas veces gana cada celda
    win_counts = winners.value_counts()
    win_arr = np.zeros((len(LAMBDA_GRID), len(M_GRID)))
    for (lam, m), c in win_counts.items():
        i = list(LAMBDA_GRID).index(lam)
        j = list(M_GRID).index(m)
        win_arr[i, j] = c

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(win_arr, cmap="YlOrRd")
    ax.set_xticks(range(len(M_GRID)))
    ax.set_xticklabels([f"m={m}" for m in M_GRID])
    ax.set_yticks(range(len(LAMBDA_GRID)))
    ax.set_yticklabels([f"lam={l}" for l in LAMBDA_GRID])
    for i in range(win_arr.shape[0]):
        for j in range(win_arr.shape[1]):
            ax.text(j, i, f"{int(win_arr[i, j])}", ha="center", va="center")
    plt.colorbar(im, ax=ax, label="# escenarios ganados")
    ax.set_title("Cuantas veces gana cada celda sobre los 5 escenarios")
    fig.tight_layout()
    fig.savefig(OUT / "capa5_winner_heatmap.png", dpi=140)
    plt.close(fig)

    # Seleccion
    lam_m, m_m = res["g_mean"]
    lam_w, m_w = res["g_worst"]
    say(f"\nSeleccion:")
    say(f"  g*_mean  = (lambda={lam_m:.2f}, m={m_m:.2f})  "
        f"mean_regret=${res['g_mean_metric']:,.2f}")
    say(f"  g*_worst = (lambda={lam_w:.2f}, m={m_w:.2f})  "
        f"worst_regret=${res['g_worst_metric']:,.2f}")

    # Veredicto
    on_boundary_mean = lam_m in (LAMBDA_GRID[0], LAMBDA_GRID[-1]) or m_m in (M_GRID[0], M_GRID[-1])
    on_boundary_worst = lam_w in (LAMBDA_GRID[0], LAMBDA_GRID[-1]) or m_w in (M_GRID[0], M_GRID[-1])
    n_distinct_winners = len(win_counts)
    if on_boundary_mean or on_boundary_worst:
        veredicto("Capa 5", "REVISAR",
                  f"g* en frontera (mean={on_boundary_mean}, worst={on_boundary_worst}) -> extender grid")
    elif n_distinct_winners <= 1:
        veredicto("Capa 5", "REVISAR",
                  f"una sola celda gana en todos los escenarios — regret degenera")
    else:
        veredicto("Capa 5", "OK",
                  f"g*_mean estable, {n_distinct_winners} celdas ganan algun escenario")

    return res


# ====================================================================
# Main
# ====================================================================

def main():
    nns, dl_config = setup_pipeline()
    capa1_nns(nns, dl_config)
    contexts, assets, T_vals = capa2_contextos(nns)
    scenarios, candidates    = capa3_escenarios(nns, contexts, assets)
    V_df, policies           = capa4_politicas(contexts, scenarios, assets, T_vals)
    capa5_regret(V_df, policies, scenarios)

    # Veredicto final + dump del log
    say("\n" + "=" * 78)
    say("VEREDICTO FINAL")
    say("=" * 78)
    for capa, estado, motivo in _VERDICTOS:
        icon = {"OK": "[OK] ", "REVISAR": "[?]  ", "PROBLEMA": "[X]  "}[estado]
        say(f"  {icon}{capa:<10} {estado:<10} {motivo}")

    (OUT / "log.txt").write_text("\n".join(_LOG), encoding="utf-8")
    say(f"\nReporte completo: {OUT / 'log.txt'}")
    say(f"Plots y CSVs:     {OUT}")


if __name__ == "__main__":
    main()
