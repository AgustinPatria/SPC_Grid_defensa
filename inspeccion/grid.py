"""Diagnostico del optimizador + regret-grid (eslabon final del pipeline).

Pregunta: con el DL fuera de la ecuacion, ¿el optimizador y el regret-grid
funcionan bien? ¿O hay problemas propios del eslabon final?

Corre con:
    python -m inspeccion.grid
    (o)
    python inspeccion/grid.py

Diagnosticos (1 PNG + 1 CSV cada uno) en `inspeccion/grid_out/`:

  1. regret_heatmap        heatmap de R[g, s] sobre la grilla estandar.
                           Visualiza el paisaje de regret y donde esta el minimo.
  2. V_heatmap             heatmap de V[g, s]. Si V es monotono en lambda
                           para todo s, el grid solo elige el extremo.
  3. boundary              corre una grilla extendida (lambda hasta 3.0,
                           m hasta 1.0). Si g* sigue tocando esquina, el
                           rango original era muy chico Y el problema es del DL.
  4. politicas             trayectorias w(i, t) para las 4 esquinas del grid
                           + g*_mean. Detecta si hay rebalanceo o todo plano.
  5. turnover              heatmap de turnover total (sum |u|+|v|) por g.
                           Si turnover ≈ 0 uniformemente, m es inerte porque
                           ninguna politica rebalancea.
  6. dl_vs_optbase         resuelve la grilla con mu_mix HISTORICO en vez
                           de DL, simula en los MISMOS escenarios DL. Aisla
                           cuanto del problema es del DL vs del optimizador.
"""
from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from inspeccion._common import save_fig, save_csv, out_dir

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
from Regret_Grid import (
    build_dl_context,
    compute_regret_and_select,
    load_market_data,
    run_regret_grid,
    simulate_capital_on_scenario,
)


SUBDIR = "grid"


def build_context():
    """Construye DL ctx + OPT base ctx + corre la grilla estandar sobre DL."""
    print("Cargando contexto DL (lleva ~10s)...")
    dl_ctx = build_dl_context(
        data_dir=DATA_DIR, checkpoint_path=CHECKPOINT_PATH,
        T=T_HORIZON, N_candidates=N_CANDIDATES,
        n_scenarios=N_SCENARIOS, seed=SCENARIO_SEED,
        summary_asset=SUMMARY_ASSET,
    )
    print("Cargando contexto OPT base (historico)...")
    opt_base_ctx = load_market_data(str(DATA_DIR))

    print("Corriendo grilla estandar sobre DL ctx (15 solves)...")
    lambda_grid = list(LAMBDA_GRID)
    m_grid = list(M_GRID)
    V_df_dl, policies_dl = run_regret_grid(dl_ctx, lambda_grid, m_grid)
    res_dl = compute_regret_and_select(V_df_dl)

    return {
        "dl_ctx":       dl_ctx,
        "opt_base_ctx": opt_base_ctx,
        "lambda_grid":  lambda_grid,
        "m_grid":       m_grid,
        "V_df_dl":      V_df_dl,
        "policies_dl":  policies_dl,
        "res_dl":       res_dl,
    }


def _pivot_metric(df, value_col):
    return df.pivot_table(
        index="lambda", columns="m", values=value_col, aggfunc="first",
    ).sort_index().sort_index(axis=1)


# ================================================================
# 1) Regret heatmap
# ================================================================
def diag_regret_heatmap(ctx):
    V_df = ctx["V_df_dl"]
    V_tab = V_df.pivot_table(index=["lambda", "m"], columns="s", values="V",
                             aggfunc="first")
    V_best = V_tab.max(axis=0)
    R_tab = V_best - V_tab                          # (g, s)
    # Mean regret por g
    mean_R = R_tab.mean(axis=1).unstack("m").sort_index().sort_index(axis=1)
    # Por escenario
    n_S = R_tab.shape[1]
    fig, axes = plt.subplots(1, n_S + 1, figsize=(3.5 * (n_S + 1), 4))
    for s in range(n_S):
        ax = axes[s]
        mat = R_tab[s].unstack("m").sort_index().sort_index(axis=1)
        im = ax.imshow(mat.values, cmap="RdYlGn_r", aspect="auto")
        ax.set_xticks(range(len(mat.columns)))
        ax.set_xticklabels([f"{m:.1f}" for m in mat.columns])
        ax.set_yticks(range(len(mat.index)))
        ax.set_yticklabels([f"{l:.2f}" for l in mat.index])
        ax.set_xlabel("m"); ax.set_ylabel("lambda")
        ax.set_title(f"R[g, s={s}]")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"${mat.values[i, j]:,.0f}",
                        ha="center", va="center", fontsize=7)
    ax = axes[n_S]
    im = ax.imshow(mean_R.values, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(len(mean_R.columns)))
    ax.set_xticklabels([f"{m:.1f}" for m in mean_R.columns])
    ax.set_yticks(range(len(mean_R.index)))
    ax.set_yticklabels([f"{l:.2f}" for l in mean_R.index])
    ax.set_xlabel("m"); ax.set_ylabel("lambda")
    ax.set_title("mean(R[g, ·])")
    for i in range(mean_R.shape[0]):
        for j in range(mean_R.shape[1]):
            ax.text(j, i, f"${mean_R.values[i, j]:,.0f}",
                    ha="center", va="center", fontsize=7)
    fig.tight_layout()
    save_fig(fig, "1_regret_heatmap", SUBDIR)
    save_csv(R_tab.reset_index(), "1_regret_table", SUBDIR)


# ================================================================
# 2) V heatmap
# ================================================================
def diag_V_heatmap(ctx):
    V_df = ctx["V_df_dl"]
    V_tab = V_df.pivot_table(index=["lambda", "m"], columns="s", values="V")
    n_S = V_tab.shape[1]
    mean_V = V_tab.mean(axis=1).unstack("m").sort_index().sort_index(axis=1)
    fig, axes = plt.subplots(1, n_S + 1, figsize=(3.5 * (n_S + 1), 4))
    for s in range(n_S):
        ax = axes[s]
        mat = V_tab[s].unstack("m").sort_index().sort_index(axis=1)
        ax.imshow(mat.values, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(mat.columns)))
        ax.set_xticklabels([f"{m:.1f}" for m in mat.columns])
        ax.set_yticks(range(len(mat.index)))
        ax.set_yticklabels([f"{l:.2f}" for l in mat.index])
        ax.set_xlabel("m"); ax.set_ylabel("lambda")
        ax.set_title(f"V[g, s={s}]")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"${mat.values[i, j]:,.0f}",
                        ha="center", va="center", fontsize=7, color="white")
    ax = axes[n_S]
    ax.imshow(mean_V.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(mean_V.columns)))
    ax.set_xticklabels([f"{m:.1f}" for m in mean_V.columns])
    ax.set_yticks(range(len(mean_V.index)))
    ax.set_yticklabels([f"{l:.2f}" for l in mean_V.index])
    ax.set_xlabel("m"); ax.set_ylabel("lambda")
    ax.set_title("mean V")
    for i in range(mean_V.shape[0]):
        for j in range(mean_V.shape[1]):
            ax.text(j, i, f"${mean_V.values[i, j]:,.0f}",
                    ha="center", va="center", fontsize=7, color="white")
    fig.tight_layout()
    save_fig(fig, "2_V_heatmap", SUBDIR)
    save_csv(V_tab.reset_index(), "2_V_table", SUBDIR)


# ================================================================
# 3) Boundary test (extended grid)
# ================================================================
def diag_boundary(ctx):
    dl_ctx = ctx["dl_ctx"]
    lambda_ext = [0.30, 0.90, 1.50, 2.00, 2.50, 3.00]
    m_ext = [0.0, 0.5, 1.0]
    print(f"  corriendo grilla extendida {len(lambda_ext)}x{len(m_ext)}="
          f"{len(lambda_ext) * len(m_ext)} solves...")
    V_df_ext, _ = run_regret_grid(dl_ctx, lambda_ext, m_ext)
    res_ext = compute_regret_and_select(V_df_ext)

    R_tab = res_ext["R_table"]
    mean_R = R_tab.mean(axis=1).unstack("m").sort_index().sort_index(axis=1)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(mean_R.values, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(len(mean_R.columns)))
    ax.set_xticklabels([f"{m:.1f}" for m in mean_R.columns])
    ax.set_yticks(range(len(mean_R.index)))
    ax.set_yticklabels([f"{l:.2f}" for l in mean_R.index])
    ax.set_xlabel("m"); ax.set_ylabel("lambda")
    lam_m, m_m = res_ext["g_mean"]
    lam_w, m_w = res_ext["g_worst"]
    ax.set_title(f"mean(R) en grilla extendida\n"
                 f"g*_mean=({lam_m:.2f}, {m_m:.1f})   "
                 f"g*_worst=({lam_w:.2f}, {m_w:.1f})")
    for i in range(mean_R.shape[0]):
        for j in range(mean_R.shape[1]):
            ax.text(j, i, f"${mean_R.values[i, j]:,.0f}",
                    ha="center", va="center", fontsize=8)
    # Marcar el optimo
    j_opt = list(mean_R.columns).index(m_m)
    i_opt = list(mean_R.index).index(lam_m)
    ax.scatter([j_opt], [i_opt], s=200, marker="*", color="cyan",
               edgecolor="k", linewidth=1.5, zorder=3, label="g*_mean")
    ax.legend(loc="upper right")
    save_fig(fig, "3_boundary_extended", SUBDIR)

    rows = [{"lambda": float(lam_m), "m": float(m_m), "tipo": "g_mean_extendido",
             "regret": float(res_ext["g_mean_metric"])},
            {"lambda": float(lam_w), "m": float(m_w), "tipo": "g_worst_extendido",
             "regret": float(res_ext["g_worst_metric"])}]
    lam_m_std, m_m_std = ctx["res_dl"]["g_mean"]
    lam_w_std, m_w_std = ctx["res_dl"]["g_worst"]
    rows += [{"lambda": float(lam_m_std), "m": float(m_m_std), "tipo": "g_mean_estandar",
              "regret": float(ctx["res_dl"]["g_mean_metric"])},
             {"lambda": float(lam_w_std), "m": float(m_w_std), "tipo": "g_worst_estandar",
              "regret": float(ctx["res_dl"]["g_worst_metric"])}]
    save_csv(pd.DataFrame(rows), "3_boundary_seleccion", SUBDIR)
    save_csv(V_df_ext, "3_boundary_V_long", SUBDIR)


# ================================================================
# 4) Politicas w(t)
# ================================================================
def diag_politicas(ctx):
    policies = ctx["policies_dl"]
    dl_ctx = ctx["dl_ctx"]
    assets = dl_ctx["assets"]
    T_vals = dl_ctx["T_vals"]
    lams = sorted({lam for (lam, _) in policies})
    ms = sorted({m for (_, m) in policies})
    # Seleccion: las 4 esquinas + g*_mean
    lam_m, m_m = ctx["res_dl"]["g_mean"]
    selected = {
        (lams[0],  ms[0]):  "low_lambda_low_m",
        (lams[0],  ms[-1]): "low_lambda_high_m",
        (lams[-1], ms[0]):  "high_lambda_low_m",
        (lams[-1], ms[-1]): "high_lambda_high_m",
        (lam_m,    m_m):    "g_mean",
    }
    fig, axes = plt.subplots(len(assets), 1, figsize=(13, 4.5 * len(assets)))
    if len(assets) == 1:
        axes = [axes]

    rows = []
    colors = plt.cm.tab10(np.linspace(0, 1, len(selected)))
    for ai, a in enumerate(assets):
        ax = axes[ai]
        for idx, ((lam, m_), label) in enumerate(selected.items()):
            if (lam, m_) not in policies:
                continue
            w_sol, _u, _v, _z = policies[(lam, m_)]
            w_t = [w_sol[a, t] for t in T_vals]
            ax.plot(T_vals, w_t, color=colors[idx], lw=1.3,
                    label=f"{label} (lam={lam:.2f}, m={m_:.1f})")
            for t, w in zip(T_vals, w_t):
                rows.append({"asset": a, "lambda": float(lam), "m": float(m_),
                             "label": label, "t": t, "w": float(w)})
        ax.axhline(0.5, color="grey", lw=0.5)
        ax.set_title(f"{a} - trayectoria w(t)")
        ax.set_xlabel("t"); ax.set_ylabel(f"w_{a}(t)")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8, loc="best")
    save_fig(fig, "4_politicas_w", SUBDIR)
    save_csv(pd.DataFrame(rows), "4_politicas_w", SUBDIR)


# ================================================================
# 5) Turnover por g
# ================================================================
def diag_turnover(ctx):
    policies = ctx["policies_dl"]
    dl_ctx = ctx["dl_ctx"]
    assets = dl_ctx["assets"]
    T_vals = dl_ctx["T_vals"]

    rows = []
    for (lam, m_), (w_sol, u_sol, v_sol, _z) in policies.items():
        # Total turnover sumando u+v sobre t y activos
        total_uv = sum(u_sol[a, t] + v_sol[a, t]
                       for a in assets for t in T_vals)
        # Tambien turnover por activo
        per_a = {a: sum(u_sol[a, t] + v_sol[a, t] for t in T_vals)
                 for a in assets}
        rows.append({
            "lambda": float(lam), "m": float(m_),
            "turnover_total": float(total_uv),
            **{f"turnover_{a}": float(v) for a, v in per_a.items()},
        })
    df = pd.DataFrame(rows)
    pivot_total = df.pivot_table(index="lambda", columns="m",
                                  values="turnover_total")

    fig, axes = plt.subplots(1, 1 + len(assets), figsize=(14, 4.5))
    if len(assets) == 0:
        axes = [axes]
    ax = axes[0]
    im = ax.imshow(pivot_total.values, cmap="magma", aspect="auto")
    ax.set_xticks(range(len(pivot_total.columns)))
    ax.set_xticklabels([f"{m:.1f}" for m in pivot_total.columns])
    ax.set_yticks(range(len(pivot_total.index)))
    ax.set_yticklabels([f"{l:.2f}" for l in pivot_total.index])
    ax.set_xlabel("m"); ax.set_ylabel("lambda")
    ax.set_title("turnover total = Σ_t Σ_i (u + v)")
    for i in range(pivot_total.shape[0]):
        for j in range(pivot_total.shape[1]):
            ax.text(j, i, f"{pivot_total.values[i, j]:.3f}",
                    ha="center", va="center", fontsize=8, color="white")

    for ai, a in enumerate(assets):
        pivot_a = df.pivot_table(index="lambda", columns="m",
                                   values=f"turnover_{a}")
        ax = axes[1 + ai]
        ax.imshow(pivot_a.values, cmap="magma", aspect="auto")
        ax.set_xticks(range(len(pivot_a.columns)))
        ax.set_xticklabels([f"{m:.1f}" for m in pivot_a.columns])
        ax.set_yticks(range(len(pivot_a.index)))
        ax.set_yticklabels([f"{l:.2f}" for l in pivot_a.index])
        ax.set_xlabel("m"); ax.set_ylabel("lambda")
        ax.set_title(f"turnover {a}")
        for i in range(pivot_a.shape[0]):
            for j in range(pivot_a.shape[1]):
                ax.text(j, i, f"{pivot_a.values[i, j]:.3f}",
                        ha="center", va="center", fontsize=8, color="white")
    save_fig(fig, "5_turnover", SUBDIR)
    save_csv(df, "5_turnover", SUBDIR)


# ================================================================
# 6) DL vs OPT base sobre los MISMOS escenarios
# ================================================================
def diag_dl_vs_optbase(ctx):
    dl_ctx = ctx["dl_ctx"]
    opt_base_ctx = ctx["opt_base_ctx"]
    lambda_grid = ctx["lambda_grid"]
    m_grid = ctx["m_grid"]

    # Contexto hibrido: mu/sigma del historico, escenarios del DL.
    hybrid_ctx = dict(opt_base_ctx)
    hybrid_ctx["scenarios"] = dl_ctx["scenarios"]

    print(f"  corriendo grilla {len(lambda_grid)}x{len(m_grid)} "
          f"con mu/sigma HISTORICO + escenarios DL...")
    V_df_base, _policies_base = run_regret_grid(
        hybrid_ctx, lambda_grid, m_grid,
    )
    res_base = compute_regret_and_select(V_df_base)

    res_dl = ctx["res_dl"]
    C0 = dl_ctx["Capital_inicial"]

    # Comparacion en mismo escenario: V_DL[g*_mean_DL, s] vs V_BASE[g*_mean_BASE, s]
    lam_dl, m_dl = res_dl["g_mean"]
    lam_b, m_b = res_base["g_mean"]
    V_dl_row = res_dl["V_table"].loc[(lam_dl, m_dl)]
    V_base_row = res_base["V_table"].loc[(lam_b, m_b)]

    n_S = len(V_dl_row)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(n_S)
    width = 0.4
    ax.bar(x - width / 2, V_dl_row.values, width, color="C0",
           label=f"DL pipeline g*=({lam_dl:.2f}, {m_dl:.1f}) — mean=${V_dl_row.mean():,.0f}")
    ax.bar(x + width / 2, V_base_row.values, width, color="C2",
           label=f"OPT base g*=({lam_b:.2f}, {m_b:.1f}) — mean=${V_base_row.mean():,.0f}")
    ax.axhline(C0, color="grey", ls="--", lw=1, label=f"C0=${C0:,.0f}")
    ax.set_xticks(x); ax.set_xticklabels([f"s={s}" for s in range(n_S)])
    ax.set_ylabel("V terminal")
    ax.set_title("Capital terminal por escenario: pipeline DL vs OPT base "
                 "(mismos 5 escenarios DL)")
    ax.legend(fontsize=8)
    save_fig(fig, "6_dl_vs_optbase", SUBDIR)

    rows = []
    for s in range(n_S):
        rows.append({
            "s": s,
            "V_dl":   float(V_dl_row.iloc[s]),
            "V_base": float(V_base_row.iloc[s]),
            "delta":  float(V_base_row.iloc[s] - V_dl_row.iloc[s]),
            "ret_dl_%":   float((V_dl_row.iloc[s] / C0 - 1) * 100),
            "ret_base_%": float((V_base_row.iloc[s] / C0 - 1) * 100),
        })
    rows.append({"s": "mean",
                 "V_dl":   float(V_dl_row.mean()),
                 "V_base": float(V_base_row.mean()),
                 "delta":  float(V_base_row.mean() - V_dl_row.mean()),
                 "ret_dl_%":   float((V_dl_row.mean() / C0 - 1) * 100),
                 "ret_base_%": float((V_base_row.mean() / C0 - 1) * 100)})
    rows.append({"s": "worst",
                 "V_dl":   float(V_dl_row.min()),
                 "V_base": float(V_base_row.min()),
                 "delta":  float(V_base_row.min() - V_dl_row.min()),
                 "ret_dl_%":   float((V_dl_row.min() / C0 - 1) * 100),
                 "ret_base_%": float((V_base_row.min() / C0 - 1) * 100)})
    save_csv(pd.DataFrame(rows), "6_dl_vs_optbase", SUBDIR)
    save_csv(V_df_base, "6_dl_vs_optbase_V_long", SUBDIR)


# ================================================================
# Main
# ================================================================
def main():
    print("=" * 70)
    print("INSPECCION DEL OPTIMIZADOR + REGRET-GRID")
    print("=" * 70)
    print(f"  lambda_grid : {list(LAMBDA_GRID)}")
    print(f"  m_grid      : {list(M_GRID)}")
    print(f"  T_HORIZON   : {T_HORIZON}")
    print("-" * 70)
    ctx = build_context()
    print(f"  V_df_dl shape : {ctx['V_df_dl'].shape}")
    print(f"  policies      : {len(ctx['policies_dl'])} entries")
    print(f"  g*_mean DL    : {ctx['res_dl']['g_mean']}")
    print(f"  g*_worst DL   : {ctx['res_dl']['g_worst']}")
    print(f"  output dir    : {out_dir(SUBDIR)}")
    print("-" * 70)

    print("[1/6] regret heatmap")
    diag_regret_heatmap(ctx)
    print("[2/6] V heatmap")
    diag_V_heatmap(ctx)
    print("[3/6] boundary (grilla extendida)")
    diag_boundary(ctx)
    print("[4/6] politicas w(t)")
    diag_politicas(ctx)
    print("[5/6] turnover por g")
    diag_turnover(ctx)
    print("[6/6] DL vs OPT base sobre mismos escenarios")
    diag_dl_vs_optbase(ctx)

    print("-" * 70)
    print(f"Listo. Resultados en {out_dir(SUBDIR)}")


if __name__ == "__main__":
    main()
