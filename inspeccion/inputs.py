"""Inspecciona lo que LLEGA al optimizador y verifica su coherencia.

`solve_portfolio` recibe un `context` con mu_mix(i, t), sigma_mix(i, j, t),
c_base, w0, V_max y (para la simulacion ex-post) escenarios. Este modulo
compara los inputs del pipeline DL con los del OPT base donde aplica, y
chequea propiedades matematicas + consistencia entre el input ex-ante
(mu_mix) y el input ex-post (escenarios).

Corre con:
    python -m inspeccion.inputs
    (o)
    python inspeccion/inputs.py

Diagnosticos (1 PNG + 1 CSV cada uno) en `inspeccion/inputs_out/`:

  1. mu_mix_serie    serie temporal de mu_mix(i, t), DL vs OPT base.
                     Detecta sesgo sistematico del DL.
  2. sigma_serie     varianza diagonal y correlacion cruzada de sigma_mix(t),
                     DL vs OPT base. La correlacion cross-asset es decisiva
                     para la diversificacion.
  3. mu_vs_realizado mu_mix DL en cada t comparado con el retorno realizado
                     en el mismo t. Sanity check: ¿el optimizador esta viendo
                     algo cercano a la realidad?
  4. psd_check       autovalores de sigma_mix(t) a cada t. Si algun
                     autovalor es negativo, la matriz NO es PSD y el FO
                     cuadratico esta mal planteado.
  5. risk_return     scatter (sigma_ii, mu_i) por t y por activo. Muestra
                     el "menu" semanal que el FO esta optimizando. Si un
                     activo domina (alto mu y baja sigma) el optimizador
                     no tiene eleccion real.
  6. coherencia      consistencia entre mu_mix (ex-ante) y mean de los
                     escenarios DL (ex-post). Si difieren mucho, el FO y
                     la simulacion viven en mundos distintos.
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
    SCENARIO_SEED,
    SUMMARY_ASSET,
    T_HORIZON,
)
from Regret_Grid import build_dl_context, load_market_data


SUBDIR = "inputs"


def build_context():
    print("Cargando contexto DL...")
    dl_ctx = build_dl_context(
        data_dir=DATA_DIR, checkpoint_path=CHECKPOINT_PATH,
        T=T_HORIZON, N_candidates=N_CANDIDATES,
        n_scenarios=N_SCENARIOS, seed=SCENARIO_SEED,
        summary_asset=SUMMARY_ASSET,
    )
    print("Cargando contexto OPT base (historico)...")
    opt_ctx = load_market_data(str(DATA_DIR))
    return {"dl": dl_ctx, "opt": opt_ctx}


def _series_to_arr(series_dict, asset, T_vals):
    """Extrae mu_mix[asset] alineado a T_vals como np.array."""
    s = series_dict[asset]
    return np.asarray([s.loc[t] for t in T_vals], dtype=np.float64)


# ================================================================
# 1) mu_mix(t) serie temporal: DL vs OPT base
# ================================================================
def diag_mu_mix_serie(ctx):
    dl, opt = ctx["dl"], ctx["opt"]
    assets = dl["assets"]
    T_vals = dl["T_vals"]
    fig, axes = plt.subplots(len(assets), 1, figsize=(13, 4 * len(assets)))
    if len(assets) == 1:
        axes = [axes]
    rows = []
    for ai, a in enumerate(assets):
        mu_dl  = _series_to_arr(dl["mu_mix"],  a, T_vals)
        mu_opt = _series_to_arr(opt["mu_mix"], a, T_vals)
        ax = axes[ai]
        ax.plot(T_vals, mu_dl,  color="C0", lw=1.1,
                label=f"DL (mean={mu_dl.mean():+.4f})")
        ax.plot(T_vals, mu_opt, color="C2", lw=1.1,
                label=f"OPT base (mean={mu_opt.mean():+.4f})")
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_title(f"{a} - mu_mix(t) por pipeline")
        ax.set_xlabel("t"); ax.set_ylabel("mu_mix")
        ax.legend()
        rows.append({"asset": a, "src": "DL",
                     "mean": float(mu_dl.mean()),
                     "std":  float(mu_dl.std()),
                     "min":  float(mu_dl.min()),
                     "max":  float(mu_dl.max())})
        rows.append({"asset": a, "src": "OPT_base",
                     "mean": float(mu_opt.mean()),
                     "std":  float(mu_opt.std()),
                     "min":  float(mu_opt.min()),
                     "max":  float(mu_opt.max())})
    save_fig(fig, "1_mu_mix_serie", SUBDIR)
    save_csv(pd.DataFrame(rows), "1_mu_mix_serie", SUBDIR)


# ================================================================
# 2) sigma_mix: varianza diagonal y correlacion cross-asset
# ================================================================
def diag_sigma_serie(ctx):
    dl, opt = ctx["dl"], ctx["opt"]
    assets = dl["assets"]
    T_vals = dl["T_vals"]
    A = len(assets)

    def diag_corr(sig_mix):
        var = {a: _series_to_arr(sig_mix[a],         a, T_vals) for a in assets}
        if A < 2:
            return var, None
        cov = _series_to_arr(sig_mix[assets[0]], assets[1], T_vals)
        denom = np.sqrt(var[assets[0]] * var[assets[1]])
        denom = np.where(denom <= 0, np.nan, denom)
        corr = cov / denom
        return var, corr

    var_dl,  corr_dl  = diag_corr(dl["sigma_mix"])
    var_opt, corr_opt = diag_corr(opt["sigma_mix"])

    fig, axes = plt.subplots(2, 1, figsize=(13, 8))
    ax = axes[0]
    for ai, a in enumerate(assets):
        c = f"C{ai}"
        ax.plot(T_vals, var_dl[a],  color=c, lw=1.1,
                label=f"{a} DL (mean={var_dl[a].mean():.5f})")
        ax.plot(T_vals, var_opt[a], color=c, lw=1.1, ls="--",
                label=f"{a} OPT (mean={var_opt[a].mean():.5f})")
    ax.set_title("sigma_mix(i, i, t) - varianza diagonal")
    ax.set_xlabel("t"); ax.set_ylabel("var")
    ax.legend(fontsize=8)

    if corr_dl is not None:
        ax = axes[1]
        ax.plot(T_vals, corr_dl,  color="C0", lw=1.2,
                label=f"DL (mean={np.nanmean(corr_dl):+.3f})")
        ax.plot(T_vals, corr_opt, color="C2", lw=1.2,
                label=f"OPT base (mean={np.nanmean(corr_opt):+.3f})")
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_title(f"corr({assets[0]}, {assets[1]})(t) implicita en sigma_mix")
        ax.set_xlabel("t"); ax.set_ylabel("corr")
        ax.set_ylim(-1.05, 1.05)
        ax.legend()
    save_fig(fig, "2_sigma_serie", SUBDIR)

    rows = []
    for a in assets:
        rows.append({"src": "DL",  "metric": f"var_{a}",
                     "mean": float(var_dl[a].mean()),
                     "min":  float(var_dl[a].min()),
                     "max":  float(var_dl[a].max())})
        rows.append({"src": "OPT", "metric": f"var_{a}",
                     "mean": float(var_opt[a].mean()),
                     "min":  float(var_opt[a].min()),
                     "max":  float(var_opt[a].max())})
    if corr_dl is not None:
        rows.append({"src": "DL",  "metric": "corr_cross",
                     "mean": float(np.nanmean(corr_dl)),
                     "min":  float(np.nanmin(corr_dl)),
                     "max":  float(np.nanmax(corr_dl))})
        rows.append({"src": "OPT", "metric": "corr_cross",
                     "mean": float(np.nanmean(corr_opt)),
                     "min":  float(np.nanmin(corr_opt)),
                     "max":  float(np.nanmax(corr_opt))})
    save_csv(pd.DataFrame(rows), "2_sigma_serie", SUBDIR)


# ================================================================
# 3) mu_mix DL vs retorno realizado en el mismo t
# ================================================================
def diag_mu_vs_realizado(ctx):
    dl = ctx["dl"]
    assets = dl["assets"]
    T_vals = dl["T_vals"]
    r_hist = dl["r"]

    fig, axes = plt.subplots(len(assets), 1, figsize=(13, 4 * len(assets)))
    if len(assets) == 1:
        axes = [axes]
    rows = []
    for ai, a in enumerate(assets):
        mu_dl = _series_to_arr(dl["mu_mix"], a, T_vals)
        realiz = np.asarray([r_hist[a].loc[t] for t in T_vals], dtype=np.float64)
        ax = axes[ai]
        ax.plot(T_vals, realiz, color="k", lw=0.9, alpha=0.7,
                label=f"realizado (mean={realiz.mean():+.4f})")
        ax.plot(T_vals, mu_dl, color="C0", lw=1.4,
                label=f"mu_mix DL (mean={mu_dl.mean():+.4f})")
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_title(f"{a} - mu_mix DL vs retorno realizado")
        ax.set_xlabel("t"); ax.set_ylabel("retorno")
        ax.legend()
        bias = mu_dl.mean() - realiz.mean()
        hit_rate = float(((mu_dl > 0) == (realiz > 0)).mean())
        rows.append({
            "asset": a,
            "mu_dl_mean":     float(mu_dl.mean()),
            "realiz_mean":    float(realiz.mean()),
            "bias":           float(bias),
            "hit_rate_signo": hit_rate,
            "corr_mu_realiz": float(np.corrcoef(mu_dl, realiz)[0, 1]),
        })
    save_fig(fig, "3_mu_vs_realizado", SUBDIR)
    save_csv(pd.DataFrame(rows), "3_mu_vs_realizado", SUBDIR)


# ================================================================
# 4) PSD check: autovalores de sigma_mix(t)
# ================================================================
def diag_psd_check(ctx):
    dl, opt = ctx["dl"], ctx["opt"]
    assets = dl["assets"]
    T_vals = dl["T_vals"]
    A = len(assets)
    T = len(T_vals)

    def eigvals(sig_mix):
        ev = np.empty((T, A))
        for k, t in enumerate(T_vals):
            S = np.empty((A, A))
            for ai, ai_name in enumerate(assets):
                for aj, aj_name in enumerate(assets):
                    S[ai, aj] = sig_mix[ai_name][aj_name].loc[t]
            # Simetrizar por robustez
            S = 0.5 * (S + S.T)
            ev[k] = np.sort(np.linalg.eigvalsh(S))
        return ev

    ev_dl  = eigvals(dl["sigma_mix"])
    ev_opt = eigvals(opt["sigma_mix"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for axi, (ev, name) in enumerate([(ev_dl, "DL"), (ev_opt, "OPT base")]):
        ax = axes[axi]
        for k in range(A):
            ax.plot(T_vals, ev[:, k], lw=1, label=f"lambda_{k+1}(t)")
        ax.axhline(0, color="red", lw=0.7, ls="--")
        ax.set_title(f"sigma_mix(t) - autovalores ({name})\n"
                     f"min_t lambda_min = {ev[:, 0].min():.2e}")
        ax.set_xlabel("t"); ax.set_ylabel("autovalor")
        ax.legend()
    save_fig(fig, "4_psd_check", SUBDIR)

    rows = []
    for name, ev in [("DL", ev_dl), ("OPT", ev_opt)]:
        rows.append({"src": name,
                     "min_eigenvalue_global": float(ev.min()),
                     "frac_t_negative":       float((ev.min(axis=1) < 0).mean()),
                     "min_t_lambda_min":      float(ev[:, 0].min()),
                     "max_t_lambda_max":      float(ev[:, -1].max())})
    save_csv(pd.DataFrame(rows), "4_psd_check", SUBDIR)


# ================================================================
# 5) Risk-return scatter: el "menu" del FO
# ================================================================
def diag_risk_return(ctx):
    dl, opt = ctx["dl"], ctx["opt"]
    assets = dl["assets"]
    T_vals = dl["T_vals"]
    A = len(assets)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    rows = []
    for axi, (name, ctx_) in enumerate([("DL", dl), ("OPT base", opt)]):
        ax = axes[axi]
        for ai, a in enumerate(assets):
            mu = _series_to_arr(ctx_["mu_mix"], a, T_vals)
            var = _series_to_arr(ctx_["sigma_mix"][a], a, T_vals)
            sig = np.sqrt(np.maximum(var, 0))
            ax.scatter(sig, mu, s=12, alpha=0.5,
                       label=f"{a} (mu={mu.mean():+.4f}, sigma={sig.mean():.4f})")
            rows.append({
                "src": name, "asset": a,
                "sigma_mean": float(sig.mean()),
                "mu_mean":    float(mu.mean()),
                "sharpe_imp": float(mu.mean() / sig.mean()) if sig.mean() > 0 else 0.0,
            })
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_title(f"{name} - menu semanal (mu, sigma) por activo y t")
        ax.set_xlabel("sigma_t (vol semanal)")
        ax.set_ylabel("mu_t")
        ax.legend(fontsize=9)
    save_fig(fig, "5_risk_return", SUBDIR)
    save_csv(pd.DataFrame(rows), "5_risk_return", SUBDIR)


# ================================================================
# 6) Coherencia: ex-ante (mu_mix) vs ex-post (mean scenarios)
# ================================================================
def diag_coherencia(ctx):
    dl = ctx["dl"]
    opt = ctx["opt"]
    assets = dl["assets"]
    T_vals = dl["T_vals"]
    scenarios = dl["scenarios"]                # (S, T, A)
    scen_mean = scenarios.mean(axis=0)         # (T, A)

    fig, axes = plt.subplots(len(assets), 1, figsize=(13, 4 * len(assets)))
    if len(assets) == 1:
        axes = [axes]
    rows = []
    for ai, a in enumerate(assets):
        mu_dl = _series_to_arr(dl["mu_mix"], a, T_vals)
        scn_m = scen_mean[:, ai]
        ax = axes[ai]
        ax.plot(T_vals, mu_dl, color="C0", lw=1.4,
                label=f"mu_mix DL (mean={mu_dl.mean():+.4f})")
        ax.plot(T_vals, scn_m, color="C1", lw=1.2, ls="--",
                label=f"mean escenarios (mean={scn_m.mean():+.4f})")
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_title(f"{a} - input ex-ante (mu_mix) vs input ex-post "
                     f"(mean de {scenarios.shape[0]} escenarios)")
        ax.set_xlabel("t"); ax.set_ylabel("retorno")
        ax.legend()
        rows.append({
            "asset": a,
            "mu_mix_dl_mean":     float(mu_dl.mean()),
            "scenarios_mean":     float(scn_m.mean()),
            "gap_ex_ante_ex_post": float(scn_m.mean() - mu_dl.mean()),
            "corr_mu_scen":       float(np.corrcoef(mu_dl, scn_m)[0, 1])
                                  if scn_m.std() > 1e-12 else 0.0,
        })
    save_fig(fig, "6_coherencia_ex_ante_ex_post", SUBDIR)

    # Resumen de constantes del context
    const_rows = [
        {"campo": "V_max",            "valor": float(dl["V_max"])},
        {"campo": "Capital_inicial",  "valor": float(dl["Capital_inicial"])},
    ]
    for a in assets:
        const_rows.append({"campo": f"c_base[{a}]", "valor": float(dl["c_base"][a])})
        const_rows.append({"campo": f"w0[{a}]",     "valor": float(dl["w0"][a])})
    const_rows.append({"campo": "sum(w0)",
                       "valor": float(sum(dl["w0"][a] for a in assets))})

    save_csv(pd.DataFrame(rows), "6_coherencia_ex_ante_ex_post", SUBDIR)
    save_csv(pd.DataFrame(const_rows), "6_constantes_del_context", SUBDIR)


# ================================================================
# Main
# ================================================================
def main():
    print("=" * 70)
    print("INSPECCION DE INPUTS AL OPTIMIZADOR")
    print("=" * 70)
    ctx = build_context()
    dl = ctx["dl"]
    print(f"  assets        : {dl['assets']}")
    print(f"  T_vals        : t1..t{len(dl['T_vals'])}")
    print(f"  V_max         : {dl['V_max']}")
    print(f"  Capital_inic  : {dl['Capital_inicial']}")
    print(f"  c_base        : {dict(dl['c_base'])}")
    print(f"  w0            : {dict(dl['w0'])}")
    print(f"  scenarios     : {dl['scenarios'].shape}")
    print(f"  output        : {out_dir(SUBDIR)}")
    print("-" * 70)

    print("[1/6] mu_mix(t) serie: DL vs OPT base")
    diag_mu_mix_serie(ctx)
    print("[2/6] sigma_mix(t) varianza + correlacion cross-asset")
    diag_sigma_serie(ctx)
    print("[3/6] mu_mix DL vs retorno realizado")
    diag_mu_vs_realizado(ctx)
    print("[4/6] PSD check de sigma_mix(t)")
    diag_psd_check(ctx)
    print("[5/6] menu risk-return")
    diag_risk_return(ctx)
    print("[6/6] coherencia ex-ante vs ex-post")
    diag_coherencia(ctx)

    print("-" * 70)
    print(f"Listo. Resultados en {out_dir(SUBDIR)}")


if __name__ == "__main__":
    main()
