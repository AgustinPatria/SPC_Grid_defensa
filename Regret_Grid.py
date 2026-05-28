"""
Regret-grid: seleccion de parametros (lambda, m) del portafolio usando
escenarios generados por el pipeline DL (PDF seccion 3).

Pipeline NUEVO (per-cell NN):
  1. Entrena UNA NN por celda g = (lambda, m) con seed deterministica
     (train_per_cell_nns).
  2. Para cada celda: construye su propio mu_mix/sigma_mix usando su NN +
     walking + p_hist (build_per_cell_context).
  3. ESCENARIOS COMPARTIDOS: ensemble de las 15 NNs (build_ensemble_model)
     genera N candidatos; se reducen a 5 representativos rankeando por
     retorno de portafolio bajo w_ref = w0 (build_shared_scenarios).
  4. Para cada g: resuelve solve_portfolio con su contexto -> w_g.
  5. Para cada (g, s): simula capital -> V[g, s] sobre los escenarios
     COMPARTIDOS (ec. 19). Asi V_best_s esta bien definido.
  6. Regret R[g, s] = V_best_s - V[g, s] (ec. 22), seleccion g*_mean,
     g*_worst (ec. 23-24).

Pipeline LEGACY (1 NN compartida): build_dl_context + run_regret_grid
sigue disponible para los scripts de inspeccion.
"""
import hashlib
from pathlib import Path
from typing import Dict, Sequence, Tuple

import gamspy as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from config import (
    ASSETS,
    CAPITAL_INICIAL,
    C_BASE,
    CHECKPOINT_PATH,
    DATA_DIR,
    DLConfig,
    LAMBDA_GRID,
    LAMBDA_RIESGO_DEFAULT,
    M_GRID,
    MODELS_DIR,
    N_CANDIDATES,
    N_SCENARIOS,
    PROB_CSV,
    REGIMES,
    RESULTS_DIR,
    RETURN_COL,
    RETURN_CSV,
    SCENARIO_SEED,
    SOLVER,
    SPLIT,
    SUMMARY_ASSET,
    T_HORIZON,
    V_MAX_BUFFER,
    V_MAX_REF_ASSET,
    W0,
)
from dl.generador_escenarios import (
    generate_candidate_scenarios,
    generate_representative_scenarios,
    reduce_by_fo_outcome,
    reduce_by_portfolio_return,
    reduce_to_representatives,
)
from dl.prediccion_deciles import (
    LoadedModel,
    load_checkpoint,
    save_checkpoint,
    train_deciles,
)
from dl.regimen_predicted import regimen_from_deciles


# ================================================================
# 0) Carga de datos historicos + momentos por regimen (Opcion B del PDF)
# ================================================================

def load_market_data(base_dir_str: str | Path | None = None,
                     moments_window=None):
    """
    Carga los CSVs, procesa probabilidades y retornos,
    y calcula momentos mezclados (mu_mix, sigma_mix).
    Replica exactamente la Seccion 2 del GAMS (Opcion B del PDF).

    Nombres de archivos, activos, regimenes, w0, c_base y Capital_inicial
    se leen de `config.py`.

    moments_window: tupla (t_start, t_end) inclusive sobre el indice t de los
        CSVs. Si None (default) estima mu_hat y sigma_hat sobre toda la serie
        — es lo que hereda OPT base del GAMS original. Si se pasa, restringe
        la estimacion al rango indicado (p.ej. solo TRAIN del LSTM para una
        comparacion sin leakage de info futura).
    """
    BASE_DIR = Path(base_dir_str) if base_dir_str is not None else DATA_DIR

    assets  = list(ASSETS)
    regimes = list(REGIMES)

    prob = {}
    ret  = {}
    for a in assets:
        df_p = pd.read_csv(BASE_DIR / PROB_CSV[a], sep=",")
        df_r = pd.read_csv(BASE_DIR / RETURN_CSV[a], sep=",")
        df_p.columns = [c.strip() for c in df_p.columns]
        df_r.columns = [c.strip() for c in df_r.columns]
        # Acepta ambos formatos del CSV: "t1, t2, ..." (string con prefijo)
        # o "1, 2, ..." (entero literal). Strip del prefijo "t" si existe.
        df_p["t"] = df_p["t"].astype(str).str.replace(r"^t", "", regex=True).astype(int)
        df_r["t"] = df_r["t"].astype(str).str.replace(r"^t", "", regex=True).astype(int)
        prob[a] = df_p
        ret[a]  = df_r

    T_vals = sorted(prob[assets[0]]["t"].unique())

    r = {a: ret[a].set_index("t")[RETURN_COL[a]] for a in assets}
    p = {a: prob[a].set_index("t")[regimes]      for a in assets}

    def _slice_mom(s):
        if moments_window is None:
            return s
        a, b = moments_window
        return s.loc[a:b]

    mu_hat    = {}
    sigma_hat = {}

    for i in assets:
        for k in regimes:
            p_w = _slice_mom(p[i][k])
            r_w = _slice_mom(r[i])
            den = p_w.sum()
            mu_hat[(i, k)] = ((p_w * r_w).sum() / den) if den > 0 else 0.0

    for i in assets:
        for j in assets:
            for k in regimes:
                pi_w = _slice_mom(p[i][k])
                pj_w = _slice_mom(p[j][k])
                ri_w = _slice_mom(r[i])
                rj_w = _slice_mom(r[j])
                den  = (pi_w * pj_w).sum()
                if den > 0:
                    term = pi_w * pj_w * (ri_w - mu_hat[(i, k)]) * (rj_w - mu_hat[(j, k)])
                    sigma_hat[(i, j, k)] = term.sum() / den
                else:
                    sigma_hat[(i, j, k)] = 0.0

    mu_mix    = {i: pd.Series(0.0, index=T_vals) for i in assets}
    sigma_mix = {i: {j: pd.Series(0.0, index=T_vals) for j in assets} for i in assets}

    for i in assets:
        for k in regimes:
            mu_mix[i] += p[i][k] * mu_hat[(i, k)]

    for i in assets:
        for j in assets:
            for k in regimes:
                sigma_mix[i][j] += p[i][k] * p[j][k] * sigma_hat[(i, j, k)]

    for i in assets:
        for j in assets:
            sym = 0.5 * (sigma_mix[i][j] + sigma_mix[j][i])
            sigma_mix[i][j] = sym
            sigma_mix[j][i] = sym

    # Presupuesto de riesgo V_max: varianza muestral del activo de referencia
    # (el "estable") escalada por V_MAX_BUFFER. Entra en la FO como penalizacion
    # lambda*(Riesgo - V_max). Es una constante => no altera el optimo w*, solo
    # desplaza el valor de z. (ddof=1, sample variance, igual que pandas default).
    # Respeta moments_window para no filtrar varianza del test set en OOS.
    V_max = float(_slice_mom(r[V_MAX_REF_ASSET]).var() * V_MAX_BUFFER)

    return {
        "mu_mix":          mu_mix,
        "sigma_mix":       sigma_mix,
        "T_vals":          T_vals,
        "nT":              len(T_vals),
        "assets":          assets,
        "c_base":          dict(C_BASE),
        "w0":              dict(W0),
        "r":               r,
        "p_hist":          p,
        "Capital_inicial": CAPITAL_INICIAL,
        "V_max":           V_max,
    }


# ================================================================
# 0.b) Solucionador GAMSPy + IPOPT
# ================================================================

def solve_portfolio(context: dict,
                    lambda_riesgo: float = LAMBDA_RIESGO_DEFAULT,
                    costo_mult:    float = 1.0,
                    verbose:       bool  = False):
    """
    Resuelve el modelo media-varianza con presupuesto de riesgo V_max y costos,
    usando GAMSPy + IPOPT.

    max z = sum_t [ sum_i w(i,t)*mu_mix(i,t)
                  - lambda * ( sum_(i,j) w(i,t)*w(j,t)*sigma_mix(i,j,t) - V_max )
                  - sum_i c_base(i)*costo_mult*(u(i,t)+v(i,t)) ]

    s.t.  sum_i w(i,t) = 1                           para todo t
          w(i,t) - w(i,t-1) = u(i,t) - v(i,t)       para t > t1
          w(i,t1) - w0(i)   = u(i,t1) - v(i,t1)     anclaje inicial
          0 <= w(i,t) <= 1;  u(i,t), v(i,t) >= 0

    V_max es una constante (no depende de las decisiones) => no altera el optimo
    w*, solo desplaza el valor de z en -lambda*T*V_max respecto a la version
    sin presupuesto. Se lee de context['V_max'].
    """
    mu_base   = context["mu_mix"]
    sigma_mix = context["sigma_mix"]
    T_vals    = context["T_vals"]
    assets    = context["assets"]
    c_base    = context["c_base"]
    w0_dict   = context["w0"]
    V_max     = float(context["V_max"])

    T_labels = [f"t{n}" for n in T_vals]   # "t1" .. "t163"

    m = gp.Container()

    i_set = gp.Set(m, "i", records=assets,           description="activos")
    j_set = gp.Alias(m, "j", i_set)
    t_set = gp.Set(m, "t", records=T_labels,          description="periodos")

    mu_records = [
        [i, f"t{t}", mu_base[i].loc[t]]
        for i in assets for t in T_vals
    ]
    mu_p = gp.Parameter(
        m, "mu_mix", domain=[i_set, t_set],
        records=pd.DataFrame(mu_records, columns=["i", "t", "value"]),
        description="media mixta por periodo",
    )

    sig_records = [
        [i, j, f"t{t}", sigma_mix[i][j].loc[t]]
        for i in assets for j in assets for t in T_vals
    ]
    sig_p = gp.Parameter(
        m, "sigma_mix", domain=[i_set, j_set, t_set],
        records=pd.DataFrame(sig_records, columns=["i", "j", "t", "value"]),
        description="covarianza mixta por periodo",
    )

    c_base_p = gp.Parameter(
        m, "c_base", domain=[i_set],
        records=pd.DataFrame(
            [[i, c_base[i]] for i in assets],
            columns=["i", "value"],
        ),
        description="costo base de transaccion",
    )

    c_mult_p = gp.Parameter(m, "c_mult", records=costo_mult,
                            description="multiplicador de costo en FO")

    w0_p = gp.Parameter(
        m, "w0", domain=[i_set],
        records=pd.DataFrame(
            [[i, w0_dict[i]] for i in assets],
            columns=["i", "value"],
        ),
        description="portafolio inicial 50/50",
    )

    lam_p = gp.Parameter(m, "lambda_riesgo", records=lambda_riesgo,
                          description="aversion al riesgo")
    v_max_p = gp.Parameter(m, "V_max", records=V_max,
                           description="presupuesto de riesgo")

    z_var = gp.Variable(m, "z",                              description="valor objetivo")
    w_var = gp.Variable(m, "w", domain=[i_set, t_set], type="positive", description="peso")
    u_var = gp.Variable(m, "u", domain=[i_set, t_set], type="positive", description="compras")
    v_var = gp.Variable(m, "v", domain=[i_set, t_set], type="positive", description="ventas")

    w_var.up[i_set, t_set] = 1.0
    # Acotar compras/ventas: en cualquier optimo no-degenerado |w(t)-w(t-1)| <= 1
    # implica u, v <= 1. Sin esta cota, si costo_mult=0 el termino c*(u+v) se anula
    # y IPOPT puede devolver u, v arbitrariamente grandes (pares u_i = v_i = 1e9
    # que satisfacen u-v=Δw). El simulador con c_base real explota a V absurdos.
    u_var.up[i_set, t_set] = 1.0
    v_var.up[i_set, t_set] = 1.0

    fo = gp.Equation(m, "FO_media_var_costo",
                     description="FO: retorno - lambda*(var - V_max) - costos")
    fo[...] = z_var == gp.Sum(
        t_set,
        gp.Sum(i_set, w_var[i_set, t_set] * mu_p[i_set, t_set])
        - lam_p * (gp.Sum((i_set, j_set),
                          w_var[i_set, t_set] * w_var[j_set, t_set] * sig_p[i_set, j_set, t_set])
                   - v_max_p)
        - gp.Sum(i_set, c_base_p[i_set] * c_mult_p * (u_var[i_set, t_set] + v_var[i_set, t_set]))
    )

    norm = gp.Equation(m, "normalizacion_pesos", domain=[t_set],
                       description="suma de pesos = 1")
    norm[t_set] = gp.Sum(i_set, w_var[i_set, t_set]) == 1

    rebal = gp.Equation(m, "rebalanceo_lineal", domain=[i_set, t_set],
                        description="identidad de rebalanceo")
    rebal[i_set, t_set].where[gp.Ord(t_set) > 1] = (
        w_var[i_set, t_set] - w_var[i_set, t_set.lag(1)]
        == u_var[i_set, t_set] - v_var[i_set, t_set]
    )

    anclaje = gp.Equation(m, "anclaje_inicial", domain=[i_set],
                          description="anclaje al portafolio inicial")
    t_anchor = T_labels[0]  # primer periodo del horizonte; "t1" en el caso base.
    anclaje[i_set] = (
        w_var[i_set, t_anchor] - w0_p[i_set]
        == u_var[i_set, t_anchor] - v_var[i_set, t_anchor]
    )

    portfolio = gp.Model(
        m,
        name="PortafolioEstadosCostos",
        equations=m.getEquations(),
        problem="QCP",
        sense=gp.Sense.MAX,
        objective=z_var,
    )

    output = None if not verbose else __import__("sys").stdout
    portfolio.solve(solver=SOLVER, output=output)

    if portfolio.status not in (
        gp.ModelStatus.OptimalLocal,
        gp.ModelStatus.OptimalGlobal,
    ):
        raise RuntimeError(
            f"GAMSPy/IPOPT no encontro solucion optima. Status: {portfolio.status}"
        )

    z_val = float(z_var.toValue())

    def _records_to_dict(var):
        sol = {}
        for _, row in var.records.iterrows():
            i_key = row["i"]
            t_key = int(row["t"][1:])
            sol[i_key, t_key] = float(row["level"])
        return sol

    w_sol = _records_to_dict(w_var)
    u_sol = _records_to_dict(u_var)
    v_sol = _records_to_dict(v_var)

    status = ("optimal" if portfolio.status == gp.ModelStatus.OptimalGlobal
              else "optimal_local")
    return z_val, w_sol, u_sol, v_sol, status


# ================================================================
# 1) DL: walking-window sobre el historico para p_bull(t)
# ================================================================

def predict_pbull_walking(model, returns_history, T):
    """p_bull(t) por ventana real del historico, sin autoalimentar.

    Para cada t en H+1..T se aplica el LSTM a la ventana real
    [r_{t-H}..r_{t-1}] y se deriva p_bull(t) via la ec. 15 del PDF.
    Para t=1..H no hay H retornos previos en el dataset; esas posiciones
    se rellenan por padding con p_bull(H+1) (decision documentada para
    no romper la alineacion temporal con el GAMS base, T_vals=1..T).

    returns_history: (T, A) retornos reales por activo, indexados 0..T-1
                     (returns_history[k] corresponde a la semana t=k+1).
    return:          (T, A) p_bull alineado a t=1..T.
    """
    cfg = model.config
    H   = cfg.H
    A   = cfg.n_assets
    if returns_history.shape != (T, A):
        raise ValueError(
            f"returns_history shape {returns_history.shape} != (T={T}, A={A})"
        )
    if T <= H:
        raise ValueError(f"T={T} debe ser > H={H} para tener al menos una ventana real.")

    p_bull = np.empty((T, A), dtype=np.float32)

    for idx in range(H, T):                          # idx 0-based; semana t1 = idx+1
        window = returns_history[idx - H : idx, :]   # (H, A) ventana real previa
        x = ((window - model.mean) / model.std).astype(np.float32)[None, :, :]
        x_tensor = torch.from_numpy(x)
        with torch.no_grad():
            outs = [net(x_tensor).numpy()[0] for net in model.nets]  # K * (A, Q)
        preds = np.mean(np.stack(outs, axis=0), axis=0)              # (A, Q)
        p_bull_step, _ = regimen_from_deciles(preds)                 # (A,)
        p_bull[idx] = p_bull_step

    # Padding para t=1..H (no hay ventana real previa de tamaño H).
    p_bull[:H] = p_bull[H]
    return p_bull


def predict_pbull_rollout(model, initial_window, T):
    """p_bull(t) por rollout autoregresivo desde la ULTIMA ventana observada
    (lectura alineada al PDF, sec. 2.5 paso 1 + ec. 15).

    Para t = 1..T, en cada paso:
      1. aplica el LSTM a la ventana actual y obtiene quintiles r_hat^(q),
      2. calcula p_bull(t) = (1/|Q|) sum_q 1{r_hat^(q) >= 0}  (ec. 15),
      3. avanza la ventana con el QUINTIL MEDIANO (proxy puntual; equivale
         a un "escenario mediano" sin muestreo aleatorio),
      4. repite.

    A diferencia de `predict_pbull_walking`, aqui el LSTM solo ve retornos
    REALES en la ventana inicial. Para t > 1 la ventana ya esta poblada con
    proyecciones propias — no hay info futura entrando como input.

    initial_window: (H, A) - los H retornos mas recientes del historico.
    return:         (T, A) - p_bull alineado a t = 1..T del FUTURO.
    """
    cfg = model.config
    H, A, Q = cfg.H, cfg.n_assets, cfg.n_quantiles
    if initial_window.shape != (H, A):
        raise ValueError(
            f"initial_window shape {initial_window.shape} != (H={H}, A={A})"
        )

    window = initial_window.astype(np.float32).copy()       # (H, A)
    p_bull = np.empty((T, A), dtype=np.float32)
    median_idx = Q // 2                                     # quintil 0.5

    for t in range(T):
        x = ((window - model.mean) / model.std).astype(np.float32)[None, :, :]
        x_tensor = torch.from_numpy(x)
        with torch.no_grad():
            outs = [net(x_tensor).numpy()[0] for net in model.nets]   # K * (A, Q)
        preds = np.mean(np.stack(outs, axis=0), axis=0)               # (A, Q)
        preds = np.sort(preds, axis=-1)                               # monotonicidad

        p_bull[t] = (preds >= 0.0).mean(axis=-1).astype(np.float32)   # ec. 15

        # Avanzar ventana con la mediana (proxy puntual del rollout).
        r_next = preds[:, median_idx]                                 # (A,)
        window = np.concatenate([window[1:, :], r_next[None, :]], axis=0)

    return p_bull


# ================================================================
# 2) Contexto DL -> dict compatible con solve_portfolio
# ================================================================

def _compute_hist_moments(r_hist, p_hist, assets, regimes, moments_window=None):
    """Replica el calculo de mu_hat/sigma_hat de load_market_data (Opcion B).

    moments_window: tupla (t_start, t_end) inclusive sobre el indice temporal
        de r_hist/p_hist. Si None (default) usa toda la serie disponible.
        Cuando se pasa, mu_hat y sigma_hat se estiman SOLO con datos en la
        ventana indicada — util para evitar leakage de info futura en la
        estimacion de momentos (p.ej. fijar la ventana a TRAIN del LSTM).
    """
    def _slice(s):
        if moments_window is None:
            return s
        a, b = moments_window
        return s.loc[a:b]

    mu_hat = {}
    for i in assets:
        for k in regimes:
            p_w = _slice(p_hist[i][k])
            r_w = _slice(r_hist[i])
            den = p_w.sum()
            mu_hat[(i, k)] = ((p_w * r_w).sum() / den
                              if den > 0 else 0.0)

    sigma_hat = {}
    for i in assets:
        for j in assets:
            for k in regimes:
                pi_w = _slice(p_hist[i][k])
                pj_w = _slice(p_hist[j][k])
                ri_w = _slice(r_hist[i])
                rj_w = _slice(r_hist[j])
                den  = (pi_w * pj_w).sum()
                if den > 0:
                    term = (pi_w * pj_w
                            * (ri_w - mu_hat[(i, k)])
                            * (rj_w - mu_hat[(j, k)]))
                    sigma_hat[(i, j, k)] = term.sum() / den
                else:
                    sigma_hat[(i, j, k)] = 0.0
    return mu_hat, sigma_hat


def build_dl_context(data_dir, checkpoint_path, T=T_HORIZON,
                     N_candidates=N_CANDIDATES, n_scenarios=N_SCENARIOS,
                     seed=SCENARIO_SEED, summary_asset=SUMMARY_ASSET,
                     position=None, moments_window=None,
                     p_method: str = "walking",
                     mu_hat_source: str = "p_sign"):
    """
    Construye un contexto compatible con solve_portfolio siguiendo la
    descomposicion por regimen del PDF (ec. 2-5).

    Defaults PDF-aligned (ver HALLAZGOS_ALINEAMIENTO_PDF.md). Para
    reproducir el comportamiento legacy del codigo, usar
    p_method='walking', mu_hat_source='p_dl'.

    p_method: 'walking' (default — captura variacion temporal en la practica),
        'rollout' (PDF-pure pero estructuralmente roto), 'scenarios'.
        Ver HALLAZGOS_ALINEAMIENTO_PDF.md secciones 4 y 12 para la justificacion
        completa del default.
        - 'walking': aplica el LSTM a ventanas REALES r_hist[t-H..t-1] para
          construir p_dl(t). Usa retornos reales como input — eso es "info que
          un trader real no tendria al planificar el futuro" si se interpreta
          t=1..T como periodos futuros. Sin embargo, validado en
          synthetic_walking_test que walking captura la senal temporal cuando
          existe; rollout no lo hace incluso con data sintetica perfecta.
        - 'rollout': avanza autoregresivamente desde la ultima ventana
          observada con quintil mediano como proxy. Deterministico. **COLAPSA A
          UN PUNTO FIJO** despues de H pasos (exposure bias clasico de modelos
          autoregresivos). mu_mix(t) queda casi constante => el regret-grid
          degenera (todas las celdas dan misma V). Demostrado en
          synthetic_walking_test: incluso con data sintetica de senal clara
          (sinusoide period=30, p_hist std=0.18), rollout produce p_dl
          constante = 0.60. Usar solo como referencia de "lo que el PDF
          parece pedir literalmente".
        - 'scenarios': calcula p_dl(t, A) = (1/N) Σ_s 1{candidates[s, t, A]
          >= 0}, promediando la ec. 15 sobre los N escenarios candidatos.
          Tambien tiende a aplanar por la ley de grandes numeros del promedio.

    mu_hat_source: 'p_sign' (default, resuelve inconsistencia de regimen
        ver HALLAZGOS_ALINEAMIENTO_PDF.md), 'p_hist' (PDF literal), o
        'p_dl' (legacy).
        - 'p_sign': oraculo historico con la MISMA regla del LSTM
          (ec. 15: bull si r >= BULL_THRESHOLD). Da gap real entre regimenes
          (SPX ~3.7%/sem, CMC ~14%/sem) y resuelve la inconsistencia entre
          la definicion de regimen del estimador y la del LSTM. Es un
          estimador historico (no leakage del futuro), conceptualmente analogo
          a estimar media o varianza muestral sobre datos disponibles.
        - 'p_hist': lectura literal del PDF sec. 1.3 — usa las probabilidades
          del CSV historico (HMM-derived). Caveat: empiricamente la accuracy
          de p_hist(bull) vs (r >= 0) es ~52-56% (apenas mejor que azar),
          entonces mu_hat(bull) ≈ mu_hat(bear) (incluso con signo invertido)
          y la mezcla por periodo no genera variacion temporal en mu_mix.
          Disponible para reproducir el comportamiento literal del PDF.
        - 'p_dl': estima mu_hat con las probabilidades del LSTM, agregando
          dependencia circular entre el estimador y la prediccion. Legacy.

    Otros:
      - mu_hat[(i,k)], sigma_hat[(i,j,k)] = ec. (2)-(3).
      - mu_mix(i,t), sigma_mix(i,j,t)     = ec. (4)-(5).
      - 5 escenarios = quintiles del rollout autoregresivo (sec. 2.5).
    """
    if p_method not in {"walking", "rollout", "scenarios"}:
        raise ValueError(f"p_method invalido: {p_method!r}")
    if mu_hat_source not in {"p_dl", "p_hist", "p_sign"}:
        raise ValueError(f"mu_hat_source invalido: {mu_hat_source!r}")
    data_dir = Path(data_dir)
    base_ctx = load_market_data(str(data_dir))
    assets   = list(base_ctx["assets"])
    regimes  = list(REGIMES)
    r_hist   = base_ctx["r"]

    # --- modelo DL + historico ---
    model = load_checkpoint(checkpoint_path)
    H     = model.config.H
    returns_history = np.stack(
        [r_hist[i].sort_index().values[:T] for i in assets], axis=1,
    ).astype(np.float32)                                 # (T, A)
    initial_window = returns_history[-H:, :]             # (H, A)
    T_vals = list(range(1, T + 1))

    # --- p_{i,k,t} segun p_method ---
    # En modo 'scenarios' necesitamos los N candidatos antes para promediar
    # la ec. 15 sobre ellos. Los reusamos despues para reducir a 5 reps =>
    # no se duplica cómputo y la coherencia ex-ante/ex-post es exacta.
    candidates_cache = None
    if p_method == "scenarios":
        candidates_cache = generate_candidate_scenarios(
            model, initial_window, N=N_candidates, T=T, seed=seed,
        )                                                    # (N, T, A)
        # ec. 15 promediada: para cada (t, A), fraccion de escenarios con r >= 0.
        p_bull_arr = (candidates_cache >= 0.0).mean(axis=0).astype(np.float32)
    elif p_method == "walking":
        # Legacy: aplica el LSTM a ventana REAL previa. Le da al LSTM info
        # futura via los retornos reales como input (ver docstring).
        p_bull_arr = predict_pbull_walking(model, returns_history, T)
    else:
        # 'rollout': autoregresivo desde initial_window con quintil mediano
        # como proxy. Determinístico. Suele colapsar a punto fijo.
        p_bull_arr = predict_pbull_rollout(model, initial_window, T)
    p_bear_arr = 1.0 - p_bull_arr
    p_dl = {
        asset: pd.DataFrame(
            {"bear": p_bear_arr[:, ai], "bull": p_bull_arr[:, ai]},
            index=T_vals,
        )
        for ai, asset in enumerate(assets)
    }

    # r_hist truncado a t=1..T y reindexado a T_vals (sirve para mu_hat y
    # para el simulador historico, no para el FO en modo rollout).
    r_hist_T = {
        i: pd.Series(r_hist[i].sort_index().values[:T], index=T_vals)
        for i in assets
    }

    # --- Momentos por regimen (ec. 2-3) ---
    # mu_hat_source elige que probabilidades alimentan al estimador.
    if mu_hat_source == "p_dl":
        p_for_moments = p_dl
    elif mu_hat_source == "p_hist":
        # Usar las p del CSV historico (HMM-derived) truncadas a 1..T.
        # Interpretacion literal del PDF sec. 1.3.
        p_for_moments = {
            i: pd.DataFrame(
                base_ctx["p_hist"][i].sort_index().values[:T, :],
                index=T_vals,
                columns=list(base_ctx["p_hist"][i].columns),
            )
            for i in assets
        }
    else:
        # 'p_sign': oraculo historico con la regla del LSTM (ec. 15:
        # bull si r >= BULL_THRESHOLD). Da estimador consistente con la
        # definicion de regimen que predice p_dl. No es leakage — es un
        # estimador historico, analogo a media muestral.
        from config import BULL_THRESHOLD
        p_for_moments = {}
        for i in assets:
            r_i = r_hist_T[i]
            bull_i = (r_i >= BULL_THRESHOLD).astype(float)
            p_for_moments[i] = pd.DataFrame(
                {"bear": (1.0 - bull_i).values, "bull": bull_i.values},
                index=T_vals,
            )
    mu_hat, sigma_hat = _compute_hist_moments(
        r_hist_T, p_for_moments, assets, regimes, moments_window=moments_window,
    )

    # --- Mezcla por periodo (ec. 4-5) ---
    mu_mix    = {i: pd.Series(0.0, index=T_vals) for i in assets}
    sigma_mix = {i: {j: pd.Series(0.0, index=T_vals) for j in assets} for i in assets}

    for i in assets:
        for k in regimes:
            mu_mix[i] += p_dl[i][k] * mu_hat[(i, k)]

    for i in assets:
        for j in assets:
            for k in regimes:
                sigma_mix[i][j] += p_dl[i][k] * p_dl[j][k] * sigma_hat[(i, j, k)]

    # Simetrizacion numerica (la formula ya es simetrica analiticamente).
    for i in assets:
        for j in assets:
            sym = 0.5 * (sigma_mix[i][j] + sigma_mix[j][i])
            sigma_mix[i][j] = sym
            sigma_mix[j][i] = sym

    # --- N candidatos del LSTM (rollout) para generar los 5 escenarios ---
    # Si ya se generaron arriba (modo 'scenarios'), los reutilizamos.
    if candidates_cache is not None:
        candidates = candidates_cache
    else:
        candidates = generate_candidate_scenarios(
            model, initial_window, N=N_candidates, T=T, seed=seed,
        )                                                # (N, T, A)

    from config import SCENARIO_POSITION
    pos = position if position is not None else SCENARIO_POSITION
    summary_idx = assets.index(summary_asset)
    scenarios = reduce_to_representatives(
        candidates, summary_asset_idx=summary_idx,
        n_quintiles=n_scenarios, position=pos,
    )                                                    # (n_scenarios, T, A)

    return {
        "mu_mix":          mu_mix,
        "sigma_mix":       sigma_mix,
        "T_vals":          T_vals,
        "nT":              T,
        "assets":          assets,
        "c_base":          base_ctx["c_base"],
        "w0":              base_ctx["w0"],
        "Capital_inicial": base_ctx["Capital_inicial"],
        "V_max":           base_ctx["V_max"],
        "r":               r_hist,
        "scenarios":       scenarios,
        "p_dl":            p_dl,
        # Diagnostico adicional especifico de esta variante:
        "mu_hat":          mu_hat,
        "sigma_hat":       sigma_hat,
    }


# ================================================================
# 2.b) Trim del padding del walking: recorta el ctx a t = H+1..T
# ================================================================

def trim_post_warmup(ctx: dict, H: int, T_max: int | None = None,
                     trim_scenarios: bool = True) -> dict:
    """Recorta un contexto a t = H+1..T_max y lo re-indexa a t = 1..(T_max-H).

    Objetivo: correr el optimizador (y el backtest historico) SIN el padding
    de `predict_pbull_walking` que rellena p_bull[:H] con p_bull[H]. El padding
    se introdujo para mantener la alineacion T_vals = 1..T del GAMS base; aqui
    asumimos que el optimizador puede trabajar sobre un horizonte mas corto
    (T_eff = T - H) que solo contiene t con prediccion LSTM real.

    Funciona tanto con ctx de `build_dl_context` (tiene 'p_dl' y 'scenarios')
    como con el de `load_market_data` (no los tiene). Las series time-indexed
    (mu_mix, sigma_mix, p_dl, r) se trimean y re-indexan al rango 1..T_eff.
    Los escenarios (n_S, T, A) se truncan a (n_S, T_eff, A): conceptualmente
    son los primeros T_eff pasos del rollout autoregresivo desde el final
    del historico (no se ven afectados por el padding, pero se truncan para
    quedar alineados con el horizonte de la FO).

    `r` se desplaza: r_new[i].loc[k] = r_old[i].loc[H+k]. De este modo
    simulate_capital_opt con el ctx trimmed y la politica trimmed evalua
    sobre los retornos del calendario *post-warmup* (semanas H+1..T_max
    del historico), que es la ventana para la que el LSTM hizo predicciones
    walking-window reales.

    V_max, c_base, w0, Capital_inicial, assets se preservan tal cual.
    """
    assets = ctx["assets"]
    if T_max is None:
        T_max = ctx["nT"]
    T_eff = T_max - H
    if T_eff <= 0:
        raise ValueError(f"T_max={T_max} <= H={H}: nada para trimear.")

    new_T_vals = list(range(1, T_eff + 1))

    def _trim_series(s):
        return pd.Series(s.iloc[H:T_max].values, index=new_T_vals)

    mu_mix    = {i: _trim_series(ctx["mu_mix"][i]) for i in assets}
    sigma_mix = {
        i: {j: _trim_series(ctx["sigma_mix"][i][j]) for j in assets}
        for i in assets
    }

    new_ctx = dict(ctx)
    new_ctx["mu_mix"]    = mu_mix
    new_ctx["sigma_mix"] = sigma_mix
    new_ctx["T_vals"]    = new_T_vals
    new_ctx["nT"]        = T_eff

    if "p_dl" in ctx:
        new_p_dl = {}
        for i in assets:
            df = ctx["p_dl"][i].iloc[H:T_max].copy()
            df.index = new_T_vals
            new_p_dl[i] = df
        new_ctx["p_dl"] = new_p_dl

    if "p_hist" in ctx:
        new_p_hist = {}
        for i in assets:
            df = ctx["p_hist"][i].iloc[H:T_max].copy()
            df.index = new_T_vals
            new_p_hist[i] = df
        new_ctx["p_hist"] = new_p_hist

    if trim_scenarios and "scenarios" in ctx:
        new_ctx["scenarios"] = ctx["scenarios"][:, :T_eff, :]

    if "r" in ctx:
        new_r = {}
        for i in assets:
            s_full = ctx["r"][i].sort_index()
            new_r[i] = pd.Series(s_full.iloc[H:T_max].values, index=new_T_vals)
        new_ctx["r"] = new_r

    return new_ctx


# ================================================================
# 3) Simulacion de capital en un escenario
# ================================================================

def simulate_capital_on_scenario(w_sol, u_sol, v_sol, scenario,
                                 assets, c_base, C0, T_vals):
    """
    Ec. (19): x_{t+1} = x_t (1 + sum_i w(i,t)*r^s_{i,t})
                      - x_t sum_i c_i * |w(i,t) - w(i,t-1)|
    El costo cobrado en el paso t->t+1 es el del rebalanceo HACIA w(t),
    es decir u(i,t)+v(i,t). En el primer paso (t=t1) eso captura el
    rebalanceo inicial w0 -> w(t1).
    scenario: (T, A) — scenario[k, ai] es el retorno en el periodo T_vals[k].
    """
    cap = {T_vals[0]: C0}
    for idx in range(1, len(T_vals)):
        t      = T_vals[idx]
        t_prev = T_vals[idx - 1]
        r_port = sum(w_sol[i, t_prev] * scenario[idx - 1, ai]
                     for ai, i in enumerate(assets))
        turn   = sum(c_base[i] * (u_sol[i, t_prev] + v_sol[i, t_prev]) for i in assets)
        cap[t] = cap[t_prev] * (1.0 + r_port) - cap[t_prev] * turn
    return cap


# ================================================================
# 4) Regret-grid
# ================================================================

def run_regret_grid(context, lambda_grid, m_grid):
    """Alg. 1 del PDF 3.5: un solve por g, simulacion por (g, s)."""
    assets    = context["assets"]
    T_vals    = context["T_vals"]
    c_base    = context["c_base"]
    C0        = context["Capital_inicial"]
    scenarios = context["scenarios"]
    n_S       = scenarios.shape[0]

    rows     = []
    policies = {}
    total    = len(lambda_grid) * len(m_grid)
    run      = 0

    for lam in lambda_grid:
        for cm in m_grid:
            run += 1
            print(f"  [{run:>2}/{total}] lambda={lam:.2f}  m={cm:.2f} ...",
                  end=" ", flush=True)
            z, w_sol, u_sol, v_sol, _ = solve_portfolio(
                context, lambda_riesgo=lam, costo_mult=cm,
            )
            policies[(lam, cm)] = (w_sol, u_sol, v_sol, z)

            Vs = []
            for s in range(n_S):
                cap = simulate_capital_on_scenario(
                    w_sol, u_sol, v_sol, scenarios[s],
                    assets, c_base, C0, T_vals,
                )
                Vs.append(cap[T_vals[-1]])
                rows.append({"lambda": lam, "m": cm, "s": s,
                             "V": Vs[-1], "z": z})
            print(f"z={z:.4f}  V=[${min(Vs):,.0f} .. ${max(Vs):,.0f}]")

    return pd.DataFrame(rows), policies


# ================================================================
# 5) Regret y seleccion
# ================================================================

def compute_regret_and_select(V_df):
    """V_best_s (ec. 21) y R[g, s] (ec. 22); elige g* por promedio y peor caso.

    Valida que V sea finito y razonable: ratio max/median > 1000 indica una
    degeneracion (tipicamente IPOPT con costo_mult=0 o restriccion mal
    configurada). Si g* cae en la frontera del grid, emite un warning para
    sugerir extender LAMBDA_GRID o M_GRID.
    """
    V_table = V_df.pivot_table(
        index=["lambda", "m"], columns="s", values="V", aggfunc="first",
    )
    v_arr = V_table.values
    if not np.isfinite(v_arr).all():
        raise ValueError(
            f"V_table contiene valores no finitos. Probable patologia en "
            f"solve_portfolio (revisar costo_mult > 0).\n{V_table}"
        )
    median = np.median(v_arr)
    if median > 0 and v_arr.max() / median > 1000:
        raise ValueError(
            f"V_table contiene valores absurdos: max={v_arr.max():.2e}, "
            f"median={median:.2e}, ratio={v_arr.max()/median:.0f}x. "
            "Probable degeneracion de IPOPT (revisar que M_GRID > 0)."
        )

    V_best_s     = V_table.max(axis=0)
    R_table      = V_best_s - V_table
    mean_regret  = R_table.mean(axis=1)
    worst_regret = R_table.max(axis=1)
    summary = pd.DataFrame({
        "mean_regret":  mean_regret,
        "worst_regret": worst_regret,
    })
    g_mean  = mean_regret.idxmin()
    g_worst = worst_regret.idxmin()

    lambda_values = sorted(V_table.index.get_level_values("lambda").unique())
    m_values      = sorted(V_table.index.get_level_values("m").unique())
    boundary_lam  = {lambda_values[0], lambda_values[-1]}
    boundary_m    = {m_values[0],      m_values[-1]}
    import warnings
    for label, (lam, m_) in [("g*_mean", g_mean), ("g*_worst", g_worst)]:
        on_boundary = []
        if lam in boundary_lam: on_boundary.append(f"lambda={lam:.2f}")
        if m_  in boundary_m:   on_boundary.append(f"m={m_:.2f}")
        if on_boundary:
            warnings.warn(
                f"{label} cae en frontera del grid ({', '.join(on_boundary)}). "
                "Considera extender LAMBDA_GRID/M_GRID en config.py.",
                stacklevel=2,
            )

    return {
        "V_table":        V_table,
        "R_table":        R_table,
        "V_best_s":       V_best_s,
        "regret_summary": summary,
        "g_mean":         g_mean,
        "g_worst":        g_worst,
        "g_mean_metric":  mean_regret.min(),
        "g_worst_metric": worst_regret.min(),
    }


# ================================================================
# 6) Plot: evolucion del capital por escenario bajo una politica
# ================================================================

def simulate_capital_opt(w_sol, u_sol, v_sol, context):
    """Capital ex-post bajo retornos historicos (ec. 19, version historica).

    cap[t] = cap[t-1] * (1 + sum_i w(i,t-1)*r(i,t-1))
           - cap[t-1] * sum_i c_base(i)*(u(i,t-1)+v(i,t-1)).
    El costo del paso t-1 -> t es el del rebalanceo HACIA w(t-1) (incluye
    el rebalanceo inicial w0 -> w(t1) en el primer paso).
    Siempre usa c_base (sin multiplicador): el costo_mult solo penaliza ex-ante en la FO.
    """
    T_vals  = context["T_vals"]
    assets  = context["assets"]
    r       = context["r"]
    c_base  = context["c_base"]
    Capital = context["Capital_inicial"]

    cap = {T_vals[0]: Capital}
    for idx in range(1, len(T_vals)):
        t      = T_vals[idx]
        t_prev = T_vals[idx - 1]
        r_port = sum(w_sol[i, t_prev] * r[i].loc[t_prev] for i in assets)
        turn   = sum(c_base[i] * (u_sol[i, t_prev] + v_sol[i, t_prev]) for i in assets)
        cap[t] = cap[t_prev] * (1.0 + r_port) - cap[t_prev] * turn
    return cap


def simulate_naive_bh(context):
    """Naive 50/50 buy & hold sobre retornos historicos (sin rebalanceo ni costos)."""
    T_vals  = context["T_vals"]
    assets  = context["assets"]
    r       = context["r"]
    Capital = context["Capital_inicial"]
    w_naive = {i: 0.5 for i in assets}

    cap = {T_vals[0]: Capital}
    for idx in range(1, len(T_vals)):
        t      = T_vals[idx]
        t_prev = T_vals[idx - 1]
        r_port = sum(w_naive[i] * r[i].loc[t_prev] for i in assets)
        cap[t] = cap[t_prev] * (1.0 + r_port)
    return cap


def simulate_naive_rb(context):
    """Naive 50/50 con rebalanceo semanal y costos sobre retornos historicos."""
    T_vals   = context["T_vals"]
    assets   = context["assets"]
    r        = context["r"]
    c_base   = context["c_base"]
    Capital  = context["Capital_inicial"]
    w_target = 0.5

    cap = {T_vals[0]: Capital}
    for idx in range(1, len(T_vals)):
        t      = T_vals[idx]
        t_prev = T_vals[idx - 1]
        r_port = sum(w_target * r[i].loc[t_prev] for i in assets)
        w_bh   = {i: w_target * (1.0 + r[i].loc[t_prev]) / (1.0 + r_port) for i in assets}
        turn   = sum(c_base[i] * abs(w_target - w_bh[i]) for i in assets)
        cap[t] = cap[t_prev] * (1.0 + r_port) - cap[t_prev] * turn
    return cap


def plot_capital_evolution_historical(cap_opt, cap_rb, cap_bh, cap_regret,
                                      T_vals, lam_star, m_star, out_path):
    """OPT vs Naive 50/50 (rebal) vs Naive 50/50 (B&H) vs Regret-Grid g*_mean.

    Las cuatro politicas se simulan sobre la misma serie de retornos historicos;
    la diferencia es la (lambda, m) que produjo cada w(i,t). El "OPT" es el
    optimo media-varianza con (lambda=1.00, m=1.0); el "Regret-Grid" usa los
    parametros seleccionados ex-ante por el pipeline DL + regret.
    """
    x = list(T_vals)
    y_opt    = [cap_opt[t]    for t in x]
    y_rb     = [cap_rb[t]     for t in x]
    y_bh     = [cap_bh[t]     for t in x]
    y_regret = [cap_regret[t] for t in x]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, y_opt,    label="OPT",                       color="#F2B705", linewidth=1.8)
    ax.plot(x, y_rb,     label="NAIVE 50/50 (rebal)",       color="#8B3A1F", linewidth=1.2)
    ax.plot(x, y_bh,     label="NAIVE 50/50 (buy&hold)",    color="#E63946", linewidth=1.2)
    ax.plot(x, y_regret,
            label=f"Regret-Grid g*_mean (lambda={lam_star:.2f}, m={m_star:.2f})",
            color="#1f77b4", linewidth=1.8)

    ax.set_title("Evolucion de capital")
    ax.set_xlabel("t")
    ax.set_ylabel("Capital")
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Grafico guardado en: {out_path}")


def plot_capital_curves(w_sol, u_sol, v_sol, context, title, out_path):
    assets    = context["assets"]
    T_vals    = context["T_vals"]
    c_base    = context["c_base"]
    C0        = context["Capital_inicial"]
    scenarios = context["scenarios"]
    n_S       = scenarios.shape[0]

    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.get_cmap("viridis")
    for s in range(n_S):
        cap = simulate_capital_on_scenario(
            w_sol, u_sol, v_sol, scenarios[s], assets, c_base, C0, T_vals,
        )
        ax.plot(T_vals, [cap[t] for t in T_vals],
                color=cmap(s / max(n_S - 1, 1)),
                label=f"Escenario s{s + 1}", linewidth=1.3)
    ax.axhline(C0, color="#666", linestyle="--", linewidth=0.8,
               label=f"Capital inicial (${C0:,.0f})")
    ax.set_title(title)
    ax.set_xlabel("t (forward)")
    ax.set_ylabel("Capital")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Grafico guardado en: {out_path}")


# ================================================================
# 6.b) Backtest historico (politica DL aplicada a r_hist)
# ================================================================

def run_historical_backtest(w_star, u_star, v_star, lam_m, m_m,
                            V_mean_row, n_scenarios,
                            data_dir, out_path, hist_ctx=None):
    """Compara OPT, Naive (rebal/B&H) y Regret-Grid g*_mean sobre retornos historicos.

    El "OPT" es el optimo media-varianza con (lambda=1.00, m=1.0) sobre el contexto
    historico (replica el caso base del GAMS). La linea del Regret-Grid usa los
    pesos w_star ya calculados por el grid DL (no se re-optimiza), simulados sobre
    r_hist — es la decision real del modelo enfrentada a la trayectoria observada.
    """
    if hist_ctx is None:
        hist_ctx = load_market_data(data_dir)
    C0 = hist_ctx["Capital_inicial"]

    _, w_opt, u_opt, v_opt, _ = solve_portfolio(
        hist_ctx, lambda_riesgo=1.00, costo_mult=1.0,
    )
    cap_opt = simulate_capital_opt(w_opt, u_opt, v_opt, hist_ctx)
    cap_rg  = simulate_capital_opt(w_star, u_star, v_star, hist_ctx)
    cap_rb  = simulate_naive_rb(hist_ctx)
    cap_bh  = simulate_naive_bh(hist_ctx)

    T_h     = hist_ctx["T_vals"]
    t_final = T_h[-1]
    cf_opt  = cap_opt[t_final]
    cf_rg   = cap_rg [t_final]
    cf_rb   = cap_rb [t_final]
    cf_bh   = cap_bh [t_final]

    print("\n--- Backtest historico (politica aplicada a r_hist) ---")
    print(f"  Capital inicial = ${C0:,.2f}   horizonte = t1..t{t_final} ({len(T_h)} periodos)")
    print(f"  {'politica':<42}  {'cap_final':>12}  {'ret_acum':>9}  {'inc_cap':>12}")
    print(f"  {'-' * 80}")
    fmt = "  {:<42}  ${:>11,.2f}  {:>+8.2%}  ${:>+11,.2f}"
    print(fmt.format("OPT (lambda=1.00, m=1.0)",
                     cf_opt, cf_opt/C0 - 1, cf_opt - C0))
    print(fmt.format(f"Regret-Grid g*_mean (lambda={lam_m:.2f}, m={m_m:.2f})",
                     cf_rg,  cf_rg /C0 - 1, cf_rg  - C0))
    print(fmt.format("Naive 50/50 rebalanceo",
                     cf_rb,  cf_rb /C0 - 1, cf_rb  - C0))
    print(fmt.format("Naive 50/50 buy & hold",
                     cf_bh,  cf_bh /C0 - 1, cf_bh  - C0))
    print(f"\n  Nota: el ret_acum de Regret-Grid sobre r_hist NO es comparable")
    print(f"        con el +{V_mean_row.mean()/C0 - 1:.2%} promedio sobre los "
          f"{n_scenarios} escenarios DL del bloque anterior — uno es backtest")
    print(f"        sobre la trayectoria observada, el otro es promedio sobre las")
    print(f"        trayectorias forecast del LSTM.")

    plot_capital_evolution_historical(
        cap_opt, cap_rb, cap_bh, cap_rg,
        T_h, lam_m, m_m, out_path=out_path,
    )

    return {
        "cap_opt": cap_opt, "cap_rg": cap_rg,
        "cap_rb":  cap_rb,  "cap_bh": cap_bh,
        "hist_ctx": hist_ctx,
    }


# ================================================================
# 6.b) Helpers para evaluacion out-of-sample (test segment del split DL)
# ================================================================

def compute_test_start_t(
    T: int = T_HORIZON,
    H: int = None,
    split: Tuple[float, float, float] = SPLIT,
) -> int:
    """Primer t (1-indexed sobre el indice del CSV) cuyo target NO se uso
    durante el entrenamiento del LSTM.

    El split del LSTM se aplica sobre las VENTANAS (no sobre los retornos):
    para H ventanas-de-largo-H sobre T retornos, hay N = T - H ventanas;
    la ventana en posicion n tiene target en t_vals[n + H]. Por lo tanto el
    primer target del test set esta en t = H + n_train + n_valid + 1
    (en 1-indexed). Con T=163, H=60, split=(0.70,0.15,0.15) => t_test_start=148.
    """
    if H is None:
        from config import H_WINDOW
        H = H_WINDOW
    N_windows = T - H
    if N_windows <= 0:
        raise ValueError(f"T={T} debe ser > H={H}")
    n_train = int(N_windows * split[0])
    n_valid = int(N_windows * split[1])
    return H + n_train + n_valid + 1


def build_hist_ctx_oos(
    data_dir: Path,
    t_test_start: int,
    T: int = T_HORIZON,
) -> dict:
    """Contexto historico restringido a [t_test_start..T] para el backtest OOS.

    - mu_hat / sigma_hat / V_max se estiman SOLO sobre [1..t_test_start-1]
      (train+valid del split) para no filtrar info del test set.
    - p_hist(t) en [t_test_start..T] se usa para el mix mu_mix/sigma_mix.
    - r_hist(t) en [t_test_start..T] se usa para simular las curvas de capital.
    """
    ctx_full = load_market_data(
        str(data_dir),
        moments_window=(1, t_test_start - 1),
    )
    assets = list(ctx_full["assets"])
    out = dict(ctx_full)
    T_vals_new = [t for t in ctx_full["T_vals"] if t_test_start <= t <= T]
    out["T_vals"]    = T_vals_new
    out["nT"]        = len(T_vals_new)
    out["mu_mix"]    = {
        i: ctx_full["mu_mix"][i].loc[t_test_start:T] for i in assets
    }
    out["sigma_mix"] = {
        i: {j: ctx_full["sigma_mix"][i][j].loc[t_test_start:T] for j in assets}
        for i in assets
    }
    out["r"]         = {
        i: ctx_full["r"][i].loc[t_test_start:T] for i in assets
    }
    out["p_hist"]    = {
        i: ctx_full["p_hist"][i].loc[t_test_start:T] for i in assets
    }
    out["t_start"]   = t_test_start
    out["moments_window"] = (1, t_test_start - 1)
    return out


def run_historical_backtest_oos(
    w_star, u_star, v_star, lam_m, m_m,
    V_mean_row, n_scenarios,
    data_dir, t_test_start, T, out_path,
):
    """Backtest OOS: compara OPT_oos / Naive RB / Naive BH / Regret-Grid g*_mean
    sobre r_hist[t_test_start..T] (segmento test del split del LSTM).

    OPT_oos se re-resuelve con lambda=1.0 sobre un contexto donde mu_hat/sigma_hat
    se estimaron solo en train+valid. Esto evita leakage de la varianza del test
    en el estimador.

    Las políticas RG vienen del regret-grid OOS (ya resuelto con horizonte=16
    semanas y anchor w0=50/50 en t=t_test_start).
    """
    hist_ctx_oos = build_hist_ctx_oos(data_dir, t_test_start, T)
    C0 = hist_ctx_oos["Capital_inicial"]

    _, w_opt, u_opt, v_opt, _ = solve_portfolio(
        hist_ctx_oos, lambda_riesgo=1.00, costo_mult=1.0,
    )
    cap_opt = simulate_capital_opt(w_opt, u_opt, v_opt, hist_ctx_oos)
    cap_rg  = simulate_capital_opt(w_star, u_star, v_star, hist_ctx_oos)
    cap_rb  = simulate_naive_rb(hist_ctx_oos)
    cap_bh  = simulate_naive_bh(hist_ctx_oos)

    T_h     = hist_ctx_oos["T_vals"]
    t_final = T_h[-1]
    cf_opt  = cap_opt[t_final]
    cf_rg   = cap_rg [t_final]
    cf_rb   = cap_rb [t_final]
    cf_bh   = cap_bh [t_final]

    print("\n--- Backtest OUT-OF-SAMPLE (politica aplicada a r_hist[test]) ---")
    print(f"  Segmento test del split LSTM: t={t_test_start}..t{T} "
          f"({len(T_h)} periodos)")
    print(f"  Capital inicial = ${C0:,.2f}   anchor w0 = 50/50 en t={t_test_start}")
    print(f"  mu_hat/sigma_hat estimados solo sobre t=1..{t_test_start-1} (train+valid)")
    print(f"  {'politica':<42}  {'cap_final':>12}  {'ret_acum':>9}  {'inc_cap':>12}")
    print(f"  {'-' * 80}")
    fmt = "  {:<42}  ${:>11,.2f}  {:>+8.2%}  ${:>+11,.2f}"
    print(fmt.format("OPT_oos (lambda=1.00, m=1.0)",
                     cf_opt, cf_opt/C0 - 1, cf_opt - C0))
    print(fmt.format(f"Regret-Grid_oos g*_mean (lambda={lam_m:.2f}, m={m_m:.2f})",
                     cf_rg,  cf_rg /C0 - 1, cf_rg  - C0))
    print(fmt.format("Naive 50/50 rebalanceo",
                     cf_rb,  cf_rb /C0 - 1, cf_rb  - C0))
    print(fmt.format("Naive 50/50 buy & hold",
                     cf_bh,  cf_bh /C0 - 1, cf_bh  - C0))
    print(f"\n  Nota: V_mean (DL forecast) para g*_mean OOS = "
          f"{V_mean_row.mean()/C0 - 1:+.2%} (promedio sobre {n_scenarios} escenarios DL OOS).")

    plot_capital_evolution_historical(
        cap_opt, cap_rb, cap_bh, cap_rg,
        T_h, lam_m, m_m, out_path=out_path,
    )

    return {
        "cap_opt": cap_opt, "cap_rg": cap_rg,
        "cap_rb":  cap_rb,  "cap_bh": cap_bh,
        "hist_ctx_oos": hist_ctx_oos,
    }


# ================================================================
# 7) Pipeline PER-CELL: una NN por celda + escenarios compartidos
# ================================================================

def cell_seed(lam: float, m: float) -> int:
    """Seed deterministica para entrenar la NN de la celda (lam, m).

    Hash SHA256 sobre la representacion textual con 6 decimales -> int. Misma
    (lam, m) => misma seed entre corridas (reproducibilidad).
    """
    s = f"lam={lam:.6f}_m={m:.6f}".encode()
    return int(hashlib.sha256(s).hexdigest()[:8], 16) % (2**31)


def _per_cell_ckpt_path(lam: float, m: float, base_dir: Path) -> Path:
    return Path(base_dir) / f"decile_predictor_l{lam:.2f}_m{m:.2f}.pt"


def train_per_cell_nns(
    lambda_grid: Sequence[float],
    m_grid: Sequence[float],
    dl_config: DLConfig | None = None,
    models_dir: Path | None = None,
    force_retrain: bool = False,
) -> Dict[Tuple[float, float], LoadedModel]:
    """Entrena una NN por celda (lam, m) con seed deterministica = cell_seed(lam, m).

    Cachea checkpoints en {models_dir}/per_cell/decile_predictor_l{lam}_m{m}.pt.
    Si force_retrain=False y el checkpoint existe, lo reusa (no re-entrena).

    Returns: dict {(lam, m): LoadedModel}. Cada LoadedModel tiene UNA red
    (no se aplica seed averaging interno: el proposito del per-cell es
    introducir heterogeneidad entre celdas).
    """
    if dl_config is None:
        dl_config = DLConfig()
    if models_dir is None:
        models_dir = MODELS_DIR
    per_cell_dir = Path(models_dir) / "per_cell"
    per_cell_dir.mkdir(parents=True, exist_ok=True)

    nns: Dict[Tuple[float, float], LoadedModel] = {}
    total = len(lambda_grid) * len(m_grid)
    run = 0

    for lam in lambda_grid:
        for m in m_grid:
            run += 1
            ckpt_path = _per_cell_ckpt_path(lam, m, per_cell_dir)
            seed = cell_seed(lam, m)
            print(f"  [{run:>2}/{total}] cell lambda={lam:.2f} m={m:.2f} "
                  f"seed={seed}", flush=True)

            if ckpt_path.exists() and not force_retrain:
                print(f"    cache hit -> {ckpt_path.name}")
            else:
                result = train_deciles(dl_config, seed=seed)
                save_checkpoint(result, ckpt_path)
                print(f"    trained best_valid={result.best_valid:.6f}  "
                      f"-> {ckpt_path.name}")

            nns[(lam, m)] = load_checkpoint(ckpt_path)

    return nns


def build_ensemble_model(
    nns: Dict[Tuple[float, float], LoadedModel],
) -> LoadedModel:
    """Combina las NNs por celda en un LoadedModel ensemble (promedio de logits).

    predict_deciles_batch y predict_pbull_walking ya promedian sobre
    model.nets, asi que un LoadedModel con la concatenacion de todos los
    nets da exactamente "promedio de logits sobre las 15 redes".

    Pre-requisitos: todas las NNs deben compartir DLConfig, mean y std (cierto
    cuando todas se entrenan con el mismo dl_config sobre el mismo dataset).
    """
    items = list(nns.values())
    if not items:
        raise ValueError("nns vacio: no hay redes para combinar")
    ref = items[0]
    all_nets = []
    for lm in items:
        all_nets.extend(lm.nets)
    return LoadedModel(
        nets=all_nets,
        config=ref.config,
        mean=ref.mean,
        std=ref.std,
    )


def build_per_cell_context(
    nn: LoadedModel,
    data_dir: Path = DATA_DIR,
    T: int = T_HORIZON,
    mu_hat_source: str = "p_hist",
    t_start: int = 1,
    moments_window: Tuple[int, int] | None = None,
) -> dict:
    """Construye un contexto para solve_portfolio usando UNA NN especifica.

    Variante de build_dl_context pensada para el pipeline per-cell:
    - p_method = 'walking' (default validado en HALLAZGOS sec 4).
    - mu_hat_source = 'p_hist' (lectura PDF literal sec 1.3: p del CSV).
    - NO genera escenarios (esos vienen del paso compartido en
      build_shared_scenarios).

    t_start: primer t (1-indexed sobre el indice del CSV) en T_vals del output.
        Default 1 (in-sample completo). Para evaluacion OOS pasar
        compute_test_start_t(T, H, SPLIT) (p.ej. 148 con SPLIT=(0.70,0.15,0.15)
        y T=163, H=60).
    moments_window: tupla (a, b) inclusive sobre el indice de r_hist para
        estimar mu_hat/sigma_hat. None (default) = toda la serie. Para
        evaluacion OOS limpia pasar (1, t_start-1) para que el estimador
        no vea el segmento de test.

    Devuelve un dict compatible con solve_portfolio (mu_mix, sigma_mix, etc.).
    Los campos 'scenarios' y similares se rellenan despues por el orquestador.
    """
    if mu_hat_source not in {"p_hist", "p_dl", "p_sign"}:
        raise ValueError(f"mu_hat_source invalido: {mu_hat_source!r}")
    if not (1 <= t_start <= T):
        raise ValueError(f"t_start={t_start} fuera de [1, T={T}]")
    data_dir = Path(data_dir)
    base_ctx = load_market_data(str(data_dir))
    assets   = list(base_ctx["assets"])
    regimes  = list(REGIMES)
    r_hist   = base_ctx["r"]

    H = nn.config.H
    returns_history = np.stack(
        [r_hist[i].sort_index().values[:T] for i in assets], axis=1,
    ).astype(np.float32)                                       # (T, A)

    # p_dl(t) por walking sobre la ventana real previa (PDF-aligned, validado).
    # Computamos para todo t=1..T y luego cortamos: walking necesita la ventana
    # real previa de tamano H, independientemente de t_start.
    p_bull_arr = predict_pbull_walking(nn, returns_history, T)  # (T, A)
    p_bear_arr = 1.0 - p_bull_arr
    T_full = list(range(1, T + 1))
    p_dl_full = {
        asset: pd.DataFrame(
            {"bear": p_bear_arr[:, ai], "bull": p_bull_arr[:, ai]},
            index=T_full,
        )
        for ai, asset in enumerate(assets)
    }

    r_hist_T = {
        i: pd.Series(r_hist[i].sort_index().values[:T], index=T_full)
        for i in assets
    }

    # mu_hat con p_hist (PDF literal sec 1.3): probabilidades del CSV.
    if mu_hat_source == "p_hist":
        p_for_moments = {
            i: pd.DataFrame(
                base_ctx["p_hist"][i].sort_index().values[:T, :],
                index=T_full,
                columns=list(base_ctx["p_hist"][i].columns),
            )
            for i in assets
        }
    elif mu_hat_source == "p_sign":
        from config import BULL_THRESHOLD
        p_for_moments = {}
        for i in assets:
            r_i = r_hist_T[i]
            bull_i = (r_i >= BULL_THRESHOLD).astype(float)
            p_for_moments[i] = pd.DataFrame(
                {"bear": (1.0 - bull_i).values, "bull": bull_i.values},
                index=T_full,
            )
    else:  # 'p_dl' (legacy, no recomendado)
        p_for_moments = p_dl_full

    mu_hat, sigma_hat = _compute_hist_moments(
        r_hist_T, p_for_moments, assets, regimes,
        moments_window=moments_window,
    )

    mu_mix_full = {i: pd.Series(0.0, index=T_full) for i in assets}
    sigma_mix_full = {
        i: {j: pd.Series(0.0, index=T_full) for j in assets} for i in assets
    }
    for i in assets:
        for k in regimes:
            mu_mix_full[i] += p_dl_full[i][k] * mu_hat[(i, k)]
    for i in assets:
        for j in assets:
            for k in regimes:
                sigma_mix_full[i][j] += p_dl_full[i][k] * p_dl_full[j][k] * sigma_hat[(i, j, k)]
    for i in assets:
        for j in assets:
            sym = 0.5 * (sigma_mix_full[i][j] + sigma_mix_full[j][i])
            sigma_mix_full[i][j] = sym
            sigma_mix_full[j][i] = sym

    # Cortar al rango [t_start..T] para output.
    T_vals = [t for t in T_full if t >= t_start]
    mu_mix = {i: mu_mix_full[i].loc[t_start:T] for i in assets}
    sigma_mix = {
        i: {j: sigma_mix_full[i][j].loc[t_start:T] for j in assets}
        for i in assets
    }
    p_dl = {i: p_dl_full[i].loc[t_start:T] for i in assets}
    r_sliced = {i: r_hist[i].loc[t_start:T] for i in assets}

    # V_max consistente con moments_window (si se restringio mu_hat al train,
    # V_max tambien debe usar solo train para no filtrar varianza futura).
    if moments_window is None:
        V_max = base_ctx["V_max"]
    else:
        a, b = moments_window
        V_max = float(r_hist[V_MAX_REF_ASSET].loc[a:b].var() * V_MAX_BUFFER)

    return {
        "mu_mix":          mu_mix,
        "sigma_mix":       sigma_mix,
        "T_vals":          T_vals,
        "nT":              len(T_vals),
        "assets":          assets,
        "c_base":          base_ctx["c_base"],
        "w0":              base_ctx["w0"],
        "Capital_inicial": base_ctx["Capital_inicial"],
        "V_max":           V_max,
        "r":               r_sliced,
        "p_dl":            p_dl,
        "mu_hat":          mu_hat,
        "sigma_hat":       sigma_hat,
        "t_start":         t_start,
        "moments_window":  moments_window,
    }


def build_shared_scenarios(
    ensemble_nn: LoadedModel,
    initial_window: np.ndarray,
    w_ref: np.ndarray,
    c_base: np.ndarray | None = None,
    N: int = N_CANDIDATES,
    T: int = T_HORIZON,
    n_scenarios: int = N_SCENARIOS,
    seed: int = SCENARIO_SEED,
    position: str = "median",
) -> np.ndarray:
    """Genera escenarios compartidos para evaluacion ex-post (FO-aligned).

    1. Usa el ensemble (promedio de logits de las 15 NNs) para rolling forward
       desde initial_window.
    2. Reduce N candidatos a n_scenarios representativos rankeando por
       capital terminal bajo rebalanceo a w_ref CON COSTOS DE TRANSACCION
       (ec. 19 del PDF, FO-aligned).

    Si c_base es None, se omite la penalizacion por costos y se rankea solo
    por retorno acumulado de portafolio (comportamiento legacy).

    Returns: (n_scenarios, T, A) — los mismos escenarios para TODAS las celdas,
    necesario para que V_best_s y la formula de regret (ec. 22) esten bien
    definidos.
    """
    candidates = generate_candidate_scenarios(
        ensemble_nn, initial_window, N=N, T=T, seed=seed,
    )                                                          # (N, T, A)
    if c_base is None:
        scenarios = reduce_by_portfolio_return(
            candidates, w_ref=w_ref,
            n_quintiles=n_scenarios, position=position,
        )
    else:
        scenarios = reduce_by_fo_outcome(
            candidates, w_ref=w_ref, c_base=c_base,
            n_quintiles=n_scenarios, position=position,
        )
    return scenarios                                            # (n_S, T, A)


def run_per_cell_regret_grid(
    per_cell_contexts: Dict[Tuple[float, float], dict],
    scenarios: np.ndarray,
    lambda_grid: Sequence[float],
    m_grid: Sequence[float],
) -> Tuple[pd.DataFrame, dict]:
    """Orquesta fase 4-5 del pipeline per-cell.

    Para cada g = (lam, m):
      - Resuelve solve_portfolio con per_cell_contexts[g] (su mu_mix, sigma_mix
        propios) y los parametros (lam, m) -> w_g, u_g, v_g, z_g.
      - Simula V[g, s] sobre los escenarios COMPARTIDOS.

    Returns: (V_df long-format, policies dict).
    """
    n_S = scenarios.shape[0]
    rows = []
    policies = {}
    total = len(lambda_grid) * len(m_grid)
    run = 0

    for lam in lambda_grid:
        for cm in m_grid:
            run += 1
            ctx_g = per_cell_contexts[(lam, cm)]
            assets = ctx_g["assets"]
            T_vals = ctx_g["T_vals"]
            c_base = ctx_g["c_base"]
            C0     = ctx_g["Capital_inicial"]

            print(f"  [{run:>2}/{total}] lambda={lam:.2f}  m={cm:.2f} ...",
                  end=" ", flush=True)
            z, w_sol, u_sol, v_sol, _ = solve_portfolio(
                ctx_g, lambda_riesgo=lam, costo_mult=cm,
            )
            policies[(lam, cm)] = (w_sol, u_sol, v_sol, z)

            Vs = []
            for s in range(n_S):
                cap = simulate_capital_on_scenario(
                    w_sol, u_sol, v_sol, scenarios[s],
                    assets, c_base, C0, T_vals,
                )
                Vs.append(cap[T_vals[-1]])
                rows.append({"lambda": lam, "m": cm, "s": s,
                             "V": Vs[-1], "z": z})
            print(f"z={z:.4f}  V=[${min(Vs):,.0f} .. ${max(Vs):,.0f}]")

    return pd.DataFrame(rows), policies


def pdl_dispersion_diagnostic(
    nns: Dict[Tuple[float, float], LoadedModel],
    data_dir: Path = DATA_DIR,
    T: int = T_HORIZON,
) -> pd.DataFrame:
    """Diagnostico: dispersion de p_dl(t) entre las 15 NNs por celda.

    Si las 15 NNs convergen al mismo modelo, p_dl es identico => el paradigma
    per-cell colapsa al equivalente legacy (1 NN). Esta funcion expone si hay
    heterogeneidad real entre celdas.

    Returns: DataFrame con columnas (asset, t, mean, std, min, max) de p_bull
    a lo largo de las 15 celdas.
    """
    base_ctx = load_market_data(str(Path(data_dir)))
    assets = list(base_ctx["assets"])
    r_hist = base_ctx["r"]
    H_any  = next(iter(nns.values())).config.H
    returns_history = np.stack(
        [r_hist[i].sort_index().values[:T] for i in assets], axis=1,
    ).astype(np.float32)

    # p_bull(t, asset) por NN (G celdas, T pasos, A activos)
    p_stack = []
    for nn in nns.values():
        p_bull_arr = predict_pbull_walking(nn, returns_history, T)  # (T, A)
        p_stack.append(p_bull_arr)
    arr = np.stack(p_stack, axis=0)                                  # (G, T, A)

    rows = []
    for ai, asset in enumerate(assets):
        for t in range(T):
            col = arr[:, t, ai]
            rows.append({
                "asset": asset, "t": t + 1,
                "mean":  float(col.mean()), "std": float(col.std()),
                "min":   float(col.min()),  "max": float(col.max()),
            })
    return pd.DataFrame(rows)


# ================================================================
# 8) Bloque principal
# ================================================================

if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("REGRET-GRID — DL -> optimizador -> seleccion (lambda, m)")
    print("=" * 70)
    print("Cargando datos y construyendo contexto DL ...")
    ctx = build_dl_context(
        data_dir=DATA_DIR,
        checkpoint_path=CHECKPOINT_PATH,
        T=T_HORIZON,
        N_candidates=N_CANDIDATES,
        n_scenarios=N_SCENARIOS,
        seed=SCENARIO_SEED,
        summary_asset=SUMMARY_ASSET,
    )
    print(f"  Assets     : {ctx['assets']}")
    print(f"  T          : {ctx['nT']} periodos forward (t1..t{ctx['nT']})")
    print(f"  Scenarios  : {ctx['scenarios'].shape} (S, T, A)")
    for i in ctx["assets"]:
        col = ctx["p_dl"][i]["bull"]
        print(f"  p_bull {i:<7}: min={col.min():.3f}  max={col.max():.3f}  "
              f"mean={col.mean():.3f}")

    lambda_grid = list(LAMBDA_GRID)
    m_grid      = list(M_GRID)

    print("\n" + "-" * 70)
    print(f"Corriendo {len(lambda_grid)}x{len(m_grid)}="
          f"{len(lambda_grid) * len(m_grid)} puntos x "
          f"{ctx['scenarios'].shape[0]} escenarios")
    print("-" * 70)
    V_df, policies = run_regret_grid(ctx, lambda_grid, m_grid)

    res = compute_regret_and_select(V_df)

    print("\n" + "=" * 70)
    print("RESULTADOS")
    print("=" * 70)

    print("\n--- V[g, s] — capital terminal por (lambda, m) y escenario ---")
    print(res["V_table"].to_string(float_format="${:,.2f}".format))

    print("\n--- R[g, s] = V_best_s - V[g, s] ---")
    print(res["R_table"].to_string(float_format="${:,.2f}".format))

    print("\n--- Resumen de regret por g ---")
    print(res["regret_summary"].to_string(float_format="${:,.2f}".format))

    lam_m, m_m = res["g_mean"]
    lam_w, m_w = res["g_worst"]
    C0 = ctx["Capital_inicial"]
    V_mean_row  = res["V_table"].loc[(lam_m, m_m)]
    V_worst_row = res["V_table"].loc[(lam_w, m_w)]
    print("\n--- Seleccion de g* ---")
    print(f"  g*_mean  (ec. 23): lambda={lam_m:.2f}  m={m_m:.2f}  "
          f"mean_regret=${res['g_mean_metric']:,.2f}")
    print(f"      V: mean=${V_mean_row.mean():>12,.2f}  "
          f"worst=${V_mean_row.min():>12,.2f}  "
          f"best=${V_mean_row.max():>12,.2f}  "
          f"(capital inicial=${C0:,.2f})")
    print(f"      retorno promedio sobre escenarios = {V_mean_row.mean()/C0 - 1:+.2%}")
    print(f"  g*_worst (ec. 24): lambda={lam_w:.2f}  m={m_w:.2f}  "
          f"worst_regret=${res['g_worst_metric']:,.2f}")
    print(f"      V: mean=${V_worst_row.mean():>12,.2f}  "
          f"worst=${V_worst_row.min():>12,.2f}  "
          f"best=${V_worst_row.max():>12,.2f}  "
          f"(capital inicial=${C0:,.2f})")
    print(f"      retorno en el peor escenario     = {V_worst_row.min()/C0 - 1:+.2%}")

    # --- Persistencia ---
    out_V = RESULTS_DIR / "regret_grid_results.csv"
    V_df.to_csv(out_V, index=False)
    print(f"\n  V_df (long)           : {out_V}")

    out_R = RESULTS_DIR / "regret_table.csv"
    res["R_table"].to_csv(out_R)
    print(f"  Tabla de regret       : {out_R}")

    out_summary = RESULTS_DIR / "regret_summary.csv"
    res["regret_summary"].to_csv(out_summary)
    print(f"  Resumen por g         : {out_summary}")

    # --- Plot capital bajo g*_mean ---
    w_star, u_star, v_star, _z = policies[(lam_m, m_m)]
    plot_capital_curves(
        w_star, u_star, v_star, ctx,
        title=f"Capital por escenario con g*_mean (lambda={lam_m:.2f}, m={m_m:.2f})",
        out_path=RESULTS_DIR / "regret_capital_curves.png",
    )

    # --- Backtest historico: OPT vs Naive vs Regret-Grid g*_mean ---
    run_historical_backtest(
        w_star, u_star, v_star, lam_m, m_m,
        V_mean_row=V_mean_row,
        n_scenarios=ctx["scenarios"].shape[0],
        data_dir=DATA_DIR,
        out_path=RESULTS_DIR / "evolucion_capital.png",
    )

    print("\n" + "=" * 70)
    print("Done.")
    print("=" * 70)
