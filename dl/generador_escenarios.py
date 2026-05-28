"""Generador de escenarios de retornos futuros (PDF sección 2.5).

Pipeline en dos pasos:

1. `generate_candidate_scenarios`: desde la última ventana observada, simula
   N trayectorias (por defecto N = 1000) de largo T (por defecto `T_HORIZON`,
   que coincide con t1..t163 del GAMS). En cada paso:
     - se predicen los deciles con el LSTM congelado;
     - se muestrea uniformemente un nivel q ∈ Q **comun a todos los activos**
       (lectura literal del PDF). El q independiente por activo se probo y
       descartado: rompia la correlacion SPX-CMC a ~0 (historico +0.31), lo
       que generaba escenarios "imposibles" (SPX -40% mientras CMC +1000%)
       que secuestraban la seleccion por regret. Mismo q da corr ~0.85
       (mas cerca del historico que 0) y escenarios ordenados de peor a
       mejor para ambos activos;
     - se fija r_cand_{i,t} = r_hat^(q_i)_{i,t};
     - se rola la ventana (se descarta el retorno más viejo y se agrega el
       nuevo) para poder predecir el paso siguiente.

2. `reduce_to_representatives`: ordena los N candidatos por un resumen
   económico (retorno acumulado del activo de referencia — SPX por defecto),
   los parte en 5 quintiles del peor al mejor, y elige 1 escenario mediano
   por quintil. Resultado: S con |S| = 5 trayectorias explicables que
   alimentan el regret-grid de ps.gms."""

from typing import Optional

import numpy as np
import torch

from config import T_HORIZON
from .prediccion_deciles import LoadedModel


def generate_candidate_scenarios(
    model: LoadedModel,
    initial_window: np.ndarray,
    N: int = 1000,
    T: int = T_HORIZON,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Genera N trayectorias de largo T partiendo de `initial_window`.

    initial_window: (H, n_assets)
    return:         (N, T, n_assets)
    """
    cfg = model.config
    if initial_window.shape != (cfg.H, cfg.n_assets):
        raise ValueError(
            f"initial_window shape {initial_window.shape} != (H={cfg.H}, A={cfg.n_assets})"
        )

    rng = np.random.default_rng(seed)
    A   = cfg.n_assets
    Q   = cfg.n_quantiles

    # Una ventana independiente por escenario (todas parten iguales).
    windows   = np.tile(initial_window.astype(np.float32), (N, 1, 1))   # (N, H, A)
    scenarios = np.empty((N, T, A), dtype=np.float32)

    for t in range(T):
        # Predice deciles para los N escenarios en paralelo (ensemble de seeds).
        x = ((windows - model.mean) / model.std).astype(np.float32)
        x_tensor = torch.from_numpy(x)
        with torch.no_grad():
            preds_list = [net(x_tensor).numpy() for net in model.nets]  # K * (N, A, Q)
        preds = np.mean(np.stack(preds_list, axis=0), axis=0)
        # Garantizamos monotonicidad: q_idx=0 debe ser el peor caso para cada activo.
        preds = np.sort(preds, axis=-1)

        # Mismo q para todos los activos en cada paso (lectura literal del PDF).
        # El q independiente por activo se descarto: daba corr SPX-CMC ~0 y
        # generaba el escenario artefacto (SPX cae / CMC explota) que
        # secuestraba la seleccion por regret. Mismo q => corr ~0.85.
        q_idx = np.repeat(rng.integers(low=0, high=Q, size=(N, 1)),
                          A, axis=1)                                    # (N, A) comonotonia
        r_t   = np.take_along_axis(
            preds, q_idx[:, :, None], axis=2
        ).squeeze(-1)                                                   # (N, A)

        scenarios[:, t, :] = r_t

        # Roll de la ventana: desecha el retorno mas viejo, agrega el recien muestreado.
        windows = np.concatenate([windows[:, 1:, :], r_t[:, None, :]], axis=1)

    return scenarios


def reduce_to_representatives(
    scenarios: np.ndarray,
    summary_asset_idx: int = 0,
    n_quintiles: int = 5,
    position: str = "median",
) -> np.ndarray:
    """
    Reduce los N candidatos a n_quintiles representativos.

    scenarios:         (N, T, n_assets)
    summary_asset_idx: indice del activo usado como resumen economico
                       (PDF ec. 17, default SPX = 0).
    position:          que escenario tomar dentro de cada quintil.
                       - "median" (default, ejemplo del PDF): el mediano del bucket
                       - "min":    el peor del bucket (mas pesimista en cada quintil)
                       - "max":    el mejor del bucket
    return:            (n_quintiles, T, n_assets)
    """
    if scenarios.ndim != 3:
        raise ValueError(f"scenarios debe ser (N, T, A); recibi {scenarios.shape}")
    N = scenarios.shape[0]
    if N < n_quintiles:
        raise ValueError(f"N={N} insuficiente para {n_quintiles} quintiles")
    if position not in {"median", "min", "max"}:
        raise ValueError(f"position invalido: {position!r}")

    # Retorno acumulado del activo resumen por escenario (PDF ec. 17).
    cum   = np.prod(1.0 + scenarios[:, :, summary_asset_idx], axis=1) - 1.0  # (N,)
    order = np.argsort(cum)                                                  # peor -> mejor

    # Particion en n_quintiles (el ultimo absorbe el remanente si N no divide).
    edges = np.linspace(0, N, n_quintiles + 1, dtype=int)
    reps  = []
    for k in range(n_quintiles):
        lo, hi = edges[k], edges[k + 1]
        bucket = order[lo:hi]
        if   position == "median": idx = bucket[len(bucket) // 2]
        elif position == "min":    idx = bucket[0]
        else:                      idx = bucket[-1]   # max
        reps.append(scenarios[idx])

    return np.stack(reps, axis=0)                                            # (n_q, T, A)


def reduce_by_portfolio_return(
    scenarios: np.ndarray,
    w_ref: np.ndarray,
    n_quintiles: int = 5,
    position: str = "median",
) -> np.ndarray:
    """Reduce N candidatos a n_quintiles representativos rankeando por
    retorno acumulado de PORTAFOLIO bajo una politica de referencia w_ref.

    Variante PDF-aligned del ranking por activo unico (`reduce_to_representatives`):
    el "resumen economico" del PDF sec 2.5 deja de ser el retorno del activo
    de referencia (ej. SPX) y pasa a ser el retorno del portafolio entero bajo
    w_ref. Esto desliga la seleccion de escenarios del activo de mas peso en
    el universo y la alinea con la FO.

    scenarios:   (N, T, n_assets)
    w_ref:       (n_assets,) — pesos del portafolio de referencia (sum=1, no
                 necesariamente; el ranking es invariante a escala).
    n_quintiles: numero de quintiles (default 5).
    position:    "median" (default), "min" o "max" dentro de cada quintil.

    return:      (n_quintiles, T, n_assets) — representativos compartidos.
    """
    if scenarios.ndim != 3:
        raise ValueError(f"scenarios debe ser (N, T, A); recibi {scenarios.shape}")
    N, _T, A = scenarios.shape
    if N < n_quintiles:
        raise ValueError(f"N={N} insuficiente para {n_quintiles} quintiles")
    if w_ref.shape != (A,):
        raise ValueError(f"w_ref shape {w_ref.shape} != (A={A},)")
    if position not in {"median", "min", "max"}:
        raise ValueError(f"position invalido: {position!r}")

    # Retorno de portafolio por paso: r_port[n, t] = sum_i w_ref[i] * scenarios[n, t, i].
    r_port = np.einsum("nti,i->nt", scenarios, w_ref.astype(np.float32))      # (N, T)
    # Log-retorno acumulado (aditivo en el tiempo, estable numericamente).
    # Clip evita log(<=0) en escenarios catastroficos sinteticos.
    log_ret_cum = np.log(np.clip(1.0 + r_port, 1e-8, None)).sum(axis=1)        # (N,)
    order = np.argsort(log_ret_cum)                                            # peor -> mejor

    edges = np.linspace(0, N, n_quintiles + 1, dtype=int)
    reps  = []
    for k in range(n_quintiles):
        lo, hi = edges[k], edges[k + 1]
        bucket = order[lo:hi]
        if   position == "median": idx = bucket[len(bucket) // 2]
        elif position == "min":    idx = bucket[0]
        else:                      idx = bucket[-1]
        reps.append(scenarios[idx])

    return np.stack(reps, axis=0)                                              # (n_q, T, A)


def reduce_by_fo_outcome(
    scenarios: np.ndarray,
    w_ref: np.ndarray,
    c_base: np.ndarray,
    n_quintiles: int = 5,
    position: str = "median",
) -> np.ndarray:
    """Reduce N candidatos rankeando por capital terminal bajo rebalanceo a w_ref
    con costos de transaccion (FO-aligned, sin termino lambda*var).

    Sigue la ec. 19 del PDF (`simulate_naive_rb`):
      r_port[n, t]   = sum_i w_ref[i] * scenarios[n, t, i]
      w_drift[n,t,i] = w_ref[i] * (1 + scenarios[n,t,i]) / (1 + r_port[n,t])
      cost[n, t]     = sum_i c_base[i] * |w_ref[i] - w_drift[n, t, i]|
      cap[n, t+1]    = cap[n, t] * (1 + r_port[n, t] - cost[n, t])

    El agente de referencia mantiene w_ref via rebalanceo semanal, incurriendo
    costos realistas. Esto refleja el "outcome economico" pedido por la FO en
    el sentido de "retorno menos costos" — la parte que SI se puede medir sobre
    una trayectoria realizada (el termino lambda*var requiere expectativas y
    se omite del ranking).

    scenarios:    (N, T, A) — candidatos del rollout
    w_ref:        (A,)      — pesos de referencia (ej. w0 = 50/50)
    c_base:       (A,)      — costos de transaccion por activo (config.C_BASE)
    n_quintiles:  numero de quintiles (default 5)
    position:     'median' / 'min' / 'max' dentro de cada quintil

    return:       (n_quintiles, T, A) — representativos compartidos
    """
    if scenarios.ndim != 3:
        raise ValueError(f"scenarios debe ser (N, T, A); recibi {scenarios.shape}")
    N, _T, A = scenarios.shape
    if N < n_quintiles:
        raise ValueError(f"N={N} insuficiente para {n_quintiles} quintiles")
    if w_ref.shape != (A,):
        raise ValueError(f"w_ref shape {w_ref.shape} != (A={A},)")
    if c_base.shape != (A,):
        raise ValueError(f"c_base shape {c_base.shape} != (A={A},)")
    if position not in {"median", "min", "max"}:
        raise ValueError(f"position invalido: {position!r}")

    w_ref32 = w_ref.astype(np.float32)
    c32     = c_base.astype(np.float32)

    # Retorno de portafolio por paso (con w_ref constante al inicio del periodo).
    r_port = np.einsum("nti,i->nt", scenarios, w_ref32)                  # (N, T)

    # Drift de los pesos al final del periodo (antes de rebalancear).
    # Evita division por 0 cuando r_port = -1 (escenario catastrofico).
    denom = np.clip(1.0 + r_port, 1e-8, None)                            # (N, T)
    w_drift = (w_ref32[None, None, :] * (1.0 + scenarios)) / denom[:, :, None]  # (N, T, A)

    # Costo de rebalancear de w_drift de vuelta a w_ref: c_i * |w_ref - w_drift|.
    cost = np.einsum("nta,a->nt", np.abs(w_ref32[None, None, :] - w_drift), c32)  # (N, T)

    # Capital terminal compuesto: cap_T = prod_t (1 + r_port - cost).
    # log-cap aditivo para estabilidad numerica.
    growth     = np.clip(1.0 + r_port - cost, 1e-8, None)                # (N, T)
    log_cap    = np.log(growth).sum(axis=1)                              # (N,)
    cap_terminal = np.exp(log_cap)                                       # (N,)

    order = np.argsort(cap_terminal)                                     # peor -> mejor

    edges = np.linspace(0, N, n_quintiles + 1, dtype=int)
    reps  = []
    for k in range(n_quintiles):
        lo, hi = edges[k], edges[k + 1]
        bucket = order[lo:hi]
        if   position == "median": idx = bucket[len(bucket) // 2]
        elif position == "min":    idx = bucket[0]
        else:                      idx = bucket[-1]
        reps.append(scenarios[idx])

    return np.stack(reps, axis=0)                                         # (n_q, T, A)


def generate_representative_scenarios(
    model: LoadedModel,
    initial_window: np.ndarray,
    N: int = 1000,
    T: int = T_HORIZON,
    n_quintiles: int = 5,
    summary_asset: str = "SPX",
    seed: Optional[int] = None,
    position: str = "median",
) -> np.ndarray:
    """Pipeline completo: genera N candidatos y devuelve n_quintiles representativos."""
    assets = tuple(model.config.assets)
    if summary_asset not in assets:
        raise ValueError(f"summary_asset {summary_asset!r} no esta en {assets}")
    idx = assets.index(summary_asset)

    candidates = generate_candidate_scenarios(
        model, initial_window, N=N, T=T, seed=seed,
    )
    return reduce_to_representatives(
        candidates, summary_asset_idx=idx, n_quintiles=n_quintiles, position=position,
    )
