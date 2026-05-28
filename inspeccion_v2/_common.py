"""Utilidades compartidas para inspeccion_v2.

- Estilo identico a inspeccion/_common.py (save_fig / save_csv / out_dir).
- Funciones para cargar el pipeline una sola vez por capa:
    load_nns()             -> dict (lam, m) -> LoadedModel  (usa cache .pt)
    load_contexts_is()     -> dict (lam, m) -> ctx in-sample
    load_contexts_oos()    -> dict (lam, m) -> ctx out-of-sample
    load_scenarios_is()    -> (n_S, T, A) escenarios shared IS
    load_scenarios_oos()   -> (n_S, T_oos, A) escenarios shared OOS
    load_initial_window()  -> (H, A) ventana inicial IS (ultimas H del hist)
    load_initial_window_oos() -> (H, A) ventana inicial OOS (H antes del test)
"""
from pathlib import Path
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INSPECCION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = INSPECCION_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (  # noqa: E402
    DATA_DIR,
    DLConfig,
    LAMBDA_GRID,
    M_GRID,
    MODELS_DIR,
    N_CANDIDATES,
    N_SCENARIOS,
    SCENARIO_SEED,
    SPLIT,
    T_HORIZON,
    W0,
)
from Regret_Grid import (  # noqa: E402
    build_ensemble_model,
    build_per_cell_context,
    build_shared_scenarios,
    compute_regret_and_select,
    compute_test_start_t,
    load_market_data,
    run_per_cell_regret_grid,
    train_per_cell_nns,
)


# --------------------------------------------------------------- IO helpers

def out_dir(name: str) -> Path:
    d = INSPECCION_DIR / f"{name}_out"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_fig(fig, name: str, subdir: str) -> Path:
    p = out_dir(subdir) / f"{name}.png"
    fig.tight_layout()
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {p.relative_to(PROJECT_ROOT)}")
    return p


def save_csv(df: pd.DataFrame, name: str, subdir: str, index: bool = False) -> Path:
    p = out_dir(subdir) / f"{name}.csv"
    df.to_csv(p, index=index)
    print(f"  -> {p.relative_to(PROJECT_ROOT)}")
    return p


# --------------------------------------------------------------- Pipeline


def clear_grid_cache(verbose: bool = True) -> None:
    """Borra inspeccion_v2/_cache/ (pickles del regret grid).

    Las policies/V_df cacheadas dependen de las NNs concretas; al reentrenar
    o cambiar la grilla, quedan stale. Esta funcion garantiza que la proxima
    corrida de L4/L5 regenere todo.
    """
    if CACHE_DIR.exists():
        import shutil
        shutil.rmtree(CACHE_DIR)
        if verbose:
            print(f"  [retrain] cache pickle eliminado: "
                  f"{CACHE_DIR.relative_to(PROJECT_ROOT)}")


def load_nns(verbose: bool = True, force_retrain: bool = False):
    """Carga (o reentrena) las 15 NNs.

    force_retrain=True borra ademas el cache pickle de _cache/ (sino L4/L5
    leerian policies derivadas de las NNs viejas).
    """
    if force_retrain:
        clear_grid_cache(verbose=verbose)
        if verbose:
            print("  [retrain] re-entrenando 15 NNs (force_retrain=True) ...")
    elif verbose:
        print("  Cargando 15 NNs cacheadas ...")
    return train_per_cell_nns(
        lambda_grid=LAMBDA_GRID,
        m_grid=M_GRID,
        dl_config=DLConfig(),
        models_dir=MODELS_DIR,
        force_retrain=force_retrain,
    )


def load_contexts_is(nns: dict, mu_hat_source: str = "p_hist") -> dict:
    """Construye contextos in-sample por celda."""
    contexts = {}
    for g, nn in nns.items():
        contexts[g] = build_per_cell_context(
            nn=nn, data_dir=DATA_DIR, T=T_HORIZON,
            mu_hat_source=mu_hat_source,
        )
    return contexts


def load_contexts_oos(nns: dict, mu_hat_source: str = "p_hist") -> dict:
    """Construye contextos OOS por celda (split test del LSTM)."""
    H = next(iter(nns.values())).config.H
    t_test_start = compute_test_start_t(T=T_HORIZON, H=H, split=SPLIT)
    contexts = {}
    for g, nn in nns.items():
        contexts[g] = build_per_cell_context(
            nn=nn, data_dir=DATA_DIR, T=T_HORIZON,
            mu_hat_source=mu_hat_source,
            t_start=t_test_start,
            moments_window=(1, t_test_start - 1),
        )
    return contexts


def load_initial_window_is(ensemble) -> np.ndarray:
    """Ventana inicial IS = ultimas H semanas del historico."""
    base_ctx = load_market_data(str(DATA_DIR))
    assets = list(base_ctx["assets"])
    r_hist = base_ctx["r"]
    H = ensemble.config.H
    returns_history = np.stack(
        [r_hist[i].sort_index().values[:T_HORIZON] for i in assets], axis=1,
    ).astype(np.float32)
    return returns_history[-H:, :]


def load_initial_window_oos(ensemble) -> tuple[np.ndarray, int]:
    """Ventana inicial OOS = H semanas inmediatamente previas a t_test_start."""
    base_ctx = load_market_data(str(DATA_DIR))
    assets = list(base_ctx["assets"])
    r_hist = base_ctx["r"]
    H = ensemble.config.H
    t_test_start = compute_test_start_t(T=T_HORIZON, H=H, split=SPLIT)
    initial_window = np.stack(
        [r_hist[i].loc[t_test_start - H : t_test_start - 1].values
         for i in assets], axis=1,
    ).astype(np.float32)
    return initial_window, t_test_start


def w_ref_vector(assets) -> np.ndarray:
    return np.array([W0[i] for i in assets], dtype=np.float32)


def _c_base_vector(assets) -> np.ndarray:
    base_ctx = load_market_data(str(DATA_DIR))
    return np.array([base_ctx["c_base"][i] for i in assets], dtype=np.float32)


def build_scenarios_is(ensemble) -> np.ndarray:
    base_ctx = load_market_data(str(DATA_DIR))
    assets = list(base_ctx["assets"])
    initial_window = load_initial_window_is(ensemble)
    return build_shared_scenarios(
        ensemble_nn=ensemble,
        initial_window=initial_window,
        w_ref=w_ref_vector(assets),
        c_base=_c_base_vector(assets),
        N=N_CANDIDATES, T=T_HORIZON, n_scenarios=N_SCENARIOS,
        seed=SCENARIO_SEED,
    )


def build_scenarios_oos(ensemble) -> tuple[np.ndarray, int]:
    base_ctx = load_market_data(str(DATA_DIR))
    assets = list(base_ctx["assets"])
    initial_window, t_test_start = load_initial_window_oos(ensemble)
    T_oos = T_HORIZON - t_test_start + 1
    scenarios = build_shared_scenarios(
        ensemble_nn=ensemble,
        initial_window=initial_window,
        w_ref=w_ref_vector(assets),
        c_base=_c_base_vector(assets),
        N=N_CANDIDATES, T=T_oos, n_scenarios=N_SCENARIOS,
        seed=SCENARIO_SEED,
    )
    return scenarios, t_test_start


def hist_returns(assets, T: int = T_HORIZON) -> dict:
    """{asset: np.array (T,)} retornos historicos t=1..T."""
    base_ctx = load_market_data(str(DATA_DIR))
    return {i: base_ctx["r"][i].sort_index().values[:T].astype(np.float32)
            for i in assets}


# ----------------------------------------------- cache pickled policies/V_df

CACHE_DIR = INSPECCION_DIR / "_cache"


def _cache_path(label: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"grid_{label}.pkl"


def run_or_load_grid(label: str, nns: dict, force: bool = False) -> dict:
    """Corre el regret grid (IS o OOS) con cache pickle.

    label: 'is' o 'oos'.

    Devuelve dict con:
        contexts: dict {g: ctx}
        scenarios: np.ndarray
        policies: dict {g: (w_sol, u_sol, v_sol, z)}
        V_df: pd.DataFrame long-format
        res: output de compute_regret_and_select
        t_test_start: int (solo OOS, None en IS)
    """
    path = _cache_path(label)
    if path.exists() and not force:
        print(f"  [cache] cargando {path.name} ...")
        with open(path, "rb") as f:
            return pickle.load(f)

    print(f"  [run]   resolviendo grid {label.upper()} (15 solves GAMS) ...")
    ensemble = build_ensemble_model(nns)

    if label == "is":
        contexts = load_contexts_is(nns, mu_hat_source="p_hist")
        scenarios = build_scenarios_is(ensemble)
        t_test_start = None
    elif label == "oos":
        contexts = load_contexts_oos(nns, mu_hat_source="p_hist")
        scenarios, t_test_start = build_scenarios_oos(ensemble)
    else:
        raise ValueError(f"label invalido: {label}")

    V_df, policies = run_per_cell_regret_grid(
        contexts, scenarios, list(LAMBDA_GRID), list(M_GRID),
    )
    res = compute_regret_and_select(V_df)

    payload = {
        "contexts": contexts,
        "scenarios": scenarios,
        "policies": policies,
        "V_df": V_df,
        "res": res,
        "t_test_start": t_test_start,
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    print(f"  [run]   cache guardado en {path.name}")
    return payload
