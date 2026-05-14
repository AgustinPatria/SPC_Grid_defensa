# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

SPC_Grid3 implements a portfolio optimization pipeline that:

1. Ports a GAMS model (`ps.gms`, media-varianza with transaction costs and weekly rebalancing) to Python using **GAMSPy + IPOPT**.
2. Extends it with a **Deep Learning layer** (quantile LSTM over weekly returns) that produces forward regime probabilities `p_bull(t)` and scenario trajectories.
3. Performs **regret-based selection** of the `(lambda, m)` hyperparameters of the optimizer over a grid `G = Λ × M`, using the DL-generated scenarios for ex-post capital simulation.

Codebase language is Spanish in docstrings, variable names, and CLI output. References like "PDF ec. 14" or "seccion 2.5" in docstrings point to the project's reference PDF (not in the repo). Follow the same Spanish style when editing.

## Commands

Python 3.11 is used. On Windows the interpreter is typically
`C:/Users/aunanue/AppData/Local/Programs/Python/Python311/python.exe` — use plain `python` in bash commands when the alias is set, otherwise the absolute path above.

Install deps:
```
python -m pip install -r requirements.txt
```
`gamspy` also requires a GAMS installation with the IPOPT solver available; the optimizer will fail at import/solve time without it.

Run the three main entrypoints from the project root (they rely on relative paths via `config.py`):

```
python basemodelGAMS.py          # base GAMS port: solve + sensitivity grid + plots
python Regret_Grid.py            # DL -> solve per g -> regret selection + results in resultados/
python dl/prediccion_deciles.py  # train the quantile LSTM -> models/decile_predictor.pt
python verify_optimum.py         # sanity check: z(IPOPT) vs naive policies at lambda=0.10
```

Diagnostic / inspection scripts (each writes PNGs + CSVs next to itself under `inspeccion/<topic>/`):

```
python inspeccion/prediccion_deciles/inspeccionar_deciles.py
python inspeccion/regimen_predicted/inspeccionar_regimen.py
python inspeccion/generador_escenarios/inspeccionar_escenarios.py
python inspeccion/regret_grid/inspeccionar_regret.py
```

There is no test suite, linter, or build configured.

## Architecture

### Central config (`config.py`)
All knobs live in a single module at the project root. Every other file — including `dl/` submodules — imports from there. Constants in UPPERCASE are global defaults; they are wrapped into dataclasses (`DLConfig`, `ScenarioConfig`, `OptConfig`, `RegretGridConfig`) so callers can override them at runtime without editing the file. When adding a hyperparameter, add the UPPERCASE default *and* a field on the relevant dataclass.

Key paths derived from `PROJECT_ROOT`: `DATA_DIR` (CSV inputs), `MODELS_DIR` (checkpoints), `RESULTS_DIR` (regret outputs). Checkpoint lives at `models/decile_predictor.pt`.

### Data flow

```
  data/                               models/                    resultados/
  ├─ ret_semanal_{spx,cmc200}.csv     decile_predictor.pt        regret_*.csv / .png
  └─ prob_{spx,cmc200}.csv                 ▲
         │                                 │
         ▼                                 │
  load_market_data ──► mu_hat, sigma_hat   │
  (basemodelGAMS)       per (i, j, regime) │
         │                                 │
         ├── mu_mix/sigma_mix (historic p) ─────► solve_portfolio ──► z, w, u, v
         │                                 │       (GAMSPy + IPOPT)        │
         │                                 │                               ▼
         │                                 │                   simulate_capital_*
         │                                 │
  dl/prediccion_deciles ─train─────────────┘
         │ (QuantileLSTM + pinball loss)
         ▼
  dl/regimen_predicted.regimen_from_deciles  → p_bull(t) forward
  dl/generador_escenarios.generate_*         → N candidate trajectories,
                                                reduce to 5 quintile-representatives
         │
         ▼
  Regret_Grid.build_dl_context → ctx with DL mu_mix/sigma_mix + scenarios
  Regret_Grid.run_regret_grid  → one solve per g, simulate V[g, s] per scenario
  Regret_Grid.compute_regret_and_select → g*_mean (avg regret) + g*_worst (minimax)
```

### Optimizer (`basemodelGAMS.py`)
`solve_portfolio(theta, context, lambda_riesgo, costo_mult)` builds the GAMSPy model:
- Vars: `w(i,t)`, `u(i,t)` (compras), `v(i,t)` (ventas), `z` (objective).
- Constraints: `sum_i w(i,t) = 1`, rebalance identity `w(i,t) - w(i,t-1) = u - v` for `t>1`, anchor `w(i,t1) - w0 = u - v`.
- Objective: `max Σ_t [ Σ_i w·mu·theta  -  λ·Σ_{ij} w_i·w_j·σ  -  Σ_i c_eff·(u+v) ]` where `c_eff = c_base * costo_mult`.
- `simulate_capital_opt` always uses `c_base` (no `costo_mult`) — the multiplier only penalizes ex-ante inside the FO.

Time labels are strings `"t1".."t163"` in GAMSPy but the Python solution dicts use integer keys `(asset, int_t)`. Keep that convention when extending — the conversion is in `_records_to_dict`.

### DL layer (`dl/`)
- `prediccion_deciles.py`: windowed (`H`) LSTM emitting `(B, n_assets, n_deciles)`; trained with pinball loss + chronological split + seed averaging. `save_checkpoint` / `load_checkpoint` persist everything needed for inference (state dict, config, standardizer mean/std).
- `regimen_predicted.py`: converts decile predictions to `p_bull(t) = fraction of deciles ≥ BULL_THRESHOLD`, preserving `p_bull + p_bear = 1`.

### Definicion de regimen — caveat importante

El **modulo DL** define el regimen via `BULL_THRESHOLD = 0.0`: una observacion es "bull"
si `retorno >= 0`. Esto vive en `dl/regimen_predicted.py` y en el rollout del LSTM.

Los archivos `data/prob_*.csv` (probabilidades historicas), en cambio, **NO siguen esa
regla**. Fueron generados por un proceso externo (probablemente HMM sobre volatilidad u
otro modelo de cambio de regimen) y empiricamente:
  - `accuracy(p_bull > 0.5  ↔  r >= 0)` ≈ 52-56% (apenas mejor que azar).
  - `mu_hat(bear, CMC200) = +0.87%/sem` y `mu_hat(bull, CMC200) = +0.28%/sem` —
    el regimen "bear" del CSV tiene MAYOR retorno medio que el "bull" para CMC200,
    porque la definicion del CSV no es "signo del retorno".

Implicaciones:
  - `load_market_data` y la antigua formulacion de `mu_mix = p_dl × mu_hat`
    (pre-unificacion) heredan esa definicion de regimen no-alineada con el LSTM.
  - **Tras la unificacion** (`build_dl_context` actual), `mu_mix` y `sigma_mix` se
    derivan de los candidatos del LSTM y **NO usan `p_hist` ni `mu_hat`**. El pipeline
    DL es inmune a esta inconsistencia.
  - `OPT base` (benchmark con `load_market_data`) sigue usando `p_hist` tal cual.
- `generador_escenarios.py`: two-stage — generate N trajectories by rolling the window forward and sampling the same decile index across assets per step, then reduce to 5 quintile-medians ranked by cumulative return of `SUMMARY_ASSET` (default SPX).
- `dl/__init__.py` aliases `dl.config → config` so older checkpoints pickled under `dl.config.DLConfig` still deserialize.

### Regret grid (`Regret_Grid.py`)
`build_dl_context` is the bridge: it mixes **historical** `mu_hat/sigma_hat` (per `(i, j, regime)`) with **DL-forecasted** `p_dl(t)` to build `mu_mix/sigma_mix(t)` over the forward horizon (t=1..T_HORIZON), and generates the 5 representative scenarios. The returned dict is drop-in compatible with `solve_portfolio` (same shape as `load_market_data`'s output) plus extra keys `scenarios` and `p_dl`.

`run_regret_grid` runs one solve per `g = (λ, m)` (15 by default) and, for each, simulates `V[g, s]` on all 5 scenarios. `compute_regret_and_select` picks `g*_mean` (argmin average regret, ec. 23) and `g*_worst` (argmin worst-case regret, ec. 24).

## Conventions and gotchas

- Never hardcode paths, hyperparameters, or asset lists in downstream modules — add them to `config.py` first.
- The PDF referenced throughout docstrings is external. If something is labelled "ec. N" or "seccion X", treat it as spec — don't rename or reformulate the math without checking the reference.
- Spanish accented characters appear in strings (`Replica`, `óptima`, `rebalanceo`). Preserve encoding when editing.
- `SOLVER = "IPOPT"` is the only tested solver; `solve_portfolio` raises on anything other than `OptimalLocal`/`OptimalGlobal` status.
- `verify_optimum.py` exists because IPOPT returns `optimal_local` — it evaluates the objective for several naive policies to sanity-check against the IPOPT solution.
- Inspection scripts insert the project root into `sys.path` via `Path(__file__).resolve().parent.parent.parent`, so they must keep their two-level-deep location under `inspeccion/<topic>/`.
