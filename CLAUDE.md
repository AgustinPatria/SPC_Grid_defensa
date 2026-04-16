# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**ProyectoV3** is a Python-based dynamic portfolio optimization model (thesis project). It implements a multi-period, regime-aware portfolio optimizer using Gurobi's quadratic programming solver, applied to two assets: SPX (S&P 500) and CMC200 (crypto index).

## Running the Project

```bash
# Run the main script directly (requires Gurobi license)
python basemodel.py
```

There are no build, lint, or test commands — this is a research/thesis script with no formal tooling infrastructure.

## Dependencies

- `gurobipy` — commercial solver, requires a valid Gurobi license installed locally
- `pandas`, `numpy` — data processing
- `pathlib` — path handling (stdlib)

No `requirements.txt` or `pyproject.toml` exists; install dependencies manually.

## Architecture

All logic lives in `basemodel.py` with three sections:

### 1. `load_market_data(base_dir_str)`
Reads 4 CSV files and computes mixed-regime market statistics:
- `prob_spx.csv` / `prob_cmc200.csv` — bear/bull regime probabilities per time period `t`
- `ret_semanal_spx.csv` / `ret_semanal_cmc200.csv` — weekly returns per `t`
- Computes per-regime mean (`mu`) and covariance (`sigma`), then mixes them using regime probabilities into `mu_mix[i,t]` and `sigma_mix[i,j,t]`
- Returns a `context` dict with all static data needed by the solver

### 2. `solve_portfolio_gurobi(context, theta, lambda_riesgo, costo_mult, mip_gap)`
Gurobi QP model over 163 weekly time periods:
- **Decision variables**: `w[i,t]` (weights), `u[i,t]` (buys), `v[i,t]` (sells)
- **Objective**: maximize expected return (scaled by `theta` sentiment multipliers) minus quadratic risk penalty minus transaction costs
- **Key constraints**: portfolio sums to 1 at each `t`; rebalancing identity `w[i,t] - w[i,t-1] = u[i,t] - v[i,t]`
- Returns optimal weights and objective value

### 3. Main block
Runs two scenarios to compare:
- **Neutral**: `theta = 1.0` for all assets
- **Bullish SPX**: `theta["SPX"] = 1.1`

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `theta` | `{SPX: 1.0, CMC200: 1.0}` | Sentiment multipliers on expected returns |
| `lambda_riesgo` | `0.10` | Risk aversion coefficient |
| `costo_mult` | `1.0` | Transaction cost multiplier |
| `c_base` | `{SPX: 0.005, CMC200: 0.010}` | Base transaction costs (0.5%, 1.0%) |
| `w0` | `{SPX: 0.5, CMC200: 0.5}` | Initial portfolio weights |


ProyectoV3/
├── data/                  CSVs (prob_*, ret_semanal_*)
├── results/               outputs
├── legacy/                basemodel original, GAMS, regret_grid_dl viejo
│
├── optimizer/             CAPA 1 — §1 PDF (GAMS equivalente, Gurobi)
│   ├── data_loader.py     §1.2–1.3  mu_mix, sigma_mix
│   ├── model.py           §1.4      QP media-varianza con costos
│   ├── simulation.py      §1.5      capital ex-post (opt, BH, RB)
│   └── sensitivity.py     grid interno λ×m (GAMS §5)
│
├── prediction/            CAPA 2 — §2 PDF (Deep Learning) — stubs
│   ├── dataset.py         §2.2  ventanas H, split cronológico
│   ├── model_dl.py        §2.3  red cuantil (LSTM/Transformer)
│   ├── train.py           §2.3  pinball loss + early stopping
│   ├── regime_probs.py    §2.4  cuantiles → p_{i,k,t}
│   └── scenarios.py       §2.5  N escenarios → 5 quintiles
│
├── calibration/           CAPA 3 — §3 PDF (Regret-Grid) — stubs
│   ├── grid.py            §3.2  G = Λ × M
│   ├── evaluate.py        §3.3  V_{g,s}
│   └── regret.py          §3.4  R_{g,s}, reglas avg / worst-case
│
├── main.py                corre solo CAPA 1 (equivalente al GAMS actual)
└── pipeline.py            Algorithm 1 completo (skeleton, pendiente)
