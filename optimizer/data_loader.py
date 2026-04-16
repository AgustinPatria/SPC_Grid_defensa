"""§1.2–1.3 del PDF: carga de datos y cálculo de momentos mixtos por periodo.

Flujo:
  r_{i,t}, p_{i,k,t}  -->  mu_hat_{i,k}, sigma_hat_{i,j,k}  -->  mu_mix_{i,t}, sigma_mix_{i,j,t}

Este módulo NO optimiza ni simula: solo produce el `context` que consumen los
demás módulos. Estudiarlo aisladamente permite verificar las ecuaciones (2)–(5).
"""
from pathlib import Path
import pandas as pd


def load_market_data(data_dir: str):
    BASE_DIR = Path(data_dir)

    prob_spx = pd.read_csv(BASE_DIR / "prob_spx.csv")
    prob_cmc = pd.read_csv(BASE_DIR / "prob_cmc200.csv")
    ret_spx  = pd.read_csv(BASE_DIR / "ret_semanal_spx.csv")
    ret_cmc  = pd.read_csv(BASE_DIR / "ret_semanal_cmc200.csv")

    for df in (prob_spx, prob_cmc, ret_spx, ret_cmc):
        df.columns = [c.strip() for c in df.columns]
        df["t"] = df["t"].astype(int)

    T_vals  = sorted(prob_spx["t"].unique())
    assets  = ["SPX", "CMC200"]
    regimes = ["bear", "bull"]

    r = {
        "SPX":    ret_spx.set_index("t")["ret_semanal_spx"],
        "CMC200": ret_cmc.set_index("t")["ret_semanal_cmc200"],
    }
    p = {
        "SPX":    prob_spx.set_index("t")[regimes],
        "CMC200": prob_cmc.set_index("t")[regimes],
    }

    mu_hat, sigma_hat = _regime_moments(assets, regimes, r, p)
    mu_mix, sigma_mix = _mix_by_period(assets, regimes, T_vals, p, mu_hat, sigma_hat)

    return {
        "mu_mix":          mu_mix,
        "sigma_mix":       sigma_mix,
        "mu_hat":          mu_hat,
        "sigma_hat":       sigma_hat,
        "T_vals":          T_vals,
        "nT":              len(T_vals),
        "assets":          assets,
        "regimes":         regimes,
        "r":               r,
        "p":               p,
        "c_base":          {"SPX": 0.005, "CMC200": 0.010},
        "w0":              {"SPX": 0.5,   "CMC200": 0.5},
        "Capital_inicial": 10000.0,
    }


def _regime_moments(assets, regimes, r, p):
    """Ecuaciones (2) y (3): media y covarianza condicional por régimen."""
    mu_hat, sigma_hat = {}, {}

    for i in assets:
        for k in regimes:
            den = p[i][k].sum()
            mu_hat[(i, k)] = (p[i][k] * r[i]).sum() / den if den > 0 else 0.0

    for i in assets:
        for j in assets:
            for k in regimes:
                pi_k, pj_k = p[i][k], p[j][k]
                den = (pi_k * pj_k).sum()
                if den > 0:
                    term = pi_k * pj_k * (r[i] - mu_hat[(i, k)]) * (r[j] - mu_hat[(j, k)])
                    sigma_hat[(i, j, k)] = term.sum() / den
                else:
                    sigma_hat[(i, j, k)] = 0.0

    return mu_hat, sigma_hat


def _mix_by_period(assets, regimes, T_vals, p, mu_hat, sigma_hat):
    """Ecuaciones (4) y (5): mezcla por periodo usando probabilidades de régimen."""
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

    return mu_mix, sigma_mix
