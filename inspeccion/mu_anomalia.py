"""Evidencia de la anomalia mu_hat(bear) > mu_hat(bull).

Calcula mu_hat por regimen (PDF ec. 2) tal como lo hace
`load_market_data` en Regret_Grid.py, y contrasta la definicion de regimen
de los CSV `prob_*.csv` contra la regla del PDF (sec. 2.4 ec. 15: bull si
r >= 0). Sirve de respaldo para la consulta a Javier Meier.

Uso:  python inspeccion/mu_anomalia.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

ASSETS = ("SPX", "CMC200")
RET_CSV = {"SPX": "ret_semanal_spx.csv", "CMC200": "ret_semanal_cmc200.csv"}
RET_COL = {"SPX": "ret_semanal_spx", "CMC200": "ret_semanal_cmc200"}
PROB_CSV = {"SPX": "prob_spx.csv", "CMC200": "prob_cmc200.csv"}


def mu_hat(p_k: pd.Series, r: pd.Series) -> float:
    """PDF ec. (2): mu_hat[i,k] = sum_t p[i,k,t]*r[i,t] / sum_t p[i,k,t]."""
    den = p_k.sum()
    return float((p_k * r).sum() / den) if den > 0 else 0.0


def main() -> None:
    rows_csv, rows_sign, diag = [], [], []
    for a in ASSETS:
        dp = pd.read_csv(DATA / PROB_CSV[a]).set_index("t")
        dr = pd.read_csv(DATA / RET_CSV[a]).set_index("t")[RET_COL[a]]
        p_bear, p_bull = dp["bear"], dp["bull"]

        # --- (A) mu_hat con las probabilidades del CSV (lo que hace el codigo)
        mu_bear_csv = mu_hat(p_bear, dr)
        mu_bull_csv = mu_hat(p_bull, dr)
        rows_csv.append((a, mu_bear_csv, mu_bull_csv, mu_bull_csv - mu_bear_csv))

        # --- (B) mu_hat con la regla del PDF ec. 15: bull si r >= 0 (regimen duro)
        ind_bull = (dr >= 0).astype(float)
        ind_bear = 1.0 - ind_bull
        mu_bear_sign = mu_hat(ind_bear, dr)
        mu_bull_sign = mu_hat(ind_bull, dr)
        rows_sign.append((a, mu_bear_sign, mu_bull_sign, mu_bull_sign - mu_bear_sign))

        # --- (C) diagnostico: coincide el regimen del CSV con el signo del retorno?
        csv_dice_bull = (p_bull > 0.5)
        retorno_es_pos = (dr >= 0)
        acc = float((csv_dice_bull == retorno_es_pos).mean())
        # retorno medio segun lo que dice el CSV
        r_dice_bull = float(dr[csv_dice_bull].mean())
        r_dice_bear = float(dr[~csv_dice_bull].mean())
        # corr(p_bull, r): determina el signo de mu(bull)-mu(bear) en la ec. (2).
        # Si corr <= 0 => el promedio ponderado invierte el orden de regimen.
        corr_pbull_r = float(np.corrcoef(p_bull.values, dr.values)[0, 1])
        diag.append((a, acc, r_dice_bull, r_dice_bear, corr_pbull_r, len(dr)))

    pct = lambda x: f"{x * 100:+.3f}%"

    print("=" * 72)
    print("ANOMALIA mu_hat(bear) > mu_hat(bull)  —  evidencia")
    print("=" * 72)
    print(f"Serie: {len(pd.read_csv(DATA / RET_CSV['SPX']))} semanas. "
          "Formula: PDF ec. (2)  mu_hat[i,k] = sum_t(p_ikt*r_it) / sum_t(p_ikt)\n")

    print("(A) mu_hat con las probabilidades de prob_*.csv  [lo que usa el modelo]")
    print(f"  {'activo':<8}{'mu(bear)':>12}{'mu(bull)':>12}{'bull-bear':>12}   anomalia?")
    for a, mb, mu, d in rows_csv:
        flag = "  <-- SI (bear>bull)" if d < 0 else ""
        print(f"  {a:<8}{pct(mb):>12}{pct(mu):>12}{pct(d):>12}{flag}")

    print("\n(B) mu_hat con la regla del PDF ec. (15): bull si r>=0  [regimen duro]")
    print(f"  {'activo':<8}{'mu(bear)':>12}{'mu(bull)':>12}{'bull-bear':>12}")
    for a, mb, mu, d in rows_sign:
        print(f"  {a:<8}{pct(mb):>12}{pct(mu):>12}{pct(d):>12}")

    print("\n(C) ¿el regimen de prob_*.csv corresponde al signo del retorno?")
    print(f"  {'activo':<8}{'accuracy':>10}{'r|CSV=bull':>13}{'r|CSV=bear':>13}"
          f"{'corr(p_bull,r)':>16}")
    for a, acc, rbu, rbe, corr, n in diag:
        print(f"  {a:<8}{acc * 100:>9.1f}%{pct(rbu):>13}{pct(rbe):>13}"
              f"{corr:>16.3f}")

    print("\n" + "-" * 72)
    print("LECTURA:")
    print("  (A) Con las probabilidades del CSV, el retorno medio del regimen")
    print("      'bear' resulta MAYOR que el del 'bull' -> anomalia.")
    print("  (B) Con la regla del PDF (bull = r>=0) la anomalia desaparece:")
    print("      mu(bull) > mu(bear) por construccion.")
    print("  (C) La definicion de regimen de prob_*.csv NO es el signo del")
    print("      retorno (accuracy ~ azar). corr(p_bull, r) <= 0 => el")
    print("      promedio ponderado de la ec. (2) invierte el orden de regimen.")
    print("=" * 72)


if __name__ == "__main__":
    main()
