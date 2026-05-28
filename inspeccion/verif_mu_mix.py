"""mu_mix usando las probabilidades de prob_*.csv.

Muestra, con UNICAMENTE los datos de los CSV (prob_*.csv y ret_semanal_*.csv),
el calculo de mu_hat y mu_mix, y compara el promedio temporal de mu_mix(i,t)
contra la media empirica de los retornos r(i).

mu_hat (PDF ec. 2):  mu_hat(i,k) = sum_t p(i,k,t)*r(i,t) / sum_t p(i,k,t)
mu_mix (PDF ec. 4):  mu_mix(i,t) = sum_k p(i,k,t)*mu_hat(i,k)

Uso:  python inspeccion/verif_mu_mix.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

RET_CSV = {"SPX": "ret_semanal_spx.csv", "CMC200": "ret_semanal_cmc200.csv"}
RET_COL = {"SPX": "ret_semanal_spx", "CMC200": "ret_semanal_cmc200"}
PROB_CSV = {"SPX": "prob_spx.csv", "CMC200": "prob_cmc200.csv"}
ASSETS = ("SPX", "CMC200")
REGIMES = ("bear", "bull")


def main():
    print("=" * 70)
    print("mu_mix CON LAS PROBABILIDADES DE prob_*.csv")
    print("=" * 70)

    for a in ASSETS:
        r = pd.read_csv(DATA / RET_CSV[a])[RET_COL[a]]
        p = pd.read_csv(DATA / PROB_CSV[a])[list(REGIMES)]

        # mu_hat por regimen (PDF ec. 2), con la p del CSV
        mu_hat = {}
        for k in REGIMES:
            mu_hat[k] = float((p[k] * r).sum() / p[k].sum())

        # mu_mix por periodo (PDF ec. 4), con la p del CSV
        mu_mix = sum(p[k].values * mu_hat[k] for k in REGIMES)

        media_mu_mix = mu_mix.mean()
        media_r = r.mean()

        print(f"\n  [{a}]   ({len(r)} semanas)")
        print(f"    mu_hat(bear)         = {mu_hat['bear']:+.6%}")
        print(f"    mu_hat(bull)         = {mu_hat['bull']:+.6%}")
        print(f"    media_t[ mu_mix(t) ] = {media_mu_mix:+.6%}")
        print(f"    media empirica de r  = {media_r:+.6%}")
        print(f"    diferencia           = {media_mu_mix - media_r:+.2e}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
