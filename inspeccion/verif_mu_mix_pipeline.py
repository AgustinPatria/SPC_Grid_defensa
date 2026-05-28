"""Verificacion de mu_mix en el pipeline DL real (build_dl_context).

Comprueba la identidad media_t[mu_mix] = media empirica de r usando el
mu_mix que realmente construye build_dl_context, para las dos fuentes de
mu_hat: 'p_dl' (misma p que mu_mix) y 'p_sign' (default actual, p distinta).

Uso:  python inspeccion/verif_mu_mix_pipeline.py
"""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT))

from config import CHECKPOINT_PATH, DATA_DIR, T_HORIZON
from Regret_Grid import build_dl_context, load_market_data

ASSETS = ("SPX", "CMC200")


def main():
    base = load_market_data(str(DATA_DIR))
    r_emp = {a: base["r"][a].sort_index().values[:T_HORIZON].mean() for a in ASSETS}

    print("=" * 74)
    print("VERIFICACION mu_mix EN EL PIPELINE DL REAL (build_dl_context)")
    print("=" * 74)
    print(f"  media empirica de r (t=1..{T_HORIZON}):")
    for a in ASSETS:
        print(f"    {a:<8} {r_emp[a]:.12f}")

    for src in ("p_dl", "p_sign"):
        print("\n" + "-" * 74)
        print(f"  mu_hat_source = '{src}'"
              + ("   [misma p que mu_mix]" if src == "p_dl"
                 else "   [DEFAULT ACTUAL - p distinta de mu_mix]"))
        print("-" * 74)
        ctx = build_dl_context(
            data_dir=DATA_DIR, checkpoint_path=CHECKPOINT_PATH, T=T_HORIZON,
            N_candidates=20, n_scenarios=5, seed=0,
            p_method="walking", mu_hat_source=src,
        )
        print(f"  {'activo':<8}{'media_t[mu_mix]':>20}{'media emp. r':>18}"
              f"{'diferencia':>16}")
        for a in ASSETS:
            m_mix = float(np.mean(ctx["mu_mix"][a].values))
            d = m_mix - r_emp[a]
            print(f"  {a:<8}{m_mix:>20.12f}{r_emp[a]:>18.10f}{d:>16.2e}")


if __name__ == "__main__":
    main()
