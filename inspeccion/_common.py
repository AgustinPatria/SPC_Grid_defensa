"""Utilidades compartidas para los modulos de inspeccion."""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

INSPECCION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = INSPECCION_DIR.parent

# Permite ejecutar tanto `python -m inspeccion.escenarios`
# como `python inspeccion/escenarios.py`.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def out_dir(name: str) -> Path:
    """Devuelve (y crea) `inspeccion/<name>_out/`."""
    d = INSPECCION_DIR / f"{name}_out"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_fig(fig, name: str, subdir: str) -> Path:
    """Guarda fig como PNG en `inspeccion/<subdir>_out/<name>.png` y cierra."""
    p = out_dir(subdir) / f"{name}.png"
    fig.tight_layout()
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {p.relative_to(PROJECT_ROOT)}")
    return p


def save_csv(df: pd.DataFrame, name: str, subdir: str, index: bool = False) -> Path:
    """Guarda DataFrame como CSV en `inspeccion/<subdir>_out/<name>.csv`."""
    p = out_dir(subdir) / f"{name}.csv"
    df.to_csv(p, index=index)
    print(f"  -> {p.relative_to(PROJECT_ROOT)}")
    return p
