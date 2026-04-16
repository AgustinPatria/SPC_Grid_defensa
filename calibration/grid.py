"""Paso 4a del Pipeline SPC-Grid: definición de la grilla G = Λ × M (§3.2).

Λ: valores de aversión al riesgo lambda.
M: valores del multiplicador de costo m (c_eff_i = m * c_i).
"""
from itertools import product
from dataclasses import dataclass


@dataclass(frozen=True)
class GridPoint:
    lam: float
    m:   float

    def label(self) -> str:
        return f"lam={self.lam:.2f}_m={self.m:.1f}"


def build_grid(lambdas=(0.05, 0.10, 0.20, 0.50, 1.00),
               m_values=(0.5, 1.0, 2.0)):
    return [GridPoint(lam=l, m=m) for l, m in product(lambdas, m_values)]
