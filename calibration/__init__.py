"""CAPA 3 — Calibración por Regret-Grid (§3 del PDF).

Selecciona la configuración de parámetros g* = (lambda*, m*) que minimiza
el regret frente a un conjunto pequeño de escenarios S (provistos por la
capa de predicción).

Módulos:
- grid:     §3.2  definición de G = Λ × M
- evaluate: §3.3  optimizar una vez por g, simular sobre cada s, construir V_{g,s}
- regret:   §3.4  R_{g,s} = V_s^best - V_{g,s}; reglas de selección (avg / worst)
"""
