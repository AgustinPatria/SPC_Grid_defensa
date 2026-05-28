# Verificación de `mu_mix`

**Autor:** Agustín · **Fecha:** 2026-05-22

## Objetivo

Verificar el comportamiento de `mu_mix` (el retorno esperado que entra al optimizador) a partir de los datos de entrada `prob_*.csv` y `ret_semanal_*.csv`.

## Método

Cálculo directo según el PDF, con `p` las probabilidades de régimen de `prob_*.csv` y `r` los retornos de `ret_semanal_*.csv` (163 semanas):

- `mu_hat(i,k) = Σ_t p(i,k,t)·r(i,t) / Σ_t p(i,k,t)` &nbsp;(ec. 2)
- `mu_mix(i,t) = Σ_k p(i,k,t)·mu_hat(i,k)` &nbsp;(ec. 4)

## Resultado

| activo | `mu_hat(bear)` | `mu_hat(bull)` | `media_t[mu_mix(t)]` | media empírica de `r` | diferencia |
|---|--:|--:|--:|--:|--:|
| SPX | +0.2053% | +0.1747% | +0.187112% | +0.187112% | `0` |
| CMC200 | +0.8654% | +0.2832% | +0.419598% | +0.419598% | `8.7e-19` |

## Conclusión

El promedio temporal de `mu_mix(i,t)` es **exactamente igual** a la media empírica de los retornos `r(i)` (la diferencia está al nivel del epsilon de máquina, `~1e-19`).

Es una identidad algebraica: al promediar la ec. (4) sobre `t`, el factor `Σ_t p(i,k,t)` se cancela con el denominador de la ec. (2) y, como `Σ_k p(i,k,t) = 1`, queda la media simple de `r`. Se cumple para cualquier `p`.

Implicancia: la descomposición por régimen **redistribuye** el retorno esperado a lo largo del tiempo, pero **no altera su nivel promedio**, que queda fijado por construcción a la media histórica de los retornos.

*Script reproducible: `inspeccion/verif_mu_mix.py`.*
