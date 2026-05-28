# Informe inspeccion_v2 — Apertura del motor SPC_Grid

Run de referencia: `python main.py` (`resultados/main_run_20260527_120051.log`).
Configuracion vigente: `mu_hat_source='p_hist'`, `p_method='walking'`, grid
`λ ∈ (0.30, 0.70, 1.10, 1.50, 1.80)`, `m ∈ (0.01, 0.20, 0.50)`.

Tanda actual: L2 + L3 + L4 + L5. La L1 (datos crudos) queda pendiente.

---

## Resultados-marco (de `main.py`)

| Politica IS (t1..t163, 163 sem) | Capital | Retorno |
|---|---:|---:|
| OPT base (λ=1.0, m=1.0) | $14,522 | **+45.22%** |
| Naive 50/50 buy&hold | $12,258 | +22.58% |
| Naive 50/50 rebal | $12,092 | +20.92% |
| **RG g\*_mean (λ=0.30, m=0.5)** | $7,938 | **−20.62%** |

| Politica OOS (t148..t163, 16 sem) | Capital | Retorno |
|---|---:|---:|
| Naive 50/50 buy&hold | $12,986 | +29.86% |
| Naive 50/50 rebal | $12,975 | +29.75% |
| **RG_oos g\*_mean (λ=0.70, m=0.0)** | $12,821 | **+28.21%** |
| OPT_oos (λ=1.0, m=1.0) | $12,226 | +22.26% |

`g*` cayo en frontera del grid en los 4 casos (IS λ=0.30, OOS m=0.01/0.50).
La seleccion no tiene un optimo interior real bajo la grilla actual.

---

## L2 — Motor DL

Outputs: [`L2_dl_out/`](L2_dl_out/). Genera, por activo, 15 curvas
superpuestas + ensemble (negro) y tablas de calibracion direccional.

### L2.1 — mu_hat es identico en las 15 celdas

`05_mu_hat_per_cell_IS.csv` — las 15 filas son **identicas**:

| activo | mu_hat(bear) | mu_hat(bull) |
|---|---:|---:|
| SPX    | +0.2053% / sem | +0.1747% / sem |
| CMC200 | +0.8654% / sem | +0.2832% / sem |

`mu_hat` depende solo de `p_hist` (CSV) y `r_hist`. La NN no lo toca. Como
`mu_hat_source='p_hist'` esta vigente, **la unica fuente de heterogeneidad
entre celdas es `p_dl(t)`** (curva temporal). Toda la diferencia entre
celdas viene de como cada NN modula los mismos dos numeros a lo largo de
t=1..T.

Y los dos numeros estan **mal alineados**: `mu_hat(bear) > mu_hat(bull)`
en ambos activos. El "regimen bear" del CSV tiene mayor retorno medio que
"bull". Caveat documentado en CLAUDE.md — `p_hist` no usa la regla
`r >= 0` del LSTM, sale de un HMM externo sobre volatilidad u otro
criterio. Empiricamente accuracy(`p_hist>0.5` ↔ `r >= 0`) ≈ 52-56%.

**Consecuencia**: cuando `p_dl(t)` sube (modelo "bullish"), `mu_mix`
**baja** (porque pondera mas a `mu_hat(bull)` que es menor). El optimizador
recibe una senal con signo contrario al esperado.

### L2.2 — Las 15 NNs discrepan fuertemente entre si

`02_dispersion.png`, resumen sobre t = 1..163 (incluye padding warmup):

| activo | std promedio entre NNs | max (max-min) |
|---|---:|---:|
| SPX    | 0.1038 | 0.6000 |
| CMC200 | 0.1016 | 0.6000 |

En algun t, una NN dice p_bull=0.9 y otra dice p_bull=0.3 para la misma
ventana de input. Cada NN se entrena con seed propio (sin ensemble interno)
y converge a optimos locales distintos del pinball loss. El ensemble
(promedio de logits) suaviza esto, pero cada celda usa su NN sola — asi
que los 15 contextos son materialmente distintos.

### L2.3 — Calibracion direccional es coin-flip

`03_calibration_IS.csv` — accuracy de `p_bull(t) > 0.5` vs `r_real(t) >= 0`
en t=H+1..T (post-warmup, 103 obs):

| | SPX | CMC200 |
|---|---:|---:|
| ensemble | 0.476 | 0.553 |
| rango per-cell | 0.417 - 0.563 | 0.447 - 0.563 |
| coin flip | 0.500 | 0.500 |

El ensemble esta **por debajo de moneda en SPX**. Ninguna celda supera
+10 pp a la moneda en ningun activo. El LSTM cuantilico, en 163 semanas de
training data (split 0.7/0.15/0.15), no esta aprendiendo direccionalidad.

OOS (`03_calibration_OOS.csv`, t=148..163, 16 obs) — ruido puro: ensemble
SPX 0.250, CMC200 0.375; el rango per-cell es [0.06, 0.94] (pequeno N).

### L2.4 — `mu_mix(t)` no se mueve con el mercado

`04_mu_mix_by_cell.png`: las 15 curvas grises (`mu_mix(t)`) oscilan en
banda estrecha alrededor del promedio. La curva naranja (`r_real(t) MA(4)`)
tiene oscilaciones mucho mas grandes. **No hay correlacion visible** — el
"forecast" que ve el optimizador es esencialmente plano y desconectado
de lo que el mercado va a hacer.

---

## L3 — Motor escenarios

Outputs: [`L3_escenarios_out/`](L3_escenarios_out/).

### L3.1 — Correlacion SPX-CMC200 ≈ 0.99 en todos los escenarios

`06_correlation_IS.csv` / `06_correlation_OOS.csv`:

| escenario | corr(SPX, CMC200) |
|---|---:|
| s=0..4 IS  | 0.985 - 0.990 |
| s=0..4 OOS | 0.972 - 0.992 |
| historico (toda la serie) | **0.311** |

`generate_candidate_scenarios` usa **el mismo `q ∈ Q` para todos los
activos en cada paso** (decision documentada en su docstring: q
independiente daba corr ~0, generando escenarios artefacto tipo
"SPX −40% / CMC +1000%"). El precio de pegar la corr es que los 5
escenarios reps no diversifican: SPX y CMC200 se mueven juntos. El termino
`-λ·sum_ij w_i·w_j·σ_ij` de la FO ve covarianza maxima entre activos.
**Diversificar entre SPX y CMC no reduce riesgo en los escenarios.** Eso
ayuda a entender por que `m` (multiplicador de costos) parece inerte:
no hay incentivo de re-balancear porque no hay diversificacion.

### L3.2 — Sesgo bear sistematico en la distribucion de candidatos

`00_summary_IS_vs_OOS.csv`:

| metric | IS | OOS |
|---|---:|---:|
| min cum_port candidatos    | −84.75% | −42.74% |
| **mediana cum_port**       | **−10.62%** | **−1.26%** |
| max cum_port candidatos    | +665.43% | +85.44% |

Si el LSTM fuera direccionalmente neutral, la mediana de cum_port deberia
estar cerca de 0%. En IS esta en −10.6%; en OOS en −1.3%. El ensemble
genera mas mass del lado bear. Esto va de la mano con L2.3 (p_bull
ensemble IS = 0.475 — apenas por debajo de 0.5 — y agravado por el
compounding sobre T=163 pasos).

### L3.3 — Cola derecha extrema domina mean_regret

5 reps IS (`01_cum_final_IS.csv`):

| s | SPX | CMC200 | portafolio w_ref |
|--:|---:|---:|---:|
| 0 | −48.77% |  −73.85% |  **−62.70%** |
| 1 | −23.94% |  −48.62% |  −36.35% |
| 2 |  −2.77% |  −20.46% |  −10.51% |
| 3 | +16.48% |  +26.21% |  +23.47% |
| 4 | +65.89% | +136.75% | **+102.41%** |

3 de 5 reps son negativos. El escenario s=4 (+102%) pesa fuerte en el
mean_regret porque solo λ bajo lo captura — eso explica por que `g*_mean`
IS = (λ=0.30, m=0.50) cae en la frontera inferior de λ.

5 reps OOS son mas estrechos (`01_cum_final_OOS.csv`): [−24.00%, +26.71%].
T_oos=16 (vs T=163) reduce el compounding y los extremos.

### L3.4 — No-leakage OOS confirmado

`05_initial_windows.png`. Ventana inicial IS = t=104..163 (las H=60
ultimas semanas del historico, incluye periodo test). Ventana inicial OOS
= t=88..147 (las H=60 inmediatamente previas a `t_test_start=148`, todas
dentro de train+valid). **No hay leakage** del segmento de test en la
inicializacion OOS.

---

---

## L4 — Motor del optimizador

Outputs: [`L4_optimizador_out/`](L4_optimizador_out/). Sanity check
`|z_check − z_reported| = 0.0` en los 30 solves (15 IS + 15 OOS): la
descomposicion analitica reproduce a IPOPT exactamente.

### L4.1 — La FO esta dominada por el termino de retorno; el costo es despreciable

`01_z_decomposition_IS.csv`. Para todas las 15 celdas IS:

| celda | retorno | riesgo  | costo  | z |
|---|---:|---:|---:|---:|
| λ=0.30, m=0.01 | +1.018 | +0.201 | +0.00003 | +0.817 |
| λ=0.70, m=0.50 | +0.595 | +0.104 | +0.00008 | +0.491 |
| λ=1.80, m=0.50 | +0.433 | −0.043 | +0.00083 | +0.475 |

**Costo es < 0.13% del retorno en todas las celdas**. Eso explica por que
`m` (costo_mult) es practicamente inerte: el termino que `m` multiplica
es 10⁴ veces menor que los otros dos. **`m` no controla nada en este
problema**. La frontera de `m` en `g*` no significa que algo este mal
calibrado — significa que la dimension `m` aporta ruido (~$1 sobre
$1,053 de mean_regret IS).

### L4.2 — λ alto produce riesgo NEGATIVO (V_max budget desplazado)

Para λ ≥ 1.50, el termino de riesgo es **negativo** (`−0.043` en
λ=1.80, m=0.50). Por la FO `riesgo = λ * (var_total − T·V_max)`, un
riesgo negativo implica `var_total < T·V_max`: el portafolio queda
**debajo** del budget. Como `V_max` es constante, esto suma al objetivo
sin afectar `w*`. Estos +$0.04 son artefacto del budget,
no diversificacion real.

### L4.3 — Asignaciones extremas: nunca diversifica

`04_w_mean_plane_IS.png`. Peso medio en CMC200 sobre t=1..163:

| λ | w_mean_CMC200 | w_mean_SPX |
|---:|---:|---:|
| 0.30 | **1.000** | 0.000 |
| 0.70 | 0.53 - 0.65 | 0.35 - 0.47 |
| 1.10 | 0.36 - 0.44 | 0.56 - 0.64 |
| 1.50 | 0.26 - 0.32 | 0.68 - 0.74 |
| 1.80 | **0.21 - 0.22** | 0.78 - 0.79 |

**Con λ=0.30 el optimizador apila todo en CMC200 desde t=1 y no vuelve
a tocar**. CMC200 tiene `mu_hat(bear)=+0.87%/sem` (el mayor de los 4
numeros de la tabla mu_hat); como `p_dl(t)` es esencialmente plano
alrededor de 0.5, `mu_mix(CMC) ≈ 0.5·(0.87 + 0.28) ≈ 0.57%/sem`
constante — domina cualquier consideracion de varianza.

A medida que λ sube, el termino `−λ·var` empuja a SPX (`mu_hat(bear)
≈ 0.20%/sem`, varianza mas baja). **En NINGUN regimen del grid el
optimizador elige una mezcla "diversificada" 50/50** — siempre cae en
una de las dos esquinas. Esto es directamente consecuencia de L3.1
(corr SPX-CMC ≈ 0.99 en los escenarios): no hay nada que diversificar.

### L4.4 — Turnover spike con m bajo

`03_turnover_IS.csv`:

| (λ, m) | turnover_total |
|---|---:|
| (0.70, 0.01) | **12.32** |
| (1.10, 0.01) | 5.60 |
| (1.50, 0.01) | 7.12 |
| (cualquiera, 0.20 - 0.50) | 0.07 - 1.05 |

Cuando `m=0.01`, IPOPT no paga practicamente nada por re-balancear y
"juguetea" con `w(t)` aunque la mejora en retorno sea marginal. La
variable `w(i,t)` se mueve mucho mas que con `m≥0.20`. Esto **no cambia
materialmente `V[g,s]`** (por eso `m=0.01` y `m=0.50` dan V casi
identicos para una misma λ — ver L5.1) pero distorsiona la
interpretacion economica.

OOS muestra el mismo patron pero atenuado (`turnover_total ≤ 1.87`
porque T_oos=16 es mucho mas corto).

---

## L5 — Regret grid y seleccion g*

Outputs: [`L5_regret_out/`](L5_regret_out/).

### L5.1 — `m` es inerte; la dimension util es solo λ

`05_regret_summary_IS.csv`, mean_regret IS pivot (filas=λ, cols=m):

| λ \ m | 0.01 | 0.20 | 0.50 |
|---:|---:|---:|---:|
| 0.30 | **$1,053** | **$1,053** | **$1,053** |
| 0.70 | $1,453 | $1,326 | $1,243 |
| 1.10 | $1,453 | $1,287 | $1,359 |
| 1.50 | $1,631 | $1,379 | $1,455 |
| 1.80 | $1,433 | $1,440 | $1,450 |

- **Fila λ=0.30 es identica en m** ($1,053 exacto): tiene sentido — el
  optimizador apila todo en CMC desde t=1, no rebalancea, y `m` no
  toca nada (ver L4.3).
- Las otras filas oscilan ~$200 en m: irrelevante frente a la
  pendiente en λ.
- **El grid no tiene minimo interior**. mean_regret crece monotonico
  con λ desde $1,053 (λ=0.30) hasta $1,500 (λ=1.80). La seleccion
  `g*_mean = (0.30, 0.50)` (cualquier m sirve) es la frontera inferior
  de λ.

### L5.2 — IS y OOS dan selecciones contradictorias

| | g\*_mean | g\*_worst |
|---|---|---|
| IS  | (λ=0.30, m=0.50) | (λ=0.30, m=0.01) |
| OOS | (λ=0.70, m=0.01) | (λ=0.70, m=0.50) |

**Las selecciones IS y OOS no coinciden en ninguna dimension**. En IS λ
bajo gana porque el escenario s=4 (+102% portafolio) domina mean_regret
y solo λ=0.30 (que apila CMC) lo captura. En OOS s=4 (+27%) es mucho
mas moderado y la pendiente en λ es plana ($553 → $267 cayendo). Esto
es esperable cuando los escenarios cambian de forma (T=163 vs T_oos=16)
y refuerza que la seleccion del RG depende fuertemente de los pocos
extremos en la distribucion de candidatos.

### L5.3 — El "peor escenario" no es siempre el mismo

`06_worst_scenario_IS.png`. Por cada celda, en cual escenario s da
peor V:

- IS λ=0.30 (full CMC): s=0 (s=0 = `worst_port = −62.7%`)
- IS λ=1.80 (full SPX): s=4 (s=4 = `best_port = +102%` — pero el SPX
  individual es solo +66%, y como la celda apuesta a SPX no captura
  el upside de CMC). El peor caso de una politica conservadora es
  **el escenario bullish**, no el bearish.

Esto cierra el circulo: en escenarios comonotonos (L3.1) la politica
conservadora es vulnerable al rally — exactamente lo opuesto a la
intuicion clasica de "minimax = proteger contra crash".

### L5.4 — Frontera del grid en TODOS los casos

4/4 selecciones (`g*_mean` IS, `g*_worst` IS, `g*_mean` OOS, `g*_worst`
OOS) cayeron en la frontera. Extender λ por debajo de 0.30 daria mas
upside en IS; extender m fuera de 0.50 daria mas pivoteo a OOS — pero
dado L4.1 (costo es 10⁴ veces menor que retorno), extender `m`
practicamente no afecta `V[g, s]`. **El grid efectivo es 1D
(solo λ)**.

---

## Sintesis y proxima capa

### Diagnostico hasta acá

El pipeline esta operando sobre **5 capas defectuosas en cascada**:

1. **mu_hat invertido** (L2.1): `mu_hat(bear) > mu_hat(bull)` por
   inconsistencia entre la definicion de regimen del CSV (`p_hist`) y la
   del LSTM (`r>=0`). El optimizador resuelve un problema en el que "ser
   bullish" le baja el retorno esperado.
2. **`p_dl(t)` es esencialmente ruido** (L2.3): 0.5 ± 0.1, sin capacidad
   direccional (calibracion 0.48). No corrige la inconsistencia anterior
   — solo agrega varianza inter-celda.
3. **Escenarios comonotonos y bear-skewed** (L3.1, L3.2): SPX y CMC se
   mueven juntos (corr ≈ 0.99 vs 0.31 historico), eliminando la
   diversificacion. Mediana de candidatos negativa (−10.6% IS) +
   cola derecha extrema secuestran `g*_mean` hacia λ bajo.
4. **Optimizador colapsa a soluciones de esquina** (L4.3): nunca elige
   mezcla — apila todo en CMC (λ bajo) o en SPX (λ alto). Diversificar
   no diversifica en este universo de escenarios.
5. **`m` es inerte** (L4.1, L5.1): el termino de costo es 10⁴ veces
   menor que el retorno. El grid efectivo del regret-grid es 1D, no
   2D. Las "selecciones por `m`" son ruido.

**Hipotesis original validada**: el regret-grid esta funcionando
correctamente sobre los datos que recibe. El problema NO esta en la
formulacion ni en el optimizador — esta en la informacion que entra:
mu_hat invertido (L2.1) + LSTM coin-flip (L2.3) + corr artificial en
escenarios (L3.1). Las 3 generan un mundo donde la unica decision util
es "cuanto cargar SPX vs CMC", controlada por λ exclusivamente.

### Decision sugerida (queda para vos)

Antes de seguir abriendo capas, vale la pena un experimento estructural:
correr `main.py` con **`mu_hat_source='p_sign'`** (oraculo historico que
usa la misma regla del LSTM) y comparar L2/L3 vs el run actual. Si
`p_sign` arregla la inconsistencia bear/bull, deberian moverse:

- `mu_hat(bull) > mu_hat(bear)` (gap positivo en ambos activos).
- `mu_mix(t)` correlacionado con `p_dl(t)` con signo correcto.
- `g*` posiblemente fuera de frontera.

La memoria persistente ya tiene anotado que el profesor pidio volver a
`p_hist`. Si esto se mantiene, las opciones siguientes serian:
(a) intentar mejorar el LSTM (mas data, mas hidden, regularizacion) para
sacarlo del coin-flip;
(b) reemplazar el q-comun por una mezcla controlada (correlacion-target)
en `generate_candidate_scenarios`;
(c) aceptar que el RG opera sobre senales debiles y reportar contra Naive
y OPT como benchmarks (que es lo que el run actual ya hace).
