# inspeccion/

Modulos de diagnostico del pipeline SPC_Grid3. Cada uno ataca un eslabon de la
cadena y deja sus salidas en `inspeccion/<modulo>_out/` (PNG + CSV).

| Modulo | Eslabon que diagnostica | Como correrlo |
|--------|--------------------------|---------------|
| `escenarios.py` | Generador de los 5 escenarios DL que alimentan el regret-grid | `python -m inspeccion.escenarios` |
| `regimen.py`    | LSTM cuantilico + p_bull (walking vs rollout, calibracion, sesgo) | `python -m inspeccion.regimen` |
| `lstm.py`       | Interno del LSTM (sensibilidad, lag importance, hidden state, historia) | `python -m inspeccion.lstm` |
| `grid.py`       | Optimizador + regret-grid (heatmaps, boundary, politicas, turnover, DL vs OPT base) | `python -m inspeccion.grid` |
| `inputs.py`     | Inputs al optimizador (mu_mix, sigma_mix, escenarios, constantes, coherencia matematica) | `python -m inspeccion.inputs` |

## escenarios.py

Genera 6 diagnosticos (1 PNG + 1 CSV cada uno):

1. **sesgo** - `cumret_T` de los 5 reps vs los N=1000 candidatos vs el historico
   realizado en el mismo horizonte.
   *Si los reps caen sistematicamente por debajo del historico, el regret-grid
   premia la politica defensiva y la `lambda` se va a la esquina alta.*

2. **dispersion** - fan chart de los 1000 candidatos con los 5 reps superpuestos.
   *Si el quintil bajo es muy extremo, minimax-regret es trivial.*

3. **correlacion** - histograma de `corr(SPX, CMC200)` por escenario.
   *Si pese al q independiente por activo los retornos quedan comonotonos, el
   optimizador no puede diversificar y `m` queda inerte.*

4. **path_dependence** - histograma de retornos semanales por bloque temporal
   (`t=1..40, 41..80, 81..120, 121..163`).
   *Si la media/varianza colapsa con t, el rollout se vuelve constante.*

5. **sesgo_resumen** - scatter `(cumret SPX, cumret CMC200)` con los 5 reps
   marcados.
   *Si los reps caen alineados solo en el eje SPX, los "quintiles" no
   representan a CMC200 y el optimizador ve un mundo mas pobre que el real.*

6. **sanity** - reproducibilidad con el mismo seed + comparacion con otro seed.

## regimen.py

Genera 6 diagnosticos (1 PNG + 1 CSV cada uno):

1. **pbull_walking_vs_rollout** - p_bull(t) calculado por ventana real previa
   (lo que alimenta `mu_mix`) vs p_bull(t) durante el rollout autoregresivo
   (lo que viven los 5 escenarios). *Si las dos series son distintas, el
   optimizador y la simulacion ex-post estan mirando mundos distintos.*

2. **pbull_serie_dist** - serie temporal + histograma de p_bull walking.
   *Detecta colapso a 0 (todo bear) o a 1 (todo bull).*

3. **rollout_step_by_step** - los 5 deciles predichos a cada paso del rollout,
   mas el retorno muestreado. *Si el rollout esta envenenado, los deciles
   colapsan o se desplazan con t.*

4. **sensibilidad_threshold** - p_bull medio para varios `BULL_THRESHOLD`
   (-0.03..+0.03). *Si CMC200 esta en p_bull=0 con thr=0 pero salta con
   thr=-0.01, el umbral esta corriendo al sesgo bear.*

5. **calibracion_deciles** - fraccion empirica de `r_real <= r_hat^q` vs q
   nominal, in-sample. *Curva por debajo de la diagonal = el modelo sobreestima
   retornos; por encima = los subestima (sesgo bear).*

6. **sesgo_deciles** - histograma de la distribucion realizada vs la distribucion
   predicha (todos los deciles apilados) in-sample. *Sesgo bruto.*

Diagnosticos 5 y 6 son in-sample (mismas semanas con las que se entreno) -
sirven para detectar el sesgo aprendido, no para evaluar generalizacion.
