1# Pipeline SPC-Grid

## Flujo secuencial (Algorithm 1 del PDF)

### 1. Preparación de Datos y Partición
- Obtener precios históricos P_{i,t}, calcular retornos semanales r_{i,t}
- Split cronológico: train → valid → test (sin fuga de información)

### 2. Deep Learning (prediction/)
- Entrenar red cuantil (LSTM/Transformer) con pinball loss
- Congelar modelo (no se reentrena después)
- Convertir cuantiles predichos → probabilidades de régimen bull/bear

### 3. Generación y Reducción de Escenarios (prediction/)
- Generar N=1000 escenarios candidatos (muestreo de cuantiles)
- Ordenar por retorno acumulado SPX, dividir en 5 quintiles
- Seleccionar escenario mediano de cada quintil → |S| = 5

### 4. Ejecución de la Grilla (optimizer/)
- Definir G = Λ × M (lambda × multiplicador de costo)
- Para cada g ∈ G: correr optimizador → política w^g_{i,t}

### 5. Calibración Regret-Grid (calibration/)
- Simular capital x^g_{t,s} para cada (g, s) con costos base reales
- V_{g,s} = capital terminal
- V^best_s = max_g V_{g,s}
- R_{g,s} = V^best_s - V_{g,s}
- Seleccionar g* por regret promedio o peor caso

### 6. Salida
- Reportar (lambda_{g*}, m_{g*}) — configuración más robusta
