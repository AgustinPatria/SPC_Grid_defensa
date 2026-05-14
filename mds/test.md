Ex-ante vs ex-post vs backtest histórico

  Ex-ante ("antes del hecho") — la información disponible al decidir. En el pipeline:
  - mu_mix(t), sigma_mix(t) son ex-ante: lo que el modelo cree que va a pasar.
  - El FO de Markowitz usa estos inputs ex-ante para resolver w*(t). El optimizador "decide hoy" con su expectativa.

  Ex-post ("después del hecho") — lo que efectivamente sucede en una realización concreta del futuro. En el pipeline:
  - Cada scenario[s, t, i] es una realización ex-post hipotética: "supongamos que esto pasa".
  - simulate_capital_on_scenario aplica la política a esa realización y calcula el capital terminal V[g, s] ex-post.
  - Los 5 escenarios son 5 "futuros posibles según el LSTM". Sirven para evaluar la robustez de la política.

  Backtest histórico — un caso particular de evaluación ex-post, pero usando los retornos realizados del histórico (lo que de hecho pasó), no escenarios sintéticos:
  - run_historical_backtest aplica w*(t) a r_hist(t) y calcula la curva de capital "como si hubieras invertido durante esas semanas".
  - Es la prueba más exigente: no son futuros posibles según un modelo, son los retornos reales.

  Resumen:

  ┌──────────────────────┬─────────────────────────────────┬───────────────────────────────────────────────────────────┐
  │                      │            Lo que es            │                       En el código                        │
  ├──────────────────────┼─────────────────────────────────┼───────────────────────────────────────────────────────────┤
  │ Ex-ante              │ Lo que el modelo predice/cree   │ mu_mix, sigma_mix → input al FO                           │
  ├──────────────────────┼─────────────────────────────────┼───────────────────────────────────────────────────────────┤
  │ Ex-post (escenarios) │ Realización hipotética del LSTM │ scenarios[s, t, i] → input a simulate_capital_on_scenario │
  ├──────────────────────┼─────────────────────────────────┼───────────────────────────────────────────────────────────┤
  │ Backtest histórico   │ Realización real del pasado     │ r_hist(t) → input a run_historical_backtest               │
  └──────────────────────┴─────────────────────────────────┴───────────────────────────────────────────────────────────┘

  La grieta que vimos es exactamente: el FO optimiza para un mundo (ex-ante con mu_mix DL) pero la regret-grid lo evalúa en otro mundo (ex-post con scenarios LSTM). Con opción
  (a), ambos mundos comparten la misma fuente: los candidatos del LSTM.