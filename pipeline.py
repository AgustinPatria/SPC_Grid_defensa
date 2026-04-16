"""Pipeline SPC-Grid completo (Algorithm 1 del PDF).

Flujo:
  1. Preparacion de datos y particion temporal
  2. Entrenar modelo DL (cuantiles) + congelar
  3. Generar N escenarios -> reducir a 5 quintiles
  4. Ejecutar grilla G sobre el optimizador
  5. Regret-grid: evaluar V_{g,s}, calcular R_{g,s}, seleccionar g*
  6. Reportar (lambda*, m*)
"""
from pathlib import Path

from prediction.dataset      import load_returns, build_windows, split_chronological
from prediction.model_dl     import QuantileLSTM, QUANTILES
from prediction.train        import train_model
from prediction.scenarios    import generate_candidate_scenarios, reduce_to_quintiles

from optimizer.data_loader   import load_market_data
from calibration.grid        import build_grid
from calibration.evaluate    import evaluate_grid
from calibration.regret      import compute_regret, select_best


ROOT     = Path(__file__).parent
DATA_DIR = ROOT / "data"

H           = 52      # ventana de historia (semanas)
N_SCENARIOS = 1000    # escenarios candidatos
T_FUTURE    = 163     # horizonte de simulacion


def run_pipeline():
    # === PASO 1: Datos y particion ===
    print("=" * 65)
    print("PASO 1 - Preparacion de datos")
    print("=" * 65)
    returns = load_returns(str(DATA_DIR))
    X, y, t_idx = build_windows(returns, H=H)
    splits = split_chronological(X, y, t_idx)
    for name, (Xs, ys, ts) in splits.items():
        print(f"  {name:>6}: {len(Xs)} muestras  (t = {ts[0]}..{ts[-1]})")

    # === PASO 2: Entrenar DL ===
    print("\n" + "=" * 65)
    print("PASO 2 - Entrenamiento del modelo DL (cuantiles)")
    print("=" * 65)
    model = QuantileLSTM(
        n_assets=2,
        hidden_size=64,
        num_layers=2,
        dropout=0.1,
        quantiles=QUANTILES,
    )
    model, history = train_model(model, splits, max_epochs=500, patience=30)
    print(f"  Epochs: {len(history['train'])}  "
          f"best_val_loss: {min(history['valid']):.6f}")
    print("  Modelo congelado.")

    # === PASO 3: Generar y reducir escenarios ===
    print("\n" + "=" * 65)
    print(f"PASO 3 - Generacion de {N_SCENARIOS} escenarios -> 5 quintiles")
    print("=" * 65)
    last_window = X[-1]
    candidates = generate_candidate_scenarios(
        model, last_window, N=N_SCENARIOS, T_future=T_FUTURE
    )
    scenarios = reduce_to_quintiles(candidates, summary_asset_idx=0)
    print(f"  Escenarios generados: {candidates.shape}")
    print(f"  Escenarios finales:   {scenarios.shape}")

    # === PASO 4: Grilla de parametros ===
    print("\n" + "=" * 65)
    print("PASO 4 - Ejecucion de la grilla sobre el optimizador")
    print("=" * 65)
    context = load_market_data(str(DATA_DIR))
    grid = build_grid()
    print(f"  |G| = {len(grid)} puntos")

    # === PASO 5: Regret-grid ===
    print("\n" + "=" * 65)
    print("PASO 5 - Evaluacion y Regret-Grid")
    print("=" * 65)
    V, policies = evaluate_grid(grid, context, scenarios)
    R, V_best = compute_regret(V)

    g_avg,   info_avg   = select_best(R, grid, rule="avg")
    g_worst, info_worst = select_best(R, grid, rule="worst")

    # === PASO 6: Reporte ===
    print("\n" + "=" * 65)
    print("PASO 6 - Resultados")
    print("=" * 65)

    print("\n--- V_{g,s} (capital terminal) ---")
    print(V.to_string(float_format="${:,.2f}".format))

    print("\n--- R_{g,s} (regret) ---")
    print(R.to_string(float_format="${:,.2f}".format))

    print(f"\n--- Seleccion por regret PROMEDIO ---")
    print(f"  g* = ({g_avg.lam:.2f}, {g_avg.m:.1f})")
    print(f"  regret promedio = ${info_avg['regret_avg']:,.2f}")
    print(f"  regret peor caso = ${info_avg['regret_worst']:,.2f}")

    print(f"\n--- Seleccion por regret PEOR CASO ---")
    print(f"  g* = ({g_worst.lam:.2f}, {g_worst.m:.1f})")
    print(f"  regret promedio = ${info_worst['regret_avg']:,.2f}")
    print(f"  regret peor caso = ${info_worst['regret_worst']:,.2f}")

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    V.to_csv(results_dir / "V_matrix.csv")
    R.to_csv(results_dir / "R_matrix.csv")
    print(f"\nMatrices guardadas en {results_dir}/")

    return g_avg, g_worst


if __name__ == "__main__":
    run_pipeline()
