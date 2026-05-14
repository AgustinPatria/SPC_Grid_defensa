"""Diagnostico interno del LSTM cuantilico.

Investiga por que el LSTM colapso a una distribucion practicamente incondicional
(salida casi independiente de la ventana de entrada). Hipotesis a testear:

  H1) La salida del LSTM es insensible al input -> el modelo aprendio una
      constante por activo+cuantil (los pesos de la cabeza absorben la media,
      el LSTM aporta poco).
  H2) El espacio de hidden states colapsa -> el encoder mapea todas las
      ventanas a estados muy parecidos.
  H3) El entrenamiento se detuvo demasiado temprano (early stopping en
      patience=15 + dataset chico = picks de un modelo casi trivial).
  H4) El modelo NO le gana al baseline constante (cuantiles del train set).

Corre con:
    python -m inspeccion.lstm
    (o)
    python inspeccion/lstm.py

Diagnosticos (1 PNG + 1 CSV cada uno) en `inspeccion/lstm_out/`:

  1. pinball_vs_baseline  pinball del LSTM vs predictor constante (cuantiles
                          del train set) en train/valid/test. Si son
                          comparables, el LSTM no agrega valor sobre la
                          distribucion marginal.
  2. jacobian             norma de d(out)/d(input) por lag. Si las normas son
                          chicas, el LSTM ignora el input.
  3. lag_importance       pinball delta al permutar cada lag. Si el delta es
                          casi 0 en todos los lags, ningun lag aporta info.
  4. hidden_state         PCA del hidden state final sobre las ventanas. Si la
                          varianza acumulada se concentra en 1 componente o las
                          distancias entre estados son chicas, el encoder
                          comprime trivialmente.
  5. history              curvas pinball train/valid del checkpoint guardado.
                          Detecta convergencia prematura.
  6. pred_vs_feature      scatter del q=0.5 predicho vs feature simple de la
                          ventana (mean, std, ultimo retorno). Si es plano,
                          el feature no se usa.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from inspeccion._common import save_fig, save_csv, out_dir

from config import CHECKPOINT_PATH, DATA_DIR
from dl.prediccion_deciles import (
    build_windows,
    chrono_split,
    fit_standardizer,
    load_checkpoint,
    load_returns,
    pinball_loss,
)


SUBDIR = "lstm"


def build_context():
    """Carga el LSTM + datos y construye un split chrono compatible con
    la configuracion entrenada."""
    model = load_checkpoint(CHECKPOINT_PATH)
    cfg = model.config
    df_ret = load_returns(DATA_DIR)
    X, Y, t_idx = build_windows(df_ret, cfg.H)
    split = chrono_split(X, Y, t_idx, cfg.split)
    scaler = fit_standardizer(split.X_train)

    # Compatibilidad: usar mean/std del checkpoint (los del scaler reentrenado
    # deberian coincidir porque el split es determinista).
    return {
        "model":  model,
        "cfg":    cfg,
        "split":  split,
        "scaler": scaler,
        "df_ret": df_ret,
        "assets": list(cfg.assets),
    }


def _to_torch(x: np.ndarray, mean, std) -> torch.Tensor:
    """Estandariza con (mean, std) del checkpoint y devuelve tensor float32."""
    return torch.from_numpy(((x - mean) / std).astype(np.float32))


def _ensemble_forward(model, X_t):
    """Promedio del ensemble de seeds del checkpoint."""
    outs = [net(X_t).detach().numpy() for net in model.nets]
    return np.mean(np.stack(outs, axis=0), axis=0)


# ================================================================
# 1) Pinball: LSTM vs baseline constante
# ================================================================
def diag_pinball_vs_baseline(ctx):
    model = ctx["model"]
    cfg = ctx["cfg"]
    split = ctx["split"]
    quantiles = list(cfg.quantiles)
    assets = ctx["assets"]
    A = len(assets)
    Q = len(quantiles)

    # Baseline: cuantiles nominales calculados sobre el target del train set.
    Y_tr = split.Y_train                                  # (n_tr, A)
    q_baseline = np.quantile(Y_tr, quantiles, axis=0).T   # (A, Q)

    def pinball(y_pred, y_true):
        # y_pred: (N, A, Q)  y_true: (N, A)
        diff = y_true[:, :, None] - y_pred                # (N, A, Q)
        q_arr = np.asarray(quantiles)                     # (Q,)
        loss = np.maximum(q_arr * diff, (q_arr - 1.0) * diff)
        return loss

    rows = []
    for name, X, Y in [("train", split.X_train, split.Y_train),
                       ("valid", split.X_valid, split.Y_valid),
                       ("test",  split.X_test,  split.Y_test)]:
        if len(X) == 0:
            continue
        Xs = _to_torch(X, model.mean, model.std)
        with torch.no_grad():
            y_pred = _ensemble_forward(model, Xs)         # (N, A, Q)
        y_pred = np.sort(y_pred, axis=-1)
        loss_lstm = pinball(y_pred, Y)                    # (N, A, Q)

        y_pred_base = np.broadcast_to(q_baseline[None, :, :], y_pred.shape)
        loss_base   = pinball(y_pred_base, Y)             # (N, A, Q)

        for ai, a in enumerate(assets):
            rows.append({"split": name, "asset": a,
                         "pinball_lstm":     float(loss_lstm[:, ai, :].mean()),
                         "pinball_baseline": float(loss_base[:, ai, :].mean()),
                         "skill":            float(1.0 - loss_lstm[:, ai, :].mean()
                                                          / loss_base[:, ai, :].mean())})

    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    width = 0.35
    splits_order = [s for s in ("train", "valid", "test") if s in df.split.unique()]
    x_pos = np.arange(len(splits_order) * A)
    labels = [f"{s}\n{a}" for s in splits_order for a in assets]
    lstm_vals = [df[(df.split == s) & (df.asset == a)].pinball_lstm.iloc[0]
                 for s in splits_order for a in assets]
    base_vals = [df[(df.split == s) & (df.asset == a)].pinball_baseline.iloc[0]
                 for s in splits_order for a in assets]
    ax.bar(x_pos - width / 2, lstm_vals, width, color="C0", label="LSTM")
    ax.bar(x_pos + width / 2, base_vals, width, color="C3", label="constante")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("pinball (mean)")
    ax.set_title("Pinball: LSTM vs baseline constante (cuantiles del train)")
    ax.legend()
    save_fig(fig, "1_pinball_vs_baseline", SUBDIR)
    save_csv(df, "1_pinball_vs_baseline", SUBDIR)


# ================================================================
# 2) Jacobiano: norma de d(out)/d(input) por lag
# ================================================================
def diag_jacobian(ctx):
    model = ctx["model"]
    cfg = ctx["cfg"]
    split = ctx["split"]
    H = cfg.H
    A = len(ctx["assets"])
    Q = cfg.n_quantiles

    # Submuestra train (jacobiano es caro).
    rng = np.random.default_rng(0)
    n_use = min(150, len(split.X_train))
    idx = rng.choice(len(split.X_train), size=n_use, replace=False)
    Xs = _to_torch(split.X_train[idx], model.mean, model.std)

    # Acumulamos norma por lag promediada sobre el ensemble + muestras.
    norms_per_lag = np.zeros((H, A), dtype=np.float32)
    for net in model.nets:
        for n in range(n_use):
            x = Xs[n:n + 1].clone().detach().requires_grad_(True)
            out = net(x)                                # (1, A, Q)
            # Suma escalar -> backprop.
            out.sum().backward()
            g = x.grad.detach().numpy()[0]              # (H, A)
            norms_per_lag += np.abs(g) / (n_use * len(model.nets))

    fig, axes = plt.subplots(1, A, figsize=(13, 4.5))
    if A == 1:
        axes = [axes]
    lags = np.arange(1, H + 1)
    rows = []
    for ai, a in enumerate(ctx["assets"]):
        ax = axes[ai]
        ax.plot(lags, norms_per_lag[:, ai], "o-", color="C0", lw=1.4)
        ax.set_title(f"{a} - |d(sum out) / d(input_lag)|")
        ax.set_xlabel("lag (1 = mas antiguo, H = mas reciente)")
        ax.set_ylabel("|gradient|")
        for k in range(H):
            rows.append({"asset": a, "lag": int(lags[k]),
                         "abs_grad": float(norms_per_lag[k, ai])})
    save_fig(fig, "2_jacobian", SUBDIR)
    save_csv(pd.DataFrame(rows), "2_jacobian", SUBDIR)


# ================================================================
# 3) Permutation lag importance
# ================================================================
def diag_lag_importance(ctx):
    model = ctx["model"]
    cfg = ctx["cfg"]
    split = ctx["split"]
    H = cfg.H
    A = len(ctx["assets"])
    quantiles = list(cfg.quantiles)
    rng = np.random.default_rng(0)

    def pinball_np(y_pred, y_true):
        q_arr = np.asarray(quantiles)
        diff = y_true[:, :, None] - y_pred
        return np.maximum(q_arr * diff, (q_arr - 1.0) * diff).mean()

    # Baseline pinball sobre train (sin permutar).
    Xs_tr = _to_torch(split.X_train, model.mean, model.std)
    with torch.no_grad():
        y_pred_base = _ensemble_forward(model, Xs_tr)
    y_pred_base = np.sort(y_pred_base, axis=-1)
    base_loss = pinball_np(y_pred_base, split.Y_train)

    rows = []
    for k in range(H):
        # Permutar el lag k (todos los activos) y recomputar.
        X_perm = split.X_train.copy()
        for ai in range(A):
            perm = rng.permutation(len(X_perm))
            X_perm[:, k, ai] = split.X_train[perm, k, ai]
        Xs = _to_torch(X_perm, model.mean, model.std)
        with torch.no_grad():
            y_pred = _ensemble_forward(model, Xs)
        y_pred = np.sort(y_pred, axis=-1)
        loss = pinball_np(y_pred, split.Y_train)
        rows.append({"lag": k + 1,
                     "pinball": float(loss),
                     "delta_vs_baseline": float(loss - base_loss)})

    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(df.lag, df.delta_vs_baseline, color="C0",
           label="Δ pinball al permutar lag")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title(f"Permutation importance por lag (pinball baseline={base_loss:.5f})")
    ax.set_xlabel("lag (1 = mas antiguo, H = mas reciente)")
    ax.set_ylabel("Δ pinball")
    ax.legend()
    save_fig(fig, "3_lag_importance", SUBDIR)
    save_csv(df, "3_lag_importance", SUBDIR)


# ================================================================
# 4) Hidden state diversity (PCA)
# ================================================================
def diag_hidden_state(ctx):
    model = ctx["model"]
    cfg = ctx["cfg"]
    split = ctx["split"]
    hidden = cfg.lstm_hidden

    # Concat train+valid+test para representar todo el dataset.
    X_all = np.concatenate([split.X_train, split.X_valid, split.X_test], axis=0)
    Xs = _to_torch(X_all, model.mean, model.std)
    splits_idx = np.r_[
        np.full(len(split.X_train), 0),
        np.full(len(split.X_valid), 1),
        np.full(len(split.X_test),  2),
    ]

    # Hidden final del primer net del ensemble.
    net = model.nets[0]
    with torch.no_grad():
        lstm_out, _ = net.lstm(Xs)                      # (N, H, hidden)
    last_h = lstm_out[:, -1, :].numpy()                 # (N, hidden)

    # PCA por descomposicion svd.
    h_centered = last_h - last_h.mean(axis=0, keepdims=True)
    _, sv, vh = np.linalg.svd(h_centered, full_matrices=False)
    explained = (sv ** 2) / max((sv ** 2).sum(), 1e-12)
    cum_explained = np.cumsum(explained)

    # Proyectar sobre PC1/PC2 para visualizar.
    pcs = h_centered @ vh.T[:, :2]                      # (N, 2)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(np.arange(1, len(explained) + 1), cum_explained,
                 "o-", color="C0")
    axes[0].axhline(0.95, color="grey", ls="--", lw=1)
    axes[0].set_title(f"Varianza explicada acumulada (hidden={hidden})")
    axes[0].set_xlabel("componente")
    axes[0].set_ylabel("var. acum.")
    axes[0].set_ylim(0, 1.05)

    colors = ["C0", "C1", "C3"]
    names  = ["train", "valid", "test"]
    for s in range(3):
        mask = splits_idx == s
        if mask.any():
            axes[1].scatter(pcs[mask, 0], pcs[mask, 1], s=18, alpha=0.6,
                            color=colors[s], label=names[s])
    axes[1].set_title("Hidden state proyectado en PC1/PC2")
    axes[1].set_xlabel("PC1"); axes[1].set_ylabel("PC2")
    axes[1].legend(fontsize=9)
    save_fig(fig, "4_hidden_state", SUBDIR)

    rows = []
    for k in range(min(10, len(explained))):
        rows.append({"componente": k + 1,
                     "varianza_explicada": float(explained[k]),
                     "varianza_acumulada": float(cum_explained[k])})
    save_csv(pd.DataFrame(rows), "4_hidden_state", SUBDIR)


# ================================================================
# 5) Curvas de loss del checkpoint
# ================================================================
def diag_history(ctx):
    # Cargar el payload bruto del checkpoint (load_checkpoint descarta history).
    payload = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    history = payload.get("history", {})
    train_hist = history.get("train", [])
    valid_hist = history.get("valid", [])

    rows = []
    fig, ax = plt.subplots(figsize=(11, 4.5))
    if train_hist:
        ax.plot(np.arange(1, len(train_hist) + 1), train_hist,
                color="C0", lw=1.2, label="train")
        for k, v in enumerate(train_hist):
            rows.append({"epoch": k + 1, "split": "train", "pinball": float(v)})
    if valid_hist:
        ax.plot(np.arange(1, len(valid_hist) + 1), valid_hist,
                color="C3", lw=1.2, label="valid")
        for k, v in enumerate(valid_hist):
            rows.append({"epoch": k + 1, "split": "valid", "pinball": float(v)})
    best_valid = payload.get("best_valid")
    best_seed = payload.get("best_seed")
    if valid_hist:
        best_epoch = int(np.argmin(valid_hist)) + 1
        ax.axvline(best_epoch, color="grey", ls="--", lw=1,
                   label=f"best epoch ({best_epoch})")
    title = "Curvas de pinball (best seed"
    if best_seed is not None:
        title += f" = {best_seed}"
    if best_valid is not None:
        title += f", best_valid={best_valid:.5f}"
    title += ")"
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel("pinball")
    ax.legend()
    save_fig(fig, "5_history", SUBDIR)
    save_csv(pd.DataFrame(rows), "5_history", SUBDIR)


# ================================================================
# 6) Predicho q=0.5 vs feature simple de la ventana
# ================================================================
def diag_pred_vs_feature(ctx):
    model = ctx["model"]
    cfg = ctx["cfg"]
    split = ctx["split"]
    assets = ctx["assets"]
    A = len(assets)

    # Tomar todo el train para que haya N grande.
    X = split.X_train
    Xs = _to_torch(X, model.mean, model.std)
    with torch.no_grad():
        y_pred = _ensemble_forward(model, Xs)             # (N, A, Q)
    y_pred = np.sort(y_pred, axis=-1)
    # mediana predicha por activo
    q05_idx = np.argmin(np.abs(np.asarray(cfg.quantiles) - 0.5))
    med_pred = y_pred[:, :, q05_idx]                       # (N, A)

    # Features de la ventana: mean, std, last value (por activo).
    feat_mean = X.mean(axis=1)                             # (N, A)
    feat_std  = X.std(axis=1)                              # (N, A)
    feat_last = X[:, -1, :]                                # (N, A)

    fig, axes = plt.subplots(A, 3, figsize=(15, 4.5 * A))
    if A == 1:
        axes = axes[None, :]

    rows = []
    feature_names = ["window_mean", "window_std", "last_return"]
    feature_arrs  = [feat_mean, feat_std, feat_last]
    for ai, a in enumerate(assets):
        for fi, (fname, farr) in enumerate(zip(feature_names, feature_arrs)):
            ax = axes[ai, fi]
            ax.scatter(farr[:, ai], med_pred[:, ai], s=10, alpha=0.5, color="C0")
            r = float(np.corrcoef(farr[:, ai], med_pred[:, ai])[0, 1])
            ax.set_title(f"{a} - q=0.5 vs {fname} (r={r:+.3f})")
            ax.set_xlabel(fname)
            ax.set_ylabel("q=0.5 predicho")
            rows.append({"asset": a, "feature": fname, "pearson_r": r})
    save_fig(fig, "6_pred_vs_feature", SUBDIR)
    save_csv(pd.DataFrame(rows), "6_pred_vs_feature", SUBDIR)


# ================================================================
# Main
# ================================================================
def main():
    print("=" * 70)
    print("INSPECCION INTERNA DEL LSTM")
    print("=" * 70)
    print("Construyendo contexto (carga LSTM + dataset + split chrono)...")
    ctx = build_context()
    cfg = ctx["cfg"]
    print(f"  assets   : {ctx['assets']}")
    print(f"  H        : {cfg.H}")
    print(f"  hidden   : {cfg.lstm_hidden}")
    print(f"  layers   : {cfg.lstm_layers}")
    print(f"  quantiles: {list(cfg.quantiles)}")
    print(f"  split    : {cfg.split}  (n_tr={len(ctx['split'].X_train)},"
          f" n_va={len(ctx['split'].X_valid)},"
          f" n_te={len(ctx['split'].X_test)})")
    print(f"  output   : {out_dir(SUBDIR)}")
    print("-" * 70)

    print("[1/6] pinball LSTM vs baseline constante")
    diag_pinball_vs_baseline(ctx)
    print("[2/6] jacobiano por lag")
    diag_jacobian(ctx)
    print("[3/6] permutation importance por lag")
    diag_lag_importance(ctx)
    print("[4/6] hidden state diversity (PCA)")
    diag_hidden_state(ctx)
    print("[5/6] curvas pinball train/valid")
    diag_history(ctx)
    print("[6/6] q=0.5 predicho vs feature de la ventana")
    diag_pred_vs_feature(ctx)

    print("-" * 70)
    print(f"Listo. Resultados en {out_dir(SUBDIR)}")


if __name__ == "__main__":
    main()
