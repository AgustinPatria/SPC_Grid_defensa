"""Mini-experimento: comparar el LSTM actual con variantes mas chicas y un
baseline lineal cuantilico contra el predictor constante.

Pregunta a contestar: bajando hidden/layers/H, o pasando a un modelo lineal
con features agregados, ¿alguna configuracion supera al predictor constante
(que es el techo trivial mostrado en `inspeccion/lstm`)?

Cada modelo se entrena con seed averaging (`seeds=(0,1,2)`, mejor seed por
pinball-valid) y se evalua en train/valid/test con su propio constante como
baseline. La metrica clave es **skill = 1 - pinball / pinball_constante**:
positivo = mejor que el baseline.

Corre con:
    python -m experimentos.sweep_lstm

Salidas en `experimentos/sweep_lstm_out/`:
    results.csv          tabla larga model x split x asset
    skill_test.png       barras de skill en test
    skill_heatmap.png    heatmap skill en (model, split, asset)
"""
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from config import DATA_DIR, DLConfig
from dl.prediccion_deciles import (
    LoadedModel,
    QuantileLSTM,
    build_windows,
    chrono_split,
    load_returns,
    pinball_loss,
    train_deciles,
)


OUT_DIR = PROJECT_ROOT / "experimentos" / "sweep_lstm_out"


# ================================================================
# Pinball helpers
# ================================================================
def pinball_np(y_pred, y_true, quantiles):
    """y_pred (N, A, Q), y_true (N, A) -> escalar (media)."""
    q = np.asarray(quantiles, dtype=np.float32)
    diff = y_true[:, :, None] - y_pred                  # (N, A, Q)
    return float(np.maximum(q * diff, (q - 1.0) * diff).mean())


def constant_quantiles(Y_train, quantiles):
    """Cuantiles nominales del train target. (A, Q)"""
    return np.quantile(Y_train, quantiles, axis=0).T


# ================================================================
# Linear quantile baseline
# ================================================================
def extract_features(X):
    """X (N, H, A) -> features (N, F).
    Por activo: mean, std, last, max, min de la ventana."""
    N, H, A = X.shape
    feats = []
    for ai in range(A):
        feats.append(X[:, :, ai].mean(axis=1))
        feats.append(X[:, :, ai].std(axis=1))
        feats.append(X[:, -1, ai])
        feats.append(X[:, :, ai].max(axis=1))
        feats.append(X[:, :, ai].min(axis=1))
    return np.stack(feats, axis=1).astype(np.float32)


class LinearQuantile(nn.Module):
    def __init__(self, n_features, A, Q):
        super().__init__()
        self.A = A
        self.Q = Q
        self.linear = nn.Linear(n_features, A * Q)

    def forward(self, x):
        return self.linear(x).view(-1, self.A, self.Q)


def train_linear(F_tr, Y_tr, F_va, Y_va, quantiles,
                 seeds=(0, 1, 2), epochs=500, lr=5e-3, weight_decay=1e-3):
    A = Y_tr.shape[1]
    Q = len(quantiles)

    f_mean = F_tr.mean(axis=0)
    f_std = F_tr.std(axis=0)
    f_std = np.where(f_std < 1e-8, 1.0, f_std)
    Xt = torch.from_numpy(((F_tr - f_mean) / f_std).astype(np.float32))
    Yt = torch.from_numpy(Y_tr)
    Xv = torch.from_numpy(((F_va - f_mean) / f_std).astype(np.float32))
    Yv = torch.from_numpy(Y_va)

    best_seed = None
    best_valid = float("inf")
    best_state = None
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = LinearQuantile(F_tr.shape[1], A, Q)
        optim = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay,
        )
        seed_best = float("inf")
        seed_state = None
        for _ in range(epochs):
            model.train()
            optim.zero_grad()
            loss = pinball_loss(model(Xt), Yt, quantiles)
            loss.backward()
            optim.step()
            model.eval()
            with torch.no_grad():
                v = pinball_loss(model(Xv), Yv, quantiles).item()
            if v < seed_best:
                seed_best = v
                seed_state = {k: vv.clone() for k, vv in model.state_dict().items()}
        if seed_best < best_valid:
            best_valid = seed_best
            best_seed = seed
            best_state = seed_state

    model = LinearQuantile(F_tr.shape[1], A, Q)
    model.load_state_dict(best_state)
    model.eval()
    return model, f_mean.astype(np.float32), f_std.astype(np.float32), best_seed, best_valid


def predict_linear(model, F, f_mean, f_std):
    Fs = ((F - f_mean) / f_std).astype(np.float32)
    with torch.no_grad():
        out = model(torch.from_numpy(Fs)).numpy()
    return np.sort(out, axis=-1)


# ================================================================
# Eval helpers
# ================================================================
def n_params(net):
    return sum(p.numel() for p in net.parameters())


def eval_lstm_split(loaded, X, Y, quantiles):
    """Pinball por activo en (X, Y)."""
    if len(X) == 0:
        return None
    Xs = ((X - loaded.mean) / loaded.std).astype(np.float32)
    with torch.no_grad():
        outs = [net(torch.from_numpy(Xs)).numpy() for net in loaded.nets]
    y_pred = np.mean(np.stack(outs, axis=0), axis=0)
    y_pred = np.sort(y_pred, axis=-1)
    return y_pred  # (N, A, Q)


def pinball_per_asset(y_pred, Y, quantiles):
    """y_pred (N, A, Q), Y (N, A) -> list of A floats."""
    A = Y.shape[1]
    out = []
    for ai in range(A):
        out.append(pinball_np(y_pred[:, ai:ai+1, :],
                              Y[:, ai:ai+1], quantiles))
    return out


def q05_vs_last(y_pred, X, ai, quantiles):
    """Pearson r entre q=0.5 predicho y last_return de la ventana, por activo."""
    if len(X) == 0:
        return float("nan")
    q05_idx = int(np.argmin(np.abs(np.asarray(quantiles) - 0.5)))
    med = y_pred[:, ai, q05_idx]
    last = X[:, -1, ai]
    if med.std() < 1e-12 or last.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(med, last)[0, 1])


# ================================================================
# Main
# ================================================================
def run_sweep():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_ret = load_returns(DATA_DIR)
    base_cfg = DLConfig()
    quantiles = list(base_cfg.quantiles)
    assets = list(base_cfg.assets)
    A = len(assets)

    print("=" * 70)
    print("SWEEP LSTM + LINEAR BASELINE")
    print("=" * 70)
    print(f"  quantiles: {quantiles}")
    print(f"  assets   : {assets}")
    print(f"  T weeks  : {len(df_ret)}")
    print(f"  output   : {OUT_DIR}")
    print("-" * 70)

    sweep = [
        ("LSTM_h24_l2_H60", dict(lstm_hidden=24, lstm_layers=2, H=60)),
        ("LSTM_h16_l1_H60", dict(lstm_hidden=16, lstm_layers=1, H=60)),
        ("LSTM_h8_l1_H24",  dict(lstm_hidden=8,  lstm_layers=1, H=24)),
        ("LSTM_h4_l1_H12",  dict(lstm_hidden=4,  lstm_layers=1, H=12)),
    ]
    linear_Hs = [12, 24, 60]

    rows = []

    # ---- LSTMs ----
    for name, overrides in sweep:
        cfg = replace(base_cfg, **overrides)
        print(f"\n[{name}]  H={cfg.H}  hidden={cfg.lstm_hidden}  layers={cfg.lstm_layers}")
        result = train_deciles(cfg)

        net = QuantileLSTM(cfg)
        net.load_state_dict(result.state_dict)
        net.eval()
        loaded = LoadedModel(nets=[net], config=cfg,
                             mean=result.mean, std=result.std)
        params = n_params(net)

        X, Y, t_idx = build_windows(df_ret, cfg.H)
        split = chrono_split(X, Y, t_idx, cfg.split)
        q_const = constant_quantiles(split.Y_train, quantiles)
        n_tr_eff = len(split.X_train) * A  # puntos efectivos
        print(f"  n_train={len(split.X_train)}  n_valid={len(split.X_valid)}  "
              f"n_test={len(split.X_test)}  params={params}  "
              f"ratio_params_data={params/max(n_tr_eff,1):.2f}")
        print(f"  best_seed={result.best_seed}  best_valid={result.best_valid:.5f}")

        for split_name, X_, Y_ in [("train", split.X_train, split.Y_train),
                                   ("valid", split.X_valid, split.Y_valid),
                                   ("test",  split.X_test,  split.Y_test)]:
            if len(X_) == 0:
                continue
            y_pred = eval_lstm_split(loaded, X_, Y_, quantiles)
            pbs = pinball_per_asset(y_pred, Y_, quantiles)
            n_obs = Y_.shape[0]
            for ai, a in enumerate(assets):
                y_const = np.broadcast_to(q_const[None, ai:ai+1, :],
                                          (n_obs, 1, len(quantiles)))
                pb_c = pinball_np(y_const, Y_[:, ai:ai+1], quantiles)
                r = q05_vs_last(y_pred, X_, ai, quantiles)
                rows.append({
                    "model":          name,
                    "params":         params,
                    "split":          split_name,
                    "asset":          a,
                    "n":              n_obs,
                    "pinball":        pbs[ai],
                    "pinball_const":  pb_c,
                    "skill":          1.0 - pbs[ai] / pb_c if pb_c > 0 else float("nan"),
                    "r_q05_vs_last":  r,
                })

    # ---- Linear baselines ----
    for H in linear_Hs:
        cfg_lin = replace(base_cfg, H=H)
        X, Y, t_idx = build_windows(df_ret, cfg_lin.H)
        split = chrono_split(X, Y, t_idx, cfg_lin.split)
        F_tr = extract_features(split.X_train)
        F_va = extract_features(split.X_valid)
        F_te = extract_features(split.X_test)
        name = f"Linear_H{H}"
        print(f"\n[{name}]  features={F_tr.shape[1]}  "
              f"n_train={len(F_tr)}  n_valid={len(F_va)}  n_test={len(F_te)}")
        model, f_mean, f_std, best_seed, best_valid = train_linear(
            F_tr, split.Y_train, F_va, split.Y_valid, quantiles,
        )
        params = n_params(model)
        print(f"  params={params}  best_seed={best_seed}  best_valid={best_valid:.5f}")
        q_const = constant_quantiles(split.Y_train, quantiles)

        for split_name, F_, Y_, X_ in [
            ("train", F_tr, split.Y_train, split.X_train),
            ("valid", F_va, split.Y_valid, split.X_valid),
            ("test",  F_te, split.Y_test,  split.X_test),
        ]:
            if len(F_) == 0:
                continue
            y_pred = predict_linear(model, F_, f_mean, f_std)
            pbs = pinball_per_asset(y_pred, Y_, quantiles)
            n_obs = Y_.shape[0]
            for ai, a in enumerate(assets):
                y_const = np.broadcast_to(q_const[None, ai:ai+1, :],
                                          (n_obs, 1, len(quantiles)))
                pb_c = pinball_np(y_const, Y_[:, ai:ai+1], quantiles)
                r = q05_vs_last(y_pred, X_, ai, quantiles)
                rows.append({
                    "model":          name,
                    "params":         params,
                    "split":          split_name,
                    "asset":          a,
                    "n":              n_obs,
                    "pinball":        pbs[ai],
                    "pinball_const":  pb_c,
                    "skill":          1.0 - pbs[ai] / pb_c if pb_c > 0 else float("nan"),
                    "r_q05_vs_last":  r,
                })

    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "results.csv"
    df.to_csv(csv_path, index=False)

    # ---- Resumen en pantalla ----
    print("\n" + "=" * 70)
    print("SKILL POR (modelo, split, asset)  -- positivo = mejor que constante")
    print("=" * 70)
    pivot = df.pivot_table(
        index=["model", "params"],
        columns=["split", "asset"],
        values="skill",
    )
    pivot = pivot[[c for c in [("train", "SPX"), ("train", "CMC200"),
                                ("valid", "SPX"), ("valid", "CMC200"),
                                ("test",  "SPX"), ("test",  "CMC200")]
                   if c in pivot.columns]]
    print(pivot.round(4).to_string())

    print("\n" + "-" * 70)
    print("CORRELACION q=0.5 predicho vs last_return (train set)")
    print("-" * 70)
    tr = df[df.split == "train"][["model", "asset", "r_q05_vs_last"]]
    print(tr.pivot(index="model", columns="asset", values="r_q05_vs_last")
            .round(3).to_string())

    # ---- Plots ----
    # 1) Skill en test (positivo verde, negativo rojo).
    df_test = df[df.split == "test"].copy()
    fig, axes = plt.subplots(1, A, figsize=(13, 5))
    if A == 1:
        axes = [axes]
    for ai, a in enumerate(assets):
        ax = axes[ai]
        sub = df_test[df_test.asset == a].sort_values("skill")
        colors = ["C2" if s > 0 else "C3" for s in sub.skill]
        ax.barh(sub.model, sub.skill, color=colors)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(f"{a} - skill en test")
        ax.set_xlabel("skill = 1 - pinball/baseline")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "skill_test.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # 2) Heatmap skill (model x split-asset).
    fig, ax = plt.subplots(figsize=(11, max(4, 0.6 * len(pivot.index))))
    vals = pivot.values
    im = ax.imshow(vals, cmap="RdYlGn", vmin=-0.1, vmax=0.1, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{s}/{a}" for (s, a) in pivot.columns],
                       rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{m} ({p}p)" for (m, p) in pivot.index])
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.3f}", ha="center", va="center",
                        fontsize=8, color="black")
    plt.colorbar(im, ax=ax, label="skill")
    ax.set_title("Skill (1 - pinball/constante) por modelo")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "skill_heatmap.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    print("\n" + "-" * 70)
    print(f"Resultados : {csv_path}")
    print(f"Skill test : {OUT_DIR / 'skill_test.png'}")
    print(f"Heatmap    : {OUT_DIR / 'skill_heatmap.png'}")


if __name__ == "__main__":
    run_sweep()
