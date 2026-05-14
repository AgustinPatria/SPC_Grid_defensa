"""Estudio comparativo: variantes de LSTM vs variantes de regresion cuantilica.

Memoria: la capa DL es requisito de la arquitectura, pero el tipo (LSTM o
regresion) es flexible. Este experimento corre 6 LSTMs + 6 regresiones contra
el predictor constante, computa skill por split y activo, y selecciona los
2 mejores de cada categoria.

Modelos:

  LSTMs:
    LSTM_h24_l2_H60   actual (control)
    LSTM_h16_l1_H60   1 capa, mismo H
    LSTM_h16_l1_H24   medio (hidden alto, H medio)
    LSTM_h8_l1_H24    chico, H medio
    LSTM_h8_l1_H12    chico, H corto
    LSTM_h4_l1_H12    minimo

  Regresiones cuantilicas (lineales + pinball loss):
    Linear_plain_H12        sin regularizacion (control - overfits)
    Linear_ridge_H12        L2 fuerte (weight_decay=0.5)
    Linear_minimal_H12      solo 2 features por activo (last, std)
    Linear_shrinkage_H12    aprende alpha = mezcla con cuantil empirico
    Linear_minimal_H24      minimal con H mas largo
    Linear_shrinkage_H24    shrinkage con H mas largo

Metricas reportadas:
  - pinball y skill (1 - pinball / pinball_constante) por split y activo
  - correlacion q=0.5 vs last_return (test de anti-momentum espurio)
  - alpha aprendido para los modelos shrinkage
  - ranking por test_skill_avg (promedio sobre activos en test)

Corre con:
    python -m experimentos.sweep_modelos

Salidas en `experimentos/sweep_modelos_out/`:
    results.csv      tabla larga (model x split x asset)
    seleccion.csv    top-2 por categoria
    skill_test.png   barras de skill en test
    skill_heatmap.png  heatmap modelo x split-asset
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


OUT_DIR = PROJECT_ROOT / "experimentos" / "sweep_modelos_out"


# ============================================================
# Pinball helpers
# ============================================================
def pinball_np(y_pred, y_true, quantiles):
    q = np.asarray(quantiles, dtype=np.float32)
    diff = y_true[:, :, None] - y_pred
    return float(np.maximum(q * diff, (q - 1.0) * diff).mean())


def constant_quantiles(Y_train, quantiles):
    """Cuantiles nominales del train target. (A, Q)"""
    return np.quantile(Y_train, quantiles, axis=0).T.astype(np.float32)


def pinball_per_asset(y_pred, Y, quantiles):
    return [pinball_np(y_pred[:, ai:ai+1, :], Y[:, ai:ai+1], quantiles)
            for ai in range(Y.shape[1])]


def n_params(net):
    return sum(p.numel() for p in net.parameters())


def q05_vs_last(y_pred, X, ai, quantiles):
    if len(X) == 0:
        return float("nan")
    q05_idx = int(np.argmin(np.abs(np.asarray(quantiles) - 0.5)))
    med = y_pred[:, ai, q05_idx]
    last = X[:, -1, ai]
    if med.std() < 1e-12 or last.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(med, last)[0, 1])


# ============================================================
# Features
# ============================================================
def features_full(X):
    """X (N, H, A) -> (N, 5A). Por activo: mean, std, last, max, min."""
    A = X.shape[2]
    feats = []
    for ai in range(A):
        feats.append(X[:, :, ai].mean(axis=1))
        feats.append(X[:, :, ai].std(axis=1))
        feats.append(X[:, -1, ai])
        feats.append(X[:, :, ai].max(axis=1))
        feats.append(X[:, :, ai].min(axis=1))
    return np.stack(feats, axis=1).astype(np.float32)


def features_minimal(X):
    """X (N, H, A) -> (N, 2A). Solo last_return y window_std por activo."""
    A = X.shape[2]
    feats = []
    for ai in range(A):
        feats.append(X[:, -1, ai])
        feats.append(X[:, :, ai].std(axis=1))
    return np.stack(feats, axis=1).astype(np.float32)


# ============================================================
# Modelos lineales
# ============================================================
class LinearQuantile(nn.Module):
    def __init__(self, n_features, A, Q):
        super().__init__()
        self.A = A; self.Q = Q
        self.linear = nn.Linear(n_features, A * Q)

    def forward(self, x):
        return self.linear(x).view(-1, self.A, self.Q)


class ShrinkageQuantile(nn.Module):
    """q_hat = alpha * linear(x) + (1 - alpha) * q_emp.
    alpha es escalar aprendido via sigmoid. q_emp es fijo (cuantiles del train).
    Si f no aporta senal, el entrenamiento empuja alpha -> 0 y recuperamos
    el predictor constante."""

    def __init__(self, n_features, A, Q, q_empirical):
        super().__init__()
        self.A = A; self.Q = Q
        self.linear = nn.Linear(n_features, A * Q)
        self.alpha_logit = nn.Parameter(torch.zeros(1))
        self.register_buffer(
            "q_emp", torch.from_numpy(q_empirical.astype(np.float32))
        )

    def forward(self, x):
        a = torch.sigmoid(self.alpha_logit)
        f = self.linear(x).view(-1, self.A, self.Q)
        return a * f + (1 - a) * self.q_emp[None, :, :]

    def alpha(self):
        return float(torch.sigmoid(self.alpha_logit).item())


def train_regression(model_factory, F_tr, Y_tr, F_va, Y_va, quantiles,
                     seeds=(0, 1, 2), epochs=500, lr=5e-3, weight_decay=1e-3):
    """Seed averaging; devuelve (modelo, f_mean, f_std, best_seed, best_valid, alpha)."""
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
        torch.manual_seed(seed); np.random.seed(seed)
        model = model_factory()
        optim = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay,
        )
        seed_best = float("inf")
        seed_state = None
        for _ in range(epochs):
            model.train()
            optim.zero_grad()
            pinball_loss(model(Xt), Yt, quantiles).backward()
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

    model = model_factory()
    model.load_state_dict(best_state)
    model.eval()
    alpha = model.alpha() if hasattr(model, "alpha") else None
    return (model, f_mean.astype(np.float32), f_std.astype(np.float32),
            best_seed, best_valid, alpha)


# ============================================================
# Inferencia
# ============================================================
def predict_linear(model, F, f_mean, f_std):
    Fs = ((F - f_mean) / f_std).astype(np.float32)
    with torch.no_grad():
        out = model(torch.from_numpy(Fs)).numpy()
    return np.sort(out, axis=-1)


def eval_lstm(loaded, X):
    Xs = ((X - loaded.mean) / loaded.std).astype(np.float32)
    with torch.no_grad():
        outs = [net(torch.from_numpy(Xs)).numpy() for net in loaded.nets]
    y_pred = np.mean(np.stack(outs, axis=0), axis=0)
    return np.sort(y_pred, axis=-1)


def evaluate_model(name, params, category, predict_fn, X_dict, Y_dict,
                   assets, quantiles, q_const_tr, alpha=None):
    rows = []
    for split_name in ("train", "valid", "test"):
        X_, Y_ = X_dict[split_name], Y_dict[split_name]
        if len(X_) == 0:
            continue
        y_pred = predict_fn(X_)
        pbs = pinball_per_asset(y_pred, Y_, quantiles)
        n_obs = Y_.shape[0]
        for ai, a in enumerate(assets):
            y_const = np.broadcast_to(q_const_tr[None, ai:ai+1, :],
                                      (n_obs, 1, len(quantiles)))
            pb_c = pinball_np(y_const, Y_[:, ai:ai+1], quantiles)
            r = q05_vs_last(y_pred, X_, ai, quantiles)
            rows.append({
                "model":           name,
                "category":        category,
                "params":          params,
                "split":           split_name,
                "asset":           a,
                "n":               n_obs,
                "pinball":         pbs[ai],
                "pinball_const":   pb_c,
                "skill":           1.0 - pbs[ai] / pb_c if pb_c > 0 else float("nan"),
                "r_q05_vs_last":   r,
                "alpha_shrinkage": alpha if alpha is not None else float("nan"),
            })
    return rows


# ============================================================
# Main
# ============================================================
def run_sweep():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_ret = load_returns(DATA_DIR)
    base_cfg = DLConfig()
    quantiles = list(base_cfg.quantiles)
    assets = list(base_cfg.assets)
    A = len(assets)
    Q = len(quantiles)

    print("=" * 70)
    print("SWEEP DL: LSTM vs REGRESION CUANTILICA")
    print("=" * 70)
    print(f"  assets   : {assets}")
    print(f"  quantiles: {quantiles}")
    print(f"  T weeks  : {len(df_ret)}")
    print(f"  output   : {OUT_DIR}")
    print("-" * 70)

    rows = []

    # ============================================================
    # LSTMs
    # ============================================================
    lstm_configs = [
        ("LSTM_h24_l2_H60", dict(lstm_hidden=24, lstm_layers=2, H=60)),
        ("LSTM_h16_l1_H60", dict(lstm_hidden=16, lstm_layers=1, H=60)),
        ("LSTM_h16_l1_H24", dict(lstm_hidden=16, lstm_layers=1, H=24)),
        ("LSTM_h8_l1_H24",  dict(lstm_hidden=8,  lstm_layers=1, H=24)),
        ("LSTM_h8_l1_H12",  dict(lstm_hidden=8,  lstm_layers=1, H=12)),
        ("LSTM_h4_l1_H12",  dict(lstm_hidden=4,  lstm_layers=1, H=12)),
    ]

    for name, overrides in lstm_configs:
        cfg = replace(base_cfg, **overrides)
        print(f"\n[LSTM] {name}  H={cfg.H} hidden={cfg.lstm_hidden} layers={cfg.lstm_layers}")
        result = train_deciles(cfg)
        net = QuantileLSTM(cfg)
        net.load_state_dict(result.state_dict)
        net.eval()
        loaded = LoadedModel(nets=[net], config=cfg,
                             mean=result.mean, std=result.std)
        params = n_params(net)

        X, Y, t_idx = build_windows(df_ret, cfg.H)
        split = chrono_split(X, Y, t_idx, cfg.split)
        q_const_tr = constant_quantiles(split.Y_train, quantiles)
        print(f"  n_tr={len(split.X_train)} n_va={len(split.X_valid)} n_te={len(split.X_test)}  params={params}  best_valid={result.best_valid:.5f}")

        X_dict = {"train": split.X_train, "valid": split.X_valid, "test": split.X_test}
        Y_dict = {"train": split.Y_train, "valid": split.Y_valid, "test": split.Y_test}

        def predict_fn(X_, _loaded=loaded):
            return eval_lstm(_loaded, X_)

        rows.extend(evaluate_model(
            name, params, "LSTM", predict_fn,
            X_dict, Y_dict, assets, quantiles, q_const_tr,
        ))

    # ============================================================
    # Regresiones
    # ============================================================
    regr_configs = [
        # (name, H, feature_fn, kind, weight_decay)
        ("Linear_plain_H12",     12, features_full,    "plain",     1e-3),
        ("Linear_ridge_H12",     12, features_full,    "plain",     5e-1),
        ("Linear_minimal_H12",   12, features_minimal, "plain",     1e-3),
        ("Linear_shrinkage_H12", 12, features_full,    "shrinkage", 1e-3),
        ("Linear_minimal_H24",   24, features_minimal, "plain",     1e-3),
        ("Linear_shrinkage_H24", 24, features_full,    "shrinkage", 1e-3),
    ]

    for name, H, feat_fn, kind, wd in regr_configs:
        cfg = replace(base_cfg, H=H)
        X, Y, t_idx = build_windows(df_ret, cfg.H)
        split = chrono_split(X, Y, t_idx, cfg.split)
        F_tr = feat_fn(split.X_train)
        F_va = feat_fn(split.X_valid)
        F_te = feat_fn(split.X_test)
        q_const_tr = constant_quantiles(split.Y_train, quantiles)
        print(f"\n[REG]  {name}  H={H}  features={F_tr.shape[1]}  kind={kind}  wd={wd}")

        if kind == "shrinkage":
            factory = lambda nf=F_tr.shape[1], qe=q_const_tr: \
                ShrinkageQuantile(nf, A, Q, qe)
        else:
            factory = lambda nf=F_tr.shape[1]: LinearQuantile(nf, A, Q)

        model, f_mean, f_std, best_seed, best_valid, alpha = train_regression(
            factory, F_tr, split.Y_train, F_va, split.Y_valid, quantiles,
            weight_decay=wd,
        )
        params = n_params(model)
        alpha_str = f"  alpha={alpha:.3f}" if alpha is not None else ""
        print(f"  n_tr={len(F_tr)} n_va={len(F_va)} n_te={len(F_te)}  params={params}  best_valid={best_valid:.5f}{alpha_str}")

        X_dict = {"train": split.X_train, "valid": split.X_valid, "test": split.X_test}
        Y_dict = {"train": split.Y_train, "valid": split.Y_valid, "test": split.Y_test}

        def predict_fn(X_, _model=model, _fmean=f_mean, _fstd=f_std,
                       _feat=feat_fn):
            return predict_linear(_model, _feat(X_), _fmean, _fstd)

        rows.extend(evaluate_model(
            name, params, "Regression", predict_fn,
            X_dict, Y_dict, assets, quantiles, q_const_tr, alpha=alpha,
        ))

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "results.csv", index=False)

    # ============================================================
    # Resumen y seleccion
    # ============================================================
    print("\n" + "=" * 70)
    print("SKILL POR (modelo, split, asset)")
    print("=" * 70)
    pivot = df.pivot_table(
        index=["category", "model", "params"],
        columns=["split", "asset"], values="skill",
    )
    cols = [c for c in [("train", "SPX"), ("train", "CMC200"),
                        ("valid", "SPX"), ("valid", "CMC200"),
                        ("test",  "SPX"), ("test",  "CMC200")]
            if c in pivot.columns]
    pivot = pivot[cols]
    print(pivot.round(4).to_string())

    df_test = df[df.split == "test"]
    score = (df_test.groupby(["category", "model", "params"])
                    .agg(test_skill_avg=("skill", "mean"))
                    .reset_index()
                    .sort_values(["category", "test_skill_avg"],
                                 ascending=[True, False]))
    print("\n" + "-" * 70)
    print("RANKING POR test_skill_avg (promedio sobre activos en test)")
    print("-" * 70)
    print(score.round(4).to_string(index=False))

    print("\n" + "=" * 70)
    print("SELECCION: TOP 2 POR CATEGORIA")
    print("=" * 70)
    top_rows = []
    for cat in score.category.unique():
        sub = score[score.category == cat].head(2)
        for _, row in sub.iterrows():
            top_rows.append(row)
            print(f"  {cat:11s}  {row.model:25s}  "
                  f"params={int(row.params):>5d}  "
                  f"test_skill_avg={row.test_skill_avg:+.4f}")
    pd.DataFrame(top_rows).to_csv(OUT_DIR / "seleccion.csv", index=False)

    # ============================================================
    # Plots
    # ============================================================
    fig, axes = plt.subplots(1, A, figsize=(14, 7))
    if A == 1:
        axes = [axes]
    cat_colors = {"LSTM": "C0", "Regression": "C4"}
    for ai, a in enumerate(assets):
        ax = axes[ai]
        sub = df_test[df_test.asset == a].sort_values("skill")
        colors = []
        edge_colors = []
        for cat, sk in zip(sub.category, sub.skill):
            colors.append("C2" if sk > 0 else "C3")
            edge_colors.append(cat_colors[cat])
        ax.barh(sub.model, sub.skill, color=colors,
                edgecolor=edge_colors, linewidth=2)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(f"{a} - skill en test  "
                     f"(borde azul=LSTM, morado=Regr)")
        ax.set_xlabel("skill")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "skill_test.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    vals = pivot.values
    fig, ax = plt.subplots(figsize=(11, max(5, 0.45 * len(pivot.index))))
    im = ax.imshow(vals, cmap="RdYlGn", vmin=-0.1, vmax=0.1, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{s}/{a}" for (s, a) in pivot.columns],
                       rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"[{c}] {m} ({p}p)" for (c, m, p) in pivot.index])
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.3f}", ha="center", va="center",
                        fontsize=7)
    plt.colorbar(im, ax=ax, label="skill (clipped ±0.1)")
    ax.set_title("Skill por modelo y split-asset")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "skill_heatmap.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"\n--> resultados : {OUT_DIR / 'results.csv'}")
    print(f"--> seleccion  : {OUT_DIR / 'seleccion.csv'}")
    print(f"--> skill_test : {OUT_DIR / 'skill_test.png'}")
    print(f"--> heatmap    : {OUT_DIR / 'skill_heatmap.png'}")


if __name__ == "__main__":
    run_sweep()
