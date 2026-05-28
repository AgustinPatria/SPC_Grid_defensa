"""Analisis ultra profundo de los 4 CSV de entrada del modelo SPC_Grid.

CSVs analizados (carpeta data/):
  - ret_semanal_spx.csv      retorno semanal SPX
  - ret_semanal_cmc200.csv   retorno semanal CMC200
  - prob_spx.csv             prob. de regimen (bear/bull) SPX
  - prob_cmc200.csv          prob. de regimen (bear/bull) CMC200

Cubre: integridad de datos, estadistica descriptiva, estructura temporal
(autocorrelacion, estacionariedad, clustering de volatilidad), relacion
cruzada entre activos, y -- para los prob_*.csv -- que tipo de regimen
representan (signo del retorno vs volatilidad), persistencia y calibracion.

Uso:  python inspeccion/csv_analisis.py
Salidas: inspeccion/csv_analisis_out/  (reporte.txt + PNGs + CSVs)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.stattools import acf, adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = Path(__file__).resolve().parent / "csv_analisis_out"
OUT.mkdir(exist_ok=True)

RET_CSV = {"SPX": "ret_semanal_spx.csv", "CMC200": "ret_semanal_cmc200.csv"}
RET_COL = {"SPX": "ret_semanal_spx", "CMC200": "ret_semanal_cmc200"}
PROB_CSV = {"SPX": "prob_spx.csv", "CMC200": "prob_cmc200.csv"}
ASSETS = ("SPX", "CMC200")

_LOG = []


def say(*args):
    line = " ".join(str(a) for a in args)
    print(line)
    _LOG.append(line)


def h1(txt):
    say("\n" + "=" * 78)
    say(txt)
    say("=" * 78)


def h2(txt):
    say("\n" + "-" * 78)
    say(txt)
    say("-" * 78)


def pct(x):
    return f"{x * 100:+.4f}%"


# =====================================================================
# Carga
# =====================================================================
def load():
    ret, prob = {}, {}
    for a in ASSETS:
        dr = pd.read_csv(DATA / RET_CSV[a])
        dp = pd.read_csv(DATA / PROB_CSV[a])
        dr.columns = [c.strip() for c in dr.columns]
        dp.columns = [c.strip() for c in dp.columns]
        ret[a] = dr
        prob[a] = dp
    return ret, prob


# =====================================================================
# 0) Integridad de datos
# =====================================================================
def seccion_integridad(ret, prob):
    h1("0) INTEGRIDAD DE DATOS")
    frames = {f"ret_{a}": ret[a] for a in ASSETS}
    frames.update({f"prob_{a}": prob[a] for a in ASSETS})

    for name, df in frames.items():
        say(f"\n  [{name}]")
        say(f"    filas={len(df)}  columnas={list(df.columns)}")
        say(f"    dtypes={dict(df.dtypes.astype(str))}")
        say(f"    NaN por columna={dict(df.isna().sum())}")
        ninf = int(np.isinf(df.select_dtypes('number')).sum().sum())
        say(f"    valores inf={ninf}   filas duplicadas={int(df.duplicated().sum())}")
        t = df["t"].values
        contig = np.array_equal(t, np.arange(t.min(), t.min() + len(t)))
        say(f"    t: min={t.min()} max={t.max()} contiguo_sin_saltos={contig}")
        say(f"    t duplicados={int(pd.Series(t).duplicated().sum())}")

    h2("Alineacion entre los 4 CSV")
    tsets = {n: set(df["t"]) for n, df in frames.items()}
    base = tsets[f"ret_{ASSETS[0]}"]
    todos_iguales = all(s == base for s in tsets.values())
    say(f"  todos los CSV cubren el mismo conjunto de t : {todos_iguales}")
    say(f"  largo comun de la serie                     : {len(base)} semanas")


# =====================================================================
# 1) Retornos: estadistica descriptiva
# =====================================================================
def seccion_retornos_descriptiva(r):
    h1("1) RETORNOS - ESTADISTICA DESCRIPTIVA")
    tab = []
    for a in ASSETS:
        x = r[a].values
        n = len(x)
        ann_mean = x.mean() * 52
        ann_vol = x.std(ddof=1) * np.sqrt(52)
        cum = np.prod(1 + x) - 1
        # max drawdown sobre capital compuesto
        cap = np.cumprod(1 + x)
        dd = (cap - np.maximum.accumulate(cap)) / np.maximum.accumulate(cap)
        jb = stats.jarque_bera(x)
        tab.append({
            "activo": a, "n": n,
            "media": x.mean(), "mediana": np.median(x), "std": x.std(ddof=1),
            "min": x.min(), "max": x.max(),
            "p1": np.percentile(x, 1), "p5": np.percentile(x, 5),
            "p95": np.percentile(x, 95), "p99": np.percentile(x, 99),
            "skew": stats.skew(x), "kurt_exc": stats.kurtosis(x),
            "%pos": (x >= 0).mean() * 100,
            "ann_mean": ann_mean, "ann_vol": ann_vol,
            "sharpe_ann": ann_mean / ann_vol if ann_vol else np.nan,
            "cumret": cum, "max_dd": dd.min(),
            "JB_stat": jb.statistic, "JB_p": jb.pvalue,
        })
    df = pd.DataFrame(tab).set_index("activo")
    for a in ASSETS:
        row = df.loc[a]
        say(f"\n  [{a}]  n={int(row['n'])} semanas")
        say(f"    media semanal   = {pct(row['media'])}    mediana = {pct(row['mediana'])}")
        say(f"    std semanal     = {pct(row['std'])}")
        say(f"    min / max       = {pct(row['min'])} / {pct(row['max'])}")
        say(f"    p1/p5/p95/p99   = {pct(row['p1'])} / {pct(row['p5'])} / "
            f"{pct(row['p95'])} / {pct(row['p99'])}")
        say(f"    skew            = {row['skew']:+.3f}     "
            f"kurtosis(exc) = {row['kurt_exc']:+.3f}")
        say(f"    semanas r>=0    = {row['%pos']:.1f}%")
        say(f"    media anualiz.  = {pct(row['ann_mean'])}   "
            f"vol anualiz. = {pct(row['ann_vol'])}")
        say(f"    Sharpe anualiz. = {row['sharpe_ann']:+.3f}")
        say(f"    retorno acumul. = {pct(row['cumret'])}   "
            f"(USD 1 -> USD {1 + row['cumret']:.3f})")
        say(f"    max drawdown    = {pct(row['max_dd'])}")
        verdict = ("NO normal (colas gruesas)" if row["JB_p"] < 0.05
                   else "compatible con normal")
        say(f"    Jarque-Bera     = {row['JB_stat']:.1f}  p={row['JB_p']:.4g}  "
            f"-> {verdict}")
    df.to_csv(OUT / "1_retornos_descriptiva.csv")
    return df


# =====================================================================
# 2) Retornos: estructura temporal
# =====================================================================
def seccion_retornos_temporal(r):
    h1("2) RETORNOS - ESTRUCTURA TEMPORAL (premisa del LSTM)")

    h2("2a) Autocorrelacion del retorno r_t  (¿predice r el proprio r pasado?)")
    say(f"  {'lag':>4}" + "".join(f"{a:>14}" for a in ASSETS))
    acfs = {a: acf(r[a].values, nlags=10, fft=False) for a in ASSETS}
    for lag in range(1, 11):
        say(f"  {lag:>4}" + "".join(f"{acfs[a][lag]:>+14.4f}" for a in ASSETS))
    # banda ~ +-1.96/sqrt(n) => fuera de eso es significativo
    for a in ASSETS:
        n = len(r[a])
        band = 1.96 / np.sqrt(n)
        signif = [lag for lag in range(1, 11) if abs(acfs[a][lag]) > band]
        say(f"  {a}: banda 95% = +-{band:.4f}  -> lags significativos: "
            f"{signif if signif else 'ninguno'}")

    h2("2b) Test Ljung-Box  (H0: retornos son ruido blanco, sin autocorrelacion)")
    for a in ASSETS:
        lb = acorr_ljungbox(r[a].values, lags=[5, 10], return_df=True)
        for lag in (5, 10):
            p = lb.loc[lag, "lb_pvalue"]
            ver = "RECHAZA H0 (hay estructura)" if p < 0.05 else "no rechaza (ruido blanco)"
            say(f"  {a:<8} lags={lag:<3} stat={lb.loc[lag,'lb_stat']:8.2f}  "
                f"p={p:.4g}  -> {ver}")

    h2("2c) Predictibilidad: R2 de un AR(4)  (regresion r_t ~ r_{t-1..t-4})")
    for a in ASSETS:
        x = r[a].values
        X = np.column_stack([x[4 - k - 1:-k - 1] for k in range(4)])
        X = np.column_stack([np.ones(len(X)), X])
        y = x[4:]
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        yhat = X @ beta
        ss_res = np.sum((y - yhat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        say(f"  {a:<8} R2(AR4) = {r2:+.4f}   "
            f"-> {'algo de senal' if r2 > 0.05 else 'practicamente cero'}")
    say("  (un R2 ~ 0 implica que el retorno pasado casi no predice el futuro)")

    h2("2d) Clustering de volatilidad: autocorrelacion de r^2")
    say(f"  {'lag':>4}" + "".join(f"{a:>14}" for a in ASSETS))
    acf2 = {a: acf(r[a].values ** 2, nlags=10, fft=False) for a in ASSETS}
    for lag in range(1, 11):
        say(f"  {lag:>4}" + "".join(f"{acf2[a][lag]:>+14.4f}" for a in ASSETS))
    for a in ASSETS:
        lb = acorr_ljungbox(r[a].values ** 2, lags=[10], return_df=True)
        p = lb.loc[10, "lb_pvalue"]
        ver = ("SI hay clustering de volatilidad" if p < 0.05
               else "sin clustering detectable")
        say(f"  {a:<8} Ljung-Box(r^2, lag10) p={p:.4g}  -> {ver}")

    h2("2e) Estacionariedad: test ADF  (H0: serie NO estacionaria)")
    for a in ASSETS:
        adf = adfuller(r[a].values, autolag="AIC")
        ver = "estacionaria" if adf[1] < 0.05 else "NO estacionaria"
        say(f"  {a:<8} ADF stat={adf[0]:8.3f}  p={adf[1]:.4g}  -> {ver}")

    h2("2f) No-estacionariedad por bloques: media/std en 3 tercios de la serie")
    for a in ASSETS:
        x = r[a].values
        tercios = np.array_split(x, 3)
        say(f"  [{a}]")
        for idx, t in enumerate(tercios, 1):
            say(f"    tercio {idx} (n={len(t):3d}): media={pct(t.mean())}  "
                f"std={pct(t.std(ddof=1))}  cumret={pct(np.prod(1+t)-1)}")


# =====================================================================
# 3) Retornos: relacion cruzada SPX vs CMC200
# =====================================================================
def seccion_retornos_cruzados(r):
    h1("3) RETORNOS - RELACION CRUZADA SPX vs CMC200")
    a, b = ASSETS
    x, y = r[a].values, r[b].values

    corr0 = np.corrcoef(x, y)[0, 1]
    corr_abs = np.corrcoef(np.abs(x), np.abs(y))[0, 1]
    say(f"\n  Correlacion contemporanea  corr(r_SPX, r_CMC)   = {corr0:+.4f}")
    say(f"  Correlacion de magnitudes  corr(|r_SPX|,|r_CMC|) = {corr_abs:+.4f}")

    h2("3a) Cross-correlacion con rezagos  (¿un activo adelanta al otro?)")
    say(f"  {'lag':>5}  {'corr(r_SPX_t, r_CMC_t+lag)':>30}")
    for lag in range(-5, 6):
        if lag < 0:
            c = np.corrcoef(x[-lag:], y[:lag])[0, 1]
        elif lag > 0:
            c = np.corrcoef(x[:-lag], y[lag:])[0, 1]
        else:
            c = corr0
        marca = "  <- contemporaneo" if lag == 0 else ""
        say(f"  {lag:>5}  {c:>+30.4f}{marca}")
    say("  (lag>0: SPX adelanta a CMC ; lag<0: CMC adelanta a SPX)")

    h2("3b) Correlacion movil (ventana 26 sem) - ¿es estable?")
    sx, sy = r[a], r[b]
    roll = sx.rolling(26).corr(sy).dropna()
    say(f"  corr movil: min={roll.min():+.3f}  max={roll.max():+.3f}  "
        f"media={roll.mean():+.3f}  std={roll.std():.3f}")


# =====================================================================
# 4) Probabilidades: descripcion
# =====================================================================
def seccion_prob_descriptiva(prob):
    h1("4) PROBABILIDADES DE REGIMEN - DESCRIPCION")
    for a in ASSETS:
        dp = prob[a]
        suma = (dp["bear"] + dp["bull"])
        err = float((suma - 1.0).abs().max())
        pb = dp["bull"].values
        say(f"\n  [{a}]")
        say(f"    bear+bull=1 ? error maximo = {err:.2e}")
        say(f"    p_bull: min={pb.min():.4f}  max={pb.max():.4f}  "
            f"media={pb.mean():.4f}  mediana={np.median(pb):.4f}  std={pb.std(ddof=1):.4f}")
        bins = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0001]
        hist = np.histogram(pb, bins=bins)[0]
        say("    distribucion de p_bull por tramo:")
        for lo, hi, c in zip(bins[:-1], bins[1:], hist):
            barra = "#" * c
            say(f"      [{lo:.1f},{min(hi,1.0):.1f}) {c:>4}  {barra}")
        say(f"    semanas p_bull>0.5 = {(pb>0.5).sum()}/{len(pb)}")
        extremos = ((pb < 0.1) | (pb > 0.9)).sum()
        mushy = ((pb > 0.4) & (pb < 0.6)).sum()
        say(f"    p_bull extremo (<0.1 o >0.9) = {extremos}/{len(pb)}   "
            f"p_bull ambiguo (0.4-0.6) = {mushy}/{len(pb)}")


# =====================================================================
# 5) Probabilidades: ¿que tipo de regimen son?
# =====================================================================
def seccion_prob_naturaleza(r, prob):
    h1("5) PROBABILIDADES - ¿QUE REGIMEN REPRESENTAN?")

    h2("5a) ¿El regimen corresponde al SIGNO del retorno?")
    say(f"  {'activo':<8}{'corr(p_bull,r)':>16}{'accuracy>0.5':>15}"
        f"{'r|CSVbull':>13}{'r|CSVbear':>13}")
    for a in ASSETS:
        pb = prob[a]["bull"].values
        x = r[a].values
        corr = np.corrcoef(pb, x)[0, 1]
        cls = pb > 0.5
        acc = (cls == (x >= 0)).mean()
        say(f"  {a:<8}{corr:>+16.4f}{acc*100:>14.1f}%"
            f"{pct(x[cls].mean()):>13}{pct(x[~cls].mean()):>13}")
    say("  -> si corr ~ 0 y accuracy ~ 50%, el regimen NO es el signo del retorno")

    h2("5b) ¿El regimen corresponde a la VOLATILIDAD?")
    say(f"  {'activo':<8}{'corr(p_bear,|r|)':>18}{'corr(p_bear,r^2)':>18}"
        f"{'corr(p_bear,vol13)':>20}")
    for a in ASSETS:
        pbear = prob[a]["bear"].values
        x = r[a].values
        vol13 = pd.Series(x).rolling(13).std().bfill().values
        c_abs = np.corrcoef(pbear, np.abs(x))[0, 1]
        c_sq = np.corrcoef(pbear, x ** 2)[0, 1]
        c_vol = np.corrcoef(pbear, vol13)[0, 1]
        say(f"  {a:<8}{c_abs:>+18.4f}{c_sq:>+18.4f}{c_vol:>+20.4f}")
    say("  -> si p_bear correlaciona con |r|/vol, 'bear' = regimen de ALTA VOLATILIDAD")
    say("     (eso explicaria que mu(bear) no sea necesariamente negativo)")

    h2("5c) ¿El regimen ADELANTA al retorno? corr(p_bull_t, r_t+1)")
    for a in ASSETS:
        pb = prob[a]["bull"].values[:-1]
        x_next = r[a].values[1:]
        c = np.corrcoef(pb, x_next)[0, 1]
        say(f"  {a:<8} corr(p_bull_t, r_t+1) = {c:+.4f}  "
            f"-> {'algo predictivo' if abs(c)>0.15 else 'sin poder predictivo'}")

    h2("5d) Persistencia del regimen  (un regimen real es persistente en t)")
    for a in ASSETS:
        pb = prob[a]["bull"].values
        ac1 = acf(pb, nlags=3, fft=False)
        cls = (pb > 0.5).astype(int)
        cambios = int(np.sum(np.abs(np.diff(cls))))
        say(f"  {a:<8} ACF(p_bull) lag1={ac1[1]:+.3f} lag2={ac1[2]:+.3f} "
            f"lag3={ac1[3]:+.3f}")
        say(f"  {'':<8} cambios de regimen (cruces de 0.5) = {cambios}/{len(pb)-1} "
            f"transiciones  -> {'MUY inestable' if cambios>len(pb)*0.35 else 'persistente'}")

    h2("5e) Calibracion: cuando p_bull=X, ¿ocurre r>=0 una fraccion X de las veces?")
    for a in ASSETS:
        pb = prob[a]["bull"].values
        x = r[a].values
        say(f"  [{a}]  {'tramo p_bull':<16}{'n':>5}{'p_bull medio':>14}"
            f"{'frac r>=0 real':>17}")
        bins = [0, .2, .4, .6, .8, 1.0001]
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (pb >= lo) & (pb < hi)
            if m.sum() == 0:
                continue
            say(f"  {'':<8}[{lo:.1f},{min(hi,1.0):.1f})      {m.sum():>5}"
                f"{pb[m].mean():>14.3f}{(x[m]>=0).mean():>17.3f}")
    say("  -> si 'p_bull medio' y 'frac r>=0 real' no se parecen, esta MAL calibrado")


# =====================================================================
# 6) Probabilidades: relacion cruzada
# =====================================================================
def seccion_prob_cruzada(prob):
    h1("6) PROBABILIDADES - RELACION CRUZADA SPX vs CMC200")
    a, b = ASSETS
    pa, pb = prob[a]["bull"].values, prob[b]["bull"].values
    say(f"\n  corr(p_bull_SPX, p_bull_CMC) = {np.corrcoef(pa, pb)[0,1]:+.4f}")
    cls_a, cls_b = pa > 0.5, pb > 0.5
    say(f"  semanas con AMBOS en bull   = {(cls_a & cls_b).sum()}/{len(pa)}")
    say(f"  semanas con AMBOS en bear   = {(~cls_a & ~cls_b).sum()}/{len(pa)}")
    say(f"  semanas en regimen OPUESTO  = {(cls_a ^ cls_b).sum()}/{len(pa)}")


# =====================================================================
# Graficos
# =====================================================================
def graficos(r, prob):
    # retornos: serie + histograma
    fig, ax = plt.subplots(2, 2, figsize=(14, 8))
    for col, a in enumerate(ASSETS):
        x = r[a].values
        ax[0, col].plot(x, lw=0.8)
        ax[0, col].axhline(0, color="k", lw=0.5)
        ax[0, col].set_title(f"Retorno semanal {a}")
        ax[1, col].hist(x, bins=40, density=True, alpha=0.7)
        ax[1, col].set_title(f"Distribucion retorno {a}")
    fig.tight_layout()
    fig.savefig(OUT / "1_retornos.png", dpi=110)
    plt.close(fig)

    # prob: serie p_bull + retorno
    fig, ax = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    for a in ASSETS:
        ax[0].plot(prob[a]["bull"].values, label=f"p_bull {a}", lw=1)
    ax[0].axhline(0.5, color="k", lw=0.5, ls="--")
    ax[0].set_title("p_bull(t) - probabilidad de regimen alcista")
    ax[0].legend()
    for a in ASSETS:
        ax[1].plot(r[a].values, label=f"r {a}", lw=0.8)
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set_title("Retorno semanal")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(OUT / "2_prob_vs_retorno.png", dpi=110)
    plt.close(fig)

    # vol movil vs p_bear
    fig, ax = plt.subplots(1, 2, figsize=(14, 4))
    for col, a in enumerate(ASSETS):
        x = r[a].values
        vol = pd.Series(x).rolling(13).std()
        axa = ax[col]
        axa.plot(vol.values, color="tab:red", label="vol movil 13s")
        axb = axa.twinx()
        axb.plot(prob[a]["bear"].values, color="tab:blue", lw=0.8,
                 label="p_bear")
        axa.set_title(f"{a}: volatilidad movil vs p_bear")
    fig.tight_layout()
    fig.savefig(OUT / "3_vol_vs_pbear.png", dpi=110)
    plt.close(fig)


# =====================================================================
def main():
    ret_df, prob_df = load()
    r = {a: ret_df[a].set_index("t")[RET_COL[a]] for a in ASSETS}

    seccion_integridad(ret_df, prob_df)
    seccion_retornos_descriptiva(r)
    seccion_retornos_temporal(r)
    seccion_retornos_cruzados(r)
    seccion_prob_descriptiva(prob_df)
    seccion_prob_naturaleza(r, prob_df)
    seccion_prob_cruzada(prob_df)
    graficos(r, prob_df)

    h1("FIN DEL ANALISIS")
    say(f"  Reporte y figuras en: {OUT}")
    (OUT / "reporte.txt").write_text("\n".join(_LOG), encoding="utf-8")


if __name__ == "__main__":
    main()
