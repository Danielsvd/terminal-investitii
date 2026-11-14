"""Testul istoric al ideilor pe ETF-uri (secțiunea „Idei și strategii”): A. trend pe clase de active și
B. rotația sectoarelor SUA cu momentum dublu. Se rulează local (Yahoo și FRED blochează mediul din cloud).

Exemplu (din rădăcina repo-ului):
    python -m tools.backtest_idei --out rezultat_idei.csv
    python -m tools.backtest_idei --out rezultat_idei.csv --reincarca     # descarcă din nou prețurile

Ce face:
  1. Descarcă prețurile ajustate (dividende incluse) ale ETF-urilor din analytics/ideas.py, plus SPY, și randamentul
     titlurilor SUA la 3 luni din FRED (DGS3MO). Totul în tools/cache/ (a doua rulare nu mai descarcă).
  2. La fiecare sfârșit de lună aplică EXACT funcțiile din aplicație (ideas.trend_signal, ideas.dual_momentum_table)
     pe datele de până atunci și ține poziția o lună:
       A. fiecare clasă de active are o felie egală; felia e în activ doar dacă starea e „ÎN TREND”, altfel în numerar.
          Comparat cu aceleași active ținute permanent, ponderi egale.
       B. primele 3 sectoare care trec filtrul absolut, ponderi egale; locurile neocupate stau în numerar.
          Comparat cu SPY.
  3. Cost: 0,10% din valoarea tranzacționată (rotația ponderilor) la fiecare rebalansare.
  4. Rezultatele pe perioade: construire (până la 2012 inclusiv) și verificare (2013 → sfârșit), plus tot intervalul:
     randament anualizat, volatilitate, Sharpe (față de numerar), scăderea maximă, % luni pozitive, N luni.
     Seria lunară se scrie în --out, rezumatul în <out>_rezumat.csv.

Limite: ETF-urile au istorii diferite (UUP din 2007, XLC din 2018), deci universul crește în timp; numerarul e un
proxy (DGS3MO ÷ 12 pe lună); fără impozite și fără efect valutar (totul în USD). Rezultatul e orientativ.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analytics import ideas  # noqa: E402
from analytics.backtest import month_end_dates  # noqa: E402
from tools.backtest_scor import CACHE, download  # noqa: E402

COST = 0.001
SPLIT = "2012-12-31"
BENCH = "SPY"


def download_cash(refresh=False):
    """FRED DGS3MO (procente, zilnic), din cache dacă există. Serie goală la eșec."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / "fred_DGS3MO.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0].dropna()
    for attempt in (1, 2):
        try:
            import pandas_datareader.data as web
            ser = web.DataReader("DGS3MO", "fred", "1995-01-01", pd.Timestamp.today()).iloc[:, 0].dropna()
            ser.to_frame("DGS3MO").to_csv(path)
            return ser
        except Exception as e:  # noqa: BLE001 — pandas_datareader aruncă tipuri diferite
            print(f"  FRED DGS3MO: eroare ({type(e).__name__}: {e})")
            if attempt == 2:
                return pd.Series(dtype=float)
            time.sleep(5)


def _at(close, day):
    part = close.loc[:day].dropna()
    return None if part.empty else float(part.iloc[-1])


def monthly_returns(closes, dates):
    """Randamentele lunare (de la o dată din `dates` la următoarea) pentru fiecare simbol: DataFrame dată × simbol.
    NaN dacă simbolul nu are preț la ambele capete."""
    rows = {}
    for d0, d1 in zip(dates[:-1], dates[1:]):
        rows[d0] = {s: (lambda p0, p1: p1 / p0 - 1 if (p0 and p1) else np.nan)(_at(c, d0), _at(c, d1))
                    for s, c in closes.items()}
    return pd.DataFrame.from_dict(rows, orient="index")


def run_strategies(closes, cash_rates, start, end, cost=COST):
    """Simularea lunară a strategiilor A și B. `closes`: {simbol: serie ajustată}; `cash_rates`: DGS3MO (%).
    Întoarce DataFrame pe lună: trend, trend_bh, sector, spy, cash (randamente lunare nete de costuri) și
    n_trend / sectors (ce s-a ținut)."""
    ref = closes[BENCH]
    dates = month_end_dates(ref.index, start, end)
    rets = monthly_returns(closes, dates)
    cash_rates = cash_rates.dropna()
    out, w_trend_prev, w_sec_prev = [], {}, {}
    for d0 in dates[:-1]:
        rate = _at(cash_rates, d0)
        cash_m = (rate / 100 / 12) if rate is not None else 0.0
        cash12 = ideas.cash_return_12m(cash_rates.loc[:d0])
        r = rets.loc[d0]
        # A. trend: felii egale pe activele cu randament în luna următoare
        assets = [s for s, _ in ideas.ASSET_CLASSES if s in closes and pd.notna(r.get(s))]
        w_trend, held = {}, []
        for s in assets:
            st_ = ideas.trend_signal(closes[s].loc[:d0], cash12)["state"]
            if st_ == "ÎN TREND":
                w_trend[s] = 1 / len(assets)
                held.append(s)
        trend_r = sum(w * r[s] for s, w in w_trend.items()) + (1 - sum(w_trend.values())) * cash_m if assets else np.nan
        bh_r = float(np.mean([r[s] for s in assets])) if assets else np.nan
        # B. sectoare: primele 3 care trec filtrul absolut
        sec = {s: closes[s].loc[:d0] for s in ideas.US_SECTORS if s in closes and pd.notna(r.get(s))}
        tab = ideas.dual_momentum_table(sec, cash12) if sec else pd.DataFrame(columns=["symbol", "selected"])
        chosen = list(tab.loc[tab["selected"], "symbol"]) if len(tab) else []
        w_sec = {s: 1 / ideas.TOP_SECTORS for s in chosen}
        sec_r = sum(w * r[s] for s, w in w_sec.items()) + (1 - sum(w_sec.values())) * cash_m
        turn_a = sum(abs(w_trend.get(s, 0) - w_trend_prev.get(s, 0)) for s in set(w_trend) | set(w_trend_prev))
        turn_b = sum(abs(w_sec.get(s, 0) - w_sec_prev.get(s, 0)) for s in set(w_sec) | set(w_sec_prev))
        out.append({"date": d0, "trend": trend_r - cost * turn_a, "trend_bh": bh_r,
                    "sector": sec_r - cost * turn_b, "spy": r.get(BENCH), "cash": cash_m,
                    "n_trend": f"{len(held)}/{len(assets)}", "sectors": " ".join(chosen)})
        w_trend_prev, w_sec_prev = w_trend, w_sec
    return pd.DataFrame(out)


def perf_stats(rets, cash):
    """Randament anualizat, volatilitate anualizată, Sharpe (exces față de numerar), scădere maximă, % luni
    pozitive, N. Pe randamente lunare; None sub 12 luni."""
    df = pd.concat([pd.Series(rets, dtype=float), pd.Series(cash, dtype=float)], axis=1).dropna()
    n = len(df)
    if n < 12:
        return {"N_luni": n, "anualizat": None, "volatilitate": None, "sharpe": None, "scadere_max": None,
                "luni_pozitive": None}
    r, c = df.iloc[:, 0], df.iloc[:, 1]
    curve = (1 + r).cumprod()
    ex = r - c
    vol = float(r.std(ddof=1) * np.sqrt(12))
    return {"N_luni": n, "anualizat": float(curve.iloc[-1] ** (12 / n) - 1), "volatilitate": vol,
            "sharpe": float(ex.mean() / ex.std(ddof=1) * np.sqrt(12)) if ex.std(ddof=1) > 0 else None,
            "scadere_max": float((curve / curve.cummax() - 1).min()), "luni_pozitive": float((r > 0).mean())}


def summary(res, split=SPLIT):
    rows = []
    periods = {"construire": res["date"] <= pd.Timestamp(split), "verificare": res["date"] > pd.Timestamp(split),
               "tot": res["date"].notna()}
    for per, mask in periods.items():
        part = res[mask]
        for col, label in (("trend", "A. trend (regula)"), ("trend_bh", "A. aceleași active, ținute"),
                           ("sector", "B. rotație sectoare"), ("spy", "SPY")):
            st_ = perf_stats(part[col], part["cash"])
            rows.append({"perioada": per, "strategie": label,
                         "start": part["date"].min(), "sfarsit": part["date"].max(), **st_})
    return pd.DataFrame(rows)


def _fmt(x, pct=True, sign=True):
    """Procent („+4,5%”, fără semn la volatilitate și luni pozitive) sau număr („0,54”); lipsă → „N/A”."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "N/A"
    if not pct:
        return f"{x:.2f}".replace(".", ",")
    return (f"{x * 100:+.1f}%" if sign else f"{x * 100:.1f}%").replace(".", ",")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Testul istoric al ideilor pe ETF-uri (trend și rotație sectoare).")
    ap.add_argument("--start", default="2000-01-01")
    ap.add_argument("--end", default=str(pd.Timestamp.today().date()))
    ap.add_argument("--out", required=True)
    ap.add_argument("--reincarca", action="store_true", help="descarcă din nou prețurile și FRED")
    args = ap.parse_args(argv)

    symbols = [s for s, _ in ideas.ASSET_CLASSES] + list(ideas.US_SECTORS) + [BENCH]
    closes, missing = {}, []
    hist_start = str((pd.Timestamp(args.start) - pd.DateOffset(years=2)).date())
    for s in dict.fromkeys(symbols):
        df = download(s, hist_start, args.end, refresh=args.reincarca)
        if df is None or df.empty:
            missing.append(s)
            continue
        closes[s] = df["adj"].dropna()
    if BENCH not in closes:
        print("SPY lipsește: testul nu poate rula.")
        return 1
    cash = download_cash(refresh=args.reincarca)
    if cash.empty:
        print("FRED DGS3MO lipsește: testul nu poate rula (filtrul absolut are nevoie de numerar).")
        return 1
    res = run_strategies(closes, cash, args.start, args.end)
    res.to_csv(args.out, index=False)
    rez = summary(res)
    rez_path = str(args.out).replace(".csv", "_rezumat.csv")
    rez.to_csv(rez_path, index=False)

    if missing:
        print("Fără date: " + ", ".join(missing))
    for per in ("construire", "verificare", "tot"):
        part = rez[rez["perioada"] == per]
        if part.empty or part["N_luni"].max() == 0:
            continue
        print(f"\n== {per}: {part['start'].min():%m.%Y} – {part['sfarsit'].max():%m.%Y} ==")
        for _, r in part.iterrows():
            print(f"  {r['strategie']:<30} anualizat {_fmt(r['anualizat'])}  vol {_fmt(r['volatilitate'], sign=False)}  "
                  f"Sharpe {_fmt(r['sharpe'], False)}  scădere max {_fmt(r['scadere_max'])}  "
                  f"luni pozitive {_fmt(r['luni_pozitive'], sign=False)}  N = {r['N_luni']}")
    print(f"\nSeria lunară: {args.out}\nRezumat: {rez_path}")
    print("Limite: universul crește în timp (istorii diferite), numerarul e proxy, fără taxe și fără efect valutar.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
