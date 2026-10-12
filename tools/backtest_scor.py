"""Testul istoric al sub-scorului tehnic. Se rulează local (Yahoo blochează mediul din cloud).

Exemple (din rădăcina repo-ului):
    python -m tools.backtest_scor --piata us  --simboluri tools/univers_sua.txt simboluri_privat.txt --out rezultat_sua.csv
    python -m tools.backtest_scor --piata bvb --simboluri tools/univers_bvb.txt --out rezultat_bvb.csv

Ce face:
  1. Descarcă din Yahoo istoricul zilnic al fiecărui simbol (o singură dată; păstrat în tools/cache/).
  2. La fiecare sfârșit de lună din perioadă calculează sub-scorul tehnic DOAR din datele de până atunci
     (aceeași funcție ca în aplicație: analytics/scoring.py).
  3. Notează randamentul din următoarele 1, 3, 6, 12 luni (prețuri ajustate pentru dividende și split-uri).
  4. Scrie toate observațiile în fișierul --out și un rezumat pe trepte de scor în <out>_rezumat.csv.

Fișierele rezultate și lista privată de simboluri nu se pun în repo (sunt în .gitignore).
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analytics.backtest import (HORIZONS, bucket_summary, forward_return, month_end_dates,  # noqa: E402
                                rank_ic, score_rows)

CACHE = ROOT / "tools" / "cache"
BENCHMARKS = {"us": "SPY", "bvb": None}   # BVB: nu există în Yahoo un istoric lung al BET (TVBETETF.RO e din 2021)
SCORE_COLUMNS = ("tech_score", "trend", "momentum", "rsi_c")
WARMUP = pd.DateOffset(months=14)        # istoric înainte de start: SMA200 și momentum 12-1


def _ic_columns(ic):
    """Câmpurile IC cu prefix `ic_`, ca să nu suprascrie coloanele tabelului (pct_pozitiv, nr_date)."""
    return {k if k.startswith("ic_") else f"ic_{k}": v for k, v in ic.items()}


def read_symbols(paths):
    """Simbolurile din fișiere text: unul pe rând, „#" începe un comentariu, dubluri eliminate."""
    out = []
    for p in paths:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            sym = line.split("#", 1)[0].strip().upper()
            if sym and sym not in out:
                out.append(sym)
    return out


def download(symbol, start, end, refresh=False):
    """Istoricul zilnic: coloanele adj (ajustat), raw (neajustat), volume. DataFrame gol la eșec.

    Un singur apel cu auto_adjust=False dă atât Close (doar split) cât și Adj Close (dividend + split).
    """
    import yfinance as yf

    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{symbol.replace('^', '_')}.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path, index_col=0, parse_dates=True)
    for attempt in (1, 2):
        try:
            h = yf.Ticker(symbol).history(start=start, end=end, auto_adjust=False, actions=False)
            break
        except Exception as e:  # noqa: BLE001 — yfinance aruncă tipuri diferite între versiuni
            print(f"  {symbol}: eroare la descărcare ({type(e).__name__}: {e})")
            if attempt == 2:
                return pd.DataFrame()
            time.sleep(10)
    if h is None or h.empty or "Close" not in h.columns:
        return pd.DataFrame()
    h.index = pd.DatetimeIndex(h.index).tz_localize(None).normalize()
    adj = h["Adj Close"] if "Adj Close" in h.columns else None
    if adj is None:
        print(f"  {symbol}: Yahoo nu a trimis Adj Close; randamentele vor fi pe prețuri neajustate.")
        adj = h["Close"]
    out = pd.DataFrame({"adj": adj, "raw": h["Close"], "volume": h.get("Volume")})
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out.to_csv(path)
    time.sleep(1)  # pauză între simboluri, ca Yahoo să nu blocheze cu 429
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Testul istoric al sub-scorului tehnic.")
    ap.add_argument("--piata", choices=["us", "bvb"], required=True)
    ap.add_argument("--simboluri", nargs="+", required=True, help="fișiere text cu simboluri")
    ap.add_argument("--start", default="2012-01-01")
    ap.add_argument("--end", default="2025-12-31", help="ultima dată de scor (randamentele merg după ea)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reincarca", action="store_true", help="ignoră cache-ul local și descarcă din nou")
    args = ap.parse_args(argv)

    symbols = read_symbols(args.simboluri)
    bench = BENCHMARKS[args.piata]
    dl_start = (pd.Timestamp(args.start) - WARMUP).date().isoformat()
    dl_end = (pd.Timestamp.today().normalize() + pd.Timedelta(days=1)).date().isoformat()
    print(f"{len(symbols)} simboluri, piața {args.piata}, scor {args.start} → {args.end}")

    bench_close = None
    if bench:
        b = download(bench, dl_start, dl_end, args.reincarca)
        bench_close = None if b.empty else b["adj"].dropna()
        if bench_close is None:
            print(f"  Indicele {bench} nu a putut fi descărcat; coloanele bench_* rămân goale.")

    rows, failed, short = [], [], []
    for i, sym in enumerate(symbols, 1):
        data = download(sym, dl_start, dl_end, args.reincarca)
        if data.empty or data["adj"].dropna().empty:
            failed.append(sym)
            print(f"[{i}/{len(symbols)}] {sym}: fără date")
            continue
        dates = month_end_dates(data["adj"].dropna().index, args.start, args.end)
        sym_rows = score_rows(sym, data["adj"], dates,
                              close_raw=data["raw"] if args.piata == "bvb" else None,
                              volume=data["volume"])
        if len(sym_rows) < 24:
            short.append(f"{sym} ({len(sym_rows)} luni)")
        for r in sym_rows:
            r["market"] = args.piata
            if bench_close is not None:
                for h in HORIZONS:
                    r[f"bench_fwd_{h}m"] = forward_return(bench_close, r["date"], h)
        rows.extend(sym_rows)
        print(f"[{i}/{len(symbols)}] {sym}: {len(sym_rows)} luni")

    if not rows:
        print("Nicio observație. Verifică simbolurile și conexiunea.")
        return 1
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    summary = []
    cols = list(SCORE_COLUMNS) + (["tech_score_unadj"] if "tech_score_unadj" in df.columns else [])
    for col in cols:
        scaled = df.copy()
        if col in ("trend", "momentum", "rsi_c"):
            scaled[col] = pd.to_numeric(scaled[col], errors="coerce") * 100
        for h in HORIZONS:
            t = bucket_summary(scaled, col, f"fwd_{h}m").reset_index()
            t.insert(0, "orizont_luni", h)
            t.insert(0, "scor", col)
            for k, v in _ic_columns(rank_ic(scaled, col, f"fwd_{h}m")).items():
                t[k] = v
            summary.append(t)
    if "dividend_20d" in df.columns:
        div = df[df["dividend_20d"] == True]  # noqa: E712 — coloana poate avea None
        for col in ("tech_score", "tech_score_unadj"):
            ic = _ic_columns(rank_ic(div, col, "fwd_3m"))
            summary.append(pd.DataFrame([{"scor": f"{col} [doar luni cu dividend]", "orizont_luni": 3, **ic,
                                          "N": int(div[col].notna().sum())}]))
    rez = pd.concat(summary, ignore_index=True)
    rez_path = str(Path(args.out).with_suffix("")) + "_rezumat.csv"
    rez.to_csv(rez_path, index=False)

    print()
    print(f"Observații: {len(df)}  |  simboluri cu date: {df['symbol'].nunique()}  |  "
          f"luni: {df['date'].nunique()}")
    if failed:
        print("Fără date: " + ", ".join(failed))
    if short:
        print("Istoric scurt (sub 24 de luni): " + ", ".join(short))
    main_t = rez[(rez["scor"] == "tech_score") & (rez["orizont_luni"] == 6)]
    if not main_t.empty:
        print("\nSub-scor tehnic → exces față de media universului, 6 luni:")
        print(main_t[["treapta", "N", "exces_mediu", "pct_pozitiv", "nr_date"]].to_string(index=False))
        ic_m, ic_t = main_t["ic_mediu"].iloc[0], main_t["ic_t"].iloc[0]
        print("IC mediu (corelația de rang scor → randament, pe lună): "
              + ("N/A" if pd.isna(ic_m) else f"{ic_m:+.3f}") + "  |  t: " + ("N/A" if pd.isna(ic_t) else f"{ic_t:+.2f}"))
    print(f"\nScris: {args.out}\nScris: {rez_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
