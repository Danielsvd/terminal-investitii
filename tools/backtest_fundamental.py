"""Testul istoric al scorului fundamental SUA/UE. Se rulează local (Yahoo și FRED blochează mediul din cloud).

Exemplu (din rădăcina repo-ului):
    python -m tools.backtest_fundamental --simboluri tools/univers_sua.txt simboluri_privat.txt --out rezultat_fund_sua.csv

Ce face:
  1. Descarcă din Yahoo prețurile zilnice, situațiile financiare anuale (~4 ani), sectorul și split-urile
     fiecărui simbol, plus SPY (beta), ^TNX (rata fără risc) și din FRED randamentul AAA lunar.
     Totul se păstrează în tools/cache/ (a doua rulare nu mai descarcă; --reincarca forțează).
  2. La fiecare sfârșit de lună calculează scorul fundamental DOAR din ce se știa atunci
     (an fiscal publicat = încheiere + 90 de zile), cu aceeași funcție ca aplicația.
  3. Notează randamentul din următoarele 1, 3, 6, 12 luni și îl compară cu media universului la aceeași dată.
  4. Scrie observațiile în --out și rezumatul (pe benzile de verdict, pe cincimi și IC) în <out>_rezumat.csv.

Limite (afișate și la final): ~4 ani de situații din Yahoo → puțini ani fiscali distincți; firmele de azi
(bias de supraviețuire); situații posibil retratate. Rezultatul e orientativ, nu o validare.
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analytics.backtest import HORIZONS, bucket_summary, forward_return, month_end_dates, rank_ic  # noqa: E402
from analytics.backtest_fundamental import (beta_as_of, inputs_at, score_row, share_basis,  # noqa: E402
                                            split_adjust_shares, split_factor_after, value_as_of)
from tools.backtest_scor import CACHE, _ic_columns, download, read_symbols  # noqa: E402

BENCH = "SPY"
RF_SYMBOL = "^TNX"
VERDICT_BINS = (0, 40, 70, 100.0001)          # NEFAVORABIL / NEUTRU / FAVORABIL (scoring.VERDICT_BANDS)
QUINTILE_BINS = (0, 20, 40, 60, 80, 100.0001)
FRED_LAG = pd.DateOffset(months=1)             # media lunii M e publicată la începutul lunii M+1
STATEMENTS = ("income", "balance", "cashflow")


def download_fundamentals(symbol, refresh=False):
    """Situațiile anuale, sectorul și split-urile: {"income", "balance", "cashflow", "sector", "splits"}.
    DataFrame-uri goale / None la eșec."""
    import yfinance as yf

    CACHE.mkdir(parents=True, exist_ok=True)
    base = CACHE / f"fund_{symbol.replace('^', '_')}"
    paths = {k: Path(f"{base}_{k}.csv") for k in STATEMENTS}
    meta_path, splits_path = Path(f"{base}_meta.csv"), Path(f"{base}_splits.csv")
    if not refresh and all(p.exists() for p in paths.values()) and meta_path.exists():
        out = {k: pd.read_csv(p, index_col=0) for k, p in paths.items()}
        meta = pd.read_csv(meta_path)
        out["sector"] = None if meta.empty or pd.isna(meta.loc[0, "sector"]) else str(meta.loc[0, "sector"])
        sp = pd.read_csv(splits_path, index_col=0, parse_dates=True) if splits_path.exists() else pd.DataFrame()
        out["splits"] = sp.iloc[:, 0] if not sp.empty else pd.Series(dtype=float)
        return out
    out = {k: pd.DataFrame() for k in STATEMENTS}
    out.update(sector=None, splits=pd.Series(dtype=float))
    for attempt in (1, 2):
        try:
            t = yf.Ticker(symbol)
            out["income"], out["balance"], out["cashflow"] = t.income_stmt, t.balance_sheet, t.cashflow
            info = t.info or {}
            out["sector"] = info.get("sector")
            out["splits"] = t.splits if t.splits is not None else pd.Series(dtype=float)
            break
        except Exception as e:  # noqa: BLE001 — yfinance aruncă tipuri diferite între versiuni
            print(f"  {symbol}: eroare la situațiile financiare ({type(e).__name__}: {e})")
            if attempt == 2:
                return out
            time.sleep(10)
    for k, p in paths.items():
        df = out[k] if isinstance(out[k], pd.DataFrame) else pd.DataFrame()
        df.to_csv(p)
    pd.DataFrame([{"sector": out["sector"]}]).to_csv(meta_path, index=False)
    sp = pd.Series(out["splits"], dtype=float)
    if len(sp):
        idx = pd.DatetimeIndex(sp.index)
        sp.index = idx.tz_localize(None) if idx.tz is not None else idx
    sp.rename("split").to_frame().to_csv(splits_path)
    out["splits"] = sp
    time.sleep(1)
    return out


def download_fred(code, refresh=False):
    """Seria lunară FRED (procente), din cache sau prin pandas_datareader. Serie goală la eșec."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"FRED_{code}.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    try:
        import pandas_datareader.data as web
        s = web.DataReader(code, "fred", "2005-01-01", pd.Timestamp.today()).iloc[:, 0].dropna()
    except Exception as e:  # noqa: BLE001 — pandas_datareader aruncă tipuri diferite
        print(f"  FRED {code}: eroare ({type(e).__name__}: {e})")
        return pd.Series(dtype=float)
    s.to_frame(code).to_csv(path)
    return s


def prepare_statements(fd):
    """Aduce acțiunile din situații pe baza de split a prețului Yahoo. Întoarce (income, balance, mod, notă)."""
    inc, bal, splits = fd["income"], fd["balance"], fd["splits"]
    mode = share_basis(inc, bal, splits)
    if mode == "reported":
        return split_adjust_shares(inc, splits), split_adjust_shares(bal, splits), mode, None
    if mode in ("restated", "none"):
        return inc, bal, mode, None
    # Nedecis: se păstrează doar anii de după ultimul split (acolo baza e aceeași).
    last = pd.DatetimeIndex(pd.Series(splits).dropna().index).max()
    keep = lambda df: df.loc[:, [c for c in df.columns if pd.Timestamp(c) > last]] if df is not None and not df.empty else df  # noqa: E731
    return keep(inc), keep(bal), "nedecis", f"split nedecis; folosiți doar anii de după {last:%Y-%m-%d}"


def summarize(df):
    """Rezumatul: pe benzile de verdict și pe cincimi pentru scor și piloni, plus IC; orizonturile 1–12 luni."""
    tables = []
    score_cols = ["fund_score"] + [c for c in df.columns if c.startswith("p_")] + [c for c in df.columns if c.startswith("c_")]
    for col in score_cols:
        if pd.to_numeric(df[col], errors="coerce").notna().sum() == 0:
            continue
        for h in HORIZONS:
            ic = _ic_columns(rank_ic(df, col, f"fwd_{h}m"))
            for name, bins in (("benzi", VERDICT_BINS), ("cincimi", QUINTILE_BINS)):
                if name == "benzi" and col != "fund_score":
                    continue
                t = bucket_summary(df, col, f"fwd_{h}m", bins=bins).reset_index()
                t.insert(0, "grupare", name)
                t.insert(0, "orizont_luni", h)
                t.insert(0, "scor", col)
                for k, v in ic.items():
                    t[k] = v
                tables.append(t)
    return pd.DataFrame.from_records([r for t in tables for r in t.to_dict("records")])


def main(argv=None):
    ap = argparse.ArgumentParser(description="Testul istoric al scorului fundamental SUA/UE.")
    ap.add_argument("--simboluri", nargs="+", required=True, help="fișiere text cu simboluri")
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--end", default=pd.Timestamp.today().date().isoformat(), help="ultima dată de scor")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reincarca", action="store_true", help="ignoră cache-ul local și descarcă din nou")
    args = ap.parse_args(argv)

    symbols = read_symbols(args.simboluri)
    dl_start = (pd.Timestamp(args.start) - pd.DateOffset(years=3)).date().isoformat()
    dl_end = (pd.Timestamp.today().normalize() + pd.Timedelta(days=1)).date().isoformat()
    print(f"{len(symbols)} simboluri, scor fundamental {args.start} → {args.end}")

    bench = download(BENCH, dl_start, dl_end, args.reincarca)
    bench_adj = None if bench.empty else bench["adj"].dropna()
    tnx = download(RF_SYMBOL, dl_start, dl_end, args.reincarca)
    tnx_close = None if tnx.empty else tnx["raw"].dropna()
    aaa = download_fred("AAA", args.reincarca)
    missing_global = [n for n, s in ((BENCH, bench_adj), (RF_SYMBOL, tnx_close), ("FRED AAA", aaa))
                      if s is None or len(s) == 0]
    if missing_global:
        print("Lipsesc seriile comune: " + ", ".join(missing_global) + ". Beta, WACC sau E/P față de AAA vor fi N/A.")

    rows, failed, notes = [], [], []
    for i, sym in enumerate(symbols, 1):
        prices = download(sym, dl_start, dl_end, args.reincarca)
        fd = download_fundamentals(sym, args.reincarca)
        if prices.empty or prices["adj"].dropna().empty or fd["balance"] is None or fd["balance"].empty:
            failed.append(sym)
            print(f"[{i}/{len(symbols)}] {sym}: fără prețuri sau fără situații")
            continue
        inc, bal, mode, note = prepare_statements(fd)
        if note:
            notes.append(f"{sym}: {note}")
        adj, raw = prices["adj"].dropna(), prices["raw"].dropna()
        sym_rows, skipped = [], {}
        for d in month_end_dates(adj.index, args.start, args.end):
            price = value_as_of(raw, d)
            rf_pct = value_as_of(tnx_close, d) if tnx_close is not None else None
            rf = None if rf_pct is None or not 0 < rf_pct < 25 else rf_pct / 100
            aaa_pct = value_as_of(aaa, d, publication_lag=FRED_LAG) if len(aaa) else None
            beta = beta_as_of(adj, bench_adj, d) if bench_adj is not None else None
            inp, fy = inputs_at(d, price, inc, bal, fd["cashflow"], fd["sector"], sym, rf, beta, aaa_pct)
            if inp is None:
                skipped[fy] = skipped.get(fy, 0) + 1
                continue
            row = score_row(sym, d, inp, fy)
            row["share_basis"] = mode
            for h in HORIZONS:
                row[f"fwd_{h}m"] = forward_return(adj, d, h)
                row[f"bench_fwd_{h}m"] = forward_return(bench_adj, d, h) if bench_adj is not None else None
            sym_rows.append(row)
        rows.extend(sym_rows)
        msg = f"[{i}/{len(symbols)}] {sym}: {len(sym_rows)} luni"
        if skipped:
            msg += " (sărite: " + ", ".join(f"{k} ×{v}" for k, v in skipped.items()) + ")"
        print(msg)

    if not rows:
        print("Nicio observație. Verifică simbolurile și conexiunea.")
        return 1
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    rez = summarize(df)
    rez_path = str(Path(args.out).with_suffix("")) + "_rezumat.csv"
    rez.to_csv(rez_path, index=False)

    scored = df[pd.to_numeric(df["fund_score"], errors="coerce").notna()]
    print()
    print(f"Observații: {len(df)}  |  cu scor: {len(scored)}  |  simboluri: {df['symbol'].nunique()}  |  "
          f"luni: {df['date'].nunique()}  |  ani fiscali distincți: {df['fiscal_year'].str[:4].nunique()}")
    if failed:
        print("Fără date: " + ", ".join(failed))
    for n in notes:
        print("Notă: " + n)
    for h in (6, 12):
        t = rez[(rez["scor"] == "fund_score") & (rez["orizont_luni"] == h) & (rez["grupare"] == "benzi")]
        if t.empty:
            continue
        print(f"\nScor fundamental → exces față de media universului, {h} luni (benzile de verdict):")
        print(t[["treapta", "N", "exces_mediu", "exces_median", "pct_pozitiv", "nr_date"]].to_string(index=False))
        ic_m, ic_t = t["ic_mediu"].iloc[0], t["ic_t"].iloc[0]
        print("IC mediu: " + ("N/A" if pd.isna(ic_m) else f"{ic_m:+.3f}") + "  |  t: " + ("N/A" if pd.isna(ic_t) else f"{ic_t:+.2f}")
              + "  (t optimist: luni suprapuse și scor schimbat o dată pe an)")
    piv = rez[(rez["orizont_luni"] == 12) & (rez["grupare"] == "cincimi")].drop_duplicates("scor")
    if not piv.empty:
        print("\nIC la 12 luni, pe piloni și componente:")
        for _, r in piv.iterrows():
            print(f"  {r['scor']:<40} " + ("N/A" if pd.isna(r["ic_mediu"]) else f"{r['ic_mediu']:+.3f}")
                  + f"  (date: {int(r['ic_nr_date']) if pd.notna(r['ic_nr_date']) else 0})")
    print("\nLimite: ~4 ani de situații Yahoo, firmele de azi (supraviețuire), situații posibil retratate. Orientativ.")
    print(f"\nScris: {args.out}\nScris: {rez_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
