"""Testul istoric al scorului fundamental SUA pe date SEC (Faza 3c). Se rulează pe MacBook (SEC și Yahoo blochează
mediul din cloud). Propunerea: `claude/propunere_faza3c_test_istoric_2026-11-27.md` (E1–E7 acceptate 27.11.2026).

Prima rulare (descarcă arhivele SEC, ~câțiva GB, și lista simbolurilor; apoi prețurile Yahoo pe ~830 de simboluri):
    python -m tools.backtest_sec --faza construire --descarca
Rulări următoare (din cache, `tools/cache/`):
    python -m tools.backtest_sec --faza construire
Probă rapidă pe câteva simboluri:
    python -m tools.backtest_sec --faza construire --simboluri AAPL MSFT JPM XOM FB
Durata [de verificat]: prețurile Yahoo ~1 s/simbol; versiunile situațiilor ~0,3 s × ~20–70 de versiuni pe companie,
în paralel pe `--procese` (implicit nuclee − 2); a doua rulare citește totul din cache.

Ce face (faza „construire”, 2012–2019; observațiile din 2020+ NU se calculează):
  1. Universul C = membrii S&P 500 la fiecare sfârșit de lună (`tools/univers/sp500_intervale.csv`), B = membrii de
     azi, aplicați pe toată perioada. Simbolurile redenumite: `tools/univers/sp500_alias.csv`.
  2. Pentru fiecare simbol: CIK (lista SEC + alias), situațiile din `companyfacts.zip` „cunoscute la t”
     (`analytics/sec_statements.py`), sectorul din SIC (`submissions.zip`), prețurile și split-urile din Yahoo.
     Perechea se acceptă doar după protecția la simboluri refolosite (`analytics.backtest_sec.accept_mapping`).
  3. La fiecare sfârșit de lună: scorul fundamental cu ACELEAȘI funcții ca aplicația (`inputs_at` → `score_row`),
     plus varianta (2) cu creșterea livrată pe 5 ani; varianta (3), pe rang în sector, se calculează în rezumat.
  4. Randamentele la 1/3/6/12 luni, doar cu capătul ≤ 31.12.2019 (fără prețuri din perioada de verificare);
     delistările din fereastră → randament trunchiat la ultimul preț, marcat.
  5. Scrie în `--out-dir`: observațiile, rezumatul (IC + t Newey-West, cincimi pe ani, benzi față de mediana
     universului și a sectorului; pe B și C; cu și fără trunchiate), acoperirea pe ani, simbolurile fără date.

Limite: companiile fără istoric Yahoo (multe preluate / falimentate) lipsesc din C — acoperirea pe ani o arată; SIC-ul
e cel de azi; situațiile doar anuale (aplicația folosește TTM când are 4 trimestre); etichetele proprii ale companiilor
nu sunt citite.
"""
import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analytics import backtest_sec as bs  # noqa: E402
from analytics import fundamentals as fund  # noqa: E402
from analytics import sec_statements as ss  # noqa: E402
from analytics.backtest import month_end_dates  # noqa: E402
from analytics.backtest_fundamental import beta_as_of, inputs_at, score_row, value_as_of  # noqa: E402
from analytics.global_fundamental import fundamental_score  # noqa: E402
from analytics.insiders import cik_for_ticker  # noqa: E402
from analytics.sic_sector import sic_sector  # noqa: E402
from tools.backtest_fundamental import FRED_LAG, download_fred  # noqa: E402
from tools.backtest_scor import CACHE  # noqa: E402
from tools.backtest_scor import download as _download  # noqa: E402

UNIV = ROOT / "tools" / "univers"
SEC_DIR = CACHE / "sec"
PHASES = {"construire": ("2012-01-01", "2019-12-31"), "verificare": ("2020-01-01", None)}
HORIZONS = (1, 3, 6, 12)
MAIN_HORIZONS = (6, 12)
BENCH, RF_SYMBOL = "SPY", "^TNX"
DL_START = "2009-01-01"
PRICES_DIR = CACHE / "sec" / "preturi"      # cache separat: fișierele altor teste încep mai târziu (ex. 2018)
NO_SCORE_LAG = pd.DateOffset(days=0)        # situațiile SEC sunt deja „cunoscute la dată”


# --- Date ----------------------------------------------------------------------

def read_intervals(path=UNIV / "sp500_intervale.csv"):
    iv = pd.read_csv(path, comment="#")
    iv["de_la"] = pd.to_datetime(iv["de_la"])
    iv["pana_la"] = pd.to_datetime(iv["pana_la"])
    return iv


def read_alias(path=UNIV / "sp500_alias.csv"):
    if not Path(path).exists():
        return {}
    a = pd.read_csv(path, comment="#", dtype=str).fillna("")
    return {r["simbol_istoric"].strip().upper(): (r["simbol_actual"].strip().upper(), r["cik"].strip() or None)
            for _, r in a.iterrows()}


def download(symbol, start, end, refresh=False):
    """Prețurile Yahoo din cache-ul propriu al testului (de la `DL_START`), nu din cel comun `tools/cache/`."""
    return _download(symbol, start, end, refresh, cache_dir=PRICES_DIR)


def download_splits(symbol, refresh=False):
    """Split-urile Yahoo (Series dată → raport), cu cache CSV. Serie goală la eșec sau fără split-uri."""
    import yfinance as yf

    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"splits_{symbol.replace('^', '_')}.csv"
    if path.exists() and not refresh:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df.iloc[:, 0] if not df.empty else pd.Series(dtype=float)
    try:
        sp = yf.Ticker(symbol).splits
    except Exception as e:  # noqa: BLE001 — yfinance aruncă tipuri diferite între versiuni
        print(f"  {symbol}: eroare la split-uri ({type(e).__name__}: {e})")
        return pd.Series(dtype=float)
    sp = pd.Series(sp if sp is not None else [], dtype=float)
    if len(sp):
        idx = pd.DatetimeIndex(sp.index)
        sp.index = idx.tz_localize(None) if idx.tz is not None else idx
    sp.rename("split").to_frame().to_csv(path)
    return sp


def company_tickers(user_agent, refresh=False):
    path = SEC_DIR / "company_tickers.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text(encoding="utf-8"))
    from data import sec_edgar, sec_insiders
    data = sec_edgar.EdgarClient(user_agent).get_json(sec_insiders.COMPANY_TICKERS_URL)
    SEC_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return data


def _splits_sig(splits):
    return tuple((str(pd.Timestamp(d).date()), float(v)) for d, v in pd.Series(splits, dtype=float).items())


def _versions_path(cik, ysym):
    return SEC_DIR / "versions" / f"{cik}_{ysym}.pkl"


def versions_cached(cik, ysym, splits):
    """Versiunile sunt deja în cache pentru aceleași split-uri?"""
    path = _versions_path(cik, ysym)
    if not path.exists():
        return False
    with open(path, "rb") as fh:
        saved_sig, _ = pickle.load(fh)
    return saved_sig == _splits_sig(splits)


def _versions_worker(args):
    """Pentru ProcessPoolExecutor: citește situațiile din cache, calculează și scrie versiunile."""
    facts_path, cik, ysym, splits = args
    cached_versions(None, cik, ysym, splits, facts_path)
    return ysym


def cached_versions(bulk_cf, cik, ysym, splits, facts_path=None):
    """Versiunile incrementale ale situațiilor, cu cache pe (CIK, simbol, split-uri). `bulk_cf` = None → se citește
    din `facts_path` (cache-ul `BulkSec`) doar dacă trebuie calculate."""
    sig = _splits_sig(splits)
    path = _versions_path(cik, ysym)
    if path.exists():
        with open(path, "rb") as fh:
            saved_sig, versions = pickle.load(fh)
        if saved_sig == sig:
            return versions
    from data.sec_bulk import load_pickle
    cf = bulk_cf if bulk_cf is not None else load_pickle(facts_path)
    versions = ss.asof_versions(cf, splits=splits)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump((sig, versions), fh, protocol=pickle.HIGHEST_PROTOCOL)
    return versions


# --- Rezumat -------------------------------------------------------------------

def score_columns(df):
    cols = ["fund_score", "v5_fund_score", "v3_sector_rank"]
    cols += [c for c in df.columns if c.startswith("p_") or c.startswith("c_")]
    return [c for c in cols if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().any()]


def summarize(df):
    """Rezumatul pe universuri (B, C) × trunchiate (cu / fără) × orizonturi × scoruri. Întoarce (rezumat, cincimi
    pe ani). Benzile și cincimile: pe randamentul în exces față de mediana universului (și a sectorului)."""
    rows, years = [], []
    for univ in ("C", "B"):
        base = df[df[f"in_{univ}"]].copy()
        if base.empty:
            continue
        base["v3_sector_rank"] = bs.sector_rank_score(base, "fund_score")
        for trunc in ("cu", "fara"):
            data = base.copy()
            for h in HORIZONS:
                if trunc == "fara":
                    data.loc[data[f"trunc_{h}m"].fillna(False).astype(bool), f"fwd_{h}m"] = None
                data = bs.add_median_excess(data, f"fwd_{h}m", by=("date",), suffix="_exu")
                data = bs.add_median_excess(data, f"fwd_{h}m", by=("date", "sector"), suffix="_exs")
            for col in score_columns(data):
                for h in HORIZONS:
                    ic = bs.ic_summary(data, col, f"fwd_{h}m")
                    ic_s = bs.ic_summary(data, col, f"fwd_{h}m_exs")
                    row = {"univers": univ, "trunchiate": trunc, "scor": col, "orizont_luni": h,
                           "N": int(pd.to_numeric(data[col], errors="coerce").notna().sum()),
                           "ic_mediu": ic["ic_mediu"], "ic_t_nw": ic["t_nw"], "ic_nr_date": ic["nr_date"],
                           "ic_pct_pozitiv": ic["pct_pozitiv"],
                           "ic_sector_mediu": ic_s["ic_mediu"], "ic_sector_t_nw": ic_s["t_nw"]}
                    if col in ("fund_score", "v5_fund_score", "v3_sector_rank") and h in MAIN_HORIZONS:
                        q = bs.quintile_spread_by_year(data, col, f"fwd_{h}m_exu")
                        row["ani_sus_bate_jos"] = int(q["sus_bate_jos"].sum()) if not q.empty else 0
                        row["ani_total"] = int(len(q))
                        for _, r in q.iterrows():
                            years.append({"univers": univ, "trunchiate": trunc, "scor": col, "orizont_luni": h,
                                          "an": int(r["an"]), "spread_mediu": r["spread_mediu"],
                                          "nr_date": int(r["nr_date"])})
                        for ref, suf in (("univers", "_exu"), ("sector", "_exs")):
                            t, mono = bs.band_medians(data, col, f"fwd_{h}m{suf}")
                            for _, b in t.iterrows():
                                row[f"banda_{b['banda']}_med_{ref}"] = b["mediana_exces"]
                                row[f"banda_{b['banda']}_N_{ref}"] = int(b["N"])
                            row[f"benzi_monotone_{ref}"] = mono
                    rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(years)


def coverage(status, months, intervals, b_set):
    """Pe an: membri medii pe lună (C) și câți au CIK, situații, prețuri acceptate și scor; B la fel."""
    out = []
    by_year = {}
    for d in months:
        by_year.setdefault(d.year, []).append(d)
    for univ in ("C", "B"):
        for y, ds in sorted(by_year.items()):
            acc = {"membri": 0, "cu_cik": 0, "cu_situatii": 0, "acceptati": 0, "cu_scor": 0}
            for d in ds:
                mem = bs.members_at(intervals, d) if univ == "C" else b_set
                for s in mem:
                    st = status.get(s, {})
                    acc["membri"] += 1
                    acc["cu_cik"] += bool(st.get("cik"))
                    acc["cu_situatii"] += bool(st.get("facts"))
                    acc["acceptati"] += bool(st.get("accepted"))
                    acc["cu_scor"] += d in st.get("scored_dates", set())
            out.append({"univers": univ, "an": y, **{k: round(v / len(ds), 1) for k, v in acc.items()},
                        "pct_cu_scor": round(acc["cu_scor"] / acc["membri"] * 100, 1) if acc["membri"] else None})
    return pd.DataFrame(out)


# --- Rulare --------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="Testul istoric al scorului SUA pe date SEC (Faza 3c).")
    ap.add_argument("--faza", choices=tuple(PHASES), required=True)
    ap.add_argument("--descarca", action="store_true", help="descarcă arhivele SEC și lista simbolurilor (o dată)")
    ap.add_argument("--secrets", default=str(ROOT / ".streamlit" / "secrets.toml"), help="pentru User-Agent-ul SEC")
    ap.add_argument("--user-agent", help="User-Agent SEC (nume + e-mail), în loc de secrets.toml")
    ap.add_argument("--simboluri", nargs="*", help="doar aceste simboluri istorice (probă)")
    ap.add_argument("--out-dir", default=str(CACHE / "sec" / "rezultate"))
    ap.add_argument("--reincarca", action="store_true", help="descarcă din nou prețurile Yahoo")
    ap.add_argument("--procese", type=int, default=max(1, (os.cpu_count() or 2) - 2),
                    help="procese pentru versiunile situațiilor (implicit: nuclee − 2)")
    args = ap.parse_args(argv)

    if args.faza == "verificare":
        print("Faza „verificare” vine în PR-ul următor (3c-3), după fixarea variantelor în varianta_fixata.json.")
        return 2
    start, end = PHASES[args.faza]
    horizon_limit_phase = pd.Timestamp(end)

    from data import sec_bulk
    facts_zip, subs_zip = SEC_DIR / "companyfacts.zip", SEC_DIR / "submissions.zip"
    ua = args.user_agent
    need = args.descarca or not facts_zip.exists() or not subs_zip.exists() or not (SEC_DIR / "company_tickers.json").exists()
    if need and not ua:
        from tools.sec.run import read_secrets
        ua, _ = read_secrets(args.secrets)
    if args.descarca or not facts_zip.exists():
        print("Descarc companyfacts.zip (o singură dată, poate dura) …")
        sec_bulk.download_zip(sec_bulk.COMPANYFACTS_ZIP_URL, facts_zip, ua, refresh=args.descarca)
    if args.descarca or not subs_zip.exists():
        print("Descarc submissions.zip …")
        sec_bulk.download_zip(sec_bulk.SUBMISSIONS_ZIP_URL, subs_zip, ua, refresh=args.descarca)
    for p in (facts_zip, subs_zip):
        print(f"  {p.name}: {p.stat().st_size / 1e9:.2f} GB".replace(".", ","))
    tickers = company_tickers(ua, refresh=args.descarca)
    bulk = sec_bulk.BulkSec(facts_zip, subs_zip, cache_dir=SEC_DIR)

    intervals = read_intervals()
    alias = read_alias()
    b_set = bs.current_members(intervals)
    t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
    active = intervals[(intervals["de_la"] <= t1) & (intervals["pana_la"].isna() | (intervals["pana_la"] > t0))]
    symbols = sorted(set(active["simbol"]) | b_set)
    if args.simboluri:
        want = {s.upper() for s in args.simboluri}
        symbols = [s for s in symbols if s in want]
    dl_end = (pd.Timestamp.today().normalize() + pd.Timedelta(days=1)).date().isoformat()

    bench = download(BENCH, DL_START, dl_end, args.reincarca)
    bench_adj = None if bench.empty else bench["adj"].dropna()
    tnx = download(RF_SYMBOL, DL_START, dl_end, args.reincarca)
    tnx_close = None if tnx.empty else tnx["raw"].dropna()
    aaa = download_fred("AAA", args.reincarca)
    if bench_adj is None:
        print("Lipsește SPY: nu se pot stabili datele de sfârșit de lună.")
        return 1
    late = [n for n, ser in ((BENCH, bench_adj), (RF_SYMBOL, tnx_close))
            if ser is not None and ser.index[0] > pd.Timestamp(start) - pd.DateOffset(years=2)]
    if late or tnx_close is None:
        print(f"ATENȚIE: {', '.join(late) or RF_SYMBOL} nu acoperă 2 ani înainte de {start} → beta / WACC lipsesc la început. "
              "Rulează cu --reincarca.")
    data_end = bench_adj.index[-1]
    horizon_limit = min(horizon_limit_phase, data_end)
    months = month_end_dates(bench_adj.index, start, end)
    members = {d: bs.members_at(intervals, d) for d in months}
    print(f"Faza {args.faza}: {start} → {end}, {len(months)} luni, {len(symbols)} simboluri "
          f"(C: membrii la fiecare dată; B: {len(b_set)} membri de azi). Randamente doar cu capătul ≤ {horizon_limit.date()}.")

    rows, status, jobs = [], {}, []
    t_start = time.time()
    # 1) simbol → CIK, situații, prețuri, protecția la simboluri refolosite
    for i, sym in enumerate(symbols, 1):
        actual, cik = alias.get(sym, (sym, None))
        ysym = bs.yahoo_symbol(actual or sym)
        cik = cik or cik_for_ticker(tickers, ysym)
        st = status.setdefault(sym, {"yahoo": ysym, "cik": cik, "scored_dates": set()})
        if cik is None:
            st["reason"] = "fără CIK (simbol ieșit sau redenumit: de completat în sp500_alias.csv)"
            continue
        cf = bulk.facts(cik, ysym)
        st["facts"] = cf is not None
        if cf is None:
            st["reason"] = "CIK absent din companyfacts.zip (fără XBRL)"
            continue
        prices = download(ysym, DL_START, dl_end, args.reincarca)
        adj = None if prices.empty else prices["adj"].dropna()
        raw = None if prices.empty else prices["raw"].dropna()
        first = None if adj is None or adj.empty else adj.index[0]
        ok, why = bs.accept_mapping(intervals[intervals["simbol"] == sym], sec_bulk.annual_filed_dates(cf),
                                    first, start)
        st["accepted"] = ok
        if not ok:
            st["reason"] = why
            continue
        sub = bulk.submission(cik) or {}
        sector, industry = sic_sector(sub.get("sic"))
        st.update(sic=sub.get("sic"), sector=sector, name=sub.get("name"))
        splits = download_splits(ysym, args.reincarca)
        jobs.append((sym, ysym, cik, bulk.facts_cache_path(cik, ysym), adj, raw, sector, industry, sub, splits))
        del cf                                            # memoria: situațiile se recitesc din cache la nevoie
        if i % 50 == 0:
            print(f"[{i}/{len(symbols)}] pregătit; {time.time() - t_start:.0f} s")
    bulk.close()

    # 2) versiunile incrementale ale situațiilor (cache pe disc; în paralel la prima rulare)
    todo = [j for j in jobs if not versions_cached(j[2], j[1], j[9])]
    print(f"Versiuni de calculat: {len(todo)} din {len(jobs)} (procese: {args.procese})")
    if todo and args.procese > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.procese) as ex:
            for k, _ in enumerate(ex.map(_versions_worker, [(j[3], j[2], j[1], j[9]) for j in todo]), 1):
                if k % 50 == 0:
                    print(f"  versiuni {k}/{len(todo)}; {time.time() - t_start:.0f} s")
    else:
        for j in todo:
            _versions_worker((j[3], j[2], j[1], j[9]))

    # 3) scorul la fiecare sfârșit de lună
    for i, (sym, ysym, cik, facts_path, adj, raw, sector, industry, sub, splits) in enumerate(jobs, 1):
        st = status[sym]
        versions = cached_versions(None, cik, ysym, splits, facts_path)
        n = 0
        for d in months:
            in_c, in_b = sym in members[d], sym in b_set
            if not (in_c or in_b):
                continue
            stm = ss.version_at(versions, d)
            price = value_as_of(raw, d)
            if stm is None or price is None:
                continue
            rf_pct = value_as_of(tnx_close, d) if tnx_close is not None else None
            rf = None if rf_pct is None or not 0 < rf_pct < 25 else rf_pct / 100
            aaa_pct = value_as_of(aaa, d, publication_lag=FRED_LAG) if len(aaa) else None
            beta = beta_as_of(adj, bench_adj, d)
            inp, fy = inputs_at(d, price, stm["income"], stm["balance"], stm["cashflow"], sector, ysym, rf, beta,
                                aaa_pct, lag=NO_SCORE_LAG)
            if inp is None:
                continue
            if industry and not inp.get("financial"):           # REIT (SIC 6798): DCF nu se aplică
                inp["dcf_na"] = fund.dcf_not_applicable_reason(sector, industry, ysym)
            row = score_row(sym, d, inp, fy)
            # varianta (2): creșterea livrată pe 5 ani (în loc de 3) în DCF invers și PEG
            if inp.get("financial"):
                row["v5_fund_score"] = row["fund_score"]
            else:
                g5 = fund.statement_cagr(stm["income"], fund.REVENUE_ROWS, max_years=5, min_years=5)
                e5 = fund.statement_cagr(stm["income"], fund.EPS_ROWS, max_years=5, min_years=5)
                inp5 = dict(inp, rev_growth=(g5 or {}).get("growth"), rev_growth_years=(g5 or {}).get("years"),
                            eps_growth=(e5 or {}).get("growth"), eps_growth_years=(e5 or {}).get("years"))
                row["v5_fund_score"] = fundamental_score(inp5)["score"]
            row.update(yahoo=ysym, cik=cik, sic=sub.get("sic"), sector=sector, in_B=in_b, in_C=in_c,
                       n_conflicts=len(stm["notes"]["conflicts"]))
            for h in HORIZONS:
                r, tr = bs.forward_return_at(adj, d, h, horizon_limit, data_end)
                row[f"fwd_{h}m"], row[f"trunc_{h}m"] = r, tr
            rows.append(row)
            st["scored_dates"].add(d)
            n += 1
        if i % 50 == 0 or args.simboluri:
            print(f"[{i}/{len(jobs)}] {sym} ({ysym}, CIK {cik}): {n} luni; {time.time() - t_start:.0f} s")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.faza
    missing = pd.DataFrame([{"simbol": s, "yahoo": v.get("yahoo"), "cik": v.get("cik"), "motiv": v.get("reason")}
                            for s, v in status.items() if v.get("reason")])
    missing.to_csv(out_dir / f"fara_date_{tag}.csv", index=False)
    cov = coverage(status, months, intervals, b_set)
    cov.to_csv(out_dir / f"acoperire_{tag}.csv", index=False)
    if not rows:
        print("Nicio observație. Vezi fara_date_*.csv.")
        return 1
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"observatii_{tag}.csv", index=False)
    rez, years = summarize(df)
    rez.to_csv(out_dir / f"rezumat_{tag}.csv", index=False)
    years.to_csv(out_dir / f"cincimi_pe_ani_{tag}.csv", index=False)

    print(f"\nObservații: {len(df)} | simboluri cu scor: {df['symbol'].nunique()} | fără date: {len(missing)}")
    print("\nAcoperirea (medii pe lună):")
    print(cov.to_string(index=False))
    main_rows = rez[rez["scor"].isin(["fund_score", "v5_fund_score", "v3_sector_rank"])
                    & rez["orizont_luni"].isin(MAIN_HORIZONS)]
    cols = ["univers", "trunchiate", "scor", "orizont_luni", "N", "ic_mediu", "ic_t_nw", "ic_nr_date",
            "ani_sus_bate_jos", "ani_total", "benzi_monotone_univers", "benzi_monotone_sector"]
    print("\nScorurile (construire; criteriile §10.3 se aplică DOAR pe verificare):")
    print(main_rows[[c for c in cols if c in main_rows.columns]].to_string(index=False))
    comp = rez[(rez["orizont_luni"] == 12) & (rez["trunchiate"] == "cu") & (rez["univers"] == "C")
               & rez["scor"].str.match(r"^(p_|c_)")]
    if not comp.empty:
        print("\nIC la 12 luni pe piloni și componente (C, cu trunchiate) — baza variantei (4):")
        print(comp[["scor", "N", "ic_mediu", "ic_t_nw", "ic_sector_mediu"]].to_string(index=False))
    print(f"\nScris în {out_dir}: observatii_{tag}.csv, rezumat_{tag}.csv, cincimi_pe_ani_{tag}.csv, "
          f"acoperire_{tag}.csv, fara_date_{tag}.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
