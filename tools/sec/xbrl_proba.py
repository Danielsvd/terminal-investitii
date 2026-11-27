"""Proba Fazei 3a: situațiile XBRL de la SEC pe simbolurile urmărite, comparate cu Yahoo. NU scrie nimic nicăieri.

Rulare pe server (din clona aplicației, cu mediul aplicației, care are yfinance și gspread):
    cd ~/apps/terminal-investitii && .venv/bin/python -m tools.sec.xbrl_proba
    .venv/bin/python -m tools.sec.xbrl_proba --simboluri AAPL NVDA JPM TSM --detaliu AAPL NVDA

Fără `--simboluri`: aceleași simboluri ca jobul SEC (acțiunile americane deținute + watchlist, din `portofoliu_db`).

Ce afișează, pe simbol:
  * tipul emitentului (10-K / 20-F / fără XBRL), anii fiscali acoperiți, ultimul raport;
  * cifra de afaceri, profitul net, EPS (calculat, ajustat cu split-urile), FCF pe ultimul an;
  * controale: etichete în conflict, EPS calculat vs. raportat (> 2%), TTM vs. anual, diferențe față de Yahoo (> 2%);
  * CAGR pe 5 și 10 ani (cifra de afaceri, EPS) — orientativ, de recalculat manual pe câteva simboluri.
La final: rezumatul acoperirii și al diferențelor. `--detaliu`: tabelul anual complet pentru simbolurile date.
"""
import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analytics import xbrl_facts as xf  # noqa: E402
from analytics.growth import cagr  # noqa: E402
from data import sec_edgar, sec_xbrl  # noqa: E402

YAHOO_ROWS = {
    "revenue": ("financials", ("Total Revenue", "Operating Revenue")),
    "net_income": ("financials", ("Net Income", "Net Income Common Stockholders")),
    "cfo": ("cashflow", ("Operating Cash Flow", "Cash Flow From Continuing Operating Activities")),
    "capex": ("cashflow", ("Capital Expenditure",)),
}
DIFF_TOL = 0.02
EPS_TOL = 0.02
MATCH_DAYS = 40


def _f(x, unit=1e9, nd=2):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "N/A"
    return f"{x / unit:,.{nd}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _pct(x):
    return "N/A" if x is None else f"{x * 100:+.1f}%".replace(".", ",")


def yahoo_annual(symbol):
    """{indicator: {data_încheiere: valoare}} din Yahoo, plus split-urile. ({}, None) la eșec."""
    import yfinance as yf
    t = yf.Ticker(symbol)
    out = {}
    for ind, (attr, rows) in YAHOO_ROWS.items():
        try:
            df = getattr(t, attr)
        except Exception as exc:  # Yahoo întoarce erori variate; proba nu se oprește
            print(f"    Yahoo {symbol} {attr}: {type(exc).__name__}")
            continue
        if df is None or df.empty:
            continue
        for r in rows:
            if r in df.index:
                out[ind] = {c.date(): float(v) for c, v in df.loc[r].items() if v == v}
                break
    try:
        splits = t.splits
    except Exception:
        splits = None
    return out, splits


def compare(table, yahoo):
    """[(indicator, an, sec, yahoo, diferență relativă)] cu diferența > DIFF_TOL, pe anii comuni (±40 de zile)."""
    diffs, compared = [], 0
    for ind, ys in yahoo.items():
        for yend, yval in ys.items():
            match = [e for e in table.index if abs((e - yend).days) <= MATCH_DAYS]
            if not match:
                continue
            sval = table.loc[match[0], ind]
            if sval != sval:
                continue
            yv = abs(yval) if ind == "capex" else yval
            compared += 1
            den = max(abs(sval), abs(yv))
            if den and abs(sval - yv) / den > DIFF_TOL:
                diffs.append((ind, match[0].year, sval, yv, (sval - yv) / abs(yv) if yv else None))
    return diffs, compared


def growth(table, col, years):
    s = table[col].dropna()
    if s.empty:
        return None
    end = s.index[-1]
    old = [e for e in s.index if abs((end - e).days - years * 365.25) <= 45]
    return cagr(s[end], s[old[0]], years) if old else None


def probe_symbol(client, symbol, use_yahoo, detail):
    cik, cf = sec_xbrl.load_for_symbol(client, symbol)
    if cik is None:
        return {"symbol": symbol, "kind": "fără CIK"}
    kind = xf.filer_kind(cf)
    if kind != "10-K":
        return {"symbol": symbol, "kind": kind or "fără XBRL"}
    splits = None
    yahoo = {}
    if use_yahoo:
        yahoo, splits = yahoo_annual(symbol)
    t = xf.annual_table(cf, splits=splits)
    if t.empty:
        return {"symbol": symbol, "kind": "10-K, fără ani"}
    last = t.iloc[-1]
    conflicts = sum(len(c) for c in t["conflicts"])
    eps_bad = int((t["eps_check"] > EPS_TOL).sum())
    ttm_rev = xf.ttm(cf, "revenue")
    diffs, compared = compare(t, yahoo) if yahoo else ([], 0)
    res = {
        "symbol": symbol, "kind": "10-K", "cik": cik, "years": len(t),
        "first": t.index[0].year, "last": t.index[-1], "filed": last["filed"],
        "revenue": last["revenue"], "net_income": last["net_income"], "eps": last["eps"], "fcf": last["fcf"],
        "conflicts": conflicts, "eps_bad": eps_bad,
        "ttm_check": None if ttm_rev is None else ttm_rev["check"],
        "diffs": diffs, "compared": compared,
        "rev5": growth(t, "revenue", 5), "rev10": growth(t, "revenue", 10),
        "eps5": growth(t, "eps", 5), "eps10": growth(t, "eps", 10),
    }
    if detail:
        cols = ["revenue", "net_income", "eps", "eps_reported", "fcf", "dps", "bvps", "shares_diluted", "filed"]
        print(f"\n--- {symbol}: tabel anual (mld. USD; EPS/DPS/BVPS în USD; acțiuni în mil.) ---")
        for e, r in t.iterrows():
            print(f"  {e}  venit {_f(r['revenue'])}  profit {_f(r['net_income'])}  EPS {_f(r['eps'], 1)} "
                  f"(raportat {_f(r['eps_reported'], 1)})  FCF {_f(r['fcf'])}  DPS {_f(r['dps'], 1)}  "
                  f"BVPS {_f(r['bvps'], 1)}  acț. {_f(r['shares_diluted'], 1e6, 0)}  depus {r['filed']}"
                  + (f"  CONFLICT: {', '.join(r['conflicts'])}" if r["conflicts"] else ""))
    return res


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--simboluri", nargs="*", help="simbolurile de verificat (implicit: ca jobul SEC)")
    p.add_argument("--secrets", default=str(ROOT / ".streamlit" / "secrets.toml"))
    p.add_argument("--fara-yahoo", action="store_true", help="fără comparația cu Yahoo")
    p.add_argument("--detaliu", nargs="*", default=[], help="tabelul anual complet pentru aceste simboluri")
    args = p.parse_args(argv)

    from tools.sec.run import held_us_symbols, read_secrets, watchlist_symbols
    from analytics import sec_alerts
    ua, _ = read_secrets(args.secrets)
    if args.simboluri:
        symbols = [s.strip().upper() for s in args.simboluri]
    else:
        from tools.pilon2.run import open_spreadsheet
        sh = open_spreadsheet(args.secrets)
        symbols = sec_alerts.watched_symbols(held_us_symbols(sh.worksheet("Tranzactii").get_all_values()),
                                             watchlist_symbols(sh.worksheet("watchlist").get_all_values(
                                                 value_render_option="UNFORMATTED_VALUE")))
    detail = {s.upper() for s in args.detaliu}
    print(f"Simboluri ({len(symbols)}): {', '.join(symbols)}")
    client = sec_edgar.EdgarClient(ua)
    results = []
    for sym in symbols:
        try:
            r = probe_symbol(client, sym, not args.fara_yahoo, sym in detail)
        except sec_edgar.EdgarError as exc:
            r = {"symbol": sym, "kind": f"EROARE SEC: {exc}"}
        results.append(r)
        if r["kind"] != "10-K":
            print(f"{sym}: {r['kind']}")
            continue
        flags = []
        if r["conflicts"]:
            flags.append(f"{r['conflicts']} conflicte de etichete")
        if r["eps_bad"]:
            flags.append(f"EPS calculat ≠ raportat în {r['eps_bad']} ani")
        if r["ttm_check"] is not None and r["ttm_check"] > xf.TTM_CHECK_TOL:
            flags.append(f"TTM ≠ anual ({_pct(r['ttm_check'])})")
        if r["diffs"]:
            flags.append(f"{len(r['diffs'])} diferențe față de Yahoo din {r['compared']}")
        print(f"{sym}: {r['years']} ani ({r['first']}–{r['last'].year}), ultimul an încheiat {r['last']}, depus {r['filed']} | "
              f"venit {_f(r['revenue'])} mld. $, profit {_f(r['net_income'])} mld. $, EPS {_f(r['eps'], 1)} $, "
              f"FCF {_f(r['fcf'])} mld. $ | CAGR venit 5/10 ani {_pct(r['rev5'])} / {_pct(r['rev10'])}, "
              f"EPS {_pct(r['eps5'])} / {_pct(r['eps10'])} | " + ("; ".join(flags) if flags else "fără probleme"))
        for ind, yr, s, y, d in r["diffs"]:
            print(f"    {ind} {yr}: SEC {_f(s)} vs Yahoo {_f(y)} mld. ({_pct(d)})")

    ok = [r for r in results if r["kind"] == "10-K"]
    print("\n--- Rezumat ---")
    print(f"Emitenți 10-K cu date: {len(ok)} din {len(results)}")
    for kind in sorted({r["kind"] for r in results if r["kind"] != "10-K"}):
        print(f"  {kind}: {', '.join(r['symbol'] for r in results if r['kind'] == kind)}")
    if ok:
        print(f"Ani fiscali: minim {min(r['years'] for r in ok)}, median "
              f"{sorted(r['years'] for r in ok)[len(ok) // 2]}, maxim {max(r['years'] for r in ok)}")
        comp = sum(r["compared"] for r in ok)
        nd = sum(len(r["diffs"]) for r in ok)
        if comp:
            print(f"Comparații cu Yahoo: {comp}, diferențe > 2%: {nd} ({nd / comp * 100:.1f}%)".replace(".", ","))
        print(f"Simboluri cu conflicte de etichete: {sum(1 for r in ok if r['conflicts'])}; "
              f"cu EPS calculat ≠ raportat: {sum(1 for r in ok if r['eps_bad'])}")
    print("Nimic scris.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
