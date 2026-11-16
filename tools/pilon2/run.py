"""Jobul Pilon II: rapoartele noi → foile `Pilon2_detineri` și `ISIN_map`.

Pași: (1) descarcă rapoartele lipsă din arhivă (`fetch.py`); (2) le citește (`parse.py`) și le respinge pe cele
cu probleme (suma acțiunilor ≠ totalul secțiunii, data din raport ≠ data fișierului, alt cod de fond);
(3) caută simbolul Yahoo al ISIN-urilor noi și le trece în `ISIN_map`; (4) derivă numărul de acțiuni la BVB din
valoare / prețul de închidere neajustat de la data raportului; (5) adaugă doar rândurile care lipsesc.

Rulează pe server, din rădăcina repo-ului, cu mediul separat (vezi `tools/pilon2/requirements.txt`):
    ~/apps/pilon2/.venv/bin/python -m tools.pilon2.run --data-dir ~/apps/pilon2 --proba          # nu scrie
    ~/apps/pilon2/.venv/bin/python -m tools.pilon2.run --data-dir ~/apps/pilon2 --luni 30 --istoric
    ~/apps/pilon2/.venv/bin/python -m tools.pilon2.run --data-dir ~/apps/pilon2                  # lunar

Reguli: o valoare lipsă rămâne goală (niciodată 0); un raport cu probleme nu se scrie deloc (se raportează);
la BVB cantitatea se derivă doar cu preț găsit în cel mult 7 zile înainte de data raportului.
"""
import argparse
import sys
import time
import tomllib
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.pilon2 import (HOLDINGS_COLUMNS, HOLDINGS_SHEET, MAP_COLUMNS, MAP_SHEET,  # noqa: E402
                         parse_isin_map)
from tools.pilon2 import fetch, parse  # noqa: E402

SPREADSHEET_NAME = "portofoliu_db"
SCOPES = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
MAX_PRICE_LAG_DAYS = 7
SEARCH_PAUSE_S = 1.0


# --- Funcții pure ---------------------------------------------------------------------------------

def report_problems(rep, archive_date, expected_code):
    """Problemele raportului, inclusiv potrivirea cu fișierul din arhivă și cu fondul așteptat.

    Un raport fără cod de fond (NN din arhiva .zip, Aripi din 12.2025) primește codul așteptat al sursei: fișierul
    vine de pe site-ul fondului, iar data și suma acțiunilor se verifică în continuare. Un cod diferit rămâne problemă.
    """
    if rep.fund_code is None and expected_code:
        rep.fund_code = expected_code
    problems = parse.check(rep)
    if rep.report_date and rep.report_date != archive_date:
        problems.append(f"data din raport {rep.report_date} ≠ data fișierului {archive_date}")
    if expected_code and rep.fund_code and rep.fund_code != expected_code:
        problems.append(f"codul fondului {rep.fund_code} ≠ {expected_code} (alt fond?)")
    return problems


def equity_by_isin(rep):
    """Acțiunile raportului, însumate pe ISIN (un ISIN apare o dată): {ISIN: {Denumire, Valoare_RON, Pondere}}."""
    out = {}
    for r in rep.equities():
        cur = out.setdefault(r["ISIN"], {"Denumire": r["Denumire"], "Valoare_RON": None, "Pondere": None})
        if not cur["Denumire"] and r["Denumire"]:
            cur["Denumire"] = r["Denumire"]
        for k in ("Valoare_RON", "Pondere"):
            if r[k] is not None:
                cur[k] = (cur[k] or 0.0) + r[k]
    return out


def derive_quantity(value, price):
    """(cantitate rotunjită, abaterea câtului de la un întreg) sau (None, None) dacă lipsește ceva."""
    if value is None or price is None or not price > 0 or value < 0:
        return None, None
    q = value / price
    n = round(q)
    return int(n), round(q - n, 4)


def choose_symbol(quotes):
    """Simbolul Yahoo pentru un ISIN, din rezultatele căutării: întâi o cotație .RO, apoi prima acțiune."""
    quotes = [q for q in (quotes or []) if q.get("symbol")]
    for q in quotes:
        if str(q["symbol"]).upper().endswith(".RO"):
            return str(q["symbol"]).upper()
    for q in quotes:
        if str(q.get("quoteType", "")).upper() in ("EQUITY", ""):
            return str(q["symbol"]).upper()
    return ""


def map_row(isin, symbol, name):
    piata = "BVB" if symbol.endswith(".RO") else ("extern" if symbol else "")
    return [isin, symbol, name, piata, "yahoo" if symbol else "negasit"]


def price_on_or_before(close, symbol, d, max_lag_days=MAX_PRICE_LAG_DAYS):
    """Închiderea din ultima ședință ≤ d, cel mult `max_lag_days` în urmă; altfel None."""
    if close is None or symbol not in getattr(close, "columns", []):
        return None, None
    s = close[symbol].dropna()
    s = s[(s.index.date <= d) & (s.index.date >= d - timedelta(days=max_lag_days))]
    if s.empty:
        return None, None
    v = float(s.iloc[-1])
    return (v, s.index[-1].date()) if v > 0 else (None, None)


def build_rows(fund_name, rep, isin_map, close, published):
    """Rândurile pentru `Pilon2_detineri` (ordinea din HOLDINGS_COLUMNS), doar acțiuni."""
    rows = []
    d = rep.report_date
    for isin, rec in sorted(equity_by_isin(rep).items()):
        m = isin_map.get(isin, {})
        sym = m.get("Symbol", "")
        price = qty = dev = None
        if sym.endswith(".RO"):
            price, _ = price_on_or_before(close, sym, d)
            qty, dev = derive_quantity(rec["Valoare_RON"], price)
        rows.append([d.isoformat(), fund_name, isin, sym, rec["Denumire"] or m.get("Denumire", ""), parse.EQUITY,
                     _cell(rec["Valoare_RON"]), _cell(rec["Pondere"]), _cell(qty), _cell(price), _cell(dev),
                     published.isoformat() if published else ""])
    return rows


def _cell(v):
    return "" if v is None else v


def existing_keys(values):
    """Cheile (Date, Fond, ISIN) deja în foaie, din rândurile ei (antet + rânduri)."""
    if not values:
        return set()
    header = [str(h).strip() for h in values[0]]
    try:
        i_d, i_f, i_i = header.index("Date"), header.index("Fond"), header.index("ISIN")
    except ValueError:
        return set()
    return {(str(r[i_d]).strip(), str(r[i_f]).strip(), str(r[i_i]).strip().upper())
            for r in values[1:] if len(r) > max(i_d, i_f, i_i)}


# --- Rețea și foi ---------------------------------------------------------------------------------

def open_spreadsheet(secrets_path):
    import gspread
    from google.oauth2.service_account import Credentials

    with open(secrets_path, "rb") as f:
        info = tomllib.load(f)["gcp_service_account"]
    creds = Credentials.from_service_account_info(dict(info), scopes=SCOPES)
    return gspread.authorize(creds).open(SPREADSHEET_NAME)


def ensure_worksheet(sh, title, header):
    """Foaia `title` (creată cu antetul dat dacă lipsește — aprobat de Daniel, 15.11.2026)."""
    import gspread

    try:
        return sh.worksheet(title)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=title, rows=1000, cols=len(header))
        ws.append_row(header, value_input_option="RAW")
        print(f"Foaie nouă creată: {title}")
        return ws


def yahoo_symbol(isin):
    import yfinance as yf

    try:
        return choose_symbol(yf.Search(isin, max_results=8).quotes)
    except Exception as exc:  # căutarea Yahoo e neoficială: eșecul lasă ISIN-ul nerezolvat, nu oprește jobul
        print(f"  ! căutare Yahoo {isin}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return ""


def download_closes(symbols, dates):
    """Închideri neajustate (auto_adjust=False: fondurile evaluează la prețul de închidere al zilei)."""
    import yfinance as yf

    from data.helpers import close_frame

    if not symbols or not dates:
        return None
    start = min(dates) - timedelta(days=MAX_PRICE_LAG_DAYS + 3)
    end = max(dates) + timedelta(days=1)
    data = yf.download(sorted(symbols), start=start.isoformat(), end=end.isoformat(), auto_adjust=False,
                       progress=False, threads=False)
    close = close_frame(data, sorted(symbols))
    if close is not None and not close.empty and getattr(close.index, "tz", None) is not None:
        close.index = close.index.tz_localize(None)
    return close


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True, help="folderul dedicat (ex. ~/apps/pilon2)")
    p.add_argument("--secrets", default=str(ROOT / ".streamlit" / "secrets.toml"))
    p.add_argument("--luni", type=int, default=6, help="câte sfârșituri de lună, înapoi de azi")
    p.add_argument("--fond", default="", help="doar aceste chei (vital,bt,bcr,metropolitan,nn,azt,aripi)")
    p.add_argument("--istoric", action="store_true", help="completare retroactivă: Data_publicare rămâne goală")
    p.add_argument("--proba", action="store_true", help="descarcă și citește, fără să scrie în foi")
    args = p.parse_args(argv)
    data_dir = Path(args.data_dir).expanduser()
    keys = [k.strip() for k in args.fond.split(",") if k.strip()] or None
    sources = fetch.load_sources(data_dir / "surse.json")

    archive = fetch.fetch_all(data_dir, args.luni, keys, sources=sources)
    good, failed = [], 0
    for key, items in archive.items():
        src = sources[key]
        for d, path, is_new in items:
            try:
                rep = parse.parse_pdf(path)
            except Exception as exc:  # PDF stricat / necitibil: se raportează, restul continuă
                print(f"{key} {d}: PDF necitibil ({type(exc).__name__}: {exc})")
                failed += 1
                continue
            problems = report_problems(rep, d, src.get("cod"))
            if problems:
                failed += 1
                print(f"{key} {d}: RESPINS — " + "; ".join(problems))
                continue
            if not src.get("cod"):
                print(f"{key} {d}: codul fondului din raport = {rep.fund_code} (de trecut în SOURCES)")
            published = rep.published_date or (date.today() if is_new and not args.istoric else None)
            good.append((key, src["fond"], rep, published))
            print(f"{key} {d}: OK, {len(rep.equities())} rânduri de acțiuni")

    if args.proba:
        print(f"Probă: {len(good)} rapoarte bune, {failed} respinse. Nimic scris.")
        return 1 if failed else 0
    if not good:
        print(f"Niciun raport de scris ({failed} respinse).")
        return 1 if failed else 0

    sh = open_spreadsheet(args.secrets)
    ws_map = ensure_worksheet(sh, MAP_SHEET, MAP_COLUMNS)
    ws_hold = ensure_worksheet(sh, HOLDINGS_SHEET, HOLDINGS_COLUMNS)
    isin_map = parse_isin_map(ws_map.get_all_values())

    names = {}
    for _k, _f, rep, _p in good:
        for isin, rec in equity_by_isin(rep).items():
            names.setdefault(isin, rec["Denumire"])
    new_map_rows = []
    for isin in sorted(set(names) - set(isin_map)):
        sym = yahoo_symbol(isin)
        time.sleep(SEARCH_PAUSE_S)
        row = map_row(isin, sym, names[isin])
        new_map_rows.append(row)
        isin_map[isin] = dict(zip(MAP_COLUMNS[1:], row[1:]))
    if new_map_rows:
        ws_map.append_rows(new_map_rows, value_input_option="RAW")
        unresolved = [r[0] for r in new_map_rows if not r[1]]
        print(f"ISIN_map: {len(new_map_rows)} ISIN-uri noi ({len(unresolved)} fără simbol: {', '.join(unresolved[:10])})")

    bvb = {m["Symbol"] for m in isin_map.values() if m.get("Symbol", "").endswith(".RO")}
    dates = sorted({rep.report_date for _k, _f, rep, _p in good})
    close = download_closes(bvb, dates)

    have = existing_keys(ws_hold.get_all_values())
    rows = []
    for _key, fund_name, rep, published in good:
        for row in build_rows(fund_name, rep, isin_map, close, published):
            if (row[0], row[1], row[2]) not in have:
                have.add((row[0], row[1], row[2]))       # și fără dubluri în aceeași rulare
                rows.append(row)
    if rows:
        ws_hold.append_rows(rows, value_input_option="RAW")
    no_qty = sum(1 for r in rows if r[3].endswith(".RO") and r[8] == "")
    print(f"{HOLDINGS_SHEET}: {len(rows)} rânduri noi ({no_qty} acțiuni BVB fără preț → fără cantitate); "
          f"{failed} rapoarte respinse.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
