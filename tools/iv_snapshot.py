"""Jobul zilnic pentru foaia `IV_istoric`: salvează IV-ul la bani al simbolurilor SUA din `Portofoliu` și `watchlist`.

Rulează pe serverul lui Daniel (timer systemd, luni–vineri, în timpul ședinței de la New York), cu aceeași
cheie ca aplicația (`.streamlit/secrets.toml`). Folosește exact aceleași calcule ca aplicația
(`analytics.options.choose_expiration` + `atm_iv`, expirarea 20–45 de zile), deci valorile sunt comparabile.

Exemple (din rădăcina repo-ului):
    .venv/bin/python -m tools.iv_snapshot --dry-run      # calculează și afișează, nu scrie nimic
    .venv/bin/python -m tools.iv_snapshot                # scrie în IV_istoric (doar în timpul ședinței SUA)

Reguli:
  - Scrie cel mult un rând pe (data ședinței la New York, simbol); dacă aplicația a salvat deja azi, sare.
  - În afara ședinței (luni–vineri, 9:30–16:00 New York) nu scrie nimic; --dry-run calculează oricum.
  - Un simbol fără opțiuni sau fără IV calculabil e sărit și raportat; nu se scrie niciodată 0 sau o valoare inventată.
"""
import argparse
import re
import sys
import time
import tomllib
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analytics import options as opt_calc  # noqa: E402
from data import iv_history  # noqa: E402

RO_TZ = ZoneInfo("Europe/Bucharest")
SPREADSHEET_NAME = "portofoliu_db"
SCOPES = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
US_SYMBOL = re.compile(r"^[A-Z]{1,5}(-[A-Z])?$")   # AAPL, BRK-B; fără sufix de bursă (.RO, .DE) și fără ^indici
PAUSE_S = 1.5                                       # pauză între simboluri, ca Yahoo să nu limiteze


def _column(values, name):
    """Valorile coloanei `name` (fără antet), după antetul din primul rând; [] dacă lipsește."""
    if not values:
        return []
    header = [str(h).strip().lower() for h in values[0]]
    if name.lower() not in header:
        return []
    i = header.index(name.lower())
    return [row[i] if i < len(row) else None for row in values[1:]]


def us_symbols(portfolio_values, watchlist_values):
    """Simbolurile SUA, unice și sortate, din `Portofoliu` (doar Currency USD sau gol) și `watchlist`.

    Funcție pură: primește rândurile foilor (antet + rânduri)."""
    out = set()
    syms = _column(portfolio_values, "Symbol")
    curs = _column(portfolio_values, "Currency") or [None] * len(syms)
    for sym, cur in zip(syms, curs):
        s, c = str(sym or "").strip().upper(), str(cur or "").strip().upper()
        if US_SYMBOL.match(s) and c in ("", "USD"):
            out.add(s)
    for sym in _column(watchlist_values, "Symbol"):
        s = str(sym or "").strip().upper()
        if US_SYMBOL.match(s):
            out.add(s)
    return sorted(out)


def rows_to_write(history_df, ivs, session_date):
    """Rândurile noi pentru IV_istoric: doar simbolurile cu IV valid și fără rând în ziua `session_date`.

    `ivs`: {simbol: IV fracție sau None}. Funcție pură."""
    rows = []
    for sym in sorted(ivs):
        iv = ivs[sym]
        if iv is None or not (0 < iv < iv_history.MAX_IV):
            continue
        if iv_history.should_save(history_df, sym, session_date):
            rows.append(iv_history.make_row(session_date, sym, iv))
    return rows


def fetch_atm_iv(symbol, today):
    """(IV la bani ca fracție sau None, explicație). Aceeași expirare și aceeași interpolare ca aplicația."""
    import yfinance as yf

    t = yf.Ticker(symbol)
    expirations = t.options
    if not expirations:
        return None, "fără listă de expirări în Yahoo"
    chosen = opt_calc.choose_expiration(expirations, today)
    if chosen is None:
        return None, "nicio expirare cu 20–45 de zile"
    expiration, days = chosen
    chain = t.option_chain(expiration)
    spot = t.fast_info.last_price
    iv, sides = opt_calc.atm_iv(chain.calls, chain.puts, spot)
    if iv is None:
        return None, f"IV la bani necalculabil (expirarea {expiration})"
    return iv, f"expirarea {expiration} ({days} zile), preț {float(spot):.2f}, {sides} părți"


def open_spreadsheet(secrets_path):
    import gspread
    from google.oauth2.service_account import Credentials

    with open(secrets_path, "rb") as f:
        info = tomllib.load(f)["gcp_service_account"]
    creds = Credentials.from_service_account_info(dict(info), scopes=SCOPES)
    return gspread.authorize(creds).open(SPREADSHEET_NAME)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--secrets", default=str(ROOT / ".streamlit" / "secrets.toml"))
    p.add_argument("--dry-run", action="store_true", help="calculează și afișează, fără să scrie în foaie")
    args = p.parse_args(argv)

    now = datetime.now(RO_TZ)
    session = iv_history.us_session_date(now)
    is_open = iv_history.us_session_open(now)
    print(f"{now:%Y-%m-%d %H:%M} ora României · ședința New York {session} · bursa {'DESCHISĂ' if is_open else 'ÎNCHISĂ'}")
    if not is_open and not args.dry_run:
        print("Bursa SUA e închisă: nu se scrie nimic (pentru probă: --dry-run).")
        return 0

    book = open_spreadsheet(args.secrets)
    symbols = us_symbols(book.sheet1.get_all_values(), book.worksheet("watchlist").get_all_values())
    print(f"Simboluri SUA ({len(symbols)}): {', '.join(symbols) or '-'}")

    ws = book.worksheet("IV_istoric")
    history_df, problems = iv_history.parse_iv_history(
        ws.get_all_values(value_render_option="UNFORMATTED_VALUE", date_time_render_option="FORMATTED_STRING"))
    for msg in problems:
        print(msg)

    ivs, skipped = {}, []
    today = now.date()   # ca în aplicație: data din România alege expirarea
    for i, sym in enumerate(symbols):
        if i:
            time.sleep(PAUSE_S)
        try:
            iv, why = fetch_atm_iv(sym, today)
        except Exception as e:  # yfinance ridică tipuri nedocumentate (HTTP, JSON, 429): simbolul e sărit, jobul continuă
            iv, why = None, f"eroare {type(e).__name__}: {e}"
        ivs[sym] = iv
        if iv is None:
            skipped.append(sym)
            print(f"  {sym:<6} sărit: {why}")
        else:
            print(f"  {sym:<6} IV {iv * 100:6.2f}% · {why}")

    rows = rows_to_write(history_df, ivs, session)
    already = [s for s, v in ivs.items() if v is not None and not iv_history.should_save(history_df, s, session)]
    if already:
        print(f"Deja salvate azi (de aplicație sau de o rulare anterioară): {', '.join(sorted(already))}")
    if args.dry_run:
        print(f"PROBĂ: s-ar scrie {len(rows)} rânduri; nu s-a scris nimic.")
    elif rows:
        ws.append_rows(rows, value_input_option="RAW")
        print(f"Scrise {len(rows)} rânduri în IV_istoric.")
    else:
        print("Nimic nou de scris.")
    print(f"Rezumat: {len(symbols)} simboluri, {len(rows)} noi, {len(already)} existente, {len(skipped)} sărite.")
    if symbols and len(skipped) == len(symbols):
        print("EȘEC: niciun IV calculat (probabil Yahoo indisponibil sau limitat).")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
