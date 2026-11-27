"""Jobul SEC (Faza 2c): Form 4 + 13F pe acțiunile americane deținute și din watchlist → jurnal + alerte Telegram.

Pași:
  (1) simbolurile: acțiunile americane deținute acum (foaia `Tranzactii`) + `watchlist`;
  (2) Form 4 depuse în ultimele `--zile` zile, doar rapoartele care nu sunt deja în `SEC_insideri_jurnal`
      (deduplicare pe accession) → rânduri noi în jurnal;
  (3) alertă: cumpărare pe piață (cod P) ≥ 50.000 $ pe raport și insider;
  (4) 13F: ultimele două trimestre ale celor 14 administratori concentrați; alertă când unul deschide sau închide
      poziția într-un simbol urmărit (o singură dată pe fond, trimestru, emitent și acțiune — `<state-dir>/state.json`);
  (5) un singur mesaj Telegram pe rulare, doar dacă are ceva de spus.
Prima rulare (fără `state.json`) doar înregistrează: jurnalul se umple, alertele pornesc de la rularea următoare.

Rulează pe server, din rădăcina repo-ului, cu mediul jobului Pilon II (aceleași pachete):
    ~/apps/pilon2/.venv/bin/python -m tools.sec.run --state-dir ~/apps/sec --proba        # nu scrie, nu trimite
    ~/apps/pilon2/.venv/bin/python -m tools.sec.run --state-dir ~/apps/sec               # zilnic (timer)

Secretele (`sec_user_agent`, `openfigi_api_key`, `gcp_service_account`) vin din secrets.toml al aplicației; botul
Telegram din `~/apps/pilon2/telegram.json` (același chat ca Pilon II — decizia lui Daniel, 27.11.2026).
"""
import argparse
import json
import sys
import tomllib
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analytics import insiders, sec_13f, sec_alerts, sec_view  # noqa: E402
from analytics import ledger as ledger_calc  # noqa: E402
from analytics import watchlist as watchlist_calc  # noqa: E402
from data import cusip_map, sec_edgar, sec_funds, sec_insiders  # noqa: E402
from tools.pilon2.run import ensure_worksheet, load_telegram, open_spreadsheet, send_telegram  # noqa: E402

STATE_FILE = "state.json"
DEFAULT_DAYS = 10            # acoperă weekendul și o zi-două în care serverul a fost oprit


# --- Funcții pure (testate) -----------------------------------------------------------------------

def held_us_symbols(ledger_values):
    """Acțiunile americane deținute acum, din `Tranzactii` (aceeași regulă ca aplicația: cantitate > 0, USD)."""
    if not ledger_values:
        return []
    df, _ = ledger_calc.parse_ledger(ledger_values)
    if df.empty:
        return []
    pos, _ = ledger_calc.build_positions(df)
    held = pos[(pos["Quantity"] > ledger_calc.QTY_EPS) & pos["Currency"].astype(str).str.upper().eq("USD")]
    return [str(s).upper() for s in held["Symbol"]]


def watchlist_symbols(values):
    df, _ = watchlist_calc.parse_watchlist(values)
    return list(df["Symbol"]) if not df.empty else []


def load_state(path):
    """{"seen_13f": [...]} sau None dacă fișierul lipsește (→ rulare inițială)."""
    p = Path(path)
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"{p}: stare ilizibilă ({type(exc).__name__}); nu o suprascriu") from exc
    return {"seen_13f": list(data.get("seen_13f") or [])}


def save_state(path, seen_13f):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps({"seen_13f": sorted(set(seen_13f)), "actualizat": date.today().isoformat()},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(p)


# --- Rețea ----------------------------------------------------------------------------------------

def read_secrets(path):
    with open(path, "rb") as f:
        s = tomllib.load(f)
    ua = str(s.get("sec_user_agent") or "").strip()
    if "@" not in ua:
        raise RuntimeError("secrets.toml: lipsește sec_user_agent (cu e-mail)")
    return ua, (str(s.get("openfigi_api_key") or "").strip() or None)


def form4_step(client, symbols, known, days, today):
    """(rânduri noi pentru jurnal, alerte P, probleme, note). Notele (simbol fără CIK, ex. un ETF, sau listă
    trunchiată) doar se tipăresc; problemele (cereri sau rapoarte eșuate) apar și în mesajul Telegram."""
    tickers = client.get_json(sec_insiders.COMPANY_TICKERS_URL)
    rows, alerts, problems, notes = [], [], [], []
    for sym in symbols:
        cik = insiders.cik_for_ticker(tickers, sym)
        if cik is None:
            notes.append(f"{sym}: nu apare în lista SEC de simboluri (fără Form 4)")
            continue
        try:
            docs, probs, truncated = sec_insiders.load_form4(client, cik, days=days, today=today, skip=known)
        except sec_edgar.EdgarError as exc:
            problems.append(f"{sym}: Form 4 {exc}")
            continue
        problems.extend(f"{sym}: {p}" for p in probs)
        if truncated:
            notes.append(f"{sym}: doar cele mai noi {sec_insiders.MAX_FILINGS} rapoarte")
        tx = insiders.transactions_frame(docs)
        rows += sec_alerts.journal_rows(tx, sym, cik)
        alerts += sec_alerts.p_buy_alerts(tx, sym)
        print(f"{sym}: {len(docs)} rapoarte Form 4 noi, {len(tx)} tranzacții")
    return rows, alerts, problems, notes


def fund_step(client, symbols, cusip_values, figi_key):
    """(evenimente 13F: intrări/ieșiri ale concentraților pe simbolurile urmărite, probleme)."""
    mg = sec_funds.managers_frame()
    conc = mg[mg["Clasa"] == sec_funds.CONCENTRAT]
    books, moves_by_cik, period_of, problems = {}, {}, {}, []
    for cik in conc["CIK"]:
        try:
            _name, quarters, probs = sec_funds.load_manager(client, cik, 2)
        except sec_edgar.EdgarError as exc:
            problems.append(f"13F {cik}: {exc}")
            continue
        problems.extend(f"13F {cik}: {p}" for p in probs)
        if len(quarters) < 2:
            continue
        curr = sec_13f.equity_book(quarters[0]["info"], 1.0)
        prev = sec_13f.equity_book(quarters[1]["info"], 1.0)
        if curr.empty or prev.empty:
            continue
        books[cik] = curr
        moves_by_cik[cik] = sec_13f.quarter_moves(prev, curr)       # intrare/ieșire: nu depind de split-uri
        period_of[cik] = quarters[0]["period"]
    holdings = sec_view.holdings_panel(books, conc)
    moves = sec_view.moves_panel(moves_by_cik)
    cmap = cusip_map.parse_map(cusip_values)
    im = sec_view.issuer_moves(moves)
    keys_needed = set(im.loc[im["Actiune"].isin([sec_13f.ACT_NEW, sec_13f.ACT_CLOSED]), "CUSIP6"]) if not im.empty else set()
    missing = sorted({c for c in moves["CUSIP"] if c not in cmap and moves.loc[moves["CUSIP"] == c, "CUSIP6"].iloc[0]
                      in keys_needed}) if not moves.empty else []
    if missing:
        found, failed = cusip_map.openfigi_lookup(missing, api_key=figi_key)
        cmap.update({c: sym for c, (sym, _n) in found.items()})
        if failed:
            print(f"13F: {len(failed)} coduri CUSIP netraduse (OpenFIGI): {', '.join(failed[:10])}")
    fund_of = dict(zip(conc["CIK"].astype(str), conc["Fond"]))
    events = sec_alerts.fund_move_events(symbols, cmap, holdings, moves, fund_of, period_of)
    print(f"13F: {len(books)} administratori concentrați cu două trimestre; {len(events)} intrări/ieșiri pe simboluri urmărite")
    return events, problems


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--state-dir", required=True, help="folderul jobului (ex. ~/apps/sec)")
    p.add_argument("--secrets", default=str(ROOT / ".streamlit" / "secrets.toml"))
    p.add_argument("--telegram", default=str(Path.home() / "apps" / "pilon2" / "telegram.json"))
    p.add_argument("--zile", type=int, default=DEFAULT_DAYS, help="Form 4 depuse în ultimele N zile")
    p.add_argument("--fara-13f", action="store_true", help="doar Form 4")
    p.add_argument("--proba", action="store_true", help="citește tot, nu scrie în foaie/stare și nu trimite")
    args = p.parse_args(argv)
    tg_path = Path(args.telegram).expanduser()
    if tg_path.name != "telegram.json":
        p.error("--telegram trebuie să fie un fișier numit telegram.json")
    tg = None if args.proba else load_telegram(tg_path.parent)
    try:
        return _run(args, tg)
    except Exception as exc:
        send_telegram(tg, f"SEC — rulare {date.today():%d.%m.%Y}: EROARE {type(exc).__name__}: {str(exc)[:300]}\n"
                          "Detalii: journalctl -u sec-snapshot -n 80 --no-pager")
        raise


def _run(args, tg):
    today = date.today()
    state_path = Path(args.state_dir).expanduser() / STATE_FILE
    state = load_state(state_path)
    seed = state is None
    ua, figi_key = read_secrets(args.secrets)

    sh = open_spreadsheet(args.secrets)
    symbols = sec_alerts.watched_symbols(held_us_symbols(sh.worksheet("Tranzactii").get_all_values()),
                                         watchlist_symbols(sh.worksheet("watchlist").get_all_values(
                                             value_render_option="UNFORMATTED_VALUE")))
    print(f"Simboluri SUA urmărite ({len(symbols)}): {', '.join(symbols)}")
    if args.proba:
        try:
            journal_values = sh.worksheet(sec_alerts.JOURNAL_SHEET).get_all_values()
        except Exception:  # foaia nu există încă: la probă nu se creează
            journal_values = []
        ws_j = None
    else:
        ws_j = ensure_worksheet(sh, sec_alerts.JOURNAL_SHEET, sec_alerts.JOURNAL_COLUMNS)
        journal_values = ws_j.get_all_values()
    known = sec_alerts.known_accessions(journal_values)

    client = sec_edgar.EdgarClient(ua)
    rows, p_alerts, problems, notes = form4_step(client, symbols, known, args.zile, today)
    for n in notes:
        print(f"  notă: {n}")

    events = []
    if not args.fara_13f:
        try:
            cusip_values = sh.worksheet(cusip_map.MAP_SHEET).get_all_values()
        except Exception:  # foaia CUSIP_map lipsește: totul trece prin OpenFIGI
            cusip_values = []
        events, fprob = fund_step(client, symbols, cusip_values, figi_key)
        problems += fprob
    seen = set(state["seen_13f"]) if state else set()
    fresh = sec_alerts.new_events(events, seen)

    if args.proba:
        print(f"Probă: {len(rows)} rânduri noi de jurnal, {len(p_alerts)} alerte P, {len(fresh)} intrări/ieșiri 13F noi, "
              f"{len(problems)} probleme. Nimic scris.")
        msg = sec_alerts.alert_message(today, p_alerts, fresh, len(symbols), len(rows), problems, seed=seed)
        print("--- mesajul care ar fi plecat ---\n" + (msg or "(niciun mesaj)"))
        return 0

    if rows:
        ws_j.append_rows(rows, value_input_option="RAW")
    save_state(state_path, seen | {e["key"] for e in events})
    print(f"{sec_alerts.JOURNAL_SHEET}: {len(rows)} rânduri noi; alerte P: {len(p_alerts)}; 13F noi: {len(fresh)}; "
          f"probleme: {len(problems)}" + (" (rulare inițială: fără alerte)" if seed else ""))
    if seed:
        msg = sec_alerts.alert_message(today, [], [], len(symbols), len(known) + len(rows), problems, seed=True)
    else:
        msg = sec_alerts.alert_message(today, p_alerts, fresh, len(symbols), len(rows), problems)
    if msg:
        send_telegram(tg, msg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
