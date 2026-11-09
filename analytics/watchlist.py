"""Reguli pure pentru pagina Watchlist (fără rețea, fără `st.*`)."""
import math

import numpy as np
import pandas as pd

from data.helpers import NEAR_TARGET_PCT, positive_or_none, target_zone

WATCHLIST_COLUMNS = ("Symbol", "TargetPrice", "Notes")
OPTIONAL_COLUMNS = ("Date",)   # data la care ai pus ținta (Daniel, 09.11.2026); rândurile vechi fără ea → N/A


def _positive(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) and x > 0 else None


def last_quotes(close):
    """{simbol: {"price": float, "date": pd.Timestamp}} din închiderile `close_frame` (o coloană per simbol).

    Pentru fiecare simbol: ultima valoare strict pozitivă și data ei (fără fus orar, ora 00:00). Coloanele
    au calendare diferite (BVB, SUA, Europa), deci ultima dată se caută pe fiecare coloană separat, nu pe tabel.
    Simbolurile fără nicio valoare lipsesc din rezultat: în interfață apar „N/A”, nu un preț 0.
    """
    out = {}
    if close is None or len(close) == 0:
        return out
    for sym in close.columns:
        s = pd.to_numeric(close[sym], errors="coerce")
        s = s[s > 0].dropna()
        if s.empty:
            continue
        ts = pd.Timestamp(s.index[-1])
        if ts.tzinfo is not None:
            ts = ts.tz_localize(None)
        out[sym] = {"price": float(s.iloc[-1]), "date": ts.normalize()}
    return out


def parse_watchlist(values):
    """Foaia `watchlist` (antet + rânduri, din `get_all_values`) → (DataFrame, probleme).

    Coloanele se caută după nume în antet (ordinea din foaie nu contează). Rezultat: Symbol (text, fără spații,
    cu majuscule), TargetPrice (număr > 0 sau None: celula goală/0/text nu devine țintă 0), Notes (text).
    Rândurile fără simbol se ignoră. `values` None → foaie necitită: DataFrame gol și o problemă.
    """
    problems = []
    empty = pd.DataFrame(columns=list(WATCHLIST_COLUMNS) + list(OPTIONAL_COLUMNS))
    if values is None:
        return empty, ["Foaia watchlist nu a putut fi citită."]
    if not values:
        return empty, problems
    header = [str(h).strip() for h in values[0]]
    missing = [c for c in WATCHLIST_COLUMNS if c not in header]
    if "Symbol" in missing or "TargetPrice" in missing:
        return empty, [f"Foaia watchlist nu are coloana {', '.join(c for c in missing if c != 'Notes')} în antet (rândul 1)."]
    idx = {c: header.index(c) for c in WATCHLIST_COLUMNS if c in header}

    def cell(row, col):
        i = idx.get(col)
        return row[i] if i is not None and i < len(row) else ""

    if "Date" in header:
        idx["Date"] = header.index("Date")

    rows = []
    for row in values[1:]:
        sym = str(cell(row, "Symbol")).strip().upper()
        if not sym:
            continue
        rows.append({"Symbol": sym, "TargetPrice": positive_or_none(cell(row, "TargetPrice")),
                     "Notes": str(cell(row, "Notes") or "").strip(), "Date": parse_sheet_date(cell(row, "Date"))})
    cols = list(WATCHLIST_COLUMNS) + list(OPTIONAL_COLUMNS)
    df = pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)
    return df, problems


def parse_sheet_date(value):
    """Data țintei din foaie: „2026-11-09” (scrisă de aplicație) sau „09.11.2026” (scrisă de mână, format românesc)
    → pd.Timestamp; goală sau de necitit → None (afișată „N/A”, nu data de azi)."""
    text = str(value or "").strip()
    if not text:
        return None
    ts = pd.to_datetime(text, dayfirst=not text[:4].isdigit(), errors="coerce")
    return None if pd.isna(ts) else pd.Timestamp(ts).normalize()


def sheet_row(header, symbol, target, note, date_text):
    """Rândul de adăugat, în ordinea coloanelor din antet (nu pe poziție fixă): Symbol, TargetPrice, Notes și,
    dacă antetul are coloana Date, data. Coloanele necunoscute rămân goale. Fără Symbol/TargetPrice în antet → None."""
    header = [str(h).strip() for h in (header or [])]
    if "Symbol" not in header or "TargetPrice" not in header:
        return None
    by_name = {"Symbol": symbol, "TargetPrice": float(target), "Notes": note or "", "Date": date_text}
    return [by_name.get(h, "") for h in header]


def normalize_symbol(symbol):
    """Simbolul așa cum se păstrează în foaie: fără spații, cu majuscule; None/gol → ""."""
    return str(symbol or "").strip().upper()


def symbol_rows(values, symbol):
    """Numerele de rând din foaie (1 = antetul) care au `symbol` în coloana Symbol, de jos în sus.

    Se caută doar în coloana Symbol (găsită după antet), nu în toată foaia: o notă care conține simbolul
    nu duce la ștergerea altui rând. Comparația ignoră spațiile și literele mici. De jos în sus, ca la
    ștergerea mai multor rânduri numerele celor rămase să nu se schimbe. Fără coloana Symbol → [].
    """
    if not values:
        return []
    header = [str(h).strip() for h in values[0]]
    if "Symbol" not in header:
        return []
    i = header.index("Symbol")
    target = normalize_symbol(symbol)
    if not target:
        return []
    rows = [n for n, row in enumerate(values[1:], start=2) if i < len(row) and normalize_symbol(row[i]) == target]
    return sorted(rows, reverse=True)


def symbol_candidates(symbol):
    """Simbolurile încercate la adăugare, în ordine, ca în pagina de Analiză (`get_stock_data`):
    întâi cel scris, apoi cu `.RO` (BVB) dacă nu îl are deja. „TLV” → ["TLV", "TLV.RO"]; gol → []."""
    s = normalize_symbol(symbol)
    if not s:
        return []
    return [s] if s.endswith(".RO") else [s, s + ".RO"]


# Statusul rândului: o alertă de preț față de ținta ta, nu o recomandare (ținta e a ta, nu un calcul al aplicației)
STATUS_LABELS = {
    "sub": "✅ La/sub țintă",
    "aproape": f"🟡 Aproape (<{NEAR_TARGET_PCT:.0f}%)",
    "peste": "⏳ Peste țintă",
}
STALE_BDAYS = 3   # ultimul preț mai vechi de 3 zile lucrătoare primește ⚠️ (titlu netranzacționat, simbol suspendat)


def status_label(price, target):
    """Eticheta din coloana Status: zona din `target_zone` sau ce lipsește („Fără preț” / „Fără țintă”)."""
    zone = target_zone(price, target)
    if zone is not None:
        return STATUS_LABELS[zone]
    return "Fără preț" if positive_or_none(price) is None else "Fără țintă"


def price_date_text(date, today, partial=False):
    """Textul coloanei „Data prețului”: „06.11.2026”, plus „⏳ ședință în curs” când prețul e din bara de azi a unei
    burse deschise (preț cu întârziere, se mai schimbă) și „⚠️ veche (N zile lucrătoare)” peste STALE_BDAYS.
    Zilele lucrătoare nu țin cont de sărbători: o zi liberă a bursei poate adăuga 1. Lipsă → „N/A”."""
    if date is None or pd.isna(date):
        return "N/A"
    d = pd.Timestamp(date).date()
    text = d.strftime("%d.%m.%Y")
    if partial:
        text += " ⏳ ședință în curs"
    age = int(np.busday_count(d, pd.Timestamp(today).date()))
    if age > STALE_BDAYS:
        text += f" ⚠️ veche ({age} zile lucrătoare)"
    return text


def needs_yahoo_currency(symbol):
    """True când moneda nu se poate deduce sigur din sufix: orice bursă în afara SUA (fără sufix → USD) și a BVB
    (.RO → RON). La Londra, Amsterdam, Zürich etc. același emitent are linii în monede diferite (un ETF pe aur
    poate cota în USD sau EUR la Londra, nu în pence), deci moneda se cere de la Yahoo."""
    s = normalize_symbol(symbol)
    return "." in s and not s.endswith(".RO")


def currency_label(yahoo_currency, symbol):
    """Eticheta din coloana Monedă: moneda raportată de Yahoo („GBp” = pence, „GBP”, „USD”…) sau, fără ea,
    cea dedusă din sufix (`briefing.quote_currency`), marcată „(dedusă)” dacă trebuia cerută de la Yahoo.
    Sufix necunoscut și fără Yahoo → „N/A”."""
    from analytics.briefing import quote_currency
    if yahoo_currency:
        return str(yahoo_currency).strip()
    guess = quote_currency(symbol)
    if guess is None:
        return "N/A"
    return f"{guess} (dedusă)" if needs_yahoo_currency(symbol) else guess
