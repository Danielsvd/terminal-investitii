"""Reguli pure pentru pagina Watchlist (fără rețea, fără `st.*`)."""
import math

import pandas as pd

from data.helpers import positive_or_none

WATCHLIST_COLUMNS = ("Symbol", "TargetPrice", "Notes")


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
    empty = pd.DataFrame(columns=list(WATCHLIST_COLUMNS))
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

    rows = []
    for row in values[1:]:
        sym = str(cell(row, "Symbol")).strip().upper()
        if not sym:
            continue
        rows.append({"Symbol": sym, "TargetPrice": positive_or_none(cell(row, "TargetPrice")),
                     "Notes": str(cell(row, "Notes") or "").strip()})
    df = pd.DataFrame(rows, columns=list(WATCHLIST_COLUMNS)) if rows else empty
    return df, problems


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
