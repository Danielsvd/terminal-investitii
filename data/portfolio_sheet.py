"""Foaia `Portofoliu` din `portofoliu_db`: sursa pozițiilor deschise (acțiuni și ETF-uri).

Daniel o actualizează manual după fiecare tranzacție: Symbol, Date, Quantity, AvgPrice, Currency.
Registrul `Tranzactii` rămâne sursa pentru ce foaia nu are: realizat, dividende, impozite,
comisioane, numerar, XIRR, TWR și titlurile de stat (care nu se trec în `Portofoliu`).

Funcții pure: fără Streamlit, fără rețea. O celulă lipsă sau invalidă nu devine 0.
"""
import math
import re
import unicodedata

import numpy as np
import pandas as pd

from data.helpers import smart_to_float

PORTFOLIO_COLUMNS = ["Symbol", "Date", "Quantity", "AvgPrice", "Currency"]
QTY_TOL = 1e-6


def _norm(text):
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", "", text).lower()


def cell_number(value):
    """Număr dintr-o celulă: valorile numerice (citite cu UNFORMATTED_VALUE) trec neschimbate;
    textul trece prin smart_to_float. Gol, eroare de foaie sau text fără cifre → None."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        v = float(value)
        return v if math.isfinite(v) else None
    s = str(value).strip()
    if not s or s.startswith("#") or not re.search(r"\d", s):
        return None
    return smart_to_float(s)


def parse_portfolio_values(values):
    """Rândurile foii `Portofoliu` (antet + rânduri, ca din get_all_values) → (DataFrame, probleme).

    Coloane: Symbol, Date, Quantity, AvgPrice, Currency. Simbolurile și monedele sunt cu majuscule.
    Rândurile fără simbol sunt ignorate. Cantitate lipsă sau ≤ 0 → rândul e sărit și semnalat.
    Preț mediu lipsă → AvgPrice NaN (costul necunoscut, nu 0), semnalat.
    Un simbol trecut de mai multe ori e adunat: cantitățile se însumează, prețul mediu e ponderat.
    """
    problems = []
    empty = pd.DataFrame(columns=PORTFOLIO_COLUMNS)
    if not values:
        return empty, ["Foaia Portofoliu e goală."]
    header = [_norm(h) for h in values[0]]
    idx = {}
    for col in PORTFOLIO_COLUMNS:
        if _norm(col) in header:
            idx[col] = header.index(_norm(col))
    missing = [c for c in ("Symbol", "Quantity", "AvgPrice", "Currency") if c not in idx]
    if missing:
        return empty, [f"Foaia Portofoliu nu are coloanele: {', '.join(missing)}."]

    def cell(row, col):
        i = idx.get(col)
        return row[i] if i is not None and i < len(row) else None

    rows = []
    for n, row in enumerate(values[1:], start=2):
        sym = str(cell(row, "Symbol") or "").strip().upper()
        if not sym:
            continue
        qty = cell_number(cell(row, "Quantity"))
        avg = cell_number(cell(row, "AvgPrice"))
        cur = str(cell(row, "Currency") or "").strip().upper()
        if qty is None or qty <= 0:
            problems.append(f"Portofoliu, rândul {n} ({sym}): cantitate lipsă sau ≤ 0; rândul nu intră în calcul.")
            continue
        if not cur:
            problems.append(f"Portofoliu, rândul {n} ({sym}): moneda lipsește; rândul nu intră în calcul.")
            continue
        if avg is None or avg <= 0:
            problems.append(f"Portofoliu, rândul {n} ({sym}): preț mediu lipsă; costul și nerealizatul rămân N/A.")
            avg = np.nan
        date = cell(row, "Date")
        rows.append({"Symbol": sym, "Date": "" if date is None else str(date).strip(),
                     "Quantity": float(qty), "AvgPrice": float(avg), "Currency": cur})
    if not rows:
        return empty, problems or ["Foaia Portofoliu nu are poziții."]

    merged = {}
    for r in rows:
        m = merged.get(r["Symbol"])
        if m is None:
            merged[r["Symbol"]] = dict(r, _cost=r["Quantity"] * r["AvgPrice"], _n=1)
            continue
        m["_n"] += 1
        m["Quantity"] += r["Quantity"]
        m["_cost"] += r["Quantity"] * r["AvgPrice"]    # NaN dacă oricare preț lipsește
        m["Date"] = r["Date"] or m["Date"]
    dups = sorted(s for s, m in merged.items() if m["_n"] > 1)
    if dups:
        problems.append(f"Simboluri trecute de mai multe ori în Portofoliu (adunate): {', '.join(dups)}.")
    df = pd.DataFrame([{**{k: v for k, v in m.items() if not k.startswith("_")},
                        "AvgPrice": m["_cost"] / m["Quantity"]} for m in merged.values()])
    return df[PORTFOLIO_COLUMNS].reset_index(drop=True), problems


def infer_class(symbol, currency, known=None):
    """Clasa unui simbol: din registru dacă e cunoscut, altfel dedusă (.RO → Actiuni RO,
    USD → Actiuni SUA, restul → Actiuni UE). Întoarce (clasă, dedusă: bool)."""
    if known and symbol in known and known[symbol]:
        return known[symbol], False
    if symbol.endswith(".RO"):
        return "Actiuni RO", True
    if currency == "USD":
        return "Actiuni SUA", True
    return "Actiuni UE", True


def merge_positions(sheet, ledger_positions, keep_from_ledger=("Obligatiuni",)):
    """Pozițiile deschise după foaia `Portofoliu`, cu istoricul din registru.

    - Cantitatea și costul (cantitate × preț mediu) vin din foaie pentru fiecare simbol din foaie.
    - Simbolurile din registru care nu sunt în foaie au cantitate 0 (închise), cu excepția
      claselor din `keep_from_ledger` (titlurile de stat nu se trec în foaie): acelea rămân din registru.
    - Realizatul, dividendele, impozitele și comisioanele rămân din registru, pe simbol.
    Întoarce (poziții în formatul `ledger.build_positions`, diferențe, simboluri cu clasă dedusă).
    `diferențe`: listă de (simbol, cantitate în foaie, cantitate în registru), pentru afișare.
    """
    cols = ["Symbol", "Class", "Currency", "Quantity", "CostBasis", "Realized", "Income", "Taxes", "Fees", "Bought", "Sold"]
    base = (ledger_positions.copy() if ledger_positions is not None and len(ledger_positions)
            else pd.DataFrame(columns=cols + ["AvgCost"]))
    base = base.reindex(columns=cols)
    ledger_qty = dict(zip(base["Symbol"], base["Quantity"].astype(float)))
    known = dict(zip(base["Symbol"], base["Class"]))
    from_sheet = set(sheet["Symbol"]) if sheet is not None and len(sheet) else set()

    keep = base["Class"].isin(keep_from_ledger) & ~base["Symbol"].isin(from_sheet)
    base.loc[~keep, ["Quantity", "CostBasis"]] = 0.0

    inferred, new_rows = [], []
    for rec in (sheet.itertuples(index=False) if from_sheet else []):
        cost = rec.Quantity * rec.AvgPrice if pd.notna(rec.AvgPrice) else np.nan
        if rec.Symbol in known:
            mask = base["Symbol"] == rec.Symbol
            base.loc[mask, "Quantity"] = rec.Quantity
            base.loc[mask, "CostBasis"] = cost
            base.loc[mask, "Currency"] = rec.Currency
        else:
            cls, was_inferred = infer_class(rec.Symbol, rec.Currency, known)
            if was_inferred:
                inferred.append(rec.Symbol)
            new_rows.append({"Symbol": rec.Symbol, "Class": cls, "Currency": rec.Currency, "Quantity": rec.Quantity,
                             "CostBasis": cost, "Realized": 0.0, "Income": 0.0, "Taxes": 0.0, "Fees": 0.0,
                             "Bought": 0.0, "Sold": 0.0})
    if new_rows:
        extra = pd.DataFrame(new_rows, columns=cols)
        base = extra if base.empty else pd.concat([base, extra], ignore_index=True)
    base["Quantity"] = base["Quantity"].astype(float)
    base["AvgCost"] = np.where(base["Quantity"] > QTY_TOL,
                               base["CostBasis"] / base["Quantity"].where(base["Quantity"] > QTY_TOL), np.nan)

    diffs = []
    sheet_qty = dict(zip(sheet["Symbol"], sheet["Quantity"])) if from_sheet else {}
    for sym in sorted(set(sheet_qty) | {s for s, q in ledger_qty.items() if q > QTY_TOL}):
        if sym not in sheet_qty and known.get(sym) in keep_from_ledger:
            continue
        sq, lq = sheet_qty.get(sym, 0.0), ledger_qty.get(sym, 0.0)
        if abs(sq - lq) > QTY_TOL:
            diffs.append((sym, sq, lq))
    return base.sort_values(["Class", "Symbol"]).reset_index(drop=True), diffs, sorted(inferred)


def classes_with_diffs(positions, diffs):
    """Perechile (clasă, monedă) care conțin un simbol unde foaia diferă de registru."""
    syms = {d[0] for d in diffs}
    part = positions[positions["Symbol"].isin(syms)]
    return set(zip(part["Class"], part["Currency"]))
