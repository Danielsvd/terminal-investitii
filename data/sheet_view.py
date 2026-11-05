"""Tabelele din tabul „Import Date”: foile `BVB` și `GLOBAL` din `portofoliu_db`, așa cum sunt în foaie.

Foile se calculează singure (prețul vine din GOOGLEFINANCE în foaie), deci aici nu se recalculează nimic:
celulele se transformă doar în numere, ca tabelul să se sorteze corect. Fără Streamlit, fără rețea.

Reguli:
- celulă goală, „-”, „-%”, „#DIV/0!”, „#N/A” → None (afișat „N/A”), niciodată 0;
- o coloană e numerică dacă toate celulele ei completate sunt numere sau lipsuri; altfel rămâne text;
- moneda se citește din fiecare celulă („$” USD, „€” EUR, „£” GBP, „lei” RON), nu se presupune.
"""
import re
import statistics

import pandas as pd

from data.bvb_sheet import normalize_label, sheet_number

CURRENCY_MARKS = (("$", "USD"), ("€", "EUR"), ("£", "GBP"), ("lei", "RON"), ("ron", "RON"))
MISSING_TEXTS = {"", "-", "-%", "—", "n/a", "na"}
AVERAGE_ROW = "Media (coloana A din foaie)"
MEDIAN_ROW = "Mediana (calculată aici)"
GLOBAL_TEXT_COLUMNS = {"industrie", "companii", "recomandare", "sector"}
CAP_COLUMN = "capitalizare"


def _text(cell):
    return "" if cell is None else str(cell).strip()


def is_missing(cell):
    """True pentru celulele care înseamnă „lipsă”: goale, liniuță, erori de foaie."""
    text = _text(cell)
    return text.casefold() in MISSING_TEXTS or text.startswith("#")


def cell_currency(cell):
    """Moneda scrisă în celulă (USD, EUR, GBP, RON) sau None."""
    text = _text(cell).casefold()
    for mark, code in CURRENCY_MARKS:
        if mark in text:
            return code
    return None


_NUMBER_RE = re.compile(r"^[-+]?[\d.,]*\d[\d.,]*$|^[-+]?[.,]\d+$")


def looks_numeric(cell):
    """True dacă celula e un număr, eventual cu „%”, monedă sau „lei” („1.003,40€”, „-,34 $”, „3,17%”).
    „Q2 26” sau „T2 2025” nu sunt numere (sheet_number le-ar citi 226 / 22025)."""
    text = _text(cell).casefold().replace("\u00a0", "").replace(" ", "")
    for mark in ("%", "$", "€", "£", "lei", "ron"):
        text = text.replace(mark, "")
    return bool(_NUMBER_RE.match(text))


def cell_number(cell):
    """Numărul din celulă sau None (lipsă = None, nu 0). „3,26%%” → 3.26; „,97 $” → 0.97."""
    if isinstance(cell, (int, float)) and not isinstance(cell, bool):
        return sheet_number(cell)
    if is_missing(cell) or not looks_numeric(cell):
        return None
    return sheet_number(cell)


def column_kind(cells):
    """(numeric, unitate) pentru o coloană. numeric = toate celulele completate sunt numere sau lipsuri;
    unitate = „%” dacă majoritatea celulelor cu număr au „%”, „ lei” dacă au „lei”, altfel „”."""
    filled = [c for c in cells if not is_missing(c)]
    if not filled:
        return True, ""                     # doar lipsuri: N/A peste tot
    if any(cell_number(c) is None for c in filled):
        return False, ""
    pct = sum("%" in _text(c) for c in filled)
    lei = sum("lei" in _text(c).casefold() for c in filled)
    unit = "%" if pct * 2 > len(filled) else (" lei" if lei * 2 > len(filled) else "")
    return True, unit


def bvb_view(values):
    """Foaia BVB (indicatori pe rânduri, companii pe coloane) → tabel cu companiile pe rânduri.

    Întoarce (df, units, dropped):
    - df: index = simbolurile din primul rând (de la coloana C), apoi rândul mediei din coloana A și rândul medianei
      calculate aici din companiile cu valoare; coloane = indicatorii, în ordinea din foaie (numele fără spații la capete);
    - units: {indicator: „%” / „ lei” / „”} pentru coloanele numerice; cele care lipsesc rămân text;
    - dropped: indicatorii care apar a doua oară (contează prima apariție, ca în restul aplicației).
    Rândurile fără nume și rândurile-titlu (fără nicio valoare la companii) sunt sărite.
    """
    if not values or len(values) < 2:
        return pd.DataFrame(), {}, []
    header = [_text(c) for c in values[0]]
    sym_cols = [(i, name.upper()) for i, name in enumerate(header) if i >= 2 and name]
    symbols = [s for _, s in sym_cols]
    data, units, dropped, seen = {}, {}, [], set()
    for row in values[1:]:
        label = _text(row[1]) if len(row) > 1 else ""
        key = normalize_label(label)
        if not key:
            continue
        cells = [row[i] if i < len(row) else "" for i, _ in sym_cols]
        if all(not _text(c) for c in cells):
            continue                                    # rând-titlu de secțiune
        if key in seen:
            dropped.append(label)
            continue
        seen.add(key)
        average = row[0] if row else ""
        numeric, unit = column_kind(cells + [average])
        if numeric:
            nums = [cell_number(c) for c in cells]
            have = [n for n in nums if n is not None]
            data[label] = nums + [cell_number(average), statistics.median(have) if have else None]
            units[label] = unit
        else:
            data[label] = [_text(c) or None for c in cells] + [None, None]
    df = pd.DataFrame(data, index=symbols + [AVERAGE_ROW, MEDIAN_ROW])
    return df, units, dropped


def global_view(values):
    """Foaia GLOBAL (o companie pe rând) → (df, units, currencies).

    - df: aceleași coloane ca în foaie (nume fără spații la capete), index 0..n-1; „Companii” fără spații la capete;
      „Industrie” completată în jos (în foaie e scrisă doar la prima companie din grup); coloana „Capitalizare” devine
      „Capitalizare (mld.)” = valoarea / 1e9; coloanele numerice au float sau None.
    - units: {coloană: „%” / „”} pentru coloanele numerice (moneda nu intră aici);
    - currencies: DataFrame de aceeași formă cu moneda fiecărei celule (None unde nu e scrisă).
    Rândurile fără nume de companie sunt sărite.
    """
    if not values or len(values) < 2:
        return pd.DataFrame(), {}, pd.DataFrame()
    header = [_text(c) for c in values[0]]
    keep = [i for i, h in enumerate(header) if h and not h.startswith("Unnamed")]
    names = [header[i] for i in keep]
    norm = [normalize_label(n) for n in names]
    comp = norm.index("companii") if "companii" in norm else None
    rows = []
    for row in values[1:]:
        cells = [row[i] if i < len(row) else "" for i in keep]
        if comp is not None and not _text(cells[comp]):
            continue
        rows.append(cells)
    data, cur, units, out_names = {}, {}, {}, []
    for j, (name, key) in enumerate(zip(names, norm)):
        cells = [r[j] for r in rows]
        numeric, unit = (False, "") if key in GLOBAL_TEXT_COLUMNS else column_kind(cells)
        col = "Capitalizare (mld.)" if key == CAP_COLUMN and numeric else name
        if numeric:
            nums = [cell_number(c) for c in cells]
            if key == CAP_COLUMN:
                nums = [None if n is None else n / 1e9 for n in nums]
            data[col] = nums
            units[col] = unit
        else:
            texts = [_text(c) or None for c in cells]
            if key == "industrie":
                last = None
                texts = [last := (t or last) for t in texts]
            data[col] = texts
        cur[col] = [cell_currency(c) for c in cells]
        out_names.append(col)
    df = pd.DataFrame(data, columns=out_names)
    return df, units, pd.DataFrame(cur, columns=out_names)


def currency_groups(currencies, column):
    """{monedă: [indici de rând]} pentru o coloană; rândurile fără monedă scrisă nu apar."""
    groups = {}
    if column not in getattr(currencies, "columns", []):
        return groups
    for idx, code in currencies[column].items():
        if code:
            groups.setdefault(code, []).append(idx)
    return groups


def mixed_currency_rows(df, currencies, name_column="Companii"):
    """Companiile cu mai multe monede pe același rând (de ex. preț în EUR, țintă în USD).
    Listă de (nume, moneda majoritară pe rând, {coloană: moneda diferită})."""
    out = []
    for idx in currencies.index:
        row = {c: v for c, v in currencies.loc[idx].items() if v}
        codes = list(row.values())
        if len(set(codes)) > 1:
            main = max(set(codes), key=lambda c: (codes.count(c), c))
            name = df.at[idx, name_column] if name_column in df.columns else idx
            out.append((name, main, {c: v for c, v in row.items() if v != main}))
    return out
