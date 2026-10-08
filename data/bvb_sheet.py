"""Citirea foii `BVB` din `portofoliu_db`. Fără Streamlit, fără rețea.

Structura foii: indicatorii pe rânduri (numele în coloana B), companiile pe coloane
(simbolul BVB în primul rând, de la coloana C). Coloana A e un agregat și nu se citește.
Între indicatori sunt rânduri-titlu de secțiune și rânduri goale. Numerele sunt în
format românesc, cu sufixe („24,49", „3,17%", „0,8699 lei"), iar celulele pot conține
erori de foaie („#DIV/0!").
"""
import re
import unicodedata

from data.helpers import smart_to_float


def normalize_label(text):
    """Nume de indicator comparabil: fără diacritice, litere mici, spații comprimate.
    Numele din foaie au spații la capăt și greșeli de scriere, deci nu se compară exact."""
    text = unicodedata.normalize("NFKD", str(text or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", text).strip().lower()


def sheet_number(value):
    """Numărul dintr-o celulă sau None. Celula goală, textul și erorile de foaie dau None
    (nu 0, cum face `smart_to_float`): un „#DIV/0!" nu este un P/E de zero."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return None if value != value else float(value)
    text = str(value).strip()
    if not text or text.startswith("#") or not re.search(r"\d", text):
        return None
    return smart_to_float(text)


# Indicator din foaie (nume normalizat) -> (cheia din `info`, împărțitor). Procentele din
# foaie devin fracții, ca în Yahoo. Doar indicatorii cu același înțeles ca în `info`:
# „Levier financiar" e datorii totale / capital propriu, nu `debtToEquity` (datorie
# purtătoare de dobândă), iar „P/E <an>" nu e Forward P/E, deci nu se mapează.
INFO_MAP = {
    "p/e ttm": ("trailingPE", 1.0),
    "p/bv ttm": ("priceToBook", 1.0),
    "eps ttm": ("trailingEps", 1.0),
    "rentabilitate active (roa)": ("returnOnAssets", 100.0),
    "rentabilitate capital (roe)": ("returnOnEquity", 100.0),
    "marja neta ttm": ("profitMargins", 100.0),
    "marja operationala": ("operatingMargins", 100.0),
    "lichiditate curenta": ("currentRatio", 1.0),
    "lichiditatea imediata": ("quickRatio", 1.0),
}
PERIOD_LABEL = "raportare"


def parse_bvb_sheet(values):
    """Transformă `get_all_values()` al foii BVB într-un dict pe simbol.

    Rezultat: {"SNP": {"indicators": [(nume afișat, text din foaie, număr sau None), ...],
                        "info": {cheie info: valoare}, "period": "Q2 26" sau None}, ...}
    Rândurile fără nume de indicator și rândurile-titlu (fără nicio valoare) sunt sărite.
    Dacă un indicator apare de două ori, contează prima apariție.
    """
    if not values or len(values) < 2:
        return {}
    header = [str(cell or "").strip() for cell in values[0]]
    columns = {idx: name.upper() for idx, name in enumerate(header) if idx >= 2 and name}
    out = {sym: {"indicators": [], "info": {}, "period": None} for sym in columns.values()}
    seen = set()
    for row in values[1:]:
        label = str(row[1]).strip() if len(row) > 1 and row[1] is not None else ""
        key = normalize_label(label)
        if not key or key in seen:
            continue
        cells = {idx: (str(row[idx]).strip() if idx < len(row) and row[idx] is not None else "") for idx in columns}
        if not any(cells.values()):
            continue                      # rând-titlu de secțiune
        seen.add(key)
        for idx, sym in columns.items():
            raw = cells[idx]
            if key == PERIOD_LABEL:
                out[sym]["period"] = raw or None
                continue
            number = sheet_number(raw)
            out[sym]["indicators"].append((label, raw, number))
            if key in INFO_MAP and number is not None:
                info_key, divisor = INFO_MAP[key]
                out[sym]["info"][info_key] = number / divisor
    return out


def bvb_symbol(symbol):
    """Simbolul din foaie pentru un simbol Yahoo: 'SNP.RO' -> 'SNP'. None dacă nu e de la BVB."""
    sym = str(symbol or "").upper().strip()
    return sym[:-3] if sym.endswith(".RO") and len(sym) > 3 else None
