"""Citirea foii `BVB` din `portofoliu_db`. Fără Streamlit, fără rețea.

Structura foii: indicatorii pe rânduri (numele în coloana B), companiile pe coloane
(simbolul BVB în primul rând, de la coloana C). Coloana A este media tuturor companiilor
pe fiecare indicator, calculată în foaie; se afișează ca reper, alături de mediana calculată aici.
Între indicatori sunt rânduri-titlu de secțiune și rânduri goale. Numerele sunt în
format românesc, cu sufixe („24,49", „3,17%", „0,8699 lei"), iar celulele pot conține
erori de foaie („#DIV/0!").
"""
import re
import statistics
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

    Rezultat: {"SNP": {"indicators": [(nume afișat, text din foaie, număr sau None,
                                         media din foaie ca text, mediana ca text), ...],
                        "info": {cheie info: valoare}, "period": "Q2 26" sau None}, ...}
    Media e cea din coloana A a foii; mediana se calculează aici din companiile cu valoare
    și e mai puțin sensibilă la extreme (un P/E de 200 trage media, nu și mediana).
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
        numbers = [n for n in (sheet_number(raw) for raw in cells.values()) if n is not None]
        sample = next((raw for raw in cells.values() if sheet_number(raw) is not None), "")
        median_text = _format_like(statistics.median(numbers), sample) if numbers else "N/A"
        average_raw = str(row[0]).strip() if row and row[0] is not None else ""
        average_text = average_raw if sheet_number(average_raw) is not None else "N/A"
        for idx, sym in columns.items():
            raw = cells[idx]
            if key == PERIOD_LABEL:
                out[sym]["period"] = raw or None
                continue
            number = sheet_number(raw)
            out[sym]["indicators"].append((label, raw, number, average_text, median_text))
            if key in INFO_MAP and number is not None:
                info_key, divisor = INFO_MAP[key]
                out[sym]["info"][info_key] = number / divisor
    return out


def _format_like(value, sample):
    """Formatează un număr ca celulele din foaie: „3,17%", „0,8699 lei" sau „24,49"."""
    if "%" in sample:
        text = f"{value:.2f}%"
    elif "lei" in sample.lower():
        text = f"{value:.4f} lei"
    else:
        text = f"{value:.2f}"
    return text.replace(".", ",")


def reprice(sheet_info, price):
    """P/E și P/BV la prețul curent, din EPS-ul și valoarea contabilă din foaie.

    Foaia dă multiplii la prețul din ziua actualizării ei. Valoarea contabilă pe acțiune
    nu e în foaie, dar rezultă din ea: P/E = preț / EPS și P/BV = preț / BVPS la același
    preț, deci BVPS = EPS × (P/E) / (P/BV). Întoarce doar cheile care se pot recalcula:
    P/E cere EPS > 0; BVPS cere EPS > 0, P/E și P/BV pozitive. Fără preț, dict gol.
    """
    out = {}
    eps, pe, pbv = sheet_info.get("trailingEps"), sheet_info.get("trailingPE"), sheet_info.get("priceToBook")
    if price is None or price != price or price <= 0:
        return out
    if eps is not None and eps > 0:
        out["trailingPE"] = price / eps
        if pe is not None and pe > 0 and pbv is not None and pbv > 0:
            book = eps * pe / pbv
            out["bookValue"] = book
            out["priceToBook"] = price / book
    return out


def bvb_symbol(symbol):
    """Simbolul din foaie pentru un simbol Yahoo: 'SNP.RO' -> 'SNP'. None dacă nu e de la BVB."""
    sym = str(symbol or "").upper().strip()
    return sym[:-3] if sym.endswith(".RO") and len(sym) > 3 else None
