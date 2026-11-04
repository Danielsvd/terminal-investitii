"""Cursurile oficiale BNR (lei pe unitate de valută) din fișierele XML publicate de BNR. Funcții pure.

Surse (publice, fără cheie; BNR publică zilnic, puțin după ora 13:00, în zilele bancare):
- https://www.bnr.ro/nbrfxrates10days.xml — ultimele 10 zile bancare;
- https://www.bnr.ro/files/xml/years/nbrfxrates<AAAA>.xml — un an întreg (anul curent ajunge la ziua precedentă).

Format: <DataSet xmlns="http://www.bnr.ro/xsd"> … <Body><Cube date="AAAA-LL-ZZ"><Rate currency="EUR">5.0866</Rate>
<Rate currency="HUF" multiplier="100">1.3122</Rate> … </Cube></Body>. Cu `multiplier`, cursul e pentru 100 de unități.
"""
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

BNR_10DAYS_URL = "https://www.bnr.ro/nbrfxrates10days.xml"
BNR_YEAR_URL = "https://www.bnr.ro/files/xml/years/nbrfxrates{year}.xml"

# Valutele din tabel, în ordinea afișată: (cod, nume, unitatea afișată). Unitatea 100 pentru valutele
# cu valoare mică pe unitate (cum le publică și BNR). XAU = gramul de aur, în lei.
FX_CURRENCIES = [
    ("EUR", "Euro", 1), ("USD", "Dolar american", 1), ("GBP", "Liră sterlină", 1), ("CHF", "Franc elvețian", 1),
    ("JPY", "Yen japonez", 100), ("CNY", "Yuan chinezesc", 1), ("CAD", "Dolar canadian", 1),
    ("AUD", "Dolar australian", 1), ("SEK", "Coroană suedeză", 1), ("NOK", "Coroană norvegiană", 1),
    ("DKK", "Coroană daneză", 1), ("PLN", "Zlot polonez", 1), ("HUF", "Forint maghiar", 100),
    ("CZK", "Coroană cehă", 1), ("TRY", "Liră turcească", 1), ("MDL", "Leu moldovenesc", 1),
    ("XAU", "Aur (1 gram)", 1),
]
FX_HORIZONS = (("1 lună", 1), ("1 an", 12))


def _local(tag):
    return tag.rsplit("}", 1)[-1]


def parse_bnr_xml(text):
    """Conținutul unui fișier XML BNR → DataFrame (index: data, coloane: codurile valutelor), lei pentru O unitate
    (multiplicatorul aplicat). Un fișier gol, invalid sau fără niciun curs → DataFrame gol. Cursurile nenumerice sau ≤ 0
    rămân NaN (nu 0)."""
    if not text:
        return pd.DataFrame()
    try:
        root = ET.fromstring(text.encode("utf-8") if isinstance(text, str) else text)
    except ET.ParseError:
        return pd.DataFrame()
    rows = {}
    for cube in root.iter():
        if _local(cube.tag) != "Cube" or not cube.get("date"):
            continue
        try:
            day = pd.Timestamp(cube.get("date"))
        except (ValueError, TypeError):
            continue
        rates = {}
        for rate in cube:
            if _local(rate.tag) != "Rate" or not rate.get("currency"):
                continue
            try:
                value = float((rate.text or "").strip())
                mult = float(rate.get("multiplier") or 1)
            except ValueError:
                continue
            if value > 0 and mult > 0:
                rates[rate.get("currency").upper()] = value / mult
        if rates:
            rows[day] = rates
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()


def merge_histories(*frames):
    """Unește istoricele (de ex. anul trecut, anul curent, ultimele 10 zile); la aceeași zi păstrează ultima sursă dată."""
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames)
    out = out[~out.index.duplicated(keep="last")]
    return out.sort_index()


def _pct(now, then):
    if then is None or now is None or not np.isfinite(then) or not np.isfinite(now) or then <= 0:
        return None
    return (now / then - 1.0) * 100.0


def _value_on_or_before(series, day):
    before = series[series.index <= day]
    return float(before.iloc[-1]) if len(before) else None


def fx_table(history, currencies=FX_CURRENCIES):
    """Tabelul cursurilor față de leu, din istoricul BNR (`parse_bnr_xml` / `merge_histories`).

    Pe valută: cursul din ultima zi publicată (pentru unitatea afișată), data lui, variația față de ziua bancară
    precedentă, față de aceeași zi de acum o lună / un an (ultima publicare de la sau înainte de acea zi) și de la
    începutul anului (față de ultimul curs din anul precedent). Un orizont pe care istoricul nu-l acoperă → None (N/A).
    Valuta absentă din istoric → rând cu toate valorile None.
    Variație pozitivă = leul s-a depreciat față de valută (valuta costă mai mulți lei).
    """
    rows = []
    for code, name, unit in currencies:
        row = {"Valută": name, "Cod": code, "Unitate": unit, "Curs BNR": None, "Data": None,
               "Zi %": None, "1 lună %": None, "De la 1 ian. %": None, "1 an %": None}
        s = history[code].dropna() if (history is not None and code in getattr(history, "columns", [])) else pd.Series(dtype=float)
        if len(s):
            last_day, last = s.index[-1], float(s.iloc[-1])
            row["Curs BNR"] = last * unit
            row["Data"] = last_day
            if len(s) >= 2:
                row["Zi %"] = _pct(last, float(s.iloc[-2]))
            for label, months in FX_HORIZONS:
                cutoff = last_day - pd.DateOffset(months=months)
                if s.index[0] <= cutoff:
                    row[f"{label} %"] = _pct(last, _value_on_or_before(s, cutoff))
            prev_year_end = pd.Timestamp(year=last_day.year - 1, month=12, day=31)
            if s.index[0] <= prev_year_end:
                row["De la 1 ian. %"] = _pct(last, _value_on_or_before(s, prev_year_end))
        rows.append(row)
    return pd.DataFrame(rows)
