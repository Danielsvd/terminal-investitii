"""Utilitare de citire sigură a datelor. Fără Streamlit, fără rețea."""
import calendar
import re
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

RO_TZ = ZoneInfo("Europe/Bucharest")


# --- Timp -------------------------------------------------------------------

def now_ro():
    """Ora curentă în România. Serverul Streamlit Cloud rulează în UTC."""
    return datetime.now(RO_TZ)


def struct_time_utc_to_ro(struct_time):
    """Convertește un `time.struct_time` în UTC (cum întoarce feedparser) în ora României."""
    return datetime.fromtimestamp(calendar.timegm(struct_time), tz=RO_TZ)


# --- Valori numerice --------------------------------------------------------

def num(mapping, key):
    """Valoarea numerică de la `key` sau None.

    `info.get(key, 0)` din yfinance nu protejează: cheia poate exista cu valoarea
    None. Aici lipsa, None, NaN și textul nenumeric întorc toate None, niciodată 0.
    """
    if not mapping:
        return None
    value = mapping.get(key)
    if value is None or isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if value != value or value in (float("inf"), float("-inf")):
        return None
    return value


# --- Prețuri din yf.download ------------------------------------------------

def field_frame(data, field, tickers):
    """Extrage un câmp (ex. 'Close') din rezultatul `yf.download`, ca DataFrame
    cu o coloană per simbol.

    Tratează toate formele întoarse de yfinance: coloane simple (un simbol, fără
    MultiIndex), MultiIndex (câmp, simbol) și MultiIndex (simbol, câmp) de la
    `group_by='ticker'`. Simbolurile fără nicio valoare sunt eliminate.
    """
    if isinstance(tickers, str):
        tickers = [tickers]
    tickers = list(tickers)
    if data is None or len(data) == 0:
        return pd.DataFrame()

    cols = data.columns
    if isinstance(cols, pd.MultiIndex):
        if field in cols.get_level_values(0):
            out = data[field]
        elif field in cols.get_level_values(1):
            out = data.xs(field, axis=1, level=1)
        else:
            return pd.DataFrame()
    else:
        if field not in cols:
            return pd.DataFrame()
        out = data[field]

    if isinstance(out, pd.Series):
        out = out.to_frame(name=tickers[0] if tickers else field)
    out = out.copy()
    out.columns = [str(c) for c in out.columns]
    return out.dropna(axis=1, how="all").dropna(axis=0, how="all")


def close_frame(data, tickers):
    """Prețurile de închidere, o coloană per simbol. Vezi `field_frame`."""
    return field_frame(data, "Close", tickers)


# --- Ferestre de timp -------------------------------------------------------

WINDOW_OFFSETS = {
    "1S": pd.DateOffset(days=7),
    "1L": pd.DateOffset(months=1),
    "3L": pd.DateOffset(months=3),
    "6L": pd.DateOffset(months=6),
    "1A": pd.DateOffset(years=1),
    "3A": pd.DateOffset(years=3),
    "5A": pd.DateOffset(years=5),
    "1 Lună": pd.DateOffset(months=1),
    "3 Luni": pd.DateOffset(months=3),
    "6 Luni": pd.DateOffset(months=6),
    "1 An": pd.DateOffset(years=1),
    "3 Ani": pd.DateOffset(years=3),
    "5 Ani": pd.DateOffset(years=5),
}


def slice_window(obj, label, default="1A"):
    """Ultima fereastră calendaristică dintr-o serie/DataFrame cu index de date.

    `label` este o cheie din WINDOW_OFFSETS ("1L", "1A", "3 Luni", ...).
    "1Z" întoarce ultimele două ședințe (ziua curentă și cea precedentă).

    Tăierea se face pe date, nu pe număr de rânduri: un an înseamnă ~252 de
    ședințe, nu 365 de rânduri.
    """
    if obj is None or len(obj) == 0:
        return obj
    if label == "1Z":
        return obj.iloc[-2:]
    offset = WINDOW_OFFSETS.get(label, WINDOW_OFFSETS[default])
    start = obj.index[-1] - offset
    return obj.loc[obj.index >= start]


# --- Numere din Google Sheets -----------------------------------------------

def smart_to_float(val):
    """Transformă un număr scris în format US sau RO/EU în float.

    Exemple: "1.000,50" -> 1000.5, "1,000.50" -> 1000.5, "50,5" -> 50.5,
    "3,17%" -> 3.17, "0,8699 lei" -> 0.8699. Valorile deja numerice trec neschimbate.
    Celulele goale, textul nenumeric și erorile de foaie ("#DIV/0!") dau 0.0.

    Limită cunoscută: un singur separator e ambiguu ("1.250" poate fi 1,25 sau 1250).
    Un singur punct e citit ca zecimală US, o singură virgulă ca zecimală RO.
    """
    if isinstance(val, bool):
        return 0.0
    if isinstance(val, (int, float)):
        return 0.0 if val != val else float(val)
    if val is None or pd.isna(val) or val == '':
        return 0.0
    s = str(val).strip()
    # Păstrăm doar cifre, punct, virgulă și minus
    s = re.sub(r'[^\d.,-]', '', s)
    if not s:
        return 0.0

    # Logică de detecție a formatului
    if ',' in s and '.' in s:
        if s.rfind(',') > s.rfind('.'):  # Format EU: 1.000,50
            s = s.replace('.', '').replace(',', '.')
        else:  # Format US: 1,000.50
            s = s.replace(',', '')
    elif ',' in s:
        if s.count(',') > 1:  # US Thousands: 1,000,000
            s = s.replace(',', '')
        else:  # RO Decimal: 50,5
            s = s.replace(',', '.')
    elif '.' in s:
        if s.count('.') > 1:  # RO Thousands: 1.000.000
            s = s.replace('.', '')
        # Altfel e US Decimal: 50.5

    try:
        return float(s)
    except ValueError:
        return 0.0
