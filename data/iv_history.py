"""Foaia `IV_istoric` (Date, Symbol, IV_ATM): istoricul zilnic al volatilității implicite la bani.

Yahoo nu oferă istoric de IV, deci se salvează zilnic: din aplicație (când bursa SUA e deschisă
și simbolul e afișat) și din jobul programat. Un rând pe (dată, simbol). `IV_ATM` e fracție
(0,2772 = 27,72%), pe expirarea de 20–45 de zile. `Date` e data ședinței la New York.

Funcții pure: fără Streamlit, fără rețea.
"""
from datetime import datetime, time
from zoneinfo import ZoneInfo

import pandas as pd

from data.portfolio_sheet import cell_number

NY_TZ = ZoneInfo("America/New_York")
COLUMNS = ["Date", "Symbol", "IV_ATM"]
MAX_IV = 5.0          # peste 500% e aproape sigur o valoare în procente trecută greșit


def parse_iv_history(values):
    """Rândurile foii → (DataFrame Date, Symbol, IV, probleme). Rândurile invalide sunt sărite și numărate."""
    empty = pd.DataFrame({"Date": pd.Series(dtype="datetime64[ns]"), "Symbol": pd.Series(dtype=str),
                          "IV": pd.Series(dtype=float)})
    if not values:
        return empty, []
    header = [str(h).strip().lower() for h in values[0]]
    try:
        i_date, i_sym, i_iv = (header.index(c.lower()) for c in COLUMNS)
    except ValueError:
        return empty, ["Foaia IV_istoric nu are antetul Date, Symbol, IV_ATM."]
    rows, bad = [], 0
    for row in values[1:]:
        cell = lambda i: row[i] if i < len(row) else None
        sym = str(cell(i_sym) or "").strip().upper()
        date = pd.to_datetime(str(cell(i_date) or "").strip(), format="%Y-%m-%d", errors="coerce")
        iv = cell_number(cell(i_iv))
        if not sym or pd.isna(date) or iv is None or not (0 < iv < MAX_IV):
            if any(str(c).strip() for c in row):
                bad += 1
            continue
        rows.append({"Date": date, "Symbol": sym, "IV": float(iv)})
    problems = [f"IV_istoric: {bad} rânduri invalide sărite (dată AAAA-LL-ZZ, simbol, IV ca fracție)."] if bad else []
    if not rows:
        return empty, problems
    df = pd.DataFrame(rows).drop_duplicates(["Date", "Symbol"], keep="last")
    return df.sort_values(["Symbol", "Date"]).reset_index(drop=True), problems


def history_for(df, symbol):
    """Seria IV (fracție) a unui simbol, indexată pe dată."""
    if df is None or df.empty:
        return pd.Series(dtype=float)
    part = df[df["Symbol"] == str(symbol).upper()]
    return pd.Series(part["IV"].to_numpy(), index=pd.DatetimeIndex(part["Date"]), dtype=float)


def us_session_date(now):
    """Data ședinței la New York pentru un moment cu fus orar."""
    return now.astimezone(NY_TZ).date()


def us_session_open(now):
    """True luni–vineri, 9:30–16:00 ora New York. Sărbătorile bursiere nu sunt excluse
    (în zilele acelea cotațiile sunt vechi; rândul poate fi șters manual)."""
    ny = now.astimezone(NY_TZ)
    return ny.weekday() < 5 and time(9, 30) <= ny.time() < time(16, 0)


def should_save(df, symbol, session_date):
    """True dacă nu există încă un rând pentru (session_date, symbol)."""
    if df is None or df.empty:
        return True
    day = pd.Timestamp(session_date)
    return not ((df["Symbol"] == str(symbol).upper()) & (df["Date"] == day)).any()


def make_row(session_date, symbol, iv):
    """Rândul de scris: [AAAA-LL-ZZ, SIMBOL, IV fracție cu 6 zecimale]."""
    return [session_date.isoformat(), str(symbol).upper(), round(float(iv), 6)]
