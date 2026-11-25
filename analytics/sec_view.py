"""Tabelele paginii „Fonduri SUA (13F)”: prezentarea administratorilor, panoul comun al deținerilor, mișcările pentru
consens, căutarea după simbol. Funcții pure (fără Streamlit, fără rețea); calculele de bază sunt în `sec_13f`.
"""
import re
import unicodedata

import numpy as np
import pandas as pd

from analytics import sec_13f

_STOP = {"LLC", "LP", "L.P.", "INC", "CO", "THE", "LTD", "LLP", "MANAGEMENT", "MGMT", "CAPITAL", "GROUP", "ADVISORS",
         "ASSOCIATES", "INVESTMENT", "INVESTMENTS", "ASSET", "FUND", "PARTNERS", "/", "&"}


def _tokens(name):
    s = unicodedata.normalize("NFKD", str(name or "")).encode("ascii", "ignore").decode().upper()
    return {t for t in re.split(r"[^A-Z0-9]+", s) if t and t not in _STOP}


def name_matches(expected, sec_name):
    """Numele de la SEC conține un cuvânt distinctiv din numele așteptat (ex. „BERKSHIRE HATHAWAY INC” ↔
    „Berkshire Hathaway”). Un CIK greșit dă alt nume → semnalat în pagină."""
    a, b = _tokens(expected), _tokens(sec_name)
    return bool(a and b and a & b)


def delay_days(period, filing_date):
    """Zilele dintre sfârșitul trimestrului și depunere (13F: cel mult 45)."""
    try:
        return int((pd.Timestamp(filing_date) - pd.Timestamp(period)).days)
    except (TypeError, ValueError):
        return None


def overview_row(manager, sec_name, curr, prev_book, curr_book, cmap):
    """Un rând din tabelul de prezentare. `curr` = trimestrul curent din `sec_funds.load_quarter` (sau None)."""
    row = {
        "Fond": manager["Fond"], "Administrator": manager["Administrator"], "Stil": manager["Stil"],
        "Clasa": manager["Clasa"], "Clasa_auto": sec_13f.classify_manager(curr_book),
        "Nume_SEC": sec_name, "Nume_ok": name_matches(manager["Fond"], sec_name) if sec_name else None,
        "Perioada": None, "Depus": None, "Decalaj_zile": None, "Pozitii": np.nan, "Valoare_USD": np.nan,
        "Top": None, "Top_pondere": np.nan,
    }
    if curr is not None:
        row.update(Perioada=curr["period"], Depus=curr["last_filing_date"],
                   Decalaj_zile=delay_days(curr["period"], curr["filing_date"]))
    if curr_book is not None and not curr_book.empty:
        top = curr_book.iloc[0]
        row.update(Pozitii=len(curr_book), Valoare_USD=curr_book["Valoare_USD"].sum(min_count=1),
                   Top=cmap.get(top["CUSIP"]) or top["Emitent"], Top_pondere=top["Pondere"])
    return row


def with_symbols(df, cmap):
    """Adaugă coloana `Symbol` (din harta CUSIP → simbol; lipsă = None, nu un simbol ghicit)."""
    out = df.copy()
    out["Symbol"] = pd.Series([cmap.get(c) for c in out["CUSIP"]], index=out.index, dtype=object)
    return out


def holdings_panel(books, managers):
    """Deținerile curente ale tuturor administratorilor încărcați, într-un singur tabel cu CIK, Fond, Clasa.
    `books`: {CIK: equity_book curent}; `managers`: DataFrame cu CIK, Fond, Clasa."""
    frames = []
    meta = managers.set_index("CIK")
    for cik, book in books.items():
        if book is None or book.empty or cik not in meta.index:
            continue
        b = book.copy()
        b["CIK"], b["Fond"], b["Clasa"] = cik, meta.at[cik, "Fond"], meta.at[cik, "Clasa"]
        frames.append(b)
    if not frames:
        return pd.DataFrame(columns=sec_13f.BOOK_COLUMNS + ["CIK", "Fond", "Clasa"])
    return pd.concat(frames, ignore_index=True)


def moves_panel(moves_by_cik):
    """Mișcările din ultimul trimestru ale tuturor administratorilor, cu coloana CIK (pentru `consensus`)."""
    frames = []
    for cik, m in moves_by_cik.items():
        if m is None or m.empty:
            continue
        x = m.copy()
        x["CIK"] = cik
        frames.append(x)
    if not frames:
        return pd.DataFrame(columns=sec_13f.MOVE_COLUMNS + ["CIK"])
    return pd.concat(frames, ignore_index=True)


def split_check_cusips(moves_panel_df):
    """CUSIP-urile „mărite” cu ≥ +50% în acțiuni: un split le poate face să arate ca o cumpărare de convingere.
    Pentru ele se cer split-urile din trimestru înainte de consens."""
    if moves_panel_df is None or moves_panel_df.empty:
        return []
    m = moves_panel_df
    sel = m["Actiune"].eq(sec_13f.ACT_UP) & m["Delta_pct"].ge(sec_13f.CONVICTION_BUY_GROWTH)
    return sorted(set(m.loc[sel, "CUSIP"]))


def cusip6_for_symbol(symbol, cmap, holdings=None):
    """Cheile de emitent (`sec_13f.issuer_key`) pentru simbolul dat. Cu `holdings`, cheile se iau din panou (corect și
    pentru ETF-uri, cheiate pe CUSIP complet); fără, din primele 6 caractere. GOOG și GOOGL dau același emitent."""
    s = (symbol or "").strip().upper()
    cusips = {c for c, sym in cmap.items() if sym == s}
    if holdings is not None and not holdings.empty:
        return sorted(set(holdings.loc[holdings["CUSIP"].isin(cusips), "CUSIP6"]))
    return sorted({c[:6] for c in cusips})


def symbols_by_key(cmap, holdings):
    """Cheie de emitent → simbolurile ei („GOOG/GOOGL”), din panoul de dețineri."""
    if holdings is None or holdings.empty:
        return {}
    out = {}
    for c, k in zip(holdings["CUSIP"], holdings["CUSIP6"]):
        if cmap.get(c):
            out.setdefault(k, set()).add(cmap[c])
    return {k: "/".join(sorted(v)) for k, v in out.items()}


def stale_periods(periods, max_lag_days=60):
    """Perioadele rămase în urmă față de cea mai frecventă (raport vechi → CIK greșit sau fond care depune sub alt
    CIK). Întoarce mulțimea perioadelor vechi."""
    ps = [p for p in periods if p]
    if not ps:
        return set()
    mode = pd.Series(ps).mode().max()
    return {p for p in ps if (pd.Timestamp(mode) - pd.Timestamp(p)).days > max_lag_days}
