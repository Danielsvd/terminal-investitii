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


def keys_for_symbol(symbol, cmap, holdings=None, moves=None):
    """Cheile de emitent ale simbolului din deținerile curente și din mișcări (inclusiv pozițiile închise, care nu mai
    sunt în deținerile curente)."""
    s = (symbol or "").strip().upper()
    cusips = {c for c, sym in cmap.items() if sym == s}
    keys = set()
    for df in (holdings, moves):
        if df is not None and not df.empty:
            keys |= set(df.loc[df["CUSIP"].isin(cusips), "CUSIP6"])
    return sorted(keys) if keys else sorted({c[:6] for c in cusips})


def issuer_moves(moves):
    """Mișcările pe (CIK, emitent): clasele de acțiuni (GOOG + GOOGL) se adună înainte de clasificare, ca un fond care a
    închis o clasă și o păstrează pe cealaltă să nu apară cu „Închisă”. Coloane: CIK, CUSIP6, Actiune, Delta_pct."""
    cols = ["CIK", "CUSIP6", "Actiune", "Delta_pct"]
    if moves is None or moves.empty:
        return pd.DataFrame(columns=cols)
    g = moves.groupby(["CIK", "CUSIP6"]).agg(prev=("Actiuni_ant", "sum"), curr=("Actiuni", "sum")).reset_index()
    acts, pcts = [], []
    for p, c in zip(g["prev"], g["curr"]):
        if not p:
            acts.append(sec_13f.ACT_NEW), pcts.append(np.nan)
        elif not c:
            acts.append(sec_13f.ACT_CLOSED), pcts.append(-1.0)
        else:
            pct = (c - p) / p
            pcts.append(pct)
            acts.append(sec_13f.ACT_UP if pct > sec_13f.MOVE_THRESHOLD else
                        sec_13f.ACT_DOWN if pct < -sec_13f.MOVE_THRESHOLD else sec_13f.ACT_HELD)
    g["Actiune"], g["Delta_pct"] = acts, pcts
    return g[cols]


SIGNAL_COLUMNS = ["Simbol", "Detin", "Convingere", "Intrari", "Mariri", "Reduceri", "Iesiri", "Fonduri", "Miscari"]


def holdings_signals(symbols, cmap, holdings, moves, concentrated):
    """Pentru fiecare simbol: câți administratori concentrați îl dețin (și cu convingere) și ce au făcut în ultimul
    trimestru (intrări = poziții noi, măriri, reduceri, ieșiri = poziții închise). `concentrated` = mulțimea CIK-urilor
    concentrate; `holdings` / `moves` = panourile din `holdings_panel` / `moves_panel` (cu `Fond` în holdings)."""
    conc = set(concentrated)
    h = holdings[holdings["CIK"].isin(conc)] if holdings is not None and not holdings.empty else holdings
    m = issuer_moves(moves[moves["CIK"].isin(conc)]) if moves is not None and not moves.empty else None
    fund_of = {}
    if holdings is not None and not holdings.empty:
        fund_of = dict(zip(holdings["CIK"], holdings["Fond"]))
    rows = []
    for sym in symbols:
        keys = keys_for_symbol(sym, cmap, holdings, moves)
        hh = h[h["CUSIP6"].isin(keys)] if h is not None and not h.empty else pd.DataFrame(columns=["CIK"])
        mm = m[m["CUSIP6"].isin(keys)] if m is not None and not m.empty else pd.DataFrame(columns=["CIK", "Actiune"])
        per = hh.groupby("CIK").agg(Pondere=("Pondere", "sum"), Rang=("Rang", "min")) if not hh.empty else None
        n_conv = int((per["Pondere"].ge(sec_13f.CONVICTION_WEIGHT) | per["Rang"].le(sec_13f.CONVICTION_TOP_N)).sum()) \
            if per is not None else 0
        acts = mm.groupby("CIK")["Actiune"].first() if not mm.empty else pd.Series(dtype=object)
        cnt = acts.value_counts()
        rows.append({
            "Simbol": sym, "Detin": int(hh["CIK"].nunique()), "Convingere": n_conv,
            "Intrari": int(cnt.get(sec_13f.ACT_NEW, 0)), "Mariri": int(cnt.get(sec_13f.ACT_UP, 0)),
            "Reduceri": int(cnt.get(sec_13f.ACT_DOWN, 0)), "Iesiri": int(cnt.get(sec_13f.ACT_CLOSED, 0)),
            "Fonduri": ", ".join(sorted({fund_of.get(c, c) for c in hh["CIK"]})),
            "Miscari": "; ".join(f"{fund_of.get(c, c)}: {a}" for c, a in acts.items() if a != sec_13f.ACT_HELD),
        })
    return pd.DataFrame(rows, columns=SIGNAL_COLUMNS)
