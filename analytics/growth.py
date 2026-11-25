"""Creșterea afacerii (CAGR) și sursa randamentului acționarului — card informativ în Analiză companie.

Cum gândește un manager de portofoliu (decizia lui Daniel, 25.11.2026: A + B + C):
  A. Creșterea afacerii, pe acțiune: cifra de afaceri, EPS, FCF și dividendul pe acțiune (diluarea și răscumpărările
     contează pentru acționar). Totalul se arată alături, ca să se vadă efectul numărului de acțiuni.
  B. Randamentul total anualizat al acțiunii (prețul + dividendele reinvestite) față de un benchmark de randament total
     din aceeași piață și valută.
  C. Sursa randamentului pe perioada situațiilor financiare: (1 + randament total) = (1 + creștere EPS) ×
     (1 + schimbarea P/E) × (1 + contribuția dividendului). Identitatea e exactă în formă multiplicativă; suma
     componentelor e o aproximare, iar diferența se afișează.

Reguli: CAGR = (final / inițial)^(1/n) − 1; un capăt lipsă sau ≤ 0 → None („N/A”), nu un număr fără sens.
Numărul de acțiuni schimbat mult între capete (raport în afara 0,8–1,25: split, acțiuni gratuite, majorare) → valorile
pe acțiune și descompunerea sunt None (aceeași regulă ca `company_files.annual_eps_growth`), totalurile rămân.
Informativ: nu intră în scor (momentul e deja în pasul tehnic). Funcții pure, fără rețea și fără Streamlit.
"""
import math

import numpy as np
import pandas as pd

from analytics import fundamentals as fund

SHARES_OK = (0.8, 1.25)          # raportul acțiunilor între capete în care valorile pe acțiune sunt comparabile
PRICE_TOL_DAYS = 10              # prețul de la o dată: ultima cotație cu cel mult atâtea zile înainte
HORIZONS = (3, 5, 10)


def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def cagr(new, old, years):
    """Creșterea anuală compusă. None dacă un capăt lipsește, e ≤ 0, sau perioada e ≤ 0."""
    new, old, years = _num(new), _num(old), _num(years)
    if new is None or old is None or years is None or years <= 0 or new <= 0 or old <= 0:
        return None
    return (new / old) ** (1.0 / years) - 1.0


def shares_comparable(sh_new, sh_old):
    """Adevărat dacă numărul de acțiuni s-a schimbat puțin (raport în 0,8–1,25); None dacă lipsește un capăt."""
    a, b = _num(sh_new), _num(sh_old)
    if a is None or b is None or a <= 0 or b <= 0:
        return None
    return SHARES_OK[0] <= a / b <= SHARES_OK[1]


# --------------------------------------------------------------------------- A. Situațiile Yahoo (SUA/UE)

def _points(df, rows):
    """{data fiscală: valoare} pentru primul rând găsit, pe toate coloanele situației."""
    out = {}
    d = fund._sorted_desc(df)
    if d is None:
        return out
    for col in range(d.shape[1]):
        v, dt = fund.stmt_value(d, rows, col), fund.stmt_date(d, col)
        if v is not None and dt is not None:
            out[dt] = v
    return out


def _match(points, date, tol_days=40):
    """Valoarea de la data fiscală cea mai apropiată (situațiile au uneori date diferite cu câteva zile)."""
    best = None
    for d, v in points.items():
        gap = abs((d - date).days)
        if gap <= tol_days and (best is None or gap < best[0]):
            best = (gap, v)
    return None if best is None else best[1]


def statement_growth(income, cashflow, min_years=2, max_years=3):
    """CAGR din situațiile anuale Yahoo (~4 ani → de obicei 3). Întoarce dict cu `years`, `date_old`, `date_new` și
    rândurile {indicator: {"total": CAGR, "per_share": CAGR, "note": text}}; None fără două capete utilizabile."""
    rev = _points(income, fund.REVENUE_ROWS)
    ni = _points(income, fund.NET_INCOME_ROWS)
    eps = _points(income, fund.EPS_ROWS)
    sh = _points(income, fund.DILUTED_SHARES_ROWS) or _points(income, ("Basic Average Shares",))
    cfo = _points(cashflow, fund.CFO_ROWS)
    capex = _points(cashflow, fund.CAPEX_ROWS)
    dates = sorted(rev or ni, reverse=True)
    if len(dates) < 2:
        return None
    d_new = dates[0]
    spans = [((d_new - d).days / 365.25, d) for d in dates[1:]]
    ok = [s for s in spans if min_years - 0.5 <= s[0] <= max_years + 0.5]
    if not ok:
        return None
    span, d_old = max(ok, key=lambda s: s[0])
    years = round(span)

    def at(points, d):
        return _match(points, d)

    s_new, s_old = at(sh, d_new), at(sh, d_old)
    comp = shares_comparable(s_new, s_old)
    note_sh = "" if comp else ("numărul de acțiuni lipsește" if comp is None
                               else f"acțiunile s-au schimbat de {s_new / s_old:.2f}× (split/emisiune): pe acțiune N/A")

    def fcf(d):
        c, x = at(cfo, d), at(capex, d)
        return None if c is None or x is None else c + x        # capex e negativ în Yahoo

    def ps(v, s):
        return None if v is None or s is None or s <= 0 else v / s

    rows = {}
    r_new, r_old = at(rev, d_new), at(rev, d_old)
    rows["Cifră de afaceri"] = {
        "total": cagr(r_new, r_old, years),
        "per_share": cagr(ps(r_new, s_new), ps(r_old, s_old), years) if comp else None, "note": note_sh}
    e_new, e_old = at(eps, d_new), at(eps, d_old)
    if e_new is None or e_old is None:                          # EPS raportat lipsă: profit net / acțiuni diluate
        e_new, e_old = ps(at(ni, d_new), s_new), ps(at(ni, d_old), s_old)
    eps_note = note_sh or ("EPS ≤ 0 la un capăt: CAGR nedefinit" if (e_new is not None and e_old is not None
                                                                     and min(e_new, e_old) <= 0) else "")
    rows["EPS diluat"] = {"total": None, "per_share": cagr(e_new, e_old, years) if comp else None, "note": eps_note}
    f_new, f_old = fcf(d_new), fcf(d_old)
    fcf_note = "FCF ≤ 0 la un capăt: CAGR nedefinit" if (f_new is not None and f_old is not None
                                                          and min(f_new, f_old) <= 0) else ""
    rows["FCF (CFO − capex)"] = {
        "total": cagr(f_new, f_old, years),
        "per_share": cagr(ps(f_new, s_new), ps(f_old, s_old), years) if comp else None,
        "note": "; ".join(x for x in (fcf_note, note_sh) if x)}
    return {"years": years, "date_old": d_old, "date_new": d_new, "rows": rows,
            "eps_new": e_new if comp else None, "eps_old": e_old if comp else None}


# --------------------------------------------------------------------------- A. Fișierele „Analiză companii” (BVB)

def file_growth(df, horizons=(3, 5)):
    """CAGR din foaia anuală a fișierului BVB (luni = 12), pe 3 și 5 ani. {indicator: {n: {"total", "per_share"}}},
    plus `year_new`, `shares_note` și EPS-urile capetelor pe 3 ani (pentru descompunere). None fără date anuale.
    Dividendul pe acțiune = profit distribuit / acțiuni emise (din fișier)."""
    if df is None or df.empty:
        return None
    a = df[(df["months"] == 12)].set_index("year").sort_index()
    a = a[a["rev"].notna()]
    if a.empty:
        return None
    y_new = int(a.index.max())

    def val(y, f):
        return _num(a.at[y, f]) if y in a.index and f in a.columns else None

    out = {"year_new": y_new, "rows": {"Cifră de afaceri": {}, "EPS (profit net / acțiuni)": {},
                                       "Dividend pe acțiune (profit distribuit / acțiuni)": {}}, "notes": {}}
    for n in horizons:
        y_old = y_new - n
        s_new, s_old = val(y_new, "shares"), val(y_old, "shares")
        comp = shares_comparable(s_new, s_old)
        if comp is False:
            out["notes"][n] = f"acțiunile s-au schimbat de {s_new / s_old:.2f}× ({y_old}–{y_new}): pe acțiune N/A"
        elif comp is None:
            out["notes"][n] = f"lipsesc acțiunile sau anul {y_old}"

        def ps(v, s):
            return None if v is None or s is None or s <= 0 else v / s
        r_new, r_old = val(y_new, "rev"), val(y_old, "rev")
        out["rows"]["Cifră de afaceri"][n] = {
            "total": cagr(r_new, r_old, n), "per_share": cagr(ps(r_new, s_new), ps(r_old, s_old), n) if comp else None}
        e_new, e_old = ps(val(y_new, "ni"), s_new), ps(val(y_old, "ni"), s_old)
        out["rows"]["EPS (profit net / acțiuni)"][n] = {"total": cagr(val(y_new, "ni"), val(y_old, "ni"), n),
                                                       "per_share": cagr(e_new, e_old, n) if comp else None}
        d_new, d_old = ps(val(y_new, "div_total"), s_new), ps(val(y_old, "div_total"), s_old)
        out["rows"]["Dividend pe acțiune (profit distribuit / acțiuni)"][n] = {
            "total": cagr(val(y_new, "div_total"), val(y_old, "div_total"), n),
            "per_share": cagr(d_new, d_old, n) if comp else None}
        if n == 3:
            out["eps_new"], out["eps_old"] = (e_new, e_old) if comp else (None, None)
            out["date_new"] = pd.Timestamp(y_new, 12, 31)
            out["date_old"] = pd.Timestamp(y_old, 12, 31)
    return out


# --------------------------------------------------------------------------- dividende Yahoo (SUA/UE)

def dividend_growth(dividends, asof, horizons=HORIZONS):
    """CAGR-ul dividendului pe acțiune din plățile Yahoo (ajustate pentru split), pe ani calendaristici încheiați:
    ultimul an complet față de cu n ani înainte. {n: CAGR sau None}, plus anul de referință. Un an fără plată → None."""
    if dividends is None or len(dividends) == 0:
        return None, {n: None for n in horizons}
    s = pd.Series(dividends).dropna()
    s = s[s > 0]
    if s.empty:
        return None, {n: None for n in horizons}
    idx = pd.DatetimeIndex(s.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    by_year = pd.Series(s.values, index=idx).groupby(idx.year).sum()
    y_ref = pd.Timestamp(asof).year - 1
    return y_ref, {n: cagr(by_year.get(y_ref), by_year.get(y_ref - n), n) for n in horizons}


# --------------------------------------------------------------------------- B. randament total

def _naive(series):
    s = pd.Series(series).dropna()
    s = s[s > 0]
    if len(s) and getattr(s.index, "tz", None) is not None:
        s.index = s.index.tz_localize(None)
    return s.sort_index()


def price_at(series, date, tol_days=PRICE_TOL_DAYS):
    """Ultima cotație la sau înainte de `date`, cu cel mult `tol_days` zile în urmă; altfel None."""
    s = _naive(series)
    if s.empty:
        return None
    s = s[s.index <= pd.Timestamp(date)]
    if s.empty or (pd.Timestamp(date) - s.index[-1]).days > tol_days:
        return None
    return float(s.iloc[-1])


def total_return(adj_close, years, end=None):
    """Randamentul total anualizat pe `years` ani calendaristici până la ultima cotație (sau `end`), din închiderile
    ajustate pentru dividende și split. dict {cagr, start, end, n_obs} sau None (istoric prea scurt)."""
    s = _naive(adj_close)
    if s.empty:
        return None
    end = pd.Timestamp(end) if end is not None else s.index[-1]
    s = s[s.index <= end]
    if s.empty:
        return None
    start = end - pd.DateOffset(years=years)
    p0, p1 = price_at(s, start), float(s.iloc[-1])
    if p0 is None:
        return None
    return {"cagr": cagr(p1, p0, years), "start": start, "end": s.index[-1],
            "n_obs": int(((s.index > start) & (s.index <= end)).sum())}


def tr_benchmark(symbol, currency=None):
    """Benchmarkul de randament total (cu dividende) din aceeași piață și valută, sau (None, None).
    S&P 500 prin SPY ajustat; DAX e deja indice de randament total; restul zonei euro prin EXSA.DE (STOXX 600,
    ajustat); BVB prin TVBETETF.RO ajustat (proxy pentru BET; ajustarea depinde de dividendele înregistrate de Yahoo)."""
    sym, cur = (symbol or "").upper(), (currency or "").upper()
    if sym.endswith(".RO"):
        return "TVBETETF.RO", "BET (ETF TVBETETF, ajustat; proxy)"
    if sym.endswith(".DE"):
        return "^GDAXI", "DAX (indice de randament total)"
    if cur == "EUR":
        return "EXSA.DE", "STOXX Europe 600 (EXSA.DE, ajustat)"
    if cur == "USD":
        return "SPY", "S&P 500 (SPY, ajustat)"
    return None, None


# --------------------------------------------------------------------------- C. sursa randamentului

def decompose(close, adj_close, eps_new, eps_old, date_new, date_old):
    """Sursa randamentului între două date fiscale. `close` = prețul ajustat doar pentru split (fără dividende),
    `adj_close` = prețul ajustat și pentru dividende. Identitate: (1 + total) = (1 + EPS) × (1 + P/E) × (1 + dividend).
    dict cu componentele anualizate, P/E la capete și diferența dintre suma componentelor și total; None dacă lipsește
    un preț sau EPS ≤ 0 la un capăt."""
    years = round((pd.Timestamp(date_new) - pd.Timestamp(date_old)).days / 365.25)
    e1, e0 = _num(eps_new), _num(eps_old)
    if years <= 0 or e1 is None or e0 is None or e1 <= 0 or e0 <= 0:
        return None
    p1, p0 = price_at(close, date_new), price_at(close, date_old)
    t1, t0 = price_at(adj_close, date_new), price_at(adj_close, date_old)
    if None in (p1, p0, t1, t0):
        return None
    g_tr, g_p, g_e = cagr(t1, t0, years), cagr(p1, p0, years), cagr(e1, e0, years)
    g_pe = (1 + g_p) / (1 + g_e) - 1
    g_div = (1 + g_tr) / (1 + g_p) - 1
    return {"years": years, "total": g_tr, "price": g_p, "eps": g_e, "pe": g_pe, "div": g_div,
            "pe_old": p0 / e0, "pe_new": p1 / e1, "residual": g_tr - (g_e + g_pe + g_div)}
