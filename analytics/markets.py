"""Piețele în care investește Daniel: regimul tehnic al fiecărui indice și randamentul lui în lei. Funcții pure.

Regimul folosește exact regula „Moment” din Analiză Companie (analytics/scoring.py: trend SMA 50/200 +
momentum 12-1, sub-scor 0–100). Pragurile au fost testate pe acțiuni (BVB), nu pe indici: pe indici scorul
e descriptiv.

Randamentul în lei descompune: (1 + randament local) × (1 + variația cursului valutei față de leu) − 1.
Indicii sunt de preț (fără dividende); ETF-ul BET (TVBETETF.RO) e proxy pentru BET.
"""
import numpy as np
import pandas as pd

from analytics.scoring import momentum_12_1, technical_subscore

# (cheie, nume, simbol Yahoo, valuta în care e cotat, notă)
MARKETS = [
    ("sp500", "S&P 500", "^GSPC", "USD", "indice de preț"),
    ("nasdaq", "Nasdaq Composite", "^IXIC", "USD", "indice de preț"),
    ("stoxx600", "STOXX Europe 600", "^STOXX", "EUR", "indice de preț"),
    ("dax", "DAX", "^GDAXI", "EUR", "indice cu dividende reinvestite (DAX e indice de performanță)"),
    ("bet", "BET (prin ETF TVBETETF)", "TVBETETF.RO", "RON", "proxy: ETF-ul care urmărește BET"),
    ("gold", "Aur (futures)", "GC=F", "USD", "contract futures, USD/uncie"),
]
HORIZONS = {"1 lună": ("months", 1), "De la 1 ian.": ("ytd", None), "1 an": ("months", 12)}
HIGH_WINDOW_WEEKS = 52


def _clean(close):
    if close is None:
        return pd.Series(dtype="float64")
    s = pd.to_numeric(pd.Series(close, dtype="float64"), errors="coerce").dropna()
    s = s[s > 0]
    if len(s) and isinstance(s.index, pd.DatetimeIndex):
        idx = s.index.tz_localize(None) if s.index.tz is not None else s.index
        s = pd.Series(s.values, index=idx.normalize())
        s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


def _on_or_before(s, day):
    before = s[s.index <= day]
    return float(before.iloc[-1]) if len(before) else None


def trend_count(close):
    """Câte din cele 3 condiții de trend sunt îndeplinite (Close > SMA50, Close > SMA200, SMA50 > SMA200),
    sau None sub 200 de ședințe."""
    c = _clean(close)
    if len(c) < 200:
        return None
    last, sma50, sma200 = c.iloc[-1], c.iloc[-50:].mean(), c.iloc[-200:].mean()
    return int(last > sma50) + int(last > sma200) + int(sma50 > sma200)


def market_regime(close):
    """Regimul unui indice la ultima ședință: dict last, date, trend_n (0–3 sau None), momentum (12-1, fracție),
    score (sub-scorul tehnic 0–100), from_high (ultima ÷ maximul pe 52 de săptămâni − 1), n (ședințe).
    Valorile care nu se pot calcula sunt None."""
    c = _clean(close)
    out = {"last": None, "date": None, "trend_n": None, "momentum": None, "score": None, "from_high": None, "n": len(c)}
    if c.empty:
        return out
    out["last"], out["date"] = float(c.iloc[-1]), c.index[-1]
    out["trend_n"] = trend_count(c)
    out["momentum"] = momentum_12_1(c)
    out["score"] = technical_subscore(c)
    window = c[c.index > c.index[-1] - pd.DateOffset(weeks=HIGH_WINDOW_WEEKS)]
    if c.index[0] <= c.index[-1] - pd.DateOffset(weeks=HIGH_WINDOW_WEEKS) and len(window):
        out["from_high"] = float(c.iloc[-1] / window.max() - 1.0)
    return out


def horizon_start(last_day, horizon):
    """Data de pornire a unui orizont din HORIZONS: aceeași zi de acum N luni sau 31 decembrie din anul precedent."""
    kind, months = HORIZONS[horizon]
    if kind == "ytd":
        return pd.Timestamp(year=last_day.year - 1, month=12, day=31)
    return last_day - pd.DateOffset(months=months)


def ron_return(close, fx_ron=None, horizon="1 an"):
    """Randamentul pe orizont: dict local, fx, ron (fracții) sau None unde nu se poate calcula.

    `close`: închiderile în valuta pieței. `fx_ron`: lei pe o unitate de valută (de ex. cursul BNR), cu index de date;
    None pentru o piață cotată în lei (efect valutar 0). Prețul și cursul de pornire sunt ultimele de la sau înainte
    de data de pornire; cursul final e ultimul de la sau înainte de ultima ședință a indicelui (aceeași zi, nu cursul de azi).
    """
    c = _clean(close)
    out = {"local": None, "fx": None, "ron": None, "start": None, "end": None}
    if len(c) < 2:
        return out
    last_day = c.index[-1]
    start = horizon_start(last_day, horizon)
    if c.index[0] > start:
        return out
    p0 = _on_or_before(c, start)
    out["start"], out["end"] = start, last_day
    out["local"] = float(c.iloc[-1] / p0 - 1.0)
    if fx_ron is None:
        out["fx"] = 0.0
    else:
        f = _clean(fx_ron)
        if len(f) and f.index[0] <= start:
            f0, f1 = _on_or_before(f, start), _on_or_before(f, last_day)
            if f0 and f1:
                out["fx"] = float(f1 / f0 - 1.0)
    if out["fx"] is not None:
        out["ron"] = float((1.0 + out["local"]) * (1.0 + out["fx"]) - 1.0)
    return out


# Lățimea pieței SUA: cât de larg e participată creșterea. ETF-uri pe aceleași 500 de companii cu ponderi egale (RSP)
# și pe companii mici (IWM), comparate cu S&P 500 ponderat cu capitalizarea (SPY). Total return (ajustate).
BREADTH_PAIRS = [
    ("RSP", "SPY", "S&P 500 ponderi egale vs ponderat cu capitalizarea"),
    ("IWM", "SPY", "Companii mici (Russell 2000) vs S&P 500"),
]
BREADTH_MONTHS = (3, 12)


def relative_return(a_close, b_close, months):
    """Randamentul lui A, al lui B pe ultimele `months` luni calendaristice și diferența A − B (fracții).
    Ambele se măsoară până la ultima zi comună. None unde istoricul nu acoperă perioada."""
    a, b = _clean(a_close), _clean(b_close)
    common = a.index.intersection(b.index)
    out = {"a": None, "b": None, "diff": None, "end": None}
    if len(common) < 2:
        return out
    end = common[-1]
    start = end - pd.DateOffset(months=months)
    if a.index[0] > start or b.index[0] > start:
        return out
    ra = float(a.loc[end] / _on_or_before(a, start) - 1.0)
    rb = float(b.loc[end] / _on_or_before(b, start) - 1.0)
    out.update({"a": ra, "b": rb, "diff": ra - rb, "end": end})
    return out


def earnings_yield(eps, price):
    """EPS ÷ preț (fracție). Profitul negativ dă randament negativ (compania intră în mediană, nu e exclusă).
    None fără preț pozitiv sau fără EPS."""
    try:
        e, p = float(eps), float(price)
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(e) and np.isfinite(p)) or p <= 0:
        return None
    return e / p


def equity_risk_gap(yields, bond_yield_pct):
    """Prima de risc a acțiunilor (proxy): mediana randamentelor profiturilor − randamentul titlurilor de stat pe 10 ani.

    `yields`: randamentele profiturilor (fracții) ale companiilor; `bond_yield_pct`: randamentul pe 10 ani, în procente.
    Întoarce dict ep (mediana, %), n (companii), bond (%), gap (puncte procentuale) — None unde lipsesc datele."""
    ys = [float(y) for y in (yields or []) if y is not None and np.isfinite(y)]
    ep = float(np.median(ys)) * 100.0 if ys else None
    try:
        bond = float(bond_yield_pct) if bond_yield_pct is not None and np.isfinite(float(bond_yield_pct)) else None
    except (TypeError, ValueError):
        bond = None
    gap = ep - bond if (ep is not None and bond is not None) else None
    return {"ep": ep, "n": len(ys), "bond": bond, "gap": gap}
