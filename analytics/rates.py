"""Dobânzi comparate (SUA, zona euro/Germania, România), dobânzi reale și prima de risc de țară. Funcții pure.

Seriile vin în procente: zilnice (FRED, BCE) indexate pe date, lunare (BCE, Eurostat) indexate pe „AAAA-LL”.
Diferențele între două serii (spread-uri, dobânzi reale) se fac doar pe aceeași lună (valoarea de la sfârșitul
lunii pentru seriile zilnice), niciodată amestecând luni diferite.
"""
import numpy as np
import pandas as pd


def to_monthly(series):
    """Serie (zilnică sau lunară) → serie lunară pe PeriodIndex, cu ultima observație din fiecare lună."""
    if series is None:
        return pd.Series(dtype="float64")
    s = pd.to_numeric(pd.Series(series), errors="coerce").dropna()
    if s.empty:
        return pd.Series(dtype="float64")
    s.index = pd.PeriodIndex([pd.Period(str(i)[:7], "M") for i in s.index])
    s = s[~s.index.duplicated(keep="last")]
    return s.sort_index().astype("float64")


def to_daily(series):
    """Serie zilnică (index de date sau „AAAA-LL-ZZ”) → DatetimeIndex fără fus orar, sortată, fără duplicate."""
    if series is None:
        return pd.Series(dtype="float64")
    s = pd.to_numeric(pd.Series(series), errors="coerce").dropna()
    if s.empty:
        return pd.Series(dtype="float64")
    idx = pd.DatetimeIndex(pd.to_datetime([str(i)[:10] for i in s.index]))
    s = pd.Series(s.values, index=idx.normalize())
    return s[~s.index.duplicated(keep="last")].sort_index().astype("float64")


def monthly_diff(a, b):
    """a − b pe lunile comune (puncte procentuale), serie lunară; goală dacă nu există lună comună."""
    both = pd.concat([to_monthly(a).rename("a"), to_monthly(b).rename("b")], axis=1, join="inner").dropna()
    return (both["a"] - both["b"]).astype("float64")


def daily_diff(a, b):
    """a − b pe zilele comune (puncte procentuale)."""
    both = pd.concat([to_daily(a).rename("a"), to_daily(b).rename("b")], axis=1, join="inner").dropna()
    return (both["a"] - both["b"]).astype("float64")


def yoy_monthly(level):
    """Inflația an/an (%) dintr-un indice lunar de preț: nivel_t / nivel_{t−12 luni} − 1, doar unde există exact luna de acum un an."""
    m = to_monthly(level)
    if m.empty:
        return m
    base = m.copy()
    base.index = base.index + 12
    both = pd.concat([m.rename("now"), base.rename("base")], axis=1, join="inner").dropna()
    both = both[both["base"] > 0]
    return ((both["now"] / both["base"] - 1.0) * 100.0).astype("float64")


def rate_stats(series):
    """Ultima valoare, perioada ei și variația în puncte procentuale față de acum 3 luni și acum un an.

    Serie lunară (PeriodIndex): comparația e cu exact luna de acum 3 / 12 luni (lipsă → None).
    Serie zilnică (DatetimeIndex): cu ultima observație de la sau înainte de aceeași zi de acum 3 / 12 luni
    (dacă seria nu acoperă perioada → None). Serie goală → toate None.
    """
    out = {"value": None, "period": None, "d3m": None, "d1y": None, "monthly": None}
    if series is None or len(series) == 0:
        return out
    s = pd.Series(series).dropna()
    if s.empty:
        return out
    out["value"], out["period"] = float(s.iloc[-1]), s.index[-1]
    if isinstance(s.index, pd.PeriodIndex):
        out["monthly"] = True
        for key, k in (("d3m", 3), ("d1y", 12)):
            prev = s.index[-1] - k
            if prev in s.index:
                out[key] = float(s.iloc[-1] - s.loc[prev])
    else:
        out["monthly"] = False
        for key, months in (("d3m", 3), ("d1y", 12)):
            cutoff = s.index[-1] - pd.DateOffset(months=months)
            if s.index[0] <= cutoff:
                out[key] = float(s.iloc[-1] - s[s.index <= cutoff].iloc[-1])
    return out


def period_label(period):
    """„AAAA-LL” → „LL.AAAA”; dată → „ZZ.LL.AAAA”; lipsă → None."""
    if period is None:
        return None
    if isinstance(period, pd.Period):
        return f"{period.month:02d}.{period.year}"
    try:
        return f"{pd.Timestamp(period):%d.%m.%Y}"
    except (TypeError, ValueError):
        return None


def curve_inversion_summary(spread, months=12):
    """Panta 10 ani − 3 luni (puncte procentuale, zilnic): ultima valoare și data, câte ședințe din ultimele
    `months` luni (fereastră calendaristică) au avut curba inversată (sub 0), din câte, și ultima zi inversată
    din tot istoricul. None pe serie goală."""
    s = to_daily(spread)
    if s.empty:
        return None
    cutoff = s.index[-1] - pd.DateOffset(months=months)
    win = s[s.index > cutoff]
    neg = s[s < 0]
    return {"value": float(s.iloc[-1]), "date": s.index[-1], "start": s.index[0],
            "n_window": int(len(win)), "inverted_window": int((win < 0).sum()),
            "last_inverted": None if neg.empty else neg.index[-1]}
