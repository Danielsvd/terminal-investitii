"""Indicatori de stres al pieței: structura VIX și spread-urile de credit. Funcții pure.

Pragurile sunt convenții uzuale de piață, netestate în aplicație.
"""
import numpy as np
import pandas as pd

from analytics.rates import to_daily

VIX_TERM_STRESS = 1.0      # VIX ≥ VIX3M: frica pe termen scurt depășește pe cea pe 3 luni (backwardation)
VIX_TERM_TENSE = 0.9       # 0,90–1,00: tensiune; sub 0,90: structură normală (contango)


def vix_term(vix, vix3m):
    """Raportul VIX / VIX3M pe ultima zi comună: dict ratio, date, vix, vix3m, state ('stres'/'tensiune'/'normal'),
    plus seria raportului. None dacă nu există zi comună."""
    both = pd.concat([to_daily(vix).rename("v"), to_daily(vix3m).rename("v3")], axis=1, join="inner").dropna()
    both = both[(both["v"] > 0) & (both["v3"] > 0)]
    if both.empty:
        return None
    ratio = both["v"] / both["v3"]
    r = float(ratio.iloc[-1])
    state = "stres" if r >= VIX_TERM_STRESS else "tensiune" if r >= VIX_TERM_TENSE else "normal"
    return {"ratio": r, "date": ratio.index[-1], "vix": float(both["v"].iloc[-1]), "vix3m": float(both["v3"].iloc[-1]),
            "state": state, "series": ratio}


def percentile_rank(series, value=None):
    """Procentul de observații din serie mai mici sau egale cu `value` (implicit ultima valoare), 0–100;
    None pe serie goală."""
    s = pd.to_numeric(pd.Series(series), errors="coerce").dropna() if series is not None else pd.Series(dtype=float)
    if s.empty:
        return None
    v = float(s.iloc[-1]) if value is None else float(value)
    return float((s <= v).mean() * 100.0)


def spread_stats(series):
    """Spread de credit (procente): ultima valoare, data, Δ 1 lună și Δ 1 an (puncte procentuale, față de ultima
    observație de la sau înainte de aceeași zi), percentila în istoricul disponibil și câte zile are istoricul."""
    s = to_daily(series)
    out = {"value": None, "date": None, "d1m": None, "d1y": None, "pctile": None, "n": int(len(s)), "start": None}
    if s.empty:
        return out
    out["value"], out["date"], out["start"] = float(s.iloc[-1]), s.index[-1], s.index[0]
    for key, months in (("d1m", 1), ("d1y", 12)):
        cutoff = s.index[-1] - pd.DateOffset(months=months)
        if s.index[0] <= cutoff:
            out[key] = float(s.iloc[-1] - s[s.index <= cutoff].iloc[-1])
    out["pctile"] = percentile_rank(s)
    return out
