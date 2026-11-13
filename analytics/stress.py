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


# --- Starea zilei: VIX și Fear & Greed (CNN) ---------------------------------------------------------

VIX_BANDS = ((15.0, "calm"), (20.0, "normal"), (30.0, "tensiune"))   # peste 30: frică
FG_BANDS = ((25.0, "frică extremă"), (45.0, "frică"), (55.0, "neutru"), (75.0, "lăcomie"))   # peste 75: lăcomie extremă
FG_RATINGS = {"extreme fear": "frică extremă", "fear": "frică", "neutral": "neutru", "greed": "lăcomie",
              "extreme greed": "lăcomie extremă"}


def vix_state(vix):
    """Eticheta VIX: sub 15 calm, 15–20 normal, 20–30 tensiune, peste 30 frică; None la lipsă."""
    if vix is None or not np.isfinite(vix):
        return None
    for limit, label in VIX_BANDS:
        if vix < limit:
            return label
    return "frică"


def fg_state(score):
    """Eticheta Fear & Greed după benzile CNN (0–25, 25–45, 45–55, 55–75, 75–100); None la lipsă."""
    if score is None or not np.isfinite(score):
        return None
    for limit, label in FG_BANDS:
        if score < limit:
            return label
    return "lăcomie extremă"


def parse_cnn_fear_greed(payload):
    """Răspunsul JSON CNN (`fear_and_greed`: score, rating, timestamp, previous_close, previous_1_week)
    -> dict {score, label, date, prev_close, prev_week} sau None dacă lipsește scorul.

    Eticheta vine din scor (benzile CNN), nu din textul lor, ca să fie mereu în română și consecventă."""
    fg = (payload or {}).get("fear_and_greed") if isinstance(payload, dict) else None
    if not isinstance(fg, dict):
        return None

    def _val(key):
        try:
            v = float(fg.get(key))
        except (TypeError, ValueError):
            return None
        return v if np.isfinite(v) and 0 <= v <= 100 else None

    score = _val("score")
    if score is None:
        return None
    date = pd.to_datetime(fg.get("timestamp"), utc=True, errors="coerce")
    return {"score": score, "label": fg_state(score), "date": None if pd.isna(date) else date,
            "prev_close": _val("previous_close"), "prev_week": _val("previous_1_week")}
