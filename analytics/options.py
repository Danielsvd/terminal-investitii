"""Calcule pe lanțul de opțiuni. Funcții pure."""
import numpy as np
import pandas as pd


def max_pain(calls, puts):
    """Prețul Max Pain pentru o singură expirare.

    Pentru fiecare strike K candidat se calculează valoarea totală pe care
    opțiunile ar avea-o la expirare dacă prețul ar închide la K:
        sum(OI_call * max(K - strike, 0)) + sum(OI_put * max(strike - K, 0))
    Max Pain este strike-ul la care această sumă este minimă (cumpărătorii de
    opțiuni pierd cel mai mult).

    `calls` și `puts` sunt DataFrame-uri cu coloanele `strike` și `openInterest`.
    Întoarce None dacă nu există open interest.
    """
    def _clean(df):
        if df is None or len(df) == 0:
            return np.array([]), np.array([])
        strikes = pd.to_numeric(df["strike"], errors="coerce")
        oi = pd.to_numeric(df["openInterest"], errors="coerce").fillna(0)
        mask = strikes.notna()
        return strikes[mask].to_numpy(dtype="float64"), oi[mask].to_numpy(dtype="float64")

    c_strike, c_oi = _clean(calls)
    p_strike, p_oi = _clean(puts)
    if c_oi.sum() + p_oi.sum() <= 0:
        return None

    candidates = np.unique(np.concatenate([c_strike, p_strike]))
    best_k, best_pain = None, None
    for k in candidates:
        pain = (c_oi * np.maximum(k - c_strike, 0)).sum() + (p_oi * np.maximum(p_strike - k, 0)).sum()
        if best_pain is None or pain < best_pain:
            best_k, best_pain = float(k), pain
    return best_k
