"""Calcule macroeconomice. Funcții pure."""
import pandas as pd


def yoy_pct(series, max_gap_days=45):
    """Variația procentuală an/an a unei serii lunare (ex. indicele CPI).

    Compară ultima observație cu observația de acum 12 luni. Întoarce None dacă
    seria nu acoperă 12 luni sau dacă observația de referință e mai veche cu
    peste `max_gap_days` față de data-țintă.

    Inflația se calculează din NIVELUL indicelui: CPI_t / CPI_{t-12} - 1.
    Nivelul indicelui (ex. 320) nu este o rată a inflației.
    """
    if series is None:
        return None
    s = pd.Series(series).dropna()
    if len(s) < 2:
        return None
    last_date = s.index[-1]
    target = last_date - pd.DateOffset(years=1)
    base = s[s.index <= target]
    if base.empty:
        return None
    base_date = base.index[-1]
    if (target - base_date).days > max_gap_days:
        return None
    base_val = float(base.iloc[-1])
    if base_val == 0:
        return None
    return (float(s.iloc[-1]) / base_val - 1) * 100


def real_rate(policy_rate_pct, inflation_yoy_pct):
    """Rata reală a dobânzii = dobânda de politică monetară - inflația an/an (ambele în %)."""
    if policy_rate_pct is None or inflation_yoy_pct is None:
        return None
    return float(policy_rate_pct) - float(inflation_yoy_pct)
