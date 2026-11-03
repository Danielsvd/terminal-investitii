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


# --- Curba randamentelor pe piața acțiunii -------------------------------------

EUROZONE_SUFFIXES = (".DE", ".F", ".PA", ".AS", ".MI", ".MC", ".BR", ".LS", ".HE", ".VI", ".IR", ".AT")


def market_of(symbol, currency=None):
    """Piața pentru curba randamentelor: "RO" (.RO), "EU" (bursă din zona euro sau acțiune în EUR),
    "US" (acțiune în USD fără sufix de bursă străină, inclusiv ADR-urile din SUA) sau None (altă piață:
    Londra, Elveția etc., pentru care nu există încă o curbă în aplicație)."""
    sym = str(symbol or "").upper()
    cur = str(currency or "").upper()
    if sym.endswith(".RO"):
        return "RO"
    if sym.endswith(EUROZONE_SUFFIXES) or cur == "EUR":
        return "EU"
    if cur == "USD" and "." not in sym.replace("-", ""):
        return "US"
    return None


def monthly_spread(long_rate, short_rate):
    """Dobânda lungă − dobânda scurtă pe ultima lună prezentă în ambele serii lunare (procente, indexate
    pe perioadă „AAAA-LL” sau pe dată). Nu amestecă luni diferite. {"spread", "long", "short", "month"}
    (month = "AAAA-LL") sau None dacă nu există nicio lună comună."""
    def _monthly(series):
        s = pd.to_numeric(pd.Series(series), errors="coerce").dropna()
        if s.empty:
            return s
        s.index = pd.PeriodIndex([pd.Period(str(i)[:7], "M") for i in s.index])
        return s[~s.index.duplicated(keep="last")]

    if long_rate is None or short_rate is None:
        return None
    both = pd.concat([_monthly(long_rate).rename("l"), _monthly(short_rate).rename("s")], axis=1, join="inner").dropna()
    if both.empty:
        return None
    month = both.index.max()
    lv, sv = float(both.loc[month, "l"]), float(both.loc[month, "s"])
    return {"spread": lv - sv, "long": lv, "short": sv, "month": str(month)}


def same_day_spread(long_close, short_close):
    """10 ani − 3 luni din închideri zilnice (ex. ^TNX, ^IRX, în procente), pe ultima zi prezentă în ambele.
    {"spread", "long", "short", "date"} sau None."""
    if long_close is None or short_close is None:
        return None
    def _clean(s):
        s = pd.to_numeric(pd.Series(s), errors="coerce").dropna()
        idx = pd.DatetimeIndex(s.index)
        s.index = (idx.tz_localize(None) if idx.tz is not None else idx).normalize()
        return s[~s.index.duplicated(keep="last")]
    both = pd.concat([_clean(long_close).rename("l"), _clean(short_close).rename("s")], axis=1, join="inner").dropna()
    if both.empty:
        return None
    d = both.index.max()
    lv, sv = float(both.loc[d, "l"]), float(both.loc[d, "s"])
    return {"spread": lv - sv, "long": lv, "short": sv, "date": d}
