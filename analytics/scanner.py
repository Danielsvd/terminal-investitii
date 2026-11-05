"""Reguli pure pentru Scanner Volum (fără rețea, fără `st.*`)."""
import math

from analytics.briefing import MIN_VALUE_RON

RVOL_DEFAULT = 1.2           # pragul RVOL implicit al cursorului (Daniel, 05.11.2026: de la 1,5 la 1,2)
MIN_AVG_SHARES = 5000        # SUA / Europa: media volumului pe 20 de ședințe, în acțiuni (regula existentă)


def _num(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def liquidity_ok(symbol, last_close, last_volume, avg_volume_20):
    """True dacă titlul trece filtrul de lichiditate al Scanner-ului.

    BVB (`.RO`): valoarea tranzacționată în ultima ședință (preț × volum) ≥ MIN_VALUE_RON (50.000 lei), aceeași regulă
    ca topurile din Rezumatul Zilei. Restul: media volumului pe 20 de ședințe ≥ MIN_AVG_SHARES acțiuni.
    Valoare lipsă → False (titlul nu se afișează, nu primește o valoare inventată).
    """
    if str(symbol).upper().endswith(".RO"):
        p, v = _num(last_close), _num(last_volume)
        return p is not None and v is not None and p * v >= MIN_VALUE_RON
    a = _num(avg_volume_20)
    return a is not None and a >= MIN_AVG_SHARES
