"""Reguli pure pentru Scanner Volum (fără rețea, fără `st.*`)."""
import math

import numpy as np

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


RVOL_WINDOW = 20             # media volumului: ultimele 20 de ședințe cu volum, fără ședința evaluată


def rvol_stats(df, window=RVOL_WINDOW):
    """Volumul relativ al ultimei ședințe dintr-un DataFrame zilnic cu coloanele `Close` și `Volume`.

    - Prețul și volumul se aliniază pe aceeași zi: se păstrează doar zilele cu ambele valori, preț > 0, volum ≥ 0
      (înainte se curățau separat, deci prețul de ieri și volumul de ieri puteau veni din zile diferite).
    - Media volumului: ultimele `window` ședințe CU VOLUM (> 0) dinaintea ședinței evaluate. Zilele cu volum 0
      (BVB, titluri fără tranzacții) nu trag media în jos.
    - Variația: ultima închidere față de închiderea din ziua aliniată anterioară.
    Întoarce dict {date, close, prev_close, change_pct, volume, avg_volume, rvol, n_avg} sau None dacă nu sunt
    `window` ședințe cu volum pentru medie (fără valoare inventată)."""
    if df is None or len(df) == 0 or not {"Close", "Volume"} <= set(df.columns):
        return None
    d = df[["Close", "Volume"]].apply(lambda s: s.astype("float64"))
    d = d[np.isfinite(d["Close"]) & np.isfinite(d["Volume"]) & (d["Close"] > 0) & (d["Volume"] >= 0)]
    if len(d) < 2:
        return None
    last, prev = d.iloc[-1], d.iloc[-2]
    hist = d["Volume"].iloc[:-1]
    hist = hist[hist > 0].iloc[-window:]
    if len(hist) < window:
        return None
    avg = float(hist.mean())
    vol = float(last["Volume"])
    return {"date": d.index[-1], "close": float(last["Close"]), "prev_close": float(prev["Close"]),
            "change_pct": (float(last["Close"]) / float(prev["Close"]) - 1.0) * 100.0,
            "volume": vol, "avg_volume": avg, "rvol": vol / avg, "n_avg": len(hist)}
