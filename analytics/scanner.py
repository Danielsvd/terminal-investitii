"""Reguli pure pentru Scanner Volum (fără rețea, fără `st.*`)."""
import math

import numpy as np
import pandas as pd

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


def hours_key(symbol):
    """Bursa pentru orarul ședinței (cheile din `briefing.MARKET_HOURS`): .RO → BVB, fără sufix → NYSE, restul → XETRA
    (reper pentru Europa: Londra, Paris, Amsterdam au aceeași oră a României pentru deschidere și închidere)."""
    sym = str(symbol or "").upper()
    if sym.endswith(".RO"):
        return "BVB"
    return "XETRA" if "." in sym.replace("-", "") else "NYSE"


def rvol_session(df, is_partial, include_partial=False, window=RVOL_WINDOW):
    """`rvol_stats` pe ședința potrivită.

    is_partial(date) → True dacă ședința din acea zi e încă în desfășurare (volum parțial).
    - implicit (include_partial=False): dacă ultima bară e parțială, se evaluează ultima ședință ÎNCHEIATĂ
      (volumul parțial ÷ media pe zile întregi ar subestima RVOL-ul);
    - include_partial=True: se evaluează bara în curs, marcată cu partial=True.
    Rezultatul are în plus cheia `partial` (bool)."""
    rs = rvol_stats(df, window)
    if rs is None:
        return None
    partial = bool(is_partial(rs["date"]))
    if partial and not include_partial:
        rs = rvol_stats(df[df.index < rs["date"]], window)
        if rs is None:
            return None
        partial = False
    rs["partial"] = partial
    return rs


# Etichetele de stare (praguri fixe, NETESTATE istoric: nu s-a măsurat ce urmează după ele)
STRONG_RVOL = 2.0            # BREAKOUT / PANIC SELL: RVOL peste 2
STRONG_MOVE_PCT = 1.5        # ... și variația peste +1,5% / sub −1,5%
ACCUM_RVOL = 1.2             # ACUMULARE: RVOL peste 1,2 și variație pozitivă


def status_label(rvol, change_pct):
    """Eticheta descriptivă a ședinței după RVOL și variație, cu „(netestat)”. Lipsă → „N/A”."""
    r, c = _num(rvol), _num(change_pct)
    if r is None or c is None:
        return "N/A"
    if r > STRONG_RVOL and c > STRONG_MOVE_PCT:
        return "🚀 BREAKOUT (netestat)"
    if r > STRONG_RVOL and c < -STRONG_MOVE_PCT:
        return "⚠️ PANIC SELL (netestat)"
    if r > ACCUM_RVOL and c > 0:
        return "✅ ACUMULARE (netestat)"
    return "Normal"


# Modelul de anomalii (Isolation Forest): NEVALIDAT. `contamination` fixează pragul astfel încât ~5% din zilele de
# antrenare să fie marcate anomalii, prin construcție, indiferent de piață.
ANOMALY_CONTAMINATION = 0.05
ANOMALY_MIN_TRAIN = 20


def anomaly_features(df):
    """Trăsăturile pe zi: randament (%), volum, volatilitatea randamentului pe 5 zile. Doar zilele complete."""
    if df is None or len(df) == 0 or not {"Close", "Volume"} <= set(df.columns):
        return pd.DataFrame(columns=["Return", "Volume", "Volatility"])
    close = df["Close"].astype("float64")
    out = pd.DataFrame({"Return": close.pct_change(fill_method=None) * 100.0, "Volume": df["Volume"].astype("float64")})
    out["Volatility"] = out["Return"].rolling(window=5).std()
    return out.replace([np.inf, -np.inf], np.nan).dropna()


def volume_anomaly(df, contamination=ANOMALY_CONTAMINATION, min_train=ANOMALY_MIN_TRAIN, random_state=42):
    """True/False: ultima zi din `df` e anomalie față de zilele DINAINTEA ei (modelul nu vede ziua evaluată la
    antrenare). Sub `min_train` zile de antrenare → None (fără verdict, nu False)."""
    from sklearn.ensemble import IsolationForest

    feats = anomaly_features(df)
    if len(feats) < min_train + 1:
        return None
    train, last = feats.iloc[:-1], feats.iloc[[-1]]
    model = IsolationForest(contamination=contamination, random_state=random_state).fit(train.values)
    return bool(model.predict(last.values)[0] == -1)
