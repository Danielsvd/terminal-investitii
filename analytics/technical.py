"""Indicatori tehnici. Funcții pure: primesc serii/DataFrame-uri și întorc obiecte noi."""
import numpy as np
import pandas as pd


def rma(series, window):
    """Media netezită Wilder (RMA).

    Prima valoare este media simplă a primelor `window` observații valide,
    apoi: rma_t = (rma_{t-1} * (window - 1) + x_t) / window.
    Este netezirea folosită de RSI și ATR în TradingView și în platformele de brokeraj.
    """
    values = pd.Series(series, dtype="float64")
    out = pd.Series(np.nan, index=values.index, dtype="float64")
    valid = values.dropna()
    if window < 1 or len(valid) < window:
        return out
    prev = valid.iloc[:window].mean()
    out.loc[valid.index[window - 1]] = prev
    for idx, x in valid.iloc[window:].items():
        prev = (prev * (window - 1) + x) / window
        out.loc[idx] = prev
    return out


def true_range(df):
    """True Range = max(High-Low, |High-Close_prev|, |Low-Close_prev|)."""
    prev_close = df["Close"].shift()
    ranges = pd.concat(
        [df["High"] - df["Low"], (df["High"] - prev_close).abs(), (df["Low"] - prev_close).abs()],
        axis=1,
    )
    return ranges.max(axis=1, skipna=True)


def atr_wilder(df, window=14):
    """Average True Range cu netezire Wilder."""
    return rma(true_range(df), window)


def atr_trailing_stop(df, window=14, multiplier=2.5):
    """Stop-loss mobil pe ATR pentru o poziție long, cu resetare la atingere.

    Reguli (pe prețuri de închidere):
      - nivel brut = Close - multiplier * ATR;
      - cât timp Close rămâne peste stopul din ziua precedentă, stopul doar urcă;
      - când Close închide la sau sub stopul precedent, stopul este considerat atins
        (`ATR_Stop_Hit` = True) și pornește din nou de la nivelul brut al zilei.

    Fără resetare, un stop care doar urcă rămâne blocat deasupra prețului după
    prima corecție mai mare de `multiplier` x ATR și nu mai spune nimic.

    Întoarce o copie a lui `df` cu coloanele `ATR`, `ATR_Stop`, `ATR_Stop_Hit`,
    sau None dacă nu sunt destule date.
    """
    if df is None or len(df) < window + 1:
        return None

    out = df.copy()
    out["ATR"] = atr_wilder(out, window)
    raw = (out["Close"] - multiplier * out["ATR"]).to_numpy(dtype="float64")
    close = out["Close"].to_numpy(dtype="float64")

    stop = np.full(len(out), np.nan)
    hit = np.zeros(len(out), dtype=bool)
    prev = np.nan
    for i in range(len(out)):
        if np.isnan(raw[i]) or np.isnan(close[i]):
            stop[i] = prev
            continue
        if np.isnan(prev):
            prev = raw[i]
        elif close[i] <= prev:
            hit[i] = True
            prev = raw[i]
        else:
            prev = max(prev, raw[i])
        stop[i] = prev

    out["ATR_Stop"] = stop
    out["ATR_Stop_Hit"] = hit
    return out


def rsi_wilder(close, window=14):
    """RSI cu netezire Wilder (varianta standard, aceeași ca în TradingView/XTB)."""
    close = pd.Series(close, dtype="float64")
    delta = close.diff()
    avg_gain = rma(delta.clip(lower=0), window)
    avg_loss = rma(-delta.clip(upper=0), window)
    rs = avg_gain / avg_loss
    rsi = 100 - 100 / (1 + rs)
    # Fără nicio scădere în fereastră: RSI = 100 (rs = inf dă deja 100; 0/0 rămâne NaN)
    rsi = rsi.where(~((avg_loss == 0) & (avg_gain > 0)), 100.0)
    return rsi


def macd(close, fast=12, slow=26, signal=9):
    """MACD standard: EMA(fast) - EMA(slow), semnal EMA(signal). Toate cu adjust=False."""
    close = pd.Series(close, dtype="float64")
    line = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    sig = line.ewm(span=signal, adjust=False).mean()
    return line, sig
