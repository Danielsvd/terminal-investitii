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
    arr = values.to_numpy()
    out = np.full(len(arr), np.nan)
    valid_pos = np.flatnonzero(~np.isnan(arr))
    if window < 1 or len(valid_pos) < window:
        return pd.Series(out, index=values.index)
    prev = arr[valid_pos[:window]].mean()
    out[valid_pos[window - 1]] = prev
    for pos in valid_pos[window:]:
        prev = (prev * (window - 1) + arr[pos]) / window
        out[pos] = prev
    return pd.Series(out, index=values.index)


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


def price_changes(window_close, full_close):
    """Cardurile „Interval” și „Evoluție Azi” din Analiză Companie.

    `window_close` = închiderile din intervalul ales; `full_close` = tot istoricul (pentru ședința precedentă).
    Întoarce dict: last (ultima închidere), diff / diff_pct (față de prima închidere din interval),
    day / day_pct (față de închiderea precedentă). Orice valoare care nu se poate calcula e None
    (interval gol, o singură ședință, preț lipsă sau ≤ 0): niciodată 0, care ar arăta „0,00 (0,00%)”.
    """
    out = dict.fromkeys(("last", "diff", "diff_pct", "day", "day_pct"))
    win = pd.to_numeric(pd.Series(window_close, dtype="float64"), errors="coerce").dropna()
    full = pd.to_numeric(pd.Series(full_close, dtype="float64"), errors="coerce").dropna()
    if win.empty:
        return out
    last = float(win.iloc[-1])
    out["last"] = last
    start = float(win.iloc[0])
    if len(win) >= 2 and start > 0:
        out["diff"] = last - start
        out["diff_pct"] = (last - start) / start * 100.0
    if len(full) >= 2:
        prev = float(full.iloc[-2])
        if prev > 0:
            out["day"] = last - prev
            out["day_pct"] = (last - prev) / prev * 100.0
    return out


def regime_hit_rates(fwd_returns, clusters, current):
    """Rata de reușită a regimului curent față de rata de bază (toate zilele).

    `fwd_returns`: randamentul următoarelor 20 de ședințe pentru fiecare zi (NaN la final);
    `clusters`: eticheta regimului pe aceeași axă; `current`: regimul de azi.
    Întoarce {regime_rate, regime_mean, regime_n, base_rate, base_mean, base_n, edge_pp} (rate în %),
    sau None dacă regimul curent nu are zile cu randament viitor cunoscut.
    Atenție: rezultatul e în eșantion (regimurile sunt învățate pe tot istoricul) și ferestrele de
    20 de zile se suprapun, deci N nu înseamnă N observații independente.
    """
    df = pd.DataFrame({"fwd": pd.Series(fwd_returns, dtype="float64").to_numpy(),
                       "cl": pd.Series(clusters).to_numpy()}).dropna(subset=["fwd"])
    reg = df.loc[df["cl"] == current, "fwd"]
    if reg.empty:
        return None
    base = df["fwd"]
    regime_rate, base_rate = float((reg > 0).mean() * 100), float((base > 0).mean() * 100)
    return {"regime_rate": regime_rate, "regime_mean": float(reg.mean()), "regime_n": int(len(reg)),
            "base_rate": base_rate, "base_mean": float(base.mean()), "base_n": int(len(base)),
            "edge_pp": regime_rate - base_rate}
