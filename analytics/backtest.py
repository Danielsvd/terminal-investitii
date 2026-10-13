"""Testul istoric al sub-scorului tehnic. Funcții pure (fără rețea, fără st.*).

Pentru fiecare simbol și fiecare sfârșit de lună se calculează scorul doar din datele
de până la acea dată, apoi randamentul din următoarele 1, 3, 6, 12 luni. Rezultatele se
grupează pe trepte de scor, cu randamentul în exces față de media universului la aceeași
dată (exces transversal: nu depinde de un indice de referință și elimină mișcarea pieței).
"""
import numpy as np
import pandas as pd

from analytics.scoring import subscore_from_components, technical_components
from analytics.technical import rsi_wilder

HORIZONS = (1, 3, 6, 12)
SCORE_BINS = (0, 20, 40, 60, 80, 100.0001)
MIN_SYMBOLS_PER_DATE = 5     # sub acest număr, media transversală a datei nu e folosită
DIVIDEND_JUMP = 0.01         # o schimbare > 1% a raportului ajustat/neajustat = eveniment (dividend, split)


def month_end_dates(index, start, end):
    """Ultima ședință din fiecare lună, între `start` și `end` (inclusiv)."""
    idx = pd.DatetimeIndex(index).sort_values()
    idx = idx[(idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))]
    if len(idx) == 0:
        return []
    s = pd.Series(idx, index=idx)
    return list(s.groupby([idx.year, idx.month]).max())


def price_at(close, date):
    """Ultima închidere la sau înainte de `date`; None dacă nu există."""
    part = close.loc[:pd.Timestamp(date)].dropna()
    return None if part.empty else float(part.iloc[-1])


def forward_return(close, date, months):
    """Randamentul de la `date` până la `date + months` luni calendaristice.

    None dacă orizontul depășește ultima dată din serie sau prețul de pornire nu e pozitiv.
    """
    date = pd.Timestamp(date)
    end = date + pd.DateOffset(months=months)
    if close.dropna().empty or end > close.dropna().index[-1]:
        return None
    p0, p1 = price_at(close, date), price_at(close, end)
    if p0 is None or p1 is None or not p0 > 0:
        return None
    return p1 / p0 - 1


def dividend_flags(close_adj, close_raw, window=20):
    """Seria booleană: a existat un eveniment de ajustare (dividend, split) în ultimele `window` ședințe.

    Evenimentul se vede ca schimbare a raportului ajustat/neajustat cu peste DIVIDEND_JUMP.
    """
    both = pd.concat([close_adj, close_raw], axis=1, keys=["adj", "raw"]).dropna()
    if both.empty:
        return pd.Series(dtype=bool)
    ratio = both["adj"] / both["raw"]
    event = ratio.pct_change(fill_method=None).abs() > DIVIDEND_JUMP
    return event.rolling(window, min_periods=1).max().astype(bool)


def score_rows(symbol, close_adj, dates, close_raw=None, volume=None, horizons=HORIZONS):
    """Rândurile testului pentru un simbol: scor la fiecare dată + randamente viitoare.

    `close_adj` (obligatoriu): închideri ajustate, din care se calculează scorul și randamentele.
    `close_raw` (opțional): închideri neajustate; dă scorul „neajustat" și semnalul de dividend.
    `volume` (opțional): pentru indicatorii de lichiditate (ședințe fără volum).
    """
    close_adj = pd.Series(close_adj, dtype="float64").dropna().sort_index()
    if close_adj.empty:
        return []
    rsi_adj = rsi_wilder(close_adj, 14)
    raw = None if close_raw is None else pd.Series(close_raw, dtype="float64").dropna().sort_index()
    rsi_raw = None if raw is None or raw.empty else rsi_wilder(raw, 14)
    divs = dividend_flags(close_adj, raw) if raw is not None and not raw.empty else None
    vol = None if volume is None else pd.Series(volume, dtype="float64").sort_index()

    rows = []
    for d in dates:
        d = pd.Timestamp(d)
        hist = close_adj.loc[:d]
        if hist.empty or hist.index[-1] < d - pd.DateOffset(days=7):
            continue                                  # fără preț în ultima săptămână: simbol inactiv
        comp = technical_components(hist, rsi_adj.loc[:d].iloc[-1])
        row = {
            "symbol": symbol, "date": d.date().isoformat(), "close": float(hist.iloc[-1]),
            "tech_score": subscore_from_components(comp),
            "trend": comp["trend"], "momentum": comp["momentum"], "rsi_c": comp["rsi"],
            "mom_raw": comp["raw"]["momentum_12_1"], "rsi_raw": comp["raw"]["rsi"],
            "sessions_12m": int(hist.loc[d - pd.DateOffset(months=12):].shape[0]),
        }
        if raw is not None and not raw.empty and not raw.loc[:d].empty:
            comp_r = technical_components(raw.loc[:d], rsi_raw.loc[:d].iloc[-1])
            row["tech_score_unadj"] = subscore_from_components(comp_r)
            row["rsi_unadj"] = comp_r["raw"]["rsi"]
            flag = divs.loc[:d]
            row["dividend_20d"] = bool(flag.iloc[-1]) if not flag.empty else None
        if vol is not None:
            last3 = vol.loc[d - pd.DateOffset(months=3):d]
            row["zero_vol_frac_3m"] = float((last3.fillna(0) <= 0).mean()) if len(last3) else None
        for h in horizons:
            row[f"fwd_{h}m"] = forward_return(close_adj, d, h)
        rows.append(row)
    return rows


def add_excess(df, ret_col, min_symbols=MIN_SYMBOLS_PER_DATE):
    """Coloana `<ret_col>_ex`: randamentul minus media universului la aceeași dată.

    Datele cu mai puțin de `min_symbols` randamente valide primesc NaN.
    """
    out = df.copy()
    r = pd.to_numeric(out[ret_col], errors="coerce")
    grp = r.groupby(out["date"])
    mean, count = grp.transform("mean"), grp.transform("count")
    out[f"{ret_col}_ex"] = (r - mean).where(count >= min_symbols)
    return out


def bucket_summary(df, score_col, ret_col, bins=SCORE_BINS, min_symbols=MIN_SYMBOLS_PER_DATE):
    """Tabelul pe trepte de scor: N, exces mediu, exces median, % cazuri cu exces pozitiv, nr. de date.

    Excesul e față de media universului la aceeași dată (vezi `add_excess`).
    """
    data = add_excess(df, ret_col, min_symbols)
    ex = f"{ret_col}_ex"
    data = data[pd.to_numeric(data[score_col], errors="coerce").notna() & data[ex].notna()].copy()
    labels = [f"{int(bins[i])}–{int(round(bins[i + 1]))}" for i in range(len(bins) - 1)]
    data["treapta"] = pd.cut(pd.to_numeric(data[score_col]), bins=list(bins), labels=labels,
                             right=False, include_lowest=True)
    g = data.groupby("treapta", observed=False)[ex]
    table = pd.DataFrame({
        "N": g.count(),
        "exces_mediu": g.mean(),
        "exces_median": g.median(),
        "pct_pozitiv": g.apply(lambda s: float((s > 0).mean()) if len(s) else np.nan),
        "nr_date": data.groupby("treapta", observed=False)["date"].nunique(),
    })
    table["N"] = table["N"].astype(int)
    table["nr_date"] = table["nr_date"].fillna(0).astype(int)
    return table


def rank_ic(df, score_col, ret_col, min_symbols=MIN_SYMBOLS_PER_DATE):
    """Corelația de rang (Spearman) între scor și randament, calculată pe fiecare dată, apoi mediată.

    Întoarce dict: ic_mediu, ic_std, nr_date, t (= ic_mediu / ic_std × √nr_date), pct_pozitiv.
    Lunile se suprapun la orizonturi > 1 lună, deci t-ul e optimist (observații dependente).
    """
    data = df[["date", score_col, ret_col]].copy()
    data[score_col] = pd.to_numeric(data[score_col], errors="coerce")
    data[ret_col] = pd.to_numeric(data[ret_col], errors="coerce")
    data = data.dropna()
    ics = []
    for _, part in data.groupby("date"):
        if len(part) < min_symbols or part[score_col].nunique() < 2 or part[ret_col].nunique() < 2:
            continue
        ics.append(part[score_col].rank().corr(part[ret_col].rank()))
    ics = pd.Series(ics, dtype="float64").dropna()
    n = int(len(ics))
    if n < 2:
        return {"ic_mediu": None, "ic_std": None, "nr_date": n, "t": None, "pct_pozitiv": None}
    mean, std = float(ics.mean()), float(ics.std(ddof=1))
    return {"ic_mediu": mean, "ic_std": std, "nr_date": n,
            "t": None if std == 0 else mean / std * np.sqrt(n), "pct_pozitiv": float((ics > 0).mean())}
