"""Testul istoric al scorului SUA pe date SEC (Faza 3c): funcții pure (fără rețea, fără Streamlit).

Universuri (decizia D9): C = membrii S&P 500 la fiecare dată (reconstituiți din `fja05680/sp500`), B = membrii de azi
(ultima dată din aceeași sursă), aplicați pe toată perioada. Construire 2012–2019, verificare 2020–2026 o singură dată
(D10, criteriile din `claude/propunere_faza3_xbrl_2026-11-27.md` §10.3).

Aici: intervalele de apartenență, protecția la simboluri refolosite, randamentul trunchiat (delistări), excesul față
de mediană (univers / sector), IC lunar cu t Newey-West, cincimile pe ani, benzile, scorul pe rang în sector.
"""
import math

import numpy as np
import pandas as pd

from analytics.backtest import price_at

MIN_SYMBOLS_PER_DATE = 5          # sub atât, mediana / IC-ul datei nu se folosesc (ca în `analytics.backtest`)
NW_LAGS = 12                      # Newey-West: randamentele la 12 luni se suprapun lună de lună
DELIST_GAP = pd.Timedelta(days=10)   # seria unui simbol oprită cu > 10 zile înaintea datelor = delistat
PRICE_START_TOL = pd.Timedelta(days=31)
VERDICT_BINS = (0, 40, 70, 100.0001)
VERDICT_LABELS = ("NEFAVORABIL", "NEUTRU", "FAVORABIL")


# --- Universul -----------------------------------------------------------------

def yahoo_symbol(sym):
    """Simbolul din lista S&P (BRK.B) în forma Yahoo / SEC (BRK-B)."""
    return str(sym).strip().upper().replace(".", "-")


def intervals_from_snapshots(snapshots, start=None):
    """Intervalele de apartenență [de_la, pana_la) din instantaneele `date, tickers` (o listă separată prin virgulă
    la fiecare schimbare). `pana_la` = NaT pentru membrii de la ultima dată. Cu `start`: doar intervalele care se
    termină după `start` (cele încheiate înainte nu contează pentru test)."""
    snaps = snapshots.copy()
    snaps["date"] = pd.to_datetime(snaps["date"])
    snaps = snaps.sort_values("date")
    open_ = {}
    rows = []
    for d, tickers in zip(snaps["date"], snaps["tickers"]):
        now = {t.strip() for t in str(tickers).split(",") if t.strip()}
        for t in list(open_):
            if t not in now:
                rows.append((t, open_.pop(t), d))
        for t in now:
            open_.setdefault(t, d)
    for t, d0 in open_.items():
        rows.append((t, d0, pd.NaT))
    out = pd.DataFrame(rows, columns=["simbol", "de_la", "pana_la"])
    if start is not None:
        out = out[out["pana_la"].isna() | (out["pana_la"] > pd.Timestamp(start))]
    return out.sort_values(["simbol", "de_la"]).reset_index(drop=True)


def members_at(intervals, when):
    """Simbolurile membre la data `when`: de_la ≤ when < pana_la."""
    w = pd.Timestamp(when)
    m = (intervals["de_la"] <= w) & (intervals["pana_la"].isna() | (intervals["pana_la"] > w))
    return set(intervals.loc[m, "simbol"])


def current_members(intervals):
    """Membrii de la ultima dată a sursei (universul B)."""
    return set(intervals.loc[intervals["pana_la"].isna(), "simbol"])


def accept_mapping(intervals_sym, tenk_filed_dates, first_price_date, test_start):
    """Protecția la simboluri refolosite: perechea simbol istoric → CIK / prețuri Yahoo se acceptă doar dacă
    (a) CIK-ul are cel puțin un 10-K depus într-un interval de apartenență (în perioada testului) și
    (b) istoricul Yahoo începe cel târziu la max(intrarea în indice, începutul testului) + 31 de zile.
    Întoarce (acceptat, motiv)."""
    if first_price_date is None:
        return False, "fără prețuri Yahoo"
    t0 = pd.Timestamp(test_start)
    filed = [pd.Timestamp(d) for d in tenk_filed_dates]
    has_10k, price_ok = False, False
    for _, r in intervals_sym.iterrows():
        lo = max(pd.Timestamp(r["de_la"]), t0)
        hi = None if pd.isna(r["pana_la"]) else pd.Timestamp(r["pana_la"])
        if hi is not None and hi <= lo:
            continue
        # 10-K depus în interval sau în anul de după ieșire (raportul pe ultimul an de apartenență)
        hi_10k = None if hi is None else hi + pd.DateOffset(years=1)
        if any(lo <= f and (hi_10k is None or f <= hi_10k) for f in filed):
            has_10k = True
        if pd.Timestamp(first_price_date) <= lo + PRICE_START_TOL:
            price_ok = True
    if not has_10k:
        return False, "CIK fără 10-K în perioada de apartenență"
    if not price_ok:
        return False, "istoricul Yahoo începe după intrarea în indice (simbol posibil refolosit)"
    return True, None


# --- Randamente --------------------------------------------------------------

def forward_return_at(close, date, months, horizon_limit, data_end):
    """(randament, trunchiat). Randamentul de la `date` la `date + months`, doar dacă acel capăt ≤ `horizon_limit`
    (sfârșitul fazei sau ultima dată cu prețuri). Dacă seria simbolului s-a oprit înainte de capăt și cu peste
    `DELIST_GAP` înaintea `data_end` (delistare / preluare), randamentul se ia până la ultimul preț și se marchează
    trunchiat. (None, False) altfel."""
    date = pd.Timestamp(date)
    end = date + pd.DateOffset(months=months)
    if end > pd.Timestamp(horizon_limit):
        return None, False
    s = close.dropna()
    if s.empty:
        return None, False
    p0 = price_at(s, date)
    if p0 is None or not p0 > 0:
        return None, False
    last = s.index[-1]
    if end <= last:
        p1 = price_at(s, end)
        return (None, False) if p1 is None else (p1 / p0 - 1, False)
    if last > date and last < pd.Timestamp(data_end) - DELIST_GAP:
        return float(s.iloc[-1]) / p0 - 1, True
    return None, False


def add_median_excess(df, ret_col, by=("date",), min_n=MIN_SYMBOLS_PER_DATE, suffix="_ex"):
    """Coloana `<ret_col><suffix>` = randamentul − mediana grupului (`by`: data, sau data + sector).
    Grupurile cu mai puțin de `min_n` randamente primesc NaN."""
    out = df.copy()
    r = pd.to_numeric(out[ret_col], errors="coerce")
    keys = [out[k] for k in by]
    grp = r.groupby(keys)
    med, cnt = grp.transform("median"), grp.transform("count")
    out[f"{ret_col}{suffix}"] = (r - med).where(cnt >= min_n)
    return out


# --- Statistici --------------------------------------------------------------

def newey_west_t(x, lags=NW_LAGS):
    """t-ul mediei unei serii (ex. IC lunar), cu eroarea standard Newey-West (nucleu Bartlett, `lags` decalaje).
    None sub 3 observații sau varianță zero."""
    v = np.asarray(pd.Series(x, dtype="float64").dropna(), dtype=float)
    n = len(v)
    if n < 3:
        return None
    e = v - v.mean()
    s = float(e @ e) / n
    for k in range(1, min(lags, n - 1) + 1):
        w = 1.0 - k / (lags + 1.0)
        s += 2.0 * w * float(e[k:] @ e[:-k]) / n
    if not s > 0:
        return None
    return float(v.mean() / math.sqrt(s / n))


def ic_series(df, score_col, ret_col, min_n=MIN_SYMBOLS_PER_DATE):
    """IC-ul de rang (Spearman) pe fiecare dată: Series indexată pe dată."""
    data = df[["date", score_col, ret_col]].copy()
    data[score_col] = pd.to_numeric(data[score_col], errors="coerce")
    data[ret_col] = pd.to_numeric(data[ret_col], errors="coerce")
    data = data.dropna()
    out = {}
    for d, part in data.groupby("date"):
        if len(part) < min_n or part[score_col].nunique() < 2 or part[ret_col].nunique() < 2:
            continue
        out[d] = part[score_col].rank().corr(part[ret_col].rank())
    return pd.Series(out, dtype="float64").dropna().sort_index()


def ic_summary(df, score_col, ret_col, lags=NW_LAGS, min_n=MIN_SYMBOLS_PER_DATE):
    """{ic_mediu, t_nw, nr_date, pct_pozitiv} pe IC-ul lunar."""
    ics = ic_series(df, score_col, ret_col, min_n)
    n = int(len(ics))
    if n == 0:
        return {"ic_mediu": None, "t_nw": None, "nr_date": 0, "pct_pozitiv": None}
    return {"ic_mediu": float(ics.mean()), "t_nw": newey_west_t(ics, lags), "nr_date": n,
            "pct_pozitiv": float((ics > 0).mean())}


def quintile_spread_by_year(df, score_col, ex_col, min_n=10):
    """Pe fiecare dată: media excesului în cincimea de sus minus cea de jos (cincimi pe rangul scorului); apoi media
    pe an calendaristic. DataFrame: an, spread_mediu, nr_date, sus_bate_jos (spread_mediu > 0)."""
    data = df[["date", score_col, ex_col]].copy()
    data[score_col] = pd.to_numeric(data[score_col], errors="coerce")
    data[ex_col] = pd.to_numeric(data[ex_col], errors="coerce")
    data = data.dropna()
    spreads = {}
    for d, part in data.groupby("date"):
        if len(part) < min_n or part[score_col].nunique() < 5:
            continue
        q = pd.qcut(part[score_col].rank(method="first"), 5, labels=False)
        spreads[pd.Timestamp(d)] = part.loc[q == 4, ex_col].mean() - part.loc[q == 0, ex_col].mean()
    if not spreads:
        return pd.DataFrame(columns=["an", "spread_mediu", "nr_date", "sus_bate_jos"])
    s = pd.Series(spreads)
    g = s.groupby(s.index.year)
    out = pd.DataFrame({"an": g.mean().index, "spread_mediu": g.mean().values, "nr_date": g.count().values})
    out["sus_bate_jos"] = out["spread_mediu"] > 0
    return out


def band_medians(df, score_col, ex_col, bins=VERDICT_BINS, labels=VERDICT_LABELS):
    """Mediana excesului pe benzile de verdict și dacă e monotonă (FAVORABIL > NEUTRU > NEFAVORABIL).
    Întoarce (DataFrame banda, N, mediana_exces; monoton: bool sau None când lipsește o bandă)."""
    data = df[[score_col, ex_col]].apply(pd.to_numeric, errors="coerce").dropna()
    data["banda"] = pd.cut(data[score_col], bins=list(bins), labels=list(labels), right=False, include_lowest=True)
    g = data.groupby("banda", observed=False)[ex_col]
    t = pd.DataFrame({"N": g.count(), "mediana_exces": g.median()}).reset_index()
    med = list(t["mediana_exces"])
    if any(pd.isna(m) for m in med):
        return t, None
    return t, bool(all(a < b for a, b in zip(med, med[1:])))


def sector_rank_score(df, score_col, sector_col="sector", min_n=MIN_SYMBOLS_PER_DATE):
    """Varianta (3): scorul ca percentilă (0–100) în interiorul sectorului, la fiecare dată. NaN pentru sectoarele
    cu mai puțin de `min_n` scoruri la dată sau fără sector."""
    s = pd.to_numeric(df[score_col], errors="coerce")
    keys = [df["date"], df[sector_col].fillna("__fara__")]
    grp = s.groupby(keys)
    pct = grp.rank(pct=True) * 100.0
    cnt = grp.transform("count")
    return pct.where((cnt >= min_n) & df[sector_col].notna())


__all__ = ["yahoo_symbol", "intervals_from_snapshots", "members_at", "current_members", "accept_mapping",
           "forward_return_at", "add_median_excess", "newey_west_t", "ic_series", "ic_summary",
           "quintile_spread_by_year", "band_medians", "sector_rank_score"]
