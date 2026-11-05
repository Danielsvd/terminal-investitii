"""Rezumatul Zilei: mișcarea fiecărui simbol în ultima ședință, lățimea pieței, lichiditatea și starea bursei.
Funcții pure (fără Streamlit, fără rețea).

Reguli:
- „ultima ședință” a unei piețe = cea mai recentă dată cu preț din universul ei; un simbol fără tranzacții în acea
  zi (fără rând sau cu volum 0) nu intră în lățime și nici în topuri: variația lui ar fi de altă zi;
- variația zilei e pe prețul de închidere neajustat (cum o raportează bursa); σ = variația / abaterea standard a
  variațiilor zilnice din cele SIGMA_WINDOW ședințe dinainte (fără ziua curentă), cu minimum SIGMA_MIN_OBS;
- lichiditatea = valoarea tranzacționată (preț × volum), nu numărul de acțiuni;
- pragurile (1 σ, 2 σ, ±0,5 pp) sunt descriptive, netestate.
"""
from datetime import datetime, time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from data.helpers import format_ro

SIGMA_WINDOW = 60
SIGMA_MIN_OBS = 20
TURNOVER_LOOKBACK = 20
TURNOVER_MIN_OBS = 10
MIN_VALUE_RON = 50_000           # pragul de valoare tranzacționată pentru topurile BVB (Daniel, 05.11.2026: coborât de la 100.000)
SESSION_MIN_SHARE = 0.2          # ședința = ultima zi cu preț la cel puțin 20% din simboluri (nu un singur simbol)
TOP_N = 10                       # câte creșteri / scăderi arată fiecare top (Daniel, 05.11.2026: de la 5 la 10)

# Orele ședinței continue, ora locală a bursei. Fără sărbători legale: data ultimei ședințe vine din date.
MARKET_HOURS = {
    "BVB": ("Europe/Bucharest", time(10, 0), time(17, 45)),
    "NYSE": ("America/New_York", time(9, 30), time(16, 0)),
    "XETRA": ("Europe/Berlin", time(9, 0), time(17, 30)),    # reper pentru Europa (LSE: 08:00–16:30 Londra, aceeași oră)
}


def parse_universe(text):
    """Simbolurile dintr-un fișier de univers (un simbol pe rând; rândurile goale și cele cu # se ignoră), fără dubluri."""
    out = []
    for line in str(text or "").splitlines():
        sym = line.split("#", 1)[0].strip().upper()
        if sym and sym not in out:
            out.append(sym)
    return out


def universe_sections(text):
    """{simbol: secțiune} dintr-un fișier de univers. O secțiune începe cu un rând „# == Nume ==”;
    simbolurile de dinaintea primei secțiuni primesc None. Aceleași reguli de citire ca `parse_universe`."""
    out, section = {}, None
    for line in str(text or "").splitlines():
        raw = line.strip()
        if raw.startswith("#"):
            body = raw.lstrip("#").strip()
            if body.startswith("==") and body.endswith("==") and len(body) > 4:
                section = body.strip("=").strip() or None
            continue
        sym = raw.split("#", 1)[0].strip().upper()
        if sym and sym not in out:
            out[sym] = section
    return out


def missing_symbols(universe, table):
    """Simbolurile din univers care nu au niciun preț în date (lipsesc din `table`), în ordinea universului."""
    have = set() if table is None else set(table.index)
    return [s for s in (universe or []) if s not in have]


def full_table(table, sections=None):
    """Toate simbolurile cu date, cu coloana `section` (piața), ordonate pe secțiune (ordinea din fișier),
    apoi tranzacționate înaintea celor fără tranzacții în ședință, apoi după variație descrescător."""
    if table is None or table.empty:
        return table
    sections = sections or {}
    order = {}
    for sec in sections.values():
        order.setdefault(sec, len(order))
    t = table.copy()
    t["section"] = [sections.get(s) for s in t.index]
    t["_o"] = [order.get(sec, len(order)) for sec in t["section"]]
    t["_nt"] = ~t["traded"].astype(bool)
    t["_p"] = -t["pct"].astype("float64").fillna(-np.inf)
    return t.sort_values(["_o", "_nt", "_p"], kind="mergesort").drop(columns=["_o", "_nt", "_p"])


def _clean_frame(frame):
    if frame is None or len(frame) == 0:
        return pd.DataFrame()
    df = frame.apply(pd.to_numeric, errors="coerce")
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    return df[~df.index.duplicated(keep="last")].sort_index()


def sigma_of_last(returns, window=SIGMA_WINDOW, min_obs=SIGMA_MIN_OBS):
    """Ultima variație ÷ abaterea standard a celor `window` variații dinainte (fără ultima). None sub `min_obs`."""
    r = pd.Series(returns, dtype="float64").dropna()
    if len(r) < min_obs + 1:
        return None
    prior = r.iloc[-(window + 1):-1]
    if len(prior) < min_obs:
        return None
    sd = float(prior.std(ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        return None
    return float(r.iloc[-1] / sd)


def daily_move(close):
    """Mișcarea unei serii la ultima dată: dict date, last, prev, pct (fracție), sigma, n (variații în fereastră).
    Valorile care nu se pot calcula sunt None."""
    s = pd.to_numeric(pd.Series(close, dtype="float64"), errors="coerce") if close is not None else pd.Series(dtype=float)
    s = s[s > 0].dropna()
    out = {"date": None, "last": None, "prev": None, "pct": None, "sigma": None, "n": 0}
    if s.empty:
        return out
    if isinstance(s.index, pd.DatetimeIndex):
        idx = s.index.tz_localize(None) if s.index.tz is not None else s.index
        s = pd.Series(s.values, index=idx.normalize())
        s = s[~s.index.duplicated(keep="last")].sort_index()
    out["date"], out["last"] = s.index[-1], float(s.iloc[-1])
    if len(s) < 2:
        return out
    out["prev"] = float(s.iloc[-2])
    rets = s.pct_change(fill_method=None).dropna()
    out["pct"] = float(rets.iloc[-1])
    out["n"] = int(min(len(rets) - 1, SIGMA_WINDOW))
    out["sigma"] = sigma_of_last(rets)
    return out


def session_date(close, min_share=SESSION_MIN_SHARE):
    """Data ultimei ședințe a universului: cea mai recentă zi în care au preț (> 0) cel puțin `min_share` din simbolurile
    cu date (minimum 1). Un singur simbol cu un rând mai nou (pre-market, OTC, alt fus) nu mută ședința. None la gol."""
    c = _clean_frame(close)
    if c.empty:
        return None
    has = c.gt(0)
    n_syms = int(has.any(axis=0).sum())
    if n_syms == 0:
        return None
    counts = has.sum(axis=1)
    ok = counts[counts >= max(1, int(np.ceil(min_share * n_syms)))]
    return ok.index[-1] if len(ok) else None


def session_table(close, volume):
    """Tabelul ultimei ședințe pe un univers. `close`, `volume`: DataFrame cu o coloană pe simbol, index de date.

    Întoarce (df, session_date): df cu index = simbol și coloanele date (ultima dată cu preț), close, prev, pct
    (fracție), volume, value (close × volume, în moneda pieței), traded (are preț și volum > 0 în ziua ședinței),
    sigma. session_date = `session_date(close)`; rândurile de după ea se ignoră (None la univers gol).
    """
    c = _clean_frame(close)
    session = session_date(c)
    if session is not None:
        c = c.loc[:session]
    v = _clean_frame(volume).reindex(index=c.index, columns=c.columns) if len(c) else pd.DataFrame()
    cols = ["date", "close", "prev", "pct", "volume", "value", "traded", "sigma"]
    if c.empty:
        return pd.DataFrame(columns=cols), None
    rows = {}
    for sym in c.columns:
        s = c[sym]
        s = s[s > 0].dropna()
        if s.empty:
            continue
        m = daily_move(s)
        vol = v[sym].get(s.index[-1]) if sym in v.columns else None
        vol = float(vol) if vol is not None and np.isfinite(vol) else None
        rows[sym] = {"date": s.index[-1], "close": m["last"], "prev": m["prev"], "pct": m["pct"], "volume": vol,
                     "value": (m["last"] * vol) if vol is not None else None, "sigma": m["sigma"]}
    if not rows:
        return pd.DataFrame(columns=cols), None
    df = pd.DataFrame.from_dict(rows, orient="index")
    df["traded"] = (df["date"] == session) & df["volume"].fillna(0).gt(0) & df["pct"].notna()
    return df[cols], session


def breadth_counts(table):
    """Lățimea pe simbolurile tranzacționate în ședință: dict up, down, flat, n (tranzacționate), not_traded,
    up_pct, down_pct, flat_pct (procente din n; None la n = 0). Nemodificat = variație exact 0."""
    if table is None or table.empty:
        return {"up": 0, "down": 0, "flat": 0, "n": 0, "not_traded": 0, "up_pct": None, "down_pct": None, "flat_pct": None}
    t = table[table["traded"]]
    up, down = int((t["pct"] > 0).sum()), int((t["pct"] < 0).sum())
    flat, n = int((t["pct"] == 0).sum()), int(len(t))
    pct = (lambda k: k / n * 100.0) if n else (lambda k: None)
    return {"up": up, "down": down, "flat": flat, "n": n, "not_traded": int(len(table) - n),
            "up_pct": pct(up), "down_pct": pct(down), "flat_pct": pct(flat)}


def top_moves(table, n=5, min_value=None):
    """(creșteri, scăderi): doar simbolurile tranzacționate, cu valoarea ≥ `min_value` (dacă e dat);
    creșterile au variație > 0 (descrescător), scăderile < 0 (crescător). Cel mult `n` pe fiecare."""
    if table is None or table.empty:
        return table, table
    t = table[table["traded"]]
    if min_value is not None:
        t = t[t["value"].fillna(-1) >= min_value]
    return (t[t["pct"] > 0].sort_values("pct", ascending=False).head(n),
            t[t["pct"] < 0].sort_values("pct", ascending=True).head(n))


def liquidity_top(table, n=10):
    """Primele `n` simboluri tranzacționate după valoarea tranzacționată."""
    if table is None or table.empty:
        return table
    t = table[table["traded"]]
    return t.sort_values("value", ascending=False).head(n)


def turnover_vs_median(close, volume, lookback=TURNOVER_LOOKBACK, min_obs=TURNOVER_MIN_OBS):
    """Valoarea totală tranzacționată în univers în ultima ședință față de mediana celor `lookback` ședințe dinainte.

    Întoarce dict date, today (valoarea), median, ratio, n (ședințe în mediană); None unde nu se poate calcula.
    Zilele cu valoare totală 0 (fără tranzacții în tot universul) nu sunt ședințe și se scot."""
    c, v = _clean_frame(close), _clean_frame(volume)
    session = session_date(c)
    if session is not None:
        c = c.loc[:session]
    out = {"date": None, "today": None, "median": None, "ratio": None, "n": 0}
    if c.empty or v.empty:
        return out
    common = [col for col in c.columns if col in v.columns]
    total = (c[common] * v[common].reindex(c.index)).sum(axis=1, min_count=1).dropna()
    total = total[total > 0]
    if total.empty:
        return out
    out["date"], out["today"] = total.index[-1], float(total.iloc[-1])
    prior = total.iloc[:-1].iloc[-lookback:]
    out["n"] = int(len(prior))
    if len(prior) >= min_obs:
        out["median"] = float(prior.median())
        out["ratio"] = out["today"] / out["median"] if out["median"] > 0 else None
    return out


def market_state(market, now=None):
    """Starea bursei la `now` (datetime cu fus orar; implicit acum): dict state ('deschisă' / 'închisă' /
    'weekend'), local (ora locală a bursei), opens, closes (ora României, text „HH:MM”). Fără sărbători legale."""
    tz_name, t_open, t_close = MARKET_HOURS[market]
    tz, ro_tz = ZoneInfo(tz_name), ZoneInfo("Europe/Bucharest")
    now = now or datetime.now(tz)
    local = now.astimezone(tz)
    day = local.date()
    o = datetime.combine(day, t_open, tz).astimezone(ro_tz)
    c = datetime.combine(day, t_close, tz).astimezone(ro_tz)
    if local.weekday() >= 5:
        state = "weekend"
    elif t_open <= local.time() < t_close:
        state = "deschisă"
    else:
        state = "închisă"
    return {"state": state, "local": local, "opens": o.strftime("%H:%M"), "closes": c.strftime("%H:%M"), "day": day}


def session_is_partial(session_date, state):
    """True când ultima ședință din date e ziua de azi a bursei și bursa e încă deschisă (date în timpul ședinței)."""
    if session_date is None or state is None:
        return False
    return state["state"] == "deschisă" and pd.Timestamp(session_date).date() == state["day"]


def relative_day(a_close, b_close):
    """Variația lui A minus variația lui B în ultima zi comună (puncte procentuale) și cele două variații (fracții).
    dict a, b, diff_pp, date; None unde lipsesc datele."""
    out = {"a": None, "b": None, "diff_pp": None, "date": None}
    both = pd.concat([pd.Series(a_close, dtype="float64").rename("a"), pd.Series(b_close, dtype="float64").rename("b")],
                     axis=1, join="inner").dropna()
    both = both[(both["a"] > 0) & (both["b"] > 0)]
    if len(both) < 2:
        return out
    ra = float(both["a"].iloc[-1] / both["a"].iloc[-2] - 1.0)
    rb = float(both["b"].iloc[-1] / both["b"].iloc[-2] - 1.0)
    out.update({"a": ra, "b": rb, "diff_pp": (ra - rb) * 100.0, "date": both.index[-1]})
    return out


def sigma_label(sigma):
    """Text descriptiv pentru o mișcare în σ (praguri 1 și 2, netestate)."""
    if sigma is None or not np.isfinite(sigma):
        return "N/A"
    a = abs(sigma)
    size = "neobișnuită (peste 2σ)" if a >= 2 else ("mai mare decât de obicei (1–2σ)" if a >= 1 else "obișnuită (sub 1σ)")
    return f"mișcare {size}"


def money_short(value, currency):
    """Sumă mare, scurt, în format românesc: „45,2 mil. lei”, „1,23 mld. USD”, „850.000 lei”. Lipsă → „N/A”."""
    if value is None or not np.isfinite(value):
        return "N/A"
    unit = "lei" if currency == "RON" else currency
    a = abs(value)
    if a >= 1e9:
        return f"{format_ro(value / 1e9, 2)} mld. {unit}"
    if a >= 1e6:
        return f"{format_ro(value / 1e6, 1)} mil. {unit}"
    return f"{format_ro(value, 0)} {unit}"
