"""Secțiunea „Idei și strategii” (propunerea din 14.11.2026, aprobată de Daniel: totul deodată, risc 1% pe idee,
jurnal în foaia `Idei_jurnal`). Funcții pure: fără Streamlit, fără rețea.

Straturi:
  1. Regimul: cadranul creștere × inflație (CFNAI-MA3 și accelerarea inflației de bază, FRED).
  2. ETF-uri: trend pe clase de active (Faber 2007; Moskowitz, Ooi & Pedersen 2012), rotația sectoarelor cu
     momentum dublu (Moskowitz & Grinblatt 1999; Antonacci 2014), regiuni, episoade de stres (VIX / VIX3M).
  3. Acțiuni: filtru de calitate (excludere, nu clasare) + catalizatori: deriva după rezultate (Bernard & Thomas
     1989), revizuirile estimărilor (Chan, Jegadeesh & Lakonishok 1996), apropierea de maximul pe 52 de săptămâni
     (George & Hwang 2004).
  4. Dimensionare: risc fix pe idee ÷ distanța până la stop, plafon pe poziție.
  5. Jurnal: fiecare idee salvată e evaluată ulterior (stop fix la intrare sau orizontul ideii).

Toate pragurile de aici sunt NETESTATE în aplicație: studiile citate sunt baza, testul istoric propriu e
`tools/backtest_idei.py` (doar straturile A și B pe ETF-uri). Lipsă = None, niciodată 0.
"""
import math

import numpy as np
import pandas as pd

from analytics.scoring import momentum_12_1
from data.helpers import format_ro

UNTESTED = "netestat în aplicație"

# --- Universuri ----------------------------------------------------------------------------------------------------

# Clasele de active pentru regula de trend: (simbol semnal, nume). ETF-uri SUA: istorie lungă și date curate.
ASSET_CLASSES = (
    ("SPY", "Acțiuni SUA (S&P 500)"),
    ("VGK", "Acțiuni Europa"),
    ("EEM", "Acțiuni piețe emergente"),
    ("EWJ", "Acțiuni Japonia"),
    ("IEF", "Trezorerie SUA 7–10 ani"),
    ("TLT", "Trezorerie SUA 20+ ani"),
    ("TIP", "Obligațiuni SUA indexate cu inflația (TIPS)"),
    ("GLD", "Aur"),
    ("DBC", "Mărfuri diversificate"),
    ("UUP", "Dolarul american (indice)"),
)

US_SECTORS = {
    "XLK": "Tehnologie", "XLC": "Comunicații", "XLY": "Consum discreționar", "XLP": "Consum de bază",
    "XLF": "Financiar", "XLV": "Sănătate", "XLI": "Industrial", "XLE": "Energie", "XLB": "Materiale",
    "XLU": "Utilități", "XLRE": "Imobiliare",
}

# Sectoarele STOXX Europe 600 (iShares, Xetra). Simbolurile Yahoo NU sunt verificate din mediul de lucru:
# cele fără date apar „fără date” în tabel.
EU_SECTORS = {
    "EXV1.DE": "Bănci", "EXH5.DE": "Asigurări", "EXV3.DE": "Tehnologie", "EXV4.DE": "Sănătate",
    "EXH1.DE": "Petrol și gaze", "EXH4.DE": "Bunuri și servicii industriale", "EXV6.DE": "Resurse de bază",
    "EXH9.DE": "Utilități", "EXV5.DE": "Auto și componente", "EXH3.DE": "Alimente și băuturi",
    "EXV2.DE": "Telecomunicații", "EXV8.DE": "Construcții și materiale",
}

# Regiunile: (simbol, nume, valuta cotației). Comparația se face în USD (TVBETETF convertit cu USDRON);
# ordinea e aceeași și în lei, pentru că toate se înmulțesc cu același curs USD/RON.
REGIONS = (
    ("SPY", "SUA", "USD"),
    ("VGK", "Europa", "USD"),
    ("EEM", "Piețe emergente", "USD"),
    ("EWJ", "Japonia", "USD"),
    ("TVBETETF.RO", "România (BET, prin ETF)", "RON"),
)

# --- Strat 1: cadranul creștere × inflație --------------------------------------------------------------------------

GROWTH_BORDER = 0.10      # |CFNAI-MA3| < 0,10: „la graniță” (netestat)
INFLATION_BORDER = 0.25   # |accelerare| < 0,25 pp: „la graniță” (netestat)

QUADRANTS = {
    "Q1": "Creștere ↑, inflație ↓",
    "Q2": "Creștere ↑, inflație ↑",
    "Q3": "Creștere ↓, inflație ↑",
    "Q4": "Creștere ↓, inflație ↓",
}

# Înclinarea istorică pe cadrane (Ilmanen, Expected Returns, 2011; Neville et al., Man Group, 2021). Orientativă.
QUADRANT_TILTS = {
    "Q1": {"favor": ("SPY", "EEM", "XLK", "XLY", "XLI"), "against": ("GLD", "DBC", "XLU", "XLP")},
    "Q2": {"favor": ("DBC", "XLE", "XLB", "XLF", "TIP"), "against": ("TLT", "IEF", "XLU")},
    "Q3": {"favor": ("GLD", "TIP", "DBC", "XLE"), "against": ("SPY", "TLT", "XLY", "XLK")},
    "Q4": {"favor": ("TLT", "IEF", "XLU", "XLV", "XLP"), "against": ("DBC", "XLE", "XLB", "XLY")},
}

QUADRANT_TEXT = {
    "Q1": "istoric au mers relativ bine acțiunile și ciclicele (tehnologie, consum discreționar); aurul și mărfurile mai slab",
    "Q2": "istoric au mers relativ bine mărfurile, energia, materialele, financiarele și TIPS; obligațiunile lungi mai slab",
    "Q3": "istoric au mers relativ bine aurul, mărfurile, TIPS și numerarul; acțiunile și obligațiunile nominale lungi mai slab",
    "Q4": "istoric au mers relativ bine obligațiunile lungi și defensivele (utilități, sănătate, consum de bază); ciclicele și mărfurile mai slab",
}


def _series(s):
    """Serie numerică curată, index de date fără fus orar, sortată, fără dubluri."""
    if s is None:
        return pd.Series(dtype="float64")
    out = pd.to_numeric(pd.Series(s, dtype="float64"), errors="coerce").dropna()
    if len(out) and isinstance(out.index, pd.DatetimeIndex):
        idx = out.index.tz_localize(None) if out.index.tz is not None else out.index
        out = pd.Series(out.to_numpy(), index=idx)
        out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def _monthly(s):
    """Ultima valoare din fiecare lună calendaristică, indexată pe perioada lunii."""
    s = _series(s)
    if s.empty:
        return pd.Series(dtype="float64")
    return s.groupby(s.index.to_period("M")).last()


def _quadrant_key(growth_up, inflation_up):
    if growth_up is None or inflation_up is None:
        return None
    return {(True, False): "Q1", (True, True): "Q2", (False, True): "Q3", (False, False): "Q4"}[(growth_up, inflation_up)]


def inflation_acceleration(core_cpi):
    """Seria lunară a accelerării inflației de bază, în puncte procentuale: ritmul anualizat pe 3 luni minus ritmul
    pe 12 luni, ambele din nivelul indicelui (CPI_t / CPI_{t−3})^4 − 1 și CPI_t / CPI_{t−12} − 1."""
    m = _monthly(core_cpi)
    m = m[m > 0]
    if len(m) < 13:
        return pd.Series(dtype="float64")
    full = m.reindex(pd.period_range(m.index[0], m.index[-1], freq="M"))
    ann3 = (full / full.shift(3)) ** 4 - 1
    yoy = full / full.shift(12) - 1
    return ((ann3 - yoy) * 100).dropna()


def growth_inflation_quadrant(cfnai_ma3, core_cpi):
    """Cadranul curent. dict: growth (ultima valoare CFNAI-MA3), growth_date, growth_up (True/False/None la graniță),
    accel (pp), ann3, yoy (fracții), inflation_date, inflation_up, key („Q1”…„Q4” sau None), label, months
    (de câte luni consecutive, inclusiv ultima lună comună, e același cadran; pe semn, fără graniță) și text."""
    g = _monthly(cfnai_ma3)
    acc = inflation_acceleration(core_cpi)
    out = {"growth": None, "growth_date": None, "growth_up": None, "accel": None, "ann3": None, "yoy": None,
           "inflation_date": None, "inflation_up": None, "key": None, "label": None, "months": None, "text": None}
    if len(g):
        out["growth"], out["growth_date"] = float(g.iloc[-1]), g.index[-1].to_timestamp()
        if abs(out["growth"]) >= GROWTH_BORDER:
            out["growth_up"] = out["growth"] > 0
    if len(acc):
        out["accel"], out["inflation_date"] = float(acc.iloc[-1]), acc.index[-1].to_timestamp()
        m = _monthly(core_cpi)
        last = acc.index[-1]
        out["ann3"] = float((m[last] / m[last - 3]) ** 4 - 1) if (last - 3) in m.index else None
        out["yoy"] = float(m[last] / m[last - 12] - 1) if (last - 12) in m.index else None
        if abs(out["accel"]) >= INFLATION_BORDER:
            out["inflation_up"] = out["accel"] > 0
    key = _quadrant_key(out["growth_up"], out["inflation_up"])
    out["key"] = key
    if key:
        out["label"], out["text"] = QUADRANTS[key], QUADRANT_TEXT[key]
        common = g.index.intersection(acc.index)
        hist = [_quadrant_key(bool(g[p] > 0), bool(acc[p] > 0)) for p in common]
        n = 0
        for k in reversed(hist):
            if k != key:
                break
            n += 1
        out["months"] = n if hist else None
    return out


def alignment(symbol, quadrant_key):
    """„aliniat” / „contra” / „neutru” față de înclinarea cadranului; None fără cadran."""
    if not quadrant_key:
        return None
    tilt = QUADRANT_TILTS[quadrant_key]
    if symbol in tilt["favor"]:
        return "aliniat"
    if symbol in tilt["against"]:
        return "contra"
    return "neutru"


# --- Strat 2: ETF-uri -----------------------------------------------------------------------------------------------

TREND_SMA_MONTHS = 10     # Faber (2007): media închiderilor lunare pe 10 luni
SMA_DAYS = 200
TOP_SECTORS = 3


def _on_or_before(s, day):
    before = s[s.index <= day]
    return float(before.iloc[-1]) if len(before) else None


def cash_return_12m(rates_pct, end=None):
    """Randamentul numerarului pe ultimele 12 luni, proxy: media randamentelor anuale (%) din fereastră ÷ 100
    (fără capitalizare). None dacă seria nu acoperă 12 luni calendaristice până la `end`."""
    r = _series(rates_pct)
    if r.empty:
        return None
    end = r.index[-1] if end is None else pd.Timestamp(end)
    start = end - pd.DateOffset(months=12)
    if r.index[0] > start + pd.DateOffset(days=10):
        return None
    window = r[(r.index > start) & (r.index <= end)]
    return float(window.mean() / 100.0) if len(window) else None


def return_months(close, months):
    """Randamentul pe `months` luni calendaristice până la ultima ședință (fracție) sau None."""
    c = _series(close)
    c = c[c > 0]
    if len(c) < 2:
        return None
    start = c.index[-1] - pd.DateOffset(months=months)
    if c.index[0] > start:
        return None
    p0 = _on_or_before(c, start)
    return float(c.iloc[-1] / p0 - 1.0) if p0 else None


def above_sma(close, days=SMA_DAYS):
    """True/False: ultima închidere peste media pe `days` ședințe; None sub `days` ședințe."""
    c = _series(close)
    if len(c) < days:
        return None
    return bool(c.iloc[-1] > c.iloc[-days:].mean())


def trend_signal(close, cash12):
    """Regula de trend pe o clasă de active, la ultima ședință.

    ÎN TREND: randamentul pe 12 luni > numerarul pe 12 luni ȘI ultima închidere > media ultimelor 10 închideri lunare
    (inclusiv luna curentă). IEȘIT DIN TREND: ambele false. MIXT: una din două. None fără date.
    `months_above`: de câte luni consecutive închiderea lunară e peste media ei pe 10 luni (inclusiv luna curentă)."""
    c = _series(close)
    c = c[c > 0]
    out = {"last": None, "date": None, "r12": None, "sma10m": None, "above_sma10m": None, "beats_cash": None,
           "state": None, "months_above": None}
    if c.empty:
        return out
    out["last"], out["date"] = float(c.iloc[-1]), c.index[-1]
    m = _monthly(c)
    if len(m) >= TREND_SMA_MONTHS:
        sma = m.rolling(TREND_SMA_MONTHS).mean()
        out["sma10m"] = float(sma.iloc[-1])
        out["above_sma10m"] = bool(out["last"] > out["sma10m"])
        flags = (m > sma).iloc[TREND_SMA_MONTHS - 1:]
        n = 0
        for f in reversed(flags.tolist()):
            if f != out["above_sma10m"]:
                break
            n += 1
        out["months_above"] = n if out["above_sma10m"] else 0
    out["r12"] = return_months(c, 12)
    if out["r12"] is not None and cash12 is not None:
        out["beats_cash"] = bool(out["r12"] > cash12)
    a, b = out["above_sma10m"], out["beats_cash"]
    if a is not None and b is not None:
        out["state"] = "ÎN TREND" if (a and b) else "IEȘIT DIN TREND" if not (a or b) else "MIXT"
    return out


def dual_momentum_table(closes, cash12, top=TOP_SECTORS):
    """Clasamentul după momentum 12-1 cu filtrul absolut (preț > SMA 200 ȘI randament 12 luni > numerar).

    `closes`: {simbol: serie de închideri}. Întoarce DataFrame cu coloanele symbol, mom (12-1), r12, above200,
    beats_cash, passes (True/False/None), rank (1 = cel mai mare momentum, între cei cu momentum), selected
    (primii `top` după momentum dintre cei care trec filtrul). Simbolurile fără date au valorile None."""
    rows = []
    for sym, close in closes.items():
        c = _series(close)
        c = c[c > 0]
        mom = momentum_12_1(c) if len(c) else None
        r12 = return_months(c, 12)
        a200 = above_sma(c)
        beats = None if (r12 is None or cash12 is None) else bool(r12 > cash12)
        passes = None if (a200 is None or beats is None) else bool(a200 and beats)
        rows.append({"symbol": sym, "mom": mom, "r12": r12, "above200": a200, "beats_cash": beats, "passes": passes,
                     "last": float(c.iloc[-1]) if len(c) else None, "date": c.index[-1] if len(c) else None})
    df = pd.DataFrame(rows, columns=["symbol", "mom", "r12", "above200", "beats_cash", "passes", "last", "date"])
    if df.empty:
        df["rank"], df["selected"] = pd.Series(dtype="float64"), pd.Series(dtype=bool)
        return df
    mom = pd.to_numeric(df["mom"], errors="coerce")
    df["rank"] = mom.rank(ascending=False, method="first")
    eligible = df[(df["passes"] == True) & mom.notna()]  # noqa: E712 (None nu trece)
    chosen = set(eligible.assign(_m=mom[eligible.index]).sort_values("_m", ascending=False).head(top)["symbol"])
    df["selected"] = df["symbol"].isin(chosen)
    return df.sort_values("rank", na_position="last").reset_index(drop=True)


def to_usd(close_ron, usdron):
    """Închiderile în lei convertite în USD cu cursul lei/USD din aceeași zi sau ultima zi anterioară (≤ 5 zile)."""
    c, fx = _series(close_ron), _series(usdron)
    fx = fx[fx > 0]
    if c.empty or fx.empty:
        return pd.Series(dtype="float64")
    aligned = fx.reindex(fx.index.union(c.index)).ffill(limit=5).reindex(c.index)
    return (c / aligned).dropna()


def in_usd(close, currency, usdron, eurron=None, gbpron=None):
    """Prețurile unei linii UCITS convertite în USD prin cursurile în lei (EUR/USD = EURRON ÷ USDRON etc.), ca
    verificarea corelației să compare același lucru: pe obligațiuni, mișcarea EUR/USD e cât mișcarea ETF-ului și
    ar coborî corelația fără ca instrumentul să fie greșit. USD → neschimbat; valută fără curs → serie goală."""
    c = _series(close)
    if c.empty or currency == "USD":
        return c
    per_ron = {"EUR": eurron, "GBP": gbpron, "GBp": gbpron}.get(currency)
    if per_ron is None:
        return pd.Series(dtype="float64")
    ron = c * _series(per_ron).reindex(c.index.union(_series(per_ron).index)).ffill(limit=5).reindex(c.index)
    out = to_usd(ron.dropna(), usdron)
    return out / 100 if currency == "GBp" else out


def tracking_check(signal_close, ucits_close, min_weeks=26):
    """Corelația randamentelor săptămânale (vineri, pe ultimele 12 luni) dintre ETF-ul semnal și echivalentul UCITS.
    dict corr, n (săptămâni comune), ok (corr ≥ 0,80; None sub `min_weeks`). Liniile în altă valută se trec întâi prin `in_usd`."""
    a, b = _series(signal_close), _series(ucits_close)
    out = {"corr": None, "n": 0, "ok": None}
    if a.empty or b.empty:
        return out
    end = min(a.index[-1], b.index[-1])
    start = end - pd.DateOffset(months=12)
    wa = a[(a.index > start) & (a.index <= end)].resample("W-FRI").last().pct_change()
    wb = b[(b.index > start) & (b.index <= end)].resample("W-FRI").last().pct_change()
    both = pd.concat([wa, wb], axis=1).dropna()
    out["n"] = len(both)
    if len(both) >= min_weeks:
        out["corr"] = float(both.iloc[:, 0].corr(both.iloc[:, 1]))
        out["ok"] = bool(out["corr"] >= 0.80)
    return out


# Episoadele de stres: VIX / VIX3M ≥ 1 (structură inversată), după cel puțin 20 de ședințe sub prag.
STRESS_RATIO = 1.0
STRESS_GAP = 20
STRESS_MIN_N = 20
STRESS_HORIZONS = (3, 6)   # luni calendaristice


def _forward(c, day, months):
    end = day + pd.DateOffset(months=months)
    if end > c.index[-1]:
        return None
    p0, p1 = _on_or_before(c, day), _on_or_before(c, end)
    return float(p1 / p0 - 1.0) if p0 and p1 else None


def stress_episodes(vix, vix3m, bench):
    """Statistica episoadelor de stres pe istoricul disponibil (descriptivă, în eșantion).

    dict: n (episoade cu randament ulterior pe 3 luni), episodes (lista datelor de început), stats {luni: {mean, pos,
    base_mean, base_pos, n}}, ratio (ultimul VIX/VIX3M), in_stress (True/False/None), last_episode, enough (n ≥ 20).
    `base_*`: aceleași statistici pe toate ședințele (randamentul ulterior necondiționat), pentru comparație."""
    v, v3, b = _series(vix), _series(vix3m), _series(bench)
    out = {"n": 0, "episodes": [], "stats": {}, "ratio": None, "in_stress": None, "last_episode": None,
           "enough": False}
    both = pd.concat([v, v3], axis=1, keys=["v", "v3"]).dropna()
    both = both[both["v3"] > 0]
    if both.empty or b.empty:
        return out
    ratio = both["v"] / both["v3"]
    out["ratio"] = float(ratio.iloc[-1])
    out["in_stress"] = bool(out["ratio"] >= STRESS_RATIO)
    starts, below = [], STRESS_GAP
    for day, r in ratio.items():
        if r >= STRESS_RATIO:
            if below >= STRESS_GAP:
                starts.append(day)
            below = 0
        else:
            below += 1
    out["episodes"] = starts
    out["last_episode"] = starts[-1] if starts else None
    for months in STRESS_HORIZONS:
        fw = [_forward(b, d, months) for d in starts]
        fw = [x for x in fw if x is not None]
        base_days = b.index[b.index <= b.index[-1] - pd.DateOffset(months=months)]
        base_days = base_days[base_days >= ratio.index[0]]
        base = [_forward(b, d, months) for d in base_days[::5]]   # din 5 în 5 ședințe: aceeași medie, calcul mai rapid
        base = [x for x in base if x is not None]
        out["stats"][months] = {
            "n": len(fw),
            "mean": float(np.mean(fw)) if fw else None,
            "pos": float(np.mean([x > 0 for x in fw])) if fw else None,
            "base_mean": float(np.mean(base)) if base else None,
            "base_pos": float(np.mean([x > 0 for x in base])) if base else None,
        }
    out["n"] = out["stats"][STRESS_HORIZONS[0]]["n"]
    out["enough"] = out["n"] >= STRESS_MIN_N
    return out


# --- Strat 3: acțiuni -----------------------------------------------------------------------------------------------

PREFILTER_US = 12          # câte acțiuni SUA trec la analiza detaliată (cost: ~5 cereri Yahoo pe acțiune)
PREFILTER_EU = 8
HIGH_WEEKS = 52
NEAR_HIGH = -0.05          # la cel mult 5% sub maximul pe 52 de săptămâni
REVISION_MIN = 0.01        # estimarea EPS pe anul curent urcată cu cel puțin 1% în 90 de zile
PEAD_MAX_AGE_DAYS = 90     # deriva după rezultate se măsoară pe ~60 de ședințe
MIN_CATALYSTS = 2
QUALITY_PIOTROSKI_MAX_FAIL = 3   # Piotroski ≤ 3 (din cel puțin 7 criterii evaluabile) → exclus
QUALITY_MIN_EVALUABLE = 7
QUALITY_MAX_ND_EBITDA = 4.0


def from_high(close, weeks=HIGH_WEEKS):
    """Ultima închidere ÷ maximul pe `weeks` săptămâni − 1 (≤ 0) sau None dacă istoricul nu acoperă fereastra."""
    c = _series(close)
    c = c[c > 0]
    if c.empty:
        return None
    start = c.index[-1] - pd.DateOffset(weeks=weeks)
    if c.index[0] > start:
        return None
    window = c[c.index > start]
    return float(c.iloc[-1] / window.max() - 1.0)


def stock_prefilter(closes, max_n):
    """Preselecția ieftină (o singură descărcare de prețuri): momentum 12-1 > 0 și preț > SMA 200, apoi primele
    `max_n` după momentum. DataFrame symbol, mom, from_high, sortat descrescător după momentum."""
    rows = []
    for sym in closes.columns:
        c = _series(closes[sym])
        mom = momentum_12_1(c) if len(c) else None
        a200 = above_sma(c)
        if mom is not None and mom > 0 and a200:
            rows.append({"symbol": sym, "mom": mom, "from_high": from_high(c)})
    df = pd.DataFrame(rows, columns=["symbol", "mom", "from_high"])
    return df.sort_values("mom", ascending=False).head(max_n).reset_index(drop=True)


def earnings_reaction(close, bench, ann_date):
    """Reacția la anunțul de rezultate, fereastra [−1, +1]: de la ultima închidere dinaintea zilei anunțului până la
    prima închidere de după ea (acoperă anunțul înainte de deschidere și după închidere). dict stock, bench, abnormal
    (fracții) sau None dacă nu există ambele închideri (de ex. anunț de ieri, fără ședință după)."""
    c, b = _series(close), _series(bench)
    if c.empty or b.empty or ann_date is None:
        return None
    day = pd.Timestamp(ann_date)
    day = day.tz_localize(None) if day.tzinfo is not None else day
    day = day.normalize()
    def window(s):
        before, after = s[s.index < day], s[s.index > day]
        if before.empty or after.empty:
            return None
        return float(after.iloc[0] / before.iloc[-1] - 1.0)
    rs, rb = window(c), window(b)
    if rs is None or rb is None:
        return None
    return {"stock": rs, "bench": rb, "abnormal": rs - rb}


def last_announcement(earnings_dates, today):
    """(data ultimului anunț trecut, surpriza % sau None, data următorului anunț sau None) din tabelul
    `get_earnings_dates` (index = data anunțului, coloane „Surprise(%)”, „Reported EPS”, „EPS Estimate”)."""
    if earnings_dates is None or not isinstance(earnings_dates, pd.DataFrame) or earnings_dates.empty:
        return None, None, None
    df = earnings_dates.copy()
    idx = pd.DatetimeIndex(df.index)
    df.index = (idx.tz_localize(None) if idx.tz is not None else idx).normalize()
    today = pd.Timestamp(today).normalize()
    reported = df["Reported EPS"] if "Reported EPS" in df.columns else pd.Series(np.nan, index=df.index)
    past = df[(df.index <= today) & pd.to_numeric(reported, errors="coerce").notna()].sort_index()
    future = df[df.index > today].sort_index()
    nxt = future.index[0] if len(future) else None
    if past.empty:
        return None, None, nxt
    row = past.iloc[-1]
    surprise = pd.to_numeric(pd.Series([row.get("Surprise(%)")]), errors="coerce").iloc[0]
    if pd.isna(surprise):
        est = pd.to_numeric(pd.Series([row.get("EPS Estimate")]), errors="coerce").iloc[0]
        act = pd.to_numeric(pd.Series([row.get("Reported EPS")]), errors="coerce").iloc[0]
        surprise = (act - est) / abs(est) * 100 if (pd.notna(est) and pd.notna(act) and est != 0) else np.nan
    return past.index[-1], (None if pd.isna(surprise) else float(surprise)), nxt


def _sgn(x, decimals=1):
    """Număr cu semn, în format românesc („+5,0”, „−0,3” scris „-0,3”)."""
    return ("+" if x > 0 else "") + format_ro(x, decimals)


def _check(name, passed, detail):
    return {"name": name, "passed": passed, "detail": detail}


def pead_catalyst(surprise_pct, reaction, ann_date, today):
    """Deriva după rezultate: surpriza > 0 ȘI reacția anormală în [−1, +1] > 0, anunț din ultimele 90 de zile."""
    name = "Deriva după rezultate"
    if ann_date is None or surprise_pct is None or reaction is None:
        return _check(name, None, "N/A: lipsește data anunțului, surpriza sau reacția prețului")
    age = (pd.Timestamp(today).normalize() - pd.Timestamp(ann_date).normalize()).days
    detail = (f"anunț {pd.Timestamp(ann_date):%d.%m.%Y} (acum {age} zile), surpriză {_sgn(surprise_pct)}%, "
              f"reacție față de piață {_sgn(reaction['abnormal'] * 100)} pp")
    if age > PEAD_MAX_AGE_DAYS:
        return _check(name, False, detail + f"; peste {PEAD_MAX_AGE_DAYS} de zile, deriva nu mai e activă")
    return _check(name, bool(surprise_pct > 0 and reaction["abnormal"] > 0), detail)


def eps_revision(eps_trend):
    """Revizuirea estimării EPS pe 90 de zile (fracție), pe anul fiscal curent („0y”, altfel „+1y”), din
    `Ticker.eps_trend` (coloane current, 90daysAgo). None fără date sau cu estimarea veche zero."""
    if eps_trend is None or not isinstance(eps_trend, pd.DataFrame) or eps_trend.empty:
        return None, None
    for period in ("0y", "+1y"):
        if period in eps_trend.index and {"current", "90daysAgo"} <= set(eps_trend.columns):
            cur = pd.to_numeric(pd.Series([eps_trend.loc[period, "current"]]), errors="coerce").iloc[0]
            old = pd.to_numeric(pd.Series([eps_trend.loc[period, "90daysAgo"]]), errors="coerce").iloc[0]
            if pd.notna(cur) and pd.notna(old) and old != 0:
                return float((cur - old) / abs(old)), period
    return None, None


def revision_catalyst(revision, period):
    name = "Revizuiri ale estimărilor"
    if revision is None:
        return _check(name, None, "N/A: Yahoo nu are tendința estimărilor EPS")
    per = {"0y": "anul fiscal curent", "+1y": "anul fiscal următor"}.get(period, period)
    return _check(name, bool(revision >= REVISION_MIN), f"estimarea EPS pe {per}: {_sgn(revision * 100)}% în 90 de zile")


def high_catalyst(dist):
    name = "Aproape de maximul pe 52 de săptămâni"
    if dist is None:
        return _check(name, None, "N/A: istoric sub un an")
    return _check(name, bool(dist >= NEAR_HIGH), f"{_sgn(dist * 100)}% față de maximul pe 52 de săptămâni")


def catalyst_summary(checks):
    """dict passed, evaluable, is_idea (cel puțin MIN_CATALYSTS trecuți). Catalizatorii N/A nu se numără."""
    passed = sum(1 for c in checks if c["passed"] is True)
    evaluable = sum(1 for c in checks if c["passed"] is not None)
    return {"passed": passed, "evaluable": evaluable, "is_idea": passed >= MIN_CATALYSTS}


def quality_filter(altman_zone, piotroski_passed, piotroski_evaluable, nd_ebitda, financial):
    """Filtrul de calitate (excludere, nu clasare). Exclus dacă: Altman în zona de dificultate (nefinanciare),
    Piotroski ≤ 3 din cel puțin 7 criterii evaluabile, datorie netă / EBITDA > 4. O valoare lipsă nu exclude.
    dict excluded, checks (listă name/passed/detail)."""
    checks = []
    if financial:
        checks.append(_check("Altman Z", None, "nu se aplică sectorului financiar"))
    elif altman_zone is None:
        checks.append(_check("Altman Z", None, "N/A"))
    else:
        txt = {"safe": "zona sigură", "grey": "zona gri", "distress": "zona de dificultate"}[altman_zone]
        checks.append(_check("Altman Z", altman_zone != "distress", txt))
    if piotroski_passed is None or not piotroski_evaluable or piotroski_evaluable < QUALITY_MIN_EVALUABLE:
        checks.append(_check("Piotroski F-Score", None,
                             f"N/A: {piotroski_evaluable or 0} criterii evaluabile (minim {QUALITY_MIN_EVALUABLE})"))
    else:
        checks.append(_check("Piotroski F-Score", piotroski_passed > QUALITY_PIOTROSKI_MAX_FAIL,
                             f"{piotroski_passed} din {piotroski_evaluable}"))
    if financial:
        checks.append(_check("Datorie netă / EBITDA", None, "nu se aplică sectorului financiar"))
    elif nd_ebitda is None:
        checks.append(_check("Datorie netă / EBITDA", None, "N/A"))
    else:
        checks.append(_check("Datorie netă / EBITDA", nd_ebitda <= QUALITY_MAX_ND_EBITDA, format_ro(nd_ebitda, 2, "×")))
    return {"excluded": any(c["passed"] is False for c in checks), "checks": checks}


# --- Strat 4: evenimente ------------------------------------------------------------------------------------------

HIGH_IMPACT_KEYS = ("fomc", "cpi", "nfp", "pce")
EARNINGS_SOON_DAYS = 10


def high_impact_events(calendar_rows):
    """Publicările cu impact mare (FOMC, CPI, NFP, PCE) din rândurile `fred_calendar.upcoming`."""
    return [r for r in (calendar_rows or []) if r.get("key") in HIGH_IMPACT_KEYS]


def earnings_in_days(next_date, today):
    """Zilele până la următorul anunț de rezultate sau None."""
    if next_date is None:
        return None
    return int((pd.Timestamp(next_date).normalize() - pd.Timestamp(today).normalize()).days)


# --- Dimensionare ---------------------------------------------------------------------------------------------------

RISK_PER_IDEA = 0.01      # decizia lui Daniel, 14.11.2026
MAX_POSITION = 0.25       # același plafon ca în optimizator


def position_size(capital_ron, entry, stop, fx_ron, risk_pct=RISK_PER_IDEA, max_weight=MAX_POSITION):
    """Cantitatea pentru riscul fix pe idee: (capital × risc) ÷ (intrare − stop), în valuta instrumentului
    (`fx_ron` = lei pe o unitate de valută; 1 pentru lei). Plafon: valoarea poziției ≤ `max_weight` × capital.

    dict qty (unități întregi), risk_ron (pierderea la stop pentru qty), value_ron, weight, capped, reason (text
    când nu se poate calcula)."""
    out = {"qty": None, "risk_ron": None, "value_ron": None, "weight": None, "capped": False, "reason": None}
    vals = (capital_ron, entry, stop, fx_ron)
    if any(v is None or not math.isfinite(float(v)) for v in vals):
        out["reason"] = "lipsește capitalul, prețul, stopul sau cursul valutar"
        return out
    if capital_ron <= 0 or fx_ron <= 0 or entry <= 0:
        out["reason"] = "capitalul, prețul și cursul trebuie să fie pozitive"
        return out
    if stop >= entry:
        out["reason"] = "stopul nu e sub prețul de intrare"
        return out
    per_unit_ron = (entry - stop) * fx_ron
    qty = math.floor(capital_ron * risk_pct / per_unit_ron)
    max_qty = math.floor(capital_ron * max_weight / (entry * fx_ron))
    if qty > max_qty:
        qty, out["capped"] = max_qty, True
    if qty < 1:
        out["reason"] = "riscul pe idee nu acoperă nici o unitate"
        return out
    out.update(qty=int(qty), risk_ron=qty * per_unit_ron, value_ron=qty * entry * fx_ron,
               weight=qty * entry * fx_ron / capital_ron)
    return out


# --- Strat 5: jurnalul ideilor ------------------------------------------------------------------------------------

STRATEGIES = {
    "trend": {"name": "Trend clase de active", "horizon": 30, "benchmark": "AOR",
              "evidence": "Faber (2007), regula pe media de 10 luni; Moskowitz, Ooi & Pedersen (2012), "
                          "time-series momentum pe 58 de piețe",
              "exit": "închiderea lunară sub media pe 10 luni sau randamentul pe 12 luni sub numerar; revizie lunară",
              "test": "Testat 2000 – oct. 2026 (tools/backtest_idei, cost 0,1%): față de aceleași active ținute, scăderea "
                      "maximă −13,8% în loc de −43,5% și Sharpe 0,54 în loc de 0,29, la randament apropiat (4,6% față de "
                      "4,4% pe an); la fel în ambele perioade (2000–2012 și 2013–2026). Reduce riscul, nu crește randamentul"},
    "sector_us": {"name": "Rotație sectoare SUA", "horizon": 30, "benchmark": "SPY",
                  "evidence": "Moskowitz & Grinblatt (1999), momentum pe industrii; Antonacci (2014), momentum dublu",
                  "exit": "iese din primele 3 sau pierde filtrul absolut (sub SMA 200 sau sub numerar); revizie lunară",
                  "test": "Testat 2000 – oct. 2026: 6,6% pe an față de 8,4% la SPY, Sharpe 0,44 față de 0,48; a bătut SPY în "
                          "2000–2012 (3,1% față de 2,4%), a rămas în urmă în 2013–2026 (10,0% față de 14,3%). Reduce scăderea "
                          "maximă (−29,4% față de −50,8%), nu aduce randament în plus"},
    "sector_eu": {"name": "Rotație sectoare Europa", "horizon": 30, "benchmark": "EXSA.DE",
                  "evidence": "Moskowitz & Grinblatt (1999), momentum pe industrii; Antonacci (2014), momentum dublu",
                  "exit": "iese din primele 3 sau pierde filtrul absolut (sub SMA 200 sau sub numerar); revizie lunară"},
    "region": {"name": "Regiuni (momentum relativ)", "horizon": 30, "benchmark": "ACWI",
               "evidence": "Asness, Moskowitz & Pedersen (2013), momentum (și valoare) pe țări",
               "exit": "nu mai e pe primul loc sau pierde filtrul absolut; revizie lunară"},
    "stock": {"name": "Acțiuni cu catalizator", "horizon": 90, "benchmark": None,
              "evidence": "Bernard & Thomas (1989), deriva după rezultate; Chan, Jegadeesh & Lakonishok (1996), "
                          "revizuiri; George & Hwang (2004), maximul pe 52 de săptămâni; filtrul de calitate: "
                          "Asness, Frazzini & Pedersen (2019)",
              "exit": "stop 2,5 × ATR sau după ~60 de ședințe (90 de zile calendaristice)"},
    "bvb": {"name": "BVB doi pași (testat)", "horizon": 182, "benchmark": "TVBETETF.RO",
            "evidence": "testul istoric al aplicației: 31 de emitenți BVB, mai 2017 – sept. 2026, exces la 6 luni",
            "exit": "stop 2,5 × ATR sau după 6 luni; treapta fundamentală scade sau momentul iese din banda ≥ 80"},
}
ATR_MULTIPLIER = 2.5


def strategy_name(key):
    return STRATEGIES[key]["name"]


JOURNAL_SHEET = "Idei_jurnal"
JOURNAL_COLUMNS = ["Date", "Symbol", "Instrument", "Strategie", "Pret_intrare", "Stop", "Orizont_zile",
                   "Benchmark", "Semnale"]
JOURNAL_MIN_N = 20


def journal_row(date, symbol, instrument, strategy, entry, stop, horizon_days, benchmark, signals):
    """Rândul de scris în `Idei_jurnal` (date AAAA-LL-ZZ; prețul și stopul cu 4 zecimale: Yahoo dă 50,099998)."""
    return [pd.Timestamp(date).strftime("%Y-%m-%d"), str(symbol).upper(), str(instrument or ""), str(strategy),
            round(float(entry), 4), round(float(stop), 4), int(horizon_days), str(benchmark or ""), str(signals)[:400]]


def parse_journal(values, cell_number):
    """Rândurile foii → (DataFrame, probleme). `cell_number`: cititorul de numere din data/portfolio_sheet."""
    cols = ["Date", "Symbol", "Instrument", "Strategie", "Entry", "Stop", "Horizon", "Benchmark", "Semnale"]
    empty = pd.DataFrame(columns=cols)
    if not values:
        return empty, []
    header = [str(h).strip().lower() for h in values[0]]
    try:
        pos = [header.index(c.lower()) for c in JOURNAL_COLUMNS]
    except ValueError:
        return empty, [f"Foaia {JOURNAL_SHEET} nu are antetul {', '.join(JOURNAL_COLUMNS)}."]
    rows, bad = [], 0
    for raw in values[1:]:
        cell = lambda i: raw[i] if i < len(raw) else None
        date = pd.to_datetime(str(cell(pos[0]) or "").strip(), format="%Y-%m-%d", errors="coerce")
        sym = str(cell(pos[1]) or "").strip().upper()
        entry, stop, hz = cell_number(cell(pos[4])), cell_number(cell(pos[5])), cell_number(cell(pos[6]))
        if pd.isna(date) or not sym or entry is None or entry <= 0 or hz is None or hz <= 0:
            if any(str(c).strip() for c in raw):
                bad += 1
            continue
        rows.append({"Date": date, "Symbol": sym, "Instrument": str(cell(pos[2]) or ""),
                     "Strategie": str(cell(pos[3]) or "").strip(), "Entry": float(entry),
                     "Stop": None if stop is None else float(stop), "Horizon": int(hz),
                     "Benchmark": str(cell(pos[7]) or "").strip().upper(), "Semnale": str(cell(pos[8]) or "")})
    problems = [f"{JOURNAL_SHEET}: {bad} rânduri invalide sărite."] if bad else []
    return (pd.DataFrame(rows, columns=cols) if rows else empty), problems


def already_logged(df, date, symbol, strategy):
    """True dacă există deja rândul (dată, simbol, strategie): o idee se salvează o dată pe zi."""
    if df is None or df.empty:
        return False
    day = pd.Timestamp(date).normalize()
    return bool(((df["Date"] == day) & (df["Symbol"] == str(symbol).upper()) & (df["Strategie"] == strategy)).any())


def evaluate_entry(close, entry_date, entry, stop, horizon_days, today, bench=None):
    """Rezultatul unei idei din jurnal: ieșire la prima închidere ≤ stop (stop fix, cel de la intrare), altfel la
    ultima închidere de la sau înainte de data orizontului; înainte de orizont, poziția e deschisă (evaluată la
    ultima închidere). dict status, exit_date, ret, bench_ret, excess (fracții; None unde lipsesc date)."""
    c = _series(close)
    out = {"status": None, "exit_date": None, "ret": None, "bench_ret": None, "excess": None}
    start = pd.Timestamp(entry_date).normalize()
    path = c[c.index > start]
    if c.empty or entry is None or entry <= 0:
        return out
    # Seria e ajustată cu dividendele: după o distribuire, prețurile vechi scad. Intrarea și stopul salvate se
    # rescalează cu raportul dintre închiderea ajustată de azi a zilei de intrare și prețul salvat, ca randamentul
    # să includă dividendul și stopul să rămână la aceeași distanță procentuală.
    ref = _on_or_before(c, start)
    if ref:
        factor = ref / entry
        entry = ref
        stop = None if stop is None else stop * factor
    horizon_end = start + pd.Timedelta(days=int(horizon_days))
    exit_date, status = None, None
    if stop is not None:
        hit = path[(path <= stop) & (path.index <= horizon_end)]
        if len(hit):
            exit_date, status = hit.index[0], "închisă la stop"
    if exit_date is None:
        if pd.Timestamp(today).normalize() >= horizon_end and len(c[c.index <= horizon_end]):
            exit_date, status = c[c.index <= horizon_end].index[-1], "închisă la orizont"
        elif len(path):
            exit_date, status = path.index[-1], "deschisă"
        else:
            return {**out, "status": "deschisă"}
    exit_px = float(c.loc[exit_date])
    out.update(status=status, exit_date=exit_date, ret=exit_px / entry - 1.0)
    b = _series(bench)
    if len(b):
        b0, b1 = _on_or_before(b, start), _on_or_before(b, exit_date)
        if b0 and b1:
            out["bench_ret"] = b1 / b0 - 1.0
            out["excess"] = out["ret"] - out["bench_ret"]
    return out


def journal_summary(results):
    """Pe strategie, doar ideile închise: n, medie, exces mediu față de benchmark, % pozitive; plus câte sunt deschise.
    `results`: DataFrame cu Strategie, status, ret, excess. `small`: sub 20 de idei închise (fără verdict)."""
    rows = []
    if results is None or results.empty:
        return pd.DataFrame(columns=["Strategie", "Închise", "Deschise", "Randament mediu", "Exces mediu",
                                     "Pozitive", "small"])
    for strat, g in results.groupby("Strategie", sort=True):
        closed = g[g["status"].isin(["închisă la stop", "închisă la orizont"]) & g["ret"].notna()]
        exc = pd.to_numeric(closed["excess"], errors="coerce").dropna()
        rows.append({"Strategie": strat, "Închise": len(closed), "Deschise": int((g["status"] == "deschisă").sum()),
                     "Randament mediu": float(closed["ret"].mean()) if len(closed) else None,
                     "Exces mediu": float(exc.mean()) if len(exc) else None,
                     "Pozitive": float((closed["ret"] > 0).mean()) if len(closed) else None,
                     "small": len(closed) < JOURNAL_MIN_N})
    return pd.DataFrame(rows)
