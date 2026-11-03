"""Testul istoric al scorului fundamental SUA/UE. Funcții pure (fără rețea, fără st.*).

La fiecare sfârșit de lună se reconstruiește, doar din ce se știa atunci, aceeași intrare ca în
aplicație (`global_fundamental_inputs` din app.py) și se calculează scorul cu aceeași funcție
(`analytics.global_fundamental.fundamental_score`). Diferențele față de aplicație, toate spuse:

- Situațiile sunt doar cele anuale din Yahoo (~4 ani fiscali). Un an fiscal devine „cunoscut”
  după data de încheiere + 90 de zile (decalaj de publicare, prudent: 10-K se depune în 60–90 de zile).
- FCF, EBITDA, EBIT și dobânzile sunt pe ultimul an fiscal publicat (aplicația folosește TTM când are
  4 trimestre). Yahoo dă doar ~5 trimestre, insuficient pentru istoric.
- P/E, P/BV și capitalizarea se calculează din prețul de la dată × acțiunile din situații, nu din
  `info`. ROE la bănci = profit net / capital propriu de la sfârșitul anului (aplicația ia ROE din `info`).
- Beta: săptămânal, pe ultimii 2 ani dinaintea datei, față de SPY (aplicația ia beta din Yahoo).
- Rata fără risc: ^TNX la dată. AAA: media lunară Moody's (FRED) din ultima lună publicată
  (luna M devine cunoscută la începutul lunii M+1).
- Prețul pentru evaluare e închiderea neajustată pentru dividende (ajustată doar pentru split-uri, ca
  în Yahoo), iar acțiunile din situații se aduc pe aceeași bază de split (vezi `split_adjust_shares`).
  Randamentele viitoare se calculează pe prețuri ajustate și pentru dividende.

Situațiile din Yahoo pot fi retratate de companie după publicare: un mic avantaj de informație
(look-ahead) pe care testul nu-l poate elimina.
"""
import math

import pandas as pd

from analytics import fundamentals as fund
from analytics.global_fundamental import fundamental_score
from analytics.risk import beta_weekly

PUBLICATION_LAG = pd.DateOffset(days=90)
BETA_WINDOW = pd.DateOffset(years=2)
BETA_MIN_WEEKS = 52
SHARE_ROWS = tuple(dict.fromkeys(fund.DILUTED_SHARES_ROWS + fund.BASIC_SHARES_ROWS + fund.SHARES_COUNT_ROWS))
SPLIT_MATCH_TOLERANCE = 0.25      # acțiunile din doi ani diferiți „se potrivesc” dacă diferă cu < 25%


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


# --- Ce se știa la o dată ------------------------------------------------------

def statements_as_of(df, date, lag=PUBLICATION_LAG):
    """Coloanele (anii fiscali) publicate până la `date`: data de încheiere + `lag` ≤ `date`.
    Cea mai recentă prima. None dacă nu există niciuna."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    try:
        cols = pd.to_datetime(df.columns)
    except (TypeError, ValueError):
        return None
    keep = [i for i, c in enumerate(cols) if c + lag <= pd.Timestamp(date)]
    if not keep:
        return None
    out = df.iloc[:, keep]
    order = pd.to_datetime(out.columns).argsort()[::-1]
    return out.iloc[:, order]


def value_as_of(series, date, publication_lag=None):
    """Ultima valoare a seriei cunoscută la `date`. Cu `publication_lag`, observația de la data t
    devine cunoscută la t + lag (ex. media lunară FRED: index = începutul lunii, lag = 1 lună)."""
    if series is None or len(series) == 0:
        return None
    s = pd.Series(series).dropna()
    idx = pd.DatetimeIndex(s.index)
    if publication_lag is not None:
        idx = idx + publication_lag
    s.index = idx
    part = s.loc[:pd.Timestamp(date)]
    return None if part.empty else float(part.iloc[-1])


# --- Split-uri -----------------------------------------------------------------

def split_factor_after(splits, date):
    """Produsul split-urilor de după `date` (2.0 = 2 la 1). 1.0 fără split-uri."""
    if splits is None or len(splits) == 0:
        return 1.0
    s = pd.Series(splits).dropna()
    idx = pd.DatetimeIndex(s.index)
    s.index = idx.tz_localize(None) if idx.tz is not None else idx
    after = s[(s.index > pd.Timestamp(date)) & (s > 0)]
    return float(after.prod()) if len(after) else 1.0


def _first_share_value(df, col):
    for name in SHARE_ROWS:
        if name in df.index:
            v = pd.to_numeric(df.loc[name].iloc[col], errors="coerce")
            if pd.notna(v) and v > 0:
                return float(v)
    return None


def share_basis(income, balance, splits):
    """Cum raportează Yahoo acțiunile din anii dinaintea unui split: „reported” (cifra din anul acela,
    trebuie înmulțită cu split-urile ulterioare ca să se potrivească cu prețul ajustat la split),
    „restated” (deja pe baza de azi) sau „none” (fără split în perioada situațiilor).
    None dacă nu se poate decide; atunci anii dinaintea split-ului nu se folosesc."""
    df = fund._sorted_desc(income) if fund.stmt_value(income, SHARE_ROWS) is not None else fund._sorted_desc(balance)
    if df is None:
        return None
    dates = pd.to_datetime(df.columns)
    factors = [split_factor_after(splits, d) for d in dates]
    if all(abs(f - 1.0) < 1e-9 for f in factors):
        return "none"
    latest = _first_share_value(df, 0)
    for col in range(1, df.shape[1]):
        if factors[col] / factors[0] > 1.0 + 1e-9:
            old = _first_share_value(df, col)
            if latest is None or old is None:
                return None
            ratio = factors[col] / factors[0]
            as_reported = abs(old * ratio / latest - 1) < SPLIT_MATCH_TOLERANCE
            restated = abs(old / latest - 1) < SPLIT_MATCH_TOLERANCE
            if as_reported and not restated:
                return "reported"
            if restated and not as_reported:
                return "restated"
            return None
    return "none"


def split_adjust_shares(df, splits):
    """Copia situației cu rândurile de acțiuni înmulțite, pe fiecare an, cu split-urile ulterioare
    (pentru modul „reported”). Profitul, capitalul etc. nu se schimbă."""
    if df is None or df.empty:
        return df
    out = df.copy()
    for j, c in enumerate(pd.to_datetime(out.columns)):
        f = split_factor_after(splits, c)
        if abs(f - 1.0) < 1e-9:
            continue
        for name in SHARE_ROWS:
            if name in out.index:
                out.iloc[out.index.get_loc(name), j] = pd.to_numeric(out.iloc[out.index.get_loc(name), j], errors="coerce") * f
    return out


# --- Beta la o dată ------------------------------------------------------------

def beta_as_of(close_adj, bench_adj, date, window=BETA_WINDOW, min_weeks=BETA_MIN_WEEKS):
    """Beta săptămânal pe `window` înainte de `date`. None sub `min_weeks` săptămâni sau beta ≤ 0."""
    if close_adj is None or bench_adj is None:
        return None
    d = pd.Timestamp(date)
    a = pd.Series(close_adj).loc[d - window:d]
    b = pd.Series(bench_adj).loc[d - window:d]
    res = beta_weekly(a, b, min_obs=min_weeks)
    return res["beta"] if res is not None and res["beta"] > 0 else None


# --- Intrarea scorului la o dată -----------------------------------------------

def inputs_at(date, price, income, balance, cashflow, sector, symbol, rf, beta, aaa_pct):
    """Intrarea pentru `fundamental_score`, doar din situațiile publicate până la `date`.

    `price`: închiderea la dată (pe aceeași bază de split ca acțiunile din situații).
    `rf`: fracție (0,042); `aaa_pct`: procente (5,3). Întoarce (inp, fiscal_year) sau (None, motiv).
    """
    inc = statements_as_of(income, date)
    bal = statements_as_of(balance, date)
    cf = statements_as_of(cashflow, date)
    if inc is None or bal is None:
        return None, "nicio situație anuală publicată la dată"
    if not _num(price) or price <= 0:
        return None, "fără preț"
    fy = fund.stmt_date(bal)
    shares, _ = fund.diluted_shares(inc, bal)
    equity = fund.stmt_value(bal, fund.EQUITY_ROWS)
    ni = fund.stmt_value(inc, fund.NET_INCOME_ROWS)
    market_cap = price * shares if _num(shares) and shares > 0 else None
    pbv = market_cap / equity if _num(market_cap) and _num(equity) and equity > 0 else None
    eps = ni / shares if _num(ni) and _num(shares) and shares > 0 else None
    pe = price / eps if _num(eps) and eps > 0 else None
    financial = fund.is_financial_issuer(sector, symbol)
    inp = {"financial": financial, "price": price, "pbv": pbv, "rf": rf, "beta": beta,
           "rf_label": "^TNX la dată", "beta_label": "săptămânal 2 ani față de SPY"}
    if financial:
        roe = ni / equity if _num(ni) and _num(equity) and equity > 0 else None
        inp.update(roe=roe, ke=fund.cost_of_equity(rf, beta))
        return inp, fy

    dcf_in = fund.dcf_inputs(inc, bal, cf)
    wacc_res = fund.wacc(rf, beta, market_cap, dcf_in["total_debt"], dcf_in["interest_expense"], dcf_in["tax_rate"])
    fcf, basis = dcf_in["fcf"], "ultimul an fiscal"
    if (fund.fcf_distortion_warning(dcf_in["cfo"], dcf_in["capex"], dcf_in["fcf"], dcf_in["fcf_average"])
            and dcf_in["fcf_average"] is not None):
        fcf, basis = dcf_in["fcf_average"], f"media ultimilor {dcf_in['fcf_average_years']} ani fiscali"
    lev = fund.leverage_ratios(inc, bal)
    variant = fund.altman_variant(sector, symbol)
    alt = {}
    if variant is not None:
        az = fund.altman_z(inc, bal, market_cap=market_cap)
        value = az["z"] if variant == "Z" else az["z2"]
        if value is None and variant == "Z" and az["z2"] is not None:
            variant, value = "Z''", az["z2"]
        if value is not None:
            alt = {"value": value, "variant": variant, "zone": fund.altman_zone(value, variant)}
    inp.update(dcf_na=fund.dcf_not_applicable_reason(sector, None, symbol),
               wacc=wacc_res["wacc"] if wacc_res else None, fcf=fcf, fcf_basis=basis,
               net_debt=dcf_in["net_debt"], shares=shares, pe=pe, eps=eps, aaa=aaa_pct,
               aaa_label="FRED AAA, ultima lună publicată",
               piotroski=fund.piotroski(inc, bal, cf), altman=alt,
               ebitda=lev["ebitda"], coverage=lev["interest_coverage"])
    return inp, fy


def score_row(symbol, date, inp, fiscal_year):
    """Rândul de rezultat: scorul, pilonii și componentele principale (valori brute, pentru analiză)."""
    res = fundamental_score(inp)
    row = {"symbol": symbol, "date": pd.Timestamp(date).date().isoformat(),
           "fiscal_year": None if fiscal_year is None else pd.Timestamp(fiscal_year).date().isoformat(),
           "financial": res["financial"], "fund_score": res["score"], "label": res["label"],
           "n_pillars": sum(1 for p in res["pillars"] if p["score"] is not None)}
    for p in res["pillars"]:
        row[f"p_{p['key']}"] = None if p["score"] is None else p["score"] * 100
        for c in p["components"]:
            if not c.get("info"):
                row[f"c_{c['label']}"] = None if c["score"] is None else c["score"] * 100
    rev = res.get("reverse_dcf")
    row["reverse_dcf_growth"] = None if not rev else rev["growth"]
    row["pe"], row["pbv"] = inp.get("pe"), inp.get("pbv")
    return row


__all__ = ["statements_as_of", "value_as_of", "split_factor_after", "share_basis", "split_adjust_shares",
           "beta_as_of", "inputs_at", "score_row", "PUBLICATION_LAG"]
