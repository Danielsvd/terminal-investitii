"""Calcule pe lanțul de opțiuni. Funcții pure."""
import numpy as np
import pandas as pd


def max_pain(calls, puts):
    """Prețul Max Pain pentru o singură expirare.

    Pentru fiecare strike K candidat se calculează valoarea totală pe care
    opțiunile ar avea-o la expirare dacă prețul ar închide la K:
        sum(OI_call * max(K - strike, 0)) + sum(OI_put * max(strike - K, 0))
    Max Pain este strike-ul la care această sumă este minimă (cumpărătorii de
    opțiuni pierd cel mai mult).

    `calls` și `puts` sunt DataFrame-uri cu coloanele `strike` și `openInterest`.
    Întoarce None dacă nu există open interest.
    """
    def _clean(df):
        if df is None or len(df) == 0:
            return np.array([]), np.array([])
        strikes = pd.to_numeric(df["strike"], errors="coerce")
        oi = pd.to_numeric(df["openInterest"], errors="coerce").fillna(0)
        mask = strikes.notna()
        return strikes[mask].to_numpy(dtype="float64"), oi[mask].to_numpy(dtype="float64")

    c_strike, c_oi = _clean(calls)
    p_strike, p_oi = _clean(puts)
    if c_oi.sum() + p_oi.sum() <= 0:
        return None

    candidates = np.unique(np.concatenate([c_strike, p_strike]))
    best_k, best_pain = None, None
    for k in candidates:
        pain = (c_oi * np.maximum(k - c_strike, 0)).sum() + (p_oi * np.maximum(p_strike - k, 0)).sum()
        if best_pain is None or pain < best_pain:
            best_k, best_pain = float(k), pain
    return best_k


# ---------------------------------------------------------------------------
# Lanțul de opțiuni: expirarea, IV la bani, greeks, put/call, GEX, IV Rank.
# Toate funcțiile sunt pure (fără rețea, fără st.*). O intrare lipsă dă None.
# Convenții:
#   - impliedVolatility vine de la Yahoo ca fracție (0,25 = 25%).
#   - T în ani = zile calendaristice până la expirare / 365.
#   - r și q sunt rate anuale continue, ca fracție (0,05 = 5%).
# ---------------------------------------------------------------------------
import math
from datetime import date as _date

# Sub acest prag IV-ul raportat de Yahoo e un substituent (de ex. 1e-5), nu o cotație.
MIN_VALID_IV = 0.01
# Sub acest număr de observații IV Rank / Percentile nu se calculează.
MIN_IV_OBS = 20


def _num(x):
    """Float finit sau None."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def choose_expiration(expirations, today, min_days=20, max_days=45, target=30):
    """Expirarea din intervalul [min_days, max_days] zile calendaristice cea mai
    apropiată de `target`. La egalitate, cea mai apropiată în timp.

    `expirations`: șiruri 'AAAA-LL-ZZ' (formatul yfinance `Ticker.options`) sau date.
    Întoarce (expirarea ca șir, zile) sau None dacă nicio expirare nu e în interval.
    """
    best = None
    for e in expirations or []:
        try:
            d = e if isinstance(e, _date) else _date.fromisoformat(str(e)[:10])
        except ValueError:
            continue
        days = (d - today).days
        if min_days <= days <= max_days:
            key = (abs(days - target), days)
            if best is None or key < best[0]:
                best = (key, d.isoformat(), days)
    return None if best is None else (best[1], best[2])


def _norm_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _d1_d2(S, K, T, r, q, sigma):
    vals = [_num(v) for v in (S, K, T, r, q, sigma)]
    if any(v is None for v in vals):
        return None
    S, K, T, r, q, sigma = vals
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return None
    sq = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / sq
    return d1, d1 - sq


def bs_price(S, K, T, r, q, sigma, kind):
    """Prețul Black-Scholes-Merton (dividend continuu q). kind: 'call' sau 'put'."""
    d = _d1_d2(S, K, T, r, q, sigma)
    if d is None or kind not in ("call", "put"):
        return None
    d1, d2 = d
    S, K, T, r, q = float(S), float(K), float(T), float(r), float(q)
    if kind == "call":
        return S * math.exp(-q * T) * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * math.exp(-q * T) * _norm_cdf(-d1)


def bs_greeks(S, K, T, r, q, sigma, kind):
    """Greeks Black-Scholes-Merton pentru un contract (o acțiune suport).

    delta: variația prețului opțiunii la +1 în prețul suport.
    gamma: variația deltei la +1 în prețul suport.
    theta_day: variația prețului într-o zi calendaristică (theta anual / 365).
    vega_pt: variația prețului la +1 punct procentual de volatilitate.
    Întoarce None dacă vreo intrare lipsește sau e invalidă.
    """
    d = _d1_d2(S, K, T, r, q, sigma)
    if d is None or kind not in ("call", "put"):
        return None
    d1, d2 = d
    S, K, T, r, q, sigma = (float(v) for v in (S, K, T, r, q, sigma))
    eq, er = math.exp(-q * T), math.exp(-r * T)
    pdf = _norm_pdf(d1)
    gamma = eq * pdf / (S * sigma * math.sqrt(T))
    vega_pt = S * eq * pdf * math.sqrt(T) / 100.0
    common = -S * eq * pdf * sigma / (2.0 * math.sqrt(T))
    if kind == "call":
        delta = eq * _norm_cdf(d1)
        theta = common - r * K * er * _norm_cdf(d2) + q * S * eq * _norm_cdf(d1)
    else:
        delta = eq * (_norm_cdf(d1) - 1.0)
        theta = common + r * K * er * _norm_cdf(-d2) - q * S * eq * _norm_cdf(-d1)
    return {"delta": delta, "gamma": gamma, "theta_day": theta / 365.0, "vega_pt": vega_pt}


def _quoted(df):
    """Rândurile cu strike, bid > 0 și IV valid, sortate după strike."""
    if df is None or len(df) == 0 or "strike" not in df or "impliedVolatility" not in df:
        return pd.DataFrame(columns=["strike", "iv"])
    out = pd.DataFrame({
        "strike": pd.to_numeric(df["strike"], errors="coerce"),
        "iv": pd.to_numeric(df["impliedVolatility"], errors="coerce"),
        "bid": pd.to_numeric(df["bid"], errors="coerce") if "bid" in df else np.nan,
    })
    mask = out["strike"].notna() & (out["iv"] >= MIN_VALID_IV) & (out["bid"] > 0)
    return out.loc[mask, ["strike", "iv"]].sort_values("strike").reset_index(drop=True)


def _iv_at(df, spot):
    """IV interpolat liniar la spot între strike-urile care îl încadrează.
    None dacă spotul nu e încadrat de cotații valide."""
    q = _quoted(df)
    if q.empty:
        return None
    exact = q[q["strike"] == spot]
    if not exact.empty:
        return float(exact["iv"].mean())
    below, above = q[q["strike"] < spot], q[q["strike"] > spot]
    if below.empty or above.empty:
        return None
    k1, v1 = below["strike"].iloc[-1], below["iv"].iloc[-1]
    k2, v2 = above["strike"].iloc[0], above["iv"].iloc[0]
    return float(v1 + (v2 - v1) * (spot - k1) / (k2 - k1))


def atm_iv(calls, puts, spot):
    """Volatilitatea implicită la bani: media IV call și IV put interpolate la spot.

    Se folosesc doar cotațiile cu bid > 0 și IV >= MIN_VALID_IV. Dacă doar o
    parte (call sau put) încadrează spotul, se folosește doar ea.
    Întoarce (iv, nr_părți_folosite) sau (None, 0).
    """
    s = _num(spot)
    if s is None or s <= 0:
        return None, 0
    vals = [v for v in (_iv_at(calls, s), _iv_at(puts, s)) if v is not None]
    if not vals:
        return None, 0
    return float(sum(vals) / len(vals)), len(vals)


def put_call_ratio(calls, puts, column):
    """Σ column pe put-uri / Σ column pe call-uri ('volume' sau 'openInterest').
    Valorile lipsă nu contează. None dacă numitorul e 0 sau lipsește."""
    def _sum(df):
        if df is None or len(df) == 0 or column not in df:
            return 0.0
        return float(pd.to_numeric(df[column], errors="coerce").clip(lower=0).fillna(0).sum())
    c, p = _sum(calls), _sum(puts)
    if c <= 0:
        return None
    return p / c


def gamma_exposure(calls, puts, spot, T, r, q):
    """Expunerea netă la gamma (GEX), în unități monetare per mișcare de 1% a suportului.

    Pe strike: gamma_BS × OI × 100 × S² × 0,01, cu IV-ul acelui strike.
    Semn: call +, put −. Convenția presupune că dealerii sunt long call și
    short put; este o estimare, nu poziția reală a dealerilor.
    Contractele fără IV valid sau fără open interest nu intră.
    Întoarce (gex, nr_contracte_folosite) sau (None, 0).
    """
    S = _num(spot)
    if S is None or S <= 0:
        return None, 0
    total, n = 0.0, 0
    for df, sign in ((calls, 1.0), (puts, -1.0)):
        if df is None or len(df) == 0:
            continue
        strikes = pd.to_numeric(df.get("strike"), errors="coerce")
        ivs = pd.to_numeric(df.get("impliedVolatility"), errors="coerce")
        ois = pd.to_numeric(df.get("openInterest"), errors="coerce")
        for k, iv, oi in zip(strikes, ivs, ois):
            if not (pd.notna(k) and pd.notna(iv) and pd.notna(oi)) or iv < MIN_VALID_IV or oi <= 0:
                continue
            g = bs_greeks(S, k, T, r, q, iv, "call")
            if g is None:
                continue
            total += sign * g["gamma"] * oi * 100.0 * S * S * 0.01
            n += 1
    return (total, n) if n else (None, 0)


def expected_move(spot, iv, days):
    """Mișcarea de ±1 abatere standard implicată până la expirare: S × IV × √(zile/365)."""
    S, v, d = _num(spot), _num(iv), _num(days)
    if S is None or v is None or d is None or S <= 0 or v <= 0 or d <= 0:
        return None
    return S * v * math.sqrt(d / 365.0)


def realized_vol(close, as_of=None, days=30):
    """Volatilitatea realizată anualizată (√252) din randamentele logaritmice zilnice
    din ultimele `days` zile calendaristice. None sub 15 randamente."""
    if close is None or len(close) == 0:
        return None
    s = pd.to_numeric(pd.Series(close), errors="coerce").dropna()
    s = s[s > 0].sort_index()
    if s.empty:
        return None
    end = pd.Timestamp(as_of) if as_of is not None else s.index[-1]
    s = s[(s.index > end - pd.DateOffset(days=days)) & (s.index <= end)]
    rets = np.log(s / s.shift(1)).dropna()
    if len(rets) < 15:
        return None
    return float(rets.std(ddof=1) * math.sqrt(252))


def iv_rank_percentile(history, current_iv, as_of, min_obs=MIN_IV_OBS):
    """IV Rank și IV Percentile din istoricul IV la bani pe ultimul an calendaristic.

    `history`: Series indexată pe dată, cu IV ca fracție (din foaia IV_istoric).
    IV Rank = (IV curent − min) / (max − min) × 100.
    IV Percentile = % din zilele istoricului cu IV < IV curent.
    Sub `min_obs` observații (sau max = min) rank și percentile sunt None.
    Întoarce {'rank', 'percentile', 'n', 'first', 'last'}.
    """
    out = {"rank": None, "percentile": None, "n": 0, "first": None, "last": None}
    cur = _num(current_iv)
    if history is None or len(history) == 0:
        return out
    h = pd.to_numeric(pd.Series(history), errors="coerce").dropna()
    h.index = pd.to_datetime(h.index)
    end = pd.Timestamp(as_of)
    h = h[(h.index > end - pd.DateOffset(years=1)) & (h.index <= end) & (h >= MIN_VALID_IV)].sort_index()
    out["n"] = int(len(h))
    if len(h):
        out["first"], out["last"] = h.index[0].date(), h.index[-1].date()
    if cur is None or len(h) < min_obs:
        return out
    lo, hi = float(h.min()), float(h.max())
    if hi > lo:
        out["rank"] = max(0.0, min(100.0, (cur - lo) / (hi - lo) * 100.0))
    out["percentile"] = float((h < cur).mean() * 100.0)
    return out
