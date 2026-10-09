"""Indicatori de risc calculați din prețuri. Fără Streamlit, fără rețea."""
import pandas as pd


def beta_benchmark(symbol, currency=None):
    """(simbol Yahoo al benchmarkului, eticheta lui) pentru calculul beta, sau (None, None).

    Benchmarkul trebuie să fie din aceeași piață și în aceeași valută cu acțiunea.
    Pentru BVB nu există indicele BET în Yahoo: se folosește ETF-ul TVBETETF.RO, care
    îl urmărește, și eticheta spune explicit că e un proxy.
    """
    sym = (symbol or "").upper()
    cur = (currency or "").upper()
    if sym.endswith(".RO"):
        return "TVBETETF.RO", "BET, prin ETF-ul TVBETETF.RO (proxy)"
    if sym.endswith(".DE"):
        return "^GDAXI", "DAX"
    if cur == "EUR":
        return "^STOXX50E", "Euro Stoxx 50"
    if cur == "USD":
        return "^GSPC", "S&P 500"
    return None, None


def _weekly_returns(close):
    series = pd.Series(close).dropna()
    if series.empty:
        return series
    index = pd.DatetimeIndex(series.index)
    if index.tz is not None:
        index = index.tz_localize(None)
    series.index = index.normalize()
    # Săptămânile fără nicio tranzacție rămân goale (nu se completează): un randament
    # zero inventat ar trage beta în jos exact la acțiunile nelichide.
    return series.resample("W-FRI").last().pct_change(fill_method=None).dropna()


def beta_weekly(asset_close, bench_close, min_obs=52):
    """Beta = cov(activ, benchmark) / var(benchmark), pe randamente săptămânale.

    Randamentele săptămânale (închidere de vineri) reduc eroarea de tranzacționare
    nesincronizată, care trage în jos beta zilnic la acțiunile puțin lichide (BVB).
    Întoarce {"beta", "n", "start", "end"} sau None dacă sunt mai puțin de `min_obs`
    săptămâni comune ori benchmarkul nu variază. Perioada e toată suprapunerea seriilor.
    """
    if asset_close is None or bench_close is None:
        return None
    joined = pd.concat([_weekly_returns(asset_close), _weekly_returns(bench_close)], axis=1, join="inner").dropna()
    if len(joined) < max(min_obs, 2):
        return None
    asset, bench = joined.iloc[:, 0], joined.iloc[:, 1]
    variance = bench.var()
    if not variance or variance != variance:
        return None
    return {"beta": float(asset.cov(bench) / variance), "n": int(len(joined)),
            "start": joined.index[0], "end": joined.index[-1]}


def _daily_close(close):
    series = pd.Series(close).dropna()
    if series.empty:
        return series
    index = pd.DatetimeIndex(series.index)
    if index.tz is not None:
        index = index.tz_localize(None)
    series.index = index.normalize()
    return series[~series.index.duplicated(keep="last")]


def jensen_alpha(asset_close, bench_close, beta, risk_free, years=1, min_obs=120):
    """Alpha Jensen pe ultimul an: R_activ − [rf + β × (R_benchmark − rf)].

    Randamentele activului și benchmarkului sunt pe aceeași fereastră calendaristică
    (ședințele comune din ultimul an), iar `risk_free` e rata anuală în aceeași valută
    (fracție). Întoarce {"alpha", "asset_return", "bench_return", "n", "start", "end"}
    sau None dacă lipsește beta ori rf, dacă sunt sub `min_obs` ședințe comune sau dacă
    fereastra acoperă mai puțin de ~11 luni (un randament pe 4 luni nu se compară cu
    o rată anuală).
    """
    if beta is None or risk_free is None or beta != beta or risk_free != risk_free:
        return None
    if asset_close is None or bench_close is None:
        return None
    joined = pd.concat([_daily_close(asset_close), _daily_close(bench_close)], axis=1, join="inner").dropna()
    if joined.empty:
        return None
    window = joined.loc[joined.index >= joined.index[-1] - pd.DateOffset(years=years)]
    if len(window) < min_obs or (window.index[-1] - window.index[0]).days < 330 * years:
        return None
    asset_return = float(window.iloc[-1, 0] / window.iloc[0, 0] - 1)
    bench_return = float(window.iloc[-1, 1] / window.iloc[0, 1] - 1)
    alpha = asset_return - (risk_free + beta * (bench_return - risk_free))
    return {"alpha": alpha, "asset_return": asset_return, "bench_return": bench_return,
            "n": int(len(window)), "start": window.index[0], "end": window.index[-1]}
