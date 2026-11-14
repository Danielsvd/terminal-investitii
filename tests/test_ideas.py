"""Testele pentru analytics/ideas.py și data/ucits.py (secțiunea „Idei și strategii”). Valori sintetice,
rezultatele de referință calculate de mână în comentarii."""
import math

import numpy as np
import pandas as pd
import pytest

from analytics import ideas
from data.portfolio_sheet import cell_number
from data.ucits import ucits_for, ucits_symbols


def _monthly_index(n, end="2026-10-01"):
    return pd.date_range(end=end, periods=n, freq="MS")


# --- Strat 1 -------------------------------------------------------------------------------------------------------

def test_inflation_acceleration_constant_rate_is_zero():
    # 0,25% pe lună constant: ritmul pe 3 luni anualizat = ritmul pe 12 luni = 1,0025^12 − 1 → accelerare 0
    idx = _monthly_index(24)
    cpi = pd.Series(100 * 1.0025 ** np.arange(24), index=idx)
    acc = ideas.inflation_acceleration(cpi)
    assert len(acc) == 12
    assert acc.abs().max() < 1e-9


def test_quadrant_q2_with_month_count():
    # Inflația: 0,25%/lună, apoi ultimele 3 luni 0,5%/lună.
    idx = _monthly_index(30)
    steps = [1.0025] * 27 + [1.005] * 3
    cpi = pd.Series(100 * np.cumprod([1.0] + steps[1:]), index=idx)
    # ann3 = 1,005^12 − 1 = 6,1678%; yoy = 1,0025^9 · 1,005^3 − 1 = 3,8154%; accelerare = 2,3524 pp
    ann3 = 1.005 ** 12 - 1
    yoy = 1.0025 ** 9 * 1.005 ** 3 - 1
    # Creșterea: ultimele 2 luni +0,30, înainte −0,30 → cadranul Q2 de 2 luni (luna −3: creștere ↓, inflație ↑ = Q3)
    cfnai = pd.Series([-0.3] * 28 + [0.3, 0.3], index=idx)
    q = ideas.growth_inflation_quadrant(cfnai, cpi)
    assert q["key"] == "Q2"
    assert q["growth"] == pytest.approx(0.3)
    assert q["ann3"] == pytest.approx(ann3)
    assert q["yoy"] == pytest.approx(yoy)
    assert q["accel"] == pytest.approx((ann3 - yoy) * 100)
    assert q["months"] == 2
    assert q["growth_date"] == pd.Timestamp("2026-10-01")


def test_quadrant_border_gives_no_key():
    idx = _monthly_index(30)
    cpi = pd.Series(100 * 1.0025 ** np.arange(30), index=idx)   # accelerare 0 → la graniță
    q = ideas.growth_inflation_quadrant(pd.Series([0.5] * 30, index=idx), cpi)
    assert q["growth_up"] is True and q["inflation_up"] is None and q["key"] is None and q["months"] is None


def test_quadrant_missing_data():
    q = ideas.growth_inflation_quadrant(None, None)
    assert q["key"] is None and q["growth"] is None and q["accel"] is None


def test_alignment():
    assert ideas.alignment("GLD", "Q3") == "aliniat"
    assert ideas.alignment("SPY", "Q3") == "contra"
    assert ideas.alignment("XLV", "Q3") == "neutru"
    assert ideas.alignment("SPY", None) is None


# --- Strat 2 -------------------------------------------------------------------------------------------------------

def _daily(values, end="2026-11-13"):
    return pd.Series(values, index=pd.bdate_range(end=end, periods=len(values)), dtype=float)


def test_cash_return_12m():
    s = _daily([4.0] * 300)
    assert ideas.cash_return_12m(s) == pytest.approx(0.04)
    assert ideas.cash_return_12m(_daily([4.0] * 100)) is None   # nu acoperă 12 luni
    assert ideas.cash_return_12m(None) is None


def test_trend_signal_up_and_down():
    up = _daily(np.linspace(100, 150, 520))
    t = ideas.trend_signal(up, 0.04)
    assert t["state"] == "ÎN TREND" and t["above_sma10m"] and t["beats_cash"]
    assert t["months_above"] >= 10
    down = _daily(np.linspace(150, 100, 520))
    d = ideas.trend_signal(down, 0.04)
    assert d["state"] == "IEȘIT DIN TREND" and d["months_above"] == 0
    assert d["r12"] < 0


def test_trend_signal_mixed_and_missing_cash():
    up = _daily(np.linspace(100, 101, 520))    # +~0,5% pe 12 luni, sub numerar 4%, dar peste media pe 10 luni
    t = ideas.trend_signal(up, 0.04)
    assert t["above_sma10m"] is True and t["beats_cash"] is False and t["state"] == "MIXT"
    assert ideas.trend_signal(up, None)["state"] is None
    assert ideas.trend_signal(None, 0.04)["state"] is None


def test_dual_momentum_table_selects_top_passing():
    closes = {
        "AAA": _daily(np.linspace(100, 160, 520)),
        "BBB": _daily(np.linspace(100, 125, 520)),
        "CCC": _daily(np.linspace(100, 80, 520)),
        "DDD": _daily(np.linspace(100, 101, 520)),   # trece SMA 200, dar nu bate numerarul
        "EEE": pd.Series(dtype=float),
    }
    df = ideas.dual_momentum_table(closes, 0.02, top=3)
    sel = df.set_index("symbol")
    assert list(df["symbol"][:2]) == ["AAA", "BBB"]
    assert sel.loc["AAA", "selected"] and sel.loc["BBB", "selected"]
    assert not sel.loc["CCC", "selected"] and sel.loc["CCC", "passes"] is False
    assert not sel.loc["DDD", "selected"] and sel.loc["DDD", "beats_cash"] is False
    assert sel.loc["EEE", "passes"] is None and pd.isna(sel.loc["EEE", "rank"])


def test_to_usd():
    idx = pd.bdate_range("2026-01-05", periods=3)
    ron = pd.Series([45.0, 46.0, 47.0], index=idx)
    fx = pd.Series([4.5, 4.6], index=idx[:2])      # a treia zi: ultimul curs (ffill)
    usd = ideas.to_usd(ron, fx)
    assert list(usd.round(6)) == [10.0, 10.0, round(47 / 4.6, 6)]


def test_tracking_check():
    rng = np.random.default_rng(1)
    s = _daily(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300))))
    r = ideas.tracking_check(s, s * 0.9)
    assert r["corr"] == pytest.approx(1.0) and r["ok"] is True and r["n"] >= 50
    short = ideas.tracking_check(s.iloc[-40:], s.iloc[-40:])
    assert short["corr"] is None and short["ok"] is None


def test_stress_episodes_counts_and_forward_returns():
    idx = pd.bdate_range("2020-01-01", "2023-12-29")
    vix3m = pd.Series(20.0, index=idx)
    vix = pd.Series(15.0, index=idx)
    ep1, ep2 = pd.Timestamp("2020-03-02"), pd.Timestamp("2021-06-01")
    vix[ep1:ep1 + pd.Timedelta(days=10)] = 25.0
    vix[pd.Timestamp("2020-03-20"):pd.Timestamp("2020-03-24")] = 25.0   # la mai puțin de 20 de ședințe: același episod
    vix[ep2:ep2 + pd.Timedelta(days=3)] = 22.0
    bench = pd.Series(np.linspace(100, 200, len(idx)), index=idx)
    out = ideas.stress_episodes(vix, vix3m, bench)
    assert out["episodes"] == [ep1, ep2]
    assert out["n"] == 2 and out["enough"] is False
    assert out["in_stress"] is False and out["ratio"] == pytest.approx(0.75)
    # randamentul pe 3 luni de la ep1: prețul la 2020-06-02 ÷ prețul la 2020-03-02 − 1
    p = lambda d: float(bench[bench.index <= d].iloc[-1])
    exp1 = p(pd.Timestamp("2020-06-02")) / p(ep1) - 1
    exp2 = p(pd.Timestamp("2021-09-01")) / p(ep2) - 1
    assert out["stats"][3]["mean"] == pytest.approx((exp1 + exp2) / 2)
    assert out["stats"][3]["pos"] == 1.0
    assert out["stats"][3]["base_mean"] is not None


# --- Strat 3 -------------------------------------------------------------------------------------------------------

def test_from_high_and_prefilter():
    vals = list(np.linspace(100, 200, 400)) + list(np.linspace(200, 180, 50))
    s = _daily(vals)
    assert ideas.from_high(s) == pytest.approx(180 / 200 - 1)
    assert ideas.from_high(s.iloc[-100:]) is None
    closes = pd.DataFrame({"UP": _daily(np.linspace(100, 200, 450)), "DOWN": _daily(np.linspace(200, 100, 450)),
                           "UP2": _daily(np.linspace(100, 130, 450))})
    pf = ideas.stock_prefilter(closes, 5)
    assert list(pf["symbol"]) == ["UP", "UP2"]
    assert len(ideas.stock_prefilter(closes, 1)) == 1


def test_earnings_reaction_window():
    idx = pd.bdate_range("2026-10-26", periods=6)   # 26.10 (luni) … 02.11
    close = pd.Series([100, 100, 104, 110, 111, 112], index=idx, dtype=float)
    bench = pd.Series([50, 50, 50.5, 51, 51, 51], index=idx, dtype=float)
    # anunț miercuri 28.10: ultima închidere dinainte = 27.10 (100), prima după = 29.10 (110)
    r = ideas.earnings_reaction(close, bench, pd.Timestamp("2026-10-28 16:05", tz="America/New_York"))
    assert r["stock"] == pytest.approx(0.10)
    assert r["bench"] == pytest.approx(0.02)
    assert r["abnormal"] == pytest.approx(0.08)
    assert ideas.earnings_reaction(close, bench, pd.Timestamp("2026-11-02")) is None   # fără ședință după


def test_last_announcement():
    idx = pd.DatetimeIndex(["2027-01-28 16:00", "2026-10-28 16:00", "2026-07-29 16:00"], tz="America/New_York")
    df = pd.DataFrame({"EPS Estimate": [1.5, 1.2, 1.1], "Reported EPS": [np.nan, 1.32, 1.0],
                       "Surprise(%)": [np.nan, 10.0, np.nan]}, index=idx)
    last, surprise, nxt = ideas.last_announcement(df, pd.Timestamp("2026-11-14"))
    assert last == pd.Timestamp("2026-10-28") and surprise == 10.0 and nxt == pd.Timestamp("2027-01-28")
    df2 = df.drop(columns="Surprise(%)")
    _, s2, _ = ideas.last_announcement(df2, pd.Timestamp("2026-11-14"))
    assert s2 == pytest.approx((1.32 - 1.2) / 1.2 * 100)
    assert ideas.last_announcement(None, pd.Timestamp("2026-11-14")) == (None, None, None)


def test_catalysts():
    today = pd.Timestamp("2026-11-14")
    reac = {"stock": 0.05, "bench": 0.01, "abnormal": 0.04}
    assert ideas.pead_catalyst(8.0, reac, pd.Timestamp("2026-10-28"), today)["passed"] is True
    assert ideas.pead_catalyst(-2.0, reac, pd.Timestamp("2026-10-28"), today)["passed"] is False
    assert ideas.pead_catalyst(8.0, reac, pd.Timestamp("2026-07-01"), today)["passed"] is False   # 136 zile
    assert ideas.pead_catalyst(8.0, None, pd.Timestamp("2026-10-28"), today)["passed"] is None
    et = pd.DataFrame({"current": [2.10, 2.50], "90daysAgo": [2.00, 2.60]}, index=["0y", "+1y"])
    rev, per = ideas.eps_revision(et)
    assert rev == pytest.approx(0.05) and per == "0y"
    assert ideas.revision_catalyst(rev, per)["passed"] is True
    assert ideas.revision_catalyst(0.005, "0y")["passed"] is False
    assert ideas.revision_catalyst(None, None)["passed"] is None
    et_neg = pd.DataFrame({"current": [-0.5], "90daysAgo": [-1.0]}, index=["0y"])
    assert ideas.eps_revision(et_neg)[0] == pytest.approx(0.5)     # pierdere mai mică: (−0,5 − (−1)) / 1
    assert ideas.high_catalyst(-0.03)["passed"] is True
    assert ideas.high_catalyst(-0.08)["passed"] is False
    assert ideas.high_catalyst(None)["passed"] is None
    s = ideas.catalyst_summary([{"passed": True}, {"passed": None}, {"passed": True}])
    assert s == {"passed": 2, "evaluable": 2, "is_idea": True}
    assert ideas.catalyst_summary([{"passed": True}, {"passed": False}, {"passed": None}])["is_idea"] is False


def test_quality_filter():
    ok = ideas.quality_filter("safe", 6, 9, 1.5, False)
    assert ok["excluded"] is False and [c["passed"] for c in ok["checks"]] == [True, True, True]
    assert ideas.quality_filter("distress", 6, 9, 1.5, False)["excluded"] is True
    assert ideas.quality_filter("safe", 3, 9, 1.5, False)["excluded"] is True
    assert ideas.quality_filter("safe", 3, 6, 1.5, False)["excluded"] is False      # sub 7 criterii: N/A
    assert ideas.quality_filter("safe", 6, 9, 4.5, False)["excluded"] is True
    fin = ideas.quality_filter(None, 5, 7, None, True)
    assert fin["excluded"] is False and fin["checks"][0]["passed"] is None
    none = ideas.quality_filter(None, None, 0, None, False)
    assert none["excluded"] is False and all(c["passed"] is None for c in none["checks"])


# --- Strat 4 -------------------------------------------------------------------------------------------------------

def test_events():
    rows = [{"key": "cpi"}, {"key": "houst"}, {"key": "fomc"}]
    assert [r["key"] for r in ideas.high_impact_events(rows)] == ["cpi", "fomc"]
    assert ideas.earnings_in_days(pd.Timestamp("2026-11-20"), pd.Timestamp("2026-11-14")) == 6
    assert ideas.earnings_in_days(None, pd.Timestamp("2026-11-14")) is None


# --- Dimensionare ------------------------------------------------------------------------------------------------

def test_position_size():
    # 100.000 lei, 1% = 1.000 lei; (100 − 95) USD × 4,5 = 22,5 lei/unitate → 44 unități (44,44 rotunjit în jos)
    r = ideas.position_size(100_000, 100.0, 95.0, 4.5)
    assert r["qty"] == 44 and r["risk_ron"] == pytest.approx(990.0) and r["value_ron"] == pytest.approx(19_800.0)
    assert r["weight"] == pytest.approx(0.198) and r["capped"] is False
    # stop foarte strâns: 444 unități > plafonul de 25% (25.000 lei ÷ 450 = 55)
    c = ideas.position_size(100_000, 100.0, 99.5, 4.5)
    assert c["qty"] == 55 and c["capped"] is True
    assert ideas.position_size(100_000, 100.0, 101.0, 4.5)["qty"] is None
    assert ideas.position_size(None, 100.0, 95.0, 4.5)["reason"]
    assert ideas.position_size(1_000, 5_000.0, 4_000.0, 4.5)["qty"] is None     # nici o unitate


# --- Jurnal --------------------------------------------------------------------------------------------------------

def test_journal_roundtrip_and_dedupe():
    row = ideas.journal_row("2026-11-14", "xlk", "ZPDT.DE", "Rotație sectoare SUA", 250.123456789, 230.5, 30,
                            "SPY", "mom 12-1 +25%")
    assert row[:4] == ["2026-11-14", "XLK", "ZPDT.DE", "Rotație sectoare SUA"] and row[4] == 250.1235 and ideas.journal_row("2026-11-14", "EL.RO", "", "x", 50.099998, 48.756601, 182, "", "")[4:6] == [50.1, 48.7566]
    values = [ideas.JOURNAL_COLUMNS, row, ["2026-11-15", "", "", "x", "1", "", "5", "", ""], ["", "", ""]]
    df, problems = ideas.parse_journal(values, cell_number)
    assert len(df) == 1 and df.iloc[0]["Horizon"] == 30 and df.iloc[0]["Stop"] == 230.5
    assert problems == ["Idei_jurnal: 1 rânduri invalide sărite."]
    assert ideas.already_logged(df, "2026-11-14", "XLK", "Rotație sectoare SUA")
    assert not ideas.already_logged(df, "2026-11-14", "XLK", "Trend clase de active")
    bad, p = ideas.parse_journal([["Date", "Symbol"]], cell_number)
    assert bad.empty and p


def test_evaluate_entry_stop_horizon_open():
    idx = pd.bdate_range("2026-01-05", periods=40)
    close = pd.Series(np.linspace(100, 139, 40), index=idx)
    bench = pd.Series(np.linspace(50, 55, 40), index=idx)
    today = idx[-1]
    # orizont 14 zile de la 05.01 → 19.01; ultima închidere ≤ 19.01 = 19.01 (index 10, preț 110)
    h = ideas.evaluate_entry(close, idx[0], 100.0, 90.0, 14, today, bench)
    assert h["status"] == "închisă la orizont" and h["exit_date"] == pd.Timestamp("2026-01-19")
    assert h["ret"] == pytest.approx(0.10)
    assert h["bench_ret"] == pytest.approx(bench.iloc[10] / 50 - 1)
    # stop atins: scădere sub 95 în a treia zi
    s_close = close.copy()
    s_close.iloc[2] = 94.0
    s = ideas.evaluate_entry(s_close, idx[0], 100.0, 95.0, 30, today, bench)
    assert s["status"] == "închisă la stop" and s["ret"] == pytest.approx(-0.06)
    o = ideas.evaluate_entry(close, idx[0], 100.0, 90.0, 365, today, None)
    assert o["status"] == "deschisă" and o["ret"] == pytest.approx(0.39) and o["excess"] is None


def test_journal_summary():
    res = pd.DataFrame({"Strategie": ["A", "A", "A", "B"],
                        "status": ["închisă la stop", "închisă la orizont", "deschisă", "deschisă"],
                        "ret": [-0.05, 0.15, 0.02, 0.01], "excess": [-0.06, 0.10, 0.0, None]})
    s = ideas.journal_summary(res).set_index("Strategie")
    assert s.loc["A", "Închise"] == 2 and s.loc["A", "Deschise"] == 1
    assert s.loc["A", "Randament mediu"] == pytest.approx(0.05)
    assert s.loc["A", "Exces mediu"] == pytest.approx(0.02)
    assert s.loc["A", "Pozitive"] == 0.5 and s.loc["A", "small"]
    assert s.loc["B", "Închise"] == 0 and pd.isna(s.loc["B", "Randament mediu"])


# --- UCITS -------------------------------------------------------------------------------------------------------

def test_ucits_map():
    assert ucits_for("spy")[0] == "SXR8.DE"
    assert ucits_for("UUP")[0] is None
    assert ucits_for("EXV1.DE")[0] == "EXV1.DE"
    assert ucits_for("ZZZ") == (None, "fără corespondență în tabel")
    assert ucits_symbols(["SPY", "UUP", "SPY", "XLK"]) == ["SXR8.DE", "ZPDT.DE"]


def test_no_streamlit_in_pure_modules():
    import inspect
    from data import ucits
    for mod in (ideas, ucits):
        src = inspect.getsource(mod)
        assert "import streamlit" not in src and "requests" not in src


def test_evaluate_entry_rescales_after_dividend():
    # Intrare salvată la 100 (stop 95); după un dividend, seria ajustată arată 98 în ziua intrării.
    # Factor 0,98: stopul devine 93,1; ieșirea la orizont 107,8 → randament 107,8 / 98 − 1 = 10%.
    idx = pd.bdate_range("2026-01-05", periods=15)
    close = pd.Series([98.0] + [100.0] * 9 + [107.8] * 5, index=idx)
    close.iloc[3] = 94.0          # peste stopul rescalat (93,1): nu se iese
    r = ideas.evaluate_entry(close, idx[0], 100.0, 95.0, 14, idx[-1], None)
    assert r["status"] == "închisă la orizont" and r["ret"] == pytest.approx(0.10)


def test_ucits_currency_and_strategies():
    from data.ucits import ucits_currency, UCITS
    assert ucits_currency("SXR8.DE") == "EUR" and ucits_currency("SGLN.L") == "USD"
    assert ucits_currency("TVBETETF.RO") == "RON" and ucits_currency("SPY") == "USD" and ucits_currency(None) is None
    for sym, (u, _) in UCITS.items():
        assert u is None or ucits_currency(u) in ("EUR", "USD")
    for key, s in ideas.STRATEGIES.items():
        assert s["horizon"] > 0 and s["evidence"] and s["exit"]


def test_details_use_romanian_format():
    today = pd.Timestamp("2026-11-14")
    d = ideas.pead_catalyst(5.0, {"stock": 0.0, "bench": 0.003, "abnormal": -0.003}, pd.Timestamp("2026-10-25"), today)
    assert "surpriză +5,0%" in d["detail"] and "reacție față de piață -0,3 pp" in d["detail"]
    assert "+4,5% în 90 de zile" in ideas.revision_catalyst(0.045, "0y")["detail"]
    assert ideas.high_catalyst(-0.018)["detail"].startswith("-1,8%")
    assert ideas.quality_filter("safe", 6, 9, 5.3419, False)["checks"][2]["detail"] == "5,34×"
