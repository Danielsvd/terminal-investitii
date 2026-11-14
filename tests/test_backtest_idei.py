"""tools/backtest_idei.py pe date sintetice (fără rețea)."""
import numpy as np
import pandas as pd
import pytest

from analytics import ideas
from tools import backtest_idei as bt


def _closes():
    idx = pd.bdate_range("2010-01-01", "2014-12-31")
    n = len(idx)
    up = lambda r: pd.Series(100 * (1 + r) ** np.arange(n), index=idx)       # creștere zilnică constantă
    out = {"SPY": up(0.0004), "GLD": up(-0.0003), "IEF": up(0.0002)}
    for i, s in enumerate(ideas.US_SECTORS):
        out[s] = up(0.0008 - 0.0001 * i)                                       # XLK cel mai puternic, apoi XLC...
    return out


def test_run_strategies_hand_checked():
    closes = _closes()
    cash = pd.Series(2.4, index=pd.bdate_range("2009-01-01", "2014-12-31"))   # 2,4%/an → 0,2% pe lună
    res = bt.run_strategies(closes, cash, "2012-01-01", "2014-12-31")
    assert len(res) == 35                                                     # 36 de sfârșituri de lună → 35 de luni
    first = res.iloc[0]
    # A: SPY (+~10,6%/an) și IEF (+~5,2%/an) bat numerarul 2,4% și sunt peste media pe 10 luni; GLD scade → numerar
    assert first["n_trend"] == "2/3"
    # B: primele 3 după momentum sunt XLK, XLC, XLY (ordinea din US_SECTORS, creșteri descrescătoare)
    assert first["sectors"] == "XLK XLC XLY"
    assert first["cash"] == pytest.approx(0.002)
    # Randamentul lunii 1 la A, recalculat: 1/3 SPY + 1/3 IEF + 1/3 numerar − 0,1% × rotația (2/3 la prima lună)
    d0, d1 = res["date"].iloc[0], res["date"].iloc[1]
    r = lambda s: closes[s].loc[:d1].iloc[-1] / closes[s].loc[:d0].iloc[-1] - 1
    exp_a = (r("SPY") + r("IEF")) / 3 + 0.002 / 3 - 0.001 * (2 / 3)
    assert first["trend"] == pytest.approx(exp_a)
    exp_b = (r("XLK") + r("XLC") + r("XLY")) / 3 - 0.001 * 1.0
    assert first["sector"] == pytest.approx(exp_b)
    # luna a doua: aceleași poziții, fără rotație, fără cost
    second = res.iloc[1]
    d2 = res["date"].iloc[2]
    r2 = lambda s: closes[s].loc[:d2].iloc[-1] / closes[s].loc[:d1].iloc[-1] - 1
    assert second["sector"] == pytest.approx((r2("XLK") + r2("XLC") + r2("XLY")) / 3)


def test_perf_stats_and_summary():
    rets = pd.Series([0.01] * 24)
    cash = pd.Series([0.002] * 24)
    st = bt.perf_stats(rets, cash)
    assert st["anualizat"] == pytest.approx(1.01 ** 12 - 1)
    assert st["scadere_max"] == pytest.approx(0.0) and st["luni_pozitive"] == 1.0 and st["N_luni"] == 24
    assert bt.perf_stats(rets[:6], cash[:6])["anualizat"] is None
    # curba: 1,10 → 0,88 → 0,924 → constant; scăderea maximă = 0,88 / 1,10 − 1 = −20%
    dd = bt.perf_stats(pd.Series([0.1, -0.2, 0.05] + [0.0] * 9), pd.Series([0.0] * 12))
    assert dd["scadere_max"] == pytest.approx(-0.2)
    res = pd.DataFrame({"date": pd.date_range("2012-01-01", periods=24, freq="MS") + pd.offsets.MonthEnd(0),
                        "trend": 0.01, "trend_bh": 0.0, "sector": 0.02, "spy": 0.005, "cash": 0.001})
    sm = bt.summary(res)
    tot = sm[(sm["perioada"] == "tot") & (sm["strategie"] == "B. rotație sectoare")].iloc[0]
    assert tot["N_luni"] == 24 and tot["anualizat"] == pytest.approx(1.02 ** 12 - 1)
    assert sm[(sm["perioada"] == "construire")]["N_luni"].max() == 12


def test_main_end_to_end(tmp_path, monkeypatch, capsys):
    closes = _closes()
    monkeypatch.setattr(bt, "download", lambda s, a, b, refresh=False:
                        pd.DataFrame({"adj": closes[s]}) if s in closes else pd.DataFrame())
    monkeypatch.setattr(bt, "download_cash", lambda refresh=False: pd.Series(2.4, index=pd.bdate_range("2009-01-01", "2014-12-31")))
    out = tmp_path / "rez.csv"
    assert bt.main(["--start", "2011-01-01", "--end", "2014-12-31", "--out", str(out)]) == 0
    rez = pd.read_csv(str(out).replace(".csv", "_rezumat.csv"))
    assert set(rez["perioada"]) == {"construire", "verificare", "tot"}
    assert "A. trend (regula)" in set(rez["strategie"])
    text = capsys.readouterr().out
    assert "Fără date:" in text and "EEM" in text                             # simbolurile lipsă sunt raportate


def test_fmt_sign_only_on_returns():
    assert bt._fmt(0.045) == "+4,5%" and bt._fmt(0.057, sign=False) == "5,7%"
    assert bt._fmt(0.54, pct=False) == "0,54" and bt._fmt(None) == "N/A"
