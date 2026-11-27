"""Faza 3c-2: funcțiile pure ale testului istoric pe date SEC + o rulare cap-coadă a `tools/backtest_sec.py` pe date
sintetice (arhive zip false, prețuri false). Valorile de referință sunt calculate de mână în comentarii."""
import json
import math
import zipfile

import numpy as np
import pandas as pd
import pytest

from analytics import backtest_sec as bs

T = pd.Timestamp


# --- universul -----------------------------------------------------------------

def _snaps():
    return pd.DataFrame({"date": ["2010-01-04", "2012-06-01", "2015-03-02"],
                         "tickers": ["A,B,C", "A,C,D", "A,D,B"]})


def test_intervals_from_snapshots_si_reintrari():
    iv = bs.intervals_from_snapshots(_snaps())
    got = {(r.simbol, r.de_la.date().isoformat(), None if pd.isna(r.pana_la) else r.pana_la.date().isoformat())
           for r in iv.itertuples()}
    assert got == {("A", "2010-01-04", None), ("B", "2010-01-04", "2012-06-01"), ("B", "2015-03-02", None),
                   ("C", "2010-01-04", "2015-03-02"), ("D", "2012-06-01", None)}
    iv2 = bs.intervals_from_snapshots(_snaps(), start="2013-01-01")      # B 2010–2012 se încheie înainte
    assert len(iv2) == 4 and not ((iv2["simbol"] == "B") & iv2["pana_la"].notna()).any()


def test_members_at_capete_si_b():
    iv = bs.intervals_from_snapshots(_snaps())
    assert bs.members_at(iv, "2012-05-31") == {"A", "B", "C"}
    assert bs.members_at(iv, "2012-06-01") == {"A", "C", "D"}               # ieșirea: pana_la exclus
    assert bs.members_at(iv, "2016-01-01") == {"A", "B", "D"}
    assert bs.current_members(iv) == {"A", "B", "D"}
    assert bs.yahoo_symbol("brk.b") == "BRK-B"


def test_accept_mapping():
    iv = pd.DataFrame({"simbol": ["X"], "de_la": [T("2014-05-01")], "pana_la": [T("2018-01-02")]})
    ok = bs.accept_mapping(iv, ["2015-02-20", "2016-02-19"], T("2005-01-03"), "2012-01-01")
    assert ok == (True, None)
    # simbol refolosit: istoricul Yahoo începe în 2020, după intrarea în indice
    assert bs.accept_mapping(iv, ["2015-02-20"], T("2020-03-02"), "2012-01-01")[0] is False
    # CIK fără 10-K în perioada de apartenență (+ un an după ieșire)
    assert bs.accept_mapping(iv, ["2010-02-20", "2020-02-20"], T("2005-01-03"), "2012-01-01")[0] is False
    assert bs.accept_mapping(iv, ["2018-11-20"], T("2005-01-03"), "2012-01-01")[0] is True    # anul de după ieșire
    assert bs.accept_mapping(iv, ["2015-02-20"], None, "2012-01-01") == (False, "fără prețuri Yahoo")
    # membru din 1996: istoricul trebuie să înceapă cel târziu la începutul testului (+31 de zile)
    old = pd.DataFrame({"simbol": ["Y"], "de_la": [T("1996-01-02")], "pana_la": [pd.NaT]})
    assert bs.accept_mapping(old, ["2013-02-20"], T("2012-01-20"), "2012-01-01")[0] is True
    assert bs.accept_mapping(old, ["2013-02-20"], T("2012-03-01"), "2012-01-01")[0] is False


# --- randamente ----------------------------------------------------------------

def _close(start, end, v0=100.0, step=1.0):
    idx = pd.bdate_range(start, end)
    return pd.Series(v0 + step * np.arange(len(idx)), index=idx)


def test_forward_return_at_normal_limita_si_trunchiat():
    c = _close("2012-01-02", "2020-06-30")
    r, tr = bs.forward_return_at(c, T("2012-01-31"), 6, T("2019-12-31"), T("2020-06-30"))
    p0, p1 = c.loc[:"2012-01-31"].iloc[-1], c.loc[:"2012-07-31"].iloc[-1]
    assert r == pytest.approx(p1 / p0 - 1) and tr is False
    # capătul (2020-01-31) peste sfârșitul fazei → nimic (fără prețuri din verificare)
    assert bs.forward_return_at(c, T("2019-07-31"), 6, T("2019-12-31"), T("2020-06-30")) == (None, False)
    # delistat: seria se oprește 2016-03-15, cu mult înaintea datelor → până la ultimul preț, trunchiat
    d = _close("2012-01-02", "2016-03-15")
    r, tr = bs.forward_return_at(d, T("2016-01-29"), 12, T("2019-12-31"), T("2020-06-30"))
    assert tr is True and r == pytest.approx(d.iloc[-1] / d.loc[:"2016-01-29"].iloc[-1] - 1)
    # seria se oprește la sfârșitul datelor (prezent) → orizont neîncheiat, nu delistare
    e = _close("2012-01-02", "2020-06-25")
    assert bs.forward_return_at(e, T("2020-03-31"), 6, T("2025-01-01"), T("2020-06-30")) == (None, False)


def test_add_median_excess_univers_si_sector():
    df = pd.DataFrame({"date": ["d"] * 6, "sector": ["X"] * 5 + ["Y"], "r": [0.1, 0.2, 0.3, 0.4, 0.5, 0.9]})
    out = bs.add_median_excess(df, "r")                                   # mediana datei = 0,35
    assert out["r_ex"].tolist() == pytest.approx([-0.25, -0.15, -0.05, 0.05, 0.15, 0.55])
    out = bs.add_median_excess(df, "r", by=("date", "sector"), suffix="_s")
    assert out["r_s"].iloc[:5].tolist() == pytest.approx([-0.2, -0.1, 0.0, 0.1, 0.2])   # mediana X = 0,3
    assert math.isnan(out["r_s"].iloc[5])                                 # sectorul Y: 1 < 5 observații


# --- statistici ------------------------------------------------------------------

def test_newey_west_t_de_mana():
    # x = 1,2,3,4: medie 2,5; abateri −1,5 −0,5 0,5 1,5; γ0 = 5/4 = 1,25 → t = 2,5 / √(1,25/4) = 4,4721
    assert bs.newey_west_t([1, 2, 3, 4], lags=0) == pytest.approx(4.472136, abs=1e-6)
    # γ1 = (0,75 − 0,25 + 0,75) / 4 = 0,3125; ponderea 1 − 1/2 = 0,5 → S = 1,25 + 2·0,5·0,3125 = 1,5625
    # t = 2,5 / √(1,5625/4) = 2,5 / 0,625 = 4,0
    assert bs.newey_west_t([1, 2, 3, 4], lags=1) == pytest.approx(4.0)
    assert bs.newey_west_t([1, 2]) is None and bs.newey_west_t([2, 2, 2]) is None


def _panel(n_dates=24, n=10, sign=1.0, sectors=("A", "B")):
    rows = []
    for i, d in enumerate(pd.date_range("2013-01-31", periods=n_dates, freq="M")):
        for k in range(n):
            score = 10.0 * k + 5
            rows.append({"date": d, "symbol": f"S{k}", "sector": sectors[k % len(sectors)], "score": score,
                         "ret": sign * score / 1000 + 0.001 * ((i * 7 + k * 3) % 5)})
    return pd.DataFrame(rows)


def test_ic_si_cincimi_pe_ani():
    df = _panel()
    s = bs.ic_summary(df, "score", "ret")
    assert s["nr_date"] == 24 and s["ic_mediu"] > 0.9 and s["pct_pozitiv"] == 1.0
    df = bs.add_median_excess(df, "ret")
    q = bs.quintile_spread_by_year(df, "score", "ret_ex")
    assert q["an"].tolist() == [2013, 2014] and q["sus_bate_jos"].all() and (q["nr_date"] == 12).all()
    neg = bs.add_median_excess(_panel(sign=-1.0), "ret")
    assert not bs.quintile_spread_by_year(neg, "score", "ret_ex")["sus_bate_jos"].any()


def test_band_medians_monotone_si_banda_lipsa():
    df = pd.DataFrame({"s": [10, 20, 50, 60, 80, 90], "ex": [-0.1, -0.05, 0.0, 0.01, 0.05, 0.08]})
    t, mono = bs.band_medians(df, "s", "ex")
    assert mono is True and t["N"].tolist() == [2, 2, 2]
    assert t["mediana_exces"].tolist() == pytest.approx([-0.075, 0.005, 0.065])
    t, mono = bs.band_medians(df[df["s"] < 70], "s", "ex")
    assert mono is None                                                    # fără banda FAVORABIL


def test_sector_rank_score():
    df = _panel(n_dates=1, n=12)
    r = bs.sector_rank_score(df, "score")
    # sectorul A: scorurile 5, 25, …, 105 (6 valori) → percentilele 1/6 … 6/6
    a = r[df["sector"] == "A"].tolist()
    assert a == pytest.approx([100 * k / 6 for k in range(1, 7)])
    df.loc[0, "sector"] = None
    assert math.isnan(bs.sector_rank_score(df, "score").iloc[0])


# --- rulare cap-coadă pe date sintetice ---------------------------------------------

def test_backtest_sec_cap_coada(tmp_path, monkeypatch):
    import tools.backtest_sec as tb
    from tests.test_sec_statements import Q_EXTRA, _company

    # arhivele SEC false: o companie (CIK 1) cu 10-K 2021–2023 (din testele adaptorului), mutată în 2013–2015
    cf = _company(Q_EXTRA)
    text = json.dumps(cf).replace("2021", "2013").replace("2022", "2014").replace("2023", "2015") \
        .replace("2024", "2016")
    sec_dir = tmp_path / "sec"
    sec_dir.mkdir()
    with zipfile.ZipFile(sec_dir / "companyfacts.zip", "w") as z:
        z.writestr("CIK0000000001.json", text)
    with zipfile.ZipFile(sec_dir / "submissions.zip", "w") as z:
        z.writestr("CIK0000000001.json", json.dumps({"sic": "3571", "name": "Test", "tickers": ["TST"]}))
    (sec_dir / "company_tickers.json").write_text(json.dumps({"0": {"cik_str": 1, "ticker": "TST"}}))
    univ = tmp_path / "univ"
    univ.mkdir()
    (univ / "sp500_intervale.csv").write_text("# test\nsimbol,de_la,pana_la\nTST,2010-01-04,\nOLD,2010-01-04,2014-01-02\n")
    (univ / "sp500_alias.csv").write_text("simbol_istoric,simbol_actual,cik,nota\n")

    idx = pd.bdate_range("2009-01-02", "2020-06-30")
    px = pd.DataFrame({"adj": np.linspace(10, 40, len(idx)), "raw": np.linspace(10, 40, len(idx)), "volume": 1.0},
                      index=idx)
    monkeypatch.setattr(tb, "SEC_DIR", sec_dir)
    monkeypatch.setattr(tb, "UNIV", univ)
    monkeypatch.setattr(tb, "read_intervals", lambda path=None: _read_iv(univ / "sp500_intervale.csv"))
    monkeypatch.setattr(tb, "read_alias", lambda path=None: {})
    monkeypatch.setattr(tb, "download", lambda sym, *a, **k: px.copy() if sym in ("TST", "SPY") else (
        pd.DataFrame({"adj": 2.0, "raw": 2.0, "volume": 1.0}, index=idx) if sym == "^TNX" else pd.DataFrame()))
    monkeypatch.setattr(tb, "download_splits", lambda sym, refresh=False: pd.Series(dtype=float))
    monkeypatch.setattr(tb, "download_fred", lambda code, refresh=False: pd.Series(
        5.0, index=pd.date_range("2009-01-01", "2020-06-01", freq="MS")))
    out = tmp_path / "rez"
    assert tb.main(["--faza", "construire", "--out-dir", str(out), "--procese", "1"]) == 0
    obs = pd.read_csv(out / "observatii_construire.csv")
    # primul 10-K (FY2013) e depus marți 18.02.2014, cunoscut din 19.02 → prima lună cu scor: februarie 2014
    assert obs["date"].min() == "2014-02-28" and obs["date"].max() == "2019-12-31"
    assert obs["in_B"].all() and obs["in_C"].all() and (obs["sector"] == "Technology").all()
    assert obs["fiscal_year"].iloc[0] == "2013-12-31"
    # cu un singur an fiscal: Piotroski și creșterea nu se pot calcula → un singur pilon → fără scor (ca în aplicație);
    # din februarie 2015 (al doilea 10-K) scorul există
    fs = pd.to_numeric(obs["fund_score"], errors="coerce")
    assert fs[obs["date"] < "2015-02-27"].isna().all() and fs[obs["date"] >= "2015-02-27"].notna().all()
    # randamentul la 12 luni: doar cu capătul ≤ 31.12.2019
    last12 = obs[obs["fwd_12m"].notna()]["date"].max()
    assert last12 == "2018-12-31"
    miss = pd.read_csv(out / "fara_date_construire.csv")
    assert miss.loc[miss["simbol"] == "OLD", "motiv"].iloc[0].startswith("fără CIK")
    cov = pd.read_csv(out / "acoperire_construire.csv")
    c2015 = cov[(cov["univers"] == "C") & (cov["an"] == 2015)].iloc[0]
    assert c2015["membri"] == 1.0 and c2015["pct_cu_scor"] == 100.0       # OLD a ieșit în ian. 2014
    assert (out / "rezumat_construire.csv").exists()


def _read_iv(path):
    iv = pd.read_csv(path, comment="#")
    iv["de_la"] = pd.to_datetime(iv["de_la"])
    iv["pana_la"] = pd.to_datetime(iv["pana_la"])
    return iv


def test_preturile_din_cache_propriu(monkeypatch):
    """Cache-ul comun `tools/cache/` are fișiere ale altor teste care încep mai târziu (ex. SPY din 2018):
    testul SEC citește prețurile din `tools/cache/sec/preturi/`."""
    import tools.backtest_sec as tb
    seen = {}
    monkeypatch.setattr(tb, "_download", lambda sym, s, e, r, cache_dir=None: seen.update(dir=cache_dir) or "ok")
    assert tb.download("SPY", "2009-01-01", "2020-01-01") == "ok"
    assert seen["dir"] == tb.PRICES_DIR and tb.PRICES_DIR.parts[-2:] == ("sec", "preturi")
