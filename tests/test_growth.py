"""analytics/growth.py: CAGR, creștere pe acțiune, dividende, randament total, sursa randamentului. Valori de mână."""
import numpy as np
import pandas as pd
import pytest

from analytics import growth as g

D = pd.to_datetime(["2026-09-30", "2025-09-30", "2024-09-30", "2023-09-30"])


def _stmt(rows):
    return pd.DataFrame(rows, index=D).T


def _income(shares=(100, 100, 100, 100), eps=(2.0, 1.8, 1.6, 1.0)):
    return _stmt({"Total Revenue": [1331.0, 1210, 1100, 1000], "Net Income": [200.0, 180, 160, 100],
                  "Diluted EPS": list(eps), "Diluted Average Shares": list(shares)})


def _cash():
    return _stmt({"Operating Cash Flow": [300.0, 250, 220, 200], "Capital Expenditure": [-84.0, -50, -40, -50]})


def test_cagr_edges():
    assert g.cagr(1331, 1000, 3) == pytest.approx(0.10)
    assert g.cagr(-1, 1000, 3) is None and g.cagr(100, 0, 3) is None and g.cagr(100, None, 3) is None
    assert g.cagr(100, 50, 0) is None


def test_statement_growth_per_share_and_total():
    r = g.statement_growth(_income(), _cash())
    assert r["years"] == 3 and r["date_old"] == D[3]
    rows = r["rows"]
    assert rows["Cifră de afaceri"]["total"] == pytest.approx(0.10)           # 1331 / 1000 pe 3 ani
    assert rows["Cifră de afaceri"]["per_share"] == pytest.approx(0.10)       # acțiuni constante
    assert rows["EPS diluat"]["per_share"] == pytest.approx(2.0 ** (1 / 3) - 1)
    assert rows["FCF (CFO − capex)"]["total"] == pytest.approx((216 / 150) ** (1 / 3) - 1)
    assert r["eps_new"] == 2.0 and r["eps_old"] == 1.0


def test_statement_growth_buybacks_raise_per_share():
    r = g.statement_growth(_income(shares=(80, 90, 95, 100)), _cash())["rows"]
    # venit pe acțiune: (1331/80) / (1000/100) = 1,66375 → CAGR 18,5%; totalul rămâne 10%
    assert r["Cifră de afaceri"]["per_share"] == pytest.approx(1.66375 ** (1 / 3) - 1)
    assert r["Cifră de afaceri"]["total"] == pytest.approx(0.10)


def test_statement_growth_split_makes_per_share_na():
    r = g.statement_growth(_income(shares=(400, 400, 100, 100)), _cash())
    rows = r["rows"]
    assert rows["Cifră de afaceri"]["per_share"] is None and rows["Cifră de afaceri"]["total"] == pytest.approx(0.10)
    assert rows["EPS diluat"]["per_share"] is None and "4.00×" in rows["EPS diluat"]["note"]
    assert r["eps_new"] is None


def test_statement_growth_negative_eps_and_short_history():
    rows = g.statement_growth(_income(eps=(2.0, 1.0, 0.5, -0.5)), _cash())["rows"]
    assert rows["EPS diluat"]["per_share"] is None and "≤ 0" in rows["EPS diluat"]["note"]
    assert g.statement_growth(_income().iloc[:, :1], _cash()) is None


def _files():
    rows = []
    for y, rev, ni, div, sh in ((2020, 1000, 100, 50, 10), (2022, 1100, 110, 60, 10), (2023, 1331, 121, 66, 10),
                                (2025, 1610.51, 146.41, 72.6, 10)):
        rows.append({"year": y, "months": 12, "rev": rev, "ni": ni, "div_total": div, "shares": sh})
    rows.append({"year": 2025, "months": 6, "rev": 800, "ni": 70, "div_total": None, "shares": 10})
    return pd.DataFrame(rows)


def test_file_growth_3_and_5_years():
    r = g.file_growth(_files())
    assert r["year_new"] == 2025
    rev = r["rows"]["Cifră de afaceri"]
    assert rev[5]["per_share"] == pytest.approx((1610.51 / 1000) ** 0.2 - 1)    # 2025 vs 2020 → 10%
    assert rev[3]["total"] == pytest.approx((1610.51 / 1100) ** (1 / 3) - 1)    # 2025 vs 2022
    assert r["rows"]["EPS (profit net / acțiuni)"][5]["per_share"] == pytest.approx((14.641 / 10) ** 0.2 - 1)
    assert r["rows"]["Dividend pe acțiune (profit distribuit / acțiuni)"][5]["per_share"] == pytest.approx((7.26 / 5) ** 0.2 - 1)


def test_file_growth_three_years_and_share_change():
    df = _files()
    df.loc[df.year == 2022, "rev"] = 1210.0
    r = g.file_growth(df)
    assert r["rows"]["Cifră de afaceri"][3]["total"] == pytest.approx((1610.51 / 1210) ** (1 / 3) - 1)
    df.loc[df.year == 2025, "shares"] = 20                                      # acțiuni gratuite 1:1
    r2 = g.file_growth(df)
    assert r2["rows"]["Cifră de afaceri"][5]["per_share"] is None and "2.00×" in r2["notes"][5]
    assert r2["rows"]["Cifră de afaceri"][5]["total"] == pytest.approx(0.10)


def test_dividend_growth_calendar_years():
    idx = pd.to_datetime(["2020-03-01", "2020-09-01", "2022-03-01", "2023-03-01", "2025-03-01", "2025-09-01",
                          "2026-03-01"])
    div = pd.Series([0.5, 0.5, 1.2, 1.3, 0.8, 0.8, 0.9], index=idx)
    y, out = g.dividend_growth(div, "2026-11-25", horizons=(3, 5))
    assert y == 2025
    assert out[5] == pytest.approx(1.6 ** 0.2 - 1)          # 1,6 (2025) vs 1,0 (2020)
    assert out[3] == pytest.approx((1.6 / 1.2) ** (1 / 3) - 1)    # 2025 vs 2022 (o singură plată de 1,2)
    _y, out4 = g.dividend_growth(div, "2026-11-25", horizons=(4,))
    assert out4[4] is None                                  # 2021: fără plată → N/A, nu 0


def test_total_return_and_tolerance():
    idx = pd.bdate_range("2016-01-01", "2026-11-24")
    s = pd.Series(100 * 1.10 ** ((idx - idx[0]).days / 365.25), index=idx)
    r = g.total_return(s, 5)
    assert r["cagr"] == pytest.approx(0.10, abs=2e-4) and r["n_obs"] > 1200
    assert g.total_return(s, 12) is None                    # istoric mai scurt de 12 ani


def test_price_at_tolerance():
    s = pd.Series([10.0, 11.0], index=pd.to_datetime(["2026-01-02", "2026-02-20"]))
    assert g.price_at(s, "2026-01-08") == 10.0
    assert g.price_at(s, "2026-02-01") is None              # 30 de zile fără cotație


def test_decompose_identity():
    idx = pd.to_datetime(["2023-09-29", "2026-09-30"])
    close = pd.Series([100.0, 172.8], index=idx)            # preț +20%/an
    adj = pd.Series([90.0, 163.296], index=idx)             # randament total: 1,8144 → +22%/an
    d = g.decompose(close, adj, 2.0 * 1.331, 2.0, idx[1], idx[0])      # EPS +10%/an
    assert d["years"] == 3
    assert d["price"] == pytest.approx(0.20) and d["eps"] == pytest.approx(0.10)
    assert d["pe"] == pytest.approx(1.20 / 1.10 - 1)
    assert (1 + d["eps"]) * (1 + d["pe"]) * (1 + d["div"]) == pytest.approx(1 + d["total"])
    assert d["pe_old"] == pytest.approx(50.0) and d["pe_new"] == pytest.approx(172.8 / 2.662)
    assert d["residual"] == pytest.approx(d["total"] - d["eps"] - d["pe"] - d["div"])
    assert g.decompose(close, adj, 2.0, -1.0, idx[1], idx[0]) is None


def test_tr_benchmark():
    assert g.tr_benchmark("TLV.RO")[0] == "TVBETETF.RO"
    assert g.tr_benchmark("SAP.DE", "EUR")[0] == "^GDAXI"
    assert g.tr_benchmark("ASML.AS", "EUR")[0] == "EXSA.DE"
    assert g.tr_benchmark("AAPL", "USD")[0] == "SPY"
    assert g.tr_benchmark("X.L", "GBP") == (None, None)


def test_dividend_growth_partial_first_year_is_na():
    idx = pd.to_datetime(["2015-11-05", "2016-02-05", "2016-05-05", "2016-08-05", "2016-11-05", "2025-02-05"])
    div = pd.Series([0.13, 0.13, 0.14, 0.14, 0.14, 1.0], index=idx)
    # istoricul începe pe 25.11.2015: anul 2015 e parțial → N/A; 2016 e complet
    _y, out = g.dividend_growth(div, "2026-11-25", horizons=(9, 10), history_start="2015-11-25")
    assert out[10] is None
    assert out[9] == pytest.approx((1.0 / 0.55) ** (1 / 9) - 1)
