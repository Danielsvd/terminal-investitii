"""Faza 3b: creșterea pe 3/5/10 ani din tabelul anual SEC (analytics/growth.sec_growth). Valori calculate de mână."""
from datetime import date

import pandas as pd
import pytest

from analytics import growth


def table(n=11, start_year=2015, **over):
    """Tabel anual sintetic ca `xbrl_facts.annual_table`: venit 100 × 1,10^i, acțiuni 10 (fără schimbare)."""
    ends = [date(start_year + i, 12, 31) for i in range(n)]
    t = pd.DataFrame(index=pd.Index(ends, name="end"))
    t["revenue"] = [100.0 * 1.10 ** i for i in range(n)]
    t["net_income"] = [10.0 * 1.08 ** i for i in range(n)]
    t["shares_diluted"] = 10.0
    t["eps"] = t["net_income"] / 10.0
    t["revenue_ps"] = t["revenue"] / 10.0
    t["cfo"] = [20.0 + i for i in range(n)]
    t["capex"] = 5.0
    t["fcf"] = t["cfo"] - t["capex"]
    t["fcf_ps"] = t["fcf"] / 10.0
    t["sbc"] = 1.0
    t["dps"] = [0.5 + 0.05 * i for i in range(n)]
    t["equity"] = [50.0 + 5 * i for i in range(n)]
    t["bvps"] = t["equity"] / 10.0
    t["filed"] = [date(start_year + i + 1, 2, 20) for i in range(n)]
    t["conflicts"] = [[] for _ in range(n)]
    for k, v in over.items():
        t[k] = v
    return t


def row(res, label, measure):
    return next(r for r in res["rows"] if r["Indicator"] == label and r["Măsură"] == measure)


def test_cagr_pe_capete_3_5_10():
    res = growth.sec_growth(table())
    r = row(res, "Cifră de afaceri", "total")
    assert r["3 ani"] == pytest.approx(0.10) and r["5 ani"] == pytest.approx(0.10) and r["10 ani"] == pytest.approx(0.10)
    assert row(res, "EPS diluat", "pe acțiune")["10 ani"] == pytest.approx(0.08)
    assert r["Consecvență"] == "10 din 10 ani"
    assert res["date_new"] == pd.Timestamp("2025-12-31") and res["date_old"] == pd.Timestamp("2022-12-31")
    assert res["eps_new"] == pytest.approx(1.0 * 1.08 ** 10) and res["eps_old"] == pytest.approx(1.0 * 1.08 ** 7)


def test_fcf_pe_medii_de_3_ani():
    # FCF = 15 + i; capătul nou (i = 8, 9, 10) medie 24; capătul vechi pe 5 ani (i = 3, 4, 5) medie 19
    res = growth.sec_growth(table())
    r = row(res, "FCF (CFO − capex)", "total")
    assert r["5 ani"] == pytest.approx((24 / 19) ** (1 / 5) - 1)
    assert "medii pe 3 ani" in r["Notă"]
    # 10 ani: capătul vechi (2015) nu are 3 ani înainte → N/A
    assert r["10 ani"] is None
    sbc = row(res, "FCF − compensarea în acțiuni (informativ)", "pe acțiune")
    assert sbc["5 ani"] == pytest.approx((2.3 / 1.8) ** (1 / 5) - 1)       # (24 − 1)/10 vs (19 − 1)/10


def test_capat_negativ_si_lipsa():
    ni = [10.0 * 1.08 ** i for i in range(11)]
    ni[5] = -3.0                                       # 2020: pierdere → EPS 5 ani nedefinit
    t = table(net_income=ni)
    t["eps"] = t["net_income"] / 10.0
    r = row(growth.sec_growth(t), "EPS diluat", "pe acțiune")
    assert r["5 ani"] is None and "≤ 0" in r["Notă"]
    assert r["3 ani"] == pytest.approx(0.08)
    assert r["Consecvență"] == "9 din 10 ani"          # scăderea în 2020 (și revenirea în 2021 contează ca creștere)


def test_conflict_de_etichete_in_interval():
    # 2016: cusătură în conflict la cifra de afaceri (eticheta din 2016 diferă de cea folosită după)
    conf = [[] for _ in range(11)]
    conf[1] = ["revenue"]
    res = growth.sec_growth(table(conflicts=conf))
    r = row(res, "Cifră de afaceri", "total")
    assert r["3 ani"] == pytest.approx(0.10) and r["5 ani"] == pytest.approx(0.10)
    assert r["10 ani"] is None and "10 ani: etichete XBRL în conflict" in r["Notă"]
    assert row(res, "Profit net", "total")["10 ani"] == pytest.approx(0.08)      # alt indicator: neafectat
    # capătul vechi pe 3 ani chiar la cusătură → și 3 ani N/A
    conf2 = [[] for _ in range(11)]
    conf2[7] = ["revenue"]
    assert row(growth.sec_growth(table(conflicts=conf2)), "Cifră de afaceri", "total")["3 ani"] is None


def test_financiare_fara_fcf():
    labels = {r["Indicator"] for r in growth.sec_growth(table(), financial=True)["rows"]}
    assert not any("FCF" in x for x in labels) and "Valoare contabilă pe acțiune" in labels


def test_istoric_scurt():
    res = growth.sec_growth(table(n=4, start_year=2022))
    r = row(res, "Cifră de afaceri", "total")
    assert r["3 ani"] == pytest.approx(0.10) and r["5 ani"] is None and r["10 ani"] is None
    assert growth.sec_growth(table(n=1)) is None
    assert growth.sec_growth(None) is None


def test_sec_yahoo_diffs():
    t = table()
    inc = pd.DataFrame({pd.Timestamp("2025-12-31"): {"Total Revenue": t["revenue"].iloc[-1] * 1.05,
                                                      "Net Income": t["net_income"].iloc[-1]}})
    cf = pd.DataFrame({pd.Timestamp("2025-12-31"): {"Operating Cash Flow": 30.0, "Capital Expenditure": -5.0}})
    d = growth.sec_yahoo_diffs(t, inc, cf)
    assert [x[0] for x in d] == ["Cifră de afaceri"]
    assert d[0][3] == pytest.approx(1 / 1.05 - 1)
    assert growth.sec_yahoo_diffs(t, None, None) == []
