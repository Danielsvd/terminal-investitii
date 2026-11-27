"""Faza 3 SEC: XBRL `companyfacts` → serii anuale, trimestriale, TTM, point-in-time. Date sintetice, calculate de mână."""
from datetime import date

import pandas as pd
import pytest

from analytics import xbrl_facts as xf
from data import sec_xbrl


def F(start, end, val, filed, form="10-K", accn=None):
    r = {"end": end, "val": val, "form": form, "filed": filed, "accn": accn or f"acc-{filed}-{form}"}
    if start:
        r["start"] = start
    return r


def cf_of(**concepts):
    """concepts: eticheta → listă de fapte (unitatea dedusă din CONCEPTS); `ifrs=True` adaugă taxonomia IFRS."""
    facts = {}
    units = {name: unit for unit, _, lst in xf.CONCEPTS.values() for _, name in lst}
    taxes = {name: tax for _, _, lst in xf.CONCEPTS.values() for tax, name in lst}
    for name, rows in concepts.items():
        facts.setdefault(taxes[name], {})[name] = {"units": {units[name]: rows}}
    return {"cik": 1, "entityName": "Test", "facts": facts}


D = date.fromisoformat


# --- data publicării -------------------------------------------------------

def test_available_on_ziua_lucratoare_urmatoare():
    assert xf.available_on("2019-11-01") == D("2019-11-04")     # vineri → luni
    assert xf.available_on("2019-11-05") == D("2019-11-06")     # marți → miercuri
    assert xf.available_on(None) is None


# --- etichete, conflict, retratare, point-in-time --------------------------

def _revenue_cf():
    return cf_of(
        SalesRevenueNet=[
            F("2016-01-01", "2016-12-31", 80.0, "2017-02-20"),
            F("2017-01-01", "2017-12-31", 90.0, "2018-02-20"),
            F("2016-01-01", "2016-12-31", 80.0, "2018-02-20"),
        ],
        RevenueFromContractWithCustomerExcludingAssessedTax=[
            F("2018-01-01", "2018-12-31", 100.0, "2019-02-20"),
            F("2017-01-01", "2017-12-31", 90.5, "2019-02-20"),       # ASC 606: +0,56% față de SalesRevenueNet → fără conflict
            F("2019-01-01", "2019-12-31", 110.0, "2020-02-21"),       # prima raportare
            F("2019-01-01", "2019-12-31", 107.0, "2021-02-19"),       # retratată în 10-K-ul următor
            F("2020-01-01", "2020-12-31", 120.0, "2021-02-19"),
            F("2020-01-01", "2020-03-31", 28.0, "2020-05-01", form="10-Q"),
            F("2020-01-01", "2020-12-31", 999.0, "2021-03-01", form="8-K"),   # 8-K: ignorat
        ],
    )


def test_annual_combina_etichetele_si_ia_ultima_versiune():
    a = xf.annual(_revenue_cf(), "revenue")
    assert list(a.index) == [D("2016-12-31"), D("2017-12-31"), D("2018-12-31"), D("2019-12-31"), D("2020-12-31")]
    assert list(a["val"]) == [80.0, 90.5, 100.0, 107.0, 120.0]
    assert a.loc[D("2016-12-31"), "concept"] == "SalesRevenueNet"
    assert a.loc[D("2017-12-31"), "concept"] == "RevenueFromContractWithCustomerExcludingAssessedTax"
    assert not a["conflict"].any()


def test_annual_point_in_time():
    cf = _revenue_cf()
    # înainte de retratare: valoarea din primul raport
    assert xf.annual(cf, "revenue", asof="2020-06-01").loc[D("2019-12-31"), "val"] == 110.0
    # depus vineri 21.02.2020 → cunoscut abia luni 24.02.2020
    assert D("2019-12-31") not in xf.annual(cf, "revenue", asof="2020-02-21").index
    assert D("2019-12-31") in xf.annual(cf, "revenue", asof="2020-02-24").index
    # 2017 înainte de 10-K-ul din 2019: doar SalesRevenueNet
    a = xf.annual(cf, "revenue", asof="2018-06-01")
    assert a.loc[D("2017-12-31"), "val"] == 90.0 and a.loc[D("2017-12-31"), "concept"] == "SalesRevenueNet"


def test_conflict_intre_etichete_peste_1_la_suta():
    cf = cf_of(
        RevenueFromContractWithCustomerExcludingAssessedTax=[F("2018-01-01", "2018-12-31", 100.0, "2019-02-20")],
        Revenues=[F("2018-01-01", "2018-12-31", 103.0, "2019-02-20")],
    )
    a = xf.annual(cf, "revenue")
    assert a.loc[D("2018-12-31"), "val"] == 100.0 and bool(a.loc[D("2018-12-31"), "conflict"])


def test_alta_unitate_si_formular_strain_ignorate():
    cf = cf_of(Revenues=[F("2018-01-01", "2018-12-31", 5.0, "2019-04-01", form="20-F")])
    cf["facts"]["us-gaap"]["NetIncomeLoss"] = {"units": {"EUR": [F("2018-01-01", "2018-12-31", 1.0, "2019-02-01")]}}
    assert xf.annual(cf, "revenue").empty and xf.annual(cf, "net_income").empty


# --- perioade: trimestre derivate, T4, TTM ---------------------------------

def _cfo_fiscal_sept():
    # exercițiu 01.10.2022 – 30.09.2023 (ca AAPL); fluxul de numerar apare în 10-Q doar cumulat
    s = "2022-10-01"
    return cf_of(NetCashProvidedByUsedInOperatingActivities=[
        F(s, "2022-12-31", 10.0, "2023-02-03", form="10-Q"),
        F(s, "2023-04-01", 25.0, "2023-05-05", form="10-Q"),
        F(s, "2023-07-01", 45.0, "2023-08-04", form="10-Q"),
        F(s, "2023-09-30", 70.0, "2023-11-03"),
    ])


def test_trimestre_derivate_din_cumulat_si_t4():
    q = xf.quarterly(_cfo_fiscal_sept(), "cfo")
    assert list(q["val"]) == [10.0, 15.0, 20.0, 25.0]               # T4 = 70 − 45
    assert list(q["derived"]) == [False, True, True, True]
    assert q.loc[D("2023-09-30"), "start"] == D("2023-07-02")


def test_ttm_si_controlul_fata_de_anual():
    t = xf.ttm(_cfo_fiscal_sept(), "cfo")
    assert t["val"] == 70.0 and t["end"] == D("2023-09-30") and t["check"] == 0.0
    # înainte de 10-K: doar 3 trimestre → fără TTM
    assert xf.ttm(_cfo_fiscal_sept(), "cfo", asof="2023-09-01") is None


def test_trimestru_direct_are_prioritate():
    cf = cf_of(Revenues=[
        F("2023-01-01", "2023-03-31", 10.0, "2023-05-01", form="10-Q"),
        F("2023-04-01", "2023-06-30", 12.0, "2023-08-01", form="10-Q"),       # T2 raportat direct
        F("2023-01-01", "2023-06-30", 23.0, "2023-08-01", form="10-Q"),       # cumulat diferit (rotunjire): nu înlocuiește
    ])
    q = xf.quarterly(cf, "revenue")
    assert q.loc[D("2023-06-30"), "val"] == 12.0 and not q.loc[D("2023-06-30"), "derived"]


# --- bilanț, split-uri, tabelul anual --------------------------------------

def _full_cf():
    return cf_of(
        Revenues=[F("2019-01-01", "2019-12-31", 1000.0, "2020-02-20"), F("2020-01-01", "2020-12-31", 1200.0, "2021-02-19")],
        NetIncomeLoss=[F("2019-01-01", "2019-12-31", 200.0, "2020-02-20"), F("2020-01-01", "2020-12-31", 240.0, "2021-02-19")],
        EarningsPerShareDiluted=[F("2019-01-01", "2019-12-31", 2.0, "2020-02-20"),     # înainte de split-ul 4:1
                                 F("2020-01-01", "2020-12-31", 0.6, "2021-02-19")],
        WeightedAverageNumberOfDilutedSharesOutstanding=[F("2019-01-01", "2019-12-31", 100.0, "2020-02-20"),
                                                         F("2020-01-01", "2020-12-31", 400.0, "2021-02-19")],
        NetCashProvidedByUsedInOperatingActivities=[F("2019-01-01", "2019-12-31", 300.0, "2020-02-20"),
                                                    F("2020-01-01", "2020-12-31", 330.0, "2021-02-19")],
        PaymentsToAcquirePropertyPlantAndEquipment=[F("2019-01-01", "2019-12-31", -50.0, "2020-02-20"),   # cu minus
                                                    F("2020-01-01", "2020-12-31", 60.0, "2021-02-19")],
        StockholdersEquity=[F(None, "2019-12-28", 800.0, "2020-02-20"),       # 3 zile distanță de încheiere: acceptat
                            F(None, "2020-12-31", 900.0, "2021-02-19")],
        CommonStockDividendsPerShareDeclared=[
            F("2019-01-01", "2019-03-31", 0.25, "2019-05-01", form="10-Q"),
            F("2019-04-01", "2019-06-30", 0.25, "2019-08-01", form="10-Q"),
            F("2019-07-01", "2019-09-30", 0.25, "2019-11-01", form="10-Q"),
            F("2019-10-01", "2019-12-31", 0.25, "2020-02-20"),
            F("2020-01-01", "2020-12-31", 0.30, "2021-02-19")],
    )


def test_annual_table_split_si_valori_pe_actiune():
    splits = pd.Series({pd.Timestamp("2020-08-31"): 4.0})
    t = xf.annual_table(_full_cf(), splits=splits)
    r19, r20 = t.loc[D("2019-12-31")], t.loc[D("2020-12-31")]
    assert r19["shares_diluted"] == 400.0 and r20["shares_diluted"] == 400.0       # 100 × 4
    assert r19["eps"] == pytest.approx(0.5) and r19["eps_reported"] == pytest.approx(0.5) and r19["eps_check"] == 0
    assert r20["eps"] == pytest.approx(0.6)
    assert r19["capex"] == 50.0 and r19["fcf"] == 250.0 and r20["fcf"] == 270.0
    assert r19["fcf_ps"] == pytest.approx(0.625) and r19["revenue_ps"] == pytest.approx(2.5)
    assert r19["equity"] == 800.0 and r19["bvps"] == pytest.approx(2.0)
    assert r19["dps"] == pytest.approx(0.25)           # 4 × 0,25 = 1,00 înainte de split → 0,25 după 4:1
    assert r20["dps"] == pytest.approx(0.30)
    assert r19["filed"] == D("2020-02-20")
    assert pd.isna(r19["cash"])                        # lipsă rămâne lipsă


def test_annual_table_point_in_time_fara_anul_nepublicat():
    t = xf.annual_table(_full_cf(), asof="2020-06-30")
    assert list(t.index) == [D("2019-12-31")]


def test_bilant_departe_de_incheiere_ignorat():
    cf = _full_cf()
    cf["facts"]["us-gaap"]["StockholdersEquity"]["units"]["USD"] = [F(None, "2019-11-30", 800.0, "2020-02-20")]
    assert pd.isna(xf.annual_table(cf).loc[D("2019-12-31"), "equity"])


def test_retratare_bilant_point_in_time():
    cf = _full_cf()
    cf["facts"]["us-gaap"]["StockholdersEquity"]["units"]["USD"].append(F(None, "2019-12-28", 780.0, "2021-02-19"))
    assert xf.annual_table(cf).loc[D("2019-12-31"), "equity"] == 780.0
    assert xf.annual_table(cf, asof="2020-12-31").loc[D("2019-12-31"), "equity"] == 800.0


# --- diverse ---------------------------------------------------------------

def test_filer_kind():
    assert xf.filer_kind(_full_cf()) == "10-K"
    assert xf.filer_kind({"facts": {"ifrs-full": {"Revenue": {}}}}) == "20-F"
    assert xf.filer_kind({"facts": {}}) is None
    assert xf.filer_kind(None) is None


def test_consistency():
    assert xf.consistency([1, 2, 3, 2, float("nan"), 5]) == (2, 3)
    assert xf.consistency([]) == (0, 0)


def test_split_factor_after():
    s = pd.Series({pd.Timestamp("2021-07-20"): 4.0, pd.Timestamp("2024-06-10"): 10.0})
    assert xf.split_factor_after(s, D("2021-01-31")) == 40.0
    assert xf.split_factor_after(s, D("2022-01-30")) == 10.0
    assert xf.split_factor_after(None, D("2022-01-30")) == 1.0


def test_slim_pastreaza_doar_etichetele_folosite():
    cf = _full_cf()
    cf["facts"]["us-gaap"]["SomethingElse"] = {"units": {"USD": [F("2019-01-01", "2019-12-31", 1.0, "2020-02-20")]}}
    cf["facts"]["ifrs-full"] = {"Revenue": {"units": {}}}
    s = sec_xbrl.slim(cf)
    assert "SomethingElse" not in s["facts"]["us-gaap"] and "NetIncomeLoss" in s["facts"]["us-gaap"]
    assert s["facts"]["ifrs-full"] == {}
    assert xf.annual_table(s).loc[D("2019-12-31"), "net_income"] == 200.0


def test_companyfacts_url():
    assert sec_xbrl.companyfacts_url(320193) == "https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json"
