"""Faza 3c-1: adaptorul SEC → situații în formatul Yahoo, versiunile incrementale, SIC → sector.
Date sintetice; valorile de referință sunt calculate de mână în comentarii."""
from datetime import date

import pandas as pd
import pytest

from analytics import backtest_fundamental as bt
from analytics import fundamentals as fund
from analytics import sec_statements as ss
from analytics import xbrl_facts as xf
from analytics.global_fundamental import fundamental_score
from analytics.sic_sector import sic_sector
from data import sec_xbrl

D = date.fromisoformat
T = pd.Timestamp

# an fiscal → valorile din 10-K (fluxuri pe an calendaristic, bilanț la 31.12)
FLOWS = {   # Revenues, NetIncomeLoss, EPS diluat, acțiuni diluate, GrossProfit, OperatingIncomeLoss, profit brut,
            # impozit, dobânzi, amortizare, CFO, capex
    2021: (800.0, 60.0, 0.60, 100.0, 300.0, 90.0, 75.0, 15.0, 20.0, 40.0, 100.0, 40.0),
    2022: (900.0, 80.0, 0.80, 100.0, 342.0, 120.0, 100.0, 20.0, 20.0, 45.0, 130.0, 50.0),
    2023: (1000.0, 100.0, 1.00, 100.0, 400.0, 150.0, 130.0, 30.0, 20.0, 50.0, 160.0, 60.0),
}
FLOW_TAGS = ("Revenues", "NetIncomeLoss", "EarningsPerShareDiluted", "WeightedAverageNumberOfDilutedSharesOutstanding",
             "GrossProfit", "OperatingIncomeLoss",
             "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
             "IncomeTaxExpenseBenefit", "InterestExpense", "DepreciationDepletionAndAmortization",
             "NetCashProvidedByUsedInOperatingActivities", "PaymentsToAcquirePropertyPlantAndEquipment")
STOCK = {   # Assets, AssetsCurrent, LiabilitiesCurrent, Liabilities, StockholdersEquity, RetainedEarnings, InventoryNet,
            # numerar, ShortTermInvestments, LongTermDebtCurrent, LongTermDebtNoncurrent, LongTermDebt, acțiuni în circulație
    2021: (1800.0, 500.0, 280.0, 1080.0, 720.0, 350.0, 80.0, 100.0, 30.0, 50.0, 550.0, 600.0, 100.0),
    2022: (1900.0, 550.0, 300.0, 1100.0, 800.0, 420.0, 90.0, 150.0, 40.0, 50.0, 500.0, 550.0, 100.0),
    2023: (2000.0, 540.0, 300.0, 1100.0, 900.0, 500.0, 100.0, 200.0, 50.0, 50.0, 450.0, 500.0, 99.0),
}
STOCK_TAGS = ("Assets", "AssetsCurrent", "LiabilitiesCurrent", "Liabilities", "StockholdersEquity",
              "RetainedEarningsAccumulatedDeficit", "InventoryNet", "CashAndCashEquivalentsAtCarryingValue",
              "ShortTermInvestments", "LongTermDebtCurrent", "LongTermDebtNoncurrent", "LongTermDebt",
              "CommonStockSharesOutstanding")
FILED = {2021: "2022-02-18", 2022: "2023-02-17", 2023: "2024-02-16"}      # vineri → cunoscute luni


def _unit(tag):
    for unit, _, lst in xf.CONCEPTS.values():
        for tax, name in lst:
            if name == tag:
                return tax, unit
    raise KeyError(tag)


def _company(extra=None):
    """Fiecare 10-K are anul curent și comparativele anului anterior (aceleași valori)."""
    facts = {}

    def put(tag, row):
        tax, unit = _unit(tag)
        facts.setdefault(tax, {}).setdefault(tag, {"units": {unit: []}})["units"][unit].append(row)

    for fy, filed in FILED.items():
        for y in (fy - 1, fy):
            if y not in FLOWS:
                continue
            for tag, v in zip(FLOW_TAGS, FLOWS[y]):
                put(tag, {"start": f"{y}-01-01", "end": f"{y}-12-31", "val": v, "form": "10-K", "filed": filed,
                          "accn": f"k{fy}"})
            for tag, v in zip(STOCK_TAGS, STOCK[y]):
                put(tag, {"end": f"{y}-12-31", "val": v, "form": "10-K", "filed": filed, "accn": f"k{fy}"})
    for tag, row in (extra or []):
        put(tag, row)
    return {"cik": 1, "entityName": "Test", "facts": facts}


Q_EXTRA = [
    # 10-Q T1 2024: capitalurile de la 31.12.2023 retratate (900 → 910) → schimbă situațiile din 06.05.2024
    ("StockholdersEquity", {"end": "2023-12-31", "val": 910.0, "form": "10-Q", "filed": "2024-05-03", "accn": "q1"}),
    # aceeași valoare a activelor de la 31.12.2023 → nicio schimbare
    ("Assets", {"end": "2023-12-31", "val": 2000.0, "form": "10-Q", "filed": "2024-05-03", "accn": "q1"}),
    # bilanț trimestrial, etichetă unică, departe de încheierea exercițiului → nu contează
    ("StockholdersEquity", {"end": "2023-06-30", "val": 850.0, "form": "10-Q", "filed": "2023-08-04", "accn": "q2"}),
    # flux trimestrial din 10-Q → nu intră în anual
    ("Revenues", {"start": "2023-04-01", "end": "2023-06-30", "val": 250.0, "form": "10-Q", "filed": "2023-08-04",
                  "accn": "q2"}),
    # etichetă cu mai multe variante (plasamente): orice bilanț trimestrial poate muta ancora → contează
    ("ShortTermInvestments", {"end": "2023-09-30", "val": 45.0, "form": "10-Q", "filed": "2023-11-03", "accn": "q3"}),
]


# --- adaptorul ---------------------------------------------------------------

def test_randurile_yahoo_si_valorile_calculate():
    st = ss.statements_from_facts(_company())
    inc, bal, cf = st["income"], st["balance"], st["cashflow"]
    assert list(inc.columns) == [T("2023-12-31"), T("2022-12-31"), T("2021-12-31")]     # cel mai recent primul
    y = T("2023-12-31")
    assert inc.loc["Total Revenue", y] == 1000.0 and inc.loc["Diluted Average Shares", y] == 100.0
    assert inc.loc["EBIT", y] == 150.0                     # 130 profit brut + 20 dobânzi (definiția Yahoo)
    assert inc.loc["EBITDA", y] == 200.0                   # 150 + 50 amortizare
    assert inc.loc["Interest Expense", y] == 20.0
    assert cf.loc["Capital Expenditure", y] == -60.0 and cf.loc["Free Cash Flow", y] == 100.0
    assert bal.loc["Current Debt", y] == 50.0              # partea curentă; fără împrumuturi pe termen scurt
    assert bal.loc["Long Term Debt", y] == 450.0
    assert bal.loc["Total Debt", y] == 500.0               # 50 + 450
    assert bal.loc["Cash Cash Equivalents And Short Term Investments", y] == 250.0
    assert bal.loc["Ordinary Shares Number", y] == 99.0
    assert st["notes"]["current_debt_derived"] == [D("2021-12-31"), D("2022-12-31"), D("2023-12-31")]
    assert st["notes"]["gross_profit_proxy"] == [] and st["notes"]["conflicts"] == []


def test_functiile_aplicatiei_pe_adaptor_valori_de_mana():
    st = ss.statements_from_facts(_company())
    inc, bal, cf = st["income"], st["balance"], st["cashflow"]
    # Altman, capitalizare 2.200: X1 = (540 − 300) / 2000 = 0,12; X2 = 500 / 2000 = 0,25; X3 = 150 / 2000 = 0,075;
    # X4 = 2200 / 1100 = 2; X5 = 1000 / 2000 = 0,5
    # Z  = 1,2·0,12 + 1,4·0,25 + 3,3·0,075 + 0,6·2 + 1,0·0,5 = 0,144 + 0,35 + 0,2475 + 1,2 + 0,5 = 2,4415
    # Z'' = 6,56·0,12 + 3,26·0,25 + 6,72·0,075 + 1,05·(900 / 1100) = 0,7872 + 0,815 + 0,504 + 0,859091 = 2,965291
    az = fund.altman_z(inc, bal, market_cap=2200.0)
    assert az["z"] == pytest.approx(2.4415) and az["z2"] == pytest.approx(2.965291, abs=1e-6)
    assert az["retained_is_proxy"] is False
    # Piotroski 2023 față de 2022: ROA 0,05 > 0 ✓; CFO > 0 ✓; ROA 0,05 > 0,0421 ✓; CFO 160 > 100 ✓;
    # datorie TL / active 0,225 < 0,263 ✓; lichiditate 540/300 = 1,80 < 550/300 = 1,83 ✗; acțiuni 99 ≤ 100 ✓;
    # marjă brută 0,40 > 0,38 ✓; rotație 0,50 > 0,4737 ✓ → 8 din 9
    p = fund.piotroski(inc, bal, cf)
    assert (p["passed"], p["evaluable"]) == (8, 9)
    assert [c["passed"] for c in p["criteria"]][5] is False
    # Îndatorare: datorie netă = 500 − (200 + 50) = 250; / EBITDA 200 = 1,25; acoperire = 150 / 20 = 7,5
    lev = fund.leverage_ratios(inc, bal)
    assert lev["net_debt"] == 250.0 and lev["net_debt_to_ebitda"] == 1.25 and lev["interest_coverage"] == 7.5
    d = fund.dcf_inputs(inc, bal, cf)
    assert d["fcf"] == 100.0 and d["net_debt"] == 250.0 and d["shares"] == 100.0 and d["shares_diluted"] is True
    assert d["tax_rate"] == pytest.approx(30.0 / 130.0)
    assert d["fcf_average"] == pytest.approx((60.0 + 80.0 + 100.0) / 3)


def test_rezerve_marcate_marja_bruta_datorii_ebitda():
    c = _company()
    g = c["facts"]["us-gaap"]
    del g["GrossProfit"], g["Liabilities"], g["IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest"]
    rows = []
    for fy, filed in FILED.items():
        rows.append(("CostOfRevenue", {"start": f"{fy}-01-01", "end": f"{fy}-12-31", "val": 600.0 if fy == 2023 else 500.0,
                                       "form": "10-K", "filed": filed, "accn": f"k{fy}"}))
        rows.append(("LiabilitiesAndStockholdersEquity", {"end": f"{fy}-12-31", "val": STOCK[fy][0], "form": "10-K",
                                                          "filed": filed, "accn": f"k{fy}"}))
        rows.append(("StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
                     {"end": f"{fy}-12-31", "val": STOCK[fy][4] + 10.0, "form": "10-K", "filed": filed,
                      "accn": f"k{fy}"}))
    for tag, row in rows:
        tax, unit = _unit(tag)
        g.setdefault(tag, {"units": {unit: []}})["units"][unit].append(row)
    st = ss.statements_from_facts(c)
    y = T("2023-12-31")
    assert st["income"].loc["Gross Profit", y] == 400.0                 # 1000 − 600
    assert st["balance"].loc["Total Liabilities Net Minority Interest", y] == 1090.0   # 2000 − 910
    assert "EBIT" not in st["income"].index                              # fără profit brut → fără EBIT
    assert st["income"].loc["EBITDA", y] == 200.0                        # profit din exploatare 150 + 50
    assert D("2023-12-31") in st["notes"]["gross_profit_proxy"]
    assert D("2023-12-31") in st["notes"]["liabilities_proxy"]
    assert D("2023-12-31") in st["notes"]["ebitda_on_operating_income"]
    # Altman ia EBIT-ul din „Operating Income” (ca la Yahoo fără rândul EBIT): X3 = 150 / 2000
    assert fund.altman_z(st["income"], st["balance"])["components"]["X3"] == pytest.approx(0.075)


def test_datoria_curenta_debtcurrent_si_imprumuturi():
    c = _company([("DebtCurrent", {"end": "2023-12-31", "val": 80.0, "form": "10-K", "filed": FILED[2023],
                                   "accn": "k2023"}),
                  ("ShortTermBorrowings", {"end": "2022-12-31", "val": 25.0, "form": "10-K", "filed": FILED[2023],
                                           "accn": "k2023"})])
    st = ss.statements_from_facts(c)
    assert st["balance"].loc["Current Debt", T("2023-12-31")] == 80.0           # DebtCurrent are prioritate
    assert st["balance"].loc["Total Debt", T("2023-12-31")] == 530.0             # 80 + 450
    assert st["balance"].loc["Current Debt", T("2022-12-31")] == 75.0           # 50 + 25 împrumuturi
    assert st["balance"].loc["Total Debt", T("2022-12-31")] == 575.0             # 75 + 500


def test_actiunile_bilant_split_si_scara():
    c = _company()
    rows = c["facts"]["us-gaap"]["CommonStockSharesOutstanding"]["units"]["shares"]
    for r in rows:
        if r["end"] == "2021-12-31":
            r["val"] = 0.1                                                    # raportate în mii (100 / 1.000)
    splits = pd.Series({T("2024-06-10"): 2.0})
    st = ss.statements_from_facts(c, splits=splits)
    b, i = st["balance"], st["income"]
    assert b.loc["Ordinary Shares Number", T("2023-12-31")] == 198.0             # 99 × 2 (split după depunere)
    assert i.loc["Diluted Average Shares", T("2023-12-31")] == 200.0
    assert b.loc["Ordinary Shares Number", T("2021-12-31")] == pytest.approx(200.0)   # 0,1 × 2 × 1.000


def test_point_in_time_si_niciun_an():
    c = _company()
    assert ss.statements_from_facts(c, asof=D("2022-02-18")) is None              # depus vineri, cunoscut luni
    st = ss.statements_from_facts(c, asof=D("2022-02-21"))
    assert list(st["income"].columns) == [T("2021-12-31")]
    st = ss.statements_from_facts(c, asof=D("2023-12-31"))
    assert list(st["income"].columns) == [T("2022-12-31"), T("2021-12-31")]


# --- incremental -------------------------------------------------------------

# a doua etichetă a plasamentelor: întâi doar la un trimestru (nu schimbă nimic la 31.12), apoi și în coloana
# comparativă de la 31.12.2023 → ancora de la 31.12.2023 devine `MarketableSecuritiesCurrent` (52 în loc de 50)
Q_SWITCH = [
    ("MarketableSecuritiesCurrent", {"end": "2024-03-31", "val": 55.0, "form": "10-Q", "filed": "2024-05-03",
                                     "accn": "q1"}),
    ("MarketableSecuritiesCurrent", {"end": "2023-12-31", "val": 52.0, "form": "10-Q", "filed": "2024-08-02",
                                     "accn": "q2b"}),
    ("MarketableSecuritiesCurrent", {"end": "2024-06-30", "val": 56.0, "form": "10-Q", "filed": "2024-08-02",
                                     "accn": "q2b"}),
]


def test_change_dates_doar_10k_si_bilanturile_relevante():
    # 10-K: 18.02.2022, 17.02.2023, 16.02.2024 (vineri) → luni; 10-Q 03.05.2024 (capitaluri retratate) → luni.
    # NU: 04.08.2023 (capitaluri la 30.06, venit trimestrial), 03.11.2023 (plasamente la 30.09, aceeași etichetă).
    c = _company(Q_EXTRA)
    assert ss.change_dates(c) == [D("2022-02-21"), D("2023-02-20"), D("2024-02-19"), D("2024-05-06")]
    # cu a doua etichetă: 02.08.2024 schimbă ancora de la 31.12.2023 → luni 05.08.2024
    c = _company(Q_EXTRA + Q_SWITCH)
    assert ss.change_dates(c) == [D("2022-02-21"), D("2023-02-20"), D("2024-02-19"), D("2024-05-06"),
                                  D("2024-08-05")]


def test_versiunile_incrementale_identice_cu_apelul_direct():
    c = _company(Q_EXTRA + Q_SWITCH)
    splits = pd.Series({T("2024-06-10"): 2.0})
    versions = ss.asof_versions(c, splits=splits)
    dates = pd.date_range("2021-12-15", "2024-12-31", freq="47D").date.tolist() + \
        [D("2022-02-21"), D("2024-05-03"), D("2024-05-06"), D("2023-11-06"), D("2024-08-02"), D("2024-08-05")]
    for d in dates:
        direct = ss.statements_from_facts(c, asof=d, splits=splits)
        inc = ss.version_at(versions, d)
        if direct is None:
            assert inc is None, d
            continue
        for k in ("income", "balance", "cashflow"):
            pd.testing.assert_frame_equal(inc[k], direct[k], obj=f"{k} la {d}")
        assert inc["notes"] == direct["notes"]
    assert ss.version_at(versions, D("2024-05-03"))["balance"].loc["Stockholders Equity", T("2023-12-31")] == 900.0
    assert ss.version_at(versions, D("2024-05-06"))["balance"].loc["Stockholders Equity", T("2023-12-31")] == 910.0
    row = "Cash Cash Equivalents And Short Term Investments"
    before, after = ss.version_at(versions, D("2024-08-02")), ss.version_at(versions, D("2024-08-05"))
    assert before["balance"].loc[row, T("2023-12-31")] == 250.0 and after["balance"].loc[row, T("2023-12-31")] == 252.0


# --- legătura cu testul istoric ----------------------------------------------

def test_inputs_at_lag_zero_pe_sec_si_90_implicit():
    st = ss.statements_from_facts(_company())
    args = (st["income"], st["balance"], st["cashflow"], "Industrials", "TEST", 0.04, 1.0, 5.0)
    inp, fy = bt.inputs_at(D("2024-02-20"), 22.0, *args, lag=pd.DateOffset(days=0))
    assert fy == T("2023-12-31")
    assert inp["shares"] == 100.0 and inp["pe"] == pytest.approx(22.0) and inp["piotroski"]["passed"] == 8
    assert inp["altman"]["variant"] == "Z"
    res = fundamental_score(inp)
    assert res["financial"] is False and res["score"] is not None
    _, fy90 = bt.inputs_at(D("2024-02-20"), 22.0, *args)                          # implicit: 31.12 + 90 zile
    assert fy90 == T("2022-12-31")


def test_slim_implicit_neschimbat_si_cu_toti_indicatorii():
    c = _company()
    assert "Assets" not in sec_xbrl.slim(c)["facts"]["us-gaap"]                 # cardul: ca înainte
    full = sec_xbrl.slim(c, tuple(xf.CONCEPTS))
    assert "Assets" in full["facts"]["us-gaap"] and "Revenues" in full["facts"]["us-gaap"]
    assert set(xf.CARD_INDICATORS).isdisjoint(xf.STATEMENT_CONCEPTS)


# --- SIC → sector -------------------------------------------------------------

@pytest.mark.parametrize("sic, sector, industry", [
    (6021, "Financial Services", None),        # bancă națională (JPM)
    (6331, "Financial Services", None),        # asigurător
    (6798, "Real Estate", "REIT"),
    (6512, "Real Estate", None),
    (2911, "Energy", None),                    # rafinare (XOM)
    (1311, "Energy", None),
    (3571, "Technology", None),                # calculatoare (AAPL)
    (3674, "Technology", None),                # semiconductori (NVDA)
    (3600, "Industrials", None),               # GE (SIC 3600)
    (3661, "Technology", None),                # echipamente de comunicații
    (2834, "Healthcare", None),
    (3841, "Healthcare", None),
    (5331, "Consumer Defensive", None),        # magazine cu reducere
    (5812, "Consumer Cyclical", None),         # restaurante
    (3711, "Consumer Cyclical", None),         # autovehicule
    (3721, "Industrials", None),               # aeronave
    (4911, "Utilities", None),
    (4953, "Industrials", None),               # deșeuri
    (7372, "Technology", None),                # software
    ("4813", "Communication Services", None),
    (None, None, None), ("", None, None), ("abc", None, None), (9999, None, None),
])
def test_sic_sector(sic, sector, industry):
    assert sic_sector(sic) == (sector, industry)


def test_sic_sector_se_potriveste_cu_regulile_aplicatiei():
    bank, _ = sic_sector(6021)
    assert fund.is_financial_issuer(bank) and fund.altman_variant(bank) is None
    reit, ind = sic_sector(6798)
    assert not fund.is_financial_issuer(reit) and fund.dcf_not_applicable_reason(reit, ind) is not None
    assert fund.altman_variant(sic_sector(2911)[0]) == "Z" and fund.altman_variant(sic_sector(7372)[0]) == "Z''"
