"""Teste pentru analytics/fundamentals.py. Valorile de referință sunt calculate de mână
(calculul e scris în comentarii), nu cu funcțiile testate."""
import math

import pandas as pd
import pytest

from analytics import fundamentals as F


# --- Date de test ------------------------------------------------------------

def _frame(rows, dates):
    return pd.DataFrame(rows, index=pd.to_datetime(dates)).T


QUARTERS = ["2026-06-30", "2026-03-31", "2025-12-31", "2025-09-30"]
YEARS = ["2025-12-31", "2024-12-31", "2023-12-31"]


def quarterly_cashflow():
    return _frame({"Operating Cash Flow": [30.0, 25.0, 35.0, 20.0],
                   "Capital Expenditure": [-5.0, -6.0, -4.0, -5.0]}, QUARTERS)


def annual_cashflow():
    return _frame({"Operating Cash Flow": [120.0, 100.0, 90.0],
                   "Capital Expenditure": [-20.0, -25.0, -26.0]}, YEARS)


def annual_balance():
    return _frame({"Total Debt": [300.0, 320.0, 310.0],
                   "Cash Cash Equivalents And Short Term Investments": [100.0, 80.0, 70.0],
                   "Ordinary Shares Number": [11.0, 11.0, 11.0]}, YEARS)


def annual_income():
    return _frame({"Diluted Average Shares": [10.0, 10.5, 11.0],
                   "Interest Expense": [15.0, 16.0, 15.5],
                   "Tax Provision": [20.0, 18.0, 17.0],
                   "Pretax Income": [100.0, 90.0, 85.0]}, YEARS)


# --- Citire ------------------------------------------------------------------

def test_stmt_value_ia_cea_mai_recenta_coloana_indiferent_de_ordine():
    df = annual_cashflow()
    assert F.stmt_value(df, F.CFO_ROWS) == 120.0
    assert F.stmt_value(df[df.columns[::-1]], F.CFO_ROWS) == 120.0   # coloane inversate
    assert F.stmt_value(df, F.CFO_ROWS, col=2) == 90.0


def test_stmt_value_lipsa_da_none_nu_zero():
    df = annual_cashflow()
    assert F.stmt_value(df, ("Rând Inexistent",)) is None
    assert F.stmt_value(df, F.CFO_ROWS, col=9) is None
    assert F.stmt_value(None, F.CFO_ROWS) is None
    assert F.stmt_value(pd.DataFrame(), F.CFO_ROWS) is None
    df.loc["Operating Cash Flow", df.columns[0]] = float("nan")
    assert F.stmt_value(df, F.CFO_ROWS) is None


def test_stmt_value_incearca_numele_alternative():
    df = _frame({"Total Cash From Operating Activities": [50.0]}, ["2025-12-31"])
    assert F.stmt_value(df, F.CFO_ROWS) == 50.0


def test_ttm_sum_aduna_patru_trimestre():
    # 30 + 25 + 35 + 20 = 110
    assert F.ttm_sum(quarterly_cashflow(), F.CFO_ROWS) == 110.0


def test_ttm_sum_respinge_trimestru_lipsa_si_raportari_semestriale():
    df = quarterly_cashflow()
    df.loc["Operating Cash Flow", df.columns[1]] = float("nan")
    assert F.ttm_sum(df, F.CFO_ROWS) is None
    assert F.ttm_sum(quarterly_cashflow().iloc[:, :3], F.CFO_ROWS) is None
    semestrial = _frame({"Operating Cash Flow": [60.0, 55.0, 50.0, 45.0]},
                        ["2026-06-30", "2025-12-31", "2025-06-30", "2024-12-31"])
    assert F.ttm_sum(semestrial, F.CFO_ROWS) is None   # 4 semestre = 2 ani, nu TTM


# --- FCF ---------------------------------------------------------------------

def test_fcf_este_cfo_minus_capex_indiferent_de_semn():
    assert F.free_cash_flow(120.0, -20.0) == 100.0   # convenția yfinance
    assert F.free_cash_flow(120.0, 20.0) == 100.0    # capex raportat pozitiv
    assert F.free_cash_flow(10.0, -30.0) == -20.0    # FCF negativ rămâne negativ


def test_fcf_lipsa_nu_devine_zero():
    assert F.free_cash_flow(None, -20.0) is None
    assert F.free_cash_flow(120.0, None) is None     # capex lipsă NU e capex zero
    assert F.free_cash_flow(float("nan"), -20.0) is None


def test_fcf_ttm():
    # CFO 110, capex 5 + 6 + 4 + 5 = 20  ->  FCF 90
    assert F.fcf_ttm(quarterly_cashflow()) == 90.0
    assert F.fcf_ttm(None) is None


def test_fcf_history_cronologic_si_fara_ani_incompleti():
    hist = F.fcf_history(annual_cashflow())
    # 2023: 90 - 26 = 64; 2024: 100 - 25 = 75; 2025: 120 - 20 = 100
    assert list(hist.values) == [64.0, 75.0, 100.0]
    assert list(hist.index.year) == [2023, 2024, 2025]
    df = annual_cashflow()
    df.loc["Capital Expenditure", pd.Timestamp("2024-12-31")] = float("nan")
    assert list(F.fcf_history(df).index.year) == [2023, 2025]


def test_fcf_cagr():
    hist = F.fcf_history(annual_cashflow())
    # 64 -> 100 în 2 ani: (100/64)^(1/2) - 1 = 1,25 - 1 = 25%
    assert F.fcf_cagr(hist) == pytest.approx(0.25, abs=1e-3)
    assert F.fcf_cagr(pd.Series([-5.0, 100.0], index=pd.to_datetime(["2023-12-31", "2025-12-31"]))) is None
    assert F.fcf_cagr(hist.iloc[:1]) is None


def test_fcf_average():
    hist = F.fcf_history(annual_cashflow())           # 64, 75, 100
    assert F.fcf_average(hist) == pytest.approx((64 + 75 + 100) / 3)
    assert F.fcf_average(hist, years=2) == pytest.approx(87.5)
    assert F.fcf_average(hist.iloc[:1]) is None
    assert F.fcf_average(None) is None


def test_fcf_distortion_warning():
    # capex 7,6 din CFO 8,23 = 92% (cazul unui ciclu de investiții)
    assert "92%" in F.fcf_distortion_warning(8.23, -7.60, 0.63, 5.0)
    # capex modest, dar FCF curent 40 față de media 100
    assert "40%" in F.fcf_distortion_warning(100.0, -60.0, 40.0, 100.0)
    assert F.fcf_distortion_warning(100.0, -20.0, 80.0, 90.0) is None
    assert F.fcf_distortion_warning(None, None, None, None) is None


# --- Datorie netă și acțiuni -------------------------------------------------

def test_net_debt():
    assert F.net_debt(annual_balance()) == 200.0     # 300 - 100
    cash_net = _frame({"Total Debt": [50.0], "Cash And Cash Equivalents": [80.0]}, ["2025-12-31"])
    assert F.net_debt(cash_net) == -30.0             # numerar net
    assert F.net_debt(_frame({"Total Debt": [50.0]}, ["2025-12-31"])) is None


def test_total_debt_din_componente_doar_cand_exista_ambele():
    both = _frame({"Long Term Debt": [70.0], "Current Debt": [30.0]}, ["2025-12-31"])
    assert F.total_debt(both) == 100.0
    assert F.total_debt(_frame({"Long Term Debt": [70.0]}, ["2025-12-31"])) is None


def test_diluted_shares_si_rezerva_nediluata():
    assert F.diluted_shares(annual_income(), annual_balance()) == (10.0, True)
    assert F.diluted_shares(None, annual_balance()) == (11.0, False)
    assert F.diluted_shares(None, None) == (None, False)


# --- Excluderi ---------------------------------------------------------------

def test_dcf_nu_se_aplica_sectorului_financiar():
    assert F.dcf_not_applicable_reason("Financial Services", "Banks - Diversified", "JPM")
    assert F.dcf_not_applicable_reason("Real Estate", "REIT - Industrial", "PLD")
    assert F.dcf_not_applicable_reason(None, None, "TLV.RO")      # Yahoo nu dă sector la BVB
    assert F.dcf_not_applicable_reason(None, None, "tlv.ro")
    assert F.dcf_not_applicable_reason("Technology", "Consumer Electronics", "AAPL") is None
    assert F.dcf_not_applicable_reason(None, None, "SNP.RO") is None
    assert F.dcf_not_applicable_reason("Real Estate", "Real Estate Services", "CBRE") is None


# --- Costul capitalului ------------------------------------------------------

def test_cost_of_equity_capm():
    # 4% + 1,2 × 5% = 10%
    assert F.cost_of_equity(0.04, 1.2, 0.05) == pytest.approx(0.10)
    assert F.cost_of_equity(None, 1.2) is None
    assert F.cost_of_equity(0.04, None) is None


def test_effective_tax_rate():
    assert F.effective_tax_rate(annual_income()) == pytest.approx(0.20)   # 20 / 100
    high = _frame({"Tax Provision": [60.0], "Pretax Income": [100.0]}, ["2025-12-31"])
    assert F.effective_tax_rate(high) == pytest.approx(0.35)              # plafonat
    loss = _frame({"Tax Provision": [5.0], "Pretax Income": [-10.0]}, ["2025-12-31"])
    assert F.effective_tax_rate(loss) is None


def test_wacc():
    # ke = 10%; kd = 10 / 200 = 5%; t = 20%; E = 800, D = 200 -> ponderi 0,8 / 0,2
    # WACC = 0,8 × 10% + 0,2 × 5% × (1 - 0,2) = 8% + 0,8% = 8,8%
    w = F.wacc(0.04, 1.2, 800.0, 200.0, interest_expense=-10.0, tax_rate=0.20, erp=0.05)
    assert w["wacc"] == pytest.approx(0.088)
    assert w["ke"] == pytest.approx(0.10) and w["kd"] == pytest.approx(0.05)
    assert w["w_e"] == pytest.approx(0.8) and w["notes"] == []


def test_wacc_fara_datorie_este_costul_capitalului_propriu():
    w = F.wacc(0.04, 1.0, 1000.0, 0.0, erp=0.05)
    assert w["wacc"] == pytest.approx(0.09) and w["w_d"] == 0.0


def test_wacc_aproximarile_sunt_declarate():
    # kd proxy = 4% + 1,5 pp = 5,5%; t lipsă = 0 -> 0,8 × 10% + 0,2 × 5,5% = 9,1%
    w = F.wacc(0.04, 1.2, 800.0, 200.0, interest_expense=None, tax_rate=None, erp=0.05)
    assert w["wacc"] == pytest.approx(0.091)
    assert len(w["notes"]) == 2


def test_wacc_cost_al_datoriei_nerealist_trece_pe_proxy():
    # „Dobândă" 75 la datorie 200 = 37,5% (cazul SNP.RO: costuri financiare, nu doar dobânzi)
    # -> kd = 4% + 1,5 pp = 5,5%; WACC = 0,8 × 10% + 0,2 × 5,5% × 0,8 = 8,88%
    w = F.wacc(0.04, 1.2, 800.0, 200.0, interest_expense=75.0, tax_rate=0.20, erp=0.05)
    assert w["kd"] == pytest.approx(0.055) and w["wacc"] == pytest.approx(0.0888)
    assert any("37.5%" in n for n in w["notes"])
    # prea mic: 1 la 200 = 0,5% < jumătate din rf
    assert F.wacc(0.04, 1.2, 800.0, 200.0, interest_expense=1.0, tax_rate=0.20)["kd"] == pytest.approx(0.055)


def test_wacc_fara_rf_beta_sau_capitalizare_este_none():
    assert F.wacc(None, 1.2, 800.0, 200.0) is None
    assert F.wacc(0.04, None, 800.0, 200.0) is None     # beta lipsă NU se presupune 1
    assert F.wacc(0.04, 1.2, None, 200.0) is None


# --- DCF ---------------------------------------------------------------------

def test_growth_path_scade_liniar_spre_g_terminal():
    assert F.growth_path(0.10, 0.02, 5) == pytest.approx([0.10, 0.08, 0.06, 0.04, 0.02])
    assert F.growth_path(0.10, 0.02, 1) == [0.10]


def test_dcf_exemplu_calculat_de_mana():
    # FCF0 = 100, creștere 10% -> 2% (10, 8, 6, 4, 2), r = 9%, g = 2%, datorie netă 200, 10 acțiuni
    # FCF:  110,000  118,800  125,928  130,96512  133,5844224
    # PV:   100,9174  99,9916  97,2395  92,7790   86,8207      suma = 477,7482
    # TV = 133,5844224 × 1,02 / (0,09 - 0,02) = 1946,5159;  PV(TV) = 1946,5159 / 1,09^5 = 1265,1018
    # EV = 477,7482 + 1265,1018 = 1742,8500; capital = 1542,8500; pe acțiune = 154,2850
    r = F.dcf_fcf(100.0, 0.10, 0.09, 0.02, 200.0, 10.0)
    assert [round(f["fcf"], 4) for f in r["flows"]] == [110.0, 118.8, 125.928, 130.9651, 133.5844]
    assert r["pv_fcf"] == pytest.approx(477.7482, abs=1e-3)
    assert r["pv_terminal"] == pytest.approx(1265.1018, abs=1e-3)
    assert r["enterprise_value"] == pytest.approx(1742.8500, abs=1e-3)
    assert r["equity_value"] == pytest.approx(1542.8500, abs=1e-3)
    assert r["per_share"] == pytest.approx(154.2850, abs=1e-3)
    assert r["terminal_share"] == pytest.approx(1265.1018 / 1742.85, abs=1e-4)
    assert r["reason"] is None and r["warnings"] == []


def test_dcf_numerar_net_creste_valoarea_capitalului():
    r = F.dcf_fcf(100.0, 0.10, 0.09, 0.02, -100.0, 10.0)
    assert r["per_share"] == pytest.approx((1742.8500 + 100.0) / 10.0, abs=1e-3)


@pytest.mark.parametrize("kwargs, fragment", [
    (dict(fcf0=None), "indisponibil"),
    (dict(fcf0=-50.0), "negativ"),
    (dict(fcf0=0.0), "negativ"),
    (dict(shares=None), "acțiuni"),
    (dict(shares=0.0), "acțiuni"),
    (dict(net_debt_value=None), "Datoria netă"),
    (dict(terminal_growth=0.04), "3%"),
    (dict(discount_rate=0.02, terminal_growth=0.02), "mai mare"),
    (dict(discount_rate=0.01, terminal_growth=0.02), "mai mare"),
    (dict(net_debt_value=5000.0), "capital propriu negativ"),
])
def test_dcf_refuza_cazurile_fara_sens_si_spune_de_ce(kwargs, fragment):
    base = dict(fcf0=100.0, growth=0.10, discount_rate=0.09, terminal_growth=0.02,
                net_debt_value=200.0, shares=10.0)
    base.update(kwargs)
    r = F.dcf_fcf(**base)
    assert r["per_share"] is None
    assert fragment in r["reason"]


def test_dcf_avertizeaza_cand_r_minus_g_e_sub_2pp():
    r = F.dcf_fcf(100.0, 0.05, 0.045, 0.03, 0.0, 10.0)
    assert r["per_share"] is not None
    assert any("instabil" in w for w in r["warnings"])
    assert F.dcf_fcf(100.0, 0.05, 0.05, 0.03, 0.0, 10.0)["warnings"] != ["x"]  # exact 2 pp: fără instabil
    assert not any("instabil" in w for w in F.dcf_fcf(100.0, 0.05, 0.05, 0.03, 0.0, 10.0)["warnings"])


def test_dcf_sensitivity():
    table = F.dcf_sensitivity(100.0, 0.10, 200.0, 10.0, [0.08, 0.09, 0.02], [0.02, 0.03])
    assert table.shape == (3, 2)
    assert table.loc[0.09, 0.02] == pytest.approx(154.2850, abs=1e-3)
    assert table.loc[0.08, 0.02] > table.loc[0.09, 0.02]      # scont mai mic -> valoare mai mare
    assert table.loc[0.09, 0.03] > table.loc[0.09, 0.02]      # g mai mare -> valoare mai mare
    assert math.isnan(table.loc[0.02, 0.02]) and math.isnan(table.loc[0.02, 0.03])


# --- Asamblarea intrărilor ---------------------------------------------------

def test_dcf_inputs_prefera_ttm_si_bilantul_cel_mai_recent():
    q_balance = _frame({"Total Debt": [280.0], "Cash And Cash Equivalents": [130.0]}, ["2026-06-30"])
    d = F.dcf_inputs(annual_income(), annual_balance(), annual_cashflow(), quarterly_cashflow(), q_balance)
    assert d["fcf"] == 90.0 and d["fcf_basis"].startswith("TTM")
    assert d["net_debt"] == 150.0 and d["balance_date"] == pd.Timestamp("2026-06-30")
    assert d["shares"] == 10.0 and d["shares_diluted"] is True
    assert d["tax_rate"] == pytest.approx(0.20) and d["interest_expense"] == 15.0
    assert d["fcf_cagr"] == pytest.approx(0.25, abs=1e-3)
    assert d["fcf_average"] == pytest.approx(239 / 3) and d["fcf_average_years"] == 3


def test_dcf_inputs_cade_pe_anual_cand_lipsesc_trimestrele():
    d = F.dcf_inputs(annual_income(), annual_balance(), annual_cashflow(), None, None)
    assert d["fcf"] == 100.0 and "2025-12-31" in d["fcf_basis"]
    assert d["net_debt"] == 200.0 and d["balance_date"] == pd.Timestamp("2025-12-31")


def test_dcf_inputs_fara_date_nu_inventeaza_nimic():
    d = F.dcf_inputs(None, None, None, None, None)
    assert d["fcf"] is None and d["fcf_basis"] is None
    assert d["net_debt"] is None and d["shares"] is None and d["tax_rate"] is None
    assert d["fcf_history"].empty and d["fcf_cagr"] is None and d["fcf_average"] is None


# --- Indicatori de bază din situații -----------------------------------------

def _ratio_frames():
    income = _frame({"Total Revenue": [1000.0], "Operating Income": [200.0], "Net Income": [120.0],
                     "Diluted Average Shares": [60.0]}, ["2025-12-31"])
    balance = _frame({"Total Assets": [2000.0], "Stockholders Equity": [800.0], "Total Debt": [400.0],
                      "Current Assets": [600.0], "Current Liabilities": [400.0], "Inventory": [100.0],
                      "Cash And Cash Equivalents": [150.0], "Ordinary Shares Number": [50.0]}, ["2025-12-31"])
    return income, balance


def test_ratios_exemplu_calculat_de_mana():
    income, balance = _ratio_frames()
    r = F.ratios_from_statements(income, balance, price=30.0)
    assert r["returnOnEquity"] == pytest.approx(0.15)        # 120 / 800
    assert r["returnOnAssets"] == pytest.approx(0.06)        # 120 / 2000
    assert r["profitMargins"] == pytest.approx(0.12)         # 120 / 1000
    assert r["operatingMargins"] == pytest.approx(0.20)      # 200 / 1000
    assert r["debtToEquity"] == pytest.approx(50.0)          # 400 / 800, în procente ca la Yahoo
    assert r["currentRatio"] == pytest.approx(1.5)           # 600 / 400
    assert r["quickRatio"] == pytest.approx(1.25)            # (600 - 100) / 400
    assert r["trailingEps"] == pytest.approx(2.0)            # 120 / 60 acțiuni diluate
    assert r["bookValue"] == pytest.approx(16.0)             # 800 / 50 acțiuni la sfârșit de perioadă
    assert r["trailingPE"] == pytest.approx(15.0)            # 30 / 2
    assert r["priceToBook"] == pytest.approx(1.875)          # 30 / 16
    assert r["totalDebt"] == 400.0 and r["totalCash"] == 150.0


def test_ratios_prefera_ttm_si_bilantul_trimestrial():
    income, balance = _ratio_frames()
    q_income = _frame({"Total Revenue": [300.0, 300.0, 300.0, 300.0], "Net Income": [45.0, 45.0, 45.0, 45.0]}, QUARTERS)
    q_balance = _frame({"Total Assets": [2400.0], "Stockholders Equity": [900.0]}, ["2026-06-30"])
    r = F.ratios_from_statements(income, balance, quarterly_income=q_income, quarterly_balance=q_balance)
    assert r["netIncomeToCommon"] == 180.0 and r["totalRevenue"] == 1200.0
    assert r["returnOnEquity"] == pytest.approx(0.20)        # 180 / 900
    assert r["profitMargins"] == pytest.approx(0.15)


def test_ratios_fara_moneda_verificata_nu_compara_pretul_cu_contabilitatea():
    income, balance = _ratio_frames()
    r = F.ratios_from_statements(income, balance, price=30.0, per_share_ok=False)
    assert r["trailingPE"] is None and r["priceToBook"] is None
    assert r["trailingEps"] is None and r["bookValue"] is None
    assert r["returnOnEquity"] == pytest.approx(0.15)        # rapoartele fără preț rămân valabile


def test_ratios_pierdere_capital_negativ_si_lipsuri():
    income, balance = _ratio_frames()
    income.loc["Net Income"] = -50.0
    r = F.ratios_from_statements(income, balance, price=30.0)
    assert r["trailingPE"] is None and r["trailingEps"] == pytest.approx(-50 / 60)
    assert r["returnOnEquity"] == pytest.approx(-0.0625)
    balance.loc["Stockholders Equity"] = -100.0
    r = F.ratios_from_statements(income, balance, price=30.0)
    assert r["returnOnEquity"] is None and r["debtToEquity"] is None and r["priceToBook"] is None
    r = F.ratios_from_statements(income, balance.drop(index="Inventory"), price=30.0)
    assert r["quickRatio"] is None                            # stoc lipsă NU e stoc zero
    assert all(v is None for v in F.ratios_from_statements(None, None).values())


# --- Graham ------------------------------------------------------------------

def test_graham_number():
    # √(22,5 × 2 × 16) = √720 = 26,8328
    assert F.graham_number(2.0, 16.0) == pytest.approx(26.8328, abs=1e-4)
    for eps, bv in ((0.0, 16.0), (-1.0, 16.0), (2.0, -3.0), (None, 16.0), (2.0, None)):
        assert F.graham_number(eps, bv) is None            # EPS ≤ 0 -> N/A, nu 0


def test_graham_revised():
    # 2 × (8,5 + 2 × 5) × 4,4 / 5,5 = 2 × 18,5 × 0,8 = 29,6
    assert F.graham_revised(2.0, 5.0, 5.5) == pytest.approx(29.6)
    # la Y = 4,4% factorul de dobândă e 1: 2 × 18,5 = 37
    assert F.graham_revised(2.0, 5.0, 4.4) == pytest.approx(37.0)
    # creșterea e plafonată la 15%: 2 × (8,5 + 30) × 0,8 = 61,6
    assert F.graham_revised(2.0, 40.0, 5.5) == pytest.approx(61.6)
    assert F.graham_revised(0.0, 5.0, 5.5) is None
    assert F.graham_revised(-1.0, 5.0, 5.5) is None
    assert F.graham_revised(2.0, 5.0, None) is None
    assert F.graham_revised(2.0, -5.0, 5.5) is None        # 8,5 − 10 < 0


# --- Altman ------------------------------------------------------------------

def _altman_frames():
    income = _frame({"Total Revenue": [1500.0], "EBIT": [150.0]}, ["2025-12-31"])
    balance = _frame({"Total Assets": [1000.0], "Current Assets": [400.0], "Current Liabilities": [200.0],
                      "Retained Earnings": [300.0], "Total Liabilities Net Minority Interest": [600.0],
                      "Stockholders Equity": [400.0]}, ["2025-12-31"])
    return income, balance


def test_altman_exemplu_calculat_de_mana():
    income, balance = _altman_frames()
    r = F.altman_z(income, balance, market_cap=1200.0)
    # X1 = 200/1000 = 0,2; X2 = 0,3; X3 = 0,15; X4 = 1200/600 = 2; X5 = 1,5
    # Z = 0,24 + 0,42 + 0,495 + 1,2 + 1,5 = 3,855
    assert r["z"] == pytest.approx(3.855)
    # X4' = 400/600 = 0,6667; Z'' = 1,312 + 0,978 + 1,008 + 0,7 = 3,998
    assert r["z2"] == pytest.approx(3.998)
    assert r["missing"] == []


def test_altman_lipsa_unei_componente_da_none_nu_scor_partial():
    income, balance = _altman_frames()
    r = F.altman_z(income, balance, market_cap=None)
    assert r["z"] is None and r["z2"] == pytest.approx(3.998) and r["missing"] == ["X4"]
    r = F.altman_z(income, balance.drop(index="Retained Earnings"), market_cap=1200.0)
    assert r["z"] is None and r["z2"] is None and "X2" in r["missing"]
    r = F.altman_z(None, None)
    assert r["z"] is None and r["z2"] is None


def test_altman_nu_plafoneaza_scorul():
    income, balance = _altman_frames()
    assert F.altman_z(income, balance, market_cap=600000.0)["z"] == pytest.approx(602.655)   # X4 = 1000


def test_altman_varianta_si_zone():
    assert F.altman_variant("Financial Services", "JPM") is None
    assert F.altman_variant(None, "TLV.RO") is None
    assert F.altman_variant("Energy", "XOM") == "Z"
    assert F.altman_variant("Energy", "SNP.RO") == "Z''"       # piață emergentă
    assert F.altman_variant("Technology", "MSFT") == "Z''"
    assert F.altman_variant(None, "XYZ") == "Z''"
    assert F.altman_zone(3.0, "Z") == "safe" and F.altman_zone(2.5, "Z") == "grey" and F.altman_zone(1.8, "Z") == "distress"
    assert F.altman_zone(2.7, "Z''") == "safe" and F.altman_zone(1.5, "Z''") == "grey" and F.altman_zone(1.0, "Z''") == "distress"
    assert F.altman_zone(None, "Z") is None and F.altman_zone(2.0, None) is None


# --- Piotroski ---------------------------------------------------------------

def _piotroski_frames():
    years = ["2025-12-31", "2024-12-31"]
    income = _frame({"Net Income": [100.0, 80.0], "Total Revenue": [1000.0, 900.0],
                     "Gross Profit": [400.0, 342.0]}, years)
    balance = _frame({"Total Assets": [1000.0, 1000.0], "Long Term Debt": [200.0, 250.0],
                      "Current Assets": [300.0, 280.0], "Current Liabilities": [150.0, 160.0],
                      "Ordinary Shares Number": [50.0, 50.0]}, years)
    cashflow = _frame({"Operating Cash Flow": [130.0, 90.0]}, years)
    return income, balance, cashflow


def test_piotroski_toate_criteriile_trecute():
    # ROA 10% > 0 și > 8%; CFO 130 > 0 și > profit 100; datorie TL/active 20% < 25%;
    # lichiditate 2,00 > 1,75; acțiuni neschimbate; marjă brută 40% > 38%; rotație 1,00 > 0,90
    r = F.piotroski(*_piotroski_frames())
    assert r["passed"] == 9 and r["evaluable"] == 9
    assert [c["passed"] for c in r["criteria"]] == [True] * 9
    assert r["year"] == pd.Timestamp("2025-12-31") and r["prior_year"] == pd.Timestamp("2024-12-31")


def test_piotroski_criterii_picate():
    income, balance, cashflow = _piotroski_frames()
    income.loc["Net Income", pd.Timestamp("2025-12-31")] = 60.0        # ROA 6% < 8% -> criteriul 3 picat
    balance.loc["Ordinary Shares Number", pd.Timestamp("2025-12-31")] = 55.0   # +10% acțiuni -> 7 picat
    cashflow.loc["Operating Cash Flow", pd.Timestamp("2025-12-31")] = 50.0     # CFO 50 < profit 60 -> 4 picat
    r = F.piotroski(income, balance, cashflow)
    assert [c["passed"] for c in r["criteria"]] == [True, True, False, False, True, True, False, True, True]
    assert r["passed"] == 6 and r["evaluable"] == 9


def test_piotroski_date_lipsa_sunt_na_nu_picate():
    income, balance, cashflow = _piotroski_frames()
    r = F.piotroski(income.drop(index="Gross Profit"), balance.drop(index=["Current Assets", "Current Liabilities"]), cashflow)
    assert r["criteria"][5]["passed"] is None and r["criteria"][7]["passed"] is None     # ca la o bancă
    assert r["passed"] == 7 and r["evaluable"] == 7
    one_year = F.piotroski(income.iloc[:, :1], balance.iloc[:, :1], cashflow.iloc[:, :1])
    assert [c["passed"] for c in one_year["criteria"]] == [True, True, None, True, None, None, None, None, None]
    empty = F.piotroski(None, None, None)
    assert empty["passed"] == 0 and empty["evaluable"] == 0


def test_piotroski_fara_datorie_in_ambii_ani_trece_criteriul_5():
    income, balance, cashflow = _piotroski_frames()
    balance.loc["Long Term Debt"] = 0.0
    assert F.piotroski(income, balance, cashflow)["criteria"][4]["passed"] is True


# --- Îndatorare --------------------------------------------------------------

def test_leverage_ratios():
    income = _frame({"EBITDA": [250.0], "EBIT": [200.0], "Interest Expense": [-25.0]}, ["2025-12-31"])
    balance = _frame({"Total Debt": [600.0], "Cash And Cash Equivalents": [100.0]}, ["2025-12-31"])
    r = F.leverage_ratios(income, balance)
    assert r["net_debt_to_ebitda"] == pytest.approx(2.0)       # (600 - 100) / 250
    assert r["interest_coverage"] == pytest.approx(8.0)        # 200 / 25


def test_leverage_ratios_cazuri_limita():
    balance = _frame({"Total Debt": [50.0], "Cash And Cash Equivalents": [150.0]}, ["2025-12-31"])
    income = _frame({"EBITDA": [100.0], "EBIT": [80.0]}, ["2025-12-31"])
    r = F.leverage_ratios(income, balance)
    assert r["net_debt_to_ebitda"] == pytest.approx(-1.0)      # numerar net
    assert r["interest_coverage"] is None                       # dobânda nu e raportată
    loss = _frame({"EBITDA": [-10.0], "EBIT": [-30.0], "Interest Expense": [5.0]}, ["2025-12-31"])
    r = F.leverage_ratios(loss, balance)
    assert r["net_debt_to_ebitda"] is None                      # EBITDA negativ
    assert r["interest_coverage"] == pytest.approx(-6.0)        # EBIT negativ nu acoperă dobânda
    assert F.leverage_ratios(None, None)["net_debt_to_ebitda"] is None
