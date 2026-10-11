"""Teste pentru registrul de tranzacții. Toate datele sunt inventate (repo-ul e public)."""
import numpy as np
import pandas as pd
import pytest

from analytics import ledger as L

HEADER = L.COLUMNS


def sheet(*rows):
    """Construiește valorile unei foi: antet + rânduri, cu numerele ca text în format românesc."""
    return [HEADER] + [list(r) for r in rows]


BASE = sheet(
    # Date, Symbol, Class, Type, Quantity, Price, Currency, Fees, Amount, Broker, ID, Notes
    ["2024-01-10 10:00:00", "", "Numerar", "DEPOSIT", "", "", "RON", "0", "3000", "TV", "d1", ""],
    ["2024-01-15 10:00:00", "AAA.RO", "Actiuni RO", "BUY", "100", "10", "RON", "5", "-1005", "TV", "t1", ""],
    ["2024-03-01 10:00:00", "AAA.RO", "Actiuni RO", "BUY", "100", "12", "RON", "6", "-1206", "TV", "t2", ""],
    ["2024-06-01 09:00:00", "AAA.RO", "Actiuni RO", "BONUS", "50", "", "RON", "0", "0", "TV", "t3", "mcs gratuite"],
    ["2024-06-20 12:00:00", "AAA.RO", "Actiuni RO", "DIV", "", "", "RON", "1", "49,5", "TV", "t4", "dividend"],
    ["2024-09-02 11:00:00", "AAA.RO", "Actiuni RO", "SELL", "125", "11", "RON", "7", "1368", "TV", "t5", ""],
    ["2024-02-01 10:00:00", "BBB", "Actiuni SUA", "BUY", "2", "50,25", "USD", "0", "-100,5", "XTB", "x1", ""],
    ["2024-05-01 10:00:00", "BBB", "Actiuni SUA", "DIV", "", "", "USD", "0", "1,2", "XTB", "x2", ""],
    ["2024-05-01 10:00:00", "BBB", "Actiuni SUA", "TAX", "", "", "USD", "0", "-0,12", "XTB", "x3", "WHT 10%"],
    ["2024-07-01 10:00:00", "BBB", "Actiuni SUA", "SELL", "2", "60", "USD", "0", "120", "XTB", "x4", ""],
    ["2024-07-02 10:00:00", "BBB", "Actiuni SUA", "FEE", "", "", "USD", "0", "-0,01", "XTB", "x5", "Sec fee"],
    ["2024-01-01 00:00:00", "", "Numerar", "FX", "", "", "USD", "0", "100,5", "XTB", "x0", ""],
)


def test_parse_ledger_tipuri_si_ordine():
    df, problems = L.parse_ledger(BASE)
    assert problems == []
    assert len(df) == 12
    assert df["Date"].is_monotonic_increasing
    assert df.iloc[0]["ID"] == "x0"                                  # sortat cronologic, nu în ordinea foii
    buy = df[df["ID"] == "x1"].iloc[0]
    assert buy["Price"] == pytest.approx(50.25) and buy["Amount"] == pytest.approx(-100.5)
    div = df[df["ID"] == "t4"].iloc[0]
    assert np.isnan(div["Quantity"]) and np.isnan(div["Price"])      # celulă goală = lipsă, nu 0
    assert div["Amount"] == pytest.approx(49.5)


def test_parse_ledger_raporteaza_randurile_gresite():
    values = sheet(
        ["2024-01-15 10:00:00", "AAA.RO", "Actiuni RO", "BUY", "10", "1", "RON", "0", "-10", "TV", "a", ""],
        ["nu e dată", "AAA.RO", "Actiuni RO", "BUY", "10", "1", "RON", "0", "-10", "TV", "b", ""],
        ["2024-01-16 10:00:00", "AAA.RO", "Actiuni RO", "CUMPAR", "10", "1", "RON", "0", "-10", "TV", "c", ""],
        ["2024-01-17 10:00:00", "AAA.RO", "Actiuni RO", "SELL", "", "1", "RON", "0", "10", "TV", "d", ""],
        ["2024-01-18 10:00:00", "AAA.RO", "Actiuni RO", "BUY", "5", "1", "RON", "0", "-5", "TV", "a", ""],
        ["", "", "", "", "", "", "", "", "", "", "", ""],
    )
    df, problems = L.parse_ledger(values)
    assert list(df["ID"]) == ["a"]
    assert problems == ["Rândul 3: dată necitibilă.", "Rândul 4: tip necunoscut „CUMPAR”.",
                        "Rândul 5: SELL fără cantitate.", "Rândul 6: ID duplicat „a” (rând ignorat)."]


def test_parse_ledger_antet_incomplet_si_foaie_goala():
    df, problems = L.parse_ledger([["Date", "Symbol"]])
    assert df.empty and "Coloane lipsă" in problems[0]
    assert L.parse_ledger([])[1] == ["Foaia e goală."]


def test_build_positions_cost_mediu_bonus_si_vanzare():
    df, _ = L.parse_ledger(BASE)
    pos, problems = L.build_positions(df)
    assert problems == []
    a = pos.set_index("Symbol").loc["AAA.RO"]
    # cost: 1005 + 1206 = 2211 pe 200 acțiuni; bonusul de 50 nu schimbă costul -> 2211 / 250 = 8,844 pe acțiune
    # vânzare 125 (jumătate): cost vândut 1105,5 ; realizat 1368 - 1105,5 = 262,5 ; rămân 125 la cost 1105,5
    assert a["Quantity"] == pytest.approx(125.0)
    assert a["CostBasis"] == pytest.approx(1105.5)
    assert a["AvgCost"] == pytest.approx(8.844)
    assert a["Realized"] == pytest.approx(262.5)
    assert a["Income"] == pytest.approx(49.5)
    assert a["Bought"] == pytest.approx(2211.0) and a["Sold"] == pytest.approx(1368.0)

    b = pos.set_index("Symbol").loc["BBB"]
    # poziție închisă complet: realizat 120 - 100,5 = 19,5 ; costul mediu nu se mai afișează
    assert b["Quantity"] == 0.0 and b["CostBasis"] == 0.0
    assert np.isnan(b["AvgCost"])
    assert b["Realized"] == pytest.approx(19.5)
    assert b["Income"] == pytest.approx(1.2) and b["Taxes"] == pytest.approx(-0.12) and b["Fees"] == pytest.approx(-0.01)


def test_build_positions_sold_de_deschidere_si_consolidare():
    values = sheet(
        ["2020-01-01 00:00:00", "TTT.RO", "Actiuni RO", "OPEN", "1000", "2,5", "RON", "0", "0", "TV", "o1", ""],
        ["2020-06-01 09:00:00", "TTT.RO", "Actiuni RO", "SPLIT", "-1000", "", "RON", "0", "0", "TV", "s1", "1/10"],
        ["2020-06-01 09:01:00", "TTT.RO", "Actiuni RO", "SPLIT", "100", "", "RON", "0", "0", "TV", "s2", "1/10"],
        ["2020-07-01 09:00:00", "TTTR01", "Drepturi", "RIGHTS", "100", "", "RON", "0", "0", "TV", "r1", ""],
    )
    df, _ = L.parse_ledger(values)
    pos, problems = L.build_positions(df)
    assert problems == []
    assert list(pos["Symbol"]) == ["TTT.RO"]                         # drepturile nu sunt dețineri
    t = pos.iloc[0]
    # 1000 × 2,5 = 2500 cost; după consolidarea 1/10 rămân 100 acțiuni la 25 fiecare
    assert t["Quantity"] == pytest.approx(100.0)
    assert t["CostBasis"] == pytest.approx(2500.0)
    assert t["AvgCost"] == pytest.approx(25.0)


def test_sold_de_deschidere_in_numerar_fara_cantitate():
    values = sheet(
        ["2020-01-01 00:00:00", "", "Numerar", "OPEN", "", "", "RON", "0", "1000,00", "TV", "o0", ""],
        ["2020-01-03 10:00:00", "AAA.RO", "Actiuni RO", "BUY", "10", "5", "RON", "1", "-51", "TV", "a", ""],
    )
    df, problems = L.parse_ledger(values)
    assert problems == [] and len(df) == 2                           # rândul de numerar nu e respins
    pos, _ = L.build_positions(df)
    assert list(pos["Symbol"]) == ["AAA.RO"]                         # și nu devine o deținere
    assert L.cash_balances(df)["Cash"].iloc[0] == pytest.approx(1000.00 - 51)


def test_build_positions_vanzare_fara_istoric_e_semnalata():
    values = sheet(
        ["2024-01-15 10:00:00", "ZZZ.RO", "Actiuni RO", "BUY", "10", "5", "RON", "0", "-50", "TV", "a", ""],
        ["2024-02-15 10:00:00", "ZZZ.RO", "Actiuni RO", "SELL", "30", "6", "RON", "0", "180", "TV", "b", ""],
    )
    df, _ = L.parse_ledger(values)
    pos, problems = L.build_positions(df)
    assert len(problems) == 1 and "ZZZ.RO" in problems[0] and "lipsește istoric" in problems[0]
    assert pos.iloc[0]["Quantity"] == 0.0                            # nu rămâne cantitate negativă


def test_class_summary_si_numerar():
    df, _ = L.parse_ledger(BASE)
    pos, _ = L.build_positions(df)
    summary = L.class_summary(pos).set_index(["Class", "Currency"])
    ro = summary.loc[("Actiuni RO", "RON")]
    assert ro["OpenPositions"] == 1
    assert ro["CostBasis"] == pytest.approx(1105.5)
    assert ro["Result"] == pytest.approx(262.5 + 49.5)
    us = summary.loc[("Actiuni SUA", "USD")]
    assert us["OpenPositions"] == 0
    assert us["Result"] == pytest.approx(19.5 + 1.2 - 0.12 - 0.01)

    cash = L.cash_balances(df).set_index(["Broker", "Currency"])["Cash"]
    assert cash[("TV", "RON")] == pytest.approx(3000 - 1005 - 1206 + 49.5 + 1368)      # 2206,5
    assert cash[("XTB", "USD")] == pytest.approx(100.5 - 100.5 + 1.2 - 0.12 + 120 - 0.01)


def test_class_cashflows_include_deschiderea():
    values = sheet(
        ["2020-01-01 00:00:00", "TTT.RO", "Actiuni RO", "OPEN", "100", "2", "RON", "0", "0", "TV", "o1", ""],
        ["2020-03-01 00:00:00", "TTT.RO", "Actiuni RO", "BONUS", "10", "", "RON", "0", "0", "TV", "b1", ""],
        ["2021-01-01 00:00:00", "TTT.RO", "Actiuni RO", "SELL", "110", "2", "RON", "0", "220", "TV", "s1", ""],
        ["2020-05-01 00:00:00", "EEE.DE", "ETF", "BUY", "1", "50", "EUR", "0", "-50", "XTB", "e1", ""],
    )
    df, _ = L.parse_ledger(values)
    flows = L.class_cashflows(df, "Actiuni RO", "RON")
    assert flows == [(pd.Timestamp("2020-01-01"), -200.0), (pd.Timestamp("2021-01-01"), 220.0)]
    assert L.class_cashflows(df, "ETF", "RON") == []


def test_xirr_valori_de_referinta():
    # -1000 azi, +1100 peste exact 365 de zile -> 10%
    assert L.xirr([("2023-01-01", -1000), ("2024-01-01", 1100)]) == pytest.approx(0.10, abs=1e-7)
    # -1000, +1210 peste 730 de zile (2 ani de 365) -> 10% pe an
    assert L.xirr([("2021-01-01", -1000), ("2023-01-01", 1210)]) == pytest.approx(0.10, abs=1e-7)
    # două investiții: -1000 la t0, -1000 la 0,5 ani, +2200 la 1 an.
    # la r = 13,0662%: 1000 + 1000/1,130662^0,5 = 1940,44 ; 2200/1,130662 = 1945,76 (verificat mai jos numeric)
    r = L.xirr([("2023-01-01", -1000), ("2023-07-02 12:00:00", -1000), ("2024-01-01", 2200)])
    npv = -1000 - 1000 / (1 + r) ** 0.5 + 2200 / (1 + r)
    assert npv == pytest.approx(0.0, abs=1e-5) and 0.13 < r < 0.14
    # pierdere: -1000, +900 peste un an -> -10%
    assert L.xirr([("2023-01-01", -1000), ("2024-01-01", 900)]) == pytest.approx(-0.10, abs=1e-7)


def test_xirr_fara_solutie():
    assert L.xirr([]) is None
    assert L.xirr([("2023-01-01", -1000)]) is None
    assert L.xirr([("2023-01-01", -1000), ("2024-01-01", -50)]) is None      # doar ieșiri
    assert L.xirr([("2023-01-01", -1000), ("2023-01-01", 1100)]) is None     # aceeași zi
    assert L.xirr([("2023-01-01", -1000), ("2024-01-01", float("nan"))]) is None


# --- Evaluare la prețuri curente ---------------------------------------------

VALUED_SHEET = sheet(
    ["2024-01-01 10:00:00", "AAA.RO", "Actiuni RO", "BUY", "100", "10", "RON", "0", "-1000", "TV", "a1", ""],
    ["2024-01-01 10:00:00", "CCC.RO", "Actiuni RO", "BUY", "10", "20", "RON", "0", "-200", "TV", "c1", ""],
    ["2024-01-01 10:00:00", "R9901AE", "Obligatiuni", "BUY", "5", "98", "EUR", "0", "-495", "TV", "r1", ""],
    ["2024-07-01 10:00:00", "R9901AE", "Obligatiuni", "COUPON", "", "", "EUR", "0", "25", "TV", "r2", ""],
    ["2024-01-01 10:00:00", "BBB", "Actiuni SUA", "BUY", "2", "50", "USD", "0", "-100", "XTB", "x1", ""],
    ["2024-06-01 10:00:00", "BBB", "Actiuni SUA", "SELL", "2", "60", "USD", "0", "120", "XTB", "x2", ""],
    ["2024-01-01 09:00:00", "", "Numerar", "DEPOSIT", "", "", "RON", "0", "1500", "TV", "d1", ""],
    ["2024-01-01 09:00:00", "", "Numerar", "DEPOSIT", "", "", "EUR", "0", "500", "TV", "d2", ""],
)


def _valued(prices):
    df, _ = L.parse_ledger(VALUED_SHEET)
    pos, _ = L.build_positions(df)
    return df, pos, L.value_open_positions(pos, prices)


def test_value_open_positions_pret_live_nominal_si_lipsa():
    _, _, valued = _valued({"AAA.RO": 12.0, "CCC.RO": None, "BBB": 70.0})
    v = valued.set_index("Symbol")
    assert list(v.index) == ["AAA.RO", "CCC.RO", "R9901AE"]          # BBB e închisă, nu apare
    # AAA: 100 × 12 = 1200 ; cost 1000 ; nerealizat 200 (20%)
    assert v.loc["AAA.RO", "MarketValue"] == pytest.approx(1200.0)
    assert v.loc["AAA.RO", "Unrealized"] == pytest.approx(200.0)
    assert v.loc["AAA.RO", "UnrealizedPct"] == pytest.approx(20.0)
    assert v.loc["AAA.RO", "PriceSource"] == L.PRICE_LIVE
    # CCC: fără preț -> lipsă, nu 0
    assert np.isnan(v.loc["CCC.RO", "MarketValue"]) and v.loc["CCC.RO", "PriceSource"] == L.PRICE_MISSING
    # obligațiune: 5 × 100 nominal = 500 ; cost 495 ; nerealizat 5
    assert v.loc["R9901AE", "MarketValue"] == pytest.approx(500.0)
    assert v.loc["R9901AE", "Unrealized"] == pytest.approx(5.0)
    assert v.loc["R9901AE", "PriceSource"] == L.PRICE_NOMINAL


def test_value_open_positions_foloseste_ultima_inchidere_cand_lipseste_pretul_live():
    _, _, valued = _valued({"AAA.RO": None, "CCC.RO": None})
    assert set(valued.set_index("Symbol").loc[["AAA.RO", "CCC.RO"], "PriceSource"]) == {L.PRICE_MISSING}
    df, pos, _ = _valued({})
    v = L.value_open_positions(pos, {"AAA.RO": None, "CCC.RO": 21.0}, {"AAA.RO": 11.0, "CCC.RO": 99.0}).set_index("Symbol")
    assert v.loc["AAA.RO", "Price"] == 11.0 and v.loc["AAA.RO", "PriceSource"] == L.PRICE_CLOSE
    assert v.loc["CCC.RO", "Price"] == 21.0 and v.loc["CCC.RO", "PriceSource"] == L.PRICE_LIVE    # live are prioritate


def test_class_results_total_si_xirr():
    df, pos, valued = _valued({"AAA.RO": 12.0, "CCC.RO": 22.0})
    res = L.class_results(df, pos, valued, "2025-01-01 10:00:00").set_index(["Class", "Currency"])
    ro = res.loc[("Actiuni RO", "RON")]
    # valoare 1200 + 220 = 1420 ; cost 1200 ; nerealizat 220 ; total 220
    assert ro["MarketValue"] == pytest.approx(1420.0) and ro["Total"] == pytest.approx(220.0)
    # fluxuri: -1200 la 2024-01-01, +1420 la 2025-01-01 (366 de zile, an bisect):
    # (1420/1200)^(365/366) - 1 = 18,2796%
    assert ro["XIRR"] == pytest.approx(((1420 / 1200) ** (365 / 366) - 1) * 100, abs=1e-4)
    assert ro["Flows"] == 2 and ro["MissingPrices"] == "" and not ro["Proxy"]
    us = res.loc[("Actiuni SUA", "USD")]
    # poziție închisă: realizat 20, fără valoare de piață ; -100 -> +120 în 152 de zile
    assert us["OpenPositions"] == 0 and us["MarketValue"] == 0.0 and us["Total"] == pytest.approx(20.0)
    assert us["XIRR"] == pytest.approx((1.2 ** (365 / 152) - 1) * 100, abs=1e-4)
    bond = res.loc[("Obligatiuni", "EUR")]
    # nerealizat 5 + cupon 25 = 30 ; marcat ca proxy
    assert bond["Total"] == pytest.approx(30.0) and bond["Proxy"]


def test_class_results_fara_total_cand_lipseste_un_pret():
    df, pos, valued = _valued({"AAA.RO": 12.0, "CCC.RO": None})
    ro = L.class_results(df, pos, valued, "2025-01-01").set_index(["Class", "Currency"]).loc[("Actiuni RO", "RON")]
    assert ro["MissingPrices"] == "CCC.RO"
    assert np.isnan(ro["MarketValue"]) and np.isnan(ro["Total"]) and np.isnan(ro["XIRR"])
    assert ro["Realized"] == 0.0                                      # ce nu depinde de preț rămâne calculat


def test_consolidate_in_ron():
    df, pos, valued = _valued({"AAA.RO": 12.0, "CCC.RO": 22.0})
    res = L.class_results(df, pos, valued, "2025-01-01")
    cash = L.cash_balances(df)
    table, total, missing = L.consolidate(res, cash, {"EUR": 5.0, "USD": 4.5})
    t = table.set_index("Currency")
    # RON: poziții 1420 + numerar 1500 - 1000 - 200 = 300 -> 1720
    assert t.loc["RON", "ValueBase"] == pytest.approx(1720.0)
    # EUR: poziții 500 + numerar 500 - 495 + 25 = 30 -> 530 × 5 = 2650
    assert t.loc["EUR", "ValueBase"] == pytest.approx(2650.0)
    # USD: fără poziții, numerar -100 + 120 = 20 -> 20 × 4,5 = 90
    assert t.loc["USD", "ValueBase"] == pytest.approx(90.0)
    assert total == pytest.approx(1720.0 + 2650.0 + 90.0) and missing == []

    # fără curs USD: totalul nu se afișează parțial
    _, total2, missing2 = L.consolidate(res, cash, {"EUR": 5.0, "USD": None})
    assert total2 is None and missing2 == ["USD"]


# --- Portofoliul întreg: depuneri, XIRR, curbă zilnică, TWR --------------------

def test_xirr_pe_clasa_nu_se_afiseaza_sub_90_de_zile():
    values = sheet(
        ["2024-01-01 10:00:00", "QQQ.L", "ETF", "BUY", "1", "50", "USD", "0", "-50", "XTB", "q1", ""],
        ["2024-01-07 10:00:00", "QQQ.L", "ETF", "SELL", "1", "49", "USD", "0", "49", "XTB", "q2", ""],
    )
    df, _ = L.parse_ledger(values)
    pos, _ = L.build_positions(df)
    res = L.class_results(df, pos, L.value_open_positions(pos, {}), "2025-01-01").iloc[0]
    assert res["Total"] == pytest.approx(-1.0) and np.isnan(res["XIRR"])      # 6 zile: fără randament anualizat
    assert L.flows_span_days([("2024-01-01", -1), ("2024-04-10", 2)]) == pytest.approx(100.0)
    assert L.flows_span_days([("2024-01-01", -1)]) == 0.0


FX = pd.DataFrame({"EUR": [5.0, 5.0, 5.1, 5.2], "USD": [4.5, 4.5, 4.6, 4.4]},
                  index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-06-03", "2025-01-01"]))


def test_fx_on_ia_ultima_cotatie_de_pana_la_data():
    assert L.fx_on(FX, "RON", "2024-03-01") == 1.0
    assert L.fx_on(FX, "EUR", "2024-01-02 15:30:00") == 5.0
    assert L.fx_on(FX, "EUR", "2024-03-01") == 5.0                       # sâmbătă/gol: ultima cotație dinainte
    assert L.fx_on(FX, "EUR", "2024-06-03 09:00:00") == 5.1
    assert L.fx_on(FX, "EUR", "2023-12-31") is None                      # nu se ia un curs din viitor
    assert L.fx_on(FX, "GBP", "2024-03-01") is None


FLOW_SHEET = sheet(
    ["2024-01-01 00:00:00", "", "Numerar", "OPEN", "", "", "RON", "0", "100", "TV", "o0", ""],
    ["2024-01-01 00:00:00", "AAA.RO", "Actiuni RO", "OPEN", "10", "20", "RON", "0", "0", "TV", "o1", ""],
    ["2024-01-02 10:00:00", "", "Numerar", "DEPOSIT", "", "", "EUR", "0", "100", "XTB", "d1", ""],
    ["2024-01-02 11:00:00", "", "Numerar", "FX", "", "", "EUR", "0", "-50", "XTB", "f1", ""],
    ["2024-01-02 11:00:00", "", "Numerar", "FX", "", "", "USD", "0", "55", "XTB", "f2", ""],
    ["2024-06-03 10:00:00", "", "Numerar", "WITHDRAW", "", "", "EUR", "0", "-10", "XTB", "w1", ""],
)


def test_external_flows_la_cursul_din_ziua_lor():
    df, _ = L.parse_ledger(FLOW_SHEET)
    ext, problems = L.external_flows(df, FX)
    assert problems == []
    # OPEN numerar 100 RON ; OPEN poziție 10 × 20 = 200 RON ; depunere 100 EUR × 5,0 = 500 ; retragere -10 EUR × 5,1 = -51
    assert list(ext["AmountBase"]) == pytest.approx([100.0, 200.0, 500.0, -51.0])
    assert len(ext) == 4                                                  # schimbul valutar nu e flux extern

    res = L.money_weighted_result(ext, 900.0, "2025-01-01")
    assert res["net_invested"] == pytest.approx(749.0)
    assert res["profit"] == pytest.approx(151.0)
    # verificare: NPV la rata găsită e zero
    r = res["xirr"]
    days = [0, 0, 1 + 10 / 24, 154 + 10 / 24, 366]                        # depunerea și retragerea sunt la ora 10:00
    npv = sum(a / (1 + r) ** (d / 365) for a, d in zip([-100, -200, -500, 51, 900], days))
    assert npv == pytest.approx(0.0, abs=1e-4) and 0.15 < r < 0.25


def test_external_flows_fara_curs_nu_da_rezultat():
    df, _ = L.parse_ledger(FLOW_SHEET)
    ext, problems = L.external_flows(df, FX[["USD"]])
    assert len(problems) == 2 and "EUR/RON" in problems[0]
    assert L.money_weighted_result(ext, 900.0, "2025-01-01") == {"net_invested": None, "profit": None, "xirr": None}
    assert L.money_weighted_result(L.external_flows(df, FX)[0], None, "2025-01-01")["profit"] is None


def test_unadjust_splits_consolidare():
    idx = pd.to_datetime(["2022-08-16", "2022-08-17", "2022-08-18", "2022-08-19"])
    close = pd.Series([23.0, 23.5, 24.0, 24.5], index=idx)               # ajustat: ca și cum consolidarea ar fi existat mereu
    splits = pd.Series([0.0, 0.0, 0.1, 0.0], index=idx)                  # consolidare 1:10 pe 18 august
    raw = L.unadjust_splits(close, splits)
    # înainte de consolidare o acțiune costa de 10 ori mai puțin
    assert list(raw) == pytest.approx([2.3, 2.35, 24.0, 24.5])
    assert L.unadjust_splits(close, pd.Series(dtype="float64")).equals(close)


CURVE_SHEET = sheet(
    ["2024-01-01 09:00:00", "", "Numerar", "DEPOSIT", "", "", "RON", "0", "1000", "TV", "d1", ""],
    ["2024-01-02 10:00:00", "AAA.RO", "Actiuni RO", "BUY", "100", "5", "RON", "0", "-500", "TV", "a1", ""],
    ["2024-01-03 09:00:00", "", "Numerar", "DEPOSIT", "", "", "EUR", "0", "100", "XTB", "d2", ""],
    ["2024-01-03 10:00:00", "EEE.DE", "ETF", "BUY", "2", "40", "EUR", "0", "-80", "XTB", "e1", ""],
    ["2024-01-04 10:00:00", "R9901AE", "Obligatiuni", "BUY", "1", "99", "EUR", "0", "-20", "XTB", "r1", ""],
)
CURVE_IDX = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
CURVE_FX = pd.DataFrame({"EUR": [5.0, 5.0, 5.0, 5.0, 5.0]}, index=CURVE_IDX)


def test_curba_zilnica_si_twr():
    df, _ = L.parse_ledger(CURVE_SHEET)
    qty = L.daily_quantities(df, CURVE_IDX)
    assert list(qty["AAA.RO"]) == [0, 100, 100, 100, 100] and list(qty["EEE.DE"]) == [0, 0, 2, 2, 2]
    cash = L.daily_cash(df, CURVE_IDX)
    assert list(cash["RON"]) == [1000, 500, 500, 500, 500] and list(cash["EUR"]) == [0, 0, 20, 0, 0]

    market = pd.DataFrame({"AAA.RO": [np.nan, 5.0, 5.5, np.nan, 6.0]}, index=CURVE_IDX)   # 4 ian. fără ședință
    trade = L.ledger_trade_prices(df, CURVE_IDX)
    classes = {"AAA.RO": "Actiuni RO", "EEE.DE": "ETF", "R9901AE": "Obligatiuni"}
    prices, sources = L.build_price_table(CURVE_IDX, classes, market, trade)
    assert sources == {"AAA.RO": "market", "EEE.DE": "proxy", "R9901AE": "nominal"}
    assert list(prices["AAA.RO"].iloc[1:]) == [5.0, 5.5, 5.5, 6.0]          # dus înainte, nu înapoi
    assert np.isnan(prices["AAA.RO"].iloc[0])
    assert list(prices["EEE.DE"].iloc[2:]) == [40.0, 40.0, 40.0]            # fără piață: prețul din tranzacție

    curve, blockers = L.portfolio_curve_base(qty, prices, cash, {"AAA.RO": "RON", "EEE.DE": "EUR", "R9901AE": "EUR"}, CURVE_FX)
    assert blockers == []
    # 1 ian: 1000 numerar
    # 2 ian: 100×5 + 500 = 1000
    # 3 ian: 100×5,5 + 500 + (2×40 + 20)×5 = 550 + 500 + 500 = 1550
    # 4 ian: 550 + 500 + (80 + 1×100 + 0)×5 = 1950   (obligațiunea la nominal 100, plătită 20: +80 EUR × 5 = +400)
    # 5 ian: 100×6 + 500 + 900 = 2000
    assert list(curve) == pytest.approx([1000.0, 1000.0, 1550.0, 1950.0, 2000.0])

    ext, _ = L.external_flows(df, CURVE_FX)
    flows = ext.groupby(ext["Date"].dt.normalize())["AmountBase"].sum()
    cumulative, total = L.twr(curve, flows)
    # r: 0 ; 1000/1000-1 = 0 ; 1550/(1000+500)-1 = 3,333% ; 1950/1550-1 = 25,806% ; 2000/1950-1 = 2,564%
    assert total == pytest.approx((1550 / 1500) * (1950 / 1550) * (2000 / 1950) - 1)
    assert total == pytest.approx(2000 / 1500 - 1)
    assert cumulative.iloc[2] == pytest.approx(1550 / 1500 - 1)


def test_curba_fara_pret_ramane_lipsa_si_twr_porneste_dupa_gol():
    df, _ = L.parse_ledger(CURVE_SHEET)
    qty, cash = L.daily_quantities(df, CURVE_IDX), L.daily_cash(df, CURVE_IDX)
    prices = pd.DataFrame({"AAA.RO": [np.nan, np.nan, 5.5, 5.5, 6.0], "EEE.DE": 40.0, "R9901AE": 100.0}, index=CURVE_IDX)
    curve, blockers = L.portfolio_curve_base(qty, prices, cash, {"AAA.RO": "RON", "EEE.DE": "EUR", "R9901AE": "EUR"}, CURVE_FX)
    assert blockers == ["AAA.RO"]
    assert np.isnan(curve.iloc[1]) and curve.iloc[0] == 1000.0 and curve.iloc[2] == pytest.approx(1550.0)
    cumulative, total = L.twr(curve, pd.Series(dtype="float64"))
    assert cumulative.index[0] == CURVE_IDX[2]                              # lanțul începe după ultima zi fără valoare
    assert total == pytest.approx(2000 / 1550 - 1)
    assert L.twr(pd.Series([np.nan, 5.0]), pd.Series(dtype="float64"))[1] is None    # o singură zi cu valoare


def test_flows_on_index_muta_weekendul_pe_ziua_urmatoare():
    idx = pd.to_datetime(["2024-01-05", "2024-01-08", "2024-01-09"])        # vineri, luni, marți
    ext = pd.DataFrame({"Date": pd.to_datetime(["2024-01-05 10:00", "2024-01-06 12:00", "2024-01-08 09:00", "2024-02-01 00:00"]),
                        "AmountBase": [100.0, 50.0, 25.0, 999.0]})
    out = L.flows_on_index(ext, idx)
    assert list(out) == [100.0, 75.0, 0.0]                                 # sâmbăta trece pe luni; februarie e în afara indexului


# --- Vârfuri de date (split datat diferit, cotație greșită) ---------------------

def test_despike_scoate_varful_care_revine_si_lasa_schimbarea_reala():
    idx = pd.bdate_range("2022-08-15", periods=7)
    # consolidare 1:10 datată cu o zi mai devreme în sursa de prețuri: o zi cu valoare ×10
    spike = pd.Series([2400.0, 2410.0, 24200.0, 2430.0, 2440.0, 2450.0, 2460.0], index=idx)
    clean, fixed = L.despike(spike)
    assert list(clean) == [2400.0, 2410.0, 2410.0, 2430.0, 2440.0, 2450.0, 2460.0]
    assert fixed == [idx[2]]
    # vârf de două zile, în jos
    dip = pd.Series([1000.0, 100.0, 105.0, 1010.0, 1000.0, 990.0, 1000.0], index=idx)
    clean, fixed = L.despike(dip)
    assert list(clean)[:4] == [1000.0, 1000.0, 1000.0, 1010.0] and len(fixed) == 2
    # o cumpărare care dublează poziția nu revine: rămâne
    step = pd.Series([1000.0, 1000.0, 2100.0, 2120.0, 2090.0, 2110.0, 2100.0], index=idx)
    clean, fixed = L.despike(step)
    assert clean.equals(step) and fixed == []
    # mișcare normală (sub prag) și zerouri (poziție închisă): neatinse
    calm = pd.Series([100.0, 120.0, 95.0, 0.0, 0.0, 100.0, 101.0], index=idx)
    assert L.despike(calm)[0].equals(calm)


def test_despike_nu_atinge_o_vanzare_urmata_de_recumparare():
    idx = pd.bdate_range("2024-03-04", periods=5)
    series = pd.Series([1000.0, 400.0, 410.0, 1000.0, 1010.0], index=idx)      # vândut 60%, recumpărat după două zile
    assert L.despike(series, blocked_days={idx[1]})[0].equals(series)
    assert L.despike(series, blocked_days={idx[3]})[0].equals(series)          # și ziua revenirii contează
    assert not L.despike(series)[0].equals(series)                             # fără informația asta ar fi netezit


def test_curba_nu_sare_la_o_consolidare_datata_diferit():
    values = sheet(
        ["2022-08-01 09:00:00", "", "Numerar", "DEPOSIT", "", "", "RON", "0", "3000", "TV", "d1", ""],
        ["2022-08-01 10:00:00", "TTT.RO", "Actiuni RO", "BUY", "1000", "2,4", "RON", "0", "-2400", "TV", "b1", ""],
        ["2022-08-18 09:00:00", "TTT.RO", "Actiuni RO", "SPLIT", "-1000", "", "RON", "0", "0", "TV", "s1", "1/10"],
        ["2022-08-18 09:01:00", "TTT.RO", "Actiuni RO", "SPLIT", "100", "", "RON", "0", "0", "TV", "s2", "1/10"],
    )
    df, _ = L.parse_ledger(values)
    idx = pd.bdate_range("2022-08-15", "2022-08-22")                          # 15,16,17,18,19,22
    qty, cash = L.daily_quantities(df, idx), L.daily_cash(df, idx)
    # sursa de prețuri datează consolidarea pe 17, brokerul pe 18: pe 17 sunt încă 1000 de acțiuni la prețul nou
    prices = pd.DataFrame({"TTT.RO": [2.4, 2.4, 24.0, 24.0, 24.5, 24.5]}, index=idx)
    fixes = []
    curve, _ = L.portfolio_curve_base(qty, prices, cash, {"TTT.RO": "RON"}, None,
                                      trade_days=L.cash_trade_days(df), corrections=fixes)
    # fără corecție, 17 august ar fi 1000 × 24 + 600 = 24.600
    assert list(curve) == pytest.approx([3000.0, 3000.0, 3000.0, 3000.0, 3050.0, 3050.0])
    assert fixes == [("TTT.RO", idx[2])]
