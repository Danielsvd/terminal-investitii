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
        ["2020-01-01 00:00:00", "", "Numerar", "OPEN", "", "", "RON", "0", "836,86", "TV", "o0", ""],
        ["2020-01-03 10:00:00", "AAA.RO", "Actiuni RO", "BUY", "10", "5", "RON", "1", "-51", "TV", "a", ""],
    )
    df, problems = L.parse_ledger(values)
    assert problems == [] and len(df) == 2                           # rândul de numerar nu e respins
    pos, _ = L.build_positions(df)
    assert list(pos["Symbol"]) == ["AAA.RO"]                         # și nu devine o deținere
    assert L.cash_balances(df)["Cash"].iloc[0] == pytest.approx(836.86 - 51)


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
