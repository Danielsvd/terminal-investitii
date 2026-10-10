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
