"""Teste pentru importul exporturilor de la brokeri. Toate datele sunt inventate (repo-ul e public)."""
import datetime as dt

import pytest

from analytics import ledger as L
from data import broker_import as B

TV_HEADER = ("data", "data", "simbol", "cant", "pret", "comis", "suma", "soldlei", "soldact", "profit", "nrtranz",
             "valuta", "obs", "costm", "txcnlei", "idord", "cont", "market", "ytm", "impozlei", "simbolb", "dirty",
             "piata", "cant1an", "cost", "tcont", "comisproc", "nrzecFractionar")


def tv(ts, tip, simbol, cant, pret, comis, suma, valuta="RON", obs=None, nrtranz=None, market=None, simbolb=None, tcont="RO"):
    row = dict.fromkeys(TV_HEADER)
    row.update(simbol=simbol, cant=cant, pret=pret, comis=comis, suma=suma, valuta=valuta, obs=obs, nrtranz=nrtranz,
               idord="A1", cont="ZZZ999" + ("-RE" if tcont == "RE" else ""), market=market, simbolb=simbolb, tcont=tcont)
    values = [row[c] for c in TV_HEADER]
    values[0], values[1] = ts, tip
    return tuple(values)


# fișierul Tradeville e în ordine descrescătoare
TV_ROWS = [
    TV_HEADER,
    tv(dt.datetime(2024, 9, 10, 11, 57, 32, 340000), "in", "RON", 524.88, None, 0, 524.88, obs="TF 100 EUR DIN RE CURS 5.2488"),
    tv(dt.datetime(2024, 9, 10, 11, 57, 31, 597000), "out", "EUR", 100, None, 0, -100, valuta="EUR", obs="TF 524.88 RON IN RO CURS 5.2488", tcont="RE"),
    tv(dt.datetime(2024, 8, 18, 9, 1, 20, 350000), "in", "AAA", 8, 23.0, None, 0, obs="Consolidare (raport 1/10)", simbolb="AAA"),
    tv(dt.datetime(2024, 8, 18, 9, 0, 40, 370000), "out", "AAA", 80, None, None, 0, obs="Consolidare (raport 1/10)", simbolb="AAA"),
    tv(dt.datetime(2024, 7, 20, 9, 17, 25, 680000), "in", "AAA", 20, 0, None, 0, obs="mcs gratuite", simbolb="AAA"),
    tv(dt.datetime(2024, 7, 5, 10, 1, 15, 517000), "out", "AAAR01", 50, None, None, 0, obs="Retragere drepturi subscriere AAAR01", simbolb="AAAR01"),
    tv(dt.datetime(2024, 7, 3, 15, 38, 2, 973000), "div", "EUR", 26.5, None, None, 26.5, valuta="EUR", obs="plata cupon R9907AE", nrtranz="R9907AE", simbolb="R9907AE", tcont="RE"),
    tv(dt.datetime(2024, 6, 30, 11, 33, 2, 597000), "div", "RON", 34.62, None, 0.3462, 34.2738, obs="dividend AAA", nrtranz="AAA", simbolb="AAA"),
    tv(dt.datetime(2024, 6, 20, 10, 14, 34, 203000), "in", "EUR", 20.0, None, 0, 20.0, valuta="EUR", obs="TRANSFER DIN ZZZ999.UE IN ZZZ999-RE", tcont="RE"),
    tv(dt.datetime(2024, 5, 7, 9, 0, 8, 690000), "in", "RON", 0.09, 0, 0, 0.09, obs="Returnare taxa ASF APR24 (Discount)"),
    tv(dt.datetime(2024, 3, 3, 21, 15, 40, 293000), "comis", "RON", 0, None, -7.37, 7.37, obs="Compensatie 25.01.2024", nrtranz="alta"),
    tv(dt.datetime(2024, 2, 1, 9, 20, 58, 890000), "cump", "R9907AE", 5, 100, 0, -500, valuta="EUR", nrtranz="222", market="EPOFB", simbolb="R9907AE", tcont="RE"),
    tv(dt.datetime(2024, 1, 15, 10, 0, 0, 1000), "vanz", "AAA", 40, 2.5, 1.9, 98.1, nrtranz="112", market="REGS", simbolb="AAA"),
    tv(dt.datetime(2024, 1, 10, 14, 25, 43, 207000), "cump", "AAA", 100, 2.4, 1.9, -241.9, nrtranz="111", market="REGS", simbolb="AAA", obs="Promo"),
    tv(dt.datetime(2024, 1, 9, 16, 11, 50, 280000), "in", "RON", 1000, None, None, 1000, obs="DEPUNERE CONT CLIENT"),
    tv(dt.datetime(2024, 1, 8, 8, 0, 0, 0), "xyz", "RON", 1, None, None, 1, obs="ceva nou"),
]


def test_parse_tradeville_tipuri_simboluri_si_clase():
    records, problems = B.parse_tradeville(TV_ROWS)
    assert problems == ["Tradeville 2024-01-08: operațiune necunoscută (xyz RON „ceva nou”), rând ignorat."]
    by_type = [(r["Type"], r["Symbol"], r["Class"]) for r in records]
    assert by_type == [
        ("DEPOSIT", "", "Numerar"), ("BUY", "AAA.RO", "Actiuni RO"), ("SELL", "AAA.RO", "Actiuni RO"),
        ("BUY", "R9907AE", "Obligatiuni"), ("FEE", "", "Numerar"), ("FEE", "", "Numerar"), ("TRANSFER", "", "Numerar"),
        ("DIV", "AAA.RO", "Actiuni RO"), ("COUPON", "R9907AE", "Obligatiuni"), ("RIGHTS", "AAAR01", "Drepturi"),
        ("BONUS", "AAA.RO", "Actiuni RO"), ("SPLIT", "AAA.RO", "Actiuni RO"), ("SPLIT", "AAA.RO", "Actiuni RO"),
        ("FX", "", "Numerar"), ("FX", "", "Numerar"),
    ]
    buy = records[1]
    assert (buy["Date"], buy["Quantity"], buy["Price"], buy["Fees"], buy["Amount"]) == ("2024-01-10 14:25:43", 100, 2.4, 1.9, -241.9)
    assert buy["ID"] == "TV-20240110142543207" and buy["Currency"] == "RON" and buy["Broker"] == "Tradeville"
    div = records[7]
    assert div["Amount"] == 34.2738 and div["Fees"] == 0.3462 and div["Quantity"] == ""          # dividend net, comision separat
    split_out, split_in = records[11], records[12]
    assert split_out["Quantity"] == -80 and split_in["Quantity"] == 8                            # consolidare: ies 80, intră 8
    assert records[9]["Quantity"] == -50                                                          # drepturi retrase
    assert all("ZZZ999" not in str(v) for r in records for v in r.values())                       # codul de client nu ajunge în foaie
    assert records[6]["Notes"] == "TRANSFER DIN cont UE IN cont RE"


def test_parse_tradeville_antet_gresit():
    records, problems = B.parse_tradeville([("data", "data", "simbol")])
    assert records == [] and "Coloane lipsă" in problems[0]


XTB_CASH = [
    ("Account number", "11111111"), ("Cash Operations", ""), ("Date from (UTC)", dt.datetime(2006, 1, 1)),
    ("Date to (UTC)", dt.datetime(2026, 1, 1)),
    ("Type", "Instrument", "Ticker", "Category", "Time", "Amount", "ID", "Comment", "Product", "Position ID"),
    ("Free funds interest", "", "", "", dt.datetime(2024, 7, 3, 11, 36, 22), 1.21, "9007", "Free-funds Interest 2024-06", "My Trades", ""),
    ("RO tax", "", "", "", dt.datetime(2024, 6, 20, 5, 0, 0), -0.23, "9006", "RO tax BBB.US 2024-06-19 (1.00 RON) ///OMI/123456789/1.00//", "My Trades", ""),
    ("Stock sell", "Bbb Corp", "BBB.US", "STOCK", dt.datetime(2024, 6, 19, 15, 23, 28), 50.26, "9005", "CLOSE BUY 1/2.0132 @ 50.26", "My Trades", "77"),
    ("Withholding tax", "Bbb Corp", "BBB.US", "STOCK", dt.datetime(2024, 5, 1, 10, 30, 46), -0.1, "9004", "BBB.US USD WHT 10%", "My Trades", ""),
    ("Dividend", "Bbb Corp", "BBB.US", "STOCK", dt.datetime(2024, 5, 1, 10, 30, 46), 0.96, "9003", "BBB.US USD 0.32/ SHR", "My Trades", ""),
    ("Stock purchase", "Gold ETC", "GGG.UK", "ETC", dt.datetime(2024, 1, 20, 14, 10, 43), -48.7, "9002", "OPEN BUY GGG.UK 1 @ 48.7000", "My Trades", "78"),
    ("Stock purchase", "Bbb Corp", "BBB.US", "STOCK", dt.datetime(2024, 1, 15, 22, 30, 0), -90.5, "9001", "OPEN BUY 2.0132 @ 44.95", "My Trades", "77"),
    ("Transfer", "", "", "", dt.datetime(2024, 1, 10, 13, 51, 3), 114.8, "9000", "Currency conversion, EUR to USD from TA: 22222222 to: 11111111, Exchange rate:1.148049", "My Trades", ""),
    ("Bonus nou", "", "", "", dt.datetime(2024, 1, 9, 13, 0, 0), 5.0, "8999", "ceva nou", "My Trades", ""),
    ("Total", None, None, None, None, 32.7, None, None, None, None),
]
XTB_SHEETS = {
    "Cash Operations": XTB_CASH,
    "Closed Positions": [("Instrument", "Ticker", "Category"), ("Bbb Corp", "BBB.US", "STOCK")],
    "Open Positions": [("Product", "Metric", "Amount", "Currency"), ("My Trades", "Open position value", 100.0, "USD")],
}


def test_parse_xtb_tipuri_ore_si_control():
    records, problems, control = B.parse_xtb(XTB_SHEETS)
    assert len(problems) == 1 and "Bonus nou" in problems[0]
    assert [(r["Type"], r["Symbol"], r["Class"]) for r in records] == [
        ("FX", "", "Numerar"), ("BUY", "BBB", "Actiuni SUA"), ("BUY", "GGG.L", "ETF"), ("DIV", "BBB", "Actiuni SUA"),
        ("TAX", "BBB", "Actiuni SUA"), ("SELL", "BBB", "Actiuni SUA"), ("TAX", "BBB", "Actiuni SUA"), ("INTEREST", "", "Numerar"),
    ]
    buy = records[1]
    # 22:30 UTC în ianuarie = 00:30 a doua zi în România (UTC+2)
    assert buy["Date"] == "2024-01-16 00:30:00"
    assert (buy["Quantity"], buy["Price"], buy["Amount"], buy["Currency"], buy["ID"]) == (2.0132, 44.95, -90.5, "USD", "XTB-9001")
    sell = records[5]
    # 15:23 UTC în iunie = 18:23 în România (UTC+3); „1/2.0132” înseamnă 1 din 2,0132 deținute
    assert sell["Date"] == "2024-06-19 18:23:28" and sell["Quantity"] == 1 and sell["Price"] == 50.26
    assert records[0]["Notes"] == "Schimb valutar EUR>USD curs 1.148049"
    assert records[6]["Notes"] == "RO tax BBB.US 2024-06-19 (1.00 RON)"
    assert all("11111111" not in str(v) and "22222222" not in str(v) for r in records for v in r.values())
    # suma rândurilor citite: 114,8 - 90,5 - 48,7 + 0,96 - 0,1 + 50,26 - 0,23 + 1,21 = 27,70
    # raportul spune 32,70: rândul necunoscut (+5) nu a intrat, iar diferența semnalează importul incomplet
    assert control == {"reported": 32.7, "parsed": pytest.approx(27.7)}


def test_parse_xtb_fara_moneda_si_detectie():
    sheets = dict(XTB_SHEETS, **{"Open Positions": []})
    records, problems, _ = B.parse_xtb(sheets)
    assert records == [] and "moneda contului" in problems[0]
    assert len(B.parse_xtb(sheets, "usd")[0]) == 8                              # moneda aleasă manual
    assert B.detect_broker(XTB_SHEETS) == "xtb"
    assert B.detect_broker({"sheet1": TV_ROWS}) == "tradeville"
    assert B.detect_broker({"x": [("a", "b")]}) is None
    assert B.parse_export({"x": [("a", "b")]})[0] is None
    assert B.parse_export(XTB_SHEETS)[0] == "XTB USD" and B.parse_export({"sheet1": TV_ROWS})[0] == "Tradeville"


def test_new_records_sare_peste_ce_exista_si_iese_in_formatul_foii():
    records, _ = B.parse_tradeville(TV_ROWS)
    existing = {records[0]["ID"], records[1]["ID"]}
    fresh = B.new_records(records + records, existing)                          # același fișier încărcat de două ori
    assert len(fresh) == len(records) - 2
    assert [r["Date"] for r in fresh] == sorted(r["Date"] for r in fresh)
    assert B.new_records(records, {r["ID"] for r in records}) == []
    rows = B.to_sheet_rows(fresh)
    assert all(len(r) == len(L.COLUMNS) for r in rows)
    # ce produce importul trebuie să fie citit fără probleme de registru
    df, problems = L.parse_ledger([L.COLUMNS] + B.to_sheet_rows(records))
    assert problems == [] and len(df) == len(records)
    pos, pos_problems = L.build_positions(df)
    assert pos_problems == []
    aaa = pos.set_index("Symbol").loc["AAA.RO"]
    # 100 cumpărate - 40 vândute + 20 gratuite = 80 ; consolidare 1/10 -> 8
    assert aaa["Quantity"] == pytest.approx(8.0)
    # cost: 241,9 pe 100 ; vândute 40 -> rămâne 60% din cost = 145,14 ; gratuitele și consolidarea nu schimbă costul
    assert aaa["CostBasis"] == pytest.approx(145.14)


# --- IBKR (Flex Query CSV cu coduri de secțiune). Cont, ID-uri și sume inventate. -------------

IB_TRNT = ["ClientAccountID", "CurrencyPrimary", "AssetClass", "SubCategory", "Symbol", "Description", "ISIN",
           "ListingExchange", "Multiplier", "DateTime", "TradeDate", "Exchange", "Quantity", "TradePrice", "Proceeds",
           "Taxes", "IBCommission", "IBCommissionCurrency", "NetCash", "Open/CloseIndicator", "Buy/Sell", "TradeID",
           "TransactionID", "LevelOfDetail"]
IB_CTRN = ["ClientAccountID", "CurrencyPrimary", "AssetClass", "SubCategory", "Symbol", "Description", "ISIN",
           "ListingExchange", "Date/Time", "SettleDate", "Amount", "Type", "TransactionID", "LevelOfDetail"]
IB_CRTT = ["ClientAccountID", "CurrencyPrimary", "LevelOfDetail", "StartingCash", "EndingCash"]
IB_POST = ["ClientAccountID", "CurrencyPrimary", "AssetClass", "SubCategory", "Symbol", "ISIN", "ListingExchange",
           "Multiplier", "Quantity", "LevelOfDetail"]


def ib(code, header, **values):
    return ["DATA", code] + [str(values.get(col, "")) for col in header]


def ib_trade(**v):
    base = dict(ClientAccountID="U00000000", Multiplier="1", Taxes="0", LevelOfDetail="EXECUTION")
    return ib("TRNT", IB_TRNT, **{**base, **v})


def ib_cash(**v):
    return ib("CTRN", IB_CTRN, **{**dict(ClientAccountID="U00000000", LevelOfDetail="DETAIL"), **v})


IB_ROWS = [
    ["HEADER", "TRNT"] + IB_TRNT,
    ib_trade(CurrencyPrimary="EUR", AssetClass="STK", SubCategory="ETF", Symbol="EGLN", Description="ISHARES PHYSICAL GOLD ETC",
             ISIN="IE00B4ND3602", ListingExchange="FWB2", DateTime="2026-11-12;03:38:44", TradeDate="2026-11-12",
             Exchange="IBIS2", Quantity="3", TradePrice="50", Proceeds="-150", IBCommission="-1.25",
             IBCommissionCurrency="EUR", NetCash="-151.25", **{"Buy/Sell": "BUY", "Open/CloseIndicator": "O"},
             TradeID="11", TransactionID="1001"),
    ib_trade(CurrencyPrimary="USD", AssetClass="CASH", Symbol="EUR.USD", DateTime="2026-11-12;04:25:05", Quantity="-500",
             TradePrice="1.1", Proceeds="550", IBCommission="-2", IBCommissionCurrency="EUR", NetCash="0",
             **{"Buy/Sell": "SELL"}, TradeID="12", TransactionID="1002"),
    ib_trade(CurrencyPrimary="USD", AssetClass="STK", SubCategory="COMMON", Symbol="BRK B", ISIN="US0000000001",
             ListingExchange="NYSE", DateTime="2026-07-01;10:00:00", Quantity="1", TradePrice="400", Proceeds="-400",
             IBCommission="-1", IBCommissionCurrency="USD", **{"Buy/Sell": "BUY", "Open/CloseIndicator": "O"},
             TradeID="13", TransactionID="1003"),
    ib_trade(CurrencyPrimary="USD", AssetClass="OPT", Symbol="XYZ   261218C00050000", ListingExchange="CBOE",
             Multiplier="100", DateTime="2026-11-13;10:00:00", Quantity="2", TradePrice="1.5", Proceeds="-300",
             IBCommission="-1.3", IBCommissionCurrency="USD", **{"Buy/Sell": "BUY", "Open/CloseIndicator": "O"},
             TradeID="14", TransactionID="1004"),
    ib_trade(CurrencyPrimary="USD", AssetClass="OPT", Symbol="XYZ   261218P00040000", ListingExchange="CBOE",
             Multiplier="100", DateTime="2026-11-13;10:05:00", Quantity="-1", TradePrice="0.8", Proceeds="80",
             IBCommission="-1", IBCommissionCurrency="USD", **{"Buy/Sell": "SELL", "Open/CloseIndicator": "O"},
             TradeID="15", TransactionID="1005"),
    ib_trade(CurrencyPrimary="EUR", AssetClass="STK", SubCategory="COMMON", Symbol="ABC", ListingExchange="IBIS",
             DateTime="2026-11-12;05:00:00", Quantity="1", TradePrice="10", Proceeds="-10", IBCommission="-1",
             IBCommissionCurrency="EUR", **{"Buy/Sell": "BUY"}, TradeID="16", TransactionID="1006",
             LevelOfDetail="ORDER"),                                       # nivel ORDER: dublură a execuției, ignorat
    ["HEADER", "CTRN"] + IB_CTRN,
    ib_cash(CurrencyPrimary="EUR", Description="CASH RECEIPTS", **{"Date/Time": "2026-11-06"}, Amount="1000",
            Type="Deposits/Withdrawals", TransactionID="2001"),
    ib_cash(ClientAccountID="-", CurrencyPrimary="EUR", **{"Date/Time": "2026-11-06"}, Amount="1000",
            Type="Deposits/Withdrawals", LevelOfDetail="SUMMARY"),
    ib_cash(CurrencyPrimary="USD", AssetClass="STK", SubCategory="COMMON", Symbol="BRK B", ListingExchange="NYSE",
            Description="BRK B CASH DIVIDEND", **{"Date/Time": "2026-11-10;20:20:00"}, Amount="2",
            Type="Dividends", TransactionID="2002"),
    ib_cash(CurrencyPrimary="USD", AssetClass="STK", SubCategory="COMMON", Symbol="BRK B", ListingExchange="NYSE",
            Description="BRK B US TAX", **{"Date/Time": "2026-11-10;20:20:00"}, Amount="-0.3",
            Type="Withholding Tax", TransactionID="2003"),
    ib_cash(CurrencyPrimary="EUR", **{"Date/Time": "2026-11-11"}, Amount="-0.5", Type="Ceva Nou", TransactionID="2004"),
    ["HEADER", "CRTT"] + IB_CRTT,
    ib("CRTT", IB_CRTT, ClientAccountID="U00000000", CurrencyPrimary="BASE_SUMMARY", LevelOfDetail="BaseCurrency",
       StartingCash="0", EndingCash="999"),
    ib("CRTT", IB_CRTT, ClientAccountID="U00000000", CurrencyPrimary="EUR", LevelOfDetail="Currency",
       StartingCash="100", EndingCash="446.75"),
    ib("CRTT", IB_CRTT, ClientAccountID="U00000000", CurrencyPrimary="USD", LevelOfDetail="Currency",
       StartingCash="0", EndingCash="-73.6"),
    ["HEADER", "POST"] + IB_POST,
    ib("POST", IB_POST, ClientAccountID="U00000000", CurrencyPrimary="EUR", AssetClass="STK", SubCategory="ETF",
       Symbol="EGLN", ISIN="IE00B4ND3602", ListingExchange="FWB2", Multiplier="1", Quantity="3", LevelOfDetail="SUMMARY"),
    ib("POST", IB_POST, ClientAccountID="U00000000", CurrencyPrimary="USD", AssetClass="OPT",
       Symbol="XYZ   261218P00040000", ListingExchange="CBOE", Multiplier="100", Quantity="-1", LevelOfDetail="SUMMARY"),
]


def test_parse_ibkr_tipuri_simboluri_clase_si_sume():
    records, problems, control = B.parse_ibkr(IB_ROWS)
    rows = {r["ID"]: r for r in records}
    assert set(rows) == {"IB-1001", "IB-1002-EUR", "IB-1002-USD", "IB-1002-FEE", "IB-1003", "IB-1004", "IB-1005",
                         "IB-2001", "IB-2002", "IB-2003"}
    gold = rows["IB-1001"]                                   # simbolul Yahoo vine din ISIN, ora New York -> România
    assert (gold["Symbol"], gold["Class"], gold["Type"], gold["Quantity"], gold["Price"], gold["Fees"], gold["Amount"]) == \
        ("PPFB.DE", "ETF", "BUY", 3, 50, 1.25, -151.25)
    assert gold["Date"] == "2026-11-12 10:38:44" and gold["Currency"] == "EUR" and gold["Broker"] == "IBKR"
    assert "ETC aur" in gold["Notes"] and "EGLN" in gold["Notes"]
    assert (rows["IB-1002-EUR"]["Amount"], rows["IB-1002-USD"]["Amount"], rows["IB-1002-FEE"]["Amount"]) == (-500, 550, -2)
    assert rows["IB-1002-FEE"]["Currency"] == "EUR" and rows["IB-1002-USD"]["Type"] == "FX"
    brk = rows["IB-1003"]                                    # vară: New York UTC-4, București UTC+3
    assert (brk["Symbol"], brk["Class"], brk["Date"], brk["Amount"], brk["Fees"]) == ("BRK-B", "Actiuni SUA", "2026-07-01 17:00:00", -401, 1)
    call = rows["IB-1004"]                                   # opțiuni: cantitatea în acțiuni-suport, prețul pe acțiune
    assert (call["Symbol"], call["Class"], call["Quantity"], call["Price"], call["Amount"]) == ("XYZ261218C00050000", "Optiuni", 200, 1.5, -301.3)
    assert rows["IB-2001"]["Type"] == "DEPOSIT" and rows["IB-2001"]["Amount"] == 1000 and rows["IB-2001"]["Date"] == "2026-11-06 00:00:00"
    assert (rows["IB-2002"]["Type"], rows["IB-2002"]["Symbol"], rows["IB-2003"]["Type"], rows["IB-2003"]["Amount"]) == ("DIV", "BRK-B", "TAX", -0.3)
    assert all("U00000000" not in str(v) for r in records for v in r.values())     # contul nu ajunge în foaie
    assert any("short" in p and "XYZ261218P00040000" in p for p in problems)
    assert any("Ceva Nou" in p for p in problems)
    assert any("CBOE" in p for p in problems) is False                          # opțiunile nu cer bursă cunoscută
    # numerar: EUR 1000 − 151,25 − 500 − 2 = 346,75 = 446,75 − 100 ; USD 550 − 401 − 301,3 + 79 + 2 − 0,3 = −71,6 ≠ −73,6
    assert control["cash"]["EUR"] == (346.75, 346.75)
    assert control["cash"]["USD"] == (-73.6, -71.6)
    assert control["positions"] == {"PPFB.DE": 3.0, "XYZ261218P00040000": -100.0}


def test_parse_ibkr_bursa_necunoscuta_si_fara_coduri_de_sectiune():
    rows = [["HEADER", "TRNT"] + IB_TRNT,
            ib_trade(CurrencyPrimary="EUR", AssetClass="STK", Symbol="QQQ1", ListingExchange="XYZEX", DateTime="2026-11-12;05:00:00",
                     Quantity="1", TradePrice="5", Proceeds="-5", IBCommission="-1", IBCommissionCurrency="EUR",
                     **{"Buy/Sell": "BUY"}, TradeID="1", TransactionID="9")]
    records, problems, _ = B.parse_ibkr(rows)
    assert records[0]["Symbol"] == "QQQ1" and any("XYZEX" in p for p in problems)
    records, problems, control = B.parse_ibkr([["ClientAccountID", "Symbol"], ["U1", "AAA"]])
    assert records == [] and "Include section code" in problems[0] and control == {}


def test_ibkr_detectie_csv_si_registru():
    text = "﻿" + "\n".join(",".join(f'"{c}"' for c in row) for row in IB_ROWS)
    rows = B.read_csv_rows(text.encode("utf-8"))
    assert rows[0][0] == "HEADER"
    broker, records, _, _ = B.parse_export({"csv": rows})
    assert broker == "IBKR" and len(records) == 10
    assert B.detect_broker({"x": [("data", "data", "simbol")]}) is None
    ledger, problems = L.parse_ledger([L.COLUMNS] + B.to_sheet_rows(records))
    assert problems == []
    cash = L.cash_balances(ledger).set_index("Currency")["Cash"].round(2).to_dict()
    assert cash == {"EUR": 346.75, "USD": -71.6}
    positions, _ = L.build_positions(ledger)
    gold = positions.set_index("Symbol").loc["PPFB.DE"]
    assert gold["Quantity"] == 3 and gold["CostBasis"] == pytest.approx(151.25)


def test_position_gaps():
    records = [dict(Broker="IBKR", Symbol="AAA", Type="BUY", Quantity=10), dict(Broker="IBKR", Symbol="AAA", Type="SELL", Quantity=4),
               dict(Broker="XTB", Symbol="AAA", Type="BUY", Quantity=99), dict(Broker="IBKR", Symbol="", Type="FX", Quantity=""),
               dict(Broker="IBKR", Symbol="BBB", Type="BUY", Quantity=float("nan"))]
    assert B.position_gaps(records, {"AAA": 6.0}) == []
    assert B.position_gaps(records, {"AAA": 12.0, "CCC": 1.0}) == [("AAA", 6.0, 12.0), ("CCC", 0.0, 1.0)]
