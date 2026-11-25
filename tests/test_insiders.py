"""analytics/insiders.py: parsare Form 4, 4/A, plan 10b5-1, indicatori pe 90 de zile. XML sintetic."""
import pandas as pd
import pytest

from analytics import insiders as ins

NS = ""   # Form 4 nu folosește namespace


def _form4(owners, txs, form="4", period="2026-11-02", plan=None, footnotes=(), derivs=0):
    """owners: [(cik, nume, titlu_ofițer|None, director)]; txs: [(data, cod, acț, preț|None, A/D, după, note_ids)]"""
    own = "".join(
        f"<reportingOwner><reportingOwnerId><rptOwnerCik>{c}</rptOwnerCik><rptOwnerName>{n}</rptOwnerName>"
        f"</reportingOwnerId><reportingOwnerRelationship><isDirector>{int(d)}</isDirector>"
        + (f"<isOfficer>1</isOfficer><officerTitle>{t}</officerTitle>" if t else "")
        + "</reportingOwnerRelationship></reportingOwner>"
        for c, n, t, d in owners)
    rows = ""
    for dt, code, sh, px, ad, after, fids in txs:
        price = f"<value>{px}</value>" if px is not None else ""
        fn = "".join(f'<footnoteId id="{i}"/>' for i in fids)
        rows += (f"<nonDerivativeTransaction><securityTitle><value>Common Stock</value></securityTitle>"
                 f"<transactionDate><value>{dt}</value></transactionDate>"
                 f"<transactionCoding><transactionFormType>4</transactionFormType><transactionCode>{code}"
                 f"</transactionCode>{fn}</transactionCoding>"
                 f"<transactionAmounts><transactionShares><value>{sh}</value></transactionShares>"
                 f"<transactionPricePerShare>{price}</transactionPricePerShare>"
                 f"<transactionAcquiredDisposedCode><value>{ad}</value></transactionAcquiredDisposedCode>"
                 f"</transactionAmounts><postTransactionAmounts><sharesOwnedFollowingTransaction><value>{after}"
                 f"</value></sharesOwnedFollowingTransaction></postTransactionAmounts>"
                 f"<ownershipNature><directOrIndirectOwnership><value>D</value></directOrIndirectOwnership>"
                 f"</ownershipNature></nonDerivativeTransaction>")
    der = "<derivativeTable>" + "<derivativeTransaction/>" * derivs + "</derivativeTable>" if derivs else ""
    fns = "".join(f'<footnote id="{i}">{t}</footnote>' for i, t in footnotes)
    aff = f"<aff10b5One>{plan}</aff10b5One>" if plan is not None else ""
    xml = (f"<?xml version='1.0'?><ownershipDocument><schemaVersion>X0508</schemaVersion>"
           f"<documentType>{form}</documentType><periodOfReport>{period}</periodOfReport>{aff}"
           f"<issuer><issuerCik>0000320193</issuerCik><issuerName>Apple Inc.</issuerName>"
           f"<issuerTradingSymbol>AAPL</issuerTradingSymbol></issuer>{own}"
           f"<nonDerivativeTable>{rows}</nonDerivativeTable>{der}<footnotes>{fns}</footnotes></ownershipDocument>")
    return xml.encode()


def test_parse_purchase_ceo_and_fields():
    d = ins.parse_form4(_form4([("111", "Doe John", "Chief Executive Officer", False)],
                               [("2026-11-02", "P", "1,000", "150.5", "A", "11000", ())], derivs=2),
                        accession="0001-26-1", filing_date="2026-11-04")
    assert d["simbol"] == "AAPL" and d["forma"] == "4" and d["ceo_cfo"] and d["derivate"] == 2
    t = d["tranzactii"][0]
    assert (t["code"], t["shares"], t["price"], t["after"], t["plan"]) == ("P", 1000.0, 150.5, 11000.0, False)
    assert "Chief Executive Officer" in d["functie"]


def test_parse_plan_flag_and_footnote():
    flag = ins.parse_form4(_form4([("2", "A", "SVP", False)], [("2026-11-03", "S", 100, 350, "D", 0, ())], plan=1))
    note = ins.parse_form4(_form4([("3", "B", None, True)], [("2026-11-03", "S", 100, 350, "D", 0, ("F1",))],
                                  footnotes=[("F1", "Sold pursuant to a Rule 10b5-1 trading plan adopted ...")]))
    none = ins.parse_form4(_form4([("4", "C", None, True)], [("2026-11-03", "S", 100, 350, "D", 0, ())], plan=0))
    assert flag["tranzactii"][0]["plan"] and note["tranzactii"][0]["plan"] and not none["tranzactii"][0]["plan"]


def test_parse_rejects_other_xml_and_missing_price_is_none():
    with pytest.raises(ins.ParseForm4Error):
        ins.parse_form4(b"<informationTable/>")
    with pytest.raises(ins.ParseForm4Error):
        ins.parse_form4(b"<nu e xml")
    d = ins.parse_form4(_form4([("5", "D", None, True)], [("2026-11-03", "G", 50000, None, "D", 0, ())]))
    assert d["tranzactii"][0]["price"] is None


def _docs():
    """AAPL în fereastra 07.08–05.11.2026: 3 cumpărători P (unul CEO, unul în comun), 1 S discreționar,
    2 S prin plan, un cadou G, o exercitare M; un P vechi în afara ferestrei; un 4/A care înlocuiește un 4."""
    d = [
        ins.parse_form4(_form4([("10", "CEO X", "CEO", False)], [("2026-11-02", "P", 1000, 100, "A", 5000, ())]),
                        "a1", "2026-11-03"),
        ins.parse_form4(_form4([("11", "Dir Y", None, True)], [("2026-10-01", "P", 200, 110, "A", 1200, ())]),
                        "a2", "2026-10-02"),
        ins.parse_form4(_form4([("12", "Fund Z", None, False), ("13", "Fund Z GP", None, False)],
                               [("2026-09-15", "P", 300, 120, "A", 300, ())]), "a3", "2026-09-16"),
        ins.parse_form4(_form4([("14", "VP S", "SVP", False)], [("2026-10-20", "S", 2399, 353, "D", 0, ()),
                                                                 ("2026-10-27", "S", 2399, 353, "D", 0, ())], plan=1),
                        "a4", "2026-10-28"),
        ins.parse_form4(_form4([("15", "Dir Q", None, True)], [("2026-10-10", "S", 100, 300, "D", 0, ()),
                                                                ("2026-10-10", "G", 50000, None, "D", 0, ()),
                                                                ("2026-10-10", "M", 400, 0, "A", 400, ())]),
                        "a5", "2026-10-12"),
        ins.parse_form4(_form4([("16", "Old Buyer", None, True)], [("2026-07-01", "P", 999, 99, "A", 999, ())]),
                        "a6", "2026-07-02"),
        # 4 inițial cu o greșeală, corectat de 4/A (același insider, aceeași dată a raportului)
        ins.parse_form4(_form4([("11", "Dir Y", None, True)], [("2026-10-05", "S", 9999, 1, "D", 0, ())],
                               period="2026-10-05"), "a7", "2026-10-06"),
        ins.parse_form4(_form4([("11", "Dir Y", None, True)], [("2026-10-05", "S", 99, 100, "D", 0, ())],
                               form="4/A", period="2026-10-05"), "a8", "2026-10-09"),
    ]
    return d


def test_transactions_frame_amendment_and_order():
    df = ins.transactions_frame(_docs())
    assert "a7" not in set(df["Accession"]) and "a8" in set(df["Accession"])
    assert df["Data_tranz"].is_monotonic_decreasing
    joint = df[df["Accession"] == "a3"].iloc[0]
    assert joint["Insider"] == "Fund Z / Fund Z GP" and joint["Cheie_insider"] == "12|13"
    assert df.loc[df["Cod"] == "G", "Valoare_USD"].isna().all()


def test_window_summary_reference_values():
    df = ins.transactions_frame(_docs())
    s = ins.window_summary(df, "2026-11-05")
    assert s["nr_P"] == 3 and s["cumparatori"] == 3 and s["grup"] and s["ceo_cfo_P"]
    assert s["valoare_P"] == pytest.approx(1000 * 100 + 200 * 110 + 300 * 120)          # 158.000; P din iulie exclus
    # % din deținerea anterioară: 1000/4000 = 25%, 200/1000 = 20%, 300/0 → exclus → mediana 22,5%
    assert s["pct_din_detinere_P"] == pytest.approx(0.225)
    assert s["nr_S_plan"] == 2 and s["valoare_S_plan"] == pytest.approx(2 * 2399 * 353)
    assert s["nr_S_discretionar"] == 2 and s["valoare_S_discretionar"] == pytest.approx(100 * 300 + 99 * 100)
    assert s["alte_coduri"] == {"G": 1, "M": 1} and s["fara_pret"] == 0
    total_signal = (df["Data_tranz"] > s["start"]) & df["Cod"].isin(["P", "S"])
    assert s["nr_P"] + s["nr_S_plan"] + s["nr_S_discretionar"] == int(total_signal.sum())
    txt = ins.describe(s)
    assert "Grup" in txt and "CEO/CFO" in txt and "masiv" not in txt.lower()


def test_window_summary_empty_and_no_buys():
    e = ins.window_summary(ins.transactions_frame([]), "2026-11-05")
    assert e["tranzactii"] == 0 and e["valoare_P"] is None and not e["grup"]
    only_sells = ins.transactions_frame([_docs()[3]])
    s = ins.window_summary(only_sells, "2026-11-05")
    assert s["cumparatori"] == 0 and s["valoare_P"] is None and s["pct_din_detinere_P"] is None


def test_raw_xml_name_and_cik_lookup():
    assert ins.raw_xml_name("xslF345X05/wk-form4_1730.xml") == "wk-form4_1730.xml"
    assert ins.raw_xml_name("form4.xml") == "form4.xml"
    tick = {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple"},
            "1": {"cik_str": 1067983, "ticker": "BRK-B", "title": "Berkshire"}}
    assert ins.cik_for_ticker(tick, "aapl") == "320193" and ins.cik_for_ticker(tick, "BRK.B") == "1067983"
    assert ins.cik_for_ticker(tick, "ZZZZ") is None


# --------------------------------------------------------------------------- data/sec_insiders.py (client fals)

class _FakeClient:
    def __init__(self, subs, files):
        self.subs, self.files, self.urls = subs, files, []

    def get_json(self, url):
        self.urls.append(url)
        return self.subs

    def get(self, url):
        self.urls.append(url)
        name = url.rsplit("/", 1)[-1]
        if name not in self.files:
            from data.sec_edgar import EdgarError
            raise EdgarError(f"{url}: HTTP 404")
        return self.files[name]


def test_load_form4_filters_dates_and_collects_problems():
    from datetime import date
    from data import sec_insiders as si
    subs = {"filings": {"recent": {
        "accessionNumber": ["0001-26-000003", "0001-26-000002", "0001-26-000001", "0001-26-000009"],
        "form": ["4", "4/A", "4", "10-Q"],
        "filingDate": ["2026-11-04", "2026-10-09", "2026-06-01", "2026-11-01"],
        "reportDate": ["2026-11-02", "2026-10-05", "2026-05-29", "2026-09-30"],
        "primaryDocument": ["xslF345X05/f3.xml", "xslF345X05/f2.xml", "xslF345X05/f1.xml", "q.htm"]}}}
    files = {"f3.xml": _form4([("10", "CEO X", "CEO", False)], [("2026-11-02", "P", 1000, 100, "A", 5000, ())])}
    c = _FakeClient(subs, files)
    docs, problems, truncated = si.load_form4(c, "320193", today=date(2026, 11, 5))
    assert [d["accession"] for d in docs] == ["0001-26-000003"] and not truncated
    assert len(problems) == 1 and "0001-26-000002" in problems[0]                    # f2 lipsește → problemă
    assert not any("000001" in u.rsplit("/", 2)[-2] for u in c.urls[1:])             # iunie: în afara ferestrei
    assert c.urls[1].endswith("/320193/000126000003/f3.xml")
    rows, trunc = si.form4_filings(subs, "2026-01-01", max_filings=2)
    assert len(rows) == 2 and trunc
