"""Schedule 13D / 13G: parsare XML (ambele spații de nume), antet SGML, deținătorii actuali, încărcare cu client fals.
Doar date sintetice."""
from datetime import date

import pandas as pd
import pytest

from analytics import schedule13 as s


def _xml_13d(persons, filer="0000000111", issuer="0000320193", event="11/12/2026", form="SCHEDULE 13D", amend=None):
    """13D: spațiul de nume schedule13D; reportingPersonInfo cu percentOfClass / aggregateAmountOwned."""
    ps = "".join(
        f"<reportingPersonInfo><reportingPersonCIK>{c}</reportingPersonCIK><reportingPersonName>{n}</reportingPersonName>"
        f"<typeOfReportingPerson>IA</typeOfReportingPerson><aggregateAmountOwned>{a}</aggregateAmountOwned>"
        f"<percentOfClass>{p}</percentOfClass></reportingPersonInfo>" for c, n, a, p in persons)
    am = f"<amendmentNo>{amend}</amendmentNo>" if amend else ""
    return (f'<?xml version="1.0" encoding="UTF-8"?><edgarSubmission xmlns="http://www.sec.gov/edgar/schedule13D" '
            f'xmlns:com="http://www.sec.gov/edgar/common"><headerData><submissionType>{form}</submissionType>'
            f"<filerInfo><filer><filerCredentials><cik>{filer}</cik><ccc>XXXXXXXX</ccc></filerCredentials></filer>"
            f"</filerInfo></headerData><formData><coverPageHeader>{am}<securitiesClassTitle>Common Stock"
            f"</securitiesClassTitle><dateOfEvent>{event}</dateOfEvent><issuerInfo><issuerCIK>{issuer}</issuerCIK>"
            f"<issuerCUSIP>037833100</issuerCUSIP><issuerName>X Inc</issuerName></issuerInfo></coverPageHeader>"
            f"<reportingPersons>{ps}</reportingPersons></formData></edgarSubmission>").encode()


def _xml_13g(persons, filer="0000000222", issuer="0000320193", event="09/30/2026", form="SCHEDULE 13G/A"):
    """13G: spațiul de nume schedule13g; coverPageHeaderReportingPersonDetails cu classPercent."""
    ps = "".join(
        f"<coverPageHeaderReportingPersonDetails><reportingPersonName>{n}</reportingPersonName>"
        f"<reportingPersonBeneficiallyOwnedAggregateNumberOfShares>{a}"
        f"</reportingPersonBeneficiallyOwnedAggregateNumberOfShares><classPercent>{p}</classPercent>"
        f"<typeOfReportingPerson>IA</typeOfReportingPerson></coverPageHeaderReportingPersonDetails>"
        for n, a, p in persons)
    return (f'<?xml version="1.0" encoding="UTF-8"?><edgarSubmission xmlns="http://www.sec.gov/edgar/schedule13g" '
            f'xmlns:com="http://www.sec.gov/edgar/common"><headerData><submissionType>{form}</submissionType>'
            f"<filerInfo><filer><filerCredentials><cik>{filer}</cik></filerCredentials></filer></filerInfo>"
            f"</headerData><formData><coverPageHeader><amendmentNo>2</amendmentNo><securitiesClassTitle>Common Stock"
            f"</securitiesClassTitle><eventDateRequiresFilingThisStatement>{event}</eventDateRequiresFilingThisStatement>"
            f"<issuerInfo><issuerCik>{issuer}</issuerCik><issuerName>X Inc</issuerName><issuerCusip>037833100"
            f"</issuerCusip></issuerInfo></coverPageHeader>{ps}<signatureInformation><reportingPersonName>Semnatar"
            f"</reportingPersonName></signatureInformation></formData></edgarSubmission>").encode()


HDR_OLD = """<SEC-HEADER>0000950123-23-000001.hdr.sgml : 20230214
ACCESSION NUMBER:		0000950123-23-000001
CONFORMED SUBMISSION TYPE:	SC 13G/A
PUBLIC DOCUMENT COUNT:		1
FILED AS OF DATE:		20230214

SUBJECT COMPANY:

	COMPANY DATA:
		COMPANY CONFORMED NAME:			X Inc
		CENTRAL INDEX KEY:			0000320193

FILED BY:

	COMPANY DATA:
		COMPANY CONFORMED NAME:			VANGUARD GROUP INC
		CENTRAL INDEX KEY:			0000102909
</SEC-HEADER>
"""


# --------------------------------------------------------------------------- parsare

def test_parse_13d_group_takes_max_not_sum():
    d = s.parse_schedule_xml(_xml_13d([("0000000111", "Fund LP", 13100807, 16.9),
                                       ("0000000112", "Fund GP LLC", 10657337, 13.8)]),
                             accession="A-1", filing_date="2026-11-14")
    assert d["kind"] == "13D" and not d["amendment"] and d["format"] == "XML"
    assert d["issuer_cik"] == "320193" and d["filer_cik"] == "111" and d["cusip"] == "037833100"
    assert d["event_date"] == pd.Timestamp("2026-11-12") and d["filing_date"] == pd.Timestamp("2026-11-14")
    assert [p["percent"] for p in d["persons"]] == [16.9, 13.8]
    f = s.filing_summary(d)
    assert f["percent"] == 16.9 and f["name"] == "Fund LP" and f["shares"] == 13100807 and f["n_persons"] == 2
    # suma ar fi 30,7% — greșit: aceleași acțiuni raportate de fond și de administrator


def test_parse_13g_namespace_and_ignores_signature_names():
    d = s.parse_schedule_xml(_xml_13g([("VANGUARD GROUP INC", 1.32e9, 8.9)]), accession="G-1", filing_date="2026-11-10")
    assert d["kind"] == "13G" and d["amendment"] and d["amendment_no"] == 2
    assert d["event_date"] == pd.Timestamp("2026-09-30") and d["issuer_cik"] == "320193"
    assert len(d["persons"]) == 1 and d["persons"][0]["name"] == "VANGUARD GROUP INC"     # nu „Semnatar”
    assert d["persons"][0]["shares"] == 1.32e9 and d["persons"][0]["percent"] == 8.9


def test_parse_errors():
    with pytest.raises(s.ParseScheduleError):
        s.parse_schedule_xml(b"<nu e xml")
    with pytest.raises(s.ParseScheduleError):
        s.parse_schedule_xml(b"<edgarSubmission><headerData><submissionType>4</submissionType></headerData></edgarSubmission>")
    with pytest.raises(s.ParseScheduleError):                       # 13D fără persoane raportoare
        s.parse_schedule_xml(_xml_13d([]))


def test_missing_percent_stays_none():
    xml = _xml_13d([("0000000111", "Fund LP", 100, "")])
    d = s.parse_schedule_xml(xml)
    assert d["persons"][0]["percent"] is None
    assert s.filing_summary(d)["percent"] is None                    # N/A, nu 0


def test_parse_sgml_header_old_filing():
    d = s.parse_sgml_header(HDR_OLD.encode(), accession="0000950123-23-000001", filing_date="2023-02-14")
    assert d["kind"] == "13G" and d["amendment"] and d["format"] == "text"
    assert d["filer_cik"] == "102909" and d["issuer_cik"] == "320193"
    assert d["persons"][0]["name"] == "VANGUARD GROUP INC" and d["persons"][0]["percent"] is None
    with pytest.raises(s.ParseScheduleError):
        s.parse_sgml_header("CONFORMED SUBMISSION TYPE:\t10-K\n")


def test_form_helpers():
    assert s.form_kind("SC 13D/A") == "13D" and s.form_kind("SCHEDULE 13G") == "13G" and s.form_kind("4") is None
    assert s.is_amendment("SCHEDULE 13G/A") and not s.is_amendment("SC 13D")


# --------------------------------------------------------------------------- deținătorii actuali

def _docs():
    old_vg = s.parse_sgml_header(HDR_OLD, accession="0000950123-23-000001", filing_date="2023-02-14")
    new_vg = s.parse_schedule_xml(_xml_13g([("VANGUARD GROUP INC", 1.3e9, 8.9)], filer="0000102909"),
                                  accession="0000102909-26-000010", filing_date="2026-11-10")
    act_g = s.parse_schedule_xml(_xml_13g([("Activ Capital", 5e7, 5.4)], filer="0000000333", form="SCHEDULE 13G"),
                                 accession="0000000333-26-000001", filing_date="2026-05-10")
    act_d = s.parse_schedule_xml(_xml_13d([("0000000333", "Activ Capital", 6e7, 6.1)], filer="0000000333"),
                                 accession="0000000333-26-000002", filing_date="2026-11-01")
    exit_g = s.parse_schedule_xml(_xml_13g([("Iesit LP", 1e6, 0.2)], filer="0000000444"),
                                  accession="0000000444-26-000003", filing_date="2026-08-14")
    other = s.parse_schedule_xml(_xml_13g([("Alt Emitent", 1e6, 9.0)], filer="0000000555", issuer="0000000999"),
                                 accession="0000000555-26-000001", filing_date="2026-10-01")
    old_only = s.parse_sgml_header(HDR_OLD.replace("VANGUARD GROUP INC", "BLACKROCK INC").replace("0000102909", "0001364742"),
                                   accession="0001364742-24-000001", filing_date="2024-02-01")
    return [old_vg, new_vg, act_g, act_d, exit_g, other, old_only]


def test_current_holders_latest_per_filer():
    h = s.current_holders(_docs(), issuer_cik="0000320193").set_index("Cheie")
    assert "CIK555" not in h.index                                   # alt emitent: exclus
    vg = h.loc["CIK102909"]
    assert vg["Procent"] == 8.9 and vg["Depuneri"] == 2 and vg["Tip"] == "13G" and vg["Format"] == "XML"
    a = h.loc["CIK333"]
    assert a["Tip"] == "13D" and a["Trecere"] == "13G → 13D" and a["Procent"] == 6.1
    assert h.loc["CIK444", "Sub_prag"] is True or h.loc["CIK444", "Sub_prag"] == True       # 0,2% < 5%
    br = h.loc["CIK1364742"]
    assert br["Detinator"] == "BLACKROCK INC" and pd.isna(br["Procent"]) and br["Sub_prag"] is None


def test_current_holders_order():
    h = s.current_holders(_docs(), issuer_cik="320193")
    # peste prag după procent (N/A la final), apoi cei sub prag
    assert h["Cheie"].tolist() == ["CIK102909", "CIK333", "CIK1364742", "CIK444"]


def test_holders_summary():
    h = s.current_holders(_docs(), issuer_cik="320193")
    sm = s.holders_summary(h, pd.Timestamp("2026-11-27"))
    assert sm["peste_prag"] == 3 and sm["fara_procent"] == 1 and sm["activisti"] == 1 and sm["pasivi"] == 2
    assert sm["sub_prag"] == 1 and sm["d13_noi"] == ["Activ Capital"]          # 13G → 13D în ultimele 90 de zile
    assert sm["ultima_depunere"] == pd.Timestamp("2026-11-10")
    sm2 = s.holders_summary(h, pd.Timestamp("2027-06-01"))
    assert sm2["d13_noi"] == []                                                  # trecerea e mai veche de 90 de zile
    empty = s.holders_summary(s.current_holders([]), pd.Timestamp("2026-11-27"))
    assert empty["peste_prag"] == 0 and empty["ultima_depunere"] is None


def test_amended_13d_inside_window_is_not_new():
    a1 = s.parse_schedule_xml(_xml_13d([("0000000777", "Vechi Activist", 1e7, 7.0)], filer="0000000777"),
                              accession="0000000777-25-000001", filing_date="2025-03-01")
    a2 = s.parse_schedule_xml(_xml_13d([("0000000777", "Vechi Activist", 1.1e7, 7.5)], filer="0000000777",
                                       form="SCHEDULE 13D/A", amend=3), accession="0000000777-26-000002",
                              filing_date="2026-11-20")
    sm = s.holders_summary(s.current_holders([a1, a2]), pd.Timestamp("2026-11-27"))
    assert sm["activisti"] == 1 and sm["d13_noi"] == []                        # amendament, nu 13D nou


# --------------------------------------------------------------------------- data/sec_ownership.py (client fals)

class _FakeClient:
    def __init__(self, jsons, files):
        self.jsons, self.files, self.urls = jsons, files, []

    def get_json(self, url):
        self.urls.append(url)
        name = url.rsplit("/", 1)[-1]
        if name not in self.jsons:
            from data.sec_edgar import EdgarError
            raise EdgarError(f"{url}: HTTP 404")
        return self.jsons[name]

    def get(self, url):
        self.urls.append(url)
        name = url.rsplit("/", 1)[-1]
        if name not in self.files:
            from data.sec_edgar import EdgarError
            raise EdgarError(f"{url}: HTTP 404")
        return self.files[name]


def test_load_schedules_xml_old_pages_and_problems():
    from data import sec_ownership as so
    subs = {"filings": {
        "recent": {
            "accessionNumber": ["0000102909-26-000010", "0000000333-26-000002", "0000000888-26-000001", "0001-26-000004"],
            "form": ["SCHEDULE 13G/A", "SCHEDULE 13D", "SCHEDULE 13G", "4"],
            "filingDate": ["2026-11-10", "2026-11-01", "2026-10-02", "2026-11-20"],
            "reportDate": ["", "", "", "2026-11-18"],
            "primaryDocument": ["xslSCHEDULE_13G_X01/primary_doc.xml", "xslSCHEDULE_13D_X01/primary_doc.xml",
                                "xslSCHEDULE_13G_X01/primary_doc.xml", "xslF345X05/f.xml"]},
        "files": [{"name": "CIK0000320193-submissions-001.json", "filingFrom": "2023-01-01", "filingTo": "2025-06-30"},
                  {"name": "CIK0000320193-submissions-002.json", "filingFrom": "2001-01-01", "filingTo": "2022-12-31"}]}}
    page1 = {"accessionNumber": ["0000950123-23-000001", "0000950123-22-000009"], "form": ["SC 13G/A", "SC 13G"],
             "filingDate": ["2023-12-14", "2022-02-14"], "reportDate": ["", ""], "primaryDocument": ["d.txt", "e.txt"]}
    jsons = {"CIK0000320193.json": subs, "CIK0000320193-submissions-001.json": page1}
    files = {
        "primary_doc.xml": _xml_13g([("VANGUARD GROUP INC", 1.3e9, 8.9)], filer="0000102909"),
        "0000950123-23-000001.hdr.sgml": HDR_OLD.encode(),
    }
    c = _FakeClient(jsons, files)
    docs, problems, truncated = so.load_schedules(c, "320193", today=date(2026, 11, 27))
    # pagina 002 se termină în 2022 < 2023-11-28 → nu se descarcă; 2022-02-14 e oricum în afara ferestrei
    assert not any("submissions-002" in u for u in c.urls)
    accs = [d["accession"] for d in docs]
    assert "0000950123-23-000001" in accs and "0000950123-22-000009" not in accs and not truncated
    assert all("0001-26-000004" not in u for u in c.urls)                          # Form 4 ignorat
    assert any(u.endswith("/320193/000010290926000010/primary_doc.xml") for u in c.urls)
    assert any(u.endswith("/320193/000095012323000001/0000950123-23-000001.hdr.sgml") for u in c.urls)
    assert len(docs) == 4 and problems == []                       # 3 XML (același fișier fals) + 1 antet
    rows = so.schedule_filings(subs, "2026-10-15")
    assert [r["accession"] for r in rows] == ["0000102909-26-000010", "0000000333-26-000002"]


def test_load_schedules_problem_and_truncation():
    from data import sec_ownership as so
    subs = {"filings": {"recent": {
        "accessionNumber": ["0000000001-26-000001", "0000000002-26-000001"],
        "form": ["SC 13D", "SCHEDULE 13D"], "filingDate": ["2026-11-01", "2026-10-01"], "reportDate": ["", ""],
        "primaryDocument": ["a.htm", "xslSCHEDULE_13D_X01/primary_doc.xml"]}}}
    c = _FakeClient({"CIK0000320193.json": subs}, {"primary_doc.xml": b"<nu e xml"})
    docs, problems, truncated = so.load_schedules(c, "320193", today=date(2026, 11, 27), max_filings=5)
    assert docs == [] and len(problems) == 2 and not truncated                      # antet lipsă + XML invalid
    _d, _p, truncated = so.load_schedules(c, "320193", today=date(2026, 11, 27), max_filings=1)
    assert truncated


def test_display_frame():
    v = s.display_frame(s.current_holders(_docs(), issuer_cik="320193"))
    assert v["Deținător"].tolist() == ["VANGUARD GROUP INC", "Activ Capital", "BLACKROCK INC", "Iesit LP"]
    assert v["Tip"].tolist()[:2] == ["13G (pasiv)", "13D (activist) · 13G → 13D"]
    assert v["Procent"].iloc[0] == 8.9 and pd.isna(v["Procent"].iloc[2])            # N/A rămâne NaN, nu 0
    assert v["Sub 5%"].tolist() == ["", "", "", "da"]
    assert s.display_frame(s.current_holders([])).empty
