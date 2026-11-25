"""data/sec_funds.py, data/cusip_map.py și prețurile pentru mișcări (analytics/sec_13f). Fără rețea."""
import json

import numpy as np
import pandas as pd
import pytest

from analytics import sec_13f
from data import cusip_map as cm
from data import sec_funds as sf
from tests.test_sec_13f import INFO_XML, _cover


# --------------------------------------------------------------------------- administratori

def test_managers_unique_and_classes():
    m = sf.managers_frame()
    assert m["CIK"].is_unique and len(m) == 20
    assert set(m.loc[m["Clasa"] == sf.LARG, "Fond"]) == {
        "Bridgewater Associates", "Citadel Advisors", "Renaissance Technologies", "Point72 Asset Management"}
    assert m.loc[m["Fond"] == "Discovery Capital Management", "CIK"].item() == "1389507"


# --------------------------------------------------------------------------- încărcare EDGAR (client fals)

class _Client:
    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def get(self, url):
        self.calls.append(url)
        for key, body in self.routes.items():
            if url.endswith(key):
                return body
        raise sf.sec_edgar.EdgarError(f"{url}: HTTP 404")

    def get_json(self, url):
        return json.loads(self.get(url))


def _index(*names):
    return json.dumps({"directory": {"item": [{"name": n} for n in names]}}).encode()


def test_latest_quarters_groups_and_amendments():
    filings = [
        {"accession": "a4", "form": "13F-HR/A", "filing_date": "2026-11-20", "report_date": "2026-09-30"},
        {"accession": "a3", "form": "13F-HR", "filing_date": "2026-11-14", "report_date": "2026-09-30"},
        {"accession": "a2", "form": "13F-HR", "filing_date": "2026-08-14", "report_date": "2026-06-30"},
        {"accession": "a1", "form": "13F-HR", "filing_date": "2026-05-15", "report_date": "2026-03-31"},
    ]
    q = sf.latest_quarters(filings, n=2)
    assert [x["period"] for x in q] == ["2026-09-30", "2026-06-30"]
    assert q[0]["base"]["accession"] == "a3" and [a["accession"] for a in q[0]["amendments"]] == ["a4"]


def test_load_quarter_valid_and_space_in_filename():
    acc = "0000000000-26-000001"
    routes = {
        "/000000000026000001/index.json": _index("primary_doc.xml", "HCI 13F Q3.xml"),
        "/000000000026000001/primary_doc.xml": _cover(),
        "/000000000026000001/HCI%2013F%20Q3.xml": INFO_XML,
    }
    c = _Client(routes)
    q = {"period": "2026-09-30", "base": {"accession": acc, "form": "13F-HR", "filing_date": "2026-11-14"},
         "amendments": []}
    res = sf.load_quarter(c, "123", q)
    assert res["problems"] == [] and len(res["info"]) == 6
    assert res["filer_name"] == "FOND TEST LLC"
    book = sec_13f.equity_book(res["info"], 1.0)
    assert book["Valoare_USD"].sum() == 1500


def test_load_quarter_thousands_before_2023():
    routes = {
        "/000000000022000001/index.json": _index("primary_doc.xml", "t.xml"),
        "/000000000022000001/primary_doc.xml": _cover(),
        "/000000000022000001/t.xml": INFO_XML,
    }
    q = {"period": "2022-09-30", "base": {"accession": "0000000000-22-000001", "form": "13F-HR",
                                          "filing_date": "2022-11-14"}, "amendments": []}
    res = sf.load_quarter(_Client(routes), "123", q)
    assert sec_13f.equity_book(res["info"], 1.0)["Valoare_USD"].sum() == 1_500_000


def test_load_quarter_invalid_base_is_empty():
    routes = {
        "/000000000026000002/index.json": _index("primary_doc.xml", "t.xml"),
        "/000000000026000002/primary_doc.xml": _cover(entry=9),
        "/000000000026000002/t.xml": INFO_XML,
    }
    q = {"period": "2026-09-30", "base": {"accession": "0000000000-26-000002", "form": "13F-HR",
                                          "filing_date": "2026-11-14"}, "amendments": []}
    res = sf.load_quarter(_Client(routes), "123", q)
    assert res["info"].empty and "tableEntryTotal" in res["problems"][0]


def test_load_quarter_restatement_replaces():
    one_row = (b'<informationTable xmlns="x"><infoTable><nameOfIssuer>ZZZ</nameOfIssuer><titleOfClass>COM'
               b'</titleOfClass><cusip>000009ZZ9</cusip><value>70</value><shrsOrPrnAmt><sshPrnamt>7</sshPrnamt>'
               b'<sshPrnamtType>SH</sshPrnamtType></shrsOrPrnAmt><investmentDiscretion>SOLE</investmentDiscretion>'
               b'</infoTable></informationTable>')
    routes = {
        "/000000000026000003/index.json": _index("primary_doc.xml", "t.xml"),
        "/000000000026000003/primary_doc.xml": _cover(),
        "/000000000026000003/t.xml": INFO_XML,
        "/000000000026000004/index.json": _index("primary_doc.xml", "a.xml"),
        "/000000000026000004/primary_doc.xml": _cover(entry=1, value=70, amend="true", amend_type="RESTATEMENT"),
        "/000000000026000004/a.xml": one_row,
    }
    q = {"period": "2026-09-30",
         "base": {"accession": "0000000000-26-000003", "form": "13F-HR", "filing_date": "2026-11-14"},
         "amendments": [{"accession": "0000000000-26-000004", "form": "13F-HR/A", "filing_date": "2026-11-20"}]}
    res = sf.load_quarter(_Client(routes), "123", q)
    assert res["info"]["CUSIP"].tolist() == ["000009ZZ9"] and res["last_filing_date"] == "2026-11-20"


# --------------------------------------------------------------------------- OpenFIGI

class _Post:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append((json, headers))
        return self.responses.pop(0)


class _R:
    def __init__(self, code, payload):
        self.status_code, self._p = code, payload

    def json(self):
        return self._p


def test_pick_listing_prefers_us_composite_and_converts_slash():
    data = [{"ticker": "BRK/B", "exchCode": "UN", "marketSector": "Equity", "name": "BERKSHIRE"},
            {"ticker": "BRK/B", "exchCode": "US", "marketSector": "Equity", "name": "BERKSHIRE HATHAWAY"},
            {"ticker": "BRYN", "exchCode": "GR", "marketSector": "Equity", "name": "BERKSHIRE DE"}]
    assert cm.pick_listing(data) == ("BRK-B", "BERKSHIRE HATHAWAY")
    assert cm.pick_listing([{"ticker": "X", "exchCode": "GR"}]) == (None, None)


def test_openfigi_lookup_batches_without_key_and_reports_failures():
    cusips = [f"{i:09d}" for i in range(12)]          # fără cheie: 10 + 2
    r1 = [{"data": [{"ticker": f"T{i}", "exchCode": "US", "name": f"N{i}"}]} for i in range(10)]
    r1[3] = {"warning": "No identifier found."}
    r2 = [{"data": [{"ticker": "T10", "exchCode": "US"}]}, {"data": [{"ticker": "Z", "exchCode": "LN"}]}]
    sess, sleeps = _Post([_R(200, r1), _R(200, r2)]), []
    found, failed = cm.openfigi_lookup(cusips, session=sess, sleep=sleeps.append)
    assert len(sess.calls) == 2 and len(sess.calls[0][0]) == 10
    assert "X-OPENFIGI-APIKEY" not in sess.calls[0][1]
    assert sorted(failed) == ["000000003", "000000011"]
    assert found["000000000"] == ("T0", "N0") and len(found) == 10
    assert sleeps == [pytest.approx(2.5)]


def test_openfigi_lookup_with_key_and_429_retry():
    sess, sleeps = _Post([_R(429, None), _R(200, [{"data": [{"ticker": "AAPL", "exchCode": "US"}]}])]), []
    found, failed = cm.openfigi_lookup(["037833100"], api_key="k", session=sess, sleep=sleeps.append)
    assert found == {"037833100": ("AAPL", None)} and failed == []
    assert sess.calls[0][1]["X-OPENFIGI-APIKEY"] == "k" and sleeps == [6.0]


def test_map_sheet_roundtrip():
    from datetime import date
    rows = cm.map_rows({"037833100": ("AAPL", "APPLE INC")}, today=date(2026, 11, 25))
    assert rows == [["037833100", "AAPL", "APPLE INC", "openfigi", "2026-11-25"]]
    values = [cm.MAP_COLUMNS] + rows + [["000000001", "", "", "openfigi", "2026-11-25"]]
    assert cm.parse_map(values) == {"037833100": "AAPL"}
    assert cm.parse_map([["X", "Y"]]) == {}


# --------------------------------------------------------------------------- prețuri și split-uri

def _px():
    idx = pd.to_datetime(["2026-06-30", "2026-07-15", "2026-08-15", "2026-09-30", "2026-10-15"])
    # Yahoo: închideri ajustate pentru split-ul 2:1 din 15.08 (prețurile dinainte sunt deja înjumătățite)
    close = pd.DataFrame({"AAA": [50.0, 51.0, 52.0, 54.0, 56.0], "BBB": [10.0, np.nan, 12.0, 14.0, 15.0]}, index=idx)
    splits = pd.DataFrame({"AAA": [0, 0, 2.0, 0, 0], "BBB": [0, 0, 0, 0, 0]}, index=idx, dtype=float)
    return close, splits


def test_raw_closes_undo_later_splits():
    close, splits = _px()
    raw = sec_13f.raw_closes(close, splits)
    assert raw["AAA"].tolist() == [100.0, 102.0, 52.0, 54.0, 56.0]     # înainte de split: ×2
    assert raw["BBB"].tolist()[2:] == [12.0, 14.0, 15.0]


def test_period_inputs_split_factor_and_avg():
    close, splits = _px()
    sf_, avg = sec_13f.period_inputs(close, splits, "2026-06-30", "2026-09-30",
                                     {"c1": "AAA", "c2": "BBB", "c3": "LIPSA", "c4": None})
    assert sf_ == {"c1": 2.0}
    assert avg["c1"] == pytest.approx((102 + 52 + 54) / 3)         # (30.06, 30.09], prețuri brute
    assert avg["c2"] == pytest.approx((12 + 14) / 2)               # NaN exclus
    assert "c3" not in avg and "c4" not in avg
