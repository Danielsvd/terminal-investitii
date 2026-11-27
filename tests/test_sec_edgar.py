"""data/sec_edgar.py: User-Agent, ritm, reîncercări, liste de depuneri. Sesiune falsă, fără rețea."""
import pytest
import requests

from data import sec_edgar as e


class _Resp:
    def __init__(self, code, body=b"{}"):
        self.status_code, self.content = code, body


class _Session:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def get(self, url, headers=None, timeout=None):
        self.calls.append((url, headers))
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def _client(responses, **kw):
    sleeps = []
    c = e.EdgarClient("Terminal Investitii test@exemplu.ro", session=_Session(responses),
                      sleep=sleeps.append, clock=lambda: 100.0, **kw)
    return c, sleeps


def test_user_agent_requires_email():
    with pytest.raises(ValueError):
        e.EdgarClient("Terminal Investitii")


def test_get_sends_user_agent_and_throttles():
    c, sleeps = _client([_Resp(200, b"a"), _Resp(200, b"b")])
    assert c.get("u1") == b"a" and c.get("u2") == b"b"
    assert c.session.calls[0][1]["User-Agent"] == "Terminal Investitii test@exemplu.ro"
    assert sleeps == [pytest.approx(1 / 8)]          # ceasul nu avansează → a doua cerere așteaptă 0,125 s


def test_retry_on_429_then_ok():
    c, sleeps = _client([_Resp(429), requests.ConnectionError("x"), _Resp(200, b"ok")], backoff_s=1.0)
    assert c.get("u") == b"ok"
    assert len(c.session.calls) == 3
    assert 1.0 in sleeps and 2.0 in sleeps           # backoff 1 s, apoi 2 s


def test_404_fails_without_retry():
    c, _ = _client([_Resp(404)])
    with pytest.raises(e.EdgarError, match="HTTP 404"):
        c.get("u")
    assert len(c.session.calls) == 1


def test_get_json_invalid():
    c, _ = _client([_Resp(200, b"nu e json")])
    with pytest.raises(e.EdgarError, match="JSON invalid"):
        c.get_json("u")


def test_urls():
    assert e.cik10(1067983) == "0001067983"
    assert e.submissions_url("1067983") == "https://data.sec.gov/submissions/CIK0001067983.json"
    assert e.filing_index_url("0001067983", "0000950123-26-012345") == \
        "https://www.sec.gov/Archives/edgar/data/1067983/000095012326012345/index.json"
    with pytest.raises(ValueError):
        e.cik10("BRK")


def test_recent_filings_filters_and_sorts():
    sub = {"filings": {"recent": {
        "accessionNumber": ["a1", "a2", "a3", "a4"],
        "form": ["4", "13F-HR", "13F-HR/A", "13F-HR"],
        "filingDate": ["2026-11-20", "2026-08-14", "2026-11-14", "2026-11-14"],
        "reportDate": ["", "2026-06-30", "2026-09-30", "2026-09-30"],
        "primaryDocument": ["x.xml", "p.xml", "p.xml", "p.xml"],
    }, "files": [{"name": "CIK0001067983-submissions-001.json"}]}}
    f = e.recent_filings(sub)
    assert [r["accession"] for r in f] == ["a4", "a3", "a2"]
    assert f[0]["report_date"] == "2026-09-30"
    assert e.older_pages(sub) == ["CIK0001067983-submissions-001.json"]
    assert e.recent_filings({}) == []


def test_info_table_candidates():
    idx = {"directory": {"item": [{"name": "primary_doc.xml"}, {"name": "50240.xml"},
                                  {"name": "0000950123-26-012345.txt"}, {"name": "index.json"}]}}
    assert e.info_table_candidates(idx) == ("primary_doc.xml", ["50240.xml"])
