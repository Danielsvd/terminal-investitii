"""Găsirea și descărcarea rapoartelor Pilon II, fără rețea (sesiune falsă)."""
import json
from datetime import date

from tools.pilon2 import fetch


class FakeResp:
    def __init__(self, status=200, content=b"", ctype="application/pdf", text=""):
        self.status_code, self._content, self.text = status, content, text
        self.headers = {"Content-Type": ctype}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_content(self, _n):
        yield self._content

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeSession:
    """url → FakeResp; orice altceva → 404. Ține evidența cererilor."""
    def __init__(self, routes, head_not_allowed=False):
        self.routes, self.calls, self.head_not_allowed = routes, [], head_not_allowed

    def _get(self, url):
        return self.routes.get(url, FakeResp(404, ctype="text/html"))

    def head(self, url, **_kw):
        self.calls.append(("HEAD", url))
        if self.head_not_allowed:
            return FakeResp(405, ctype="text/html")
        return self._get(url)

    def get(self, url, **_kw):
        self.calls.append(("GET", url))
        return self._get(url)


def test_month_ends_skips_unfinished_month():
    assert fetch.month_ends(date(2026, 11, 15), 3) == [date(2026, 10, 31), date(2026, 9, 30), date(2026, 8, 31)]
    assert fetch.month_ends(date(2026, 10, 31), 1) == [date(2026, 10, 31)]
    assert fetch.month_ends(date(2024, 3, 10), 2) == [date(2024, 2, 29), date(2024, 1, 31)]


def test_candidate_urls_fill_zero_padded_dates():
    urls = fetch.candidate_urls(fetch.SOURCES["vital"], date(2026, 2, 28))
    assert urls == ["https://www.carpathiapensii.ro/wp-content/uploads/rapoarte_vital/2026/20260228_A12_detaliata_FP2-54.pdf"]
    bt = fetch.candidate_urls(fetch.SOURCES["bt"], date(2025, 8, 31))
    assert bt[0].endswith("/2025/FPAP%20structura%20detaliata%20a%20portofoliului%20la%2031.08.2025.pdf")
    assert bt[1].endswith("/2025/FPAP_structura%20detaliata%20a%20portofoliului%20la%2031.08.2025.pdf")
    assert len(fetch.candidate_urls(fetch.SOURCES["metropolitan"], date(2025, 7, 31))) == 12


def test_date_from_text_variants():
    d = fetch.date_from_text
    assert d("Str. Inv. Raport Anual 28.02.2026_FPAP NN.pdf") == date(2026, 2, 28)
    assert d("situatia-detaliata-a-portofoliului-de-investitii-la-data-de-31-03-2025.pdf") == date(2025, 3, 31)
    assert d("20260131_A12_detaliata_FP2-54.pdf") == date(2026, 1, 31)
    assert d("Mai 2025 FPAP NN.pdf") == date(2025, 5, 31)
    assert d("Structura portofoliu februarie 2024") == date(2024, 2, 29)
    assert d("Structura_Portofoliu_Investitii_NN_P2_052022.pdf") == date(2022, 5, 31)
    assert d("raport fara data") is None
    assert d("31.02.2026") is None


def test_pdf_links_filters_and_dates():
    html = """
      <a href="/files/2026-06/Str.%20Inv.%20Raport%20Anual%2028.02.2026_FPAP%20NN.pdf">Structura 28.02.2026</a>
      <a href="https://x.ro/buletin_lunar_nn_fpap_jun_2026.pdf">Buletin lunar iunie 2026</a>
      <a href="/doc/regulament.pdf">Prospect</a>
      <a href="/files/structura-martie-2026.pdf">Structura portofoliului – martie 2026</a>
      <a href="/files/structura-martie-2026.pdf">dublură</a>
      <a href="/pagina.html">Structura (pagină)</a>
    """
    links = fetch.pdf_links(html, "https://www.nn.ro/pensii/")
    assert [(u, d) for u, _t, d in links] == [
        ("https://www.nn.ro/files/2026-06/Str.%20Inv.%20Raport%20Anual%2028.02.2026_FPAP%20NN.pdf", date(2026, 2, 28)),
        ("https://www.nn.ro/files/structura-martie-2026.pdf", date(2026, 3, 31)),
    ]


def test_discover_template_tries_in_order_and_stops_at_first_pdf():
    src = {"tip": "sablon", "sabloane": ["https://a/{Y}{m}{d}.pdf", "https://b/{d}.{m}.{Y}.pdf"]}
    s = FakeSession({"https://b/31.01.2026.pdf": FakeResp(200),
                     "https://a/20251231.pdf": FakeResp(200),
                     "https://a/20251130.pdf": FakeResp(200, ctype="text/html")})   # pagină de eroare, nu PDF
    found = fetch.discover("x", src, [date(2026, 1, 31), date(2025, 12, 31), date(2025, 11, 30)], s, pause=0)
    assert found == [(date(2026, 1, 31), "https://b/31.01.2026.pdf"), (date(2025, 12, 31), "https://a/20251231.pdf")]
    assert ("HEAD", "https://b/31.12.2025.pdf") not in s.calls       # oprit după primul PDF găsit


def test_probe_falls_back_to_get_when_head_not_allowed():
    s = FakeSession({"https://a/x.pdf": FakeResp(200)}, head_not_allowed=True)
    assert fetch.probe(s, "https://a/x.pdf") is True
    assert ("GET", "https://a/x.pdf") in s.calls


def test_discover_page_source_and_missing_page():
    page = "https://bcr.ro/rapoarte"
    html = '<a href="/r/Situatia_detaliata_FPAP_BCR_la_data_de_31.07.2026.pdf">31.07.2026</a>'
    s = FakeSession({page: FakeResp(200, ctype="text/html", text=html)})
    src = {"tip": "pagina", "pagina": page}
    assert fetch.discover("bcr", src, [date(2026, 7, 31), date(2026, 6, 30)], s, pause=0) == [
        (date(2026, 7, 31), "https://bcr.ro/r/Situatia_detaliata_FPAP_BCR_la_data_de_31.07.2026.pdf")]
    assert fetch.discover("nn", {"tip": "pagina", "pagina": None}, [date(2026, 7, 31)], s, pause=0) == []


def test_download_rejects_non_pdf_and_writes_pdf(tmp_path):
    s = FakeSession({"https://a/ok.pdf": FakeResp(200, content=b"%PDF-1.7 ..."),
                     "https://a/html.pdf": FakeResp(200, content=b"<html>eroare</html>")})
    ok = tmp_path / "pdf" / "vital" / "2026-01-31.pdf"
    assert fetch.download(s, "https://a/ok.pdf", ok) and ok.read_bytes().startswith(b"%PDF")
    bad = tmp_path / "pdf" / "vital" / "2025-12-31.pdf"
    assert not fetch.download(s, "https://a/html.pdf", bad) and not bad.exists()
    assert not list(tmp_path.rglob("*.part"))


def test_fetch_all_skips_archived_and_uses_override(tmp_path):
    (tmp_path / "surse.json").write_text(json.dumps({"vital": {"sabloane": ["https://v/{Y}{m}{d}.pdf"]}}))
    archived = fetch.archive_path(tmp_path, "vital", date(2026, 9, 30))
    archived.parent.mkdir(parents=True)
    archived.write_bytes(b"%PDF vechi")
    s = FakeSession({"https://v/20261031.pdf": FakeResp(200, content=b"%PDF nou")})
    res = fetch.fetch_all(tmp_path, 2, keys=["vital"], session=s, today=date(2026, 11, 15), pause=0)
    assert [(d, new) for d, _p, new in res["vital"]] == [(date(2026, 10, 31), True), (date(2026, 9, 30), False)]
    assert not any("20260930" in u for _m, u in s.calls)        # cel arhivat nu se mai cere
