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


def test_discover_template_then_page_for_missing_months(capsys):
    """BCR, 16.11.2026: șablonul găsește iulie, nu și mai; luna lipsă se ia de pe pagina de listare."""
    page = "https://bcr.ro/rapoarte"
    html = ('<a href="https://cdn/r/la_data_de_31.05.2026.pdf?forceDownload=1">Structura detaliată 31.05.2026</a>'
            '<a href="https://cdn/r/la_data_de_31.07.2026.pdf?forceDownload=1">Structura detaliată 31.07.2026</a>')
    s = FakeSession({"https://cdn/t/31.07.2026.pdf": FakeResp(200),
                     page: FakeResp(200, ctype="text/html", text=html)})
    src = {"tip": "sablon", "sabloane": ["https://cdn/t/{d}.{m}.{Y}.pdf"], "pagina": page}
    found = fetch.discover("bcr", src, [date(2026, 7, 31), date(2026, 6, 30), date(2026, 5, 31)], s, pause=0)
    assert found == [(date(2026, 7, 31), "https://cdn/t/31.07.2026.pdf"),
                     (date(2026, 5, 31), "https://cdn/r/la_data_de_31.05.2026.pdf?forceDownload=1")]
    assert ("GET", page) in s.calls
    # toate lunile găsite prin șablon → pagina nu se mai cere
    s2 = FakeSession({"https://cdn/t/31.07.2026.pdf": FakeResp(200)})
    assert fetch.discover("bcr", src, [date(2026, 7, 31)], s2, pause=0) == [(date(2026, 7, 31), "https://cdn/t/31.07.2026.pdf")]
    assert ("GET", page) not in s2.calls


def test_discover_page_without_matching_reports_says_so(capsys):
    page = "https://azt.ro/rap"
    s = FakeSession({page: FakeResp(200, ctype="text/html", text='<a href="/x/Raportari_contabile.pdf">contabil</a>')})
    assert fetch.discover("azt", {"tip": "pagina", "pagina": page}, [date(2026, 7, 31)], s, pause=0) == []
    assert "pagina nu are rapoarte" in capsys.readouterr().err


def test_discover_page_one_link_per_month():
    """NN, 16.11.2026: același raport apare în mai multe taburi → o singură descărcare pe lună."""
    page = "https://nn.ro/r"
    html = ('<a href="/a/Str.%20Inv.%2031.07.2026_FPAP%20NN.pdf">Structura</a>'
            '<a href="/b/Str.%20Inv.%2031.07.2026_FPAP%20NN.pdf">Structura</a>'
            '<a href="/c/Str.%20Inv.%2031.07.2026_FPAP%20NN.pdf">Structura</a>')
    s = FakeSession({page: FakeResp(200, ctype="text/html", text=html)})
    assert fetch.discover("nn", {"tip": "pagina", "pagina": page}, [date(2026, 7, 31)], s, pause=0) == [
        (date(2026, 7, 31), "https://nn.ro/a/Str.%20Inv.%2031.07.2026_FPAP%20NN.pdf")]


def test_sources_aripi_nn_bcr_16_11():
    assert fetch.candidate_urls(fetch.SOURCES["aripi"], date(2024, 2, 29)) == [
        "https://www.generali.ro/wp-content/uploads/pension-documents/pensii/pensii-obligatorii/"
        "acte-si-rapoarte/date-financiare-fondul-aripi/20240229_pII_inv_det.pdf"]
    assert fetch.SOURCES["bcr"]["cod"] == "FP2-123"
    assert fetch.SOURCES["nn"]["pagina"] == "https://www.nn.ro/rapoarte-financiare"
    # legătura curentă NN (16.11.2026): data vine din numele fișierului, nu din „(2026)” din text
    html = ('<a href="https://www.nn.ro/sites/default/files/2026-11/Str.%20Inv.%20Raport%20Anual%2031.07.2026_FPAP%20NN.pdf">'
            'Structura detaliata a portofoliului de investitii fond pensii private curenta (2026)</a>'
            '<a href="https://www.nn.ro/sites/default/files/2026-11/Arhiva%20Structura%20detaliata.zip">Arhiva</a>')
    assert [d for _u, _t, d in fetch.pdf_links(html, "https://www.nn.ro/rapoarte-financiare")] == [date(2026, 7, 31)]


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


def _zip_bytes(members):
    import io
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in members.items():
            z.writestr(name, data)
    return buf.getvalue()


NN_PAGE = "https://www.nn.ro/rapoarte-financiare"
NN_HTML = (
    '<a href="/sites/default/files/2026-11/Str.%20Inv.%20Raport%20Anual%2031.07.2026_FPAP%20NN.pdf">Structura curenta</a>'
    '<a href="/sites/default/files/2026-11/Arhiva%20Structura%20detaliata%20a%20portofoliului%202021%20-%202026.zip">'
    'Arhiva Structura detaliata a portofoliului de investitii fond pensii private 2021 - 2026</a>'
    '<a href="/sites/default/files/2026-11/Arhiva%20situatiilor%20valorilor%20activului%20net.zip">Arhiva VAN</a>')
NN_ZIP_URL = ("https://www.nn.ro/sites/default/files/2026-11/"
              "Arhiva%20Structura%20detaliata%20a%20portofoliului%202021%20-%202026.zip")


def test_zip_links_and_member_dates():
    assert fetch.zip_links(NN_HTML, NN_PAGE) == [NN_ZIP_URL]            # arhiva VAN nu e luată
    names = ["Iulie 2026.pdf", "Februarie 2023 FPAP NN.pdf", "Iunie 2019.pdf", "readme.txt", "dir/Mai 2026.pdf"]
    assert fetch.zip_members_by_date(names) == {
        date(2026, 7, 31): "Iulie 2026.pdf", date(2023, 2, 28): "Februarie 2023 FPAP NN.pdf",
        date(2019, 6, 30): "Iunie 2019.pdf", date(2026, 5, 31): "dir/Mai 2026.pdf"}


def test_fetch_all_takes_missing_months_from_zip_and_caches_it(tmp_path):
    """NN: luna curentă de pe pagină, restul din arhiva .zip; arhiva nu se descarcă din nou la a doua rulare."""
    z = _zip_bytes({"Iulie 2026.pdf": b"%PDF iulie din zip", "Iunie 2026.pdf": b"%PDF iunie",
                    "Mai 2026.pdf": b"<html>stricat</html>"})
    routes = {NN_PAGE: FakeResp(200, ctype="text/html", text=NN_HTML),
              "https://www.nn.ro/sites/default/files/2026-11/Str.%20Inv.%20Raport%20Anual%2031.07.2026_FPAP%20NN.pdf":
                  FakeResp(200, content=b"%PDF iulie"),
              NN_ZIP_URL: FakeResp(200, ctype="application/zip", content=z)}
    s = FakeSession(routes)
    src = {"nn": dict(fetch.SOURCES["nn"])}
    res = fetch.fetch_all(tmp_path, 3, session=s, sources=src, today=date(2026, 8, 15), pause=0)
    assert [(d, new) for d, _p, new in res["nn"]] == [(date(2026, 7, 31), True), (date(2026, 6, 30), False)]
    assert fetch.archive_path(tmp_path, "nn", date(2026, 7, 31)).read_bytes() == b"%PDF iulie"   # de pe pagină
    assert fetch.archive_path(tmp_path, "nn", date(2026, 6, 30)).read_bytes() == b"%PDF iunie"   # din arhivă
    assert not fetch.archive_path(tmp_path, "nn", date(2026, 5, 31)).exists()                  # nu e PDF
    assert ("GET", NN_ZIP_URL) in s.calls
    s2 = FakeSession(routes)
    fetch.fetch_all(tmp_path, 3, session=s2, sources=src, today=date(2026, 8, 15), pause=0)
    assert ("GET", NN_ZIP_URL) not in s2.calls                                                  # din cache


def test_fetch_zip_refreshes_old_cache_and_keeps_it_when_download_fails(tmp_path):
    cache = tmp_path / "pdf" / "nn" / "_arhiva.zip"
    cache.parent.mkdir(parents=True)
    cache.write_bytes(_zip_bytes({"Iunie 2026.pdf": b"%PDF"}))
    old = cache.stat().st_mtime + (fetch.ZIP_MAX_AGE_DAYS + 1) * 86400
    s = FakeSession({NN_PAGE: FakeResp(200, ctype="text/html", text=NN_HTML),
                     NN_ZIP_URL: FakeResp(200, ctype="text/html", content=b"<html>eroare</html>")})
    assert fetch.fetch_zip(s, fetch.SOURCES["nn"], tmp_path, "nn", now=old) == cache          # cea veche rămâne
    assert ("GET", NN_ZIP_URL) in s.calls and zipfile_ok(cache)


def zipfile_ok(path):
    import zipfile
    return zipfile.is_zipfile(path)


def test_date_from_text_ddmmyyyy_azt():
    assert fetch.date_from_text("AZT_Viitorul_Tau_31072026.pdf") == date(2026, 7, 31)
    assert fetch.date_from_text("Raportare_semestriala_AZT_Viitorul_30062026.pdf") == date(2026, 6, 30)
    assert fetch.date_from_text("20260731_pII_inv_det.pdf") == date(2026, 7, 31)          # aaaallzz rămâne primul
    assert fetch.date_from_text("x_99999999.pdf") is None


def test_discover_monthly_pages_azt():
    base = "https://azt/rap?p_fond=VIITORUL&p_an={Y}&p_luna={mi}"
    jul = "https://azt/rap?p_fond=VIITORUL&p_an=2026&p_luna=7"
    jun = "https://azt/rap?p_fond=VIITORUL&p_an=2026&p_luna=6"
    html_jul = ('<a href="/f/VIITORUL/2026/7/31/vuan_vt_31.07.2026.pdf">vuan</a>'
                '<a href="/f/VIITORUL/2026/7/Numar_participanti/Structura_participanti_AZT_Viitorul_Tau_Iulie.pdf">x</a>'
                '<a href="/f/VIITORUL/2026/7/Structura_portofoliu_investitii/AZT_Viitorul_Tau_31072026.pdf">s</a>'
                '<a href="/f/VIITORUL/2026/trim_2/Rata_de_rentabilitate/Rata_rentab_30.06.2026.pdf">r</a>')
    s = FakeSession({jul: FakeResp(200, ctype="text/html", text=html_jul),
                     jun: FakeResp(200, ctype="text/html", text="<p>nimic</p>")})
    src = {"tip": "pagina_lunara", "pagina": base}
    assert fetch.discover("azt", src, [date(2026, 7, 31), date(2026, 6, 30)], s, pause=0) == [
        (date(2026, 7, 31), "https://azt/f/VIITORUL/2026/7/Structura_portofoliu_investitii/AZT_Viitorul_Tau_31072026.pdf")]
    assert fetch.SOURCES["azt"]["tip"] == "pagina_lunara"
    assert fetch.SOURCES["azt"]["pagina"].format(**fetch._fill(date(2026, 7, 31))).endswith("p_an=2026&p_luna=7")


def test_metropolitan_page_fallback_names():
    """Metropolitan, 16.11.2026: lunile cu „_31_05_2025” lipsesc din șabloane și se iau de pe pagină."""
    assert fetch.SOURCES["metropolitan"]["pagina"].startswith("https://www.metropolitanlife.ro/pensie-pilon-2/")
    html = ('<a href="https://www.metropolitanlife.ro/content/dam/metlifecom/ro/PDFs/'
            'situatia_detaliata_a_portofoliului_de_investitii_la_data_de_31_05_2025.pdf">'
            'Situatia detaliata a portofoliului de investitii la data de 31.05.2025</a>')
    links = fetch.pdf_links(html, fetch.SOURCES["metropolitan"]["pagina"])
    assert [d for _u, _t, d in links] == [date(2025, 5, 31)]
