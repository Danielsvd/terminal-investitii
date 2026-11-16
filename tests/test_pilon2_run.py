"""Jobul Pilon II (tools/pilon2/run.py) și foile lui (data/pilon2.py). Fără rețea; valori sintetice."""
from datetime import date

import pandas as pd
import pytest

from data.pilon2 import HOLDINGS_COLUMNS, MAP_COLUMNS, parse_holdings, parse_isin_map
from tools.pilon2 import parse, run
from tools.pilon2.parse import parse_lines


def _rep(lines):
    return parse_lines(lines.splitlines())


REP_TEXT = """Cod fond de pensii: FP2-54
la data de: 31.07.2026
2. Valori mobiliare tranzactionate, din care: 3,500.00 35.000%
d) Actiuni 3,500.00 35.000%
Banca Transilvania SA ROTLVAACNOR1 2,450.00 24.500%
OMV Petrom SA ROSNPPACNOR9 1,000.00 10.000%
OMV Petrom SA ROSNPPACNOR9 50.00 0.500%
"""
REP = _rep(REP_TEXT)


def test_derive_quantity():
    assert run.derive_quantity(2450.0, 24.5) == (100, 0.0)
    assert run.derive_quantity(1000.0, 0.81) == (1235, -0.4321)   # 1234,5679 → abatere mare = de verificat
    assert run.derive_quantity(None, 24.5) == (None, None)
    assert run.derive_quantity(100.0, None) == (None, None)
    assert run.derive_quantity(100.0, 0.0) == (None, None)


def test_equity_by_isin_sums_duplicates():
    eq = run.equity_by_isin(REP)
    assert eq["ROSNPPACNOR9"]["Valoare_RON"] == 1050.0 and eq["ROSNPPACNOR9"]["Pondere"] == pytest.approx(10.5)


def test_report_problems_date_and_code_mismatch():
    assert run.report_problems(REP, date(2026, 7, 31), "FP2-54") == []
    probs = run.report_problems(REP, date(2026, 6, 30), "FP2-31")
    assert any("data din raport" in p for p in probs) and any("alt fond" in p for p in probs)
    # raport fără cod (NN din arhivă, Aripi 2026): primește codul sursei; data se verifică în continuare
    no_code = _rep(REP_TEXT.replace("Cod fond de pensii: FP2-54\n", ""))
    assert no_code.fund_code is None
    assert run.report_problems(no_code, date(2026, 7, 31), "FP2-54", code_missing_ok=True) == []
    assert no_code.fund_code == "FP2-54"
    no_code2 = _rep(REP_TEXT.replace("Cod fond de pensii: FP2-54\n", ""))
    assert any("data din raport" in p for p in run.report_problems(no_code2, date(2026, 6, 30), "FP2-54", True))
    # fără marcajul sursei (NN, 16.11.2026: arhiva .zip avea alt fond, fără cod) → respins
    no_code3 = _rep(REP_TEXT.replace("Cod fond de pensii: FP2-54\n", ""))
    assert any("codul fondului" in p for p in run.report_problems(no_code3, date(2026, 7, 31), "FP2-54"))
    assert no_code3.fund_code is None
    assert any("codul fondului" in p for p in run.report_problems(
        _rep(REP_TEXT.replace("Cod fond de pensii: FP2-54\n", "")), date(2026, 7, 31), None))


def test_choose_symbol_and_map_row():
    quotes = [{"symbol": "TLV.F", "quoteType": "EQUITY"}, {"symbol": "TLV.RO", "quoteType": "EQUITY"}]
    assert run.choose_symbol(quotes) == "TLV.RO"
    assert run.choose_symbol([{"symbol": "SAP.DE", "quoteType": "EQUITY"}]) == "SAP.DE"
    assert run.choose_symbol([{"symbol": "X", "quoteType": "MUTUALFUND"}]) == ""
    assert run.choose_symbol([]) == ""
    assert run.map_row("ROTLVAACNOR1", "TLV.RO", "Banca Transilvania SA") == [
        "ROTLVAACNOR1", "TLV.RO", "Banca Transilvania SA", "BVB", "yahoo"]
    assert run.map_row("XX0000000000", "", "Necunoscut")[3:] == ["", "negasit"]


def _close():
    idx = pd.to_datetime(["2026-07-29", "2026-07-30", "2026-08-03"])
    return pd.DataFrame({"TLV.RO": [24.0, 24.5, 25.0], "SNP.RO": [float("nan")] * 3}, index=idx)


def test_price_on_or_before_uses_last_session_within_lag():
    close = _close()
    assert run.price_on_or_before(close, "TLV.RO", date(2026, 7, 31)) == (24.5, date(2026, 7, 30))
    assert run.price_on_or_before(close, "SNP.RO", date(2026, 7, 31)) == (None, None)      # fără preț
    assert run.price_on_or_before(close, "TLV.RO", date(2026, 7, 20)) == (None, None)      # prea vechi / înainte
    assert run.price_on_or_before(close, "H2O.RO", date(2026, 7, 31)) == (None, None)
    assert run.price_on_or_before(None, "TLV.RO", date(2026, 7, 31)) == (None, None)


def test_build_rows_derives_only_bvb_with_price():
    isin_map = {"ROTLVAACNOR1": {"Symbol": "TLV.RO"}, "ROSNPPACNOR9": {"Symbol": "SNP.RO"}}
    rows = run.build_rows("Vital", REP, isin_map, _close(), None)
    assert len(rows[0]) == len(HOLDINGS_COLUMNS)
    by = {r[2]: dict(zip(HOLDINGS_COLUMNS, r)) for r in rows}
    tlv = by["ROTLVAACNOR1"]
    assert (tlv["Date"], tlv["Fond"], tlv["Symbol"], tlv["Clasa"]) == ("2026-07-31", "Vital", "TLV.RO", "actiuni")
    assert (tlv["Valoare_RON"], tlv["Cantitate_derivata"], tlv["Pret_folosit"], tlv["Abatere_intreg"]) == (2450.0, 100, 24.5, 0.0)
    snp = by["ROSNPPACNOR9"]
    assert snp["Cantitate_derivata"] == "" and snp["Pret_folosit"] == ""          # lipsă rămâne goală, nu 0
    assert tlv["Data_publicare"] == ""
    assert run.build_rows("Vital", REP, {}, _close(), date(2026, 11, 14))[0][-1] == "2026-11-14"


def test_existing_keys():
    vals = [HOLDINGS_COLUMNS, ["2026-07-31", "Vital", "rotlvaacnor1"] + [""] * 9, ["scurt"]]
    assert run.existing_keys(vals) == {("2026-07-31", "Vital", "ROTLVAACNOR1")}
    assert run.existing_keys([]) == set()


def test_parse_isin_map_and_holdings_roundtrip():
    m = parse_isin_map([MAP_COLUMNS, ["rotlvaacnor1", "tlv.ro", "Banca Transilvania", "BVB", "yahoo"], ["", "X"]])
    assert m == {"ROTLVAACNOR1": {"Symbol": "TLV.RO", "Denumire": "Banca Transilvania", "Piata": "BVB", "Sursa": "yahoo"}}
    rows = run.build_rows("Vital", REP, {"ROTLVAACNOR1": {"Symbol": "TLV.RO"}}, _close(), None)
    # foaia întoarce text în format românesc la get_all_values
    as_text = [[str(c).replace(".", ",") if isinstance(c, float) else str(c) for c in r] for r in rows]
    df, problems = parse_holdings([HOLDINGS_COLUMNS] + as_text + [["fără dată", "Vital"]])
    assert len(problems) == 1 and len(df) == 2
    tlv = df[df["ISIN"] == "ROTLVAACNOR1"].iloc[0]
    assert tlv["Valoare_RON"] == 2450.0 and tlv["Cantitate_derivata"] == 100 and tlv["Pret_folosit"] == 24.5
    snp = df[df["ISIN"] == "ROSNPPACNOR9"].iloc[0]
    assert pd.isna(snp["Cantitate_derivata"]) and pd.isna(snp["Data_publicare"])
    assert parse_holdings([["Date", "X"]])[1]                                    # antet greșit → problemă


class FakeWS:
    def __init__(self, values):
        self.values, self.appended = values, []

    def get_all_values(self):
        return self.values

    def append_rows(self, rows, value_input_option=None):
        self.appended.extend(rows)
        self.values = self.values + rows


class FakeSheet:
    def __init__(self):
        self.ws = {"ISIN_map": FakeWS([MAP_COLUMNS, ["ROTLVAACNOR1", "TLV.RO", "Banca Transilvania", "BVB", "manual"]]),
                   "Pilon2_detineri": FakeWS([HOLDINGS_COLUMNS])}


def test_main_end_to_end_with_fakes(tmp_path, monkeypatch, capsys):
    good = tmp_path / "good.pdf"
    bad = tmp_path / "bad.pdf"
    good.write_bytes(b"%PDF")
    bad.write_bytes(b"%PDF")
    reports = {str(good): REP, str(bad): _rep("Cod fond de pensii: FP2-54\nla data de: 30.06.2026\n")}
    monkeypatch.setattr(run.fetch, "fetch_all", lambda *a, **k: {
        "vital": [(date(2026, 7, 31), good, True), (date(2026, 6, 30), bad, False)]})
    monkeypatch.setattr(run.parse, "parse_pdf", lambda path: reports[str(path)])
    sheet = FakeSheet()
    monkeypatch.setattr(run, "open_spreadsheet", lambda _s: sheet)
    monkeypatch.setattr(run, "ensure_worksheet", lambda sh, title, header: sh.ws[title])
    monkeypatch.setattr(run, "yahoo_symbol", lambda isin: {"ROSNPPACNOR9": "SNP.RO"}.get(isin, ""))
    monkeypatch.setattr(run, "download_closes", lambda syms, dates: _close())
    monkeypatch.setattr(run, "SEARCH_PAUSE_S", 0)

    code = run.main(["--data-dir", str(tmp_path), "--fond", "vital"])
    out = capsys.readouterr().out
    assert code == 1 and "RESPINS" in out                     # raportul din iunie n-are acțiuni
    assert sheet.ws["ISIN_map"].appended == [["ROSNPPACNOR9", "SNP.RO", "OMV Petrom SA", "BVB", "yahoo"]]
    rows = sheet.ws["Pilon2_detineri"].appended
    assert [r[2] for r in rows] == ["ROSNPPACNOR9", "ROTLVAACNOR1"]
    assert all(r[-1] == date.today().isoformat() for r in rows)        # raport nou, fără nr. de ieșire
    # a doua rulare nu dublează nimic
    run.main(["--data-dir", str(tmp_path), "--fond", "vital"])
    assert len(sheet.ws["Pilon2_detineri"].appended) == 2


def test_main_proba_writes_nothing(tmp_path, monkeypatch):
    pdf = tmp_path / "r.pdf"
    pdf.write_bytes(b"%PDF")
    monkeypatch.setattr(run.fetch, "fetch_all", lambda *a, **k: {"vital": [(date(2026, 7, 31), pdf, True)]})
    monkeypatch.setattr(run.parse, "parse_pdf", lambda path: REP)
    monkeypatch.setattr(run, "open_spreadsheet", lambda _s: pytest.fail("--proba nu deschide foile"))
    assert run.main(["--data-dir", str(tmp_path), "--proba"]) == 0
    assert parse.EQUITY == "actiuni"


def test_main_same_report_twice_in_one_run_written_once(tmp_path, monkeypatch):
    """NN, 16.11.2026: același raport ajuns de două ori în aceeași rulare → rândurile se scriu o dată."""
    a, b = tmp_path / "a.pdf", tmp_path / "b.pdf"
    a.write_bytes(b"%PDF")
    b.write_bytes(b"%PDF")
    monkeypatch.setattr(run.fetch, "fetch_all", lambda *x, **k: {
        "vital": [(date(2026, 7, 31), a, True), (date(2026, 7, 31), b, True)]})
    monkeypatch.setattr(run.parse, "parse_pdf", lambda path: REP)
    sheet = FakeSheet()
    monkeypatch.setattr(run, "open_spreadsheet", lambda _s: sheet)
    monkeypatch.setattr(run, "ensure_worksheet", lambda sh, title, header: sh.ws[title])
    monkeypatch.setattr(run, "yahoo_symbol", lambda isin: "")
    monkeypatch.setattr(run, "download_closes", lambda syms, dates: _close())
    monkeypatch.setattr(run, "SEARCH_PAUSE_S", 0)
    assert run.main(["--data-dir", str(tmp_path), "--fond", "vital"]) == 0
    rows = sheet.ws["Pilon2_detineri"].appended
    assert len(rows) == len({(r[0], r[1], r[2]) for r in rows}) == 2


def test_price_rounded_to_bvb_ticks_gives_integer_quantities():
    """16.11.2026, foaia reală: TLV.RO la 31.07.2026, prețul Yahoo 36.900002 → cantități cu abateri de până la 0,46.
    Valorile (rotunjite la leu) din rapoartele AZT, Aripi, BCR, Metropolitan, NN, Vital se împart exact la 36,90."""
    idx = pd.to_datetime(["2026-07-30", "2026-07-31"])
    close = pd.DataFrame({"TLV.RO": [36.5, 36.900002]}, index=idx)
    price, day = run.price_on_or_before(close, "TLV.RO", date(2026, 7, 31))
    assert price == 36.9 and day == date(2026, 7, 31)
    expected = {2291057753.0: 62088286, 1125317935.0: 30496421, 662525847.0: 17954630,
                1702237738.0: 46131104, 4050795728.0: 109777662, 1019671132.0: 27633364}
    for value, qty in expected.items():
        q, dev = run.derive_quantity(value, price)
        assert q == qty and abs(dev) < 0.02
    assert run.price_on_or_before(pd.DataFrame({"X.RO": [0.86990003]}, index=idx[1:]), "X.RO", date(2026, 7, 31))[0] == 0.8699


def test_summary_message_and_telegram_config(tmp_path, capsys):
    msg = run.summary_message(date(2026, 12, 5), [("vital", date(2026, 8, 31)), ("bt", date(2026, 8, 31))],
                              ["aripi 31.08.2026: nicio acțiune găsită"], 40, 1, 2, ["XS0000000000"])
    assert msg.splitlines() == [
        "Pilon II — rulare 05.12.2026",
        "Rapoarte noi: 2 — vital 31.08.2026, bt 31.08.2026",
        "RESPINSE: 1",
        "  aripi 31.08.2026: nicio acțiune găsită",
        "Pilon2_detineri: 40 rânduri noi (1 acțiuni BVB fără preț)",
        "ISIN_map: 2 ISIN-uri noi, fără simbol: XS0000000000"]
    assert "niciunul" in run.summary_message(date(2026, 12, 5), [], [], 0, 0, 0, [])
    assert len(run.summary_message(date(2026, 12, 5), [], [f"r{i} " + "x" * 80 for i in range(100)], 0, 0, 0, [])) \
        <= run.TELEGRAM_MAX
    assert run.load_telegram(tmp_path) is None                                       # fără fișier: nicio notificare
    (tmp_path / run.TELEGRAM_FILE).write_text('{"token": "T"}')
    assert run.load_telegram(tmp_path) is None
    (tmp_path / run.TELEGRAM_FILE).write_text('{"token": "123:ABC", "chat_id": 42}')
    assert run.load_telegram(tmp_path) == {"token": "123:ABC", "chat_id": "42"}


def test_send_telegram_never_raises_and_never_prints_token(capsys):
    cfg = {"token": "123:SECRET", "chat_id": "42"}
    calls = []

    class R:
        def __init__(self, code):
            self.status_code = code

    assert run.send_telegram(cfg, "x", post=lambda url, **k: calls.append((url, k)) or R(200)) is True
    assert calls[0][0] == "https://api.telegram.org/bot123:SECRET/sendMessage" and calls[0][1]["data"]["chat_id"] == "42"
    assert run.send_telegram(cfg, "x", post=lambda url, **k: R(401)) is False

    def boom(url, **k):
        raise ConnectionError(f"failed for {url}")
    assert run.send_telegram(cfg, "x", post=boom) is False
    assert run.send_telegram(None, "x", post=boom) is False
    assert "SECRET" not in capsys.readouterr().err


def test_main_sends_summary_but_not_on_proba(tmp_path, monkeypatch):
    pdf = tmp_path / "r.pdf"
    pdf.write_bytes(b"%PDF")
    (tmp_path / run.TELEGRAM_FILE).write_text('{"token": "1:A", "chat_id": "7"}')
    monkeypatch.setattr(run.fetch, "fetch_all", lambda *a, **k: {"vital": [(date(2026, 7, 31), pdf, True)]})
    monkeypatch.setattr(run.parse, "parse_pdf", lambda path: REP)
    sheet = FakeSheet()
    monkeypatch.setattr(run, "open_spreadsheet", lambda _s: sheet)
    monkeypatch.setattr(run, "ensure_worksheet", lambda sh, title, header: sh.ws[title])
    monkeypatch.setattr(run, "yahoo_symbol", lambda isin: "")
    monkeypatch.setattr(run, "download_closes", lambda syms, dates: _close())
    monkeypatch.setattr(run, "SEARCH_PAUSE_S", 0)
    sent = []
    monkeypatch.setattr(run, "send_telegram", lambda cfg, text, post=None: sent.append((cfg, text)) or True)
    assert run.main(["--data-dir", str(tmp_path), "--proba"]) == 0
    assert all(cfg is None for cfg, _t in sent)                                       # la probă: fără Telegram
    sent.clear()
    assert run.main(["--data-dir", str(tmp_path), "--fond", "vital"]) == 0
    assert sent and sent[-1][0] == {"token": "1:A", "chat_id": "7"}
    assert "Rapoarte noi: 1 — vital 31.07.2026" in sent[-1][1] and "Pilon2_detineri: 2 rânduri noi" in sent[-1][1]
    monkeypatch.setattr(run.fetch, "fetch_all", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("rețea")))
    sent.clear()
    with pytest.raises(RuntimeError):
        run.main(["--data-dir", str(tmp_path)])
    assert "EROARE RuntimeError: rețea" in sent[-1][1]
