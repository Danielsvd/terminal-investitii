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
    assert run.report_problems(no_code, date(2026, 7, 31), "FP2-54") == [] and no_code.fund_code == "FP2-54"
    no_code2 = _rep(REP_TEXT.replace("Cod fond de pensii: FP2-54\n", ""))
    assert any("data din raport" in p for p in run.report_problems(no_code2, date(2026, 6, 30), "FP2-54"))
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
