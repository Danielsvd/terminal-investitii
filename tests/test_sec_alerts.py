"""Jobul SEC (Faza 2c): funcțiile pure din analytics/sec_alerts.py și o rulare completă a tools/sec/run.py cu
foaie, EDGAR și Telegram false. Doar date sintetice."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from analytics import insiders, sec_13f, sec_alerts as sa, sec_view
from tests.test_insiders import _form4

LEDGER_HEADER = ["Date", "Symbol", "Class", "Type", "Quantity", "Price", "Currency", "Fees", "Amount", "Broker", "ID", "Notes"]
LEDGER = [LEDGER_HEADER,
          ["2026-01-10 10:00:00", "AAPL", "Actiuni SUA", "BUY", "10", "200", "USD", "1", "-2001", "XTB", "x1", ""],
          ["2026-01-11 10:00:00", "MSFT", "Actiuni SUA", "BUY", "5", "400", "USD", "1", "-2001", "XTB", "x2", ""],
          ["2026-02-11 10:00:00", "MSFT", "Actiuni SUA", "SELL", "5", "410", "USD", "1", "2049", "XTB", "x3", ""],
          ["2026-01-12 10:00:00", "TLV.RO", "Actiuni RO", "BUY", "100", "30", "RON", "1", "-3001", "TV", "t1", ""]]
WATCH = [["Symbol", "TargetPrice", "Notes"], ["NVDA", 120, ""], ["SAP.DE", 150, ""], ["aapl", "", ""], ["BTC-USD", "", ""]]


# --------------------------------------------------------------------------- funcții pure

def test_watched_symbols_us_only():
    assert sa.watched_symbols(["AAPL", "TLV.RO"], ["nvda", "SAP.DE", "BTC-USD", "^GSPC", "AAPL"]) == ["AAPL", "NVDA"]


def test_held_and_watchlist_symbols():
    from tools.sec import run
    assert run.held_us_symbols(LEDGER) == ["AAPL"]                       # MSFT vândut, TLV.RO în RON
    assert sa.watched_symbols(run.held_us_symbols(LEDGER), run.watchlist_symbols(WATCH)) == ["AAPL", "NVDA"]


def test_known_accessions():
    vals = [sa.JOURNAL_COLUMNS, ["2026-11-02"] + [""] * 11 + ["A-1"], ["x"] * 12 + [""]]
    assert sa.known_accessions(vals) == {"A-1"}
    assert sa.known_accessions([]) == set() and sa.known_accessions([["Alt"]]) == set()


def _tx():
    docs = [
        insiders.parse_form4(_form4([("10", "CEO X", "Chief Executive Officer", False)],
                                    [("2026-11-20", "P", 300, 100.0, "A", 5300, ()),
                                     ("2026-11-20", "P", 250, 100.0, "A", 5550, ())]),
                             accession="ACC-1", filing_date="2026-11-24"),          # 55.000 $ în total → alertă
        insiders.parse_form4(_form4([("11", "Dir Y", None, True)], [("2026-11-21", "P", 100, 100.0, "A", 100, ())]),
                             accession="ACC-2", filing_date="2026-11-24"),          # 10.000 $ → sub prag
        insiders.parse_form4(_form4([("12", "Dir Z", None, True)], [("2026-11-21", "P", 9000, None, "A", 9000, ())]),
                             accession="ACC-3", filing_date="2026-11-24"),          # fără preț → nu devine 0
        insiders.parse_form4(_form4([("13", "CFO W", "CFO", False)], [("2026-11-22", "S", 1000, 300.0, "D", 0, ())]),
                             accession="ACC-4", filing_date="2026-11-24"),
    ]
    return insiders.transactions_frame(docs)


def test_p_buy_alerts_threshold_per_filing_and_insider():
    al = sa.p_buy_alerts(_tx(), "AAPL")
    assert len(al) == 1
    a = al[0]
    assert a["insider"] == "CEO X" and a["valoare"] == pytest.approx(55_000) and a["actiuni"] == 550
    assert a["accession"] == "ACC-1" and a["data"] == pd.Timestamp("2026-11-20")
    assert sa.p_buy_alerts(_tx(), "AAPL", min_usd=5_000)[1]["insider"] == "Dir Y"
    assert sa.p_buy_alerts(insiders.transactions_frame([]), "AAPL") == []


def test_journal_rows_empty_cells_not_zero():
    rows = sa.journal_rows(_tx(), "AAPL", "320193")
    assert len(rows) == 5 and all(len(r) == len(sa.JOURNAL_COLUMNS) for r in rows)
    by_acc = {r[-1]: r for r in rows}
    z = by_acc["ACC-3"]
    assert z[2] == "AAPL" and z[3] == "320193" and z[6] == "P" and z[9] == "" and z[10] == ""   # preț/valoare lipsă
    assert by_acc["ACC-4"][0] == "2026-11-22" and by_acc["ACC-4"][7] == "NU"


def _book(rows):
    df = pd.DataFrame(rows, columns=["CUSIP", "Emitent", "Actiuni", "Valoare_USD"])
    df["Clasa_titlu"], df["CUSIP6"] = "COM", df["CUSIP"].str[:6]
    df["Pondere"] = df["Valoare_USD"] / df["Valoare_USD"].sum()
    df = df.sort_values("Valoare_USD", ascending=False).reset_index(drop=True)
    df["Rang"] = range(1, len(df) + 1)
    return df[sec_13f.BOOK_COLUMNS]


def _panels():
    # F1 deschide AAPL; F2 închide AAPL; F2 trece GOOG→GOOGL (aceeași emitent: nu e ieșire)
    p1 = _book([("594918104", "MSFT", 10, 100)])
    c1 = _book([("037833100", "APPLE", 10, 900), ("594918104", "MSFT", 10, 100)])
    p2 = _book([("037833100", "APPLE", 100, 50), ("02079K107", "ALPHABET C", 5, 950)])
    c2 = _book([("02079K305", "ALPHABET A", 5, 975), ("000009ZZ9", "Z", 1, 25)])
    mg = pd.DataFrame({"CIK": ["1", "2"], "Fond": ["F1", "F2"], "Clasa": ["Concentrat", "Concentrat"]})
    hold = sec_view.holdings_panel({"1": c1, "2": c2}, mg)
    mov = sec_view.moves_panel({"1": sec_13f.quarter_moves(p1, c1), "2": sec_13f.quarter_moves(p2, c2)})
    cmap = {"037833100": "AAPL", "594918104": "MSFT", "02079K107": "GOOG", "02079K305": "GOOGL"}
    return hold, mov, cmap


def test_fund_move_events_and_dedup():
    hold, mov, cmap = _panels()
    ev = sa.fund_move_events(["AAPL", "GOOGL", "NVDA"], cmap, hold, mov, {"1": "F1", "2": "F2"},
                             {"1": "2026-09-30", "2": "2026-09-30"})
    assert [(e["symbol"], e["fond"], e["actiune"]) for e in ev] == [("AAPL", "F1", sec_13f.ACT_NEW),
                                                                     ("AAPL", "F2", sec_13f.ACT_CLOSED)]
    assert ev[0]["key"] == "1|2026-09-30|037833|Nouă"
    assert sa.new_events(ev, {ev[0]["key"]}) == [ev[1]]
    assert sa.new_events(ev + ev, set()) == ev                          # fără dubluri în aceeași rulare
    assert sa.fund_move_events(["AAPL"], cmap, hold, mov.iloc[0:0], {}, {}) == []


def test_alert_message():
    assert sa.alert_message(date(2026, 11, 27), [], [], 3, 0, []) is None          # nimic nou → fără mesaj
    al = sa.p_buy_alerts(_tx(), "AAPL")
    ev = [{"symbol": "AAPL", "fond": "F1", "actiune": sec_13f.ACT_NEW, "perioada": "2026-09-30", "key": "k"}]
    m = sa.alert_message(date(2026, 11, 27), al, ev, 3, 5, ["X: HTTP 500"])
    assert "CUMPĂRARE INSIDER AAPL: CEO X" in m and "550 acț." in m and "55.000 $" in m and "20.11.2026" in m
    assert "13F AAPL: F1 a DESCHIS poziția (trimestrul 2026-09-30)" in m and "Probleme: 1" in m
    s = sa.alert_message(date(2026, 11, 27), [], [], 3, 40, [], seed=True)
    assert "Rulare inițială" in s and "40 tranzacții" in s
    big = sa.alert_message(date(2026, 11, 27), al * 200, [], 3, 0, [])
    assert len(big) <= sa.TELEGRAM_MAX and big.endswith("(trunchiat)")


def test_state_roundtrip(tmp_path):
    from tools.sec import run
    p = tmp_path / "sec" / "state.json"
    assert run.load_state(p) is None
    run.save_state(p, {"b", "a", "a"})
    assert run.load_state(p) == {"seen_13f": ["a", "b"]}
    p.write_text("{stricat", encoding="utf-8")
    with pytest.raises(RuntimeError):
        run.load_state(p)


def test_load_form4_skip_known():
    from data import sec_insiders as si
    from tests.test_insiders import _FakeClient
    subs = {"filings": {"recent": {
        "accessionNumber": ["0001-26-000003", "0001-26-000002"], "form": ["4", "4"],
        "filingDate": ["2026-11-04", "2026-11-03"], "reportDate": ["", ""],
        "primaryDocument": ["xslF345X05/f3.xml", "xslF345X05/f2.xml"]}}}
    files = {"f3.xml": _form4([("10", "CEO X", "CEO", False)], [("2026-11-02", "P", 1, 1, "A", 1, ())])}
    c = _FakeClient(subs, files)
    docs, problems, _t = si.load_form4(c, "320193", today=date(2026, 11, 5), skip={"0001-26-000002"})
    assert [d["accession"] for d in docs] == ["0001-26-000003"] and problems == []
    assert not any("f2.xml" in u for u in c.urls)                       # cel cunoscut nu se mai descarcă


# --------------------------------------------------------------------------- rulare completă cu falsuri

class _WS:
    def __init__(self, values):
        self.values, self.appended = values, []

    def get_all_values(self, **_kw):
        return self.values

    def append_rows(self, rows, **_kw):
        self.appended += rows
        self.values = self.values + rows


class _SH:
    def __init__(self, sheets):
        self.sheets = sheets

    def worksheet(self, name):
        import gspread
        if name not in self.sheets:
            raise gspread.WorksheetNotFound(name)
        return self.sheets[name]


class _Client:
    def __init__(self, files):
        self.files = files

    def get_json(self, url):
        if url.endswith("company_tickers.json"):
            return {"0": {"ticker": "AAPL", "cik_str": 320193}, "1": {"ticker": "NVDA", "cik_str": 1045810}}
        return {"filings": {"recent": {
            "accessionNumber": ["ACC-1"] if "320193" in url else [], "form": ["4"] if "320193" in url else [],
            "filingDate": ["2026-11-24"] if "320193" in url else [], "reportDate": [""] if "320193" in url else [],
            "primaryDocument": ["xslF345X05/a.xml"] if "320193" in url else []}}}

    def get(self, url):
        return self.files[url.rsplit("/", 1)[-1]]


def _patch(monkeypatch, sheets, sent):
    from tools.sec import run
    hold, mov, cmap = _panels()
    books = {"1067983": None}
    monkeypatch.setattr(run, "open_spreadsheet", lambda _p: _SH(sheets))
    monkeypatch.setattr(run, "ensure_worksheet", lambda sh, t, h: sh.sheets.setdefault(t, _WS([h])))
    monkeypatch.setattr(run, "read_secrets", lambda _p: ("Test test@example.com", None))
    monkeypatch.setattr(run.sec_edgar, "EdgarClient", lambda ua: _Client(
        {"a.xml": _form4([("10", "CEO X", "Chief Executive Officer", False)],
                         [("2026-11-20", "P", 600, 100.0, "A", 600, ())])}))
    monkeypatch.setattr(run, "send_telegram", lambda cfg, text: sent.append(text) or True)
    monkeypatch.setattr(run, "load_telegram", lambda d: {"token": "t", "chat_id": "c"})

    def fake_fund_step(client, symbols, cusip_values, figi_key):
        ev = sa.fund_move_events(symbols, cmap, hold, mov, {"1": "F1", "2": "F2"}, {"1": "2026-09-30", "2": "2026-09-30"})
        return ev, []
    monkeypatch.setattr(run, "fund_step", fake_fund_step)
    return run, books


def test_run_seed_then_alerts(monkeypatch, tmp_path):
    sheets = {"Tranzactii": _WS(LEDGER), "watchlist": _WS(WATCH)}
    sent = []
    run, _ = _patch(monkeypatch, sheets, sent)
    args = ["--state-dir", str(tmp_path), "--telegram", str(tmp_path / "telegram.json")]
    assert run.main(args) == 0                                             # rulare inițială
    j = sheets[sa.JOURNAL_SHEET]
    assert len(j.appended) == 1 and j.appended[0][-1] == "ACC-1"
    assert len(sent) == 1 and "Rulare inițială" in sent[0] and "CUMPĂRARE" not in sent[0]
    # a doua rulare: ACC-1 e cunoscut, 13F deja văzut → nimic nou, niciun mesaj
    assert run.main(args) == 0
    assert len(j.appended) == 1 and len(sent) == 1
    # un raport nou cu cumpărare de 60.000 $ → alertă
    j.values = [sa.JOURNAL_COLUMNS]                                         # simulăm un accession necunoscut
    assert run.main(args) == 0
    assert len(sent) == 2 and "CUMPĂRARE INSIDER AAPL: CEO X" in sent[1] and "60.000 $" in sent[1]
    assert "13F" not in sent[1]                                             # intrarea F1 a fost înregistrată la seed


def test_run_proba_writes_nothing(monkeypatch, tmp_path, capsys):
    sheets = {"Tranzactii": _WS(LEDGER), "watchlist": _WS(WATCH)}
    sent = []
    run, _ = _patch(monkeypatch, sheets, sent)
    assert run.main(["--state-dir", str(tmp_path), "--telegram", str(tmp_path / "telegram.json"), "--proba"]) == 0
    assert sa.JOURNAL_SHEET not in sheets and sent == [] and not (tmp_path / "state.json").exists()
    out = capsys.readouterr().out
    assert "Probă: 1 rânduri noi" in out and "Rulare inițială" in out


def test_usd_format_ro():
    # 213.437×142,9122 + 341.423×143,6841 + 295.240×144,6068 ≈ 122.254.xxx $ (Form 4 PLTR, 20.11.2026)
    assert sa._usd(122_254_123.0) == "122,25 mil. $"
    assert sa._usd(1_234_567_890.0) == "1.234,57 mil. $"
    assert sa._usd(320_900.0) == "320.900 $"
    assert sa._usd(None) == "N/A"
