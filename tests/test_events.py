"""Testele pentru analytics/events.py și data/entities.py (Radar evenimente). Titluri și prețuri sintetice."""
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from analytics import events as ev
from data import universes
from data.entities import COMMON_TICKER_WORDS, COMPETITOR_GROUPS, NAMES

NOW = datetime(2026, 11, 14, 20, 0)
SYMS = ["NVDA", "AMD", "AVGO", "INTC", "MSFT", "AAPL", "XOM", "CVX", "SNP.RO", "SNG.RO", "TLV.RO", "BRD.RO",
        "H2O.RO", "SNN.RO", "EL.RO", "AIR.PA", "BA", "SIE.DE", "ENR.DE", "T"]


def _matcher(symbols=SYMS):
    return ev.EntityMatcher(symbols, NAMES, COMMON_TICKER_WORDS)


def _item(title, hours_ago=1, source="Sursa A"):
    return {"title": title, "link": "https://example.com/x", "source": source,
            "date_obj": NOW - pd.Timedelta(hours=hours_ago)}


def test_normalize():
    assert ev.normalize("  Anchetă   DIICOT la Hidroelectrica ") == "ancheta diicot la hidroelectrica"


def test_classify_types_and_relief():
    assert ev.classify("Nvidia CEO faces espionage probe")[0] == "frauda_ancheta"
    assert ev.classify("Boeing recalls 737 jets after accident")[0] == "operational"
    assert ev.classify("Romgaz raportează profit record")[0] == "ghidaj_pozitiv"
    assert ev.classify("Apple cleared of fraud charges by court") == ["rezolvare"]
    assert ev.classify("Microsoft unveils new Surface") == []
    assert ev.classify("Hidroelectrica: anchetă a Consiliului Concurenței")[0] == "frauda_ancheta"
    # „SPY” (ETF) nu mai e „spionaj”; „în proces de” nu e litigiu
    assert ev.classify("SPY rallies as tech gains") == []
    assert ev.classify("Banca Transilvania este în proces de extindere") == []


def test_matcher_longest_case_and_ticker():
    m = _matcher()
    assert m.find("Siemens Energy shares jump; Siemens flat") == ["ENR.DE", "SIE.DE"]
    assert m.find("NVIDIA AND AMD SLIDE") == ["NVDA", "AMD"]
    assert m.find("intel agencies warn") == []                 # cu literă mică nu e compania
    assert m.find("AT&T fined by FCC") == ["T"]
    assert m.find("Hidroelectrica și Nuclearelectrica") == ["H2O.RO", "SNN.RO"]
    assert m.find("energia electrica se scumpește") == []      # nu e Electrica
    assert m.find("OMV Petrom investește") == ["SNP.RO"]


def test_detect_company_event_with_beneficiaries():
    items = [_item("Nvidia CEO Jensen Huang faces espionage probe", 2),
             _item("Nvidia shares fall amid espionage scandal", 1, "Sursa B"),
             _item("Microsoft unveils new Surface", 1)]
    evs = ev.detect_events(items, _matcher(), COMPETITOR_GROUPS, NOW, universes.market_of)
    assert len(evs) == 1
    e = evs[0]
    assert e["type"] == "frauda_ancheta" and e["affected"] == [("NVDA", -1)]
    assert set(e["beneficiaries"]) == {"AMD", "AVGO", "INTC"}
    assert len(e["headlines"]) == 2 and e["severity_total"] == 4   # 3 + două surse
    assert e["markets"] == {"us"}
    assert ev.touches(e, {"AMD", "NVDA", "TLV.RO"}) == [("AMD", "beneficiar"), ("NVDA", "afectat")]


def test_detect_primary_subject_and_no_share_shift():
    evs = ev.detect_events([_item("Nvidia faces espionage probe as AMD gains", 1)], _matcher(), COMPETITOR_GROUPS,
                           NOW, universes.market_of)
    # AMD e menționat, dar e concurent în același grup: trece la beneficiari, nu la „menționat”
    assert evs[0]["affected"] == [("NVDA", -1)] and "AMD" in evs[0]["beneficiaries"]
    # Subiectul = compania cea mai apropiată de cuvântul-cheie, nu prima numită (cazul real din 14.11)
    near = ev.detect_events([_item("AMD Just Crushed It as the Nvidia Scandal Grows", 1)], _matcher(),
                            COMPETITOR_GROUPS, NOW, universes.market_of)[0]
    assert near["affected"][0] == ("NVDA", -1) and "AMD" in near["beneficiaries"]
    # Mențiunile fără direcție nu dau alertă pe deținute
    amb = ev.detect_events([_item("Investor buys drilling firm with contracts for Romgaz and Petrom in acquisition", 1)],
                           _matcher(), COMPETITOR_GROUPS, NOW, universes.market_of)[0]
    assert amb["direction"] == 0 and ev.touches(amb, {"SNG.RO", "SNP.RO"}) == []
    juridic = ev.detect_events([_item("Apple fined in EU antitrust case", 1)], _matcher(), COMPETITOR_GROUPS, NOW,
                               universes.market_of)
    assert juridic[0]["type"] == "juridic" and juridic[0]["beneficiaries"] == {}


def test_mixed_timezones():
    from zoneinfo import ZoneInfo
    aware = {"title": "Boeing recalls jets", "link": "", "source": "S",
             "date_obj": NOW.replace(tzinfo=ZoneInfo("Europe/Bucharest")) - pd.Timedelta(hours=1)}
    out = ev.clean_items([aware, _item("Airbus fined in EU", 2)], NOW)
    assert [o["title"] for o in out] == ["Boeing recalls jets", "Airbus fined in EU"] and out[0]["date_obj"].tzinfo is None


def test_detect_sector_events_and_age_filter():
    items = [_item("Oil prices surge after OPEC decision", 3),
             _item("Guvernul introduce taxa pe banci", 5),
             _item("Oil prices surge after OPEC decision", 2),               # dublură
             _item("Brent falls on demand worries", 60)]                     # peste 48 de ore
    evs = ev.detect_events(items, _matcher(), COMPETITOR_GROUPS, NOW, universes.market_of)
    by = {e["type"]: e for e in evs}
    assert set(by) == {"petrol", "taxe_banci"}
    assert by["petrol"]["direction"] == 1 and ("XOM", 1) in by["petrol"]["affected"] and ("SNP.RO", 1) in by["petrol"]["affected"]
    assert len(by["petrol"]["headlines"]) == 1
    assert ("TLV.RO", -1) in by["taxe_banci"]["affected"] and by["taxe_banci"]["severity"] == 2
    assert ev.sector_direction("Oil prices mixed", "petrol") == 0
    assert ev.sector_direction("Brent hits record low", "petrol") == -1
    assert ev.sector_direction("Lower Oil Prices Could Lift Uber Stock", "petrol") == -1


def test_reference_session():
    idx = pd.bdate_range("2026-11-09", "2026-11-13")       # luni–vineri
    assert ev.reference_session(idx, datetime(2026, 11, 12, 15, 0), "us") == pd.Timestamp("2026-11-11")
    assert ev.reference_session(idx, datetime(2026, 11, 12, 23, 30), "us") == pd.Timestamp("2026-11-12")
    assert ev.reference_session(idx, datetime(2026, 11, 12, 18, 10), "bvb") == pd.Timestamp("2026-11-12")
    assert ev.reference_session(idx, datetime(2026, 11, 9, 9, 0), "eu") is None


def test_reaction_z_and_labels():
    idx = pd.bdate_range("2026-08-03", "2026-11-13")
    rng = np.random.default_rng(3)
    b = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, len(idx))), index=idx)
    noise = rng.normal(0, 0.01, len(idx))
    c = pd.Series(50 * np.cumprod(1 + (b.pct_change().fillna(0).to_numpy() + noise)), index=idx)
    c.iloc[-1] = c.iloc[-2] * (b.iloc[-1] / b.iloc[-2]) * 0.90      # −10% față de piață în ultima ședință
    vol = pd.Series(1000.0, index=idx)
    vol.iloc[-1] = 3500.0
    r = ev.reaction(c, b, vol, datetime(2026, 11, 13, 10, 0), "us", -1)
    assert r["ref"] == pd.Timestamp("2026-11-12") and r["sessions"] == 1
    # anormal = (0,9·g − 1) − (g − 1) = −0,1·g, cu g = randamentul benchmarkului în ultima ședință + 1
    assert r["abnormal"] == pytest.approx(-0.10 * b.iloc[-1] / b.iloc[-2])
    hist = pd.concat([c[c.index <= r["ref"]].pct_change(), b[b.index <= r["ref"]].pct_change()], axis=1).dropna().iloc[-60:]
    sigma = (hist.iloc[:, 0] - hist.iloc[:, 1]).std(ddof=1)
    assert r["z"] == pytest.approx(r["abnormal"] / sigma)
    assert r["rvol"] == pytest.approx(3.5) and r["label"] == "deja în preț"
    assert ev.reaction(c, b, vol, datetime(2026, 11, 13, 10, 0), "us", 1)["label"] == "contrazis de piață"
    none = ev.reaction(c, b, vol, datetime(2026, 11, 13, 23, 30), "us", -1)
    assert none["label"] == "fără ședință de la știre" and none["sessions"] == 0
    assert ev.reaction_label(0.5, -1) == "încă nereflectat" and ev.reaction_label(None, 1) == "N/A"
    assert ev.reaction_label(-2.5, 0) == "mișcare puternică"


def test_big_moves():
    idx = pd.bdate_range("2026-08-03", "2026-11-13")
    rng = np.random.default_rng(5)
    b = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, len(idx))), index=idx)
    calm = pd.Series(50 * np.cumprod(1 + b.pct_change().fillna(0).to_numpy() + rng.normal(0, 0.005, len(idx))), index=idx)
    jump = calm.copy()
    jump.iloc[-1] = jump.iloc[-2] * (b.iloc[-1] / b.iloc[-2]) * 1.08
    closes = pd.DataFrame({"CALM": calm, "JUMP": jump, "NEWS": jump})
    mv = ev.big_moves(closes, {s: b for s in closes.columns}, exclude={"NEWS"})
    assert [m["symbol"] for m in mv] == ["JUMP"] and mv[0]["z"] > 3


def test_journal_rows_parse_evaluate_summary():
    e = ev.detect_events([_item("Nvidia faces espionage probe as AMD gains", 1)], _matcher(), COMPETITOR_GROUPS, NOW,
                         universes.market_of)[0]
    reac = {(e["key"], "NVDA"): {"ref": pd.Timestamp("2026-11-13")}, (e["key"], "AVGO"): {"ref": pd.Timestamp("2026-11-13")}}
    moves = [{"symbol": "TLV.RO", "date": pd.Timestamp("2026-11-13"), "z": 3.4}]
    rows = ev.journal_rows([e], reac, {"us": "SPY", "eu": "EXSA.DE", "bvb": "TVBETETF.RO"}, universes.market_of, moves)
    roles = {(r[2], r[3]): r for r in rows}
    assert set(roles) == {("NVDA", "afectat"), ("AVGO", "beneficiar"), ("TLV.RO", "fara_stire")}   # AMD, INTC: fără ref
    assert roles[("NVDA", "afectat")][4] == -1 and roles[("AVGO", "beneficiar")][4] == 1
    assert roles[("TLV.RO", "fara_stire")][4] == -1 and roles[("TLV.RO", "fara_stire")][7] == "TVBETETF.RO"
    df, probs = ev.parse_journal([ev.JOURNAL_COLUMNS] + rows + [["x"] * 10])
    assert len(df) == 3 and probs
    idx = pd.bdate_range("2026-11-02", periods=40)
    c = pd.Series(np.linspace(100, 139, 40), index=idx)
    b = pd.Series(100.0, index=idx)
    r = ev.evaluate_event(c, b, idx[9], sessions=(5, 20))
    assert r[5] == pytest.approx(c.iloc[14] / c.iloc[9] - 1) and r[20] == pytest.approx(c.iloc[29] / c.iloc[9] - 1)
    assert ev.evaluate_event(c, b, idx[-3])[5] is None
    res = pd.DataFrame({"Rol": ["afectat", "afectat", "beneficiar", "afectat"], "Directie": [-1, -1, 1, 0],
                        "ab5": [-0.02, 0.01, 0.03, 0.5], "ab20": [-0.05, None, 0.01, 0.5]})
    s = ev.journal_summary(res).set_index("Rol")
    assert s.loc["afectat", "N 5"] == 2 and s.loc["afectat", "Nimerite 5"] == 0.5
    assert s.loc["afectat", "Medie 5"] == pytest.approx((0.02 - 0.01) / 2)
    assert s.loc["afectat", "N 20"] == 1 and s.loc["afectat", "small"]
    assert s.loc["beneficiar", "Nimerite 5"] == 1.0


def test_entities_cover_groups_and_universes():
    known = set(universes.symbols("us")) | set(universes.symbols("eu")) | set(universes.symbols("bvb"))
    for g, members in COMPETITOR_GROUPS.items():
        assert set(members) <= known, (g, set(members) - known)
    assert set(NAMES) <= known, set(NAMES) - known


def test_no_streamlit():
    import inspect
    from data import entities
    for mod in (ev, entities):
        src = inspect.getsource(mod)
        assert "import streamlit" not in src and "requests" not in src
