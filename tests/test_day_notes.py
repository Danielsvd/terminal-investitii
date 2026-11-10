"""„3 lucruri de știut azi” (analytics/day_notes.py): valori calculate de mână."""
from datetime import datetime

import pandas as pd

from analytics import day_notes as dn


def test_portofoliu_cu_pozitia_care_a_contat():
    tot = {"pnl_ron": -383.4, "pct": -0.0115, "top": ["AG"], "bottom": ["TLV.RO", "SNG.RO"]}
    rows = {"TLV.RO": {"Contribution": -0.0053}, "AG": {"Contribution": 0.0005}}
    n = dn.portfolio_note(tot, rows)
    assert n["text"] == "Portofoliul (acțiuni și ETF): -383 lei (-1,15%) în ultima ședință; cel mai mult a contat TLV (-0,53 pp)."
    tot_up = {"pnl_ron": 120.0, "pct": 0.004, "top": ["AG"], "bottom": []}
    assert "cel mai mult a contat AG (+0,05 pp)" in dn.portfolio_note(tot_up, rows)["text"]
    assert dn.portfolio_note({"pnl_ron": None}) is None and dn.portfolio_note(None) is None


def test_piata_indicele_cu_cea_mai_mare_sigma():
    d = pd.Timestamp("2026-11-05")
    moves = [("BET", {"pct": -0.012, "sigma": -1.1, "date": d}),
             ("STOXX 600", {"pct": -0.004, "sigma": -0.5, "date": d}),
             ("S&P 500", {"pct": 0.011, "sigma": 2.3, "date": d})]
    n = dn.market_note(moves)
    assert n["text"] == ("Piața: cea mai mare mișcare la S&P 500 (05.11), +1,10% (+2,3σ, mișcare neobișnuită față de "
                         "ultimele 60 de ședințe); BET -1,20%, STOXX 600 -0,40%.")
    # fără σ: cea mai mare variație absolută
    n2 = dn.market_note([("BET", {"pct": -0.02, "sigma": None, "date": None}), ("S&P 500", {"pct": 0.01, "sigma": None})])
    assert n2["text"].startswith("Piața: cea mai mare mișcare la BET, -2,00%;")
    assert dn.market_note([("BET", {"pct": None})]) is None


def test_ordinea_regulilor_si_completarea():
    p = {"tag": "portofoliu", "text": "P"}
    m = {"tag": "piață", "text": "M"}
    ev = dn.events_note([{"in_days": 1, "label": "Piața muncii (NFP, șomaj)"}])
    assert ev["text"] == "Calendar SUA: mâine Piața muncii (NFP, șomaj)."
    al = dn.alerts_note(["DFNC.DE", "X"], {"DFNC.DE": {"Pct": 0.041, "Sigma": 2.34}, "X": {"Pct": None, "Sigma": None}})
    assert al["text"] == "Mișcări neobișnuite (peste 2σ) la pozițiile tale: DFNC.DE +4,10% (+2,3σ)."
    sec = dn.sector_note([{"name": "Energie", "etf": "XLE", "pct": -0.02, "vs_index_pp": -2.5},
                          {"name": "Tehnologie", "etf": "XLK", "pct": 0.02, "vs_index_pp": 1.5}])
    assert sec["text"] == "Sectorul Energie (XLE) -2,00%, -2,50 pp față de S&P 500."
    assert dn.sector_note([{"name": "A", "etf": "XLA", "pct": 0.01, "vs_index_pp": 0.4}]) is None   # sub 1 pp
    nw = dn.news_note([{"symbol": "TLV.RO", "title": "Profit record", "source": "ZF", "time": datetime(2026, 11, 5, 9, 0)},
                       {"symbol": "AAPL", "title": "x", "source": "Y", "time": datetime(2026, 11, 5, 8, 0)}])
    assert nw["text"] == "Știre despre TLV: „Profit record” (ZF, 05.11 09:00); încă 1 în tabul „Știri”."

    # 1 și 2 fixe, al treilea = primul disponibil (calendarul)
    assert [x["text"] for x in dn.three_things(p, m, ev, al, sec, nw)] == ["P", "M", ev["text"]]
    # fără calendar: alertele
    assert dn.three_things(p, m, None, al, sec, nw)[2] is al
    # fără portofoliu: piața + primele două din rest
    assert dn.three_things(None, m, None, al, sec, nw) == [m, al, sec]
    # fără nimic: listă goală, niciun text inventat
    assert dn.three_things() == []
