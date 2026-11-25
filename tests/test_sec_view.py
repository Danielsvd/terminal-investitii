"""analytics/sec_view.py: tabelele paginii Fonduri SUA. Valori sintetice."""
import numpy as np
import pandas as pd

from analytics import sec_13f, sec_view as v
from tests.test_sec_13f import CURR, PREV


def test_name_matches():
    assert v.name_matches("Berkshire Hathaway", "BERKSHIRE HATHAWAY INC")
    assert v.name_matches("Icahn Carl C", "ICAHN CARL C")
    assert v.name_matches("Discovery Capital Management", "DISCOVERY CAPITAL MANAGEMENT, LLC / CT")
    assert not v.name_matches("Third Point", "SOME OTHER FUND LLC")
    assert not v.name_matches("Capital Management", "CAPITAL MANAGEMENT LLC")   # doar cuvinte generice


def test_delay_days():
    assert v.delay_days("2026-09-30", "2026-11-14") == 45
    assert v.delay_days(None, "2026-11-14") is None


def test_overview_row():
    mgr = {"Fond": "Fond Test", "Administrator": "X", "Stil": "Value", "Clasa": "Concentrat"}
    curr = {"period": "2026-09-30", "filing_date": "2026-11-14", "last_filing_date": "2026-11-20"}
    r = v.overview_row(mgr, "FOND TEST LLC", curr, PREV, CURR, {"000001AA1": "AAA"})
    assert r["Nume_ok"] and r["Decalaj_zile"] == 45 and r["Depus"] == "2026-11-20"
    assert r["Pozitii"] == 3 and r["Valoare_USD"] == 1500 and r["Top"] == "AAA"
    assert r["Top_pondere"] == 1000 / 1500 and r["Clasa_auto"] == "Concentrat"
    empty = v.overview_row(mgr, None, None, None, None, {})
    assert empty["Nume_ok"] is None and np.isnan(empty["Pozitii"])


def test_with_symbols_missing_is_none():
    out = v.with_symbols(CURR, {"000001AA1": "AAA"})
    assert out["Symbol"].tolist() == ["AAA", None, None]


def test_panels_and_split_candidates():
    mg = pd.DataFrame({"CIK": ["1", "2"], "Fond": ["F1", "F2"], "Clasa": ["Concentrat", "Larg"]})
    h = v.holdings_panel({"1": CURR, "2": PREV, "9": CURR}, mg)
    assert set(h["CIK"]) == {"1", "2"} and h.loc[h.CIK == "2", "Clasa"].iloc[0] == "Larg"
    m = v.moves_panel({"1": sec_13f.quarter_moves(PREV, CURR)})      # AAA 50→100 (+100%), fără split
    assert v.split_check_cusips(m) == ["000001AA1"]
    assert v.moves_panel({}).empty and v.split_check_cusips(None) == []


def test_cusip6_for_symbol_merges_classes():
    cmap = {"02079K305": "GOOGL", "02079K107": "GOOG", "037833100": "AAPL"}
    assert v.cusip6_for_symbol("googl", cmap) == ["02079K"]
    assert v.cusip6_for_symbol("MSFT", cmap) == []


def test_keys_and_symbols_from_holdings():
    h = pd.DataFrame({"CUSIP": ["02079K305", "02079K107", "464286400", "464286822"],
                      "CUSIP6": ["02079K", "02079K", "464286400", "464286822"]})
    cmap = {"02079K305": "GOOGL", "02079K107": "GOOG", "464286400": "EWZ", "464286822": "EWW"}
    assert v.cusip6_for_symbol("EWZ", cmap, h) == ["464286400"]
    assert v.cusip6_for_symbol("GOOG", cmap, h) == ["02079K"]
    assert v.symbols_by_key(cmap, h) == {"02079K": "GOOG/GOOGL", "464286400": "EWZ", "464286822": "EWW"}


def test_stale_periods():
    ps = ["2026-09-30"] * 5 + ["2026-03-31", "2023-12-31", None, "2026-06-30"]
    assert v.stale_periods(ps) == {"2026-03-31", "2023-12-31", "2026-06-30"}   # și un trimestru în urmă = vechi
    assert v.stale_periods([None]) == set()
