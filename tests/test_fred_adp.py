"""ADP din FRED: seria nouă ADPMNUSNERSA (persoane) adusă la mii, ca PAYEMS; regula de divergență NFP–ADP."""
import pandas as pd

from analytics.macro import FRED_ADP, fred_in_thousands, interpret_fred


def test_adp_code_is_new_series():
    assert FRED_ADP == "ADPMNUSNERSA"


def test_adp_persons_to_thousands():
    # Valori reale FRED: sep. 2026 = 132.885.000, oct. 2026 = 132.971.000 → +86.000 persoane = +86,0K
    s = fred_in_thousands(FRED_ADP, pd.Series([132_885_000.0, 132_971_000.0]))
    assert s.tolist() == [132_885.0, 132_971.0]
    assert s.iloc[-1] - s.iloc[-2] == 86.0


def test_other_series_unchanged():
    s = fred_in_thousands("PAYEMS", pd.Series([159_000.0, 159_250.0]))
    assert s.tolist() == [159_000.0, 159_250.0]


def _df(rows):
    return pd.DataFrame([{"_simbol": c, "Valoare Curentă": v, "_change_raw": ch, "_yoy": None} for c, v, ch in rows])


def test_divergence_rule_uses_new_adp():
    # NFP +250K, ADP +44K (mii) → divergență privat-stat
    out = interpret_fred(_df([("PAYEMS", 159_250.0, 250.0), (FRED_ADP, 132_763.0, 44.0)]))
    assert any("DIVERGENȚĂ PRIVAT-STAT" in b for b in out)
    assert not any("ADP" in b and "Fără date" in b for b in out)


def test_no_divergence_when_adp_strong():
    # ADP +150K ≥ 100K → fără divergență
    out = interpret_fred(_df([("PAYEMS", 159_250.0, 250.0), (FRED_ADP, 132_850.0, 150.0)]))
    assert not any("DIVERGENȚĂ PRIVAT-STAT" in b for b in out)


def test_adp_missing_is_reported_not_zero():
    out = interpret_fred(_df([("PAYEMS", 159_250.0, 250.0)]))
    assert not any("DIVERGENȚĂ PRIVAT-STAT" in b for b in out)
    assert any("Fără date" in b and "ADP" in b for b in out)
