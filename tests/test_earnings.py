"""Teste pentru analytics.earnings.earnings_table (valori sintetice)."""
import math

import pandas as pd

from analytics.earnings import earnings_table

LABELS = ["Estimare", "Realizat", "Diferență", "Surpriză %"]


def _idx():
    return pd.to_datetime(["2026-03-31", "2026-06-30"])


def test_toate_coloanele():
    df = pd.DataFrame({"epsEstimate": [1.0, 1.1], "epsActual": [1.05, 1.0],
                       "epsDifference": [0.05, -0.1], "surprisePercent": [0.05, -0.0909]},
                      index=_idx())
    out, has_est = earnings_table(df)
    assert has_est is True
    assert list(out.columns) == LABELS
    assert out["Estimare"].tolist() == [1.0, 1.1]
    assert out["Realizat"].tolist() == [1.05, 1.0]
    assert out["Surpriză %"].iloc[1] == -0.0909


def test_doar_eps_raportat_fara_analisti():
    # cazul TEL.RO: Yahoo are EPS-ul raportat, dar nu are estimări
    df = pd.DataFrame({"epsActual": [0.42, 0.55]}, index=_idx())
    out, has_est = earnings_table(df)
    assert has_est is False
    assert list(out.columns) == LABELS
    assert out["Realizat"].tolist() == [0.42, 0.55]
    for col in ["Estimare", "Diferență", "Surpriză %"]:
        assert out[col].isna().all()  # lipsă = NaN, nu 0


def test_estimare_prezenta_dar_goala():
    df = pd.DataFrame({"epsEstimate": [None, None], "epsActual": [0.42, 0.55]}, index=_idx())
    out, has_est = earnings_table(df)
    assert has_est is False
    assert out["Estimare"].isna().all()


def test_tabel_gol_sau_none():
    for df in (None, pd.DataFrame()):
        out, has_est = earnings_table(df)
        assert out.empty and has_est is False
        assert list(out.columns) == LABELS


def test_fara_coloane_cunoscute():
    out, has_est = earnings_table(pd.DataFrame({"altceva": [1, 2]}))
    assert out.empty and has_est is False


def test_valori_nenumerice_devin_nan():
    df = pd.DataFrame({"epsEstimate": ["1.0", "x"], "epsActual": [1.05, 1.0]}, index=_idx())
    out, has_est = earnings_table(df)
    assert has_est is True
    assert out["Estimare"].iloc[0] == 1.0
    assert math.isnan(out["Estimare"].iloc[1])
