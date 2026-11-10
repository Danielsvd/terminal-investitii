"""Graficul pilonului „Piața de credit”: aceeași medie și bandă ca regula pilonului."""
import numpy as np
import pandas as pd
import pytest

from analytics import macro as m


def test_banda_egala_cu_regula_pilonului():
    idx = pd.bdate_range("2026-01-01", periods=60)
    s = pd.Series(np.r_[np.full(59, 0.80), 0.82], index=idx)
    f = m.credit_band_frame(s)
    assert f["sma"].iloc[:m.CREDIT_SMA_DAYS - 1].isna().all()          # fără medie înainte de 50 de ședințe
    sma_last = (0.80 * 49 + 0.82) / 50                                  # 0,8004
    assert f["sma"].iloc[-1] == pytest.approx(sma_last)
    assert f["upper"].iloc[-1] == pytest.approx(sma_last * 1.01)
    assert f["lower"].iloc[-1] == pytest.approx(sma_last * 0.99)
    # 0,82 / 0,8004 − 1 = +2,45% > +1% → aceeași concluzie ca pilonul (100)
    assert f["ratio"].iloc[-1] > f["upper"].iloc[-1]
    p = m.pillar_credit(s)
    assert p["score"] == 100.0 and p["value"] == pytest.approx(0.82 / sma_last - 1)


def test_serie_goala_sau_cu_lipsa():
    assert m.credit_band_frame(None).empty
    assert m.credit_band_frame(pd.Series(dtype=float)).empty
    s = pd.Series([0.8, np.nan, 0.81], index=pd.bdate_range("2026-01-01", periods=3))
    f = m.credit_band_frame(s)
    assert len(f) == 2 and f["sma"].isna().all()                        # lipsa nu devine 0
