"""Verdictul macro pe piloni: scoruri 0–100, ponderi reale, lipsă = exclus (nu neutru)."""
import numpy as np
import pandas as pd
import pytest

from analytics import macro as m


def test_vix_liniar_si_capete():
    assert m.pillar_vix(15)["score"] == 100
    assert m.pillar_vix(10)["score"] == 100        # plafonat
    assert m.pillar_vix(20)["score"] == pytest.approx(50)
    assert m.pillar_vix(22.5)["score"] == pytest.approx(25)
    assert m.pillar_vix(30)["score"] == 0
    for bad in (None, np.nan, 0, -1, "x"):
        assert m.pillar_vix(bad)["score"] is None


def test_curba():
    assert m.pillar_curve(-0.2)["score"] == 0
    assert m.pillar_curve(0.0)["score"] == 50
    assert m.pillar_curve(0.49)["score"] == 50
    assert m.pillar_curve(0.5)["score"] == 100
    assert m.pillar_curve(None)["score"] is None


def test_dolar():
    assert m.pillar_dollar(-0.8)["score"] == 0
    assert m.pillar_dollar(-0.35)["score"] == pytest.approx(50)
    assert m.pillar_dollar(0.3)["score"] == 100
    assert m.pillar_dollar(np.nan)["score"] is None


def test_sentiment():
    assert m.pillar_sentiment(0.2)["score"] == 100
    assert m.pillar_sentiment(0.0)["score"] == 50
    assert m.pillar_sentiment(-0.16)["score"] == 0
    assert m.pillar_sentiment(None)["score"] is None


def test_credit_fata_de_sma50():
    base = pd.Series([1.0] * 60)
    up = base.copy(); up.iloc[-1] = 1.03         # sma50 = (49 + 1,03)/50 = 1,0006 → +2,94%
    down = base.copy(); down.iloc[-1] = 0.97     # −2,94%
    flat = base.copy(); flat.iloc[-1] = 1.005    # +0,49%: în bandă
    assert m.pillar_credit(up)["score"] == 100
    assert m.pillar_credit(up)["value"] == pytest.approx(1.03 / 1.0006 - 1)
    assert m.pillar_credit(down)["score"] == 0
    assert m.pillar_credit(flat)["score"] == 50
    assert m.pillar_credit(pd.Series([1.0] * 49))["score"] is None
    assert m.pillar_credit(None)["score"] is None


def test_sectoare():
    rets = {"Tehnologie": 0.05, "Financiar": 0.04, "Energie": 0.03, "Utilități": -0.01,
            "Sănătate": 0.0, "Industrial": 0.01, "Consum de Bază": -0.02}
    assert m.pillar_sectors(rets)["score"] == 100        # 2 ciclice în top 3
    rets2 = dict(rets, **{"Utilități": 0.10, "Financiar": -0.05})
    assert m.pillar_sectors(rets2)["score"] == 0         # Utilități, Tehnologie, Energie
    rets3 = dict(rets, **{"Financiar": -0.05, "Sănătate": 0.045})
    assert m.pillar_sectors(rets3)["score"] == 50        # Tehnologie, Sănătate, Energie
    assert m.pillar_sectors({"Tehnologie": 0.1})["score"] is None


def test_verdict_media_ponderata():
    p = {"credit": m._pillar(100, None, ""), "vix": m._pillar(50, None, ""), "dolar": m._pillar(0, None, ""),
         "sectoare": m._pillar(100, None, ""), "curba": m._pillar(0, None, ""), "sentiment": m._pillar(50, None, "")}
    v = m.macro_verdict(p)
    # (30·100 + 20·50 + 15·0 + 15·100 + 10·0 + 10·50) / 100 = 60
    assert v["score"] == pytest.approx(60.0)
    assert v["label"] == "FAVORABIL" and v["missing"] == [] and v["weight_used"] == 100


def test_verdict_lipsa_e_exclusa_nu_neutra():
    p = {"credit": m._pillar(100, None, ""), "vix": m.pillar_vix(None), "curba": m._pillar(0, None, "")}
    v = m.macro_verdict(p)
    # doar credit (30) și curba (10): (3000 + 0) / 40 = 75; o lipsă tratată ca 50 ar fi dat altceva
    assert v["weight_used"] == 40
    assert v["score"] is None                            # 40 < 50: fără verdict
    p["vix"] = m._pillar(50, None, "")
    v = m.macro_verdict(p)
    # (3000 + 1000 + 0) / 60 = 66,67
    assert v["score"] == pytest.approx(200 / 3)
    assert set(v["missing"]) == {"dolar", "sectoare", "sentiment"}


def test_etichete_praguri():
    one = lambda s: m.macro_verdict({"credit": m._pillar(s, None, ""), "vix": m._pillar(s, None, "")})
    assert one(60)["label"] == "FAVORABIL"
    assert one(59.9)["label"] == "MIXT"
    assert one(40)["label"] == "MIXT"
    assert one(39.9)["label"] == "DEFAVORABIL"


def test_randament_pe_o_luna_calendaristica():
    idx = pd.to_datetime(["2026-09-01", "2026-10-01", "2026-10-02", "2026-10-15", "2026-11-02"])
    s = pd.Series([90.0, 100.0, 101.0, 105.0, 110.0], index=idx)
    # cutoff 2026-10-02 → 101 ; 110/101 − 1
    assert m.period_return(s) == pytest.approx(110 / 101 - 1)
    assert m.period_return(s.iloc[-2:]) is None           # istoricul nu acoperă o lună
    assert m.period_return(None) is None
