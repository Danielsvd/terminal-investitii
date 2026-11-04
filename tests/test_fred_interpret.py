"""Interpretarea Barometrului FED: un indicator lipsă nu mai e 0 (regula se sare)."""
import pandas as pd

from analytics.macro import interpret_fred


def _df(rows):
    return pd.DataFrame([{"_simbol": k, "Valoare Curentă": v, "_change_raw": c, "_yoy": y} for k, v, c, y in rows])


def test_somaj_lipsa_nu_declanseaza_goldilocks():
    # înainte: UNRATE lipsă = 0 < 4 → „GOLDILOCKS” cu PCE 0,1 și retail +0,4
    out = interpret_fred(_df([("PCEPILFE", 125.0, 0.1, None), ("RSAFS", 720000.0, 0.4, None)]))
    assert not any("GOLDILOCKS" in b for b in out)
    assert any("șomaj" in b and "Fără date" in b for b in out)


def test_goldilocks_cu_date_complete():
    out = interpret_fred(_df([("UNRATE", 3.8, -0.1, None), ("PCEPILFE", 125.0, 0.1, None), ("RSAFS", 720000.0, 0.4, None)]))
    assert any("GOLDILOCKS" in b for b in out)


def test_inflatie_si_dobanda_reala():
    out = interpret_fred(_df([("PCEPILFE", 125.0, 0.3, None)]))
    assert any("INFLAȚIE REZISTENTĂ" in b for b in out)
    out2 = interpret_fred(_df([("PCEPILFE", 125.0, 0.1, None), ("FEDFUNDS", 5.3, 0.0, None), ("CPIAUCSL", 320.0, 0.2, 3.0)]))
    assert any("DOBÂNZI REALE MARI" in b and "2,3%" in b for b in out2)    # 5,3 − 3,0


def test_somaj_in_crestere():
    out = interpret_fred(_df([("UNRATE", 4.3, 0.2, None)]))
    assert any("ȘOMAJ ÎN CREȘTERE" in b for b in out)


def test_gol():
    assert interpret_fred(pd.DataFrame()) == []
