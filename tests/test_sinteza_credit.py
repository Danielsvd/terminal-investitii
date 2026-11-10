"""Sinteza AI: concluzia despre credit vine din pilonul HYG/IEF, nu din curbă."""
import pandas as pd

from analytics import macro as m


def _ratio(last):
    s = pd.Series([1.0] * 60)
    s.iloc[-1] = last
    return s


def test_credit_sentence_pe_banda():
    # sma50 = (49 + x) / 50; 0,97 → −2,94%; 1,03 → +2,94%; 1,005 → +0,49%
    assert m.credit_sentence(m.pillar_credit(_ratio(0.97))) == (
        "Piața de credit: retragere din risc (HYG/IEF -2,94% față de media pe 50 de ședințe, sub banda de ±1%).")
    assert m.credit_sentence(m.pillar_credit(_ratio(1.03))).startswith("Piața de credit: apetit pentru risc (HYG/IEF +2,94%")
    assert m.credit_sentence(m.pillar_credit(_ratio(1.005))).startswith("Piața de credit: neutră (HYG/IEF +0,49%")
    assert m.credit_sentence(m.pillar_credit(pd.Series([1.0] * 10))) is None
    assert m.credit_sentence(None) is None


def test_sinteza_nu_mai_spune_credit_sanatos_cand_pilonul_e_0():
    import ai_engine
    with open(ai_engine.__file__, encoding="utf-8") as f:
        src = f.read()
    assert ":.2f}). Creditarea corporativă este sănătoasă." not in src  # fraza veche din bullet (comentariul o citează)
