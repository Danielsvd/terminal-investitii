"""Garda pentru bannerul „Mediul de piață SUA” (Piață Globală): pe telefon, eticheta verdictului
(„DEFAVORABIL”) nu se mai rupe în mijlocul cuvântului, iar scorul trece sub text când nu încape."""
from pathlib import Path

SRC = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")


def _banner():
    i = SRC.index("Mediul de piață SUA (verdict macro, netestat)")
    return SRC[i - 600:i + 1200]


def test_randul_se_poate_impacheta():
    assert "flex-wrap: wrap" in _banner()


def test_eticheta_nu_se_rupe_in_cuvant():
    b = _banner()
    assert "word-break:keep-all" in b and "font-size:clamp(26px, 8vw, 38px)" in b


def test_scorul_se_micsoreaza_pe_ecran_ingust():
    assert "font-size:clamp(40px, 11vw, 56px)" in _banner()
