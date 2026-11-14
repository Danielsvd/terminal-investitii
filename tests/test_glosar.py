"""Glosarul economic: structura capitolelor și căutarea (fără diacritice, toate cuvintele)."""
from ui import glosar


def test_capitole_complete():
    titles = [c["title"] for c in glosar.CHAPTERS]
    assert len(titles) == len(set(titles)) >= 12
    for ch in glosar.CHAPTERS:
        assert ch["sections"] and all(t and body.strip() for t, body in ch["sections"])


def test_continut_din_notitele_lui_daniel():
    text = " ".join(body for ch in glosar.CHAPTERS for _, body in ch["sections"])
    for must in ("31,1 grame", "USD, CHF, JPY", "1.000 lei / ordin", "maximum 70%", "minimum 15%", "VUAA",
                 "minimum 1 mld. lei", "AROBS"):
        assert must in text, must


def test_cautare():
    assert glosar.search(glosar.CHAPTERS, "") == glosar.CHAPTERS
    hits = glosar.search(glosar.CHAPTERS, "dobanzi reale")          # fără diacritice
    assert hits and any("Aur" in ch["title"] for ch in hits)
    assert all(all("dobanzi" in glosar._norm(ch["title"] + t + b) for t, b in ch["sections"]) for ch in hits)
    assert glosar.search(glosar.CHAPTERS, "xyzinexistent") == []
