"""Graficul Venit / Venit net: axa și hover-ul marjei cu o zecimală, ca etichetele („26,2%”, nu „26,186%”)."""
from pathlib import Path

SRC = Path(__file__).resolve().parents[1].joinpath("app.py").read_text(encoding="utf-8")


def test_marja_o_zecimala():
    assert 'title_text="Marjă (%)", showgrid=False, ticksuffix="%", tickformat=".1f"' in SRC
    block = SRC[SRC.index("name='Marja Netă (%)'"):SRC.index('title_text="Marjă (%)"')]
    assert 'hovertemplate="%{y:.1f}%"' in block
