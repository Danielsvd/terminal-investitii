"""Portofoliu → Risc pe curba reală: procentele în format românesc (virgulă zecimală)."""
from pathlib import Path

from data.helpers import ro

SRC = Path(__file__).resolve().parents[1].joinpath("app.py").read_text(encoding="utf-8")


def test_pct_din_riscul_pe_curba_reala_foloseste_ro():
    block = SRC[SRC.index('st.markdown("**Risc, pe curba reală**")'):SRC.index("def _ledger_num(")]
    assert 'f"{ro(v * 100):.{d}f}%"' in block
    assert 'f"{v * 100:' not in block


def test_formatul_asteptat():
    assert f"{ro(-0.0070 * 100):.2f}%" == "-0,70%" and f"{ro(0.093 * 100):.1f}%" == "9,3%"
