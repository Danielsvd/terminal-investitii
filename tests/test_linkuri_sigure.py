"""Gardă: orice link construit din date externe (RSS) într-un bloc HTML trece prin `safe_link`.

`html.escape` oprește ieșirea din atribut, dar nu un link „javascript:...”; `safe_link` lasă doar http(s).
"""
import re
from pathlib import Path

from data.news import safe_link

ROOT = Path(__file__).resolve().parents[1]
FILES = ["app.py", "ai_engine.py", *sorted(str(p.relative_to(ROOT)) for p in ROOT.glob("data/*.py")),
         *sorted(str(p.relative_to(ROOT)) for p in ROOT.glob("analytics/*.py")),
         *sorted(str(p.relative_to(ROOT)) for p in ROOT.glob("ui/*.py"))]
HREF = re.compile(r'href="\{')


def test_safe_link_lasa_doar_http():
    assert safe_link("https://finance.yahoo.com/x") == "https://finance.yahoo.com/x"
    assert safe_link(" http://a.ro ") == "http://a.ro"
    assert safe_link("javascript:alert(1)") is None
    assert safe_link("JaVaScRiPt:alert(1)") is None
    assert safe_link("data:text/html,x") is None
    assert safe_link(None) is None
    assert safe_link("") is None


def test_href_din_variabila_trece_prin_safe_link():
    bad = []
    for name in FILES:
        for no, line in enumerate((ROOT / name).read_text(encoding="utf-8").splitlines(), 1):
            if HREF.search(line) and "safe_link(" not in line and "html.escape(link" not in line:
                bad.append(f"{name}:{no}")
    assert not bad, "href construit fără safe_link: " + ", ".join(bad)
