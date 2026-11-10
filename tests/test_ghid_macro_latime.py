"""Piață Globală: ghidul macro, matricea de sensibilitate și rotația nu depind de indicatorul ales (10.11).

Înainte stăteau în coloana graficului, în `if not series.empty:`, deci dispăreau când indicatorul
ales nu avea date. Testul citește structura din app.py (AST): blocul graficului nu le mai conține,
iar ele sunt după el, pe nivelul secțiunii.
"""
import ast
from pathlib import Path

SRC = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
TREE = ast.parse(SRC)
MARKERS = ("Analiza Corelațiilor (Ghid Macro)", "Matricea de Sensibilitate Economică", "Indicator Rotație Sectoare")


def _strings(node):
    return {n.value for n in ast.walk(node) if isinstance(n, ast.Constant) and isinstance(n.value, str)}


def _macro_chart_if():
    """`if not series.empty:` din Piață Globală (cel care desenează `fig_macro`)."""
    for node in ast.walk(TREE):
        if (isinstance(node, ast.If) and isinstance(node.test, ast.UnaryOp) and isinstance(node.test.op, ast.Not)
                and ast.unparse(node.test.operand) == "series.empty"
                and any(isinstance(n, ast.Name) and n.id == "fig_macro" for n in ast.walk(node))):
            return node
    raise AssertionError("blocul graficului macro nu a fost găsit")


def test_ghidurile_nu_sunt_in_blocul_graficului():
    inside = " ".join(_strings(_macro_chart_if()))
    for marker in MARKERS:
        assert marker not in inside, f"„{marker}” e încă în `if not series.empty:`"


def test_ghidurile_sunt_dupa_grafic_si_in_ordine():
    chart_line = _macro_chart_if().lineno
    pos = [SRC.index(m) for m in MARKERS]
    assert all(SRC[:p].count("\n") + 1 > chart_line for p in pos)
    assert pos == sorted(pos)                         # ordinea blocurilor rămâne aceeași
