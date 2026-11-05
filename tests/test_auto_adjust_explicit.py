"""Fiecare apel yfinance `.download(...)` / `.history(...)` are `auto_adjust` explicit.

Regula din instrucțiunile proiectului: valoarea implicită s-a schimbat între versiunile
yfinance (False → True la download), deci se scrie de fiecare dată.
"""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ["app.py", "ai_engine.py", *sorted(str(p.relative_to(ROOT)) for d in ("analytics", "data", "tools")
                                         for p in (ROOT / d).glob("*.py"))]


def _missing(path):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    out = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("download", "history")):
            kws = {k.arg for k in node.keywords}
            if "auto_adjust" not in kws and None not in kws:  # None = **kwargs, nu se poate verifica
                out.append(f"{path}:{node.lineno}")
    return out


def test_auto_adjust_explicit_peste_tot():
    missing = [m for f in FILES for m in _missing(f)]
    assert missing == [], f"apeluri fără auto_adjust explicit: {missing}"
