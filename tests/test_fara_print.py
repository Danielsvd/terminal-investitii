"""Standard de cod: erorile se loghează cu `logger`, nu cu `print` (app.py, ai_engine.py, analytics/, data/).

`tools/` e exclus: acolo sunt scripturi de linie de comandă, unde `print` e ieșirea normală.
"""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _files():
    yield ROOT / "app.py"
    yield ROOT / "ai_engine.py"
    for folder in ("analytics", "data"):
        yield from (ROOT / folder).rglob("*.py")


def test_fara_print_in_cod():
    gasite = []
    for path in _files():
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print":
                gasite.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert not gasite, "print în loc de logger în " + ", ".join(gasite)
