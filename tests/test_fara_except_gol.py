"""Garda: niciun `except:` gol în app.py, ai_engine.py, analytics/, data/, tools/.

Un `except:` gol prinde și KeyboardInterrupt/SystemExit și ascunde erorile reale.
Se prinde `Exception` (sau mai specific) și se loghează.
"""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _files():
    yield ROOT / "app.py"
    yield ROOT / "ai_engine.py"
    for folder in ("analytics", "data", "tools"):
        yield from (ROOT / folder).rglob("*.py")


def test_fara_except_gol():
    gasite = []
    for path in _files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                gasite.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert not gasite, "except: gol în " + ", ".join(gasite)
