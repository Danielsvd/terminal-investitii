"""get_market_curve: un eșec nu se salvează în cache (excepția iese din funcția cu cache)."""
import ast
import pathlib

SRC = (pathlib.Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
TREE = ast.parse(SRC)


def _fn(name):
    return next(n for n in ast.walk(TREE) if isinstance(n, ast.FunctionDef) and n.name == name)


def test_functia_cu_cache_nu_intoarce_esecul():
    fn = _fn("_market_curve_cached")
    assert any(isinstance(d, ast.Call) and getattr(d.func, "attr", "") == "cache_data" for d in fn.decorator_list)
    for node in ast.walk(fn):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Tuple):
            first = node.value.elts[0]
            assert not (isinstance(first, ast.Constant) and first.value is None), "eșecul nu trebuie întors (s-ar salva în cache)"


def test_wrapperul_nu_are_cache_si_prinde_exceptia():
    fn = _fn("get_market_curve")
    assert not fn.decorator_list
    assert any(isinstance(n, ast.ExceptHandler) and getattr(n.type, "id", "") == "CurveUnavailable" for n in ast.walk(fn))
