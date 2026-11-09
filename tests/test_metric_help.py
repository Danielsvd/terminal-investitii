"""Gardă: fiecare card `.metric(...)` din app.py are „?” (`help=`), cum cer standardele proiectului.

Un card nou fără explicație face testul să pice, cu rândul și eticheta lui."""
import ast
from pathlib import Path

APP = Path(__file__).resolve().parent.parent / "app.py"


def _metrics_without_help(source):
    tree = ast.parse(source)
    out = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "metric"
                and not any(k.arg == "help" for k in node.keywords)
                and not any(k.arg is None for k in node.keywords)):   # **kwargs: nu se poate verifica static
            label = ast.get_source_segment(source, node.args[0]) if node.args else "?"
            out.append(f"app.py:{node.lineno} {label}")
    return out


def test_fiecare_card_metric_are_help():
    missing = _metrics_without_help(APP.read_text(encoding="utf-8"))
    assert not missing, "carduri fără help=:\n" + "\n".join(missing)


def test_garda_prinde_cardul_fara_help():
    src = 'import streamlit as st\nst.metric("A", 1, help="x")\nc.metric("B", 2)\n'
    assert _metrics_without_help(src) == ['app.py:3 "B"']
