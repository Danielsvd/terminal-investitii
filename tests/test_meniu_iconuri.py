"""Garda pentru meniul lateral cu iconuri Material (DESIGN.md §6): valorile interne nu se schimbă,
doar eticheta afișată (format_func). Toate ramurile `sectiune == ...` trebuie să aibă o etichetă."""
import ast
import re
from pathlib import Path

SRC = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")


def _menu():
    m = re.search(r"_MENU_LABEL = (\{.*?\})\r?\n", SRC, re.S)
    return ast.literal_eval(m.group(1))


def test_fiecare_sectiune_are_eticheta():
    menu = _menu()
    used = set(re.findall(r'sectiune == "([^"]+)"', SRC))
    assert used and used == set(menu), (used ^ set(menu))


def test_etichete_cu_icon_material_fara_numar():
    for value, label in _menu().items():
        assert re.match(r"\d\. ", value), value
        assert re.match(r":material/[a-z_]+: [A-ZȘȚ]", label), label
        assert not re.search(r"\d\.", label), label


def test_radio_foloseste_format_func():
    assert 'st.sidebar.radio("Mergi la:", list(_MENU_LABEL), format_func=_MENU_LABEL.get)' in SRC
