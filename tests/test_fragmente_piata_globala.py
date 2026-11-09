"""Etapa 5 (viteză): widget-urile din „4. Piață Globală” stau în fragmente (`@st.fragment`), ca un clic să reruleze
doar secțiunea lor, nu toată pagina (FRED, BCE, Yahoo, FinBERT, toate graficele). Test de structură, pe sursă."""
import ast
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "app.py"


def _fragments():
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and any(
                isinstance(d, ast.Attribute) and d.attr == "fragment" for d in node.decorator_list):
            out[node.name] = ast.unparse(node)
    return out


def test_widgeturile_sunt_in_fragmente():
    frs = _fragments()
    assert set(frs) >= {"_pg_ron_returns_fragment", "_pg_macro_chart_fragment", "_pg_sector_heat_fragment"}, sorted(frs)
    assert "key='mk_horizon'" in frs["_pg_ron_returns_fragment"]
    assert "st.radio('Indicator'" in frs["_pg_macro_chart_fragment"]
    assert "st.select_slider('Perioadă'" in frs["_pg_macro_chart_fragment"]
    assert "key='sec_heat_hz'" in frs["_pg_sector_heat_fragment"]


def test_fragmentele_nu_scriu_in_sidebar_si_nu_descarca_fara_cache():
    # un fragment nu poate scrie în afara corpului său (sidebar); descărcările directe ar rula la fiecare clic
    for name, code in _fragments().items():
        if not name.startswith("_pg_"):
            continue
        assert "st.sidebar" not in code, name
        assert "yf.download(" not in code and "yf.Ticker(" not in code and "requests.get(" not in code, name


def test_fiecare_fragment_e_apelat():
    text = SRC.read_text(encoding="utf-8")
    for name in ("_pg_ron_returns_fragment", "_pg_macro_chart_fragment", "_pg_sector_heat_fragment"):
        assert text.count(f"{name}()") == 2, name     # definiția + apelul
