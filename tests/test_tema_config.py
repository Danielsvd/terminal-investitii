"""Garda pentru tema aplicației (.streamlit/config.toml), direcția „Bleumarin + Revolut” din DESIGN.md.

Verifică:
- ambele moduri (light și dark) sunt definite, cu aceleași chei;
- `base` nu apare în [theme.light] / [theme.dark] (Streamlit 1.65 îl respinge acolo);
- culorile sunt hex valide;
- contrastul WCAG text/fundal este de cel puțin 4,5:1 în ambele moduri;
- CSS-ul global din app.py nu mai forțează fundalul paginii (ar bloca modul luminos).
"""
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / ".streamlit" / "config.toml"
HEX = re.compile(r"^#[0-9A-Fa-f]{6}$")


def _cfg():
    return tomllib.loads(CONFIG.read_text(encoding="utf-8"))


def _lum(h):
    h = h.lstrip("#")
    c = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    c = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def contrast(a, b):
    la, lb = _lum(a), _lum(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


def test_contrast_valori_de_referinta():
    # Valori cunoscute: alb/negru = 21; alb/alb = 1.
    assert round(contrast("#FFFFFF", "#000000"), 2) == 21.0
    assert round(contrast("#FFFFFF", "#FFFFFF"), 2) == 1.0


def test_ambele_moduri_definite_cu_aceleasi_chei():
    th = _cfg()["theme"]
    assert "light" in th and "dark" in th
    assert set(th["light"]) == set(th["dark"])


def test_base_nu_apare_in_subsectiuni():
    th = _cfg()["theme"]
    assert "base" not in th["light"]
    assert "base" not in th["dark"]


def test_culori_hex_valide():
    th = _cfg()["theme"]
    for mod in ("light", "dark"):
        for k, v in th[mod].items():
            vals = v if isinstance(v, list) else [v]
            for x in vals:
                if isinstance(x, str) and x.startswith("#"):
                    assert HEX.match(x), f"{mod}.{k}: {x}"


def test_contrast_text_pe_fundal_si_card():
    th = _cfg()["theme"]
    for mod in ("light", "dark"):
        t = th[mod]
        assert contrast(t["textColor"], t["backgroundColor"]) >= 4.5, mod
        assert contrast(t["textColor"], t["secondaryBackgroundColor"]) >= 4.5, mod
        for k in ("greenColor", "redColor", "linkColor"):
            assert contrast(t[k], t["secondaryBackgroundColor"]) >= 4.5, f"{mod}.{k}"


def test_css_global_nu_forteaza_fundalul_paginii():
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    assert not re.search(r"\.stApp\s*\{\s*background-color", src)


def test_signed_key_si_css():
    from ui import theme
    assert theme.signed_key("mom-S&P 500", 0.143) == "semn-poz-mom-s-p-500"
    assert theme.signed_key("bet-sigma", -0.4) == "semn-neg-bet-sigma"
    assert theme.signed_key("x", 0) == "semn-zero-x" and theme.signed_key("x", None) == "semn-zero-x"
    assert theme.signed_key("x", float("nan")) == "semn-zero-x"
    css = theme.premium_css()
    assert '[class*="st-key-semn-poz-"] [data-testid="stMetricValue"] { color: var(--tk-up); }' in css
    assert '[class*="st-key-semn-neg-"] [data-testid="stMetricValue"] { color: var(--tk-down); }' in css


def test_tooltip_incape_pe_telefon():
    from ui import theme
    css = theme.premium_css()
    assert '[data-testid="stTooltipContent"] { max-width: min(560px, calc(100vw - 24px)) !important;' in css
