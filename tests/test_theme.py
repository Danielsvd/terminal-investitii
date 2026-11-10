"""Teste pentru ui/theme.py: tokenurile temei luminoase/întunecate și conversia culorilor vechi."""
import re
import tomllib
from pathlib import Path

import pytest

from ui import theme

ROOT = Path(__file__).resolve().parents[1]


def _lum(h):
    h = h.lstrip("#")
    c = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    c = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def contrast(a, b):
    la, lb = _lum(a), _lum(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


# --- tokenuri --------------------------------------------------------------

def test_tokenurile_coincid_cu_config_toml():
    cfg = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))["theme"]
    for mode, idx in (("light", 0), ("dark", 1)):
        t = cfg[mode]
        assert theme.TOKENS["bg"][idx] == t["backgroundColor"]
        assert theme.TOKENS["surface"][idx] == t["secondaryBackgroundColor"]
        assert theme.TOKENS["text"][idx] == t["textColor"]
        assert theme.TOKENS["border"][idx] == t["borderColor"]
        assert theme.TOKENS["up"][idx] == t["greenColor"]
        assert theme.TOKENS["muted"][idx] == t["dataframeHeaderTextColor"]
        assert theme.TOKENS["down"][idx] == t["redColor"]


@pytest.mark.parametrize("tok", ["text", "muted", "accent-text", "up", "down", "warn"])
def test_contrast_text_pe_card_ambele_moduri(tok):
    for idx in (0, 1):
        for bg in ("surface", "surface-2", "bg"):
            assert contrast(theme.TOKENS[tok][idx], theme.TOKENS[bg][idx]) >= 4.5, (tok, bg, idx)


def test_on_fill_pe_culorile_semantice():
    # textul de pe un fundal plin (verdict, GEX) trebuie citit în ambele moduri
    for idx in (0, 1):
        for fill in ("up", "down", "warn"):
            assert contrast(theme.TOKENS["on-fill"][idx], theme.TOKENS[fill][idx]) >= 4.5, (fill, idx)


def test_root_css_are_light_dark_pentru_fiecare_token():
    css = theme.root_css()
    for k, (lt, dk) in theme.TOKENS.items():
        assert f"--tk-{k}: light-dark({lt}, {dk});" in css


def test_root_css_si_premium_css_nu_sunt_modificate_de_themed_html():
    for css in (f"<style>{theme.root_css()}</style>", f"<style>{theme.premium_css()}</style>"):
        assert theme.themed_html(css) == css


def test_premium_css_foloseste_doar_tokenuri_definite():
    used = set(re.findall(r"var\(--tk-([a-z0-9-]+)\)", theme.premium_css()))
    assert used and used <= set(theme.TOKENS), used - set(theme.TOKENS)


def test_premium_css_tinteste_cardurile_inline_dar_nu_surface_2():
    css = theme.premium_css()
    assert 'div[style*="background:var(--tk-surface)"]' in css
    assert "surface-2)\"]" not in css


def test_accentul_principal_cu_text_alb():
    import tomllib as _t
    cfg = _t.loads((ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))["theme"]
    for mode in ("light", "dark"):
        # butonul principal Streamlit are text alb pe primaryColor
        assert contrast("#FFFFFF", cfg[mode]["primaryColor"]) >= 4.4, mode


def test_var_token_necunoscut():
    with pytest.raises(KeyError):
        theme.var("nu-exista")


# --- conversia HTML-ului vechi ---------------------------------------------

def test_hex_vechi_devine_token_indiferent_de_majuscule():
    out = theme.themed_html('<div style="background:#161b22; color:#8B949E; border:1px solid #30363D">')
    assert out == ('<div style="background:var(--tk-surface); color:var(--tk-muted); '
                   'border:1px solid var(--tk-border)">')


def test_culorile_semantice():
    assert theme.themed_html("#3FB950 #F85149 #D29922 #58A6FF") == (
        "var(--tk-up) var(--tk-down) var(--tk-warn) var(--tk-accent-text)")


def test_culoare_necunoscuta_ramane():
    s = '<span style="color:#BF91FF; background:#123456">x</span>'
    assert theme.themed_html(s) == s


def test_rgba_vechi_devine_color_mix():
    out = theme.themed_html("background: rgba(63, 185, 80, 0.1); border: rgba(248,81,73,0.15)")
    assert out == ("background: color-mix(in srgb, var(--tk-up) 10%, transparent); "
                   "border: color-mix(in srgb, var(--tk-down) 15%, transparent)")


def test_rgba_negru_ramane():
    s = "box-shadow: 0 4px 6px rgba(0,0,0,0.3)"
    assert theme.themed_html(s) == s


def test_ancora_html_si_text_fara_culori_neatinse():
    s = "<a href='https://ex.ro/#top'>Legătură</a> 50% din 100"
    assert theme.themed_html(s) == s
    assert theme.themed_html(None) is None


# --- install ---------------------------------------------------------------

class _FakeDG:
    def markdown(self, body, unsafe_allow_html=False, *, help=None):
        return body


class _FakeSt:
    pass


def test_install_converteste_doar_html():
    st = _FakeSt()
    cls = type("DG", (_FakeDG,), {})
    st._main = cls()
    theme.install(st, cls)
    assert st.markdown('<b style="color:#3FB950">x</b>', unsafe_allow_html=True) == '<b style="color:var(--tk-up)">x</b>'
    assert st.markdown("text #3FB950", unsafe_allow_html=False) == "text #3FB950"
    assert cls().markdown("#8B949E", True) == "var(--tk-muted)"


def test_install_idempotent():
    st = _FakeSt()
    cls = type("DG", (_FakeDG,), {})
    st._main = cls()
    theme.install(st, cls)
    first = cls.markdown
    theme.install(st, cls)
    assert cls.markdown is first


# --- tabele (canvas) -------------------------------------------------------

@pytest.mark.parametrize("col", [theme.DF_UP, theme.DF_DOWN, theme.DF_MUTED, theme.DF_WARN])
def test_culorile_de_tabel_lizibile_pe_ambele_fundaluri(col):
    # canvas-ul tabelelor nu acceptă var(); culori de mijloc, ≈ 4:1 pe alb și pe bleumarin
    for bg in ("#FFFFFF", "#F4F4FB", "#141230"):
        assert contrast(col, bg) >= 3.9, (col, bg)


def test_app_tabele_fara_verdele_si_rosul_vechi_in_styler():
    # funcțiile Styler întorc șiruri „color: #...”; pe canvas culorile vechi (#3FB950 / #F85149) nu se citesc pe alb
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    bad = re.findall(r"""['"](?:font-weight: bold; )?color: #(?:3FB950|F85149)""", src, re.I)
    assert not bad, bad
    assert not re.search(r"background-color: #21262D", src)


# --- grafice ---------------------------------------------------------------

def test_theme_figure_dict():
    fig = {"layout": {"paper_bgcolor": "#0E1117", "plot_bgcolor": "#0e1117",
                      "xaxis": {"gridcolor": "#30363D", "tickcolor": "white"}},
           "data": [{"line": {"color": "yellow"}, "textfont": {"color": "#ffffff"},
                     "marker": {"color": ["#3FB950", "cyan"]}, "y": [1, 2], "name": "#30363D"}]}
    out = theme.theme_figure_dict(fig)
    assert out["layout"]["paper_bgcolor"] == "rgba(0,0,0,0)"
    assert out["layout"]["plot_bgcolor"] == "rgba(0,0,0,0)"
    assert out["layout"]["xaxis"]["gridcolor"] == "rgba(128,128,128,0.25)"
    assert out["layout"]["xaxis"]["tickcolor"] == "rgba(128,128,128,0.6)"
    assert out["data"][0]["line"]["color"] == "#F59E0B"
    assert out["data"][0]["textfont"]["color"] == "#ffffff"       # text alb pe celule colorate rămâne
    assert out["data"][0]["marker"]["color"] == ["#3FB950", "#0891B2"]
    assert out["data"][0]["y"] == [1, 2] and out["data"][0]["name"] == "#30363D"   # date și nume neatinse
    assert fig["layout"]["paper_bgcolor"] == "#0E1117"            # originalul nu se modifică


def test_df_colors_dupa_mod():
    assert theme.df_colors("dark") == (theme.TOKENS["up"][1], theme.TOKENS["down"][1], theme.TOKENS["muted"][1])
    assert theme.df_colors("light") == (theme.TOKENS["up"][0], theme.TOKENS["down"][0], theme.TOKENS["muted"][0])
    assert theme.df_colors(None) == (theme.DF_UP, theme.DF_DOWN, theme.DF_MUTED)


def test_df_colors_contrast_pe_fundalul_tabelului():
    # cu mod cunoscut, verdele și roșul din tabele au contrast deplin (≥ 4,5:1) pe fundalul paginii
    for mode, i in (("light", 0), ("dark", 1)):
        for col in theme.df_colors(mode):
            assert contrast(col, theme.TOKENS["bg"][i]) >= 4.5, (mode, col)


def test_app_tabele_folosesc_df_colors():
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "DF_UP, DF_DOWN, DF_MUTED = ui_theme.df_colors(_THEME_TYPE)" in src
    assert not re.search(r"#3A8854|#E82C2C|#707A8F", src)


def test_metricile_centrate_si_compacte():
    css = theme.premium_css()
    assert 'div[data-testid="stMetric"] { padding: 10px 12px; text-align: center; }' in css
    assert "justify-content: center" in css


def test_sign_css_dupa_semn():
    up, down = "#0A0", "#A00"
    assert theme.sign_css("+1,23%", up, down) == "color: #0A0"
    assert theme.sign_css("-0,71%", up, down) == "color: #A00"
    assert theme.sign_css("\u22120,5 pp", up, down) == "color: #A00"
    assert theme.sign_css("+73 lei", up, down) == "color: #0A0"
    assert theme.sign_css("-8 lei", up, down) == "color: #A00"
    assert theme.sign_css("+0,4σ", up, down) == "color: #0A0"


def test_sign_css_fara_culoare():
    up, down = "#0A0", "#A00"
    for v in ("+0,00%", "0 lei", "0,00 pp", "+0,0σ", "-0,0%", "N/A", "", "   ", "0,4σ", None, float("nan"), 1.5):
        assert theme.sign_css(v, up, down) == "", v


def test_theme_figure_dict_verde_crestere():
    fig = {"data": [{"type": "bar", "marker": {"color": ["#3FB950", "#F85149", "#199E70"]}},
                    {"type": "scatter", "fillcolor": "rgba(63, 185, 80, 0.1)", "line": {"color": "#238636"}},
                    {"type": "candlestick", "x": [1], "open": [1], "close": [2]},
                    {"type": "candlestick", "increasing": {"line": {"color": "#123456"}}}]}
    out = theme.theme_figure_dict(fig, up="#34D399")
    assert out["data"][0]["marker"]["color"] == ["#34D399", "#F85149", "#199E70"]   # roșul și categoriile rămân
    assert out["data"][1]["fillcolor"] == "rgba(52, 211, 153, 0.1)"
    assert out["data"][1]["line"]["color"] == "#34D399"
    assert out["data"][2]["increasing"] == {"line": {"color": "#34D399"}}
    assert "decreasing" not in out["data"][2]
    assert out["data"][3]["increasing"]["line"]["color"] == "#123456"             # culoare explicită păstrată
    assert theme.theme_figure_dict(fig)["data"][0]["marker"]["color"][0] == "#3FB950"   # fără up: neschimbat
