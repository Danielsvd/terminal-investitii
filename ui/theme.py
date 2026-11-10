"""Tema aplicației (DESIGN.md, „Bleumarin + Revolut”): tokenuri de culoare pentru modul luminos și întunecat.

De ce așa:
- Cardurile HTML proprii (st.markdown cu unsafe_allow_html) aveau culori întunecate scrise direct
  (paleta veche #161B22, #8B949E, #3FB950...). În modul luminos textul moștenit din temă (închis)
  ajungea pe fundaluri închise și nu se mai citea.
- Streamlit pune `color-scheme: light|dark` pe containerul aplicației, deci funcția CSS `light-dark()`
  alege singură valoarea potrivită, inclusiv imediat după schimbarea temei din meniu (fără rerulare).
  `st.context.theme.type` nu e folosit: documentația spune că poate fi greșit la prima încărcare
  și imediat după schimbarea temei.
- `themed_html()` înlocuiește, la afișare, culorile paletei vechi cu variabilele `--tk-*`. Codul care
  produce HTML-ul nu se schimbă, iar graficele Plotly (care nu trec prin st.markdown) nu sunt atinse.

Funcții pure, fără `st.*`, cu teste în tests/test_theme.py. Singura legătură cu Streamlit e `install()`.
"""
import re

# (luminos, întunecat) — valorile și rolurile sunt în DESIGN.md §1
TOKENS = {
    "bg": ("#F3F5F9", "#0B1220"),
    "surface": ("#FFFFFF", "#111A2E"),
    "surface-2": ("#F8FAFC", "#18233B"),
    "border": ("#DCE2EC", "#1F2A44"),
    "border-strong": ("#C4CDDB", "#2E3B5C"),
    "text": ("#0F172A", "#E5ECF6"),
    "muted": ("#526079", "#8A97AD"),
    "accent": ("#1D4ED8", "#3B82F6"),
    "accent-text": ("#1D4ED8", "#60A5FA"),
    "up": ("#15803D", "#22C55E"),
    "down": ("#B91C1C", "#F87171"),
    "warn": ("#B45309", "#FBBF24"),
    # textul de pe un fundal plin colorat (verdict, GEX): alb pe culorile închise din modul luminos,
    # bleumarin închis pe culorile luminoase din modul întunecat
    "on-fill": ("#FFFFFF", "#0B1220"),
}

# Paleta veche (GitHub dark) → token. Culorile care nu sunt aici rămân neschimbate.
LEGACY = {
    "#0E1117": "bg", "#0D1117": "bg",
    "#161B22": "surface",
    "#21262D": "surface-2",
    "#30363D": "border", "#383835": "border",
    "#484F58": "border-strong",
    "#C9D1D9": "text", "#B0B8C4": "text",
    "#8B949E": "muted", "#6E7681": "muted",
    "#3FB950": "up", "#238636": "up", "#199E70": "up", "#008300": "up",
    "#F85149": "down", "#DA3633": "down",
    "#D29922": "warn", "#DBAB09": "warn", "#FFAB00": "warn", "#C98500": "warn",
    "#58A6FF": "accent-text", "#3987E5": "accent-text", "#0678FA": "accent-text",
}

# rgba(r, g, b, a) din paleta veche → color-mix cu tokenul (fundaluri și margini translucide)
LEGACY_RGB = {
    (63, 185, 80): "up",
    (248, 81, 73): "down",
    (210, 153, 34): "warn", (219, 171, 9): "warn",
    (139, 148, 158): "muted",
    (88, 166, 255): "accent-text",
    (255, 255, 255): "text",   # albul translucid de pe cardurile închise = o nuanță a textului
}

# Tabele (st.dataframe cu Styler): grila se desenează pe canvas, unde var() și light-dark() nu merg.
# Culori de mijloc, lizibile pe ambele fundaluri (contrast ≈ 4:1 pe alb și pe bleumarin; DESIGN.md §5.6).
DF_UP = "#3A8854"
DF_DOWN = "#E82C2C"
DF_MUTED = "#707A8F"
DF_WARN = "#A0701C"
DF_HIGHLIGHT_BG = "rgba(59, 130, 246, 0.14)"   # rândul evidențiat (compania analizată, mediana)

_HEX = re.compile(r"#[0-9A-Fa-f]{6}\b")
_RGBA = re.compile(r"rgba\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*([0-9.]+)\s*\)")


def var(name):
    """`var(--tk-<name>)`, pentru HTML scris direct cu tokenuri."""
    if name not in TOKENS:
        raise KeyError(name)
    return f"var(--tk-{name})"


def root_css():
    """Blocul `:root { --tk-*: light-dark(luminos, întunecat); }`."""
    body = " ".join(f"--tk-{k}: light-dark({lt}, {dk});" for k, (lt, dk) in TOKENS.items())
    return f":root {{ {body} }}"


def _hex_sub(m):
    tok = LEGACY.get(m.group(0).upper())
    return var(tok) if tok else m.group(0)


def _rgba_sub(m):
    rgb = tuple(int(m.group(i)) for i in (1, 2, 3))
    tok = LEGACY_RGB.get(rgb)
    if tok is None:
        return m.group(0)
    pct = float(m.group(4)) * 100
    pct_txt = f"{pct:.1f}".rstrip("0").rstrip(".")
    return f"color-mix(in srgb, {var(tok)} {pct_txt}%, transparent)"


def themed_html(html):
    """Înlocuiește culorile paletei vechi cu tokenuri care urmează tema. Restul textului rămâne identic."""
    if not isinstance(html, str) or ("#" not in html and "rgba(" not in html):
        return html
    return _RGBA.sub(_rgba_sub, _HEX.sub(_hex_sub, html))


def install(st_module, delta_generator_cls):
    """Trece prin `themed_html` tot HTML-ul afișat cu `markdown(..., unsafe_allow_html=True)`:
    `st.markdown`, `st.sidebar.markdown` și `col.markdown`. Textul fără HTML nu e atins.
    Idempotent: a doua apelare nu mai împachetează încă o dată."""
    orig = delta_generator_cls.markdown
    if getattr(orig, "_tk_themed", False):
        return

    def markdown(self, body, unsafe_allow_html=False, *args, **kwargs):
        if unsafe_allow_html:
            body = themed_html(body)
        return orig(self, body, unsafe_allow_html, *args, **kwargs)

    markdown._tk_themed = True
    markdown.__doc__ = orig.__doc__
    delta_generator_cls.markdown = markdown
    main = getattr(st_module, "_main", None)
    if main is not None:
        st_module.markdown = main.markdown


# ---------------------------------------------------------------------------
# Grafice Plotly: se desenează în SVG cu culori fixe (fără var()). Streamlit aplică deja tema lui
# (luminos/întunecat) peste șablonul figurii; rămân de corectat culorile scrise direct pentru modul
# întunecat, care în modul luminos dau un bloc negru sau linii invizibile.
# ---------------------------------------------------------------------------
_TRANSPARENT = "rgba(0,0,0,0)"
_GRID = "rgba(128,128,128,0.25)"        # grilă neutră, vizibilă pe alb și pe bleumarin
PLOT_COLOR_MAP = {
    "#0E1117": _TRANSPARENT,             # fundal închis forțat → fundalul temei
    "#30363D": _GRID,                    # grilă / contur închis → gri translucid
    "YELLOW": "#F59E0B",                 # galben pur: invizibil pe alb → chihlimbar (DESIGN.md §1.5)
    "CYAN": "#0891B2", "#00E5FF": "#0891B2",
    "RGBA(255, 255, 255, 0.4)": "rgba(128,128,128,0.5)",
    "RGBA(255,255,255,0.08)": "rgba(128,128,128,0.12)",
}
# alb doar unde e contur/tick pe fundalul graficului; textul alb de pe celulele colorate rămâne alb
_WHITE_KEYS = {"tickcolor"}


def _plot_color(key, val):
    up = val.strip().upper()
    if up in ("WHITE", "#FFFFFF"):
        return "rgba(128,128,128,0.6)" if key in _WHITE_KEYS else val
    return PLOT_COLOR_MAP.get(up, val)


def theme_figure_dict(obj, key=None):
    """Copie a dicționarului unei figuri Plotly, cu culorile pentru fundal închis înlocuite (PLOT_COLOR_MAP).
    Nu modifică datele, doar valorile text ale cheilor care conțin „color”."""
    if isinstance(obj, dict):
        return {k: theme_figure_dict(v, k) for k, v in obj.items()}
    if isinstance(obj, list):
        return [theme_figure_dict(v, key) for v in obj]
    if isinstance(obj, str) and key is not None and "color" in key.lower():
        return _plot_color(key, obj)
    return obj
