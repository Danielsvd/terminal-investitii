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

# (luminos, întunecat) — valorile și rolurile sunt în DESIGN.md §1 (paleta „Indigo profund”, aleasă pe 10.11)
TOKENS = {
    "bg": ("#F4F4FB", "#141230"),
    "surface": ("#FFFFFF", "#1C1940"),
    "surface-top": ("#FFFFFF", "#221F4C"),      # capătul de sus al gradientului fin de pe carduri
    "surface-2": ("#F8F8FD", "#242150"),
    "border": ("#E3E1F3", "#2E2A5C"),
    "border-strong": ("#CFCCE8", "#3D3874"),
    "text": ("#16133A", "#ECEAFB"),
    "muted": ("#5F5A8A", "#A09BC8"),
    "accent": ("#4F46E5", "#818CF8"),
    "accent-text": ("#4338CA", "#A5B4FC"),
    "up": ("#047857", "#34D399"),
    "down": ("#B91C1C", "#F87171"),
    "warn": ("#B45309", "#FBBF24"),
    # textul de pe un fundal plin colorat (verdict, GEX): alb pe culorile închise din modul luminos,
    # indigo închis pe culorile luminoase din modul întunecat
    "on-fill": ("#FFFFFF", "#141230"),
    # adâncime: linia luminoasă de pe muchia de sus, umbra de sub card, strălucirea din capul paginii
    # (hex cu 8 cifre = culoare + transparență; nu sunt atinse de themed_html)
    "highlight": ("#FFFFFFE6", "#FFFFFF0F"),
    "shadow": ("#16133A12", "#05041473"),
    "glow": ("#4F46E50F", "#818CF81A"),
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
# Culori de mijloc, lizibile pe ambele fundaluri (contrast ≈ 4:1 pe alb și pe fundalul indigo; DESIGN.md §5.6).
DF_UP = "#3A8854"
DF_DOWN = "#E82C2C"
DF_MUTED = "#707A8F"
DF_WARN = "#A0701C"
DF_HIGHLIGHT_BG = "rgba(59, 130, 246, 0.14)"   # rândul evidențiat (compania analizată, mediana)

def df_colors(mode):
    """(câștig, pierdere, neutru) pentru tabele, după modul temei (`st.context.theme.type`).
    Cu mod cunoscut se folosesc tokenurile pline (contrast ≥ 5:1); necunoscut (None) → culorile
    de mijloc, lizibile pe ambele fundaluri. Documentația Streamlit avertizează că modul poate fi
    greșit chiar în momentul schimbării temei; la următoarea rerulare se corectează."""
    if mode in ("light", "dark"):
        i = 0 if mode == "light" else 1
        return TOKENS["up"][i], TOKENS["down"][i], TOKENS["muted"][i]
    return DF_UP, DF_DOWN, DF_MUTED


_NONZERO = re.compile(r"[1-9]")


def sign_css(text, up, down):
    """Culoarea CSS a unei celule de variație deja formatate ca text („+1,23%”, „-0,71%”, „-8 lei”).
    Semnul vine din primul caracter: „+” → `up`, „-” sau „−” → `down`. Fără nicio cifră nenulă
    („+0,00%”, „0 lei”, „0,0σ”) sau fără semn → fără culoare. „N/A” și valorile lipsă → fără culoare.
    Se folosește cu `Styler.map(..., subset=coloanele de variație)` peste un `text_frame`."""
    if not isinstance(text, str):
        return ""
    t = text.strip()
    if not t or not _NONZERO.search(t):
        return ""
    if t[0] == "+":
        return f"color: {up}"
    if t[0] in "-\u2212":
        return f"color: {down}"
    return ""


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


_CARD_SHADOW = "inset 0 1px 0 var(--tk-highlight), 0 1px 2px var(--tk-shadow), 0 10px 28px var(--tk-shadow)"


def premium_css():
    """Tratamentul „premium” al suprafețelor (DESIGN.md §4): gradient fin pe carduri, contur de 1px,
    linie luminoasă sus, umbră moale, colțuri de 14px și o strălucire discretă în capul paginii.
    Se injectează DUPĂ CSS-ul vechi din app.py, ca să-l înlocuiască la aceeași specificitate."""
    card = ("background: linear-gradient(180deg, var(--tk-surface-top) 0%, var(--tk-surface) 100%);"
            " border: 1px solid var(--tk-border); border-radius: 14px;"
            f" box-shadow: {_CARD_SHADOW};")
    # cardurile HTML vechi au fundalul scris inline; după themed_html() el conține „var(--tk-surface)”
    inline = ", ".join(f'div[style*="{p}var(--tk-surface)"]' for p in (
        "background:", "background: ", "background-color:", "background-color: ",
        "linear-gradient(90deg, ", "linear-gradient(135deg, "))
    return (
        '[data-testid="stAppViewContainer"] { background:'
        " radial-gradient(1100px 420px at 50% -160px, var(--tk-glow), transparent 70%), var(--tk-bg); }"
        f" .fin-card, .news-card, div[data-testid=\"stMetric\"] {{ {card} }}"
        # metricile (un singur număr): mai compacte și centrate (cerut de Daniel pe 10.11, 19:36)
        ' div[data-testid="stMetric"] { padding: 10px 12px; text-align: center; }'
        ' div[data-testid="stMetric"] [data-testid="stMetricLabel"],'
        ' div[data-testid="stMetric"] [data-testid="stMetricValue"],'
        ' div[data-testid="stMetric"] [data-testid="stMetricDelta"] { justify-content: center; text-align: center; }'
        ' div[data-testid="stMetric"] [data-testid="stMetricLabel"] { grid-template-columns: auto auto; }'
        ' div[data-testid="stMetric"] div:has(> [data-testid="stMetricDelta"]) { justify-content: center; }'
        " .fin-card:hover, .news-card:hover { border-color: var(--tk-border-strong); }"
        ' [data-testid="stExpander"] details { border: 1px solid var(--tk-border); border-radius: 14px;'
        " background: var(--tk-surface); }"
        f" {inline} {{ box-shadow: {_CARD_SHADOW}; }}"
    )


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
