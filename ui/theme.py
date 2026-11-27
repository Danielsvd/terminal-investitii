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

# (luminos, întunecat) — luminos = „Neon Carbon”, întunecat = „Nordic Frost” (decizia din 27.11, DESIGN.md §1.0)
TOKENS = {
    "bg": ("#EEF1EC", "#0E1620"),
    "surface": ("#FFFFFF", "#16212D"),
    "surface-top": ("#FFFFFF", "#1A2734"),      # ziua plat; noaptea gradient fin de sus în jos
    "surface-2": ("#F4F6F2", "#1D2A38"),
    "border": ("#DCE2DA", "#273647"),
    "border-strong": ("#C6CFC3", "#34485C"),
    "text": ("#0F1411", "#E8EEF4"),
    "muted": ("#566059", "#9DAEC0"),
    "accent": ("#0F1411", "#7CC4E8"),          # ziua: cerneală; noaptea: albastru-gheață
    "accent-text": ("#3D5E0B", "#9AD3F0"),
    "on-accent": ("#C6F432", "#FFFFFF"),       # textul butonului principal: lime pe cerneală / alb pe fiord
    "up": ("#116B37", "#6EE7B7"),
    "down": ("#C0262D", "#FCA5A5"),
    "warn": ("#8A5A00", "#FCD34D"),
    # textul de pe un fundal plin colorat (verdict, GEX)
    "on-fill": ("#FFFFFF", "#0E1620"),
    # adâncime: ziua plat (transparente); noaptea linie luminoasă sus, umbră moale, strălucire rece
    "highlight": ("#FFFFFF00", "#FFFFFF0D"),
    "shadow": ("#00000000", "#02081266"),
    "glow": ("#00000000", "#7CC4E812"),
}

# Colțurile cardurilor, metricilor și expanderelor și ale opțiunilor din meniu: comune celor două moduri
# (CSS nu poate alege o lungime după mod), la mijloc între Nordic (18/12) și Neon (24/16)
CARD_RADIUS = "20px"
NAV_RADIUS = "14px"

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
DF_HIGHLIGHT_BG = "rgba(124, 196, 232, 0.18)"  # rândul evidențiat (compania analizată, mediana): albastru translucid, lizibil în ambele moduri

CORR_NEG = "#3987E5"   # corelație negativă (pozițiile se compensează): albastru
CORR_POS = "#D95926"   # corelație pozitivă (se mișcă împreună): portocaliu


def corr_heatmap_colors(mode):
    """(colorscale, culoarea textului) pentru matricile de corelație (zmin −1, zmax 1).
    Capetele rămân albastru / portocaliu în ambele moduri; mijlocul (0) e `surface-2` al modului, iar textul
    din celule e `text` al modului: întunecat pe alb (≥ 4,7:1 pe tot intervalul), alb noaptea.
    Mod necunoscut: scala veche, cu mijloc gri închis și text alb (lizibilă, dar un bloc închis ziua)."""
    if mode in ("light", "dark"):
        i = 0 if mode == "light" else 1
        mid, txt = TOKENS["surface-2"][i], ("#FFFFFF" if mode == "dark" else TOKENS["text"][0])
    else:
        mid, txt = "#383835", "#FFFFFF"
    return [[0.0, CORR_NEG], [0.5, mid], [1.0, CORR_POS]], txt


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


# radio-ul de navigare din bara laterală (singurul st.radio din sidebar)
_NAV = 'section[data-testid="stSidebar"] [data-testid="stRadio"]'

_CARD_SHADOW = "inset 0 1px 0 var(--tk-highlight), 0 1px 2px var(--tk-shadow), 0 10px 28px var(--tk-shadow)"


def premium_css():
    """Tratamentul „premium” al suprafețelor (DESIGN.md §4): gradient fin pe carduri, contur de 1px,
    linie luminoasă sus, umbră moale, colțuri `CARD_RADIUS` și o strălucire discretă în capul paginii
    (valorile în TOKENS: highlight, shadow, glow).
    Se injectează DUPĂ CSS-ul vechi din app.py, ca să-l înlocuiască la aceeași specificitate."""
    card = ("background: linear-gradient(180deg, var(--tk-surface-top) 0%, var(--tk-surface) 100%);"
            f" border: 1px solid var(--tk-border); border-radius: {CARD_RADIUS};"
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
        # valoarea principală colorată după semn (st.metric colorează doar delta): containerul cu cheia
        # „semn-poz-…” / „semn-neg-…” primește clasa st-key-…, vezi signed_key()
        ' [class*="st-key-semn-poz-"] [data-testid="stMetricValue"] { color: var(--tk-up); }'
        ' [class*="st-key-semn-neg-"] [data-testid="stMetricValue"] { color: var(--tk-down); }'
        # explicațiile „?” încap pe lățimea telefonului (pe S25 ieșeau din ecran: containerul are
        # width: max-content, iar max-width-ul Streamlit e relativ la el, nu la ecran)
        ' [data-testid="stTooltipContent"] { max-width: min(560px, calc(100vw - 24px)) !important;'
        " white-space: normal !important; overflow-wrap: anywhere; }"
        " .fin-card:hover, .news-card:hover { border-color: var(--tk-border-strong); }"
        f' [data-testid="stExpander"] details {{ border: 1px solid var(--tk-border); border-radius: {CARD_RADIUS};'
        " background: var(--tk-surface); }"
        f" {inline} {{ box-shadow: {_CARD_SHADOW}; }}"
        # bannerul „Mediul de piață SUA”: pe telefon verdictul și scorul se pun unul sub altul, centrate,
        # ca cardurile VIX și Fear & Greed de dedesubt (cerut de Daniel pe 14.11); pe desktop rămâne neschimbat
        " @media (max-width: 640px) { .macro-hero { justify-content: center !important; text-align: center; }"
        " .macro-hero > div { flex-basis: 100% !important; } }"
        # meniul lateral: fiecare opțiune e un rând de min. 44px (ținta de atingere pe telefon), text mai mare,
        # fundal la hover și rândul ales evidențiat cu accentul (cerut de Daniel pe 15.11). Streamlit 1.65 desenează
        # radio-ul cu react-aria: grupul = stRadioGroup, opțiunea = <label data-testid="stRadioOption"> cu data-selected
        # butonul principal: textul din on-accent (Streamlit îl face alb; pe lime nu s-ar citi)
        ' [data-testid^="stBaseButton-primary"], [data-testid^="stBaseButton-primary"] p {'
        " color: var(--tk-on-accent) !important; }"
        f' {_NAV} [data-testid="stRadioGroup"] {{ align-items: stretch; gap: 4px; }}'
        f' {_NAV} [data-testid="stRadioOption"] {{ width: 100%; min-height: 44px; display: flex; align-items: center;'
        f" box-sizing: border-box; padding: 8px 12px; border: 1px solid transparent; border-radius: {NAV_RADIUS};"
        " transition: background-color .15s, border-color .15s; }"
        f' {_NAV} [data-testid="stRadioOption"]:hover {{ background: var(--tk-surface-2); }}'
        f' {_NAV} [data-testid="stRadioOption"][data-selected] {{'
        " background: color-mix(in srgb, var(--tk-accent) 16%, transparent);"
        " border-color: color-mix(in srgb, var(--tk-accent) 45%, transparent); }"
        f' {_NAV} [data-testid="stRadioOption"] [data-testid="stMarkdownContainer"] p {{'
        " font-size: 1.05rem; line-height: 1.35; }"
        f' {_NAV} [data-testid="stRadioOption"] [data-testid="stIconMaterial"] {{ font-size: 1.3rem; }}'
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
    "#446E9E": _GRID,                    # grila albastră a graficului de performanță relativă
}

# Culorile seriilor în grafice, după modul temei (luminos, întunecat). Liniile trebuie să aibă contrast ≥ 3:1
# pe fundal (WCAG, obiecte grafice): culorile vechi gândite pentru fundal închis (portocaliu, chihlimbar,
# gri deschis, violet deschis) dispăreau pe alb, iar `purple` dispărea pe fundalul de noapte.
# Nuanțele „blue/amber/violet/teal” sunt cele din `chartCategoricalColors` din config.toml.
PLOT_TOKENS = {
    "text": TOKENS["text"], "muted": TOKENS["muted"],
    "surface": TOKENS["surface"], "surface-2": TOKENS["surface-2"], "border-strong": TOKENS["border-strong"],
    "up": TOKENS["up"], "down": TOKENS["down"], "warn": TOKENS["warn"],
    "blue": ("#1D4ED8", "#7CC4E8"),
    "amber": ("#B45309", "#F4B26A"),
    "violet": ("#7C3AED", "#B4A7F5"),
    "teal": ("#0F766E", "#7CC4E8"),
}
# culoare scrisă în cod → token din PLOT_TOKENS (doar cu modul cunoscut). Paletele de categorii
# (#3987E5, #199E70, #C98500, #D95926...) NU sunt aici: s-ar contopi seriile între ele.
PLOT_MODE_MAP = {
    "#C9D1D9": "text", "#B0B8C4": "text",
    "#8B949E": "muted", "#6E7681": "muted",
    "#161B22": "surface", "#21262D": "surface-2", "#484F58": "border-strong",
    "GREEN": "up",
    "#F85149": "down", "#DA3633": "down", "RED": "down",
    "#D29922": "warn", "#DBAB09": "warn", "#FFAB00": "warn", "YELLOW": "warn",
    "#58A6FF": "blue",
    "ORANGE": "amber",
    "PURPLE": "violet", "#BF91FF": "violet", "#A371F7": "violet",
    "CYAN": "teal", "#00E5FF": "teal",
}
_PLOT_MODE_RGB = {tuple(int(h[i:i + 2], 16) for i in (1, 3, 5)): tok
                  for h, tok in PLOT_MODE_MAP.items() if h.startswith("#")}
# alb doar unde e contur/tick pe fundalul graficului; textul alb de pe celulele colorate rămâne alb
_WHITE_KEYS = {"tickcolor"}


# Verdele de creștere din paleta veche, scris direct în grafice (bare, lumânări, linii): devine verdele
# temei active (`up`, același ca în tabele și carduri). Verdele din paletele de categorii nu e aici.
PLOT_UP_HEX = {"#3FB950", "#238636"}
_PLOT_UP_RGB = (63, 185, 80)
_RGBA_FULL = re.compile(r"^rgba\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*([0-9.]+)\s*\)$", re.I)


def _rgb(hex_color):
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _plot_mode_color(val, mode):
    """Culoarea din PLOT_TOKENS pentru `mode` ("light"/"dark"), sau None dacă `val` nu e în PLOT_MODE_MAP.
    `rgba(...)` cu nuanța unei culori din hartă păstrează transparența."""
    i = 0 if mode == "light" else 1
    norm = val.strip().upper()
    tok = PLOT_MODE_MAP.get(norm)
    if tok is not None:
        return PLOT_TOKENS[tok][i]
    m = _RGBA_FULL.match(val.strip())
    if m:
        tok = _PLOT_MODE_RGB.get(tuple(int(m.group(k)) for k in (1, 2, 3)))
        if tok is not None:
            r, g, b = _rgb(PLOT_TOKENS[tok][i])
            return f"rgba({r}, {g}, {b}, {m.group(4)})"
    return None


def _plot_color(key, val, up=None, mode=None):
    norm = val.strip().upper()
    if norm in ("WHITE", "#FFFFFF"):
        return "rgba(128,128,128,0.6)" if key in _WHITE_KEYS else val
    if mode in ("light", "dark"):
        themed = _plot_mode_color(val, mode)
        if themed is not None:
            return themed
    if up is not None:
        if norm in PLOT_UP_HEX:
            return up
        m = _RGBA_FULL.match(val.strip())
        if m and tuple(int(m.group(i)) for i in (1, 2, 3)) == _PLOT_UP_RGB:
            r, g, b = _rgb(up)
            return f"rgba({r}, {g}, {b}, {m.group(4)})"
    return PLOT_COLOR_MAP.get(norm, val)


def theme_figure_dict(obj, key=None, up=None, mode=None):
    """Copie a dicționarului unei figuri Plotly, cu culorile pentru fundal închis înlocuite (PLOT_COLOR_MAP).
    Cu `up` (hex, verdele temei active): verdele vechi de creștere și lumânările în creștere primesc `up`.
    Cu `mode` ("light"/"dark", din `st.context.theme.type`): culorile din PLOT_MODE_MAP primesc nuanța
    modului (PLOT_TOKENS); fără mod (necunoscut) rămân ca înainte.
    Nu modifică datele, doar valorile text ale cheilor care conțin „color”."""
    if isinstance(obj, dict):
        out = {k: theme_figure_dict(v, k, up, mode) for k, v in obj.items()}
        if up is not None and out.get("type") == "candlestick":
            inc = dict(out.get("increasing") or {})
            line = dict(inc.get("line") or {})
            line.setdefault("color", up)
            inc["line"] = line
            out["increasing"] = inc
        return out
    if isinstance(obj, list):
        return [theme_figure_dict(v, key, up, mode) for v in obj]
    if isinstance(obj, str) and key is not None and "color" in key.lower():
        return _plot_color(key, obj, up, mode)
    return obj


def signed_key(name, value):
    """Cheia unui `st.container(key=...)` care colorează valoarea `st.metric` din el după semnul lui `value`
    (CSS în premium_css): pozitiv → verde, negativ → roșu, zero sau lipsă → culoarea normală.
    `name` trebuie să fie unic pe pagină (cheile Streamlit nu se pot repeta)."""
    slug = re.sub(r"[^a-z0-9]+", "-", str(name).lower()).strip("-")
    if value is None or value != value or value == 0:
        return f"semn-zero-{slug}"
    return f"semn-{'poz' if value > 0 else 'neg'}-{slug}"
