"""Utilitare de citire sigură a datelor. Fără Streamlit, fără rețea."""
import calendar
import csv
import html
import io
import re
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

RO_TZ = ZoneInfo("Europe/Bucharest")


# --- Timp -------------------------------------------------------------------

def now_ro():
    """Ora curentă în România. Serverul Streamlit Cloud rulează în UTC."""
    return datetime.now(RO_TZ)


def struct_time_utc_to_ro(struct_time):
    """Convertește un `time.struct_time` în UTC (cum întoarce feedparser) în ora României."""
    return datetime.fromtimestamp(calendar.timegm(struct_time), tz=RO_TZ)


# --- Valori numerice --------------------------------------------------------

def num(mapping, key):
    """Valoarea numerică de la `key` sau None.

    `info.get(key, 0)` din yfinance nu protejează: cheia poate exista cu valoarea
    None. Aici lipsa, None, NaN și textul nenumeric întorc toate None, niciodată 0.
    """
    if not mapping:
        return None
    value = mapping.get(key)
    if value is None or isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if value != value or value in (float("inf"), float("-inf")):
        return None
    return value


# --- Prețuri din yf.download ------------------------------------------------

def field_frame(data, field, tickers):
    """Extrage un câmp (ex. 'Close') din rezultatul `yf.download`, ca DataFrame
    cu o coloană per simbol.

    Tratează toate formele întoarse de yfinance: coloane simple (un simbol, fără
    MultiIndex), MultiIndex (câmp, simbol) și MultiIndex (simbol, câmp) de la
    `group_by='ticker'`. Simbolurile fără nicio valoare sunt eliminate.
    """
    if isinstance(tickers, str):
        tickers = [tickers]
    tickers = list(tickers)
    if data is None or len(data) == 0:
        return pd.DataFrame()

    cols = data.columns
    if isinstance(cols, pd.MultiIndex):
        if field in cols.get_level_values(0):
            out = data[field]
        elif field in cols.get_level_values(1):
            out = data.xs(field, axis=1, level=1)
        else:
            return pd.DataFrame()
    else:
        if field not in cols:
            return pd.DataFrame()
        out = data[field]

    if isinstance(out, pd.Series):
        out = out.to_frame(name=tickers[0] if tickers else field)
    out = out.copy()
    out.columns = [str(c) for c in out.columns]
    return out.dropna(axis=1, how="all").dropna(axis=0, how="all")


def close_frame(data, tickers):
    """Prețurile de închidere, o coloană per simbol. Vezi `field_frame`."""
    return field_frame(data, "Close", tickers)


# --- Ferestre de timp -------------------------------------------------------

WINDOW_OFFSETS = {
    "1S": pd.DateOffset(days=7),
    "1L": pd.DateOffset(months=1),
    "3L": pd.DateOffset(months=3),
    "6L": pd.DateOffset(months=6),
    "1A": pd.DateOffset(years=1),
    "3A": pd.DateOffset(years=3),
    "5A": pd.DateOffset(years=5),
    "1 Lună": pd.DateOffset(months=1),
    "3 Luni": pd.DateOffset(months=3),
    "6 Luni": pd.DateOffset(months=6),
    "1 An": pd.DateOffset(years=1),
    "3 Ani": pd.DateOffset(years=3),
    "5 Ani": pd.DateOffset(years=5),
}


def slice_window(obj, label, default="1A"):
    """Ultima fereastră calendaristică dintr-o serie/DataFrame cu index de date.

    `label` este o cheie din WINDOW_OFFSETS ("1L", "1A", "3 Luni", ...).
    "1Z" întoarce ultimele două ședințe (ziua curentă și cea precedentă).

    Tăierea se face pe date, nu pe număr de rânduri: un an înseamnă ~252 de
    ședințe, nu 365 de rânduri.
    """
    if obj is None or len(obj) == 0:
        return obj
    if label == "1Z":
        return obj.iloc[-2:]
    offset = WINDOW_OFFSETS.get(label, WINDOW_OFFSETS[default])
    start = obj.index[-1] - offset
    return obj.loc[obj.index >= start]


# --- Numere din Google Sheets -----------------------------------------------

def smart_to_float(val):
    """Transformă un număr scris în format US sau RO/EU în float.

    Exemple: "1.000,50" -> 1000.5, "1,000.50" -> 1000.5, "50,5" -> 50.5,
    "3,17%" -> 3.17, "0,8699 lei" -> 0.8699. Valorile deja numerice trec neschimbate.
    Celulele goale, textul nenumeric și erorile de foaie ("#DIV/0!") dau 0.0.

    Limită cunoscută: un singur separator e ambiguu ("1.250" poate fi 1,25 sau 1250).
    Un singur punct e citit ca zecimală US, o singură virgulă ca zecimală RO.
    """
    if isinstance(val, bool):
        return 0.0
    if isinstance(val, (int, float)):
        return 0.0 if val != val else float(val)
    if val is None or pd.isna(val) or val == '':
        return 0.0
    s = str(val).strip()
    # Păstrăm doar cifre, punct, virgulă și minus
    s = re.sub(r'[^\d.,-]', '', s)
    if not s:
        return 0.0

    # Logică de detecție a formatului
    if ',' in s and '.' in s:
        if s.rfind(',') > s.rfind('.'):  # Format EU: 1.000,50
            s = s.replace('.', '').replace(',', '.')
        else:  # Format US: 1,000.50
            s = s.replace(',', '')
    elif ',' in s:
        if s.count(',') > 1:  # US Thousands: 1,000,000
            s = s.replace(',', '')
        else:  # RO Decimal: 50,5
            s = s.replace(',', '.')
    elif '.' in s:
        if s.count('.') > 1:  # RO Thousands: 1.000.000
            s = s.replace('.', '')
        # Altfel e US Decimal: 50.5

    try:
        return float(s)
    except ValueError:
        return 0.0


# --- BCE (Data Portal, format csvdata) --------------------------------------

def parse_ecb_csv(text):
    """Ultima observație dintr-un răspuns CSV al BCE: (perioadă, valoare) sau None.

    Răspunsul are antet și coloanele TIME_PERIOD (ex. "2026-08") și OBS_VALUE.
    Rândurile fără valoare numerică sunt sărite. Orice altă formă (HTML de eroare,
    text gol, coloane lipsă) dă None, niciodată o valoare presupusă.
    """
    if not text or not isinstance(text, str):
        return None
    try:
        rows = list(csv.DictReader(io.StringIO(text)))
    except csv.Error:
        return None
    valid = []
    for row in rows:
        period = (row.get("TIME_PERIOD") or "").strip()
        try:
            value = float(row.get("OBS_VALUE"))
        except (TypeError, ValueError):
            continue
        if period and value == value:
            valid.append((period, value))
    if not valid:
        return None
    return max(valid, key=lambda item: item[0])


def parse_ecb_series(text):
    """Toate observațiile dintr-un răspuns CSV al BCE, ca serie indexată pe perioadă (text, ex. "2026-08"
    sau "2026-08-14"), sortată crescător. Serie goală dacă răspunsul nu are forma așteptată."""
    if not text or not isinstance(text, str):
        return pd.Series(dtype=float)
    try:
        rows = list(csv.DictReader(io.StringIO(text)))
    except csv.Error:
        return pd.Series(dtype=float)
    out = {}
    for row in rows:
        period = (row.get("TIME_PERIOD") or "").strip()
        try:
            value = float(row.get("OBS_VALUE"))
        except (TypeError, ValueError):
            continue
        if period and value == value:
            out[period] = value
    return pd.Series(out, dtype=float).sort_index()


# --- Eurostat (API de diseminare, JSON-stat 2.0) -----------------------------

def parse_eurostat_monthly(obj):
    """Seria lunară dintr-un răspuns JSON-stat Eurostat filtrat pe o singură combinație (o țară, un indicator):
    indexată pe perioada "AAAA-LL", sortată crescător. Toate dimensiunile în afară de `time` trebuie să aibă
    un singur element; altfel (sau la orice formă neașteptată) serie goală, nu o valoare ghicită."""
    try:
        ids, sizes = list(obj["id"]), list(obj["size"])
        if any(size != 1 for dim, size in zip(ids, sizes) if dim != "time"):
            return pd.Series(dtype=float)
        times = obj["dimension"]["time"]["category"]["index"]
        values = obj["value"]
    except (KeyError, TypeError):
        return pd.Series(dtype=float)
    out = {}
    for label, pos in times.items():
        raw = values.get(str(pos)) if isinstance(values, dict) else (values[pos] if pos < len(values) else None)
        try:
            v = float(raw)
        except (TypeError, ValueError):
            continue
        m = re.fullmatch(r"(\d{4})-?M?(\d{2})", str(label))
        if m and v == v:
            out[f"{m.group(1)}-{m.group(2)}"] = v
    return pd.Series(out, dtype=float).sort_index()


def positive_or_none(value):
    """Prețul țintă dintr-o celulă de foaie: număr strict pozitiv sau None.

    `smart_to_float` dă 0 pentru celula goală, text sau eroare de foaie; un preț țintă
    de 0 nu există, deci 0, negativul și NaN înseamnă „fără țintă", nu „țintă 0".
    """
    number = smart_to_float(value)
    if number is None or number != number or number <= 0:
        return None
    return float(number)


def entry_target_view(price, target):
    """Textele și culoarea cardului „Țintă intrare": (text țintă, status, culoare).

    Fără țintă (None) cardul arată „N/A" și nu calculează nicio distanță: înainte, ținta
    lipsă apărea ca „0.00" cu „+0.0% peste țintă". Cu țintă: verde dacă prețul e la sau sub
    ea, galben dacă e la mai puțin de 5% peste, gri altfel.
    """
    if target is None or target != target or target <= 0:
        return "N/A", "Fără preț țintă în watchlist", "#8B949E"
    if price <= target:
        return f"{target:.2f}", "🚀 ZONĂ ACHIZIȚIE", "#3FB950"
    dist_pct = (price - target) / target * 100
    color = "#D29922" if dist_pct < 5 else "#8B949E"
    return f"{target:.2f}", f"⏳ +{dist_pct:.1f}% peste țintă", color


def scale_number(value):
    """Număr scalat pentru afișare: „4.97 T", „76.89 B", „57.07 M" sau „1,234.50".
    Pragul se compară cu modulul, ca valorile negative mari (o pierdere de 11,29 mld.)
    să fie scalate la fel ca cele pozitive, cu semnul păstrat."""
    size = abs(value)
    if size >= 1e12:
        return f"{ro(value / 1e12):.2f} T"
    if size >= 1e9:
        return f"{ro(value / 1e9):.2f} B"
    if size >= 1e6:
        return f"{ro(value / 1e6):.2f} M"
    return f"{ro(value):,.2f}"


def watchlist_status(price, target):
    """Rândul din tabelul Watchlist: (preț, țintă, distanță %, e în zona de cumpărare).

    Prețul sau ținta lipsă (None, NaN, ≤ 0) rămân None și nu produc nici distanță, nici semnal:
    un preț 0 ar arăta ca o scădere de 100% până sub orice țintă. Distanța e față de țintă,
    (preț − țintă) / țintă × 100, la fel ca în cardul „Țintă intrare”.
    """
    price = positive_or_none(price)
    target = positive_or_none(target)
    if price is None or target is None:
        return price, target, None, False
    return price, target, (price - target) / target * 100, price <= target


def md_bold_html(text):
    """Text pentru o casetă HTML (st.markdown cu unsafe_allow_html): caracterele HTML se escapează,
    iar **bold** din Markdown, care nu se interpretează în interiorul unui <div>, devine <b>bold</b>."""
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", html.escape(str(text), quote=False))


def ro_separators(text):
    """Separatorii unui număr formatat american, schimbați în cei românești: „1,234.5” → „1.234,5”."""
    return text.replace(",", "\x00").replace(".", ",").replace("\x00", ".")


class ro:
    """Număr afișat în format românesc direct dintr-un f-string, cu specificația obișnuită.

    f"{ro(x):,.2f}" → „1.234,56”; f"{ro(x):+.1%}" → „+12,3%”; f"{ro(n):,}" → „12.345”.
    Schimbă doar afișarea: valoarea nu se modifică. O valoare nenumerică (None, text) dă
    aceeași eroare ca f"{x:.2f}", ca să nu ascundă o lipsă tratată greșit în apelant.
    """
    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value

    def __format__(self, spec):
        return ro_separators(format(self.value, spec))


def ro_spec(spec, suffix=""):
    """Formatter pentru pandas Styler / map: ro_spec(",.2f")(1234.5) → „1.234,50”; lipsă → „N/A”.

    În Styler păstrează valorile numerice dedesubt, deci sortarea pe coloană rămâne numerică.
    `suffix` se adaugă doar la numere („ USD”, „%”), nu și la „N/A”.
    """
    def _fmt(v):
        if v is None or isinstance(v, str):
            return "N/A" if v is None else v
        try:
            if pd.isna(v):
                return "N/A"
        except (TypeError, ValueError):
            pass
        return format(ro(v), spec) + suffix
    return _fmt


def format_ro(value, decimals=2, suffix="", trim=False):
    """Număr în format românesc: punct la mii, virgulă la zecimale („1.250”, „4.562,50”).

    trim=True scoate zerourile de la final („1.250,0000” → „1.250”, „0,3048” rămâne).
    Lipsă, NaN sau text nenumeric → „N/A”, niciodată 0.
    """
    if value is None or isinstance(value, bool):
        return "N/A"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if v != v or v in (float("inf"), float("-inf")):
        return "N/A"
    text = f"{v:,.{decimals}f}"
    if trim and "." in text:
        text = text.rstrip("0").rstrip(".")
    text = ro_separators(text)
    if text in ("-0", "-0,00") or (text.startswith("-") and set(text[1:]) <= set("0,.")):
        text = text[1:]
    return text + suffix
