"""Calendarul publicărilor macro SUA din FRED (endpoint `fred/releases/dates`, cere cheie API). Funcții pure.

FRED dă doar ziua publicării, fără oră. Valorile publicate rămân în tabelul „Barometrul Economiei Reale”
(aceleași serii FRED); calendarul spune doar când se actualizează.

Excepție: deciziile FOMC. Publicarea FRED „FOMC Press Release” conține seriile zilnice ale intervalului țintă
(DFEDTARU/DFEDTARL), deci FRED o listează în fiecare zi; datele ședințelor se iau din calendarul oficial al Fed.
"""
from datetime import date, timedelta

FRED_RELEASE_DATES_URL = "https://api.stlouisfed.org/fred/releases/dates"
CALENDAR_DAYS = 14
WEEKDAYS_RO = ("luni", "marți", "miercuri", "joi", "vineri", "sâmbătă", "duminică")

# (cheie, nume afișat, numele publicării în FRED, ce actualizează în „Barometrul Economiei Reale”)
# Numele FRED se potrivesc exact (fără majuscule și spații la capete). Dacă FRED redenumește o publicare,
# ea apare la „fără dată în interval”, iar lista completă din interval e afișată pentru verificare.
TRACKED = (
    ("cpi", "Inflația CPI", "Consumer Price Index", "CPI General, Core CPI"),
    ("ppi", "Inflația PPI (prețuri de producție)", "Producer Price Index", "— (nu e în tabel)"),
    ("pce", "Venituri și cheltuieli (PCE)", "Personal Income and Outlays", "Core PCE"),
    ("nfp", "Piața muncii (NFP, șomaj)", "Employment Situation", "Nonfarm Payrolls, Rata Șomajului"),
    ("adp", "Raportul ADP", "ADP National Employment Report", "ADP Private Payrolls"),
    ("jolts", "Locuri de muncă vacante (JOLTS)", "Job Openings and Labor Turnover Survey", "JOLTS"),
    ("gdp", "PIB", "Gross Domestic Product", "— (nu e în tabel)"),
    ("retail", "Vânzări cu amănuntul", "Advance Monthly Sales for Retail and Food Services", "Vânzări de Retail"),
    ("indpro", "Producția industrială", "Industrial Production and Capacity Utilization", "Producție Industrială"),
    ("houst", "Construcții noi de locuințe", "New Residential Construction", "Housing Starts"),
    ("umich", "Sentimentul consumatorilor (U. Mich)", "Surveys of Consumers", "Sentiment Consumatori"),
)

# Deciziile FOMC: a doua zi a fiecărei ședințe programate, comunicat la 14:00 ora New York.
# Sursa: federalreserve.gov/monetarypolicy/fomccalendars.htm (verificat 05.11.2026; 2027 = program provizoriu,
# confirmat la ședința precedentă). Ședințele neprogramate nu apar. Lista se actualizează anual (Fed publică anul
# următor de obicei în august–septembrie); după ultima dată, `fomc_outdated` semnalează lipsa.
FOMC_LABEL = "Decizia de dobândă FOMC (14:00 ora New York)"
FOMC_UPDATES = "Dobânda Cheie FED (media lunară; se schimbă la publicarea lunii)"
FOMC_DECISIONS = (
    date(2026, 1, 28), date(2026, 3, 18), date(2026, 4, 29), date(2026, 6, 17),
    date(2026, 7, 29), date(2026, 9, 16), date(2026, 10, 28), date(2026, 12, 9),
    date(2027, 1, 27), date(2027, 3, 17), date(2027, 4, 28), date(2027, 6, 9),
    date(2027, 7, 28), date(2027, 9, 15), date(2027, 10, 27), date(2027, 12, 8),
)


def _norm(name):
    return " ".join(str(name or "").split()).casefold()


def _to_date(value):
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def window(today, days=CALENDAR_DAYS):
    """(prima zi, ultima zi) a intervalului cerut lui FRED: de azi, inclusiv, pe `days` zile."""
    return today, today + timedelta(days=days - 1)


def fomc_in_window(today, days=CALENDAR_DAYS, decisions=FOMC_DECISIONS):
    """Datele deciziilor FOMC din intervalul [today, today + days − 1], sortate."""
    start, end = window(today, days)
    return sorted(d for d in decisions if start <= d <= end)


def fomc_outdated(today, days=CALENDAR_DAYS, decisions=FOMC_DECISIONS):
    """True când intervalul trece de ultima decizie FOMC din listă: lista trebuie actualizată."""
    return not decisions or window(today, days)[1] > max(decisions)


def upcoming(release_dates, today, days=CALENDAR_DAYS):
    """Publicările urmărite din interval: cele din FRED (TRACKED) plus deciziile FOMC (FOMC_DECISIONS).

    release_dates: lista `release_dates` din răspunsul FRED (dict cu release_name și date „AAAA-LL-ZZ”).
    „FOMC Press Release” din FRED e ignorată (apare zilnic, vezi docstring-ul modulului).
    Întoarce (rows, missing): rows = dict date, weekday, in_days, label, fred_name, updates, sortate după dată
    (o publicare apare o singură dată pe zi); missing = numele afișate ale publicărilor urmărite fără dată în interval.
    """
    start, end = window(today, days)
    by_name = {_norm(fred): (key, label, fred, upd) for key, label, fred, upd in TRACKED}
    seen, rows = set(), []
    for item in release_dates or []:
        if not isinstance(item, dict):
            continue
        hit = by_name.get(_norm(item.get("release_name")))
        d = _to_date(item.get("date"))
        if hit is None or d is None or not (start <= d <= end) or (hit[0], d) in seen:
            continue
        seen.add((hit[0], d))
        rows.append({"date": d, "weekday": WEEKDAYS_RO[d.weekday()], "in_days": (d - today).days,
                     "label": hit[1], "fred_name": hit[2], "updates": hit[3], "key": hit[0]})
    for d in fomc_in_window(today, days):
        rows.append({"date": d, "weekday": WEEKDAYS_RO[d.weekday()], "in_days": (d - today).days,
                     "label": FOMC_LABEL, "fred_name": None, "updates": FOMC_UPDATES, "key": "fomc"})
    rows.sort(key=lambda r: (r["date"], r["label"]))
    found = {r["key"] for r in rows}
    missing = [label for key, label, _, _ in TRACKED if key not in found]
    if "fomc" not in found:
        missing.append("Decizia de dobândă FOMC")
    return rows, missing


def all_release_names(release_dates):
    """Numele distincte ale tuturor publicărilor din răspuns, sortate (pentru verificarea potrivirii)."""
    names = {str(i.get("release_name")).strip() for i in (release_dates or [])
             if isinstance(i, dict) and i.get("release_name")}
    return sorted(names, key=str.casefold)


def error_text(status_code, body, api_key):
    """Mesajul de eroare FRED pentru interfață, fără cheia API (înlocuită cu ***)."""
    msg = ""
    if isinstance(body, dict):
        msg = str(body.get("error_message") or "")
    text = f"FRED a răspuns HTTP {status_code}" + (f": {msg}" if msg else "")
    return text.replace(api_key, "***") if api_key else text


def near(rows, days=2):
    """Publicările de azi și de mâine (in_days < days) din `upcoming` și, dacă nu e niciuna, prima următoare.
    Întoarce (lista de azi/mâine, următoarea publicare sau None)."""
    rows = list(rows or [])
    soon = [r for r in rows if 0 <= r.get("in_days", -1) < days]
    later = [r for r in rows if r.get("in_days", -1) >= days]
    return soon, (None if soon or not later else later[0])
