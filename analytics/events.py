"""„Radar evenimente” (decizia lui Daniel, 14.11.2026: doar reguli, fără AI; jurnal `Evenimente_jurnal` da;
SEC 8-K mai târziu). Propunerea: claude/propunere_radar_evenimente_2026-11-14.md. Funcții pure: fără Streamlit,
fără rețea.

Fluxul:
  1. titlurile (RSS) din ultimele 48 de ore, fără dubluri;
  2. potrivirea cu companiile (data/entities.py) și taxonomia evenimentelor materiale (RO + EN, doar pe titlu);
  3. direcția pentru companie din tipul evenimentului; beneficiarii = concurenții din grupul îngust, doar la
     evenimentele „de cotă de piață”; evenimente sectoriale (petrol, gaze, taxe pe bănci, apărare...);
  4. reacția pieței din prețuri: randament anormal față de benchmark în multipli de volatilitate și volum relativ;
  5. mișcări mari fără știre potrivită (Chan, 2003: după mișcări fără știre prețul tinde să revină);
  6. jurnalul: direcția așteptată pe fiecare companie, evaluată la 5 și 20 de ședințe.

Limite: regulile citesc cuvinte, nu sens (negații, ironie, „X dă în judecată Y” au direcții ambigue); titlurile vin
cu întârziere față de piață; grupurile de concurenți sunt fixe. Pragurile sunt netestate până la jurnal.
"""
import hashlib
import math
import re
import unicodedata
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

UNTESTED = "netestat în aplicație"
MAX_AGE_HOURS = 48


def normalize(text):
    """Litere mici, fără diacritice (ș → s, ț → t, ă/â → a, î → i), spații comprimate."""
    t = unicodedata.normalize("NFKD", str(text or "")).encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"\s+", " ", t).strip()


def _kw_regex(words):
    """O expresie regulată pentru cuvintele/expresiile (normalizate), ca cuvinte întregi."""
    alts = sorted({normalize(w) for w in words if w}, key=len, reverse=True)
    return re.compile(r"(?<![a-z0-9])(?:" + "|".join(re.escape(a) for a in alts) + r")(?![a-z0-9])")


# --- Taxonomia evenimentelor pe companie ---------------------------------------------------------------------------
# direction: efectul așteptat asupra companiei numite (−1 / 0 ambiguu / +1); severity 1–3; share_shift: concurenții
# din grupul îngust pot câștiga cotă (doar la evenimente care scot compania din joc temporar sau îi strică imaginea).
EVENT_TYPES = {
    "rezolvare": {
        "label": "Rezolvare favorabilă (anchetă închisă, achitare)", "direction": 1, "severity": 1, "share_shift": False,
        "words": ("cleared of", "acquitted", "drops probe", "dropped probe", "ends probe", "charges dropped",
                  "dismissed lawsuit", "lawsuit dismissed", "wins lawsuit", "wins appeal", "achitat", "achitata",
                  "dosar clasat", "a castigat procesul", "castiga procesul", "inchide ancheta")},
    "frauda_ancheta": {
        "label": "Anchetă, fraudă sau scandal", "direction": -1, "severity": 3, "share_shift": True,
        "words": ("scandal", "fraud", "probe", "investigation", "investigated", "indicted", "indictment", "charged",
                  "espionage", "spying", "bribery", "raid", "raided", "whistleblower", "accounting irregularities",
                  "short seller report", "arrested", "ancheta", "anchetat", "anchetata", "frauda", "spionaj", "mita",
                  "perchezitii", "perchezitie", "procurorii", "dosar penal", "diicot", "retinut", "arestat", "inculpat", "trimis in judecata")},
    "interdictie": {
        "label": "Interdicție, sancțiuni, restricții de export", "direction": -1, "severity": 3, "share_shift": True,
        "words": ("ban", "bans", "banned", "export controls", "export curbs", "export restrictions", "entity list",
                  "blacklist", "blacklisted", "sanctions", "sanctioned", "embargo", "interdictie", "interzis",
                  "interzice", "sanctiuni", "sanctionat", "restrictii la export", "embargou")},
    "operational": {
        "label": "Problemă operațională (rechemare, accident, atac, grevă)", "direction": -1, "severity": 2,
        "share_shift": True,
        "words": ("recall", "recalls", "recalled", "outage", "data breach", "breach", "hack", "hacked", "cyberattack",
                  "cyber attack", "ransomware", "strike", "walkout", "explosion", "fire at", "accident",
                  "grounded", "halts production", "production halt", "shutdown", "plant closure", "contamination",
                  "rechemare", "recheama", "avarie", "incendiu", "explozie", "greva", "atac cibernetic",
                  "oprirea productiei", "opreste productia", "intrerupere", "accident")},
    "juridic": {
        "label": "Litigiu, amendă, antitrust", "direction": -1, "severity": 2, "share_shift": False,
        "words": ("lawsuit", "sued", "sues", "fined", "penalty", "antitrust", "class action", "settlement",
                  "settle", "settles", "violation", "amenda", "amendat", "amendata", "dat in judecata", "proces penal",
                  "consiliul concurentei",
                  "sanctionat de anaf", "litigiu")},
    "conducere": {
        "label": "Plecare din conducere", "direction": -1, "severity": 1, "share_shift": False,
        "words": ("resigns", "resigned", "steps down", "stepping down", "ousted", "fired", "departure", "quits",
                  "demisie", "demisioneaza", "demisionat", "revocat", "revocare", "pleaca de la conducere")},
    "ghidaj_negativ": {
        "label": "Rezultate sau ghidaj sub așteptări", "direction": -1, "severity": 2, "share_shift": False,
        "words": ("profit warning", "cuts guidance", "cut guidance", "lowers guidance", "lowers outlook", "cuts outlook",
                  "misses estimates", "misses expectations", "weak guidance", "downgrade", "downgraded", "layoffs",
                  "job cuts", "loss widens", "avertisment", "scaderea profitului", "profitul scade", "profit in scadere",
                  "pierdere", "pierderi", "retrogradat", "concedieri", "disponibilizari")},
    "ghidaj_pozitiv": {
        "label": "Rezultate sau ghidaj peste așteptări", "direction": 1, "severity": 2, "share_shift": False,
        "words": ("raises guidance", "raised guidance", "lifts guidance", "raises outlook", "beats estimates",
                  "beats expectations", "tops estimates", "record profit", "record revenue", "upgrade", "upgraded",
                  "profit record", "profit in crestere", "profitul creste", "venituri record", "crestere a profitului",
                  "depaseste estimarile")},
    "contract": {
        "label": "Contract sau parteneriat important", "direction": 1, "severity": 1, "share_shift": False,
        "words": ("wins contract", "awarded", "contract worth", "partnership", "strikes deal", "signs deal",
                  "major order", "contract de", "a castigat licitatia", "castiga licitatia",
                  "parteneriat", "comanda de")},
    "fuziune": {
        "label": "Achiziție, fuziune, preluare (direcție ambiguă)", "direction": 0, "severity": 2, "share_shift": False,
        "words": ("acquire", "acquires", "acquisition", "merger", "merge", "takeover", "buyout", "bid for",
                  "achizitie", "achizitioneaza", "preia", "preluare", "fuziune", "oferta publica de cumparare")},
    "capital_pozitiv": {
        "label": "Răscumpărare sau dividend mărit", "direction": 1, "severity": 1, "share_shift": False,
        "words": ("buyback", "share repurchase", "raises dividend", "special dividend", "dividend hike",
                  "rascumparare", "dividend special", "majoreaza dividendul")},
    "dilutie": {
        "label": "Emisiune de acțiuni (diluare)", "direction": -1, "severity": 1, "share_shift": False,
        "words": ("share offering", "stock offering", "secondary offering", "capital increase", "dilution",
                  "majorare de capital", "majorarea capitalului", "emisiune de actiuni")},
}
_TYPE_ORDER = list(EVENT_TYPES)
_TYPE_RE = {k: _kw_regex(v["words"]) for k, v in EVENT_TYPES.items()}
_LEGAL = ("frauda_ancheta", "juridic", "interdictie")


def classify(title):
    """Tipurile de eveniment din titlu, în ordinea severității (cel mai grav primul). O rezolvare favorabilă
    („cleared”, „achitat”) anulează tipurile juridice din același titlu."""
    t = normalize(title)
    found = [k for k in _TYPE_ORDER if _TYPE_RE[k].search(t)]
    if "rezolvare" in found:
        found = [k for k in found if k not in _LEGAL]
    return sorted(found, key=lambda k: (-EVENT_TYPES[k]["severity"], _TYPE_ORDER.index(k)))


# --- Potrivirea cu companiile ---------------------------------------------------------------------------------------

class EntityMatcher:
    """Găsește simbolurile numite într-un titlu, cu o singură expresie regulată (cea mai lungă expresie câștigă).

    `symbols`: simbolurile urmărite; `names`: {simbol: (nume, alias...)}; `common`: tickere care nu se potrivesc
    după ticker. Numele: cu majuscule exact sau totul cu majuscule; tickerul: ≥ 3 litere, doar cu majuscule."""

    def __init__(self, symbols, names, common=frozenset(), weak=frozenset()):
        self.term_to_symbols = {}
        self.weak_terms = {v for w in weak for v in (w, w.upper())}   # mărci/produse: mențiune slabă
        for sym in symbols:
            for n in names.get(sym, ()):
                for variant in {n, n.upper()}:
                    self.term_to_symbols.setdefault(variant, set()).add(sym)
            base = str(sym).split(".")[0].split("-")[0]
            if base.isalpha() and len(base) >= 3 and base.upper() == base and base not in common:
                self.term_to_symbols.setdefault(base, set()).add(sym)
        terms = sorted(self.term_to_symbols, key=len, reverse=True)
        self.regex = re.compile(r"(?<![\w&])(?:" + "|".join(re.escape(t) for t in terms) + r")(?![\w&])") if terms else None

    def positions(self, text):
        """{simbol: [pozițiile de început ale mențiunilor]} în text."""
        out = {}
        if not self.regex or not text:
            return out
        for m in self.regex.finditer(str(text)):
            for sym in self.term_to_symbols.get(m.group(0), ()):
                out.setdefault(sym, []).append(m.start())
        return out

    def strong(self, text):
        """Simbolurile numite prin numele companiei, persoane sau ticker (nu doar printr-o marcă/un produs)."""
        out = set()
        if not self.regex or not text:
            return out
        for m in self.regex.finditer(str(text)):
            if m.group(0) not in self.weak_terms:
                out |= set(self.term_to_symbols.get(m.group(0), ()))
        return out

    def find(self, text):
        """Simbolurile numite, în ordinea apariției, fără dubluri."""
        if not self.regex or not text:
            return []
        out = []
        for m in self.regex.finditer(str(text)):
            for sym in sorted(self.term_to_symbols.get(m.group(0), ())):
                if sym not in out:
                    out.append(sym)
        return out


def subject(title, type_key, syms, matcher):
    """Compania la care se referă evenimentul: mențiunea cea mai apropiată de cuvântul-cheie al tipului (în „AMD Just
    Crushed It as Nvidia Scandal Grows” scandalul e lângă Nvidia), cu penalizare pentru mențiunile de după el. La egalitate sau fără poziții: prima numită.
    Distanța e în caractere; textul normalizat are aproape aceeași lungime ca originalul (diacriticele rămân litere)."""
    if len(syms) == 1:
        return syms[0]
    kws = [m.start() for m in _TYPE_RE[type_key].finditer(normalize(title))]
    pos = matcher.positions(title)
    if not kws or not pos:
        return syms[0]
    # O mențiune DUPĂ cuvântul-cheie contează de 3 ori mai departe: subiectul stă de obicei înaintea verbului
    # („Nvidia faces espionage probe as AMD gains” → Nvidia), dar „the scandal at Nvidia” rămâne Nvidia.
    dist = {s: min((k - p) if p <= k else 3 * (p - k) for p in pos.get(s, [10 ** 6]) for k in kws) for s in syms}
    return min(syms, key=lambda s: (dist[s], syms.index(s)))


def competitors(symbol, groups):
    """Concurenții din grupurile înguste ale simbolului: {simbol: [grupuri]} (fără simbolul însuși)."""
    out = {}
    for g, members in groups.items():
        if symbol in members:
            for s in members:
                if s != symbol:
                    out.setdefault(s, []).append(g)
    return out


# --- Evenimente sectoriale -----------------------------------------------------------------------------------------
UP_WORDS = _kw_regex(("rise", "rises", "rising", "jump", "jumps", "surge", "surges", "soar", "soars", "rally",
                      "rallies", "climb", "climbs", "gain", "gains", "record high", "spike", "spikes", "hike", "boost",
                      "increase", "higher", "crestere", "creste", "cresc", "urca", "urcat", "explodeaza", "maxim record", "majorare",
                      "majoreaza", "scumpire", "se scumpeste"))
DOWN_WORDS = _kw_regex(("fall", "falls", "falling", "drop", "drops", "plunge", "plunges", "slump", "slumps", "tumble",
                        "tumbles", "sink", "sinks", "slide", "slides", "decline", "declines", "record low", "lowest", "lower",
                        "scade", "scad", "scadere", "coboara", "prabusire", "se prabuseste", "ieftinire", "se ieftineste"))

# trigger: expresiile care identifică tema; groups: efectul pe grup când tema „crește” (+1) sau efectul fix (fixed=True).
SECTOR_RULES = {
    "petrol": {"label": "Prețul petrolului", "fixed": False,
               "trigger": ("oil price", "oil prices", "crude", "brent", "wti", "opec", "pretul petrolului",
                           "petrolul", "titeiul", "barilul"),
               "groups": {"Petrol integrate": 1, "Servicii petroliere": 1, "Petrol și gaze România": 1}},
    "gaze": {"label": "Prețul gazelor naturale", "fixed": False,
             "trigger": ("natural gas", "gas prices", "gas price", "ttf", "pretul gazelor", "gazele naturale",
                         "pretul gazului"),
             "groups": {"Gaz natural lichefiat": 1, "Petrol și gaze România": 1}},
    "energie": {"label": "Prețul energiei electrice", "fixed": False,
                "trigger": ("electricity price", "electricity prices", "power prices", "pretul energiei",
                            "energia electrica", "pretul curentului"),
                "groups": {"Producători de energie România": 1, "Utilități Europa": 1}},
    "taxe_energie": {"label": "Taxe sau plafonări pe energie", "fixed": True,
                     "trigger": ("windfall tax", "price cap", "supraimpozitare", "contributia de solidaritate",
                                 "plafonare", "plafonarea preturilor", "taxa pe profitul suplimentar"),
                     "groups": {"Producători de energie România": -1, "Petrol și gaze România": -1}},
    "taxe_banci": {"label": "Taxe pe bănci", "fixed": True,
                   "trigger": ("bank tax", "tax on banks", "banking tax", "taxa pe banci", "impozit pe banci",
                               "impozitul pe cifra de afaceri a bancilor", "impozitare suplimentara a bancilor"),
                   "groups": {"Bănci România": -1, "Bănci Europa": -1}},
    "aparare": {"label": "Cheltuieli de apărare", "fixed": False,
                "trigger": ("defense spending", "defence spending", "military spending", "rearmament", "rearm",
                            "cheltuieli de aparare", "cheltuielile militare", "bugetul apararii", "inzestrare"),
                "groups": {"Apărare": 1}},
    "export_cipuri": {"label": "Restricții la exportul de cipuri", "fixed": True,
                      "trigger": ("chip export", "chip exports", "semiconductor export", "chip ban", "chip curbs",
                                  "export controls on chips", "restrictii la exportul de cipuri"),
                      "groups": {"Acceleratoare AI / cipuri de calcul": -1, "Echipamente pentru semiconductori": -1}},
    "aur": {"label": "Prețul aurului", "fixed": False,
            "trigger": ("gold price", "gold prices", "pretul aurului", "aurul"),
            "groups": {"Minerit diversificat": 1}},
}
_SECTOR_RE = {k: _kw_regex(v["trigger"]) for k, v in SECTOR_RULES.items()}


def sector_direction(title, rule_key):
    """+1 / −1 / 0 pentru tema sectorială din titlu: fix pentru taxe și interdicții, altfel după verbul de direcție
    (crește / scade). Ambele sau niciunul → 0 (de urmărit)."""
    rule = SECTOR_RULES[rule_key]
    if rule["fixed"]:
        return 1
    t = normalize(title)
    up, down = bool(UP_WORDS.search(t)), bool(DOWN_WORDS.search(t))
    return 1 if up and not down else -1 if down and not up else 0


def sector_rules_in(title):
    t = normalize(title)
    return [k for k in SECTOR_RULES if _SECTOR_RE[k].search(t)]


# --- Construirea evenimentelor -------------------------------------------------------------------------------------

def clean_items(items, now, max_age_hours=MAX_AGE_HOURS):
    """Titlurile din ultimele `max_age_hours` ore, fără dubluri (titlu normalizat), cele mai noi primele.
    `items`: dict-uri cu title, link, source, date_obj (datetime naiv, ora României)."""
    seen, out = set(), []
    limit = now - timedelta(hours=max_age_hours)
    # Datele vin uneori cu fus orar (ora României), alteori fără: se compară toate fără fus.
    dated = [(it.get("date_obj").replace(tzinfo=None), it) for it in items or [] if isinstance(it.get("date_obj"), datetime)]
    for d, it in sorted(dated, key=lambda x: x[0], reverse=True):
        if d < limit or d > now + timedelta(minutes=30):
            continue
        key = normalize(it.get("title"))
        if not key or key in seen:
            continue
        seen.add(key)
        out.append({**it, "date_obj": d})
    return out


def _event_key(kind, symbols, day):
    raw = f"{kind}|{','.join(sorted(symbols))}|{day}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def detect_events(items, matcher, groups, now, market_of, max_age_hours=MAX_AGE_HOURS):
    """Evenimentele pe companie și cele sectoriale din titluri. Întoarce lista de dict-uri:
    key, kind („companie” / „sector”), type, label, direction, severity, affected [(simbol, direcție)],
    beneficiaries {simbol: [grupuri]}, markets (set), headlines [titlu, link, sursă, oră], time (cel mai recent).
    Titlurile cu același tip și aceleași companii în aceeași zi se adună într-un eveniment."""
    events = {}
    for it in clean_items(items, now, max_age_hours):
        title = str(it.get("title") or "")
        head = {"title": title, "link": it.get("link"), "source": it.get("source"), "time": it["date_obj"]}
        syms = matcher.find(title)
        types = classify(title)
        if syms and types:
            # Subiectul titlului = prima companie numită; celelalte sunt doar menționate (direcție 0): în „Nvidia
            # CEO faces probe as AMD gains” AMD nu e afectată negativ.
            t = types[0]
            spec = EVENT_TYPES[t]
            primary = subject(title, t, syms, matcher)
            direction = spec["direction"] if primary in matcher.strong(title) else 0   # doar prin produs → fără direcție
            key = _event_key(t, syms, it["date_obj"].date())
            ev = events.get(key)
            if ev is None:
                ben = {}
                if spec["share_shift"] and direction < 0:
                    # concurenții, inclusiv cei menționați în același titlu („AMD crushes as Nvidia scandal...”)
                    for c, gs in competitors(primary, groups).items():
                        ben[c] = list(gs)
                ev = events[key] = {
                    "key": key, "kind": "companie", "type": t, "label": spec["label"], "direction": direction,
                    "severity": spec["severity"] if direction else 1,
                    "affected": [(primary, direction)] + [(s, 0) for s in syms if s != primary and s not in ben],
                    "beneficiaries": ben, "markets": {market_of(s) for s in syms}, "headlines": [],
                    "time": it["date_obj"], "other_types": types[1:]}
            ev["headlines"].append(head)
            ev["time"] = max(ev["time"], it["date_obj"])
            continue
        for rk in sector_rules_in(title):
            rule = SECTOR_RULES[rk]
            d = sector_direction(title, rk)
            key = _event_key(f"sector:{rk}:{d}", [], it["date_obj"].date())
            ev = events.get(key)
            if ev is None:
                affected = []
                for g, sign in rule["groups"].items():
                    for s in groups.get(g, ()):
                        if s not in [a for a, _ in affected]:
                            affected.append((s, sign * d))
                ev = events[key] = {
                    "key": key, "kind": "sector", "type": rk, "label": rule["label"], "direction": d,
                    "severity": 2 if rule["fixed"] else 1, "affected": affected, "beneficiaries": {},
                    "markets": {market_of(s) for s, _ in affected}, "headlines": [], "time": it["date_obj"],
                    "other_types": [], "groups": list(rule["groups"])}
            ev["headlines"].append(head)
            ev["time"] = max(ev["time"], it["date_obj"])
    out = list(events.values())
    for ev in out:
        ev["severity_total"] = ev["severity"] + (1 if len({h["source"] for h in ev["headlines"]}) >= 2 else 0)
    return sorted(out, key=lambda e: (-e["severity_total"], -e["time"].timestamp()))


def touches(event, holdings):
    """Deținerile atinse de eveniment cu o direcție (afectate cu direcție ≠ 0 sau beneficiari): [(simbol, rol)].
    Mențiunile fără direcție (fuziuni ambigue, companii doar numite) nu dau alertă."""
    held = set(holdings or ())
    out = [(s, "afectat") for s, d in event["affected"] if d and s in held]
    out += [(s, "beneficiar") for s in event["beneficiaries"] if s in held]
    return sorted(out)


# --- Reacția pieței ------------------------------------------------------------------------------------------------
CLOSE_TIME_RO = {"us": (23, 0), "eu": (18, 30), "bvb": (18, 0)}   # ora României, aproximativ (fără schimbarea orei)
SIGMA_SESSIONS = 60
Z_PRICED = 2.0
RVOL_HIGH = 2.0
Z_BIG_MOVE = 3.0


def _series(s):
    if s is None:
        return pd.Series(dtype="float64")
    out = pd.to_numeric(pd.Series(s, dtype="float64"), errors="coerce").dropna()
    if len(out) and isinstance(out.index, pd.DatetimeIndex):
        idx = out.index.tz_localize(None) if out.index.tz is not None else out.index
        out = pd.Series(out.to_numpy(), index=idx.normalize())
        out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def reference_session(index, event_time, market):
    """Ultima ședință dinaintea știrii: dacă știrea e după închiderea pieței (ora României), ședința din aceeași zi;
    altfel ședința anterioară zilei știrii. None dacă nu există."""
    idx = pd.DatetimeIndex(index)
    h, m = CLOSE_TIME_RO.get(market, (23, 0))
    day = pd.Timestamp(event_time.date())
    after_close = (event_time.hour, event_time.minute) >= (h, m)
    cand = idx[idx <= day] if after_close else idx[idx < day]
    return cand[-1] if len(cand) else None


def reaction(close, bench, volume, event_time, market, direction):
    """Reacția de la ședința de referință până la ultima închidere.

    dict: ref (data), sessions (câte ședințe după ref), ret, bench_ret, abnormal (fracții), z (anormal ÷ volatilitatea
    zilnică a randamentului anormal pe 60 de ședințe înainte de ref × √ședințe), rvol (volumul ultimei ședințe ÷ media
    celor 20 dinainte), label. Fără date → valorile None și label „N/A”."""
    c, b, v = _series(close), _series(bench), _series(volume)
    out = {"ref": None, "sessions": 0, "ret": None, "bench_ret": None, "abnormal": None, "z": None, "rvol": None,
           "label": "N/A"}
    if c.empty or b.empty:
        return out
    ref = reference_session(c.index, event_time, market)
    if ref is None:
        return out
    out["ref"] = ref
    after = c[c.index > ref]
    out["sessions"] = len(after)
    if len(v) >= 21:
        base = v.iloc[-21:-1].mean()
        out["rvol"] = float(v.iloc[-1] / base) if base > 0 else None
    if not len(after):
        out["label"] = "fără ședință de la știre"
        return out
    b_ref = b[b.index <= ref]
    if b_ref.empty:
        return out
    out["ret"] = float(c.iloc[-1] / c.loc[ref] - 1)
    out["bench_ret"] = float(b.iloc[-1] / b_ref.iloc[-1] - 1)
    out["abnormal"] = out["ret"] - out["bench_ret"]
    hist = pd.concat([c[c.index <= ref].pct_change(), b[b.index <= ref].pct_change()], axis=1).dropna().iloc[-SIGMA_SESSIONS:]
    if len(hist) >= 20:
        sigma = float((hist.iloc[:, 0] - hist.iloc[:, 1]).std(ddof=1))
        if sigma > 0:
            out["z"] = out["abnormal"] / (sigma * math.sqrt(out["sessions"]))
    out["label"] = reaction_label(out["z"], direction)
    return out


def reaction_label(z, direction):
    """deja în preț (≥ 2σ în direcția așteptată) / contrazis de piață (≥ 2σ invers) / încă nereflectat / mișcare
    puternică (direcție ambiguă) / N/A."""
    if z is None:
        return "N/A"
    if not direction:
        return "mișcare puternică" if abs(z) >= Z_PRICED else "fără reacție clară"
    s = z * direction
    return "deja în preț" if s >= Z_PRICED else "contrazis de piață" if s <= -Z_PRICED else "încă nereflectat"


def big_moves(closes, bench_of, exclude, z_min=Z_BIG_MOVE):
    """Mișcările mari ale ultimei ședințe FĂRĂ știre potrivită (simbolurile din `exclude` sunt sărite).

    `closes`: DataFrame închideri; `bench_of`: {simbol: serie benchmark}. Pentru fiecare simbol: randamentul anormal
    al ultimei ședințe ÷ volatilitatea lui pe 60 de ședințe. Întoarce listă de dict symbol, date, ret, abnormal, z,
    sortată după |z|."""
    out = []
    for sym in closes.columns:
        if sym in exclude:
            continue
        c, b = _series(closes[sym]), _series(bench_of.get(sym))
        if len(c) < 30 or b.empty:
            continue
        r = pd.concat([c.pct_change(), b.pct_change()], axis=1).dropna()
        if len(r) < 25 or r.index[-1] != c.index[-1]:
            continue
        ab = r.iloc[:, 0] - r.iloc[:, 1]
        sigma = float(ab.iloc[-SIGMA_SESSIONS - 1:-1].std(ddof=1))
        if not sigma > 0:
            continue
        z = float(ab.iloc[-1] / sigma)
        if abs(z) >= z_min:
            out.append({"symbol": sym, "date": c.index[-1], "ret": float(r.iloc[-1, 0]), "abnormal": float(ab.iloc[-1]),
                        "z": z})
    return sorted(out, key=lambda x: -abs(x["z"]))


# --- Jurnalul evenimentelor ----------------------------------------------------------------------------------------
JOURNAL_SHEET = "Evenimente_jurnal"
JOURNAL_COLUMNS = ["Data", "Ora", "Symbol", "Rol", "Directie", "Tip", "Data_ref", "Benchmark", "Titlu", "Cheie"]
EVAL_SESSIONS = (5, 20)
JOURNAL_MIN_N = 20


def journal_rows(events, reactions, bench_of_market, market_of, moves=()):
    """Rândurile de jurnal: câte unul pe (eveniment, simbol, rol). `reactions`: {(cheie, simbol): reaction dict} —
    data de referință vine de acolo; fără ea, rândul nu se scrie (nu se poate evalua). `moves`: mișcările fără știre
    (rol „fara_stire”, direcția așteptată = invers mișcării, după Chan 2003)."""
    rows = []
    for ev in events:
        t = ev["time"]
        title = ev["headlines"][0]["title"][:200] if ev["headlines"] else ""
        for role, pairs in (("afectat", ev["affected"]), ("beneficiar", [(s, 1) for s in ev["beneficiaries"]])):
            for sym, d in pairs:
                if not d:            # menționat sau direcție ambiguă: nu are ce evalua
                    continue
                r = reactions.get((ev["key"], sym))
                if not r or r.get("ref") is None:
                    continue
                rows.append([t.strftime("%Y-%m-%d"), t.strftime("%H:%M"), sym, role, int(d), ev["type"],
                             pd.Timestamp(r["ref"]).strftime("%Y-%m-%d"), bench_of_market.get(market_of(sym), ""),
                             title, f"{ev['key']}:{sym}:{role}"])
    for m in moves:
        d = -1 if m["z"] > 0 else 1
        rows.append([pd.Timestamp(m["date"]).strftime("%Y-%m-%d"), "", m["symbol"], "fara_stire", d, "fara_stire",
                     pd.Timestamp(m["date"]).strftime("%Y-%m-%d"), bench_of_market.get(market_of(m["symbol"]), ""),
                     f"mișcare {m['z']:+.1f}σ fără știre", f"fara_stire:{m['symbol']}:{pd.Timestamp(m['date']):%Y%m%d}"])
    return rows


def parse_journal(values):
    """Rândurile foii → (DataFrame Data_ref, Symbol, Rol, Directie, Tip, Benchmark, Cheie, Titlu; probleme)."""
    cols = ["Data_ref", "Symbol", "Rol", "Directie", "Tip", "Benchmark", "Cheie", "Titlu", "Data"]
    empty = pd.DataFrame(columns=cols)
    if not values:
        return empty, []
    header = [str(h).strip() for h in values[0]]
    try:
        pos = {c: header.index(c) for c in JOURNAL_COLUMNS}
    except ValueError:
        return empty, [f"Foaia {JOURNAL_SHEET} nu are antetul {', '.join(JOURNAL_COLUMNS)}."]
    rows, bad = [], 0
    for raw in values[1:]:
        cell = lambda c: raw[pos[c]] if pos[c] < len(raw) else ""
        ref = pd.to_datetime(str(cell("Data_ref")).strip(), format="%Y-%m-%d", errors="coerce")
        sym = str(cell("Symbol")).strip().upper()
        try:
            d = int(float(str(cell("Directie")).replace(",", ".")))
        except ValueError:
            d = None
        if pd.isna(ref) or not sym or d not in (-1, 0, 1):
            if any(str(x).strip() for x in raw):
                bad += 1
            continue
        rows.append({"Data_ref": ref, "Symbol": sym, "Rol": str(cell("Rol")).strip(), "Directie": d,
                     "Tip": str(cell("Tip")).strip(), "Benchmark": str(cell("Benchmark")).strip().upper(),
                     "Cheie": str(cell("Cheie")).strip(), "Titlu": str(cell("Titlu")),
                     "Data": str(cell("Data")).strip()})
    problems = [f"{JOURNAL_SHEET}: {bad} rânduri invalide sărite."] if bad else []
    return (pd.DataFrame(rows, columns=cols) if rows else empty), problems


def evaluate_event(close, bench, ref_date, sessions=EVAL_SESSIONS):
    """Randamentul anormal (activ − benchmark) de la ședința de referință la +k ședințe (ale activului).
    {k: anormal sau None dacă nu au trecut k ședințe}."""
    c, b = _series(close), _series(bench)
    out = {k: None for k in sessions}
    if c.empty or b.empty:
        return out
    ref = pd.Timestamp(ref_date)
    base = c[c.index <= ref]
    after = c[c.index > ref]
    b0 = b[b.index <= ref]
    if base.empty or b0.empty:
        return out
    for k in sessions:
        if len(after) >= k:
            day = after.index[k - 1]
            b1 = b[b.index <= day]
            out[k] = float(after.iloc[k - 1] / base.iloc[-1] - 1) - float(b1.iloc[-1] / b0.iloc[-1] - 1)
    return out


def journal_summary(results):
    """Pe rol (afectat / beneficiar / fara_stire): N evaluate la 5 și 20 de ședințe, rata de nimerire a direcției,
    randamentul anormal mediu în direcția așteptată. Direcția 0 nu intră. `small`: sub 20 de observații."""
    cols = ["Rol", "N 5", "Nimerite 5", "Medie 5", "N 20", "Nimerite 20", "Medie 20", "small"]
    if results is None or results.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for role, g in results[results["Directie"] != 0].groupby("Rol", sort=True):
        row = {"Rol": role}
        for k in EVAL_SESSIONS:
            x = pd.to_numeric(g[f"ab{k}"], errors="coerce")
            ok = x.notna()
            signed = (x[ok] * g.loc[ok, "Directie"])
            row[f"N {k}"] = int(ok.sum())
            row[f"Nimerite {k}"] = float((signed > 0).mean()) if ok.any() else None
            row[f"Medie {k}"] = float(signed.mean()) if ok.any() else None
        row["small"] = row[f"N {EVAL_SESSIONS[-1]}"] < JOURNAL_MIN_N
        rows.append(row)
    return pd.DataFrame(rows, columns=cols)
