"""Filtrul știrilor despre o companie: păstrează doar titlurile care chiar o menționează.

Fluxul RSS Yahoo pe simbol (`headline?s=SAP.DE`) întoarce des știri generale despre
piață sau despre alte companii (SAP.DE → IBM/ServiceNow; SNP.RO → „French Bank Stocks”).
Sentimentul calculat pe ele nu spune nimic despre emitent. Aici se păstrează doar
știrile al căror titlu sau rezumat conține numele companiei sau simbolul ei.

Funcții pure, fără rețea și fără `st.*`.
"""
import re
from datetime import datetime, timedelta

# Forme juridice și cuvinte care apar în numele multor companii: nu identifică emitentul.
LEGAL_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "co", "company", "ltd", "limited", "plc", "llc", "lp",
    "sa", "s.a", "s.a.", "se", "ag", "nv", "n.v", "bv", "spa", "s.p.a", "ab", "asa", "as", "oyj", "kgaa",
    "srl", "holding", "holdings", "group", "groupe", "the", "and", "of", "de", "des", "du", "la", "le",
    "societatea", "comerciala", "compania", "nationala", "national", "international", "global",
    "banca", "bank", "banco", "energy", "capital", "financial", "industries", "industrial", "technologies",
    "technology", "systems", "american", "romania", "romanian", "class", "ordinary", "shares", "adr",
}
MIN_WORD = 4        # un cuvânt din nume, scris normal, contează de la 4 litere
MIN_ACRONYM = 3     # un acronim/simbol cu majuscule (SAP, OMV, BRD) contează de la 3 litere


def company_terms(symbol, names=()):
    """Termenii care identifică emitentul: (cuvinte comparate fără majuscule, acronime comparate exact).

    - din simbol: partea dinaintea punctului (SAP.DE → SAP), dacă are cel puțin 3 litere;
      se compară exact, cu majuscule, ca „ALL” sau „ON” să nu prindă cuvinte obișnuite;
    - din nume: cuvintele care nu sunt forme juridice sau cuvinte generice; cele scrise
      integral cu majuscule (OMV, BRD) sunt acronime și se compară exact.
    """
    words, acronyms = set(), set()
    base = str(symbol or "").split(".")[0].strip()
    if base.isalpha() and len(base) >= MIN_ACRONYM and not base.startswith("^"):
        acronyms.add(base.upper())
    for name in names or ():
        if not name:
            continue
        # Numele întreg, fără forma juridică de la final („OMV Petrom”, „AT&T”), ca expresie:
        # doar când are mai multe cuvinte sau semne (un singur cuvânt e acoperit mai jos).
        parts = str(name).replace(",", " ").split()
        while parts and parts[-1].lower().strip(".") in LEGAL_SUFFIXES | {"s.a"}:
            parts.pop()
        phrase = " ".join(parts)
        if len(phrase) >= MIN_ACRONYM and not phrase.isalpha():
            words.add(phrase.lower())
        for raw in re.split(r"[\s,/()&\-–]+", str(name)):
            token = raw.strip(".'’\"")
            if not token or token.lower().strip(".") in LEGAL_SUFFIXES or token.lower() in LEGAL_SUFFIXES:
                continue
            if token.isupper() and token.isalpha() and len(token) >= MIN_ACRONYM:
                acronyms.add(token)
            elif token.isalpha() and len(token) >= MIN_WORD:
                words.add(token.lower())
    return words, acronyms


def mentions_company(text, terms):
    """True dacă textul conține unul dintre termeni ca cuvânt întreg."""
    words, acronyms = terms
    text = str(text or "")
    lowered = text.lower()
    for w in words:
        if re.search(rf"(?<![\w]){re.escape(w)}(?![\w])", lowered):
            return True
    for a in acronyms:
        if re.search(rf"(?<![\w]){re.escape(a)}(?![\w])", text):
            return True
    return False


def filter_company_news(items, symbol, names=()):
    """Știrile (dict-uri cu `title` și opțional `summary`) care menționează emitentul.

    Fără niciun termen de identificare întoarce o listă goală: o știre nepotrivită ar da
    un sentiment fals, iar lipsa știrilor se afișează ca N/A.
    """
    terms = company_terms(symbol, names)
    if not terms[0] and not terms[1]:
        return []
    return [it for it in items or []
            if mentions_company(f"{it.get('title', '')} {it.get('summary', '')}", terms)]


# --- Rezumatul Zilei, etapa D: știrile despre pozițiile din registru -------------------------------------------------
NEWS_MAX_AGE_HOURS = 48       # știrile mai vechi de 2 zile nu mai sunt „de azi”
NEWS_PER_SYMBOL = 3           # cel mult 3 știri pe poziție, ca o companie cu multe titluri să nu le acopere pe celelalte
NEWS_TOTAL = 15


def news_time(item):
    """Momentul publicării (datetime naiv, ora României): `date_obj` dacă există, altfel `date_str` „AAAA-LL-ZZ HH:MM”.
    Neinterpretabil → None (știrea nu intră în filtrul de vechime, nu primește o dată inventată)."""
    d = item.get("date_obj") if isinstance(item, dict) else None
    if isinstance(d, datetime):
        return d.replace(tzinfo=None)
    try:
        return datetime.strptime(str(item.get("date_str", ""))[:16], "%Y-%m-%d %H:%M")
    except (ValueError, AttributeError):
        return None


def _bare_name(name):
    """Numele fără forma juridică și fără cuvintele generice de la final („Bank of America Corporation” → „Bank of
    America”, „Novo Nordisk A/S” → „Novo Nordisk”, „OMV PETROM SA” → „OMV PETROM”)."""
    parts = str(name or "").replace(",", " ").split()
    while parts and parts[-1].lower().replace("/", "").strip(".") in LEGAL_SUFFIXES | {"sa", "nv"}:
        parts.pop()
    return " ".join(parts)


def strict_terms(symbol, names=()):
    """Termenii stricți pentru „Știri despre acțiunile tale”: numele întreg al companiei (fără forma juridică), ca
    expresie, și simbolul (≥ 3 litere, cu majuscule, exact). Un cuvânt singur din nume NU ajunge când numele are mai
    multe cuvinte: „America” din Bank of America prindea „America Latină”, „First” din First Majestic orice „first”.
    Numele dintr-un singur cuvânt (CoreWeave, Alphabet) rămâne cuvânt, de la 4 litere. Întoarce (expresii, acronime)."""
    phrases, acronyms = set(), set()
    base = str(symbol or "").split(".")[0].strip()
    if base.isalpha() and len(base) >= MIN_ACRONYM and not base.startswith("^"):
        acronyms.add(base.upper())
    for name in names or ():
        bare = _bare_name(name)
        if not bare:
            continue
        if " " in bare:
            phrases.add(bare.lower())
        elif bare.isupper() and bare.isalpha() and len(bare) >= MIN_ACRONYM:
            acronyms.add(bare)
        elif len(bare) >= MIN_WORD and bare.lower() not in LEGAL_SUFFIXES:
            phrases.add(bare.lower())
    return phrases, acronyms


def holdings_news(holdings, general, own=None, now=None, max_age_hours=NEWS_MAX_AGE_HOURS,
                  per_symbol=NEWS_PER_SYMBOL, total=NEWS_TOTAL):
    """Știrile recente care menționează pozițiile tale.

    holdings: {simbol: (nume, ...)} — numele companiei (din Yahoo); fără nume → doar simbolul (`strict_terms`).
    general: știrile din fluxurile generale (RO și internaționale).
    own: {simbol: știri} din fluxul Yahoo pe simbol. Toate se filtrează aici cu `strict_terms` (numele întreg sau
    simbolul), mai strict decât `filter_company_news` (care rămâne neschimbat pentru sentiment).
    now: datetime naiv în ora României (implicit acum, pentru teste se dă explicit).

    Întoarce lista de dict symbol, title, link, source, time (datetime), cea mai nouă întâi: cel mult `per_symbol`
    pe simbol și `total` în total; aceeași știre (același titlu) o singură dată, la primul simbol în ordinea dată.
    Știrile fără dată sau mai vechi de `max_age_hours` se scot."""
    own = own or {}
    now = now or datetime.now()
    oldest = now - timedelta(hours=max_age_hours)
    seen, out = set(), []
    for sym, names in (holdings or {}).items():
        terms = strict_terms(sym, tuple(n for n in names or () if n))
        if not terms[0] and not terms[1]:
            continue
        cands = [it for it in list(own.get(sym) or []) + list(general or [])
                 if mentions_company(f"{it.get('title', '')} {it.get('summary', '')}", terms)]
        picked = []
        for it in sorted(cands, key=lambda x: news_time(x) or datetime.min, reverse=True):
            t, title = news_time(it), str(it.get("title", "")).strip()
            key = title.casefold()
            if t is None or t < oldest or t > now + timedelta(hours=1) or not title or key in seen:
                continue
            seen.add(key)
            picked.append({"symbol": sym, "title": title, "link": it.get("link", ""),
                           "source": it.get("source") or it.get("publisher") or "", "time": t})
            if len(picked) >= per_symbol:
                break
        out.extend(picked)
    out.sort(key=lambda x: x["time"], reverse=True)
    return out[:total]
