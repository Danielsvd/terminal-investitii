"""Filtrul știrilor despre o companie: păstrează doar titlurile care chiar o menționează.

Fluxul RSS Yahoo pe simbol (`headline?s=SAP.DE`) întoarce des știri generale despre
piață sau despre alte companii (SAP.DE → IBM/ServiceNow; SNP.RO → „French Bank Stocks”).
Sentimentul calculat pe ele nu spune nimic despre emitent. Aici se păstrează doar
știrile al căror titlu sau rezumat conține numele companiei sau simbolul ei.

Funcții pure, fără rețea și fără `st.*`.
"""
import re

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
