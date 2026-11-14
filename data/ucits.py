"""Corespondența ETF semnal (SUA) → ETF UCITS cumpărabil de un investitor de retail din UE (Daniel, 14.11.2026).

De ce: ETF-urile domiciliate în SUA nu au KID (PRIIPs), deci XTB și IBKR Irlanda nu le vând clienților de retail
din UE. Semnalul se calculează pe ETF-ul SUA (istorie lungă, date curate); execuția se face pe echivalentul UCITS.

Broker pentru ETF-uri: IBKR (Daniel, 14.11.2026; la XTB nu mai cumpără ETF-uri), cu preferință pentru Xetra.

ATENȚIE: simbolurile de mai jos NU au fost verificate din mediul de lucru (fără acces la Yahoo). Aplicația verifică
singură, la afișare, dacă Yahoo are date pentru simbolul UCITS și cât de corelate sunt randamentele săptămânale cu
ETF-ul semnal (analytics/ideas.tracking_check). Înainte de o tranzacție: verifică ISIN-ul și indicele la emitent și
la broker. Fără date de portofoliu aici (repo public).
"""

# simbol semnal → (simbol Yahoo UCITS sau None, descriere)
UCITS = {
    "SPY": ("SXR8.DE", "iShares Core S&P 500 UCITS (Acc), Xetra"),
    "VGK": ("EXSA.DE", "iShares STOXX Europe 600 UCITS (Dist), Xetra"),
    "EEM": ("IS3N.DE", "iShares Core MSCI EM IMI UCITS (Acc), Xetra"),
    "EWJ": ("SJPA.L", "iShares Core MSCI Japan IMI UCITS (Acc), Londra"),
    "IEF": ("CBU0.DE", "iShares $ Treasury Bond 7-10yr UCITS, Xetra — linia de la IBKR (Daniel, 14.11)"),
    "TLT": ("IS04.DE", "iShares $ Treasury Bond 20+yr UCITS, Xetra — linia de la IBKR (Daniel, 14.11)"),
    "TIP": ("ITPS.L", "iShares $ TIPS UCITS (Acc), Londra"),
    "GLD": ("SGLN.L", "iShares Physical Gold ETC, Londra"),
    "DBC": ("CMOD.L", "Invesco Bloomberg Commodity UCITS (Acc), Londra"),
    "UUP": (None, "fără echivalent UCITS simplu: expunerea pe dolar se ia prin active în USD"),
    "XLK": ("ZPDT.DE", "SPDR S&P U.S. Technology Select Sector UCITS, Xetra"),
    "XLV": ("ZPDH.DE", "SPDR S&P U.S. Health Care Select Sector UCITS, Xetra"),
    "XLF": ("ZPDF.DE", "SPDR S&P U.S. Financials Select Sector UCITS, Xetra"),
    "XLE": ("ZPDE.DE", "SPDR S&P U.S. Energy Select Sector UCITS, Xetra"),
    "XLY": ("ZPDD.DE", "SPDR S&P U.S. Consumer Discretionary Select Sector UCITS, Xetra"),
    "XLP": ("ZPDS.DE", "SPDR S&P U.S. Consumer Staples Select Sector UCITS, Xetra"),
    "XLI": ("ZPDI.DE", "SPDR S&P U.S. Industrials Select Sector UCITS, Xetra"),
    "XLB": ("ZPDM.DE", "SPDR S&P U.S. Materials Select Sector UCITS, Xetra"),
    "XLU": ("ZPDU.DE", "SPDR S&P U.S. Utilities Select Sector UCITS, Xetra"),
    "XLC": ("ZPDX.DE", "SPDR S&P U.S. Communication Services Select Sector UCITS, Xetra"),
    "XLRE": (None, "fără echivalent UCITS sectorial identificat: de căutat la broker"),
}


def ucits_for(symbol):
    """(simbol UCITS sau None, descriere). Simbolurile europene (.DE, .RO...) sunt deja cumpărabile: se întorc ele."""
    s = str(symbol).upper()
    if s in UCITS:
        return UCITS[s]
    if "." in s:
        return s, "cotat în Europa: se cumpără direct"
    return None, "fără corespondență în tabel"


def ucits_currency(symbol):
    """Valuta DEDUSĂ din sufix, doar ca rezervă când Yahoo nu răspunde: .DE → EUR; .L → USD; .RO → RON; fără sufix →
    USD. La Londra deducția poate greși (SJPA.L, SGLN.L cotează în pence, GBp): aplicația cere întâi moneda de la Yahoo."""
    s = str(symbol or "").upper()
    if s.endswith(".DE"):
        return "EUR"
    if s.endswith(".RO"):
        return "RON"
    if s.endswith(".L") or "." not in s:
        return "USD" if s else None
    return None


def ucits_symbols(symbols):
    """Simbolurile UCITS (fără None, fără dubluri) pentru o listă de simboluri semnal."""
    out = []
    for s in symbols:
        u = UCITS.get(str(s).upper(), (None, ""))[0]
        if u and u not in out:
            out.append(u)
    return out
