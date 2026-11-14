"""Corespondența ETF semnal (SUA) → ETF UCITS cumpărabil de un investitor de retail din UE (Daniel, 14.11.2026).

De ce: ETF-urile domiciliate în SUA nu au KID (PRIIPs), deci XTB și IBKR Irlanda nu le vând clienților de retail
din UE. Semnalul se calculează pe ETF-ul SUA (istorie lungă, date curate); execuția se face pe echivalentul UCITS.

ATENȚIE: simbolurile de mai jos NU au fost verificate din mediul de lucru (fără acces la Yahoo). Aplicația verifică
singură, la afișare, dacă Yahoo are date pentru simbolul UCITS și cât de corelate sunt randamentele săptămânale cu
ETF-ul semnal (analytics/ideas.tracking_check). Înainte de o tranzacție: verifică ISIN-ul și indicele la emitent și
la broker. Fără date de portofoliu aici (repo public).
"""

# simbol semnal → (simbol Yahoo UCITS sau None, descriere)
UCITS = {
    "SPY": ("SXR8.DE", "iShares Core S&P 500 UCITS (Acc), Xetra, EUR"),
    "VGK": ("EXSA.DE", "iShares STOXX Europe 600 UCITS (Dist), Xetra, EUR"),
    "EEM": ("IS3N.DE", "iShares Core MSCI EM IMI UCITS (Acc), Xetra, EUR"),
    "EWJ": ("SJPA.L", "iShares Core MSCI Japan IMI UCITS (Acc), Londra, USD"),
    "IEF": ("IBTM.L", "iShares $ Treasury Bond 7-10yr UCITS (Dist), Londra, USD"),
    "TLT": ("IDTL.L", "iShares $ Treasury Bond 20+yr UCITS (Dist), Londra, USD"),
    "TIP": ("ITPS.L", "iShares $ TIPS UCITS (Acc), Londra, USD"),
    "GLD": ("SGLN.L", "iShares Physical Gold ETC, Londra, USD"),
    "DBC": ("CMOD.L", "Invesco Bloomberg Commodity UCITS (Acc), Londra, USD"),
    "UUP": (None, "fără echivalent UCITS simplu: expunerea pe dolar se ia prin active în USD"),
    "XLK": ("ZPDT.DE", "SPDR S&P U.S. Technology Select Sector UCITS, Xetra, EUR"),
    "XLV": ("ZPDH.DE", "SPDR S&P U.S. Health Care Select Sector UCITS, Xetra, EUR"),
    "XLF": ("ZPDF.DE", "SPDR S&P U.S. Financials Select Sector UCITS, Xetra, EUR"),
    "XLE": ("ZPDE.DE", "SPDR S&P U.S. Energy Select Sector UCITS, Xetra, EUR"),
    "XLY": ("ZPDD.DE", "SPDR S&P U.S. Consumer Discretionary Select Sector UCITS, Xetra, EUR"),
    "XLP": ("ZPDS.DE", "SPDR S&P U.S. Consumer Staples Select Sector UCITS, Xetra, EUR"),
    "XLI": ("ZPDI.DE", "SPDR S&P U.S. Industrials Select Sector UCITS, Xetra, EUR"),
    "XLB": ("ZPDM.DE", "SPDR S&P U.S. Materials Select Sector UCITS, Xetra, EUR"),
    "XLU": ("ZPDU.DE", "SPDR S&P U.S. Utilities Select Sector UCITS, Xetra, EUR"),
    "XLC": ("ZPDX.DE", "SPDR S&P U.S. Communication Services Select Sector UCITS, Xetra, EUR"),
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
    """Valuta liniei de cotare: .DE → EUR; .L → USD (liniile din tabel sunt în USD); .RO → RON; fără sufix → USD."""
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
