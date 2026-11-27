"""Codul SIC al SEC → sectorul în vocabularul Yahoo (Faza 3c, decizia E5). Funcție pură.

Pentru companiile ieșite din indice nu există sector Yahoo; testul istoric ia sectorul din codul SIC (`submissions`),
ca `fundamentals.is_financial_issuer`, `altman_variant` și `dcf_not_applicable_reason` să funcționeze neschimbate.
Contează în scor: (1) financiar sau nu (bănci, asigurători, fonduri: scor pe P/BV și ROE, fără DCF, Altman, Piotroski
pe flux), (2) REIT (fără DCF), (3) „producție” sau nu (Altman Z față de Z''), (4) gruparea pe sector la varianta pe rang.

Limite (spuse în raport): SIC-ul e cel de azi (o companie care și-a schimbat activitatea are codul nou); tabelul e o
aproximare pe intervale — câteva coduri se împart altfel în Yahoo (ex. internetul, SIC 7370, e „Technology” aici, la
Yahoo uneori „Communication Services”). Financiare = SIC 6000–6799, fără imobiliare (6500–6553) și REIT (6798).
"""

REIT_INDUSTRY = "REIT"      # `dcf_not_applicable_reason` caută „reit” în industrie

# (de la, până la, sector Yahoo) — primul interval care conține codul câștigă; cele specifice stau înaintea celor largi
_RANGES = (
    (100, 999, "Consumer Defensive"),
    (1000, 1099, "Basic Materials"),
    (1200, 1399, "Energy"),
    (1400, 1499, "Basic Materials"),
    (1520, 1531, "Consumer Cyclical"),          # constructori de locuințe
    (1500, 1799, "Industrials"),
    (2000, 2199, "Consumer Defensive"),
    (2200, 2399, "Consumer Cyclical"),
    (2400, 2499, "Basic Materials"),
    (2500, 2599, "Consumer Cyclical"),
    (2600, 2699, "Basic Materials"),
    (2700, 2799, "Communication Services"),
    (2830, 2836, "Healthcare"),
    (2840, 2844, "Consumer Defensive"),
    (2800, 2899, "Basic Materials"),
    (2900, 2999, "Energy"),
    (3000, 3199, "Consumer Cyclical"),
    (3200, 3399, "Basic Materials"),
    (3400, 3499, "Industrials"),
    (3570, 3579, "Technology"),
    (3500, 3599, "Industrials"),
    (3600, 3629, "Industrials"),                # echipamente electrice, motoare (GE, Emerson: SIC 3600)
    (3630, 3639, "Consumer Cyclical"),
    (3640, 3649, "Industrials"),                # iluminat, instalații electrice
    (3600, 3699, "Technology"),
    (3710, 3716, "Consumer Cyclical"),
    (3700, 3799, "Industrials"),
    (3812, 3812, "Industrials"),
    (3840, 3851, "Healthcare"),
    (3800, 3899, "Technology"),
    (3900, 3999, "Consumer Cyclical"),
    (4950, 4959, "Industrials"),                # deșeuri
    (4900, 4999, "Utilities"),
    (4800, 4899, "Communication Services"),
    (4000, 4799, "Industrials"),
    (5122, 5122, "Healthcare"),
    (5140, 5149, "Consumer Defensive"),
    (5170, 5172, "Energy"),
    (5000, 5199, "Industrials"),
    (5331, 5331, "Consumer Defensive"),         # magazine cu reducere (Yahoo: „Discount Stores”)
    (5399, 5399, "Consumer Defensive"),
    (5400, 5499, "Consumer Defensive"),
    (5912, 5912, "Healthcare"),
    (5200, 5999, "Consumer Cyclical"),
    (6500, 6553, "Real Estate"),
    (6798, 6798, "Real Estate"),
    (6000, 6799, "Financial Services"),
    (7000, 7299, "Consumer Cyclical"),
    (7310, 7319, "Communication Services"),
    (7370, 7379, "Technology"),
    (7300, 7399, "Industrials"),
    (7500, 7599, "Consumer Cyclical"),
    (7800, 7899, "Communication Services"),
    (7900, 7999, "Consumer Cyclical"),
    (8000, 8099, "Healthcare"),
    (8200, 8299, "Consumer Defensive"),
    (8731, 8731, "Healthcare"),
    (8100, 8999, "Industrials"),
)


def _code(sic):
    try:
        c = int(str(sic).strip())
    except (TypeError, ValueError):
        return None
    return c if 100 <= c <= 9999 else None


def sic_sector(sic):
    """(sector Yahoo, industrie) pentru codul SIC; (None, None) dacă codul lipsește sau nu e în tabel.
    Industria e completată doar la REIT (6798), singura folosită de funcțiile aplicației."""
    c = _code(sic)
    if c is None:
        return None, None
    for lo, hi, sector in _RANGES:
        if lo <= c <= hi:
            return sector, (REIT_INDUSTRY if c == 6798 else None)
    return None, None


__all__ = ["sic_sector", "REIT_INDUSTRY"]
