"""Comparabili pe sector și regiune. Fără Streamlit, fără rețea.

Listele sunt fixe și îmbătrânesc (delistări, schimbări de simbol): aplicația raportează
simbolurile pentru care Yahoo nu mai trimite date, ca lista să poată fi corectată.
Sectoarele poartă denumirile Yahoo. Companiile dintr-o listă sunt din aceeași regiune,
pentru că multiplii diferă structural între SUA și Europa.
"""
import math
import statistics

PEERS = {
    "US": {
        "Technology": ["MSFT", "AAPL", "NVDA", "AVGO", "ORCL", "CRM", "ADBE", "AMD", "INTC", "QCOM", "TXN", "CSCO", "IBM", "MU"],
        "Communication Services": ["GOOGL", "META", "NFLX", "DIS", "T", "VZ", "CMCSA", "TMUS"],
        "Financial Services": ["JPM", "BAC", "WFC", "C", "GS", "MS", "AXP", "SCHW", "BLK", "V", "MA"],
        "Energy": ["XOM", "CVX", "COP", "OXY", "EOG", "SLB", "DVN", "LNG", "MPC", "PSX"],
        "Healthcare": ["LLY", "JNJ", "MRK", "PFE", "ABBV", "UNH", "TMO", "ABT", "AMGN", "BMY"],
        "Industrials": ["CAT", "GE", "RTX", "LMT", "BA", "HON", "UNP", "DE", "MMM", "GD", "NOC", "UPS"],
        "Basic Materials": ["LIN", "FCX", "NEM", "APD", "SHW", "DOW", "NUE", "MP", "ALB"],
        "Consumer Defensive": ["WMT", "PG", "KO", "PEP", "COST", "PM", "MO", "CL", "KHC", "MDLZ"],
        "Consumer Cyclical": ["AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "LOW", "GM", "F", "CMG", "BKNG"],
        "Utilities": ["NEE", "DUK", "SO", "D", "AEP", "EXC", "CEG", "VST", "SRE"],
        "Real Estate": ["PLD", "AMT", "EQIX", "O", "SPG", "PSA", "WELL", "CCI"],
    },
    "EU": {
        "Technology": ["SAP.DE", "ASML.AS", "IFX.DE", "STMPA.PA", "CAP.PA", "DSY.PA", "NOKIA.HE", "ERIC-B.ST", "ADYEN.AS"],
        "Communication Services": ["DTE.DE", "ORA.PA", "TEF.MC", "VOD.L", "PUB.PA", "UMG.AS"],
        "Financial Services": ["ALV.DE", "BNP.PA", "SAN.MC", "ISP.MI", "UCG.MI", "DBK.DE", "INGA.AS", "CS.PA", "MUV2.DE", "HSBA.L"],
        "Energy": ["SHEL.L", "TTE.PA", "BP.L", "ENI.MI", "EQNR.OL", "REP.MC", "OMV.VI"],
        "Healthcare": ["NOVO-B.CO", "ROG.SW", "NOVN.SW", "AZN.L", "SAN.PA", "BAYN.DE", "GSK.L", "MRK.DE"],
        "Industrials": ["SIE.DE", "AIR.PA", "SU.PA", "SAF.PA", "ABBN.SW", "DHL.DE", "RR.L", "VOLV-B.ST", "DG.PA"],
        "Basic Materials": ["AI.PA", "BAS.DE", "RIO.L", "GLEN.L", "HOLN.SW", "AKZA.AS", "SIKA.SW"],
        "Consumer Defensive": ["NESN.SW", "OR.PA", "ULVR.L", "ABI.BR", "DGE.L", "BN.PA", "HEIA.AS", "CA.PA"],
        "Consumer Cyclical": ["MC.PA", "RMS.PA", "VOW3.DE", "BMW.DE", "MBG.DE", "ITX.MC", "ADS.DE", "STLAM.MI", "KER.PA", "RACE.MI"],
        "Utilities": ["IBE.MC", "ENEL.MI", "RWE.DE", "EOAN.DE", "ENGI.PA", "NG.L", "ORSTED.CO"],
        "Real Estate": ["VNA.DE", "URW.PA", "SGRO.L", "LEG.DE"],
    },
}

EU_SUFFIXES = frozenset({"DE", "F", "PA", "AS", "BR", "MI", "MC", "LS", "L", "SW", "VI", "ST", "CO", "HE", "OL", "IR", "AT", "WA", "PR", "BD"})

# Coloanele tabelului de comparabili: (cheia din `info`, eticheta, înmulțitor la afișare)
METRICS = (
    ("trailingPE", "P/E", 1.0),
    ("priceToBook", "P/BV", 1.0),
    ("returnOnEquity", "ROE (%)", 100.0),
    ("returnOnAssets", "ROA (%)", 100.0),
    ("profitMargins", "Marjă netă (%)", 100.0),
    ("debtToEquity", "Datorii/Capital (%)", 1.0),
)
MIN_PEERS_FOR_MEDIAN = 3


def peer_region(symbol):
    """"US", "EU", "BVB" sau None (altă piață) după sufixul simbolului Yahoo."""
    sym = str(symbol or "").upper().strip()
    if not sym or sym.startswith("^"):
        return None
    if "." not in sym:
        return "US"
    suffix = sym.rsplit(".", 1)[1]
    if suffix == "RO":
        return "BVB"
    return "EU" if suffix in EU_SUFFIXES else None


def peer_list(symbol, sector):
    """Comparabilii din același sector și aceeași regiune, fără simbolul însuși. Listă goală
    dacă regiunea sau sectorul nu au listă (nu se înlocuiește cu ETF-uri sau cu alt sector)."""
    region = peer_region(symbol)
    names = PEERS.get(region, {}).get(sector or "", [])
    own = str(symbol or "").upper().strip()
    return [name for name in names if name.upper() != own]


def _clean(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def median_of(values, positive_only=False):
    """(mediană, număr de observații) peste valorile disponibile, sau (None, n) sub
    MIN_PEERS_FOR_MEDIAN. `positive_only` exclude valorile ≤ 0 (un P/E negativ nu are sens
    ca multiplu și ar trage mediana în jos)."""
    usable = [v for v in (_clean(v) for v in values) if v is not None and (v > 0 or not positive_only)]
    if len(usable) < MIN_PEERS_FOR_MEDIAN:
        return None, len(usable)
    return statistics.median(usable), len(usable)


def peer_medians(rows):
    """Mediana fiecărui indicator peste comparabili: {eticheta: (mediană, n)}.
    `rows` = listă de dict-uri cu etichetele din METRICS drept chei (valori afișate)."""
    return {label: median_of([row.get(label) for row in rows], positive_only=label in ("P/E", "P/BV"))
            for _, label, _ in METRICS}


def versus_median(value, median):
    """Diferența relativă față de mediană, ca fracție (0,25 = cu 25% peste), sau None."""
    value, median = _clean(value), _clean(median)
    if value is None or median is None or median == 0:
        return None
    return (value - median) / abs(median)


# --- BVB ---------------------------------------------------------------------------------
# Yahoo nu trimite sectorul pentru simbolurile .RO, iar foaia `BVB` nu are sector: harta
# de mai jos e ținută de mână, cu denumirile de sector Yahoo (aceleași ca la SUA și UE).
# Confirmată de Daniel pe 09.10.2026. Un simbol nou din foaie apare „fără sector" până e adăugat.
BVB_SECTORS = {
    "SNP": "Energy", "SNG": "Energy", "COTE": "Energy",
    "H2O": "Utilities", "SNN": "Utilities", "EL": "Utilities", "TEL": "Utilities", "TGN": "Utilities", "PE": "Utilities",
    "TLV": "Financial Services", "BRD": "Financial Services", "BVB": "Financial Services", "TBK": "Financial Services",
    "ATB": "Healthcare", "BIO": "Healthcare", "M": "Healthcare",
    "AROBS": "Technology", "BENTO": "Technology", "SAFE": "Technology", "ALW": "Technology",
    "ONE": "Real Estate", "IMP": "Real Estate",
    "AQ": "Consumer Defensive", "WINE": "Consumer Defensive", "DN": "Consumer Defensive",
    "CFH": "Consumer Defensive", "AG": "Consumer Defensive", "BONA": "Consumer Defensive",
    "SFG": "Consumer Cyclical",
    "ARS": "Industrials", "TTS": "Industrials", "SMTL": "Industrials", "TRP": "Industrials",
}

# Comparabili din Europa Centrală și de Est, doar pentru sectoarele unde există companii cu
# același tip de afacere. La „Financial Services" lista conține bănci. Simbolurile sunt Yahoo
# și pot îmbătrâni: aplicația le raportează pe cele fără date.
BVB_REGION = "CEE"
BVB_REGIONAL_PEERS = {
    "Energy": ["OMV.VI", "MOL.BD", "PKN.WA"],
    "Utilities": ["CEZ.PR", "VER.VI", "EVN.VI", "PGE.WA", "TPE.WA", "ENA.WA"],
    "Financial Services": ["EBS.VI", "RBI.VI", "OTP.BD", "PKO.WA", "PEO.WA", "KOMB.PR"],
}
# Bursa și brokerul de asigurări nu au modelul de afaceri al unei bănci: nu primesc comparabili
# (nici regionali, nici din BVB) și nu intră în grupul de comparabili al băncilor.
BVB_STANDALONE = frozenset({"BVB", "TBK"})
BVB_NO_REGIONAL = BVB_STANDALONE            # nume vechi, păstrat pentru compatibilitate


def _bvb_code(symbol):
    """'snp.ro' sau 'SNP' -> 'SNP'."""
    sym = str(symbol or "").upper().strip()
    return sym[:-3] if sym.endswith(".RO") else sym


def bvb_sector(symbol):
    """Sectorul (denumire Yahoo) al unui simbol BVB din harta proprie, sau None dacă nu e în hartă."""
    return BVB_SECTORS.get(_bvb_code(symbol))


def bvb_sector_peers(symbol):
    """Simbolurile BVB (ca în foaie, fără '.RO') din același sector, fără simbolul însuși.
    Simbolurile din BVB_STANDALONE nu au comparabili și nu sunt comparabilii nimănui."""
    own = _bvb_code(symbol)
    sector = BVB_SECTORS.get(own)
    if sector is None or own in BVB_STANDALONE:
        return []
    return [sym for sym, sec in BVB_SECTORS.items()
            if sec == sector and sym != own and sym not in BVB_STANDALONE]


def bvb_regional_peers(symbol):
    """Comparabilii regionali (simboluri Yahoo) pentru un simbol BVB. Listă goală dacă sectorul
    nu are listă regională sau simbolul e exclus (BVB_NO_REGIONAL)."""
    own = _bvb_code(symbol)
    if own in BVB_NO_REGIONAL:
        return []
    return list(BVB_REGIONAL_PEERS.get(BVB_SECTORS.get(own) or "", []))


def peer_symbols(region, sector):
    """Lista de simboluri Yahoo pentru o regiune și un sector; regiunea BVB_REGION folosește
    BVB_REGIONAL_PEERS. Listă goală dacă nu există."""
    source = BVB_REGIONAL_PEERS if region == BVB_REGION else PEERS.get(region, {})
    return list(source.get(sector or "", []))


def sheet_peer_row(sheet_symbol, sheet_info):
    """Rândul din tabelul de comparabili pentru o companie BVB, din indicatorii foii `BVB`
    (`entry["info"]` din data/bvb_sheet.py). None dacă foaia nu are niciun indicator pentru ea.
    „Datorii/Capital" nu există în foaie cu înțelesul din Yahoo, deci rămâne None (N/A)."""
    row = {"Simbol": f"{_bvb_code(sheet_symbol)}.RO", "Capitalizare": None, "Monedă": "RON"}
    for key, label, mult in METRICS:
        value = _clean((sheet_info or {}).get(key))
        row[label] = None if value is None else value * mult
    if all(row[label] is None for _, label, _ in METRICS):
        return None
    return row


PEER_FORMATS = {"P/E": "{:.1f}", "P/BV": "{:.2f}", "ROE (%)": "{:.1f}%", "ROA (%)": "{:.1f}%",
                "Marjă netă (%)": "{:.1f}%", "Datorii/Capital (%)": "{:.0f}%"}


def format_peer_value(label, value):
    """Textul unei celule din tabelul de comparabili; valoarea lipsă (None, NaN) devine „N/A".
    Formatarea se face aici pentru că `st.dataframe` afișează „None" la celulele goale."""
    value = _clean(value)
    return "N/A" if value is None else PEER_FORMATS[label].format(value)
