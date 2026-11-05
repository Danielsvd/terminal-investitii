"""Listele de simboluri nu au voie să scadă (Daniel, 05.11.2026: „peste tot mai multe companii, niciodată mai puține”).

Instantaneul de mai jos = listele din `main` (0e3b82f) la 05.11.2026. Fiecare listă curentă trebuie să le conțină pe
toate. Din 05.11.2026 listele sunt unice, câte un fișier pe piață (data/universes.py): fiecare simbol din instantaneu
trebuie să fie în fișierul pieței lui (`.RO` → BVB, sufix european → Europa, fără sufix → SUA, inclusiv ADR-urile).
Se pot adăuga simboluri oricând; un simbol scos intenționat (delistat, redenumit) se scoate și de aici, cu mențiune
în PR. Fără rețea.
"""
import ast
from pathlib import Path

from data import universes

ROOT = Path(__file__).resolve().parent.parent

BVB_MIN = {
    'AG.RO', 'ALW.RO', 'AQ.RO', 'AROBS.RO', 'ARS.RO', 'ASC.RO', 'AST.RO', 'ATB.RO', 'BENTO.RO', 'BIO.RO', 'BRD.RO',
    'BRK.RO', 'CFH.RO', 'COTE.RO', 'DIGI.RO', 'DN.RO', 'EBS.RO', 'EL.RO', 'FP.RO', 'H2O.RO', 'IARV.RO', 'IMP.RO',
    'M.RO', 'MILK.RO', 'ONE.RO', 'PBK.RO', 'PE.RO', 'SAFE.RO', 'SFG.RO', 'SMTL.RO', 'SNG.RO', 'SNN.RO', 'SNP.RO',
    'TBK.RO', 'TEL.RO', 'TGN.RO', 'TLV.RO', 'TRP.RO', 'TTS.RO', 'WINE.RO',
}

US_BRIEFING_MIN = {
    'AAPL', 'AIG', 'AMD', 'AMGN', 'AMZN', 'ANET', 'ARM', 'ASML', 'AXP', 'BA', 'BAC', 'BHP', 'BIIB', 'C', 'CEG', 'CG',
    'CMG', 'COO', 'COP', 'CRM', 'CRWV', 'CVX', 'D', 'DVN', 'DXCM', 'FN', 'FSLR', 'GD', 'GOOGL', 'GPOR', 'GS', 'HON',
    'INOD', 'INTC', 'JNJ', 'JPM', 'KO', 'KTOS', 'LLY', 'LMT', 'LNG', 'MA', 'MCD', 'MDB', 'META', 'METC', 'MP', 'MRK',
    'MS', 'MSFT', 'MU', 'NET', 'NFLX', 'NVDA', 'NVO', 'NVS', 'OKLO', 'ORCL', 'OXY', 'PANW', 'PEP', 'PFE', 'PG',
    'PLTR', 'PM', 'QCOM', 'RIO', 'RTX', 'SCHW', 'SMR', 'SNDK', 'SNOW', 'SNY', 'SOFI', 'SPCX', 'T', 'TSLA', 'TSM',
    'TTE', 'UUUU', 'V', 'VALE', 'VG', 'WFC', 'WMT', 'XOM',
}

US_STOCKS_MIN = {
    'AAPL', 'AIG', 'AMD', 'AMGN', 'AMZN', 'ANET', 'ARM', 'ASML', 'AXP', 'BA', 'BAC', 'BHP', 'BIIB', 'C', 'CEG', 'CG',
    'CMG', 'COO', 'COP', 'CRM', 'CRWV', 'CVX', 'D', 'DVN', 'DXCM', 'FN', 'FSLR', 'GD', 'GOOGL', 'GPOR', 'GS', 'HON',
    'INOD', 'INTC', 'JNJ', 'JPM', 'KO', 'KTOS', 'LLY', 'LMT', 'LNG', 'MA', 'MCD', 'MDB', 'META', 'METC', 'MP', 'MRK',
    'MS', 'MSFT', 'MU', 'NET', 'NFLX', 'NVDA', 'NVO', 'NVS', 'OKLO', 'ORCL', 'OXY', 'PANW', 'PEP', 'PFE', 'PG',
    'PLTR', 'PM', 'QCOM', 'RIO', 'RTX', 'SCHW', 'SMR', 'SNDK', 'SNOW', 'SNY', 'SOFI', 'SPCX', 'T', 'TSLA', 'TSM',
    'TTE', 'UUUU', 'V', 'VALE', 'VG', 'WFC', 'WMT', 'XOM',
}

EU_STOCKS_MIN = {
    'AIR.PA', 'ALV.DE', 'ASML', 'AZN', 'BA.L', 'BATS.L', 'BAYN.DE', 'BMW.DE', 'BNP.PA', 'BSP.DE', 'DBK.DE', 'DGE.L',
    'DTE.DE', 'ENR.DE', 'GLNCY', 'HSBA.L', 'LDO.MI', 'MBG.DE', 'MC.PA', 'OR.PA', 'RACE.MI', 'REL.L', 'RHM.DE',
    'RNO.PA', 'SAN.PA', 'SAP.DE', 'SHEL.L', 'SIE.DE', 'TTE.PA', 'UCG.MI', 'ULVR.L', 'VOW3.DE', 'ZAL.DE',
}

SCANNER_MIN = {
    '🇷🇴 BVB (România - BET)': {
        'AG.RO', 'ALW.RO', 'AQ.RO', 'AROBS.RO', 'ARS.RO', 'ASC.RO', 'AST.RO', 'ATB.RO', 'BENTO.RO', 'BIO.RO', 'BRD.RO',
        'BRK.RO', 'CFH.RO', 'COTE.RO', 'DIGI.RO', 'DN.RO', 'EBS.RO', 'EL.RO', 'FP.RO', 'H2O.RO', 'IARV.RO', 'IMP.RO',
        'M.RO', 'MILK.RO', 'ONE.RO', 'PBK.RO', 'PE.RO', 'SAFE.RO', 'SFG.RO', 'SMTL.RO', 'SNG.RO', 'SNN.RO', 'SNP.RO',
        'TBK.RO', 'TEL.RO', 'TGN.RO', 'TLV.RO', 'TRP.RO', 'TTS.RO', 'TVBETETF.RO', 'WINE.RO',
    },
    '🇺🇸 SUA - Tech & Growth (Nasdaq 100)': {
        'AAPL', 'AFRM', 'AMD', 'AMZN', 'ANET', 'ARM', 'ASML', 'AVGO', 'BKNG', 'CDNS', 'CLS', 'CMCSA', 'COST', 'CRWD',
        'CRWV', 'CSCO', 'CSX', 'GILD', 'GOOGL', 'HOOD', 'INOD', 'INTC', 'ISRG', 'KLAC', 'LRCX', 'MDLZ', 'META', 'MSFT',
        'MSTR', 'MU', 'NET', 'NFLX', 'NVDA', 'ORCL', 'PANW', 'PEP', 'PLTR', 'PYPL', 'QCOM', 'ROKU', 'SBUX', 'SNDK',
        'SNPS', 'TMUS', 'TSLA', 'TSM', 'TXN', 'U', 'ZS',
    },
    '🇺🇸 SUA - Industrial & Finance (Dow/S&P)': {
        'ABBV', 'AEM', 'AMGN', 'AXP', 'BA', 'BAC', 'BHP', 'BIIB', 'BLK', 'BMY', 'BRK-B', 'C', 'CAT', 'CMG', 'COP', 'CVS',
        'CVX', 'DE', 'DHR', 'DVN', 'EOG', 'GD', 'GE', 'GPOR', 'GS', 'HAL', 'HON', 'JNJ', 'JPM', 'LLY', 'LMT', 'MA',
        'METC', 'MMM', 'MP', 'MPC', 'MRK', 'MS', 'NOC', 'NVO', 'NVS', 'OKLO', 'OXY', 'PFE', 'RIO', 'RTX', 'SLB', 'SNY',
        'SPCX', 'T', 'TMO', 'UNH', 'UNP', 'UUUU', 'V', 'VG', 'VLO', 'WFC', 'XOM',
    },
    '🇪🇺 Europa - Germania (DAX 40)': {
        'ADS.DE', 'AIR.DE', 'ALV.DE', 'BAS.DE', 'BAYN.DE', 'BEI.DE', 'BMW.DE', 'BSP.DE', 'CBK.DE', 'CON.DE', 'DB1.DE',
        'DBK.DE', 'DHL.DE', 'DTE.DE', 'DTG.DE', 'ENR.DE', 'EOAN.DE', 'HEI.DE', 'HEN3.DE', 'IFX.DE', 'MBG.DE', 'MTX.DE',
        'MUV2.DE', 'PUM.DE', 'RHM.DE', 'RWE.DE', 'SAP.DE', 'SIE.DE', 'SY1.DE', 'VNA.DE', 'VOW3.DE', 'ZAL.DE',
    },
    '🇪🇺 Europa - Franța (CAC 40)': {
        'ACA.PA', 'AI.PA', 'AIR.PA', 'BN.PA', 'BNP.PA', 'CAP.PA', 'CS.PA', 'DG.PA', 'DGE.L', 'DSY.PA', 'EL.PA', 'EN.PA',
        'ENG.PA', 'GLNCY', 'HO.PA', 'KER.PA', 'LR.PA', 'MC.PA', 'ML.PA', 'OR.PA', 'ORA.PA', 'RI.PA', 'RMS.PA', 'RNO.PA',
        'SAF.PA', 'SAN.PA', 'STLAP.PA', 'SU.PA', 'TTE.PA', 'VIV.PA',
    },
    '🇬🇧 UK & Others (FTSE/Global)': {
        'AAL.L', 'ASML', 'AZN', 'BA.L', 'BABA', 'BARC.L', 'BATS.L', 'BIDU', 'BP', 'DGE.L', 'GLNCY', 'GSK.L', 'HSBA.L',
        'JD', 'LDO.MI', 'LLOY.L', 'LSEG.L', 'NVO', 'NWG.L', 'REL.L', 'RIO', 'RR.L', 'SHEL.L', 'SONY', 'TCEHY', 'TSCO.L',
        'TSM', 'ULVR.L', 'VOD.L',
    },
}



def _all_old():
    out = set(BVB_MIN) | US_BRIEFING_MIN | US_STOCKS_MIN | EU_STOCKS_MIN
    for v in SCANNER_MIN.values():
        out |= v
    return out - {"TVBETETF.RO"}                       # ETF-ul BET: proxy de indice, nu companie (adăugat separat)


def test_niciun_simbol_vechi_nu_lipseste_din_listele_unice():
    now = {m: set(universes.symbols(m)) for m in universes.UNIVERSE_FILES}
    lost = {s for s in _all_old() if s not in now[universes.market_of(s)]}
    assert lost == set()
    assert {"BENTO.RO", "AG.RO", "SMTL.RO"} <= now["bvb"]                  # deținerile lui Daniel


def test_listele_unice_nu_se_amesteca():
    seen = {}
    for m in universes.UNIVERSE_FILES:
        syms = universes.symbols(m)
        assert syms, m
        assert all(universes.market_of(s) == m for s in syms), m           # BVB nu intră în Europa și invers
        for s in syms:
            assert s not in seen, (s, seen.get(s), m)                       # un simbol într-o singură piață
            seen[s] = m
        assert sum(len(v) for v in universes.groups(m).values()) == len(syms)   # fiecare simbol o dată în grupuri


def test_app_nu_mai_are_liste_scrise_de_mana():
    """Lista se adaugă într-un singur loc: în app.py nu mai există liste literale lungi de simboluri."""
    tree = ast.parse((ROOT / "app.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ("BRIEFING_US_UNIVERSE", "us_stocks", "eu_stocks", "tickers_map"):
                literal = [e for lst in ast.walk(node.value) if isinstance(lst, ast.List) for e in lst.elts
                           if isinstance(e, ast.Constant) and isinstance(e.value, str)]
                assert len(literal) < 5, (node.targets[0].id, literal[:5])


def test_piata_dupa_sufix():
    assert [universes.market_of(s) for s in ("TLV.RO", "SAP.DE", "BRK-B", "AZN", "HSBA.L")] == ["bvb", "eu", "us", "us", "eu"]
