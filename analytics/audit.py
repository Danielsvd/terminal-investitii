"""Auditul instituțional pe 6 piloni și pragurile pe sector. Funcții pure (fără st.*).

Mutate din app.py fără schimbarea numelor sau a semnăturilor. Regula: un indicator
lipsă (None) nu devine 0; pilonul respectiv e sărit și apare în lista „DATE LIPSĂ".
"""
from data.helpers import num


def get_sector_benchmarks(sector):
    """Definește pragurile 'normale' în funcție de industrie."""
    # Benchmarks implicite (Standard)
    benchmarks = {"pe_threshold": 20, "roe_target": 0.12, "de_max": 150}

    # Ajustări pe sectoare specifice
    sector_maps = {
        "Technology": {"pe_threshold": 35, "roe_target": 0.20, "de_max": 100},
        "Energy": {"pe_threshold": 12, "roe_target": 0.10, "de_max": 200},
        "Financial Services": {"pe_threshold": 15, "roe_target": 0.10, "de_max": 400},
        "Utilities": {"pe_threshold": 18, "roe_target": 0.08, "de_max": 300}
    }

    return sector_maps.get(sector, benchmarks)


def generate_advanced_audit_v2(info, alpha, beta, h_score):
    """
    Audit Instituțional Complet (6 Piloni).

    Valorile lipsă rămân None și pilonul respectiv e sărit (sau evaluat parțial, cu
    mențiune). Înainte, ROE lipsă cu P/E sub 0,8 × pragul sectorului arunca TypeError
    (`None > float`), iar P/E, current ratio și alpha lipsă deveneau 0.
    """
    info = info or {}
    sector = info.get('sector') or 'sector necunoscut'
    limits = get_sector_benchmarks(info.get('sector'))

    pe = num(info, 'trailingPE')
    roe = num(info, 'returnOnEquity')
    de = num(info, 'debtToEquity')
    cr = num(info, 'currentRatio')

    audit = []
    missing = [name for name, v in (("P/E", pe), ("ROE", roe), ("datorii", de),
                                     ("lichiditate", cr), ("alpha", alpha), ("beta", beta)) if v is None]
    if missing:
        audit.append(f"ℹ️ **DATE LIPSĂ:** {', '.join(missing)}. Pilonii respectivi nu sunt evaluați.")

    # --- 1. EVALUARE VS SECTOR ---
    # P/E negativ (pierdere) nu se compară cu pragul sectorului.
    if pe is not None and pe > 0:
        rel_price = pe / limits['pe_threshold']
        if rel_price < 0.8 and roe is not None and roe > limits['roe_target']:
            audit.append(f"💰 **EVALUARE:** Subevaluată în {sector}. P/E {pe:.1f} este atractiv față de media de {limits['pe_threshold']}.")
        elif rel_price < 0.8 and roe is None:
            audit.append(f"📊 **EVALUARE:** P/E {pe:.1f} sub pragul sectorului ({limits['pe_threshold']}), "
                         f"dar ROE lipsește: nu se poate confirma că prețul mic nu reflectă o rentabilitate slabă.")
        elif rel_price > 1.3:
            audit.append(f"⚖️ **EVALUARE:** Scumpă raportat la sector ({pe:.1f} vs {limits['pe_threshold']}).")
        else:
            audit.append(f"📊 **EVALUARE:** Preț corect în contextul {sector} (P/E {pe:.1f}).")

    # --- 2. PROFITABILITATE & EFICIENȚĂ ---
    if roe is not None and roe > 0.25:
        audit.append(f"🚀 **PROFITABILITATE:** Eficiență de elită (ROE {roe*100:.1f}%). Management performant.")
    elif roe is not None and roe < 0.10:
        audit.append(f"📉 **PROFITABILITATE:** Eficiență sub-optimă ({roe*100:.1f}%). Capitalul nu produce suficient.")

    # --- 3. SOLVABILITATE (DATORII) ---
    if de is not None and de > limits['de_max']:
        audit.append(f"🚩 **SOLVABILITATE:** Îndatorare peste limita sectorului ({de:.1f}%). Risc structural ridicat.")
    elif de is not None:
        audit.append(f"✅ **SOLVABILITATE:** Structură de capital sănătoasă ({de:.1f}% debt/equity).")

    # --- 4. LICHIDITATE (CASH-FLOW) ---
    if cr is not None and cr > 0:
        if cr < 1.0:
            audit.append(f"❌ **LICHIDITATE:** Critică ({cr:.2f}). Firma depinde de finanțări externe pe termen scurt.")
        elif cr > 1.5:
            audit.append(f"💧 **LICHIDITATE:** Solidă ({cr:.2f}). Există suficient 'cash' pentru siguranță.")

    # --- 5. PERFORMANȚĂ PIAȚĂ (ALPHA) ---
    if alpha is not None:
        alpha_p = alpha * 100
        if alpha > 0.02:
            audit.append(f"📈 **PERFORMANȚĂ:** Alpha Pozitiv ({alpha_p:.1f}%). Randament peste indexul de referință.")
        elif alpha < -0.02:
            audit.append(f"🥀 **PERFORMANȚĂ:** Subperformanță ({alpha_p:.1f}%). Activul pierde în fața pieței.")

    # --- 6. RISC DE PIAȚĂ (BETA) ---
    if beta is None:
        pass
    elif beta > 1.3:
        audit.append(f"🎢 **VOLATILITATE (Beta {beta:.2f}):** Risc ridicat. Mișcări mult mai ample decât piața.")
    elif beta < 0.8:
        audit.append(f"🛡️ **VOLATILITATE (Beta {beta:.2f}):** Profil defensiv. Stabilă în perioade de criză.")
    else:
        audit.append(f"⚖️ **VOLATILITATE (Beta {beta:.2f}):** Mișcare sincronizată cu piața generală.")

    return audit
