"""Teste pentru analytics/audit.py: valori sintetice, calculate de mână."""
from analytics.audit import generate_advanced_audit_v2, get_sector_benchmarks


def _lines(audit, tag):
    return [line for line in audit if tag in line]


def test_benchmarks_sector_cunoscut_si_implicit():
    assert get_sector_benchmarks("Technology") == {"pe_threshold": 35, "roe_target": 0.20, "de_max": 100}
    assert get_sector_benchmarks("Sector inventat") == {"pe_threshold": 20, "roe_target": 0.12, "de_max": 150}
    assert get_sector_benchmarks(None) == {"pe_threshold": 20, "roe_target": 0.12, "de_max": 150}


def test_roe_lipsa_cu_pe_mic_nu_mai_arunca():
    # Technology: prag P/E 35. P/E 10 → 10/35 = 0,286 < 0,8. Înainte: TypeError (None > 0.20).
    info = {"sector": "Technology", "trailingPE": 10, "returnOnEquity": None, "debtToEquity": 50, "currentRatio": 2}
    audit = generate_advanced_audit_v2(info, 0.01, 1.0, 6)
    assert "ROE" in _lines(audit, "DATE LIPSĂ")[0]
    evaluare = _lines(audit, "EVALUARE")
    assert len(evaluare) == 1 and "ROE lipsește" in evaluare[0]
    assert not _lines(audit, "Subevaluată")


def test_subevaluata_cand_roe_peste_tinta():
    # P/E 10 < 0,8 × 35 = 28 și ROE 30% > 20% → subevaluată.
    info = {"sector": "Technology", "trailingPE": 10, "returnOnEquity": 0.30, "debtToEquity": 50, "currentRatio": 2}
    audit = generate_advanced_audit_v2(info, 0.0, 1.0, 8)
    assert "Subevaluată" in _lines(audit, "EVALUARE")[0]
    assert "30.0%" in _lines(audit, "PROFITABILITATE")[0]


def test_scumpa_peste_1_3_din_prag():
    # Prag implicit 20; P/E 27 → 1,35 > 1,3.
    info = {"sector": "Industrials", "trailingPE": 27, "returnOnEquity": 0.15, "debtToEquity": 80, "currentRatio": 1.2}
    audit = generate_advanced_audit_v2(info, 0.0, 1.0, 6)
    assert "Scumpă" in _lines(audit, "EVALUARE")[0]


def test_toate_lipsa_doar_linia_de_date_lipsa():
    audit = generate_advanced_audit_v2({}, None, None, None)
    assert len(audit) == 1
    linie = audit[0]
    for nume in ("P/E", "ROE", "datorii", "lichiditate", "alpha", "beta"):
        assert nume in linie


def test_info_none_nu_arunca():
    assert len(generate_advanced_audit_v2(None, None, None, None)) == 1


def test_alpha_lipsa_nu_devine_zero_si_nu_apare_pilonul():
    info = {"sector": "Technology", "trailingPE": 30, "returnOnEquity": 0.18, "debtToEquity": 50, "currentRatio": 1.2}
    audit = generate_advanced_audit_v2(info, None, 1.0, 6)
    assert not _lines(audit, "PERFORMANȚĂ")
    assert "alpha" in _lines(audit, "DATE LIPSĂ")[0]
    audit = generate_advanced_audit_v2(info, 0.05, 1.0, 6)
    assert "5.0%" in _lines(audit, "PERFORMANȚĂ")[0]


def test_lichiditate_lipsa_si_critica():
    baza = {"sector": "Technology", "trailingPE": 30, "returnOnEquity": 0.18, "debtToEquity": 50}
    assert not _lines(generate_advanced_audit_v2(dict(baza, currentRatio=None), 0.0, 1.0, 6), "LICHIDITATE")
    assert "Critică (0.50)" in _lines(generate_advanced_audit_v2(dict(baza, currentRatio=0.5), 0.0, 1.0, 6), "LICHIDITATE")[0]


def test_pe_negativ_nu_se_compara_cu_sectorul():
    info = {"sector": "Technology", "trailingPE": -5, "returnOnEquity": -0.1, "debtToEquity": 50, "currentRatio": 1.2}
    assert not _lines(generate_advanced_audit_v2(info, 0.0, 1.0, 3), "EVALUARE")


def test_solvabilitate_pe_limita_sectorului():
    # Financial Services: limită 400 → 450 peste; Technology: limită 100 → 150 peste, 90 sub.
    fin = {"sector": "Financial Services", "trailingPE": 10, "returnOnEquity": 0.12, "debtToEquity": 450, "currentRatio": None}
    assert "🚩" in _lines(generate_advanced_audit_v2(fin, 0.0, 1.0, 5), "SOLVABILITATE")[0]
    tech = {"sector": "Technology", "trailingPE": 30, "returnOnEquity": 0.22, "debtToEquity": 150, "currentRatio": 1.6}
    assert "🚩" in _lines(generate_advanced_audit_v2(tech, 0.0, 1.0, 5), "SOLVABILITATE")[0]
    assert "✅" in _lines(generate_advanced_audit_v2(dict(tech, debtToEquity=90), 0.0, 1.0, 5), "SOLVABILITATE")[0]


def test_beta_praguri():
    info = {"sector": "Technology"}
    assert "Risc ridicat" in _lines(generate_advanced_audit_v2(info, None, 1.5, None), "VOLATILITATE")[0]
    assert "defensiv" in _lines(generate_advanced_audit_v2(info, None, 0.6, None), "VOLATILITATE")[0]
    assert "sincronizată" in _lines(generate_advanced_audit_v2(info, None, 1.0, None), "VOLATILITATE")[0]
