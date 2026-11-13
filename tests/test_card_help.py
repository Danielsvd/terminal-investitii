"""Textele „?” din Indicatori Fundamentali: sursa corectă pentru fiecare valoare."""
from ui.card_help import FUND_HELP, fund_help, value_source


def test_yahoo_implicit():
    assert value_source({}, "trailingPE") == "**Sursă:** Yahoo Finance."


def test_bvb_recalculat_are_prioritate():
    info = {"_from_bvb_sheet": ["trailingPE", "priceToBook"], "_bvb_repriced": ["trailingPE"], "_bvb_period": "S1 2026"}
    assert value_source(info, "trailingPE") == "**Sursă:** foaia BVB, recalculat la prețul curent."
    assert value_source(info, "priceToBook") == "**Sursă:** foaia BVB (raportare S1 2026)."


def test_din_situatii():
    info = {"_from_statements": ["returnOnEquity"]}
    assert "situațiile financiare" in value_source(info, "returnOnEquity")


def test_gn_ia_sursa_componentelor():
    info = {"_from_bvb_sheet": ["trailingPE", "priceToBook"], "_bvb_repriced": ["trailingPE", "priceToBook"]}
    assert value_source(info, "gn") == "**Sursă:** foaia BVB, recalculat la prețul curent."
    mixed = {"_bvb_repriced": ["trailingPE"]}
    assert value_source(mixed, "gn") == "**Sursă:** foaia BVB, recalculat la prețul curent; Yahoo Finance."


def test_fiecare_text_are_si_sursa():
    for key in FUND_HELP:
        text = fund_help({}, key)
        assert text.startswith(FUND_HELP[key]) and text.endswith("**Sursă:** Yahoo Finance.")


def test_info_none():
    assert value_source(None, "ROE" if False else "returnOnEquity") == "**Sursă:** Yahoo Finance."


def test_explicatii_cu_praguri_pe_randuri():
    for key in ("trailingPE", "forwardPE", "peg_hist", "peg_analysts", "dividendRate", "priceToBook", "gn",
                "returnOnAssets", "returnOnEquity", "profitMargins", "operatingMargins", "debtToEquity",
                "currentRatio", "quickRatio"):
        text = FUND_HELP[key]
        assert text.startswith("**Ce e:**") and "\n\n**Calcul:**" in text and "**Praguri orientative" in text, key
    assert "**Praguri" not in FUND_HELP["trailingEps"]          # sumele nu primesc praguri
    assert "10–15 corect" in FUND_HELP["trailingPE"] and "22,5" in FUND_HELP["gn"]
