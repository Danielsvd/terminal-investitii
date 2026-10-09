"""Teste pentru data/bvb_sheet.py, pe structura reală a foii BVB (cifre inventate)."""
import pytest

from data.bvb_sheet import bvb_symbol, normalize_label, parse_bvb_sheet, reprice, sheet_number

SHEET = [
    ["Multipli", "Multilpi de preț", "AG", "SNP", "tlv", "", ""],
    ["", "Multipli de preț"],
    ["54,55", "P/E 2025", "13,46", "8,78", "7,10"],
    ["24,19", "P/E TTM ", "12,67", "30,85", "#DIV/0!"],
    ["3,97", "P/BV TTM", "1,41", "2,11", "1,90"],
    ["", "Indicatori de rentabilitate"],
    ["3,17%", "Rentabilitate active (ROA)", "3,46%", "3,29%", ""],
    ["5,50%", "Rentabilitate capital (ROE)", "12,96%", "7,41%", "22,10%"],
    [],
    ["13,99%", "Marjă netă TTM", "2,47%", "7,07%", "35,2%"],
    ["16,96%", "Marjă operațională", "6,38%", "10,70%", "n/a"],
    ["1,8417 lei", "EPS TTM", "0,1066 lei", "0,0400 lei", "1,8417 lei"],
    ["1,39", "Levier financiar", "2,74", "0,65", "9,10"],
    ["1,99", "Lichiditate curentă", "0,89", "1,34"],
    ["1,48", "Lichiditatea imediată", "0,35", "0,83"],
    ["12,28", "Debt/EBIDTA", "9,71", "0,25", ""],
    ["24,19", "P/E TTM", "99", "99", "99"],            # duplicat: se ignoră
    ["", "Raportare", "Q2 26", "Q2 26", "Q1 26"],
]


def test_sheet_number():
    assert sheet_number("24,49") == 24.49
    assert sheet_number("3,17%") == 3.17
    assert sheet_number("0,8699 lei") == 0.8699
    assert sheet_number("-0,50") == -0.5
    assert sheet_number(12.5) == 12.5
    for bad in ("", None, "#DIV/0!", "#N/A", "n/a", "lei", float("nan"), True):
        assert sheet_number(bad) is None            # lipsă, nu zero


def test_normalize_label():
    assert normalize_label("P/E TTM ") == "p/e ttm"
    assert normalize_label("Marjă  netă TTM") == "marja neta ttm"
    assert normalize_label(None) == ""


def test_parse_valorile_mapate_au_unitatile_din_info():
    data = parse_bvb_sheet(SHEET)
    assert set(data) == {"AG", "SNP", "TLV"}        # coloanele fără simbol sunt ignorate; simbolul e normalizat
    snp = data["SNP"]["info"]
    assert snp["trailingPE"] == 30.85 and snp["priceToBook"] == 2.11
    assert snp["returnOnEquity"] == pytest.approx(0.0741)     # 7,41% -> fracție
    assert snp["returnOnAssets"] == pytest.approx(0.0329)
    assert snp["profitMargins"] == pytest.approx(0.0707)
    assert snp["operatingMargins"] == pytest.approx(0.1070)
    assert snp["trailingEps"] == pytest.approx(0.04)
    assert snp["currentRatio"] == 1.34 and snp["quickRatio"] == 0.83
    assert data["SNP"]["period"] == "Q2 26" and data["TLV"]["period"] == "Q1 26"


def test_parse_erori_si_celule_goale_nu_devin_zero():
    tlv = parse_bvb_sheet(SHEET)["TLV"]
    assert "trailingPE" not in tlv["info"]          # #DIV/0!
    assert "returnOnAssets" not in tlv["info"]      # celulă goală
    assert "operatingMargins" not in tlv["info"]    # text
    assert "currentRatio" not in tlv["info"]        # rând mai scurt decât antetul
    assert tlv["info"]["returnOnEquity"] == pytest.approx(0.221)
    by_name = {row[0].strip(): (row[1], row[2]) for row in tlv["indicators"]}
    assert by_name["P/E TTM"] == ("#DIV/0!", None)


def test_parse_nu_mapeaza_indicatorii_cu_alt_inteles_si_sare_titlurile():
    data = parse_bvb_sheet(SHEET)
    assert "debtToEquity" not in data["AG"]["info"] and "forwardPE" not in data["AG"]["info"]
    names = [row[0].strip() for row in data["AG"]["indicators"]]
    assert "Levier financiar" in names and "Debt/EBIDTA" in names and "P/E 2025" in names
    assert "Multipli de preț" not in names and "Indicatori de rentabilitate" not in names
    assert names.count("P/E TTM") == 1 and data["AG"]["info"]["trailingPE"] == 12.67   # prima apariție
    assert "Raportare" not in names


def test_parse_foaie_goala():
    assert parse_bvb_sheet([]) == {} and parse_bvb_sheet(None) == {} and parse_bvb_sheet([["a", "b", "X"]]) == {}


def test_bvb_symbol():
    assert bvb_symbol("SNP.RO") == "SNP" and bvb_symbol("snp.ro") == "SNP"
    assert bvb_symbol("AAPL") is None and bvb_symbol("SAP.DE") is None and bvb_symbol(None) is None


def test_media_din_foaie_si_mediana_calculata():
    rows = {row[0].strip(): row for row in parse_bvb_sheet(SHEET)["SNP"]["indicators"]}
    # P/E TTM: AG 12,67, SNP 30,85, TLV #DIV/0! -> mediana celor două = 21,76; media e textul din coloana A
    assert rows["P/E TTM"][3] == "24,19" and rows["P/E TTM"][4] == "21,76"
    # ROE: 12,96%, 7,41%, 22,10% -> mediana 12,96%
    assert rows["Rentabilitate capital (ROE)"][3] == "5,50%" and rows["Rentabilitate capital (ROE)"][4] == "12,96%"
    # EPS: 0,1066 / 0,0400 / 1,8417 lei -> mediana 0,1066 lei
    assert rows["EPS TTM"][4] == "0,1066 lei"
    no_avg = parse_bvb_sheet([["", "x", "A", "B"], ["#DIV/0!", "P/E TTM", "#DIV/0!", ""]])["A"]["indicators"][0]
    assert no_avg[3] == "N/A" and no_avg[4] == "N/A"


def test_reprice_la_pretul_curent():
    # foaia: EPS 0,04, P/E 30,85, P/BV 2,11 -> BVPS = 0,04 × 30,85 / 2,11 = 0,58483
    # la prețul 1,23: P/E = 1,23 / 0,04 = 30,75; P/BV = 1,23 / 0,58483 = 2,1032
    out = reprice({"trailingEps": 0.04, "trailingPE": 30.85, "priceToBook": 2.11}, 1.23)
    assert out["trailingPE"] == pytest.approx(30.75)
    assert out["bookValue"] == pytest.approx(0.58483, abs=1e-5)
    assert out["priceToBook"] == pytest.approx(2.1032, abs=1e-4)


def test_reprice_nu_inventeaza():
    assert reprice({"trailingEps": -0.1, "trailingPE": None, "priceToBook": 1.5}, 2.0) == {}      # pierdere
    assert reprice({"trailingEps": 0.04, "trailingPE": 30.85, "priceToBook": 2.11}, None) == {}   # fără preț
    assert reprice({"trailingEps": 0.04, "priceToBook": 2.11}, 1.23) == {"trailingPE": pytest.approx(30.75)}
    assert reprice({}, 1.23) == {}


from data.bvb_sheet import unmapped_info_keys  # noqa: E402


def test_etichetele_roe_roa_cu_ttm_se_mapeaza():
    # Din 09.10.2026 foaia are „Rentabilitate active (ROA) TTM" și „Rentabilitate capital (ROE) TTM".
    sheet = [["Multipli", "Multilpi de preț", "SNP"],
             ["11,44%", "Rentabilitate active (ROA) TTM", "4,66%"],
             ["19,79%", "Rentabilitate capital (ROE) TTM", "7,67%"]]
    info = parse_bvb_sheet(sheet)["SNP"]["info"]
    assert info["returnOnAssets"] == pytest.approx(0.0466)
    assert info["returnOnEquity"] == pytest.approx(0.0767)


def test_unmapped_info_keys_semnaleaza_randurile_redenumite():
    assert unmapped_info_keys(SHEET) == []                      # foaia de test are toate rândurile mapate
    renamed = [row[:] for row in SHEET]
    for row in renamed:
        if len(row) > 1 and row[1] == "Rentabilitate capital (ROE)":
            row[1] = "ROE anual"
    assert unmapped_info_keys(renamed) == ["returnOnEquity"]
    assert "trailingPE" in unmapped_info_keys([])               # foaie goală: nimic găsit
