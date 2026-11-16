"""Parserul rapoartelor Anexa 12 (Pilon II). Valori sintetice; ISIN-urile sunt identificatori publici reali."""
from datetime import date

import pytest

from tools.pilon2 import parse
from tools.pilon2.parse import (EQUITY, check, isin_checksum_ok, join_split_value, normalize, parse_amount,
                                parse_lines, parse_report_date)


def test_parse_amount_formats():
    assert parse_amount("1,058,918,959.06") == pytest.approx(1058918959.06)
    assert parse_amount("-35,383,211.30") == pytest.approx(-35383211.30)
    assert parse_amount("3.996") == pytest.approx(3.996)
    assert parse_amount("0.000") == 0.0
    assert parse_amount("1.234.567,89") == pytest.approx(1234567.89)
    assert parse_amount("24,49") == pytest.approx(24.49)
    assert parse_amount("12,345") == 12345.0          # o singură virgulă urmată de 3 cifre = mii
    assert parse_amount("-") is None
    assert parse_amount("") is None
    assert parse_amount("abc") is None


def test_isin_checksum():
    assert isin_checksum_ok("ROTLVAACNOR1")
    assert isin_checksum_ok("RO4Q0Z5RO1B6")
    assert isin_checksum_ok("NL0012294474")
    assert not isin_checksum_ok("ROTLVAACNOR2")
    assert not isin_checksum_ok("DEP-RO-37395")


def test_normalize_removes_diacritics():
    assert normalize("d) Acţiuni  Ţară ș ă") == "d) actiuni tara s a"


def test_report_date_variants():
    assert parse_report_date(["Structura portofoliului de investitii", "la data de: 28.02.2026"]) == date(2026, 2, 28)
    assert parse_report_date(["Structura portofoliului de investiții la data de 31.05.2025"]) == date(2025, 5, 31)
    assert parse_report_date(["Pondere in activul", "31-01-2026"]) == date(2026, 1, 31)     # BT
    assert parse_report_date(["la data de: 05/31/2020"]) == date(2020, 5, 31)              # ll/zz/aaaa
    assert parse_report_date(["la data de 31/12/2025"]) == date(2025, 12, 31)
    assert parse_report_date(["fără dată"]) is None


# Raport complet, așezarea Vital/Metropolitan: un rând pe deținere, procent cu „%”.
VITAL = """Structura portofoliului de investitii
la data de: 28.02.2026
Cod depozitar: DEP-RO-373958
Administrator fond de pensii: CARPATHIA PENSII - S.A.F.P.P. S.A.
Fond de pensii: Fond de pensii administrat privat VITAL
Cod fond de pensii: FP2-54
Portofoliul de instrumente financiare
Denumire emitent ISIN Valoare actualizata (lei) Pondere in total active (%)
Col 1 Col 2 Col 3 Col 4 Col 5
1. Instrumente ale pietei monetare, din care: 1,000.00 0.010%
a) Conturi curente 1,000.00 0.010%
Citibank Romania 1,000.00 0.010%
2. Valori mobiliare tranzactionate, din care: 9,000,000.00 90.000%
a) Titluri de stat cu scadenta mai mare de 1 an 5,000,000.00 50.000%
Ministerul Finantelor Publice RO1227DBN011 5,000,000.00 50.000%
d) Actiuni 4,000,000.00 40.000%
Banca Transilvania SA ROTLVAACNOR1 2,500,000.00 25.000%
Axa SA FR0000120628 500,000.00 5.000%
Digi Communications NV NL0012294474 1,000,000.00 10.000%
3. OPCVM 900,000.00 9.000%
BT Maxim / BT Asset Management ROFDIN000101 900,000.00 9.000%
5. Private equity, din care: 99,000.00 0.990%
a) Actiuni la societati 99,000.00 0.990%
Exemplu Privat SA ROTLVAACNOR1 99,000.00 0.990%
8. Alte instrumente financiare - 0.000%
a) Dividende de incasat 0.00 0.000%
Total activ 10,000,000.00
""".splitlines()


def test_vital_layout_rows_and_classes():
    rep = parse_lines(VITAL)
    assert rep.fund_code == "FP2-54" and rep.fund == "Vital"
    assert rep.report_date == date(2026, 2, 28)
    eq = rep.equities()
    assert [r["ISIN"] for r in eq] == ["ROTLVAACNOR1", "FR0000120628", "NL0012294474"]
    assert eq[0]["Denumire"] == "Banca Transilvania SA"
    assert eq[0]["Valoare_RON"] == 2500000.0 and eq[0]["Pondere"] == 25.0
    classes = {r["ISIN"]: r["Clasa"] for r in rep.rows}
    assert classes["RO1227DBN011"] == "titluri_stat"
    assert classes["ROFDIN000101"] == "opcvm"
    # acțiunile din private equity (5.a) NU sunt acțiuni listate
    pe = [r for r in rep.rows if r["Denumire"] == "Exemplu Privat SA"]
    assert pe and pe[0]["Clasa"] == "private_equity"
    assert rep.equity_header_total == 4000000.0
    assert rep.total_assets == 10000000.0
    assert check(rep) == []


def test_bt_layout_names_and_numbers_on_separate_lines():
    lines = """Structura detaliata a portofoliului de investitii la data de
31-01-2026
SAP-RO-22013894
Fondul de Pensii Administrat Privat BRD
FP2-83
2. Valori mobiliare tranzacţionate, din care: 300.00 3.00%
a) Titluri de stat cu scadența mai mare de 1 an 100.00 1.00%
Ministerul Finantelor Publice
RO1631DBN055 100.00 1.00%
d) Acţiuni 200.00 2.00%
Hidroelectrica SA RO4Q0Z5RO1B6
150.00 1.50%
Henkel AG Co. KGaA DE0006048432 50.00 0.50%
3. OPCVM 30.00 0.30%
iShares STOXX Europe 600 Health Care UCITS ETF
(DE) / BlackRock Inc
DE000A0Q4R36
30.00 0.30%
""".splitlines()
    rep = parse_lines(lines)
    assert rep.fund == "BT Pilon 2" and rep.report_date == date(2026, 1, 31)
    by = {r["ISIN"]: r for r in rep.rows}
    assert by["RO1631DBN055"]["Denumire"] == "Ministerul Finantelor Publice"
    assert by["RO1631DBN055"]["Clasa"] == "titluri_stat"
    assert by["RO4Q0Z5RO1B6"]["Denumire"] == "Hidroelectrica SA"
    assert by["RO4Q0Z5RO1B6"]["Valoare_RON"] == 150.0 and by["RO4Q0Z5RO1B6"]["Clasa"] == EQUITY
    assert by["DE000A0Q4R36"]["Denumire"] == "iShares STOXX Europe 600 Health Care UCITS ETF (DE) / BlackRock Inc"
    assert by["DE000A0Q4R36"]["Clasa"] == "opcvm"
    assert check(rep) == []


def test_nn_layout_weight_without_percent_and_aripi_isin_only():
    nn = """Cod fond de pensii FP2-31
la data de 31.01.2026
2. Valori mobiliare tranzacţionate, din care: 1,500.00 1.50
d. Acţiuni 1,500.00 1.50
OMV Petrom SA ROSNPPACNOR9 1,000.00 1.00
RO4Q0Z5RO1B6 500.00 0.50
""".splitlines()
    rep = parse_lines(nn)
    eq = rep.equities()
    assert [r["Pondere"] for r in eq] == [1.0, 0.5]
    assert eq[1]["ISIN"] == "RO4Q0Z5RO1B6" and eq[1]["Denumire"] == ""   # doar ISIN: fără nume inventat
    assert check(rep) == []


def test_page_header_is_not_taken_as_issuer_name():
    lines = """FP2-54
la data de: 28.02.2026
2. Valori mobiliare tranzactionate, din care: 300.00 3.000%
d) Actiuni 300.00 3.000%
RO4Q0Z5RO1B6 100.00 1.000%
Structura portofoliului de investitii
Pondere in total active
ROSNPPACNOR9 200.00 2.000%
""".splitlines()
    eq = parse_lines(lines).equities()
    assert [r["Denumire"] for r in eq] == ["", ""]


def test_check_reports_sum_mismatch_bad_isin_and_missing_parts():
    lines = """la data de: 28.02.2026
2. Valori mobiliare tranzactionate, din care: 300.00 3.000%
d) Actiuni 300.00 3.000%
Banca Transilvania SA ROTLVAACNOR1 100.00 1.000%
Gresit SA ROTLVAACNOR2 100.00 1.000%
""".splitlines()
    problems = check(parse_lines(lines))
    assert any("codul fondului" in p for p in problems)
    assert any("diferă de totalul secțiunii" in p for p in problems)   # 200 vs 300
    assert any("cifră de control" in p for p in problems)
    assert any("nicio acțiune" in p for p in check(parse_lines(["FP2-54", "la data de: 28.02.2026"])))


# PDF real, generat aici și citit cu pdfplumber (aceeași cale ca pe server).
def _make_pdf(path, lines):
    canvas_mod = pytest.importorskip("reportlab.pdfgen.canvas")
    from reportlab.lib.pagesizes import A4

    c = canvas_mod.Canvas(str(path), pagesize=A4)
    y = 800
    for line in lines:
        if y < 60:
            c.showPage()
            y = 800
        x = 40
        for cell in line.split("|"):
            c.drawString(x, y, cell)
            x += 260 if x == 40 else 110
        y -= 14
    c.save()


def test_parse_pdf_end_to_end(tmp_path):
    pytest.importorskip("pdfplumber")
    from tools.pilon2.parse import parse_pdf

    rows = ["Structura portofoliului de investitii", "la data de: 30.04.2026", "Cod fond de pensii: FP2-96",
            "2. Valori mobiliare tranzactionate, din care:|9,000.00|90.000%",
            "d) Actiuni|9,000.00|90.000%"]
    rows += [f"Banca Transilvania SA ROTLVAACNOR1|{1000 + i:,}.00|1.000%" for i in range(0, 60, 10)]
    rows += ["Digi Communications NV NL0012294474|2,850.00|28.500%"]
    pdf = tmp_path / "anexa12.pdf"
    _make_pdf(pdf, rows)
    rep = parse_pdf(pdf)
    assert rep.fund == "Metropolitan Life" and rep.report_date == date(2026, 4, 30)
    eq = rep.equities()
    assert len(eq) == 7
    assert sum(r["Valoare_RON"] for r in eq) == pytest.approx(9000.0)
    assert check(rep) == []


def test_summary_and_cli(tmp_path, capsys):
    pytest.importorskip("pdfplumber")
    from tools.pilon2.parse import main, summary

    text = summary(parse_lines(VITAL))
    assert text[0].startswith("Fond: Vital (FP2-54) · data raportului: 2026-02-28")
    assert any("ROTLVAACNOR1" in t for t in text)
    assert text[-1] == "Probleme: niciuna"
    assert main([]) == 2
    pdf = tmp_path / "gol.pdf"
    _make_pdf(pdf, ["fara continut util"])
    assert main([str(pdf)]) == 1          # raport fără cod de fond / dată / acțiuni → cod 1
    assert "Probleme:" in capsys.readouterr().out


def test_published_date_variants():
    from tools.pilon2.parse import parse_published_date as pub
    assert pub(["Număr ieșire: 1226411/11.07.2025"]) == date(2025, 7, 11)
    assert pub(["Numar iesire 1238802/29.08.2024"]) == date(2024, 8, 29)
    assert pub(["Numar intern: 1223275/30.07.2024"]) == date(2024, 7, 30)
    assert pub(["Nr. înregistrare: 50290/27.04.2023"]) == date(2023, 4, 27)
    assert pub(["Cod depozitar: DEP-RO-374199    Nr. Inregistare: 1180237/28.07.2023"]) == date(2023, 7, 28)
    assert pub(["Nr. Inreg.: 1183975/29.04.2024"]) == date(2024, 4, 29)
    assert pub(["1307878/12.12.2025"]) == date(2025, 12, 12)
    assert pub(["la data de: 31.08.2025"]) is None



# Rapoartele BCR din 2024 (16.11.2026): pdfplumber desparte prima cifră a valorii. Așezarea e cea reală
# (31.12.2024); valorile sunt sintetice. Suma rândurilor = 1,000,000,000.00 = totalul din antet.
BCR_2024_EQUITY = [
    "d. Acţiuni 1,000,000,000.00 10.0000%",
    "Banca Transilvania SA ROTLVAACNOR1 400,000,000.00 4.0000%",
    "Hidroelectrica SA RO4Q0Z5RO1B6 3 00,000,000.00 3.0000%",
    "Digi Communications NV NL0012294474 9 9,000,000.00 0.9900%",
    "Romgaz SA ROSNGNACNOR3 150,000,000.00 1.5000%",
    "Sphera Franchise Group SA ROSFGPACNOR4 2 0,996,300.00 0.2100%",
    "DHL Group DE0005552004 4 ,732,557.70 0.0473%",
    "Conpet SA ROCOTEACNOR7 25,271,142.30 0.2527%",
    "e. Obligatiuni BERD, BEI, BM - 0.0000%",
    "f. Alte obligatiuni emise de organisme straine neguvernamentale (investment grade) 2 0,000,000.00 0.2000%",
    "Black Sea Trade and Development Bank XS2337703966 2 0,000,000.00 0.2000%",
    "2/3",
    "3. OPCVM 50,000,000.00 0.5000%",
    "Xtrackers Euro Stoxx 50 UCITS ETF 1C / Xtrackers LU0380865021 5 0,000,000.00 0.5000%",
]


def test_join_split_value():
    assert join_split_value("Hidroelectrica SA RO4Q0Z5RO1B6 3 44,525,072.00 3.1292%") == \
        "Hidroelectrica SA RO4Q0Z5RO1B6 344,525,072.00 3.1292%"
    assert join_split_value("DHL Group DE0005552004 4 ,732,557.70 0.0430%") == "DHL Group DE0005552004 4,732,557.70 0.0430%"
    assert join_split_value("3 44,525,072.00 3.1292%") == "344,525,072.00 3.1292%"
    for unchanged in ("Banca Transilvania SA ROTLVAACNOR1 449,105,094.00 4.0791%",
                      "ROTLVAACNOR1 1,058,918,959.06 2.08",
                      "381,221,274.00 3.90%",
                      "d. Acţiuni 2,373,226,297.13 21.5553%",
                      "Ministerul Finantelor Publice RO1631DBN055 117,233,612.80 1.65%",
                      "a. Titluri de stat cu scadenta mai mica de 1 an - 0.00%"):
        assert join_split_value(unchanged) == unchanged
    # antete rupte (Aripi 2025, BCR 2024): totalul real e cel lipit
    assert join_split_value("d. Actiuni 5 ,178,640,378.75 32.39%") == "d. Actiuni 5,178,640,378.75 32.39%"
    assert join_split_value("b. Participații la fonduri de investiții private de capital 9 3,187,550.03 0.8464%") == \
        "b. Participații la fonduri de investiții private de capital 93,187,550.03 0.8464%"
    assert join_split_value("2. Valori mobiliare tranzactionate, din care: 2 2,498,274,971.79 94.62%") == \
        "2. Valori mobiliare tranzactionate, din care: 22,498,274,971.79 94.62%"


def test_bcr_2024_split_digits_sum_matches_header():
    lines = ["Cod fond de pensii: FP2-123", "la data de: 31.12.2024",
             "2. Valori mobiliare admise la tranzacționare 9,000,000,000.00 90.0000%"] + BCR_2024_EQUITY
    rep = parse_lines(lines)
    eq = rep.equities()
    assert [r["ISIN"] for r in eq][:3] == ["ROTLVAACNOR1", "RO4Q0Z5RO1B6", "NL0012294474"] and len(eq) == 7
    assert round(sum(r["Valoare_RON"] for r in eq), 2) == 1000000000.00 == rep.equity_header_total
    hidro = next(r for r in eq if r["ISIN"] == "RO4Q0Z5RO1B6")
    assert hidro["Valoare_RON"] == 300000000.00 and hidro["Pondere"] == 3.0
    assert next(r for r in eq if r["ISIN"] == "DE0005552004")["Valoare_RON"] == 4732557.70
    assert check(rep) == []
    assert next(r for r in rep.rows if r["ISIN"] == "LU0380865021")["Clasa"] == "opcvm"


def test_aripi_2025_split_equity_header():
    """Aripi, 30.06.2025: antetul „d. Actiuni 5 ,178,…” rupt; după lipire, totalul se citește și suma se verifică."""
    lines = ["Cod fond FP2-49", "Structura portofoliului de investitii la data de 30/06/2025",
             "2. Valori mobiliare tranzactionate, din care: 1 0,000,000,000.00 90.00%",
             "c. Obligatiuni corporative tranzactionate 900,000,000.00 9.00%",
             "Banca Comerciala Romana RO1AQREPLMW7 9 00,000,000.00 9.00%",
             "d. Actiuni 1 ,000,000,000.00 10.00%",
             "Axa SA FR0000120628 129,367,929.67 1.29%",
             "Banca Transilvania SA ROTLVAACNOR1 7 44,043,444.56 7.44%",
             "Basf Se DE000BASF111 1 26,588,625.77 1.27%",
             "Page 2 of 3",
             "e. Obligatiuni BERD, BEI, BM - 0.00%"]
    rep = parse_lines(lines)
    assert rep.fund_code == "FP2-49" and rep.report_date == date(2025, 6, 30)
    assert rep.equity_header_total == 1000000000.00
    assert [r["ISIN"] for r in rep.equities()] == ["FR0000120628", "ROTLVAACNOR1", "DE000BASF111"]
    assert check(rep) == []


def test_aripi_ghost_section_title_and_capital_increase_section():
    """Aripi 2026 (texte sintetice): titlul-fantomă „1. Instrumente…” fără valoare, sus pe pagina 2, nu mută secțiunea;
    „g. Actiuni aferente majorarilor … -” nu șterge totalul secțiunii de acțiuni și nu e acțiune."""
    lines = ["Structura portofoliului de investitii la data de 31/07/2026",
             "2. Valori mobiliare tranzactionate, din care: 1,000.00 100.00%",
             "c. Obligatiuni corporative tranzactionate 400.00 40.00%",
             "RO1AQREPLMW7 400.00 40.00%",
             "Page 1 of 2",
             "1. Instrumente ale pietei monetare, din care:",
             "d. Actiuni 600.00 60.00%",
             "ROTLVAACNOR1 350.00 35.00%",
             "ROSNGNACNOR3 250.00 25.00%",
             "g. Actiuni aferente majorarilor de capital social - 0.00%",
             "h. Drepturi de preferinta - 0.00%",
             "3. OPCVM ETF - 0.00%"]
    rep = parse_lines(lines)
    assert [r["ISIN"] for r in rep.equities()] == ["ROTLVAACNOR1", "ROSNGNACNOR3"]
    assert rep.equity_header_total == 600.0
    assert check(rep) == [parse.MISSING_CODE]                      # codul îl completează jobul din sursă
    from tools.pilon2.parse import classify
    assert classify(2, "valori mobiliare", "actiuni aferente majorarilor de capital social") != EQUITY


def test_parse_pdf_falls_back_to_strict_lines_for_overlapping_rows(tmp_path):
    """Două rânduri suprapuse la 0,7 pt (Aripi din 12.2025): citirea obișnuită le amestecă, cea strictă le desparte."""
    pytest.importorskip("pdfplumber")
    canvas_mod = pytest.importorskip("reportlab.pdfgen.canvas")
    from reportlab.lib.pagesizes import A4

    from tools.pilon2.parse import parse_pdf

    path = tmp_path / "aripi.pdf"
    c = canvas_mod.Canvas(str(path), pagesize=A4)
    c.setFont("Helvetica", 6)
    y = 800
    for text in ("Structura portofoliului de investitii la data de 31/07/2026",
                 "2. Valori mobiliare tranzactionate, din care: 1,000.00 100.00%"):
        c.drawString(40, y, text)
        y -= 12
    c.showPage()
    c.setFont("Helvetica", 6)
    y = 800
    c.drawString(40, y, "1. Instrumente ale pietei monetare, din care:")      # fantoma
    c.drawString(40, y - 0.72, "d. Actiuni")
    c.drawString(300, y - 1.2, "1,000.00 100.00%")
    y -= 12
    for text in ("ROTLVAACNOR1 600.00 60.00%", "ROSNGNACNOR3 400.00 40.00%", "h. Drepturi de preferinta - 0.00%"):
        c.drawString(40, y, text)
        y -= 12
    c.save()
    loose = parse_lines(parse.pdf_lines(path))
    assert not loose.equities()                                    # citirea obișnuită ratează secțiunea
    rep = parse_pdf(path)
    assert [r["ISIN"] for r in rep.equities()] == ["ROTLVAACNOR1", "ROSNGNACNOR3"]
    assert rep.equity_header_total == 1000.0 and check(rep) == [parse.MISSING_CODE]
