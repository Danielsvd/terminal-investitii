"""analytics/sec_13f.py: parsare, validare, portofoliu, mișcări, consens. Valori sintetice, calculate de mână."""
import numpy as np
import pandas as pd
import pytest

from analytics import sec_13f as s

NS_IT = "http://www.sec.gov/edgar/document/thirteenf/informationtable"
NS_C = "http://www.sec.gov/edgar/thirteenffiler"


def _row(name, cls, cusip, value, sh, typ="SH", put_call=None, disc="SOLE", other=None):
    pc = f"<putCall>{put_call}</putCall>" if put_call else ""
    om = f"<otherManager>{other}</otherManager>" if other else ""
    return (f"<infoTable><nameOfIssuer>{name}</nameOfIssuer><titleOfClass>{cls}</titleOfClass>"
            f"<cusip>{cusip}</cusip><value>{value}</value><shrsOrPrnAmt><sshPrnamt>{sh}</sshPrnamt>"
            f"<sshPrnamtType>{typ}</sshPrnamtType></shrsOrPrnAmt>{pc}<investmentDiscretion>{disc}"
            f"</investmentDiscretion>{om}<votingAuthority><Sole>0</Sole><Shared>0</Shared><None>0</None>"
            f"</votingAuthority></infoTable>")


INFO_XML = (f'<?xml version="1.0"?><informationTable xmlns="{NS_IT}">'
            + _row("AAA CORP", "COM", "000001AA1", 600, 60)
            + _row("AAA CORP", "COM", "000001AA1", 400, 40, disc="DFND", other="1")
            + _row("BBB INC", "CL A", "000002BB2", 300, 30)
            + _row("BBB INC", "CL C", "000002BB9", 200, 20)
            + _row("CCC CO", "COM", "000003CC3", 50, 5, put_call="Call")
            + _row("DDD CO", "NOTE 2%", "000004DD4", 100, 100, typ="PRN")
            + "</informationTable>").encode()


def _cover(entry=6, value=1650, amend="false", amend_type=""):
    at = f"<amendmentInfo><amendmentType>{amend_type}</amendmentType></amendmentInfo>" if amend_type else ""
    return (f'<?xml version="1.0"?><edgarSubmission xmlns="{NS_C}"><headerData/><formData><coverPage>'
            f"<reportCalendarOrQuarter>09-30-2026</reportCalendarOrQuarter><isAmendment>{amend}</isAmendment>{at}"
            f"<filingManager><name>FOND TEST LLC</name></filingManager></coverPage><summaryPage>"
            f"<otherIncludedManagersCount>1</otherIncludedManagersCount><tableEntryTotal>{entry}</tableEntryTotal>"
            f"<tableValueTotal>{value}</tableValueTotal></summaryPage></formData></edgarSubmission>").encode()


def test_parse_cover():
    c = s.parse_cover(_cover())
    assert c["period"] == pd.Timestamp("2026-09-30").date()
    assert c["entry_total"] == 6 and c["value_total"] == 1650
    assert c["filer_name"] == "FOND TEST LLC" and not c["is_amendment"] and c["amendment_type"] is None
    a = s.parse_cover(_cover(amend="true", amend_type="restatement"))
    assert a["is_amendment"] and a["amendment_type"] == "RESTATEMENT"


def test_parse_cover_rejects_other_root():
    with pytest.raises(s.Parse13FError):
        s.parse_cover(INFO_XML)


def test_parse_info_table_and_detection():
    info = s.parse_info_table(INFO_XML)
    assert len(info) == 6
    assert info["Actiuni"].tolist() == [60, 40, 30, 20, 5, 100]
    assert info.loc[4, "PutCall"] == "Call" and info.loc[5, "Tip"] == "PRN"
    assert info.loc[1, "Alt_manager"] == "1"
    assert s.is_info_table(INFO_XML) and not s.is_info_table(_cover()) and not s.is_info_table(b"<nu")


def test_validate_ok_and_mismatches():
    info = s.parse_info_table(INFO_XML)
    assert s.validate(s.parse_cover(_cover()), info) == []
    probs = s.validate(s.parse_cover(_cover(entry=7, value=2000)), info)
    assert len(probs) == 2 and "tableEntryTotal" in probs[0] and "tableValueTotal" in probs[1]


def test_value_multiplier_switch_2023():
    assert s.value_multiplier("2022-11-14") == 1000.0
    assert s.value_multiplier("2023-01-02") == 1000.0
    assert s.value_multiplier("2023-01-03") == 1.0
    assert s.value_multiplier("2026-11-14") == 1.0


def test_implied_price():
    info = s.parse_info_table(INFO_XML)
    p = s.implied_price(info, 1.0)
    assert p[0] == pytest.approx(10.0)          # 600 / 60
    assert np.isnan(p[5])                       # PRN: fără preț pe acțiune


def test_equity_book_aggregates_and_excludes_options_prn():
    book = s.equity_book(s.parse_info_table(INFO_XML), 1.0)
    # AAA: 60+40 acț., 600+400 $; BBB A 300, BBB C 200; total 1500; opțiunea și PRN în afară
    assert book["CUSIP"].tolist() == ["000001AA1", "000002BB2", "000002BB9"]
    assert book["Actiuni"].tolist() == [100, 30, 20]
    assert book["Pondere"].tolist() == pytest.approx([1000 / 1500, 300 / 1500, 200 / 1500])
    assert book["Rang"].tolist() == [1, 2, 3]
    assert book["CUSIP6"].tolist() == ["000001", "000002", "000002"]


def test_equity_book_thousands_before_2023():
    book = s.equity_book(s.parse_info_table(INFO_XML), s.value_multiplier("2022-11-14"))
    assert book.loc[0, "Valoare_USD"] == 1_000_000


def test_options_book():
    o = s.options_book(s.parse_info_table(INFO_XML), 1.0)
    assert len(o) == 1 and o.loc[0, "PutCall"] == "Call" and o.loc[0, "Valoare_USD"] == 50


def test_apply_amendment():
    base = s.parse_info_table(INFO_XML)
    am = base.iloc[:1].copy()
    assert len(s.apply_amendment(base, am, "RESTATEMENT")) == 1
    assert len(s.apply_amendment(base, am, "new holdings")) == 7
    with pytest.raises(ValueError):
        s.apply_amendment(base, am, None)


def _book(rows):
    df = pd.DataFrame(rows, columns=["CUSIP", "Emitent", "Actiuni", "Valoare_USD"])
    df["Clasa_titlu"] = "COM"
    df["CUSIP6"] = df["CUSIP"].str[:6]
    df["Pondere"] = df["Valoare_USD"] / df["Valoare_USD"].sum()
    df = df.sort_values("Valoare_USD", ascending=False).reset_index(drop=True)
    df["Rang"] = np.arange(1, len(df) + 1)
    return df[s.BOOK_COLUMNS]


PREV = _book([("000001AA1", "AAA", 50, 500), ("000002BB2", "BBB A", 40, 400), ("000005EE5", "EEE", 10, 100)])
CURR = _book([("000001AA1", "AAA", 100, 1000), ("000002BB2", "BBB A", 30, 300), ("000002BB9", "BBB C", 20, 200)])
PRICES = {"000001AA1": 10.0, "000002BB2": 10.0, "000002BB9": 10.0, "000005EE5": 10.0}


def test_quarter_moves_with_split_and_prices():
    m = s.quarter_moves(PREV, CURR, split_factor={"000001AA1": 2.0}, avg_price=PRICES).set_index("CUSIP")
    assert m.loc["000001AA1", "Actiune"] == s.ACT_HELD and m.loc["000001AA1", "Delta_actiuni"] == 0
    assert m.loc["000002BB2", "Actiune"] == s.ACT_DOWN
    assert m.loc["000002BB2", "Delta_pct"] == pytest.approx(-0.25)
    assert m.loc["000002BB2", "Valoare_tranzactionata"] == pytest.approx(-100)
    assert m.loc["000002BB2", "Impact"] == pytest.approx(100 / 1500)
    assert m.loc["000002BB2", "Delta_pp"] == pytest.approx(-20)
    assert m.loc["000002BB9", "Actiune"] == s.ACT_NEW and m.loc["000002BB9", "Impact"] == pytest.approx(200 / 1500)
    assert m.loc["000005EE5", "Actiune"] == s.ACT_CLOSED and m.loc["000005EE5", "Valoare_tranzactionata"] == -100


def test_quarter_moves_order_by_impact_then_pp():
    m = s.quarter_moves(PREV, CURR, split_factor={"000001AA1": 2.0}, avg_price=PRICES)
    # BBB C (0,133), apoi BBB A și EEE la egalitate 0,067 → |Δpp| 20 > 10, apoi AAA (0)
    assert m["CUSIP"].tolist() == ["000002BB9", "000002BB2", "000005EE5", "000001AA1"]


def test_quarter_moves_without_split_and_price():
    m = s.quarter_moves(PREV, CURR).set_index("CUSIP")
    assert m.loc["000001AA1", "Actiune"] == s.ACT_UP and m.loc["000001AA1", "Delta_pct"] == pytest.approx(1.0)
    assert np.isnan(m.loc["000001AA1", "Valoare_tranzactionata"])      # fără preț: NaN, nu 0


def test_move_counts_close():
    m = s.quarter_moves(PREV, CURR, split_factor={"000001AA1": 2.0})
    c = s.move_counts(m)
    assert c == {s.ACT_NEW: 1, s.ACT_CLOSED: 1, s.ACT_UP: 0, s.ACT_DOWN: 1, s.ACT_HELD: 1}
    assert c[s.ACT_NEW] + c[s.ACT_UP] + c[s.ACT_DOWN] + c[s.ACT_HELD] == len(CURR)


def test_classify_manager():
    assert s.classify_manager(CURR) == "Concentrat"
    flat = _book([(f"{i:06d}XX1", f"E{i}", 1, 1) for i in range(150)])          # primele 10 = 6,7%
    assert s.classify_manager(flat) == "Larg"
    top = _book([(f"{i:06d}XX1", f"E{i}", 1, 100 if i < 10 else 5) for i in range(150)])
    assert s.classify_manager(top) == "Concentrat"                              # 1000 / 1700 = 58,8%
    assert s.classify_manager(pd.DataFrame(columns=s.BOOK_COLUMNS)) is None


def _holdings():
    rows = [
        # CIK, Fond, Clasa, CUSIP, Emitent, Pondere, Rang
        # F1: 20 de poziții → prag max(1/20, 1%) = 5%
        ("1", "F1", "Concentrat", "000001AA1", "AAA", 0.50, 1),
        ("1", "F1", "Concentrat", "000002BB2", "BBB", 0.03, 12),
        ("1", "F1", "Concentrat", "000002BB9", "BBB", 0.03, 13),     # clasele BBB adunate: 6% ≥ 5% → convingere
        # F2: 8 poziții → prag 12,5%; AAA 0,97% pe locul 8 (cazul Himalaya/AAPL): deține, fără convingere
        ("2", "F2", "Concentrat", "000001AA1", "AAA", 0.0097, 8),
        ("3", "F3", "Larg", "000001AA1", "AAA", 0.001, 2000),         # exclus din consens
    ]
    rows += [("1", "F1", "Concentrat", f"9{i:05d}XX1", f"F1_{i}", 0.44 / 17, 14 + i) for i in range(17)]
    rows += [("2", "F2", "Concentrat", f"8{i:05d}XX1", f"F2_{i}", 0.9903 / 7, 1 + i) for i in range(7)]
    df = pd.DataFrame(rows, columns=["CIK", "Fond", "Clasa", "CUSIP", "Emitent", "Pondere", "Rang"])
    df["CUSIP6"] = df["CUSIP"].str[:6]
    df["Actiuni"] = 1.0
    df["Valoare_USD"] = df["Pondere"] * 1000
    return df


def test_consensus_share_classes_and_exclusions():
    moves = pd.DataFrame([
        ("2", "000001", s.ACT_NEW, np.nan, 0.01),       # cumpărare de convingere (pondere ≥ 1%)
        ("3", "000001", s.ACT_NEW, np.nan, 0.05),       # Larg → nu contează
        ("1", "000001", s.ACT_UP, 0.20, 0.50),          # +20% < 50% → nu
        ("1", "000002", s.ACT_UP, 0.60, 0.02),          # +60%, 2% → da
    ], columns=["CIK", "CUSIP6", "Actiune", "Delta_pct", "Pondere"])
    c = s.consensus(_holdings(), moves).set_index("CUSIP6")
    assert c.loc["000001", "Detinatori"] == 2 and c.loc["000001", "Convingere"] == 1
    assert c.loc["000002", "Detinatori"] == 1 and c.loc["000002", "Convingere"] == 1
    assert c.loc["000001", "Cumparari_convingere"] == 1 and c.loc["000002", "Cumparari_convingere"] == 1
    assert c.loc["000001", "Fonduri"] == "F1, F2"


def test_who_holds_order_and_conviction():
    w = s.who_holds(_holdings(), "000001")
    assert w["Fond"].tolist() == ["F1", "F2", "F3"]
    assert w["Convingere"].tolist() == [True, False, False]      # F2: 0,97% < 12,5% (deși e în top 10); F3: 0,1%
    assert s.who_holds(_holdings(), "000002")["Convingere"].tolist() == [True]   # BBB: 3% + 3% ≥ 5%


def test_conviction_threshold():
    assert s.conviction_threshold(8) == pytest.approx(0.125)        # Himalaya: 8 poziții
    assert s.conviction_threshold(16) == pytest.approx(0.0625)      # Pershing
    assert s.conviction_threshold(30) == pytest.approx(1 / 30)      # Berkshire: 3,33%
    assert s.conviction_threshold(100) == pytest.approx(0.01)
    assert s.conviction_threshold(150) == pytest.approx(0.01)       # minimul de 1%
    assert np.isnan(s.conviction_threshold(0)) and np.isnan(s.conviction_threshold(np.nan))
    thr = s.conviction_threshold(pd.Series([8, 150], index=["a", "b"]))
    assert thr.index.tolist() == ["a", "b"] and thr.tolist() == pytest.approx([0.125, 0.01])


def test_conviction_mask_single_fund_cases():
    # 8 poziții (prag 12,5%): 24,2% da; 12,5% exact da; 0,97% pe locul 8 nu (regula veche: da, prin top 10)
    w8 = [0.242, 0.20, 0.15, 0.125, 0.12, 0.10, 0.0533, 0.0097]
    b8 = _book([(f"{i:06d}XX1", f"E{i}", 1, w * 1000) for i, w in enumerate(w8)])
    assert s.conviction_mask(b8).tolist() == [True, True, True, True, False, False, False, False]
    # 150 de poziții (prag 1%): 3,5% pe locul 12 da (regula „top 10 ȘI ≥ 1%” l-ar fi pierdut); 0,5% nu
    w150 = [0.05] * 11 + [0.035] + [0.01] + [(1 - 0.55 - 0.035 - 0.01) / 137] * 137
    b150 = _book([(f"{i:06d}XX1", f"E{i}", 1, w * 1e6) for i, w in enumerate(w150)])
    m = s.conviction_mask(b150)
    assert bool(m.iloc[11]) and bool(m.iloc[12]) and not m.iloc[13:].any()
    # pondere lipsă → nu e convingere
    b = b8.copy()
    b.loc[0, "Pondere"] = np.nan
    assert not bool(s.conviction_mask(b).iloc[0])


def test_conviction_mask_multi_fund_counts_per_fund():
    h = _holdings()
    m = s.conviction_mask(h)
    assert bool(m[(h["CIK"] == "1") & (h["CUSIP"] == "000001AA1")].iloc[0])        # 50% ≥ 5%
    assert m[(h["CIK"] == "1") & (h["Emitent"] == "BBB")].all()                     # clasele adunate: 3% + 3% ≥ 5%
    assert not bool(m[(h["CIK"] == "2") & (h["CUSIP"] == "000001AA1")].iloc[0])    # 0,97% < 12,5%


def test_issuer_key_funds_use_full_cusip():
    assert s.issuer_key("02079K305", "CAP STK CL A") == "02079K"
    assert s.issuer_key("464286400", "MSCI BRAZIL ETF") == "464286400"       # iShares Inc: fonduri diferite
    assert s.issuer_key("78462F103", "TR UNIT") == "78462F103"
    assert s.issuer_key("464287200", "CORE S&P500 ETF") == "464287200"


def test_equity_book_etf_rows_not_merged():
    xml = (b'<informationTable xmlns="x">'
           + b"".join(f"<infoTable><nameOfIssuer>ISHARES INC</nameOfIssuer><titleOfClass>{t}</titleOfClass>"
                      f"<cusip>{c}</cusip><value>{v}</value><shrsOrPrnAmt><sshPrnamt>10</sshPrnamt><sshPrnamtType>SH"
                      f"</sshPrnamtType></shrsOrPrnAmt></infoTable>".encode()
                      for t, c, v in (("MSCI BRAZIL ETF", "464286400", 300), ("MSCI MEXICO ETF", "464286822", 200)))
           + b"</informationTable>")
    book = s.equity_book(s.parse_info_table(xml), 1.0)
    assert sorted(book["CUSIP6"]) == ["464286400", "464286822"]


def test_detect_thousands():
    info = s.parse_info_table(INFO_XML)                    # prețuri implicite 10 $ la multiplicator 1
    assert not s.detect_thousands(info, 1.0)
    small = info.copy()
    small["Valoare_raportata"] = small["Valoare_raportata"] / 1000    # 0,01 $ / acțiune → raportat în mii
    assert s.detect_thousands(small, 1.0)
    assert not s.detect_thousands(small.iloc[:2], 1.0)                # prea puține rânduri: nu decide


def test_conviction_mask_share_classes_same_flag():
    # Berkshire-like: 30 de poziții (prag 3,33%); GOOGL 8,91% + GOOG 3,05% = 11,96% → ambele clase bifate
    w = [0.2495, 0.0891, 0.0305] + [(1 - 0.2495 - 0.0891 - 0.0305) / 27] * 27
    b = _book([(f"{i:06d}XX1", f"E{i}", 1, x * 1e6) for i, x in enumerate(w)])
    b.loc[b["Emitent"] == "E2", "CUSIP6"] = b.loc[b["Emitent"] == "E1", "CUSIP6"].iloc[0]   # E1/E2 = același emitent
    m = s.conviction_mask(b).set_axis(b["Emitent"])
    assert m["E1"] and m["E2"]                                   # E2 singur (3,05%) ar fi sub 3,33%
    assert not m.drop(["E0", "E1", "E2"]).any()                  # restul: 2,34% < 3,33%
