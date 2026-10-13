"""Teste pentru data/company_files.py. Valori sintetice, calculate de mână."""
import numpy as np
import pandas as pd
import pytest

from data.company_files import (add_ttm, is_bank_template, load_company, parse_company, parse_period,
                                period_label, ticker_from_title, validate)

LABELS = ["Datorii totale", "Cifră de afaceri", "Profit Net", "Profit distribuit", "Ebitda", "Rezultat operațional",
          "Capital propriu (Activ Net)", "Conturi la bănci", "Datorii termen scurt", "Datorii termen lung",
          "Active totale", "Active imobilizate", "Active circulante (curente)", "Stocuri", "Creanțe",
          "Preț curent acțiune", "Nr. de acțiuni emise"]


def _sheet(periods, data):
    """Foaie sintetică: antet cu perioadele și un rând per etichetă; `data[label]` = listă de valori."""
    rows = [["Criteriu"] + periods]
    rows.append(["Marjă netă"] + ["5%"] * len(periods))            # rând calculat: ignorat
    for lab in LABELS:
        rows.append([lab] + data.get(lab, [None] * len(periods)))
    rows.append(["Comparație date Q"] + [None] * len(periods))
    rows.append(["Profit Net"] + [999] * len(periods))                # a doua apariție: ignorată
    return rows


def _company():
    q_periods = ["Q1 2024", "Q2 2024", "Q3 2024", "Q1 2025", "Q2 2025", "Q3 2025", "Q1 2026"]
    q = {
        "Cifră de afaceri": [100, 210, 320, 120, 250, 380, 130],
        "Profit Net": [10, 22, 30, 12, 26, 40, 15],
        "Ebitda": [20, 40, 60, 24, 50, 76, 26],
        "Rezultat operațional": [15, 30, 45, 18, 38, 57, 20],
        "Capital propriu (Activ Net)": [500, 510, 520, 560, 570, 580, 600],
        "Datorii totale": [300, 300, 300, 290, 290, 290, 280],
        "Active totale": [800, 810, 820, 850, 860, 870, 880],
        "Active imobilizate": [500] * 7, "Active circulante (curente)": [300, 310, 320, 350, 360, 370, 380],
        "Datorii termen scurt": [100] * 7, "Datorii termen lung": [200, 200, 200, 190, 190, 190, 180],
        "Stocuri": [50] * 7, "Creanțe": [80] * 7, "Conturi la bănci": [60] * 7,
        "Preț curent acțiune": [10] * 7, "Nr. de acțiuni emise": [1000] * 7,
        "Profit distribuit": [0] * 7,
    }
    a_periods = ["2024", "2025", "2026"]
    a = {
        "Cifră de afaceri": [440, 520, None], "Profit Net": [42, 50, None], "Ebitda": [80, 100, None],
        "Rezultat operațional": [60, 75, None], "Capital propriu (Activ Net)": [550, 590, None],
        "Datorii totale": [290, 285, None], "Active totale": [840, 875, 0], "Active imobilizate": [500, 500, None],
        "Active circulante (curente)": [340, 375, None], "Datorii termen scurt": [100, 100, None],
        "Datorii termen lung": [190, 185, None], "Stocuri": [50, 50, None], "Creanțe": [80, 80, None],
        "Conturi la bănci": [60, 60, None], "Preț curent acțiune": [10, 10, 10], "Nr. de acțiuni emise": [1000, 1000, 1000],
        "Profit distribuit": [20, 25, None],
    }
    return _sheet(q_periods, q), _sheet(a_periods, a)


def test_perioade_si_titluri():
    assert parse_period("Q1 2021") == (2021, 3) and parse_period("Q3 2021") == (2021, 9)
    assert parse_period("2019") == (2019, 12) and parse_period("2025/2024") is None and parse_period("Media") is None
    assert period_label(2025, 6) == "S1 2025" and period_label(2025, 12) == "2025"
    assert ticker_from_title("DN Agrar") == "DN" and ticker_from_title("SNP") == "SNP"
    assert ticker_from_title("0-Dashboard") is None and ticker_from_title("Z-Gol") is None


def test_parsare_perioade_si_coloane_viitoare():
    q, a = _company()
    df, missing = parse_company(q, a)
    # 7 cumulate + 2 anuale (2026 are active 0 → exclus)
    assert len(df) == 9 and missing == {"Date Q": ["expense_total", "income_total"],
                                         "Date Anuale": ["expense_total", "income_total"]}
    assert list(zip(df.year, df.months))[-3:] == [(2025, 9), (2025, 12), (2026, 3)]
    # prima apariție a „Profit Net" (nu 999 din blocul de comparații)
    assert df.loc[(df.year == 2025) & (df.months == 12), "ni"].item() == 50
    assert df.loc[(df.year == 2026) & (df.months == 3), "published"].item() == pd.Timestamp(2026, 5, 15)
    assert df.loc[(df.year == 2025) & (df.months == 12), "published"].item() == pd.Timestamp(2026, 4, 30)


def test_ttm_din_cumulate():
    q, a = _company()
    df, issues, _ = load_company(q, a)
    get = lambda y, m, c: df.loc[(df.year == y) & (df.months == m), c].item()
    # T1 2026: 50 − 12 + 15 = 53;  9L 2025: 42 − 30 + 40 = 52;  anual 2025: 50
    assert get(2026, 3, "ni_ttm") == 53 and get(2025, 9, "ni_ttm") == 52 and get(2025, 12, "ni_ttm") == 50
    # cifra de afaceri S1 2025: 440 − 210 + 250 = 480
    assert get(2025, 6, "rev_ttm") == 480
    # 2024: lipsește anul 2023 → NaN
    assert np.isnan(get(2024, 6, "ni_ttm"))
    assert issues == []


def _frame(**cols):
    n = len(next(iter(cols.values())))
    base = {k: [np.nan] * n for k in ("ebitda", "ebit", "assets", "equity", "liab", "assets_fixed", "assets_cur",
                                       "rev", "inventory", "receivables", "liab_st", "liab_lt", "cash")}
    base.update(cols)
    base.setdefault("year", [2024] * n)
    base["months"] = cols.get("months", [3, 6, 9, 12][:n] + [12] * max(0, n - 4))
    return pd.DataFrame(base)


def test_validare_ebitda_si_cifra_de_afaceri():
    df = _frame(ebitda=[10, 5, 0, 30], ebit=[8, 9, 7, 25], rev=[100, 210, 200, 400], year=[2024] * 4,
                months=[3, 6, 9, 12])
    clean, issues = validate(df)
    assert np.isnan(clean.loc[1, "ebitda"]) and np.isnan(clean.loc[2, "ebitda"])     # 5 < 9; 0 = necompletat
    assert clean.loc[0, "ebitda"] == 10 and clean.loc[3, "ebitda"] == 30
    # 210 → 200 scade: ambele excluse; 400 rămâne
    assert np.isnan(clean.loc[1, "rev"]) and np.isnan(clean.loc[2, "rev"]) and clean.loc[3, "rev"] == 400
    fields = sorted(i["field"] for i in issues)
    assert fields == ["ebitda", "rev"] and all(i["excluded"] for i in issues)


def test_validare_valori_izolate_si_bilant():
    df = _frame(equity=[500, 5, 520, 530, 540, 5400], assets=[800, 810, 820, 830, 840, 850],
                liab=[300, 300, 300, 300, 300, 300], year=[2024, 2024, 2024, 2024, 2025, 2025],
                months=[3, 6, 9, 12, 3, 6], cash=[60, 61, 600, 62, 63, 64])
    clean, issues = validate(df)
    # 5 izolat (×0,01 față de ambii vecini); 5.400 ultimul, ×10 față de 540 → sub pragul de 20 → rămâne
    assert np.isnan(clean.loc[1, "equity"]) and clean.loc[5, "equity"] == 5400
    # numerarul izolat (600) e doar semnalat, nu exclus
    assert clean.loc[2, "cash"] == 600
    cash_issue = [i for i in issues if i["field"] == "cash"][0]
    assert cash_issue["excluded"] is False
    # bilanțul nu se închide la rândurile 1 (5 + 300 vs 810) și 5 (5400 + 300 vs 850): doar semnalat
    bil = [(i["year"], i["months"]) for i in issues if i["field"] == "bilant"]
    assert (2024, 6) in bil and (2025, 6) in bil


def test_sablon_banca():
    assert is_bank_template([["Criteriu"], ["Datorii la clienți (depozite)", 1]])
    assert is_bank_template([["Criteriu"], ["Depozite clienți", 1]])
    assert not is_bank_template(_company()[0])


def test_ttm_cu_termen_lipsa():
    df = pd.DataFrame({"year": [2024, 2025, 2025], "months": [12, 3, 12], "ni": [40, 10, 50],
                       **{f: [1.0, 1.0, 1.0] for f in ("rev", "ebitda", "ebit", "income_total", "expense_total")}})
    out = add_ttm(df)
    assert np.isnan(out.loc[1, "ni_ttm"])          # lipsește T1 2024
    assert out.loc[2, "ni_ttm"] == 50
