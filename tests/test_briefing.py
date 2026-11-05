"""Rezumatul Zilei: ultima ședință, lățime, topuri cu prag de valoare, σ, starea bursei (valori sintetice)."""
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from analytics.briefing import (breadth_counts, daily_move, liquidity_top, market_state, parse_universe,
                                relative_day, session_is_partial, session_table, sigma_label, sigma_of_last,
                                top_moves, turnover_vs_median)

DAYS = pd.bdate_range("2026-10-01", periods=5)          # 01–07.10.2026


def test_univers_din_fisier():
    assert parse_universe("# comentariu\nTLV.RO\n\nsnp.ro  # notă\nTLV.RO\n") == ["TLV.RO", "SNP.RO"]
    assert parse_universe(None) == []


def test_sigma_fara_ziua_curenta():
    # 20 de variații alternând ±1% (abatere standard eșantion = 1,0260%), apoi +3%
    prior = [0.01, -0.01] * 10
    r = pd.Series(prior + [0.03])
    sd = float(pd.Series(prior).std(ddof=1))
    assert sigma_of_last(r) == pytest.approx(0.03 / sd)
    assert sigma_of_last(pd.Series(prior)) is None                        # doar 19 variații dinainte
    assert sigma_of_last(pd.Series([0.01] * 25)) is None                  # abatere 0


def test_daily_move():
    m = daily_move(pd.Series([100.0, 102.0], index=DAYS[:2]))
    assert m["pct"] == pytest.approx(0.02) and m["last"] == 102.0 and m["prev"] == 100.0 and m["sigma"] is None
    assert daily_move(None)["pct"] is None and daily_move(pd.Series([5.0], index=DAYS[:1]))["pct"] is None


def _universe():
    close = pd.DataFrame({
        "AAA": [10.0, 10.0, 10.0, 10.0, 11.0],          # +10%, tranzacționat
        "BBB": [0.05, 0.05, 0.05, 0.05, 0.0495],        # −1% la 0,05 lei (vechiul prag absolut 0,001 îl dădea „neutru”)
        "CCC": [20.0, 20.0, 21.0, 22.0, np.nan],        # fără preț azi: ultima variație e de ieri
        "DDD": [5.0, 5.0, 5.0, 5.0, 5.0],               # rând azi, dar volum 0: netranzacționat
        "EEE": [8.0, 8.0, 8.0, 8.0, 8.0],               # tranzacționat, nemodificat
    }, index=DAYS)
    volume = pd.DataFrame({
        "AAA": [1000] * 5, "BBB": [10_000_000] * 5, "CCC": [500, 500, 500, 500, np.nan],
        "DDD": [100, 100, 100, 100, 0], "EEE": [20_000] * 5,
    }, index=DAYS, dtype="float64")
    return close, volume


def test_tabelul_sedintei():
    t, session = session_table(*_universe())
    assert session == DAYS[-1]
    assert list(t.index[t["traded"]]) == ["AAA", "BBB", "EEE"]
    assert t.loc["AAA", "pct"] == pytest.approx(0.10) and t.loc["AAA", "value"] == pytest.approx(11_000.0)
    assert t.loc["BBB", "value"] == pytest.approx(495_000.0)
    assert t.loc["CCC", "date"] == DAYS[-2] and not t.loc["CCC", "traded"]
    assert not t.loc["DDD", "traded"]


def test_latime_doar_tranzactionate_si_in_procente():
    t, _ = session_table(*_universe())
    b = breadth_counts(t)
    assert (b["up"], b["down"], b["flat"], b["n"], b["not_traded"]) == (1, 1, 1, 3, 2)
    assert b["up_pct"] == pytest.approx(100 / 3)
    assert breadth_counts(None)["up_pct"] is None


def test_topuri_cu_prag_de_valoare_si_semn():
    t, _ = session_table(*_universe())
    up, down = top_moves(t, n=5, min_value=100_000)
    assert list(up.index) == [] and list(down.index) == ["BBB"]           # AAA are doar 11.000 de lei
    up, down = top_moves(t, n=5)
    assert list(up.index) == ["AAA"] and list(down.index) == ["BBB"]      # EEE (0%) nu e nici creștere, nici scădere
    assert list(liquidity_top(t).index) == ["BBB", "EEE", "AAA"]           # după valoare: 495.000, 160.000, 11.000


def test_valoarea_totala_fata_de_mediana():
    idx = pd.bdate_range("2026-09-01", periods=12)
    close = pd.DataFrame({"A": [10.0] * 12, "B": [2.0] * 12}, index=idx)
    vol = pd.DataFrame({"A": [100.0] * 11 + [300.0], "B": [500.0] * 11 + [500.0]}, index=idx)
    r = turnover_vs_median(close, vol)
    # zilele anterioare: 10×100 + 2×500 = 2.000; azi: 10×300 + 2×500 = 4.000
    assert (r["today"], r["median"], r["n"]) == (4000.0, 2000.0, 11) and r["ratio"] == pytest.approx(2.0)
    short = turnover_vs_median(close.iloc[:5], vol.iloc[:5])
    assert short["ratio"] is None and short["n"] == 4


def test_starea_bursei_cu_decalaj_de_ora():
    ro = ZoneInfo("Europe/Bucharest")
    # 18.03.2026: SUA deja pe ora de vară, România nu → NYSE deschide la 15:30 ora României
    s = market_state("NYSE", datetime(2026, 3, 18, 15, 45, tzinfo=ro))
    assert s["state"] == "deschisă" and s["opens"] == "15:30" and s["closes"] == "22:00"
    s = market_state("NYSE", datetime(2026, 11, 5, 16, 15, tzinfo=ro))
    assert s["state"] == "închisă" and s["opens"] == "16:30"
    assert market_state("BVB", datetime(2026, 11, 7, 12, 0, tzinfo=ro))["state"] == "weekend"
    s = market_state("BVB", datetime(2026, 11, 5, 17, 50, tzinfo=ro))
    assert s["state"] == "închisă" and s["closes"] == "17:45"


def test_sedinta_partiala():
    ro = ZoneInfo("Europe/Bucharest")
    open_state = market_state("BVB", datetime(2026, 11, 5, 12, 0, tzinfo=ro))
    assert session_is_partial(pd.Timestamp("2026-11-05"), open_state)
    assert not session_is_partial(pd.Timestamp("2026-11-04"), open_state)
    assert not session_is_partial(pd.Timestamp("2026-11-05"), market_state("BVB", datetime(2026, 11, 5, 19, 0, tzinfo=ro)))


def test_diferenta_pe_o_zi():
    a = pd.Series([100.0, 101.0], index=DAYS[:2])
    b = pd.Series([200.0, 204.0], index=DAYS[:2])
    r = relative_day(a, b)
    assert r["diff_pp"] == pytest.approx(-1.0) and r["date"] == DAYS[1]
    assert relative_day(a.iloc[:1], b)["diff_pp"] is None


def test_eticheta_sigma():
    assert sigma_label(2.4) == "mișcare neobișnuită (peste 2σ)"
    assert sigma_label(-1.2) == "mișcare mai mare decât de obicei (1–2σ)"
    assert sigma_label(0.3) == "mișcare obișnuită (sub 1σ)" and sigma_label(None) == "N/A"


def test_suma_scurta():
    from analytics.briefing import money_short
    assert money_short(45_200_000, "RON") == "45,2 mil. lei"
    assert money_short(1_234_000_000, "USD") == "1,23 mld. USD"
    assert money_short(850_000, "RON") == "850.000 lei"
    assert money_short(None, "RON") == "N/A" and money_short(float("nan"), "USD") == "N/A"
