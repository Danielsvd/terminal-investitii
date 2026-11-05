"""Calendarul macro FRED: potrivirea publicărilor, intervalul de 14 zile, eroarea fără cheie."""
from datetime import date

from data.fred_calendar import (CALENDAR_DAYS, FOMC_DECISIONS, FOMC_LABEL, TRACKED, all_release_names, error_text,
                                fomc_in_window, fomc_outdated, upcoming, window)

TODAY = date(2026, 11, 5)   # joi


def test_interval_14_zile_inclusiv():
    assert window(TODAY) == (date(2026, 11, 5), date(2026, 11, 18))


def test_potrivire_sortare_si_lipsa():
    data = [
        {"release_id": 50, "release_name": "Employment Situation", "date": "2026-11-06"},
        {"release_id": 10, "release_name": "consumer price index ", "date": "2026-11-12"},   # majuscule/spații
        {"release_id": 10, "release_name": "Consumer Price Index", "date": "2026-11-12"},    # duplicat aceeași zi
        {"release_id": 18, "release_name": "H.15 Selected Interest Rates", "date": "2026-11-06"},  # neurmărită
        {"release_id": 53, "release_name": "Gross Domestic Product", "date": "2026-11-19"},  # după interval
        {"release_id": 53, "release_name": "Gross Domestic Product", "date": "2026-11-04"},  # ieri
        {"release_id": 9, "release_name": "Advance Monthly Sales for Retail and Food Services", "date": "nu e dată"},
        "nu e dict",
    ]
    rows, missing = upcoming(data, TODAY)
    assert [(r["date"], r["key"]) for r in rows] == [(date(2026, 11, 6), "nfp"), (date(2026, 11, 12), "cpi")]
    assert rows[0]["weekday"] == "vineri" and rows[0]["in_days"] == 1
    assert rows[1]["weekday"] == "joi" and rows[1]["in_days"] == 7
    assert rows[0]["updates"] == "Nonfarm Payrolls, Rata Șomajului"
    assert len(missing) == len(TRACKED) - 2 + 1 and "PIB" in missing and "Inflația CPI" not in missing


def test_azi_si_ultima_zi_sunt_incluse():
    data = [{"release_name": "Surveys of Consumers", "date": "2026-11-05"},
            {"release_name": "Producer Price Index", "date": "2026-11-18"}]
    rows, _ = upcoming(data, TODAY)
    assert [r["in_days"] for r in rows] == [0, CALENDAR_DAYS - 1]


def test_raspuns_gol():
    rows, missing = upcoming(None, TODAY)
    assert rows == [] and len(missing) == len(TRACKED) + 1 and "Decizia de dobândă FOMC" in missing
    assert all_release_names(None) == []


def test_lista_completa_de_nume():
    data = [{"release_name": "b"}, {"release_name": "A "}, {"release_name": "b"}, {}]
    assert all_release_names(data) == ["A", "b"]


def test_eroarea_nu_contine_cheia():
    key = "a" * 32
    body = {"error_code": 400, "error_message": f"Bad Request. api_key {key} is not registered."}
    txt = error_text(400, body, key)
    assert key not in txt and "***" in txt and txt.startswith("FRED a răspuns HTTP 400")
    assert error_text(500, None, key) == "FRED a răspuns HTTP 500"


def test_fomc_din_fred_ignorat_chiar_daca_e_zilnic():
    # bug-ul din captura 05.11: FRED listează „FOMC Press Release” în fiecare zi (seriile DFEDTARU/DFEDTARL)
    data = [{"release_id": 101, "release_name": "FOMC Press Release", "date": f"2026-11-{d:02d}"} for d in range(5, 19)]
    rows, missing = upcoming(data, TODAY)
    assert rows == [] and "Decizia de dobândă FOMC" in missing   # nicio ședință 05–18.11.2026


def test_fomc_din_calendarul_fed():
    # 01.12.2026 (marți) → interval 01–14.12: decizia din 09.12 (miercuri), peste 8 zile
    today = date(2026, 12, 1)
    rows, missing = upcoming([{"release_name": "FOMC Press Release", "date": "2026-12-02"}], today)
    assert [(r["date"], r["key"], r["weekday"], r["in_days"]) for r in rows] == [(date(2026, 12, 9), "fomc", "miercuri", 8)]
    assert rows[0]["label"] == FOMC_LABEL and "Decizia de dobândă FOMC" not in missing


def test_fomc_sortat_impreuna_cu_fred():
    today = date(2026, 12, 1)
    data = [{"release_name": "Consumer Price Index", "date": "2026-12-10"},
            {"release_name": "Employment Situation", "date": "2026-12-04"}]
    rows, _ = upcoming(data, today)
    assert [r["key"] for r in rows] == ["nfp", "fomc", "cpi"]


def test_fomc_interval_margini():
    assert fomc_in_window(date(2026, 10, 28)) == [date(2026, 10, 28)]           # azi inclus
    assert fomc_in_window(date(2026, 10, 15)) == [date(2026, 10, 28)]           # ultima zi (15 + 13 = 28) inclusă
    assert fomc_in_window(date(2026, 10, 14)) == []                             # 14 + 13 = 27
    assert fomc_in_window(date(2026, 11, 5)) == []


def test_fomc_lista_valida_si_expirare():
    assert list(FOMC_DECISIONS) == sorted(set(FOMC_DECISIONS)) and len(FOMC_DECISIONS) == 16
    assert all(d.weekday() == 2 for d in FOMC_DECISIONS)                       # deciziile sunt miercurea
    assert not fomc_outdated(date(2027, 11, 24))                               # 24.11 + 13 = 07.12 < 08.12
    assert not fomc_outdated(date(2027, 11, 25))                               # = 08.12, încă acoperit
    assert fomc_outdated(date(2027, 11, 26))
    assert fomc_outdated(TODAY, decisions=())
