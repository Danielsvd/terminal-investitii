"""Calendarul macro FRED: potrivirea publicărilor, intervalul de 14 zile, eroarea fără cheie."""
from datetime import date

from data.fred_calendar import CALENDAR_DAYS, TRACKED, all_release_names, error_text, upcoming, window

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
    assert len(missing) == len(TRACKED) - 2 and "PIB" in missing and "Inflația CPI" not in missing


def test_azi_si_ultima_zi_sunt_incluse():
    data = [{"release_name": "Surveys of Consumers", "date": "2026-11-05"},
            {"release_name": "Producer Price Index", "date": "2026-11-18"}]
    rows, _ = upcoming(data, TODAY)
    assert [r["in_days"] for r in rows] == [0, CALENDAR_DAYS - 1]


def test_raspuns_gol():
    rows, missing = upcoming(None, TODAY)
    assert rows == [] and len(missing) == len(TRACKED)
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
