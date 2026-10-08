"""Teste pentru analytics/peers.py."""
import pytest

from analytics.peers import METRICS, PEERS, median_of, peer_list, peer_medians, peer_region, versus_median


def test_peer_region():
    assert peer_region("AAPL") == "US" and peer_region("brk-b") == "US"
    assert peer_region("SAP.DE") == "EU" and peer_region("MC.PA") == "EU" and peer_region("SHEL.L") == "EU"
    assert peer_region("SNP.RO") == "BVB"
    assert peer_region("7203.T") is None and peer_region("^GSPC") is None and peer_region(None) is None


def test_peer_list_aceeasi_regiune_fara_simbolul_insusi():
    us = peer_list("AAPL", "Technology")
    assert "AAPL" not in us and "MSFT" in us and all("." not in s for s in us)
    eu = peer_list("SAP.DE", "Technology")
    assert "SAP.DE" not in eu and "ASML.AS" in eu and all("." in s for s in eu)


def test_peer_list_fara_lista_nu_cade_pe_etf_uri():
    assert peer_list("AAPL", None) == []
    assert peer_list("AAPL", "Sector Inexistent") == []
    assert peer_list("SNP.RO", "Energy") == []           # BVB se compară prin foaia BVB
    assert peer_list("7203.T", "Consumer Cyclical") == []


def test_listele_nu_au_duplicate_si_nu_amesteca_regiunile():
    for region, sectors in PEERS.items():
        for sector, names in sectors.items():
            assert len(names) == len(set(names)), (region, sector)
            assert all(peer_region(n) == region for n in names), (region, sector)
    assert set(PEERS["US"]) == set(PEERS["EU"])


def test_median_of():
    assert median_of([10, 30, 20]) == (20, 3)
    assert median_of([10, None, 30, float("nan"), 20, 40]) == (25, 4)        # lipsurile nu intră
    assert median_of([10, 20]) == (None, 2)                                  # sub 3 observații: fără mediană
    assert median_of([-5, 10, 20, 30], positive_only=True) == (20, 3)        # P/E negativ exclus
    assert median_of([-5, 10, 20, 30]) == (15, 4)
    assert median_of([]) == (None, 0)


def test_peer_medians():
    rows = [{"P/E": 10.0, "ROE (%)": 5.0}, {"P/E": 20.0, "ROE (%)": 15.0},
            {"P/E": -8.0, "ROE (%)": -10.0}, {"P/E": 30.0, "ROE (%)": None}]
    med = peer_medians(rows)
    assert med["P/E"] == (20.0, 3)            # pierderea (P/E negativ) nu intră
    assert med["ROE (%)"] == (5.0, 3)         # un ROE negativ e informație și rămâne
    assert med["P/BV"] == (None, 0)
    assert set(med) == {label for _, label, _ in METRICS}


def test_versus_median():
    assert versus_median(25.0, 20.0) == pytest.approx(0.25)
    assert versus_median(15.0, 20.0) == pytest.approx(-0.25)
    assert versus_median(5.0, -10.0) == pytest.approx(1.5)
    assert versus_median(None, 20.0) is None and versus_median(5.0, None) is None and versus_median(5.0, 0) is None
