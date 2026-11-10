"""Pragurile unice de corelație (heatmap + carduri): 0,75 / 0,40."""
import numpy as np
import pytest

from analytics.risk import CORR_BANDS, CORR_HIGH, CORR_MID, correlation_band


def test_praguri():
    assert (CORR_HIGH, CORR_MID) == (0.75, 0.40)


@pytest.mark.parametrize("v,band", [
    (0.95, "slaba"), (0.7501, "slaba"),
    (0.75, "moderata"), (0.45, "moderata"), (0.4001, "moderata"),
    (0.40, "buna"), (0.10, "buna"), (0.0, "buna"), (-0.6, "buna"),
])
def test_clase(v, band):
    assert correlation_band(v) == band


@pytest.mark.parametrize("v", [None, np.nan, float("inf"), "abc"])
def test_lipsa(v):
    assert correlation_band(v) is None


def test_etichete_pentru_toate_clasele():
    assert set(CORR_BANDS) == {"slaba", "moderata", "buna"}
