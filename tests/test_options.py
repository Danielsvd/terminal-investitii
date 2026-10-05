import numpy as np
import pandas as pd

from analytics.options import max_pain


def _chain(strikes, oi):
    return pd.DataFrame({"strike": strikes, "openInterest": oi})


def test_max_pain_symmetric_chain():
    calls = _chain([90, 100, 110], [10, 20, 30])
    puts = _chain([90, 100, 110], [30, 20, 10])
    # K=90: call 0 + put (20*10 + 10*20) = 400
    # K=100: call 10*10 + put 10*10 = 200
    # K=110: call (10*20 + 20*10) + put 0 = 400
    assert max_pain(calls, puts) == 100.0


def test_max_pain_is_not_simply_the_middle_strike():
    calls = _chain([90], [100])
    puts = _chain([110], [1])
    # K=90: put 1*20 = 20 ; K=110: call 100*20 = 2000
    assert max_pain(calls, puts) == 90.0


def test_max_pain_ignores_missing_open_interest():
    calls = _chain([90, 100, 110], [np.nan, 20, 30])
    puts = _chain([90, 100, 110], [30, 20, np.nan])
    # K=90: put 20*10 = 200 ; K=100: 0 ; K=110: call 20*10 = 200
    assert max_pain(calls, puts) == 100.0


def test_max_pain_none_without_open_interest():
    assert max_pain(_chain([90, 100], [0, 0]), _chain([90, 100], [np.nan, 0])) is None
    assert max_pain(pd.DataFrame(columns=["strike", "openInterest"]), None) is None
