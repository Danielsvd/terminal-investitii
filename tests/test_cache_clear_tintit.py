"""Garda: aplicația nu golește global cache-ul Streamlit (st.cache_data.clear()).

Golirea globală aruncă și datele de piață (Yahoo, FRED, BCE) și forțează
reîncărcarea lor. Se golește doar funcția afectată: `functie.clear()`.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_fara_clear_global():
    for name in ("app.py", "ai_engine.py"):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert "st.cache_data.clear()" not in text.replace(
            "înainte: st.cache_data.clear() global", ""
        ), f"{name} conține st.cache_data.clear() global"
