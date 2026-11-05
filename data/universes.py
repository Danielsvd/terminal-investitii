"""Listele unice de simboluri urmărite, câte una pe piață (Daniel, 05.11.2026).

Simbolurile se adaugă DOAR în fișierele de mai jos. Le folosesc Rezumatul Zilei, Piață Globală și Scanner Volum;
lista BVB o folosește și testul istoric (tools/backtest_scor.py). Piețele nu se amestecă: BVB, SUA, Europa.
Fără rețea și fără `st.*`; fișier lipsă → listă goală + logare (aplicația afișează „N/A”, nu cade).
"""
import logging
from functools import lru_cache
from pathlib import Path

from analytics.briefing import parse_universe, universe_sections

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
UNIVERSE_FILES = {
    "bvb": ROOT / "tools" / "univers_bvb.txt",
    "us": ROOT / "tools" / "univers_urmarit_sua.txt",
    "eu": ROOT / "tools" / "univers_urmarit_ue.txt",
}
MARKET_LABELS = {"bvb": "🇷🇴 BVB", "us": "🇺🇸 SUA", "eu": "🇪🇺 Europa"}


@lru_cache(maxsize=None)
def _text(market):
    path = UNIVERSE_FILES[market]
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("universul %s (%s): %s", market, path.name, exc)
        return ""


def symbols(market):
    """Lista de simboluri a pieței (`bvb`, `us`, `eu`), în ordinea din fișier, fără dubluri."""
    return parse_universe(_text(market))


def sections(market):
    """{simbol: secțiune} pentru piața dată (secțiunile „# == Nume ==” din fișier)."""
    return universe_sections(_text(market))


def groups(market):
    """{secțiune: [simboluri]} în ordinea din fișier; simbolurile fără secțiune merg sub eticheta pieței."""
    out = {}
    for sym, sec in sections(market).items():
        out.setdefault(sec or MARKET_LABELS[market], []).append(sym)
    return out


def market_of(symbol):
    """Piața după sufixul Yahoo: `.RO` → bvb; alt sufix de bursă (`.DE`, `.PA`, `.L`, `.MI`...) → eu; fără → us.
    Folosit de testul de structură (fiecare simbol în fișierul pieței lui)."""
    s = str(symbol).upper()
    if s.endswith(".RO"):
        return "bvb"
    return "eu" if "." in s else "us"
