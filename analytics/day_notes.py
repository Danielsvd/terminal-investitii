"""Rezumatul Zilei, etapa E — „3 lucruri de știut azi”: text din reguli fixe, fără model și fără scoruri noi.

Funcții pure (fără Streamlit, fără rețea); textul e simplu, ca să poată fi trimis și de botul Telegram.

Regulile (descriptive, netestate ca semnal):
1. Portofoliul: P/L-ul ultimei ședințe pe acțiuni și ETF (lei și %) și poziția care a contat cel mai mult.
2. Piața: dintre BET (proxy ETF), STOXX 600 și S&P 500, indicele cu cea mai mare mișcare în σ (|σ|).
3. Primul disponibil dintre: publicările macro SUA de azi/mâine; pozițiile tale peste 2σ; sectorul S&P 500 cu cea mai
   mare abatere față de indice (cel puțin SECTOR_MIN_PP); cea mai nouă știre despre pozițiile tale.
Dacă 1 sau 2 lipsesc, locul lor îl iau următoarele din lista de la 3, în aceeași ordine. Niciun text nu e inventat:
o regulă fără date nu produce rând.
"""
from data.helpers import format_ro

SECTOR_MIN_PP = 1.0      # sub 1 pp față de S&P 500, sectorul nu e „de știut” (prag descriptiv, netestat)
SIGMA_ALERT = 2.0
N_ITEMS = 3


def _signed(v, decimals=2, suffix="%"):
    return ("+" if v > 0 else "") + format_ro(v, decimals, suffix)


def _sigma_word(s):
    a = abs(s)
    return "neobișnuită" if a >= 2 else ("mai mare decât de obicei" if a >= 1 else "obișnuită")


def portfolio_note(totals, rows_by_symbol=None):
    """Rândul despre portofoliu sau None. totals: dict pnl_ron, pct, top, bottom (de la portfolio_day);
    rows_by_symbol: {simbol: dict PnL_RON, Contribution}."""
    if not totals or totals.get("pnl_ron") is None:
        return None
    rows_by_symbol = rows_by_symbol or {}
    pnl, pct = totals["pnl_ron"], totals.get("pct")
    text = f"Portofoliul (acțiuni și ETF): {_signed(pnl, 0, ' lei')}" + (f" ({_signed(pct * 100)})" if pct is not None else "") \
        + " în ultima ședință"
    side = totals.get("bottom") if pnl < 0 else totals.get("top")
    if side:
        sym = side[0]
        r = rows_by_symbol.get(sym) or {}
        c = r.get("Contribution")
        text += f"; cel mai mult a contat {sym.replace('.RO', '')}" + (f" ({_signed(c * 100, 2, ' pp')})" if c is not None else "")
    return {"tag": "portofoliu", "text": text + "."}


def market_note(moves):
    """Indicele cu cea mai mare |σ|. moves: listă de (nume, dict pct, sigma, date de la briefing.daily_move).
    Fără σ la niciunul → cel cu cea mai mare |variație|; fără variații → None."""
    ok = [(n, m) for n, m in moves or [] if m and m.get("pct") is not None]
    if not ok:
        return None
    with_sig = [(n, m) for n, m in ok if m.get("sigma") is not None]
    name, m = max(with_sig, key=lambda x: abs(x[1]["sigma"])) if with_sig else max(ok, key=lambda x: abs(x[1]["pct"]))
    day = f" ({m['date']:%d.%m})" if m.get("date") is not None else ""
    text = f"Piața: cea mai mare mișcare la {name}{day}, {_signed(m['pct'] * 100)}"
    if m.get("sigma") is not None:
        text += f" ({_signed(m['sigma'], 1, 'σ')}, mișcare {_sigma_word(m['sigma'])} față de ultimele 60 de ședințe)"
    others = [f"{n} {_signed(x['pct'] * 100)}" for n, x in ok if n != name]
    return {"tag": "piață", "text": text + (f"; {', '.join(others)}." if others else ".")}


def events_note(soon):
    """Publicările macro de azi/mâine (de la fred_calendar.near) sau None."""
    if not soon:
        return None
    parts = [f"{'azi' if r['in_days'] == 0 else 'mâine'} {r['label']}" for r in soon[:3]]
    return {"tag": "calendar", "text": "Calendar SUA: " + "; ".join(parts) + "."}


def alerts_note(alerts, rows_by_symbol=None):
    """Pozițiile tale cu |σ| ≥ 2 sau None. alerts: simbolurile (de la portfolio_day); rows: dict Pct, Sigma."""
    rows_by_symbol = rows_by_symbol or {}
    parts = []
    for s in alerts or []:
        r = rows_by_symbol.get(s) or {}
        if r.get("Pct") is None or r.get("Sigma") is None:
            continue
        parts.append(f"{s.replace('.RO', '')} {_signed(r['Pct'] * 100)} ({_signed(r['Sigma'], 1, 'σ')})")
    if not parts:
        return None
    return {"tag": "poziții", "text": f"Mișcări neobișnuite (peste {format_ro(SIGMA_ALERT, 0)}σ) la pozițiile tale: "
                                      + ", ".join(parts[:4]) + "."}


def sector_note(sectors):
    """Sectorul cu cea mai mare |abatere| față de S&P 500, dacă trece de SECTOR_MIN_PP. sectors: listă de dict
    name, etf, pct (fracție), vs_index_pp."""
    ok = [s for s in sectors or [] if s.get("vs_index_pp") is not None and s.get("pct") is not None]
    if not ok:
        return None
    s = max(ok, key=lambda x: abs(x["vs_index_pp"]))
    if abs(s["vs_index_pp"]) < SECTOR_MIN_PP:
        return None
    return {"tag": "sector", "text": f"Sectorul {s['name']} ({s['etf']}) {_signed(s['pct'] * 100)}, "
                                     f"{_signed(s['vs_index_pp'], 2, ' pp')} față de S&P 500."}


def news_note(news):
    """Cea mai nouă știre despre pozițiile tale (de la data.news.holdings_news) sau None."""
    if not news:
        return None
    it = news[0]
    more = len(news) - 1
    return {"tag": "știri", "text": f"Știre despre {it['symbol'].replace('.RO', '')}: „{it['title']}” ({it['source']}, "
                                    f"{it['time']:%d.%m %H:%M})" + (f"; încă {more} în tabul „📰 Știri”." if more else ".")}


def three_things(portfolio=None, market=None, events=None, alerts=None, sector=None, news=None, n=N_ITEMS):
    """Cel mult `n` rânduri, în ordinea regulilor din docstring-ul modulului. Argumentele sunt rândurile deja
    construite (dict tag, text) sau None."""
    fixed = [x for x in (portfolio, market) if x]
    rest = [x for x in (events, alerts, sector, news) if x]
    return (fixed + rest)[:n]
