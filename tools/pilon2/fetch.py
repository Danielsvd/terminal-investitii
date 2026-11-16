"""Găsirea și descărcarea rapoartelor „Anexa 12” ale fondurilor Pilon II.

Două feluri de surse:
  - „sablon”: fondul pune rapoartele la adrese previzibile, cu data în nume (Vital, BT, BCR, Metropolitan, Aripi).
    Pentru fiecare sfârșit de lună se încearcă șabloanele, în ordine, până răspunde un PDF. Dacă fondul are și
    o pagină de listare, lunile negăsite prin șablon se caută apoi pe pagină.
  - „pagina”: fondul are o pagină cu legături (NN, AZT). Se citesc legăturile spre PDF, iar data se ia
    din textul sau numele legăturii. Adresa paginii se poate suprascrie în `surse.json`.

Adresele implicite sunt în `SOURCES`; se pot suprascrie fără cod, într-un fișier JSON
(`<data-dir>/surse.json`, aceeași structură, doar cheile care se schimbă).

Data din numele fișierului e doar o indicație: data adevărată e cea din raport (`parse.py`), verificată în `run.py`.

Probă (din rădăcina repo-ului, pe server):
    ~/apps/pilon2/.venv/bin/python -m tools.pilon2.fetch --data-dir ~/apps/pilon2 --luni 3
"""
import argparse
import calendar
import json
import re
import sys
import time
import zipfile
from datetime import date
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin

USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) terminal-investitii/pilon2 (uz personal)"
TIMEOUT_S = 30
ZIP_TIMEOUT_S = 180
ZIP_MAX_AGE_DAYS = 30  # arhiva .zip (NN, ~33 MB) se descarcă din nou cel mult o dată pe lună
PAUSE_S = 0.4          # pauză între cereri către același site

_ML = "https://www.metropolitanlife.ro/content/dam/metlifecom/ro/PDFs/"
_ML_NAMES = (
    "situatia_detaliata_a_portofoliului_de_investitii_la_data_de_{d}.{m}.{Y}.pdf",
    "situatia_detaliata_a_portofoliului_de_investitii_la_data_de_{d}-{m}-{Y}.pdf",
    "Situatia_detaliata_a_portofoliului_de_investitii_la_data_de_{d}.{m}.{Y}.pdf",
    "situatia-detaliata-a-portofoliului-de-investitii-la-data-de-{d}-{m}-{Y}.pdf",
)
_GEN = ("https://www.generali.ro/wp-content/uploads/pension-documents/pensii/pensii-obligatorii/"
        "acte-si-rapoarte/date-financiare-fondul-aripi/")
_BT = ("https://btpensianoastra.ro/storage/app/media/Pilonul%20II/"
       "Structura%20detaliata%20a%20portofoliului%20de%20investitii/{Y}/")

# Cheile sunt și numele folderelor din arhivă (pdf/<cheie>/<AAAA-LL-ZZ>.pdf).
SOURCES = {
    "vital": {
        "fond": "Vital", "cod": "FP2-54", "tip": "sablon",
        "sabloane": ["https://www.carpathiapensii.ro/wp-content/uploads/rapoarte_vital/{Y}/{Y}{m}{d}_A12_detaliata_FP2-54.pdf"],
    },
    "bt": {
        "fond": "BT Pilon 2", "cod": "FP2-83", "tip": "sablon",
        "sabloane": [_BT + "FPAP%20structura%20detaliata%20a%20portofoliului%20la%20{d}.{m}.{Y}.pdf",
                     _BT + "FPAP_structura%20detaliata%20a%20portofoliului%20la%20{d}.{m}.{Y}.pdf"],
    },
    "bcr": {
        "fond": "BCR", "cod": "FP2-123", "tip": "sablon",
        "sabloane": ["https://cdn.erstegroup.com/content/dam/ro/bcr/www_pensiibcr_ro/utile/rapoarte-p2/"
                     "structura-detaliata-a-portofoliului-de-investitii/{Y}/"
                     "Situatia_detaliata_a_portofoliului_de_investitii_FPAP_BCR_la_data_de_{d}.{m}.{Y}.pdf"],
        "pagina": "https://www.pensiibcr.ro/ro/utile/pensia-obligatorie-pilon-2/rapoarte/"
                  "structura-detaliata-a-portofoliului-de-investitii",
    },
    "metropolitan": {
        "fond": "Metropolitan Life", "cod": "FP2-96", "tip": "sablon",
        "sabloane": [base + name for base in (_ML, _ML + "Informatiifinanciare/",
                                              _ML + "Informatiifinanciare/Raportare-lunara/")
                     for name in _ML_NAMES],
    },
    "aripi": {
        "fond": "Aripi", "cod": None, "tip": "sablon",
        "sabloane": [_GEN + "{Y}{m}{d}_pII_inv_det.pdf"],
        "pagina": "https://www.generali.ro/pensii/pensii-obligatorii/acte-si-rapoarte/date-financiare-fondul-aripi/",
    },
    # NN: pagina are doar raportul curent; istoricul e într-o arhivă .zip („Iulie 2026.pdf” …), legată de pe pagină.
    "nn": {"fond": "NN", "cod": "FP2-31", "tip": "pagina", "pagina": "https://www.nn.ro/rapoarte-financiare",
           "arhiva_zip": True},
    # AZT: pagina fondului Viitorul Tău (p_fond=VIITORUL); neverificată pe date reale.
    "azt": {"fond": "AZT Viitorul Tău", "cod": None, "tip": "pagina",
            "pagina": "https://www.aztpensii.ro:7777/employer/prc_rap_web_new_49?p_fond=VIITORUL"},
}

# Legături care par rapoarte de structură detaliată (după adresă sau text, normalizate).
LINK_HINTS = ("structura", "situatia detaliata", "situatia_detaliata", "situatia-detaliata", "detaliat",
              "a12", "anexa 12", "str. inv", "portofoliu")

_RO_MONTHS = {"ianuarie": 1, "ian": 1, "februarie": 2, "feb": 2, "martie": 3, "mar": 3, "aprilie": 4, "apr": 4,
              "mai": 5, "iunie": 6, "iun": 6, "iulie": 7, "iul": 7, "august": 8, "aug": 8, "septembrie": 9,
              "sep": 9, "sept": 9, "octombrie": 10, "oct": 10, "noiembrie": 11, "noi": 11, "nov": 11,
              "decembrie": 12, "dec": 12, "jan": 1, "jun": 6, "jul": 7}


def load_sources(override_path=None):
    """SOURCES, cu cheile din JSON-ul de suprascriere (dacă există) puse peste cele implicite."""
    out = {k: dict(v) for k, v in SOURCES.items()}
    if override_path and Path(override_path).is_file():
        extra = json.loads(Path(override_path).read_text(encoding="utf-8"))
        for key, cfg in extra.items():
            out.setdefault(key, {}).update(cfg)
    return out


def month_end(year, month):
    return date(year, month, calendar.monthrange(year, month)[1])


def month_ends(last, count):
    """Ultimele `count` sfârșituri de lună calendaristică, de la `last` (inclusiv luna lui) înapoi."""
    y, m = last.year, last.month
    if last < month_end(y, m):        # luna curentă nu s-a încheiat → începem cu luna trecută
        y, m = (y, m - 1) if m > 1 else (y - 1, 12)
    out = []
    for _ in range(count):
        out.append(month_end(y, m))
        y, m = (y, m - 1) if m > 1 else (y - 1, 12)
    return out


def candidate_urls(source, d):
    """Adresele de încercat pentru raportul de la data `d` (sfârșit de lună)."""
    fill = {"Y": f"{d.year:04d}", "m": f"{d.month:02d}", "d": f"{d.day:02d}"}
    return [t.format(**fill) for t in source.get("sabloane") or []]


def date_from_text(text):
    """Data raportului din textul sau numele unei legături; None dacă nu se poate deduce.

    Recunoaște: 31.07.2026, 31-07-2026, 31_07_2026, 20260731, „Mai 2025”, „jun_2026”, „_052022” (NN, vechi).
    O lună fără zi înseamnă sfârșitul lunii (rapoartele sunt la sfârșit de lună).
    """
    t = str(text or "").lower()
    m = re.search(r"(?<!\d)(\d{1,2})[._-](\d{1,2})[._-](20\d{2})(?!\d)", t)
    if m:
        try:
            return date(int(m[3]), int(m[2]), int(m[1]))
        except ValueError:
            pass
    m = re.search(r"(?<!\d)(20\d{2})(\d{2})(\d{2})(?!\d)", t)
    if m:
        try:
            return date(int(m[1]), int(m[2]), int(m[3]))
        except ValueError:
            pass
    for m in re.finditer(r"([a-z]+)[\s_-]+(20\d{2})(?!\d)", t):
        if m[1] in _RO_MONTHS:
            return month_end(int(m[2]), _RO_MONTHS[m[1]])
    m = re.search(r"_(\d{2})(20\d{2})(?:\D|$)", t)
    if m and 1 <= int(m[1]) <= 12:
        return month_end(int(m[2]), int(m[1]))
    return None


class _LinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links, self._href, self._text = [], None, []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self._href = dict(attrs).get("href")
            self._text = []

    def handle_data(self, data):
        if self._href is not None:
            self._text.append(data)

    def handle_endtag(self, tag):
        if tag == "a" and self._href is not None:
            self.links.append((self._href, " ".join(" ".join(self._text).split())))
            self._href = None


def pdf_links(html, base_url):
    """Legăturile spre PDF care par rapoarte de structură: listă de (url absolut, text, dată dedusă sau None)."""
    p = _LinkParser()
    p.feed(html or "")
    out, seen = [], set()
    for href, text in p.links:
        if not href:
            continue
        url = urljoin(base_url, href.strip())
        low = (url + " " + text).lower()
        if ".pdf" not in low or not any(h in low.replace("%20", " ") for h in LINK_HINTS):
            continue
        if url in seen:
            continue
        seen.add(url)
        out.append((url, text, date_from_text(text) or date_from_text(url.replace("%20", " "))))
    return out


def zip_links(html, base_url):
    """Legăturile spre arhive .zip cu rapoarte de structură (NN: „Arhiva Structura detaliata … 2021 - 2026.zip”)."""
    p = _LinkParser()
    p.feed(html or "")
    out = []
    for href, text in p.links:
        if not href:
            continue
        url = urljoin(base_url, href.strip())
        low = (url + " " + text).lower().replace("%20", " ")
        if ".zip" in low and "arhiva" in low and "structura" in low and url not in out:
            out.append(url)
    return out


def zip_members_by_date(names):
    """{data: numele fișierului} pentru PDF-urile dintr-o arhivă; data din nume („Iulie 2026.pdf”). Primul câștigă."""
    out = {}
    for n in names:
        if not n.lower().endswith(".pdf"):
            continue
        d = date_from_text(Path(n).name)
        if d is not None and d not in out:
            out[d] = n
    return out


def extract_from_zip(zip_path, dates, data_dir, key):
    """Scoate din arhivă rapoartele pentru `dates` în pdf/<cheie>/<data>.pdf. Întoarce [(data, cale)]."""
    got = []
    try:
        with zipfile.ZipFile(zip_path) as z:
            members = zip_members_by_date(z.namelist())
            for d in dates:
                name = members.get(d)
                if not name:
                    continue
                data = z.read(name)
                if not data.startswith(b"%PDF"):
                    print(f"  ! {key}: {name} din arhivă nu e PDF", file=sys.stderr)
                    continue
                dest = archive_path(data_dir, key, d)
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_suffix(".part")
                tmp.write_bytes(data)
                tmp.replace(dest)
                got.append((d, dest))
    except (zipfile.BadZipFile, OSError) as exc:
        print(f"  ! {key}: arhiva {zip_path}: {type(exc).__name__}: {exc}", file=sys.stderr)
    return got


def fetch_zip(session, source, data_dir, key, now=None):
    """Arhiva .zip a fondului, din cache (pdf/<cheie>/_arhiva.zip) dacă e mai nouă de ZIP_MAX_AGE_DAYS; altfel de pe pagină."""
    cache = Path(data_dir) / "pdf" / key / "_arhiva.zip"
    now = time.time() if now is None else now
    if cache.is_file() and now - cache.stat().st_mtime < ZIP_MAX_AGE_DAYS * 86400:
        return cache
    page = source.get("pagina")
    try:
        r = session.get(page, timeout=TIMEOUT_S)
        r.raise_for_status()
        links = zip_links(r.text, page)
        if not links:
            print(f"  {key}: pagina nu are arhivă .zip cu rapoarte", file=sys.stderr)
            return cache if cache.is_file() else None
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(".part")
        with session.get(links[0], timeout=ZIP_TIMEOUT_S, stream=True) as z:
            z.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in z.iter_content(1 << 20):
                    f.write(chunk)
        if not zipfile.is_zipfile(tmp):
            tmp.unlink(missing_ok=True)
            print(f"  ! {key}: {links[0]}: răspunsul nu e o arhivă .zip", file=sys.stderr)
            return cache if cache.is_file() else None
        tmp.replace(cache)
        return cache
    except Exception as exc:  # rețea: se raportează, restul rulării continuă
        print(f"  ! {key}: arhiva .zip: {type(exc).__name__}: {exc}", file=sys.stderr)
        return cache if cache.is_file() else None


def make_session():
    import requests

    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    return s


def _is_pdf_response(resp):
    ctype = (resp.headers.get("Content-Type") or "").lower()
    return resp.status_code == 200 and ("pdf" in ctype or "octet-stream" in ctype)


def probe(session, url):
    """True dacă adresa răspunde cu un PDF (HEAD; dacă serverul nu acceptă HEAD, GET parțial)."""
    try:
        r = session.head(url, timeout=TIMEOUT_S, allow_redirects=True)
        if r.status_code in (403, 405, 501):
            r = session.get(url, timeout=TIMEOUT_S, stream=True, allow_redirects=True)
            r.close()
        return _is_pdf_response(r)
    except Exception as exc:  # rețea: se raportează ca negăsit, nu oprește rularea
        print(f"  ! {url}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return False


def discover(key, source, dates, session, pause=PAUSE_S):
    """Rapoartele găsite pentru fond: listă de (data indicată, url). Datele lipsă nu apar.

    La sursele „sablon” cu pagină de listare, lunile negăsite prin șablon se caută pe pagină.
    """
    found = []
    if source.get("tip") == "sablon":
        for d in dates:
            for url in candidate_urls(source, d):
                time.sleep(pause)
                if probe(session, url):
                    found.append((d, url))
                    break
        got = {d for d, _u in found}
        missing = [d for d in dates if d not in got]
        if not missing or not source.get("pagina"):
            return found
        dates = missing
    page = source.get("pagina")
    if not page:
        print(f"  {key}: pagina cu rapoarte nu e configurată (surse.json)", file=sys.stderr)
        return found
    try:
        r = session.get(page, timeout=TIMEOUT_S)
        r.raise_for_status()
    except Exception as exc:
        print(f"  ! {key}: pagina {page}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return found
    wanted = set(dates)
    on_page, seen = [], set()
    for url, _text, d in pdf_links(r.text, page):
        if d is not None and d in wanted and d not in seen:   # o singură legătură pe lună (NN o pune în mai multe taburi)
            seen.add(d)
            on_page.append((d, url))
    if not on_page:
        print(f"  {key}: pagina nu are rapoarte pentru lunile căutate ({page})", file=sys.stderr)
    return sorted(set(found + on_page), reverse=True)


def archive_path(data_dir, key, d):
    return Path(data_dir) / "pdf" / key / f"{d.isoformat()}.pdf"


def download(session, url, dest):
    """Descarcă PDF-ul în `dest` (prin fișier temporar). True dacă a reușit și fișierul începe cu %PDF."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    try:
        with session.get(url, timeout=TIMEOUT_S, stream=True) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(65536):
                    f.write(chunk)
        with open(tmp, "rb") as f:
            if f.read(4) != b"%PDF":
                tmp.unlink(missing_ok=True)
                print(f"  ! {url}: răspunsul nu e PDF", file=sys.stderr)
                return False
        tmp.replace(dest)
        return True
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        print(f"  ! {url}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return False


def fetch_all(data_dir, months, keys=None, session=None, sources=None, today=None, pause=PAUSE_S):
    """Găsește și descarcă rapoartele lipsă din arhivă. Întoarce {cheie: [(data, cale, nou)]}."""
    sources = sources or load_sources(Path(data_dir) / "surse.json")
    session = session or make_session()
    dates = month_ends(today or date.today(), months)
    result = {}
    for key, src in sources.items():
        if keys and key not in keys:
            continue
        have = {d for d in dates if archive_path(data_dir, key, d).is_file()}
        todo = [d for d in dates if d not in have]
        got = [(d, archive_path(data_dir, key, d), False) for d in sorted(have, reverse=True)]
        for d, url in (discover(key, src, todo, session, pause) if todo else []):
            dest = archive_path(data_dir, key, d)
            if download(session, url, dest):
                got.append((d, dest, True))
        found = {d for d, *_x in got}
        rest = [d for d in todo if d not in found]
        if rest and src.get("arhiva_zip") and src.get("pagina"):
            zip_path = fetch_zip(session, src, data_dir, key)
            if zip_path:
                # din arhivă = rapoarte vechi: data publicării nu e cunoscută → „nou” = False
                got.extend((d, path, False) for d, path in extract_from_zip(zip_path, rest, data_dir, key))
        result[key] = sorted(got, reverse=True)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True, help="folderul dedicat (ex. ~/apps/pilon2)")
    p.add_argument("--luni", type=int, default=3, help="câte sfârșituri de lună, înapoi de azi")
    p.add_argument("--fond", default="", help="doar aceste chei, separate prin virgulă (ex. vital,bt)")
    args = p.parse_args(argv)
    keys = [k.strip() for k in args.fond.split(",") if k.strip()] or None
    data_dir = Path(args.data_dir).expanduser()
    result = fetch_all(data_dir, args.luni, keys)
    for key, items in result.items():
        new = sum(1 for *_x, is_new in items if is_new)
        print(f"{key}: {len(items)} rapoarte în arhivă ({new} noi)"
              + ("".join(f"\n  {d}  {path}{'  (nou)' if is_new else ''}" for d, path, is_new in items)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
