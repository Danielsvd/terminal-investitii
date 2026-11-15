"""Citirea rapoartelor ASF „Anexa 12 — structura detaliată a portofoliului” ale fondurilor Pilon II.

Toate cele 7 fonduri publică același raport, cu coloanele: denumire emitent, ISIN, valoare actualizată (lei),
pondere în activul total. Diferă doar așezarea în pagină:
  - un rând complet:          „Banca Transilvania SA ROTLVAACNOR1 894,075,849.00 3.996%”
  - numele pe rândul de sus:  „Ministerul Finantelor Publice” / „RO1631DBN055 117,233,612.80 1.65%”
  - numerele pe rândul de jos: „Hidroelectrica SA RO4Q0Z5RO1B6” / „381,221,274.00 3.90%”
  - doar ISIN (Aripi, din 31.03.2026): „RO4Q0Z5RO1B6 381,221,274.00 3.90%”
  - ponderea fără „%” (NN):   „... 1,058,918,959.06 2.08”

Rapoartele NU conțin numărul de acțiuni; cantitatea se derivă în altă parte (valoare / preț).

Funcțiile de aici sunt pure (primesc rândurile de text), cu excepția `pdf_lines` / `parse_pdf`, care citesc
fișierul cu `pdfplumber` (importat doar acolo, ca testele și aplicația să nu depindă de el).
"""
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date

# Codurile ASF ale fondurilor, văzute în rapoarte (15.11.2026). Cele lipsă (AZT, BCR, Aripi) se adaugă
# după primul raport citit; până atunci fondul apare cu codul brut.
FUND_CODES = {
    "FP2-31": "NN",
    "FP2-54": "Vital",
    "FP2-83": "BT Pilon 2",
    "FP2-96": "Metropolitan Life",
}

EQUITY = "actiuni"
SUM_TOLERANCE = 0.005          # suma rândurilor de acțiuni vs totalul secțiunii: abatere admisă 0,5%

_NUM = r"-?\d[\d,.]*"
_ISIN = r"[A-Z]{2}[A-Z0-9]{9}\d"
_ROW = re.compile(rf"^(?P<name>.*?)\s*\b(?P<isin>{_ISIN})\s+(?P<value>{_NUM})\s+(?P<weight>{_NUM})\s*%?\s*$")
_ISIN_ONLY = re.compile(rf"^(?P<name>.*?)\s*\b(?P<isin>{_ISIN})\s*$")
_NUM_ONLY = re.compile(rf"^(?P<value>{_NUM})\s+(?P<weight>{_NUM})\s*%?\s*$")
_TAIL = rf"(?:\s+(?P<value>{_NUM}|-)\s+(?P<weight>{_NUM})\s*%?)?"
_MAJOR = re.compile(rf"^(?P<major>\d{{1,2}})\.\s*(?P<title>[^\d\s].*?){_TAIL}\s*$")
_SUB = re.compile(rf"^(?P<letter>[a-z])[\)\.]\s+(?P<title>.+?){_TAIL}\s*$")
_TOTAL = re.compile(rf"^total\s+activ\w*\s*:?\s*(?P<value>{_NUM})")
_DATE = r"(\d{1,2})([./-])(\d{1,2})\2(\d{4})"
_REPORT_DATE = re.compile(rf"la\s+data\s+de\s*:?\s*{_DATE}")
_ONLY_DATE = re.compile(rf"^{_DATE}$")
_PUBLISHED = re.compile(rf"(?:numar|nr\.?)\s*(?:iesire|inreg\w*\.?|intern)\s*:?\s*\d+\s*/\s*{_DATE}")
_PUBLISHED_BARE = re.compile(rf"^\d{{6,8}}\s*/\s*{_DATE}$")      # Metropolitan 08.2025: „1307878/12.12.2025”
_FUND_CODE = re.compile(r"\bFP2-\d+\b")
_NOISE = re.compile(r"\d{5,}|\d[./-]\d{1,2}[./-]\d{2,4}|:|\bfp\d-\d+")   # coduri, date, „Etichetă:” — nu sunt nume

# Rânduri de antet / subsol care nu sunt niciodată numele unui emitent.
_NOT_A_NAME = ("structura", "portofoliul", "denumire", "pondere", "valoare actualizata", "administrator",
               "depozitar", "cod fond", "cod administrator", "cod depozitar", "fond de pensii", "isin",
               "col ", "pagina", "intocmit", "verificat", "societate de administrare", "fondurilor de pensii")


def normalize(text):
    """Litere mici, fără diacritice (ă â î ș ş ț ţ), spații simple."""
    t = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", t).strip().lower()


def parse_amount(text):
    """Număr din raport („1,058,918,959.06”, „-35,383,211.30”, „0.000”, „1.234.567,89”); „-” sau gol → None."""
    s = str(text or "").strip().replace(" ", "").replace(" ", "")
    if s in ("", "-"):
        return None
    neg = s.startswith("-")
    s = s.lstrip("-")
    if "," in s and "." in s:
        dec = "." if s.rfind(".") > s.rfind(",") else ","
        s = s.replace("," if dec == "." else ".", "").replace(dec, ".")
    elif "," in s:
        parts = s.split(",")
        s = s.replace(",", "") if (len(parts) > 2 or len(parts[-1]) == 3) else s.replace(",", ".")
    elif s.count(".") > 1:
        s = s.replace(".", "")
    try:
        v = float(s)
    except ValueError:
        return None
    return -v if neg else v


def isin_checksum_ok(isin):
    """Cifra de control ISIN (Luhn pe cifrele obținute din litere: A=10 … Z=35)."""
    if not re.fullmatch(_ISIN, str(isin or "")):
        return False
    digits = "".join(str(int(c, 36)) for c in isin[:-1])
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 0:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return (10 - total % 10) % 10 == int(isin[-1])


def _to_date(a, sep, b, year):
    """zz.ll.aaaa / zz-ll-aaaa; cu „/” și a doua parte > 12 → ll/zz/aaaa (Metropolitan, rapoartele vechi)."""
    a, b, year = int(a), int(b), int(year)
    day, month = (b, a) if (sep == "/" and b > 12) else (a, b)
    try:
        return date(year, month, day)
    except ValueError:
        return None


def parse_report_date(lines):
    """Data raportului: după „la data de”, altfel primul rând care e doar o dată (BT: „31-01-2026”)."""
    head = [normalize(x) for x in lines[:80]]
    for t in head:
        m = _REPORT_DATE.search(t)
        if m:
            return _to_date(*m.groups())
    for t in head:
        m = _ONLY_DATE.match(t)
        if m:
            return _to_date(*m.groups())
    return None


def parse_published_date(lines):
    """Data numărului de ieșire / înregistrare (Metropolitan: „Număr ieșire: 1226411/11.07.2025”), dacă există."""
    for x in lines[:80]:
        t = normalize(x)
        m = _PUBLISHED.search(t) or _PUBLISHED_BARE.match(t)
        if m:
            return _to_date(*m.groups())
    return None


def parse_fund_code(lines):
    for x in lines[:80]:
        m = _FUND_CODE.search(str(x))
        if m:
            return m.group(0)
    return None


def classify(major, major_title, sub_title):
    """Clasa rândului după secțiunea raportului (titlurile normalizate)."""
    mt, st = major_title or "", sub_title or ""
    if "opcvm" in mt or "organisme de plasament" in mt:
        return "opcvm"
    if "private equity" in mt:
        return "private_equity"
    if major in (1, 2) and "titluri de stat" in st:
        return "titluri_stat"
    if major == 2 and "actiun" in st:
        return EQUITY
    if major == 2 and "obligatiuni corporative" in st:
        return "obligatiuni_corporative"
    if major == 2 and "obligatiuni" in st:
        return "obligatiuni_locale"
    return "altele"


@dataclass
class ParsedReport:
    fund_code: str | None = None
    report_date: date | None = None
    published_date: date | None = None
    rows: list = field(default_factory=list)            # dict: Clasa, Denumire, ISIN, Valoare_RON, Pondere, ISIN_valid
    section_totals: dict = field(default_factory=dict)  # (major, litera sau None) → valoarea din antet
    total_assets: float | None = None
    equity_header_total: float | None = None            # valoarea din antetul secțiunii de acțiuni

    @property
    def fund(self):
        return FUND_CODES.get(self.fund_code, self.fund_code)

    def equities(self):
        return [r for r in self.rows if r["Clasa"] == EQUITY]


def parse_lines(lines):
    """Rândurile cu ISIN din raport, fiecare cu clasa secțiunii în care se află."""
    rep = ParsedReport(fund_code=parse_fund_code(lines), report_date=parse_report_date(lines),
                       published_date=parse_published_date(lines))
    major, major_title, sub_title = None, "", ""
    pending_name, pending_row = [], None

    def add(name, isin, value, weight):
        rep.rows.append({
            "Clasa": classify(major, major_title, sub_title),
            "Denumire": re.sub(r"\s+", " ", name).strip(),
            "ISIN": isin,
            "Valoare_RON": parse_amount(value),
            "Pondere": parse_amount(weight),
            "ISIN_valid": isin_checksum_ok(isin),
        })

    for raw in lines:
        s = str(raw or "").strip()
        if not s:
            continue
        m = _ROW.match(s)
        if m:
            add(m["name"] or " ".join(pending_name), m["isin"], m["value"], m["weight"])
            pending_name, pending_row = [], None
            continue
        m = _ISIN_ONLY.match(s)
        if m:
            pending_row = ((m["name"] or " ".join(pending_name)), m["isin"])
            pending_name = []
            continue
        m = _NUM_ONLY.match(s)
        if m:
            if pending_row:
                add(pending_row[0], pending_row[1], m["value"], m["weight"])
            pending_name, pending_row = [], None
            continue
        t = normalize(s)
        mt = _TOTAL.match(t)
        if mt:
            rep.total_assets = parse_amount(mt["value"])
            pending_name, pending_row = [], None
            continue
        m = _MAJOR.match(s)
        if m:
            major, major_title, sub_title = int(m["major"]), normalize(m["title"]), ""
            if m["value"]:
                rep.section_totals[(major, None)] = parse_amount(m["value"])
            pending_name, pending_row = [], None
            continue
        m = _SUB.match(s)
        if m:
            sub_title = normalize(m["title"])
            if m["value"]:
                rep.section_totals[(major, m["letter"])] = parse_amount(m["value"])
                if classify(major, major_title, sub_title) == EQUITY:
                    rep.equity_header_total = parse_amount(m["value"])
            pending_name, pending_row = [], None
            continue
        if any(k in t for k in _NOT_A_NAME) or _NOISE.search(t):
            pending_name, pending_row = [], None
            continue
        pending_name = (pending_name + [s])[-3:]
    return rep


def equity_section_total(rep):
    """Totalul din antetul secțiunii de acțiuni („d) Acțiuni 5,907,697,473.66 26.405%”), dacă apare."""
    return rep.equity_header_total


def check(rep):
    """Probleme de calitate; lista goală = raport citit complet."""
    problems = []
    if rep.fund_code is None:
        problems.append("codul fondului (FP2-…) lipsește")
    if rep.report_date is None:
        problems.append("data raportului lipsește")
    eq = rep.equities()
    if not eq:
        problems.append("nicio acțiune găsită (secțiunea 2, „Acțiuni”)")
        return problems
    missing = [r["ISIN"] for r in eq if r["Valoare_RON"] is None]
    if missing:
        problems.append(f"{len(missing)} rânduri de acțiuni fără valoare: {', '.join(missing[:5])}")
    header = equity_section_total(rep)
    if header:
        got = sum(r["Valoare_RON"] or 0.0 for r in eq)
        dev = abs(got - header) / abs(header)
        if dev > SUM_TOLERANCE:
            problems.append(f"suma acțiunilor {got:,.2f} diferă de totalul secțiunii {header:,.2f} ({dev:.2%})")
    else:
        problems.append("totalul secțiunii de acțiuni nu apare în antet (suma nu poate fi verificată)")
    bad = [r["ISIN"] for r in eq if not r["ISIN_valid"]]
    if bad:
        problems.append(f"ISIN cu cifră de control greșită: {', '.join(bad[:5])}")
    return problems


def pdf_lines(path):
    """Textul PDF-ului, rând cu rând (toate paginile)."""
    import pdfplumber

    out = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            out.extend((page.extract_text(x_tolerance=1.5, y_tolerance=3) or "").splitlines())
    return out


def parse_pdf(path):
    return parse_lines(pdf_lines(path))


def summary(rep):
    """Rezumatul unui raport citit, pentru verificarea de mână (rânduri de text)."""
    from collections import Counter

    eq = rep.equities()
    eq_sum = sum(r["Valoare_RON"] or 0.0 for r in eq)
    out = [f"Fond: {rep.fund or '?'} ({rep.fund_code or '?'}) · data raportului: {rep.report_date or '?'}"
           f" · nr. ieșire/înregistrare: {rep.published_date or '-'}",
           "Rânduri cu ISIN pe clase: " + ", ".join(f"{k} {v}" for k, v in sorted(Counter(r['Clasa'] for r in rep.rows).items())),
           f"Acțiuni: {len(eq)} rânduri, suma {eq_sum:,.2f} lei; totalul din antet: "
           + (f"{rep.equity_header_total:,.2f}" if rep.equity_header_total else "-")
           + (f"; total activ: {rep.total_assets:,.2f}" if rep.total_assets else "")]
    for r in sorted(eq, key=lambda r: -(r["Valoare_RON"] or 0))[:15]:
        out.append(f"  {r['ISIN']}  {r['Valoare_RON'] or 0:>18,.2f}  {r['Pondere'] if r['Pondere'] is not None else '-':>7}  {r['Denumire'][:45]}")
    problems = check(rep)
    out.append("Probleme: " + ("niciuna" if not problems else ""))
    out.extend(f"  - {p}" for p in problems)
    return out


def main(argv=None):
    """Probă de mână: python -m tools.pilon2.parse raport1.pdf [raport2.pdf ...]. Cod 1 dacă vreun raport are probleme."""
    import sys

    paths = sys.argv[1:] if argv is None else argv
    if not paths:
        print("Folosire: python -m tools.pilon2.parse raport.pdf [alt_raport.pdf ...]")
        return 2
    bad = 0
    for path in paths:
        print(f"=== {path}")
        rep = parse_pdf(path)
        for line in summary(rep):
            print(line)
        bad += bool(check(rep))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
