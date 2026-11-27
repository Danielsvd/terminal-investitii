"""Regenerează `tools/univers/sp500_intervale.csv` din sursa `fja05680/sp500` (MIT). Rulat de mână, rar.

    git clone --depth 1 https://github.com/fja05680/sp500.git /tmp/sp500
    python -m tools.univers.sp500_din_sursa /tmp/sp500

Fișierul sursă: `S&P 500 Historical Components & Changes (Updated).csv` (membrii la fiecare schimbare din 1996).
Se păstrează doar intervalele care se termină după 2011-01-01 (testul începe în 2012).
"""
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analytics.backtest_sec import intervals_from_snapshots  # noqa: E402

SOURCE_FILE = "S&P 500 Historical Components & Changes (Updated).csv"
OUT = ROOT / "tools" / "univers" / "sp500_intervale.csv"
START = "2011-01-01"


def main(argv=None):
    src = Path((argv or sys.argv[1:])[0])
    snaps = pd.read_csv(src / SOURCE_FILE)
    commit = subprocess.run(["git", "-C", str(src), "log", "-1", "--format=%H %cs"], capture_output=True,
                            text=True, check=True).stdout.strip()
    iv = intervals_from_snapshots(snaps, start=START)
    last = pd.to_datetime(snaps["date"]).max().date()
    header = (f"# Sursa: https://github.com/fja05680/sp500 (licență MIT, vezi LICENSE_sp500.txt), commit {commit}\n"
              f"# Fișier: {SOURCE_FILE}; ultima dată din sursă: {last}\n"
              f"# Intervale de apartenență [de_la, pana_la); pana_la gol = membru la ultima dată. Doar intervalele "
              f"încheiate după {START}.\n"
              f"# Simbolurile sunt cele de la data respectivă (ex. FB, nu META). Generat de tools/univers/sp500_din_sursa.py\n")
    body = iv.assign(de_la=iv["de_la"].dt.date, pana_la=iv["pana_la"].dt.date).to_csv(index=False)
    OUT.write_text(header + body, encoding="utf-8")
    print(f"{len(iv)} intervale, {iv['simbol'].nunique()} simboluri, membri la {last}: {iv['pana_la'].isna().sum()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
