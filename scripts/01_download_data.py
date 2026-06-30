"""
Download raw data for the FIFA World Cup Transitivity project.

Sources:
  - Match results: jfjelstul/worldcup (CC-BY-SA 4.0)
  - FIFA rankings history:           Dato-Futbol/fifa-ranking (scraped from FIFA)
"""

import urllib.request
import pathlib
import sys

RAW = pathlib.Path(__file__).parent.parent / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

SOURCES = {
    "matches.csv": (
        "https://raw.githubusercontent.com/jfjelstul/worldcup/master/data-csv/matches.csv"
    ),
    "fifa_rankings_historical.csv": (
        "https://raw.githubusercontent.com/Dato-Futbol/fifa-ranking/master/ranking_fifa_historical.csv"
    ),
}


def download(name: str, url: str) -> None:
    dest = RAW / name
    if dest.exists():
        print(f"  [skip] {name} already exists")
        return
    print(f"  [download] {name} ...", end=" ", flush=True)
    headers = {"User-Agent": "Mozilla/5.0 (research project)"}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as f:
            f.write(resp.read())
        size_kb = dest.stat().st_size / 1024
        print(f"done ({size_kb:.1f} KB)")
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        dest.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    print("Downloading raw data...")
    for name, url in SOURCES.items():
        download(name, url)
    print("All downloads complete.")
    print(f"Files saved to: {RAW}")
