"""Download *The Adventures of Sherlock Holmes* (Project Gutenberg #1661) to data/.

Default destination: data/sherlock.txt (plain-text UTF-8 from Project Gutenberg).

This is the script the assignment asks for in Core Task 1.b — "write a script to
download and load this text file." We keep it deliberately small: no third-party
HTTP library, no caching framework, just urllib.

Usage:
    python src/download_data.py                    # downloads to data/sherlock.txt
    python src/download_data.py --out data/foo.txt # custom destination
    python src/download_data.py --force            # re-download even if file exists
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

# Project Gutenberg's mirrored, stable plain-text UTF-8 URL for eBook #1661.
GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/1661/pg1661.txt"

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "data" / "sherlock.txt"


def download(url: str, out_path: Path, force: bool = False) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not force:
        size = out_path.stat().st_size
        print(f"Skipping download — {out_path} already exists ({size:,} bytes). "
              f"Pass --force to overwrite.")
        return out_path

    print(f"Downloading {url} → {out_path} …")
    # Project Gutenberg requires a User-Agent header; the default urllib UA gets
    # rate-limited / 403'd.
    req = urllib.request.Request(url, headers={"User-Agent": "sherlock-next-word/0.1"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
    out_path.write_bytes(data)

    sha = hashlib.sha256(data).hexdigest()[:16]
    print(f"Downloaded {len(data):,} bytes (sha256[:16] = {sha}).")
    return out_path


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default=GUTENBERG_URL)
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    download(args.url, Path(args.out), force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
