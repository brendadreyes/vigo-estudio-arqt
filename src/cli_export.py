
from __future__ import annotations

from pathlib import Path
from src.pipeline import run_all

if __name__ == "__main__":
    base = Path(__file__).resolve().parents[1]
    excel_path = base / "data" / "GENERAL.xlsx"
    out_dir = base / "artifacts"
    run_all(excel_path, out_dir)
    print(f"OK: exportados CSVs a {out_dir}")
