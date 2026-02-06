#!/usr/bin/env python3
"""Phase 2b: ESMFold + MD. 입력이 있으면 2b 스텁 결과 생성."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

def main():
    from src.phase2.run_2b_esmfold_md import run_phase2b, load_phase2_input
    inp = ROOT / "data" / "phase2_input" / "sequences.csv"
    rows = load_phase2_input(inp)
    n = len(rows) if rows else 100
    out_csv = run_phase2b(n=n, input_csv=inp if rows else None)
    print("Phase 2b 스텁 결과:", out_csv)
    return 0

if __name__ == "__main__":
    sys.exit(main())
