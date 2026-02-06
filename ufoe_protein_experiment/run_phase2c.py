#!/usr/bin/env python3
"""Phase 2c: 대조군 (UFoE / Random / RFdiffusion) 입력 및 스텁 결과 생성."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

def main():
    from src.phase2.run_2c_control import run_phase2c
    out_csv = run_phase2c(n_per_group=100)
    print("Phase 2c 대조군 입력/스텁 결과:", out_csv)
    return 0

if __name__ == "__main__":
    sys.exit(main())
