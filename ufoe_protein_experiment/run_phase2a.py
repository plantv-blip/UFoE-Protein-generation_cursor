#!/usr/bin/env python3
"""Phase 2a: UFoE Type B 서열 → OpenMM 순수 MD (입출력 생성). GPU 실행은 기준 코드로 대체."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

def main():
    from src.phase2.run_2a_openmm import run_phase2a
    n = 100
    out_csv = run_phase2a(n=n)
    print("Phase 2a 입력/스텁 결과 생성 완료:", out_csv)
    print("실제 MD 실행은 OpenMM GPU 환경에서 기준 코드(Claude)로 수행 후 동일 CSV 형식으로 채우세요.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
