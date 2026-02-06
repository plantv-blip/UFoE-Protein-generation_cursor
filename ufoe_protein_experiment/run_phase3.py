#!/usr/bin/env python3
"""
Phase 3 소거 루프 실행.
Phase 2 결과 CSV(및 구조 PDB)를 읽어 UFoE 5-필터 재적용 → K/M, 탈락 분석, 보고서 저장.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Phase 2 결과 CSV (2a/2b/2c 중 하나 지정)
DEFAULT_RESULTS = ROOT / "data" / "phase2a_output" / "phase2a_results.csv"


def main():
    import argparse
    p = argparse.ArgumentParser(description="Phase 3: UFoE elimination loop")
    p.add_argument("--results", type=Path, default=DEFAULT_RESULTS, help="Phase 2 results CSV")
    p.add_argument("--structures-dir", type=Path, default=None, help="PDB directory (default: <results_dir>/structures)")
    p.add_argument("--strict", action="store_true", help="Use strict filter thresholds")
    p.add_argument("--output", type=Path, default=None, help="Output CSV (default: data/phase3_output/phase3_report.csv)")
    args = p.parse_args()

    from src.phase3.elimination_loop import run_elimination_loop, write_phase3_report

    out_path = args.output or (ROOT / "data" / "phase3_output" / "phase3_report.csv")
    result = run_elimination_loop(
        args.results,
        structures_dir=args.structures_dir,
        strict=args.strict,
    )
    write_phase3_report(result, out_path)

    M, K = result["M"], result["K"]
    km = result["K_over_M"]
    print("=== Phase 3 소거 루프 결과 ===")
    print("M (접힘 완료 구조):", M)
    print("K (UFoE 필터 재통과):", K)
    print("K/M (자기일관성 비율): %.2f" % km)
    print("탈락 필터별:", result["failed_by_filter"])
    print("저장:", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
