#!/usr/bin/env python3
"""Phase 5: 자기일관적 구조와 PDB DB 비교 (TM-align)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main():
    from src.phase5.pdb_compare import (
        run_phase5_compare,
        write_phase5_report,
        find_tmalign,
    )

    # Phase 3 통과 구조 PDB 목록 (기준 파이프라인 출력 또는 로컬 phase2a 구조)
    query_dir = ROOT / "data" / "phase2a_output" / "structures"
    if not query_dir.exists():
        query_dir = Path(ROOT.parent) / "claude code" / "ufoe_protein_experiment" / "output" / "mock_test" / "pdb" / "ufoe_2a"
    query_list = list(query_dir.glob("*.pdb")) if query_dir.exists() else []

    # 비교 대상 PDB (워크스페이스 pdb_complexes / pdb_p53 / data/pdb_representative)
    pdb_db = ROOT.parent / "pdb_complexes"
    if not pdb_db.exists():
        pdb_db = ROOT.parent / "pdb_p53"
    if not pdb_db.exists():
        pdb_db = ROOT / "data" / "pdb_representative"

    if not query_list:
        print("Phase 5: 비교할 query PDB가 없습니다. 기준 파이프라인 mock 실행 후 output/.../pdb/ufoe_2a 를 사용하세요.")
        # 스텁 결과만 생성
        from src.phase5.pdb_compare import Phase5Result, write_phase5_report
        results = []
        out_csv = ROOT / "data" / "phase5_output" / "phase5_results.csv"
        write_phase5_report(results, out_csv, out_csv.with_suffix(".md"))
        return 0

    results = run_phase5_compare(
        query_pdb_list=query_list[:50],
        pdb_db_dir=pdb_db,
        tm_threshold=0.5,
        mock_if_no_tmalign=True,
    )
    out_csv = ROOT / "data" / "phase5_output" / "phase5_results.csv"
    out_md = ROOT / "data" / "phase5_output" / "phase5_summary.md"
    write_phase5_report(results, out_csv, out_md)

    n = len(results)
    n_matched = sum(1 for r in results if r.matched)
    print("Phase 5 요약:", n_matched, "/", n, "매칭 (TM-score > 0.5)")
    print("저장:", out_csv, out_md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
