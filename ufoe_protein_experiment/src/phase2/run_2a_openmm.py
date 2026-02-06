"""
Phase 2a: UFoE Type B 서열 → OpenMM 순수 MD (100ns).
기준 코드(Claude)와 동일한 입출력 규격 사용. 실제 OpenMM 실행은 GPU 환경에서 수행.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import csv
import json

# 프로젝트 루트 기준
def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_phase2_input(input_csv: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Phase 2 입력 CSV 로드. id, group, sequence, length."""
    path = input_csv or (_root() / "data" / "phase2_input" / "sequences.csv")
    if not path.exists():
        return []
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "id": r.get("id", ""),
                "group": r.get("group", "UFoE"),
                "sequence": r.get("sequence", ""),
                "length": int(r.get("length", len(r.get("sequence", "")))),
            })
    return rows


def run_single_openmm(record: Dict[str, Any], output_dir: Path, ns: int = 100) -> Dict[str, Any]:
    """
    단일 서열에 대해 OpenMM MD 시뮬레이션 실행 (스텁).
    실제 구현 시: OpenMM으로 초기 구조 생성 → 100ns MD → RMSD/에너지 계산.
    """
    # 스텁: 실제 OpenMM 호출은 GPU 환경에서 기준 코드로 대체
    return {
        "id": record["id"],
        "group": record["group"],
        "sequence": record["sequence"],
        "fold_success": False,
        "final_rmsd": None,
        "energy_score": None,
        "output_pdb_path": "",
        "zone_EC": None,
        "zone_TZ": None,
        "zone_BZ": None,
        "runtime_sec": None,
        "phase": "2a",
        "stub": True,
    }


def run_phase2a(n: int = 100, input_csv: Optional[Path] = None, output_dir: Optional[Path] = None) -> Path:
    """
    Phase 2a 배치: n개 서열에 대해 순수 MD 실행.
    입력이 없으면 Phase 1 생성기로 UFoE Type B n개 생성 후 CSV 저장.
    반환: 결과 CSV 경로.
    """
    root = _root()
    out_dir = output_dir or (root / "data" / "phase2a_output")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "phase2a_results.csv"

    rows = load_phase2_input(input_csv)
    if not rows:
        # Phase 1 생성기로 n개 생성
        import sys
        sys.path.insert(0, str(root))
        from src.generator.ufoef_generator import generate_candidate
        in_dir = root / "data" / "phase2_input"
        in_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for i in range(n):
            length = 60 + (i % 41)
            seq, _ = generate_candidate(length=length, type="B")
            rows.append({"id": f"ufoe_{i:04d}", "group": "UFoE", "sequence": seq, "length": len(seq)})
        in_csv = in_dir / "sequences.csv"
        with open(in_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["id", "group", "sequence", "length"])
            w.writeheader()
            w.writerows(rows)

    results = []
    struct_dir = out_dir / "structures"
    struct_dir.mkdir(parents=True, exist_ok=True)
    for rec in rows:
        res = run_single_openmm(rec, struct_dir, ns=100)
        results.append(res)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "id", "group", "sequence", "fold_success", "final_rmsd", "energy_score",
            "output_pdb_path", "zone_EC", "zone_TZ", "zone_BZ", "runtime_sec", "phase",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)

    return out_csv
