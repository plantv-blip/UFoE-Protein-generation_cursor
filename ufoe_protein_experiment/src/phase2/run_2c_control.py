"""
Phase 2c: 대조군 생성 및 실행.
그룹 A: UFoE Type B (n=100), B: 동일 조성 무작위 (n=100), C: RFdiffusion (n=100).
입출력 규격 동일. 실제 RFdiffusion/OpenMM은 GPU 환경에서.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import csv
import random


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def generate_random_same_composition(sequence: str) -> str:
    """동일 1-letter 조성으로 순서만 무작위."""
    chars = list(sequence)
    random.shuffle(chars)
    return "".join(chars)


def run_phase2c(
    n_per_group: int = 100,
    input_csv: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Path:
    """
    대조군 입력 CSV 생성 (A/B/C). 실제 MD는 2a/2b와 동일 러너 사용.
    반환: 생성된 입력 CSV 경로 (또는 2c 결과 CSV).
    """
    root = _root()
    out_dir = output_dir or (root / "data" / "phase2c_output")
    out_dir.mkdir(parents=True, exist_ok=True)
    in_dir = root / "data" / "phase2_input"
    in_dir.mkdir(parents=True, exist_ok=True)

    # UFoE Type B n개 생성
    import sys
    sys.path.insert(0, str(root))
    from src.generator.ufoef_generator import generate_candidate

    rows = []
    for i in range(n_per_group):
        length = 60 + (i % 41)
        seq, _ = generate_candidate(length=length, type="B")
        rows.append({"id": f"ufoe_{i:04d}", "group": "UFoE", "sequence": seq, "length": len(seq)})
    for i in range(n_per_group):
        ref = rows[i % len(rows)]
        seq_rand = generate_random_same_composition(ref["sequence"])
        rows.append({"id": f"random_{i:04d}", "group": "Random", "sequence": seq_rand, "length": len(seq_rand)})
    # RFdiffusion 그룹: placeholder ID/sequence (실제는 RFdiffusion 출력으로 대체)
    for i in range(n_per_group):
        rows.append({"id": f"rfdiff_{i:04d}", "group": "RFdiffusion", "sequence": "", "length": 0})

    in_csv = in_dir / "phase2c_sequences.csv"
    with open(in_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "group", "sequence", "length"])
        w.writeheader()
        w.writerows(rows)

    out_csv = out_dir / "phase2c_results.csv"
    # 스텁 결과: 실제 실행 시 2a/2b와 동일 형식으로 채움
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["id", "group", "sequence", "fold_success", "final_rmsd", "energy_score",
                       "output_pdb_path", "zone_EC", "zone_TZ", "zone_BZ", "runtime_sec", "phase"],
            extrasaction="ignore",
        )
        w.writeheader()
        for r in rows:
            w.writerow({
                "id": r["id"], "group": r["group"], "sequence": r["sequence"],
                "fold_success": False, "final_rmsd": None, "energy_score": None,
                "output_pdb_path": "", "zone_EC": None, "zone_TZ": None, "zone_BZ": None,
                "runtime_sec": None, "phase": "2c",
            })
    return out_csv
