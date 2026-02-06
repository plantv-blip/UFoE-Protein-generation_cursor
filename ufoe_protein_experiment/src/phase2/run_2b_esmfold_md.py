"""
Phase 2b: 동일 서열 → ESMFold 초기 구조 예측 → MD 정제 (10ns).
입출력 규격은 Phase 2a와 동일. 실제 ESMFold/OpenMM 호출은 GPU 환경에서.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import csv


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_phase2_input(input_csv: Optional[Path] = None) -> List[Dict[str, Any]]:
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


def run_single_esmfold_md(record: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    """ESMFold 예측 → 10ns MD 정제 (스텁)."""
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
        "phase": "2b",
        "stub": True,
    }


def run_phase2b(n: int = 100, input_csv: Optional[Path] = None, output_dir: Optional[Path] = None) -> Path:
    """Phase 2b 배치. 반환: 결과 CSV 경로."""
    root = _root()
    out_dir = output_dir or (root / "data" / "phase2b_output")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "phase2b_results.csv"
    rows = load_phase2_input(input_csv)
    if not rows:
        return out_csv
    results = [run_single_esmfold_md(r, out_dir / "structures") for r in rows]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["id", "group", "sequence", "fold_success", "final_rmsd", "energy_score",
                       "output_pdb_path", "zone_EC", "zone_TZ", "zone_BZ", "runtime_sec", "phase"],
            extrasaction="ignore",
        )
        w.writeheader()
        w.writerows(results)
    return out_csv
