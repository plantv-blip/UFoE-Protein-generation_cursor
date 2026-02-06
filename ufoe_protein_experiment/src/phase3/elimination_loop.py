"""
Phase 3: 소거 루프 (자기검증).
Phase 2 결과물 M개 구조에 UFoE 5-필터 재적용 → K개 자기일관적 구조.
탈락 구조 분석: 어떤 필터에서 탈락했는지.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import csv

def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_phase2_results(results_csv: Path) -> List[Dict[str, Any]]:
    """Phase 2 결과 CSV 로드. output_pdb_path 가 있는 행만 (실제 구조 존재)."""
    if not results_csv.exists():
        return []
    rows = []
    with open(results_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(dict(r))
    return rows


def run_elimination_loop(
    results_csv: Path,
    structures_dir: Optional[Path] = None,
    strict: bool = False,
) -> Dict[str, Any]:
    """
    Phase 2 결과 M개에 대해 UFoE 5-필터 재적용.
    반환: {
        "M": int,
        "K": int,
        "K_over_M": float,
        "passed_ids": [...],
        "failed_by_filter": { "empty_center": n, ... },
        "rows": [ { "id", "all_passed", "empty_center_passed", ... }, ... ]
    }
    """
    import sys
    sys.path.insert(0, str(_root()))
    from src.filters.ufoef_filters import apply_all_filters
    from src.utils.pdb_parser import load_structure

    rows = load_phase2_results(results_csv)
    if not rows:
        return {"M": 0, "K": 0, "K_over_M": 0.0, "passed_ids": [], "failed_by_filter": {}, "rows": []}

    base = results_csv.resolve().parent
    struct_dir = structures_dir or (base / "structures")
    filter_names = ["empty_center", "fibonacci_ratio", "zone_balance", "density_gradient", "hydrophobic_core"]
    failed_by_filter = {f: 0 for f in filter_names}
    passed_ids = []
    out_rows = []

    for r in rows:
        pid = r.get("id", "")
        pdb_path = r.get("output_pdb_path", "").strip() or (struct_dir / f"{pid}.pdb")
        pdb_path = Path(pdb_path)
        if not pdb_path.is_absolute():
            pdb_path = base / pdb_path
        row_out = {"id": pid, "group": r.get("group", ""), "phase": r.get("phase", "")}
        if not pdb_path.exists():
            row_out["all_passed"] = False
            row_out["error"] = "PDB not found"
            for f in filter_names:
                row_out[f"{f}_passed"] = False
            out_rows.append(row_out)
            continue
        try:
            structure = load_structure(str(pdb_path))
            res = apply_all_filters(structure, strict=strict)
            all_passed = all(v[0] for v in res.values())
            row_out["all_passed"] = all_passed
            for name, (passed, val) in res.items():
                row_out[f"{name}_passed"] = passed
                row_out[f"{name}_value"] = val
                if not passed:
                    failed_by_filter[name] = failed_by_filter.get(name, 0) + 1
            if all_passed:
                passed_ids.append(pid)
        except Exception as e:
            row_out["all_passed"] = False
            row_out["error"] = str(e)
            for f in filter_names:
                row_out[f"{f}_passed"] = False
        out_rows.append(row_out)

    M = len(rows)
    K = len(passed_ids)
    return {
        "M": M,
        "K": K,
        "K_over_M": (K / M) if M else 0.0,
        "passed_ids": passed_ids,
        "failed_by_filter": failed_by_filter,
        "rows": out_rows,
    }


def write_phase3_report(result: Dict[str, Any], output_csv: Path) -> None:
    """Phase 3 결과를 CSV로 저장."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = result.get("rows", [])
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
