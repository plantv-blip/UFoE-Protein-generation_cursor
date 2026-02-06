"""
Phase 3 소거 루프 (자기일관성 K/M). 기준 파이프라인 API.
K/M ≥ 70% 이면 UFoE 자기일관적 설계 원리로 인정.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional

# 기존 phase3 구현 래핑
def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_phase2_results(results_csv: Path) -> List[Dict[str, Any]]:
    from ..phase3.elimination_loop import load_phase2_results as _load
    return _load(results_csv)


def run_elimination_loop(
    results_csv: Path,
    structures_dir: Optional[Path] = None,
    strict: bool = False,
) -> Dict[str, Any]:
    from ..phase3.elimination_loop import run_elimination_loop as _run
    root = _root()
    if not results_csv.is_absolute():
        results_csv = root / results_csv
    result = _run(results_csv, structures_dir=structures_dir, strict=strict)
    result["self_consistent"] = result["K_over_M"] >= 0.70
    return result


def write_phase3_report(result: Dict[str, Any], output_csv: Path) -> None:
    from ..phase3.elimination_loop import write_phase3_report as _write
    _write(result, output_csv)
