"""
검증 지표 수집: Phase 2 결과 + Phase 3 K/M 요약.
접힘 성공률, 에너지, zone 유지, 자기일관성 비율 등.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import csv

ROOT = Path(__file__).resolve().parents[1]


def load_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def summarize_phase2(results_csv: Path) -> Dict[str, Any]:
    """Phase 2 결과 요약: 그룹별 접힘 성공률, 평균 에너지 등."""
    rows = load_csv(results_csv)
    if not rows:
        return {"n": 0, "by_group": {}}
    by_group: Dict[str, List[Dict]] = {}
    for r in rows:
        g = r.get("group", "unknown")
        by_group.setdefault(g, []).append(r)
    out = {"n": len(rows), "by_group": {}}
    for g, lst in by_group.items():
        success = sum(1 for r in lst if str(r.get("fold_success", "")).lower() in ("true", "1"))
        energies = []
        for r in lst:
            try:
                e = r.get("energy_score")
                if e is not None and str(e).strip():
                    energies.append(float(e))
            except (TypeError, ValueError):
                pass
        out["by_group"][g] = {
            "n": len(lst),
            "fold_success_rate": (success / len(lst)) if lst else 0,
            "mean_energy": (sum(energies) / len(energies)) if energies else None,
        }
    return out


def summarize_phase3(report_csv: Path) -> Dict[str, Any]:
    """Phase 3 보고서에서 K/M 및 탈락 필터 요약."""
    rows = load_csv(report_csv)
    if not rows:
        return {"M": 0, "K": 0, "K_over_M": 0.0, "failed_by_filter": {}}
    M = len(rows)
    K = sum(1 for r in rows if str(r.get("all_passed", "")).lower() in ("true", "1"))
    failed = {}
    for r in rows:
        for k, v in r.items():
            if k.endswith("_passed") and str(v).lower() in ("false", "0"):
                name = k.replace("_passed", "")
                failed[name] = failed.get(name, 0) + 1
    return {
        "M": M,
        "K": K,
        "K_over_M": (K / M) if M else 0.0,
        "failed_by_filter": failed,
    }


def main():
    data = ROOT / "data"
    out = {
        "phase2a": summarize_phase2(data / "phase2a_output" / "phase2a_results.csv"),
        "phase2b": summarize_phase2(data / "phase2b_output" / "phase2b_results.csv"),
        "phase2c": summarize_phase2(data / "phase2c_output" / "phase2c_results.csv"),
        "phase3": summarize_phase3(data / "phase3_output" / "phase3_report.csv"),
    }
    for k, v in out.items():
        print(k, v)
    return out
