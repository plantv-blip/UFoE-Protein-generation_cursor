#!/usr/bin/env python3
"""
결과 요약 리포트 템플릿 생성.

Phase 0~5 출력이 있으면 수집하여 REPORT_Phase0-5_Results.md 를 생성.
없으면 템플릿만 채워서 저장.
"""

from pathlib import Path
from datetime import datetime
import csv

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def load_phase0_summary():
    p = DATA / "phase0_validation_results.csv"
    if not p.exists():
        return None
    with open(p, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    n = len(rows)
    all_passed = sum(1 for r in rows if str(r.get("all_passed", "")).lower() in ("true", "1"))
    return {"n": n, "all_passed": all_passed, "rate": 100 * all_passed / n if n else 0}


def load_phase3_summary():
    p = DATA / "phase3_output" / "phase3_report.csv"
    if not p.exists():
        return None
    with open(p, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    n = len(rows)
    k = sum(1 for r in rows if str(r.get("all_passed", "")).lower() in ("true", "1"))
    return {"M": n, "K": k, "K_over_M": 100 * k / n if n else 0}


def load_phase5_summary():
    p = DATA / "phase5_output" / "phase5_results.csv"
    if not p.exists():
        return None
    with open(p, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    n = len(rows)
    matched = sum(1 for r in rows if str(r.get("matched", "")).lower() in ("true", "1"))
    return {"n": n, "matched": matched, "rate": 100 * matched / n if n else 0}


def main():
    p0 = load_phase0_summary()
    p3 = load_phase3_summary()
    p5 = load_phase5_summary()

    lines = [
        "# UFoE 단백질 구조 생성 실험 — 결과 요약",
        "",
        f"**생성일:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "---",
        "",
        "## Phase 0: 자연 PDB 필터 검증",
        "",
    ]
    if p0:
        lines.append(f"- 검증 PDB 수: {p0['n']}")
        lines.append(f"- 5개 필터 모두 통과: {p0['all_passed']} ({p0['rate']:.1f}%)")
        lines.append("")
    else:
        lines.append("- (데이터 없음 — `run_phase0_validation.py` 실행 후 재생성)")
        lines.append("")

    lines.extend([
        "## Phase 2-3: 접힘 및 자기일관성",
        "",
    ])
    if p3:
        lines.append(f"- M (접힘 완료): {p3['M']}")
        lines.append(f"- K (필터 재통과): {p3['K']}")
        lines.append(f"- K/M: {p3['K_over_M']:.1f}%")
        lines.append("")
    else:
        lines.append("- (데이터 없음 — 기준 파이프라인 또는 `run_phase3.py` 실행)")
        lines.append("")

    lines.extend([
        "## Phase 5: PDB 비교 (TM-align)",
        "",
    ])
    if p5:
        lines.append(f"- 비교 구조 수: {p5['n']}")
        lines.append(f"- TM-score > 0.5 매칭: {p5['matched']} ({p5['rate']:.1f}%)")
        lines.append("")
    else:
        lines.append("- (데이터 없음 — `run_phase5.py` 실행)")
        lines.append("")

    lines.extend([
        "---",
        "",
        "상세: DESIGN.md, REPORT_Phase0-2_Summary.md, REFERENCE_PIPELINE.md 참고.",
        "",
    ])

    out = ROOT / "data" / "REPORT_Phase0-5_Results.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print("저장:", out)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
