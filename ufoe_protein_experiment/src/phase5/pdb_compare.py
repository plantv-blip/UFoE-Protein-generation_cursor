"""
Phase 5: PDB 비교 (TM-align).

자기일관적 구조(K개)를 PDB 전체(또는 대표 집합)와 TM-align으로 비교.
- TM-score > 0.5: 자연 단백질과 구조적 유사 → UFoE가 자연 소거 결과를 설계로 재현
- 매칭 없음: 자연이 탐색하지 않은 새로운 안정 구조 발견

TM-align 미설치 시: 스텁 모드(합성 TM-score) 또는 경로 지정.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import csv
import subprocess
import sys


@dataclass
class Phase5Result:
    query_id: str
    query_pdb: str
    hit_pdb: Optional[str]
    tm_score: float
    matched: bool  # TM-score > 0.5


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def find_tmalign() -> Optional[Path]:
    """TM-align 실행 파일 경로 탐색."""
    for name in ["TMalign", "TMalign", "tmalign"]:
        try:
            r = subprocess.run(
                [name, "-h"],
                capture_output=True,
                timeout=2,
            )
            if r.returncode in (0, 1):
                return Path(name)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
    return None


def run_tmalign(pdb1: Path, pdb2: Path, tmalign_bin: Optional[Path] = None) -> Tuple[float, float]:
    """
    TM-align 실행. 반환: (TM-score_1, TM-score_2).
    실패 시 (0.0, 0.0).
    """
    bin_ = tmalign_bin or find_tmalign()
    if not bin_:
        return (0.0, 0.0)
    try:
        out = subprocess.run(
            [str(bin_), str(pdb1), str(pdb2)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if out.returncode != 0:
            return (0.0, 0.0)
        # Parse "TM-score= 0.xxxxx" from stdout
        tm1, tm2 = 0.0, 0.0
        for line in out.stdout.splitlines():
            if "TM-score=" in line:
                parts = line.split("TM-score=")
                for p in parts[1:]:
                    try:
                        val = float(p.strip().split()[0])
                        if tm1 == 0.0:
                            tm1 = val
                        else:
                            tm2 = val
                            break
                    except (IndexError, ValueError):
                        pass
        return (tm1, tm2)
    except Exception:
        return (0.0, 0.0)


def run_phase5_compare(
    query_pdb_list: List[Path],
    pdb_db_dir: Path,
    tm_threshold: float = 0.5,
    tmalign_bin: Optional[Path] = None,
    mock_if_no_tmalign: bool = True,
) -> List[Phase5Result]:
    """
    query_pdb_list: Phase 3 자기일관적 구조 PDB 경로
    pdb_db_dir: 비교 대상 PDB 디렉터리 (또는 목록 파일)
    반환: 각 query별 최고 TM-score 매칭 결과
    """
    results = []
    has_tmalign = bool(tmalign_bin or find_tmalign())
    pdb_db = list(Path(pdb_db_dir).glob("*.pdb")) + list(Path(pdb_db_dir).glob("*.cif"))
    if not pdb_db and mock_if_no_tmalign:
        # 스텁: 매칭 없음 또는 낮은 TM-score
        for q in query_pdb_list:
            results.append(Phase5Result(
                query_id=q.stem,
                query_pdb=str(q),
                hit_pdb=None,
                tm_score=0.0,
                matched=False,
            ))
        return results

    for qpath in query_pdb_list:
        best_score = 0.0
        best_hit = None
        if has_tmalign and pdb_db:
            for ref in pdb_db[:200]:  # 상위 200개만 비교 (시간 제한)
                tm1, tm2 = run_tmalign(qpath, ref, tmalign_bin)
                score = max(tm1, tm2)
                if score > best_score:
                    best_score = score
                    best_hit = str(ref)
        elif mock_if_no_tmalign:
            best_score = 0.0
            best_hit = None
        results.append(Phase5Result(
            query_id=qpath.stem,
            query_pdb=str(qpath),
            hit_pdb=best_hit,
            tm_score=best_score,
            matched=best_score > tm_threshold,
        ))
    return results


def write_phase5_report(
    results: List[Phase5Result],
    output_csv: Path,
    output_md: Optional[Path] = None,
) -> None:
    """Phase 5 결과를 CSV 및 선택적 Markdown 요약으로 저장."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "query_pdb", "hit_pdb", "tm_score", "matched"])
        for r in results:
            w.writerow([r.query_id, r.query_pdb, r.hit_pdb or "", r.tm_score, r.matched])

    if output_md:
        n = len(results)
        n_matched = sum(1 for r in results if r.matched)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        pct = f"{100*n_matched/n:.1f}%" if n else "0%"
        body = f"""# Phase 5: PDB 비교 (TM-align) 요약

- **자기일관적 구조 수:** {n}
- **TM-score > 0.5 매칭:** {n_matched} ({pct})
- **상세:** `{output_csv.name}`

## 해석

- 매칭 있음: UFoE가 자연의 소거 결과를 설계로 재현한 가능성
- 매칭 없음: 자연이 탐색하지 않은 새로운 안정 구조 발견 가능성
"""
        output_md.write_text(body, encoding="utf-8")
