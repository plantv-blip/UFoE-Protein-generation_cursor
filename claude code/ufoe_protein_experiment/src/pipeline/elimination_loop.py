"""
Phase 3: UFoE 자기일관성 루프 (Elimination Loop)

접힘 시뮬레이션 후 최종 구조에 대해 UFoE 5-필터를 재적용하여
자기일관성 비율(K/M)을 계산한다.

핵심 질문:
  "UFoE 조건으로 설계된 M개 서열 중,
   접힌 후에도 UFoE 조건을 유지하는 K개는 몇 개인가?"

K/M ≥ 70%이면 UFoE가 자기일관적(self-consistent) 설계 원리로 인정.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from src.filters.ufoef_filters import (
    apply_all_filters,
    all_filters_passed,
    FilterResult,
)
from src.folding.md_simulator import MDTrajectory
from src.folding.folding_metrics import FoldingMetrics
from src.utils.constants import SELF_CONSISTENCY_THRESHOLD
from src.utils.pdb_parser import parse_pdb_from_coords

logger = logging.getLogger(__name__)


# =============================================================================
# 데이터 구조
# =============================================================================

@dataclass
class EliminationResult:
    """단일 후보의 Phase 3 결과."""

    candidate_id: str

    # 접힘 성공 여부 (Phase 2)
    folding_success: bool

    # Phase 3: 접힌 구조의 UFoE 필터 결과
    post_fold_filter_results: dict[str, FilterResult] | None = None
    post_fold_all_passed: bool = False

    # 자기일관성: 설계 시 통과 + 접힘 후에도 통과
    self_consistent: bool = False

    # Zone 변화 추적
    zone_shift: dict | None = None

    metadata: dict = field(default_factory=dict)


@dataclass
class EliminationReport:
    """Phase 3 자기일관성 보고서."""

    group_name: str

    # 입력
    total_designed: int         # M: UFoE 설계 총 후보 수
    total_folded: int           # 접힘 성공 수

    # Phase 3 결과
    post_fold_passed: int       # K: 접힘 후 UFoE 통과 수
    self_consistency_ratio: float  # K / M

    # 판정
    threshold: float
    self_consistent: bool       # K/M ≥ threshold

    # 필터별 통과율 (접힘 후)
    per_filter_pass_rates: dict[str, float]

    # 개별 결과
    results: list[EliminationResult]

    metadata: dict = field(default_factory=dict)


# =============================================================================
# Phase 3 메인 로직
# =============================================================================

def run_elimination_loop(
    trajectories: list[MDTrajectory],
    folding_metrics: list[FoldingMetrics],
    residue_names_map: dict[str, list[str]],
    group_name: str = "ufoe",
    strict: bool = False,
    threshold: float = SELF_CONSISTENCY_THRESHOLD,
) -> EliminationReport:
    """Phase 3 자기일관성 루프를 실행한다.

    Parameters
    ----------
    trajectories : list[MDTrajectory]
        Phase 2에서 생성된 MD 궤적
    folding_metrics : list[FoldingMetrics]
        Phase 2 접힘 판정 결과
    residue_names_map : dict[candidate_id, list[str]]
        각 후보의 잔기 이름 (3글자 코드)
    group_name : str
    strict : bool — strict 필터 모드 사용 여부
    threshold : float — 자기일관성 판정 임계값

    Returns
    -------
    EliminationReport
    """
    logger.info(f"Phase 3 Elimination Loop 시작: {group_name}")

    # 메트릭을 ID로 인덱싱
    metrics_map = {m.candidate_id: m for m in folding_metrics}

    results = []
    total_designed = len(trajectories)

    for traj in trajectories:
        cid = traj.candidate_id
        fm = metrics_map.get(cid)

        folding_ok = fm.folding_success if fm else False

        if not folding_ok or len(traj.final_ca_coords) == 0:
            # 접힘 실패 → Phase 3 스킵
            results.append(EliminationResult(
                candidate_id=cid,
                folding_success=False,
                metadata={"reason": "folding_failed"},
            ))
            continue

        # 접힘 후 구조로 ProteinStructure 생성
        rnames = residue_names_map.get(cid, [])
        if len(rnames) != len(traj.final_ca_coords):
            logger.warning(
                f"잔기 수 불일치: {cid} | "
                f"좌표: {len(traj.final_ca_coords)}, 이름: {len(rnames)}"
            )
            results.append(EliminationResult(
                candidate_id=cid,
                folding_success=True,
                metadata={"reason": "residue_count_mismatch"},
            ))
            continue

        # 접힘 후 구조에 UFoE 필터 재적용
        post_structure = parse_pdb_from_coords(
            pdb_id=f"{cid}_post_fold",
            ca_coords=traj.final_ca_coords,
            residue_names=rnames,
        )

        filter_results = apply_all_filters(post_structure, strict=strict)
        post_all_passed = all_filters_passed(filter_results)

        # Zone 변화 추적
        zone_shift = _track_zone_shift(
            traj.initial_ca_coords, traj.final_ca_coords
        )

        results.append(EliminationResult(
            candidate_id=cid,
            folding_success=True,
            post_fold_filter_results=filter_results,
            post_fold_all_passed=post_all_passed,
            self_consistent=post_all_passed,
            zone_shift=zone_shift,
        ))

    # 통계 집계
    total_folded = sum(1 for r in results if r.folding_success)
    post_fold_passed = sum(1 for r in results if r.self_consistent)

    # K/M = 접힘 후 UFoE 통과 / 전체 설계 수
    sc_ratio = post_fold_passed / max(total_designed, 1)
    is_self_consistent = sc_ratio >= threshold

    # 필터별 통과율
    per_filter = _compute_per_filter_rates(results)

    report = EliminationReport(
        group_name=group_name,
        total_designed=total_designed,
        total_folded=total_folded,
        post_fold_passed=post_fold_passed,
        self_consistency_ratio=sc_ratio,
        threshold=threshold,
        self_consistent=is_self_consistent,
        per_filter_pass_rates=per_filter,
        results=results,
    )

    logger.info(
        f"Phase 3 완료: {group_name} | "
        f"K/M = {post_fold_passed}/{total_designed} = {sc_ratio:.1%} | "
        f"{'PASS' if is_self_consistent else 'FAIL'}"
    )

    return report


# =============================================================================
# 보조 함수
# =============================================================================

def _track_zone_shift(
    initial_coords: np.ndarray,
    final_coords: np.ndarray,
) -> dict:
    """접힘 전후의 Zone 변화를 추적한다.

    Returns
    -------
    dict with keys:
        - zone_transitions: 각 잔기의 (before_zone, after_zone) 전환
        - n_changed: Zone이 바뀐 잔기 수
        - change_ratio: 변화 비율
    """
    from src.utils.constants import EC_RADIUS, TZ_RADIUS

    def _classify(coords):
        center = coords.mean(axis=0)
        dists = np.linalg.norm(coords - center, axis=1)
        zones = []
        for d in dists:
            if d <= EC_RADIUS:
                zones.append("EC")
            elif d <= TZ_RADIUS:
                zones.append("TZ")
            else:
                zones.append("BZ")
        return zones

    if len(initial_coords) == 0 or len(final_coords) == 0:
        return {"zone_transitions": [], "n_changed": 0, "change_ratio": 0.0}

    n = min(len(initial_coords), len(final_coords))
    before_zones = _classify(initial_coords[:n])
    after_zones = _classify(final_coords[:n])

    transitions = list(zip(before_zones, after_zones))
    n_changed = sum(1 for b, a in transitions if b != a)

    return {
        "zone_transitions": transitions,
        "n_changed": n_changed,
        "change_ratio": n_changed / max(n, 1),
    }


def _compute_per_filter_rates(
    results: list[EliminationResult],
) -> dict[str, float]:
    """접힘 성공한 후보의 필터별 통과율을 계산한다."""
    filter_names = [
        "empty_center", "fibonacci_ratio", "zone_balance",
        "density_gradient", "hydrophobic_core",
    ]

    rates = {}
    folded_results = [r for r in results if r.folding_success and r.post_fold_filter_results]
    n_folded = len(folded_results)

    for fname in filter_names:
        if n_folded == 0:
            rates[fname] = 0.0
        else:
            n_pass = sum(
                1 for r in folded_results
                if r.post_fold_filter_results
                and fname in r.post_fold_filter_results
                and r.post_fold_filter_results[fname].passed
            )
            rates[fname] = n_pass / n_folded

    return rates


def print_elimination_report(report: EliminationReport) -> None:
    """Phase 3 보고서를 보기 좋게 출력한다."""
    print(f"\n{'=' * 70}")
    print(f"  Phase 3: Self-Consistency Report — {report.group_name}")
    print(f"{'=' * 70}")
    print(f"  Designed (M): {report.total_designed}")
    print(f"  Folded:       {report.total_folded}")
    print(f"  Post-fold OK: {report.post_fold_passed}")
    print(f"  K/M ratio:    {report.self_consistency_ratio:.1%}")
    print(f"  Threshold:    {report.threshold:.1%}")
    print(f"  Verdict:      {'SELF-CONSISTENT' if report.self_consistent else 'NOT SELF-CONSISTENT'}")

    print(f"\n  Per-filter pass rates (post-fold):")
    for fname, rate in report.per_filter_pass_rates.items():
        icon = "[+]" if rate >= 0.5 else "[-]"
        print(f"    {icon} {fname:<20s}: {rate:.1%}")

    print(f"{'=' * 70}\n")
