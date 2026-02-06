"""
접힘 성공 판정 및 메트릭 계산 모듈

MD 시뮬레이션 궤적에서 다음 메트릭을 계산한다:
  1. RMSD 수렴: 마지막 윈도우에서의 RMSD 안정성
  2. Rg 안정성: 회전 반경의 상대적 변화
  3. 에너지 수렴: 포텐셜 에너지의 안정화
  4. 2차 구조 보존율: 예측 vs 시뮬레이션 후 2차 구조 비교
  5. 접촉 순서(Contact Order): 잔기 쌍 접촉의 평균 서열 분리
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from src.utils.constants import FOLDING_CRITERIA
from src.folding.md_simulator import MDTrajectory, _calculate_rg, calculate_rmsd_aligned

logger = logging.getLogger(__name__)


# =============================================================================
# 데이터 구조
# =============================================================================

@dataclass
class FoldingMetrics:
    """개별 구조의 접힘 메트릭."""

    candidate_id: str

    # 핵심 메트릭
    rmsd_convergence: float         # 마지막 윈도우의 RMSD 표준편차 (Å)
    rmsd_final: float               # 최종 RMSD (Å)
    rg_change_ratio: float          # Rg 변화 비율 (|final - initial| / initial)
    rg_initial: float               # 초기 Rg (Å)
    rg_final: float                 # 최종 Rg (Å)
    energy_convergence: float       # PE 표준편차/평균 (마지막 윈도우)
    energy_final: float             # 최종 PE (kJ/mol)
    contact_order: float            # 상대 접촉 순서

    # 통과/실패 판정
    rmsd_converged: bool
    rg_stable: bool
    energy_converged: bool
    folding_success: bool           # 종합 판정

    # 추가 메트릭
    compactness: float              # Rg / sqrt(N) — 구상성 지표
    end_to_end_distance: float      # 양 말단 거리 (Å)

    # ESMFold 관련 (Phase 2b)
    plddt_mean: float = 0.0
    plddt_min: float = 0.0

    metadata: dict = field(default_factory=dict)


@dataclass
class FoldingReport:
    """배치 접힘 검증 보고서."""

    group_name: str                 # 'ufoe_2a', 'ufoe_2b', 'control_random', etc.
    total_count: int
    success_count: int
    success_rate: float

    # 그룹 통계
    mean_rmsd_final: float
    std_rmsd_final: float
    mean_rg_final: float
    std_rg_final: float
    mean_energy_final: float
    std_energy_final: float
    mean_contact_order: float

    # 개별 결과
    metrics: list[FoldingMetrics]

    metadata: dict = field(default_factory=dict)


# =============================================================================
# 메트릭 계산
# =============================================================================

def evaluate_folding(
    trajectory: MDTrajectory,
    criteria: dict | None = None,
    plddt_scores: np.ndarray | None = None,
) -> FoldingMetrics:
    """단일 궤적에서 접힘 메트릭을 계산하고 성공 여부를 판정한다.

    Parameters
    ----------
    trajectory : MDTrajectory
    criteria : dict, optional — 판정 기준. None이면 기본값 사용.
    plddt_scores : np.ndarray, optional — ESMFold pLDDT (Phase 2b)

    Returns
    -------
    FoldingMetrics
    """
    if criteria is None:
        criteria = FOLDING_CRITERIA

    cid = trajectory.candidate_id

    # --- 빈 궤적 처리 ---
    if not trajectory.converged or len(trajectory.timestamps_ns) == 0:
        logger.warning(f"빈 또는 실패한 궤적: {cid}")
        return _empty_metrics(cid, trajectory.error_message)

    timestamps = trajectory.timestamps_ns
    total_time = trajectory.total_time_ns

    # =========================================================================
    # 1. RMSD 수렴
    # =========================================================================
    window_ns = criteria["rmsd_convergence_window_ns"]
    window_mask = timestamps >= (total_time - window_ns)

    if window_mask.sum() < 2:
        # 윈도우에 프레임이 부족하면 전체 사용
        window_mask = np.ones(len(timestamps), dtype=bool)

    rmsd_window = trajectory.rmsd_to_initial[window_mask]
    rmsd_convergence = float(np.std(rmsd_window))
    rmsd_final = float(trajectory.rmsd_to_initial[-1])
    rmsd_converged = rmsd_convergence < criteria["rmsd_convergence_max_std"]

    # =========================================================================
    # 2. Rg 안정성
    # =========================================================================
    rg_values = trajectory.rg_values
    rg_initial = float(rg_values[0]) if len(rg_values) > 0 else 0.0
    rg_final = float(rg_values[-1]) if len(rg_values) > 0 else 0.0

    if rg_initial > 0:
        rg_change_ratio = abs(rg_final - rg_initial) / rg_initial
    else:
        rg_change_ratio = 0.0

    rg_stable = rg_change_ratio < criteria["rg_stability_max_change"]

    # =========================================================================
    # 3. 에너지 수렴
    # =========================================================================
    pe_window = trajectory.potential_energies[window_mask]
    pe_mean = float(np.mean(pe_window)) if len(pe_window) > 0 else 0.0
    pe_std = float(np.std(pe_window)) if len(pe_window) > 0 else 0.0

    if abs(pe_mean) > 1e-6:
        energy_convergence = abs(pe_std / pe_mean)
    else:
        energy_convergence = 0.0

    energy_final = float(trajectory.potential_energies[-1]) if len(
        trajectory.potential_energies
    ) > 0 else 0.0
    energy_converged = energy_convergence < criteria[
        "energy_convergence_max_std_ratio"
    ]

    # =========================================================================
    # 4. 접촉 순서 (Contact Order)
    # =========================================================================
    final_ca = trajectory.final_ca_coords
    if len(final_ca) > 0:
        contact_order = _calculate_contact_order(final_ca)
    else:
        contact_order = 0.0

    # =========================================================================
    # 5. 구상성 (Compactness)
    # =========================================================================
    n = trajectory.n_residues
    compactness = rg_final / np.sqrt(max(n, 1))

    # =========================================================================
    # 6. 양 말단 거리
    # =========================================================================
    if len(final_ca) >= 2:
        end_to_end = float(np.linalg.norm(final_ca[-1] - final_ca[0]))
    else:
        end_to_end = 0.0

    # =========================================================================
    # 7. 종합 판정
    # =========================================================================
    # 최소 2/3 기준 통과 시 접힘 성공으로 판정
    criteria_passed = sum([rmsd_converged, rg_stable, energy_converged])
    folding_success = criteria_passed >= 2

    # 접촉 순서가 너무 높으면 (비구상적) 실패
    if contact_order > criteria["contact_order_max"]:
        folding_success = False

    # pLDDT
    plddt_mean = float(np.mean(plddt_scores)) if plddt_scores is not None else 0.0
    plddt_min = float(np.min(plddt_scores)) if plddt_scores is not None else 0.0

    return FoldingMetrics(
        candidate_id=cid,
        rmsd_convergence=rmsd_convergence,
        rmsd_final=rmsd_final,
        rg_change_ratio=rg_change_ratio,
        rg_initial=rg_initial,
        rg_final=rg_final,
        energy_convergence=energy_convergence,
        energy_final=energy_final,
        contact_order=contact_order,
        rmsd_converged=rmsd_converged,
        rg_stable=rg_stable,
        energy_converged=energy_converged,
        folding_success=folding_success,
        compactness=compactness,
        end_to_end_distance=end_to_end,
        plddt_mean=plddt_mean,
        plddt_min=plddt_min,
        metadata={
            "criteria_passed": criteria_passed,
            "total_criteria": 3,
        },
    )


def evaluate_batch(
    trajectories: list[MDTrajectory],
    group_name: str = "unknown",
    criteria: dict | None = None,
    plddt_map: dict[str, np.ndarray] | None = None,
) -> FoldingReport:
    """배치 궤적의 접힘 메트릭을 평가하고 보고서를 생성한다.

    Parameters
    ----------
    trajectories : list[MDTrajectory]
    group_name : str
    criteria : dict, optional
    plddt_map : dict[candidate_id, plddt_scores], optional

    Returns
    -------
    FoldingReport
    """
    metrics_list = []

    for traj in trajectories:
        plddt = None
        if plddt_map and traj.candidate_id in plddt_map:
            plddt = plddt_map[traj.candidate_id]

        m = evaluate_folding(traj, criteria, plddt)
        metrics_list.append(m)

    total = len(metrics_list)
    success_count = sum(1 for m in metrics_list if m.folding_success)
    success_rate = success_count / max(total, 1)

    # 통계 (성공한 것만)
    successful = [m for m in metrics_list if m.folding_success]
    if successful:
        mean_rmsd = float(np.mean([m.rmsd_final for m in successful]))
        std_rmsd = float(np.std([m.rmsd_final for m in successful]))
        mean_rg = float(np.mean([m.rg_final for m in successful]))
        std_rg = float(np.std([m.rg_final for m in successful]))
        mean_energy = float(np.mean([m.energy_final for m in successful]))
        std_energy = float(np.std([m.energy_final for m in successful]))
        mean_co = float(np.mean([m.contact_order for m in successful]))
    else:
        mean_rmsd = std_rmsd = mean_rg = std_rg = 0.0
        mean_energy = std_energy = mean_co = 0.0

    return FoldingReport(
        group_name=group_name,
        total_count=total,
        success_count=success_count,
        success_rate=success_rate,
        mean_rmsd_final=mean_rmsd,
        std_rmsd_final=std_rmsd,
        mean_rg_final=mean_rg,
        std_rg_final=std_rg,
        mean_energy_final=mean_energy,
        std_energy_final=std_energy,
        mean_contact_order=mean_co,
        metrics=metrics_list,
    )


# =============================================================================
# 보조 함수
# =============================================================================

def _calculate_contact_order(
    ca_coords: np.ndarray,
    contact_threshold: float = 8.0,
) -> float:
    """상대 접촉 순서(Relative Contact Order)를 계산한다.

    CO = (1 / (L * N_contacts)) * Σ |i - j|
    여기서 |i-j|는 접촉 쌍의 서열 분리, L은 서열 길이.

    높은 CO는 긴 범위 접촉이 많은 복잡한 접힘을 의미한다.

    Parameters
    ----------
    ca_coords : np.ndarray of shape (N, 3)
    contact_threshold : float — 접촉 판정 거리 (Å)

    Returns
    -------
    float — 상대 접촉 순서 (0~1 범위)
    """
    n = len(ca_coords)
    if n < 3:
        return 0.0

    total_sep = 0.0
    n_contacts = 0

    for i in range(n):
        for j in range(i + 3, n):  # 최소 3잔기 분리
            dist = np.linalg.norm(ca_coords[i] - ca_coords[j])
            if dist < contact_threshold:
                total_sep += abs(j - i)
                n_contacts += 1

    if n_contacts == 0:
        return 0.0

    return total_sep / (n * n_contacts)


def _empty_metrics(candidate_id: str, error_msg: str = "") -> FoldingMetrics:
    """빈/실패 궤적에 대한 기본 메트릭."""
    return FoldingMetrics(
        candidate_id=candidate_id,
        rmsd_convergence=float("inf"),
        rmsd_final=float("inf"),
        rg_change_ratio=float("inf"),
        rg_initial=0.0,
        rg_final=0.0,
        energy_convergence=float("inf"),
        energy_final=0.0,
        contact_order=0.0,
        rmsd_converged=False,
        rg_stable=False,
        energy_converged=False,
        folding_success=False,
        compactness=0.0,
        end_to_end_distance=0.0,
        metadata={"error": error_msg},
    )


def print_folding_report(report: FoldingReport) -> None:
    """접힘 보고서를 보기 좋게 출력한다."""
    print(f"\n{'=' * 70}")
    print(f"  Folding Report: {report.group_name}")
    print(f"{'=' * 70}")
    print(f"  Total: {report.total_count} | "
          f"Success: {report.success_count} | "
          f"Rate: {report.success_rate:.1%}")
    print(f"  Mean RMSD (final): {report.mean_rmsd_final:.2f} "
          f"± {report.std_rmsd_final:.2f} Å")
    print(f"  Mean Rg (final):   {report.mean_rg_final:.2f} "
          f"± {report.std_rg_final:.2f} Å")
    print(f"  Mean PE (final):   {report.mean_energy_final:.1f} "
          f"± {report.std_energy_final:.1f} kJ/mol")
    print(f"  Mean Contact Order: {report.mean_contact_order:.3f}")
    print(f"{'=' * 70}")

    # 개별 결과 요약
    print(f"\n  {'ID':<20s} {'RMSD':<8s} {'Rg':<8s} {'CO':<8s} {'Result'}")
    print(f"  {'-'*60}")
    for m in report.metrics:
        result_str = "PASS" if m.folding_success else "FAIL"
        icon = "[+]" if m.folding_success else "[-]"
        print(f"  {icon} {m.candidate_id:<18s} "
              f"{m.rmsd_final:<8.2f} "
              f"{m.rg_final:<8.2f} "
              f"{m.contact_order:<8.3f} "
              f"{result_str}")
    print()
