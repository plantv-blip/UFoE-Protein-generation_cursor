"""
UFoE 5-필터 시스템

특허의 5개 기하학적 제약 조건을 Python으로 구현한다.
각 필터는 primary(특허 값)와 strict(실험 설계 값) 두 버전의 임계값을 지원한다.

필터 목록:
  1. Empty Center: 중심 5Å 내 잔기 비율 제한
  2. Fibonacci Ratio: Q3/Q1(Boundary/Center) Zone 경계 비율
  3. Zone Balance: 세 Zone의 최소 점유율 보장
  4. Density Gradient: 내부→외부 밀도 감소 확인
  5. Hydrophobic Core: 내부 소수성 > 외부 소수성

사용 예:
    structure = parse_pdb_file("1ubq.pdb")
    results = apply_all_filters(structure, strict=False)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from src.utils.constants import (
    EC_RADIUS,
    FILTER_THRESHOLDS,
    TZ_RADIUS,
)
from src.utils.pdb_parser import ProteinStructure


# =============================================================================
# 데이터 구조
# =============================================================================

@dataclass
class FilterResult:
    """단일 필터의 결과."""
    name: str
    passed: bool
    value: float          # 계산된 수치 (비율, 기울기 등)
    threshold: float | tuple  # 적용된 임계값
    detail: dict | None = None


@dataclass
class ZoneClassification:
    """3-Zone 분류 결과."""
    ec_residues: list[int]   # Empty Center 잔기 ID 목록
    tz_residues: list[int]   # Transition Zone
    bz_residues: list[int]   # Boundary Zone
    ec_ratio: float
    tz_ratio: float
    bz_ratio: float
    distances: dict[int, float]


# =============================================================================
# 기하학적 계산
# =============================================================================

def calculate_geometric_center(structure: ProteinStructure) -> np.ndarray:
    """Cα 좌표의 기하학적 중심을 계산한다.

    Returns
    -------
    np.ndarray of shape (3,)
    """
    return structure.geometric_center


def calculate_residue_distances(
    structure: ProteinStructure,
    center: np.ndarray | None = None,
) -> dict[int, float]:
    """각 잔기의 기하학적 중심으로부터의 거리를 계산한다.

    Parameters
    ----------
    structure : ProteinStructure
    center : np.ndarray, optional
        외부에서 지정한 중심. None이면 구조의 기하학적 중심 사용.

    Returns
    -------
    dict[residue_id, distance]
    """
    if center is None:
        center = structure.geometric_center
    return {r.id: r.distance_to(center) for r in structure.residues}


def classify_zones(
    distances: dict[int, float],
    ec_radius: float = EC_RADIUS,
    tz_radius: float = TZ_RADIUS,
) -> ZoneClassification:
    """잔기를 3-Zone으로 분류한다.

    Zone 정의:
      - EC (Empty Center): distance ≤ ec_radius (5Å)
      - TZ (Transition Zone): ec_radius < distance ≤ tz_radius (5~13Å)
      - BZ (Boundary Zone): distance > tz_radius (13Å+)

    Parameters
    ----------
    distances : dict[residue_id, distance]
    ec_radius : float
    tz_radius : float

    Returns
    -------
    ZoneClassification
    """
    ec, tz, bz = [], [], []
    total = len(distances)

    for rid, dist in distances.items():
        if dist <= ec_radius:
            ec.append(rid)
        elif dist <= tz_radius:
            tz.append(rid)
        else:
            bz.append(rid)

    if total == 0:
        return ZoneClassification(ec, tz, bz, 0, 0, 0, distances)

    return ZoneClassification(
        ec_residues=ec,
        tz_residues=tz,
        bz_residues=bz,
        ec_ratio=len(ec) / total,
        tz_ratio=len(tz) / total,
        bz_ratio=len(bz) / total,
        distances=distances,
    )


# =============================================================================
# 5개 필터 구현
# =============================================================================

def filter_empty_center(
    zones: ZoneClassification,
    threshold: float = 0.30,
) -> FilterResult:
    """필터 1: Empty Center — 중심 5Å 내 잔기 비율이 임계값 미만인지 확인.

    자연 단백질은 기하학적 중심 근처가 비어 있는 경향이 있다.
    병리적 구조는 중심에 잔기가 과밀집되어 있다.

    Parameters
    ----------
    zones : ZoneClassification
    threshold : float
        EC 비율 상한. primary=0.30, strict=0.25

    Returns
    -------
    FilterResult
    """
    ratio = zones.ec_ratio
    passed = ratio < threshold

    return FilterResult(
        name="empty_center",
        passed=passed,
        value=ratio,
        threshold=threshold,
        detail={"n_ec_residues": len(zones.ec_residues)},
    )


def filter_fibonacci_ratio(
    zones: ZoneClassification,
    ratio_range: tuple[float, float] = (1.0, 3.0),
) -> FilterResult:
    """필터 2: Fibonacci Ratio — Q3/Q1 비율이 허용 범위 내인지 확인.

    Zone 경계 비율(Boundary/Center)이 황금비 근방에 있는지 검증한다.
    자연 단백질은 Fibonacci Shell 구조를 따르는 경향이 있다.

    Q3/Q1 = BZ 잔기 수 / EC 잔기 수

    EC 잔기가 0이면 비율을 무한대로 간주하여 실패 처리한다.

    Parameters
    ----------
    zones : ZoneClassification
    ratio_range : (min, max)

    Returns
    -------
    FilterResult
    """
    n_ec = len(zones.ec_residues)
    n_bz = len(zones.bz_residues)

    if n_ec == 0:
        # EC에 잔기가 전혀 없는 경우
        # 비율을 계산할 수 없지만, Empty Center 조건은 만족하는 극단적 경우
        # 비율을 inf로 처리
        ratio = float("inf")
        passed = False
    else:
        ratio = n_bz / n_ec
        passed = ratio_range[0] <= ratio <= ratio_range[1]

    return FilterResult(
        name="fibonacci_ratio",
        passed=passed,
        value=ratio,
        threshold=ratio_range,
        detail={"n_ec": n_ec, "n_bz": n_bz},
    )


def filter_zone_balance(
    zones: ZoneClassification,
    min_ratio: float = 0.20,
) -> FilterResult:
    """필터 3: Zone Balance — 세 Zone 중 어느 것도 최소 비율 이하가 아닌지 확인.

    min(EC%, TZ%, BZ%) ≥ min_ratio

    자연 단백질은 세 Zone이 균형 잡혀 있다.
    특정 Zone이 극단적으로 비어있거나 과밀집된 구조는 비자연적이다.

    Parameters
    ----------
    zones : ZoneClassification
    min_ratio : float

    Returns
    -------
    FilterResult
    """
    ratios = {
        "ec": zones.ec_ratio,
        "tz": zones.tz_ratio,
        "bz": zones.bz_ratio,
    }
    min_val = min(ratios.values())
    passed = min_val >= min_ratio

    return FilterResult(
        name="zone_balance",
        passed=passed,
        value=min_val,
        threshold=min_ratio,
        detail=ratios,
    )


def filter_density_gradient(
    zones: ZoneClassification,
    structure: ProteinStructure,
) -> FilterResult:
    """필터 4: Density Gradient — 내부에서 외부로 잔기 밀도가 감소하는지 확인.

    각 Zone의 밀도 = Zone 내 잔기 수 / Zone 부피
    Zone 부피는 구 껍질(spherical shell) 부피로 근사:
      - EC: (4/3)π r_ec³
      - TZ: (4/3)π (r_tz³ - r_ec³)
      - BZ: (4/3)π (r_max³ - r_tz³)  여기서 r_max는 가장 먼 잔기 거리

    조건: density_ec ≥ density_tz ≥ density_bz (단조 감소)
    값: (density_ec - density_bz) / density_ec  (양수여야 통과)

    Parameters
    ----------
    zones : ZoneClassification
    structure : ProteinStructure

    Returns
    -------
    FilterResult
    """
    distances = zones.distances
    if not distances:
        return FilterResult("density_gradient", False, 0.0, 0.0)

    r_max = max(distances.values())
    r_ec = EC_RADIUS
    r_tz = TZ_RADIUS

    # r_max가 TZ 내에 있으면 BZ가 비어있는 것이므로 BZ 부피를 작은 값으로 설정
    if r_max <= r_tz:
        r_max = r_tz + 1.0  # 최소 1Å 바깥

    vol_ec = (4 / 3) * np.pi * r_ec ** 3
    vol_tz = (4 / 3) * np.pi * (r_tz ** 3 - r_ec ** 3)
    vol_bz = (4 / 3) * np.pi * (r_max ** 3 - r_tz ** 3)

    n_ec = len(zones.ec_residues)
    n_tz = len(zones.tz_residues)
    n_bz = len(zones.bz_residues)

    # 부피가 0이 되는 것 방지
    density_ec = n_ec / max(vol_ec, 1e-6)
    density_tz = n_tz / max(vol_tz, 1e-6)
    density_bz = n_bz / max(vol_bz, 1e-6)

    # 단조 감소 확인
    monotone_decrease = density_ec >= density_tz >= density_bz

    # 그래디언트 값: 정규화된 감소율
    if density_ec > 0:
        gradient = (density_ec - density_bz) / density_ec
    else:
        gradient = 0.0

    return FilterResult(
        name="density_gradient",
        passed=monotone_decrease and gradient > 0,
        value=gradient,
        threshold=0.0,  # 양수이면 통과
        detail={
            "density_ec": density_ec,
            "density_tz": density_tz,
            "density_bz": density_bz,
            "r_max": r_max,
        },
    )


def filter_hydrophobic_core(
    zones: ZoneClassification,
    structure: ProteinStructure,
) -> FilterResult:
    """필터 5: Hydrophobic Core — 내부 평균 소수성이 외부보다 높은지 확인.

    자연 단백질은 소수성 잔기가 내부에, 친수성 잔기가 외부에 위치한다.
    이 조건이 위반되면 접힘이 불안정하다.

    조건: mean_hydrophobicity(EC + TZ_inner) > mean_hydrophobicity(BZ)
    값: 내부 평균 소수성 - 외부 평균 소수성 (양수여야 통과)

    Parameters
    ----------
    zones : ZoneClassification
    structure : ProteinStructure

    Returns
    -------
    FilterResult
    """
    residue_map = {r.id: r for r in structure.residues}

    # 내부 잔기 (EC + TZ)의 소수성
    inner_ids = set(zones.ec_residues) | set(zones.tz_residues)
    outer_ids = set(zones.bz_residues)

    inner_hydro = [residue_map[rid].hydrophobicity for rid in inner_ids if rid in residue_map]
    outer_hydro = [residue_map[rid].hydrophobicity for rid in outer_ids if rid in residue_map]

    if not inner_hydro or not outer_hydro:
        return FilterResult(
            name="hydrophobic_core",
            passed=False,
            value=0.0,
            threshold=0.0,
            detail={"n_inner": len(inner_hydro), "n_outer": len(outer_hydro)},
        )

    mean_inner = float(np.mean(inner_hydro))
    mean_outer = float(np.mean(outer_hydro))
    diff = mean_inner - mean_outer

    return FilterResult(
        name="hydrophobic_core",
        passed=diff > 0,
        value=diff,
        threshold=0.0,  # 양수이면 통과
        detail={
            "mean_inner_hydrophobicity": mean_inner,
            "mean_outer_hydrophobicity": mean_outer,
        },
    )


# =============================================================================
# 통합 필터 적용
# =============================================================================

def apply_all_filters(
    structure: ProteinStructure,
    strict: bool = False,
) -> dict[str, FilterResult]:
    """5개 필터를 모두 적용한다.

    Parameters
    ----------
    structure : ProteinStructure
    strict : bool
        True이면 strict(실험 설계) 임계값 사용, False이면 primary(특허) 사용.

    Returns
    -------
    dict[filter_name, FilterResult]
    """
    mode = "strict" if strict else "primary"
    thresholds = FILTER_THRESHOLDS[mode]

    # 1. 기하학적 중심 및 거리 계산
    center = calculate_geometric_center(structure)
    distances = calculate_residue_distances(structure, center)

    # 2. Zone 분류
    zones = classify_zones(distances)

    # 3. 5개 필터 적용
    results = {}

    results["empty_center"] = filter_empty_center(
        zones, threshold=thresholds["empty_center_max_ratio"]
    )
    results["fibonacci_ratio"] = filter_fibonacci_ratio(
        zones, ratio_range=thresholds["fibonacci_ratio_range"]
    )
    results["zone_balance"] = filter_zone_balance(
        zones, min_ratio=thresholds["zone_balance_min_ratio"]
    )
    results["density_gradient"] = filter_density_gradient(zones, structure)
    results["hydrophobic_core"] = filter_hydrophobic_core(zones, structure)

    return results


def all_filters_passed(results: dict[str, FilterResult]) -> bool:
    """5개 필터가 모두 통과했는지 확인한다."""
    return all(r.passed for r in results.values())


def summarize_results(results: dict[str, FilterResult]) -> dict:
    """필터 결과를 요약 딕셔너리로 변환한다."""
    summary = {}
    for name, result in results.items():
        summary[name] = {
            "passed": result.passed,
            "value": result.value,
            "threshold": result.threshold,
        }
    summary["all_passed"] = all_filters_passed(results)
    return summary


# =============================================================================
# 배치 검증
# =============================================================================

def batch_validate(
    structures: list[ProteinStructure],
    strict: bool = False,
) -> pd.DataFrame:
    """여러 구조에 대해 필터를 배치로 적용하고 결과를 DataFrame으로 반환한다.

    Parameters
    ----------
    structures : list[ProteinStructure]
    strict : bool

    Returns
    -------
    pd.DataFrame
        컬럼: pdb_id, n_residues, ec_passed, ec_value, fib_passed, fib_value,
              zb_passed, zb_value, dg_passed, dg_value, hc_passed, hc_value,
              all_passed
    """
    rows = []
    for structure in structures:
        results = apply_all_filters(structure, strict=strict)

        row = {
            "pdb_id": structure.pdb_id,
            "n_residues": structure.n_residues,
        }
        for name, result in results.items():
            prefix = {
                "empty_center": "ec",
                "fibonacci_ratio": "fib",
                "zone_balance": "zb",
                "density_gradient": "dg",
                "hydrophobic_core": "hc",
            }[name]
            row[f"{prefix}_passed"] = result.passed
            row[f"{prefix}_value"] = result.value

        row["all_passed"] = all_filters_passed(results)
        rows.append(row)

    return pd.DataFrame(rows)


def print_filter_report(
    structure: ProteinStructure,
    strict: bool = False,
) -> None:
    """필터 결과를 보기 좋게 출력한다."""
    mode = "STRICT" if strict else "PRIMARY"
    results = apply_all_filters(structure, strict=strict)

    print(f"\n{'='*60}")
    print(f"  UFoE 5-Filter Report [{mode}]")
    print(f"  PDB: {structure.pdb_id} | Residues: {structure.n_residues}")
    print(f"{'='*60}")

    for name, result in results.items():
        status = "PASS" if result.passed else "FAIL"
        icon = "[+]" if result.passed else "[-]"
        value_str = f"{result.value:.4f}" if isinstance(result.value, float) else str(result.value)
        print(f"  {icon} {name:20s} | {status} | value={value_str}")

    overall = all_filters_passed(results)
    overall_icon = "[+]" if overall else "[-]"
    print(f"{'='*60}")
    print(f"  {overall_icon} OVERALL: {'ALL PASSED' if overall else 'FAILED'}")
    print(f"{'='*60}\n")
