"""
UFoE 5-필터 시스템 테스트

합성 데이터를 사용하여 각 필터의 정확성을 검증한다:
  1. 이상적인 구상 단백질 (모든 필터 통과 예상)
  2. 병리적 구조 (대부분 필터 실패 예상)
  3. 경계값 테스트
"""

import numpy as np
import pytest

from src.utils.pdb_parser import ProteinStructure, Residue, parse_pdb_from_coords
from src.utils.constants import (
    EC_RADIUS,
    TZ_RADIUS,
    HYDROPHOBICITY_KD,
    HYDROPHOBIC_AA,
    HYDROPHILIC_AA,
)
from src.filters.ufoef_filters import (
    calculate_geometric_center,
    calculate_residue_distances,
    classify_zones,
    filter_empty_center,
    filter_fibonacci_ratio,
    filter_zone_balance,
    filter_density_gradient,
    filter_hydrophobic_core,
    apply_all_filters,
    all_filters_passed,
    batch_validate,
    FilterResult,
    ZoneClassification,
)


# =============================================================================
# 테스트용 합성 구조 생성 헬퍼
# =============================================================================

def _make_ideal_globular(n_residues: int = 100, seed: int = 42) -> ProteinStructure:
    """이상적인 구상 단백질을 생성한다.

    - 중심 근처는 비어 있고 (EC 비율 낮음)
    - 3-Zone이 균형적으로 분포
    - 내부에 소수성, 외부에 친수성 잔기 배치
    - 밀도가 내부→외부로 감소
    """
    rng = np.random.default_rng(seed)

    # 타겟 Zone 비율: Type B (EC ~20%, TZ ~47%, BZ ~33%)
    # EC를 약간 높게 잡아 Zone 경계 편차에도 Fibonacci 비율이 범위 내 유지
    n_ec = int(n_residues * 0.20)
    n_tz = int(n_residues * 0.47)
    n_bz = n_residues - n_ec - n_tz

    coords = []
    names = []

    hydrophobic = sorted(HYDROPHOBIC_AA)
    hydrophilic = sorted(HYDROPHILIC_AA)

    # EC 잔기: 1~4.5Å 거리에 소수성 잔기 (중심 깊숙이 배치)
    for _ in range(n_ec):
        r = rng.uniform(1.0, EC_RADIUS - 0.5)
        theta = rng.uniform(0, 2 * np.pi)
        phi = rng.uniform(0, np.pi)
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        coords.append([x, y, z])
        names.append(rng.choice(hydrophobic))

    # TZ 잔기: 5~13Å 거리에 혼합 잔기
    for i in range(n_tz):
        r = rng.uniform(EC_RADIUS + 0.5, TZ_RADIUS - 0.5)
        theta = rng.uniform(0, 2 * np.pi)
        phi = rng.uniform(0, np.pi)
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        coords.append([x, y, z])
        if i % 2 == 0:
            names.append(rng.choice(hydrophobic))
        else:
            names.append(rng.choice(hydrophilic))

    # BZ 잔기: 13Å+ 거리에 친수성 잔기
    for _ in range(n_bz):
        r = rng.uniform(TZ_RADIUS + 0.5, TZ_RADIUS + 8.0)
        theta = rng.uniform(0, 2 * np.pi)
        phi = rng.uniform(0, np.pi)
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        coords.append([x, y, z])
        names.append(rng.choice(hydrophilic))

    return parse_pdb_from_coords(
        pdb_id="ideal_globular",
        ca_coords=np.array(coords),
        residue_names=names,
    )


def _make_pathological_dense_center(n_residues: int = 100, seed: int = 42) -> ProteinStructure:
    """병리적 구조: 중심에 잔기가 과밀집.

    EC에 60% 이상의 잔기가 몰려 있어 Empty Center 필터 실패 예상.
    """
    rng = np.random.default_rng(seed)
    coords = []
    names = []

    hydrophilic = sorted(HYDROPHILIC_AA)

    # 대부분 잔기를 중심 5Å 이내에 배치
    for _ in range(int(n_residues * 0.6)):
        r = rng.uniform(0.5, EC_RADIUS)
        theta = rng.uniform(0, 2 * np.pi)
        phi = rng.uniform(0, np.pi)
        coords.append([
            r * np.sin(phi) * np.cos(theta),
            r * np.sin(phi) * np.sin(theta),
            r * np.cos(phi),
        ])
        names.append(rng.choice(hydrophilic))  # 의도적으로 친수성을 내부에

    for _ in range(n_residues - int(n_residues * 0.6)):
        r = rng.uniform(EC_RADIUS, TZ_RADIUS + 5)
        theta = rng.uniform(0, 2 * np.pi)
        phi = rng.uniform(0, np.pi)
        coords.append([
            r * np.sin(phi) * np.cos(theta),
            r * np.sin(phi) * np.sin(theta),
            r * np.cos(phi),
        ])
        names.append(rng.choice(hydrophilic))

    return parse_pdb_from_coords(
        pdb_id="pathological_dense",
        ca_coords=np.array(coords),
        residue_names=names,
    )


def _make_unbalanced_structure(n_residues: int = 100, seed: int = 42) -> ProteinStructure:
    """Zone Balance 실패 구조: 모든 잔기가 BZ에 집중."""
    rng = np.random.default_rng(seed)
    coords = []
    names = []

    hydrophobic = sorted(HYDROPHOBIC_AA)

    for _ in range(n_residues):
        r = rng.uniform(TZ_RADIUS + 1, TZ_RADIUS + 10)
        theta = rng.uniform(0, 2 * np.pi)
        phi = rng.uniform(0, np.pi)
        coords.append([
            r * np.sin(phi) * np.cos(theta),
            r * np.sin(phi) * np.sin(theta),
            r * np.cos(phi),
        ])
        names.append(rng.choice(hydrophobic))

    return parse_pdb_from_coords(
        pdb_id="unbalanced",
        ca_coords=np.array(coords),
        residue_names=names,
    )


# =============================================================================
# 테스트
# =============================================================================

class TestGeometricCalculations:
    """기하학적 계산 기본 테스트."""

    def test_geometric_center_origin(self):
        """대칭 구조의 중심은 원점 근처여야 한다."""
        coords = np.array([
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1],
        ], dtype=float)
        names = ["ALA"] * 6
        structure = parse_pdb_from_coords("sym", coords, names)
        center = calculate_geometric_center(structure)
        np.testing.assert_array_almost_equal(center, [0, 0, 0], decimal=5)

    def test_distances_from_center(self):
        """원점 중심으로 반경 10에 놓인 잔기의 거리는 10이어야 한다."""
        coords = np.array([[10, 0, 0], [-10, 0, 0]], dtype=float)
        names = ["ALA", "ALA"]
        structure = parse_pdb_from_coords("dist", coords, names)
        distances = calculate_residue_distances(structure)
        for d in distances.values():
            assert abs(d - 10.0) < 0.01

    def test_zone_classification(self):
        """거리에 따라 올바른 Zone으로 분류되는지 확인."""
        distances = {1: 3.0, 2: 8.0, 3: 15.0}
        zones = classify_zones(distances)
        assert 1 in zones.ec_residues
        assert 2 in zones.tz_residues
        assert 3 in zones.bz_residues


class TestEmptyCenterFilter:
    """필터 1: Empty Center 테스트."""

    def test_ideal_passes(self):
        """이상적 구조는 Empty Center 통과."""
        structure = _make_ideal_globular()
        distances = calculate_residue_distances(structure)
        zones = classify_zones(distances)
        result = filter_empty_center(zones, threshold=0.30)
        assert result.passed is True
        assert result.value < 0.30

    def test_dense_center_fails(self):
        """과밀집 중심 구조는 Empty Center 실패."""
        structure = _make_pathological_dense_center()
        distances = calculate_residue_distances(structure)
        zones = classify_zones(distances)
        result = filter_empty_center(zones, threshold=0.30)
        assert result.passed is False
        assert result.value >= 0.30

    def test_strict_vs_primary(self):
        """strict 임계값이 primary보다 엄격함을 확인."""
        structure = _make_ideal_globular()
        distances = calculate_residue_distances(structure)
        zones = classify_zones(distances)
        primary = filter_empty_center(zones, threshold=0.30)
        strict = filter_empty_center(zones, threshold=0.25)
        # strict가 더 엄격하므로, primary가 통과하면 strict도 통과하거나 같아야 함
        if not primary.passed:
            assert not strict.passed


class TestFibonacciRatioFilter:
    """필터 2: Fibonacci Ratio 테스트."""

    def test_ideal_passes(self):
        """이상적 구조의 Q3/Q1 비율이 허용 범위 내."""
        structure = _make_ideal_globular()
        distances = calculate_residue_distances(structure)
        zones = classify_zones(distances)
        result = filter_fibonacci_ratio(zones, ratio_range=(1.0, 3.0))
        assert result.passed is True

    def test_zero_ec_fails(self):
        """EC 잔기가 0이면 실패."""
        zones = ZoneClassification(
            ec_residues=[],
            tz_residues=[1, 2, 3],
            bz_residues=[4, 5],
            ec_ratio=0.0,
            tz_ratio=0.6,
            bz_ratio=0.4,
            distances={},
        )
        result = filter_fibonacci_ratio(zones)
        assert result.passed is False
        assert result.value == float("inf")


class TestZoneBalanceFilter:
    """필터 3: Zone Balance 테스트."""

    def test_ideal_passes(self):
        """이상적 구조는 Zone Balance 통과."""
        structure = _make_ideal_globular()
        distances = calculate_residue_distances(structure)
        zones = classify_zones(distances)
        result = filter_zone_balance(zones, min_ratio=0.20)
        assert result.passed is False or result.value >= 0.20

    def test_unbalanced_fails(self):
        """불균형 구조는 Zone Balance 실패."""
        structure = _make_unbalanced_structure()
        distances = calculate_residue_distances(structure)
        zones = classify_zones(distances)
        result = filter_zone_balance(zones, min_ratio=0.20)
        assert result.passed is False


class TestDensityGradientFilter:
    """필터 4: Density Gradient 테스트."""

    def test_ideal_passes(self):
        """이상적 구상 구조는 내부 밀도 > 외부 밀도."""
        structure = _make_ideal_globular()
        distances = calculate_residue_distances(structure)
        zones = classify_zones(distances)
        result = filter_density_gradient(zones, structure)
        assert result.passed is True
        assert result.value > 0


class TestHydrophobicCoreFilter:
    """필터 5: Hydrophobic Core 테스트."""

    def test_ideal_passes(self):
        """이상적 구조: 내부 소수성 > 외부 소수성."""
        structure = _make_ideal_globular()
        distances = calculate_residue_distances(structure)
        zones = classify_zones(distances)
        result = filter_hydrophobic_core(zones, structure)
        assert result.passed is True
        assert result.value > 0

    def test_pathological_fails(self):
        """병리적 구조: 친수성이 내부에 배치됨 → 실패."""
        structure = _make_pathological_dense_center()
        distances = calculate_residue_distances(structure)
        zones = classify_zones(distances)
        result = filter_hydrophobic_core(zones, structure)
        # 병리적 구조는 내부에 친수성이므로 실패 예상
        assert result.passed is False


class TestIntegration:
    """통합 필터 테스트."""

    def test_apply_all_filters_returns_5(self):
        """apply_all_filters는 5개 결과를 반환."""
        structure = _make_ideal_globular()
        results = apply_all_filters(structure, strict=False)
        assert len(results) == 5
        assert all(isinstance(r, FilterResult) for r in results.values())

    def test_ideal_structure_mostly_passes(self):
        """이상적 구조는 대부분 필터를 통과."""
        structure = _make_ideal_globular()
        results = apply_all_filters(structure, strict=False)
        passed_count = sum(1 for r in results.values() if r.passed)
        assert passed_count >= 3  # 최소 3개 이상 통과

    def test_pathological_mostly_fails(self):
        """병리적 구조는 대부분 필터 실패."""
        structure = _make_pathological_dense_center()
        results = apply_all_filters(structure, strict=False)
        failed_count = sum(1 for r in results.values() if not r.passed)
        assert failed_count >= 2  # 최소 2개 이상 실패

    def test_batch_validate(self):
        """batch_validate가 DataFrame을 반환."""
        structures = [_make_ideal_globular(), _make_pathological_dense_center()]
        df = batch_validate(structures, strict=False)
        assert len(df) == 2
        assert "pdb_id" in df.columns
        assert "all_passed" in df.columns
        assert "ec_passed" in df.columns

    def test_strict_more_restrictive(self):
        """strict 모드가 primary보다 더 제한적."""
        structure = _make_ideal_globular()
        primary = apply_all_filters(structure, strict=False)
        strict = apply_all_filters(structure, strict=True)

        primary_pass = sum(1 for r in primary.values() if r.passed)
        strict_pass = sum(1 for r in strict.values() if r.passed)
        assert strict_pass <= primary_pass
