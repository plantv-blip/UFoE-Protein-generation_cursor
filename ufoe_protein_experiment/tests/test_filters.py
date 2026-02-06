"""
Phase 0 필터 단위 테스트.
"""

import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest
import numpy as np

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
)
from src.utils.constants import EC_RADIUS, TZ_OUTER_RADIUS


def _make_residue_list(n, center=(0, 0, 0), radius=10.0):
    """테스트용 residue dict 리스트: 구 표면 근처 n개."""
    rng = np.random.default_rng(42)
    residues = []
    cx, cy, cz = center
    for i in range(n):
        r = radius * (0.5 + 0.5 * rng.random())
        theta = np.arccos(2 * rng.random() - 1)
        phi = 2 * np.pi * rng.random()
        x = cx + r * np.sin(theta) * np.cos(phi)
        y = cy + r * np.sin(theta) * np.sin(phi)
        z = cz + r * np.cos(theta)
        residues.append({
            "residue_id": ("A", i + 1),
            "resseq": i + 1,
            "x": float(x), "y": float(y), "z": float(z),
            "resname_1": "A",
            "resname_3": "ALA",
        })
    return residues


def test_calculate_geometric_center():
    residues = _make_residue_list(10, center=(1, 2, 3))
    cx, cy, cz = calculate_geometric_center(residues)
    assert abs(cx - 1) < 2 and abs(cy - 2) < 2 and abs(cz - 3) < 2


def test_calculate_residue_distances():
    residues = _make_residue_list(5, center=(0, 0, 0))
    center = calculate_geometric_center(residues)
    dists = calculate_residue_distances(residues, center)
    assert len(dists) == 5
    for rid, d in dists.items():
        assert d >= 0


def test_classify_zones():
    # 거리 0~4 → EC, 5~12 → TZ, 13+ → BZ
    distances = {("A", 1): 2, ("A", 2): 7, ("A", 3): 15}
    zones = classify_zones(distances, ec_radius=5, tz_radius=13)
    assert len(zones["EC"]) == 1
    assert len(zones["TZ"]) == 1
    assert len(zones["BZ"]) == 1


def test_filter_empty_center():
    # EC 2/10 = 20% < 30% → pass primary
    zones = {"EC": [1, 2], "TZ": [3, 4, 5, 6], "BZ": [7, 8, 9, 10]}
    passed, ratio = filter_empty_center(zones, threshold=0.30)
    assert passed is True
    assert abs(ratio - 0.2) < 1e-6
    passed_strict, _ = filter_empty_center(zones, threshold=0.25)
    assert passed_strict is True


def test_filter_fibonacci_ratio():
    # 균일 분포에 가깝게 하면 Q3/Q1 비율이 특정 범위 안에 들어갈 수 있음
    distances = {i: 5.0 + i for i in range(20)}
    passed, ratio = filter_fibonacci_ratio(distances, range_vals=(1.0, 3.0))
    assert isinstance(passed, bool)
    assert ratio > 0


def test_filter_zone_balance():
    zones = {"EC": list(range(25)), "TZ": list(range(25, 75)), "BZ": list(range(75, 100))}
    passed, ratios = filter_zone_balance(zones, min_ratio=0.20)
    assert passed is True
    assert ratios["EC"] == 0.25 and ratios["TZ"] == 0.50 and ratios["BZ"] == 0.25


def test_apply_all_filters():
    # 큰 구에 잔기 골고루 배치 → 여러 필터 통과 가능
    residues = _make_residue_list(100, center=(0, 0, 0), radius=15)
    # EC 비율 낮게: 반경을 크게 해서 대부분 TZ/BZ에
    results = apply_all_filters(residues, strict=False)
    assert "empty_center" in results
    assert "fibonacci_ratio" in results
    assert "zone_balance" in results
    assert "density_gradient" in results
    assert "hydrophobic_core" in results
    for name, (passed, val) in results.items():
        assert isinstance(passed, bool)


def test_apply_all_filters_strict():
    residues = _make_residue_list(100, center=(0, 0, 0), radius=15)
    results = apply_all_filters(residues, strict=True)
    assert all(isinstance(r[0], bool) for r in results.values())
