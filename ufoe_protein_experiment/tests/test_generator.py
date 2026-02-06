"""
Phase 1 생성기 단위 테스트.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from src.generator.ufoef_generator import (
    generate_backbone_scaffold,
    assign_zone_residues,
    select_amino_acids,
    validate_ramachandran,
    generate_candidate,
    batch_generate,
)
from src.filters.ufoef_filters import apply_all_filters
from src.utils.constants import TYPE_B_TARGET


def test_generate_backbone_scaffold():
    coords = generate_backbone_scaffold(100, type="B")
    assert coords.shape == (100, 3)
    center = coords.mean(axis=0)
    assert np.allclose(center, 0, atol=0.1)


def test_assign_zone_residues():
    coords = generate_backbone_scaffold(100, type="B")
    zones = assign_zone_residues(coords)
    assert len(zones) == 100
    zone_names = [z[1] for z in zones]
    ec_ratio = zone_names.count("EC") / 100
    tz_ratio = zone_names.count("TZ") / 100
    bz_ratio = zone_names.count("BZ") / 100
    # Type B 목표에 근사
    assert 0.05 < ec_ratio < 0.35
    assert 0.30 < tz_ratio < 0.65
    assert 0.20 < bz_ratio < 0.50


def test_select_amino_acids():
    zones = [(i, "EC" if i < 17 else "TZ" if i < 64 else "BZ") for i in range(100)]
    seq = select_amino_acids(zones, hydrophobic_rules=True)
    assert len(seq) == 100
    assert all(c in "ARNDCEQGHILKMFPSTWYV" for c in seq)


def test_validate_ramachandran():
    # 자연스러운 연속 좌표 → violation 적음
    structure = [
        {"x": float(i), "y": 0.0, "z": 0.0, "residue_id": (i,)}
        for i in range(20)
    ]
    passed, violations = validate_ramachandran(structure)
    assert isinstance(passed, bool)
    assert isinstance(violations, list)


def test_generate_candidate():
    seq, structure = generate_candidate(length=50, type="B")
    assert len(seq) == 50
    assert len(structure) == 50
    for r in structure:
        assert "x" in r and "y" in r and "z" in r and "resname_1" in r and "residue_id" in r
    # 생성 구조에 필터 적용 가능
    results = apply_all_filters(structure, strict=False)
    assert len(results) == 5


def test_batch_generate():
    candidates = batch_generate(n=5, type="B", length=30)
    assert len(candidates) == 5
    for seq, structure in candidates:
        assert len(seq) == 30 and len(structure) == 30
