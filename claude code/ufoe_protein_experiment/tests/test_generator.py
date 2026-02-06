"""
UFoE Phase 1 서열 생성기 테스트

합성 데이터를 사용하여 생성기의 정확성을 검증한다:
  1. 백본 스캐폴드 생성
  2. Zone 분포 타겟 근접성
  3. 아미노산 배정 규칙 준수
  4. Ramachandran 허용 범위
  5. 충돌 검사
  6. 배치 생성
"""

import numpy as np
import pytest

from src.utils.constants import (
    EC_RADIUS,
    TZ_RADIUS,
    HYDROPHOBIC_AA,
    HYDROPHILIC_AA,
    NEUTRAL_AA,
    TYPE_TARGETS,
    ZONE_TOLERANCE,
    MIN_CONTACT_DISTANCE,
    AA_3TO1,
    STANDARD_AA,
)
from src.generator.ufoef_generator import (
    generate_backbone_scaffold,
    assign_zone_residues,
    select_amino_acids,
    validate_ramachandran,
    check_steric_clashes,
    UFoEGenerator,
    GeneratedCandidate,
    generate_type_b_candidates,
)


class TestBackboneScaffold:
    """백본 스캐폴드 생성 테스트."""

    def test_output_shape(self):
        """생성된 좌표 배열의 크기가 올바른지 확인."""
        n = 60
        coords, phi, psi, ss = generate_backbone_scaffold(n, "B", np.random.default_rng(42))
        assert coords.shape == (n, 3)
        assert phi.shape == (n,)
        assert psi.shape == (n,)
        assert len(ss) == n

    def test_different_seeds_different_structures(self):
        """다른 시드는 다른 구조를 생성."""
        coords1, *_ = generate_backbone_scaffold(50, "B", np.random.default_rng(1))
        coords2, *_ = generate_backbone_scaffold(50, "B", np.random.default_rng(2))
        assert not np.allclose(coords1, coords2)

    def test_secondary_structure_elements(self):
        """2차 구조 배정에 H, E, C가 포함."""
        _, _, _, ss = generate_backbone_scaffold(100, "B", np.random.default_rng(42))
        ss_set = set(ss)
        # 100 잔기면 H, E, C 모두 나올 확률이 높음
        assert len(ss_set) >= 2  # 최소 2종류

    def test_all_protein_types(self):
        """Type A, B, C 모두 생성 가능."""
        for ptype in ["A", "B", "C"]:
            coords, _, _, _ = generate_backbone_scaffold(50, ptype, np.random.default_rng(42))
            assert coords.shape == (50, 3)


class TestZoneAssignment:
    """Zone 배정 테스트."""

    def test_zones_are_valid(self):
        """모든 잔기가 EC, TZ, BZ 중 하나에 배정."""
        coords = np.random.default_rng(42).normal(0, 8, (50, 3))
        zones = assign_zone_residues(coords)
        assert all(z in ("EC", "TZ", "BZ") for z in zones)
        assert len(zones) == 50

    def test_center_residues_are_ec(self):
        """원점 근처 잔기는 EC로 분류."""
        coords = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
        zones = assign_zone_residues(coords)
        # 중심이 (1, 0, 0)이므로 모든 잔기가 EC
        for z in zones:
            assert z == "EC"

    def test_far_residues_are_bz(self):
        """원점에서 먼 잔기는 BZ로 분류."""
        coords = np.array([[20, 0, 0], [-20, 0, 0]], dtype=float)
        zones = assign_zone_residues(coords)
        # 중심은 (0, 0, 0), 모든 잔기가 20Å 거리
        assert all(z == "BZ" for z in zones)


class TestAminoAcidSelection:
    """아미노산 선택 테스트."""

    def test_output_length(self):
        """선택된 아미노산 수가 입력과 동일."""
        zones = ["EC"] * 10 + ["TZ"] * 30 + ["BZ"] * 20
        names = select_amino_acids(zones, np.random.default_rng(42))
        assert len(names) == 60

    def test_all_standard_aa(self):
        """모든 선택된 아미노산이 표준 20종 중 하나."""
        zones = ["EC"] * 50 + ["TZ"] * 30 + ["BZ"] * 20
        names = select_amino_acids(zones, np.random.default_rng(42))
        assert all(name in STANDARD_AA for name in names)

    def test_ec_enriched_hydrophobic(self):
        """EC Zone은 소수성 잔기가 많아야 한다."""
        rng = np.random.default_rng(42)
        zones = ["EC"] * 200
        names = select_amino_acids(zones, rng)
        hydrophobic_count = sum(1 for n in names if n in HYDROPHOBIC_AA)
        ratio = hydrophobic_count / len(names)
        # 기대치: ~70% 소수성, 최소 50% 이상이면 통과
        assert ratio > 0.50, f"EC hydrophobic ratio: {ratio:.2f}"

    def test_bz_enriched_hydrophilic(self):
        """BZ Zone은 친수성 잔기가 많아야 한다."""
        rng = np.random.default_rng(42)
        zones = ["BZ"] * 200
        names = select_amino_acids(zones, rng)
        hydrophilic_count = sum(1 for n in names if n in HYDROPHILIC_AA)
        ratio = hydrophilic_count / len(names)
        # 기대치: ~60% 친수성, 최소 40% 이상이면 통과
        assert ratio > 0.40, f"BZ hydrophilic ratio: {ratio:.2f}"


class TestRamachandranValidation:
    """Ramachandran 검증 테스트."""

    def test_valid_angles_pass(self):
        """허용 범위 내 각도는 통과."""
        phi = np.array([-60.0, -120.0, -70.0])
        psi = np.array([-45.0, 130.0, 150.0])
        names = ["ALA", "ALA", "ALA"]
        valid, violations = validate_ramachandran(phi, psi, names)
        assert valid is True
        assert len(violations) == 0

    def test_positive_phi_fails_for_non_gly(self):
        """양수 phi는 비-glycine에서 실패."""
        phi = np.array([60.0])  # 양수 phi — 일반 잔기에서 위반
        psi = np.array([0.0])
        names = ["ALA"]
        valid, violations = validate_ramachandran(phi, psi, names)
        assert valid is False
        assert 0 in violations

    def test_glycine_wide_range(self):
        """Glycine은 넓은 범위 허용."""
        phi = np.array([60.0])  # 양수 phi도 OK
        psi = np.array([120.0])
        names = ["GLY"]
        valid, violations = validate_ramachandran(phi, psi, names)
        assert valid is True


class TestStericClashCheck:
    """충돌 검사 테스트."""

    def test_no_clash_far_apart(self):
        """충분히 떨어진 잔기는 충돌 없음."""
        coords = np.array([
            [0, 0, 0],
            [3.8, 0, 0],
            [7.6, 0, 0],
            [11.4, 0, 0],
        ], dtype=float)
        no_clash, pairs = check_steric_clashes(coords)
        assert no_clash is True
        assert len(pairs) == 0

    def test_clash_detected(self):
        """가까운 비인접 잔기는 충돌 검출."""
        coords = np.array([
            [0, 0, 0],
            [3.8, 0, 0],
            [7.6, 0, 0],
            [0.5, 0, 0],   # 잔기 0과 매우 가까움 (비인접)
        ], dtype=float)
        no_clash, pairs = check_steric_clashes(coords, min_distance=2.0)
        assert no_clash is False
        assert (0, 3) in pairs


class TestUFoEGenerator:
    """UFoEGenerator 통합 테스트."""

    def test_generate_single_candidate(self):
        """단일 후보 생성."""
        gen = UFoEGenerator(length=50, protein_type="B", seed=42)
        candidate = gen.generate_candidate()

        assert isinstance(candidate, GeneratedCandidate)
        assert candidate.length == 50
        assert candidate.protein_type == "B"
        assert len(candidate.sequence) == 50
        assert len(candidate.residue_names) == 50
        assert candidate.ca_coords.shape == (50, 3)

    def test_sequence_is_valid(self):
        """생성된 서열이 유효한 아미노산으로 구성."""
        gen = UFoEGenerator(length=60, protein_type="B", seed=42)
        candidate = gen.generate_candidate()

        valid_aa_1letter = set(AA_3TO1.values())
        for aa in candidate.sequence:
            assert aa in valid_aa_1letter, f"Invalid amino acid: {aa}"

    def test_zone_assignments_present(self):
        """각 잔기에 Zone이 배정."""
        gen = UFoEGenerator(length=60, protein_type="B", seed=42)
        candidate = gen.generate_candidate()
        assert len(candidate.zone_assignments) == 60
        assert all(z in ("EC", "TZ", "BZ") for z in candidate.zone_assignments)

    def test_structure_attached(self):
        """ProteinStructure 객체가 첨부."""
        gen = UFoEGenerator(length=50, protein_type="B", seed=42)
        candidate = gen.generate_candidate()
        assert candidate.structure is not None
        assert candidate.structure.n_residues == 50

    def test_batch_generate(self):
        """배치 생성."""
        gen = UFoEGenerator(length=40, protein_type="B", seed=42)
        candidates = gen.batch_generate(n=5)
        assert len(candidates) == 5
        for c in candidates:
            assert isinstance(c, GeneratedCandidate)
            assert c.length == 40

    def test_type_a_and_c(self):
        """Type A, C도 생성 가능."""
        for ptype in ["A", "C"]:
            gen = UFoEGenerator(length=50, protein_type=ptype, seed=42)
            candidate = gen.generate_candidate()
            assert candidate.protein_type == ptype
            assert candidate.length == 50

    def test_convenience_function(self):
        """generate_type_b_candidates 편의 함수."""
        candidates = generate_type_b_candidates(n=3, length=40, seed=42)
        assert len(candidates) == 3
        for c in candidates:
            assert c.protein_type == "B"


class TestZoneDistributionQuality:
    """생성된 구조의 Zone 분포 품질 테스트."""

    def test_type_b_zone_distribution(self):
        """Type B 구조의 Zone 분포가 타겟에 근접.

        완벽한 일치를 요구하지는 않지만,
        각 Zone 비율이 타겟 ± 15% 이내여야 한다.
        """
        gen = UFoEGenerator(length=100, protein_type="B", seed=42)
        candidate = gen.generate_candidate()

        zones = candidate.zone_assignments
        n = len(zones)
        ec_ratio = zones.count("EC") / n
        tz_ratio = zones.count("TZ") / n
        bz_ratio = zones.count("BZ") / n

        target = TYPE_TARGETS["B"]
        tolerance = 0.15  # 15% 허용

        assert abs(ec_ratio - target["center"]) < tolerance, \
            f"EC: {ec_ratio:.2f} vs target {target['center']:.2f}"
        assert abs(tz_ratio - target["transition"]) < tolerance, \
            f"TZ: {tz_ratio:.2f} vs target {target['transition']:.2f}"
        assert abs(bz_ratio - target["boundary"]) < tolerance, \
            f"BZ: {bz_ratio:.2f} vs target {target['boundary']:.2f}"
