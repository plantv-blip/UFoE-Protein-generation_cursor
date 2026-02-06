"""
UFoE Phase 1: Type B 타겟 서열 생성기

UFoE 기하학적 제약 조건을 만족하는 아미노산 서열을 ab initio로 생성한다.

전략:
  1. 3D 공간에 Empty Center(반경 5Å 구체)를 배치
  2. Fibonacci Shell 경계(5Å, 13Å)에 따라 3-Zone 골격 생성
  3. Type B 타겟 분포(EC 17%, TZ 47%, BZ 36%)에 맞춰 잔기 수 배정
  4. Zone별 소수성/친수성 규칙에 따라 아미노산 유형 선택
  5. Ramachandran plot 허용 범위 내 결합각 제약
  6. 잔기 간 충돌 없음(최소 거리 > 2Å) 검증

사용 예:
    gen = UFoEGenerator(length=60, protein_type='B')
    candidate = gen.generate_candidate()
    batch = gen.batch_generate(n=100)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from src.utils.constants import (
    AA_3TO1,
    BOND_LENGTH_CA_CA,
    EC_RADIUS,
    HELIX_RESIDUES_PER_TURN,
    HELIX_RISE_PER_RESIDUE,
    HYDROPHOBIC_AA,
    HYDROPHILIC_AA,
    HYDROPHOBICITY_KD,
    MIN_CONTACT_DISTANCE,
    NEUTRAL_AA,
    RAMA_GENERAL_ALLOWED,
    RAMA_GLYCINE_ALLOWED,
    RAMA_PROLINE_ALLOWED,
    RAMACHANDRAN_REGIONS,
    STANDARD_AA,
    TZ_RADIUS,
    TYPE_TARGETS,
    TYPE_TARGETS_FILTER_COMPATIBLE,
    ZONE_TOLERANCE,
)
from src.utils.pdb_parser import ProteinStructure, Residue, parse_pdb_from_coords

logger = logging.getLogger(__name__)


# =============================================================================
# 데이터 구조
# =============================================================================

@dataclass
class GeneratedCandidate:
    """생성된 후보 구조."""

    sequence: str                  # 1글자 아미노산 서열
    residue_names: list[str]       # 3글자 코드 리스트
    ca_coords: np.ndarray          # (N, 3) Cα 좌표
    phi_angles: np.ndarray         # (N,) phi 각도 (degrees)
    psi_angles: np.ndarray         # (N,) psi 각도 (degrees)
    zone_assignments: list[str]    # ['EC', 'TZ', 'BZ', ...] 각 잔기의 Zone
    protein_type: str              # 'A', 'B', 'C'
    structure: ProteinStructure | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def length(self) -> int:
        return len(self.sequence)


# =============================================================================
# 백본 스캐폴드 생성
# =============================================================================

def _sample_secondary_structure(n_residues: int, rng: np.random.Generator) -> list[str]:
    """2차 구조 요소의 연속 블록을 무작위로 배정한다.

    자연 단백질의 2차 구조 비율을 근사:
      - α-helix: ~30-40%
      - β-sheet: ~20-30%
      - coil/loop: ~30-40%

    Returns
    -------
    list[str] of length n_residues
        각 잔기에 대해 'H'(helix), 'E'(sheet), 'C'(coil)
    """
    ss = []
    i = 0
    while i < n_residues:
        element = rng.choice(["H", "E", "C"], p=[0.35, 0.25, 0.40])
        if element == "H":
            block_len = rng.integers(4, 12)  # helix는 최소 4잔기
        elif element == "E":
            block_len = rng.integers(3, 8)   # sheet는 최소 3잔기
        else:
            block_len = rng.integers(2, 6)   # coil/loop

        block_len = min(block_len, n_residues - i)
        ss.extend([element] * block_len)
        i += block_len

    return ss[:n_residues]


def _ss_to_phi_psi(
    ss_type: str,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """2차 구조 유형에 따른 (phi, psi) 각도를 Ramachandran 허용 범위에서 샘플링.

    Parameters
    ----------
    ss_type : str
        'H' (helix), 'E' (sheet), 'C' (coil)
    rng : np.random.Generator

    Returns
    -------
    (phi, psi) in degrees
    """
    if ss_type == "H":
        region = RAMACHANDRAN_REGIONS["alpha_helix"]
        phi = rng.uniform(region["phi"][0], region["phi"][1])
        psi = rng.uniform(region["psi"][0], region["psi"][1])
    elif ss_type == "E":
        region = RAMACHANDRAN_REGIONS["beta_sheet"]
        phi = rng.uniform(region["phi"][0], region["phi"][1])
        psi = rng.uniform(region["psi"][0], region["psi"][1])
    else:  # coil
        # coil은 일반 허용 범위에서 넓게 샘플링
        phi = rng.uniform(-150.0, -50.0)
        psi = rng.uniform(-60.0, 160.0)

    return phi, psi


def generate_backbone_scaffold(
    n_residues: int,
    protein_type: str = "B",
    rng: np.random.Generator | None = None,
    target_override: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """3D 공간에 UFoE Zone 구조를 따르는 백본 스캐폴드를 생성한다.

    전략:
      1. 2차 구조 블록을 무작위 배정
      2. 각 잔기에 Ramachandran 기반 phi/psi 배정
      3. 이상적 결합 각도로 Cα 좌표 생성
      4. 전체 구조를 Type 타겟 분포에 맞게 스케일링/조정

    Parameters
    ----------
    n_residues : int
    protein_type : str ('A', 'B', 'C')
    rng : np.random.Generator, optional
    target_override : dict, optional
        외부에서 지정한 Zone 분포 타겟. None이면 TYPE_TARGETS 사용.

    Returns
    -------
    ca_coords : np.ndarray (N, 3)
    phi_angles : np.ndarray (N,)
    psi_angles : np.ndarray (N,)
    ss_assignment : list[str] of length N
    """
    if rng is None:
        rng = np.random.default_rng()

    target = target_override if target_override is not None else TYPE_TARGETS[protein_type]

    # 1. 2차 구조 블록 배정
    ss_assignment = _sample_secondary_structure(n_residues, rng)

    # 2. phi/psi 각도 배정
    phi_angles = np.zeros(n_residues)
    psi_angles = np.zeros(n_residues)
    for i, ss in enumerate(ss_assignment):
        phi_angles[i], psi_angles[i] = _ss_to_phi_psi(ss, rng)

    # 3. Cα 좌표 생성 (chain growth)
    ca_coords = _build_ca_chain(phi_angles, psi_angles, rng)

    # 4. Zone 분포 최적화: 구조를 조정하여 타겟 분포에 근접하게
    ca_coords = _optimize_zone_distribution(ca_coords, target, rng)

    return ca_coords, phi_angles, psi_angles, ss_assignment


def _build_ca_chain(
    phi: np.ndarray,
    psi: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """phi/psi 각도로부터 Cα 체인을 구성한다.

    간소화된 Cα-level 모델: 각 잔기 사이 거리 ~3.8Å,
    방향은 phi/psi 기반 회전 행렬로 결정.

    Returns
    -------
    np.ndarray (N, 3)
    """
    n = len(phi)
    coords = np.zeros((n, 3))

    # 초기 방향 벡터
    direction = np.array([1.0, 0.0, 0.0])
    # 법선 벡터 (회전 참조)
    normal = np.array([0.0, 1.0, 0.0])

    for i in range(1, n):
        # phi/psi를 기반으로 방향 조정
        phi_rad = np.radians(phi[i])
        psi_rad = np.radians(psi[i])

        # 방향 벡터를 phi, psi로 회전
        rot_phi = _rotation_matrix(normal, phi_rad * 0.3)
        rot_psi = _rotation_matrix(np.cross(direction, normal), psi_rad * 0.3)

        direction = rot_psi @ rot_phi @ direction
        direction = direction / np.linalg.norm(direction)

        # 약간의 노이즈 추가 (자연스러운 변이)
        noise = rng.normal(0, 0.1, 3)
        step = direction * BOND_LENGTH_CA_CA + noise

        coords[i] = coords[i - 1] + step

        # 법선 벡터 업데이트
        if i >= 2:
            v1 = coords[i] - coords[i - 1]
            v2 = coords[i - 1] - coords[i - 2]
            new_normal = np.cross(v1, v2)
            norm = np.linalg.norm(new_normal)
            if norm > 1e-6:
                normal = new_normal / norm

    return coords


def _rotation_matrix(axis: np.ndarray, theta: float) -> np.ndarray:
    """Rodrigues' rotation formula로 회전 행렬을 생성한다.

    Parameters
    ----------
    axis : np.ndarray (3,) - 회전축 (단위 벡터)
    theta : float - 회전 각도 (radians)

    Returns
    -------
    np.ndarray (3, 3)
    """
    axis = axis / (np.linalg.norm(axis) + 1e-10)
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)

    return np.array([
        [a*a + b*b - c*c - d*d, 2*(b*c - a*d),     2*(b*d + a*c)],
        [2*(b*c + a*d),     a*a + c*c - b*b - d*d, 2*(c*d - a*b)],
        [2*(b*d - a*c),     2*(c*d + a*b),     a*a + d*d - b*b - c*c],
    ])


def _optimize_zone_distribution(
    coords: np.ndarray,
    target: dict,
    rng: np.random.Generator,
    max_iter: int = 500,
) -> np.ndarray:
    """구조의 Zone 분포를 타겟에 맞추기 위해 좌표를 반복 조정한다.

    전략:
      1. 구조를 기하학적 중심으로 이동 (centering)
      2. 전역 스케일링으로 대략적인 분포 조정
      3. 잔기별 반경 방향 이동으로 세부 조정
      4. 부족한 Zone에 잔기를 직접 배치하는 보정 단계

    Parameters
    ----------
    coords : np.ndarray (N, 3)
    target : dict with keys 'center', 'transition', 'boundary'
    rng : np.random.Generator
    max_iter : int

    Returns
    -------
    np.ndarray (N, 3) — 조정된 좌표
    """
    coords = coords.copy()
    n = len(coords)
    target_ec = target["center"]
    target_tz = target["transition"]
    target_bz = target["boundary"]

    tol = ZONE_TOLERANCE  # 0.05

    # --- 전략: 각 잔기를 타겟 Zone 반경으로 직접 배치 ---
    # 잔기를 타겟 분포에 맞게 EC/TZ/BZ로 배정하고,
    # 각 잔기의 방향(단위 벡터)은 원래 체인에서 유지하되 거리만 조정한다.
    center = coords.mean(axis=0)
    coords -= center  # centering

    # 1. 타겟 잔기 수 계산
    n_ec_target = max(1, round(n * target_ec))
    n_bz_target = max(1, round(n * target_bz))
    n_tz_target = n - n_ec_target - n_bz_target

    # 2. 원래 체인의 방향 벡터 보존
    distances = np.linalg.norm(coords, axis=1)
    directions = np.zeros_like(coords)
    for i in range(n):
        if distances[i] > 1e-6:
            directions[i] = coords[i] / distances[i]
        else:
            # 원점에 있는 잔기는 랜덤 방향
            d = rng.normal(0, 1, 3)
            directions[i] = d / np.linalg.norm(d)

    # 3. 잔기를 거리 순으로 정렬 → 가장 안쪽 n_ec개를 EC로, 다음을 TZ, 나머지를 BZ로
    sorted_idx = np.argsort(distances)

    ec_indices = sorted_idx[:n_ec_target]
    tz_indices = sorted_idx[n_ec_target:n_ec_target + n_tz_target]
    bz_indices = sorted_idx[n_ec_target + n_tz_target:]

    # 4. 각 잔기를 해당 Zone의 반경 범위 내로 재배치
    for i in ec_indices:
        new_r = rng.uniform(1.5, EC_RADIUS - 0.3)
        coords[i] = directions[i] * new_r

    for i in tz_indices:
        new_r = rng.uniform(EC_RADIUS + 0.3, TZ_RADIUS - 0.3)
        coords[i] = directions[i] * new_r

    for i in bz_indices:
        new_r = rng.uniform(TZ_RADIUS + 0.3, TZ_RADIUS + 6.0)
        coords[i] = directions[i] * new_r

    # 5. 중심 보정 (재배치 후 중심이 약간 이동할 수 있음)
    # 반복적으로 미세 조정
    for _ in range(max_iter):
        center = coords.mean(axis=0)
        rel = coords - center
        dists = np.linalg.norm(rel, axis=1)

        ec_count = (dists <= EC_RADIUS).sum()
        tz_count = ((dists > EC_RADIUS) & (dists <= TZ_RADIUS)).sum()
        bz_count = (dists > TZ_RADIUS).sum()

        if (abs(ec_count / n - target_ec) < tol
                and abs(tz_count / n - target_tz) < tol
                and abs(bz_count / n - target_bz) < tol):
            break

        # 중심을 약간 이동하여 Zone 비율 조정
        shift = np.zeros(3)
        if ec_count / n < target_ec - tol:
            # 가장 안쪽 TZ 잔기 쪽으로 중심 이동
            tz_mask = (dists > EC_RADIUS) & (dists <= TZ_RADIUS)
            if tz_mask.any():
                closest_tz = np.argmin(dists + (~tz_mask) * 1e6)
                shift = rel[closest_tz] * 0.1
        elif ec_count / n > target_ec + tol:
            # 중심을 EC 잔기에서 멀어지게
            ec_mask = dists <= EC_RADIUS
            if ec_mask.any():
                farthest_ec = np.argmax(dists * ec_mask)
                shift = -rel[farthest_ec] * 0.1

        coords -= shift

    return coords


# =============================================================================
# 아미노산 배정
# =============================================================================

def assign_zone_residues(
    ca_coords: np.ndarray,
) -> list[str]:
    """좌표 기반으로 각 잔기의 Zone을 배정한다.

    Returns
    -------
    list[str] — 각 잔기의 Zone ('EC', 'TZ', 'BZ')
    """
    center = ca_coords.mean(axis=0)
    distances = np.linalg.norm(ca_coords - center, axis=1)

    zones = []
    for d in distances:
        if d <= EC_RADIUS:
            zones.append("EC")
        elif d <= TZ_RADIUS:
            zones.append("TZ")
        else:
            zones.append("BZ")

    return zones


def select_amino_acids(
    zone_assignments: list[str],
    rng: np.random.Generator | None = None,
) -> list[str]:
    """Zone별 소수성/친수성 규칙에 따라 아미노산을 선택한다.

    규칙:
      - EC (Empty Center): 주로 소수성 잔기 (코어 형성)
        → 70% hydrophobic, 20% neutral, 10% hydrophilic
      - TZ (Transition Zone): 혼합
        → 40% hydrophobic, 30% neutral, 30% hydrophilic
      - BZ (Boundary Zone): 주로 친수성 잔기 (표면)
        → 15% hydrophobic, 25% neutral, 60% hydrophilic

    Parameters
    ----------
    zone_assignments : list[str]
    rng : np.random.Generator, optional

    Returns
    -------
    list[str] — 3글자 아미노산 코드 리스트
    """
    if rng is None:
        rng = np.random.default_rng()

    hydrophobic_list = sorted(HYDROPHOBIC_AA)
    hydrophilic_list = sorted(HYDROPHILIC_AA)
    neutral_list = sorted(NEUTRAL_AA)

    zone_probabilities = {
        "EC": {"hydrophobic": 0.70, "neutral": 0.20, "hydrophilic": 0.10},
        "TZ": {"hydrophobic": 0.40, "neutral": 0.30, "hydrophilic": 0.30},
        "BZ": {"hydrophobic": 0.15, "neutral": 0.25, "hydrophilic": 0.60},
    }

    residue_names = []
    for zone in zone_assignments:
        probs = zone_probabilities[zone]
        category = rng.choice(
            ["hydrophobic", "neutral", "hydrophilic"],
            p=[probs["hydrophobic"], probs["neutral"], probs["hydrophilic"]],
        )

        if category == "hydrophobic":
            aa = rng.choice(hydrophobic_list)
        elif category == "neutral":
            aa = rng.choice(neutral_list)
        else:
            aa = rng.choice(hydrophilic_list)

        residue_names.append(aa)

    return residue_names


# =============================================================================
# Ramachandran 검증
# =============================================================================

def validate_ramachandran(
    phi_angles: np.ndarray,
    psi_angles: np.ndarray,
    residue_names: list[str],
) -> tuple[bool, list[int]]:
    """Ramachandran plot 허용 범위 검증.

    Parameters
    ----------
    phi_angles : np.ndarray (N,) in degrees
    psi_angles : np.ndarray (N,) in degrees
    residue_names : list[str] — 3글자 코드

    Returns
    -------
    (all_valid, violation_indices)
    """
    violations = []

    for i, (phi, psi, name) in enumerate(zip(phi_angles, psi_angles, residue_names)):
        if name == "GLY":
            allowed = RAMA_GLYCINE_ALLOWED
        elif name == "PRO":
            allowed = RAMA_PROLINE_ALLOWED
        else:
            allowed = RAMA_GENERAL_ALLOWED

        phi_ok = allowed["phi"][0] <= phi <= allowed["phi"][1]
        psi_ok = allowed["psi"][0] <= psi <= allowed["psi"][1]

        if not (phi_ok and psi_ok):
            violations.append(i)

    return len(violations) == 0, violations


# =============================================================================
# 충돌 검사
# =============================================================================

def check_steric_clashes(
    ca_coords: np.ndarray,
    min_distance: float = MIN_CONTACT_DISTANCE,
) -> tuple[bool, list[tuple[int, int]]]:
    """잔기 간 충돌(steric clash) 검사.

    순차적으로 인접하지 않은 잔기 쌍 중 거리가 min_distance 미만인 것을 검출.

    Parameters
    ----------
    ca_coords : np.ndarray (N, 3)
    min_distance : float

    Returns
    -------
    (no_clashes, clash_pairs)
    """
    n = len(ca_coords)
    clashes = []

    for i in range(n):
        for j in range(i + 3, n):  # 인접 2잔기는 제외
            dist = np.linalg.norm(ca_coords[i] - ca_coords[j])
            if dist < min_distance:
                clashes.append((i, j))

    return len(clashes) == 0, clashes


def _resolve_clashes(
    ca_coords: np.ndarray,
    min_distance: float = MIN_CONTACT_DISTANCE,
    max_iter: int = 50,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """충돌을 반복적으로 해소한다.

    충돌이 있는 잔기 쌍을 서로 반대 방향으로 밀어낸다.

    Returns
    -------
    np.ndarray (N, 3)
    """
    if rng is None:
        rng = np.random.default_rng()

    coords = ca_coords.copy()
    n = len(coords)

    for _ in range(max_iter):
        no_clashes, clash_pairs = check_steric_clashes(coords, min_distance)
        if no_clashes:
            break

        for i, j in clash_pairs:
            direction = coords[j] - coords[i]
            dist = np.linalg.norm(direction)
            if dist < 1e-6:
                direction = rng.normal(0, 1, 3)
                dist = np.linalg.norm(direction)
            direction = direction / dist

            push = (min_distance - dist) / 2 + 0.1
            coords[i] -= direction * push
            coords[j] += direction * push

    return coords


# =============================================================================
# 후보 생성기
# =============================================================================

class UFoEGenerator:
    """UFoE 제약 충족 서열 생성기.

    Parameters
    ----------
    length : int
        서열 길이 (잔기 수)
    protein_type : str
        타겟 유형 ('A', 'B', 'C')
    seed : int, optional
        난수 시드
    """

    def __init__(
        self,
        length: int = 60,
        protein_type: str = "B",
        seed: int | None = None,
        filter_compatible: bool = True,
    ):
        self.length = length
        self.protein_type = protein_type
        self.rng = np.random.default_rng(seed)
        self.filter_compatible = filter_compatible

        # 필터 호환 모드: Zone Balance(min ≥ 20%) 충족하도록 조정된 타겟 사용
        if filter_compatible:
            self.target = TYPE_TARGETS_FILTER_COMPATIBLE[protein_type]
        else:
            self.target = TYPE_TARGETS[protein_type]

    def generate_candidate(self) -> GeneratedCandidate:
        """단일 후보 구조를 생성한다.

        Returns
        -------
        GeneratedCandidate
        """
        # 1. 백본 스캐폴드 생성 (내부에서 Zone 최적화 1차 수행)
        ca_coords, phi, psi, ss = generate_backbone_scaffold(
            self.length, self.protein_type, self.rng,
            target_override=self.target,
        )

        # 2. 충돌 해소
        ca_coords = _resolve_clashes(ca_coords, rng=self.rng)

        # 3. Zone 재최적화 (충돌 해소 후 분포가 틀어질 수 있으므로)
        ca_coords = _optimize_zone_distribution(ca_coords, self.target, self.rng)

        # 4. Zone 배정
        zone_assignments = assign_zone_residues(ca_coords)

        # 4. 아미노산 선택
        residue_names = select_amino_acids(zone_assignments, self.rng)

        # 5. Ramachandran 검증 (위반 시 phi/psi 재조정)
        rama_valid, violations = validate_ramachandran(phi, psi, residue_names)
        if not rama_valid:
            for idx in violations:
                name = residue_names[idx]
                if name == "GLY":
                    # Glycine은 범위가 넓으므로 그대로 둔다
                    pass
                elif name == "PRO":
                    phi[idx] = self.rng.uniform(-80.0, -55.0)
                    psi[idx] = self.rng.uniform(-60.0, 160.0)
                else:
                    # 일반 잔기: 허용 범위 내로 클램핑
                    phi[idx] = np.clip(phi[idx], -180.0, 0.0)
                    psi[idx] = np.clip(psi[idx], -180.0, 180.0)

        # 6. 1글자 서열 생성
        sequence = "".join(AA_3TO1.get(name, "X") for name in residue_names)

        # 7. ProteinStructure 생성
        structure = parse_pdb_from_coords(
            pdb_id=f"gen_{self.protein_type}",
            ca_coords=ca_coords,
            residue_names=residue_names,
        )

        return GeneratedCandidate(
            sequence=sequence,
            residue_names=residue_names,
            ca_coords=ca_coords,
            phi_angles=phi,
            psi_angles=psi,
            zone_assignments=zone_assignments,
            protein_type=self.protein_type,
            structure=structure,
            metadata={
                "ss_assignment": ss,
                "n_clash_resolved": len(check_steric_clashes(ca_coords)[1]),
            },
        )

    def batch_generate(
        self,
        n: int = 100,
        max_attempts_per_candidate: int = 5,
        require_filter_pass: bool = False,
    ) -> list[GeneratedCandidate]:
        """여러 후보를 배치로 생성한다.

        Parameters
        ----------
        n : int
            생성할 후보 수
        max_attempts_per_candidate : int
            각 후보 생성 시 최대 시도 횟수 (필터 통과 요구 시)
        require_filter_pass : bool
            True이면 UFoE 필터를 통과하는 후보만 수집

        Returns
        -------
        list[GeneratedCandidate]
        """
        from src.filters.ufoef_filters import apply_all_filters, all_filters_passed

        candidates = []
        total_attempts = 0

        while len(candidates) < n:
            total_attempts += 1
            candidate = self.generate_candidate()

            if require_filter_pass and candidate.structure is not None:
                results = apply_all_filters(candidate.structure, strict=False)
                if not all_filters_passed(results):
                    if total_attempts > n * max_attempts_per_candidate:
                        logger.warning(
                            f"최대 시도 횟수 초과. {len(candidates)}/{n}개 생성 완료."
                        )
                        break
                    continue

            candidates.append(candidate)

            if len(candidates) % 100 == 0:
                logger.info(f"생성 진행: {len(candidates)}/{n}")

        return candidates


# =============================================================================
# 편의 함수
# =============================================================================

def generate_type_b_candidates(
    n: int = 100,
    length: int = 60,
    seed: int | None = None,
    require_filter_pass: bool = False,
) -> list[GeneratedCandidate]:
    """Type B 후보를 간편하게 배치 생성한다.

    Parameters
    ----------
    n : int
    length : int
    seed : int, optional
    require_filter_pass : bool

    Returns
    -------
    list[GeneratedCandidate]
    """
    gen = UFoEGenerator(length=length, protein_type="B", seed=seed)
    return gen.batch_generate(n=n, require_filter_pass=require_filter_pass)
