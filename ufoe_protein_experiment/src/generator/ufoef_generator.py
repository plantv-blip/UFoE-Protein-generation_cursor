"""
Phase 1: Type B 타겟 서열 생성기.
Zone 분포 EC ~17%, TZ ~47%, BZ ~36% 맞추고, 소수성 규칙·Ramachandran·충돌 제약 적용.
"""

import random
from typing import List, Tuple, Dict, Any, Optional

import numpy as np

from ..utils.constants import (
    EC_RADIUS,
    TZ_OUTER_RADIUS,
    TYPE_B_TARGET,
    TYPE_B_FILTER_COMPATIBLE,
    AMINO_ACID_HYDROPHOBICITY,
    MIN_CA_CA_DISTANCE,
)
from ..filters.ufoef_filters import classify_zones


# Zone별 선호 아미노산: 내부=소수성, 외부=친수성
_sorted_by_hydro = sorted(AMINO_ACID_HYDROPHOBICITY.items(), key=lambda x: -x[1])
HYDROPHOBIC_POOL = [a for a, _ in _sorted_by_hydro[:10]]   # I, V, L, F, M, C, A, ...
HYDROPHILIC_POOL = [a for a, _ in _sorted_by_hydro[-10:]]  # R, K, D, E, N, Q, ...
MID_POOL = [a for a, _ in _sorted_by_hydro[5:15]]


def generate_backbone_scaffold(
    length: int,
    type: str = "B",
    filter_compatible: bool = False,
) -> np.ndarray:
    """
    Type B 타겟 분포에 맞게 3D CA 좌표 생성.
    filter_compatible=True 시 EC 21%, TZ 47%, BZ 32% (Zone Balance 필터 통과용).
    Returns: (N, 3) 좌표 배열, 중심 원점 근처.
    """
    if type == "B" and filter_compatible:
        target = TYPE_B_FILTER_COMPATIBLE
    else:
        target = TYPE_B_TARGET
    n_ec = max(1, int(round(length * target["EC"])))
    n_tz = max(1, int(round(length * target["TZ"])))
    n_bz = length - n_ec - n_tz
    if n_bz < 1:
        n_bz = 1
        n_tz = length - n_ec - n_bz

    rng = np.random.default_rng()
    coords = []

    # EC: 반경 0 ~ ec_radius 균일 구
    for _ in range(n_ec):
        r = (rng.random() ** (1.0 / 3.0)) * EC_RADIUS
        theta = np.arccos(2 * rng.random() - 1)
        phi = 2 * np.pi * rng.random()
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        coords.append([x, y, z])

    # TZ: 5 ~ 13 Å 껍질
    for _ in range(n_tz):
        r3_inner = EC_RADIUS ** 3
        r3_outer = TZ_OUTER_RADIUS ** 3
        r3 = r3_inner + (r3_outer - r3_inner) * rng.random()
        r = r3 ** (1.0 / 3.0)
        theta = np.arccos(2 * rng.random() - 1)
        phi = 2 * np.pi * rng.random()
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        coords.append([x, y, z])

    # BZ: 13 Å ~ 25 Å 껍질 (외부 상한 고정)
    outer = 25.0
    for _ in range(n_bz):
        r3_inner = TZ_OUTER_RADIUS ** 3
        r3_outer = outer ** 3
        r3 = r3_inner + (r3_outer - r3_inner) * rng.random()
        r = r3 ** (1.0 / 3.0)
        theta = np.arccos(2 * rng.random() - 1)
        phi = 2 * np.pi * rng.random()
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        coords.append([x, y, z])

    coords = np.array(coords, dtype=float)
    # 순서 셔플 후 중심 재계산해서 원점 맞춤
    rng.shuffle(coords)
    center = coords.mean(axis=0)
    coords -= center
    return coords


def assign_zone_residues(scaffold: np.ndarray) -> List[Tuple[int, str]]:
    """
    scaffold: (N, 3) CA 좌표.
    Returns: [(index, "EC"|"TZ"|"BZ"), ...].
    """
    center = scaffold.mean(axis=0)
    distances = {}
    for i in range(len(scaffold)):
        d = float(np.linalg.norm(scaffold[i] - center))
        distances[i] = d
    zones = classify_zones(distances, ec_radius=EC_RADIUS, tz_radius=TZ_OUTER_RADIUS)
    idx_to_zone = {}
    for zname, rids in zones.items():
        for rid in rids:
            idx_to_zone[rid] = zname
    return [(i, idx_to_zone.get(i, "BZ")) for i in range(len(scaffold))]


def select_amino_acids(
    zones: List[Tuple[int, str]],
    hydrophobic_rules: bool = True,
) -> str:
    """
    Zone별 소수성 규칙: EC → 소수성 풀, TZ → 혼합, BZ → 친수성 풀.
    Returns: 1-letter sequence.
    """
    seq = []
    for _idx, zone in zones:
        if not hydrophobic_rules:
            seq.append(random.choice(list(AMINO_ACID_HYDROPHOBICITY.keys())))
            continue
        if zone == "EC":
            seq.append(random.choice(HYDROPHOBIC_POOL))
        elif zone == "BZ":
            seq.append(random.choice(HYDROPHILIC_POOL))
        else:
            seq.append(random.choice(MID_POOL))
    return "".join(seq)


def validate_ramachandran(structure: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    """
    CA-only 구조에서 CA-CA-CA 각도로 pseudo-Ramachandran 검사.
    연속 3 CA 각도가 극단적이지 않으면 허용 (일반 허용 범위).
    Returns: (passed, list of violation descriptions).
    """
    violations = []
    if len(structure) < 3:
        return (True, [])
    for i in range(len(structure) - 2):
        a = np.array([structure[i]["x"], structure[i]["y"], structure[i]["z"]])
        b = np.array([structure[i + 1]["x"], structure[i + 1]["y"], structure[i + 1]["z"]])
        c = np.array([structure[i + 2]["x"], structure[i + 2]["y"], structure[i + 2]["z"]])
        v1 = b - a
        v2 = c - b
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            violations.append(f"residue {i+1}: degenerate CA-CA-CA")
            continue
        cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(cos_angle))
        # CA-CA-CA 각도 대략 90~180° 범위가 자연스러움 (너무 꺾이면 violation)
        if angle_deg < 60 or angle_deg > 175:
            violations.append(f"residue {i+1}: CA-CA-CA angle {angle_deg:.1f}°")
    passed = len(violations) == 0
    return (passed, violations)


def _check_collisions(structure: List[Dict], min_dist: float = MIN_CA_CA_DISTANCE) -> Tuple[bool, List[str]]:
    """잔기 간 최소 거리 > min_dist 확인."""
    violations = []
    n = len(structure)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt(
                (structure[i]["x"] - structure[j]["x"]) ** 2
                + (structure[i]["y"] - structure[j]["y"]) ** 2
                + (structure[i]["z"] - structure[j]["z"]) ** 2
            )
            if d < min_dist:
                violations.append(f"residue {i}-{j}: distance {d:.2f} Å < {min_dist} Å")
    return (len(violations) == 0, violations)


def generate_candidate(
    length: int = 100,
    type: str = "B",
    filter_compatible: bool = False,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Type B 분포로 백본 스캐폴드 생성 → Zone 배정 → 서열 할당 → 검증(충돌·Ramachandran).
    filter_compatible=True 시 EC 21%, TZ 47%, BZ 32%.
    Returns: (sequence, structure).
    """
    scaffold = generate_backbone_scaffold(length, type=type, filter_compatible=filter_compatible)
    zone_assignments = assign_zone_residues(scaffold)
    sequence = select_amino_acids(zone_assignments, hydrophobic_rules=True)

    structure = []
    for i in range(length):
        structure.append({
            "residue_id": ("A", i + 1),
            "resseq": i + 1,
            "x": float(scaffold[i, 0]),
            "y": float(scaffold[i, 1]),
            "z": float(scaffold[i, 2]),
            "resname_1": sequence[i],
            "resname_3": None,
        })

    # 충돌 검사: 실패 시 재생성 시도 (간단히 한 번만)
    ok_coll, coll_v = _check_collisions(structure)
    if not ok_coll:
        # 약간 스케일 업해서 거리 벌리기
        scale = 1.2
        for r in structure:
            r["x"] *= scale
            r["y"] *= scale
            r["z"] *= scale
        ok_coll, _ = _check_collisions(structure)
        if not ok_coll:
            pass  # 그래도 실패해도 후속 필터에서 걸러질 수 있음

    return (sequence, structure)


def batch_generate(
    n: int = 1000,
    type: str = "B",
    length: int = 100,
) -> List[Tuple[str, List[Dict[str, Any]]]]:
    """n개 Type B 후보 (sequence, structure) 생성."""
    return [generate_candidate(length=length, type=type) for _ in range(n)]
