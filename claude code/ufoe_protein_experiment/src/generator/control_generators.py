"""
대조군 구조 생성기

Phase 2c에서 UFoE 생성 구조와 비교하기 위한 대조군을 생성한다.

대조군 유형:
  1. Random: 동일한 아미노산 조성이지만 Zone 규칙 없이 무작위 배치
  2. Shuffled: UFoE 생성 구조의 서열을 섞어서 Zone-AA 대응 파괴
  3. RFdiffusion (stub): RFdiffusion으로 생성된 구조 (외부 도구 의존)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from src.utils.constants import (
    AA_3TO1,
    STANDARD_AA,
    HYDROPHOBIC_AA,
    HYDROPHILIC_AA,
    NEUTRAL_AA,
    EC_RADIUS,
    TZ_RADIUS,
    MIN_CONTACT_DISTANCE,
)
from src.utils.pdb_parser import ProteinStructure, parse_pdb_from_coords
from src.generator.ufoef_generator import (
    GeneratedCandidate,
    generate_backbone_scaffold,
    assign_zone_residues,
    check_steric_clashes,
    _resolve_clashes,
    validate_ramachandran,
)

logger = logging.getLogger(__name__)


# =============================================================================
# 대조군 1: Random (Zone 규칙 무시)
# =============================================================================

def generate_random_candidates(
    n: int = 100,
    length: int = 60,
    seed: int | None = None,
    match_composition: dict[str, float] | None = None,
) -> list[GeneratedCandidate]:
    """Zone 규칙 없이 무작위 아미노산 배치 구조를 생성한다.

    동일한 백본 스캐폴드를 사용하되, 아미노산은 Zone과 무관하게
    균등 확률로 선택한다.

    Parameters
    ----------
    n : int — 생성할 후보 수
    length : int — 서열 길이
    seed : int, optional
    match_composition : dict, optional
        특정 아미노산 조성을 매칭 (예: UFoE 생성 구조의 평균 조성)

    Returns
    -------
    list[GeneratedCandidate]
    """
    rng = np.random.default_rng(seed)
    all_aa = sorted(STANDARD_AA)

    candidates = []

    for i in range(n):
        # 백본 생성 (Type B와 동일한 구조적 특성)
        coords, phi, psi, ss = generate_backbone_scaffold(
            length, "B", rng
        )

        # 충돌 해소
        coords = _resolve_clashes(coords, rng=rng)

        # Zone 배정 (기록용, 아미노산 선택에는 미사용)
        zones = assign_zone_residues(coords)

        # 아미노산 선택: Zone 무시, 균등 확률
        if match_composition:
            residue_names = _sample_from_composition(
                length, match_composition, rng
            )
        else:
            residue_names = [rng.choice(all_aa) for _ in range(length)]

        # Ramachandran 보정
        _, violations = validate_ramachandran(phi, psi, residue_names)
        for idx in violations:
            name = residue_names[idx]
            if name == "PRO":
                phi[idx] = rng.uniform(-80.0, -55.0)
                psi[idx] = rng.uniform(-60.0, 160.0)
            elif name != "GLY":
                phi[idx] = np.clip(phi[idx], -180.0, 0.0)

        sequence = "".join(AA_3TO1.get(name, "X") for name in residue_names)

        structure = parse_pdb_from_coords(
            pdb_id=f"ctrl_rand_{i:04d}",
            ca_coords=coords,
            residue_names=residue_names,
        )

        candidates.append(GeneratedCandidate(
            sequence=sequence,
            residue_names=residue_names,
            ca_coords=coords,
            phi_angles=phi,
            psi_angles=psi,
            zone_assignments=zones,
            protein_type="control_random",
            structure=structure,
            metadata={"control_type": "random", "index": i},
        ))

    logger.info(f"Random 대조군 {n}개 생성 완료")
    return candidates


# =============================================================================
# 대조군 2: Shuffled (Zone-AA 대응 파괴)
# =============================================================================

def generate_shuffled_candidates(
    ufoe_candidates: list[GeneratedCandidate],
    n_shuffles: int = 1,
    seed: int | None = None,
) -> list[GeneratedCandidate]:
    """UFoE 생성 구조의 서열을 셔플하여 대조군을 생성한다.

    동일한 3D 좌표를 유지하되, 아미노산 배정만 무작위로 섞는다.
    이를 통해 Zone-아미노산 대응 관계의 중요성을 검증할 수 있다.

    Parameters
    ----------
    ufoe_candidates : list[GeneratedCandidate] — 원본 UFoE 후보
    n_shuffles : int — 각 후보당 셔플 횟수
    seed : int, optional

    Returns
    -------
    list[GeneratedCandidate]
    """
    rng = np.random.default_rng(seed)
    shuffled = []

    for orig in ufoe_candidates:
        for s in range(n_shuffles):
            # 잔기 이름 셔플 (좌표는 그대로)
            names_shuffled = orig.residue_names.copy()
            rng.shuffle(names_shuffled)

            sequence = "".join(
                AA_3TO1.get(name, "X") for name in names_shuffled
            )

            structure = parse_pdb_from_coords(
                pdb_id=f"ctrl_shuf_{orig.metadata.get('index', 0):04d}_s{s}",
                ca_coords=orig.ca_coords.copy(),
                residue_names=names_shuffled,
            )

            shuffled.append(GeneratedCandidate(
                sequence=sequence,
                residue_names=names_shuffled,
                ca_coords=orig.ca_coords.copy(),
                phi_angles=orig.phi_angles.copy(),
                psi_angles=orig.psi_angles.copy(),
                zone_assignments=orig.zone_assignments.copy(),
                protein_type="control_shuffled",
                structure=structure,
                metadata={
                    "control_type": "shuffled",
                    "original_id": orig.metadata.get("index", 0),
                    "shuffle_index": s,
                },
            ))

    logger.info(
        f"Shuffled 대조군 {len(shuffled)}개 생성 완료 "
        f"(원본 {len(ufoe_candidates)}개 × {n_shuffles} 셔플)"
    )
    return shuffled


# =============================================================================
# 대조군 3: Zone-반전 (의도적 위반)
# =============================================================================

def generate_inverted_candidates(
    n: int = 100,
    length: int = 60,
    seed: int | None = None,
) -> list[GeneratedCandidate]:
    """UFoE 규칙을 의도적으로 반전한 구조를 생성한다.

    - EC (내부): 친수성 잔기 배치 (정상은 소수성)
    - BZ (외부): 소수성 잔기 배치 (정상은 친수성)
    - TZ: 변경 없음

    Parameters
    ----------
    n : int
    length : int
    seed : int, optional

    Returns
    -------
    list[GeneratedCandidate]
    """
    rng = np.random.default_rng(seed)

    hydrophobic_list = sorted(HYDROPHOBIC_AA)
    hydrophilic_list = sorted(HYDROPHILIC_AA)
    neutral_list = sorted(NEUTRAL_AA)

    candidates = []

    for i in range(n):
        coords, phi, psi, ss = generate_backbone_scaffold(
            length, "B", rng
        )
        coords = _resolve_clashes(coords, rng=rng)
        zones = assign_zone_residues(coords)

        # 반전된 아미노산 선택
        residue_names = []
        for zone in zones:
            if zone == "EC":
                # 내부에 친수성 (정상의 반대)
                cat = rng.choice(
                    ["hydrophilic", "neutral", "hydrophobic"],
                    p=[0.70, 0.20, 0.10],
                )
            elif zone == "BZ":
                # 외부에 소수성 (정상의 반대)
                cat = rng.choice(
                    ["hydrophobic", "neutral", "hydrophilic"],
                    p=[0.70, 0.20, 0.10],
                )
            else:
                cat = rng.choice(
                    ["hydrophobic", "neutral", "hydrophilic"],
                    p=[0.33, 0.34, 0.33],
                )

            if cat == "hydrophobic":
                residue_names.append(rng.choice(hydrophobic_list))
            elif cat == "neutral":
                residue_names.append(rng.choice(neutral_list))
            else:
                residue_names.append(rng.choice(hydrophilic_list))

        sequence = "".join(AA_3TO1.get(name, "X") for name in residue_names)

        structure = parse_pdb_from_coords(
            pdb_id=f"ctrl_inv_{i:04d}",
            ca_coords=coords,
            residue_names=residue_names,
        )

        candidates.append(GeneratedCandidate(
            sequence=sequence,
            residue_names=residue_names,
            ca_coords=coords,
            phi_angles=phi,
            psi_angles=psi,
            zone_assignments=zones,
            protein_type="control_inverted",
            structure=structure,
            metadata={"control_type": "inverted", "index": i},
        ))

    logger.info(f"Inverted 대조군 {n}개 생성 완료")
    return candidates


# =============================================================================
# 유틸리티
# =============================================================================

def _sample_from_composition(
    n: int,
    composition: dict[str, float],
    rng: np.random.Generator,
) -> list[str]:
    """아미노산 조성 비율에 맞춰 잔기를 샘플링한다.

    Parameters
    ----------
    n : int — 잔기 수
    composition : dict[AA_3letter, frequency]
    rng : np.random.Generator

    Returns
    -------
    list[str] — 3글자 코드
    """
    aa_names = sorted(composition.keys())
    probs = np.array([composition[aa] for aa in aa_names])
    probs = probs / probs.sum()  # 정규화

    return list(rng.choice(aa_names, size=n, p=probs))


def calculate_composition(
    candidates: list[GeneratedCandidate],
) -> dict[str, float]:
    """후보 리스트의 평균 아미노산 조성을 계산한다.

    Parameters
    ----------
    candidates : list[GeneratedCandidate]

    Returns
    -------
    dict[AA_3letter, frequency]
    """
    counts: dict[str, int] = {}
    total = 0

    for c in candidates:
        for name in c.residue_names:
            counts[name] = counts.get(name, 0) + 1
            total += 1

    return {aa: count / total for aa, count in counts.items()}
