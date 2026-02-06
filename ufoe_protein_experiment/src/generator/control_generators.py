"""
대조군 3종: Random(Zone 규칙 무시), Shuffled(서열 셔플), Inverted(소수성/친수성 반전).
"""

import random
from typing import List, Tuple
from ..utils.constants import AMINO_ACID_HYDROPHOBICITY


def generate_random_sequence(length: int) -> str:
    """완전 무작위 1-letter 서열 (Zone 규칙 무시)."""
    pool = list(AMINO_ACID_HYDROPHOBICITY.keys())
    return "".join(random.choices(pool, k=length))


def generate_shuffled_sequence(sequence: str) -> str:
    """동일 조성, 순서만 셔플."""
    chars = list(sequence)
    random.shuffle(chars)
    return "".join(chars)


def generate_inverted_sequence(sequence: str) -> str:
    """소수성/친수성 반전: 내부→친수성, 외부→소수성에 가깝게 (서열 반전 + 극성 반전)."""
    sorted_by_hydro = sorted(AMINO_ACID_HYDROPHOBICITY.items(), key=lambda x: -x[1])
    hydro = [a for a, _ in sorted_by_hydro[:10]]
    hydrophilic = [a for a, _ in sorted_by_hydro[-10:]]
    mapping = {}
    for a in hydro:
        mapping[a] = random.choice(hydrophilic)
    for a in hydrophilic:
        mapping[a] = random.choice(hydro)
    mid = [a for a, _ in sorted_by_hydro[5:15]]
    for a in mid:
        if a not in mapping:
            mapping[a] = random.choice(mid)
    inv = []
    for c in sequence:
        inv.append(mapping.get(c, random.choice(list(AMINO_ACID_HYDROPHOBICITY.keys()))))
    return "".join(inv)


def generate_control_group(
    reference_sequences: List[Tuple[str, str]],
    group_type: str,
    n_per_group: int = 100,
) -> List[Tuple[str, str, str]]:
    """
    reference_sequences = [(id, sequence), ...]
    group_type: "Random" | "Shuffled" | "Inverted"
    Returns: [(id, group, sequence), ...]
    """
    out = []
    for i, (rid, seq) in enumerate(reference_sequences):
        if i >= n_per_group:
            break
        if group_type == "Random":
            new_seq = generate_random_sequence(len(seq))
        elif group_type == "Shuffled":
            new_seq = generate_shuffled_sequence(seq)
        else:
            new_seq = generate_inverted_sequence(seq)
        out.append((f"{group_type}_{rid}", group_type, new_seq))
    return out
