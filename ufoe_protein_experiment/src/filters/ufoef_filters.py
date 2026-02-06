"""
Phase 0: UFoE 5-필터 Python 구현.
Primary(특허) / Strict(실험설계) 두 버전 지원.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Union, Any

import numpy as np

from ..utils.pdb_parser import structure_to_residue_list, get_hydrophobicity
from ..utils.constants import (
    EC_RADIUS,
    TZ_OUTER_RADIUS,
    EMPTY_CENTER_PRIMARY,
    EMPTY_CENTER_STRICT,
    FIBONACCI_RANGE_PRIMARY,
    FIBONACCI_RANGE_STRICT,
    ZONE_BALANCE_MIN_RATIO,
)

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def _get_residues(structure) -> List[Dict[str, Any]]:
    """Structure(BioPython) 또는 residue dict 리스트 → 동일한 residue 리스트."""
    if isinstance(structure, list) and structure and isinstance(structure[0], dict):
        return structure
    return structure_to_residue_list(structure)


def calculate_geometric_center(structure) -> Tuple[float, float, float]:
    """구조의 기하학적 중심 (CA 좌표 평균)."""
    residues = _get_residues(structure)
    if not residues:
        return (0.0, 0.0, 0.0)
    xs = [r["x"] for r in residues]
    ys = [r["y"] for r in residues]
    zs = [r["z"] for r in residues]
    return (float(np.mean(xs)), float(np.mean(ys)), float(np.mean(zs)))


def calculate_residue_distances(
    structure,
    center: Tuple[float, float, float],
) -> Dict[Any, float]:
    """잔기별 중심까지 거리. dict[residue_id, distance]."""
    residues = _get_residues(structure)
    cx, cy, cz = center
    out = {}
    for r in residues:
        dx = r["x"] - cx
        dy = r["y"] - cy
        dz = r["z"] - cz
        d = np.sqrt(dx * dx + dy * dy + dz * dz)
        out[r["residue_id"]] = float(d)
    return out


def classify_zones(
    distances: Dict[Any, float],
    ec_radius: float = EC_RADIUS,
    tz_radius: float = TZ_OUTER_RADIUS,
) -> Dict[str, List[Any]]:
    """
    EC: d < ec_radius, TZ: ec_radius <= d < tz_radius, BZ: d >= tz_radius.
    Returns: dict["EC"|"TZ"|"BZ", list of residue_id].
    """
    zones = {"EC": [], "TZ": [], "BZ": []}
    for rid, d in distances.items():
        if d < ec_radius:
            zones["EC"].append(rid)
        elif d < tz_radius:
            zones["TZ"].append(rid)
        else:
            zones["BZ"].append(rid)
    return zones


def filter_empty_center(
    zones: Dict[str, List],
    threshold: float = EMPTY_CENTER_PRIMARY,
) -> Tuple[bool, float]:
    """
    Empty Center: 중심 5Å 내 잔기 비율 < threshold.
    Primary 30%, Strict 25%.
    """
    total = sum(len(v) for v in zones.values())
    if total == 0:
        return (True, 0.0)
    ec_ratio = len(zones["EC"]) / total
    passed = ec_ratio < threshold
    return (passed, ec_ratio)


def filter_fibonacci_ratio(
    distances: Dict[Any, float],
    range_vals: Tuple[float, float] = FIBONACCI_RANGE_PRIMARY,
) -> Tuple[bool, float]:
    """
    Fibonacci Shell: 거리 분포의 Q3/Q1 ∈ [low, high].
    Primary [1.0, 3.0], Strict [1.2, 2.5].
    """
    if len(distances) < 4:
        return (True, 1.0)
    vals = np.array(list(distances.values()))
    q1 = float(np.percentile(vals, 25))
    q3 = float(np.percentile(vals, 75))
    if q1 <= 0:
        q1 = 1e-6
    ratio = q3 / q1
    low, high = range_vals
    passed = low <= ratio <= high
    return (passed, ratio)


def filter_zone_balance(
    zones: Dict[str, List],
    min_ratio: float = ZONE_BALANCE_MIN_RATIO,
) -> Tuple[bool, Dict[str, float]]:
    """
    3-Zone Balance: min(EC, TZ, BZ) 비율 >= min_ratio (20%).
    """
    total = sum(len(v) for v in zones.values())
    if total == 0:
        return (False, {"EC": 0, "TZ": 0, "BZ": 0})
    ratios = {z: len(zones[z]) / total for z in ["EC", "TZ", "BZ"]}
    min_val = min(ratios.values())
    passed = min_val >= min_ratio
    return (passed, ratios)


def filter_density_gradient(
    zones: Dict[str, List],
    structure,
) -> Tuple[bool, float]:
    """
    Density Gradient: 내부→외부 감소. EC 밀도 > TZ > BZ.
    볼륨 대비 잔기 수로 밀도 근사. 구 껍질 부피로 정규화.
    """
    residues = _get_residues(structure)
    n = len(residues)
    if n < 3:
        return (True, 0.0)
    # 구 껍질 부피: EC 0~5Å, TZ 5~13Å, BZ 13~R_max
    vol_ec = (4.0 / 3.0) * np.pi * (EC_RADIUS ** 3)
    vol_tz = (4.0 / 3.0) * np.pi * (TZ_OUTER_RADIUS ** 3 - EC_RADIUS ** 3)
    # BZ: 반경 상한을 거리 최대값으로
    dists = [np.sqrt((r["x"] - np.mean([x["x"] for x in residues]))**2 +
                     (r["y"] - np.mean([x["y"] for x in residues]))**2 +
                     (r["z"] - np.mean([x["z"] for x in residues]))**2) for r in residues]
    r_max = max(dists) + 1e-6
    vol_bz = (4.0 / 3.0) * np.pi * (r_max ** 3 - TZ_OUTER_RADIUS ** 3)
    vol_bz = max(vol_bz, 1e-6)
    n_ec, n_tz, n_bz = len(zones["EC"]), len(zones["TZ"]), len(zones["BZ"])
    dens_ec = n_ec / vol_ec if vol_ec > 0 else 0
    dens_tz = n_tz / vol_tz if vol_tz > 0 else 0
    dens_bz = n_bz / vol_bz if vol_bz > 0 else 0
    # 감소: EC >= TZ >= BZ
    passed = dens_ec >= dens_tz >= dens_bz
    # 점수: 감소 기울기 일관성 (클수록 좋게)
    score = (dens_ec - dens_bz) / (max(dens_ec, 1e-10))
    return (passed, float(score))


def filter_hydrophobic_core(
    zones: Dict[str, List],
    structure,
) -> Tuple[bool, float]:
    """
    Hydrophobic Core: 내부(EC) 평균 소수성 > 외부(BZ) 평균 소수성.
    """
    residues = _get_residues(structure)
    rid_to_resname = {r["residue_id"]: r["resname_1"] for r in residues}
    def mean_hydro(rids):
        if not rids:
            return 0.0
        vals = [get_hydrophobicity(rid_to_resname.get(rid, "G")) for rid in rids]
        return float(np.mean(vals))
    h_ec = mean_hydro(zones["EC"])
    h_bz = mean_hydro(zones["BZ"])
    passed = h_ec > h_bz
    diff = h_ec - h_bz
    return (passed, float(diff))


def apply_all_filters(
    structure,
    strict: bool = False,
) -> Dict[str, Tuple[bool, Union[float, Dict[str, float]]]]:
    """
    ​5개 필터 모두 적용.
    strict=True → Strict(실험설계) 임계값, False → Primary(특허).
    Returns: dict[filter_name, (passed, value)].
    """
    center = calculate_geometric_center(structure)
    distances = calculate_residue_distances(structure, center)
    zones = classify_zones(distances, ec_radius=EC_RADIUS, tz_radius=TZ_OUTER_RADIUS)

    ec_thresh = EMPTY_CENTER_STRICT if strict else EMPTY_CENTER_PRIMARY
    fib_range = FIBONACCI_RANGE_STRICT if strict else FIBONACCI_RANGE_PRIMARY

    results = {}
    results["empty_center"] = filter_empty_center(zones, threshold=ec_thresh)
    results["fibonacci_ratio"] = filter_fibonacci_ratio(distances, range_vals=fib_range)
    results["zone_balance"] = filter_zone_balance(zones, min_ratio=ZONE_BALANCE_MIN_RATIO)
    results["density_gradient"] = filter_density_gradient(zones, structure)
    results["hydrophobic_core"] = filter_hydrophobic_core(zones, structure)
    return results


def batch_validate(pdb_list: List[Union[str, Path]]) -> "pd.DataFrame":
    """
    PDB 경로 리스트에 대해 apply_all_filters 적용, 결과 DataFrame 반환.
    검증 기준: 80%+ 통과율 목표.
    """
    if not HAS_PANDAS:
        raise ImportError("pandas required for batch_validate")
    from ..utils.pdb_parser import load_structure

    rows = []
    for path in pdb_list:
        path = Path(path)
        row = {"pdb": path.stem, "path": str(path)}
        try:
            structure = load_structure(str(path))
            res = apply_all_filters(structure, strict=False)
            for name, (passed, val) in res.items():
                row[f"{name}_passed"] = passed
                row[f"{name}_value"] = val
            row["all_passed"] = all(r[0] for r in res.values())
        except Exception as e:
            row["error"] = str(e)
            row["all_passed"] = False
        rows.append(row)

    df = pd.DataFrame(rows)
    return df
