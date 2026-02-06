"""
접힘 지표: RMSD 수렴(σ<2Å), Rg 안정성(변화<30%), 에너지 수렴(CV<5%).
2개 이상 통과 시 접힘 성공.
"""

from typing import List, Tuple, Optional
import numpy as np


def compute_rmsd_convergence(
    rmsd_trajectory: List[float],
    threshold_sigma: float = 2.0,
    window_frac: float = 0.2,
) -> Tuple[bool, float]:
    """후반 window_frac 구간 RMSD 표준편차 < threshold_sigma 이면 수렴."""
    if not rmsd_trajectory or len(rmsd_trajectory) < 5:
        return (False, float("nan"))
    arr = np.array(rmsd_trajectory)
    n = len(arr)
    start = int(n * (1 - window_frac))
    tail = arr[start:]
    sigma = float(np.std(tail))
    return (sigma < threshold_sigma, sigma)


def compute_rg_stability(
    rg_trajectory: List[float],
    max_change_frac: float = 0.30,
    window_frac: float = 0.2,
) -> Tuple[bool, float]:
    """후반 Rg 변화가 max_change_frac(30%) 미만이면 안정."""
    if not rg_trajectory or len(rg_trajectory) < 5:
        return (False, float("nan"))
    arr = np.array(rg_trajectory)
    n = len(arr)
    start = int(n * (1 - window_frac))
    tail = arr[start:]
    rg_min, rg_max = float(np.min(tail)), float(np.max(tail))
    mean_rg = float(np.mean(tail))
    change = (rg_max - rg_min) / mean_rg if mean_rg > 1e-6 else 1.0
    return (change < max_change_frac, change)


def compute_energy_convergence(
    energy_trajectory: List[float],
    cv_threshold: float = 0.05,
    window_frac: float = 0.2,
) -> Tuple[bool, float]:
    """후반 에너지 변동계수(CV) < cv_threshold(5%) 이면 수렴."""
    if not energy_trajectory or len(energy_trajectory) < 5:
        return (False, float("nan"))
    arr = np.array(energy_trajectory)
    n = len(arr)
    start = int(n * (1 - window_frac))
    tail = arr[start:]
    mean_e = float(np.mean(tail))
    std_e = float(np.std(tail))
    cv = std_e / abs(mean_e) if abs(mean_e) > 1e-6 else 1.0
    return (cv < cv_threshold, cv)


def compute_contact_order(coords: np.ndarray, sequence: str) -> float:
    """Contact Order (단순화: CA-CA 거리 기반)."""
    if len(coords) < 3:
        return 0.0
    n = len(coords)
    total, count = 0.0, 0
    for i in range(n):
        for j in range(i + 2, n):
            d = np.linalg.norm(coords[i] - coords[j])
            if d < 8.0:
                total += (j - i)
                count += 1
    return (total / count / n) if count else 0.0


def folding_success_criteria(
    rmsd_sigma: float,
    rg_change: float,
    energy_cv: float,
    min_pass: int = 2,
    rmsd_thresh: float = 2.0,
    rg_thresh: float = 0.30,
    energy_thresh: float = 0.05,
) -> bool:
    """3개 중 min_pass(2)개 이상 통과 시 성공."""
    pass_rmsd = rmsd_sigma < rmsd_thresh
    pass_rg = rg_change < rg_thresh
    pass_energy = energy_cv < energy_thresh
    return (int(pass_rmsd) + int(pass_rg) + int(pass_energy)) >= min_pass
