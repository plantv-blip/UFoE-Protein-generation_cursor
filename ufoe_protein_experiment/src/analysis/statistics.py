"""
통계: t-검정, Fisher, Cohen's d, 부트스트랩 CI.
"""

from typing import List, Tuple, Optional
import math

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def t_test_independent(
    a: List[float],
    b: List[float],
) -> Tuple[float, float]:
    """독립 표본 t-검정. 반환: (t_statistic, p_value)."""
    if not HAS_NUMPY or len(a) < 2 or len(b) < 2:
        return (0.0, 1.0)
    na, nb = len(a), len(b)
    ma, mb = float(np.mean(a)), float(np.mean(b))
    va = float(np.var(a, ddof=1)) if na > 1 else 0.0
    vb = float(np.var(b, ddof=1)) if nb > 1 else 0.0
    pooled = ((na - 1) * va + (nb - 1) * vb) / (na + nb - 2)
    if pooled <= 0:
        return (0.0, 1.0)
    se = math.sqrt(pooled * (1.0 / na + 1.0 / nb))
    if se <= 0:
        return (0.0, 1.0)
    t = (ma - mb) / se
    from scipy import stats as scipy_stats
    p = 2 * (1 - scipy_stats.t.cdf(abs(t), na + nb - 2))
    return (float(t), float(p))


def cohens_d(a: List[float], b: List[float]) -> float:
    """Cohen's d (효과 크기)."""
    if not HAS_NUMPY or len(a) < 2 or len(b) < 2:
        return 0.0
    ma, mb = float(np.mean(a)), float(np.mean(b))
    va = float(np.var(a, ddof=1))
    vb = float(np.var(b, ddof=1))
    na, nb = len(a), len(b)
    pooled_std = math.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled_std <= 0:
        return 0.0
    return (ma - mb) / pooled_std


def fisher_exact(a_success: int, a_total: int, b_success: int, b_total: int) -> float:
    """Fisher 정확 검정 p-value."""
    try:
        from scipy.stats import fisher_exact as scipy_fisher
        table = [[a_success, a_total - a_success], [b_success, b_total - b_success]]
        _, p = scipy_fisher(table)
        return float(p)
    except Exception:
        return 1.0


def bootstrap_ci(
    values: List[float],
    statistic_fn=None,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
) -> Tuple[float, float]:
    """부트스트랩 신뢰구간. statistic_fn = None 이면 평균."""
    if not HAS_NUMPY or not values:
        return (0.0, 0.0)
    arr = np.array(values)
    fn = statistic_fn or (lambda x: float(np.mean(x)))
    rng = np.random.default_rng()
    stats = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(arr), size=len(arr))
        stats.append(fn(arr[idx]))
    stats = sorted(stats)
    lo = int((1 - ci) / 2 * n_bootstrap)
    hi = int((1 + ci) / 2 * n_bootstrap)
    return (float(stats[lo]), float(stats[hi]))
