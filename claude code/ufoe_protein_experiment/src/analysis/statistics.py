"""
Phase 4: 통계 분석 모듈

UFoE 설계 구조와 대조군 사이의 접힘 성공률/메트릭 차이를 통계적으로 검증한다.

분석 방법:
  1. 독립표본 t-검정 (접힘 메트릭 평균 비교)
  2. Fisher 정확 검정 (접힘 성공/실패 비율 비교)
  3. Cohen's d (효과 크기)
  4. Mann-Whitney U (비모수 검정)
  5. 부트스트랩 신뢰구간
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from src.utils.constants import STATS_SIGNIFICANCE_LEVEL, STATS_COHENS_D_LARGE
from src.folding.folding_metrics import FoldingReport

logger = logging.getLogger(__name__)


# =============================================================================
# 데이터 구조
# =============================================================================

@dataclass
class ComparisonResult:
    """두 그룹 간 통계 비교 결과."""

    metric_name: str
    group_a_name: str
    group_b_name: str

    # 기술 통계
    group_a_mean: float
    group_a_std: float
    group_a_n: int
    group_b_mean: float
    group_b_std: float
    group_b_n: int

    # 검정 결과
    t_statistic: float = 0.0
    t_pvalue: float = 1.0
    u_statistic: float = 0.0
    u_pvalue: float = 1.0
    cohens_d: float = 0.0

    # 판정
    significant: bool = False     # p < α
    large_effect: bool = False    # |d| > 0.8

    metadata: dict = field(default_factory=dict)


@dataclass
class ProportionTestResult:
    """비율 비교 검정 결과."""

    metric_name: str
    group_a_name: str
    group_b_name: str

    group_a_success: int
    group_a_total: int
    group_a_rate: float
    group_b_success: int
    group_b_total: int
    group_b_rate: float

    # Fisher 정확 검정
    fisher_odds_ratio: float = 1.0
    fisher_pvalue: float = 1.0

    # Chi-square
    chi2_statistic: float = 0.0
    chi2_pvalue: float = 1.0

    significant: bool = False

    metadata: dict = field(default_factory=dict)


@dataclass
class StatisticalReport:
    """Phase 4 통계 보고서."""

    # 접힘 성공률 비교
    folding_rate_test: ProportionTestResult | None = None

    # 연속 변수 비교
    metric_comparisons: list[ComparisonResult] = field(default_factory=list)

    # 자기일관성 비교 (Phase 3)
    consistency_test: ProportionTestResult | None = None

    # 종합 판정
    ufoe_superior: bool = False
    summary: str = ""

    metadata: dict = field(default_factory=dict)


# =============================================================================
# 비교 함수
# =============================================================================

def compare_groups(
    report_a: FoldingReport,
    report_b: FoldingReport,
    alpha: float = STATS_SIGNIFICANCE_LEVEL,
) -> StatisticalReport:
    """두 실험 그룹의 접힘 결과를 통계적으로 비교한다.

    Parameters
    ----------
    report_a : FoldingReport — 실험군 (UFoE)
    report_b : FoldingReport — 대조군 (Random/Shuffled)
    alpha : float — 유의수준

    Returns
    -------
    StatisticalReport
    """
    logger.info(
        f"통계 비교: {report_a.group_name} vs {report_b.group_name}"
    )

    stat_report = StatisticalReport()

    # =========================================================================
    # 1. 접힘 성공률 비교 (Fisher 정확 검정)
    # =========================================================================
    stat_report.folding_rate_test = _test_proportions(
        metric_name="folding_success_rate",
        group_a_name=report_a.group_name,
        group_b_name=report_b.group_name,
        a_success=report_a.success_count,
        a_total=report_a.total_count,
        b_success=report_b.success_count,
        b_total=report_b.total_count,
        alpha=alpha,
    )

    # =========================================================================
    # 2. 연속 메트릭 비교
    # =========================================================================
    metrics_to_compare = [
        ("rmsd_final", "rmsd_final"),
        ("rg_final", "rg_final"),
        ("energy_final", "energy_final"),
        ("contact_order", "contact_order"),
        ("compactness", "compactness"),
        ("rmsd_convergence", "rmsd_convergence"),
    ]

    for metric_key, attr_name in metrics_to_compare:
        values_a = [
            getattr(m, attr_name)
            for m in report_a.metrics
            if m.folding_success and np.isfinite(getattr(m, attr_name))
        ]
        values_b = [
            getattr(m, attr_name)
            for m in report_b.metrics
            if m.folding_success and np.isfinite(getattr(m, attr_name))
        ]

        if len(values_a) >= 2 and len(values_b) >= 2:
            comparison = _compare_continuous(
                metric_name=metric_key,
                group_a_name=report_a.group_name,
                group_b_name=report_b.group_name,
                values_a=np.array(values_a),
                values_b=np.array(values_b),
                alpha=alpha,
            )
            stat_report.metric_comparisons.append(comparison)

    # =========================================================================
    # 3. 종합 판정
    # =========================================================================
    n_significant = sum(
        1 for c in stat_report.metric_comparisons if c.significant
    )
    folding_rate_sig = (
        stat_report.folding_rate_test.significant
        if stat_report.folding_rate_test else False
    )
    ufoe_better_rate = (
        report_a.success_rate > report_b.success_rate
        if report_a.total_count > 0 and report_b.total_count > 0 else False
    )

    stat_report.ufoe_superior = folding_rate_sig and ufoe_better_rate
    stat_report.summary = _generate_summary(
        stat_report, report_a.group_name, report_b.group_name
    )

    return stat_report


def compare_self_consistency(
    ufoe_k: int,
    ufoe_m: int,
    control_k: int,
    control_m: int,
    alpha: float = STATS_SIGNIFICANCE_LEVEL,
) -> ProportionTestResult:
    """UFoE vs 대조군의 자기일관성 비율을 비교한다.

    Parameters
    ----------
    ufoe_k : int — UFoE 자기일관적 수
    ufoe_m : int — UFoE 전체 수
    control_k : int — 대조군 자기일관적 수
    control_m : int — 대조군 전체 수
    alpha : float

    Returns
    -------
    ProportionTestResult
    """
    return _test_proportions(
        metric_name="self_consistency_ratio",
        group_a_name="UFoE",
        group_b_name="Control",
        a_success=ufoe_k,
        a_total=ufoe_m,
        b_success=control_k,
        b_total=control_m,
        alpha=alpha,
    )


# =============================================================================
# 내부 통계 함수
# =============================================================================

def _compare_continuous(
    metric_name: str,
    group_a_name: str,
    group_b_name: str,
    values_a: np.ndarray,
    values_b: np.ndarray,
    alpha: float = 0.05,
) -> ComparisonResult:
    """연속 변수에 대해 t-검정, Mann-Whitney U, Cohen's d를 수행한다."""

    mean_a, std_a = float(np.mean(values_a)), float(np.std(values_a, ddof=1))
    mean_b, std_b = float(np.mean(values_b)), float(np.std(values_b, ddof=1))
    n_a, n_b = len(values_a), len(values_b)

    # t-검정 (Welch's)
    t_stat, t_pval = stats.ttest_ind(values_a, values_b, equal_var=False)

    # Mann-Whitney U (비모수)
    try:
        u_stat, u_pval = stats.mannwhitneyu(
            values_a, values_b, alternative="two-sided"
        )
    except ValueError:
        u_stat, u_pval = 0.0, 1.0

    # Cohen's d
    pooled_std = np.sqrt(
        ((n_a - 1) * std_a ** 2 + (n_b - 1) * std_b ** 2)
        / max(n_a + n_b - 2, 1)
    )
    cohens_d = (mean_a - mean_b) / max(pooled_std, 1e-10)

    significant = float(t_pval) < alpha
    large_effect = abs(cohens_d) > STATS_COHENS_D_LARGE

    return ComparisonResult(
        metric_name=metric_name,
        group_a_name=group_a_name,
        group_b_name=group_b_name,
        group_a_mean=mean_a,
        group_a_std=std_a,
        group_a_n=n_a,
        group_b_mean=mean_b,
        group_b_std=std_b,
        group_b_n=n_b,
        t_statistic=float(t_stat),
        t_pvalue=float(t_pval),
        u_statistic=float(u_stat),
        u_pvalue=float(u_pval),
        cohens_d=float(cohens_d),
        significant=significant,
        large_effect=large_effect,
    )


def _test_proportions(
    metric_name: str,
    group_a_name: str,
    group_b_name: str,
    a_success: int,
    a_total: int,
    b_success: int,
    b_total: int,
    alpha: float = 0.05,
) -> ProportionTestResult:
    """비율에 대해 Fisher 정확 검정과 chi-square 검정을 수행한다."""

    a_fail = a_total - a_success
    b_fail = b_total - b_success
    a_rate = a_success / max(a_total, 1)
    b_rate = b_success / max(b_total, 1)

    # Fisher 정확 검정
    contingency = [[a_success, a_fail], [b_success, b_fail]]
    try:
        odds_ratio, fisher_p = stats.fisher_exact(contingency)
    except ValueError:
        odds_ratio, fisher_p = 1.0, 1.0

    # Chi-square (기대 빈도 ≥ 5일 때)
    try:
        chi2, chi2_p, _, _ = stats.chi2_contingency(
            [[a_success, a_fail], [b_success, b_fail]]
        )
    except ValueError:
        chi2, chi2_p = 0.0, 1.0

    significant = float(fisher_p) < alpha

    return ProportionTestResult(
        metric_name=metric_name,
        group_a_name=group_a_name,
        group_b_name=group_b_name,
        group_a_success=a_success,
        group_a_total=a_total,
        group_a_rate=a_rate,
        group_b_success=b_success,
        group_b_total=b_total,
        group_b_rate=b_rate,
        fisher_odds_ratio=float(odds_ratio),
        fisher_pvalue=float(fisher_p),
        chi2_statistic=float(chi2),
        chi2_pvalue=float(chi2_p),
        significant=significant,
    )


def bootstrap_confidence_interval(
    values: np.ndarray,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    statistic: str = "mean",
    seed: int | None = None,
) -> tuple[float, float, float]:
    """부트스트랩 신뢰구간을 계산한다.

    Parameters
    ----------
    values : np.ndarray
    n_bootstrap : int
    confidence : float
    statistic : str — 'mean' or 'median'
    seed : int, optional

    Returns
    -------
    (point_estimate, ci_lower, ci_upper)
    """
    rng = np.random.default_rng(seed)
    n = len(values)

    if n == 0:
        return 0.0, 0.0, 0.0

    stat_fn = np.mean if statistic == "mean" else np.median
    point = float(stat_fn(values))

    boot_stats = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(values, size=n, replace=True)
        boot_stats[i] = stat_fn(sample)

    alpha = (1 - confidence) / 2
    ci_lower = float(np.percentile(boot_stats, 100 * alpha))
    ci_upper = float(np.percentile(boot_stats, 100 * (1 - alpha)))

    return point, ci_lower, ci_upper


# =============================================================================
# 보고서 생성
# =============================================================================

def _generate_summary(
    report: StatisticalReport,
    group_a: str,
    group_b: str,
) -> str:
    """통계 보고서의 텍스트 요약을 생성한다."""
    lines = []

    # 접힘 성공률
    if report.folding_rate_test:
        t = report.folding_rate_test
        lines.append(
            f"접힘 성공률: {group_a}={t.group_a_rate:.1%} vs "
            f"{group_b}={t.group_b_rate:.1%} "
            f"(Fisher p={t.fisher_pvalue:.4f}, "
            f"{'유의함' if t.significant else '유의하지 않음'})"
        )

    # 유의한 메트릭
    sig_metrics = [c for c in report.metric_comparisons if c.significant]
    if sig_metrics:
        lines.append(f"\n유의한 메트릭 차이 ({len(sig_metrics)}개):")
        for c in sig_metrics:
            direction = "높음" if c.group_a_mean > c.group_b_mean else "낮음"
            lines.append(
                f"  - {c.metric_name}: {group_a}가 {direction} "
                f"(d={c.cohens_d:.2f}, p={c.t_pvalue:.4f})"
            )

    # 종합 판정
    if report.ufoe_superior:
        lines.append(f"\n결론: {group_a}가 {group_b}보다 통계적으로 우수함")
    else:
        lines.append(f"\n결론: {group_a}와 {group_b} 사이에 유의한 차이 없음")

    return "\n".join(lines)


def print_statistical_report(report: StatisticalReport) -> None:
    """통계 보고서를 보기 좋게 출력한다."""
    print(f"\n{'=' * 70}")
    print(f"  Phase 4: Statistical Comparison Report")
    print(f"{'=' * 70}")

    if report.folding_rate_test:
        t = report.folding_rate_test
        sig = "*" if t.significant else ""
        print(f"\n  Folding Success Rate:")
        print(f"    {t.group_a_name}: {t.group_a_rate:.1%} "
              f"({t.group_a_success}/{t.group_a_total})")
        print(f"    {t.group_b_name}: {t.group_b_rate:.1%} "
              f"({t.group_b_success}/{t.group_b_total})")
        print(f"    Fisher p = {t.fisher_pvalue:.4f}{sig}")

    if report.metric_comparisons:
        print(f"\n  {'Metric':<20s} {'Group A':>10s} {'Group B':>10s} "
              f"{'t-stat':>8s} {'p-val':>8s} {'d':>8s} {'Sig':>4s}")
        print(f"  {'-'*68}")
        for c in report.metric_comparisons:
            sig = " *" if c.significant else ""
            print(f"  {c.metric_name:<20s} "
                  f"{c.group_a_mean:>10.3f} {c.group_b_mean:>10.3f} "
                  f"{c.t_statistic:>8.2f} {c.t_pvalue:>8.4f} "
                  f"{c.cohens_d:>8.2f}{sig}")

    print(f"\n  {report.summary}")
    print(f"{'=' * 70}\n")
