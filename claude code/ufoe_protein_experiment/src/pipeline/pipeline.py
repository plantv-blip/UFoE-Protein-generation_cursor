"""
Phase 2 통합 파이프라인

전체 실험 흐름을 오케스트레이션한다:

  Phase 2a: UFoE 생성 → 순수 MD (100ns) → 접힘 판정
  Phase 2b: UFoE 생성 → ESMFold → MD 정제 (10ns) → 접힘 판정
  Phase 2c: 대조군 생성 → MD → 접힘 판정
  Phase 3:  자기일관성 루프 (접힌 구조에 UFoE 재적용)
  Phase 4:  통계 비교 (UFoE vs 대조군)

사용 예:
    pipeline = Phase2Pipeline(output_dir="output/experiment_001")
    results = pipeline.run_full()
    pipeline.print_summary(results)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.constants import (
    MD_N_SAMPLES_2A,
    MD_N_SAMPLES_2B,
    MD_N_CONTROL_RANDOM,
    MD_SIMULATION_TIME_NS,
    MD_REFINEMENT_TIME_NS,
    MD_LENGTH_RANGE,
)
from src.utils.pdb_writer import write_pdb_from_coords, sequence_to_fasta
from src.generator.ufoef_generator import UFoEGenerator, GeneratedCandidate
from src.generator.control_generators import (
    generate_random_candidates,
    generate_inverted_candidates,
    generate_shuffled_candidates,
    calculate_composition,
)
from src.filters.ufoef_filters import apply_all_filters, all_filters_passed
from src.folding.md_simulator import MDSimulator, MDConfig, MDTrajectory
from src.folding.esmfold_predictor import ESMFoldPredictor, ESMFoldPrediction
from src.folding.folding_metrics import (
    evaluate_folding,
    evaluate_batch,
    FoldingReport,
    FoldingMetrics,
    print_folding_report,
)
from src.pipeline.elimination_loop import (
    run_elimination_loop,
    EliminationReport,
    print_elimination_report,
)
from src.analysis.statistics import (
    compare_groups,
    compare_self_consistency,
    StatisticalReport,
    print_statistical_report,
)

logger = logging.getLogger(__name__)


# =============================================================================
# 데이터 구조
# =============================================================================

@dataclass
class PipelineConfig:
    """파이프라인 설정."""

    # 출력 경로
    output_dir: str | Path = "output/phase2"

    # Phase 2a: 순수 MD
    n_samples_2a: int = MD_N_SAMPLES_2A
    md_time_ns: float = MD_SIMULATION_TIME_NS
    length_range: tuple[int, int] = MD_LENGTH_RANGE

    # Phase 2b: ESMFold + MD
    n_samples_2b: int = MD_N_SAMPLES_2B
    md_refinement_ns: float = MD_REFINEMENT_TIME_NS
    esmfold_mode: str = "mock"  # 'local', 'api', 'mock'

    # Phase 2c: 대조군
    n_control_random: int = MD_N_CONTROL_RANDOM
    n_control_inverted: int = 50

    # 공통
    protein_type: str = "B"
    seed: int = 42
    strict_filters: bool = False

    # MD 설정
    md_config: MDConfig | None = None

    # 실험 모드 ('full', 'quick', 'mock')
    # quick: 샘플 수 축소 (n=5), mock: 합성 데이터만 사용
    run_mode: str = "mock"


@dataclass
class PipelineResults:
    """전체 파이프라인 실행 결과."""

    # Phase 1: 생성 결과
    ufoe_candidates: list[GeneratedCandidate] = field(default_factory=list)
    control_random: list[GeneratedCandidate] = field(default_factory=list)
    control_inverted: list[GeneratedCandidate] = field(default_factory=list)
    control_shuffled: list[GeneratedCandidate] = field(default_factory=list)

    # Phase 2: 접힘 결과
    ufoe_2a_report: FoldingReport | None = None
    ufoe_2b_report: FoldingReport | None = None
    control_random_report: FoldingReport | None = None
    control_inverted_report: FoldingReport | None = None
    control_shuffled_report: FoldingReport | None = None

    # Phase 2b: ESMFold 예측
    esmfold_predictions: list[ESMFoldPrediction] = field(default_factory=list)

    # Phase 3: 자기일관성
    elimination_ufoe_2a: EliminationReport | None = None
    elimination_ufoe_2b: EliminationReport | None = None
    elimination_control: EliminationReport | None = None

    # Phase 4: 통계
    stats_2a_vs_random: StatisticalReport | None = None
    stats_2a_vs_inverted: StatisticalReport | None = None
    stats_2b_vs_random: StatisticalReport | None = None

    # 실행 정보
    total_wall_time: float = 0.0
    metadata: dict = field(default_factory=dict)


# =============================================================================
# 메인 파이프라인
# =============================================================================

class Phase2Pipeline:
    """Phase 2 전체 파이프라인 오케스트레이터.

    사용법:
        pipeline = Phase2Pipeline(config=PipelineConfig(run_mode='mock'))
        results = pipeline.run_full()
        pipeline.save_results(results)
    """

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 실행 모드에 따라 설정 조정
        if self.config.run_mode == "quick":
            self.config.n_samples_2a = 5
            self.config.n_samples_2b = 5
            self.config.n_control_random = 5
            self.config.n_control_inverted = 5
            self.config.md_time_ns = 1.0
            self.config.md_refinement_ns = 0.5

        # MD 설정 (run_mode=='mock'이면 실제 MD 대신 mock 궤적 사용)
        if self.config.md_config is None:
            self.config.md_config = MDConfig(
                simulation_time_ns=self.config.md_time_ns,
                output_dir=str(self.output_dir / "md"),
                force_mock=(self.config.run_mode == "mock"),
            )

        self.md_sim = MDSimulator(self.config.md_config)
        self.esmfold = ESMFoldPredictor(
            mode=self.config.esmfold_mode,
            output_dir=str(self.output_dir / "esmfold"),
        )

        self.rng = np.random.default_rng(self.config.seed)

    def run_full(self) -> PipelineResults:
        """전체 파이프라인을 실행한다.

        Returns
        -------
        PipelineResults
        """
        start_time = time.time()
        results = PipelineResults()

        logger.info(f"{'=' * 60}")
        logger.info(f"UFoE Phase 2 파이프라인 시작")
        logger.info(f"모드: {self.config.run_mode}")
        logger.info(f"출력: {self.output_dir}")
        logger.info(f"{'=' * 60}")

        # =====================================================================
        # Step 1: 구조 생성
        # =====================================================================
        logger.info("\n[Step 1] 구조 생성...")
        results.ufoe_candidates = self._generate_ufoe_candidates()
        results.control_random = self._generate_random_controls(
            results.ufoe_candidates
        )
        results.control_inverted = self._generate_inverted_controls()
        results.control_shuffled = generate_shuffled_candidates(
            results.ufoe_candidates[:min(10, len(results.ufoe_candidates))],
            n_shuffles=1,
            seed=self.config.seed + 1000,
        )

        # =====================================================================
        # Step 2a: UFoE + 순수 MD
        # =====================================================================
        logger.info("\n[Step 2a] UFoE + 순수 MD 시뮬레이션...")
        ufoe_2a_trajs = self._run_md_batch(
            results.ufoe_candidates, "ufoe_2a"
        )
        ufoe_2a_metrics = [
            evaluate_folding(t) for t in ufoe_2a_trajs
        ]
        results.ufoe_2a_report = evaluate_batch(
            ufoe_2a_trajs, "ufoe_2a"
        )

        # =====================================================================
        # Step 2b: UFoE + ESMFold + MD 정제
        # =====================================================================
        logger.info("\n[Step 2b] UFoE + ESMFold + MD 정제...")
        results.esmfold_predictions = self._run_esmfold_batch(
            results.ufoe_candidates[:self.config.n_samples_2b]
        )
        ufoe_2b_trajs = self._run_md_refinement(results.esmfold_predictions)
        ufoe_2b_metrics = [evaluate_folding(t) for t in ufoe_2b_trajs]
        results.ufoe_2b_report = evaluate_batch(
            ufoe_2b_trajs, "ufoe_2b"
        )

        # =====================================================================
        # Step 2c: 대조군 MD
        # =====================================================================
        logger.info("\n[Step 2c] 대조군 MD...")
        ctrl_rand_trajs = self._run_md_batch(
            results.control_random, "ctrl_random"
        )
        results.control_random_report = evaluate_batch(
            ctrl_rand_trajs, "control_random"
        )

        ctrl_inv_trajs = self._run_md_batch(
            results.control_inverted, "ctrl_inverted"
        )
        results.control_inverted_report = evaluate_batch(
            ctrl_inv_trajs, "control_inverted"
        )

        ctrl_shuf_trajs = self._run_md_batch(
            results.control_shuffled, "ctrl_shuffled"
        )
        results.control_shuffled_report = evaluate_batch(
            ctrl_shuf_trajs, "control_shuffled"
        )

        # =====================================================================
        # Step 3: 자기일관성 루프
        # =====================================================================
        logger.info("\n[Step 3] Phase 3 — 자기일관성 루프...")

        ufoe_rnames = {
            c.metadata.get("candidate_id", f"ufoe_2a_{i:04d}"): c.residue_names
            for i, c in enumerate(results.ufoe_candidates)
        }

        results.elimination_ufoe_2a = run_elimination_loop(
            trajectories=ufoe_2a_trajs,
            folding_metrics=ufoe_2a_metrics,
            residue_names_map=ufoe_rnames,
            group_name="ufoe_2a",
            strict=self.config.strict_filters,
        )

        ufoe_2b_rnames = {
            c.metadata.get("candidate_id", f"ufoe_2b_{i:04d}"): c.residue_names
            for i, c in enumerate(
                results.ufoe_candidates[:self.config.n_samples_2b]
            )
        }
        results.elimination_ufoe_2b = run_elimination_loop(
            trajectories=ufoe_2b_trajs,
            folding_metrics=ufoe_2b_metrics,
            residue_names_map=ufoe_2b_rnames,
            group_name="ufoe_2b",
            strict=self.config.strict_filters,
        )

        ctrl_rnames = {
            c.metadata.get("candidate_id", f"ctrl_random_{i:04d}"): c.residue_names
            for i, c in enumerate(results.control_random)
        }
        ctrl_rand_metrics = [evaluate_folding(t) for t in ctrl_rand_trajs]
        results.elimination_control = run_elimination_loop(
            trajectories=ctrl_rand_trajs,
            folding_metrics=ctrl_rand_metrics,
            residue_names_map=ctrl_rnames,
            group_name="control_random",
            strict=self.config.strict_filters,
        )

        # =====================================================================
        # Step 4: 통계 비교
        # =====================================================================
        logger.info("\n[Step 4] Phase 4 — 통계 분석...")

        if results.ufoe_2a_report and results.control_random_report:
            results.stats_2a_vs_random = compare_groups(
                results.ufoe_2a_report,
                results.control_random_report,
            )

        if results.ufoe_2a_report and results.control_inverted_report:
            results.stats_2a_vs_inverted = compare_groups(
                results.ufoe_2a_report,
                results.control_inverted_report,
            )

        if results.ufoe_2b_report and results.control_random_report:
            results.stats_2b_vs_random = compare_groups(
                results.ufoe_2b_report,
                results.control_random_report,
            )

        # =====================================================================
        # 완료
        # =====================================================================
        results.total_wall_time = time.time() - start_time
        results.metadata = {
            "config": {
                "run_mode": self.config.run_mode,
                "n_2a": self.config.n_samples_2a,
                "n_2b": self.config.n_samples_2b,
                "n_ctrl_random": self.config.n_control_random,
                "n_ctrl_inverted": self.config.n_control_inverted,
                "protein_type": self.config.protein_type,
                "md_time_ns": self.config.md_time_ns,
            },
        }

        logger.info(f"\n{'=' * 60}")
        logger.info(f"파이프라인 완료 | 총 소요: {results.total_wall_time:.1f}s")
        logger.info(f"{'=' * 60}")

        return results

    # =========================================================================
    # 내부 메서드
    # =========================================================================

    def _generate_ufoe_candidates(self) -> list[GeneratedCandidate]:
        """UFoE 후보를 생성한다."""
        n = self.config.n_samples_2a
        candidates = []

        for i in range(n):
            length = self.rng.integers(*self.config.length_range)
            gen = UFoEGenerator(
                length=int(length),
                protein_type=self.config.protein_type,
                seed=self.config.seed + i,
                filter_compatible=True,
            )
            c = gen.generate_candidate()
            c.metadata["candidate_id"] = f"ufoe_2a_{i:04d}"
            c.metadata["index"] = i

            # PDB 파일 저장
            pdb_dir = self.output_dir / "pdb" / "ufoe_2a"
            pdb_dir.mkdir(parents=True, exist_ok=True)
            write_pdb_from_coords(
                pdb_dir / f"ufoe_2a_{i:04d}.pdb",
                c.ca_coords,
                c.residue_names,
            )

            # FASTA 저장
            fasta_dir = self.output_dir / "fasta"
            fasta_dir.mkdir(parents=True, exist_ok=True)
            sequence_to_fasta(
                fasta_dir / f"ufoe_2a_{i:04d}.fasta",
                c.sequence,
                seq_id=f"ufoe_2a_{i:04d}",
            )

            candidates.append(c)

        # 필터 통과율 확인
        if candidates and candidates[0].structure:
            n_pass = sum(
                1 for c in candidates
                if c.structure and all_filters_passed(
                    apply_all_filters(c.structure, strict=False)
                )
            )
            logger.info(
                f"UFoE {n}개 생성 | 필터 통과: {n_pass}/{n} "
                f"({n_pass / n:.1%})"
            )

        return candidates

    def _generate_random_controls(
        self,
        ufoe_candidates: list[GeneratedCandidate],
    ) -> list[GeneratedCandidate]:
        """Random 대조군을 생성한다. UFoE와 동일 아미노산 조성 매칭."""
        composition = calculate_composition(ufoe_candidates)
        avg_length = int(np.mean([c.length for c in ufoe_candidates]))

        controls = generate_random_candidates(
            n=self.config.n_control_random,
            length=avg_length,
            seed=self.config.seed + 5000,
            match_composition=composition,
        )

        # PDB 저장
        pdb_dir = self.output_dir / "pdb" / "ctrl_random"
        pdb_dir.mkdir(parents=True, exist_ok=True)
        for i, c in enumerate(controls):
            c.metadata["candidate_id"] = f"ctrl_random_{i:04d}"
            write_pdb_from_coords(
                pdb_dir / f"ctrl_random_{i:04d}.pdb",
                c.ca_coords,
                c.residue_names,
            )

        return controls

    def _generate_inverted_controls(self) -> list[GeneratedCandidate]:
        """Inverted 대조군을 생성한다."""
        controls = generate_inverted_candidates(
            n=self.config.n_control_inverted,
            length=70,
            seed=self.config.seed + 6000,
        )

        pdb_dir = self.output_dir / "pdb" / "ctrl_inverted"
        pdb_dir.mkdir(parents=True, exist_ok=True)
        for i, c in enumerate(controls):
            c.metadata["candidate_id"] = f"ctrl_inv_{i:04d}"
            write_pdb_from_coords(
                pdb_dir / f"ctrl_inv_{i:04d}.pdb",
                c.ca_coords,
                c.residue_names,
            )

        return controls

    def _run_md_batch(
        self,
        candidates: list[GeneratedCandidate],
        group_prefix: str,
    ) -> list[MDTrajectory]:
        """후보 리스트에 대해 MD 시뮬레이션을 배치 실행한다."""
        trajectories = []

        for i, c in enumerate(candidates):
            cid = c.metadata.get("candidate_id", f"{group_prefix}_{i:04d}")

            # PDB 파일 경로
            pdb_path = (
                self.output_dir / "pdb" / group_prefix.split("_")[0]
                / f"{cid}.pdb"
            )

            # PDB 파일이 없으면 생성
            if not pdb_path.exists():
                pdb_path.parent.mkdir(parents=True, exist_ok=True)
                write_pdb_from_coords(
                    pdb_path, c.ca_coords, c.residue_names
                )

            traj = self.md_sim.run_from_pdb(
                pdb_path=pdb_path,
                candidate_id=cid,
                simulation_type=f"{group_prefix}_md",
                simulation_time_ns=self.config.md_time_ns,
            )
            trajectories.append(traj)

        return trajectories

    def _run_esmfold_batch(
        self,
        candidates: list[GeneratedCandidate],
    ) -> list[ESMFoldPrediction]:
        """ESMFold 배치 예측을 실행한다."""
        sequences = [
            (c.metadata.get("candidate_id", f"ufoe_2b_{i:04d}"), c.sequence)
            for i, c in enumerate(candidates)
        ]
        return self.esmfold.batch_predict(sequences)

    def _run_md_refinement(
        self,
        predictions: list[ESMFoldPrediction],
    ) -> list[MDTrajectory]:
        """ESMFold 예측 구조에 MD 정제를 실행한다."""
        trajectories = []

        for pred in predictions:
            if not pred.success or pred.pdb_path is None:
                logger.warning(f"ESMFold 실패로 MD 정제 스킵: {pred.candidate_id}")
                continue

            traj = self.md_sim.run_from_pdb(
                pdb_path=pred.pdb_path,
                candidate_id=pred.candidate_id,
                simulation_type="esmfold_md",
                simulation_time_ns=self.config.md_refinement_ns,
            )
            trajectories.append(traj)

        return trajectories

    # =========================================================================
    # 결과 저장 및 출력
    # =========================================================================

    def save_results(self, results: PipelineResults) -> Path:
        """파이프라인 결과를 디스크에 저장한다.

        Returns
        -------
        Path — 결과 디렉토리
        """
        results_dir = self.output_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        # 1. 접힘 보고서를 DataFrame으로 저장
        reports = {
            "ufoe_2a": results.ufoe_2a_report,
            "ufoe_2b": results.ufoe_2b_report,
            "ctrl_random": results.control_random_report,
            "ctrl_inverted": results.control_inverted_report,
            "ctrl_shuffled": results.control_shuffled_report,
        }

        for name, report in reports.items():
            if report is not None:
                rows = []
                for m in report.metrics:
                    rows.append({
                        "candidate_id": m.candidate_id,
                        "folding_success": m.folding_success,
                        "rmsd_final": m.rmsd_final,
                        "rmsd_convergence": m.rmsd_convergence,
                        "rg_initial": m.rg_initial,
                        "rg_final": m.rg_final,
                        "rg_change_ratio": m.rg_change_ratio,
                        "energy_final": m.energy_final,
                        "contact_order": m.contact_order,
                        "compactness": m.compactness,
                        "end_to_end": m.end_to_end_distance,
                        "plddt_mean": m.plddt_mean,
                    })
                df = pd.DataFrame(rows)
                df.to_csv(results_dir / f"{name}_metrics.csv", index=False)

        # 2. 자기일관성 결과 저장
        elim_reports = {
            "ufoe_2a": results.elimination_ufoe_2a,
            "ufoe_2b": results.elimination_ufoe_2b,
            "control": results.elimination_control,
        }
        for name, elim in elim_reports.items():
            if elim is not None:
                data = {
                    "group_name": elim.group_name,
                    "total_designed": elim.total_designed,
                    "total_folded": elim.total_folded,
                    "post_fold_passed": elim.post_fold_passed,
                    "self_consistency_ratio": elim.self_consistency_ratio,
                    "self_consistent": elim.self_consistent,
                    "per_filter_pass_rates": elim.per_filter_pass_rates,
                }
                with open(results_dir / f"elimination_{name}.json", "w") as f:
                    json.dump(data, f, indent=2, default=str)

        # 3. 요약 통계 저장
        summary = {
            "run_mode": self.config.run_mode,
            "total_wall_time_s": results.total_wall_time,
        }
        for name, report in reports.items():
            if report:
                summary[f"{name}_n"] = report.total_count
                summary[f"{name}_success"] = report.success_count
                summary[f"{name}_rate"] = report.success_rate

        with open(results_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"결과 저장 완료: {results_dir}")
        return results_dir

    @staticmethod
    def print_summary(results: PipelineResults) -> None:
        """전체 결과를 요약 출력한다."""
        print(f"\n{'#' * 70}")
        print(f"#  UFoE Phase 2 Pipeline — 최종 요약")
        print(f"#  총 소요 시간: {results.total_wall_time:.1f}s")
        print(f"{'#' * 70}")

        # 접힘 보고서
        for name, report in [
            ("UFoE 2a (순수 MD)", results.ufoe_2a_report),
            ("UFoE 2b (ESMFold+MD)", results.ufoe_2b_report),
            ("Control Random", results.control_random_report),
            ("Control Inverted", results.control_inverted_report),
            ("Control Shuffled", results.control_shuffled_report),
        ]:
            if report:
                print_folding_report(report)

        # 자기일관성
        for name, elim in [
            ("UFoE 2a", results.elimination_ufoe_2a),
            ("UFoE 2b", results.elimination_ufoe_2b),
            ("Control", results.elimination_control),
        ]:
            if elim:
                print_elimination_report(elim)

        # 통계
        for name, stat in [
            ("2a vs Random", results.stats_2a_vs_random),
            ("2a vs Inverted", results.stats_2a_vs_inverted),
            ("2b vs Random", results.stats_2b_vs_random),
        ]:
            if stat:
                print_statistical_report(stat)


# =============================================================================
# 편의 함수
# =============================================================================

def run_quick_experiment(
    output_dir: str = "output/quick_test",
    seed: int = 42,
) -> PipelineResults:
    """빠른 테스트를 위한 축소 실험을 실행한다.

    n=5 샘플, 1ns MD, mock ESMFold

    Returns
    -------
    PipelineResults
    """
    config = PipelineConfig(
        output_dir=output_dir,
        run_mode="quick",
        seed=seed,
        esmfold_mode="mock",
    )
    pipeline = Phase2Pipeline(config)
    return pipeline.run_full()


def run_mock_experiment(
    output_dir: str = "output/mock_test",
    n: int = 10,
    seed: int = 42,
) -> PipelineResults:
    """합성 데이터로 파이프라인 로직을 테스트한다.

    OpenMM/ESMFold 없이 mock 데이터로 전체 흐름을 검증.

    Returns
    -------
    PipelineResults
    """
    config = PipelineConfig(
        output_dir=output_dir,
        run_mode="mock",
        n_samples_2a=n,
        n_samples_2b=min(n, 5),
        n_control_random=n,
        n_control_inverted=n,
        seed=seed,
        esmfold_mode="mock",
    )
    pipeline = Phase2Pipeline(config)
    return pipeline.run_full()
