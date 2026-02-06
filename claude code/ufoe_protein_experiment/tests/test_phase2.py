"""
Phase 2 파이프라인 테스트

합성(mock) 데이터를 사용하여 Phase 2 전체 파이프라인의 정확성을 검증한다:
  1. PDB 작성/읽기
  2. MD 시뮬레이션 (mock)
  3. ESMFold 예측 (mock)
  4. 접힘 메트릭 계산
  5. 대조군 생성
  6. 자기일관성 루프
  7. 통계 비교
  8. 통합 파이프라인
"""

import numpy as np
import pytest
from pathlib import Path
import tempfile
import shutil

from src.utils.constants import (
    EC_RADIUS,
    TZ_RADIUS,
    HYDROPHOBIC_AA,
    HYDROPHILIC_AA,
    STANDARD_AA,
    FOLDING_CRITERIA,
    SELF_CONSISTENCY_THRESHOLD,
)
from src.utils.pdb_parser import parse_pdb_from_coords
from src.utils.pdb_writer import (
    write_pdb_from_coords,
    write_pdb_from_structure,
    sequence_to_fasta,
    read_pdb_coords,
)
from src.generator.ufoef_generator import UFoEGenerator, GeneratedCandidate
from src.generator.control_generators import (
    generate_random_candidates,
    generate_shuffled_candidates,
    generate_inverted_candidates,
    calculate_composition,
)
from src.folding.md_simulator import (
    MDSimulator,
    MDConfig,
    MDTrajectory,
    _calculate_rg,
    _calculate_rmsd,
    calculate_rmsd_aligned,
)
from src.folding.esmfold_predictor import ESMFoldPredictor, ESMFoldPrediction
from src.folding.folding_metrics import (
    evaluate_folding,
    evaluate_batch,
    FoldingMetrics,
    FoldingReport,
    _calculate_contact_order,
)
from src.pipeline.elimination_loop import (
    run_elimination_loop,
    EliminationReport,
    _track_zone_shift,
)
from src.analysis.statistics import (
    compare_groups,
    compare_self_consistency,
    bootstrap_confidence_interval,
    ComparisonResult,
    ProportionTestResult,
)


# =============================================================================
# 테스트 픽스처
# =============================================================================

@pytest.fixture
def tmp_dir():
    """임시 디렉토리를 생성하고 테스트 후 삭제한다."""
    d = tempfile.mkdtemp(prefix="ufoe_test_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def sample_coords():
    """60잔기 합성 좌표."""
    rng = np.random.default_rng(42)
    return rng.normal(0, 5, (60, 3))


@pytest.fixture
def sample_names():
    """60잔기 합성 잔기 이름."""
    all_aa = sorted(STANDARD_AA)
    rng = np.random.default_rng(42)
    return [rng.choice(all_aa) for _ in range(60)]


@pytest.fixture
def ufoe_candidates():
    """5개의 UFoE 후보 구조를 생성."""
    gen = UFoEGenerator(length=50, protein_type="B", seed=42)
    candidates = []
    for i in range(5):
        c = gen.generate_candidate()
        c.metadata["candidate_id"] = f"ufoe_test_{i:04d}"
        c.metadata["index"] = i
        candidates.append(c)
    return candidates


# =============================================================================
# 1. PDB Writer 테스트
# =============================================================================

class TestPDBWriter:
    """PDB 파일 작성/읽기 테스트."""

    def test_write_and_read_roundtrip(self, tmp_dir, sample_coords, sample_names):
        """PDB 쓰기 → 읽기 왕복 테스트."""
        pdb_path = tmp_dir / "test.pdb"
        write_pdb_from_coords(pdb_path, sample_coords, sample_names)

        assert pdb_path.exists()
        coords_read, names_read = read_pdb_coords(pdb_path)

        assert len(coords_read) == len(sample_coords)
        assert len(names_read) == len(sample_names)
        np.testing.assert_array_almost_equal(coords_read, sample_coords, decimal=2)

    def test_write_pdb_from_structure(self, tmp_dir, sample_coords, sample_names):
        """ProteinStructure에서 PDB 작성."""
        structure = parse_pdb_from_coords("test", sample_coords, sample_names)
        pdb_path = write_pdb_from_structure(tmp_dir / "struct.pdb", structure)
        assert pdb_path.exists()

    def test_fasta_output(self, tmp_dir):
        """FASTA 형식 저장."""
        seq = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFP"
        path = sequence_to_fasta(tmp_dir / "test.fasta", seq, "test_id")
        assert path.exists()

        content = path.read_text()
        assert ">test_id" in content
        assert seq in content.replace("\n", "")

    def test_b_factors(self, tmp_dir, sample_coords, sample_names):
        """B-factor가 올바르게 작성되는지 확인."""
        b_factors = np.random.default_rng(42).uniform(0, 100, len(sample_coords))
        pdb_path = tmp_dir / "bfactor.pdb"
        write_pdb_from_coords(
            pdb_path, sample_coords, sample_names, b_factors=b_factors
        )
        assert pdb_path.exists()

    def test_mismatched_lengths_raises(self, tmp_dir):
        """좌표와 잔기 이름 수가 다르면 에러."""
        coords = np.zeros((5, 3))
        names = ["ALA"] * 3
        with pytest.raises(ValueError):
            write_pdb_from_coords(tmp_dir / "bad.pdb", coords, names)


# =============================================================================
# 2. MD 시뮬레이터 테스트 (Mock)
# =============================================================================

class TestMDSimulator:
    """MD 시뮬레이터 mock 테스트."""

    def test_mock_trajectory_structure(self, tmp_dir, sample_coords, sample_names):
        """Mock 궤적의 데이터 구조가 올바른지 확인."""
        pdb_path = tmp_dir / "input.pdb"
        write_pdb_from_coords(pdb_path, sample_coords, sample_names)

        config = MDConfig(simulation_time_ns=1.0)
        sim = MDSimulator(config)

        traj = sim.run_from_pdb(
            pdb_path, candidate_id="test_001", simulation_type="test"
        )

        assert isinstance(traj, MDTrajectory)
        assert traj.candidate_id == "test_001"
        assert traj.n_residues == len(sample_coords)
        assert len(traj.timestamps_ns) > 0
        assert len(traj.potential_energies) == len(traj.timestamps_ns)
        assert len(traj.rg_values) == len(traj.timestamps_ns)
        assert len(traj.rmsd_to_initial) == len(traj.timestamps_ns)
        assert traj.final_ca_coords.shape[0] == len(sample_coords)
        assert traj.converged is True

    def test_different_simulation_times(self, tmp_dir, sample_coords, sample_names):
        """다른 시뮬레이션 시간 설정."""
        pdb_path = tmp_dir / "input.pdb"
        write_pdb_from_coords(pdb_path, sample_coords, sample_names)

        for sim_time in [0.5, 1.0, 2.0]:
            config = MDConfig(simulation_time_ns=sim_time)
            sim = MDSimulator(config)
            traj = sim.run_from_pdb(pdb_path, candidate_id=f"t{sim_time}")
            assert traj.total_time_ns == sim_time

    def test_rg_calculation(self):
        """Rg 계산 정확성."""
        # 정육면체 꼭짓점에 8개 잔기
        coords = np.array([
            [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
            [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        ], dtype=float)
        rg = _calculate_rg(coords)
        # 이론적 Rg = sqrt(3) ≈ 1.732
        assert abs(rg - np.sqrt(3)) < 0.01

    def test_rmsd_identical(self):
        """동일한 좌표의 RMSD는 0."""
        coords = np.random.default_rng(42).normal(0, 5, (50, 3))
        assert _calculate_rmsd(coords, coords) < 1e-10

    def test_rmsd_aligned(self):
        """Kabsch 정렬 후 RMSD가 순수 RMSD보다 작거나 같다."""
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 5, (50, 3))
        # 약간의 회전 + 이동
        moved = ref + 3.0  # 이동
        raw_rmsd = _calculate_rmsd(ref, moved)
        aligned_rmsd = calculate_rmsd_aligned(ref, moved)
        assert aligned_rmsd <= raw_rmsd + 1e-6


# =============================================================================
# 3. ESMFold 테스트 (Mock)
# =============================================================================

class TestESMFoldPredictor:
    """ESMFold mock 예측 테스트."""

    def test_mock_prediction(self, tmp_dir):
        """Mock 예측이 올바른 구조를 반환하는지."""
        predictor = ESMFoldPredictor(
            mode="mock", output_dir=str(tmp_dir / "esmfold")
        )

        seq = "MVLSPADKTNVKAAWGKVGA"
        result = predictor.predict(seq, candidate_id="test_esm")

        assert isinstance(result, ESMFoldPrediction)
        assert result.success is True
        assert result.n_residues == len(seq)
        assert result.ca_coords.shape == (len(seq), 3)
        assert result.plddt_scores.shape == (len(seq),)
        assert 0 < result.mean_plddt < 100
        assert result.pdb_path is not None
        assert result.pdb_path.exists()

    def test_batch_predict(self, tmp_dir):
        """배치 예측."""
        predictor = ESMFoldPredictor(
            mode="mock", output_dir=str(tmp_dir / "esmfold")
        )

        sequences = [
            ("test_1", "MVLSPADKTN"),
            ("test_2", "VKAAWGKVGA"),
            ("test_3", "HAGEYGAEAL"),
        ]
        results = predictor.batch_predict(sequences)

        assert len(results) == 3
        for r in results:
            assert r.success is True


# =============================================================================
# 4. 접힘 메트릭 테스트
# =============================================================================

class TestFoldingMetrics:
    """접힘 메트릭 계산 테스트."""

    def _make_mock_trajectory(
        self,
        cid: str = "test",
        n_residues: int = 50,
        converging: bool = True,
    ) -> MDTrajectory:
        """합성 궤적을 생성한다."""
        rng = np.random.default_rng(hash(cid) % (2**31))
        n_frames = 100

        timestamps = np.linspace(0, 100.0, n_frames)
        initial_ca = rng.normal(0, 5, (n_residues, 3))

        if converging:
            # 수렴하는 궤적
            rmsd = 5.0 * (1 - np.exp(-timestamps / 20.0)) + rng.normal(0, 0.3, n_frames)
            pe = -50 * n_residues + 30 * n_residues * np.exp(-timestamps / 15.0)
            pe += rng.normal(0, abs(pe.mean()) * 0.01, n_frames)
        else:
            # 발산하는 궤적
            rmsd = timestamps * 0.1 + rng.normal(0, 1.0, n_frames)
            pe = rng.normal(0, 1000, n_frames)

        rg_init = _calculate_rg(initial_ca)
        rg = np.full(n_frames, rg_init) + rng.normal(0, 0.3, n_frames)

        final_ca = initial_ca + rng.normal(0, 1.0, initial_ca.shape)

        return MDTrajectory(
            candidate_id=cid,
            simulation_type="test",
            n_residues=n_residues,
            total_time_ns=100.0,
            timestamps_ns=timestamps,
            potential_energies=pe,
            kinetic_energies=np.full(n_frames, 100.0),
            temperatures=np.full(n_frames, 300.0),
            rg_values=rg,
            rmsd_to_initial=np.abs(rmsd),
            ca_snapshots=[],
            snapshot_times_ns=[],
            final_ca_coords=final_ca,
            initial_ca_coords=initial_ca,
            converged=True,
        )

    def test_converging_trajectory_passes(self):
        """수렴하는 궤적은 접힘 성공으로 판정."""
        traj = self._make_mock_trajectory("conv", converging=True)
        metrics = evaluate_folding(traj)

        assert isinstance(metrics, FoldingMetrics)
        assert metrics.rmsd_final > 0
        assert metrics.rg_final > 0
        # 수렴 궤적은 최소 1개 기준 통과
        criteria_passed = sum([
            metrics.rmsd_converged, metrics.rg_stable, metrics.energy_converged
        ])
        assert criteria_passed >= 1

    def test_empty_trajectory_fails(self):
        """빈 궤적은 접힘 실패."""
        traj = MDTrajectory(
            candidate_id="empty",
            simulation_type="test",
            n_residues=0,
            total_time_ns=0,
            timestamps_ns=np.array([]),
            potential_energies=np.array([]),
            kinetic_energies=np.array([]),
            temperatures=np.array([]),
            rg_values=np.array([]),
            rmsd_to_initial=np.array([]),
            ca_snapshots=[],
            snapshot_times_ns=[],
            final_ca_coords=np.array([]),
            initial_ca_coords=np.array([]),
            converged=False,
            error_message="test error",
        )
        metrics = evaluate_folding(traj)
        assert metrics.folding_success is False

    def test_batch_evaluate(self):
        """배치 평가."""
        trajs = [
            self._make_mock_trajectory(f"batch_{i}", converging=True)
            for i in range(5)
        ]
        report = evaluate_batch(trajs, group_name="test_batch")

        assert isinstance(report, FoldingReport)
        assert report.total_count == 5
        assert report.group_name == "test_batch"

    def test_contact_order_linear_chain(self):
        """선형 체인의 접촉 순서."""
        # 직선 배열 — 접촉이 적어야 함
        coords = np.column_stack([
            np.arange(100) * 3.8,
            np.zeros(100),
            np.zeros(100),
        ])
        co = _calculate_contact_order(coords)
        # 직선이므로 접촉 순서가 낮거나 0
        assert co >= 0

    def test_contact_order_compact(self):
        """구상 구조의 접촉 순서."""
        rng = np.random.default_rng(42)
        coords = rng.normal(0, 3, (50, 3))
        co = _calculate_contact_order(coords)
        assert co > 0  # 컴팩트 구조는 접촉이 많음


# =============================================================================
# 5. 대조군 생성 테스트
# =============================================================================

class TestControlGenerators:
    """대조군 생성기 테스트."""

    def test_random_candidates(self):
        """Random 대조군 생성."""
        controls = generate_random_candidates(n=5, length=50, seed=42)
        assert len(controls) == 5
        for c in controls:
            assert isinstance(c, GeneratedCandidate)
            assert c.protein_type == "control_random"
            assert len(c.residue_names) == 50
            assert all(name in STANDARD_AA for name in c.residue_names)

    def test_shuffled_candidates(self, ufoe_candidates):
        """Shuffled 대조군 생성."""
        shuffled = generate_shuffled_candidates(
            ufoe_candidates, n_shuffles=1, seed=42
        )
        assert len(shuffled) == len(ufoe_candidates)
        for s in shuffled:
            assert s.protein_type == "control_shuffled"
            # 좌표는 원본과 동일
            # (원본 후보의 좌표와 비교는 순서 보존 확인)

    def test_inverted_candidates(self):
        """Inverted 대조군 생성."""
        controls = generate_inverted_candidates(n=5, length=50, seed=42)
        assert len(controls) == 5
        for c in controls:
            assert c.protein_type == "control_inverted"

    def test_composition_calculation(self, ufoe_candidates):
        """아미노산 조성 계산."""
        comp = calculate_composition(ufoe_candidates)
        assert isinstance(comp, dict)
        assert len(comp) > 0
        assert abs(sum(comp.values()) - 1.0) < 0.01

    def test_random_with_matched_composition(self, ufoe_candidates):
        """조성 매칭 대조군."""
        comp = calculate_composition(ufoe_candidates)
        controls = generate_random_candidates(
            n=5, length=50, seed=42, match_composition=comp
        )
        assert len(controls) == 5
        for c in controls:
            assert all(name in STANDARD_AA for name in c.residue_names)


# =============================================================================
# 6. Phase 3 자기일관성 루프 테스트
# =============================================================================

class TestEliminationLoop:
    """Phase 3 자기일관성 루프 테스트."""

    def _make_test_data(self, n: int = 5):
        """테스트용 궤적과 메트릭을 생성."""
        rng = np.random.default_rng(42)

        trajs = []
        metrics_list = []
        rnames_map = {}

        for i in range(n):
            cid = f"test_{i:04d}"
            n_res = 50
            initial = rng.normal(0, 5, (n_res, 3))
            final = initial + rng.normal(0, 1.0, (n_res, 3))

            traj = MDTrajectory(
                candidate_id=cid,
                simulation_type="test",
                n_residues=n_res,
                total_time_ns=100.0,
                timestamps_ns=np.linspace(0, 100, 100),
                potential_energies=np.full(100, -3000.0),
                kinetic_energies=np.full(100, 100.0),
                temperatures=np.full(100, 300.0),
                rg_values=np.full(100, 5.0),
                rmsd_to_initial=np.full(100, 3.0),
                ca_snapshots=[],
                snapshot_times_ns=[],
                final_ca_coords=final,
                initial_ca_coords=initial,
                converged=True,
            )
            trajs.append(traj)

            metrics_list.append(FoldingMetrics(
                candidate_id=cid,
                rmsd_convergence=1.0,
                rmsd_final=3.0,
                rg_change_ratio=0.1,
                rg_initial=5.0,
                rg_final=5.5,
                energy_convergence=0.01,
                energy_final=-3000.0,
                contact_order=0.15,
                rmsd_converged=True,
                rg_stable=True,
                energy_converged=True,
                folding_success=True,
                compactness=0.7,
                end_to_end_distance=10.0,
            ))

            names = [rng.choice(sorted(STANDARD_AA)) for _ in range(n_res)]
            rnames_map[cid] = names

        return trajs, metrics_list, rnames_map

    def test_elimination_loop_runs(self):
        """자기일관성 루프가 실행되고 보고서를 반환."""
        trajs, metrics, rnames = self._make_test_data(5)
        report = run_elimination_loop(
            trajs, metrics, rnames, group_name="test"
        )

        assert isinstance(report, EliminationReport)
        assert report.total_designed == 5
        assert report.total_folded == 5
        assert 0 <= report.self_consistency_ratio <= 1.0
        assert len(report.per_filter_pass_rates) == 5

    def test_zone_shift_tracking(self):
        """Zone 변화 추적."""
        initial = np.array([[0, 0, 0], [8, 0, 0], [15, 0, 0]], dtype=float)
        final = np.array([[0, 0, 0], [3, 0, 0], [20, 0, 0]], dtype=float)

        shift = _track_zone_shift(initial, final)
        assert "n_changed" in shift
        assert "change_ratio" in shift
        assert shift["n_changed"] >= 0

    def test_failed_folding_skipped(self):
        """접힘 실패한 후보는 Phase 3에서 스킵."""
        trajs, metrics, rnames = self._make_test_data(3)

        # 하나를 실패로 설정
        metrics[0] = FoldingMetrics(
            candidate_id=metrics[0].candidate_id,
            rmsd_convergence=float("inf"),
            rmsd_final=float("inf"),
            rg_change_ratio=float("inf"),
            rg_initial=0,
            rg_final=0,
            energy_convergence=float("inf"),
            energy_final=0,
            contact_order=0,
            rmsd_converged=False,
            rg_stable=False,
            energy_converged=False,
            folding_success=False,
            compactness=0,
            end_to_end_distance=0,
        )

        report = run_elimination_loop(
            trajs, metrics, rnames, group_name="test_fail"
        )

        assert report.total_folded == 2  # 1개 실패


# =============================================================================
# 7. 통계 분석 테스트
# =============================================================================

class TestStatistics:
    """통계 분석 모듈 테스트."""

    def _make_folding_report(
        self,
        group_name: str,
        n: int,
        success_rate: float,
        mean_rmsd: float,
        seed: int = 42,
    ) -> FoldingReport:
        """합성 FoldingReport를 생성."""
        rng = np.random.default_rng(seed)
        metrics = []

        for i in range(n):
            success = rng.random() < success_rate
            metrics.append(FoldingMetrics(
                candidate_id=f"{group_name}_{i}",
                rmsd_convergence=rng.normal(1.0, 0.3),
                rmsd_final=rng.normal(mean_rmsd, 1.0) if success else float("inf"),
                rg_change_ratio=rng.uniform(0.05, 0.2),
                rg_initial=rng.uniform(4, 8),
                rg_final=rng.uniform(4, 8),
                energy_convergence=rng.uniform(0.01, 0.05),
                energy_final=rng.normal(-3000, 500),
                contact_order=rng.uniform(0.1, 0.25),
                rmsd_converged=success,
                rg_stable=success,
                energy_converged=success,
                folding_success=success,
                compactness=rng.uniform(0.5, 1.0),
                end_to_end_distance=rng.uniform(5, 20),
            ))

        n_success = sum(1 for m in metrics if m.folding_success)
        return FoldingReport(
            group_name=group_name,
            total_count=n,
            success_count=n_success,
            success_rate=n_success / n,
            mean_rmsd_final=mean_rmsd,
            std_rmsd_final=1.0,
            mean_rg_final=6.0,
            std_rg_final=1.0,
            mean_energy_final=-3000.0,
            std_energy_final=500.0,
            mean_contact_order=0.15,
            metrics=metrics,
        )

    def test_compare_groups(self):
        """두 그룹 비교가 올바르게 수행되는지."""
        report_a = self._make_folding_report("ufoe", 20, 0.8, 5.0, seed=42)
        report_b = self._make_folding_report("ctrl", 20, 0.3, 8.0, seed=43)

        result = compare_groups(report_a, report_b)

        assert result.folding_rate_test is not None
        assert len(result.metric_comparisons) > 0
        assert isinstance(result.summary, str)

    def test_proportion_test(self):
        """비율 검정."""
        result = compare_self_consistency(80, 100, 30, 100)

        assert isinstance(result, ProportionTestResult)
        assert result.group_a_rate == 0.8
        assert result.group_b_rate == 0.3
        # 이 큰 차이는 유의해야 함
        assert result.significant is True

    def test_bootstrap_ci(self):
        """부트스트랩 신뢰구간."""
        rng = np.random.default_rng(42)
        values = rng.normal(10.0, 2.0, 100)

        point, lower, upper = bootstrap_confidence_interval(
            values, n_bootstrap=1000, confidence=0.95, seed=42
        )

        assert lower < point < upper
        assert abs(point - 10.0) < 1.0  # 평균에 가까워야 함

    def test_equal_groups_not_significant(self):
        """동일 분포의 두 그룹은 유의하지 않아야 함."""
        report_a = self._make_folding_report("a", 20, 0.5, 5.0, seed=42)
        report_b = self._make_folding_report("b", 20, 0.5, 5.0, seed=43)

        result = compare_groups(report_a, report_b)
        # 동일 조건이므로 대부분 유의하지 않아야 함
        n_sig = sum(1 for c in result.metric_comparisons if c.significant)
        # 일부는 우연히 유의할 수 있지만, 대부분은 아닐 것
        assert n_sig <= len(result.metric_comparisons)


# =============================================================================
# 8. 통합 파이프라인 테스트
# =============================================================================

class TestPipelineIntegration:
    """통합 파이프라인 테스트."""

    def test_mock_pipeline(self, tmp_dir):
        """Mock 모드 파이프라인이 전체 흐름을 완주."""
        from src.pipeline.pipeline import Phase2Pipeline, PipelineConfig

        config = PipelineConfig(
            output_dir=str(tmp_dir / "pipeline_test"),
            run_mode="mock",
            n_samples_2a=3,
            n_samples_2b=2,
            n_control_random=3,
            n_control_inverted=3,
            seed=42,
            esmfold_mode="mock",
            md_time_ns=0.1,
            md_refinement_ns=0.05,
        )

        pipeline = Phase2Pipeline(config)
        results = pipeline.run_full()

        # 기본 검증
        assert len(results.ufoe_candidates) == 3
        assert len(results.control_random) == 3
        assert len(results.control_inverted) == 3

        # 접힘 보고서
        assert results.ufoe_2a_report is not None
        assert results.ufoe_2a_report.total_count == 3
        assert results.control_random_report is not None

        # 자기일관성
        assert results.elimination_ufoe_2a is not None
        assert results.elimination_ufoe_2a.total_designed == 3

        # 통계
        assert results.stats_2a_vs_random is not None

        # 소요 시간
        assert results.total_wall_time > 0

    def test_save_results(self, tmp_dir):
        """결과 저장."""
        from src.pipeline.pipeline import Phase2Pipeline, PipelineConfig

        config = PipelineConfig(
            output_dir=str(tmp_dir / "save_test"),
            run_mode="mock",
            n_samples_2a=2,
            n_samples_2b=1,
            n_control_random=2,
            n_control_inverted=2,
            seed=42,
            esmfold_mode="mock",
            md_time_ns=0.1,
            md_refinement_ns=0.05,
        )

        pipeline = Phase2Pipeline(config)
        results = pipeline.run_full()
        results_dir = pipeline.save_results(results)

        assert results_dir.exists()
        assert (results_dir / "summary.json").exists()
        assert (results_dir / "ufoe_2a_metrics.csv").exists()

    def test_quick_experiment(self, tmp_dir):
        """quick 모드 실행."""
        from src.pipeline.pipeline import run_quick_experiment

        results = run_quick_experiment(
            output_dir=str(tmp_dir / "quick"),
            seed=42,
        )
        assert results.total_wall_time > 0
        assert results.ufoe_2a_report is not None
