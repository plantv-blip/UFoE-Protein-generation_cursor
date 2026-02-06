"""
OpenMM 기반 MD 시뮬레이션 모듈

UFoE 생성 구조의 접힘 검증을 위한 분자역학 시뮬레이션을 수행한다.

Phase 2a: Cα 좌표 → 전체 원자 재구성 → 암묵적 용매 MD → 100ns
Phase 2b: ESMFold 예측 구조 → MD 정제 → 10ns

의존성:
    - OpenMM >= 8.0
    - PDBFixer (전체 원자 모델 재구성)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from src.utils.constants import (
    MD_TEMPERATURE,
    MD_TIMESTEP_FS,
    MD_FRICTION_COEFF,
    MD_NONBONDED_CUTOFF,
    MD_SIMULATION_TIME_NS,
    MD_EQUILIBRATION_NS,
    MD_REPORT_INTERVAL_PS,
    MD_ENERGY_INTERVAL_PS,
    MD_REFINEMENT_TIME_NS,
)

logger = logging.getLogger(__name__)


# =============================================================================
# 데이터 구조
# =============================================================================

@dataclass
class MDTrajectory:
    """MD 시뮬레이션 궤적 데이터."""

    # 시뮬레이션 메타데이터
    candidate_id: str
    simulation_type: str        # 'pure_md' | 'esmfold_md' | 'control_md'
    n_residues: int
    total_time_ns: float

    # 시간별 데이터 (각 프레임에 대한 값)
    timestamps_ns: np.ndarray       # (n_frames,)
    potential_energies: np.ndarray   # (n_frames,) kJ/mol
    kinetic_energies: np.ndarray    # (n_frames,) kJ/mol
    temperatures: np.ndarray        # (n_frames,) K
    rg_values: np.ndarray           # (n_frames,) Å — 회전 반경
    rmsd_to_initial: np.ndarray     # (n_frames,) Å — 초기 구조 대비 RMSD

    # Cα 좌표 스냅샷 (핵심 프레임만)
    ca_snapshots: list[np.ndarray]   # list of (N, 3) — 주요 시점의 좌표
    snapshot_times_ns: list[float]   # 스냅샷 시간

    # 최종 구조
    final_ca_coords: np.ndarray     # (N, 3) — 시뮬레이션 종료 시 좌표
    initial_ca_coords: np.ndarray   # (N, 3) — 시작 좌표

    # 실행 정보
    wall_time_seconds: float = 0.0
    converged: bool = False
    error_message: str = ""

    # 추가 메타데이터
    metadata: dict = field(default_factory=dict)


@dataclass
class MDConfig:
    """MD 시뮬레이션 설정."""

    temperature: float = MD_TEMPERATURE           # K
    timestep_fs: float = MD_TIMESTEP_FS           # fs
    friction_coeff: float = MD_FRICTION_COEFF     # ps^-1
    nonbonded_cutoff: float = MD_NONBONDED_CUTOFF  # Å
    simulation_time_ns: float = MD_SIMULATION_TIME_NS  # ns
    equilibration_ns: float = MD_EQUILIBRATION_NS      # ns
    report_interval_ps: float = MD_REPORT_INTERVAL_PS  # ps
    energy_interval_ps: float = MD_ENERGY_INTERVAL_PS  # ps

    # 암묵적 용매 모델
    implicit_solvent: str = "GBn2"   # GBn2 or OBC2 or None
    solute_dielectric: float = 1.0
    solvent_dielectric: float = 78.5

    # 포스 필드
    force_field: str = "amber14-all.xml"
    water_model: str = ""  # 암묵적 용매 시 빈 문자열

    # 출력 경로
    output_dir: str | Path = "output/md"

    # 스냅샷 저장 간격 (ns)
    snapshot_interval_ns: float = 1.0

    @property
    def n_steps(self) -> int:
        """전체 시뮬레이션 스텝 수."""
        time_ps = self.simulation_time_ns * 1000.0
        return int(time_ps / (self.timestep_fs / 1000.0))

    @property
    def equilibration_steps(self) -> int:
        """평형화 스텝 수."""
        time_ps = self.equilibration_ns * 1000.0
        return int(time_ps / (self.timestep_fs / 1000.0))

    @property
    def report_steps(self) -> int:
        """좌표 저장 간격 (스텝)."""
        return int(self.report_interval_ps / (self.timestep_fs / 1000.0))

    @property
    def energy_steps(self) -> int:
        """에너지 저장 간격 (스텝)."""
        return int(self.energy_interval_ps / (self.timestep_fs / 1000.0))


# =============================================================================
# OpenMM 시뮬레이션 엔진
# =============================================================================

class MDSimulator:
    """OpenMM 기반 MD 시뮬레이션 수행기.

    사용법:
        sim = MDSimulator(config=MDConfig())
        trajectory = sim.run_from_pdb("input.pdb", candidate_id="ufoe_001")
        trajectory = sim.run_from_sequence("MVLSPADKTN...", candidate_id="ufoe_002")
    """

    def __init__(self, config: MDConfig | None = None):
        self.config = config or MDConfig()
        self._openmm_available = self._check_openmm()

    @staticmethod
    def _check_openmm() -> bool:
        """OpenMM 설치 여부를 확인한다."""
        try:
            import openmm  # noqa: F401
            return True
        except ImportError:
            logger.warning(
                "OpenMM이 설치되지 않았습니다. "
                "pip install openmm 또는 conda install -c conda-forge openmm"
            )
            return False

    @staticmethod
    def _check_pdbfixer() -> bool:
        """PDBFixer 설치 여부를 확인한다."""
        try:
            import pdbfixer  # noqa: F401
            return True
        except ImportError:
            logger.warning(
                "PDBFixer가 설치되지 않았습니다. "
                "pip install pdbfixer 또는 conda install -c conda-forge pdbfixer"
            )
            return False

    def prepare_structure(
        self,
        pdb_path: str | Path,
        output_path: str | Path | None = None,
    ) -> Path:
        """PDB 파일에서 전체 원자 모델을 준비한다.

        Cα-only PDB를 전체 원자 모델로 재구성하고,
        결손 원자/잔기를 보완하며, 수소를 추가한다.

        Parameters
        ----------
        pdb_path : str or Path — 입력 PDB 파일
        output_path : str or Path, optional — 출력 경로. None이면 자동 생성.

        Returns
        -------
        Path — 준비된 PDB 파일 경로
        """
        from pdbfixer import PDBFixer
        import openmm.app as app

        pdb_path = Path(pdb_path)
        if output_path is None:
            output_path = pdb_path.parent / f"{pdb_path.stem}_prepared.pdb"
        output_path = Path(output_path)

        logger.info(f"구조 준비 중: {pdb_path}")

        # PDBFixer로 구조 보완
        fixer = PDBFixer(filename=str(pdb_path))
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(self.config.temperature)

        # 준비된 구조 저장
        with open(output_path, "w") as f:
            app.PDBFile.writeFile(fixer.topology, fixer.positions, f)

        logger.info(f"준비 완료: {output_path}")
        return output_path

    def _build_system(
        self,
        pdb_path: Path,
    ) -> tuple:
        """OpenMM 시스템을 구성한다.

        Returns
        -------
        (topology, system, positions)
        """
        import openmm
        import openmm.app as app
        import openmm.unit as unit

        # PDB 읽기
        pdb = app.PDBFile(str(pdb_path))

        # 포스 필드 설정
        ff_files = [self.config.force_field]
        if self.config.implicit_solvent:
            ff_files.append(f"implicit/{self.config.implicit_solvent}.xml")

        forcefield = app.ForceField(*ff_files)

        # 시스템 생성
        system = forcefield.createSystem(
            pdb.topology,
            nonbondedMethod=app.NoCutoff if self.config.implicit_solvent
            else app.PME,
            constraints=app.HBonds,
        )

        return pdb.topology, system, pdb.positions

    def run_from_pdb(
        self,
        pdb_path: str | Path,
        candidate_id: str = "unknown",
        simulation_type: str = "pure_md",
        simulation_time_ns: float | None = None,
    ) -> MDTrajectory:
        """PDB 파일로부터 MD 시뮬레이션을 실행한다.

        Parameters
        ----------
        pdb_path : str or Path
        candidate_id : str
        simulation_type : str
        simulation_time_ns : float, optional — 기본값은 config에서 가져옴

        Returns
        -------
        MDTrajectory
        """
        if not self._openmm_available:
            return self._mock_trajectory(
                candidate_id, simulation_type, pdb_path,
                simulation_time_ns or self.config.simulation_time_ns
            )

        import openmm
        import openmm.app as app
        import openmm.unit as unit

        pdb_path = Path(pdb_path)
        sim_time = simulation_time_ns or self.config.simulation_time_ns

        logger.info(
            f"MD 시뮬레이션 시작: {candidate_id} | "
            f"유형: {simulation_type} | 시간: {sim_time}ns"
        )

        start_wall = time.time()

        try:
            # 1. 구조 준비
            prepared_path = self.prepare_structure(pdb_path)
            topology, system, positions = self._build_system(prepared_path)

            # 2. 적분기 설정 (Langevin)
            integrator = openmm.LangevinMiddleIntegrator(
                self.config.temperature * unit.kelvin,
                self.config.friction_coeff / unit.picosecond,
                self.config.timestep_fs * unit.femtosecond,
            )

            # 3. 시뮬레이션 객체 생성
            simulation = app.Simulation(topology, system, integrator)
            simulation.context.setPositions(positions)

            # 4. 에너지 최소화
            logger.info("에너지 최소화 중...")
            simulation.minimizeEnergy()

            # 초기 좌표 추출
            state = simulation.context.getState(getPositions=True)
            initial_positions = state.getPositions(asNumpy=True).value_in_unit(
                unit.angstrom
            )
            initial_ca = self._extract_ca_coords(topology, initial_positions)
            n_residues = len(initial_ca)

            # 5. 평형화
            logger.info(f"평형화 중: {self.config.equilibration_ns}ns...")
            eq_steps = self.config.equilibration_steps
            simulation.step(eq_steps)

            # 6. Production run — 데이터 수집
            total_steps = int(
                sim_time * 1000.0 / (self.config.timestep_fs / 1000.0)
            )
            report_every = self.config.report_steps
            n_frames = total_steps // report_every

            timestamps = np.zeros(n_frames)
            pot_energies = np.zeros(n_frames)
            kin_energies = np.zeros(n_frames)
            temperatures_arr = np.zeros(n_frames)
            rg_values = np.zeros(n_frames)
            rmsd_values = np.zeros(n_frames)

            snapshots = []
            snapshot_times = []
            snapshot_interval_steps = int(
                self.config.snapshot_interval_ns * 1000.0
                / (self.config.timestep_fs / 1000.0)
            )

            logger.info(f"Production run: {sim_time}ns ({total_steps} steps)...")

            for frame_idx in range(n_frames):
                simulation.step(report_every)

                # 상태 추출
                state = simulation.context.getState(
                    getPositions=True,
                    getEnergy=True,
                )

                pos = state.getPositions(asNumpy=True).value_in_unit(
                    unit.angstrom
                )
                ca_coords = self._extract_ca_coords(topology, pos)

                current_step = (frame_idx + 1) * report_every
                current_time_ns = (
                    current_step * self.config.timestep_fs / 1e6
                )

                timestamps[frame_idx] = current_time_ns
                pot_energies[frame_idx] = (
                    state.getPotentialEnergy().value_in_unit(
                        unit.kilojoule_per_mole
                    )
                )
                kin_energies[frame_idx] = (
                    state.getKineticEnergy().value_in_unit(
                        unit.kilojoule_per_mole
                    )
                )
                temperatures_arr[frame_idx] = (
                    2.0 * kin_energies[frame_idx]
                    / (3.0 * n_residues * 8.314e-3)
                )
                rg_values[frame_idx] = _calculate_rg(ca_coords)
                rmsd_values[frame_idx] = _calculate_rmsd(
                    initial_ca, ca_coords
                )

                # 스냅샷 저장
                if current_step % snapshot_interval_steps == 0:
                    snapshots.append(ca_coords.copy())
                    snapshot_times.append(current_time_ns)

                # 진행상황 로깅
                if frame_idx % max(1, n_frames // 10) == 0:
                    logger.info(
                        f"  Progress: {current_time_ns:.1f}/{sim_time:.1f}ns "
                        f"| PE: {pot_energies[frame_idx]:.1f} kJ/mol"
                    )

            wall_time = time.time() - start_wall

            # 최종 좌표 추출
            final_state = simulation.context.getState(getPositions=True)
            final_pos = final_state.getPositions(asNumpy=True).value_in_unit(
                unit.angstrom
            )
            final_ca = self._extract_ca_coords(topology, final_pos)

            trajectory = MDTrajectory(
                candidate_id=candidate_id,
                simulation_type=simulation_type,
                n_residues=n_residues,
                total_time_ns=sim_time,
                timestamps_ns=timestamps,
                potential_energies=pot_energies,
                kinetic_energies=kin_energies,
                temperatures=temperatures_arr,
                rg_values=rg_values,
                rmsd_to_initial=rmsd_values,
                ca_snapshots=snapshots,
                snapshot_times_ns=snapshot_times,
                final_ca_coords=final_ca,
                initial_ca_coords=initial_ca,
                wall_time_seconds=wall_time,
                converged=True,
                metadata={
                    "config": {
                        "temperature": self.config.temperature,
                        "timestep_fs": self.config.timestep_fs,
                        "force_field": self.config.force_field,
                        "implicit_solvent": self.config.implicit_solvent,
                    }
                },
            )

            logger.info(
                f"시뮬레이션 완료: {candidate_id} | "
                f"Wall time: {wall_time:.1f}s"
            )
            return trajectory

        except Exception as e:
            wall_time = time.time() - start_wall
            logger.error(f"시뮬레이션 오류: {candidate_id} | {e}")
            return self._error_trajectory(
                candidate_id, simulation_type, str(e), wall_time
            )

    def _extract_ca_coords(
        self, topology, positions: np.ndarray
    ) -> np.ndarray:
        """토폴로지에서 Cα 원자의 좌표를 추출한다."""
        import openmm.app as app

        ca_indices = []
        for atom in topology.atoms():
            if atom.name == "CA":
                ca_indices.append(atom.index)

        if len(ca_indices) == 0:
            # Cα가 없으면 모든 원자 좌표 반환 (Cα-only 모델)
            return np.array(positions)

        return np.array([positions[i] for i in ca_indices])

    def _mock_trajectory(
        self,
        candidate_id: str,
        simulation_type: str,
        pdb_path: str | Path,
        simulation_time_ns: float,
    ) -> MDTrajectory:
        """OpenMM이 없을 때 합성 궤적을 생성한다 (테스트/개발용).

        실제 물리 시뮬레이션 대신, 합리적인 범위의 합성 데이터를 생성하여
        파이프라인의 다운스트림 로직을 테스트할 수 있도록 한다.
        """
        from src.utils.pdb_writer import read_pdb_coords

        logger.warning(
            f"OpenMM 미설치 — 합성(mock) 궤적 생성: {candidate_id}"
        )

        pdb_path = Path(pdb_path)
        if pdb_path.exists():
            initial_ca, _ = read_pdb_coords(pdb_path)
        else:
            # 기본 합성 좌표
            initial_ca = np.random.default_rng(42).normal(0, 5, (60, 3))

        n_residues = len(initial_ca)
        rng = np.random.default_rng(hash(candidate_id) % (2**31))

        # 프레임 수
        n_frames = int(simulation_time_ns * 1000 / 10.0)  # 10ps 간격

        # 합성 시계열 데이터
        timestamps = np.linspace(0, simulation_time_ns, n_frames)

        # 포텐셜 에너지: 초기 높음 → 지수 감쇠 + 노이즈
        pe_init = -50.0 * n_residues
        pe_final = -80.0 * n_residues
        decay = 1.0 - np.exp(-timestamps / 20.0)
        pot_energies = pe_init + (pe_final - pe_init) * decay
        pot_energies += rng.normal(0, abs(pe_final) * 0.02, n_frames)

        kin_energies = np.full(n_frames, 1.5 * n_residues * 8.314e-3 * 300.0)
        kin_energies += rng.normal(0, kin_energies[0] * 0.05, n_frames)

        temperatures = np.full(n_frames, 300.0) + rng.normal(0, 5.0, n_frames)

        # Rg: 초기값 유지 + 약간의 변동
        rg_init = _calculate_rg(initial_ca)
        rg_values = np.full(n_frames, rg_init)
        rg_values += rng.normal(0, rg_init * 0.05, n_frames)

        # RMSD: 점진적 증가 후 수렴
        rmsd_plateau = rng.uniform(3.0, 8.0)  # 수렴 RMSD
        rmsd_values = rmsd_plateau * (1.0 - np.exp(-timestamps / 15.0))
        rmsd_values += rng.normal(0, 0.5, n_frames)
        rmsd_values = np.maximum(0, rmsd_values)

        # 최종 좌표: 약간 변형된 초기 좌표
        perturbation = rng.normal(0, 1.0, initial_ca.shape)
        final_ca = initial_ca + perturbation

        # 스냅샷
        snapshots = []
        snapshot_times = []
        for t_ns in np.arange(0, simulation_time_ns, 1.0):
            frac = t_ns / max(simulation_time_ns, 1.0)
            snap = initial_ca + frac * perturbation
            snap += rng.normal(0, 0.5, initial_ca.shape)
            snapshots.append(snap)
            snapshot_times.append(t_ns)

        return MDTrajectory(
            candidate_id=candidate_id,
            simulation_type=simulation_type,
            n_residues=n_residues,
            total_time_ns=simulation_time_ns,
            timestamps_ns=timestamps,
            potential_energies=pot_energies,
            kinetic_energies=kin_energies,
            temperatures=temperatures,
            rg_values=rg_values,
            rmsd_to_initial=rmsd_values,
            ca_snapshots=snapshots,
            snapshot_times_ns=snapshot_times,
            final_ca_coords=final_ca,
            initial_ca_coords=initial_ca,
            wall_time_seconds=0.1,
            converged=True,
            metadata={"mock": True},
        )

    def _error_trajectory(
        self,
        candidate_id: str,
        simulation_type: str,
        error_msg: str,
        wall_time: float,
    ) -> MDTrajectory:
        """오류 발생 시 빈 궤적을 반환한다."""
        return MDTrajectory(
            candidate_id=candidate_id,
            simulation_type=simulation_type,
            n_residues=0,
            total_time_ns=0.0,
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
            wall_time_seconds=wall_time,
            converged=False,
            error_message=error_msg,
        )


# =============================================================================
# 유틸리티 함수
# =============================================================================

def _calculate_rg(coords: np.ndarray) -> float:
    """회전 반경(Radius of Gyration)을 계산한다.

    Rg = sqrt(sum(|r_i - r_cm|^2) / N)

    Parameters
    ----------
    coords : np.ndarray of shape (N, 3)

    Returns
    -------
    float — Rg (Å)
    """
    if len(coords) == 0:
        return 0.0
    center = coords.mean(axis=0)
    diff = coords - center
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))


def _calculate_rmsd(
    ref_coords: np.ndarray,
    coords: np.ndarray,
) -> float:
    """두 좌표 세트 간의 RMSD를 계산한다 (정렬 없음).

    RMSD = sqrt(mean(|r_i - r_ref_i|^2))

    Parameters
    ----------
    ref_coords : np.ndarray of shape (N, 3)
    coords : np.ndarray of shape (N, 3)

    Returns
    -------
    float — RMSD (Å)
    """
    if len(ref_coords) != len(coords) or len(ref_coords) == 0:
        return float("inf")
    diff = coords - ref_coords
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))


def calculate_rmsd_aligned(
    ref_coords: np.ndarray,
    coords: np.ndarray,
) -> float:
    """Kabsch 알고리즘으로 정렬 후 RMSD를 계산한다.

    Parameters
    ----------
    ref_coords : np.ndarray of shape (N, 3)
    coords : np.ndarray of shape (N, 3)

    Returns
    -------
    float — 정렬된 RMSD (Å)
    """
    if len(ref_coords) != len(coords) or len(ref_coords) == 0:
        return float("inf")

    # 중심 정렬
    ref_c = ref_coords - ref_coords.mean(axis=0)
    mov_c = coords - coords.mean(axis=0)

    # 공분산 행렬
    H = mov_c.T @ ref_c

    # SVD
    U, S, Vt = np.linalg.svd(H)

    # 반전 보정 (reflection)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    diag = np.diag([1.0, 1.0, d])

    # 최적 회전 행렬
    R = Vt.T @ diag @ U.T

    # 회전 적용
    aligned = mov_c @ R.T

    # RMSD 계산
    diff = aligned - ref_c
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))
