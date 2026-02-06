"""
OpenMM 기반 MD 시뮬레이션. mock 모드: OpenMM 미설치 시 합성 궤적 반환.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np

try:
    import openmm
    HAS_OPENMM = True
except ImportError:
    HAS_OPENMM = False


@dataclass
class MDResult:
    fold_success: bool
    final_rmsd: Optional[float]
    energy_score: Optional[float]
    output_pdb_path: str
    zone_EC: Optional[float]
    zone_TZ: Optional[float]
    zone_BZ: Optional[float]
    runtime_sec: Optional[float]
    rmsd_trajectory: List[float]
    rg_trajectory: List[float]
    energy_trajectory: List[float]


def _mock_trajectory(n_frames: int, n_residues: int) -> tuple:
    """합성 궤적: RMSD 감소, Rg/에너지 안정."""
    rmsd = 5.0 * np.exp(-np.linspace(0, 3, n_frames)) + 0.5 * np.random.randn(n_frames).cumsum()
    rmsd = np.maximum(rmsd, 0.5)
    rg = 10.0 + 0.5 * np.random.randn(n_frames).cumsum()
    rg = np.maximum(rg, 5.0)
    energy = -500 - 2 * np.linspace(0, 1, n_frames) + 5 * np.random.randn(n_frames)
    return (
        rmsd.tolist(),
        rg.tolist(),
        energy.tolist(),
    )


class MDSimulator:
    def __init__(self, run_mode: str = "mock"):
        self.run_mode = run_mode  # "mock" | "full"

    def run(
        self,
        initial_structure: List[Dict[str, Any]],
        output_dir: Path,
        run_id: str,
        length_ns: float = 1.0,
    ) -> MDResult:
        if self.run_mode == "mock" or not HAS_OPENMM:
            return self._run_mock(initial_structure, output_dir, run_id, length_ns)
        return self._run_full(initial_structure, output_dir, run_id, length_ns)

    def _run_mock(
        self,
        initial_structure: List[Dict[str, Any]],
        output_dir: Path,
        run_id: str,
        length_ns: float,
    ) -> MDResult:
        n_res = len(initial_structure)
        n_frames = max(20, int(length_ns * 10))
        rmsd_t, rg_t, energy_t = _mock_trajectory(n_frames, n_res)
        from .folding_metrics import (
            compute_rmsd_convergence,
            compute_rg_stability,
            compute_energy_convergence,
            folding_success_criteria,
        )
        ok_rmsd, sigma = compute_rmsd_convergence(rmsd_t)
        ok_rg, rg_ch = compute_rg_stability(rg_t)
        ok_e, e_cv = compute_energy_convergence(energy_t)
        success = folding_success_criteria(sigma, rg_ch, e_cv)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        pdb_path = output_dir / f"{run_id}.pdb"
        from ..utils.pdb_writer import structure_to_pdb
        structure_to_pdb(initial_structure, pdb_path)
        # zone 비율 (초기 구조 기준, 단순화)
        from ..filters.ufoef_filters import (
            calculate_geometric_center,
            calculate_residue_distances,
            classify_zones,
        )
        from ..utils.constants import EC_RADIUS, TZ_OUTER_RADIUS
        center = calculate_geometric_center(initial_structure)
        dists = calculate_residue_distances(initial_structure, center)
        zones = classify_zones(dists, ec_radius=EC_RADIUS, tz_radius=TZ_OUTER_RADIUS)
        n = sum(len(zones[z]) for z in zones)
        zone_EC = len(zones["EC"]) / n if n else 0
        zone_TZ = len(zones["TZ"]) / n if n else 0
        zone_BZ = len(zones["BZ"]) / n if n else 0
        return MDResult(
            fold_success=success,
            final_rmsd=float(rmsd_t[-1]) if rmsd_t else None,
            energy_score=float(energy_t[-1]) if energy_t else None,
            output_pdb_path=str(pdb_path),
            zone_EC=zone_EC,
            zone_TZ=zone_TZ,
            zone_BZ=zone_BZ,
            runtime_sec=length_ns * 0.1,
            rmsd_trajectory=rmsd_t,
            rg_trajectory=rg_t,
            energy_trajectory=energy_t,
        )

    def _run_full(
        self,
        initial_structure: List[Dict[str, Any]],
        output_dir: Path,
        run_id: str,
        length_ns: float,
    ) -> MDResult:
        """실제 OpenMM 실행. 미구현 시 mock 대체."""
        return self._run_mock(initial_structure, output_dir, run_id, length_ns)
