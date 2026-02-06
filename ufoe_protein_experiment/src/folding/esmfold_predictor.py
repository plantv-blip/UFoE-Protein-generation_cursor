"""
ESMFold 구조 예측. local / api / mock 3모드.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class ESMFoldResult:
    coords: np.ndarray  # (N, 3) CA
    pdb_path: Optional[str]
    confidence: Optional[float]


class ESMFoldPredictor:
    def __init__(self, mode: str = "mock"):
        self.mode = mode  # "mock" | "local" | "api"

    def predict(
        self,
        sequence: str,
        output_dir: Path,
        run_id: str,
    ) -> ESMFoldResult:
        if self.mode == "mock":
            return self._predict_mock(sequence, output_dir, run_id)
        if self.mode == "api":
            return self._predict_api(sequence, output_dir, run_id)
        return self._predict_local(sequence, output_dir, run_id)

    def _predict_mock(
        self,
        sequence: str,
        output_dir: Path,
        run_id: str,
    ) -> ESMFoldResult:
        """합성 CA 좌표 (나선형 스캐폴드)."""
        n = len(sequence)
        t = np.linspace(0, 4 * np.pi, n)
        r = 5.0 + 2 * np.sin(t * 0.5)
        x = r * np.cos(t)
        y = r * np.sin(t)
        z = 2 * t
        coords = np.column_stack([x, y, z])
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        pdb_path = output_dir / f"{run_id}.pdb"
        from ..utils.pdb_writer import coords_to_residues, structure_to_pdb
        residues = coords_to_residues(coords.tolist(), sequence)
        structure_to_pdb(residues, pdb_path)
        return ESMFoldResult(coords=coords, pdb_path=str(pdb_path), confidence=0.85)

    def _predict_api(self, sequence: str, output_dir: Path, run_id: str) -> ESMFoldResult:
        """API 호출. 미구현 시 mock."""
        return self._predict_mock(sequence, output_dir, run_id)

    def _predict_local(self, sequence: str, output_dir: Path, run_id: str) -> ESMFoldResult:
        """로컬 ESMFold. 미구현 시 mock."""
        return self._predict_mock(sequence, output_dir, run_id)
